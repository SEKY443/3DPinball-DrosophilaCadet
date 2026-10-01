"""Is the oracle's best flipper timing predictable from what the agent sees?

Reads timing_oracle.npz (observation before the window -> value of each timing option) and runs a
seed-grouped 5-fold cross-validation of two value models (per-option linear ridge, small MLP). Each
model predicts every option's value and picks the argmax. Compares the mean realised value of that
choice against: the reflex's own timing (offset 0), the best single fixed offset, and the oracle.
If the learned choice doesn't beat the best fixed offset, timing isn't learnable from the
observation and imitation training would have nothing to learn.

Usage: python agents/experiments/timing/timing_learnability.py [--data timing_oracle.npz]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VALUE_SCALE = 10000.0


def _folds(seeds: np.ndarray, k: int, rng: np.random.Generator):
	uniq = rng.permutation(np.unique(seeds))
	for i in range(k):
		test_seeds = set(uniq[i::k].tolist())
		test = np.array([s in test_seeds for s in seeds])
		yield ~test, test


def _standardize(tr: np.ndarray, te: np.ndarray):
	mu, sd = tr.mean(0), tr.std(0) + 1e-6
	return (tr - mu) / sd, (te - mu) / sd


def _ridge(xtr, ytr, xte, lam=10.0):
	xtr1 = np.hstack([xtr, np.ones((len(xtr), 1))])
	xte1 = np.hstack([xte, np.ones((len(xte), 1))])
	w = np.linalg.solve(xtr1.T @ xtr1 + lam * np.eye(xtr1.shape[1]), xtr1.T @ ytr)
	return xte1 @ w


def _mlp(xtr, ytr, xte, seed=0, epochs=400):
	torch.manual_seed(seed)
	net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
	                          torch.nn.Tanh(), torch.nn.Linear(64, ytr.shape[1]))
	opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
	x, y = torch.as_tensor(xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
	for _ in range(epochs):
		opt.zero_grad()
		loss = torch.nn.functional.smooth_l1_loss(net(x), y)
		loss.backward()
		opt.step()
	with torch.no_grad():
		return net(torch.as_tensor(xte, dtype=torch.float32)).numpy()


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--data", default=os.path.join(HERE, "timing_oracle.npz"))
	ap.add_argument("--folds", type=int, default=5)
	args = ap.parse_args()
	d = np.load(args.data)
	x, v, seeds, opts = d["feat"].astype(np.float64), d["values"].astype(np.float64), d["seed"], d["options"]
	bank = d["bank"].astype(np.float64)
	n = len(v)
	i0 = int(np.where(opts == 0)[0][0])
	labels = [("none" if o == -99 else f"{o:+d}") for o in opts]
	print(f"{n} approaches from {len(np.unique(seeds))} lives; options {labels}")
	print("mean value per option: " + "  ".join(f"{l} {m:7.0f}" for l, m in zip(labels, v.mean(0))))
	print("bank hits per approach per option: " + "  ".join(f"{l} {m:.3f}" for l, m in zip(labels, bank.mean(0))))
	print("drain rate per option: " + "  ".join(f"{l} {m:.3f}" for l, m in zip(labels, d["drained"].mean(0))))
	best = v.argmax(1)
	ties = (v == v.max(1, keepdims=True)).sum(1)
	print(f"oracle best option distribution: " + "  ".join(f"{l} {np.mean(best == j):.2f}" for j, l in enumerate(labels))
	      + f"   (all-options-tied approaches: {np.mean(ties == len(opts)):.2f})")
	rng = np.random.default_rng(0)
	chosen = {"ridge": np.zeros(n), "mlp": np.zeros(n), "fixed": np.zeros(n)}
	for tr, te in _folds(seeds, args.folds, rng):
		xtr, xte = _standardize(x[tr], x[te])
		ytr = np.clip(v[tr], -20000, 30000) / VALUE_SCALE
		fixed = int(v[tr].mean(0).argmax())  # best single offset, chosen on the training folds only
		chosen["fixed"][te] = v[te, fixed]
		for name, pred in (("ridge", _ridge(xtr, ytr, xte)), ("mlp", _mlp(xtr, ytr, xte))):
			chosen[name][te] = v[te][np.arange(te.sum()), pred.argmax(1)]
	oracle, reflex = v.max(1), v[:, i0]
	print("\nmean realised value per approach (cross-validated, grouped by life):")
	for name, arr in (("oracle (upper bound)", oracle), ("reflex timing (offset 0)", reflex),
	                  ("best fixed offset", chosen["fixed"]), ("ridge choice", chosen["ridge"]), ("mlp choice", chosen["mlp"])):
		print(f"  {name:<26} {arr.mean():8.0f}")
	for name in ("ridge", "mlp"):
		diff = chosen[name] - chosen["fixed"]
		se = diff.std(ddof=1) / np.sqrt(n)
		print(f"  {name} - best fixed: {diff.mean():+.0f} +- {se:.0f}  (t {diff.mean() / se:+.2f})")
	gap = oracle - chosen["fixed"]
	print(f"  share of the oracle gap captured: ridge {(chosen['ridge'] - chosen['fixed']).mean() / gap.mean():.2f}, "
	      f"mlp {(chosen['mlp'] - chosen['fixed']).mean() / gap.mean():.2f}")


if __name__ == "__main__":
	main()
