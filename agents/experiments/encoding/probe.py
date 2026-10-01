#!/usr/bin/env python3
"""Offline diagnosis + reflex separability for each sensory encoding (encoding study).

Input: reflex rollouts recorded by collect.py (the reflex drives the engine; the circuit is run
afterwards on the stored observations - exact, since the circuit never acts). For each
(encoding, flipper-obs mode) it runs the fixed circuit over every life and reports:

Separability (the step-3 protocol of agents/experiments/step3/reflex_separability.py): per-side
pos-weighted logistic regression on the 64 standardized readout cells, fit on lives 7000-7011,
tested on 7012-7023 (and the swapped fold); ROC-AUC and precision at the label's own press rate.
Also "prec@rate+-1": a predicted press counts if a reflex press on that side is within +-1 step.

Diagnosis (flipper zone = y > 10):
  zone R^2   2-fold (by life) ridge probe of x, y, vy from the readout, zone steps only.
  d'(dy)     local resolvability of a 0.1-unit y step (a typical per-step fall is 0.1-0.4): the
             finite-difference readout change dr for y + 0.1 from the same previous state, measured
             against the zone covariance S of the readout residual after regressing out the current
             observation linearly: d' = sqrt(dr^T S^-1 dr) (ridge 1e-3 tr/n). Same for vy + 1.0 and
             for the flipper-state channel 0 -> 1 (dFlip) for scale.
  sat_in / sat_all / sat_ro  fraction of |activity| > 0.95 at input / all / readout cells in zone.
  rec/drv    median |1.4 * recurrent message| / |external drive| at the ball_y input cells.
  clip       fraction of dec.norm outputs at the +-10 clip (norm calibrated as in production).
  PR         participation ratio of the readout covariance in zone vs. over all steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

import torch  # noqa: E402

torch.set_num_threads(1)

from agents import train_pinball_circuit_cem as T  # noqa: E402
from agents.experiments.encoding.encoding import build_encoded_agent  # noqa: E402
from agents.fixed_circuit_agent import DYNAMICS_GAIN, DYNAMICS_ITERATIONS, DYNAMICS_LEAK  # noqa: E402

CONN = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")


def filtered_obs(obs, life, fobs):
	out = np.empty_like(obs)
	filt, cur = None, None
	for t in range(len(obs)):
		if life[t] != cur:
			filt, cur = T.FlipperObsFilter(fobs), life[t]
		out[t] = filt(obs[t])
	return out


def run_circuit(agent, fobs_arr, life):
	"""Returns activity H (T, n) and H_prev (state before each step)."""
	drive = agent._channel_drive(torch.as_tensor(fobs_arr))
	H = np.empty((len(life), agent.brain.n), dtype=np.float32)
	Hp = np.empty_like(H)
	h, cur = None, None
	with torch.no_grad():
		for t in range(len(life)):
			if life[t] != cur:
				h, cur = agent.initial_state(1), life[t]
			Hp[t] = h[0].numpy()
			h = agent.brain.step(h, drive[t:t + 1])
			H[t] = h[0].numpy()
	return H, Hp


def step_instrumented(agent, h_prev, drive):
	"""Replicates FixedConnectome.step, returning final activity, last external drive and message."""
	b = agent.brain
	d = torch.zeros_like(h_prev)
	d[:, b.input_cells] = drive[:, b.input_channels]
	a = h_prev
	for _ in range(DYNAMICS_ITERATIONS):
		msg = torch.zeros_like(a).index_add_(1, b.post, a[:, b.pre] * b.w)
		a = (1 - DYNAMICS_LEAK) * a + DYNAMICS_LEAK * torch.tanh(d + DYNAMICS_GAIN * msg)
	return a, d, msg


def auc_prec(s, yt, tol_mask=None):
	order = np.argsort(-s)
	ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
	pos = yt == 1; npos, nneg = pos.sum(), (~pos).sum()
	auc = 1 - (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg)
	k = max(1, int(npos)); top = order[:k]
	prec = yt[top].mean()
	prec_tol = tol_mask[top].mean() if tol_mask is not None else float("nan")
	return auc, prec, prec_tol


def fit_eval(X, LAB, tr, te, tol, epochs=800, seed=0):
	torch.manual_seed(seed)
	X = torch.as_tensor(X, dtype=torch.float32)
	mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
	X = (X - mu) / sd
	res = []
	for side in (0, 1):
		y = torch.as_tensor(LAB[:, side], dtype=torch.float32)
		model = torch.nn.Linear(X.shape[1], 1)
		opt = torch.optim.Adam(model.parameters(), lr=1e-2)
		pw = torch.tensor([(1 - y[tr].mean()) / y[tr].mean().clamp_min(1e-4)])
		for _ in range(epochs):
			loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X[tr]).squeeze(-1), y[tr], pos_weight=pw)
			opt.zero_grad(); loss.backward(); opt.step()
		with torch.no_grad():
			s = model(X[te]).squeeze(-1).numpy()
		res.append(auc_prec(s, LAB[te, side], tol[te, side]))
	return res


def ridge_r2(X, y, folds, lam=1.0):
	pred = np.empty_like(y)
	for tr, te in folds:
		mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
		A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
		w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
		pred[te] = B @ w + y[tr].mean()
	return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def participation_ratio(X):
	ev = np.clip(np.linalg.eigvalsh(np.cov(X.T)), 0, None)
	return ev.sum() ** 2 / (ev ** 2).sum()


def analyse(encoding, fobs, data, diag=True):
	obs, LAB, life = data["obs"], data["lab"], data["life"]
	agent = build_encoded_agent(CONN, 8, encoding)
	import gymnasium as gym

	space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
	agent.calibrate(space)  # same probe as production (random-walk obs) for the readout norm
	fo = filtered_obs(obs, life, fobs)
	H, Hp = run_circuit(agent, fo, life)
	idx = agent.decoder.idx.numpy()
	R = H[:, idx]
	# tolerance mask: a reflex press on that side within +-1 step (same life)
	tol = LAB.copy()
	same_prev = np.r_[False, life[1:] == life[:-1]]
	same_next = np.r_[life[:-1] == life[1:], False]
	tol[1:] |= LAB[:-1] * same_prev[1:, None]
	tol[:-1] |= LAB[1:] * same_next[:-1, None]
	A, B = life < 7012, life >= 7012
	f1 = fit_eval(R, LAB, A, B, tol)
	f2 = fit_eval(R, LAB, B, A, tol)
	out = dict(encoding=encoding, fobs=fobs,
	           L=dict(auc=f1[0][0], prec=f1[0][1], prec_tol=f1[0][2], auc_sw=f2[0][0], prec_sw=f2[0][1]),
	           R=dict(auc=f1[1][0], prec=f1[1][1], prec_tol=f1[1][2], auc_sw=f2[1][0], prec_sw=f2[1][1]))
	if not diag:
		return out
	z = obs[:, 1] > 10
	folds = [(A[z], B[z]), (B[z], A[z])]
	Rz = R[z]
	out["zone_r2"] = {k: float(ridge_r2(Rz, obs[z, c], folds)) for k, c in (("x", 0), ("y", 1), ("vy", 3))}
	out["all_r2_y"] = float(ridge_r2(R, obs[:, 1], [(A, B), (B, A)]))
	# residual covariance of the standardized readout in zone after regressing out current obs
	mu, sd = Rz.mean(0), Rz.std(0) + 1e-6
	Zs = (Rz - mu) / sd
	O = np.c_[fo[z], np.ones(z.sum())]
	resid = Zs - O @ np.linalg.lstsq(O, Zs, rcond=None)[0]
	out["linfrac_zone"] = float(1 - resid.var(0).sum() / Zs.var(0).sum())  # readout variance linear in current obs
	act = (obs[:, 1] > 10) & (obs[:, 1] < 13.9) & (obs[:, 0] > -6.5) & (np.abs(obs[:, 3]) > 0.5)
	out["active_r2"] = {k: float(ridge_r2(R[act], obs[act, c], [(A[act], B[act]), (B[act], A[act])])) for k, c in (("y", 1), ("vy", 3))}
	S = np.cov(resid.T); S += 1e-3 * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
	Sinv = np.linalg.inv(S)
	zi = np.where(z)[0]
	rng = np.random.default_rng(0)
	zi = rng.choice(zi, size=min(1500, len(zi)), replace=False)
	hp = torch.as_tensor(Hp[zi])
	dps = {}
	for key, ch, delta, absolute in (("dy0.1", 1, 0.1, False), ("dvy1", 3, 1.0, False), ("dFlip", 12, 1.0, True)):
		o0 = torch.as_tensor(fo[zi]); o1 = o0.clone()
		if absolute:
			o0[:, ch] = 0.0; o1[:, ch] = 1.0
		else:
			o1[:, ch] += delta
		with torch.no_grad():
			a0 = agent.brain.step(hp, agent._channel_drive(o0)).numpy()[:, idx]
			a1 = agent.brain.step(hp, agent._channel_drive(o1)).numpy()[:, idx]
		dr = (a1 - a0) / sd
		dps[key] = float(np.median(np.sqrt(np.einsum("ij,jk,ik->i", dr, Sinv, dr))))
	out["dprime"] = dps
	with torch.no_grad():
		a, d, msg = step_instrumented(agent, hp, agent._channel_drive(torch.as_tensor(fo[zi])))
	a, d, msg = a.numpy(), d.numpy(), msg.numpy()
	inp = agent.brain.input_cells.numpy()
	y_cells = inp[12:24]  # ball_y channel cells (per-cell layout keeps channel order)
	out["sat_in"] = float((np.abs(a[:, inp]) > 0.95).mean())
	out["sat_all"] = float((np.abs(a) > 0.95).mean())
	out["sat_ro"] = float((np.abs(a[:, idx]) > 0.95).mean())
	out["rec_over_drive_y"] = float(np.median(np.abs(DYNAMICS_GAIN * msg[:, y_cells]) / (np.abs(d[:, y_cells]) + 1e-6)))
	ya = H[z][:, y_cells]
	out["ycell_corr_y_zone"] = float(np.median([abs(np.corrcoef(ya[:, k], obs[z, 1])[0, 1]) for k in range(12)]))
	out["ycell_range_zone"] = float(np.median(np.percentile(ya, 95, axis=0) - np.percentile(ya, 5, axis=0)))
	with torch.no_grad():
		nr = agent.decoder.norm(torch.as_tensor(R)).numpy()
	out["clip"] = float((np.abs(nr) >= 9.999).mean())
	out["PR_zone"] = float(participation_ratio(Zs))
	Rs = (R - R.mean(0)) / (R.std(0) + 1e-6)
	out["PR_all"] = float(participation_ratio(Rs))
	return out


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--data", default=os.path.join(ROOT, "agents/experiments/encoding/reflex_rollouts_7000.npz"))
	ap.add_argument("--encodings", default="broadcast,gain,popcode,dspop,retino,popcode+ttc,dspop+ttc")
	ap.add_argument("--fobs", default="delay1")
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/encoding/probe.json"))
	ap.add_argument("--no-diag", action="store_true")
	args = ap.parse_args()
	data = dict(np.load(args.data))
	data["lab"] = data["lab"].astype(np.int64)
	print(f"steps {len(data['lab'])}  press rate L {data['lab'][:, 0].mean():.4f} R {data['lab'][:, 1].mean():.4f}", flush=True)
	results = []
	for fobs in args.fobs.split(","):
		for enc in args.encodings.split(","):
			r = analyse(enc, fobs, data, diag=not args.no_diag)
			results.append(r)
			L, Rr = r["L"], r["R"]
			line = (f"{enc:<14} [{fobs:<6}] L AUC {L['auc']:.3f} prec {L['prec']:.3f} (sw {L['prec_sw']:.3f}) tol {L['prec_tol']:.3f} | "
			        f"R AUC {Rr['auc']:.3f} prec {Rr['prec']:.3f} (sw {Rr['prec_sw']:.3f}) tol {Rr['prec_tol']:.3f}")
			if "dprime" in r:
				line += (f"\n    zoneR2 x {r['zone_r2']['x']:.2f} y {r['zone_r2']['y']:.2f} vy {r['zone_r2']['vy']:.2f} (allR2 y {r['all_r2_y']:.2f}) | "
				         f"d' dy0.1 {r['dprime']['dy0.1']:.2f} dvy1 {r['dprime']['dvy1']:.2f} dFlip {r['dprime']['dFlip']:.2f} | "
				         f"sat in {r['sat_in']:.2f} all {r['sat_all']:.2f} ro {r['sat_ro']:.2f} | rec/drv(y) {r['rec_over_drive_y']:.2f} | "
				         f"ycell |corr| {r['ycell_corr_y_zone']:.2f} range {r['ycell_range_zone']:.3f} | clip {r['clip']:.3f} | "
				         f"PR zone {r['PR_zone']:.1f} all {r['PR_all']:.1f} | lin {r['linfrac_zone']:.3f} | activeR2 y {r['active_r2']['y']:.2f} vy {r['active_r2']['vy']:.2f}")
			print(line, flush=True)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(results, fh, indent=1)


if __name__ == "__main__":
	main()
