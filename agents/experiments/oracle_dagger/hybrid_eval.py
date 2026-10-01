#!/usr/bin/env python3
"""Trigger-detector + hybrid policy evaluation.

Trains a small MLP on the 16 readout activities to detect the moment an oracle
correction ("release both flippers 1 step, then press both 3 steps") should be
triggered, then evaluates a hybrid policy (champion + forced intervention on
trigger) against the plain gen-125 champion on held-out seeds.

Reuses the env/agent setup and pooled-eval pattern from
agents/experiments/oracle_dagger/dagger.py.

Usage:
    .venv/bin/python agents/experiments/oracle_dagger/hybrid_eval.py
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

from agents.experiments.oracle_dagger.dagger import BINARY, CONNECTOME, MAX_STEPS, PRESS_STEPS, RELEASE_STEPS

CHAMPION_CKPT = os.path.join(ROOT, "agents/experiments/life_objective_2026-09-26/best_heldout_gen125_champion.pt")
DATA_GLOBS = [os.path.join(ROOT, f"agents/experiments/oracle_dagger/run{k}/data_iter*.npz") for k in (2, 3, 4)]
DETECTOR_PATH = os.path.join(ROOT, "agents/experiments/oracle_dagger/detector.pt")
SCORES_PATH = os.path.join(ROOT, "agents/experiments/oracle_dagger/hybrid_scores.npz")
EVAL_SEEDS = list(range(4000, 4096))
INTERVENTION_SEQ = [(0, 0)] * RELEASE_STEPS + [(1, 1)] * PRESS_STEPS
N_EPOCHS = 2000
TARGET_FPRS = (0.003, 0.01, 0.03)


# ---------------------------------------------------------------------------
# Trigger dataset
# ---------------------------------------------------------------------------

def build_trigger_dataset():
	"""Positives = first step of each corrective block (maximal run of w>1);
	negatives = all rows with w==1. Order is preserved (concat order across files)."""
	X, Y = [], []
	files = []
	for pattern in DATA_GLOBS:
		files.extend(sorted(glob.glob(pattern)))
	assert files, "no data_iter*.npz files found"
	n_pos_files = 0
	for f in files:
		d = np.load(f)
		R, w = d["R"], d["w"]
		neg_mask = w == 1
		X.append(R[neg_mask])
		Y.append(np.zeros(neg_mask.sum(), dtype=np.float32))
		mask = w > 1
		idx = np.where(mask)[0]
		i = 0
		starts = []
		while i < len(idx):
			j = i
			while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
				j += 1
			starts.append(idx[i])
			i = j + 1
		if starts:
			X.append(R[starts])
			Y.append(np.ones(len(starts), dtype=np.float32))
			n_pos_files += len(starts)
	X = np.concatenate(X).astype(np.float32)
	Y = np.concatenate(Y).astype(np.float32)
	print(f"trigger dataset: {len(files)} files, {n_pos_files} positives, {int((Y == 0).sum())} negatives")
	return X, Y


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
	"""Mann-Whitney U / rank-sum AUC, tie-averaged ranks, no external deps."""
	order = np.argsort(scores, kind="mergesort")
	sorted_scores = scores[order]
	ranks = np.empty(len(scores), dtype=np.float64)
	i = 0
	r = 1
	while i < len(sorted_scores):
		j = i
		while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
			j += 1
		avg_rank = (r + (r + (j - i))) / 2.0
		ranks[i:j + 1] = avg_rank
		r += (j - i + 1)
		i = j + 1
	rank_full = np.empty_like(ranks)
	rank_full[order] = ranks
	n_pos = labels.sum()
	n_neg = len(labels) - n_pos
	if n_pos == 0 or n_neg == 0:
		return float("nan")
	sum_ranks_pos = rank_full[labels == 1].sum()
	return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def threshold_for_fpr(probs_neg: np.ndarray, target_fpr: float) -> float:
	"""Smallest threshold s.t. fraction of negatives with prob > threshold <= target_fpr."""
	sorted_neg = np.sort(probs_neg)[::-1]
	k = max(0, int(round(target_fpr * len(sorted_neg))) - 1)
	k = min(k, len(sorted_neg) - 1)
	return float(sorted_neg[k])


class Detector(nn.Module):
	def __init__(self):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(16, 32), nn.Tanh(),
			nn.Linear(32, 32), nn.Tanh(),
			nn.Linear(32, 1),
		)

	def forward(self, x):
		return self.net(x).squeeze(-1)


def train_detector(X: torch.Tensor, Y: torch.Tensor, epochs: int = N_EPOCHS) -> Detector:
	n_pos = Y.sum().item()
	n_neg = len(Y) - n_pos
	pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))
	model = Detector()
	opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
	for _ in range(epochs):
		logits = model(X)
		loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y, pos_weight=pos_weight)
		opt.zero_grad()
		loss.backward()
		opt.step()
	return model


def fit_and_save_detector():
	X_np, Y_np = build_trigger_dataset()
	n = len(X_np)
	n_train = int(n * 0.8)
	mean = X_np[:n_train].mean(axis=0)
	std = X_np[:n_train].std(axis=0)
	std[std < 1e-6] = 1e-6
	Xn = (X_np - mean) / std

	X_t = torch.as_tensor(Xn, dtype=torch.float32)
	Y_t = torch.as_tensor(Y_np, dtype=torch.float32)
	X_train, Y_train = X_t[:n_train], Y_t[:n_train]
	X_held, Y_held = X_t[n_train:], Y_t[n_train:]

	model = train_detector(X_train, Y_train)
	with torch.no_grad():
		train_probs = torch.sigmoid(model(X_train)).numpy()
		held_probs = torch.sigmoid(model(X_held)).numpy()
	train_auc = auc(train_probs, Y_train.numpy())
	held_auc = auc(held_probs, Y_held.numpy())
	print(f"sanity split: train n={len(Y_train)} (pos {int(Y_train.sum())})  held n={len(Y_held)} (pos {int(Y_held.sum())})")
	print(f"train AUC {train_auc:.4f}  held-out AUC {held_auc:.4f}")

	held_neg_probs = held_probs[Y_held.numpy() == 0]
	thresholds = {fpr: threshold_for_fpr(held_neg_probs, fpr) for fpr in TARGET_FPRS}
	held_fpr_actual = {fpr: float((held_neg_probs > thresholds[fpr]).mean()) for fpr in TARGET_FPRS}
	for fpr, thr in thresholds.items():
		print(f"threshold for held-out FPR~{fpr:.3%}: {thr:.4f}  (actual held-out FPR {held_fpr_actual[fpr]:.4%})")

	# retrain on all data (same mean/std/threshold), final detector to ship
	final_model = train_detector(X_t, Y_t)
	torch.save({
		"state_dict": final_model.state_dict(),
		"mean": mean, "std": std,
		"threshold": thresholds[0.01],
		"thresholds": thresholds,
		"train_auc": train_auc, "held_auc": held_auc,
	}, DETECTOR_PATH)
	print(f"saved detector to {DETECTOR_PATH}")
	return thresholds, train_auc, held_auc


# ---------------------------------------------------------------------------
# Pooled evaluation workers
# ---------------------------------------------------------------------------

_ENV = None
_AGENT = None
_DETECTOR = None
_MEAN = None
_STD = None
_CHAMPION_FLAT = None


def _worker_init(readout_dim: int, detector_state, mean, std, champion_flat):
	global _ENV, _AGENT, _DETECTOR, _MEAN, _STD, _CHAMPION_FLAT
	import atexit

	import gymnasium as gym

	from agents.train_pinball_circuit_cem import build_agent, set_flat_params
	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=BINARY, headless=True, frame_skip=4), max_episode_steps=MAX_STEPS)
	_AGENT = build_agent("circuit", CONNECTOME, readout_dim)
	_AGENT.calibrate(_ENV.observation_space)
	set_flat_params(_AGENT.decoder, champion_flat)
	_CHAMPION_FLAT = champion_flat
	_DETECTOR = Detector()
	_DETECTOR.load_state_dict(detector_state)
	_DETECTOR.eval()
	_MEAN = torch.as_tensor(mean, dtype=torch.float32)
	_STD = torch.as_tensor(std, dtype=torch.float32)
	signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
	atexit.register(_ENV.close)


@torch.no_grad()
def _run_one_life(seed: int, mode: str, threshold: float = None):
	obs, _ = _ENV.reset(seed=seed)
	h = _AGENT.initial_state(1)
	dec = _AGENT.decoder
	score, steps, n_interventions = 0.0, 0, 0
	queue = []
	for _ in range(MAX_STEPS):
		h = _AGENT.brain.step(h, _AGENT._channel_drive(torch.as_tensor(obs).unsqueeze(0)))
		readout = h[0, dec.idx]
		logits = dec.dist_inputs(dec.features(h))[0]
		l, r = int(logits[0] > 0), int(logits[1] > 0)

		if mode == "hybrid":
			if queue:
				l, r = queue.pop(0)
			else:
				x = ((readout - _MEAN) / _STD).unsqueeze(0)
				prob = torch.sigmoid(_DETECTOR(x)).item()
				if prob > threshold:
					queue = list(INTERVENTION_SEQ)
					n_interventions += 1
					l, r = queue.pop(0)

		obs, _, term, trunc, info = _ENV.step(np.array([l, r, 0]))
		score += info["score_delta"]
		steps += 1
		if info["drained"] or term or trunc:
			break
	return score, steps, n_interventions


def eval_task(task):
	seed, mode, threshold = task
	return _run_one_life(seed, mode, threshold)


def run_pool_eval(pool, mode: str, threshold: float = None):
	results = pool.map(eval_task, [(s, mode, threshold) for s in EVAL_SEEDS])
	scores = np.array([r[0] for r in results], dtype=np.float64)
	lives = np.array([r[1] for r in results], dtype=np.int64)
	interventions = np.array([r[2] for r in results], dtype=np.int64)
	return scores, lives, interventions


def paired_report(name: str, scores: np.ndarray, lives: np.ndarray, interventions: np.ndarray,
                   base_scores: np.ndarray):
	log_s = np.log1p(np.maximum(scores, 0))
	log_b = np.log1p(np.maximum(base_scores, 0))
	d = log_s - log_b
	se = d.std(ddof=1) / np.sqrt(len(d))
	t = d.mean() / se if se > 0 else float("nan")
	win_rate = float((scores > base_scores).mean())
	print(f"{name}: paired log1p diff {d.mean():+.4f} (SE {se:.4f}, t={t:.2f})  "
	      f"median score {np.median(scores):.0f}  mean life {lives.mean():.0f}  "
	      f"win rate {win_rate:.3f}  interventions/life {interventions.mean():.3f}")


def main():
	t0 = time.time()
	thresholds, train_auc, held_auc = fit_and_save_detector()

	ck = torch.load(CHAMPION_CKPT, weights_only=False)
	readout_dim = ck["readout_dim"]
	champion_flat = ck["champion"]

	det = torch.load(DETECTOR_PATH, weights_only=False)
	detector_state = det["state_dict"]
	mean, std = det["mean"], det["std"]

	ctx = mp.get_context("spawn")
	pool = ctx.Pool(8, initializer=_worker_init, initargs=(readout_dim, detector_state, mean, std, champion_flat))
	try:
		print(f"\nevaluating {len(EVAL_SEEDS)} seeds ({EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]})")
		champ_scores, champ_lives, _ = run_pool_eval(pool, "champion")
		print(f"champion: median score {np.median(champ_scores):.0f}  mean life {champ_lives.mean():.0f}")

		save_data = {"champion_scores": champ_scores, "champion_lives": champ_lives, "seeds": np.array(EVAL_SEEDS)}
		for fpr in TARGET_FPRS:
			thr = thresholds[fpr]
			scores, lives, interventions = run_pool_eval(pool, "hybrid", thr)
			key = f"fpr{fpr}".replace(".", "_")
			save_data[f"hybrid_{key}_scores"] = scores
			save_data[f"hybrid_{key}_lives"] = lives
			save_data[f"hybrid_{key}_interventions"] = interventions
			save_data[f"hybrid_{key}_threshold"] = thr
			paired_report(f"hybrid @ FPR~{fpr:.1%} (thr={thr:.4f})", scores, lives, interventions, champ_scores)

		np.savez_compressed(SCORES_PATH, **save_data)
		print(f"\nsaved per-seed scores to {SCORES_PATH}")
	finally:
		pool.terminate()
		pool.join()

	print(f"\ntotal runtime {time.time() - t0:.0f}s")
	print(f"detector: train AUC {train_auc:.4f}  held-out AUC {held_auc:.4f}")


if __name__ == "__main__":
	main()
