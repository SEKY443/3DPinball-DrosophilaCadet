#!/usr/bin/env python3
"""Measures press-edge behavior (how often a policy presses, and how often those presses
actually land near a real flipper_hit) for a CMA champion checkpoint and a scripted "flail"
reference, to calibrate --press-cost in agents/train_pinball_circuit_cem.py.

Plays full ball lives (stopping at info["drained"], same as the trainer's "life" objective) over
seeds 7000-7047 - deliberately NOT 4000-4095 or 5000-5287, which this project reserves as
held-out test sets elsewhere.

Usage:
	python agents/experiments/press_cost/measure_presses.py \\
		--binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
		--connectome agents/experiments/big_circuit/connectome.json \\
		--ckpt agents/experiments/big_circuit/best_heldout_gen280.pt
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor", "nfly"))

from agents.drill import build_policy  # noqa: E402
from env_python.pinball_env import OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT  # noqa: E402

# Measurement seeds - see module docstring for why not 4000-4095/5000-5287.
SEED_LO, SEED_HI = 7000, 7047  # inclusive, 48 seeds
HELD_OUT_RANGES = [(4000, 4095), (5000, 5287)]

# A press edge counts as "coincident" with a real flipper_hit if a hit lands on the same step or
# either of the next 2 steps - close enough in time that the press plausibly caused it, rather
# than being spammed with no ball anywhere near the flipper.
HIT_COINCIDENCE_WINDOW = 2

_WORKER_ENV = None
_WORKER_POLICY = None


def _worker_init(binary_path: str, connectome_path: str, policy_spec: str, frame_skip: int, max_steps: int) -> None:
	global _WORKER_ENV, _WORKER_POLICY
	import atexit
	import signal

	import gymnasium as gym
	import torch

	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)  # see train_pinball_circuit_cem.py's _worker_init for why

	env = PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip)
	env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
	_WORKER_ENV = env
	_WORKER_POLICY = build_policy(policy_spec, connectome_path, env)

	def _on_terminate(signum, frame):
		sys.exit(0)

	signal.signal(signal.SIGTERM, _on_terminate)
	atexit.register(env.close)


def _play_life(seed: int) -> dict:
	env, policy_fn = _WORKER_ENV, _WORKER_POLICY
	obs, info = env.reset(seed=seed)
	if hasattr(policy_fn, "reset"):
		policy_fn.reset()

	prev_left = obs[OBS_FLIPPER_LEFT] > 0.5
	prev_right = obs[OBS_FLIPPER_RIGHT] > 0.5
	press_left = press_right = 0
	edge_steps: list[int] = []
	hit_steps: list[int] = []
	score = 0.0
	t = 0
	while True:
		left, right = policy_fn(obs, t)
		obs, _reward, terminated, truncated, info = env.step([bool(left), bool(right), 0])
		score += info["score_delta"]
		left_up = obs[OBS_FLIPPER_LEFT] > 0.5
		right_up = obs[OBS_FLIPPER_RIGHT] > 0.5
		if left_up and not prev_left:
			press_left += 1
			edge_steps.append(t)
		if right_up and not prev_right:
			press_right += 1
			edge_steps.append(t)
		prev_left, prev_right = left_up, right_up
		if info["flipper_hit"]:
			hit_steps.append(t)
		t += 1
		if info["drained"] or terminated or truncated:
			break

	hit_set = set(hit_steps)
	coincident = sum(1 for e in edge_steps if any((e + k) in hit_set for k in range(HIT_COINCIDENCE_WINDOW + 1)))
	steps = t
	return dict(
		seed=seed, score=float(score), steps=steps, press_left=press_left, press_right=press_right,
		press_total=press_left + press_right, coincident=coincident,
	)


def _run_policy(binary_path: str, connectome_path: str, policy_spec: str, frame_skip: int, max_steps: int,
                workers: int, seeds: list[int]) -> list[dict]:
	ctx = mp.get_context("spawn")
	pool = ctx.Pool(
		processes=workers, initializer=_worker_init,
		initargs=(binary_path, connectome_path, policy_spec, frame_skip, max_steps),
	)
	try:
		return pool.map(_play_life, seeds)
	finally:
		pool.terminate()
		pool.join()


def _summarize(label: str, records: list[dict]) -> dict:
	score = np.array([r["score"] for r in records], dtype=np.float64)
	steps = np.array([r["steps"] for r in records], dtype=np.float64)
	presses = np.array([r["press_total"] for r in records], dtype=np.float64)
	left = np.array([r["press_left"] for r in records], dtype=np.float64)
	right = np.array([r["press_right"] for r in records], dtype=np.float64)
	coincident = np.array([r["coincident"] for r in records], dtype=np.float64)
	per_100 = presses / steps * 100.0
	total_edges = presses.sum()
	coincidence_frac = coincident.sum() / total_edges if total_edges > 0 else float("nan")
	log_score = np.log1p(np.maximum(0.0, score))

	print(f"\n{label} (n={len(records)} lives)")
	print(f"  score            median {np.median(score):10.1f}  mean {score.mean():10.1f}  std {score.std():10.1f}")
	print(f"  log1p(score)     median {np.median(log_score):10.3f}  mean {log_score.mean():10.3f}  std {log_score.std():10.3f}")
	print(f"  life steps       median {np.median(steps):10.1f}  mean {steps.mean():10.1f}")
	print(f"  press edges (L)  median {np.median(left):10.1f}  mean {left.mean():10.2f}")
	print(f"  press edges (R)  median {np.median(right):10.1f}  mean {right.mean():10.2f}")
	print(f"  press edges tot  median {np.median(presses):10.1f}  mean {presses.mean():10.2f}")
	print(f"  presses/100step  median {np.median(per_100):10.3f}  mean {per_100.mean():10.3f}")
	print(f"  hit-coincidence fraction (of all press edges): {coincidence_frac:.3f}")

	return dict(
		median_presses=float(np.median(presses)), mean_presses=float(presses.mean()),
		median_log_score=float(np.median(log_score)), mean_log_score=float(log_score.mean()),
	)


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", default=os.environ.get("PINBALL_BINARY"))
	parser.add_argument("--connectome", required=True, help="connectome for the ckpt policy (unused by flail)")
	parser.add_argument("--ckpt", required=True, help="train_pinball_circuit_cem.py checkpoint to measure")
	parser.add_argument("--frame-skip", type=int, default=4)
	parser.add_argument("--max-steps", type=int, default=3000)
	parser.add_argument("--workers", type=int, default=8)
	args = parser.parse_args()

	if not args.binary:
		raise SystemExit("--binary not given and PINBALL_BINARY is not set")

	seeds = list(range(SEED_LO, SEED_HI + 1))
	assert len(seeds) == 48, f"expected 48 seeds, got {len(seeds)}"
	for lo, hi in HELD_OUT_RANGES:
		assert not (lo <= SEED_LO <= hi or lo <= SEED_HI <= hi), f"seed range overlaps held-out [{lo}, {hi}]"

	champion_records = _run_policy(
		args.binary, args.connectome, f"ckpt:{args.ckpt}", args.frame_skip, args.max_steps, args.workers, seeds
	)
	champion_stats = _summarize("champion", champion_records)

	flail_records = _run_policy(
		args.binary, args.connectome, "flail", args.frame_skip, args.max_steps, args.workers, seeds
	)
	_summarize("flail (scripted reference)", flail_records)

	# --press-cost calibration: pick values so the champion's MEDIAN life pays roughly these total
	# log-units of press cost - small relative to the champion's own run-to-run fitness variation
	# (measured "a few tenths" per the trainer's docstring) so a low setting barely perturbs a
	# genuinely efficient policy, while the top setting is comparable to that variation and should
	# start visibly discouraging pure spam.
	median_presses = champion_stats["median_presses"]
	print(f"\ncalibration: champion median presses/life = {median_presses:.1f}, "
	      f"median log1p(score/life) = {champion_stats['median_log_score']:.3f}")
	for target in (0.1, 0.3, 0.6):
		press_cost = target / median_presses if median_presses > 0 else float("nan")
		print(f"  target {target:.1f} log units over {median_presses:.1f} presses -> "
		      f"--press-cost {press_cost:.5f}  ({press_cost:.5f} * {median_presses:.1f} = {press_cost * median_presses:.3f})")


if __name__ == "__main__":
	main()
