#!/usr/bin/env python3
"""Builds a bank of real-game-sampled flipper-drill start states, for agents/drill.py's --bank
and train_pinball_circuit_cem.py's --drill-bank.

Rationale: drill.py's make_drills() synthesizes start states from hand-picked uniform ranges.
Training a policy against those raised its drill save rate but did not improve real full-game
play - most likely because synthetic drills don't look like how a ball actually approaches the
flippers in real games. This script instead plays real ball lives with a chosen policy and
records the observation (x, y, vx, vy) each time the ball genuinely approaches the flipper zone,
so drills sampled from the resulting bank match real approach geometry/velocity.

An "approach" is: the ball descends through y = 6.0 (prev obs y < 6.0 <= this obs y, with
vy > 0 - see pinball_env.py's "+vy falls toward the flippers" convention), and within the
following 60 steps actually reaches the flipper zone (y > 10.0, |x| < 4.0) before retreating back
above y < 3.0 - this filters out y=6.0 crossings that never turn into a real flipper encounter
(e.g. a shallow bounce off a bumper that falls back toward the top of the table).

Usage:
	python agents/collect_drill_bank.py --policy ckpt:agents/experiments/.../champion.pt \\
		--connectome web/connectome.json --seeds 400 --seed0 30000 --out bank.npz
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.drill import DRILL_VX_RANGE, DRILL_VY_RANGE, DRILL_X_RANGE, DRILL_Y_RANGE, build_policy  # noqa: E402
from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y  # noqa: E402

# Ball must descend through this y before an approach is even considered (see module docstring).
APPROACH_Y = 6.0
# Flipper zone an approach must actually reach to count as real (not a shallow bounce back up).
FLIPPER_ZONE_Y = 10.0
FLIPPER_ZONE_X = 4.0
RETREAT_Y = 3.0
# Steps (post frame_skip) to look ahead from the y=6.0 crossing for flipper-zone entry/flipper_hit.
APPROACH_WINDOW = 60
# Steps to look ahead for the drained metadata field - longer, since a save can still later drain.
DRAINED_WINDOW = 120

# PLC! placement bounds, mirrored from src_cpp/ipc_server.cpp's IsValidPlacement/kPlace* constants.
PLACE_X_RANGE = (-12.0, 12.0)
PLACE_Y_RANGE = (-14.0, 20.0)
PLACE_MAX_SPEED = 200.0
# Skip the plunger lane (rest position x=-7.02, see pinball_env.py's PLUNGER_REST_X) - only the
# open table is a valid flipper-drill start.
OPEN_TABLE_MAX_X = 6.5

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


def _is_valid_placement(x: float, y: float, vx: float, vy: float) -> bool:
	return (
		PLACE_X_RANGE[0] <= x <= PLACE_X_RANGE[1]
		and PLACE_Y_RANGE[0] <= y <= PLACE_Y_RANGE[1]
		and abs(vx) <= PLACE_MAX_SPEED
		and abs(vy) <= PLACE_MAX_SPEED
	)


def find_approaches(trace: list, seed: int) -> list[dict]:
	"""Scans one life's (obs, info) trace for approach events - see module docstring."""
	approaches = []
	n = len(trace)
	for i in range(1, n):
		prev_obs, _ = trace[i - 1]
		obs, _info = trace[i]
		y_prev, y = prev_obs[OBS_BALL_Y], obs[OBS_BALL_Y]
		vy = obs[OBS_BALL_VY]
		if not (y_prev < APPROACH_Y <= y and vy > 0):
			continue
		x, vx = obs[OBS_BALL_X], obs[OBS_BALL_VX]
		if abs(x) > OPEN_TABLE_MAX_X or not _is_valid_placement(x, y, vx, vy):
			continue

		window_end = min(i + 1 + APPROACH_WINDOW, n)
		entered_zone = False
		for j in range(i + 1, window_end):
			oy, ox = trace[j][0][OBS_BALL_Y], trace[j][0][OBS_BALL_X]
			if oy < RETREAT_Y:
				break
			if oy > FLIPPER_ZONE_Y and abs(ox) < FLIPPER_ZONE_X:
				entered_zone = True
				break
		if not entered_zone:
			continue

		hit = any(trace[j][1].get("flipper_hit", False) for j in range(i, window_end))
		drained_end = min(i + 1 + DRAINED_WINDOW, n)
		drained = any(trace[j][1].get("drained", False) for j in range(i, drained_end))
		approaches.append(dict(
			seed=seed, step=i, x=float(x), y=float(y), vx=float(vx), vy=float(vy),
			flipper_hit=bool(hit), drained=bool(drained),
		))
	return approaches


def _play_life(seed: int) -> list[dict]:
	env, policy_fn = _WORKER_ENV, _WORKER_POLICY
	obs, info = env.reset(seed=seed)
	if hasattr(policy_fn, "reset"):
		policy_fn.reset()

	trace = [(obs, info)]
	t = 0
	while True:
		left, right = policy_fn(obs, t)
		obs, _reward, terminated, truncated, info = env.step([bool(left), bool(right), 0])
		trace.append((obs, info))
		t += 1
		if info["drained"] or terminated or truncated:
			break

	return find_approaches(trace, seed)


def _print_stats(approaches: list[dict], n_lives: int) -> None:
	n = len(approaches)
	print(f"lives={n_lives}  approaches={n}  per_life_avg={n / n_lives if n_lives else float('nan'):.2f}")
	if n == 0:
		return
	arr = {k: np.array([a[k] for a in approaches], dtype=np.float64) for k in ("x", "y", "vx", "vy")}
	synthetic = dict(x=DRILL_X_RANGE, y=DRILL_Y_RANGE, vx=DRILL_VX_RANGE, vy=DRILL_VY_RANGE)
	print("field    mean     std   | synthetic range")
	for k in ("x", "y", "vx", "vy"):
		lo, hi = synthetic[k]
		print(f"{k:>4}  {arr[k].mean():7.3f} {arr[k].std():7.3f}  | [{lo}, {hi}]")
	print(f"flipper_hit rate={np.mean([a['flipper_hit'] for a in approaches]):.3f}  "
	      f"drained rate={np.mean([a['drained'] for a in approaches]):.3f}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", default=os.environ.get("PINBALL_BINARY"))
	parser.add_argument("--policy", required=True, help="none | flail | ckpt:<path>")
	parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "..", "web", "connectome.json"))
	parser.add_argument("--seeds", type=int, required=True, help="number of ball lives (seeds) to play")
	parser.add_argument("--seed0", type=int, default=0, help="first seed; seeds used are seed0..seed0+seeds-1")
	parser.add_argument("--out", required=True)
	parser.add_argument("--frame-skip", type=int, default=4)
	parser.add_argument("--max-steps", type=int, default=3000)
	parser.add_argument("--workers", type=int, default=8)
	args = parser.parse_args()

	if not args.binary:
		raise SystemExit("--binary not given and PINBALL_BINARY is not set")

	seeds = list(range(args.seed0, args.seed0 + args.seeds))
	ctx = mp.get_context("spawn")
	pool = ctx.Pool(
		processes=args.workers, initializer=_worker_init,
		initargs=(args.binary, args.connectome, args.policy, args.frame_skip, args.max_steps),
	)
	try:
		per_life = pool.map(_play_life, seeds)
	finally:
		pool.terminate()
		pool.join()

	approaches = [a for life in per_life for a in life]
	_print_stats(approaches, len(seeds))

	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	np.savez(
		args.out,
		seed=np.array([a["seed"] for a in approaches], dtype=np.int64),
		step=np.array([a["step"] for a in approaches], dtype=np.int64),
		x=np.array([a["x"] for a in approaches], dtype=np.float32),
		y=np.array([a["y"] for a in approaches], dtype=np.float32),
		vx=np.array([a["vx"] for a in approaches], dtype=np.float32),
		vy=np.array([a["vy"] for a in approaches], dtype=np.float32),
		flipper_hit=np.array([a["flipper_hit"] for a in approaches], dtype=bool),
		drained=np.array([a["drained"] for a in approaches], dtype=bool),
	)
	print(f"saved {len(approaches)} approaches to {args.out}")


if __name__ == "__main__":
	main()
