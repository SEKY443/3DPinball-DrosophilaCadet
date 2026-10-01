#!/usr/bin/env python3
"""Ceiling test for timing-only policies: a per-approach lookahead oracle (step 3).

Base policy: the correct-side pulse reflex (ReflexLabeler y>11.5). Each time the base policy is
about to press ("approach" = the reflex fires), the oracle branches over K press timings
(offsets --offsets, in decision steps, relative to the reflex's own press step, same flipper),
replays each branch from reset (the engine is deterministic per seed + action sequence) and runs
the base reflex after the forced press until the NEXT approach, a drain, or --horizon steps.
Branch ranking: not drained first, then points gained. The best branch's actions up to its press
+ cooldown are committed and the process repeats. The life's final score is an upper-bound-style
estimate for any policy that only chooses WHEN to press (greedy, one approach ahead).

Reports, per life and in aggregate: base reflex score/steps vs oracle score/steps, log1p means.

Usage: python agents/experiments/step3/lookahead_oracle.py --episodes 8 --seed0 7024 --workers 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

_ENV = None
_CFG: dict = {}


def _init(cfg: dict) -> None:
	global _ENV
	import atexit
	import signal

	import gymnasium as gym

	from env_python.pinball_env import PinballEnv

	_CFG.update(cfg)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=cfg["binary"], headless=True, frame_skip=4),
	                              max_episode_steps=cfg["max_steps"])
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)


def _rollout(seed: int, prefix: list, forced: dict | None, stop_after_trigger_from: int | None, horizon_end: int | None):
	"""Replays `prefix` actions, then runs the reflex (with `forced` step->action overrides).
	Stops at the first reflex trigger at step >= stop_after_trigger_from (returned as `trigger`,
	that step's action NOT executed), at drain/end, or at horizon_end. Returns dict."""
	from agents.train_pinball_circuit_cem import ReflexLabeler

	lab = ReflexLabeler(_CFG["y"], 1)
	obs, _ = _ENV.reset(seed=seed)
	actions, score, t, trigger, drained, done = [], 0, 0, None, False, False
	while not done:
		if t < len(prefix):
			a = prefix[t]
		elif forced is not None and t in forced:
			a = forced[t]
		else:
			a = lab.label(obs)
			a[2] = 0
			if a[:2].any() and stop_after_trigger_from is not None and t >= stop_after_trigger_from:
				trigger = (t, int(np.argmax(a[:2])))
				break
		if horizon_end is not None and t >= horizon_end:
			break
		lab.observe(a)
		obs, _r, term, trunc, info = _ENV.step(a)
		actions.append(np.asarray(a).copy())
		score += info["score_delta"]
		t += 1
		drained = bool(info["drained"])
		done = term or trunc or drained
	return dict(actions=actions, score=score, t=t, trigger=trigger, drained=drained, ended=done)


def _life(seed: int) -> dict:
	base = _rollout(seed, [], None, None, None)
	committed: list = []
	n_approaches = 0
	while True:
		probe = _rollout(seed, committed, None, len(committed), None)
		if probe["trigger"] is None:  # life ended under the base reflex from here on
			final = probe
			break
		t_star, side = probe["trigger"]
		prefix_score = _rollout(seed, committed, None, None, len(committed))["score"] if committed else 0
		best = None
		for off in _CFG["offsets"]:
			tp = t_star + off
			if tp < len(committed):
				continue
			forced = {}
			for t in range(len(committed), tp + 3):
				a = np.zeros(3, dtype=np.int64)
				if t == tp:
					a[side] = 1
				forced[t] = a
			# base reflex actions between committed and the branch press are "no press" (t* is
			# the reflex's FIRST press after `committed`), so forcing zeros there is exact.
			r = _rollout(seed, committed, forced, tp + 3, tp + 3 + _CFG["horizon"])
			key = (not r["drained"], r["score"] - prefix_score)
			if best is None or key > best[0]:
				best = (key, off, r, tp)
		n_approaches += 1
		_key, off, r, tp = best
		committed = [np.asarray(a).copy() for a in r["actions"][:tp + 3]]
		if len(committed) < tp + 3 or n_approaches > _CFG["max_approaches"]:
			final = _rollout(seed, committed, None, None, None)
			break
	return dict(seed=seed, base_score=int(base["score"]), base_steps=base["t"], oracle_score=int(final["score"]),
	            oracle_steps=final["t"], approaches=n_approaches)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--episodes", type=int, default=8)
	ap.add_argument("--seed0", type=int, default=7024)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--offsets", default="-2,-1,0,1,2")
	ap.add_argument("--horizon", type=int, default=400)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--max-approaches", type=int, default=200)
	ap.add_argument("--y", type=float, default=11.5)
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/step3/lookahead_oracle.json"))
	args = ap.parse_args()
	if 5000 <= args.seed0 < 6000:
		sys.exit("seeds 5000+ are the held-out confirmation set - use 7000+")
	cfg = dict(binary=args.binary, max_steps=args.max_steps, y=args.y, horizon=args.horizon,
	           offsets=[int(v) for v in args.offsets.split(",")], max_approaches=args.max_approaches)
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
		res = pool.map(_life, range(args.seed0, args.seed0 + args.episodes), chunksize=1)
	for r in res:
		print(r)
	b = np.log1p([r["base_score"] for r in res])
	o = np.log1p([r["oracle_score"] for r in res])
	d = o - b
	print(f"{len(res)} lives, {time.time() - t0:.0f}s  base reflex log1p {b.mean():.2f}  oracle log1p {o.mean():.2f}  "
	      f"paired diff {d.mean():+.2f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.2f})  "
	      f"steps base {np.mean([r['base_steps'] for r in res]):.0f} oracle {np.mean([r['oracle_steps'] for r in res]):.0f}")
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(dict(args=vars(args), lives=res, base_log1p=float(b.mean()), oracle_log1p=float(o.mean()),
		               paired_diff=float(d.mean())), f)


if __name__ == "__main__":
	main()
