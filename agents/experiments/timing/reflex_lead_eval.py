"""Paired test of a one-step-early ("lead") reflex against the plain reflex.

The timing oracle (timing_oracle.py) found pressing one decision step earlier than the reflex hits the
multiplier bank ~3x as often with no higher drain rate. The lead reflex fires when the ball's
PREDICTED y one decision step ahead (y + vy * dt_step) crosses the reflex threshold, which moves the
press one step earlier only when the ball is really about to arrive. dt_step (y-units per unit vy per
decision step) is estimated from a reflex life before the test.

Usage: python agents/experiments/timing/reflex_lead_eval.py --seeds 9000:9096 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "encoding"))

BANK = ("a_targ7", "a_targ8", "a_targ9")
POLICIES = ("reflex", "reflex_lead1")
_ENV = None
_BANK_IDS: frozenset = frozenset()


def _init(binary: str, max_steps: int) -> None:
	global _ENV, _BANK_IDS
	import atexit
	import signal

	import gymnasium as gym

	from env_python.pinball_env import PinballEnv

	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=binary, headless=True, frame_skip=4), max_episode_steps=max_steps)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		_BANK_IDS = frozenset(o["id"] for o in json.load(f)["objects"] if o["name"] in BANK)


def _estimate_dt(seed: int) -> float:
	"""Median (y[t+1]-y[t]) / vy[t] over falling steps of a reflex life = y-units per vy-unit per step."""
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	obs, _ = _ENV.reset(seed=seed)
	reflex = ReflexLabeler(11.5, 1)
	ratios = []
	for _ in range(3000):
		a = reflex.label(obs)
		a[2] = 0
		reflex.observe(a)
		nxt, _r, term, trunc, info = _ENV.step(a)
		if obs[OBS_BALL_VY] > 0.5 and 0.0 < nxt[OBS_BALL_Y] - obs[OBS_BALL_Y] < 3.0:
			ratios.append((nxt[OBS_BALL_Y] - obs[OBS_BALL_Y]) / obs[OBS_BALL_VY])
		obs = nxt
		if term or trunc or info["drained"]:
			break
	return float(np.median(ratios))


def _run(task) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y, OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

	name, seed, dt = task
	obs, _ = _ENV.reset(seed=seed)
	reflex = ReflexLabeler(11.5, 1)
	score = presses = contacts = hits = completions = 0
	prev_l, prev_r = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
	done = False
	while not done:
		view = np.array(obs, dtype=np.float32)
		if name == "reflex_lead1" and view[OBS_BALL_VY] > 0:
			view[OBS_BALL_Y] = view[OBS_BALL_Y] + view[OBS_BALL_VY] * dt
		a = reflex.label(view)
		a[2] = 0
		reflex.observe(a)
		obs, _r, term, trunc, info = _ENV.step(a)
		score += info["score_delta"]
		contacts += int(info["flipper_hit"])
		for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
			if oid in _BANK_IDS:
				hits += 1
				completions += int(base >= 1500)
		lu, ru = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
		presses += int(lu and not prev_l) + int(ru and not prev_r)
		prev_l, prev_r = lu, ru
		done = term or trunc or bool(info["drained"])
	return dict(policy=name, seed=seed, score=float(score), presses=presses, contacts=contacts, hits=hits,
	            completions=completions)


def _dt_task(seed: int) -> float:
	return _estimate_dt(seed)


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="9000:9096")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/timing/reflex_lead_9000.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary, args.max_steps)) as pool:
		dt = float(np.median(pool.map(_dt_task, [20500, 20501, 20502, 20503])))
		print(f"dt_step estimate: {dt:.4f} y-units per vy-unit per decision step", flush=True)
		res = pool.map(_run, [(p, s, dt) for s in seeds for p in POLICIES], chunksize=1)
	by = {p: {r["seed"]: r for r in res if r["policy"] == p} for p in POLICIES}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	for p in POLICIES:
		rs = [by[p][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		print(f"{p:<13} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):8.0f}  "
		      + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.2f}" for k in ("hits", "completions", "contacts", "presses")))
	for k in ("log1p", "hits", "completions"):
		if k == "log1p":
			d = np.array([np.log1p(by["reflex_lead1"][s]["score"]) - np.log1p(by["reflex"][s]["score"]) for s in seeds])
		else:
			d = np.array([by["reflex_lead1"][s][k] - by["reflex"][s][k] for s in seeds], dtype=float)
		se = d.std(ddof=1) / math.sqrt(len(d))
		print(f"  lead1 - reflex {k:<12} {d.mean():+.3f} +- {se:.3f}  t {d.mean() / se if se else float('nan'):+.2f}  "
		      f"p {confirm.wilcoxon_p(d):.3g}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), dt=dt, lives=res), fh)


if __name__ == "__main__":
	main()
