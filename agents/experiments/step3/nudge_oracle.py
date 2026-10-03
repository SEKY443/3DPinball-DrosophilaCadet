#!/usr/bin/env python3
"""Drain anatomy under the pulse reflex + nudge-save oracle (step 3, nudge copy).

Base policy: correct-side pulse reflex (ReflexLabeler y>11.5). Per life (one ball life, trainer's
`life` episode):
  - drain class: 'outlane' if an outlane rollover (a_roll4 id 29 / a_roll8 id 30) was hit within
    the last 60 decision steps before the drain, else 'center' if |x| < 1.5 just before the drain,
    else 'side'; 'alive' if the life hit --max-steps;
  - whether the ball's final approach (from the last step it was above y < 8 until the drain)
    contained a flipper contact.
Nudge oracle, for every outlane death: replay the identical life up to step s = t_out - k
(k in --lead-steps), hold nudge code c (1 +x, 2 -x, 3 bottom) for h decision steps, then continue
with the reflex. A branch SAVES the ball if it does not drain before t_drain + --save-horizon (or
the life reaches --max-steps) and the table never tilted. Reported: fraction of outlane deaths
with at least one saving branch (Wilson 95% CI), and which (k, c, h) save most often.

Usage: python agents/experiments/step3/nudge_oracle.py --episodes 48 --seed0 7024 --workers 2
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

OUTLANE_IDS = {29, 30}
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


def _run(seed: int, prefix: list, nudge_from: int | None, code: int, hold: int, stop_at: int | None) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_X, OBS_BALL_Y

	lab = ReflexLabeler(_CFG["y"], 1)
	obs, _ = _ENV.reset(seed=seed)
	recs, t, done, tilted = [], 0, False, False
	while not done:
		if t < len(prefix):
			a = prefix[t].copy()
		else:
			a = lab.label(obs)
			a[2] = 0
		lab.observe(a)
		a4 = np.array([a[0], a[1], 0, code if (nudge_from is not None and nudge_from <= t < nudge_from + hold) else 0])
		obs, _r, term, trunc, info = _ENV.step(a4)
		tilted |= bool(info["tilted"])
		recs.append(dict(a=a.copy(), x=float(obs[OBS_BALL_X]), y=float(obs[OBS_BALL_Y]), hit=bool(info["flipper_hit"]),
		                 ids=[h[0] for h in info["hit_objects"]], drained=bool(info["drained"])))
		t += 1
		done = term or trunc or info["drained"] or (stop_at is not None and t >= stop_at)
	return dict(recs=recs, drained=recs[-1]["drained"] if recs else False, t=t, tilted=tilted)


def _life(seed: int) -> dict:
	base = _run(seed, [], None, 0, 0, None)
	recs = base["recs"]
	out = dict(seed=seed, steps=base["t"], drain="alive", final_contact=None, outlane_step=None, saves=[], n_branches=0, branches=[])
	if not base["drained"]:
		return out
	# The env's `drained` flag fires only once the ball is re-fed to the plunger, which can be
	# 100+ decision steps after the ball actually fell into the drain - use the drain component's
	# own hit (table_map id 80, TDrain) as the drain time when it was reported.
	drain_hits = [t for t, r in enumerate(recs) if 80 in r["ids"]]
	td = drain_hits[-1] if drain_hits else base["t"] - 1
	out["drain_lag"] = base["t"] - 1 - td
	t_out = None
	for t in range(max(0, td - 60), td + 1):
		if OUTLANE_IDS & set(recs[t]["ids"]):
			t_out = t
	# final approach: from the last step the ball was above y < 8 until the drain
	last_up = max([t for t in range(td + 1) if recs[t]["y"] < 8.0], default=0)
	out["final_contact"] = any(recs[t]["hit"] for t in range(last_up, td + 1))
	if t_out is not None:
		out["drain"] = "outlane"
	else:
		x_before = recs[td - 1]["x"] if td >= 1 else recs[td]["x"]
		out["drain"] = "center" if abs(x_before) < 1.5 else "side"
	if t_out is None or not _CFG["oracle"]:
		return out
	out["outlane_step"] = t_out
	prefix = [r["a"] for r in recs]
	for k in _CFG["lead_steps"]:
		s = t_out - k
		if s < 1:
			continue
		for code in (1, 2, 3):
			for hold in _CFG["holds"]:
				r = _run(seed, prefix[:s], s, code, hold, base["t"] + _CFG["save_horizon"])
				out["n_branches"] += 1
				avoided = not any(OUTLANE_IDS & set(rr["ids"]) for rr in r["recs"][s:t_out + 15])
				out["branches"].append((k, code, hold, bool(r["drained"]), bool(r["tilted"]), avoided))
				if not r["drained"] and not r["tilted"]:
					out["saves"].append((k, code, hold, avoided))
	return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
	if n == 0:
		return (float("nan"), float("nan"))
	p = k / n
	den = 1 + z * z / n
	c = (p + z * z / (2 * n)) / den
	h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
	return (round(c - h, 3), round(c + h, 3))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--episodes", type=int, default=48)
	ap.add_argument("--seed0", type=int, default=7024)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--y", type=float, default=11.5)
	ap.add_argument("--lead-steps", default="30,20,12,6,3")
	ap.add_argument("--holds", default="2,4,6")
	ap.add_argument("--save-horizon", type=int, default=200)
	ap.add_argument("--no-oracle", action="store_true")
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/step3/nudge_oracle.json"))
	args = ap.parse_args()
	if 5000 <= args.seed0 < 6000:
		sys.exit("seeds 5000+ are the held-out confirmation set - use 7000+")
	cfg = dict(binary=args.binary, max_steps=args.max_steps, y=args.y, save_horizon=args.save_horizon,
	           lead_steps=[int(v) for v in args.lead_steps.split(",")], holds=[int(v) for v in args.holds.split(",")],
	           oracle=not args.no_oracle)
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
		res = pool.map(_life, range(args.seed0, args.seed0 + args.episodes), chunksize=1)
	cls = Counter(r["drain"] for r in res)
	center = [r for r in res if r["drain"] == "center"]
	print(f"{len(res)} lives, {time.time() - t0:.0f}s  drain classes: {dict(cls)}")
	print(f"center drains with NO flipper contact in the final approach: "
	      f"{sum(1 for r in center if not r['final_contact'])}/{len(center)}")
	for c in ("outlane", "side"):
		rr = [r for r in res if r["drain"] == c]
		print(f"{c} drains with no flipper contact in the final approach: {sum(1 for r in rr if not r['final_contact'])}/{len(rr)}")
	out_deaths = [r for r in res if r["drain"] == "outlane"]
	if out_deaths and not args.no_oracle:
		saved = sum(1 for r in out_deaths if r["saves"])
		print(f"nudge oracle: {saved}/{len(out_deaths)} outlane deaths have >=1 saving branch "
		      f"(Wilson 95% CI {_wilson(saved, len(out_deaths))}); branches per death {out_deaths[0]['n_branches']}")
		ctr = Counter(tuple(s[:3]) for r in out_deaths for s in r["saves"])
		print("most frequent saving (lead_steps, code, hold):", ctr.most_common(8))
		av = sum(1 for r in out_deaths if any(sv[3] for sv in r["saves"]))
		print(f"  of which the save avoided the outlane itself (vs. only delaying the drain): {av}/{len(out_deaths)}")
		tilts = sum(1 for r in out_deaths for b in r["branches"] if b[4])
		print(f"  branches that tilted: {tilts}")
		per_branch = sum(len(r["saves"]) for r in out_deaths) / sum(r["n_branches"] for r in out_deaths)
		print(f"fraction of all branches that save: {per_branch:.3f}")
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(dict(args=vars(args), lives=res), f)


if __name__ == "__main__":
	main()
