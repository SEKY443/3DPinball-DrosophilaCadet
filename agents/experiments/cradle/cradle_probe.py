"""Feasibility probe for "cradle and shoot" aiming.

Real players aim by trapping the ball on a raised flipper, letting it settle, then releasing and
re-pressing so the shot starts from a near-identical, low-chaos state. Two questions:
  1. Can a held flipper trap the ball? A cradle policy plays lives: lead1 reflex timing, but the
     pressed flipper is HELD until the ball either settles on it (speed below SETTLE_SPEED for
     SETTLE_STEPS decision steps -> a cradle event) or bounces away (y < RELEASE_Y).
  2. From a cradle, is the shot controllable? For each cradle event, replay to the cradle step
     (the engine is deterministic given seed + actions), release the flipper, wait r steps while the
     ball rolls down the flipper, then pulse it; after that the lead1 reflex plays HORIZON steps.
     Record ramp hits, multiplier-bank hits, upper-playfield base points and drains per r.
If some release delay hits the ramp or the bank far more often than the others, aiming from a cradle
is feasible and a cradle teacher is worth building.

Usage: python agents/experiments/cradle/cradle_probe.py --seeds 24000:24064 --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

DT_STEP = 0.0347
SETTLE_SPEED = 0.6      # ball speed (table units/s) below which it counts as resting
SETTLE_STEPS = 6        # consecutive slow steps on a held flipper -> cradle event
RELEASE_Y = 9.0         # ball bounced back up the table -> stop holding
FLIPPER_ZONE = (10.0, 3.5)  # y above, |x| below: ball is over the flippers
RELEASE_DELAYS = (1, 2, 3, 4, 5, 6, 8, 10)  # steps released before the shot (the flipper is up at the cradle)
PULSE_STEPS = 3
HORIZON = 120
MAX_EVENTS_PER_LIFE = 6
_ENV = None
_IDS: dict = {}


def _init(binary: str) -> None:
	global _ENV
	import atexit
	import signal

	from env_python.pinball_env import PinballEnv

	_ENV = PinballEnv(binary_path=binary, headless=True, frame_skip=4)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		objs = json.load(f)["objects"]
	_IDS["ramp"] = frozenset(o["id"] for o in objs if o["name"] == "ramp")
	_IDS["bank"] = frozenset(o["id"] for o in objs if o["name"] in ("a_targ7", "a_targ8", "a_targ9"))
	_IDS["drain"] = frozenset(o["id"] for o in objs if o["name"] == "drain")
	_IDS["upper"] = frozenset(o["id"] for o in objs if o["center"] is not None and o["center"][1] < 6.0
	                          and o["group"] not in ("drain", "flipper", "plunger"))


def _lead(reflex, obs) -> np.ndarray:
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	view = np.array(obs, dtype=np.float32)
	if view[OBS_BALL_VY] > 0:
		view[OBS_BALL_Y] += view[OBS_BALL_VY] * DT_STEP
	a = reflex.label(view)
	a[2] = 0
	return a


def _cradle_life(seed: int, max_steps: int):
	"""Cradle collection: hold BOTH flippers up from the start so the ball lands on a raised flipper
	and rolls to rest at its base (measured resting spot about (+-2.5, 11.3)). The first rest of at
	least SETTLE_STEPS slow steps over the flippers is the cradle event; the life stops there (a held
	ball would rest forever). Returns (actions, events [(t, side, x, y)]); no event = ball drained."""
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	obs, _ = _ENV.reset(seed=seed)
	actions, events = [], []
	slow = 0
	for t in range(max_steps):
		x, y = float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y])
		speed = math.hypot(float(obs[OBS_BALL_VX]), float(obs[OBS_BALL_VY]))
		on_flipper = FLIPPER_ZONE[0] < y < 12.8 and abs(x) < FLIPPER_ZONE[1]
		slow = slow + 1 if (on_flipper and speed < SETTLE_SPEED) else 0
		if slow == SETTLE_STEPS:
			events.append((t, 0 if x > 0 else 1, x, y))  # x > 0 is the LEFT flipper (mirrored table)
			break
		a = np.array([1, 1, 0], dtype=np.int64)
		actions.append(a)
		obs, _r, term, trunc, info = _ENV.step(a)
		if term or trunc or info["drained"]:
			break
	return actions, events


def _branch(seed: int, actions, t: int, side: int, delay: int) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler

	obs, _ = _ENV.reset(seed=seed)
	for k in range(t):
		obs, _r, term, trunc, info = _ENV.step(actions[k])
		if term or trunc or info["drained"]:
			return dict(invalid=1)
	plan = [np.zeros(3, dtype=np.int64) for _ in range(delay)]
	for _ in range(PULSE_STEPS):
		a = np.zeros(3, dtype=np.int64)
		a[side] = 1
		plan.append(a)
	out = dict(ramp=0, bank=0, upper=0.0, drained=0, first=None, invalid=0)
	reflex = ReflexLabeler(11.5, 1)

	def account(info):
		for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
			if oid in _IDS["upper"] and out["first"] is None:
				out["first"] = int(oid)
			out["ramp"] += int(oid in _IDS["ramp"])
			out["bank"] += int(oid in _IDS["bank"])
			out["upper"] += base if oid in _IDS["upper"] else 0
			out["drained"] |= int(oid in _IDS["drain"])
		out["drained"] |= int(bool(info["drained"]))

	for a in plan:
		obs, _r, term, trunc, info = _ENV.step(a)
		account(info)
		if term or trunc or out["drained"]:
			return out
	for _ in range(HORIZON):
		a = _lead(reflex, obs)
		reflex.observe(a)
		obs, _r, term, trunc, info = _ENV.step(a)
		account(info)
		if term or trunc or out["drained"]:
			break
	return out


def _life_task(task) -> dict:
	seed, max_steps = task
	actions, events = _cradle_life(seed, max_steps)
	rows = []
	for t, side, x, y in events:
		branches = [_branch(seed, actions, t, side, d) for d in RELEASE_DELAYS]
		if any(b["invalid"] for b in branches):
			continue
		rows.append(dict(seed=seed, t=t, side=side, x=x, y=y, branches=branches))
	return dict(seed=seed, steps=len(actions), events=len(events), rows=rows)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="24000:24064")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(HERE, "cradle_probe.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		res = pool.map(_life_task, [(s, args.max_steps) for s in range(lo, hi)], chunksize=1)
	rows = [r for x in res for r in x["rows"]]
	print(f"{hi - lo} lives, {sum(x['events'] for x in res)} cradle events ({np.mean([x['events'] for x in res]):.2f}/life), "
	      f"{len(rows)} usable, mean life steps {np.mean([x['steps'] for x in res]):.0f}  {time.time() - t0:.0f}s")
	if rows:
		print("side split: left %d, right %d" % (sum(r["side"] == 0 for r in rows), sum(r["side"] == 1 for r in rows)))
		print(f"{'delay':>5} {'ramp':>6} {'bank':>6} {'upper':>8} {'drain':>6}")
		for i, d in enumerate(RELEASE_DELAYS):
			b = [r["branches"][i] for r in rows]
			print(f"{d:>5} {np.mean([x['ramp'] > 0 for x in b]):6.2f} {np.mean([x['bank'] > 0 for x in b]):6.2f} "
			      f"{np.mean([x['upper'] for x in b]):8.0f} {np.mean([x['drained'] for x in b]):6.2f}")
		best_ramp = np.mean([max(br["ramp"] > 0 for br in r["branches"]) for r in rows])
		best_bank = np.mean([max(br["bank"] > 0 for br in r["branches"]) for r in rows])
		print(f"events where SOME delay hits the ramp: {best_ramp:.2f}; the bank: {best_bank:.2f}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=[{k: v for k, v in x.items()} for x in res]), fh)


if __name__ == "__main__":
	main()
