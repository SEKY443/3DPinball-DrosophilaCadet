"""Feasibility probe for ANALOG flipper force (engine ACTF frames, a physics change vs the original).

From the same cradle states as cradle_probe.py (both flippers held until the ball rests on one),
replay to the cradle, release for `delay` steps, then shoot the cradle-side flipper with force f.
For each (delay, force) records the ball's speed and travel angle shortly after the flipper contact,
plus ramp / multiplier-bank hits and drains over a short lead1-played horizon.
Questions: does force change the shot at all (speed), is the angle a smooth function of force, and
is it consistent across cradle events (low spread) - i.e. can force aim where timing could not?

Usage: python agents/experiments/cradle/force_probe.py --seeds 24000:24256 --workers 12
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
sys.path.insert(0, HERE)
import cradle_probe as CP  # noqa: E402  (env init, cradle collection, ids)

DELAYS = (1, 3, 5)
FORCES = (0.3, 0.45, 0.6, 0.75, 0.9, 1.0)
ANGLE_AFTER = 3   # decision steps after the flipper contact at which speed/angle are read
HORIZON = 120


def _branch(seed, actions, t, side, delay, force) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY

	env = CP._ENV
	obs, _ = env.reset(seed=seed)
	for k in range(t):
		obs, _r, term, trunc, info = env.step(actions[k])
		if term or trunc or info["drained"]:
			return dict(invalid=1)
	plan = [np.array([0, 0, 0, 1.0, 1.0]) for _ in range(delay)]
	for _ in range(CP.PULSE_STEPS):
		a = np.array([0, 0, 0, 1.0, 1.0])
		a[side] = 1
		a[3 + side] = force
		plan.append(a)
	out = dict(invalid=0, ramp=0, bank=0, drained=0, speed=None, angle=None, contact=0)
	since = None

	def account(info):
		for (oid, _p, _x, _y) in info.get("hit_objects", ()):
			out["ramp"] += int(oid in CP._IDS["ramp"])
			out["bank"] += int(oid in CP._IDS["bank"])
			out["drained"] |= int(oid in CP._IDS["drain"])
		out["drained"] |= int(bool(info["drained"]))

	def track(obs, info):
		nonlocal since
		if info["flipper_hit"] and since is None:
			since = 0
			out["contact"] = 1
		elif since is not None:
			since += 1
			if since == ANGLE_AFTER and out["speed"] is None:
				vx, vy = float(obs[OBS_BALL_VX]), float(obs[OBS_BALL_VY])
				out["speed"] = math.hypot(vx, vy)
				out["angle"] = math.degrees(math.atan2(vx, -vy))  # 0 = straight up the table

	for a in plan:
		obs, _r, term, trunc, info = env.step(a)
		account(info)
		track(obs, info)
		if term or trunc or out["drained"]:
			return out
	reflex = ReflexLabeler(11.5, 1)
	for _ in range(HORIZON):
		a = CP._lead(reflex, obs)
		reflex.observe(a)
		obs, _r, term, trunc, info = env.step(a)
		account(info)
		track(obs, info)
		if term or trunc or out["drained"]:
			break
	return out


def _parse_combos(spec: str):
	"""'L:5:0.45,R:5:0.75' -> {0: [(5, 0.45)], 1: [(5, 0.75)]} (side 0 = LEFT flipper)."""
	out = {0: [], 1: []}
	for item in spec.split(","):
		side, d, f = item.split(":")
		out[0 if side.upper() == "L" else 1].append((int(d), float(f)))
	return out


def _life_task(task) -> dict:
	seed, max_steps, combos = task
	actions, events = CP._cradle_life(seed, max_steps)
	rows = []
	for t, side, x, y in events:
		pairs = combos[side] if combos else [(d, f) for d in DELAYS for f in FORCES]
		grid = {f"{d}_{f}": _branch(seed, actions, t, side, d, f) for d, f in pairs}
		if any(b["invalid"] for b in grid.values()):
			continue
		rows.append(dict(seed=seed, side=side, x=x, y=y, grid=grid))
	return dict(seed=seed, rows=rows)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(CP.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="24000:24256")
	ap.add_argument("--workers", type=int, default=12)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--combos", default="", help="confirmation mode: only these side:delay:force combos, e.g. L:5:0.45,R:5:0.75")
	ap.add_argument("--out", default=os.path.join(HERE, "force_probe.json"))
	args = ap.parse_args()
	combos = _parse_combos(args.combos) if args.combos else None
	lo, hi = (int(v) for v in args.seeds.split(":"))
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=CP._init, initargs=(args.binary,)) as pool:
		res = pool.map(_life_task, [(s, args.max_steps, combos) for s in range(lo, hi)], chunksize=1)
	rows = [r for x in res for r in x["rows"]]
	print(f"{hi - lo} lives, {len(rows)} cradle events  {time.time() - t0:.0f}s")
	for side, name in ((0, "LEFT cradle"), (1, "RIGHT cradle")):
		rs = [r for r in rows if r["side"] == side]
		print(f"\n{name} (n={len(rs)})")
		print(f"{'delay':>5} {'force':>5} {'contact':>7} {'speed':>6} {'angle':>6} {'ang sd':>6} {'ramp':>5} {'bank':>5} {'drain':>5}")
		pairs = combos[side] if combos else [(d, f) for d in DELAYS for f in FORCES]
		for d, f in pairs:
				b = [r["grid"][f"{d}_{f}"] for r in rs]
				sp = [x["speed"] for x in b if x["speed"] is not None]
				an = [x["angle"] for x in b if x["angle"] is not None]
				print(f"{d:>5} {f:>5.2f} {np.mean([x['contact'] for x in b]):7.2f} "
				      f"{(np.mean(sp) if sp else float('nan')):6.1f} {(np.median(an) if an else float('nan')):6.0f} "
				      f"{(np.std(an) if an else float('nan')):6.0f} {np.mean([x['ramp'] > 0 for x in b]):5.2f} "
				      f"{np.mean([x['bank'] > 0 for x in b]):5.2f} {np.mean([x['drained'] for x in b]):5.2f}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
