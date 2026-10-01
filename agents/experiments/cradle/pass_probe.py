"""Left-to-right "post pass" probe (analog flippers, ACTF frames).

A LEFT cradle (ball resting on the left flipper, x > 0 on the mirrored table) almost never reaches
the ramp (~3%), while a RIGHT cradle + delay 5 + force 0.75 reaches it ~20%. Can the ball be passed
across? From each left cradle: keep the RIGHT flipper raised, release the LEFT flipper for `delay`
steps (the ball rolls toward its tip), tap it with a weak force `force` for PULSE_STEPS, keep the left
flipper down and the right one raised, and count a success if the ball then comes to rest on the
right flipper (x < 0 over the flippers, speed < SETTLE_SPEED for SETTLE_STEPS) within WAIT steps.
Failures are split into drained, bounced back up the table, and timed out.

Usage: python agents/experiments/cradle/pass_probe.py --seeds 28000:28512 --workers 16
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
import cradle_probe as CP  # noqa: E402

DELAYS = (14, 15, 16, 17, 18)  # the released ball needs ~28 steps to roll to the tip; a tap around 16 passes it
FORCES = (0.3, 0.33, 0.35, 0.38, 0.4)
PULSE_STEPS = 2
WAIT = 100


def _branch(seed, actions, t, delay, force) -> str:
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	env = CP._ENV
	obs, _ = env.reset(seed=seed)
	for k in range(t):
		obs, _r, term, trunc, info = env.step(actions[k])
		if term or trunc or info["drained"]:
			return "invalid"
	plan = [np.array([0, 1, 0, 1.0, 1.0]) for _ in range(delay)]
	plan += [np.array([1, 1, 0, force, 1.0]) for _ in range(PULSE_STEPS)]
	for a in plan:
		obs, _r, term, trunc, info = env.step(a)
		if term or trunc or info["drained"]:
			return "drained"
	slow = 0
	hold_right = np.array([0, 1, 0, 1.0, 1.0])
	for _ in range(WAIT):
		obs, _r, term, trunc, info = env.step(hold_right)
		if term or trunc or info["drained"]:
			return "drained"
		x, y = float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y])
		if y < CP.RELEASE_Y:
			return "bounced"
		speed = math.hypot(float(obs[OBS_BALL_VX]), float(obs[OBS_BALL_VY]))
		on_right = x < 0 and 10.0 < y < 12.8 and abs(x) < 3.5
		slow = slow + 1 if (on_right and speed < CP.SETTLE_SPEED) else 0
		if slow >= CP.SETTLE_STEPS:
			return "passed"
	return "timeout"


def _life_task(task) -> dict:
	seed, max_steps = task
	actions, events = CP._cradle_life(seed, max_steps)
	rows = []
	for t, side, x, y in events:
		if side != 0:
			continue
		grid = {f"{d}_{f}": _branch(seed, actions, t, d, f) for d in DELAYS for f in FORCES}
		if "invalid" in grid.values():
			continue
		rows.append(dict(seed=seed, x=x, y=y, grid=grid))
	return dict(seed=seed, rows=rows)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(CP.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="28000:28512")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(HERE, "pass_probe.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=CP._init, initargs=(args.binary,)) as pool:
		res = pool.map(_life_task, [(s, args.max_steps) for s in range(lo, hi)], chunksize=1)
	rows = [r for x in res for r in x["rows"]]
	print(f"{hi - lo} lives, {len(rows)} left cradles  {time.time() - t0:.0f}s")
	print(f"{'delay':>5} {'force':>5} {'passed':>7} {'drained':>8} {'bounced':>8} {'timeout':>8}")
	for d in DELAYS:
		for f in FORCES:
			out = [r["grid"][f"{d}_{f}"] for r in rows]
			n = max(1, len(out))
			print(f"{d:>5} {f:>5.2f} {out.count('passed') / n:7.2f} {out.count('drained') / n:8.2f} "
			      f"{out.count('bounced') / n:8.2f} {out.count('timeout') / n:8.2f}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
