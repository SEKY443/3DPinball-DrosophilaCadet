"""Cradle-and-shoot teacher, evaluated in FULL games against lead1.

cradle_probe.py showed that from a ball trapped on a raised flipper, the release delay before the
shot controls where it goes: LEFT cradle (ball at x > 0, mirrored table) + 1 step -> multiplier bank
~16%; RIGHT cradle + 5 steps -> ramp ~16%, versus ~4% per ordinary flipper shot.

Policy (stateful, one instance per ball):
  * default: lead1 reflex (press when y + vy*DT_STEP > 11.5, pulse, cooldown 2);
  * catch: a ball falling slowly (speed < catch_speed) onto a flipper from the side
    (y > CATCH_Y, CATCH_X_MIN < |x| < CATCH_X_MAX) makes that flipper rise and stay up;
  * cradle: while held, a ball slow for SETTLE_STEPS steps over the flippers is trapped -> release
    the flipper for delay[side] steps, then pulse it for PULSE_STEPS (the aimed shot);
  * abort: ball back up the table (y < RELEASE_Y) or HOLD_TIMEOUT steps without settling -> release;
  * after a shot, catching stays disarmed until the ball is back up the table (y < RELEASE_Y), so a
    shot that fails to move the ball is followed by plain lead1 play instead of an endless re-cradle.

Usage: python agents/experiments/cradle/cradle_teacher.py --seeds 64000:64064 --workers 16
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
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "encoding"))

DT_STEP = 0.0347
CATCH_Y = 9.0
CATCH_X_MIN, CATCH_X_MAX = 1.0, 5.0
SETTLE_SPEED = 0.6
SETTLE_STEPS = 6
RELEASE_Y = 9.0
HOLD_TIMEOUT = 80
PULSE_STEPS = 3
DELAYS = (1, 5)  # release steps before the shot: (left cradle -> bank, right cradle -> ramp)

POLICIES = {
	"lead1": None,
	"cradle_s3": dict(catch_speed=3.0),
	"cradle_s5": dict(catch_speed=5.0),
	# force_probe.py best combos: LEFT delay 5 force 0.45 (bank), RIGHT delay 5 force 0.75 (ramp)
	"cradle_force": dict(catch_speed=5.0, delays=(5, 5), forces=(0.45, 0.75)),
}
_BINARY = [None]
_IDS: dict = {}


class CradleTeacher:
	def __init__(self, catch_speed: float, delays=DELAYS, forces=None, post_shot: int = 0,
	             catch_y: float = CATCH_Y, catch_x_max: float = CATCH_X_MAX, left_pass=None,
	             pass_hold_timeout: int = HOLD_TIMEOUT):
		from agents.train_pinball_circuit_cem import ReflexLabeler

		self.reflex = ReflexLabeler(11.5, 1)
		self.catch_speed, self.delays = catch_speed, delays
		# forces=(left, right) in [0, 1] for the aimed shot (analog flippers, ACTF frames); None keeps
		# the original binary flippers and 3-element actions.
		self.forces = forces
		# Decision steps the routine keeps both flippers released after the aimed shot, so a hybrid
		# controller taking over afterwards cannot hit the ball again while it is leaving the flipper.
		self.post_shot = post_shot
		self.catch_y, self.catch_x_max = catch_y, catch_x_max
		# left_pass=(tap_step, tap_force): a LEFT cradle is passed to the right flipper instead of being
		# shot (pass_probe.py: keep the right flipper up, release the left one, tap it at step 18 with
		# force 0.33 -> ball rests on the right flipper ~92%), then the right-cradle shot follows.
		self.left_pass = left_pass
		self._hold_after_plan = -1
		self.passes = 0
		# settle wait for the receiving flipper after a pass (a pass takes longer to come to rest)
		self.pass_hold_timeout = pass_hold_timeout
		self._timeout = HOLD_TIMEOUT
		self.hold = -1        # flipper side being held for a catch, -1 = none
		self.hold_steps = 0
		self.slow = 0
		self.plan: list = []  # queued actions for an aimed shot
		self.cradles = 0
		# After a shot, no new catch until the ball has really left the flipper area: a shot that
		# fails to move the ball would otherwise re-cradle and re-shoot forever.
		self.catch_armed = True
		self.override = False
		self.catches = 0       # catch attempts started (flipper raised for a slow side ball)
		self.eligible = 0      # steps where a slow side ball was falling (catch condition, armed or not)
		self.abort_bounce = 0  # catch ended because the ball went back up the table
		self.abort_timeout = 0 # catch ended after HOLD_TIMEOUT steps without settling
		self.min_speed_sum = 0.0  # sum over aborted catches of the slowest speed seen on the flipper
		self._min_speed = 1e9

	def act(self, obs) -> np.ndarray:
		from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

		x, y = float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y])
		vx, vy = float(obs[OBS_BALL_VX]), float(obs[OBS_BALL_VY])
		speed = math.hypot(vx, vy)
		# override = this step is part of the cradle routine (catch, hold, release, aimed shot);
		# used by hybrid policies that let another controller play every other step.
		self.override = True
		if self.plan:
			a = self.plan.pop(0)
			if not self.plan and self._hold_after_plan >= 0:
				# pass finished: wait for the ball to settle on the receiving flipper, then shoot
				self.hold, self.hold_steps, self.slow = self._hold_after_plan, 0, 0
				self._hold_after_plan = -1
				self._timeout = self.pass_hold_timeout
		elif self.hold >= 0:
			self.hold_steps += 1
			on_flipper = 10.0 < y < 12.8 and abs(x) < 3.5
			if on_flipper:
				self._min_speed = min(self._min_speed, speed)
			self.slow = self.slow + 1 if (on_flipper and speed < SETTLE_SPEED) else 0
			if self.slow >= SETTLE_STEPS and self.hold == 0 and self.left_pass is not None:
				tap_step, tap_force = self.left_pass
				self.cradles += 1
				self.passes += 1
				# right flipper stays up, left released until the tap, then a weak left tap
				self.plan = [np.array([0, 1, 0, 1.0, 1.0]) for _ in range(tap_step)]
				self.plan += [np.array([1, 1, 0, tap_force, 1.0]) for _ in range(2)]
				self._hold_after_plan = 1
				self.hold, self.slow = -1, 0
				a = self.plan.pop(0)
			elif self.slow >= SETTLE_STEPS:
				side = self.hold
				self.cradles += 1
				self.plan = [np.zeros(3, dtype=np.int64) for _ in range(self.delays[side])]
				for _ in range(PULSE_STEPS):
					p = np.zeros(3, dtype=np.int64)
					p[side] = 1
					self.plan.append(self._with_force(p, side))
				self.plan.extend(np.zeros(3, dtype=np.int64) for _ in range(self.post_shot))
				self.hold, self.slow = -1, 0
				self.catch_armed = False
				a = self.plan.pop(0)
			elif y < RELEASE_Y or self.hold_steps > self._timeout:
				if y < RELEASE_Y:
					self.abort_bounce += 1
				else:
					self.abort_timeout += 1
				self.min_speed_sum += self._min_speed if self._min_speed < 1e9 else 0.0
				self.hold, self.slow = -1, 0
				self.override = False
				a = self._lead(obs)
			else:
				a = self._lead(obs)
				a[self.hold] = 1
		else:
			if y < RELEASE_Y:
				self.catch_armed = True
			side_ok = CATCH_X_MIN < abs(x) < self.catch_x_max
			slow_side_ball = vy > 0 and y > self.catch_y and side_ok and speed < self.catch_speed
			self.eligible += int(slow_side_ball)
			if self.catch_armed and slow_side_ball:
				self.catches += 1
				self._timeout = HOLD_TIMEOUT
				self._min_speed = 1e9
				self.hold = 0 if x > 0 else 1  # x > 0 is the LEFT flipper (mirrored table)
				self.hold_steps = self.slow = 0
				a = self._lead(obs)
				a[self.hold] = 1
			else:
				self.override = False
				a = self._lead(obs)
		self.reflex.observe(a[:3].astype(np.int64))
		return a if len(a) == 5 else self._with_force(a, None)

	def _with_force(self, a, shot_side):
		"""Full force everywhere except the aimed shot's flipper; plain 3-element action if no forces."""
		if self.forces is None or len(a) == 5:
			return a
		out = np.array([a[0], a[1], a[2], 1.0, 1.0], dtype=np.float64)
		if shot_side is not None:
			out[3 + shot_side] = self.forces[shot_side]
		return out

	def _lead(self, obs) -> np.ndarray:
		from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

		view = np.array(obs, dtype=np.float32)
		if view[OBS_BALL_VY] > 0:
			view[OBS_BALL_Y] += view[OBS_BALL_VY] * DT_STEP
		a = self.reflex.label(view)
		a[2] = 0
		return a


class Lead1:
	def __init__(self):
		self.t = CradleTeacher(catch_speed=-1.0)  # catch never triggers -> plain lead1
		self.cradles = 0

	def act(self, obs):
		return self.t.act(obs)


def _init(binary: str) -> None:
	import signal

	_BINARY[0] = binary
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		objs = json.load(f)["objects"]
	_IDS["ramp"] = frozenset(o["id"] for o in objs if o["name"] == "ramp")
	_IDS["bank"] = frozenset(o["id"] for o in objs if o["name"] in ("a_targ7", "a_targ8", "a_targ9"))


def _game(task) -> dict:
	from env_python.pinball_env import PinballEnv

	name, seed, max_steps = task
	spec = POLICIES[name]

	def fresh():
		return Lead1() if spec is None else CradleTeacher(**spec)

	env = PinballEnv(binary_path=_BINARY[0], headless=True, frame_skip=4, relaunch_at_rest=True)
	score = 0.0
	ramp = bank = completions = cradles = 0
	game_over = False
	try:
		obs, _ = env.reset(seed=seed)
		pol = fresh()
		prev_drained = False
		for _ in range(max_steps):
			obs, _r, term, _trunc, info = env.step(pol.act(obs))
			score += info["score_delta"]
			for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
				ramp += int(oid in _IDS["ramp"])
				bank += int(oid in _IDS["bank"])
				completions += int(oid in _IDS["bank"] and base >= 1500)
			if info["drained"] and not prev_drained:
				cradles += getattr(pol, "cradles", 0) if spec is not None else 0
				pol = fresh()
			prev_drained = bool(info["drained"])
			if term:
				game_over = True
				break
		if spec is not None:
			cradles += pol.cradles
	finally:
		env.close()
	return dict(name=name, seed=seed, score=float(score), ramp=ramp, bank=bank, completions=completions,
	            cradles=cradles, game_over=game_over)


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="64000:64064")
	ap.add_argument("--policies", default=",".join(POLICIES))
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=30000)
	ap.add_argument("--out", default=os.path.join(HERE, "cradle_teacher_eval.json"))
	args = ap.parse_args()
	names = args.policies.split(",")
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		res = pool.map(_game, [(n, s, args.max_steps) for s in seeds for n in names], chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["name"] == n} for n in names}
	print(f"paired full games {lo}..{hi - 1} ({len(seeds)})  {time.time() - t0:.0f}s")
	for n in names:
		rs = [by[n][s] for s in seeds]
		sc = np.array([r["score"] for r in rs])
		lg = np.log1p(sc)
		line = (f"{n:<10} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median(sc):,.0f}  "
		        f"mean {sc.mean():,.0f}  ramp {np.mean([r['ramp'] for r in rs]):.1f}  bank {np.mean([r['bank'] for r in rs]):.1f}  "
		        f"compl {np.mean([r['completions'] for r in rs]):.2f}  cradles {np.mean([r['cradles'] for r in rs]):.1f}  "
		        f"over {np.mean([r['game_over'] for r in rs]):.0%}")
		if n != "lead1" and "lead1" in by:
			d = np.array([np.log1p(by[n][s]["score"]) - np.log1p(by["lead1"][s]["score"]) for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			line += f" | vs lead1 {d.mean():+.3f} +- {se:.3f} t {d.mean() / se:+.2f} p {confirm.wilcoxon_p(d):.2g} wins {int((d > 0).sum())}/{int((d < 0).sum())}"
		print(line)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), games=res), fh)


if __name__ == "__main__":
	main()
