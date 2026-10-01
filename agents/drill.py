#!/usr/bin/env python3
"""Flipper drills: place the ball heading at the flippers from a chosen position/velocity and
score whether the policy saves it - a much less luck-dominated signal of flipper skill than a
full game's score (see place_ball in env_python/pinball_env.py and the PLC! wire frame in
src_cpp/ipc_protocol.h).

Usage:
	python agents/drill.py --policy ckpt:agents/experiments/.../champion.pt \\
		--connectome web/connectome.json --n 200 --seed 0
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from typing import Callable, Iterator, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from env_python.pinball_env import OBS_BALL_X, OBS_BALL_Y, PinballEnv  # noqa: E402

PolicyFn = Callable[[np.ndarray, int], tuple[bool, bool]]

# Drill zone (see make_drills): start position clears the plunger lane and sits above the
# flippers; downward velocity range covers gentle drops through the flippers' own max speed.
DRILL_X_RANGE = (-3.5, 3.5)
DRILL_Y_RANGE = (3.0, 7.0)
DRILL_VX_RANGE = (-6.0, 6.0)
DRILL_VY_RANGE = (6.0, 18.0)

# Ball leaving the plunger lane, per pinball_env.py's own PLUNGER_REST_X/DRAIN_FROM_Y geometry.
LANE_CLEAR_X = -5.5
LANE_CLEAR_Y = 8.0
LANE_CLEAR_GIVE_UP_STEPS = 400

# Outcome thresholds - table top is y=-12, flippers sit at y 10.2-14, drain below y=14.2 (see
# ipc_protocol.h's docstring and pinball_env.py's DRAIN_FROM_Y).
SAVED_Y = 3.0
LOST_Y = 14.2


@dataclasses.dataclass(frozen=True)
class Drill:
	seed: int
	x: float
	y: float
	vx: float
	vy: float


def make_drills(n: int, rng: np.random.Generator) -> Iterator[Drill]:
	"""Random drill starts: falling toward the flippers from above, x/y within the flipper
	zone's approach, vy always positive (falling toward increasing y, i.e. toward the flippers)."""
	for _ in range(n):
		seed = int(rng.integers(0, 2**31 - 1))
		x = float(rng.uniform(*DRILL_X_RANGE))
		y = float(rng.uniform(*DRILL_Y_RANGE))
		vx = float(rng.uniform(*DRILL_VX_RANGE))
		vy = float(rng.uniform(*DRILL_VY_RANGE))
		yield Drill(seed, x, y, vx, vy)


_BANK_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load_bank(bank_path: str) -> dict[str, np.ndarray]:
	# Cached per (worker) process so repeated calls don't re-hit disk.
	bank = _BANK_CACHE.get(bank_path)
	if bank is None:
		with np.load(bank_path) as data:
			bank = {k: data[k] for k in ("x", "y", "vx", "vy")}
		_BANK_CACHE[bank_path] = bank
	return bank


def drills_from_bank(bank_path: str, n: int, rng: np.random.Generator) -> list[Drill]:
	"""Real-game-sampled drill starts: uniformly resample (with replacement) approach states
	collected by agents/collect_drill_bank.py, instead of make_drills' synthetic ranges."""
	bank = _load_bank(bank_path)
	n_rows = bank["x"].shape[0]
	drills = []
	for _ in range(n):
		seed = int(rng.integers(0, 2**31 - 1))
		idx = int(rng.integers(0, n_rows))
		drills.append(Drill(
			seed, float(bank["x"][idx]), float(bank["y"][idx]), float(bank["vx"][idx]), float(bank["vy"][idx])
		))
	return drills


def _wait_for_lane_clear(
	env: PinballEnv, on_step: Optional[Callable[[np.ndarray, int], None]] = None
) -> tuple[Optional[np.ndarray], Optional[dict]]:
	"""Steps with no input until the ball has left the plunger lane, or gives up. Returns
	(obs, info) on success, (None, None) if the drill should be skipped. `on_step`, if given, is
	called with each waiting-phase (obs, step) - return ignored, action stays forced to no-op -
	used to keep a stateful policy's hidden state warm through the wait (see run_one_drill)."""
	obs = info = None
	for t in range(LANE_CLEAR_GIVE_UP_STEPS):
		obs, _reward, terminated, _truncated, info = env.step([0, 0, 0])
		if on_step is not None:
			on_step(obs, t)
		if terminated:
			return None, None
		if info["ball_in_play"] and obs[OBS_BALL_X] > LANE_CLEAR_X and obs[OBS_BALL_Y] < LANE_CLEAR_Y:
			return obs, info
	return None, None


def run_one_drill(
	env: PinballEnv, policy_fn: PolicyFn, drill: Drill, max_steps: int = 150, step_policy_during_wait: bool = False
) -> dict:
	"""Runs a single drill against policy_fn and classifies the outcome (see module docstring for
	the "saved"/"lost"/"timeout" definitions). `step_policy_during_wait` also calls policy_fn
	(hidden-state side effect only, action ignored) through the lane-clear wait - off by default so
	drill.py's own CLI results stay unchanged. Returns outcome plus post-placement telemetry."""
	unwrapped = env.unwrapped
	env.reset(seed=drill.seed)
	if hasattr(policy_fn, "reset"):
		policy_fn.reset()

	on_step = policy_fn if step_policy_during_wait else None
	obs, info = _wait_for_lane_clear(env, on_step=on_step)
	if obs is None:
		return {"drill": drill, "outcome": "skipped"}

	obs, _info = unwrapped.place_ball(drill.x, drill.y, drill.vx, drill.vy)

	outcome = "timeout"
	hit_occurred = False
	ticks = contacts = press_left_ticks = press_right_ticks = 0
	for t in range(max_steps):
		left, right = policy_fn(obs, t)
		obs, _reward, terminated, _truncated, info = env.step([bool(left), bool(right), 0])
		ticks += 1
		if info["flipper_hit"]:
			hit_occurred = True
			contacts += 1
		press_left_ticks += int(bool(left))
		press_right_ticks += int(bool(right))
		if info["drained"] or obs[OBS_BALL_Y] > LOST_Y:
			outcome = "lost"
			break
		if hit_occurred and obs[OBS_BALL_Y] < SAVED_Y:
			outcome = "saved"
			break
		if terminated:
			outcome = "lost"
			break

	return {
		"drill": drill, "outcome": outcome, "ticks": ticks, "contacts": contacts,
		"press_left_ticks": press_left_ticks, "press_right_ticks": press_right_ticks,
	}


def run_drills(env: PinballEnv, policy_fn: PolicyFn, drills, max_steps: int = 150) -> dict:
	"""Runs each drill via run_one_drill and aggregates the outcome counts."""
	results = [run_one_drill(env, policy_fn, d, max_steps=max_steps) for d in drills]

	n_scored = sum(1 for r in results if r["outcome"] != "skipped")
	n_saved = sum(1 for r in results if r["outcome"] == "saved")
	n_lost = sum(1 for r in results if r["outcome"] == "lost")
	n_timeout = sum(1 for r in results if r["outcome"] == "timeout")
	n_skipped = len(results) - n_scored
	save_rate = n_saved / n_scored if n_scored else float("nan")
	save_rate_se = (
		float(np.sqrt(save_rate * (1.0 - save_rate) / n_scored)) if n_scored else float("nan")
	)

	summary = {
		"n": len(results), "n_scored": n_scored, "saved": n_saved, "lost": n_lost,
		"timeout": n_timeout, "skipped": n_skipped, "save_rate": save_rate, "save_rate_se": save_rate_se,
	}
	return {"results": results, "summary": summary}


def none_policy(obs: np.ndarray, t: int) -> tuple[bool, bool]:
	return False, False


class FlailPolicy:
	"""Presses both flippers on alternate step pairs while the ball is near the flipper zone -
	a dumb, non-learned baseline (no timing skill, just "flap when close")."""

	def __call__(self, obs: np.ndarray, t: int) -> tuple[bool, bool]:
		near = obs[OBS_BALL_Y] > 9.0 and abs(obs[OBS_BALL_X]) < 4.0
		press = bool(near and (t % 4) < 2)
		return press, press


class CkptPolicy:
	"""Loads a train_pinball_circuit_cem.py checkpoint and replays its greedy action, carrying
	the circuit's hidden state across steps within one drill (reset via .reset())."""

	def __init__(self, ckpt_path: str, connectome_path: str, env: PinballEnv):
		# Imported lazily: pulls in the full training module (argparse/multiprocessing/etc.),
		# unnecessary for the "none"/"flail" policies.
		from agents.train_pinball_circuit_cem import build_agent, set_flat_params

		ckpt = torch.load(ckpt_path, weights_only=False)
		kind = ckpt.get("agent", "circuit")
		self.agent = build_agent(kind, connectome_path, ckpt["readout_dim"])
		self.agent.calibrate(env.observation_space)
		set_flat_params(self.agent.decoder, ckpt["champion"])
		self.h = None

	def reset(self) -> None:
		self.h = self.agent.initial_state(1)

	def __call__(self, obs: np.ndarray, t: int) -> tuple[bool, bool]:
		obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
		with torch.no_grad():
			action, self.h = self.agent.act(obs_t, self.h, greedy=True)
		action = np.asarray(action)
		return bool(action[0][0]), bool(action[0][1])


def build_policy(spec: str, connectome_path: str, env: PinballEnv) -> PolicyFn:
	if spec == "none":
		return none_policy
	if spec == "flail":
		return FlailPolicy()
	if spec.startswith("ckpt:"):
		return CkptPolicy(spec[len("ckpt:"):], connectome_path, env)
	raise ValueError(f"unknown --policy {spec!r} (expected none, flail, or ckpt:<path>)")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", default=os.environ.get("PINBALL_BINARY"))
	parser.add_argument("--policy", required=True, help="none | flail | ckpt:<path>")
	parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "..", "web", "connectome.json"))
	parser.add_argument("--n", type=int, default=200)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--bank", default=None, help="path to a bank .npz (agents/collect_drill_bank.py); default None = synthetic make_drills")
	parser.add_argument("--max-steps", type=int, default=150)
	# Matches train_pinball_circuit_cem.py's own default - real game-time coverage per env.step()
	# is frame_skip native ticks, and max_steps=150 was sized assuming this cadence.
	parser.add_argument("--frame-skip", type=int, default=4)
	args = parser.parse_args()

	if not args.binary:
		raise SystemExit("--binary not given and PINBALL_BINARY is not set")

	env = PinballEnv(binary_path=args.binary, headless=True, frame_skip=args.frame_skip)
	try:
		policy_fn = build_policy(args.policy, args.connectome, env)
		rng = np.random.default_rng(args.seed)
		drills = drills_from_bank(args.bank, args.n, rng) if args.bank else list(make_drills(args.n, rng))
		outcome = run_drills(env, policy_fn, drills, max_steps=args.max_steps)
	finally:
		env.close()

	s = outcome["summary"]
	print(
		f"policy={args.policy}  n={s['n']} scored={s['n_scored']} skipped={s['skipped']}  "
		f"saved={s['saved']} lost={s['lost']} timeout={s['timeout']}  "
		f"save_rate={s['save_rate']:.3f} +/- {s['save_rate_se']:.3f} (SE)"
	)


if __name__ == "__main__":
	main()
