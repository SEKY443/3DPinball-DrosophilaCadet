"""Giant-fiber body: looming drive -> fixed MaleCNS circuit (LC4/LPLC2 -> ... -> DNp01) -> flipper.

No trained readout. Per decision step: drive_L/R = gain * looming_intensity(obs, LOOM_TARGET_L/R, tau0)
is injected into the loom_L / loom_R input cells, the fixed leaky-tanh circuit (FixedConnectome from
agents/fixed_circuit_agent.py) advances one step, and the DNp01 activities a_L / a_R gate the
flippers with hysteresis (press above theta_on, release below theta_off), a hold cap and a refractory
period. GF_L drives action 0 (left flipper), GF_R action 1.

Also holds the shared environment/life-runner helpers used by tune_gf_body.py and eval_gf_body.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # step3 repo root (analog-force engine + env)
for p in (ROOT, os.path.join(ROOT, "vendor", "nfly"), os.path.join(ROOT, "agents", "experiments", "encoding")):
	if p not in sys.path:
		sys.path.insert(0, p)

BINARY = os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball")


def spectral_radius(brain) -> float:
	"""Spectral radius of the normalized signed weight matrix W[post, pre] of a FixedConnectome."""
	W = np.zeros((brain.n, brain.n))
	np.add.at(W, (brain.post.numpy(), brain.pre.numpy()), brain.w.numpy().astype(float))
	return float(np.max(np.abs(np.linalg.eigvals(W))))


def step_with_gain(brain, activity, channel_drive, gain: float):
	"""FixedConnectome.step with an explicit dynamics gain (same math, same leak/iterations)."""
	import torch

	from agents.fixed_circuit_agent import DYNAMICS_ITERATIONS, DYNAMICS_LEAK

	drive = torch.zeros(activity.shape[0], brain.n, dtype=activity.dtype)
	drive[:, brain.input_cells] = channel_drive[:, brain.input_channels]
	for _ in range(DYNAMICS_ITERATIONS):
		msg = torch.zeros_like(activity).index_add_(1, brain.post, activity[:, brain.pre] * brain.w)
		activity = (1 - DYNAMICS_LEAK) * activity + DYNAMICS_LEAK * torch.tanh(drive + gain * msg)
	return activity


class GiantFiberBody:
	def __init__(self, circuit_json: str, gain: float, tau0: float, theta_on: float, theta_off: float | None = None,
	             hold_max: int = 2, refractory: int = 2, rho_target: float | None = None,
	             force_mode: str | None = None, f0: float = 1.0, k: float = 0.0, f_min: float = 0.2):
		import torch

		from agents.fixed_circuit_agent import FixedConnectome

		with open(circuit_json, encoding="utf-8") as f:
			g = json.load(f)
		assert g["output_sides"] == ["L", "R"]
		self.brain = FixedConnectome(g)
		self.out_idx = list(g["outputs"])  # [GF_L, GF_R]
		self.gain, self.tau0 = float(gain), float(tau0)
		self.theta_on = float(theta_on)
		self.theta_off = 0.7 * self.theta_on if theta_off is None else float(theta_off)
		self.hold_max, self.refractory = int(hold_max), int(refractory)
		# force_mode None = original 3-element binary action (default physics). 'const' = 5-element action
		# with force 1.0 (255/255 = the engine's unmodified full-force flippers). 'rate' = while a side is
		# pressed, force = clip(f0 + k * (a_side - theta_on), f_min, 1.0).
		assert force_mode in (None, "const", "rate")
		self.force_mode, self.f0, self.k, self.f_min = force_mode, float(f0), float(k), float(f_min)
		self._torch = torch
		self.rho = spectral_radius(self.brain)
		# Optional sub-critical override: dynamics gain = rho_target / rho (instance-level; the shared
		# module's DYNAMICS_GAIN constant is untouched). None = the default gain 1.4 via brain.step.
		self.dyn_gain = None if rho_target is None else float(rho_target) / self.rho
		self.reset()

	def _step(self, h, drive_ch):
		if self.dyn_gain is None:
			return self.brain.step(h, drive_ch)
		return step_with_gain(self.brain, h, drive_ch, self.dyn_gain)

	def reset(self) -> None:
		self.h = self._torch.zeros(1, self.brain.n)
		self.pressed = [False, False]
		self.run = [0, 0]
		self.rest = [self.refractory, self.refractory]
		self.last_a = (0.0, 0.0)

	def act(self, obs: np.ndarray) -> np.ndarray:
		from agents.train_pinball_circuit_cem import LOOM_TARGET_LEFT, LOOM_TARGET_RIGHT, looming_intensity

		torch = self._torch
		drive = torch.tensor([[self.gain * looming_intensity(obs, LOOM_TARGET_LEFT, self.tau0),
		                       self.gain * looming_intensity(obs, LOOM_TARGET_RIGHT, self.tau0)]], dtype=torch.float32)
		with torch.no_grad():
			self.h = self._step(self.h, drive)
		a = (float(self.h[0, self.out_idx[0]]), float(self.h[0, self.out_idx[1]]))
		self.last_a = a
		action = np.zeros(3 if self.force_mode is None else 5, dtype=np.int64 if self.force_mode is None else np.float64)
		for s in (0, 1):
			if self.pressed[s]:
				if a[s] < self.theta_off or self.run[s] >= self.hold_max:
					self.pressed[s], self.run[s], self.rest[s] = False, 0, 0
				else:
					self.run[s] += 1
			else:
				self.rest[s] += 1
				if a[s] > self.theta_on and self.rest[s] > self.refractory:
					self.pressed[s], self.run[s] = True, 1
			action[s] = int(self.pressed[s])
		if self.force_mode is not None:
			action[3] = action[4] = 1.0
			if self.force_mode == "rate":
				for s in (0, 1):
					if self.pressed[s]:
						action[3 + s] = float(np.clip(self.f0 + self.k * (a[s] - self.theta_on), self.f_min, 1.0))
		return action


# ---- shared env / life runner -------------------------------------------------------------------
_ENV = None
_DT = 0.0347


def init_worker(binary: str = BINARY, max_steps: int = 3000) -> None:
	global _ENV
	import atexit
	import signal

	import gymnasium as gym
	import torch

	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=binary, headless=True, frame_skip=4), max_episode_steps=max_steps)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)


_BODIES: dict = {}


def run_life(kind: str, params: dict | None, seed: int, stats: bool = False) -> dict:
	"""kind: 'gf' (params incl. circuit path), 'lead1', 'none'."""
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y, OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

	obs, _ = _ENV.reset(seed=seed)
	body = reflex = None
	if kind == "gf":
		key = json.dumps(params, sort_keys=True)
		if key not in _BODIES:
			_BODIES[key] = GiantFiberBody(**params)
		body = _BODIES[key]
		body.reset()
	elif kind == "lead1":
		reflex = ReflexLabeler(11.5, 1)
	prev_l, prev_r = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
	score = steps = contacts = presses = 0
	far, zone = [], []
	done = False
	while not done:
		if kind == "gf":
			act = body.act(obs)
			if stats:
				(zone if obs[OBS_BALL_Y] > 10 else far).append(body.last_a)
		elif kind == "lead1":
			view = np.array(obs, dtype=np.float32)
			if view[OBS_BALL_VY] > 0:
				view[OBS_BALL_Y] = view[OBS_BALL_Y] + view[OBS_BALL_VY] * _DT
			act = reflex.label(view)
			act[2] = 0
			reflex.observe(act)
		else:
			act = np.zeros(3, dtype=np.int64)
		obs, _r, term, trunc, info = _ENV.step(act)
		score += info["score_delta"]
		contacts += int(info["flipper_hit"])
		lu, ru = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
		presses += int(lu and not prev_l) + int(ru and not prev_r)
		prev_l, prev_r = lu, ru
		steps += 1
		done = term or trunc or bool(info["drained"])
	out = dict(kind=kind, seed=seed, score=float(score), steps=steps, contacts=contacts, presses=presses)
	if stats:
		out["far"] = np.asarray(far, dtype=float).reshape(-1, 2).tolist()
		out["zone"] = np.asarray(zone, dtype=float).reshape(-1, 2).tolist()
	return out


def circuit_path(tag: str) -> str:
	"""tag: 'real' or 'shuffled_<i>' -> local circuit JSON."""
	return os.path.join(HERE, "gf_circuit.json" if tag == "real" else f"gf_circuit_{tag}.json")


def load_budget_params(tag: str) -> dict:
	"""Budget-tuned body params for a circuit (copied from pathway_body), circuit path pointed at the local copy."""
	with open(os.path.join(HERE, f"best_params_{tag}_budget.json"), encoding="utf-8") as f:
		p = dict(json.load(f)["params"])
	p["circuit_json"] = circuit_path(tag)
	return p
