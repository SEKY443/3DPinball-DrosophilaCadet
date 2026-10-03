"""Split giant-fiber body: LC4 and LPLC2 get different looming signals (Ache et al. 2019).

LC4 encodes angular VELOCITY of a looming object, LPLC2 encodes angular SIZE. Per eye (flipper point):
d = distance ball->target, closing = velocity toward target (units/s), object radius R,
theta = 2*atan(R/d), theta_dot = 2*R*closing/(d^2+R^2); both are 0 unless closing > 0.
LC4 drive = g4*tanh(theta_dot/v0), LPLC2 drive = g2*tanh(theta/s0).
Input channels are remapped in memory to ['LC4_L','LC4_R','LPLC2_L','LPLC2_R'] by node type and the
original side channel; gating (hysteresis/hold/refractory) and sub-critical dynamics are as in GiantFiberBody.
"""
from __future__ import annotations

import json
import math

import numpy as np

import gf_body as G

CHANNELS = ["LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R"]


def remap_graph(g: dict) -> dict:
	g = dict(g)
	new = []
	for cell, ch in g["inputs"]:
		t = g["nodes"][cell]["type"]
		assert t in ("LC4", "LPLC2"), t
		new.append([cell, (0 if t == "LC4" else 2) + int(ch)])
	g["inputs"] = new
	g["channels"] = list(CHANNELS)
	return g


def eye_signals(obs: np.ndarray, target: tuple, R: float) -> tuple[float, float]:
	"""(theta, theta_dot) of a ball of radius R seen from target; (0, 0) when not closing."""
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	dx, dy = target[0] - obs[OBS_BALL_X], target[1] - obs[OBS_BALL_Y]
	d = float(math.hypot(dx, dy))
	if d < 1e-6:
		return math.pi, 0.0
	closing = (obs[OBS_BALL_VX] * dx + obs[OBS_BALL_VY] * dy) / d
	if closing <= 0:
		return 0.0, 0.0
	return 2.0 * math.atan(R / d), 2.0 * R * closing / (d * d + R * R)


class SplitGiantFiberBody(G.GiantFiberBody):
	def __init__(self, circuit_json: str, R: float, g4: float, g2: float, v0: float, s0: float, theta_on: float,
	             theta_off: float | None = None, hold_max: int = 2, refractory: int = 2, rho_target: float | None = 0.8):
		import torch

		from agents.fixed_circuit_agent import FixedConnectome

		with open(circuit_json, encoding="utf-8") as f:
			g = json.load(f)
		assert g["output_sides"] == ["L", "R"]
		self.brain = FixedConnectome(remap_graph(g))
		self.out_idx = list(g["outputs"])
		self.R, self.g4, self.g2, self.v0, self.s0 = float(R), float(g4), float(g2), float(v0), float(s0)
		self.theta_on = float(theta_on)
		self.theta_off = 0.7 * self.theta_on if theta_off is None else float(theta_off)
		self.hold_max, self.refractory = int(hold_max), int(refractory)
		self._torch = torch
		self.rho = G.spectral_radius(self.brain)
		self.dyn_gain = None if rho_target is None else float(rho_target) / self.rho
		self.reset()

	def act(self, obs: np.ndarray) -> np.ndarray:
		from agents.train_pinball_circuit_cem import LOOM_TARGET_LEFT, LOOM_TARGET_RIGHT

		torch = self._torch
		l_th, l_td = eye_signals(obs, LOOM_TARGET_LEFT, self.R)
		r_th, r_td = eye_signals(obs, LOOM_TARGET_RIGHT, self.R)
		drive = torch.tensor([[self.g4 * math.tanh(l_td / self.v0), self.g4 * math.tanh(r_td / self.v0),
		                       self.g2 * math.tanh(l_th / self.s0), self.g2 * math.tanh(r_th / self.s0)]], dtype=torch.float32)
		with torch.no_grad():
			self.h = self._step(self.h, drive)
		a = (float(self.h[0, self.out_idx[0]]), float(self.h[0, self.out_idx[1]]))
		self.last_a = a
		action = np.zeros(3, dtype=np.int64)
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
		return action


def run_life_split(params: dict, seed: int, stats: bool = False) -> dict:
	"""Same life runner as gf_body.run_life but with a SplitGiantFiberBody (reuses run_life via its body cache)."""
	key = json.dumps(params, sort_keys=True)
	if key not in G._BODIES:
		G._BODIES[key] = SplitGiantFiberBody(**params)
	return G.run_life("gf", params, seed, stats=stats)
