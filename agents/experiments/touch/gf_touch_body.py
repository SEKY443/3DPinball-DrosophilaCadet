"""Giant-fiber body with a speed-independent TOUCH sense (Johnston's organ -> DNp01) on top of the sweep retina body.

Visual part: unchanged RetinaGiantFiberBody with fields="sweep" (per-cell LC4 / LPLC2 receptive fields along the flipper sweep).
Touch part: the JO cells of gf_touch_circuit.json (channels JO_L / JO_R, by antenna side) receive, per side,
    drive = gj * exp(-d / lam)      while the ball is in the lower table (y > Y_LOWER),
where d is the distance from the ball to that side's flipper swept area (the closer of the two segments pivot -> extended tip
and pivot -> rest tip). There is NO dependence on ball speed or direction. Left flipper = action 0 at x > 0 (mirrored table),
left antenna / JO_L / GF_L drive it. Dynamics (rho_target 0.8 sub-critical scaling), hysteresis and press logic are inherited.
All JO cells of a side share one input channel (same drive); LC4 / LPLC2 cells keep one channel per cell as in the retina body.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pathway_body"))
import gf_body as G  # noqa: E402
import gf_retina_body as RB  # noqa: E402

PIVOT = (2.489, 12.063)
TIP_EXT = (0.961, 11.007)
TIP_REST = (0.961, 13.131)
Y_LOWER = 9.5


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
	ab = b - a
	t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
	return float(np.hypot(*(p - (a + t * ab))))


def swept_distance(x: float, y: float, side: int) -> float:
	"""Distance from (x, y) to the swept area of one flipper (side 0 = left flipper at +x, 1 = right at -x)."""
	s = 1.0 if side == 0 else -1.0
	p = np.array([x, y])
	piv = np.array([s * PIVOT[0], PIVOT[1]])
	return min(_seg_dist(p, piv, np.array([s * TIP_EXT[0], TIP_EXT[1]])), _seg_dist(p, piv, np.array([s * TIP_REST[0], TIP_REST[1]])))


def touch_graph(g: dict) -> tuple[dict, np.ndarray, np.ndarray, int]:
	"""Retina graph for the LC4/LPLC2 inputs plus two shared JO channels. Returns (graph, targets, is_lc4, n_lc_channels)."""
	node_ch = {int(c): int(ch) for c, ch in g["inputs"]}
	lc = [[c, ch] for c, ch in g["inputs"] if g["nodes"][c]["type"] in ("LC4", "LPLC2")]
	jo = [[c, ch] for c, ch in g["inputs"] if g["nodes"][c]["type"] not in ("LC4", "LPLC2")]
	assert all(str(g["nodes"][c]["type"]).startswith("JO") for c, _ in jo) and all(ch in (2, 3) for _, ch in jo)
	graph, targets, is_lc4 = RB.retina_graph(dict(g, inputs=lc), "sweep")
	n = len(lc)
	graph = dict(graph)
	graph["inputs"] = graph["inputs"] + [[c, n + ch - 2] for c, ch in jo]
	graph["channels"] = graph["channels"] + ["JO_L", "JO_R"]
	return graph, targets, is_lc4, n


class TouchGiantFiberBody(RB.RetinaGiantFiberBody):
	def __init__(self, circuit_json: str, R: float, g4: float, g2: float, v0: float, s0: float, gj: float, lam: float,
	             theta_on: float, theta_off: float | None = None, hold_max: int = 2, refractory: int = 2,
	             rho_target: float | None = 0.8, fields: str = "sweep"):
		import torch

		from agents.fixed_circuit_agent import FixedConnectome

		assert fields == "sweep"
		with open(circuit_json, encoding="utf-8") as f:
			g = json.load(f)
		assert g["output_sides"] == ["L", "R"]
		graph, self.targets, self.is_lc4, self.n_lc = touch_graph(g)
		self.brain = FixedConnectome(graph)
		self.out_idx = list(g["outputs"])
		self.R, self.g4, self.g2, self.v0, self.s0 = float(R), float(g4), float(g2), float(v0), float(s0)
		self.gj, self.lam = float(gj), float(lam)
		self.theta_on = float(theta_on)
		self.theta_off = 0.7 * self.theta_on if theta_off is None else float(theta_off)
		self.hold_max, self.refractory = int(hold_max), int(refractory)
		self._torch = torch
		self.rho = G.spectral_radius(self.brain)
		self.dyn_gain = None if rho_target is None else float(rho_target) / self.rho
		self.reset()

	def touch(self, obs: np.ndarray) -> np.ndarray:
		from env_python.pinball_env import OBS_BALL_X, OBS_BALL_Y

		x, y = float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y])
		if y <= Y_LOWER:
			return np.zeros(2)
		return np.array([self.gj * np.exp(-swept_distance(x, y, s) / self.lam) for s in (0, 1)])

	def drives(self, obs: np.ndarray) -> np.ndarray:
		vis = super().drives(obs)
		return np.concatenate([vis, self.touch(obs)])


def run_life_touch(kind: str, params: dict, seed: int, stats: bool = False) -> dict:
	"""kind: 'touch' (TouchGiantFiberBody) | 'retina' (RetinaGiantFiberBody) | 'lead1'; same life runner as gf_body.run_life."""
	if kind == "lead1":
		return G.run_life("lead1", None, seed, stats=stats)
	key = json.dumps(params, sort_keys=True)
	if key not in G._BODIES:
		G._BODIES[key] = (TouchGiantFiberBody if kind == "touch" else RB.RetinaGiantFiberBody)(**params)
	return G.run_life("gf", params, seed, stats=stats)
