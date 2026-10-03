"""Retinotopic giant-fiber body: every LC4 / LPLC2 input cell has its own receptive-field point.

Why: the split body (gf_split_body.py) looks at ONE point per side (the flipper AABB centre x=+-1.7839, y=12.0688), so a
ball falling through the centre gap (|x| < ~1) or past a flipper tip has almost no closing component toward either point
and the giant fibers never fire (see blind_spot.py).

Flipper geometry (read from the engine table file DEMO.DAT, attributes 800/801/802 of groups a_flip1 / a_flip2; table
coordinates, +y falls toward the drain, the LEFT flipper = action 0 is at +x because the table x axis is mirrored):
  pivot (origin)        (+-2.489, 12.063)   radius 0.311
  tip at rest   (T1)    (+-0.961, 13.131)   radius 0.193
  tip extended  (T2)    (+-0.961, 11.007)   radius 0.193
  length 1.865; the mid-sweep line is the horizontal y = 12.069 (= AABB centre y); tips of the two flippers are 1.92 apart.

Receptive-field line of each side: the horizontal mid-sweep segment y = 12.0688 from the pivot x = +-2.489 across the
extended tip (x = +-0.961) and on past it toward the centre to x = -+0.25, so the left and right fields overlap by 0.5
in the middle. The k-th of n cells of a (type, side) group (cells sorted by circuit index) sits at
x = x_pivot + (x_end - x_pivot) * k / (n - 1). Drive per cell is the same LC4 / LPLC2 formula as the split body, computed
from the cell's own point; g4 / g2 keep their meaning as the per-cell drive scale (a cell sees tanh-bounded drive, so the
mean drive over a side is at most g, like the split body). Dynamics, gating and the spectral-radius rescaling are unchanged.
Node identities are preserved in the shuffled circuits, so the same map applies to gf_circuit_shuffled_*.json.
"""
from __future__ import annotations

import json
import math

import numpy as np

import gf_body as G
import gf_split_body as S

PIVOT_X = 2.489
SEG_Y = 12.0688
CENTRE_OVERLAP_X = 0.25  # each side's field ends this far past x = 0 (toward the other side)


def field_points(side: int, n: int, y: float = SEG_Y, end_x: float = CENTRE_OVERLAP_X) -> np.ndarray:
	"""(n, 2) receptive-field points of one side (0 = left flipper at +x, 1 = right flipper at -x), pivot -> centre."""
	sgn = 1.0 if side == 0 else -1.0
	xs = np.linspace(sgn * PIVOT_X, -sgn * end_x, n) if n > 1 else np.array([sgn * 1.0])
	return np.stack([xs, np.full(n, y)], axis=1)


# "sweep" fields (blind-spot patch): the "mid" line alone misses a ball at or just past a flipper TIP (it has no closing
# component toward any point of the mid-sweep line). Each side's cells are dealt round-robin over three lines that
# span the whole swept area: pivot -> extended tip, the mid-sweep line (as "mid"), and pivot -> rest tip; the two
# tip lines run TIP_OVERSHOOT of their length past the tip.
TIP_EXT = (0.961, 11.007)
TIP_REST = (0.961, 13.131)
TIP_OVERSHOOT = 0.25


def _line(side: int, end: tuple[float, float], n: int, overshoot: float = TIP_OVERSHOOT) -> np.ndarray:
	sgn = 1.0 if side == 0 else -1.0
	p0 = np.array([sgn * PIVOT_X, 12.063])
	p1 = np.array([sgn * end[0], end[1]])
	p1 = p0 + (p1 - p0) * (1.0 + overshoot)
	t = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])
	return p0[None, :] + t[:, None] * (p1 - p0)[None, :]


def sweep_field_points(side: int, n: int) -> np.ndarray:
	"""(n, 2) points: cell k goes to line k % 3 (extended tip, mid-sweep, rest tip), spread along that line."""
	pts = np.zeros((n, 2))
	for line in range(3):
		idx = list(range(line, n, 3))
		if not idx:
			continue
		if line == 0:
			pts[idx] = _line(side, TIP_EXT, len(idx))
		elif line == 1:
			pts[idx] = field_points(side, len(idx))
		else:
			pts[idx] = _line(side, TIP_REST, len(idx))
	return pts


def retina_graph(g: dict, fields: str = "mid") -> tuple[dict, np.ndarray, np.ndarray]:
	"""Per-cell-channel graph; returns (graph, targets (n_in, 2), is_lc4 (n_in,) bool). fields: "mid" | "sweep"."""
	g = dict(g)
	cells = [(int(c), int(ch)) for c, ch in g["inputs"]]
	assert len({c for c, _ in cells}) == len(cells), "duplicate input cells"
	order = sorted(range(len(cells)), key=lambda k: cells[k][0])
	targets = np.zeros((len(cells), 2))
	is_lc4 = np.zeros(len(cells), dtype=bool)
	for t in ("LC4", "LPLC2"):
		for side in (0, 1):
			idx = [k for k in order if g["nodes"][cells[k][0]]["type"] == t and cells[k][1] == side]
			if idx:
				targets[idx] = (sweep_field_points if fields == "sweep" else field_points)(side, len(idx))
			is_lc4[idx] = t == "LC4"
	assert all(g["nodes"][c]["type"] in ("LC4", "LPLC2") for c, _ in cells)
	g["inputs"] = [[c, k] for k, (c, _) in enumerate(cells)]
	g["channels"] = [f"in{k}" for k in range(len(cells))]
	return g, targets, is_lc4


def cell_signals(obs: np.ndarray, targets: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray]:
	"""Vectorised eye_signals: (theta, theta_dot) per target point; zero where the ball is not closing."""
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	dx, dy = targets[:, 0] - obs[OBS_BALL_X], targets[:, 1] - obs[OBS_BALL_Y]
	d = np.maximum(np.hypot(dx, dy), 1e-6)
	closing = (obs[OBS_BALL_VX] * dx + obs[OBS_BALL_VY] * dy) / d
	on = closing > 0
	theta = np.where(on, 2.0 * np.arctan(R / d), 0.0)
	theta_dot = np.where(on, 2.0 * R * closing / (d * d + R * R), 0.0)
	return theta, theta_dot


class RetinaGiantFiberBody(S.SplitGiantFiberBody):
	def __init__(self, circuit_json: str, R: float, g4: float, g2: float, v0: float, s0: float, theta_on: float,
	             theta_off: float | None = None, hold_max: int = 2, refractory: int = 2, rho_target: float | None = 0.8,
	             retina: bool = True, fields: str = "mid"):
		import torch

		from agents.fixed_circuit_agent import FixedConnectome

		with open(circuit_json, encoding="utf-8") as f:
			g = json.load(f)
		assert g["output_sides"] == ["L", "R"]
		graph, self.targets, self.is_lc4 = retina_graph(g, fields)
		self.brain = FixedConnectome(graph)
		self.out_idx = list(g["outputs"])
		self.R, self.g4, self.g2, self.v0, self.s0 = float(R), float(g4), float(g2), float(v0), float(s0)
		self.theta_on = float(theta_on)
		self.theta_off = 0.7 * self.theta_on if theta_off is None else float(theta_off)
		self.hold_max, self.refractory = int(hold_max), int(refractory)
		self._torch = torch
		self.rho = G.spectral_radius(self.brain)
		self.dyn_gain = None if rho_target is None else float(rho_target) / self.rho
		self.reset()

	def drives(self, obs: np.ndarray) -> np.ndarray:
		theta, td = cell_signals(obs, self.targets, self.R)
		return np.where(self.is_lc4, self.g4 * np.tanh(td / self.v0), self.g2 * np.tanh(theta / self.s0))

	def act(self, obs: np.ndarray) -> np.ndarray:
		torch = self._torch
		drive = torch.tensor(self.drives(obs)[None, :], dtype=torch.float32)
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


def run_life_retina(params: dict, seed: int, stats: bool = False) -> dict:
	"""Life runner with a RetinaGiantFiberBody; params must carry retina=True so the body cache key is distinct."""
	key = json.dumps(params, sort_keys=True)
	if key not in G._BODIES:
		G._BODIES[key] = RetinaGiantFiberBody(**params)
	return G.run_life("gf", params, seed, stats=stats)
