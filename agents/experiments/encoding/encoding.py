"""Sensory encodings for the fixed MaleCNS pinball circuit (encoding study, 2026-09-30).

The production agent (agents/fixed_circuit_agent.FixedCircuitAgent) injects each of the 15
observation channels as ONE scalar, obs/OBS_SCALE clamped to [-1, 1], broadcast identically to all
12 input cells of that channel. This module keeps the circuit, its 180 input cells, its dynamics
and its 64 readout cells unchanged, and only changes WHAT each input cell receives: every input
cell gets its own drive value, computed from the (already FlipperObsFilter-ed) 15-d observation.

Implementation: `EncodedCircuitAgent` is a FixedCircuitAgent whose connectome is rebuilt with one
virtual "channel" per input cell (input_channels = arange(180)); `_channel_drive(obs)` returns the
(B, 180) per-cell drive. Everything else (step, act, calibrate, decoder, flat params) is inherited,
so the decoder parameter vector has the same size and layout as the production agent's.

Encodings (name -> function of obs, see ENCODINGS):
  broadcast   production behaviour (sanity check: bit-identical to FixedCircuitAgent).
  gain        broadcast, but ball_y / ball_vy are re-centred and re-scaled so the flipper zone uses
              the input range: y -> (y - 10) / 3, vy -> vy / 10 (clamped to [-1, 1]).
  popcode     proprioceptor-style range fractionation on the 4 mechanosensory ball channels:
              ball_x cells: Gaussian position bumps tiling x in [-7, 7];
              ball_y cells: Gaussian position bumps tiling y, denser over the flipper zone;
              ball_vx / ball_vy cells: half-wave rectified, direction-selective speed units (6 per
              direction) with graded thresholds.
  dspop       popcode, plus the ball_vy cells become direction-selective (falling-only)
              position units: bump(y - c_k) * sigmoid(vy), a T4/T5-like "downward motion at
              retinotopic position c_k" code over the lower table.
  retino      ball position as 2-D Gaussian bumps over the 96 LC4/LPLC2 visual input cells (8
              visual channels x 12 cells, laid out as a retinotopic grid), multiplied by an
              approach (vy > 0) gate for LC4 and not for LPLC2; mechanosensory channels unchanged.
Any name may be suffixed "+ttc": the 12 tilt cells (tilt is almost always 0) are replaced by a
per-flipper time-to-contact population (6 per flipper) with graded tau preferences.
"""
from __future__ import annotations

import numpy as np
import torch

from agents.fixed_circuit_agent import OBS_SCALE, FixedCircuitAgent

N_PER_CH = 12
CH_X, CH_Y, CH_VX, CH_VY = 0, 1, 2, 3
CH_VIS = (4, 5, 6, 7, 8, 9, 10, 11)  # LC4 vel1_L/C/R, LPLC2 size1_L/C/R, LC4 vel2, LPLC2 size2
CH_FL, CH_FR, CH_TILT = 12, 13, 14
LC4_CH = (4, 5, 6, 10)

# Flipper geometry (agents/table_map.json a_flip1 / a_flip2; x axis mirrored: action 0 at x>0).
FLIP_L = (1.7839, 12.0688)
FLIP_R = (-1.7839, 12.0688)
STEP_DT = 0.04  # seconds per decision step (frame_skip 4), same constant as T.LOOM_STEP_DT

# Position tilings. X spans the lower playfield; Y has coarse coverage of the upper table and
# 0.6-unit spacing across the flipper band (10.2-14.0, TFlipperEdge YMin/YMax) where a falling ball
# moves ~0.1-0.4 units per decision step. Centres come from table geometry, not from the reflex.
X_CENTERS = np.linspace(-7.0, 7.0, N_PER_CH)
X_SIGMA = 0.75 * (X_CENTERS[1] - X_CENTERS[0])
Y_CENTERS = np.array([-10.0, -5.0, 0.0, 4.0, 7.0, 9.0, 10.2, 10.8, 11.4, 12.0, 12.6, 13.4])
Y_SIGMA = np.maximum(0.6 * np.gradient(Y_CENTERS), 0.35)
# direction-selective speed units: 6 thresholds per direction (units / s)
SPEED_THRESH = np.array([0.5, 2.0, 5.0, 10.0, 20.0, 35.0])
SPEED_SCALE = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0])
# falling-only position units for dspop: 12 bumps over the lower half of the table
DS_Y_CENTERS = np.array([4.0, 6.0, 7.5, 8.5, 9.3, 10.0, 10.6, 11.2, 11.8, 12.4, 13.0, 13.7])
DS_Y_SIGMA = np.maximum(0.6 * np.gradient(DS_Y_CENTERS), 0.35)
# TTC units (decision steps) for "+ttc": exp(-(log tau - log tau_k)^2 / (2 * 0.5^2))
TTC_PREF = np.array([0.5, 1.0, 1.5, 2.5, 4.0, 7.0])
# retinotopic grid for "retino": 96 visual cells = 8 x 12 (x columns, y rows)
RET_X = np.linspace(-7.0, 7.0, 8)
RET_Y = np.linspace(2.0, 14.0, 12)


def _sig(z):
	return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _bumps(v, centers, sigma):
	return np.exp(-((v[:, None] - centers[None, :]) ** 2) / (2 * np.asarray(sigma)[None, :] ** 2))


def _ds_speed(v):
	"""12 cells: 6 for positive velocity, 6 for negative, graded thresholds (range fractionation)."""
	pos = _sig((v[:, None] - SPEED_THRESH[None, :]) / SPEED_SCALE[None, :])
	neg = _sig((-v[:, None] - SPEED_THRESH[None, :]) / SPEED_SCALE[None, :])
	return np.concatenate([pos, neg], axis=1)


def _ttc_pop(x, y, vx, vy, target):
	dx, dy = target[0] - x, target[1] - y
	d = np.maximum(np.hypot(dx, dy), 1e-3)
	closing = (vx * dx + vy * dy) / d
	tau = np.where(closing > 1e-3, d / np.maximum(closing, 1e-3) / STEP_DT, np.inf)
	logtau = np.log(np.maximum(tau, 1e-3))
	out = np.exp(-((logtau[:, None] - np.log(TTC_PREF)[None, :]) ** 2) / (2 * 0.5 ** 2))
	return np.where(np.isfinite(tau)[:, None], out, 0.0)


class Encoder:
	"""obs (B, 15) numpy -> per-channel (B, 15, 12) drive in [-1, 1]."""

	def __init__(self, name: str):
		base, _, extra = name.partition("+")
		if base not in ("broadcast", "gain", "popcode", "dspop", "retino"):
			raise ValueError(f"unknown encoding {name!r}")
		if extra not in ("", "ttc"):
			raise ValueError(f"unknown encoding suffix {extra!r}")
		self.name, self.base, self.ttc = name, base, extra == "ttc"
		self.scale = np.asarray(OBS_SCALE, dtype=np.float32)

	def __call__(self, obs: np.ndarray) -> np.ndarray:
		obs = np.asarray(obs, dtype=np.float64)
		B = obs.shape[0]
		scalar = np.clip(obs / self.scale, -1.0, 1.0)
		out = np.repeat(scalar[:, :, None], N_PER_CH, axis=2)
		x, y, vx, vy = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
		if self.base == "gain":
			out[:, CH_Y, :] = np.clip((y - 10.0) / 3.0, -1, 1)[:, None]
			out[:, CH_VY, :] = np.clip(vy / 10.0, -1, 1)[:, None]
		elif self.base in ("popcode", "dspop"):
			out[:, CH_X, :] = _bumps(x, X_CENTERS, np.full(N_PER_CH, X_SIGMA))
			out[:, CH_Y, :] = _bumps(y, Y_CENTERS, Y_SIGMA)
			out[:, CH_VX, :] = _ds_speed(vx)
			out[:, CH_VY, :] = _ds_speed(vy)
			if self.base == "dspop":
				out[:, CH_VY, :] = _bumps(y, DS_Y_CENTERS, DS_Y_SIGMA) * _sig((vy - 1.0) / 1.0)[:, None]
		elif self.base == "retino":
			gx = _bumps(x, RET_X, np.full(8, 1.0))  # (B, 8)
			gy = _bumps(y, RET_Y, np.full(12, 0.6))  # (B, 12)
			grid = gx[:, :, None] * gy[:, None, :]  # (B, 8 visual channels as columns, 12 rows)
			approach = _sig((vy - 1.0) / 1.0)[:, None]
			for k, ch in enumerate(CH_VIS):
				out[:, ch, :] = grid[:, k, :] * (approach if ch in LC4_CH else 1.0)
		if self.ttc:
			out[:, CH_TILT, :6] = _ttc_pop(x, y, vx, vy, FLIP_L)
			out[:, CH_TILT, 6:] = _ttc_pop(x, y, vx, vy, FLIP_R)
		return out.astype(np.float32)


class EncodedCircuitAgent(FixedCircuitAgent):
	"""FixedCircuitAgent with per-input-cell drive. Same circuit, dynamics, readout and decoder."""

	def __init__(self, graph: dict, encoding: str = "broadcast", **kw):
		self.encoder = Encoder(encoding)
		# per-cell order, grouped by channel in the graph's own order (rank tier)
		per_ch = {ch: [] for ch in range(len(graph["channels"]))}
		for cell, ch in graph["inputs"]:
			per_ch[ch].append(cell)
		assert all(len(v) == N_PER_CH for v in per_ch.values()), "expects 12 input cells per channel"
		cells = [c for ch in range(len(graph["channels"])) for c in per_ch[ch]]
		g2 = dict(graph)
		g2["inputs"] = [[c, k] for k, c in enumerate(cells)]
		g2["channels"] = [f"cell{k}" for k in range(len(cells))]
		super().__init__(g2, **kw)
		self.encoding = encoding

	@classmethod
	def from_json(cls, path, **kw) -> "EncodedCircuitAgent":
		import json
		from pathlib import Path

		return cls(json.loads(Path(path).read_text()), **kw)

	def _channel_drive(self, obs: torch.Tensor) -> torch.Tensor:
		enc = self.encoder(obs.detach().cpu().numpy())  # (B, 15, 12)
		return torch.as_tensor(enc.reshape(enc.shape[0], -1), dtype=obs.dtype, device=obs.device)


def build_encoded_agent(connectome_path: str, readout_dim: int, encoding: str):
	return EncodedCircuitAgent.from_json(connectome_path, encoding=encoding, readout_dim=readout_dim)
