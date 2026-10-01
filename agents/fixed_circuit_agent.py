"""FixedCircuitAgent: a small (~80-cell), FIXED, measured MaleCNS circuit (see
build_pinball_connectome.py) feeding a small TRAINED readout - the flyjump
(cobanov/flyjump) method, ported to PinballEnv.

Unlike agents/fly_env.py's FlyAgent (nfly's full ~139k-neuron ConnectomeRNN with millions of
learned per-edge gains, built for offline research-scale training), this circuit's connectome
weights are never trained - only the small decoder/value heads are. That keeps the whole thing
small enough to export as JSON and run as a plain forward pass in the browser (see
web/connectome.js, the JS port of the step() method below).

Deliberately duck-types nfly's FlyAgent interface (forward(obs, h, weights) -> (dist, value,
h), initial_state(batch), decoder.to_env) so nfly.rl.simple.ppo.train_ppo runs against it
completely unmodified - no new training algorithm needed, just a new small "brain".
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from nfly.agent import value_head

from agents.fly_decoder import MultiBinaryDecoder

# ball_x, ball_y, ball_vx, ball_vy, vel1_L, vel1_C, vel1_R, size1_L, size1_C, size1_R, vel2,
# size2, flipper_left, flipper_right, tilted - see env_python/pinball_env.py's
# OBS_VEL1_L/.../OBS_SIZE2 (LC4/LPLC2 looming-detector channels; ball1's vel/size are now
# retinotopic, spatially gated by a 3-bin population code over ball_x - see BIN_CENTERS). Each
# vel_* is capped/bounded at [0, 20] before the (<=1) spatial gate, each size_* at [0, 1] - same
# scale either way as the pre-retinotopic single-channel version. NOTE: this diverges from the
# old DrosophilaController's (agents/snn_model.py) 7-value OBS_SCALE, which predates these
# channels and was never updated - that model's own INPUT_DIM=7 is now stale too (see its
# comment), but it's a separate, currently-unused pipeline (train.py/export_weights.py), not
# touched here.
OBS_SCALE = (20.0, 20.0, 50.0, 50.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0)

# Bit-for-bit match to web/connectome.js's DYNAMICS constant - keep both in sync by hand;
# there is no single source of truth shared between Python and JS runtimes (same constraint the
# old SNN's inference.js already lived with).
DYNAMICS_ITERATIONS = 3
DYNAMICS_LEAK = 0.7
DYNAMICS_GAIN = 1.4


class FixedConnectome(nn.Module):
	"""Signed, leaky-tanh circuit on measured (not learned) MaleCNS connectivity. No trainable
	parameters - every weight here is either a fixed connectome-derived magnitude/sign or a
	constant dynamics hyperparameter."""

	def __init__(self, graph: dict):
		super().__init__()
		self.n = len(graph["nodes"])
		self.n_channels = len(graph["channels"])
		signs = torch.tensor([node["sign"] for node in graph["nodes"]], dtype=torch.float32)
		pre = torch.tensor([e[0] for e in graph["edges"]], dtype=torch.long)
		post = torch.tensor([e[1] for e in graph["edges"]], dtype=torch.long)
		contacts = torch.tensor([e[2] for e in graph["edges"]], dtype=torch.float32)

		# Normalize each cell's incoming weight by its total unsigned incoming contact count -
		# mirrors web/connectome.js's edge normalization exactly.
		totals = torch.zeros(self.n)
		totals.index_add_(0, post, contacts * signs[pre].abs())
		w = torch.where(totals[post] > 0, contacts * signs[pre] / totals[post].clamp_min(1e-12), torch.zeros_like(contacts))

		self.register_buffer("pre", pre)
		self.register_buffer("post", post)
		self.register_buffer("w", w)
		self.register_buffer("input_cells", torch.tensor([c for c, _ in graph["inputs"]], dtype=torch.long))
		self.register_buffer("input_channels", torch.tensor([ch for _, ch in graph["inputs"]], dtype=torch.long))

	def step(self, activity: torch.Tensor, channel_drive: torch.Tensor) -> torch.Tensor:
		"""activity: (B, n) previous cell activity. channel_drive: (B, n_channels), already
		scaled to roughly [-1, 1]. Returns the new (B, n) activity."""
		batch = activity.shape[0]
		drive = torch.zeros(batch, self.n, device=activity.device, dtype=activity.dtype)
		drive[:, self.input_cells] = channel_drive[:, self.input_channels]
		for _ in range(DYNAMICS_ITERATIONS):
			msg = torch.zeros_like(activity).index_add_(1, self.post, activity[:, self.pre] * self.w)
			activity = (1 - DYNAMICS_LEAK) * activity + DYNAMICS_LEAK * torch.tanh(drive + DYNAMICS_GAIN * msg)
		return activity


class FixedCircuitAgent(nn.Module):
	def __init__(self, graph: dict, readout_dim: int = 16, obs_scale=OBS_SCALE):
		super().__init__()
		self.brain = FixedConnectome(graph)
		self.register_buffer("obs_scale", torch.tensor(obs_scale, dtype=torch.float32))
		readout_idx = torch.tensor(graph["outputs"], dtype=torch.long)
		self.decoder = MultiBinaryDecoder(readout_idx, n_actions=3, readout_dim=readout_dim)
		self.value = value_head(self.decoder.n_features)
		self.register_buffer("h_rest", torch.zeros(self.brain.n))

	@classmethod
	def from_json(cls, path: str | Path, **kw) -> "FixedCircuitAgent":
		graph = json.loads(Path(path).read_text())
		return cls(graph, **kw)

	def initial_state(self, batch: int) -> torch.Tensor:
		return self.h_rest.unsqueeze(0).expand(batch, -1).clone()

	def weights(self):
		return None  # no derived per-unroll parameters (see ConnectomeRNN.weights()) - circuit is fixed

	def _channel_drive(self, obs: torch.Tensor) -> torch.Tensor:
		return (obs / self.obs_scale).clamp(-1.0, 1.0)

	def step(self, obs: torch.Tensor, h: torch.Tensor, weights=None) -> tuple[torch.Tensor, torch.Tensor]:
		h = self.brain.step(h, self._channel_drive(obs))
		return self.decoder.features(h), h

	def forward(self, obs: torch.Tensor, h: torch.Tensor, weights=None):
		feats, h = self.step(obs, h, weights)
		return self.decoder.distribution(feats), self.value(feats).squeeze(-1), h

	def act(self, obs: torch.Tensor, h: torch.Tensor, greedy: bool = False):
		with torch.no_grad():
			dist, _, h = self(obs, h)
			a = dist.mode if greedy else dist.sample()
		return self.decoder.to_env(a), h

	@torch.no_grad()
	def calibrate(self, obs_space: gym.Space, n_probe: int = 64, steps: int = 64, seed: int = 0) -> None:
		"""Sets the readout normalization from a probe of random-walk observations (see
		ActionDecoder.calibrate / ReadoutNorm) - the fixed circuit itself needs no calibration,
		only the trainable readout that reads it does."""
		rng = np.random.default_rng(seed)
		obs_space.seed(int(rng.integers(2**31)))
		h = self.initial_state(n_probe)
		obs = torch.as_tensor(np.stack([obs_space.sample() for _ in range(n_probe)]), dtype=torch.float32)
		states = []
		for _ in range(steps):
			obs = 0.8 * obs + 0.2 * torch.as_tensor(np.stack([obs_space.sample() for _ in range(n_probe)]), dtype=torch.float32)
			h = self.brain.step(h, self._channel_drive(obs))
			states.append(h)
		self.decoder.calibrate(torch.cat(states))

	def summary(self) -> str:
		return (f"FixedCircuitAgent: {self.brain.n} cells (fixed), {self.brain.pre.numel()} edges | "
				f"MultiBinaryDecoder <- {self.decoder.n_readout} readout cells | trainable params: "
				f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
