"""Non-connectome baseline: raw observation -> small MLP -> flipper logits.

Exists to measure the achievable ceiling of the task/objective/optimizer with full information,
so the fixed-circuit agent's result can be judged against it (the circuit's 16 readout cells were
measured to carry little ball-velocity information). Duck-types the parts of FixedCircuitAgent
that train_pinball_circuit_cem.py uses: `.decoder` (the searched parameters), `initial_state`,
`act`, `calibrate`.
"""
from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn

from agents.fixed_circuit_agent import OBS_SCALE


class DirectMLPAgent(nn.Module):
	def __init__(self, obs_dim: int = len(OBS_SCALE), hidden: int = 32, n_actions: int = 3):
		super().__init__()
		self.register_buffer("obs_scale", torch.tensor(OBS_SCALE, dtype=torch.float32))
		self.decoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))

	def initial_state(self, batch: int) -> torch.Tensor:
		return torch.zeros(batch, 1)

	def calibrate(self, obs_space: gym.Space) -> None:
		pass

	@torch.no_grad()
	def act(self, obs: torch.Tensor, h: torch.Tensor, greedy: bool = True):
		logits = self.decoder((obs / self.obs_scale).clamp(-1.0, 1.0))
		return (logits > 0).to(torch.int64).cpu().numpy(), h
