"""MultiBinary action decoder for nfly's FlyAgent.

nfly's built-in ActionDecoder.for_space only handles Discrete (Categorical) and Box (squashed
Gaussian) - see nfly/interface/decoders.py. PinballEnv's action_space is MultiBinary(3)
(flipper_left, flipper_right, launch): three independent button states, not mutually exclusive
choices (Discrete) or continuous values (Box). nfly's own docs describe subclassing
ActionDecoder and passing the instance to FlyAgent.build(decoder=...) for exactly this case, so
this stays entirely outside the vendored nfly package - no modification to it required.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Bernoulli, Independent

from nfly.interface.decoders import ActionDecoder


class MultiBinaryDecoder(ActionDecoder):
	"""Readout features -> independent Bernoulli per action bit (mirrors BoxDecoder's shape)."""

	def __init__(self, readout_idx: torch.Tensor, n_actions: int, readout_dim: int | None = 32):
		super().__init__(readout_idx, readout_dim)
		self.head = nn.Linear(self.n_features, n_actions)
		nn.init.zeros_(self.head.weight)
		nn.init.zeros_(self.head.bias)

	def _rebuild_heads(self, n_features: int) -> None:
		head = nn.Linear(n_features, self.head.out_features).to(self.head.weight.device)
		nn.init.zeros_(head.weight)
		nn.init.zeros_(head.bias)
		self.head = head

	def dist_inputs(self, feats: torch.Tensor) -> torch.Tensor:
		return self.head(feats)  # logits, one per action bit

	def distribution(self, feats: torch.Tensor) -> Independent:
		return Independent(Bernoulli(logits=self.dist_inputs(feats)), 1)

	def to_env(self, action: torch.Tensor):
		return action.detach().cpu().numpy().astype("int64")
