"""Drosophila-connectome-inspired SNN controller, built on Norse (norse.torch).

Architecture: input_linear -> hidden LIF layer -> output_linear -> output LIF layer, run for
`sim_steps` inner timesteps per environment step with the observation injected as a constant
current each step (rate/current coding rather than a separate spike-encoding layer). The
mean output firing rate over the window is used as Bernoulli probabilities for the 3 binary
actions (flipper_left, flipper_right, launch) - gradients flow back through Norse's surrogate
threshold function, so this is trainable with an ordinary policy-gradient loss.

"Drosophila-inspired" names the hidden layer after the loose analogy to a fly's central-complex
integration circuitry; this is not a claim of connectome fidelity, just a small, interpretable
spiking layer sized for a single flipper-control task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import norse.torch as snn

INPUT_DIM = 7  # matches env_python.pinball_env.OBS_DIM
OUTPUT_DIM = 3  # [flipper_left, flipper_right, launch], matches ACT_* indices
OUTPUT_LABELS = ["flipper_left", "flipper_right", "launch"]

# Rough fixed normalization for the raw world-unit observations (position/velocity aren't
# calibrated against real table extents yet - see env_python/pinball_env.py). Grounded in one
# real captured episode against DEMO.DAT: ball_x/y stayed within roughly +/-12, ball_vy hit
# ~204 during a plunger launch. Dividing by these keeps typical injected currents in a range
# the LIF layers below can actually respond to, instead of saturating on launch spikes while
# staying at exactly zero at rest.
_OBS_SCALE = (20.0, 20.0, 50.0, 50.0, 1.0, 1.0, 1.0)

# Lower thresholds than Norse's LIFParameters() default (v_th=1.0): with only sim_steps inner
# timesteps of dt=1ms each, the default threshold rarely gets crossed by a freshly-initialized
# linear layer's currents, so the network starts out never spiking (verified: an earlier version
# of this model with defaults produced a firing rate stuck at 0.0 for an entire real training run
# against the engine). Lower v_th makes exploratory spikes - and therefore a nonzero policy
# gradient - reachable from the start.
_HIDDEN_LIF_PARAMS = snn.LIFParameters(v_th=torch.as_tensor(0.5))
_OUTPUT_LIF_PARAMS = snn.LIFParameters(v_th=torch.as_tensor(0.3))


class DrosophilaController(nn.Module):
	def __init__(self, hidden_dim: int = 32, sim_steps: int = 16):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.sim_steps = sim_steps

		self.input_linear = nn.Linear(INPUT_DIM, hidden_dim)
		self.hidden_lif = snn.LIFCell(p=_HIDDEN_LIF_PARAMS)
		self.output_linear = nn.Linear(hidden_dim, OUTPUT_DIM)
		self.output_lif = snn.LIFCell(p=_OUTPUT_LIF_PARAMS)

		self.register_buffer("obs_scale", torch.tensor(_OBS_SCALE))
		# Wider initial weights than nn.Linear's default (~U(-0.38, 0.38) for fan_in=7) so the
		# bias-driven resting-state current alone has a real chance of crossing v_th - without
		# this, exploration during early training is too weak to ever discover useful actions.
		nn.init.normal_(self.input_linear.weight, mean=0.0, std=1.0)
		nn.init.normal_(self.output_linear.weight, mean=0.0, std=1.0)

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		"""obs: (batch, INPUT_DIM) -> firing_rate: (batch, OUTPUT_DIM) in [0, 1],
		used as per-action Bernoulli probabilities."""
		obs = obs / self.obs_scale

		hidden_state = None
		output_state = None
		spike_sum = torch.zeros(obs.shape[0], OUTPUT_DIM, dtype=obs.dtype, device=obs.device)

		for _ in range(self.sim_steps):
			hidden_current = self.input_linear(obs)
			hidden_spikes, hidden_state = self.hidden_lif(hidden_current, hidden_state)
			output_current = self.output_linear(hidden_spikes)
			output_spikes, output_state = self.output_lif(output_current, output_state)
			spike_sum = spike_sum + output_spikes

		return spike_sum / self.sim_steps
