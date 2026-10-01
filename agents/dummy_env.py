"""A synthetic stand-in for env_python.pinball_env.PinballEnv, used ONLY to verify that
agents/train.py's training loop is mechanically correct (gradients flow, the policy actually
moves, checkpoints save) without a compiled binary or a copyrighted CADET.DAT/PINBALL.DAT file.

This is NOT a pinball simulator and learning to solve it says nothing about playing pinball -
reward is an arbitrary, trivially learnable function of the action (encourage flipper_left=1)
chosen purely so a training run against it has a visible, checkable learning signal. Swap this
for env_python.pinball_env.PinballEnv the moment a real DAT file is available.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from env_python.pinball_env import ACT_FLIPPER_LEFT, OBS_DIM


class DummyPlumbingEnv(gym.Env):
	metadata = {"render_modes": []}

	def __init__(self, episode_length: int = 50, seed: int | None = None):
		super().__init__()
		self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
		self.action_space = gym.spaces.MultiBinary(3)
		self.episode_length = episode_length
		self._rng = np.random.default_rng(seed)
		self._step_count = 0

	def reset(self, *, seed: int | None = None, options: dict | None = None):
		super().reset(seed=seed)
		if seed is not None:
			self._rng = np.random.default_rng(seed)
		self._step_count = 0
		obs = self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)
		return obs, {}

	def step(self, action):
		self._step_count += 1
		obs = self._rng.uniform(-1.0, 1.0, size=OBS_DIM).astype(np.float32)
		reward = 1.0 if action[ACT_FLIPPER_LEFT] else -1.0
		terminated = False
		truncated = self._step_count >= self.episode_length
		return obs, reward, terminated, truncated, {}

	def render(self):
		return None

	def close(self):
		pass
