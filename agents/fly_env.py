"""Vectorized, wrapped PinballEnv for nfly's rl.simple trainers.

nfly's GameSuite.finish() (nfly/suite/base.py) is the normal place this wrapping would happen,
but that requires registering PinballEnv in the gym or nfly suite registry - unnecessary
indirection for a single project-specific env. This reproduces the same two wrappers finish()
applies to any vector observation env, applied by hand around PinballEnv:

  - NormalizeObservation: PinballEnv's observation_space is unbounded (Box(-inf, inf)), so
	nfly's own VectorEncoder normalization (nfly/interface/encoders.py's _finite_bounds) is a
	no-op for it - raw ball_vx/ball_vy (up to ~200) would otherwise swamp the flipper/tilt bits
	(0 or 1) in the encoder's shared linear projection. Running-stats normalization here fixes
	that regardless of which encoder nfly picks.
  - RecordEpisodeStatistics: nfly.rl.simple.common.finished_returns() reads info["episode"],
	which only RecordEpisodeStatistics populates.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from env_python.pinball_env import ACT_FLIPPER_LEFT, ACT_FLIPPER_RIGHT, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y, OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT, PinballEnv

# Same values/reasoning as agents/train_pinball_circuit_cem.py's DRAIN_PENALTY, FLIPPER_ZONE_Y,
# PROXIMITY_BONUS, FLIPPER_HIT_MIN_VY - duplicated here rather than imported, since that module
# is PPO's sibling trainer, not a dependency of the env-wrapping layer.
_FLIPPER_ZONE_Y = 9.0
_PROXIMITY_BONUS = 0.1
_DRAIN_PENALTY = 2.0
_FLIPPER_HIT_MIN_VY = 1.0


class FlipperFocusRewardWrapper(gym.Wrapper):
	"""Replaces PinballEnv's own dense shaped reward (score_delta-dominated) with the same
	flipper-focus formula agents/train_pinball_circuit_cem.py's _evaluate() uses for CEM/CMA-ES:
	isolates reward to real flipper contact (plus proximity/drain/tilt shaping), dropping
	score_delta entirely.

	Needed because a PPO run trained on PinballEnv's unmodified reward achieved a real, sustained
	training-return improvement (388 -> ~600) yet scored ZERO real flipper_hit events across 10
	held-out episodes - it learned to keep the ball bouncing passively off bumpers for score,
	the exact failure mode --flipper-focus was invented to avoid on the CEM/CMA-ES track, just
	never previously applied to PPO's reward too. See TILT_PENALTY/FLIPPER_HIT_BONUS in
	env_python/pinball_env.py for those two terms' own values/reasoning - only DRAIN_PENALTY and
	the proximity shaping are new here, ported from _evaluate()."""

	def __init__(self, env: gym.Env):
		super().__init__(env)
		self._was_in_play = False
		self._credited_left = False
		self._credited_right = False

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		self._was_in_play = bool(info["ball_in_play"])
		self._credited_left = self._credited_right = False
		return obs, info

	def step(self, action):
		from env_python.pinball_env import FLIPPER_HIT_BONUS, TILT_PENALTY

		obs, _, terminated, truncated, info = self.env.step(action)
		reward = 0.0
		if info["tilted"]:
			reward -= TILT_PENALTY
		if info["flipper_hit"] and obs[OBS_BALL_VY] > _FLIPPER_HIT_MIN_VY:
			reward += FLIPPER_HIT_BONUS
		if self._was_in_play and not info["ball_in_play"]:
			reward -= _DRAIN_PENALTY
		self._was_in_play = info["ball_in_play"]

		# obs_y > _FLIPPER_ZONE_Y alone is NOT "near a flipper" - see
		# agents/train_pinball_circuit_cem.py's identical fix/comment: the plunger lane sits at a
		# similar Y range (x approx -7) but is nowhere near either flipper (real x =
		# -2.489/+2.489), so this alone let PROXIMITY_BONUS be farmed for free during the (very
		# common) plunger-lane dwell time. Gate on |ball_x| too.
		#
		# The zone must also be split by side (ball_x < 0 vs >= 0) - a single shared zone let a
		# policy hold ONE flipper up permanently and farm proximity credit whenever the ball
		# entered the zone from EITHER side, with zero need to ever learn the other flipper. See
		# agents/train_pinball_circuit_cem.py's identical fix for the empirical confirmation
		# (a champion trained under the shared zone always held left, never pressed right).
		in_zone_left = obs[OBS_BALL_Y] > _FLIPPER_ZONE_Y and -4.0 < obs[OBS_BALL_X] < 0.0
		in_zone_right = obs[OBS_BALL_Y] > _FLIPPER_ZONE_Y and 0.0 <= obs[OBS_BALL_X] < 4.0
		physical_left = obs[OBS_FLIPPER_LEFT] > 0.5
		physical_right = obs[OBS_FLIPPER_RIGHT] > 0.5
		if in_zone_left and physical_left and not self._credited_left:
			reward += _PROXIMITY_BONUS
			self._credited_left = True
		elif not in_zone_left:
			self._credited_left = False
		if in_zone_right and physical_right and not self._credited_right:
			reward += _PROXIMITY_BONUS
			self._credited_right = True
		elif not in_zone_right:
			self._credited_right = False

		return obs, reward, terminated, truncated, info


class ScorePerLifeRewardWrapper(gym.Wrapper):
	"""PPO reward matching the CMA/CEM track's objective (see train_pinball_circuit_cem.py): one
	episode is one ball life, reward is log-compressed real score, and the episode ends at the real drain
	(info["drained"]) rather than PinballEnv's own step/TimeLimit end. Replaces every earlier
	reward-shaping term (FLIPPER_HIT_BONUS gate, DRAIN_PENALTY, PROXIMITY_BONUS) - those measured
	the wrong events."""

	def step(self, action):
		obs, _, terminated, truncated, info = self.env.step(action)
		reward = np.log1p(max(0.0, info["score_delta"])) / 10.0
		if info["tilted"]:
			reward -= 1.0
		if info.get("drained", False):
			terminated = True
		return obs, reward, terminated, truncated, info


class LaunchCurriculumWrapper(gym.Wrapper):
	"""For the first `episodes` resets, force the flipper bits to 0 before they reach the
	engine - the policy still emits and gets gradient on all 3 action bits (nfly's decoder has
	no per-bit masking hook), but only the launch bit can affect anything, so it is the only
	behavior curriculum stage 1 can possibly reinforce. episode count is per-wrapped-instance,
	so each parallel env in the vector counts its own resets.

	NOTE: PinballEnv no longer treats launch as a learned action at all - it auto-launches every
	tick the ball isn't in play (see PinballEnv.step's `launch = not self._was_ball_in_play`),
	ignoring action[ACT_LAUNCH] entirely, following the CEM/CMA-ES track's finding that an
	explicit launch decision taught policies to idle forever rather than risk it. That makes this
	wrapper's flipper-zeroing the only thing it still does; the launch-timing curriculum it was
	originally built for no longer applies (PinballEnv.apply_launch_waste_penalty, which this
	wrapper used to toggle, was removed along with the learned launch action itself)."""

	def __init__(self, env: gym.Env, episodes: int):
		super().__init__(env)
		self.curriculum_episodes = episodes
		self._episode = 0

	def reset(self, **kwargs):
		self._episode += 1
		return self.env.reset(**kwargs)

	def step(self, action):
		if self._episode <= self.curriculum_episodes:
			action = action.copy()
			action[ACT_FLIPPER_LEFT] = 0
			action[ACT_FLIPPER_RIGHT] = 0
		return self.env.step(action)


def make_pinball_vec_env(
	binary_path: str,
	n_envs: int,
	frame_skip: int = 1,
	curriculum_episodes: int = 0,
	max_episode_steps: int = 2000,
	flipper_focus: bool = False,
	score_life: bool = False,
	normalize_obs: bool = True,
) -> gym.vector.VectorEnv:
	if score_life and flipper_focus:
		raise ValueError("score_life and flipper_focus are mutually exclusive reward wrappers")

	def thunk():
		env = PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip)
		env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
		if curriculum_episodes:
			env = LaunchCurriculumWrapper(env, episodes=curriculum_episodes)
		if score_life:
			env = ScorePerLifeRewardWrapper(env)
		if flipper_focus:
			env = FlipperFocusRewardWrapper(env)
		if normalize_obs:
			env = gym.wrappers.NormalizeObservation(env)
		env = gym.wrappers.RecordEpisodeStatistics(env)
		return env

	return gym.vector.SyncVectorEnv(
		[thunk for _ in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
	)
