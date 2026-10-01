#!/usr/bin/env python3
"""Cross-entropy neuroevolution (CEM) for the fixed-circuit agent - the actual training method
cobanov/flyjump uses (src/lib/training.ts's Trainer), ported from JS/Dino to Python/pinball.

Why CEM instead of the PPO track in train_pinball_circuit.py: PPO's policy is a per-tick
stochastic Bernoulli with an entropy bonus that actively resists collapsing to a confident
decision (see the entropy-vs-confidence trade-off documented in fixed_circuit_agent.py's
history) - a real run got stuck with launch probability hovering ~30% forever, high enough to
rack up reward via lucky sampling but never actually deployable (browser inference thresholds
deterministically). flyjump's method sidesteps this entirely: only the CIRCUIT is fixed and only
a WEIGHT VECTOR is searched over (no gradients, no entropy term); every candidate is evaluated by
its deterministic behavior, so there is no separate "confidence" problem to solve - a good
candidate's behavior already is what gets deployed.

Only agent.decoder's parameters are searched (agent.value, the PPO track's critic, is dropped
entirely - CEM has no value function). Population evaluation is parallelized across worker
processes, each owning one persistent PinballEnv + FixedCircuitAgent (avoids re-spawning the
native engine per candidate).

Fitness is built from real game outcomes (score, tilt, flipper contact, launch success), NOT
PinballEnv's dense per-tick shaped `reward` - see _evaluate()'s docstring. A first version reused
the shaped reward directly and converged the entire population to ~50 fitness within 14
generations, sigma collapsed to 0.1: BALL_IN_PLAY_BONUS accrues every tick for the whole episode,
so "launch once, then let the ball bounce passively" was a fitness-maximizing shortcut that real
scoring behavior had no easy way to beat. That bonus exists to give PPO's gradient something to
follow when nothing else has happened yet; CEM ranks whole episodes by outcome, so it doesn't
need it and the dense accumulation actively hurts by creating that shortcut.

Usage:
	python agents/train_pinball_circuit_cem.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
		--connectome web/connectome.json --workers 4 --population 16 --generations 200
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402

_WORKER_ENV = None
_WORKER_AGENT = None


class ResilientPool:
	"""Wraps a multiprocessing Pool, rebuilding it from scratch and retrying on any worker
	failure - specifically ConnectionError ("IPC socket closed by native process"), a rare but
	real crash where the long-running native engine subprocess a worker owns dies mid-episode
	after enough uptime (observed crashing an unattended overnight run after 700+ generations).
	A standard Pool has no way to recover one dead worker's broken IPC connection short of
	recreating the whole pool, so that's what this does on any exception from map() - the
	alternative (the whole training run crashing and sitting idle until someone notices) defeats
	the point of running unattended overnight."""

	def __init__(self, ctx, pool_args: tuple, workers: int, max_retries: int = 2):
		self._ctx = ctx
		self._pool_args = pool_args
		self._workers = workers
		self._max_retries = max_retries
		self._pool = self._new_pool()

	def _new_pool(self):
		return self._ctx.Pool(processes=self._workers, initializer=_worker_init, initargs=self._pool_args)

	def map(self, func, tasks):
		for attempt in range(self._max_retries + 1):
			try:
				return self._pool.map(func, tasks)
			except Exception as e:
				self._pool.terminate()
				self._pool.join()
				if attempt >= self._max_retries:
					print(f"  worker pool error ({e!r}), out of retries ({self._max_retries}) - giving up", flush=True)
					raise
				print(f"  worker pool error ({e!r}) - recreating pool and retrying "
				      f"(attempt {attempt + 2}/{self._max_retries + 1})", flush=True)
				self._pool = self._new_pool()

	def terminate(self) -> None:
		self._pool.terminate()

	def join(self) -> None:
		self._pool.join()

	def close(self) -> None:
		self._pool.close()


def get_flat_params(decoder: torch.nn.Module) -> np.ndarray:
	return torch.cat([p.detach().flatten() for p in decoder.parameters()]).numpy()


def set_flat_params(decoder: torch.nn.Module, flat: np.ndarray) -> None:
	flat_t = torch.as_tensor(flat, dtype=torch.float32)
	offset = 0
	with torch.no_grad():
		for p in decoder.parameters():
			n = p.numel()
			p.copy_(flat_t[offset:offset + n].view_as(p))
			offset += n


def _worker_init(binary_path: str, connectome_path: str, readout_dim: int, frame_skip: int, max_steps: int) -> None:
	global _WORKER_ENV, _WORKER_AGENT
	import atexit
	import signal

	import gymnasium as gym

	from env_python.pinball_env import PinballEnv

	# Without this, each worker's own torch ops default to using every visible core for internal
	# threading - fine with a tiny circuit (each op finishes near-instantly regardless), but with
	# a several-hundred-node circuit N_WORKERS processes x N_CORES threads each massively
	# oversubscribes the machine and workers spend most of their time context-switching instead
	# of computing (confirmed: CPU sat at ~90% per worker yet a single generation never finished
	# in over 7 minutes). Parallelism already comes from the worker processes themselves.
	torch.set_num_threads(1)

	_WORKER_AGENT = FixedCircuitAgent.from_json(connectome_path, readout_dim=readout_dim)
	env = PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip)
	env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
	_WORKER_ENV = env
	_WORKER_AGENT.calibrate(env.observation_space)

	# Pool.terminate()/an interrupted parent kills workers without giving them a chance to run
	# `finally` blocks, and SIGTERM's default action skips atexit hooks too - without an explicit
	# handler the native engine subprocess this worker owns gets orphaned (reparented to pid 1,
	# left spinning at ~100% CPU forever; confirmed with a real interrupted run). Catching SIGTERM
	# and routing it through sys.exit() makes atexit actually run.
	def _on_terminate(signum, frame):
		sys.exit(0)

	signal.signal(signal.SIGTERM, _on_terminate)
	atexit.register(env.close)


# CEM_LAUNCH_BONUS and LAUNCH_DELAY_PENALTY (both since removed) rewarded/penalized the
# not-in-play<->in-play transition back when launching was a decision the agent's own decoder
# output controlled. It no longer is: PinballEnv.step() now auto-launches every tick the ball
# isn't in play (see its docstring), following CMU's PinBot report (2024) - they found giving an
# agent an explicit, learnable launch action taught it to idle forever rather than risk the
# eventual drain penalty, even though launching is unconditionally required to ever score, and
# fixed it by removing the decision entirely rather than trying to shape around it. With launch
# no longer a choice, those two terms would just add an identical constant to every candidate's
# fitness every episode - harmless to CEM's ranking, but measuring nothing about the policy.

# Fixed table-coordinate threshold above which the ball is near the flippers.
#
# The previous value (-6.0, with an obs_y < threshold check) was wrong by construction: it was
# eyeballed from one rollout's y-range without ever confirming it against the engine's real
# flipper geometry. Ground truth, read directly from TFlipperEdge::XMin/XMax/YMin/YMax at
# construction time (see a one-off debug print in TFlipper.cpp, since reverted) via the real
# DEMO.DAT: both flippers sit at YMin=10.18/YMax=13.95 - i.e. HIGH positive Y, the opposite end
# of the table from where the old zone was checking. That single sign error explains why a real
# CEM run plateaued at a fixed low fitness for 130 generations, and why a targeted diagnostic
# (diagnose_flipper_geometry.py) found literally zero real flipper_hit events across ~22,000
# ticks spent with obs_y < -6.0 while both flippers toggled continuously: that region has no
# flippers in it at all. Set a little below the real YMin for margin, same spirit as the old
# constant's margin above its (wrong) empirical threshold - this is still a shaping signal, not
# a ground-truth hit test.
FLIPPER_ZONE_Y = 9.0

# Small reward for pressing a flipper while the ball is in the flipper zone, even without a hit.
# CEM has no notion of "close" - a candidate either hits the ball (FLIPPER_HIT_BONUS) or doesn't,
# nothing in between - so pure random search has to land an exact hit by chance in a ~190/800-
# tick window to ever see a fitness gradient at all (confirmed: zero hits across ~5,000 evaluated
# episodes). This gives it a slope to climb instead of a cliff.
#
# Capped at ONE credit per flipper per zone VISIT (not per press, not per tick): a first version
# was edge-triggered (fired on every off->on transition) meaning to stop a held press from
# out-earning real hits, but a candidate found it could instead rapidly TOGGLE the flipper the
# whole time the ball was in the zone, generating dozens of edges per episode (confirmed: a real
# champion scored 26.75 from ~53 proximity credits and zero actual flipper_hit ticks - pure
# oscillation, not timing). Limiting to one credit per continuous zone-visit removes that: the
# exploitable ceiling is now bounded by how many times the ball can plausibly revisit the zone in
# one episode (a handful), not by how fast the policy can flicker its own output.
#
# Cut from 0.5 to 0.1: the cap above stopped the *toggling* exploit, but a milder version of the
# same problem survived - "hold both flippers up whenever the ball is anywhere in the zone" earns
# the full 2x0.5=1.0 credit without ever needing to be correctly timed against the ball (the zone
# check is a coarse y-threshold, not a real hit test), and 1.0 stacked with CEM_LAUNCH_BONUS was
# enough on its own to plateau a real run at 6.0 fitness for 130 generations with zero actual
# contact. At 0.1 this is barely a slope to climb, not a destination worth converging to.
PROXIMITY_BONUS = 0.1

# Penalty for losing the ball (the in-play -> not-in-play edge, i.e. a drain), per PinBot's
# R_ball term (-0.3 on their own reward scale). Distinct from TILT_PENALTY - draining is normal,
# expected gameplay, not a foul, so this is sized closer to FLIPPER_HIT_BONUS (2.0) than to
# TILT_PENALTY (5.0): losing a ball should cost roughly one good hit's worth of fitness, enough to
# make "keep the ball alive" a first-order pressure distinct from just chasing score_delta, without
# being severe enough to make survival alone (versus real scoring) the dominant strategy.
DRAIN_PENALTY = 2.0

# Minimum post-collision ball_vy (Y increases toward/past the flippers, away from the drain near
# the origin - see FLIPPER_ZONE_Y's comment) for a real engine-registered flipper_hit to actually
# count. The user watched a live demo and noticed real "hits" where the ball just kept rolling
# down anyway - confirmed via telemetry: of 5 real flipper_hit events sampled, post-hit ball_vy
# was [-14.06, 0.79, 9.30, 13.62, 18.30] - one hit left the ball MOVING TOWARD THE DRAIN faster
# than before, one barely moved it, and only 3 were genuine, powerful upward redirects. The
# engine's own FlipperCollision fires on any contact during an active swing (see project memory),
# regardless of whether the resulting trajectory is actually useful - this gate makes the fitness
# signal match what a human watching would call "a real hit", not just "contact happened".
FLIPPER_HIT_MIN_VY = 1.0

# Behavior descriptor for novelty search/quality-diversity (--novelty): a coarse, normalized
# visitation histogram of ball position over the episode. Bounds are generous rather than exact
# (real table extent was never fully surveyed) - only relative distances between descriptors
# matter for novelty, not calibrated coordinates. Y follows this engine's convention where HIGH
# positive Y is near the flippers (see FLIPPER_ZONE_Y's comment: real flipper YMin/YMax is
# 10.18/13.95), so the range below covers rest/launch (~0) through past the flippers with margin.
NOVELTY_X_RANGE = (-10.0, 10.0)
NOVELTY_Y_RANGE = (-2.0, 18.0)
NOVELTY_X_BINS = 6
NOVELTY_Y_BINS = 6
NOVELTY_DESCRIPTOR_DIM = NOVELTY_X_BINS * NOVELTY_Y_BINS

# Reward-shaping redesign, round 2 (2026-09-25, re-added after round 1 - see
# agents/experiments/reward_redesign_2026-09-25.py for round 1's full design/rationale and why it
# was pulled back out): round 1's 150-generation/population-10 local validation showed the
# real-hit-rate telemetry completely FLAT (0.052/0.056/0.051 across thirds) - no trend either way.
# Two candidate explanations: (a) the budget was too small for CMA-ES to discover the swing-credit
# gradient at all in a 643-dim search, or (b) SWING_BONUS=0.3 was too weak a signal relative to
# fitness noise from the other terms (DRAIN_PENALTY=2.0, TILT_PENALTY=5.0, FLIPPER_HIT_BONUS=2.0)
# to matter even if discovered. This round tests (b) directly by raising SWING_BONUS well above
# round 1's value while ALSO running at a meaningfully larger budget (see the validation run's
# --population/--generations) - if hits still don't trend up here, that's stronger evidence the
# problem is the mechanism/geometry itself, not just under-budgeting.
#
# Real flipper origins (see FLIPPER_ZONE_Y's comment: TFlipperEdge ground truth) - left at
# x=-2.489, right at x=+2.489. Used below for a much tighter swing-credit window than
# PROXIMITY_BONUS's +-4.0 zone, since swing credit is meant to approximate "a real press attempt
# against a ball actually at the flipper", not merely "somewhere in the flipper's half of the
# table".
FLIPPER_ORIGIN_X_LEFT = -2.489
FLIPPER_ORIGIN_X_RIGHT = 2.489
SWING_ZONE_HALF_WIDTH = 2.0

# Credit for the PHYSICAL press EDGE (released -> pressed transition, not a sustained hold) while
# the ball is tightly near that flipper's real origin and moving toward the drain (obs_vy < 0,
# i.e. needs to be hit back rather than already receding from a prior hit). This exists because
# TFlipper::FlipperCollision only registers a hit while the flipper is actively ROTATING
# (deltaAngle != 0 - see FLIPPER_HIT_BONUS's comment in env_python/pinball_env.py); a flipper
# already up and holding static has deltaAngle=0 and can never produce a real hit no matter how
# long it holds or how close the ball is. Confirmed empirically on the live gen-1655 production
# champion (round 1's measurement): it holds a flipper up 3-25% of ticks most episodes yet
# produces close to zero gated hits - it is optimizing PROXIMITY_BONUS's static-hold credit, which
# structurally cannot lead to real contact.
#
# Raised from round 1's 0.3 to 1.0 (half of FLIPPER_HIT_BONUS=2.0, comparable in magnitude to
# DRAIN_PENALTY) - round 1's flat result left open whether the mechanism doesn't work or the
# signal was just too weak to matter against other terms' noise; this makes it strong enough that
# if CMA-ES can find the gradient at all, it should be visible in the telemetry trend within a
# few dozen generations rather than needing hundreds.
SWING_BONUS = 1.0

# ROUND 3 FIX: round 2 (MAX_SWING_CREDITS_PER_VISIT, re-armed on zone EXIT, mirroring
# PROXIMITY_BONUS's credited_left/right pattern) was confirmed exploited. Diagnostic on the round-2
# champion (generation 130): n_press_events_l=130-215 per 800-tick episode (mean press length 1.0
# RL-step - i.e. toggling on essentially every decision) with real_hits still 0-2 (unchanged
# baseline) yet swing credit being earned repeatedly. Root cause: PROXIMITY_BONUS's zone-visit
# re-arming works because a HELD flipper only produces one continuous visit per genuine ball
# approach - but this is edge-triggered, and the ball's natural bounce carries it back and forth
# across the zone's Y/X threshold many times per episode regardless of real timing, re-arming the
# per-visit cap far more often than real approaches occur. Rapid continuous toggling then wins
# credit on every re-arm for free.
#
# Fix: replace the zone-visit-based re-arm with a COOLDOWN measured in ticks elapsed since the
# last CREDITED swing on that side (not since the last zone visit) - this can only be reset by
# genuinely waiting, not by the zone boundary flickering as the ball bounces. SWING_COOLDOWN_TICKS
# is in RL-decision-step units (each step already covers frame_skip=4 native engine ticks) - set
# high enough that even the observed round-2 toggle rate (~1 press-event per 4-6 RL-steps) cannot
# cash in more than once per genuine ball approach, while still being short enough that two
# genuinely separate approaches within one episode can each earn credit.
SWING_COOLDOWN_TICKS = 25

# Score, at score_weight=1.0, is dominated by passive ball-vs-bumper bounces (measured: 12 fresh
# episodes of the live production champion had score std ~334 against a flipper-skill (non-score)
# term averaging ~-0.7 - score noise alone is ~450x the entire flipper-skill signal at linear
# scaling). log1p compresses that heavy-tailed luck while still rewarding genuinely higher scores:
# a 10x score gap becomes log(10) ~= 2.3 fitness (SCORE_LOG_SCALE=1.0), comparable to one real
# FLIPPER_HIT_BONUS rather than swamping it by two orders of magnitude. Applied ONCE per episode to
# the episode's total score_delta (not per-tick, unlike the old linear term), since log is only
# well-behaved applied to a whole accumulated quantity.
SCORE_LOG_SCALE = 1.0


def _evaluate(task: tuple) -> tuple[float, np.ndarray, dict]:
	"""Runs one deterministic episode with the given decoder weights, returns (fitness,
	behavior_descriptor, telemetry) built from real game outcomes only - see this module's
	docstring for why PinballEnv's own shaped `reward` (returned but unused here) is the wrong
	signal for CEM specifically. No launch-waste penalty is needed: a redundant launch=1 while
	already in play is a guarded no-op at the engine level (see ipc_protocol.h's
	relaunch_pending field) with zero effect on any of these outcomes - the penalty only ever
	mattered for gradient credit assignment, which CEM doesn't do. `telemetry` is a small dict of
	raw episode stats (real_hits, tilt_ticks, drains, prox_credits, swing_credits, ticks, score,
	p_left, p_right) for population-level logging - see the generation log line in _run_cma/main.

	`task` is (flat_weights, score_weight, seed): score_weight scales how much real score_delta
	contributes to fitness, from 0.0 (pure flipper-focus) to 1.0 (full score) - see
	--score-curriculum-gens, which ramps this across generations instead of an all-or-nothing
	--flipper-focus switch, letting a run that already learned real flipper contact under a pure
	flipper-focus objective gradually be steered back toward maximizing actual score once that
	sub-skill exists to build on. `seed` (may be None) is passed straight to env.reset(seed=...) -
	None draws a fresh OS-random seed as before; an explicit int lets callers implement common
	random numbers (CRN) across a generation's population, see _run_cma's seed rotation."""
	from env_python.pinball_env import (
		FLIPPER_HIT_BONUS,
		OBS_BALL_VY,
		OBS_BALL_X,
		OBS_BALL_Y,
		OBS_FLIPPER_LEFT,
		OBS_FLIPPER_RIGHT,
		TILT_PENALTY,
	)

	flat_weights, score_weight, seed = task
	agent, env = _WORKER_AGENT, _WORKER_ENV
	set_flat_params(agent.decoder, flat_weights)
	obs, info = env.reset(seed=seed)
	h = agent.initial_state(1)
	fitness = 0.0
	episode_score_delta = 0.0
	was_in_play = False
	credited_left = credited_right = False  # re-armed only when the ball leaves the zone
	# Start at the cooldown value (not 0) so a genuine opportunity in the first few ticks of an
	# episode isn't blocked by an artificial startup delay - see SWING_COOLDOWN_TICKS's comment.
	ticks_since_swing_left = ticks_since_swing_right = SWING_COOLDOWN_TICKS
	prev_physical_left = prev_physical_right = False
	real_hits = tilt_ticks = drains = prox_credits = swing_credits = press_left_ticks = press_right_ticks = 0
	visits = np.zeros(NOVELTY_DESCRIPTOR_DIM, dtype=np.float64)
	x_lo, x_hi = NOVELTY_X_RANGE
	y_lo, y_hi = NOVELTY_Y_RANGE
	n_ticks = 0
	done = False
	with torch.no_grad():
		while not done:
			obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
			action, h = agent.act(obs_t, h, greedy=True)
			obs, reward, terminated, truncated, info = env.step(action[0])
			# See --flipper-focus/--score-curriculum-gens docstrings above: real score is
			# dominated by passive ball-vs-bumper bounces the policy never caused and can't
			# control (confirmed empirically: a 200-episode real-play sample averaged only 0.58
			# flipper_hit events per episode, with 69.5% of episodes landing ZERO real hits, yet
			# scores ranging 3,500-325,750). score_weight lets a run isolate fitness to the one
			# sub-skill that was never being learned (score_weight=0), a compound objective
			# (score_weight=1) that conflates it with survival/launching/luck, or anywhere between.
			# Accumulated raw here, log-compressed once at episode end (see SCORE_LOG_SCALE) -
			# linear per-tick scaling let score noise (measured std ~334 on the live champion)
			# swamp the entire flipper-skill signal (non-score mean ~-0.7) by two orders of
			# magnitude at score_weight=1.0.
			episode_score_delta += info["score_delta"]
			if info["tilted"]:
				fitness -= TILT_PENALTY
				tilt_ticks += 1
			# Graded, not all-or-nothing: the old hard FLIPPER_HIT_MIN_VY gate rejected 100% of
			# real hits measured on the live production champion (12 fresh episodes, 5 raw hits,
			# 0 passing the gate) - meaning FLIPPER_HIT_BONUS was structurally unreachable as a
			# training signal even though real contact was happening. A weak redirect still beats
			# no contact; a powerful one still earns full value.
			if info["flipper_hit"]:
				graded_hit_credit = float(np.clip(obs[OBS_BALL_VY] / 10.0, 0.0, 1.0))
				fitness += FLIPPER_HIT_BONUS * graded_hit_credit
				if obs[OBS_BALL_VY] > FLIPPER_HIT_MIN_VY:
					real_hits += 1  # telemetry only - "a human watching would call this a real hit"
			if was_in_play and not info["ball_in_play"]:
				fitness -= DRAIN_PENALTY
				drains += 1
			was_in_play = info["ball_in_play"]

			# Proximity credit is based on the PHYSICAL flipper state the engine actually acted
			# on (post press/release-hold constraint - see PinballEnv.MIN_FLIPPER_PRESS_TICKS),
			# not the raw requested action, so a request that never produced a real press can't
			# earn credit.
			#
			# obs_y > FLIPPER_ZONE_Y alone is NOT "near a flipper" - the plunger lane sits at a
			# similar Y range (x approx -7, ball rests/climbs there after every launch) but is
			# nowhere near either flipper (real x = -2.489/+2.489). Confirmed via real telemetry:
			# 88.8% of all obs_y>FLIPPER_ZONE_Y ticks during real gameplay had the ball at
			# |x|>5.6, i.e. sitting in the plunger lane, not the flipper zone - meaning
			# PROXIMITY_BONUS could be farmed for free by holding a flipper up during the (very
			# common) plunger-lane dwell time, with zero connection to real flipper-timing skill.
			# Gating on |ball_x| too (generous margin beyond the real +-2.489 flipper positions)
			# excludes the plunger lane while still covering genuine flipper-zone approaches.
			#
			# CRITICAL: the zone must also be split by side (ball_x < 0 vs >= 0), not shared - a
			# single shared zone let a policy that holds ONE flipper up permanently (ignoring the
			# other entirely) farm proximity credit for free every time the ball entered the zone
			# from EITHER side. Confirmed empirically: the champion accepted under the old shared
			# zone always held the left flipper up (P(flip_left)=1.0) and never once pressed the
			# right (P(flip_right)=0.0) regardless of ball position - a free, skill-independent
			# reward loophole, not real flipper-timing skill. Splitting the zone means a
			# permanently-held flipper only earns credit while the ball is actually on that side.
			in_zone_left = obs[OBS_BALL_Y] > FLIPPER_ZONE_Y and -4.0 < obs[OBS_BALL_X] < 0.0
			in_zone_right = obs[OBS_BALL_Y] > FLIPPER_ZONE_Y and 0.0 <= obs[OBS_BALL_X] < 4.0
			physical_left = obs[OBS_FLIPPER_LEFT] > 0.5
			physical_right = obs[OBS_FLIPPER_RIGHT] > 0.5
			press_left_ticks += int(physical_left)
			press_right_ticks += int(physical_right)
			if in_zone_left and physical_left and not credited_left:
				fitness += PROXIMITY_BONUS
				credited_left = True
				prox_credits += 1
			elif not in_zone_left:
				credited_left = False
			if in_zone_right and physical_right and not credited_right:
				fitness += PROXIMITY_BONUS
				credited_right = True
				prox_credits += 1
			elif not in_zone_right:
				credited_right = False

			# Swing credit: the physical press EDGE (released -> pressed transition) inside a
			# tight window around that flipper's real origin, while the ball is moving toward the
			# drain (needs to be hit back, not already receding from a prior hit) - see
			# SWING_BONUS's module-level comment for why this targets the one motion that can
			# actually produce a real hit, unlike PROXIMITY_BONUS's static hold. Cooldown-gated
			# (SWING_COOLDOWN_TICKS's comment), not zone-visit-gated - round 2 confirmed a
			# zone-visit-based cap gets re-armed by the ball's own natural bounce across the zone
			# boundary, letting rapid continuous toggling farm credit with zero real timing.
			near_left_origin = abs(obs[OBS_BALL_X] - FLIPPER_ORIGIN_X_LEFT) < SWING_ZONE_HALF_WIDTH
			near_right_origin = abs(obs[OBS_BALL_X] - FLIPPER_ORIGIN_X_RIGHT) < SWING_ZONE_HALF_WIDTH
			approaching = obs[OBS_BALL_VY] < 0
			left_edge = physical_left and not prev_physical_left
			right_edge = physical_right and not prev_physical_right
			ticks_since_swing_left += 1
			ticks_since_swing_right += 1
			if in_zone_left and near_left_origin and approaching and left_edge and ticks_since_swing_left >= SWING_COOLDOWN_TICKS:
				fitness += SWING_BONUS
				swing_credits += 1
				ticks_since_swing_left = 0
			if in_zone_right and near_right_origin and approaching and right_edge and ticks_since_swing_right >= SWING_COOLDOWN_TICKS:
				fitness += SWING_BONUS
				swing_credits += 1
				ticks_since_swing_right = 0
			prev_physical_left, prev_physical_right = physical_left, physical_right

			bx = int(np.clip((obs[OBS_BALL_X] - x_lo) / (x_hi - x_lo) * NOVELTY_X_BINS, 0, NOVELTY_X_BINS - 1))
			by = int(np.clip((obs[OBS_BALL_Y] - y_lo) / (y_hi - y_lo) * NOVELTY_Y_BINS, 0, NOVELTY_Y_BINS - 1))
			visits[by * NOVELTY_X_BINS + bx] += 1.0
			n_ticks += 1
			done = terminated or truncated
	fitness += score_weight * SCORE_LOG_SCALE * float(np.log1p(max(0.0, episode_score_delta)))
	descriptor = (visits / n_ticks) if n_ticks else visits
	telemetry = dict(
		real_hits=real_hits, tilt_ticks=tilt_ticks, drains=drains, prox_credits=prox_credits,
		swing_credits=swing_credits, ticks=n_ticks, score=episode_score_delta,
		p_left=(press_left_ticks / n_ticks) if n_ticks else 0.0,
		p_right=(press_right_ticks / n_ticks) if n_ticks else 0.0,
	)
	return fitness, descriptor.astype(np.float32), telemetry


def gaussian_weights(rng: np.random.Generator, n: int, scale: float = 0.5) -> np.ndarray:
	return rng.standard_normal(n).astype(np.float32) * scale


def score_weight_for_generation(args: argparse.Namespace, generation: int, curriculum_start_gen: int) -> float:
	"""Real-score contribution to fitness at this generation (see _evaluate's `score_weight`).
	--score-curriculum-gens ramps 0.0 (pure flipper-focus) to 1.0 (full score) linearly over the
	given number of generations since curriculum_start_gen, not over the absolute generation
	counter - resuming from, say, generation 2111 would otherwise saturate the ramp to 1.0 on the
	very first generation, defeating the whole point of a gradual reintroduction.

	curriculum_start_gen is DELIBERATELY a separate value from the resume's start_gen (the loop's
	own starting generation): it is persisted in the checkpoint and carried forward unchanged
	across resumes, so a crash/session-loss recovery (which only bumps start_gen) does not also
	reset the curriculum ramp back to 0.0. An earlier version conflated the two - every process
	restart (this project's watchdog relaunches on ANY crash, including transient IPC errors)
	silently reset score_w to 0.00 and re-ramped over --score-curriculum-gens generations, while
	the persisted `champion_fitness` stayed on the score_w=1.0 scale the whole time: no candidate
	evaluated during that reset window could possibly be promoted (fitness is computed at whatever
	the reduced score_weight is that generation, never comparable to a champion_fitness measured
	at score_weight=1.0), and gens_since_improvement (also reset per-process) could climb toward
	--ipop-patience and fire a restart under a half-scaled objective. The idea being that once a
	run has learned real flipper contact under an isolated flipper-focus objective, real score can
	be reintroduced gradually so the search builds on that sub-skill instead of the score signal
	drowning it out again immediately - a one-time ramp for the run's whole lifetime, not a ramp
	that silently re-triggers on every crash."""
	if not args.flipper_focus:
		return 1.0
	if args.score_curriculum_gens <= 0:
		return 0.0
	return min(1.0, (generation - curriculum_start_gen) / args.score_curriculum_gens)


class NoveltyArchive:
	"""Bounded set of past behavior descriptors for novelty search/quality-diversity (--novelty).
	Each candidate's search-guiding score is its real fitness plus a bonus for how far its
	behavior descriptor sits from previously-seen behaviors (mean distance to the k nearest
	archive entries) - this only steers what the search tries next (es.tell()/CEM recombination);
	champion acceptance always checks real fitness alone (see the validation-gate logic in both
	optimizer loops), so novelty can never promote a genuinely worse candidate to champion."""

	def __init__(self, k: int, max_size: int, rng: np.random.Generator):
		self.k = k
		self.max_size = max_size
		self.rng = rng
		self._descriptors: list[np.ndarray] = []

	def novelty(self, descriptor: np.ndarray) -> float:
		if not self._descriptors:
			return 0.0
		archive = np.stack(self._descriptors)
		dists = np.linalg.norm(archive - descriptor[None, :], axis=1)
		k = min(self.k, len(dists))
		return float(np.mean(np.sort(dists)[:k]))

	def add(self, descriptor: np.ndarray) -> None:
		if len(self._descriptors) < self.max_size:
			self._descriptors.append(descriptor)
		else:
			# Reservoir-style random eviction keeps the archive an unbiased sample of behaviors
			# seen across the whole run, instead of just the most recent ones once it fills up.
			idx = int(self.rng.integers(0, self.max_size))
			self._descriptors[idx] = descriptor


# Combines vel (LC4) and size (LPLC2) the same way Ache et al. 2019 (Curr. Biol.) model the real
# Giant Fiber's looming response - a weighted sum of the two. BruceLanLan/c3s-reflex-circuits'
# NAND-netlist synthesis of this exact circuit reports a real measured synapse ratio of ~1.2-1.4
# :1 looming-speed to looming-size drive onto the giant fiber - VEL_SIZE_RATIO applies that ratio
# to the two signals normalized onto the SAME [0, 1] scale first (vel's own [0, 20] cap divides
# it down), since the ratio describes relative drive strength, not our own arbitrarily-chosen
# raw units. An earlier version multiplied size directly by 10.0 (guesswork, to bring its [0, 1]
# range up onto vel's raw [0, 20] one) without this normalization step - a ~7-8x overweighting of
# size relative to the real ratio. This circuit has essentially no redundancy to a wrong ratio -
# chris017/fly-connectome-escape-circuit found silencing just 5% of either LC4 or LPLC2 collapses
# real escape probability from ~88% to ~9% - which plausibly explains why that overweighted
# version of this heuristic produced a WORSE imitation-learning baseline (0.08 mean flipper_hits,
# 95.5% zero-hit episodes over 200 real episodes) than the original unsplit loom signal (0.58
# mean hits, 69.5% zero-hit).
VEL_CAP = 20.0
VEL_SIZE_RATIO = 1.3
ESCAPE_THRESHOLD = 0.5


def _heuristic_action(obs: np.ndarray, ball_in_play: bool) -> np.ndarray:
	"""Hand-coded reflex, not learned: press the flipper on the ball's side once its combined
	LC4 (velocity) + LPLC2 (size) escape signal crosses threshold, launch whenever the ball isn't
	in play. This is the same kind of thing FlyBrain-HalfLife hand-wires from known fly reflexes
	(steer toward optical flow, back up when stuck) rather than discovering through training - we
	have no equivalent *known* circuit for flipper timing to wire up, but we can still hand-write
	a crude version of the target behavior and use it to seed the search / imitation-learn from,
	instead of starting CEM from pure noise.

	Uses vel1_*/size1_*/vel2/size2 (see env_python.pinball_env._velocity_signal/_size_signal/
	_spatial_weight) rather than a static ball_y threshold, so it reacts to genuine approach (and,
	near the size signal's peak distance, imminent proximity) rather than merely "ball is
	somewhere in this y-range, moving or not." Ball1's channels are now retinotopic (3 spatial
	bins, see BIN_CENTERS) - this heuristic still uses raw ball_x for left/right (simpler and
	exact, still available in obs regardless), summing the 3 bins back into one aggregate escape
	signal, rather than trying to infer side FROM the bin pattern the way the trained circuit
	itself has to. Crucially, this is the first version of this heuristic that reacts to ball2 at
	all: previous versions were built before multiball observability existed and simply couldn't.
	vel2/size2 carry no left/right information (that's a real property of looming detection, not
	a shortcut we're skipping - see build_pinball_connectome.py's VEL_CELL_TYPES/SIZE_CELL_TYPES
	comment), so the only obs-consistent reaction to them is symmetric: press both flippers.
	Deliberately does NOT use ball2_x/y from the raw State to break that tie, even though
	ipc_client.py exposes it - doing so would make the imitation-learning label depend on
	information the trained decoder's own observation vector doesn't contain, an unlearnable/
	inconsistent target."""
	from env_python.pinball_env import (
		OBS_BALL_X,
		OBS_SIZE1_C,
		OBS_SIZE1_L,
		OBS_SIZE1_R,
		OBS_SIZE2,
		OBS_VEL1_C,
		OBS_VEL1_L,
		OBS_VEL1_R,
		OBS_VEL2,
	)

	# Real engine origins (see FLIPPER_ZONE_Y's comment): left flipper at x=-2.489, right
	# flipper at x=+2.489 - so the LEFT flipper is the one that reaches a ball on the negative-x
	# side, not positive.
	vel1 = obs[OBS_VEL1_L] + obs[OBS_VEL1_C] + obs[OBS_VEL1_R]
	size1 = obs[OBS_SIZE1_L] + obs[OBS_SIZE1_C] + obs[OBS_SIZE1_R]
	escape1 = (vel1 / VEL_CAP) * VEL_SIZE_RATIO + size1
	ball1_closing = escape1 > ESCAPE_THRESHOLD
	flip_left = ball1_closing and obs[OBS_BALL_X] <= 0
	flip_right = ball1_closing and obs[OBS_BALL_X] > 0
	escape2 = (obs[OBS_VEL2] / VEL_CAP) * VEL_SIZE_RATIO + obs[OBS_SIZE2]
	if escape2 > ESCAPE_THRESHOLD:
		flip_left = flip_right = True
	return np.array([flip_left, flip_right, not ball_in_play], dtype=np.int64)


def heuristic_seed(binary_path: str, connectome_path: str, readout_dim: int, frame_skip: int,
					episodes: int = 6, max_steps: int = 800, epochs: int = 300, lr: float = 1e-2,
					seed: int = 0) -> np.ndarray:
	"""Drives the real engine with `_heuristic_action` (not the agent) to collect matched
	(circuit-readout, heuristic-label) pairs, then supervised-fits agent.decoder to approximate
	that heuristic. Returns the fitted decoder's flat parameter vector, for use as CEM's
	starting champion instead of random noise - CEM then only has to REFINE a crude reflex
	into precise timing, not discover the whole behavior blind. Not guaranteed to be a GOOD
	starting point (the heuristic itself is crude, and results vary run to run) - low risk
	either way, since generation 1's population also includes perturbed and fully-random
	candidates that normal CEM selection will prefer if the seed turns out mediocre."""
	torch.manual_seed(seed)
	import gymnasium as gym

	from env_python.pinball_env import PinballEnv

	agent = FixedCircuitAgent.from_json(connectome_path, readout_dim=readout_dim)
	env = gym.wrappers.TimeLimit(
		PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip), max_episode_steps=max_steps
	)
	agent.calibrate(env.observation_space)

	all_feats, all_labels = [], []
	was_in_play = False
	try:
		with torch.no_grad():
			for _ in range(episodes):
				obs, info = env.reset()
				h = agent.initial_state(1)
				was_in_play = bool(info["ball_in_play"])
				done = False
				while not done:
					label = _heuristic_action(obs, was_in_play)
					obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
					feats, h = agent.step(obs_t, h)
					all_feats.append(feats[0])
					all_labels.append(label)
					obs, reward, terminated, truncated, info = env.step(label.astype(bool))
					was_in_play = bool(info["ball_in_play"])
					done = terminated or truncated
	finally:
		env.close()

	feats = torch.stack(all_feats)
	labels = torch.as_tensor(np.stack(all_labels), dtype=torch.float32)
	pos_counts = labels.sum(0)
	neg_counts = labels.shape[0] - pos_counts
	print(f"heuristic_seed: collected {feats.shape[0]} labeled ticks across {episodes} episodes "
	      f"(positive rate: flip_left {pos_counts[0]/labels.shape[0]:.3f}, "
	      f"flip_right {pos_counts[1]/labels.shape[0]:.3f}, launch {pos_counts[2]/labels.shape[0]:.3f})", flush=True)

	# "launch" is true for a handful of ticks per episode (right after reset/drain) out of
	# hundreds where it's false - unweighted BCE lets the optimizer trivially predict "never
	# launch" and still score ~99% aggregate accuracy (confirmed: a first version did exactly
	# this, matched the heuristic's own labels at 98%, then launched zero times in fresh
	# rollouts). pos_weight rebalances each action head's gradient by its own class imbalance.
	pos_weight = (neg_counts / pos_counts.clamp_min(1)).clamp(max=50.0)
	opt = torch.optim.Adam(agent.decoder.parameters(), lr=lr)
	loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	for epoch in range(epochs):
		logits = agent.decoder.dist_inputs(feats)
		loss = loss_fn(logits, labels)
		opt.zero_grad()
		loss.backward()
		opt.step()
		if epoch % 50 == 0 or epoch == epochs - 1:
			with torch.no_grad():
				per_label_acc = ((logits > 0).float() == labels).float().mean(0)
			print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  match-rate [flip_left {per_label_acc[0]:.3f} "
			      f"flip_right {per_label_acc[1]:.3f} launch {per_label_acc[2]:.3f}]", flush=True)

	return get_flat_params(agent.decoder)


def _run_cma(pool: mp.pool.Pool, args: argparse.Namespace, n_params: int, champion: np.ndarray,
             champion_fitness: float, start_gen: int, curriculum_start_gen: int, mean: np.ndarray) -> None:
	"""Real CMA-ES (via the pycma package) in place of this module's own hand-rolled CEM. Same
	fitness function, environment, and worker pool as the CEM path - only population
	sampling/update changes: pycma's ask()/tell() learns actual correlations between the 4179
	readout parameters (a real covariance matrix, or pycma's own rank-limited approximation at
	this dimensionality), where CEM's diagonal sigma structurally assumes every parameter varies
	independently. Champion acceptance keeps the same validation-gate logic as the CEM path (a
	candidate must beat the current champion on fresh episodes before being trusted), since that
	protects against the same kind of noisy-fitness false positive regardless of which optimizer
	proposed the candidate."""
	import cma

	# Extra wide-random candidates evaluated alongside (never told to) the ES's own ask()'d
	# population each generation - the CEM path's --fresh-fraction ported to CMA-ES for the same
	# reason: a single Gaussian search distribution, however large its step size, still only ever
	# samples around ONE current mean - raising cma-sigma0 (tried first, from 0.4 to 0.7) widens
	# that one basin's exploration but can't make the search jump to a distant, qualitatively
	# different region of the 4179-dim parameter space the way a handful of genuinely
	# independent random draws can. Champion tracking considers these too (a lucky fresh
	# candidate can still become champion), but es.tell() only ever sees what it actually asked
	# for - feeding it candidates it didn't generate would corrupt its own step-size/covariance
	# adaptation, which assumes a strict correspondence between ask() and tell().
	rng = np.random.default_rng(args.seed)
	archive = NoveltyArchive(args.novelty_k, args.novelty_archive_size, rng) if args.novelty else None

	cma_state_path = args.out + ".cma.pkl"
	if args.resume and os.path.exists(cma_state_path):
		with open(cma_state_path, "rb") as f:
			es = pickle.load(f)
		print(f"resumed CMA-ES internal state from {cma_state_path}", flush=True)
	else:
		es = cma.CMAEvolutionStrategy(mean.astype(np.float64), args.cma_sigma0, {
			"popsize": args.population, "seed": args.seed, "verbose": -9,
		})
		if args.resume:
			print(f"no {cma_state_path} found - starting CMA-ES fresh from the resumed mean "
			      f"(covariance/step-size history is lost, but the search center is not)", flush=True)

	# IPOP-CMA-ES-style restart (Auger & Hansen 2005): CMA-ES has no mechanism of its own to
	# escape a converged basin - its covariance/step-size only ever shrink once locked on. The
	# documented fix is a full restart from a wider, LARGER-population search once a plateau is
	# detected, not just nudging sigma0 (tried earlier this project - failed, since sigma re-
	# converges to the same small value regardless of its starting point once CMA-ES's own
	# adaptation takes over; see local_cma_flipperfocus/progress_notes.md). A bigger population
	# per restart gives the covariance estimate more samples to find genuine new structure with,
	# per the standard IPOP-CMA-ES prescription for global (not just local) optimization.
	gens_since_improvement = 0
	last_champion_fitness = champion_fitness
	restart_count = 0
	seed_rng = np.random.default_rng(args.seed + 0x5eed)  # separate stream from rng (novelty/fresh), so seed rotation is reproducible independent of --novelty/--fresh-fraction usage

	try:
		for generation in range(start_gen, args.generations + 1):
			t0 = time.time()
			score_weight = score_weight_for_generation(args, generation, curriculum_start_gen)
			solutions = es.ask()
			n_fresh = round(args.fresh_fraction * len(solutions))
			fresh = [gaussian_weights(rng, n_params, scale=0.7) for _ in range(n_fresh)]
			candidates = solutions + fresh

			# Common random numbers (CRN): every candidate this generation is evaluated on the SAME
			# courses_per_candidate seeds, rotated fresh each generation. Verified env.reset(seed=)
			# is fully deterministic (same seed -> identical score/hits/ticks). Measured on the live
			# production champion: independent seeds put ~98% of a single-episode fitness's variance
			# on "which seed" rather than "which candidate" (seed main-effect std ~163 vs
			# between-candidate std ~21 in a 6-candidate/6-seed decomposition) - sharing seeds across
			# the population cancels that seed effect out of the RANKING es.tell() consumes, at zero
			# extra compute (same number of episodes either way).
			episode_seeds = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(args.courses_per_candidate)]
			tasks = [(w.astype(np.float32), score_weight, s) for w in candidates for s in episode_seeds]
			results = pool.map(_evaluate, tasks)
			raw_fitness = np.array([r[0] for r in results], dtype=np.float32).reshape(len(candidates), args.courses_per_candidate).mean(axis=1)
			telemetry = [r[2] for r in results]

			if archive is not None:
				descriptors = np.stack([r[1] for r in results]).reshape(len(candidates), args.courses_per_candidate, -1).mean(axis=1)
				novelty_scores = np.array([archive.novelty(d) for d in descriptors], dtype=np.float32)
				for d in descriptors:
					archive.add(d)
				search_fitness = raw_fitness + args.novelty_weight * novelty_scores
			else:
				search_fitness = raw_fitness

			# len(solutions), not args.population: resuming from a pickled CMA-ES state (the
			# .cma.pkl companion file) restores the ES's OWN popsize as it was when first
			# created, ignoring whatever --population is passed on this invocation's command
			# line - trusting the CLI arg here crashes the reshape the moment the two disagree.
			es.tell(solutions, (-search_fitness[:len(solutions)]).tolist())  # pycma minimizes

			# Champion candidacy is always judged on raw_fitness (real game outcome), never the
			# novelty-augmented search_fitness - novelty only steers exploration, it must never be
			# able to promote a candidate that isn't actually a real improvement.
			#
			# HONEST bar, not a frozen historical max: whenever the prefilter fires, BOTH the
			# challenger and the CURRENT champion are freshly re-evaluated on the same
			# validation_episodes seeds (paired, cancelling seed noise between them same as the CRN
			# above) and champion_fitness is always overwritten with the champion's own fresh
			# measurement - never left as the old value, and never set to a selection-conditioned
			# max. The old version stored `validation` (the challenger's own re-measured mean,
			# conditioned on having beaten the bar) as champion_fitness and never re-tested it
			# downward: that's a monotone max over a long sequence of noisy estimates, which
			# ratchets into the upper noise tail and then requires an ever-larger fluke to ever
			# beat again. Measured on a stale local backup: a champion frozen at 2094.55 had a
			# TRUE mean of 270 (std 226) - beating its own frozen bar again needed a +16 sigma
			# fluke on a 4-episode mean. The live production champion (bar 913.25) measured a true
			# mean of 423 (std 334) - a ~2.2x inflation, same mechanism, smaller because this bar
			# had only been set once so far.
			best_idx = int(np.argmax(raw_fitness))
			if raw_fitness[best_idx] > champion_fitness:
				candidate = candidates[best_idx].astype(np.float32)
				validation_seeds = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(args.validation_episodes)]
				challenger_results = pool.map(_evaluate, [(candidate, score_weight, s) for s in validation_seeds])
				champion_results = pool.map(_evaluate, [(champion, score_weight, s) for s in validation_seeds])
				challenger_mean = float(np.mean([r[0] for r in challenger_results]))
				champion_mean = float(np.mean([r[0] for r in champion_results]))
				champion_fitness = champion_mean
				if challenger_mean > champion_mean:
					champion_fitness = challenger_mean
					champion = candidate.copy()

			if champion_fitness > last_champion_fitness:
				last_champion_fitness = champion_fitness
				gens_since_improvement = 0
			else:
				gens_since_improvement += 1

			# IPOP-CMA-ES restart: previously gated on `es.popsize < args.ipop_max_population`, so
			# once popsize reached the cap the restart condition became permanently False and IPOP
			# silently disabled itself for the rest of the run instead of continuing to restart AT
			# the cap (which still refreshes step-size/covariance even without growing further).
			restarted = False
			if args.ipop_patience > 0 and gens_since_improvement >= args.ipop_patience:
				new_popsize = min(args.ipop_max_population, round(es.popsize * args.ipop_growth))
				es = cma.CMAEvolutionStrategy(champion.astype(np.float64), args.cma_sigma0, {
					"popsize": new_popsize, "seed": args.seed + restart_count + 1, "verbose": -9,
				})
				restart_count += 1
				gens_since_improvement = 0
				restarted = True

			elapsed = time.time() - t0
			novelty_note = f"  novelty {novelty_scores.mean():6.3f}" if archive is not None else ""
			restart_note = f"  [IPOP RESTART #{restart_count}, popsize->{es.popsize}]" if restarted else ""
			# Behavior telemetry, averaged across the whole population's episodes: fitness
			# aggregates alone never revealed that a champion can hold a flipper up most of the
			# episode while producing near-zero real contact (confirmed via an offline measurement
			# script against the live production champion - see docs/colab_cli_notes.md) - this
			# makes that kind of degenerate basin visible in the log directly, every generation,
			# instead of requiring another offline forensic investigation to notice.
			mean_hits = np.mean([t["real_hits"] for t in telemetry])
			mean_p_left = np.mean([t["p_left"] for t in telemetry])
			mean_p_right = np.mean([t["p_right"] for t in telemetry])
			mean_drains = np.mean([t["drains"] for t in telemetry])
			print(
				f"gen {generation:4d}  best {raw_fitness[best_idx]:8.2f}  mean {raw_fitness.mean():8.2f}  "
				f"champion {champion_fitness:8.2f}  cma_sigma {es.sigma:.4f}  n_fresh {n_fresh}  "
				f"score_w {score_weight:.2f}  hits {mean_hits:.2f}  P(L) {mean_p_left:.2f}  "
				f"P(R) {mean_p_right:.2f}  drains {mean_drains:.2f}"
				f"{novelty_note}{restart_note}  {elapsed:5.1f}s",
				flush=True,
			)

			if generation % args.save_every == 0 or generation == args.generations:
				os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
				torch.save({
					"mean": es.mean.astype(np.float32), "sigma": np.full(n_params, es.sigma, dtype=np.float32),
					"champion": champion, "champion_fitness": champion_fitness,
					"generation": generation, "curriculum_start_gen": curriculum_start_gen,
					"readout_dim": args.readout_dim,
				}, args.out)
				with open(cma_state_path, "wb") as f:
					pickle.dump(es, f)
				print(f"  saved checkpoint: {args.out} (+ {cma_state_path})", flush=True)
	except BaseException:
		pool.terminate()
		pool.join()
		raise
	else:
		pool.close()
		pool.join()

	print(f"training complete, final checkpoint: {args.out}", flush=True)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", required=True)
	parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "..", "web", "connectome.json"))
	parser.add_argument("--readout-dim", type=int, default=16)
	parser.add_argument("--frame-skip", type=int, default=4)
	parser.add_argument("--max-steps", type=int, default=800, help="RL-steps per episode before TimeLimit truncation - real game-time coverage is max_steps * frame_skip native ticks, so halving frame_skip without doubling this shortens episodes")
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--population", type=int, default=16)
	parser.add_argument("--elites", type=int, default=4)
	parser.add_argument("--generations", type=int, default=300)
	parser.add_argument("--courses-per-candidate", type=int, default=2, help="episodes averaged per candidate per generation - more than 1 matters for elite selection, since a single noisy episode can promote an unlucky-good candidate into the elite set and skew mean/sigma")
	parser.add_argument("--seed", type=int, default=20260912)
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints", "pinball_circuit_cem.pt"))
	parser.add_argument("--save-every", type=int, default=5)
	parser.add_argument("--resume", default=None)
	parser.add_argument("--fresh-fraction", type=float, default=0.2,
	                     help="fraction of each generation's population drawn from fresh wide-random weights instead of mean+sigma*noise - without this, sigma shrinks every generation regardless of whether real progress is happening (confirmed via a real run: locked onto 'launch reliably, nothing else' with sigma collapsed to 0.12 by generation 11, never finding flipper-scoring), so the population has no way back out of an early local optimum once sigma is small")
	parser.add_argument("--sigma-floor", type=float, default=0.15, help="minimum per-parameter sigma - keeps a permanent amount of exploration even once the search has mostly converged")
	parser.add_argument("--validation-episodes", type=int, default=4, help="fresh episodes a candidate must be re-evaluated on before it can replace the champion")
	parser.add_argument("--no-seed-heuristic", dest="seed_heuristic", action="store_false", help="start from pure random noise instead of the hand-coded heuristic reflex (see heuristic_seed)")
	parser.add_argument("--flipper-focus", action="store_true", help="drop raw score from fitness entirely, isolating the objective to real flipper_hit events (plus proximity/drain/tilt shaping) - use when real score is dominated by passive ball-vs-bumper bounces the policy never caused, drowning out whatever gradient toward genuinely hitting the ball existed underneath it")
	parser.add_argument("--score-curriculum-gens", type=int, default=0, help="only meaningful with --flipper-focus: instead of an all-or-nothing switch, linearly ramp real score's weight in fitness from 0.0 to 1.0 over this many generations, then hold at 1.0 - lets a run that already learned real flipper contact under pure flipper-focus be steered back toward maximizing actual score once that sub-skill exists to build on. 0 (default) means no ramp - score stays at whatever --flipper-focus implies (0.0 if set, 1.0 if not) for the whole run")
	parser.add_argument("--optimizer", choices=["cem", "cma"], default="cem",
	                     help="'cem' is this module's own hand-rolled cross-entropy method (mean+diagonal-sigma, rank-weighted recombination). 'cma' delegates to the pycma package's real CMA-ES - a full (or, at this dimensionality, pycma's own default separable/rank-limited) covariance matrix that can learn correlations between readout parameters, unlike CEM's per-parameter-independent sigma. Same fitness function, environment, and worker pool either way - only the population sampling/update rule changes.")
	parser.add_argument("--cma-sigma0", type=float, default=0.5, help="CMA-ES's single initial global step-size (pycma has no per-parameter sigma vector like this module's CEM does) - only used when --optimizer cma")
	parser.add_argument("--ipop-patience", type=int, default=0, help="IPOP-CMA-ES-style restart (Auger & Hansen 2005): generations without a champion improvement before restarting CMA-ES fresh from the champion with a larger population (see cma-sigma0 for the restart's step-size, ipop-growth/ipop-max-population for the population schedule). 0 (default) disables restarts entirely - only used when --optimizer cma")
	parser.add_argument("--ipop-growth", type=float, default=2.0, help="population size multiplier applied on each IPOP restart - only used when --ipop-patience > 0")
	parser.add_argument("--ipop-max-population", type=int, default=256, help="population size cap for IPOP restarts - only used when --ipop-patience > 0")
	parser.add_argument("--novelty", action="store_true", help="quality-diversity-lite: add a novelty bonus (mean distance to the k nearest past behavior descriptors in a bounded archive) to each candidate's fitness before selection/es.tell(), to pull the search toward exploring qualitatively different ball-trajectory patterns instead of only exploiting the current best basin. Champion acceptance still checks real fitness alone - novelty only steers what the search tries next, never what gets promoted")
	parser.add_argument("--novelty-weight", type=float, default=1.0, help="multiplier on the novelty bonus added to fitness - only used with --novelty")
	parser.add_argument("--novelty-k", type=int, default=10, help="number of nearest archive entries averaged for the novelty score - only used with --novelty")
	parser.add_argument("--novelty-archive-size", type=int, default=500, help="max behavior descriptors kept in the novelty archive (reservoir eviction once full) - only used with --novelty")
	args = parser.parse_args()

	# Reference agent only to size the parameter vector - never stepped in the main process.
	ref_agent = FixedCircuitAgent.from_json(args.connectome, readout_dim=args.readout_dim)
	n_params = get_flat_params(ref_agent.decoder).size
	print(f"decoder parameter count (CEM search dimension): {n_params}", flush=True)

	rng = np.random.default_rng(args.seed)
	sigma = np.full(n_params, 0.8, dtype=np.float32)
	champion_fitness = -1e18
	start_gen = 1

	if args.seed_heuristic:
		champion = heuristic_seed(args.binary, args.connectome, args.readout_dim, args.frame_skip, max_steps=args.max_steps, seed=args.seed)
		mean = champion.copy()
	else:
		champion = gaussian_weights(rng, n_params)
		mean = np.zeros(n_params, dtype=np.float32)

	curriculum_start_gen = start_gen  # only meaningful default for a fresh (non-resumed) run
	if args.resume:
		ckpt = torch.load(args.resume, weights_only=False)
		mean, sigma = ckpt["mean"], ckpt["sigma"]
		champion, champion_fitness = ckpt["champion"], ckpt["champion_fitness"]
		start_gen = ckpt["generation"] + 1
		# .get(..., start_gen) falls back to the OLD (buggy) reset-every-resume behavior only for
		# checkpoints saved before this field existed - see score_weight_for_generation's docstring
		# for why persisting the curriculum's true origin matters.
		curriculum_start_gen = ckpt.get("curriculum_start_gen", start_gen)
		print(f"resumed from {args.resume} at generation {start_gen} (champion_fitness={champion_fitness:.2f}, "
		      f"curriculum_start_gen={curriculum_start_gen})", flush=True)

	ctx = mp.get_context("spawn")
	pool = ResilientPool(
		ctx,
		(args.binary, args.connectome, args.readout_dim, args.frame_skip, args.max_steps),
		args.workers,
	)

	if args.optimizer == "cma":
		_run_cma(pool, args, n_params, champion, champion_fitness, start_gen, curriculum_start_gen, mean)
		return

	archive = NoveltyArchive(args.novelty_k, args.novelty_archive_size, rng) if args.novelty else None
	seed_rng = np.random.default_rng(args.seed + 0x5eed)  # see _run_cma's identical CRN comment

	try:
		for generation in range(start_gen, args.generations + 1):
			t0 = time.time()
			score_weight = score_weight_for_generation(args, generation, curriculum_start_gen)
			n_fresh = round(args.fresh_fraction * args.population)
			n_perturbed = args.population - 1 - n_fresh
			population = (
				[champion.copy()]
				+ [mean + sigma * gaussian_weights(rng, n_params, scale=1.0) for _ in range(n_perturbed)]
				+ [gaussian_weights(rng, n_params, scale=0.7) for _ in range(n_fresh)]
			)
			# CRN: same seeds shared across the whole population, rotated each generation - see
			# _run_cma's identical comment for the measured seed-vs-candidate variance decomposition
			# motivating this.
			episode_seeds = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(args.courses_per_candidate)]
			tasks = [(w, score_weight, s) for w in population for s in episode_seeds]
			results = pool.map(_evaluate, tasks)
			fitness = np.array([r[0] for r in results], dtype=np.float32).reshape(args.population, args.courses_per_candidate).mean(axis=1)
			telemetry = [r[2] for r in results]

			if archive is not None:
				descriptors = np.stack([r[1] for r in results]).reshape(args.population, args.courses_per_candidate, -1).mean(axis=1)
				novelty_scores = np.array([archive.novelty(d) for d in descriptors], dtype=np.float32)
				for d in descriptors:
					archive.add(d)
				# Selection (elites/recombination) is driven by fitness+novelty; champion
				# acceptance below still checks raw `fitness` alone.
				selection_fitness = fitness + args.novelty_weight * novelty_scores
			else:
				selection_fitness = fitness

			order = np.argsort(-selection_fitness)
			elite_idx = order[:args.elites]
			elite_weights = np.stack([population[i] for i in elite_idx])
			# CMA-ES-style log-decaying rank weights instead of plain truncation-selection
			# (every elite equally weighted, as if rank #1 and rank #elites carried the same
			# information). Weight depends only on RANK, never on the raw fitness gap between
			# candidates - pinball's episode-to-episode variance means that gap is itself noisy,
			# so building the update formula on top of it (e.g. fitness-proportional weighting)
			# would just be propagating noise into the search direction. Standard CMA-ES
			# recombination weights: w_i = (ln(mu+0.5) - ln(i)) / sum(...), i=1..mu (rank 1 = best).
			mu = args.elites
			log_ranks = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
			rank_weights = (log_ranks / log_ranks.sum()).astype(np.float32)
			new_mean = (rank_weights[:, None] * elite_weights).sum(axis=0)
			new_sigma = np.sqrt((rank_weights[:, None] * (elite_weights - new_mean) ** 2).sum(axis=0))
			mean = 0.3 * mean + 0.7 * new_mean
			sigma = np.maximum(args.sigma_floor, 0.3 * sigma + 0.7 * new_sigma)

			# Champion candidacy is always judged on raw `fitness` (real game outcome), never
			# selection_fitness - novelty (if enabled) only steers exploration, it must never be
			# able to promote a candidate that isn't actually a real improvement.
			best_idx = int(np.argmax(fitness))
			if fitness[best_idx] > champion_fitness:
				# Re-evaluate on fresh episodes before trusting this candidate as the new
				# champion - matches flyjump's own Trainer.step() (separate validationSeeds
				# before accepting a champion). Needed here for more than the usual
				# overfitting-to-training-conditions reason: a real run accepted a champion
				# whose logged training-time fitness (20.0) could never be reproduced by any
				# direct re-evaluation of those exact saved weights (always exactly 5.0,
				# checked fresh, pool-based, and after 60 episodes of "wear" on the same
				# worker) - the root cause was never pinned down, but re-confirming before
				# accepting a champion catches it regardless of mechanism.
				#
				# Paired against a freshly re-measured champion (population[0] IS a copy of
				# champion, so best_idx can literally BE the champion - the old version could
				# then overwrite champion_fitness with a higher noisy re-measurement of IDENTICAL
				# weights, silently inflating the bar with zero search progress). See _run_cma's
				# identical fix for the full rationale (measured: a frozen bar can end up ~2-8x a
				# candidate's true mean). champion_fitness is always overwritten with the
				# champion's own fresh estimate, never left stale and never a selection-conditioned
				# max.
				candidate = population[best_idx]
				validation_seeds = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(args.validation_episodes)]
				challenger_results = pool.map(_evaluate, [(candidate, score_weight, s) for s in validation_seeds])
				champion_results = pool.map(_evaluate, [(champion, score_weight, s) for s in validation_seeds])
				challenger_mean = float(np.mean([r[0] for r in challenger_results]))
				champion_mean = float(np.mean([r[0] for r in champion_results]))
				champion_fitness = champion_mean
				if challenger_mean > champion_mean:
					champion_fitness = challenger_mean
					champion = candidate.copy()

			elapsed = time.time() - t0
			novelty_note = f"  novelty {novelty_scores.mean():6.3f}" if archive is not None else ""
			mean_hits = np.mean([t["real_hits"] for t in telemetry])
			mean_p_left = np.mean([t["p_left"] for t in telemetry])
			mean_p_right = np.mean([t["p_right"] for t in telemetry])
			mean_drains = np.mean([t["drains"] for t in telemetry])
			print(
				f"gen {generation:4d}  best {fitness[best_idx]:8.2f}  mean {fitness.mean():8.2f}  "
				f"champion {champion_fitness:8.2f}  sigma_mean {sigma.mean():.4f}  score_w {score_weight:.2f}  "
				f"hits {mean_hits:.2f}  P(L) {mean_p_left:.2f}  P(R) {mean_p_right:.2f}  drains {mean_drains:.2f}"
				f"{novelty_note}  {elapsed:5.1f}s",
				flush=True,
			)

			if generation % args.save_every == 0 or generation == args.generations:
				os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
				torch.save({
					"mean": mean, "sigma": sigma, "champion": champion, "champion_fitness": champion_fitness,
					"generation": generation, "curriculum_start_gen": curriculum_start_gen,
					"readout_dim": args.readout_dim,
				}, args.out)
				print(f"  saved checkpoint: {args.out}", flush=True)
	except BaseException:
		# terminate(), not close(): a hung/long-running worker must not block shutdown, and
		# terminate() sends SIGTERM (caught by _worker_init's handler, which runs the env.close()
		# atexit hook) rather than SIGKILL.
		pool.terminate()
		pool.join()
		raise
	else:
		pool.close()
		pool.join()

	print(f"training complete, final checkpoint: {args.out}", flush=True)


if __name__ == "__main__":
	main()
