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
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.direct_mlp_agent import DirectMLPAgent  # noqa: E402
from agents.drill import drills_from_bank, make_drills, run_one_drill  # noqa: E402
from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402

_WORKER_ENV = None
_WORKER_AGENT = None
_WORKER_OBJECTIVE = "life"
_WORKER_DRILL_POLICY = None
_WORKER_DRILL_BANK = None
_WORKER_PRESS_COST = 0.0
_WORKER_ACTIVE_UPPER_WEIGHT = 0.0
_WORKER_UPPER_IDS: frozenset = frozenset()
_WORKER_OBS_FILTER = None

# --active-upper-weight (step 3, scoring-object attribution): a flipper contact opens an "active"
# window that closes once the ball falls back into the flipper zone (y >= ACTIVE_END_Y moving
# down, vy > 0). Points from UPPER-playfield objects (table_map center y < UPPER_MAX_Y: bumpers,
# targets, spinners, ramp, kickouts, top lanes - not the inlane/outlane/bonus lanes at y~8 the
# ball rolls through on every fall-back, and not the drain) hit inside such a window are the
# "aimed" score. Diagnostic on the gen-280 champion (agents/experiments/step3/): 34.6% of its
# points were upper+active, 25% lower-lane+active (the fall-back, not aimed).
ACTIVE_END_Y = 9.0
UPPER_MAX_Y = 6.0
DEFAULT_TABLE_MAP = os.path.join(os.path.dirname(__file__), "table_map.json")


def load_upper_object_ids(table_map_path: str, upper_max_y: float = UPPER_MAX_Y) -> frozenset:
	"""Ids (StateFrame.hit_ids) of scoring objects in the upper playfield - see ACTIVE_END_Y."""
	import json

	with open(table_map_path, encoding="utf-8") as f:
		objects = json.load(f)["objects"]
	return frozenset(o["id"] for o in objects if o["center"] is not None and o["center"][1] < upper_max_y
	                 and o["group"] not in ("drain", "flipper", "plunger"))


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


def build_agent(kind: str, connectome_path: str, readout_dim: int):
	"""'circuit' is the fixed-connectome agent; 'mlp' is a raw-observation baseline used to measure
	the task's achievable ceiling (see agents/direct_mlp_agent.py)."""
	if kind == "mlp":
		return DirectMLPAgent()
	return FixedCircuitAgent.from_json(connectome_path, readout_dim=readout_dim)


def _worker_init(binary_path: str, connectome_path: str, readout_dim: int, frame_skip: int, max_steps: int,
                 agent_kind: str = "circuit", objective: str = "life", drill_bank: Optional[str] = None,
                 press_cost: float = 0.0, active_upper_weight: float = 0.0,
                 table_map: str = DEFAULT_TABLE_MAP, flipper_obs: str = "raw", loom_tau0: float = 2.0) -> None:
	global _WORKER_ENV, _WORKER_AGENT, _WORKER_OBJECTIVE, _WORKER_DRILL_POLICY, _WORKER_DRILL_BANK, _WORKER_PRESS_COST
	global _WORKER_ACTIVE_UPPER_WEIGHT, _WORKER_UPPER_IDS, _WORKER_OBS_FILTER
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

	_WORKER_AGENT = build_agent(agent_kind, connectome_path, readout_dim)
	env = PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip)
	env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
	_WORKER_ENV = env
	_WORKER_AGENT.calibrate(env.observation_space)
	_WORKER_OBJECTIVE = objective
	_WORKER_DRILL_POLICY = None
	_WORKER_DRILL_BANK = drill_bank
	_WORKER_PRESS_COST = press_cost
	_WORKER_ACTIVE_UPPER_WEIGHT = active_upper_weight
	_WORKER_OBS_FILTER = FlipperObsFilter(flipper_obs, loom_tau0=loom_tau0)
	_WORKER_UPPER_IDS = load_upper_object_ids(table_map) if active_upper_weight > 0 else frozenset()

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

# Fitness is the score earned during ONE ball life: an episode ends at the first real drain
# (PinballEnv's info["drained"]) or at --max-steps. Measured over 24 seeds, simple hand-coded
# flipper play changes this a lot versus doing nothing (mean ball life 432 -> 934 steps, score per
# life 22k -> 74k), so it is a real, controllable signal. The earlier shaping terms all measured
# the wrong events and were removed (2026-09-26):
#  - FLIPPER_HIT_BONUS was gated on post-hit ball_vy > +1, but +vy points TOWARD the drain on this
#    table (drain at y~14, top of table at y~-12): it paid for contacts that failed to save the
#    ball and ignored real saves.
#  - DRAIN_PENALTY fired on ball_in_play True->False, which is a distance-from-(0,0) test in
#    state_export.cpp: it fired whenever the ball crossed the table centre and never on a real
#    drain (after a drain the ball parks on the plunger, which still reads as "in play").
#  - PROXIMITY_BONUS and WASTED_PRESS_PENALTY were built on those same mis-measured events.
# The old swing-credit experiments are kept in agents/experiments/reward_redesign_2026-09-25.py.
# No per-step survival bonus: holding a flipper up can cradle the ball and farm time with no score.
#
# The score is taken as log1p(score per life), not linear: raw score per life is heavy-tailed
# (measured std 139k vs mean 97k for the gen-2315 champion over 48 seeds, driven by rare jackpot
# lives), which made 2-4 episode rankings close to coin flips. In log space the per-policy
# coefficient of variation drops from 1.44 to 0.11, and two different policies' outcomes on the
# same seed correlate 0.42 instead of 0.13, so common random numbers actually cancel noise.

# Behavior descriptor for novelty search/quality-diversity (--novelty): a coarse, normalized
# visitation histogram of ball position over the episode. Bounds are generous rather than exact
# (real table extent was never fully surveyed) - only relative distances between descriptors
# matter for novelty, not calibrated coordinates. Y follows this engine's convention where HIGH
# positive Y is near the flippers (TFlipperEdge ground truth: real flipper YMin/YMax is
# 10.18/13.95), so the range below covers rest/launch (~0) through past the flippers with margin.
NOVELTY_X_RANGE = (-10.0, 10.0)
NOVELTY_Y_RANGE = (-2.0, 18.0)
NOVELTY_X_BINS = 6
NOVELTY_Y_BINS = 6
NOVELTY_DESCRIPTOR_DIM = NOVELTY_X_BINS * NOVELTY_Y_BINS


class _CircuitPolicy:
	"""Greedy policy_fn over an already-built worker agent (--objective drill only) - lighter than
	drill.py's own CkptPolicy (no checkpoint reload): reuses the persistent per-worker agent whose
	weights _evaluate already sets via set_flat_params."""

	def __init__(self, agent):
		self.agent = agent
		self.h = None

	def reset(self) -> None:
		self.h = self.agent.initial_state(1)

	def __call__(self, obs: np.ndarray, t: int) -> tuple[bool, bool]:
		obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
		with torch.no_grad():
			action, self.h = self.agent.act(obs_t, self.h, greedy=True)
		action = np.asarray(action)
		return bool(action[0][0]), bool(action[0][1])


def _evaluate_drill(agent, env, seed: int) -> tuple[float, np.ndarray, dict]:
	"""--objective drill: one binary-outcome flipper-save drill (agents/drill.py) instead of a full
	ball life. `seed` deterministically picks the drill (make_drills), so CRN across candidates
	still holds. Fitness is 1.0 iff saved - a stuck-on-flipper timeout must not be rewarded."""
	global _WORKER_DRILL_POLICY
	if _WORKER_DRILL_POLICY is None:
		_WORKER_DRILL_POLICY = _CircuitPolicy(agent)
	if _WORKER_DRILL_BANK:
		drill = drills_from_bank(_WORKER_DRILL_BANK, 1, np.random.default_rng(seed))[0]
	else:
		drill = next(make_drills(1, np.random.default_rng(seed)))
	# drill length stays at drill.py's default so training matches `agents/drill.py` evaluation;
	# --max-steps only bounds the env's TimeLimit (lane-clear wait + drill)
	result = run_one_drill(env, _WORKER_DRILL_POLICY, drill, step_policy_during_wait=True)

	descriptor = np.zeros(NOVELTY_DESCRIPTOR_DIM, dtype=np.float32)
	if result["outcome"] == "skipped":
		telemetry = dict(contacts=0, drained=0, ticks=0, score=0.0, p_left=0.0, p_right=0.0, skipped=1)
		return 0.0, descriptor, telemetry

	ticks = result["ticks"]
	fitness = 1.0 if result["outcome"] == "saved" else 0.0
	telemetry = dict(
		contacts=result["contacts"], drained=int(result["outcome"] == "lost"), ticks=ticks, score=fitness,
		p_left=(result["press_left_ticks"] / ticks) if ticks else 0.0,
		p_right=(result["press_right_ticks"] / ticks) if ticks else 0.0,
		skipped=0,
	)
	return fitness, descriptor, telemetry


def _evaluate(task: tuple) -> tuple[float, np.ndarray, dict]:
	"""Runs one deterministic episode with the given decoder weights, returns (fitness,
	behavior_descriptor, telemetry) built from real game outcomes only - see this module's
	docstring for why PinballEnv's own shaped `reward` (returned but unused here) is the wrong
	signal for CEM specifically. No launch-waste penalty is needed: a redundant launch=1 while
	already in play is a guarded no-op at the engine level (see ipc_protocol.h's
	relaunch_pending field) with zero effect on any of these outcomes - the penalty only ever
	mattered for gradient credit assignment, which CEM doesn't do. The episode covers one ball
	life - it ends at the first real drain (info["drained"]) or at --max-steps - see the fitness
	comment above NOVELTY_X_RANGE. `telemetry` is a small dict of raw episode stats (contacts,
	tilt_ticks, drained, ticks, score, p_left, p_right, presses) for population-level logging -
	see the generation log line in _run_cma/main. `presses` counts physical flipper press EDGES
	(down->up transitions, left+right combined), the quantity --press-cost penalizes - not the
	same as p_left/p_right, which measure fraction of ticks HELD up.

	`task` is (flat_weights, score_weight, seed): score_weight scales how much real score_delta
	contributes to fitness (see --score-curriculum-gens; 1.0 unless --flipper-focus is given).
	`seed` (may be None) is passed straight to env.reset(seed=...) - None draws a fresh OS-random
	seed; an explicit int lets callers implement common random numbers (CRN) across a
	generation's population, see _run_cma's seed rotation. --press-cost is NOT part of `task` -
	like frame_skip/max_steps it's fixed for the whole run, so it's a worker-global
	(_WORKER_PRESS_COST) set once in _worker_init rather than repeated in every task tuple."""
	from env_python.pinball_env import (
		OBS_BALL_VY,
		OBS_BALL_X,
		OBS_BALL_Y,
		OBS_FLIPPER_LEFT,
		OBS_FLIPPER_RIGHT,
		TILT_PENALTY,
	)

	# Optional 4th element `shaped` (default True): False evaluates the plain objective without the
	# --active-upper-weight term - _paired_validation uses it so promotion is always judged on the
	# real one-life log1p(score), never on the shaped proxy.
	flat_weights, score_weight, seed = task[:3]
	shaped = task[3] if len(task) > 3 else True
	agent, env = _WORKER_AGENT, _WORKER_ENV
	set_flat_params(agent.decoder, flat_weights)

	if _WORKER_OBJECTIVE == "drill":
		return _evaluate_drill(agent, env, seed)

	obs, info = env.reset(seed=seed)
	h = agent.initial_state(1)
	obs_filter = _WORKER_OBS_FILTER or FlipperObsFilter("raw")
	obs_filter.reset()
	fitness = 0.0
	episode_score_delta = 0.0
	contacts = tilt_ticks = press_left_ticks = press_right_ticks = 0
	press_left_edges = press_right_edges = 0
	prev_left = obs[OBS_FLIPPER_LEFT] > 0.5
	prev_right = obs[OBS_FLIPPER_RIGHT] > 0.5
	drained = False
	in_active_window = False
	active_upper_points = 0
	visits = np.zeros(NOVELTY_DESCRIPTOR_DIM, dtype=np.float64)
	x_lo, x_hi = NOVELTY_X_RANGE
	y_lo, y_hi = NOVELTY_Y_RANGE
	n_ticks = 0
	done = False
	with torch.no_grad():
		while not done:
			obs_t = torch.as_tensor(np.asarray(obs_filter(obs), dtype=np.float32)).unsqueeze(0)
			action, h = agent.act(obs_t, h, greedy=True)
			obs, reward, terminated, truncated, info = env.step(action[0])
			episode_score_delta += info["score_delta"]
			if info["tilted"]:
				fitness -= TILT_PENALTY
				tilt_ticks += 1
			contacts += int(info["flipper_hit"])
			# Credit goes to the flipper CONTACT that launched the ball, not to presses: a press
			# that misses the ball opens no window, so spamming earns nothing here.
			if info["flipper_hit"]:
				in_active_window = True
			if in_active_window:
				# Multiplier-free value (hit_base_points), so the bonus pays for WHAT was hit, not for
				# the x2-x10 table multiplier state (the real score term still includes it).
				active_upper_points += sum(b for (oid, _p, _x, _y), b in zip(info.get("hit_objects", ()), info.get("hit_base_points", ()))
				                           if oid in _WORKER_UPPER_IDS)
				if obs[OBS_BALL_Y] >= ACTIVE_END_Y and obs[OBS_BALL_VY] > 0:
					in_active_window = False
			left_up = obs[OBS_FLIPPER_LEFT] > 0.5
			right_up = obs[OBS_FLIPPER_RIGHT] > 0.5
			press_left_ticks += int(left_up)
			press_right_ticks += int(right_up)
			# Press EDGE (physical down->up transition), not tick count - the thing --press-cost
			# should penalize is "how many times did it press", not "how long did it hold".
			press_left_edges += int(left_up and not prev_left)
			press_right_edges += int(right_up and not prev_right)
			prev_left, prev_right = left_up, right_up
			drained = info["drained"]

			bx = int(np.clip((obs[OBS_BALL_X] - x_lo) / (x_hi - x_lo) * NOVELTY_X_BINS, 0, NOVELTY_X_BINS - 1))
			by = int(np.clip((obs[OBS_BALL_Y] - y_lo) / (y_hi - y_lo) * NOVELTY_Y_BINS, 0, NOVELTY_Y_BINS - 1))
			visits[by * NOVELTY_X_BINS + bx] += 1.0
			n_ticks += 1
			done = terminated or truncated or drained
	presses = press_left_edges + press_right_edges
	base = fitness - _WORKER_PRESS_COST * presses  # tilt penalty accumulated above
	true_fitness = base + score_weight * float(np.log1p(max(0.0, episode_score_delta)))
	if shaped and _WORKER_ACTIVE_UPPER_WEIGHT > 0:
		# Stays inside the one log: aimed points simply count (1 + weight) times.
		fitness = base + score_weight * float(np.log1p(max(0.0, episode_score_delta + _WORKER_ACTIVE_UPPER_WEIGHT * active_upper_points)))
	else:
		fitness = true_fitness
	descriptor = (visits / n_ticks) if n_ticks else visits
	telemetry = dict(
		contacts=contacts, tilt_ticks=tilt_ticks, drained=int(drained), ticks=n_ticks, score=episode_score_delta,
		p_left=(press_left_ticks / n_ticks) if n_ticks else 0.0,
		p_right=(press_right_ticks / n_ticks) if n_ticks else 0.0,
		presses=presses, active_upper=active_upper_points, true_fitness=true_fitness,
	)
	return fitness, descriptor.astype(np.float32), telemetry


def _telemetry_summary(telemetry: list[dict]) -> str:
	"""Population-mean behavior stats for the generation log line (see _evaluate's telemetry)."""
	def m(key):
		return float(np.mean([t[key] for t in telemetry]))
	# .get(..., 0): the drill objective's telemetry (see _evaluate_drill) has no "presses" key.
	presses = float(np.mean([t.get("presses", 0) for t in telemetry]))
	active_upper = float(np.mean([t.get("active_upper", 0) for t in telemetry]))
	return (f"life {m('ticks'):6.0f}  score {m('score'):8.0f}  act_up {active_upper:7.0f}  contacts {m('contacts'):.2f}  "
	        f"drained {m('drained'):.2f}  P(L) {m('p_left'):.2f}  P(R) {m('p_right'):.2f}  presses {presses:.0f}")


def _true_fitness(results: list, n_candidates: int, courses: int) -> np.ndarray:
	"""Per-candidate mean of the UNSHAPED objective (telemetry true_fitness; falls back to the
	returned fitness for objectives without it, e.g. drill). Champion prefilters compare this, not
	the shaped search fitness, against champion_fitness - which _paired_validation always measures
	unshaped - so the two sides of that comparison are on the same scale."""
	vals = [r[2].get("true_fitness", r[0]) for r in results]
	return np.array(vals, dtype=np.float32).reshape(n_candidates, courses).mean(axis=1)


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

	# Side mapping: the table x axis is MIRRORED relative to the screen - the LEFT flipper
	# (action 0, a_flip1 / FlipperL) sits at POSITIVE x (agents/table_map.json: a_flip1 centre
	# x=+1.78; plunger lane, screen-right, at x=-7.2), verified by placing the ball at x=+-2 and
	# pressing each flipper. Earlier versions of this function had the sides swapped (x<=0 ->
	# left), so every heuristic-seeded decoder before 2026-09-29 was taught the wrong flipper.
	vel1 = obs[OBS_VEL1_L] + obs[OBS_VEL1_C] + obs[OBS_VEL1_R]
	size1 = obs[OBS_SIZE1_L] + obs[OBS_SIZE1_C] + obs[OBS_SIZE1_R]
	escape1 = (vel1 / VEL_CAP) * VEL_SIZE_RATIO + size1
	ball1_closing = escape1 > ESCAPE_THRESHOLD
	flip_left = ball1_closing and obs[OBS_BALL_X] > 0
	flip_right = ball1_closing and obs[OBS_BALL_X] <= 0
	escape2 = (obs[OBS_VEL2] / VEL_CAP) * VEL_SIZE_RATIO + obs[OBS_SIZE2]
	if escape2 > ESCAPE_THRESHOLD:
		flip_left = flip_right = True
	return np.array([flip_left, flip_right, not ball_in_play], dtype=np.int64)


class ReflexLabeler:
	"""Correct-side reactive flipper reflex (step-3 root-cause study): when the ball is low
	(y > threshold) and falling (vy > 0), press the flipper on the ball's side (x > 0 -> LEFT,
	see _heuristic_action's side-mapping note) for `hold` decision steps, then keep it released
	for `cooldown` steps before it may fire again. hold=1 is the one-step pulse. Measured on dev
	seeds 7000-7023 (agents/experiments/step3/): pulse at y > 11.5 scores log1p 11.40 with 13
	presses per life vs the gen-280 circuit's 11.34 with 862.

	Stateful (hold/cooldown counters per side). `label(obs)` returns the reflex's action for this
	step; `observe(action)` must then be called with the action actually sent, so the counters
	follow the real flipper history (the reflex's own label while it drives, the agent's action
	when it is only scored against the agent)."""

	def __init__(self, threshold: float = 11.5, hold: int = 1, cooldown: int = 2):
		self.threshold, self.hold, self.cooldown = threshold, hold, cooldown
		self.reset()

	def reset(self) -> None:
		self._run = [0, 0]      # consecutive pressed steps so far, per side
		self._rest = [self.cooldown, self.cooldown]  # consecutive released steps so far, per side
		self._pending = [0, 0]  # remaining forced-hold steps, per side

	def label(self, obs: np.ndarray, ball_in_play: bool = True) -> np.ndarray:
		from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

		a = np.zeros(3, dtype=np.int64)
		a[2] = int(not ball_in_play)
		trigger = obs[OBS_BALL_Y] > self.threshold and obs[OBS_BALL_VY] > 0
		ball_side = 0 if obs[OBS_BALL_X] > 0 else 1
		for side in (0, 1):
			if self._pending[side] > 0:
				a[side] = 1
			elif trigger and side == ball_side and self._rest[side] >= self.cooldown:
				a[side] = 1
		return a

	def observe(self, action) -> None:
		for side in (0, 1):
			if action[side]:
				if self._run[side] == 0 and self._pending[side] == 0:
					self._pending[side] = self.hold  # a new press starts a hold window
				self._pending[side] = max(0, self._pending[side] - 1)
				self._run[side] += 1
				self._rest[side] = 0
			else:
				self._pending[side] = 0
				self._run[side] = 0
				self._rest[side] += 1


# Per-flipper looming input (FlipperObsFilter mode 'loom'). Flipper targets are the centres of
# the flippers' collision AABBs from agents/table_map.json (a_flip1 = LEFT flipper = action 0 at
# x=+1.78; a_flip2 = RIGHT at x=-1.78; y=12.07). LOOM_STEP_DT converts the time-to-contact from
# seconds (obs velocities are table units / s) into decision steps; measured from recorded obs as
# the median dy / vy between consecutive decision steps (agents/experiments/step3/loom_probe.py).
LOOM_TARGET_LEFT = (1.7839, 12.0688)
LOOM_TARGET_RIGHT = (-1.7839, 12.0688)
LOOM_STEP_DT = 0.04


def looming_intensity(obs: np.ndarray, target: tuple, tau0_steps: float) -> float:
	"""LPLC2-style approach signal from one flipper's point of view: exp(-tau / tau0), tau = time
	to contact (distance / closing speed, in decision steps); 0 while the ball is not closing in."""
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	dx, dy = target[0] - obs[OBS_BALL_X], target[1] - obs[OBS_BALL_Y]
	d = float(np.hypot(dx, dy))
	if d < 1e-6:
		return 1.0
	closing = (obs[OBS_BALL_VX] * dx + obs[OBS_BALL_VY] * dy) / d  # units / s toward the target
	if closing <= 0:
		return 0.0
	tau_steps = d / closing / LOOM_STEP_DT
	return float(np.exp(-tau_steps / tau0_steps))


class FlipperObsFilter:
	"""What the agent sees of observation channels OBS_FLIPPER_LEFT/RIGHT (the physical flipper
	state, i.e. effectively its own previous action). Step-3 root-cause study: fed back
	unchanged, these two channels let the fixed circuit form a free-running press/release
	oscillator (period 3, left == right, blind to the ball - agents/experiments/step3/
	circuit_probe.py). 'raw' = unchanged (default, previous behaviour), 'zero' = both channels
	always 0, 'delay1' = the values from one decision step earlier, 'loom' = both channels 0 AND
	the two lateral LPLC2 size channels (OBS_SIZE1_L / OBS_SIZE1_R, which feed LPLC2 input cells in
	the connectome) replaced by per-flipper looming intensities (looming_intensity): SIZE1_L's bin
	is centred at x=-2.5, the RIGHT flipper's side (the table x axis is mirrored), SIZE1_R's at
	x=+2.5, the LEFT flipper's side. Same observation width, same input cells - no new pathway to
	the readout. Only the agent's input is filtered; telemetry keeps using the raw observation."""

	def __init__(self, mode: str = "raw", loom_tau0: float = 2.0):
		if mode not in ("raw", "zero", "delay1", "loom", "loomonly", "loom12"):
			raise ValueError(f"unknown flipper-obs mode {mode!r}")
		self.mode = mode
		self.loom_tau0 = loom_tau0
		self._prev = None

	def reset(self) -> None:
		self._prev = None

	def __call__(self, obs: np.ndarray) -> np.ndarray:
		if self.mode == "raw":
			return obs
		from env_python.pinball_env import OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

		out = np.array(obs, dtype=np.float32, copy=True)
		cur = out[[OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT]].copy()
		if self.mode == "loom12":
			# Per-flipper looming carried on the flipper-state channels themselves (which enter the
			# connectome through mechanosensory VNC cells one hop from the readout), instead of
			# through the LC4/LPLC2 cells that measurably barely reach it.
			out[OBS_FLIPPER_LEFT] = looming_intensity(obs, LOOM_TARGET_LEFT, self.loom_tau0)
			out[OBS_FLIPPER_RIGHT] = looming_intensity(obs, LOOM_TARGET_RIGHT, self.loom_tau0)
		elif self.mode in ("zero", "loom", "loomonly"):
			out[[OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT]] = 0.0
			if self.mode == "loomonly":
				# Also silence the mechanosensory channels (raw ball x/y/vx/vy, tilt). Measured
				# (agents/experiments/step3/loom_probe.py): with them driving the circuit, the
				# LC4/LPLC2 channels move the readout by ~1e-5 per step (vs ~2 for a mechanosensory
				# channel) - the shared intermediate cells saturate and mask the visual pathway.
				from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y, OBS_TILTED

				out[[OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY, OBS_TILTED]] = 0.0
			if self.mode in ("loom", "loomonly"):
				from env_python.pinball_env import OBS_SIZE1_L, OBS_SIZE1_R

				out[OBS_SIZE1_L] = looming_intensity(obs, LOOM_TARGET_RIGHT, self.loom_tau0)
				out[OBS_SIZE1_R] = looming_intensity(obs, LOOM_TARGET_LEFT, self.loom_tau0)
		else:
			out[[OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT]] = self._prev if self._prev is not None else 0.0
			self._prev = cur
		return out


def heuristic_seed(binary_path: str, connectome_path: str, readout_dim: int, frame_skip: int,
					episodes: int = 6, max_steps: int = 800, epochs: int = 300, lr: float = 1e-2,
					seed: int = 0, seed_policy: str = "escape", reflex_y: float = 11.5, reflex_hold: int = 1,
					flipper_obs: str = "raw", episode_seeds: Optional[list] = None,
					end_at_drain: bool = False, calibrate_bias: bool = False, loom_tau0: float = 2.0) -> np.ndarray:
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
	obs_filter = FlipperObsFilter(flipper_obs, loom_tau0=loom_tau0)
	reflex = ReflexLabeler(reflex_y, reflex_hold) if seed_policy == "reflex" else None
	env = gym.wrappers.TimeLimit(
		PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip), max_episode_steps=max_steps
	)
	agent.calibrate(env.observation_space)

	all_feats, all_labels = [], []
	was_in_play = False
	try:
		with torch.no_grad():
			for ep in range(episodes):
				obs, info = env.reset(seed=episode_seeds[ep] if episode_seeds else None)
				h = agent.initial_state(1)
				obs_filter.reset()
				if reflex is not None:
					reflex.reset()
				was_in_play = bool(info["ball_in_play"])
				done = False
				while not done:
					if reflex is not None:
						label = reflex.label(obs, was_in_play)
						reflex.observe(label)
					else:
						label = _heuristic_action(obs, was_in_play)
					obs_t = torch.as_tensor(np.asarray(obs_filter(obs), dtype=np.float32)).unsqueeze(0)
					_feats, h = agent.step(obs_t, h)
					# Normalized readout activity (before the trainable bottleneck `proj`), so the fit
					# below trains proj AND head. Earlier versions stored decoder.features() (already
					# projected by the random-orthogonal initial proj) and so only ever fit the 8->3
					# head on a random 8-d slice of the readout - measured to cap imitation badly.
					all_feats.append(agent.decoder.norm(h[:, agent.decoder.idx])[0])
					all_labels.append(label)
					obs, reward, terminated, truncated, info = env.step(label.astype(bool))
					was_in_play = bool(info["ball_in_play"])
					done = terminated or truncated or (end_at_drain and info["drained"])
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
		logits = agent.decoder.dist_inputs(agent.decoder.proj(feats))
		loss = loss_fn(logits, labels)
		opt.zero_grad()
		loss.backward()
		opt.step()
		if epoch % 50 == 0 or epoch == epochs - 1:
			with torch.no_grad():
				per_label_acc = ((logits > 0).float() == labels).float().mean(0)
			print(f"  epoch {epoch:4d}  loss {loss.item():.4f}  match-rate [flip_left {per_label_acc[0]:.3f} "
			      f"flip_right {per_label_acc[1]:.3f} launch {per_label_acc[2]:.3f}]", flush=True)

	if calibrate_bias:
		# pos_weight-balanced BCE thresholded at logit 0 grossly over-predicts rare presses
		# (measured: precision ~0.001 for a 0.5%-rate pulse). Shift each head's bias so the
		# fitted decoder fires at the same rate as the labels on the collected data.
		with torch.no_grad():
			logits = agent.decoder.dist_inputs(agent.decoder.proj(feats))
			for k in range(labels.shape[1]):
				rate = float(labels[:, k].mean())
				if 0.0 < rate < 1.0:
					cut = torch.quantile(logits[:, k], 1.0 - rate)
					agent.decoder.head.bias[k] -= cut
		print(f"heuristic_seed: bias-calibrated heads to label press rates {labels.mean(0).tolist()}", flush=True)

	return get_flat_params(agent.decoder)


def _paired_validation(pool, args: argparse.Namespace, seed_rng: np.random.Generator, candidate: np.ndarray,
                       champion: np.ndarray, score_weight: float) -> tuple[bool, float, float]:
	"""Re-evaluates challenger and champion on the same fresh seeds and returns (promote,
	challenger_mean, champion_mean). Promotes only if the paired mean difference exceeds
	--promotion-t standard errors: one ball life's outcome is mostly luck (paired per-episode std
	~1.2 in log-score units, measured over 48 seeds), so "challenger mean > champion mean" on a few
	episodes promotes statistically tied candidates about half the time."""
	seeds = [int(seed_rng.integers(0, 2**31 - 1)) for _ in range(args.validation_episodes)]
	challenger = np.array([r[0] for r in pool.map(_evaluate, [(candidate, score_weight, s, False) for s in seeds])])
	champ = np.array([r[0] for r in pool.map(_evaluate, [(champion, score_weight, s, False) for s in seeds])])
	diff = challenger - champ
	se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("inf")
	promote = bool(diff.mean() > args.promotion_t * se)
	print(f"  validation: challenger {challenger.mean():.3f} vs champion {champ.mean():.3f} "
	      f"(diff {diff.mean():+.3f}, SE {se:.3f}){' -> PROMOTED' if promote else ''}", flush=True)
	return promote, float(challenger.mean()), float(champ.mean())


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
			true_fitness = _true_fitness(results, len(candidates), args.courses_per_candidate)
			if true_fitness[best_idx] > champion_fitness:
				candidate = candidates[best_idx].astype(np.float32)
				promote, challenger_mean, champion_fitness = _paired_validation(pool, args, seed_rng, candidate, champion, score_weight)
				if promote:
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
			print(
				f"gen {generation:4d}  best {raw_fitness[best_idx]:8.2f}  mean {raw_fitness.mean():8.2f}  "
				f"champion {champion_fitness:8.2f}  cma_sigma {es.sigma:.4f}  n_fresh {n_fresh}  "
				f"score_w {score_weight:.2f}  {_telemetry_summary(telemetry)}"
				f"{novelty_note}{restart_note}  {elapsed:5.1f}s",
				flush=True,
			)

			if generation % args.save_every == 0 or generation == args.generations:
				os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
				torch.save({
					"mean": es.mean.astype(np.float32), "sigma": np.full(n_params, es.sigma, dtype=np.float32),
					"champion": champion, "champion_fitness": champion_fitness,
					"generation": generation, "curriculum_start_gen": curriculum_start_gen,
					"readout_dim": args.readout_dim, "agent": args.agent, "objective": args.objective,
					"drill_bank": args.drill_bank, "press_cost": args.press_cost, "flipper_obs": args.flipper_obs,
					"loom_tau0": args.loom_tau0,
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
	parser.add_argument("--agent", choices=["circuit", "mlp"], default="circuit", help="circuit: fixed-connectome agent; mlp: raw-observation baseline for measuring the task's ceiling")
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
	parser.add_argument("--promotion-t", type=float, default=0.0, help="promote a challenger only if its paired mean advantage over the champion exceeds this many standard errors (0 = any positive difference)")
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
	parser.add_argument("--objective", choices=["life", "drill"], default="life",
	                     help="'life' (default, unchanged): fitness from one full ball life's score. "
	                          "'drill': fitness is binary success on one flipper-save drill (agents/drill.py) - "
	                          "--courses-per-candidate/--validation-episodes then count drills instead of episodes")
	parser.add_argument("--drill-bank", default=None, help="only meaningful with --objective drill: sample drill "
	                     "starts from a real-game bank (agents/collect_drill_bank.py) instead of synthetic make_drills")
	parser.add_argument("--press-cost", type=float, default=0.0, help="only meaningful with --objective life: "
	                     "subtracts press_cost * presses (physical flipper down->up edges, left+right combined) "
	                     "from fitness, so the search is penalized for pressing more than it needs to instead of "
	                     "only ever being rewarded for pressing. 0.0 (default) reproduces the old unpenalized "
	                     "behaviour exactly - see agents/experiments/press_cost/measure_presses.py for calibration")
	parser.add_argument("--active-upper-weight", type=float, default=0.0, help="only meaningful with --objective "
	                     "life: search fitness becomes log1p(score + W * active_upper_points), where active_upper_points "
	                     "are points from upper-playfield objects (--table-map, center y < 6) hit between a real flipper "
	                     "CONTACT and the ball falling back (see ACTIVE_END_Y). Champion promotion (paired validation) "
	                     "still uses the plain log1p(score). 0.0 (default) = unchanged behaviour")
	parser.add_argument("--seed-policy", choices=["escape", "reflex"], default="escape", help="policy the decoder is "
	                     "imitation-seeded from (unless --no-seed-heuristic/--resume): 'escape' = _heuristic_action "
	                     "(looming escape signal), 'reflex' = ReflexLabeler (correct-side pulse/hold reflex)")
	parser.add_argument("--seed-reflex-y", type=float, default=11.5, help="ReflexLabeler trigger: ball y above this and falling")
	parser.add_argument("--seed-reflex-hold", type=int, default=1, help="ReflexLabeler press length in decision steps (1 = pulse)")
	parser.add_argument("--seed-episodes", type=int, default=6, help="rollouts collected for the imitation seed")
	parser.add_argument("--loom-tau0", type=float, default=2.0, help="--flipper-obs loom: looming time constant in decision steps")
	parser.add_argument("--flipper-obs", choices=["raw", "zero", "delay1", "loom", "loomonly", "loom12"], default="raw", help="what the agent sees of "
	                     "the flipper-state observation channels (see FlipperObsFilter); applied at seeding and at "
	                     "evaluation time. Must match the setting a --resume checkpoint was trained with")
	parser.add_argument("--table-map", default=DEFAULT_TABLE_MAP, help="scripts/gen_table_map.py output, used by --active-upper-weight")
	args = parser.parse_args()

	# Reference agent only to size the parameter vector - never stepped in the main process.
	ref_agent = build_agent(args.agent, args.connectome, args.readout_dim)
	n_params = get_flat_params(ref_agent.decoder).size
	print(f"decoder parameter count (CEM search dimension): {n_params}", flush=True)

	rng = np.random.default_rng(args.seed)
	sigma = np.full(n_params, 0.8, dtype=np.float32)
	champion_fitness = -1e18
	start_gen = 1

	if args.seed_heuristic and not args.resume and args.agent == "circuit":
		champion = heuristic_seed(args.binary, args.connectome, args.readout_dim, args.frame_skip, max_steps=args.max_steps,
		                          seed=args.seed, seed_policy=args.seed_policy, reflex_y=args.seed_reflex_y,
		                          reflex_hold=args.seed_reflex_hold, flipper_obs=args.flipper_obs,
		                          episodes=args.seed_episodes, calibrate_bias=args.seed_policy == "reflex",
		                          end_at_drain=args.seed_policy == "reflex", loom_tau0=args.loom_tau0)
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
		resumed_objective = ckpt.get("objective", "life")
		if resumed_objective != args.objective:
			# champion_fitness's scale belongs to the OLD objective (e.g. life's log1p(score) can be
			# ~11 vs drill's 0-1) - left as-is, no candidate could ever beat it and validation would
			# never fire again; resetting forces gen 1 to re-measure the champion fresh.
			print(f"resumed checkpoint's objective ({resumed_objective!r}) != --objective ({args.objective!r}) - "
			      f"resetting champion_fitness so validation re-measures it on the new scale", flush=True)
			champion_fitness = -1e18

	ctx = mp.get_context("spawn")
	pool = ResilientPool(
		ctx,
		(args.binary, args.connectome, args.readout_dim, args.frame_skip, args.max_steps, args.agent, args.objective,
		 args.drill_bank, args.press_cost, args.active_upper_weight, args.table_map, args.flipper_obs, args.loom_tau0),
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
			true_fitness = _true_fitness(results, args.population, args.courses_per_candidate)
			if true_fitness[best_idx] > champion_fitness:
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
				promote, challenger_mean, champion_fitness = _paired_validation(pool, args, seed_rng, candidate, champion, score_weight)
				if promote:
					champion_fitness = challenger_mean
					champion = candidate.copy()

			elapsed = time.time() - t0
			novelty_note = f"  novelty {novelty_scores.mean():6.3f}" if archive is not None else ""
			print(
				f"gen {generation:4d}  best {fitness[best_idx]:8.2f}  mean {fitness.mean():8.2f}  "
				f"champion {champion_fitness:8.2f}  sigma_mean {sigma.mean():.4f}  score_w {score_weight:.2f}  "
				f"{_telemetry_summary(telemetry)}"
				f"{novelty_note}  {elapsed:5.1f}s",
				flush=True,
			)

			if generation % args.save_every == 0 or generation == args.generations:
				os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
				torch.save({
					"mean": mean, "sigma": sigma, "champion": champion, "champion_fitness": champion_fitness,
					"generation": generation, "curriculum_start_gen": curriculum_start_gen,
					"readout_dim": args.readout_dim, "agent": args.agent, "objective": args.objective,
					"drill_bank": args.drill_bank, "press_cost": args.press_cost, "flipper_obs": args.flipper_obs,
					"loom_tau0": args.loom_tau0,
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
