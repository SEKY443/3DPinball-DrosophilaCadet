"""Gymnasium-shaped wrapper around the native SpaceCadetPinball binary, driven over the
Unix-domain-socket IPC bridge (env_python/ipc_client.py <-> src_cpp/ipc_server.cpp).

Requires PINBALL_BINARY to point at a binary built with the src_cpp patches applied
(scripts/build_native.sh), and that binary needs a user-supplied CADET.DAT/PINBALL.DAT
placed next to it (see project README) - this wrapper cannot function without one, since the
native process never reaches IpcServer::Init() if pb::init() fails to find game data.
"""
from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
import uuid
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from .ipc_client import Action, PinballIPCClient, State

# Index layout for both the observation array and (informally) for anyone reading raw State
# objects - flipper_left/right/tilted are represented as 0.0/1.0 floats in the observation.
OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY = 0, 1, 2, 3
# Ball1's vel/size channels are retinotopic: real LC4/LPLC2 populations are distributed across
# the visual field (different cells have different preferred positions), so a real looming
# detector's population activity pattern already tells the downstream circuit roughly WHERE the
# threat is, not just that one exists. Earlier versions of this channel broadcast the exact same
# scalar to all 25 cells in a channel regardless of ball_x - collapsing that spatial information
# entirely, using only the "downstream wiring" half of what these neurons do and none of their
# distributed spatial coding. Three bins (L/C/R) is a crude discretization of that population
# code, not the real continuous retinotopy, but it's a step toward actually using the cells'
# real computational role instead of treating them as an arbitrary scalar-injection channel.
OBS_VEL1_L, OBS_VEL1_C, OBS_VEL1_R = 4, 5, 6
OBS_SIZE1_L, OBS_SIZE1_C, OBS_SIZE1_R = 7, 8, 9
OBS_VEL2, OBS_SIZE2 = 10, 11
OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT, OBS_TILTED = 12, 13, 14
OBS_DIM = 15

# Bin centers (table x-coordinates) and tuning width for the retinotopic population code above -
# the two flippers sit at x=-2.489/+2.489 (see FLIPPER_ZONE_Y's comment; note the table x axis is
# mirrored vs the screen - the LEFT flipper, action 0, is the one at +2.489), so centers span
# slightly past that range with enough overlap (BIN_SIGMA) that a ball anywhere in the flipper
# zone drives at least two of the three bins rather than falling in a dead zone between them.
BIN_CENTERS = (-2.5, 0.0, 2.5)
BIN_SIGMA = 2.0


def _spatial_weight(ball_x: float, center: float) -> float:
	return float(np.exp(-((ball_x - center) ** 2) / (2 * BIN_SIGMA ** 2)))

# Fixed table-coordinate threshold above which the ball is near the flippers - same value and
# rationale as train_pinball_circuit_cem.py's FLIPPER_ZONE_Y (ground-truth from
# TFlipperEdge::XMin/XMax/YMin/YMax: both flippers sit at YMin=10.18/YMax=13.95, i.e. HIGH
# positive Y). Kept here too since vel/size channels are computed at observation time, not
# fitness time.
LOOM_ZONE_Y = 9.0

# Peak distance and width for the LPLC2 "size" Gaussian - see _size_signal. Not fit to any real
# data (engineered, like every other channel-injection choice in this project - see
# build_pinball_connectome.py's disclaimer), just shaped to peak shortly before the ball would
# reach the flipper zone rather than growing monotonically all the way to contact.
SIZE_PEAK_DISTANCE = 1.5
SIZE_SIGMA = 1.5


# Attention window for _velocity_signal - see its docstring for why this exists. Wider than
# SIZE_SIGMA (1.5) since velocity should matter over a longer approach than size's sharp
# near-contact peak, but still heavily discount motion far from the zone.
VEL_ATTENTION_SIGMA = 6.0


def _velocity_signal(ball_vy: float, ball_y: float, active: bool = True) -> float:
	"""LC4's channel: real LC4 neurons encode retinal angular VELOCITY of an approaching
	object - see Ache et al. 2019 (Current Biology), whose model of the Giant Fiber's looming
	response sums a LINEAR function of LC4's velocity signal with a Gaussian function of
	LPLC2's size signal (_size_signal).

	Gated by distance to the zone, NOT raw physical closing speed alone - a real fly's LC4
	encodes retinal ANGULAR velocity, which is physical speed divided by distance (a fast object
	far away subtends a slowly-growing angle; the same speed up close grows the angle fast). Raw
	ball_vy has no such distance scaling, so a ball freshly launched and climbing steadily toward
	the zone from ~20 units away registered the same "closing" signal as one about to make
	contact - confirmed via real telemetry: the hand-coded heuristic (_heuristic_action) fired
	real flip decisions for a sustained stretch starting the instant the ball left the plunger,
	20+ units of distance before reaching the flipper zone, and correspondingly got ZERO real
	flipper_hit events across 5 held-out episodes once the ball_vx/vy computation bug (see
	src_cpp/state_export.cpp) was fixed and raw velocity values stopped being noise-saturated
	spikes that had been masking this. Distance gating restores the "only urgent once actually
	close" property real angular velocity has for free."""
	if not active:
		return 0.0
	closing = min(max(ball_vy, 0.0), 20.0)
	distance = LOOM_ZONE_Y - ball_y
	if distance < 0:
		return 0.0
	attention = float(np.exp(-(distance ** 2) / (2 * VEL_ATTENTION_SIGMA ** 2)))
	return float(closing * attention)


def _size_signal(ball_y: float, active: bool = True) -> float:
	"""LPLC2's channel: real LPLC2 neurons encode retinal angular SIZE of an approaching object,
	Gaussian-tuned rather than monotonic (Ache et al. 2019) - a real fly's escape response peaks
	at a characteristic apparent size, not "bigger is always more urgent". Approximated here as a
	Gaussian in remaining distance to the flipper zone, peaked at SIZE_PEAK_DISTANCE (closer than
	that, "size" - i.e. urgency - falls back off, same qualitative shape the real neuron shows).
	Zero when the ball is past the zone already or (for the second ball) not on the table."""
	if not active:
		return 0.0
	distance = LOOM_ZONE_Y - ball_y
	if distance < 0:
		return 0.0
	return float(np.exp(-((distance - SIZE_PEAK_DISTANCE) ** 2) / (2 * SIZE_SIGMA ** 2)))

ACT_FLIPPER_LEFT, ACT_FLIPPER_RIGHT, ACT_LAUNCH = 0, 1, 2

# Scales raw score_delta (in-engine points: regular bumper/target hits are O(100-1000),
# jackpots/multiball bonuses can be tens of thousands - see MessageCode::TLightGroupJackpot
# and similar "special" scoring paths) down to the same order of magnitude as the shaping
# bonuses below. This preserves the game's own relative scoring hierarchy exactly - a jackpot
# still gives proportionally more reward than a regular bumper - it just stops that hierarchy
# from drowning out every other reward term once real scoring is happening (score reached into
# the tens of thousands once the launch-race bug below was fixed, versus the ~0.1-2.0 range
# the shaping bonuses were tuned for while score was still ~0 under the broken ball physics).
SCORE_SCALE = 0.01

# Per-tick shaping penalty while the table is tilted, on top of scaled score_delta reward.
TILT_PENALTY = 5.0

# Per-tick shaping bonus while the ball is in play. Originally 0.1, sized to bootstrap
# "launch and survive" back when score was ~0 under the (since-fixed) launch-race bug.
# Cut 10x after auditing real gameplay: with ZERO flipper input, the ball stays "in play" for
# an entire 4000-tick episode via passive bumper bounces alone (verified empirically), so at
# 0.1/tick this bonus alone (~400 over a full episode) could dwarf real scoring and
# LAUNCH_SUCCESS_BONUS combined - rewarding "a ball is bouncing somewhere" far more than
# "I am playing skillfully". Now just enough to keep favoring survival over instant drain
# without becoming the dominant reward source.
BALL_IN_PLAY_BONUS = 0.01

# One-off bonus the instant ball_in_play flips False -> True (a launch actually worked - the
# ball left the plunger lane), separate from the small per-tick BALL_IN_PLAY_BONUS above. A
# per-tick bonus alone rewards a launch only lazily, spread over many following ticks - REINFORCE
# then has to correctly assign credit for a whole trailing sequence of "in play" reward back to
# one earlier "launch" action. A large, immediate bonus tied directly to the successful
# transition is a far shorter, stronger, more attributable signal for learning this specific
# button press - this exists because a real trained run converged to a ~0 launch probability
# despite launching being clearly high-value once it happens (score reached 50k+ per episode),
# exactly the kind of premature exploration collapse this bonus (plus train.py's entropy term)
# is meant to prevent.
LAUNCH_SUCCESS_BONUS = 20.0

# One-off bonus when an active flipper physically strikes the ball (StateFrame.flipper_hit,
# see src_cpp/patches/0004). score_delta alone can't teach "use the paddles": the same score
# comes from the ball passively bouncing off bumpers on its way down from the initial launch,
# with zero flipper skill involved - that ambiguity is exactly why a trained agent learned to
# launch reliably but never learned deliberate flipper timing. This bonus is a direct, first
# reward attributable ONLY to paddle-ball contact, independent of whether it scores anything.
FLIPPER_HIT_BONUS = 2.0

# A flipper press is a real physical actuator, not an instantaneous flag: once pressed it
# must stay up for at least this many native ticks, and once released it must stay down for at
# least this many, before the policy's next request is honored. Without this, a policy can
# request an on/off flip every single decision - previously exploited to farm a proximity
# reward shaping term via rapid toggling instead of genuine timing (see
# agents/train_pinball_circuit_cem.py's history; that specific exploit is now closed at the
# reward layer too - PROXIMITY_BONUS is capped per zone-visit, not per press-edge).
#
# Kept deliberately SHORT, not long: TFlipper::FlipperCollision in the real engine only
# registers a hit while the flipper is actively ROTATING (its ray-cast is built from
# deltaAngle, the rotation delta *this tick* - a flipper already up and holding static has
# deltaAngle=0, a degenerate ray, and can never register a hit no matter how long it holds or
# how close the ball is). Confirmed directly: an always-held-up scripted policy produced zero
# flipper_hit events across 24,000 ticks. The only useful moment is the brief up-swing itself,
# so a long static hold wastes time it could instead spend re-triggering fresh swing attempts -
# these are the minimum ticks that still look like a real press/release, not a tuned "how long
# should a press feel" value.
MIN_FLIPPER_PRESS_TICKS = 2
MIN_FLIPPER_RELEASE_TICKS = 1

# Where the engine parks a fresh ball after a drain, waiting on the plunger. ball_in_play (a
# distance-from-(0,0) test in state_export.cpp) reads True there, so auto-launch never fires and
# the ball sits in the lane for the rest of the episode - a real drain is only visible as the ball
# jumping from the flipper end of the table straight to this spot. Measured on DEMO.DAT.
PLUNGER_REST_X, PLUNGER_REST_Y = -7.02, 10.09
PLUNGER_REST_TOL = 0.3
DRAIN_FROM_Y = 12.5


def _state_to_obs(state: State) -> np.ndarray:
	# Position/velocity bounds are intentionally NOT normalized here: the engine's world-unit
	# table extents haven't been calibrated against a real DAT file yet (see project plan,
	# Phase 1 verification). observation_space is correspondingly left unbounded (-inf, inf)
	# rather than asserting false precision; normalize downstream (e.g. in agents/snn_model.py)
	# once real extents are known.
	vel1 = _velocity_signal(state.ball_vy, state.ball_y)
	size1 = _size_signal(state.ball_y)
	bin_weights = [_spatial_weight(state.ball_x, c) for c in BIN_CENTERS]
	return np.array([
		state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
		bin_weights[0] * vel1, bin_weights[1] * vel1, bin_weights[2] * vel1,
		bin_weights[0] * size1, bin_weights[1] * size1, bin_weights[2] * size1,
		_velocity_signal(state.ball2_vy, state.ball2_y, active=state.ball2_active),
		_size_signal(state.ball2_y, active=state.ball2_active),
		1.0 if state.flipper_left else 0.0,
		1.0 if state.flipper_right else 0.0,
		1.0 if state.tilted else 0.0,
	], dtype=np.float32)


class PinballEnv(gym.Env):
	metadata = {"render_modes": []}

	def __init__(
		self,
		binary_path: Optional[str] = None,
		sock_path: Optional[str] = None,
		connect_timeout: float = 15.0,
		headless: bool = True,
		frame_skip: int = 1,
		relaunch_at_rest: bool = False,
	):
		super().__init__()
		# Full-game play only (default off, so training is unchanged): also auto-launch when the
		# ball sits at the plunger rest spot. After a drain the engine parks the next ball there and
		# ball_in_play reads True, so the default auto-launch never fires (see PLUNGER_REST_X).
		self.relaunch_at_rest = relaunch_at_rest
		# Repeats each chosen action for this many native physics ticks per step() call.
		# At uncapped engine speed (~400 ticks/sec) a decision every single tick is far
		# faster than any coherent flipper motion - independent Bernoulli sampling at that
		# rate produces on/off dithering rather than a real held flip. frame_skip=1 keeps
		# the old per-tick behavior; use e.g. 4-8 for training so a "hold the flipper"
		# decision actually holds for a physically meaningful window.
		self.frame_skip = max(1, frame_skip)
		self.binary_path = binary_path or os.environ.get("PINBALL_BINARY")
		self.headless = headless
		if not self.binary_path:
			raise ValueError("binary_path not given and PINBALL_BINARY is not set")
		self.sock_path = sock_path or os.path.join(
			tempfile.gettempdir(), f"pinball_ipc_{uuid.uuid4().hex[:8]}.sock"
		)
		self.connect_timeout = connect_timeout

		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
		)
		self.action_space = gym.spaces.MultiBinary(3)  # [flipper_left, flipper_right, launch]

		self._proc: Optional[subprocess.Popen] = None
		self._client: Optional[PinballIPCClient] = None
		self._pending_state: Optional[State] = None  # received but not yet actioned-upon
		self._was_ball_in_play = False  # tracks the False->True edge for LAUNCH_SUCCESS_BONUS

		# Physical flipper actuator state (see MIN_FLIPPER_PRESS_TICKS) - the ACTUAL state sent
		# to the engine each native tick, which may lag the policy's requested state.
		self._flip_left_physical = False
		self._flip_right_physical = False
		self._flip_left_ticks_in_state = 0
		self._flip_right_ticks_in_state = 0

	def _advance_flipper(self, requested: bool, physical: bool, ticks_in_state: int) -> tuple[bool, int]:
		"""One native tick of the press/release-hold state machine for one flipper. Returns the
		(possibly unchanged) physical state to actually send, and the updated hold-duration
		counter."""
		min_hold = MIN_FLIPPER_PRESS_TICKS if physical else MIN_FLIPPER_RELEASE_TICKS
		if requested != physical and ticks_in_state >= min_hold:
			return requested, 0
		return physical, ticks_in_state + 1

	def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
		super().reset(seed=seed)

		if self._client is None:
			self._spawn_and_connect()
			self._pending_state = self._client.recv_state()

		# Ack the currently-pending state with a reset pulse, then wait for the frame that
		# reflects the post-reset table - see the lockstep send/recv pairing note below.
		#
		# reset_seed reseeds the engine's own rand()/RandFloat() (see ipc_protocol.h) so this
		# episode's bumper/plunger jitter and TLightGroup score-bonus randomness depend only on
		# this seed, not on how many ticks this worker process has already run - without this, a
		# fixed set of weights measurably scored differently depending purely on process age (see
		# agents/train_pinball_circuit_cem.py's history of "unreproducible fitness" champions).
		# Drawn from self.np_random (seeded via reset(seed=...) if the caller passed one, else
		# lazily OS-seeded once per env instance) so repeated resets still vary episode to episode.
		reset_seed = int(self.np_random.integers(0, 2**32 - 1))
		self._client.send_action(Action(tick=self._pending_state.tick, reset=True, reset_seed=reset_seed))
		self._pending_state = self._client.recv_state()
		self._was_ball_in_play = bool(self._pending_state.ball_in_play)
		self._flip_left_physical = self._flip_right_physical = False
		self._flip_left_ticks_in_state = self._flip_right_ticks_in_state = 0

		obs = _state_to_obs(self._pending_state)
		info: dict[str, Any] = {
			"tick": self._pending_state.tick,
			"ball_in_play": bool(self._pending_state.ball_in_play),
			"relaunch_pending": bool(self._pending_state.relaunch_pending),
			"ball2_active": bool(self._pending_state.ball2_active),
		}
		return obs, info

	def step(self, action):
		if self._client is None or self._pending_state is None:
			raise RuntimeError("step() called before reset()")

		requested_left = bool(action[ACT_FLIPPER_LEFT])
		requested_right = bool(action[ACT_FLIPPER_RIGHT])

		# Launching is no longer a learned decision - see CMU's PinBot report (2024): giving an
		# agent an explicit launch action taught it to idle forever rather than risk the
		# eventual drain penalty, even though launching is unconditionally necessary to ever
		# score. Their fix, adopted here, is to take the decision away entirely: auto-launch
		# every tick the ball isn't in play (idempotent - see ipc_server.cpp's
		# g_relaunchPending guard against re-triggering pb::launch_ball() while a launch is
		# already pending or the ball is already out). The policy's own launch output
		# (action[ACT_LAUNCH]) is now ignored; the action space still reports MultiBinary(3)
		# for checkpoint/observation-shape compatibility, but the third slot is inert.
		launch = not self._was_ball_in_play
		if self.relaunch_at_rest and self._pending_state is not None:
			ps = self._pending_state
			# The parked ball settles further down the lane (y ~11.7) than the drain teleport spot;
			# the engine side (ipc_server.cpp BallRestsInPlungerLane) also requires it to be at rest.
			launch = launch or (abs(ps.ball_x - PLUNGER_REST_X) < PLUNGER_REST_TOL
			                    and ps.ball_y > PLUNGER_REST_Y - PLUNGER_REST_TOL)

		total_reward = 0.0
		total_score_delta = 0
		any_flipper_hit = False
		drained = False
		# (object_id, points, ball_x, ball_y) per scoring-object contact, ball position taken at
		# the native tick the contact was reported - see ipc_client.State.hits. hit_base_points is
		# the parallel list of the same contacts' points BEFORE the table score multiplier.
		hit_objects: list[tuple[int, int, float, float]] = []
		hit_base_points: list[int] = []
		unattributed_points = 0
		state = self._pending_state

		for sub_tick in range(self.frame_skip):
			# The requested flipper state is the same for all sub_ticks in this frame_skip
			# window (one decision per env step), but the PHYSICAL actuator advances one native
			# tick at a time toward it, subject to the press/release hold-duration state
			# machine - see _advance_flipper and MIN_FLIPPER_PRESS_TICKS.
			self._flip_left_physical, self._flip_left_ticks_in_state = self._advance_flipper(
				requested_left, self._flip_left_physical, self._flip_left_ticks_in_state
			)
			self._flip_right_physical, self._flip_right_ticks_in_state = self._advance_flipper(
				requested_right, self._flip_right_physical, self._flip_right_ticks_in_state
			)

			# Lockstep invariant: exactly one ActionFrame must be sent in reply to the most
			# recently received StateFrame before another one will arrive. launch is a pulse
			# (see ipc_protocol.h) - only fire it on the first sub-tick of the skip window, or
			# a held launch=True action would re-trigger pb::launch_ball() every native tick.
			self._client.send_action(Action(
				tick=self._pending_state.tick,
				flipper_left=self._flip_left_physical,
				flipper_right=self._flip_right_physical,
				launch=launch and sub_tick == 0,
			))
			prev_y = state.ball_y
			self._pending_state = self._client.recv_state()
			state = self._pending_state
			if (prev_y > DRAIN_FROM_Y
					and abs(state.ball_x - PLUNGER_REST_X) < PLUNGER_REST_TOL
					and abs(state.ball_y - PLUNGER_REST_Y) < PLUNGER_REST_TOL):
				drained = True

			total_score_delta += state.score_delta
			for obj_id, points in state.hits:
				hit_objects.append((obj_id, points, state.ball_x, state.ball_y))
			hit_base_points.extend(b for _, b in state.base_hits)
			unattributed_points += state.unattributed_points
			total_reward += float(state.score_delta) * SCORE_SCALE - (TILT_PENALTY if state.tilted else 0.0)
			if state.ball_in_play:
				total_reward += BALL_IN_PLAY_BONUS
				if not self._was_ball_in_play:
					total_reward += LAUNCH_SUCCESS_BONUS
			self._was_ball_in_play = bool(state.ball_in_play)
			if state.flipper_hit:
				any_flipper_hit = True
				total_reward += FLIPPER_HIT_BONUS

			if state.done or drained:
				break

		obs = _state_to_obs(state)
		terminated = state.done
		truncated = False
		info: dict[str, Any] = {
			"tick": state.tick,
			"score_delta": total_score_delta,
			"flipper_hit": any_flipper_hit,
			"hit_objects": hit_objects,
			"hit_base_points": hit_base_points,
			"unattributed_points": unattributed_points,
			"drained": drained,
			"ball_in_play": bool(state.ball_in_play),
			"relaunch_pending": bool(state.relaunch_pending),
			"tilted": bool(state.tilted),
			"ball2_active": bool(state.ball2_active),
		}
		return obs, total_reward, terminated, truncated, info

	def place_ball(self, x: float, y: float, vx: float, vy: float) -> tuple[np.ndarray, dict[str, Any]]:
		"""Flipper-drill primitive: teleports the ball to (x, y) with velocity (vx, vy) - table
		coordinates and table-units/sec, matching ball_x/y/vx/vy (see state_export.cpp's Capture)
		- in reply to the currently pending state, then returns the observation for the next
		tick. Sent INSTEAD of a normal action for that one tick: flipper actuator state and
		launch/reset are left untouched (see ipc_server.cpp's PLC! handling). Call after reset()
		once the ball is out of the plunger lane; a no-op engine-side if BallList[0] isn't active.
		"""
		if self._client is None or self._pending_state is None:
			raise RuntimeError("place_ball() called before reset()")

		self._client.send_place(self._pending_state.tick, x, y, vx, vy)
		self._pending_state = self._client.recv_state()
		state = self._pending_state
		self._was_ball_in_play = bool(state.ball_in_play)

		obs = _state_to_obs(state)
		info: dict[str, Any] = {
			"tick": state.tick,
			"score_delta": state.score_delta,
			"flipper_hit": bool(state.flipper_hit),
			"hit_objects": [(i, p, state.ball_x, state.ball_y) for i, p in state.hits],
			"hit_base_points": [b for _, b in state.base_hits],
			"unattributed_points": state.unattributed_points,
			"drained": False,  # placement doesn't run step()'s drain-transition detector
			"ball_in_play": bool(state.ball_in_play),
			"relaunch_pending": bool(state.relaunch_pending),
			"tilted": bool(state.tilted),
			"ball2_active": bool(state.ball2_active),
		}
		return obs, info

	def render(self):
		# No screen-capture rendering by design - run the native binary without
		# PINBALL_IPC_SOCK set (or with a real DAT file visible) to watch it in its own
		# SDL2 window; this wrapper is headless/IPC-only.
		return None

	def close(self):
		if self._client is not None:
			self._client.close()
			self._client = None
		if self._proc is not None:
			self._proc.terminate()
			try:
				self._proc.wait(timeout=5.0)
			except subprocess.TimeoutExpired:
				self._proc.kill()
				self._proc.wait(timeout=5.0)
			self._proc = None
		if os.path.exists(self.sock_path):
			os.remove(self.sock_path)

	def _spawn_and_connect(self) -> None:
		env = {**os.environ, "PINBALL_IPC_SOCK": self.sock_path}
		if self.headless:
			# No visible window/GPU context needed for IPC-driven training - avoids depending
			# on a real display/window-server session entirely (relevant for background/CI
			# runs), on top of just being the correct default for a headless RL env.
			env["SDL_VIDEODRIVER"] = "dummy"
			env["SDL_AUDIODRIVER"] = "dummy"
		# Headless training workers have no use for the native binary's own console output (game
		# version banner, and - on some SDL2 builds under the dummy video driver - a per-frame
		# "SDL Error: That operation is not supported" cursor-management spam line, see
		# winmain.cpp's ImGuiConfigFlags_NoMouseCursorChange comment). Left inherited (the
		# subprocess.Popen default) for non-headless use, where a human watching a live demo
		# might want to see it. Discarding this at the source - rather than a shell-level `2>&1`
		# into a shared training log - is what actually stops multi-GB log files: confirmed
		# three separate times in one overnight run (a 2.7GB, a 1.29GB, and an 804MB train.log,
		# each almost entirely this same spam from 6-8 concurrent workers) that redirecting the
		# whole Python process's stderr wasn't enough once the child inherited and wrote to the
		# same fd directly.
		spawn_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL} if self.headless else {}
		self._proc = subprocess.Popen([self.binary_path], env=env, **spawn_kwargs)

		deadline = time.monotonic() + self.connect_timeout
		last_error: Optional[Exception] = None
		while time.monotonic() < deadline:
			if self._proc.poll() is not None:
				raise RuntimeError(
					f"native process exited early (code {self._proc.returncode}) before "
					f"accepting an IPC connection - check that a valid DAT file is present"
				)
			if os.path.exists(self.sock_path):
				try:
					self._client = PinballIPCClient(self.sock_path, connect_timeout=1.0)
					return
				except (ConnectionRefusedError, socket.timeout, OSError) as exc:
					last_error = exc
			time.sleep(0.05)

		raise TimeoutError(
			f"could not connect to {self.sock_path} within {self.connect_timeout}s"
		) from last_error
