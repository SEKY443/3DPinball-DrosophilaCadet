"""Unix-domain-socket client for the wire format defined in src_cpp/ipc_protocol.h.

Keep STATE_FORMAT/ACTION_FORMAT in exact lockstep with that header - a field added, removed,
or reordered on the C++ side must be mirrored here (and in web/bridge.js's DataView reads).
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

STATE_MAGIC = b"PINB"
ACTION_MAGIC = b"ACT!"
PLACE_MAGIC = b"PLC!"

# <  little-endian, no alignment padding (mirrors #pragma pack(push, 1))
# I magic, I tick, f ball_x, f ball_y, f ball_vx, f ball_vy,
# B flipper_left, B flipper_right, B ball_in_play, B tilted,
# i score_delta, B done, B flipper_hit, B relaunch_pending, 1x explicit pad,
# f ball2_x, f ball2_y, f ball2_vx, f ball2_vy, B ball2_active,
# B hit_count, 4B hit_ids, 4i hit_points, i unattributed_points, 4i hit_base_points
# (appended 2026-09 - step 3)
STATE_FORMAT = "<IIffffBBBBiBBB1xffffBB4B4ii4i"
# Max scoring-object hits listed per frame (ipc_protocol.h::kMaxHits).
MAX_HITS = 4
# I magic, I tick, B flipper_left, B flipper_right, B launch, B reset, I reset_seed
ACTION_FORMAT = "<IIBBBBI"
# I magic, I tick, f x, f y, f vx, f vy - see ipc_protocol.h::PlaceFrame. Sent INSTEAD of an
# ActionFrame in reply to one StateFrame (flipper-drill ball placement), never alongside one.
PLACE_FORMAT = "<IIffff"

STATE_STRUCT = struct.Struct(STATE_FORMAT)
ACTION_STRUCT = struct.Struct(ACTION_FORMAT)
PLACE_STRUCT = struct.Struct(PLACE_FORMAT)

assert STATE_STRUCT.size == 94, "STATE_FORMAT must match ipc_protocol.h::StateFrame (94 bytes)"
assert ACTION_STRUCT.size == 16, "ACTION_FORMAT must match ipc_protocol.h::ActionFrame (16 bytes)"
assert PLACE_STRUCT.size == 24, "PLACE_FORMAT must match ipc_protocol.h::PlaceFrame (24 bytes)"

_STATE_MAGIC_INT = struct.unpack("<I", STATE_MAGIC)[0]
_ACTION_MAGIC_INT = struct.unpack("<I", ACTION_MAGIC)[0]
_PLACE_MAGIC_INT = struct.unpack("<I", PLACE_MAGIC)[0]


@dataclass(frozen=True)
class State:
	tick: int
	ball_x: float
	ball_y: float
	ball_vx: float
	ball_vy: float
	flipper_left: bool
	flipper_right: bool
	ball_in_play: bool
	tilted: bool
	score_delta: int
	done: bool
	flipper_hit: bool
	relaunch_pending: bool
	ball2_x: float
	ball2_y: float
	ball2_vx: float
	ball2_vy: float
	ball2_active: bool
	# Scoring objects the ball contacted since the previous frame, as (id, points) pairs - id is
	# the stable agents/table_map.json id (control::score_components index + 1), points are the
	# points attributed to that contact (0 for e.g. a rollover that only toggles a light).
	# hit_count may exceed len(hits) (only MAX_HITS are listed). score_delta ==
	# sum(points) + unattributed_points - see ipc_protocol.h.
	hit_count: int = 0
	hits: tuple = ()
	unattributed_points: int = 0
	# (id, points before the score multiplier) - same order/ids as `hits`.
	base_hits: tuple = ()

	@classmethod
	def from_bytes(cls, raw: bytes) -> "State":
		fields = STATE_STRUCT.unpack(raw)
		(magic, tick, bx, by, bvx, bvy, fl, fr, bip, tilt, score_delta, done, flipper_hit,
		 relaunch_pending, b2x, b2y, b2vx, b2vy, b2active, hit_count) = fields[:20]
		hit_ids = fields[20:20 + MAX_HITS]
		hit_points = fields[20 + MAX_HITS:20 + 2 * MAX_HITS]
		unattributed = fields[20 + 2 * MAX_HITS]
		hit_base = fields[21 + 2 * MAX_HITS:21 + 3 * MAX_HITS]
		if magic != _STATE_MAGIC_INT:
			raise ValueError(f"bad StateFrame magic: {magic:#010x}")
		return cls(
			tick=tick,
			ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
			flipper_left=bool(fl), flipper_right=bool(fr),
			ball_in_play=bool(bip), tilted=bool(tilt),
			score_delta=score_delta, done=bool(done), flipper_hit=bool(flipper_hit),
			relaunch_pending=bool(relaunch_pending),
			ball2_x=b2x, ball2_y=b2y, ball2_vx=b2vx, ball2_vy=b2vy,
			ball2_active=bool(b2active),
			hit_count=hit_count,
			hits=tuple((i, p) for i, p in zip(hit_ids, hit_points) if i != 0),
			unattributed_points=unattributed,
			base_hits=tuple((i, b) for i, b in zip(hit_ids, hit_base) if i != 0),
		)


@dataclass(frozen=True)
class Action:
	tick: int
	flipper_left: bool = False
	flipper_right: bool = False
	launch: bool = False
	reset: bool = False
	reset_seed: int = 0  # only consulted by the engine when reset=True
	def to_bytes(self) -> bytes:
		return ACTION_STRUCT.pack(
			_ACTION_MAGIC_INT, self.tick,
			int(self.flipper_left), int(self.flipper_right),
			int(self.launch), int(self.reset), self.reset_seed,
		)


class PinballIPCClient:
	"""Lockstep client: each recv_state() must be followed by exactly one send_action()
	before the native process's next tick can proceed (see IpcServer::SyncTick)."""

	def __init__(self, sock_path: str, connect_timeout: float = 10.0):
		self.sock_path = sock_path
		self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		self._sock.settimeout(connect_timeout)
		self._sock.connect(sock_path)
		self._sock.settimeout(None)

	def recv_state(self) -> State:
		raw = self._recv_exact(STATE_STRUCT.size)
		return State.from_bytes(raw)

	def send_action(self, action: Action) -> None:
		self._sock.sendall(action.to_bytes())

	def send_place(self, tick: int, x: float, y: float, vx: float, vy: float) -> None:
		"""Flipper-drill placement: sent INSTEAD of send_action() in reply to a recv_state(),
		never in addition to it - see ipc_protocol.h::PlaceFrame and ipc_server.cpp::SyncTick."""
		self._sock.sendall(PLACE_STRUCT.pack(_PLACE_MAGIC_INT, tick, x, y, vx, vy))

	def close(self) -> None:
		self._sock.close()

	def __enter__(self) -> "PinballIPCClient":
		return self

	def __exit__(self, *exc_info) -> None:
		self.close()

	def _recv_exact(self, n: int) -> bytes:
		chunks = []
		remaining = n
		while remaining > 0:
			chunk = self._sock.recv(remaining)
			if not chunk:
				raise ConnectionError("IPC socket closed by native process")
			chunks.append(chunk)
			remaining -= len(chunk)
		return b"".join(chunks)
