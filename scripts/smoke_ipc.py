#!/usr/bin/env python3
"""Protocol/client smoke test for env_python/ipc_client.py.

This does NOT exercise the real C++ ipc_server.cpp - that requires a compiled binary plus a
copyrighted CADET.DAT/PINBALL.DAT file the user must supply themselves (see README). Instead
this stands up a tiny fake "engine" server in Python that speaks the exact wire format from
src_cpp/ipc_protocol.h, so PinballIPCClient's framing/parsing can be verified in isolation.
Once a real DAT file is available, run the actual binary with PINBALL_IPC_SOCK set and point
env_python/pinball_env.py at it for the real integration test.
"""
import os
import socket
import struct
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env_python.ipc_client import (  # noqa: E402
	ACTION_STRUCT, STATE_STRUCT, PinballIPCClient, _ACTION_MAGIC_INT, _STATE_MAGIC_INT,
)

NUM_TICKS = 20


def fake_engine(sock_path: str, ready: threading.Event, results: list):
	srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	srv.bind(sock_path)
	srv.listen(1)
	ready.set()
	conn, _ = srv.accept()
	with conn:
		for tick in range(1, NUM_TICKS + 1):
			# Deterministic fake state: ball drifts diagonally, score increments every 5th tick.
			state_bytes = STATE_STRUCT.pack(
				_STATE_MAGIC_INT, tick,
				float(tick), float(tick) * 2.0, 1.5, -1.5,
				1 if tick % 2 == 0 else 0, 1 if tick % 3 == 0 else 0,
				1, 0,
				100 if tick % 5 == 0 else 0,
				1 if tick == NUM_TICKS else 0,
				1 if tick % 7 == 0 else 0,
				1 if tick % 11 == 0 else 0,
				0.0, 0.0, 0.0, 0.0, 0,  # ball2 x/y/vx/vy/active
				# hit_count, hit_ids[4], hit_points[4], unattributed_points: every 5th tick the
				# fake ball hits object 1 (bumper) worth 60 and object 45 worth 40 (sum = 100).
				2 if tick % 5 == 0 else 0,
				*((1, 45, 0, 0) if tick % 5 == 0 else (0, 0, 0, 0)),
				*((60, 40, 0, 0) if tick % 5 == 0 else (0, 0, 0, 0)),
				0,
				*((30, 40, 0, 0) if tick % 5 == 0 else (0, 0, 0, 0)),  # hit_base_points (x2 on id 1)
			)
			conn.sendall(state_bytes)

			raw = conn.recv(ACTION_STRUCT.size)
			if len(raw) != ACTION_STRUCT.size:
				results.append(("error", f"short action read at tick {tick}: {len(raw)} bytes"))
				return
			magic, echoed_tick, fl, fr, launch, reset, _reset_seed = ACTION_STRUCT.unpack(raw)
			if magic != _ACTION_MAGIC_INT:
				results.append(("error", f"bad action magic at tick {tick}: {magic:#010x}"))
				return
			if echoed_tick != tick:
				results.append(("error", f"tick echo mismatch: sent {tick}, got {echoed_tick}"))
				return
			results.append(("ok", tick, fl, fr, launch, reset))
	srv.close()


def main() -> int:
	sock_path = os.path.join(tempfile.mkdtemp(), "pinball_ipc_smoke.sock")
	ready = threading.Event()
	results: list = []

	server_thread = threading.Thread(target=fake_engine, args=(sock_path, ready, results), daemon=True)
	server_thread.start()
	if not ready.wait(timeout=5.0):
		print("FAIL: fake engine server never became ready")
		return 1

	client = PinballIPCClient(sock_path, connect_timeout=5.0)
	try:
		for expected_tick in range(1, NUM_TICKS + 1):
			state = client.recv_state()
			assert state.tick == expected_tick, f"expected tick {expected_tick}, got {state.tick}"
			assert state.ball_x == float(expected_tick), state
			assert state.ball_y == float(expected_tick) * 2.0, state
			assert state.ball_in_play is True, state
			assert state.tilted is False, state
			assert state.score_delta == (100 if expected_tick % 5 == 0 else 0), state
			assert state.done == (expected_tick == NUM_TICKS), state
			assert state.flipper_hit == (expected_tick % 7 == 0), state
			assert state.relaunch_pending == (expected_tick % 11 == 0), state
			expected_hits = ((1, 60), (45, 40)) if expected_tick % 5 == 0 else ()
			assert state.hits == expected_hits, state
			assert state.hit_count == len(expected_hits), state
			assert state.base_hits == (((1, 30), (45, 40)) if expected_tick % 5 == 0 else ()), state
			assert sum(p for _, p in state.hits) + state.unattributed_points == state.score_delta, state

			from env_python.ipc_client import Action
			client.send_action(Action(
				tick=state.tick,
				flipper_left=(expected_tick % 2 == 1),
				flipper_right=(expected_tick % 4 == 0),
				launch=(expected_tick == 3),
				reset=False,
			))
	finally:
		client.close()

	server_thread.join(timeout=5.0)

	errors = [r for r in results if r[0] == "error"]
	if errors:
		for _, msg in errors:
			print(f"FAIL: {msg}")
		return 1

	if len(results) != NUM_TICKS:
		print(f"FAIL: expected {NUM_TICKS} round trips, server observed {len(results)}")
		return 1

	for _, tick, fl, fr, launch, reset in results:
		assert fl == (1 if (tick % 2 == 1) else 0), (tick, fl)
		assert fr == (1 if (tick % 4 == 0) else 0), (tick, fr)
		assert launch == (1 if tick == 3 else 0), (tick, launch)
		assert reset == 0, (tick, reset)

	print(f"OK: {NUM_TICKS}/{NUM_TICKS} StateFrame/ActionFrame round trips matched the wire format")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
