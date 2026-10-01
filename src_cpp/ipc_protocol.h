#pragma once
#include <cstdint>

// Wire format shared by src_cpp/ipc_server.cpp (writer), env_python/ipc_client.py (reader/writer,
// mirrored with struct.Struct), and eventually web/bridge.js (reads the equivalent layout out of
// Wasm linear memory). Keep the three in lockstep whenever a field is added, removed, or reordered.
namespace pinball_ipc
{
	// Byte order intentionally spells out the ASCII tag on a little-endian machine (first byte sent = 'P'/'A').
	constexpr uint32_t kStateMagic = static_cast<uint32_t>('P') | (static_cast<uint32_t>('I') << 8) |
		(static_cast<uint32_t>('N') << 16) | (static_cast<uint32_t>('B') << 24); // "PINB"
	constexpr uint32_t kActionMagic = static_cast<uint32_t>('A') | (static_cast<uint32_t>('C') << 8) |
		(static_cast<uint32_t>('T') << 16) | (static_cast<uint32_t>('!') << 24); // "ACT!"
	// "Flipper drill" placement command - see ipc_server.cpp's SyncTick: sent INSTEAD of an
	// ActionFrame for one StateFrame reply, never alongside one (lockstep is still one reply
	// per StateFrame either way).
	constexpr uint32_t kPlaceMagic = static_cast<uint32_t>('P') | (static_cast<uint32_t>('L') << 8) |
		(static_cast<uint32_t>('C') << 16) | (static_cast<uint32_t>('!') << 24); // "PLC!"

#pragma pack(push, 1)
	struct StateFrame
	{
		uint32_t magic;
		uint32_t tick;
		float ball_x;
		float ball_y;
		float ball_vx;
		float ball_vy;
		uint8_t flipper_left;
		uint8_t flipper_right;
		uint8_t ball_in_play;
		uint8_t tilted;
		int32_t score_delta;
		uint8_t done;
		uint8_t flipper_hit; // pulse: an active flipper physically struck the ball this tick
		// True while a launch request is already in flight (fed but not yet resolved) - any
		// further launch=1 this tick is a guaranteed no-op (see ipc_server.cpp's
		// g_relaunchPending guard). Lets the RL client exclude redundant repeated launch
		// decisions from the training signal instead of diluting credit across every tick of
		// the ~1s feed delay when only the first press in that window actually did anything.
		uint8_t relaunch_pending;
		uint8_t _pad[1];
		// Second ball, for multiball (see TPinballTable.cpp's MultiballFlag - real DEMO.DAT
		// tables enable this). Previously the engine only ever reported BallList[0]; a second
		// ball could appear, score, and drain completely invisibly to the RL client. Zeroed
		// (with ball2_active=0) whenever BallList.size() < 2 or the second slot isn't
		// ActiveFlag, so a client can tell "no second ball" apart from "second ball at (0,0)".
		float ball2_x;
		float ball2_y;
		float ball2_vx;
		float ball2_vy;
		uint8_t ball2_active;
		// ---- Scoring-object hits (appended; every field above keeps its offset) ----
		// Ids are control::score_components index + 1 (0 = empty slot); see agents/table_map.json
		// and StateExport::BeginControl. hit_count is the number of DISTINCT scoring objects the
		// ball contacted since the previous frame (ControlCollision / ControlBallCaptured) and may
		// exceed kMaxHits, in which case only the first kMaxHits are listed. hit_points[i] is every
		// point AddScore() added while hit_ids[i]'s handler was the outermost active one (incl.
		// multipliers and mission/jackpot awards it triggered). unattributed_points is score added
		// outside any ball-contact handler (timers, drain bonus countdown, light-group bonus).
		// Points of objects beyond the first kMaxHits are folded into unattributed_points, so the
		// invariant score_delta == sum(hit_points) + unattributed_points always holds (barring the
		// engine's 1e9 score wrap, which Capture() already corrects score_delta for).
		uint8_t hit_count;
		uint8_t hit_ids[4];
		int32_t hit_points[4];
		int32_t unattributed_points;
		// hit_points[i] before the table's score multiplier (x1/x2/x3/x5/x10) was applied: the sum
		// of the raw `score` arguments AddScore() received for that object. Lets a reward value
		// the object itself rather than the multiplier state the table happens to be in.
		int32_t hit_base_points[4];
	};
	constexpr int kMaxHits = 4;

	struct ActionFrame
	{
		uint32_t magic;
		uint32_t tick; // echoes the StateFrame.tick this action responds to
		uint8_t flipper_left;
		uint8_t flipper_right;
		uint8_t launch; // pulse: set for exactly one ActionFrame to trigger pb::launch_ball()
		uint8_t reset; // pulse: set for exactly one ActionFrame to start a new game
		// Only meaningful when reset=1: reseeds libc rand()/RandFloat() (see pch.h) right before
		// the reset takes effect. The engine never seeds this RNG on its own (confirmed: no
		// srand() call anywhere in vendor/SpaceCadetPinball), so every process previously ran on
		// one continuously-advancing, unseeded stream for its whole lifetime - meaning a fixed
		// set of weights got measurably different scores (RandFloat()-driven bumper/plunger
		// jitter and TLightGroup's own RandFloat()*1e6 score bonus, pb.cpp) depending purely on
		// how many prior ticks that worker process happened to have already consumed, not on
		// anything the policy did. A client-supplied per-episode seed makes fitness reproducible
		// (same seed -> same episode) while still varying seed-to-seed for real CEM averaging.
		uint32_t reset_seed;
	};

	// Flipper-drill ball placement, in reply to a StateFrame, in place of an ActionFrame for
	// that one tick (see ipc_server.cpp::SyncTick). x/y are table coordinates; vx/vy are in the
	// same table-units-per-second as StateFrame::ball_vx/vy (see state_export.cpp::Capture).
	// Only ever applied to BallList[0], and only while it is active - see StateExport::PlaceBall.
	struct PlaceFrame
	{
		uint32_t magic;
		uint32_t tick; // echoes the StateFrame.tick this placement responds to
		float x;
		float y;
		float vx;
		float vy;
	};
#pragma pack(pop)

	static_assert(sizeof(StateFrame) == 94, "StateFrame layout changed - update env_python/ipc_client.py");
	static_assert(sizeof(ActionFrame) == 16, "ActionFrame layout changed - update env_python/ipc_client.py");
	static_assert(sizeof(PlaceFrame) == 24, "PlaceFrame layout changed - update env_python/ipc_client.py");
}
