#include "pch.h"
#include "state_export.h"

// Wasm-build-only bridge. Compiles to an empty translation unit on native builds (guarded by
// __EMSCRIPTEN__), so it's safe to include unconditionally in CMakeLists.txt's SOURCE_FILES
// alongside ipc_server.cpp (native-only in practice; see src_cpp/patches/0002).
#ifdef __EMSCRIPTEN__
#include "pb.h"
#include "TPinballTable.h"
#include "TPlunger.h"
#include <emscripten.h>

namespace
{
	pinball_ipc::StateFrame g_lastFrame{};
	uint8_t g_prevActionLeft = 0;
	uint8_t g_prevActionRight = 0;
	// See ipc_server.cpp's identical guard: without it, repeated launch requests before the
	// first one's ball arrives exhaust the engine's ball object pool and abort().
	bool g_relaunchPending = false;
	// StateExport::Capture() deliberately leaves magic/tick to the caller (see its header
	// comment) - ipc_server.cpp assigns its own g_tick the same way; without this, every
	// exported StateFrame reads tick=0 forever, which breaks any JS-side "decide every N ticks"
	// cadence (see web/bridge.js) since it would never see tick advance at all.
	uint32_t g_tick = 0;
}

void WasmBridge_CaptureTick(float timeDeltaSec)
{
	g_lastFrame = StateExport::Capture(timeDeltaSec);
	g_lastFrame.tick = ++g_tick;
	g_lastFrame.relaunch_pending = g_relaunchPending ? 1 : 0;
}

extern "C"
{
	// Returns a pointer into Wasm linear memory; JS reads sizeof(StateFrame) bytes from it
	// via a DataView (see web/bridge.js). Layout must stay in lockstep with ipc_protocol.h.
	EMSCRIPTEN_KEEPALIVE
	const void* pb_get_state_ptr()
	{
		return &g_lastFrame;
	}

	EMSCRIPTEN_KEEPALIVE
	int pb_get_state_size()
	{
		return static_cast<int>(sizeof(pinball_ipc::StateFrame));
	}

	// Sends the same held/released MessageCode pb::InputDown/InputUp send for real keyboard
	// input, and mirrors the transition into StateExport - same pattern as
	// ipc_server.cpp's ApplyFlipperEdge, just called from JS instead of a socket.
	EMSCRIPTEN_KEEPALIVE
	void pb_set_action(uint8_t flipperLeft, uint8_t flipperRight, uint8_t launch)
	{
		if (!pb::MainTable)
			return;

		if (flipperLeft != g_prevActionLeft)
		{
			pb::MainTable->Message(
				flipperLeft ? MessageCode::LeftFlipperInputPressed : MessageCode::LeftFlipperInputReleased,
				pb::time_now);
			StateExport::OnFlipperInput(StateExport::FlipperSide::Left, flipperLeft != 0);
			g_prevActionLeft = flipperLeft;
		}
		if (flipperRight != g_prevActionRight)
		{
			pb::MainTable->Message(
				flipperRight ? MessageCode::RightFlipperInputPressed : MessageCode::RightFlipperInputReleased,
				pb::time_now);
			StateExport::OnFlipperInput(StateExport::FlipperSide::Right, flipperRight != 0);
			g_prevActionRight = flipperRight;
		}

		// PlungerRelaunchBall, not pb::launch_ball() - see ipc_server.cpp for why the naive
		// path silently does nothing if the ball hasn't physically arrived at the plunger yet.
		// !StateExport::AnyBallActive() mirrors ipc_server.cpp's identical fix: ball_in_play
		// alone only reflects BallList[0]'s position and misses a ball fed into another slot by
		// the engine's own lock/sink/unstuck logic, which would otherwise stay alive while we
		// feed a second one.
		if (g_lastFrame.ball_in_play)
			g_relaunchPending = false;
		if (launch && !g_relaunchPending && !g_lastFrame.ball_in_play && !StateExport::AnyBallActive())
		{
			pb::MainTable->Plunger->Message(MessageCode::PlungerRelaunchBall, 0.1f);
			g_relaunchPending = true;
		}
	}
}

#endif // __EMSCRIPTEN__
