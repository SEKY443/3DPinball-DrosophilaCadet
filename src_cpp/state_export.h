#pragma once
#include "ipc_protocol.h"

class TPinballComponent;
enum class MessageCode;

// Reads ball physics, flipper state, and score deltas out of pb::MainTable each tick.
// Used by both ipc_server.cpp (native build) and state_export_wasm.cpp (browser build),
// so this file must stay free of POSIX/Emscripten-specific includes.
namespace StateExport
{
	enum class FlipperSide { Left, Right };

	// Called whenever a flipper's held-state changes, whether the source is real keyboard
	// input (pb::InputDown/InputUp) or an IPC/Wasm action frame. TFlipper exposes no public
	// held-state field, so this is the single source of truth Capture() reads back from.
	void OnFlipperInput(FlipperSide side, bool pressed);

	// Called from TFlipper::FlipperCollision when an active flipper physically strikes a
	// ball (see src_cpp/patches/0004). This is the unambiguous "did the paddle touch the
	// ball" signal score_delta alone can't give you - score also comes from bumpers/targets
	// the ball hits passively, with no flipper skill involved.
	void OnFlipperBallHit();

	// Scoring-object attribution (see src_cpp/patches/0007). control::handler() is the engine's
	// single dispatch point for every table object's game logic: a component's own Collision()
	// calls control::handler(ControlCollision, this) when the ball hits/rolls over it (THole uses
	// ControlBallCaptured), and every AddScore() for that hit happens inside that call (directly,
	// or nested via MissionControl/light groups). BeginControl/EndControl bracket each handler()
	// call; OnScoreAdded is called from TPinballTable::AddScore with the points actually added.
	// Returns the previous attribution id, to be passed back to EndControl (supports nesting:
	// the OUTERMOST ball-contact handler owns every point scored inside it).
	uint8_t BeginControl(MessageCode code, TPinballComponent* cmp);
	void EndControl(uint8_t previousAttribution);
	void OnScoreAdded(int addedScore, int baseScore);

	// Writes a JSON description of every control::score_components entry (stable id = index+1,
	// engine name, C++ class, collision AABB in table coordinates, runtime score values) to
	// `path`. Called once from IpcServer::Init when PINBALL_DUMP_TABLE_MAP is set; consumed by
	// scripts/gen_table_map.py. Returns false if the file could not be written.
	bool DumpTableMap(const char* path);

	// Clears held-state and rebases score-delta tracking against the current score.
	// Call on pb::init() and again after every new-game/reset.
	void ResetTrackedState();

	// Fills every StateFrame field except magic/tick (the IPC layer assigns those).
	// timeDeltaSec must be the same value passed to pb::timed_frame() this tick, since
	// ball velocity is derived from (Position - PrevPosition) / timeDeltaSec.
	pinball_ipc::StateFrame Capture(float timeDeltaSec);

	// True if any slot in pb::MainTable->BallList currently has ActiveFlag set. Unlike
	// StateFrame::ball_in_play (a position-distance heuristic on BallList[0] only, used for
	// observation/reward), this checks the engine's own authoritative per-ball flag across
	// every slot - the correct check before issuing an automated relaunch. A human's manual
	// plunger press is naturally gated by requiring an actual key-down/key-up cycle and, in
	// FullTiltMode, by TPlunger::PlungerInputPressed's own `MultiballCount > 0` condition; our
	// IPC/Wasm launch path calls PlungerRelaunchBall directly (see ipc_server.cpp's comment on
	// why), which has no such gate and will happily feed a new ball via AddBall() even while
	// another one is still alive elsewhere on the table (e.g. mid-flight right after the
	// engine's own ball-lock/sink logic in control.cpp fed one, or during the natural
	// drain-to-refeed window) - producing a real, unintended second ball.
	bool AnyBallActive();

	// Flipper-drill support: places BallList[0] at (x, y) with velocity (vx, vy) - the latter in
	// the same table-units-per-second as Capture()'s ball_vx/vy - if that ball is currently
	// active; a no-op otherwise. Mirrors the (re)positioning bookkeeping TPinballTable::AddBall
	// and TBall::Message(Reset) perform (collision state, stuck-ball watchdog) so a placed ball
	// behaves like a normally-fed one, not a half-reset one. Caller (ipc_server.cpp) is
	// responsible for validating x/y/vx/vy before calling this.
	void PlaceBall(float x, float y, float vx, float vy);

	// Clears ball-1's previous-position tracking so the next Capture() reports velocity 0
	// instead of a teleport spike. Call once right after PlaceBall.
	void NotifyBallTeleported();
}

#ifdef __EMSCRIPTEN__
// Implemented in state_export_wasm.cpp - the Wasm build's equivalent of IpcServer::SyncTick.
// No sockets/blocking: just refreshes the frame JS reads out of Wasm memory on its own
// requestAnimationFrame cadence via the pb_get_state_ptr()/pb_get_state_size() exports.
void WasmBridge_CaptureTick(float timeDeltaSec);
#endif
