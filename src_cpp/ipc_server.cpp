#include "pch.h"
#include "ipc_server.h"
#include "ipc_protocol.h"
#include "state_export.h"
#include "pb.h"
#include "TPinballTable.h"
#include "TPlunger.h"
#include "options.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdlib>

using namespace pinball_ipc;

namespace
{
	int g_listenFd = -1;
	int g_clientFd = -1;
	bool g_enabled = false;
	uint32_t g_tick = 0;
	uint8_t g_prevActionLeft = 0;
	uint8_t g_prevActionRight = 0;
	std::string g_sockPath;
	// Guards against redundant PlungerRelaunchBall calls: without this, an agent sending
	// repeated launch=1 pulses before the first one's ball has actually arrived would stack
	// up SomeCounter/BallFeedTimer schedules, eventually exhausting the engine's ball object
	// pool and crashing (assert failure in TPinballTable::AddBall) - confirmed via testing.
	bool g_relaunchPending = false;
	// Manual plunger pull for a ball already parked in the plunger lane (full-game play): after a
	// drain the engine feeds the next ball there itself and ball_in_play reads true, so the
	// relaunch path never applies. Held for kPlungerHoldTicks native ticks, like the web demo's
	// 400 ms synthetic Space key press.
	constexpr int kPlungerHoldTicks = 120; // 1 s at 120 updates/s: a 400 ms pull often fails to clear the lane
	int g_plungerHoldTicks = 0;

	bool BallRestsInPlungerLane(const StateFrame& f)
	{
		return f.ball_x < -6.5f && f.ball_y > 9.5f && std::fabs(f.ball_vx) < 0.5f && std::fabs(f.ball_vy) < 0.5f;
	}

	// Flipper-drill PlaceFrame bounds (see ipc_protocol.h) - generous outer box around every
	// table coordinate this project's own code already treats as in-bounds (see
	// train_pinball_circuit_cem.py's NOVELTY_X_RANGE/NOVELTY_Y_RANGE and pinball_env.py's
	// PLUNGER_REST_X/Y and DRAIN_FROM_Y), not a tight physical playfield edge. This crosses a
	// process boundary, so reject non-finite/wildly-out-of-range values instead of trusting them.
	constexpr float kPlaceMinX = -12.0f;
	constexpr float kPlaceMaxX = 12.0f;
	constexpr float kPlaceMinY = -14.0f;
	constexpr float kPlaceMaxY = 20.0f;
	// Well above the engine's own BallMaxSpeed (Radius*200, clamped again every physics tick
	// regardless - see pb.cpp::timed_frame) - just enough to catch garbage/NaN-adjacent input.
	constexpr float kPlaceMaxSpeed = 200.0f;

	bool IsValidPlacement(const PlaceFrame& p)
	{
		return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.vx) && std::isfinite(p.vy) &&
			p.x >= kPlaceMinX && p.x <= kPlaceMaxX &&
			p.y >= kPlaceMinY && p.y <= kPlaceMaxY &&
			std::fabs(p.vx) <= kPlaceMaxSpeed && std::fabs(p.vy) <= kPlaceMaxSpeed;
	}

	bool RecvExact(int fd, void* buf, size_t len)
	{
		auto* p = static_cast<uint8_t*>(buf);
		size_t got = 0;
		while (got < len)
		{
			auto n = recv(fd, p + got, len - got, 0);
			if (n <= 0)
				return false;
			got += static_cast<size_t>(n);
		}
		return true;
	}

	bool SendExact(int fd, const void* buf, size_t len)
	{
		auto* p = static_cast<const uint8_t*>(buf);
		size_t sent = 0;
		while (sent < len)
		{
			auto n = send(fd, p + sent, len - sent, 0);
			if (n <= 0)
				return false;
			sent += static_cast<size_t>(n);
		}
		return true;
	}

	// Sends the same held/released MessageCode pb::InputDown/InputUp send for real keyboard
	// input, and mirrors the transition into StateExport so Capture() reports it back.
	void ApplyFlipperEdge(StateExport::FlipperSide side, uint8_t prevHeld, uint8_t curHeld,
	                       MessageCode pressed, MessageCode released)
	{
		if (curHeld == prevHeld)
			return;
		pb::MainTable->Message(curHeld ? pressed : released, pb::time_now);
		StateExport::OnFlipperInput(side, curHeld != 0);
	}
}

void IpcServer::Init()
{
	// Table-map export for scripts/gen_table_map.py: independent of the IPC socket, so the dump
	// can be produced by a plain headless launch. Runs after pb::init() built MainTable and
	// control::make_links() resolved every score component.
	const char* mapPathEnv = std::getenv("PINBALL_DUMP_TABLE_MAP");
	if (mapPathEnv && mapPathEnv[0])
	{
		if (!StateExport::DumpTableMap(mapPathEnv))
			fprintf(stderr, "IPC: failed to write table map to %s\n", mapPathEnv);
	}

	const char* sockPathEnv = std::getenv("PINBALL_IPC_SOCK");
	if (!sockPathEnv || !sockPathEnv[0])
	{
		// No socket path configured: run as a normal, unmanaged build for manual play/debugging.
		g_enabled = false;
		return;
	}
	g_sockPath = sockPathEnv;

	g_listenFd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (g_listenFd < 0)
	{
		pb::ShowMessageBox(SDL_MESSAGEBOX_ERROR, "IPC", "Failed to create IPC socket");
		return;
	}

	sockaddr_un addr{};
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, g_sockPath.c_str(), sizeof(addr.sun_path) - 1);
	unlink(g_sockPath.c_str());

	if (bind(g_listenFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
		listen(g_listenFd, 1) != 0)
	{
		pb::ShowMessageBox(SDL_MESSAGEBOX_ERROR, "IPC", "Failed to bind/listen on IPC socket");
		close(g_listenFd);
		g_listenFd = -1;
		return;
	}

	printf("IPC: waiting for RL client on %s\n", g_sockPath.c_str());
	g_clientFd = accept(g_listenFd, nullptr, nullptr);
	if (g_clientFd < 0)
	{
		pb::ShowMessageBox(SDL_MESSAGEBOX_ERROR, "IPC", "Failed to accept IPC client");
		return;
	}

	g_enabled = true;
	StateExport::ResetTrackedState();

	// An IPC-driven run is always an automated client (env_python/pinball_env.py), never a
	// human watching the window - skip winmain::MainLoop's real-display-refresh pacing
	// (see winmain.cpp's `targetTimeDelta > 0 && !UncappedUpdatesPerSecond` sleep_for gate)
	// so training throughput is CPU-bound instead of capped at ~60 physics ticks/sec.
	options::Options.UncappedUpdatesPerSecond = true;
}

void IpcServer::Shutdown()
{
	if (g_clientFd >= 0)
		close(g_clientFd);
	if (g_listenFd >= 0)
		close(g_listenFd);
	if (!g_sockPath.empty())
		unlink(g_sockPath.c_str());
	g_clientFd = g_listenFd = -1;
	g_enabled = false;
}

bool IpcServer::IsEnabled()
{
	return g_enabled;
}

// A lost or misbehaving client ends the process: this engine exists only to serve that one
// client, and after Shutdown() the main loop would otherwise keep running the game unpaced
// (headless, no vsync) at 100% CPU forever, orphaned under PID 1.
static void DropClientAndQuit()
{
	IpcServer::Shutdown();
	SDL_Event event{SDL_QUIT};
	SDL_PushEvent(&event);
}

void IpcServer::SyncTick(float timeDeltaSec)
{
	if (!g_enabled)
		return;

	auto frame = StateExport::Capture(timeDeltaSec);
	frame.tick = ++g_tick;
	frame.relaunch_pending = g_relaunchPending ? 1 : 0;

	if (!SendExact(g_clientFd, &frame, sizeof(frame)))
	{
		DropClientAndQuit();
		return;
	}

	// Read the magic first so a PLC! (flipper-drill placement) reply can be told apart from a
	// normal ACT! reply before committing to either frame's remaining size on the wire - see
	// ipc_protocol.h::PlaceFrame. Exactly one reply of either kind is expected per StateFrame,
	// so the lockstep invariant holds regardless of which one arrives.
	uint32_t magic = 0;
	if (!RecvExact(g_clientFd, &magic, sizeof(magic)))
	{
		DropClientAndQuit();
		return;
	}

	if (magic == kPlaceMagic)
	{
		PlaceFrame place{};
		place.magic = magic;
		if (!RecvExact(g_clientFd, reinterpret_cast<uint8_t*>(&place) + sizeof(magic),
		               sizeof(place) - sizeof(magic)))
		{
			DropClientAndQuit();
			return;
		}
		if (IsValidPlacement(place))
		{
			StateExport::PlaceBall(place.x, place.y, place.vx, place.vy);
			StateExport::NotifyBallTeleported();
		}
		// Flipper held-state, launch, and reset are all left exactly as they were this tick -
		// a placement replaces an action, it doesn't imply one. The outer per-tick loop
		// (pb::timed_frame) continues normally from here, same as the ACT! path below.
		return;
	}

	if (magic != kActionMagic)
	{
		DropClientAndQuit();
		return;
	}

	ActionFrame action{};
	action.magic = magic;
	if (!RecvExact(g_clientFd, reinterpret_cast<uint8_t*>(&action) + sizeof(magic),
	               sizeof(action) - sizeof(magic)))
	{
		DropClientAndQuit();
		return;
	}

	ApplyFlipperEdge(StateExport::FlipperSide::Left, g_prevActionLeft, action.flipper_left,
	                  MessageCode::LeftFlipperInputPressed, MessageCode::LeftFlipperInputReleased);
	ApplyFlipperEdge(StateExport::FlipperSide::Right, g_prevActionRight, action.flipper_right,
	                  MessageCode::RightFlipperInputPressed, MessageCode::RightFlipperInputReleased);
	g_prevActionLeft = action.flipper_left;
	g_prevActionRight = action.flipper_right;

	// Only an explicit launch request with the ball resting in the lane starts a pull, so the
	// training env (which never requests a launch while ball_in_play is true) is unaffected.
	if (g_plungerHoldTicks > 0)
	{
		if (--g_plungerHoldTicks == 0)
			pb::MainTable->Message(MessageCode::PlungerInputReleased, pb::time_now);
	}
	else if (action.launch && frame.ball_in_play && BallRestsInPlungerLane(frame))
	{
		pb::MainTable->Message(MessageCode::PlungerInputPressed, pb::time_now);
		g_plungerHoldTicks = kPlungerHoldTicks;
	}

	// launch/reset are level-triggered pulses: the client is expected to set them for
	// exactly one ActionFrame, so no edge-tracking is needed for these two.
	//
	// PlungerRelaunchBall (not pb::launch_ball()/PlungerLaunchBall) deliberately: the naive
	// path sets Boost=MaxPullback and starts a short-lived ReleasedTimer boost window
	// immediately, racing against PlungerFeedBall's own ~1s ball-feed delay - if the ball
	// isn't physically fed into the plunger yet (e.g. right after reset()), the boost window
	// expires (Boost reset to 0 by ReleasedTimer) before the ball ever arrives to receive it,
	// so the "launch" silently does nothing. Confirmed via telemetry: the ball was observed
	// stuck near its rest position across entire episodes, never reaching the flippers.
	// PlungerRelaunchBall instead schedules the feed itself AND increments SomeCounter, which
	// makes TPlunger::Collision() apply a guaranteed near-max boost on first real contact with
	// the ball, however long the feed actually takes - no race.
	if (frame.ball_in_play)
		g_relaunchPending = false;
	// !StateExport::AnyBallActive() is the real safety gate here (see its declaration for why):
	// !frame.ball_in_play alone isn't enough, since it only reflects BallList[0]'s position and
	// says nothing about a ball the engine's own lock/sink/unstuck logic fed into a different
	// slot. Without this, that ball stays alive while we ALSO feed a fresh one - a real second
	// ball, not a learned strategy.
	if (action.launch && !g_relaunchPending && !frame.ball_in_play && !StateExport::AnyBallActive())
	{
		pb::MainTable->Plunger->Message(MessageCode::PlungerRelaunchBall, 0.1f);
		g_relaunchPending = true;
	}

	if (action.reset)
	{
		// See ipc_protocol.h::ActionFrame::reset_seed: reseed before Reset takes effect so this
		// episode's RandFloat()-driven physics/scoring are reproducible from the seed alone, not
		// from how long this worker process has already been running.
		srand(action.reset_seed);
		// Plain Reset (not NewGame) deliberately: NewGame's StartGamePlayer1 step triggers a
		// TLightGroup light-flourish animation (TLightGroupReset/TLightResetAndTurnOff) whose
		// scheduled timer callbacks crashed with SIGSEGV in timer::set under back-to-back
		// RL-driven resets (see project history - crash confirmed via macOS crash report,
		// faulting in TLight::schedule_timeout -> timer::set). TBall/TFlipper/TPlunger all
		// reset themselves adequately on plain Reset (ball position, flipper angle, plunger
		// timers) without going through that path - score isn't zeroed by Reset alone, but
		// that's fine since reward uses score_delta rebased by ResetTrackedState() below.
		pb::mode_change(GameModes::InGame);
		pb::MainTable->Message(MessageCode::Reset, 0.0f);
		// Ball feeding is deferred entirely to the launch action's PlungerRelaunchBall call
		// above (not fed here) - PlungerRelaunchBall already handles feed+guaranteed-launch
		// together, so pre-feeding here would just create a second, racier path to the same
		// "ball sitting in the plunger" state.
		StateExport::ResetTrackedState();
		g_relaunchPending = false;
		g_plungerHoldTicks = 0;
	}
}
