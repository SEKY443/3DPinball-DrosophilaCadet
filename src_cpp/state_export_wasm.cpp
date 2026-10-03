#include "pch.h"
#include "state_export.h"

// Wasm-build-only bridge. Compiles to an empty translation unit on native builds (guarded by
// __EMSCRIPTEN__), so it's safe to include unconditionally in CMakeLists.txt's SOURCE_FILES
// alongside ipc_server.cpp (native-only in practice; see src_cpp/patches/0002).
#ifdef __EMSCRIPTEN__
#include "pb.h"
#include "TPinballTable.h"
#include "TPlunger.h"
#include "TLight.h"
#include "TLightGroup.h"
#include "high_score.h"
#include "options.h"
#include "winmain.h"
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

	// Holds (down != 0) or releases the plunger exactly like the keyboard plunger binding does
	// (pb::InputDown/InputUp -> MainTable PlungerInputPressed/Released): while held the spring is pulled back
	// visibly and the launch power builds up; the release fires the ball. Ignored outside a running game.
	EMSCRIPTEN_KEEPALIVE
	void pb_plunger(int down)
	{
		if (!pb::MainTable)
			return;
		if (down)
		{
			if (pb::game_mode != GameModes::InGame || winmain::single_step || winmain::DemoActive)
				return;
			pb::MainTable->Message(MessageCode::PlungerInputPressed, pb::time_now);
		}
		else
		{
			pb::MainTable->Message(MessageCode::PlungerInputReleased, pb::time_now);
		}
	}

	// Read-only diagnostics for the plunger/skill-shot tuning (see web/bridge.js PLUNGER_SKILL_HOLD_TICKS):
	// 0 = number of lit skill-shot gate lights (skill_shot_lights; the count when the ball falls back into the
	// escape chute selects the Skill Shot score), 1 = "Re-Deploy" light lite200 on, 2 = plunger pull-back in
	// percent of its maximum (0 when idle), 3 = balls left (BallCount), 4 = extra balls (shoot-again).
	EMSCRIPTEN_KEEPALIVE
	int pb_diag(int what)
	{
		if (!pb::MainTable)
			return 0;
		switch (what)
		{
		case 0:
		{
			auto group = dynamic_cast<TLightGroup*>(pb::MainTable->find_component("skill_shot_lights"));
			return group ? group->Message(MessageCode::TLightGroupGetOnCount, 0.0f) : 0;
		}
		case 1:
		{
			auto light = dynamic_cast<TLight*>(pb::MainTable->find_component("lite200"));
			return light && light->light_on() ? 1 : 0;
		}
		case 2:
		{
			auto plunger = pb::MainTable->Plunger;
			return plunger && plunger->MaxPullback > 0 ? static_cast<int>(100.0f * plunger->Boost / plunger->MaxPullback) : 0;
		}
		case 3:
			return pb::MainTable->BallCount; // balls left for the current player (the one in play included)
		case 4:
			return pb::MainTable->ExtraBalls;
		default:
			return 0;
		}
	}

	// HTML menu bar bridge (web/index.html). The in-canvas ImGui menu is disabled in the browser
	// build (patch 0008); these run the same engine calls its items used to (see
	// winmain.cpp RenderUi). Commands: 1 New Game, 2 Launch Ball, 3 Pause/Resume, 4 High Scores,
	// 5 Demo, 6 Select Players (arg = 1..4), 7 Sounds, 8 Music, 9 Player Controls dialog.
	EMSCRIPTEN_KEEPALIVE
	void pb_menu_command(int cmd, int arg)
	{
		if (!pb::MainTable)
			return;
		switch (cmd)
		{
		case 1:
			winmain::new_game();
			break;
		case 2:
			winmain::end_pause();
			pb::launch_ball();
			break;
		case 3:
			winmain::pause();
			break;
		case 4:
			if (winmain::HighScoresEnabled)
			{
				winmain::pause(false);
				pb::high_scores();
			}
			break;
		case 5:
			winmain::end_pause();
			pb::toggle_demo();
			break;
		case 6:
		{
			static const Menu1 players[] = {Menu1::OnePlayer, Menu1::TwoPlayers, Menu1::ThreePlayers, Menu1::FourPlayers};
			if (arg >= 1 && arg <= 4)
			{
				options::toggle(players[arg - 1]);
				winmain::new_game();
			}
			break;
		}
		case 7:
			options::toggle(Menu1::Sounds);
			break;
		case 8:
			options::toggle(Menu1::Music);
			break;
		case 9:
			winmain::pause(false);
			options::ShowControlDialog();
			break;
		default:
			break;
		}
	}

	// Player name for the high-score table (see patch 0010): the engine inserts a qualifying score
	// directly under this name instead of opening the name-entry dialog. Empty/null clears it
	// (engine default "Player 1" is used). Truncated to the engine's 31-char name field.
	EMSCRIPTEN_KEEPALIVE
	void pb_set_player_name(const char* name)
	{
		snprintf(high_score::WasmPlayerName, sizeof high_score::WasmPlayerName, "%s", name ? name : "");
	}

	// Seeds the in-game High Scores table with a house record (the page calls it once at start-up; the browser
	// build starts with an empty table on every load). Inserted like any finished game, so it can be beaten.
	// Returns 1 if inserted, 0 if the table already holds that name/score or the score doesn't qualify.
	EMSCRIPTEN_KEEPALIVE
	int pb_seed_high_score(const char* name, int score)
	{
		if (!name)
			return 0;
		for (const auto& e : high_score::highscore_table)
			if (e.Score == score && strncmp(e.Name, name, sizeof e.Name) == 0)
				return 0;
		high_score_struct entry{};
		snprintf(entry.Name, sizeof entry.Name, "%s", name);
		entry.Score = score;
		const bool inserted = high_score::commit_score_direct(entry);
		if (inserted)
			--high_score::WasmCommitCount; // not a game the page played; keep the page's commit counter honest
		return inserted ? 1 : 0;
	}

	// High-score table readout for the page: number of scores inserted so far this session, and
	// per-rank name/score (rank 0..4; out-of-range -> "" / 0; empty slots have score -999).
	EMSCRIPTEN_KEEPALIVE
	int pb_hs_commit_count()
	{
		return high_score::WasmCommitCount;
	}

	// Number of finished games (pb::end_game calls). The engine ends a game ~3 s after the page sees
	// StateFrame.done (end-of-game timer, or immediately on New Game) and inserts the high score there.
	EMSCRIPTEN_KEEPALIVE
	int pb_game_end_count()
	{
		return high_score::WasmGameEndCount;
	}

	// Player 1's final score of the last finished game (captured inside pb::end_game, before New Game
	// resets the score), and the live score of the current player.
	EMSCRIPTEN_KEEPALIVE
	int pb_last_game_score()
	{
		return high_score::WasmLastScore;
	}

	EMSCRIPTEN_KEEPALIVE
	int pb_current_score()
	{
		return pb::MainTable ? pb::MainTable->CurScore : 0;
	}

	EMSCRIPTEN_KEEPALIVE
	const char* pb_hs_name(int rank)
	{
		return (rank >= 0 && rank < 5) ? high_score::highscore_table[rank].Name : "";
	}

	EMSCRIPTEN_KEEPALIVE
	int pb_hs_score(int rank)
	{
		return (rank >= 0 && rank < 5) ? high_score::highscore_table[rank].Score : 0;
	}

	// Bit flags: 0 paused, 1 demo active, 2 launch ball enabled, 3 high scores enabled,
	// 4 sounds on, 5 music on, 8..10 player count (1..4).
	EMSCRIPTEN_KEEPALIVE
	int pb_menu_state()
	{
		int flags = 0;
		if (winmain::single_step) flags |= 1 << 0;
		if (winmain::DemoActive) flags |= 1 << 1;
		if (winmain::LaunchBallEnabled) flags |= 1 << 2;
		if (winmain::HighScoresEnabled) flags |= 1 << 3;
		if (options::Options.Sounds) flags |= 1 << 4;
		if (options::Options.Music) flags |= 1 << 5;
		flags |= (static_cast<int>(options::Options.Players.V) & 7) << 8;
		return flags;
	}
}

#endif // __EMSCRIPTEN__
