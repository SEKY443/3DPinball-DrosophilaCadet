#include "pch.h"
#include "state_export.h"
#include "pb.h"
#include "TBall.h"
#include "TPinballTable.h"
#include "TCollisionComponent.h"
#include "control.h"

#include <cstdio>
#include <typeinfo>

namespace
{
	bool g_flipperLeftHeld = false;
	bool g_flipperRightHeld = false;
	bool g_flipperHitPending = false;
	int g_prevScore = 0;

	// Our own previous-position tracking for velocity, since TBall::PrevPosition is NOT "position
	// last frame" - see Capture()'s comment below for why that assumption was wrong.
	float g_prevBall1X = 0.0f, g_prevBall1Y = 0.0f;
	bool g_hasPrevBall1 = false;
	float g_prevBall2X = 0.0f, g_prevBall2Y = 0.0f;
	bool g_hasPrevBall2 = false;

	// Scoring-object hits accumulated since the last Capture() (see BeginControl). One entry per
	// DISTINCT object, in first-contact order; 88 score components exist, so a fixed array of
	// that size can never overflow.
	constexpr size_t kScoreComponentCount = sizeof(control::score_components) / sizeof(control::score_components[0]);
	uint8_t g_hitIds[kScoreComponentCount];
	int32_t g_hitPoints[kScoreComponentCount];
	int32_t g_hitBasePoints[kScoreComponentCount];
	size_t g_hitCount = 0;
	int32_t g_unattributedPoints = 0;
	uint8_t g_attribution = 0; // id owning points added right now (0 = none)

	// Stable id of a component: its index in control::score_components + 1, found through the
	// component_control pointer control::make_links() stored on it. 0 if not a score component.
	uint8_t ScoreComponentId(const TPinballComponent* cmp)
	{
		if (!cmp || !cmp->Control)
			return 0;
		for (size_t i = 0; i < kScoreComponentCount; ++i)
		{
			if (&control::score_components[i].Control == cmp->Control)
				return static_cast<uint8_t>(i + 1);
		}
		return 0;
	}

	size_t HitSlot(uint8_t id)
	{
		for (size_t i = 0; i < g_hitCount; ++i)
		{
			if (g_hitIds[i] == id)
				return i;
		}
		g_hitIds[g_hitCount] = id;
		g_hitPoints[g_hitCount] = 0;
		g_hitBasePoints[g_hitCount] = 0;
		return g_hitCount++;
	}

	void ClearHits()
	{
		g_hitCount = 0;
		g_unattributedPoints = 0;
	}
}

static_assert(sizeof(control::score_components) / sizeof(control::score_components[0]) < 255,
              "score component ids must fit in StateFrame::hit_ids (uint8_t)");

uint8_t StateExport::BeginControl(MessageCode code, TPinballComponent* cmp)
{
	const auto previous = g_attribution;
	if (code != MessageCode::ControlCollision && code != MessageCode::ControlBallCaptured)
		return previous;
	const auto id = ScoreComponentId(cmp);
	if (id == 0)
		return previous;
	HitSlot(id);
	if (g_attribution == 0)
		g_attribution = id; // outermost ball-contact handler owns everything scored inside it
	return previous;
}

void StateExport::EndControl(uint8_t previousAttribution)
{
	g_attribution = previousAttribution;
}

void StateExport::OnScoreAdded(int addedScore, int baseScore)
{
	if (g_attribution == 0)
	{
		g_unattributedPoints += addedScore;
		return;
	}
	const auto slot = HitSlot(g_attribution);
	g_hitPoints[slot] += addedScore;
	g_hitBasePoints[slot] += baseScore;
}

bool StateExport::DumpTableMap(const char* path)
{
	auto table = pb::MainTable;
	if (!table || !path)
		return false;
	FILE* f = std::fopen(path, "w");
	if (!f)
		return false;

	std::fprintf(f, "{\n  \"components\": [\n");
	for (size_t i = 0; i < kScoreComponentCount; ++i)
	{
		const auto& info = control::score_components[i];
		auto cmp = info.Tag.GetComponent();
		// typeid().name() is the Itanium-mangled class name ("7TBumper") on clang/gcc - strip the
		// length prefix. Only ever printed, never parsed back by the engine.
		const char* cls = cmp ? typeid(*cmp).name() : "";
		while (*cls >= '0' && *cls <= '9')
			++cls;
		std::fprintf(f, "    {\"id\": %zu, \"name\": \"%s\", \"class\": \"%s\", \"linked\": %s",
		             i + 1, info.Tag.Name, cls, cmp ? "true" : "false");

		auto coll = dynamic_cast<TCollisionComponent*>(cmp);
		// Constructor seeds AABB inverted ({XMax,YMax,XMin,YMin} = {-1e4,-1e4,1e4,1e4}); it only
		// becomes a real box once place_in_grid() merged at least one edge into it.
		if (coll && coll->AABB.XMax >= coll->AABB.XMin && coll->AABB.YMax >= coll->AABB.YMin)
		{
			std::fprintf(f, ", \"aabb\": {\"x_min\": %.4f, \"y_min\": %.4f, \"x_max\": %.4f, \"y_max\": %.4f}",
			             coll->AABB.XMin, coll->AABB.YMin, coll->AABB.XMax, coll->AABB.YMax);
		}
		else
		{
			std::fprintf(f, ", \"aabb\": null");
		}

		std::fprintf(f, ", \"scores\": [");
		for (unsigned s = 0; s < info.Control.ScoreCount; ++s)
			std::fprintf(f, "%s%d", s ? ", " : "", info.Control.Scores[s]);
		std::fprintf(f, "]}%s\n", i + 1 < kScoreComponentCount ? "," : "");
	}
	std::fprintf(f, "  ]\n}\n");
	return std::fclose(f) == 0;
}

void StateExport::OnFlipperBallHit()
{
	g_flipperHitPending = true;
}

void StateExport::OnFlipperInput(FlipperSide side, bool pressed)
{
	if (side == FlipperSide::Left)
		g_flipperLeftHeld = pressed;
	else
		g_flipperRightHeld = pressed;
}

void StateExport::ResetTrackedState()
{
	g_flipperLeftHeld = false;
	g_flipperRightHeld = false;
	g_flipperHitPending = false;
	g_prevScore = pb::MainTable ? pb::MainTable->CurScore : 0;
	g_hasPrevBall1 = false;
	g_hasPrevBall2 = false;
	ClearHits();
	g_attribution = 0;
}

void StateExport::PlaceBall(float x, float y, float vx, float vy)
{
	auto table = pb::MainTable;
	if (!table || table->BallList.empty())
		return;

	auto ball = table->BallList[0];
	if (!ball->ActiveFlag)
		return; // no live ball to place onto - ignore, per ipc_server.cpp's contract

	ball->Position.X = x;
	ball->Position.Y = y;
	// Z left untouched - it's derived from CollisionFlag/CollisionOffset elsewhere (see
	// TBall::Repaint), not part of the 2D placement contract.

	vector2 dir{vx, vy};
	// ball->Speed is already table-units/sec (see pb.cpp::timed_frame: ballStepsDistance =
	// Speed * TimeDelta, TimeDelta in seconds) - the same units Capture() reports, so no unit
	// conversion is needed here.
	ball->Speed = maths::normalize_2d(dir);
	ball->Direction.X = dir.X;
	ball->Direction.Y = dir.Y;

	// Mirror TPinballTable::AddBall / TBall::Message(Reset)'s bookkeeping for a (re)positioned
	// ball, so nothing thinks it's still mid-collision with whatever it last touched.
	ball->CollisionComp = nullptr;
	ball->CollisionFlag = 0;
	ball->CollisionDisabledFlag = false;
	ball->EdgeCollisionCount = 0;
	ball->EdgeCollisionResetFlag = false;
	ball->StuckCounter = 0;
	ball->LastActiveTime = pb::time_ticks;
}

void StateExport::NotifyBallTeleported()
{
	g_hasPrevBall1 = false;
}

bool StateExport::AnyBallActive()
{
	auto table = pb::MainTable;
	if (!table)
		return false;
	for (auto ball : table->BallList)
	{
		if (ball->ActiveFlag)
			return true;
	}
	return false;
}

pinball_ipc::StateFrame StateExport::Capture(float timeDeltaSec)
{
	pinball_ipc::StateFrame frame{};
	frame.magic = pinball_ipc::kStateMagic;

	auto table = pb::MainTable;
	if (!table || table->BallList.empty())
		return frame;

	// BallList[0] is the primary ball, tracked unconditionally as before.
	auto ball = table->BallList[0];
	frame.ball_x = ball->Position.X;
	frame.ball_y = ball->Position.Y;
	// NOT ball->PrevPosition - that field is the vanilla engine's own stuck-ball-watchdog
	// reference (see pb.cpp's timed_frame and TPinballTable::AddBall), updated only when a ball
	// is created and then only every 500+ ticks the ball spends "inactive" per the engine's own
	// bookkeeping - NOT once per frame. Using it as a per-frame previous-position produced
	// velocities that grew without bound between those rare resets (confirmed via real
	// telemetry: values up to +-2700 units/tick on a table roughly 20x28 units wide, saturating
	// the RL observation's velocity clamp 20-35% of the time) - a real bug present since this
	// channel was first added, silently corrupting the one signal the whole looming-detector
	// design (LC4/LPLC2 channels, see build_pinball_connectome.py) most depends on. Track our
	// own previous-position state instead, exactly like g_prevScore already does for
	// score_delta - reset in ResetTrackedState() so a reset's position jump isn't read as
	// velocity either.
	if (timeDeltaSec > 0.0f && g_hasPrevBall1)
	{
		frame.ball_vx = (ball->Position.X - g_prevBall1X) / timeDeltaSec;
		frame.ball_vy = (ball->Position.Y - g_prevBall1Y) / timeDeltaSec;
	}
	g_prevBall1X = ball->Position.X;
	g_prevBall1Y = ball->Position.Y;
	g_hasPrevBall1 = true;

	// Second ball, for multiball (see TPinballTable.cpp's MultiballFlag, enabled on real
	// DEMO.DAT tables). BallList can hold up to 20 slots (AddBall's cap) that get reused
	// (ActiveFlag toggled) across a game rather than freshly allocated per ball, so index 1
	// isn't necessarily "the" second ball - scan for the first OTHER active slot instead.
	bool foundBall2 = false;
	for (size_t i = 1; i < table->BallList.size(); ++i)
	{
		auto ball2 = table->BallList[i];
		if (!ball2->ActiveFlag)
			continue;
		frame.ball2_x = ball2->Position.X;
		frame.ball2_y = ball2->Position.Y;
		// Same fix as ball1 above, plus: g_hasPrevBall2 is forced false below whenever no
		// second ball was active last capture, so a newly-appeared ball's first frame reports
		// velocity 0 instead of a bogus delta against a stale/unrelated previous position.
		if (timeDeltaSec > 0.0f && g_hasPrevBall2)
		{
			frame.ball2_vx = (ball2->Position.X - g_prevBall2X) / timeDeltaSec;
			frame.ball2_vy = (ball2->Position.Y - g_prevBall2Y) / timeDeltaSec;
		}
		g_prevBall2X = ball2->Position.X;
		g_prevBall2Y = ball2->Position.Y;
		g_hasPrevBall2 = true;
		frame.ball2_active = 1;
		foundBall2 = true;
		break;
	}
	if (!foundBall2)
		g_hasPrevBall2 = false;

	frame.flipper_left = g_flipperLeftHeld ? 1 : 0;
	frame.flipper_right = g_flipperRightHeld ? 1 : 0;
	// table->BallCount is balls-remaining (lives), not "is a ball active on the table right
	// now" - it's already MaxBallCount at the start of a fresh game, before any launch. There
	// is no clean public engine flag for "ball has left the plunger lane", so use a
	// position-based heuristic instead: MessageCode::NewGame resets the ball to exactly
	// (0,0,-0.8), and real telemetry against DEMO.DAT confirms it stays there until launched,
	// then moves to genuine table coordinates (e.g. x=-7, y=10-12 during a plunger launch).
	constexpr float kPlungerRestRadius = 1.0f;
	auto distFromRest = std::sqrt(ball->Position.X * ball->Position.X + ball->Position.Y * ball->Position.Y);
	frame.ball_in_play = (pb::game_mode == GameModes::InGame && distFromRest > kPlungerRestRadius) ? 1 : 0;
	frame.tilted = table->TiltLockFlag ? 1 : 0;

	// TPinballTable::add_score() wraps CurScore by subtracting 1,000,000,000 once it exceeds
	// that (a real, intentional feature of the original 1995 game - keeps the on-screen digit
	// count bounded), not a bug. But it means a naive score-g_prevScore delta reads as a huge
	// spurious NEGATIVE spike the one tick the wrap happens - confirmed via real telemetry: a
	// held-out eval episode reported a total score of -264,500 for a game that was clearly
	// still accumulating points normally. Detect and correct for the wrap the same way the
	// engine itself defines it.
	auto score = table->CurScore;
	auto delta = score - g_prevScore;
	if (delta < -500000000)
		delta += 1000000000;
	frame.score_delta = delta;
	g_prevScore = score;

	frame.done = (pb::game_mode == GameModes::GameOver) ? 1 : 0;

	frame.flipper_hit = g_flipperHitPending ? 1 : 0;
	g_flipperHitPending = false;

	frame.hit_count = static_cast<uint8_t>(g_hitCount);
	frame.unattributed_points = g_unattributedPoints;
	for (size_t i = 0; i < g_hitCount; ++i)
	{
		if (i < static_cast<size_t>(pinball_ipc::kMaxHits))
		{
			frame.hit_ids[i] = g_hitIds[i];
			frame.hit_points[i] = g_hitPoints[i];
			frame.hit_base_points[i] = g_hitBasePoints[i];
		}
		else
		{
			frame.unattributed_points += g_hitPoints[i]; // keep the score_delta sum invariant
		}
	}
	ClearHits();

	return frame;
}
