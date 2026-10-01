"""Reverted reward-shaping redesign for agents/train_pinball_circuit_cem.py, kept here for
reference in case it's worth picking back up with a larger validation budget. NOT wired into the
training script - this file is not imported by anything, it's a snapshot of the code as it stood
before being reverted out of _evaluate() on 2026-09-25.

## Why this was proposed

An Opus-level review of the long-standing champion_fitness plateau (stuck at 913.25 for 100+
generations even at full score_weight=1.0) measured, on the live production champion:
  - True mean fitness ~423 (std 334) vs. the stored/inflated champion bar of 913.25.
  - The champion holds a flipper up 3-25% of ticks per episode, yet produces close to zero real
    flipper_hit events. Root cause: TFlipper::FlipperCollision in the real engine only registers
    a hit while the flipper is actively ROTATING (deltaAngle != 0) - a flipper already up and
    holding static has deltaAngle=0 and can never register a hit no matter how long it holds or
    how close the ball is. PROXIMITY_BONUS (the existing, still-live reward term) only rewards
    the HOLD, never the swing motion that can actually connect.
  - The existing FLIPPER_HIT_MIN_VY=1.0 hard gate rejected 100% of the real flipper_hit events
    sampled (12 fresh episodes, 5 raw hits, 0 passing the gate) - meaning FLIPPER_HIT_BONUS was
    structurally unreachable as a training signal even when real contact was happening.
  - Score, at score_weight=1.0, has std ~334 against a flipper-skill (non-score) signal averaging
    ~-0.7 - linear score scaling swamps the entire flipper-skill signal by ~2 orders of magnitude.

## Why it was reverted rather than deployed

A local 150-generation validation run (population=10, --score-curriculum-gens 40) showed the
mean real-hit rate completely FLAT across the whole run:
    gens   1-50  mean hits/episode: 0.052
    gens  51-100 mean hits/episode: 0.056
    gens 101-150 mean hits/episode: 0.051
No rising trend - the deciding signal the validation plan called for. NOT confirmed broken (this
budget - population 10 for 150 generations - is far smaller than production's population, which
was up to 128 at the time), but not confirmed working either, so it was pulled back out of
_evaluate() rather than deployed to the live Colab run on an inconclusive result. See this
project's session history (docs/colab_cli_notes.md and the training conversation) for the full
measurement details.

## What DID get deployed alongside this (kept, not reverted)

Honest paired champion re-validation (challenger vs. a freshly re-measured champion, same seeds,
instead of comparing against a stale historical max), common random numbers (the `seed` parameter
threaded through _evaluate()'s task tuple and rotated once per generation), population-level
behavior telemetry printed on the generation log line, an IPOP-restart dead-end fix, and
persisting the curriculum ramp's true origin (`curriculum_start_gen`) across resumes. All of that
is live in agents/train_pinball_circuit_cem.py as of this file's date - only the reward-shaping
changes below were pulled back out.

## To pick this back up

1. Re-add the constants and _evaluate() changes below.
2. Run a SIGNIFICANTLY larger local validation than 150gen/pop10 before considering Colab -
   consider matching production's scale more closely (population 24+, --generations 500+), or at
   minimum increase the budget until the hits telemetry trend is unambiguous either way.
3. Judge success by the `hits` telemetry column trending up, not by champion_fitness alone (which
   is noisy and, per the honest-bar fix, can legitimately fluctuate either direction).
"""

# ============================================================================
# Constants (originally placed after NOVELTY_DESCRIPTOR_DIM, before _evaluate())
# ============================================================================

# Real flipper origins (see FLIPPER_ZONE_Y's comment: TFlipperEdge ground truth) - left at
# x=-2.489, right at x=+2.489. Used below for a much tighter swing-credit window than
# PROXIMITY_BONUS's +-4.0 zone, since swing credit is meant to approximate "a real press attempt
# against a ball actually at the flipper", not merely "somewhere in the flipper's half of the
# table".
FLIPPER_ORIGIN_X_LEFT = -2.489
FLIPPER_ORIGIN_X_RIGHT = 2.489
SWING_ZONE_HALF_WIDTH = 2.0

# Credit for the PHYSICAL press EDGE (released -> pressed transition, not a sustained hold) while
# the ball is tightly near that flipper's real origin and moving toward the drain (obs_vy < 0,
# i.e. needs to be hit back rather than already receding from a prior hit). Edge-triggered by
# construction (only fires on a 0->1 transition), so a naive "toggle rapidly" exploit is already
# closed the way PROXIMITY_BONUS's toggling exploit was closed historically - but
# MIN_FLIPPER_RELEASE_TICKS=1 (vs MIN_FLIPPER_PRESS_TICKS=2) means a policy could still request
# press/release/press/release almost every tick pair and rack up several genuine edges per zone
# visit without ever timing against the ball. Capped per zone-visit (same re-arm pattern as
# PROXIMITY_BONUS's credited_left/right) for exactly that reason.
SWING_BONUS = 0.3
MAX_SWING_CREDITS_PER_VISIT = 2

# log1p compresses score's heavy-tailed luck while still rewarding genuinely higher scores: a 10x
# score gap becomes log(10) ~= 2.3 fitness (SCORE_LOG_SCALE=1.0), comparable to one real
# FLIPPER_HIT_BONUS rather than swamping it by two orders of magnitude. Applied ONCE per episode
# to the episode's total score_delta (not per-tick), since log is only well-behaved applied to a
# whole accumulated quantity.
SCORE_LOG_SCALE = 1.0


# ============================================================================
# _evaluate() body changes (relative to the current, reverted version) - shown as the
# replacements that were made, not a full function (see train_pinball_circuit_cem.py for the
# current, live _evaluate() to reconstruct the full function against)
# ============================================================================

REVERTED_EVALUATE_NOTES = r"""
1. Per-tick, instead of the linear `fitness += info["score_delta"] * SCORE_SCALE * score_weight`,
   accumulate `episode_score_delta += info["score_delta"]` and, ONCE after the episode loop ends:
       fitness += score_weight * SCORE_LOG_SCALE * float(np.log1p(max(0.0, episode_score_delta)))

2. Replace the hard-gated flipper_hit block:
       if info["flipper_hit"] and obs[OBS_BALL_VY] > FLIPPER_HIT_MIN_VY:
           fitness += FLIPPER_HIT_BONUS
           real_hits += 1
   with a graded version that still counts a HARD-gated `real_hits` for telemetry/comparability,
   but gives partial credit for any contact:
       if info["flipper_hit"]:
           graded_hit_credit = float(np.clip(obs[OBS_BALL_VY] / 10.0, 0.0, 1.0))
           fitness += FLIPPER_HIT_BONUS * graded_hit_credit
           if obs[OBS_BALL_VY] > FLIPPER_HIT_MIN_VY:
               real_hits += 1  # telemetry only

3. Track `prev_physical_left`/`prev_physical_right` (previous-tick physical flipper state) and
   per-zone-visit swing credit counters `swing_credits_left`/`swing_credits_right` (re-armed to 0
   on zone exit, same pattern as PROXIMITY_BONUS's credited_left/right), plus an episode-total
   `swing_credits` counter for telemetry. After the existing PROXIMITY_BONUS block, add:

       near_left_origin = abs(obs[OBS_BALL_X] - FLIPPER_ORIGIN_X_LEFT) < SWING_ZONE_HALF_WIDTH
       near_right_origin = abs(obs[OBS_BALL_X] - FLIPPER_ORIGIN_X_RIGHT) < SWING_ZONE_HALF_WIDTH
       approaching = obs[OBS_BALL_VY] < 0
       left_edge = physical_left and not prev_physical_left
       right_edge = physical_right and not prev_physical_right
       if in_zone_left and near_left_origin and approaching and left_edge and swing_credits_left < MAX_SWING_CREDITS_PER_VISIT:
           fitness += SWING_BONUS
           swing_credits_left += 1
           swing_credits += 1
       elif not in_zone_left:
           swing_credits_left = 0
       if in_zone_right and near_right_origin and approaching and right_edge and swing_credits_right < MAX_SWING_CREDITS_PER_VISIT:
           fitness += SWING_BONUS
           swing_credits_right += 1
           swing_credits += 1
       elif not in_zone_right:
           swing_credits_right = 0
       prev_physical_left, prev_physical_right = physical_left, physical_right

4. Add `swing_credits=swing_credits` to the returned telemetry dict.
"""
