"""Diagnostic: can a policy AIM the ball with the flippers, i.e. how much does the outgoing
direction of the ball depend on WHEN the flipper is pressed?

For each of N real-game drill starts (from the eval drill bank), replay the identical
placement many times (engine is deterministic given seed + action sequence), pressing one
side (left/right/both) for 3 steps starting at press-timing k in 0..30, releasing, and doing
nothing else for 80 steps total. Records the outgoing velocity angle and where the ball first
crosses y=0 after any resulting flipper_hit, and whether that leads to a real save (y < 3
within the run). Read-only diagnostic: does not modify any existing file.

Usage: .venv/bin/python agents/experiments/aim/aim_controllability.py
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = "/Users/seky/Developer/Git/3DPinball-DrosophilaCadet"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

BANK_PATH = os.path.join(ROOT, "agents/experiments/drill_bank/eval_bank.npz")
BINARY_PATH = os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball")

N_DRILLS = 60
N_WORKERS = 8
DRILL_SEED = 0

K_RANGE = range(0, 31)          # press-timing offsets, steps after placement
PRESS_DURATION = 3              # steps the side is held down
MAX_STEPS = 80                  # total steps simulated after placement
SAVE_WINDOW = 60                # steps after the hit within which y < 3 counts as a save
SAVED_Y = 3.0
ANGLE_DELAY = 3                 # steps after the hit at which outgoing angle is measured

PRESS_ACTIONS = {
    "left": (True, False),
    "right": (False, True),
    "both": (True, True),
}
SIDES = ("left", "right", "both")

_WORKER_ENV = None


def _worker_init():
    global _WORKER_ENV
    import atexit
    import signal

    from env_python.pinball_env import PinballEnv

    _WORKER_ENV = PinballEnv(binary_path=BINARY_PATH, headless=True, frame_skip=4)

    def _on_terminate(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_terminate)
    atexit.register(_WORKER_ENV.close)


def _replay(drill, k, side):
    """Reset + wait for lane clear + place the ball (identical every call for a given drill,
    since the engine is deterministic in seed + action sequence), then press `side` for
    PRESS_DURATION steps starting at step k, release, and step with no input up to MAX_STEPS
    total. Returns a dict describing the resulting flipper hit (if any)."""
    from agents.drill import _wait_for_lane_clear
    from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

    env = _WORKER_ENV
    env.reset(seed=drill.seed)
    obs, info = _wait_for_lane_clear(env)
    if obs is None:
        return None

    obs, info = env.unwrapped.place_ball(drill.x, drill.y, drill.vx, drill.vy)
    pl, pr = PRESS_ACTIONS[side]

    hit_step = None
    angle_deg = None
    crossed_x = None
    saved = False

    for t in range(MAX_STEPS):
        prev_x, prev_y = float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y])
        if k <= t < k + PRESS_DURATION:
            action = [pl, pr, False]
        else:
            action = [False, False, False]
        obs, _reward, terminated, _truncated, info = env.step(action)
        cur_x, cur_y = float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y])

        if hit_step is None and info["flipper_hit"] and t >= k:
            hit_step = t

        if hit_step is not None:
            if t == hit_step + ANGLE_DELAY:
                angle_deg = math.degrees(math.atan2(float(obs[OBS_BALL_VX]), -float(obs[OBS_BALL_VY])))
            if crossed_x is None and prev_y >= 0.0 and cur_y < 0.0:
                denom = prev_y - cur_y
                frac = (prev_y / denom) if denom != 0.0 else 0.0
                crossed_x = prev_x + frac * (cur_x - prev_x)
            if cur_y < SAVED_Y and (t - hit_step) <= SAVE_WINDOW:
                saved = True

        if info["drained"] or terminated:
            break

    return {
        "k": k, "side": side, "hit": hit_step is not None, "hit_step": hit_step,
        "angle_deg": angle_deg, "crossed_x": crossed_x, "saved": saved,
    }


def _process_drill(drill):
    options = []
    for k in K_RANGE:
        for side in SIDES:
            r = _replay(drill, k, side)
            if r is not None:
                options.append(r)

    if not options:
        return {"drill": drill, "skipped": True}

    saving = [o for o in options if o["saved"]]
    hitting = [o for o in options if o["hit"]]

    def spread(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, None, None
        return min(vals), max(vals), max(vals) - min(vals)

    angle_min, angle_max, angle_spread = spread([o["angle_deg"] for o in saving])
    x_min, x_max, x_spread = spread([o["crossed_x"] for o in saving])

    per_side = {}
    for side in SIDES:
        side_saving = [o for o in saving if o["side"] == side]
        a_min, a_max, a_spread = spread([o["angle_deg"] for o in side_saving])
        x_smin, x_smax, x_sspread = spread([o["crossed_x"] for o in side_saving])
        per_side[side] = {
            "n_saving": len(side_saving), "n_hitting": sum(1 for o in hitting if o["side"] == side),
            "angle_spread": a_spread, "x_spread": x_sspread,
        }

    first_hit_k = min((o["k"] for o in hitting), default=None)

    return {
        "drill": drill, "skipped": False,
        "n_saving": len(saving), "n_hitting": len(hitting), "n_options": len(options),
        "angle_min": angle_min, "angle_max": angle_max, "angle_spread": angle_spread,
        "x_min": x_min, "x_max": x_max, "x_spread": x_spread,
        "per_side": per_side,
        "first_hit_k": first_hit_k,
        "saving_options": [{"k": o["k"], "side": o["side"], "angle_deg": o["angle_deg"]} for o in saving],
    }


def _pool_worker(drill):
    return _process_drill(drill)


def _percentile(vals, p):
    return float(np.percentile(vals, p)) if vals else float("nan")


def main():
    from agents.drill import drills_from_bank

    rng = np.random.default_rng(DRILL_SEED)
    drills = drills_from_bank(BANK_PATH, N_DRILLS, rng)

    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    results = []
    try:
        pool = ctx.Pool(processes=N_WORKERS, initializer=_worker_init)
        try:
            for i, res in enumerate(pool.imap(_pool_worker, drills)):
                d = res["drill"]
                if res["skipped"]:
                    print(f"[{i+1}/{N_DRILLS}] seed={d.seed} SKIPPED (lane never cleared)", flush=True)
                else:
                    print(
                        f"[{i+1}/{N_DRILLS}] seed={d.seed} n_saving={res['n_saving']:3d}/{res['n_options']} "
                        f"angle_spread={res['angle_spread']} x_spread={res['x_spread']} "
                        f"first_hit_k={res['first_hit_k']}",
                        flush=True,
                    )
                results.append(res)
        finally:
            pool.terminate()
            pool.join()
    except Exception:
        raise
    runtime_s = time.monotonic() - t0

    scored = [r for r in results if not r["skipped"]]
    n_skipped = len(results) - len(scored)

    zero_save = [r for r in scored if r["n_saving"] == 0]
    with_save = [r for r in scored if r["n_saving"] >= 1]
    multi_save = [r for r in scored if r["n_saving"] >= 2]

    angle_spreads = [r["angle_spread"] for r in multi_save if r["angle_spread"] is not None]
    x_spreads = [r["x_spread"] for r in multi_save if r["x_spread"] is not None]

    frac_gt20 = sum(1 for s in angle_spreads if s > 20.0) / len(angle_spreads) if angle_spreads else float("nan")
    frac_gt40 = sum(1 for s in angle_spreads if s > 40.0) / len(angle_spreads) if angle_spreads else float("nan")

    per_side_agg = {}
    for side in SIDES:
        side_spreads = [r["per_side"][side]["angle_spread"] for r in scored
                         if r["per_side"][side]["angle_spread"] is not None]
        n_saving_side = sum(r["per_side"][side]["n_saving"] for r in scored)
        n_hitting_side = sum(r["per_side"][side]["n_hitting"] for r in scored)
        per_side_agg[side] = {
            "n_saving_total": n_saving_side, "n_hitting_total": n_hitting_side,
            "median_angle_spread": float(np.median(side_spreads)) if side_spreads else float("nan"),
        }

    # angle vs timing offset relative to each drill's first-possible-hit k, pooled over saving
    # options across all drills.
    offset_angles: dict[int, list[float]] = {}
    for r in scored:
        if r["first_hit_k"] is None:
            continue
        for opt in r["saving_options"]:
            if opt["angle_deg"] is None:
                continue
            off = opt["k"] - r["first_hit_k"]
            offset_angles.setdefault(off, []).append(opt["angle_deg"])

    print("\n" + "=" * 70)
    print(f"drills: n={N_DRILLS} scored={len(scored)} skipped={n_skipped}  runtime={runtime_s:.1f}s")
    print(f"drills with 0 saving (timing,side) options: {len(zero_save)}/{len(scored)}")
    print(f"drills with >=1 saving option: {len(with_save)}/{len(scored)}")
    print(f"drills with >=2 saving options (spread defined): {len(multi_save)}/{len(scored)}")

    if angle_spreads:
        print(
            f"\nangle spread across saving options (deg), per drill (n={len(angle_spreads)}):\n"
            f"  median={np.median(angle_spreads):.1f}  "
            f"q1={_percentile(angle_spreads,25):.1f}  q3={_percentile(angle_spreads,75):.1f}  "
            f"min={min(angle_spreads):.1f}  max={max(angle_spreads):.1f}"
        )
        print(f"  fraction with spread > 20 deg: {frac_gt20:.3f}")
        print(f"  fraction with spread > 40 deg: {frac_gt40:.3f}")
    else:
        print("\nno drills had >=2 saving options; no angle-spread distribution available")

    if x_spreads:
        print(
            f"\ncrossing-x spread across saving options (table units), per drill (n={len(x_spreads)}):\n"
            f"  median={np.median(x_spreads):.2f}  "
            f"q1={_percentile(x_spreads,25):.2f}  q3={_percentile(x_spreads,75):.2f}  "
            f"min={min(x_spreads):.2f}  max={max(x_spreads):.2f}"
        )

    print("\nper-side aggregates:")
    for side in SIDES:
        a = per_side_agg[side]
        print(
            f"  {side:5s}: hits={a['n_hitting_total']:4d} saves={a['n_saving_total']:4d} "
            f"median_angle_spread={a['median_angle_spread']:.1f} deg"
        )

    print("\nangle (deg) vs press timing relative to first-possible-hit k, pooled over saving options:")
    print(f"  {'offset':>7s} {'n':>5s} {'mean_angle':>11s} {'std':>7s}")
    for off in sorted(offset_angles):
        vals = offset_angles[off]
        print(f"  {off:7d} {len(vals):5d} {np.mean(vals):11.1f} {np.std(vals):7.1f}")

    never_saved_but_hit = [r for r in scored if r["n_hitting"] > 0 and r["n_saving"] == 0]
    if never_saved_but_hit:
        print(
            f"\n{len(never_saved_but_hit)} drills had flipper_hit events but ZERO saving "
            f"(timing,side) options among the {len(K_RANGE)*len(SIDES)} tried."
        )


if __name__ == "__main__":
    main()
