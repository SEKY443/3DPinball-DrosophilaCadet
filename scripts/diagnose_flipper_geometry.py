#!/usr/bin/env python3
"""Empirically maps real flipper-hit geometry, instead of guessing a zone threshold from one
eyeballed rollout (which is what FLIPPER_ZONE_Y in env_python/pinball_env.py currently is).

Runs a wide-net scripted policy (rapidly cycling both flippers on any tick the ball is anywhere
in the lower half of its observed range) across several long episodes, and records the exact
(ball_x, ball_y, ball_vx, ball_vy) of every real StateFrame.flipper_hit event. If real hits
show up, this tells us the TRUE collision region precisely, to replace the current guessed
FLIPPER_ZONE_Y=-6.0 / ball_x-sign heuristic in agents/train_pinball_circuit_cem.py with
something derived from evidence.

If this finds ZERO hits even after a long run, that is itself the important result: it would
mean the wide net still isn't wide enough, or ball_y never actually reaches flipper range within
a normal episode length, and max_episode_steps needs to grow rather than the zone/heuristic.

Usage:
    python diagnose_flipper_geometry.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
        --episodes 8 --max-steps 6000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")

from env_python.pinball_env import PinballEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=6000)
    args = parser.parse_args()

    env = PinballEnv(binary_path=args.binary, headless=True, frame_skip=1)

    hit_obs = []
    all_y_seen = []
    low_y_obs = []  # (x, y, flip_left_held, flip_right_held) whenever y > LOW_Y_THRESHOLD
    # Real flipper YMin/YMax, read directly from TFlipperEdge at construction time via a one-off
    # debug print (since reverted): YMin=10.18, YMax=13.95 - i.e. HIGH positive y, not negative.
    # The original -6.0 (with a "< threshold" check) was checking the opposite end of the table.
    LOW_Y_THRESHOLD = 9.0
    for ep in range(args.episodes):
        obs, info = env.reset()
        toggle = 0
        for t in range(args.max_steps):
            toggle = 1 - toggle
            # Relaunch on every tick the ball isn't in play, not just once at t=0 - otherwise
            # the first mid-episode drain permanently parks the ball at rest (0, 0, -0.8) (see
            # state_export.cpp) for the remainder of max_steps, wasting the episode on a single
            # attempt instead of the many relaunch-drain cycles a long episode should allow.
            action = np.array([toggle, 1 - toggle, 0 if info["ball_in_play"] else 1])
            obs, reward, terminated, truncated, info = env.step(action)
            all_y_seen.append(float(obs[1]))
            if obs[1] > LOW_Y_THRESHOLD:
                low_y_obs.append((float(obs[0]), float(obs[1]), float(obs[4]), float(obs[5])))
            if info["flipper_hit"]:
                hit_obs.append(tuple(obs.tolist()))
                print(f"  HIT at ep={ep} t={t}: x={obs[0]:.2f} y={obs[1]:.2f} vx={obs[2]:.2f} vy={obs[3]:.2f}", flush=True)
            if terminated or truncated:
                break
        print(f"episode {ep}: {t + 1} ticks, cumulative hits: {len(hit_obs)}, ticks with y>{LOW_Y_THRESHOLD}: {len(low_y_obs)}", flush=True)

    print()
    print(f"observed ball_y range across all episodes: [{min(all_y_seen):.2f}, {max(all_y_seen):.2f}]")
    if low_y_obs:
        low_arr = np.array(low_y_obs)
        print(f"ticks with y > {LOW_Y_THRESHOLD}: {len(low_y_obs)}")
        print(f"  ball_x range during those ticks: [{low_arr[:,0].min():.2f}, {low_arr[:,0].max():.2f}]  mean {low_arr[:,0].mean():.2f}")
        print(f"  flipper_left physically held rate:  {low_arr[:,2].mean():.3f}")
        print(f"  flipper_right physically held rate: {low_arr[:,3].mean():.3f}")
    else:
        print(f"ball_y NEVER rose above {LOW_Y_THRESHOLD} in any episode - the flipper zone is out of the ball's actual reachable range in this rollout.")
    if hit_obs:
        arr = np.array(hit_obs)
        print(f"=== {len(hit_obs)} REAL HITS ===")
        print(f"ball_x  range: [{arr[:,0].min():.2f}, {arr[:,0].max():.2f}]  mean {arr[:,0].mean():.2f}")
        print(f"ball_y  range: [{arr[:,1].min():.2f}, {arr[:,1].max():.2f}]  mean {arr[:,1].mean():.2f}")
        print(f"ball_vx range: [{arr[:,2].min():.2f}, {arr[:,2].max():.2f}]")
        print(f"ball_vy range: [{arr[:,3].min():.2f}, {arr[:,3].max():.2f}]")
        print(f"flipper_left held at hit rate:  {arr[:,4].mean():.2f}")
        print(f"flipper_right held at hit rate: {arr[:,5].mean():.2f}")
        print()
        print("=> update FLIPPER_ZONE_Y and the ball_x-side heuristic in "
              "agents/train_pinball_circuit_cem.py's _heuristic_action and heuristic_seed() to "
              "match these real ranges.")
    else:
        print("=== NO HITS FOUND ===")
        print("Even a wide-net toggling policy across the full observed y-range found nothing.")
        print("Next things to check: does ball_y ever get much lower than what was seen here "
              "within a longer episode (increase --max-steps further)? Is the ball_x range "
              "during low-y ticks actually within flipper reach, or off to one side?")

    env.close()


if __name__ == "__main__":
    main()
