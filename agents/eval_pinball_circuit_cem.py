#!/usr/bin/env python3
"""Load a CEM checkpoint (see train_pinball_circuit_cem.py) and replay its champion decoder
against the real engine for a handful of episodes, reporting real outcomes (score, flipper
hits, launches, tilts, survival ticks) - not the training-time fitness number, which mixes in
shaping bonuses not present here.

Usage:
    python agents/eval_pinball_circuit_cem.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
        --checkpoint agents/checkpoints/pinball_circuit_cem_colab.pt --episodes 3 --headless
"""
from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402
from agents.train_pinball_circuit_cem import set_flat_params  # noqa: E402
from env_python.pinball_env import PinballEnv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "..", "web", "connectome.json"))
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--headless", action="store_true", help="omit to open a real SDL window and watch it play")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    print(f"loaded checkpoint: generation={ckpt['generation']} champion_fitness={ckpt['champion_fitness']:.2f} "
          f"readout_dim={ckpt['readout_dim']}", flush=True)

    agent = FixedCircuitAgent.from_json(args.connectome, readout_dim=ckpt["readout_dim"])
    env = gym.wrappers.TimeLimit(
        PinballEnv(binary_path=args.binary, headless=args.headless, frame_skip=args.frame_skip),
        max_episode_steps=args.max_steps,
    )
    agent.calibrate(env.observation_space)
    set_flat_params(agent.decoder, ckpt["champion"])

    try:
        for ep in range(1, args.episodes + 1):
            obs, info = env.reset()
            h = agent.initial_state(1)
            total_score = 0.0
            flipper_hits = 0
            launches = 0
            tilted_ticks = 0
            was_in_play = bool(info["ball_in_play"])
            ticks = 0
            done = False
            with torch.no_grad():
                while not done:
                    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                    action, h = agent.act(obs_t, h, greedy=True)
                    obs, reward, terminated, truncated, info = env.step(action[0])
                    ticks += 1
                    total_score += info["score_delta"]
                    if info["flipper_hit"]:
                        flipper_hits += 1
                    if info["ball_in_play"] and not was_in_play:
                        launches += 1
                    was_in_play = info["ball_in_play"]
                    if info["tilted"]:
                        tilted_ticks += 1
                    done = terminated or truncated
            print(f"episode {ep}: ticks={ticks} score={total_score:.0f} flipper_hits={flipper_hits} "
                  f"launches={launches} tilted_ticks={tilted_ticks}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
