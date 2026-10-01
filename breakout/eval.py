#!/usr/bin/env python3
"""Load a breakout/train.py checkpoint and replay its champion for a handful of real episodes,
reporting bricks broken / paddle hits / ball losses - mirrors agents/eval_pinball_circuit_cem.py.

Usage:
    python breakout/eval.py --checkpoint breakout/checkpoints/breakout_circuit_cem.pt --episodes 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402
from agents.train_pinball_circuit_cem import set_flat_params  # noqa: E402

from breakout.env import BreakoutEnv  # noqa: E402
from breakout.train import OBS_SCALE  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "connectome.json"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3000)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    print(f"loaded checkpoint: generation={ckpt['generation']} champion_fitness={ckpt['champion_fitness']:.2f} "
          f"readout_dim={ckpt['readout_dim']}", flush=True)

    agent = FixedCircuitAgent.from_json(args.connectome, readout_dim=ckpt["readout_dim"], obs_scale=OBS_SCALE)
    env = BreakoutEnv(seed=123)
    agent.calibrate(env.observation_space)
    set_flat_params(agent.decoder, ckpt["champion"])

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=1000 + ep)
        h = agent.initial_state(1)
        paddle_hits = 0
        brick_hits = 0
        ball_losses = 0
        steps = 0
        done = False
        with torch.no_grad():
            while not done and steps < args.max_steps:
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                action, h = agent.act(obs_t, h, greedy=True)
                obs, reward, terminated, truncated, info = env.step(action[0])
                if info["paddle_hit"]:
                    paddle_hits += 1
                if info["brick_hit"]:
                    brick_hits += 1
                if info["ball_lost"]:
                    ball_losses += 1
                steps += 1
                done = terminated or truncated
        print(f"episode {ep}: steps={steps} score={info['score']} paddle_hits={paddle_hits} "
              f"brick_hits={brick_hits} ball_losses={ball_losses} bricks_left={info['bricks_left']}", flush=True)


if __name__ == "__main__":
    main()
