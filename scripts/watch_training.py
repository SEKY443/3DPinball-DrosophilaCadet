#!/usr/bin/env python3
"""Live visual preview of the currently-training policy, in a real visible SDL2 window.

Runs as a SEPARATE process from the actual (headless, uncapped-speed) training run in
agents/train.py - it does not slow training down or interfere with it. It just periodically
checks agents/checkpoints/ for a newer checkpoint and, when one appears, starts playing with
that model instead, so what's on screen tracks training progress over time.

Actions are sampled from the policy's own Bernoulli(rate) distribution (matching how
agents/train.py actually behaves during training), not thresholded - this is a diagnostic
preview of real current behavior, not a polished demo.

Usage:
    python scripts/watch_training.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.snn_model import DrosophilaController  # noqa: E402
from env_python.pinball_env import PinballEnv  # noqa: E402


def latest_checkpoint(checkpoint_dir: str) -> str | None:
	paths = glob.glob(os.path.join(checkpoint_dir, "drosophila_controller_ep*.pt"))
	if not paths:
		return None

	def ep_num(p: str) -> int:
		m = re.search(r"ep(\d+)\.pt$", p)
		return int(m.group(1)) if m else -1

	return max(paths, key=ep_num)


def load_policy(path: str) -> DrosophilaController:
	ckpt = torch.load(path, map_location="cpu", weights_only=True)
	policy = DrosophilaController(hidden_dim=ckpt["hidden_dim"], sim_steps=ckpt["sim_steps"])
	policy.load_state_dict(ckpt["model_state_dict"])
	policy.eval()
	return policy


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", required=True)
	parser.add_argument(
		"--checkpoint-dir",
		default=os.path.join(os.path.dirname(__file__), "..", "agents", "checkpoints"),
	)
	parser.add_argument("--frame-skip", type=int, default=4, help="should match the running training run's --frame-skip")
	parser.add_argument("--refresh-every", type=int, default=50, help="re-check for a newer checkpoint every N steps")
	parser.add_argument("--max-episode-steps", type=int, default=2000, help="force a reset after this many steps even if the ball never launches")
	args = parser.parse_args()

	print("[preview] connecting to engine...", flush=True)
	env = PinballEnv(binary_path=args.binary, headless=False, frame_skip=args.frame_skip)
	policy: DrosophilaController | None = None
	loaded_path: str | None = None

	obs, _info = env.reset()
	print("[preview] connected, waiting for first checkpoint...", flush=True)
	step = 0
	episode_step = 0
	episode_reward = 0.0
	episode = 0
	try:
		while True:
			if step % args.refresh_every == 0:
				newest = latest_checkpoint(args.checkpoint_dir)
				if newest and newest != loaded_path:
					try:
						policy = load_policy(newest)
						loaded_path = newest
						print(f"[preview] now playing: {os.path.basename(newest)}", flush=True)
					except Exception as exc:
						print(f"[preview] failed to load {newest}: {exc}", flush=True)

			if policy is None:
				action = [0, 0, 0]
			else:
				with torch.no_grad():
					obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
					rate = policy(obs_t).squeeze(0).clamp(1e-4, 1.0 - 1e-4)
				dist = torch.distributions.Bernoulli(probs=rate)
				action = dist.sample().int().tolist()

			obs, reward, terminated, truncated, info = env.step(action)
			episode_reward += reward
			step += 1
			episode_step += 1

			if terminated or episode_step >= args.max_episode_steps:
				episode += 1
				print(f"[preview] episode {episode} ended after {episode_step} steps, reward={episode_reward:.1f}", flush=True)
				obs, _info = env.reset()
				episode_reward = 0.0
				episode_step = 0
	finally:
		env.close()


if __name__ == "__main__":
	main()
