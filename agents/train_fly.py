#!/usr/bin/env python3
"""Train a real Drosophila connectome (nfly / Janelia MaleCNS v1.0) as the RL policy for
PinballEnv, replacing the small hand-built SNN in snn_model.py / train.py.

Uses nfly's own recurrent PPO (nfly.rl.simple.ppo) rather than a hand-rolled REINFORCE loop:
the fly network is a whole-episode recurrent policy over ~166k neurons, and PPO's truncated-BPTT
rollout + GAE + value function is what nfly is built and tuned against (see ppo.py's
PPOConfig docstring) - reimplementing that on top of the old single-episode REINFORCE loop would
mean rediscovering the same rollout/replay/GAE machinery nfly already ships.

Data: three MaleCNS v1.0 files (~1.2 GB total, CC-BY 4.0) must exist under --data-dir before
this can run - see agents/download_malecns.sh.

Usage:
	python agents/train_fly.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
		--data-dir data --subset brain --n-envs 4 --updates 2000
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from nfly import FlyAgent, load_malecns, select_subset  # noqa: E402
from nfly.interface.decoders import default_readout_nodes  # noqa: E402
from nfly.rl.simple.ppo import PPOConfig, train_ppo  # noqa: E402

from agents.fly_decoder import MultiBinaryDecoder  # noqa: E402
from agents.fly_env import make_pinball_vec_env  # noqa: E402


def build_agent(data_dir: str, subset: str, obs_space, act_space, readout_dim: int, device: str):
	conn = select_subset(load_malecns(data_dir), subset)
	decoder = MultiBinaryDecoder(default_readout_nodes(conn), n_actions=act_space.n, readout_dim=readout_dim)
	agent = FlyAgent.build(conn, obs_space, act_space, decoder=decoder)
	agent = agent.to(device)
	print(agent.summary(), flush=True)
	return agent


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", required=True, help="path to the built SpaceCadetPinball binary")
	parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
	parser.add_argument("--subset", default="brain", choices=["all", "brain", "visual", "visual_small"])
	parser.add_argument("--n-envs", type=int, default=4)
	parser.add_argument("--frame-skip", type=int, default=4)
	parser.add_argument("--max-episode-steps", type=int, default=2000)
	parser.add_argument("--curriculum-episodes", type=int, default=0,
						 help="first N episodes per env: flipper bits are zeroed before reaching the engine, so only launch timing can be reinforced")
	parser.add_argument("--readout-dim", type=int, default=32)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--rollout", type=int, default=32)
	parser.add_argument("--updates", type=int, default=1000)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--minibatch-envs", type=int, default=4)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--gamma", type=float, default=0.98)
	parser.add_argument("--lam", type=float, default=0.8)
	parser.add_argument("--entropy", type=float, default=0.01)
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints", "fly_agent.pt"))
	parser.add_argument("--resume", default=None, help="checkpoint .pt path to load agent weights from before training")
	parser.add_argument("--save-every", type=int, default=20)
	parser.add_argument("--log-every", type=int, default=1)
	args = parser.parse_args()

	venv = make_pinball_vec_env(
		args.binary,
		n_envs=args.n_envs,
		frame_skip=args.frame_skip,
		curriculum_episodes=args.curriculum_episodes,
		max_episode_steps=args.max_episode_steps,
	)
	obs_space, act_space = venv.single_observation_space, venv.single_action_space

	agent = build_agent(args.data_dir, args.subset, obs_space, act_space, args.readout_dim, args.device)
	if args.resume:
		ckpt = torch.load(args.resume, map_location=args.device, weights_only=True)
		agent.load_state_dict(ckpt["agent"])
		print(f"resumed weights from {args.resume}", flush=True)

	cfg = PPOConfig(
		rollout=args.rollout, updates=args.updates, epochs=args.epochs, minibatch_envs=args.minibatch_envs,
		lr=args.lr, gamma=args.gamma, lam=args.lam, entropy=args.entropy,
		out=args.out, save_every=args.save_every, log_every=args.log_every,
	)

	try:
		train_ppo(agent, venv, cfg, device=args.device)
	finally:
		venv.close()

	print(f"training complete, final checkpoint: {args.out}", flush=True)


if __name__ == "__main__":
	main()
