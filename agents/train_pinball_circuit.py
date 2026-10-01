#!/usr/bin/env python3
"""Train the small fixed-MaleCNS-circuit agent (agents/fixed_circuit_agent.py) against
PinballEnv, using nfly's recurrent PPO (nfly.rl.simple.ppo) - the same trainer train_fly.py
uses for the full connectome, unmodified, since FixedCircuitAgent duck-types FlyAgent's
interface. This is the model meant to actually reach the browser (see
agents/build_pinball_connectome.py's docstring for why the full connectome can't).

Prerequisite: run agents/build_pinball_connectome.py once to produce web/connectome.json.

Usage:
	python agents/train_pinball_circuit.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
		--connectome web/connectome.json --n-envs 4 --updates 500
"""
from __future__ import annotations

import argparse
import os
import signal
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from nfly.rl.simple.ppo import PPOConfig, train_ppo  # noqa: E402

from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402
from agents.fly_env import make_pinball_vec_env  # noqa: E402
from agents.train_pinball_circuit_cem import get_flat_params, set_flat_params  # noqa: E402


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--binary", required=True)
	parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "..", "web", "connectome.json"))
	parser.add_argument("--n-envs", type=int, default=4)
	parser.add_argument("--frame-skip", type=int, default=4)
	parser.add_argument("--max-episode-steps", type=int, default=2000)
	parser.add_argument("--curriculum-episodes", type=int, default=0)
	parser.add_argument("--flipper-focus", action="store_true", help="replace PinballEnv's own dense shaped reward (score_delta-dominated) with the same flipper-focus formula agents/train_pinball_circuit_cem.py's CEM/CMA-ES track uses (see agents/fly_env.py's FlipperFocusRewardWrapper) - use this; a run trained on the unmodified reward got a real, sustained training-return improvement yet scored ZERO real flipper_hit events over 10 held-out episodes, having learned to farm passive bumper-bounce score instead")
	parser.add_argument("--score-life", action="store_true", help="replace the reward with the corrected one-ball-life objective (log1p(score) per step, episode ends on info['drained']) - same objective the CMA-ES track now trains against; see agents/fly_env.py's ScorePerLifeRewardWrapper. Mutually exclusive with --flipper-focus")
	parser.add_argument("--no-normalize-obs", dest="normalize_obs", action="store_false", help="skip gym.wrappers.NormalizeObservation - FixedCircuitAgent already scales observations itself via OBS_SCALE, and the CMA track feeds raw observations, so use this when comparing/evaluating PPO against a CMA champion")
	parser.add_argument("--no-clip-reward", dest="clip_reward", action="store_false", help="disable nfly's collect() replacing each reward with np.sign(r) (vendor/nfly/nfly/rl/simple/common.py:55) - that clipping erases score magnitude, so pass this with --score-life")
	parser.add_argument("--init-head-scale", default=None, help="comma-separated per-action multipliers for the warm-started head's logits (e.g. 0.0022,0.019,1), used with --init-from-cma to restore exploration")
	parser.add_argument("--init-from-cma", default=None, help="warm-start the decoder from a CMA-format checkpoint's flat champion vector (agents/train_pinball_circuit_cem.py's format). Mutually exclusive with --resume")
	parser.add_argument("--readout-dim", type=int, default=16)
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--rollout", type=int, default=64)
	parser.add_argument("--updates", type=int, default=2000)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--minibatch-envs", type=int, default=4)
	parser.add_argument("--lr", type=float, default=3e-3)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--lam", type=float, default=0.9)
	parser.add_argument("--entropy", type=float, default=0.01)
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "checkpoints", "pinball_circuit.pt"))
	parser.add_argument("--resume", default=None)
	parser.add_argument("--save-every", type=int, default=20)
	parser.add_argument("--log-every", type=int, default=1)
	args = parser.parse_args()
	if args.init_from_cma and args.resume:
		parser.error("--init-from-cma and --resume are mutually exclusive")

	# SIGTERM's default action skips `finally` blocks (and therefore venv.close()), orphaning
	# every sub-env's native engine subprocess - confirmed via a real kill leaving 4 processes
	# spinning at ~90% CPU each for 10+ minutes unnoticed. Routing SIGTERM through sys.exit()
	# makes the `finally` below actually run - same fix as train_pinball_circuit_cem.py's
	# _worker_init's identical guard.
	signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))

	venv = make_pinball_vec_env(
		args.binary,
		n_envs=args.n_envs,
		frame_skip=args.frame_skip,
		curriculum_episodes=args.curriculum_episodes,
		max_episode_steps=args.max_episode_steps,
		flipper_focus=args.flipper_focus,
		score_life=args.score_life,
		normalize_obs=args.normalize_obs,
	)
	obs_space = venv.single_observation_space

	agent = FixedCircuitAgent.from_json(args.connectome, readout_dim=args.readout_dim).to(args.device)
	agent.calibrate(obs_space)
	print(agent.summary(), flush=True)

	if args.resume:
		ckpt = torch.load(args.resume, map_location=args.device, weights_only=True)
		agent.load_state_dict(ckpt["agent"])
		print(f"resumed weights from {args.resume}", flush=True)

	if args.init_from_cma:
		ckpt = torch.load(args.init_from_cma, weights_only=False)
		assert ckpt["readout_dim"] == args.readout_dim, (
			f"checkpoint readout_dim {ckpt['readout_dim']} != --readout-dim {args.readout_dim}")
		assert ckpt.get("agent", "circuit") == "circuit", f"checkpoint agent kind {ckpt.get('agent')!r} != 'circuit'"
		set_flat_params(agent.decoder, ckpt["champion"])
		print(f"warm-started decoder from {args.init_from_cma} (generation {ckpt.get('generation')})", flush=True)
		if args.init_head_scale:
			# CMA champions have |logits| in the hundreds (greedy-only training), which saturates the
			# Bernoulli policy so PPO cannot explore; scaling keeps every logit's sign (same greedy policy).
			scale = torch.tensor([float(v) for v in args.init_head_scale.split(",")])
			with torch.no_grad():
				agent.decoder.head.weight.mul_(scale[:, None])
				agent.decoder.head.bias.mul_(scale)
			print(f"scaled warm-started head logits by {scale.tolist()}", flush=True)

	cfg = PPOConfig(
		rollout=args.rollout, updates=args.updates, epochs=args.epochs, minibatch_envs=args.minibatch_envs,
		lr=args.lr, gamma=args.gamma, lam=args.lam, entropy=args.entropy,
		out=args.out, save_every=args.save_every, log_every=args.log_every,
		clip_reward=args.clip_reward,
	)

	try:
		train_ppo(agent, venv, cfg, device=args.device)
	finally:
		venv.close()

	print(f"training complete, final checkpoint: {args.out}", flush=True)


if __name__ == "__main__":
	main()
