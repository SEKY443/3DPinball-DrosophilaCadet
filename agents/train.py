#!/usr/bin/env python3
"""Minimal REINFORCE training loop for DrosophilaController against PinballEnv (no SB3, per
the project's chosen RL-framework option). Each of the 3 action bits is treated as an
independent Bernoulli variable parameterized by the SNN's mean output firing rate; the loss is
the standard policy-gradient estimator with a whole-episode-return baseline.

Usage:
    python agents/train.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball --episodes 500
    python agents/train.py --dummy --episodes 20   # plumbing check, no real engine/DAT needed
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.snn_model import DrosophilaController  # noqa: E402


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
	returns = [0.0] * len(rewards)
	running = 0.0
	for t in reversed(range(len(rewards))):
		running = rewards[t] + gamma * running
		returns[t] = running
	return returns


def run_episode(env, policy: DrosophilaController, device: torch.device, active_dims: Optional[list[int]] = None):
	"""active_dims restricts which action bits the policy actually controls this episode -
	used for curriculum stage 1 (launch-only): the other bits are forced to 0 before being
	sent to the env, and excluded from log_prob/entropy so no gradient flows to them from
	actions that had no effect. None means all bits are active (normal full-control training).

	Independently of active_dims, the launch bit's log_prob/entropy is ALSO excluded on any
	tick where info["ball_in_play"] or info["relaunch_pending"] is true - a launch=1 there is
	a guaranteed engine-side no-op (see ipc_protocol.h's relaunch_pending field), since only
	the first press after a drain actually does anything; crediting every redundant repeat
	during the ~1s feed delay dilutes the gradient across dozens of meaningless identical
	decisions instead of concentrating it on the one press that mattered. This does NOT affect
	what's actually sent to the env - holding launch=1 through the wait is still harmless."""
	from env_python.pinball_env import ACT_LAUNCH

	obs, info = env.reset()
	log_probs = []
	entropies = []
	rewards = []
	done = False

	if active_dims is not None:
		env_mask = torch.zeros(3)
		env_mask[active_dims] = 1.0
	else:
		env_mask = torch.ones(3)

	while not done:
		obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
		rate = policy(obs_t).squeeze(0).clamp(1e-4, 1.0 - 1e-4)
		dist = torch.distributions.Bernoulli(probs=rate)
		action = dist.sample()
		env_action = (action * env_mask).detach().cpu().numpy().astype(int)

		grad_mask = env_mask.clone()
		if info.get("ball_in_play") or info.get("relaunch_pending"):
			grad_mask[ACT_LAUNCH] = 0.0
		log_prob = (dist.log_prob(action) * grad_mask).sum()
		entropy_term = (dist.entropy() * grad_mask).sum()

		obs, reward, terminated, truncated, info = env.step(env_action)
		log_probs.append(log_prob)
		entropies.append(entropy_term)
		rewards.append(float(reward))
		done = terminated or truncated

	return log_probs, entropies, rewards


def train(
	env,
	episodes: int,
	hidden_dim: int,
	sim_steps: int,
	lr: float,
	gamma: float,
	checkpoint_dir: str,
	checkpoint_every: int,
	log_path: Optional[str],
	resume_path: Optional[str] = None,
	entropy_coef: float = 0.01,
	launch_curriculum_episodes: int = 0,
	baseline_decay: float = 0.95,
) -> DrosophilaController:
	device = torch.device("cpu")
	policy = DrosophilaController(hidden_dim=hidden_dim, sim_steps=sim_steps).to(device)
	optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

	# Running mean of each episode's average return, used as the REINFORCE baseline instead of
	# that same episode's own mean. The within-episode-mean baseline has a real failure mode:
	# an episode with identically zero reward at every tick (ball never launches, nothing ever
	# happens) has advantage = 0 - 0 = 0 for its ENTIRE length - REINFORCE gets literally zero
	# gradient signal from that failure, so once the launch probability starts to collapse
	# there is nothing pushing back (confirmed via a real run: many consecutive episodes with
	# bit-for-bit identical loss values, which is only possible if the returns really are
	# uniformly zero and advantage really is uniformly zero). A cross-episode running baseline
	# means a zero-reward episode gets a real negative advantage whenever the running average
	# is positive, which is the actual "you did worse than usual" signal REINFORCE needs.
	baseline_ema = 0.0
	start_episode = 1
	if resume_path:
		ckpt = torch.load(resume_path, map_location=device, weights_only=True)
		policy.load_state_dict(ckpt["model_state_dict"])
		if "optimizer_state_dict" in ckpt:
			optimizer.load_state_dict(ckpt["optimizer_state_dict"])
		baseline_ema = ckpt.get("baseline_ema", 0.0)
		start_episode = ckpt["episode"] + 1
		print(f"resumed from {resume_path} at episode {start_episode} (baseline_ema={baseline_ema:.3f})")

	os.makedirs(checkpoint_dir, exist_ok=True)
	log_file = None
	log_writer = None
	if log_path:
		os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
		write_header = not (resume_path and os.path.exists(log_path))
		log_file = open(log_path, "a" if resume_path else "w", newline="")
		log_writer = csv.writer(log_file)
		if write_header:
			log_writer.writerow(["episode", "total_reward", "loss"])

	try:
		for episode in range(start_episode, episodes + 1):
			# Curriculum stage 1: for the first launch_curriculum_episodes, flippers are
			# forced off and only the launch bit is trainable - the policy has nothing else
			# to learn from but "launch when the ball isn't in play", so it converges on that
			# specific button press fast, instead of it being one noisy signal among three
			# competing for gradient in a huge combined action space from episode 1. Stage 2
			# (full action space) then builds flipper skill on top of a launch habit that's
			# already reliable, and naturally reinforces relaunch-after-drain too since
			# LAUNCH_SUCCESS_BONUS fires on every ball_in_play False->True transition, not
			# just the first one each episode.
			from env_python.pinball_env import ACT_LAUNCH
			in_stage1 = episode <= launch_curriculum_episodes
			active_dims = [ACT_LAUNCH] if in_stage1 else None
			log_probs, entropies, rewards = run_episode(env, policy, device, active_dims=active_dims)

			returns = discounted_returns(rewards, gamma)
			returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
			advantage = returns_t - baseline_ema
			if advantage.numel() > 1 and advantage.std() > 1e-6:
				advantage = advantage / (advantage.std() + 1e-6)

			# Update the running baseline AFTER using it for this episode's advantage, so a
			# below-average episode is actually measured against where the average stood
			# going in, not shifted by this same episode's own outcome.
			baseline_ema = baseline_decay * baseline_ema + (1 - baseline_decay) * returns_t[0].item()

			# Entropy bonus (standard in policy-gradient methods): without it, REINFORCE can
			# prematurely collapse a Bernoulli probability all the way to ~0 or ~1 based on
			# early, noisy samples, before it's had enough tries to learn the action's true
			# value - exactly what happened to the launch probability in an earlier run (it
			# converged to 0 despite launching being worth 50k+ points when it happened).
			# Subtracting entropy from the loss rewards keeping some residual randomness,
			# which keeps rarely-sampled-but-valuable actions like "launch" reachable for
			# longer during training instead of getting locked out early.
			policy_loss = -(torch.stack(log_probs) * advantage).sum()
			entropy_bonus = torch.stack(entropies).sum()
			loss = policy_loss - entropy_coef * entropy_bonus

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_reward = sum(rewards)
			stage_tag = "stage1-launch" if in_stage1 else "stage2-full"
			print(f"episode {episode:5d}  steps {len(rewards):4d}  total_reward {total_reward:8.2f}  loss {loss.item():8.4f}  {stage_tag}")
			if log_writer:
				log_writer.writerow([episode, total_reward, loss.item()])
				log_file.flush()

			if checkpoint_every and episode % checkpoint_every == 0:
				ckpt_path = os.path.join(checkpoint_dir, f"drosophila_controller_ep{episode}.pt")
				torch.save({
					"model_state_dict": policy.state_dict(),
					"optimizer_state_dict": optimizer.state_dict(),
					"hidden_dim": hidden_dim,
					"sim_steps": sim_steps,
					"episode": episode,
					"baseline_ema": baseline_ema,
				}, ckpt_path)
				print(f"  saved checkpoint: {ckpt_path}")
	finally:
		if log_file:
			log_file.close()

	final_path = os.path.join(checkpoint_dir, "drosophila_controller_final.pt")
	torch.save({
		"model_state_dict": policy.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"hidden_dim": hidden_dim,
		"sim_steps": sim_steps,
		"episode": episodes,
		"baseline_ema": baseline_ema,
	}, final_path)
	print(f"training complete, final checkpoint: {final_path}")

	return policy


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	source = parser.add_mutually_exclusive_group(required=True)
	source.add_argument("--binary", help="path to the built SpaceCadetPinball binary (needs a DAT file next to it)")
	source.add_argument("--dummy", action="store_true", help="use the synthetic plumbing-test env instead (no engine/DAT needed)")
	parser.add_argument("--episodes", type=int, default=200)
	parser.add_argument("--max-episode-steps", type=int, default=2000, help="only applies to --binary runs")
	parser.add_argument("--frame-skip", type=int, default=1, help="repeat each action this many native ticks (only applies to --binary runs)")
	parser.add_argument("--hidden-dim", type=int, default=32)
	parser.add_argument("--sim-steps", type=int, default=8)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy bonus weight; higher keeps exploring longer before committing to a deterministic policy")
	parser.add_argument("--checkpoint-dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
	parser.add_argument("--checkpoint-every", type=int, default=50)
	parser.add_argument("--log-path", default=None, help="CSV path for the reward/loss curve (default: <checkpoint-dir>/train_log.csv)")
	parser.add_argument("--resume", default=None, help="checkpoint .pt path to resume from (continues episode numbering and optimizer state)")
	parser.add_argument("--launch-curriculum-episodes", type=int, default=0, help="for this many initial episodes, flippers are disabled and only the launch action is trained (curriculum stage 1)")
	args = parser.parse_args()

	if args.dummy:
		from agents.dummy_env import DummyPlumbingEnv
		env = DummyPlumbingEnv()
	else:
		from gymnasium.wrappers import TimeLimit
		from env_python.pinball_env import PinballEnv
		# A real game only reaches `terminated` after every ball across every player's turn is
		# lost (GameModes.GameOver) - unbounded in real time, so cap episode length for training.
		env = TimeLimit(
			PinballEnv(binary_path=args.binary, frame_skip=args.frame_skip),
			max_episode_steps=args.max_episode_steps,
		)

	log_path = args.log_path or os.path.join(args.checkpoint_dir, "train_log.csv")

	try:
		train(
			env=env,
			episodes=args.episodes,
			hidden_dim=args.hidden_dim,
			sim_steps=args.sim_steps,
			lr=args.lr,
			gamma=args.gamma,
			checkpoint_dir=args.checkpoint_dir,
			checkpoint_every=args.checkpoint_every,
			log_path=log_path,
			resume_path=args.resume,
			entropy_coef=args.entropy_coef,
			launch_curriculum_episodes=args.launch_curriculum_episodes,
		)
	finally:
		env.close()


if __name__ == "__main__":
	main()
