#!/usr/bin/env python3
"""DAgger imitation of the correct-side pulse reflex by the fixed-circuit decoder (step 3).

Plain behaviour cloning (seed_eval.py) fits the reflex's own trajectories, but the seeded agent
then visits states the reflex never did (distribution shift) and loses the ball. DAgger: round 0
rolls out the reflex; each later round rolls out the CURRENT agent on the tuning lives, labels every
visited state with a shadow reflex (fed the same observation and the agent's own actions), adds
those samples, and refits proj + head (pos-weighted BCE, then head biases rate-calibrated).

One worker process per variant (Y:HOLD:FLIPPER_OBS:READOUT_DIM); run two variants to use 2 cores.
Writes a resumable start checkpoint per variant under --ckpt-dir; score them on held-out lives
with seed_eval.py --eval-ckpts.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))


def _variant(job: dict) -> str:
	import gymnasium as gym
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	torch.manual_seed(0)
	y, hold, fobs, rdim = job["variant"]
	env = gym.wrappers.TimeLimit(PinballEnv(binary_path=job["binary"], headless=True, frame_skip=4), max_episode_steps=3000)
	agent = T.build_agent("circuit", job["connectome"], rdim)
	agent.calibrate(env.observation_space)
	dec = agent.decoder
	X, L = [], []
	log = []
	try:
		for rnd in range(job["rounds"] + 1):
			scores = []
			for seed in job["seeds"]:
				obs, info = env.reset(seed=seed)
				h = agent.initial_state(1)
				filt = T.FlipperObsFilter(fobs.split("@")[0], loom_tau0=float(fobs.split("@")[1]) if "@" in fobs else 2.0)
				lab = T.ReflexLabeler(y, hold)
				score, done = 0, False
				while not done:
					label = lab.label(obs)
					label[2] = 0
					with torch.no_grad():
						o = torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0)
						if rnd == 0:
							h = agent.brain.step(h, agent._channel_drive(o))
							act = label
						else:
							a, h = agent.act(o, h, greedy=True)
							act = np.asarray(a[0], dtype=np.int64)
							act[2] = 0
						X.append(dec.norm(h[:, dec.idx])[0].clone())
					L.append(label[:2].copy())
					lab.observe(act)
					obs, _r, term, trunc, info = env.step(act)
					score += info["score_delta"]
					done = term or trunc or info["drained"]
				scores.append(score)
			log.append(f"round {rnd}: {'reflex' if rnd == 0 else 'agent'} rollouts mean log1p {np.mean(np.log1p(scores)):.2f}, "
			           f"dataset {len(L)}")
			# refit on everything collected so far
			feats = torch.stack(X)
			labels = torch.as_tensor(np.array(L), dtype=torch.float32)
			pos = labels.sum(0)
			pw = ((labels.shape[0] - pos) / pos.clamp_min(1)).clamp(max=200.0)
			opt = torch.optim.Adam(list(dec.proj.parameters()) + list(dec.head.parameters()), lr=1e-2)
			for _ in range(job["epochs"]):
				logits = dec.dist_inputs(dec.proj(feats))[:, :2]
				loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
				opt.zero_grad()
				loss.backward()
				opt.step()
			with torch.no_grad():
				logits = dec.dist_inputs(dec.proj(feats))
				for k in range(2):
					rate = float(labels[:, k].mean())
					dec.head.bias[k] -= torch.quantile(logits[:, k], 1.0 - rate)
				dec.head.bias[2] = -10.0  # launch is automatic in PinballEnv; keep the bit off
	finally:
		env.close()
	w = T.get_flat_params(dec)
	name = f"dagger_y{y}_h{hold}_{fobs.replace('@', '_tau')}_r{rdim}"
	torch.save({"mean": w, "sigma": np.full(w.size, 0.2, dtype=np.float32), "champion": w, "champion_fitness": -1e18,
	            "generation": 0, "curriculum_start_gen": 1, "readout_dim": rdim, "agent": "circuit", "objective": "life",
	            "drill_bank": None, "press_cost": 0.0, "flipper_obs": fobs.split("@")[0],
	            "loom_tau0": float(fobs.split("@")[1]) if "@" in fobs else 2.0,
	            "seed_policy": f"DAgger({job['rounds']}) of reflex y>{y} hold {hold}", "seed_lives": job["seeds"][:1] + job["seeds"][-1:]},
	           os.path.join(job["ckpt_dir"], name + ".pt"))
	return name + "\n  " + "\n  ".join(log)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--connectome", default=os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"))
	ap.add_argument("--variants", default="11.5:1:delay1:8,11.5:1:delay1:32")
	ap.add_argument("--seeds", default="7000:7024")
	ap.add_argument("--rounds", type=int, default=3)
	ap.add_argument("--epochs", type=int, default=600)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "agents/experiments/step3/seeds"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	os.makedirs(args.ckpt_dir, exist_ok=True)
	jobs = []
	for v in args.variants.split(","):
		y, hold, fobs, rdim = v.split(":")
		jobs.append(dict(variant=(float(y), int(hold), fobs, int(rdim)), binary=args.binary, connectome=args.connectome,
		                 seeds=list(range(lo, hi)), rounds=args.rounds, epochs=args.epochs, ckpt_dir=args.ckpt_dir))
	with mp.get_context("spawn").Pool(args.workers) as pool:
		for msg in pool.map(_variant, jobs, chunksize=1):
			print(msg, flush=True)


if __name__ == "__main__":
	main()
