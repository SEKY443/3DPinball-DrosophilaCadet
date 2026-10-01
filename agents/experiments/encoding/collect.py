#!/usr/bin/env python3
"""Record correct-side pulse-reflex rollouts (ReflexLabeler y>11.5, hold 1) on tuning lives so every
encoding can be probed OFFLINE on identical observations (the reflex drives the engine; the circuit
never affects the trajectory, so running it afterwards on the stored obs is exact).

Saves obs (raw 15-d), labels (L/R), life seed, step index, reflex cooldown state and flipper_hit
to an .npz.  Usage: python agents/experiments/encoding/collect.py --seeds 7000:7024 --out ...
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="7000:7024")
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/encoding/reflex_rollouts_7000.npz"))
	args = ap.parse_args()
	import gymnasium as gym
	import torch

	torch.set_num_threads(1)
	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import PinballEnv

	lo, hi = (int(v) for v in args.seeds.split(":"))
	env = gym.wrappers.TimeLimit(PinballEnv(binary_path=args.binary, headless=True, frame_skip=4), max_episode_steps=3000)
	OBS, LAB, LIFE, STEP, REST, HIT, SCORE = [], [], [], [], [], [], {}
	try:
		for seed in range(lo, hi):
			obs, _ = env.reset(seed=seed)
			lab = T.ReflexLabeler(11.5, 1)
			done, t, score = False, 0, 0
			while not done:
				rest = list(lab._rest)
				a = lab.label(obs)
				a[2] = 0
				lab.observe(a)
				OBS.append(np.asarray(obs, dtype=np.float32)); LAB.append(a[:2].copy()); LIFE.append(seed); STEP.append(t)
				REST.append(rest)
				obs, _r, term, trunc, info = env.step(a)
				HIT.append(int(info["flipper_hit"]))
				score += info["score_delta"]; t += 1
				done = term or trunc or info["drained"]
			SCORE[seed] = score
	finally:
		env.close()
	np.savez_compressed(args.out, obs=np.array(OBS), lab=np.array(LAB), life=np.array(LIFE), step=np.array(STEP),
	                    rest=np.array(REST), hit=np.array(HIT), seeds=np.array(sorted(SCORE)),
	                    scores=np.array([SCORE[s] for s in sorted(SCORE)]))
	print(f"{len(LAB)} steps, press rate {np.array(LAB).mean(0)}, mean log1p {np.mean(np.log1p(list(SCORE.values()))):.2f}")


if __name__ == "__main__":
	main()
