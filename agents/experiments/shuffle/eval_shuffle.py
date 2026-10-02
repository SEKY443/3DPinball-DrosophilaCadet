"""Paired evaluation: popcode brains trained on the real connectome vs degree-preserving shuffles.

Each brain (seeds_<name>/dagger_popcode_delay1_r8.pt from dagger_lead.py) is loaded with ITS OWN
connectome and plays one life per seed on the same fresh seeds, delay1 flipper obs. Reports log1p
of score per life and the paired differences shuffled - real.

Usage: python agents/experiments/shuffle/eval_shuffle.py --seeds 76000:76096 --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "encoding"))

BRAINS = {
	"real": "agents/experiments/big_circuit/connectome.json",
	"s1": "agents/experiments/shuffle/connectome_shuffled_1.json",
	"s2": "agents/experiments/shuffle/connectome_shuffled_2.json",
	"s3": "agents/experiments/shuffle/connectome_shuffled_3.json",
}
_ENV = None
_AGENTS: dict = {}


def _init(binary: str, max_steps: int) -> None:
	global _ENV
	import atexit
	import signal

	import gymnasium as gym
	import torch

	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=binary, headless=True, frame_skip=4), max_episode_steps=max_steps)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)


def _agent(name: str):
	import torch

	from agents import train_pinball_circuit_cem as T
	from agents.experiments.encoding.encoding import build_encoded_agent

	if name not in _AGENTS:
		ck = torch.load(os.path.join(HERE, f"seeds_{name}", "dagger_popcode_delay1_r8.pt"), weights_only=False)
		agent = build_encoded_agent(os.path.join(ROOT, BRAINS[name]), int(ck["readout_dim"]), ck["encoding"])
		agent.calibrate(_ENV.observation_space)
		T.set_flat_params(agent.decoder, np.asarray(ck["champion"], dtype=np.float32))
		_AGENTS[name] = agent
	return _AGENTS[name]


def _run(task) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

	name, seed = task
	obs, _ = _ENV.reset(seed=seed)
	agent = _agent(name)
	h = agent.initial_state(1)
	filt = T.FlipperObsFilter("delay1")
	prev_l, prev_r = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
	score = steps = contacts = presses = 0
	done = False
	with torch.no_grad():
		while not done:
			a, h = agent.act(torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0), h, greedy=True)
			act = a[0]
			obs, _r, term, trunc, info = _ENV.step(act)
			score += info["score_delta"]
			contacts += int(info["flipper_hit"])
			lu, ru = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
			presses += int(lu and not prev_l) + int(ru and not prev_r)
			prev_l, prev_r = lu, ru
			steps += 1
			done = term or trunc or bool(info["drained"])
	return dict(name=name, seed=seed, score=float(score), steps=steps, contacts=contacts, presses=presses)


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="76000:76096")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(HERE, "eval_shuffle.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(_run, [(n, s) for s in seeds for n in BRAINS], chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["name"] == n} for n in BRAINS}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	for n in BRAINS:
		rs = [by[n][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		line = (f"{n:<5} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):8.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):.1f}  contacts {np.mean([r['contacts'] for r in rs]):.1f}")
		if n != "real":
			d = np.array([np.log1p(by[n][s]["score"]) - np.log1p(by["real"][s]["score"]) for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			line += f" | {n} - real {d.mean():+.3f} +- {se:.3f} t {d.mean() / se:+.2f} p {confirm.wilcoxon_p(d):.2g}"
		print(line)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
