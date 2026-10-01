"""Does pressing early send the ball into the outlanes?

User observation: the brain often knocks the ball into the lane beside a flipper whose gate closes
after one use (the outlanes: a_roll4/a_kick1/v_gate1 on one side, a_roll8/a_kick2/v_gate2 on the
other; the kicker saves the first ball, then the gate closes). Hypothesis: the early (lead1) press
hits the ball before it reaches the flipper's sweet spot, so the shot is too weak / badly angled.

Counts per life, for the plain reflex, the lead1 reflex and the lead-seeded popcode brain on the same
seeds: outlane entries, entries within SHOT_WINDOW steps after a flipper contact (i.e. caused by a
shot), kicker saves, and whether the life ended right after an outlane entry.

Usage: python agents/experiments/cradle/outlane_probe.py --seeds 25000:25096 --workers 8
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

DT_STEP = 0.0347
SHOT_WINDOW = 40
OUTLANE = ("a_roll4", "a_roll8")
KICKERS = ("a_kick1", "a_kick2")
BRAIN = "agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt"
CONN = "agents/experiments/big_circuit/connectome.json"
_ENV = None
_IDS: dict = {}
_AGENT = [None]


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
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		objs = json.load(f)["objects"]
	_IDS["outlane"] = frozenset(o["id"] for o in objs if o["name"] in OUTLANE)
	_IDS["kicker"] = frozenset(o["id"] for o in objs if o["name"] in KICKERS)


def _brain():
	if _AGENT[0] is None:
		import torch

		from agents import train_pinball_circuit_cem as T
		from agents.experiments.encoding.encoding import build_encoded_agent

		ck = torch.load(os.path.join(ROOT, BRAIN), weights_only=False)
		agent = build_encoded_agent(os.path.join(ROOT, CONN), int(ck["readout_dim"]), ck["encoding"])
		agent.calibrate(_ENV.observation_space)
		T.set_flat_params(agent.decoder, np.asarray(ck["champion"], dtype=np.float32))
		_AGENT[0] = agent
	return _AGENT[0]


def _life(task) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	policy, seed = task
	obs, _ = _ENV.reset(seed=seed)
	reflex = T.ReflexLabeler(11.5, 1)
	if policy == "brain":
		agent = _brain()
		h = agent.initial_state(1)
		filt = T.FlipperObsFilter("delay1")
	score = steps = entries = shot_entries = kicks = contacts = 0
	since_contact = 10 ** 9
	last_entry_step = -10 ** 9
	done = False
	info = {}
	with torch.no_grad():
		while not done:
			if policy == "brain":
				a, h = agent.act(torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0), h, greedy=True)
				act = np.asarray(a[0], dtype=np.int64)
				act[2] = 0
			else:
				view = np.array(obs, dtype=np.float32)
				if policy == "lead1" and view[OBS_BALL_VY] > 0:
					view[OBS_BALL_Y] += view[OBS_BALL_VY] * DT_STEP
				act = reflex.label(view)
				act[2] = 0
				reflex.observe(act)
			obs, _r, term, trunc, info = _ENV.step(act)
			steps += 1
			score += info["score_delta"]
			if info["flipper_hit"]:
				contacts += 1
				since_contact = 0
			else:
				since_contact += 1
			for (oid, _p, _x, _y) in info.get("hit_objects", ()):
				if oid in _IDS["outlane"]:
					entries += 1
					shot_entries += int(since_contact <= SHOT_WINDOW)
					last_entry_step = steps
				kicks += int(oid in _IDS["kicker"])
			done = term or trunc or bool(info["drained"])
	ended_by_outlane = bool(info.get("drained")) and steps - last_entry_step < 150
	return dict(policy=policy, seed=seed, score=float(score), steps=steps, contacts=contacts, entries=entries,
	            shot_entries=shot_entries, kicks=kicks, outlane_death=int(ended_by_outlane))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="25000:25096")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(HERE, "outlane_probe.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	names = ("plain", "lead1", "brain")
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(_life, [(p, s) for s in seeds for p in names], chunksize=1)
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	print(f"{'policy':<7} {'log1p':>6} {'contacts':>8} {'outlane':>8} {'after-shot':>10} {'per 100 shots':>13} {'kicks':>6} {'outlane deaths':>14}")
	for p in names:
		rs = [r for r in res if r["policy"] == p]
		c = sum(r["contacts"] for r in rs)
		print(f"{p:<7} {np.mean(np.log1p([r['score'] for r in rs])):6.2f} {np.mean([r['contacts'] for r in rs]):8.2f} "
		      f"{np.mean([r['entries'] for r in rs]):8.2f} {np.mean([r['shot_entries'] for r in rs]):10.2f} "
		      f"{100 * sum(r['shot_entries'] for r in rs) / max(1, c):13.2f} {np.mean([r['kicks'] for r in rs]):6.2f} "
		      f"{np.mean([r['outlane_death'] for r in rs]):14.0%}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
