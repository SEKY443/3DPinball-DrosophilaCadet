"""Stage-C diagnostic: where do points come from in a FULL game (all balls until game over)?

Plays complete games with the lead-seeded popcode brain and with the lead1 reflex teacher. The brain's
circuit state and flipper filter are reset at every drain (as in training and the web demo). Per game
it records: total score, balls played, score per ball, the score multiplier in effect (hit points /
base points), bank completions, and every large single award (>= 100k base points) with the object
that triggered it, so missions, jackpots and extra balls show up by name.

Usage: python agents/experiments/fullgame/full_game_eval.py --seeds 40000:40032 --workers 16
"""
from __future__ import annotations

import argparse
import collections
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

BRAIN = "agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt"
CONN = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")
DT_STEP = 0.0347
BIG_AWARD = 100000
_ENV = None
_BINARY = [None]
_GAMES_ON_ENV = [0]
_NAMES: dict = {}
_AGENT = None


def _init(binary: str) -> None:
	global _ENV
	import atexit
	import signal

	import torch

	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_BINARY[0] = binary
	_ENV = PinballEnv(binary_path=binary, headless=True, frame_skip=4, relaunch_at_rest=True)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(lambda: _ENV.close())
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		_NAMES.update({o["id"]: o["name"] for o in json.load(f)["objects"]})


def _brain():
	global _AGENT
	if _AGENT is None:
		import torch

		from agents import train_pinball_circuit_cem as T
		from agents.experiments.encoding.encoding import build_encoded_agent

		ck = torch.load(os.path.join(ROOT, BRAIN), weights_only=False)
		_AGENT = build_encoded_agent(CONN, int(ck["readout_dim"]), ck["encoding"])
		_AGENT.calibrate(_ENV.observation_space)
		T.set_flat_params(_AGENT.decoder, np.asarray(ck["champion"], dtype=np.float32))
	return _AGENT


def _game(task) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	global _ENV
	from env_python.pinball_env import PinballEnv

	policy, seed, max_steps = task
	# The env's reset() is a plain table Reset (not NewGame, see ipc_server.cpp), which does not
	# restore the remaining-ball count after a game over - so every full game gets a fresh engine.
	if _GAMES_ON_ENV[0] > 0:
		_ENV.close()
		_ENV = PinballEnv(binary_path=_BINARY[0], headless=True, frame_skip=4, relaunch_at_rest=True)
	_GAMES_ON_ENV[0] += 1
	obs, _ = _ENV.reset(seed=seed)

	def fresh():
		if policy == "brain":
			agent = _brain()
			return dict(h=agent.initial_state(1), filt=T.FlipperObsFilter("delay1"))
		return dict(reflex=T.ReflexLabeler(11.5, 1))

	st = fresh()
	score = 0.0
	ball_scores = [0.0]
	completions = 0
	mult_seen = collections.Counter()
	by_object = collections.Counter()
	big = []
	steps = 0
	terminated = False
	prev_drained = False
	with torch.no_grad():
		while steps < max_steps:
			if policy == "brain":
				o = torch.as_tensor(np.asarray(st["filt"](obs), dtype=np.float32)).unsqueeze(0)
				a, st["h"] = _brain().act(o, st["h"], greedy=True)
				act = np.asarray(a[0], dtype=np.int64)
				act[2] = 0
			else:
				view = np.array(obs, dtype=np.float32)
				if view[OBS_BALL_VY] > 0:
					view[OBS_BALL_Y] += view[OBS_BALL_VY] * DT_STEP
				act = st["reflex"].label(view)
				act[2] = 0
				st["reflex"].observe(act)
			obs, _r, terminated, _trunc, info = _ENV.step(act)
			steps += 1
			score += info["score_delta"]
			ball_scores[-1] += info["score_delta"]
			for (oid, pts, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
				name = _NAMES.get(oid, str(oid))
				by_object[name] += pts
				if base > 0 and pts > 0:
					mult_seen[round(pts / base)] += pts
				if name in ("a_targ7", "a_targ8", "a_targ9") and base >= 1500:
					completions += 1
				if base >= BIG_AWARD:
					big.append((name, int(pts), int(base), steps))
			if info.get("unattributed_points", 0) >= BIG_AWARD:
				big.append(("unattributed", int(info["unattributed_points"]), 0, steps))
			if terminated:
				break
			if info["drained"] and not prev_drained:  # rising edge only: the flag may stay up until relaunch
				ball_scores.append(0.0)
				st = fresh()
			prev_drained = bool(info["drained"])
	if ball_scores and ball_scores[-1] == 0.0 and len(ball_scores) > 1:
		ball_scores.pop()
	return dict(policy=policy, seed=seed, score=score, steps=steps, game_over=bool(terminated), balls=len(ball_scores),
	            ball_scores=ball_scores, completions=completions, multiplier_points=dict(mult_seen),
	            top_objects=by_object.most_common(8), big_awards=big)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="40000:40032")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=20000)
	ap.add_argument("--policies", default="brain,lead1")
	ap.add_argument("--out", default=os.path.join(HERE, "full_game_eval.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	policies = args.policies.split(",")
	tasks = [(p, s, args.max_steps) for s in range(lo, hi) for p in policies]
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		res = pool.map(_game, tasks, chunksize=1)
	for p in policies:
		rs = [r for r in res if r["policy"] == p]
		sc = np.array([r["score"] for r in rs])
		print(f"\n== {p}: {len(rs)} games, game over reached {np.mean([r['game_over'] for r in rs]):.0%}")
		print(f"  total score: median {np.median(sc):,.0f}  mean {sc.mean():,.0f}  max {sc.max():,.0f}  "
		      f"log1p {np.log1p(sc).mean():.2f} +- {np.log1p(sc).std(ddof=1) / math.sqrt(len(sc)):.2f}")
		print(f"  balls per game {np.mean([r['balls'] for r in rs]):.2f}  steps {np.mean([r['steps'] for r in rs]):.0f}  "
		      f"bank completions per game {np.mean([r['completions'] for r in rs]):.2f}")
		mult = collections.Counter()
		for r in rs:
			mult.update({int(k): v for k, v in r["multiplier_points"].items()})
		tot = sum(mult.values()) or 1
		print("  share of points by multiplier: " + "  ".join(f"x{k} {v / tot:.0%}" for k, v in sorted(mult.items())))
		bigs = collections.Counter()
		big_pts = 0
		for r in rs:
			for name, pts, _b, _s in r["big_awards"]:
				bigs[name] += 1
				big_pts += pts
		print(f"  big awards (>=100k base): {sum(bigs.values())} in {len(rs)} games, {big_pts / max(1, sc.sum()):.0%} of all points; "
		      + ", ".join(f"{n} x{c}" for n, c in bigs.most_common(8)))
		objs = collections.Counter()
		for r in rs:
			objs.update(dict(r["top_objects"]))
		print("  top objects by points: " + ", ".join(f"{n} {v / sc.sum():.0%}" for n, v in objs.most_common(8)))
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), games=res), fh)


if __name__ == "__main__":
	main()
