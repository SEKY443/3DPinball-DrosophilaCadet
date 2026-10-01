#!/usr/bin/env python3
"""Root-cause study (step 3 follow-up): compares the trained champions with brainless baselines on
the same one-ball lives (trainer's `life` episode), and reports each policy's press pattern.

Policy specs (comma-separated list via --policies):
  none                 never press
  metro:P:PH           open loop: press BOTH flippers on every step t with t % P == PH, else release
  reflex:Y             press the flipper on the ball's side (x<=0 -> left) while y > Y and vy > 0
  reflexboth:Y         press both flippers while y > Y and vy > 0
  reflexswap:Y         like reflex but with the side mapping mirrored (x<=0 -> right). NOTE: this is
                       the PHYSICALLY CORRECT mapping - action 0 (left flipper) sits at positive table
                       x (the table x axis is mirrored vs the screen), so `reflex` presses the wrong side
  pulse:Y              correct-side reactive pulse: press the ball-side flipper for one step when
                       y > Y and vy > 0, at most once every 3 steps per side
  gated:Y:P            metronome with period P (both flippers) only while the ball is below y > Y
  lab:Y:H              train_pinball_circuit_cem.ReflexLabeler(threshold=Y, hold=H) (H=1 == pulse:Y)
  ckpt:PATH            trained checkpoint (circuit or mlp, from the checkpoint's `agent` field)

Per policy: mean log1p(score) (the trainer's life fitness without press cost), mean score, mean
steps, presses per life, fraction of steps with left == right, share of press gaps equal to 3.

Usage:
	python agents/experiments/step3/policy_compare.py --policies none,metro:3:0,reflex:10 --episodes 24
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

_ENV = None
_AGENTS: dict = {}
_CFG: dict = {}


def _init(cfg: dict) -> None:
	global _ENV
	import gymnasium as gym
	import torch

	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_CFG.update(cfg)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=cfg["binary"], headless=True, frame_skip=cfg["frame_skip"]),
	                              max_episode_steps=cfg["max_steps"])
	import atexit
	import signal

	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)


def _agent(path: str):
	if path not in _AGENTS:
		import torch

		from agents import train_pinball_circuit_cem as T

		ckpt = torch.load(path, weights_only=False)
		kind = ckpt.get("agent", "circuit")
		agent = T.build_agent(kind, _CFG["connectome"], ckpt.get("readout_dim", 8))
		agent.calibrate(_ENV.observation_space)
		T.set_flat_params(agent.decoder, np.asarray(ckpt["champion"], dtype=np.float32))
		_AGENTS[path] = agent
	return _AGENTS[path]


def _life(task: tuple) -> dict:
	import torch

	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	spec, seed = task
	kind, *rest = spec.split(":", 1)
	agent = _agent(rest[0]) if kind == "ckpt" else None
	obs, _ = _ENV.reset(seed=seed)
	h = agent.initial_state(1) if agent is not None else None
	lab = None
	if kind == "lab":
		from agents.train_pinball_circuit_cem import ReflexLabeler

		y, hold = rest[0].split(":")
		lab = ReflexLabeler(float(y), int(hold))
	acts, score, steps, contacts, done = [], 0, 0, 0, False
	while not done:
		if kind == "none":
			a = np.zeros(3, dtype=np.int64)
		elif kind == "metro":
			period, phase = (int(v) for v in rest[0].split(":"))
			p = int(steps % period == phase)
			a = np.array([p, p, 0], dtype=np.int64)
		elif kind == "lab":
			a = lab.label(obs)
			a[2] = 0
			lab.observe(a)
		elif kind == "pulse":
			on = obs[OBS_BALL_Y] > float(rest[0]) and obs[OBS_BALL_VY] > 0
			side = 0 if obs[OBS_BALL_X] > 0 else 1
			a = np.zeros(3, dtype=np.int64)
			if on and len(acts) >= 2 and not acts[-1][side] and not acts[-2][side]:
				a[side] = 1
			elif on and len(acts) < 2:
				a[side] = 1
		elif kind == "gated":
			thr, period = rest[0].split(":")
			p = int(obs[OBS_BALL_Y] > float(thr) and steps % int(period) == 0)
			a = np.array([p, p, 0], dtype=np.int64)
		elif kind in ("reflex", "reflexboth", "reflexswap"):
			thr = float(rest[0])
			on = obs[OBS_BALL_Y] > thr and obs[OBS_BALL_VY] > 0
			if kind == "reflexboth":
				a = np.array([on, on, 0], dtype=np.int64)
			elif kind == "reflexswap":
				a = np.array([on and obs[OBS_BALL_X] > 0, on and obs[OBS_BALL_X] <= 0, 0], dtype=np.int64)
			else:
				a = np.array([on and obs[OBS_BALL_X] <= 0, on and obs[OBS_BALL_X] > 0, 0], dtype=np.int64)
		else:
			with torch.no_grad():
				out, h = agent.act(torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0), h, greedy=True)
			a = np.asarray(out[0], dtype=np.int64)
		obs, _r, term, trunc, info = _ENV.step(a)
		acts.append(a[:2].copy())
		score += info["score_delta"]
		contacts += int(info["flipper_hit"])
		steps += 1
		done = term or trunc or info["drained"]
	acts = np.array(acts, dtype=np.int64)
	edges = int(((acts[1:] == 1) & (acts[:-1] == 0)).sum() + acts[0].sum())
	gaps = []
	for side in (0, 1):
		on = np.where(acts[:, side] == 1)[0]
		starts = on[np.r_[True, np.diff(on) > 1]] if len(on) else on
		gaps.extend(np.diff(starts).tolist())
	return dict(spec=spec, seed=seed, score=int(score), steps=steps, contacts=contacts, presses=edges,
	            lr_equal=float((acts[:, 0] == acts[:, 1]).mean()), drained=bool(info["drained"]),
	            gap3=float(np.mean(np.array(gaps) == 3)) if gaps else 0.0,
	            p_left=float(acts[:, 0].mean()), p_right=float(acts[:, 1].mean()))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--policies", required=True)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--connectome", default=os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"))
	ap.add_argument("--frame-skip", type=int, default=4)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--episodes", type=int, default=24)
	ap.add_argument("--seed0", type=int, default=7000)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--out", default=None)
	args = ap.parse_args()
	if 5000 <= args.seed0 < 6000:
		sys.exit("seeds 5000+ are the held-out confirmation set - use 7000+")
	specs = args.policies.split(",")
	seeds = list(range(args.seed0, args.seed0 + args.episodes))
	cfg = dict(binary=args.binary, connectome=args.connectome, frame_skip=args.frame_skip, max_steps=args.max_steps)
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
		res = pool.map(_life, [(s, sd) for s in specs for sd in seeds], chunksize=1)
	print(f"{len(seeds)} lives per policy, seeds {seeds[0]}..{seeds[-1]}, {time.time() - t0:.0f}s")
	print(f"{'policy':<58} {'log1p':>6} {'se':>5} {'score':>8} {'steps':>6} {'cont':>5} {'press':>6} {'L==R':>5} {'gap3':>5} {'drain':>5}")
	rows = {}
	for s in specs:
		r = [x for x in res if x["spec"] == s]
		lg = np.log1p([x["score"] for x in r])
		row = dict(log1p=float(lg.mean()), se=float(lg.std(ddof=1) / np.sqrt(len(lg))), score=float(np.mean([x["score"] for x in r])),
		           steps=float(np.mean([x["steps"] for x in r])), contacts=float(np.mean([x["contacts"] for x in r])),
		           presses=float(np.mean([x["presses"] for x in r])), lr_equal=float(np.mean([x["lr_equal"] for x in r])),
		           gap3=float(np.mean([x["gap3"] for x in r])), drained=float(np.mean([x["drained"] for x in r])))
		rows[s] = row
		name = s if len(s) <= 58 else "..." + s[-55:]
		print(f"{name:<58} {row['log1p']:6.2f} {row['se']:5.2f} {row['score']:8.0f} {row['steps']:6.0f} {row['contacts']:5.1f} "
		      f"{row['presses']:6.0f} {row['lr_equal']:5.2f} {row['gap3']:5.2f} {row['drained']:5.2f}")
	if args.out:
		with open(args.out, "w", encoding="utf-8") as f:
			json.dump(dict(args=vars(args), summary=rows, lives=res), f)


if __name__ == "__main__":
	main()
