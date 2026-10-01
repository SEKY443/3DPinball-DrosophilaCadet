"""DAgger of the popcode brain toward the lead1 reflex teacher inside FULL games.

The lead-seeded brain was imitation-trained on single lives only, so it never practised the moments
after a drain (circuit reset, plunger relaunch, next ball). In full games it trails its teacher
(median ~0.82M vs ~1.15M). This continues training from that brain with DAgger rollouts of complete
games: each round the current brain plays full games (its circuit state reset at every drain, as in
the web demo), the lead1 reflex labels every step, the dataset is aggregated and the readout's
projection + head are refit exactly as in agents/experiments/encoding/dagger_enc.py. Round 0 is
driven by the teacher itself. Finally the new brain and the starting brain play the same fresh full
games for a paired comparison.

Usage: python agents/experiments/fullgame/dagger_full.py --rounds 3 --games 24 --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "encoding"))

START = os.path.join(ROOT, "agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt")
CONN = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")
DT_STEP = 0.0347
_BINARY = [None]
_AGENT = [None]


def _init(binary: str) -> None:
	import signal

	import torch

	torch.set_num_threads(1)
	_BINARY[0] = binary
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def _agent():
	if _AGENT[0] is None:
		import torch

		from agents.experiments.encoding.encoding import build_encoded_agent

		ck = torch.load(START, weights_only=False)
		agent = build_encoded_agent(CONN, int(ck["readout_dim"]), ck["encoding"])
		# calibrate() only reads the observation space; the readout's norm params are then overwritten
		# by every set_flat_params call, exactly as in the trainer's _worker_init.
		agent.calibrate(_obs_space())
		_AGENT[0] = agent
	return _AGENT[0]


def _obs_space():
	import gymnasium as gym

	from env_python.pinball_env import OBS_DIM

	return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)


TEACHER_ENV = "DAGGER_TEACHER"  # "lead1" (default) or "cradle_s5"; read in every spawned worker


class _Lead1Teacher:
	def __init__(self):
		from agents import train_pinball_circuit_cem as T

		self.reflex = T.ReflexLabeler(11.5, 1)

	def label(self, obs):
		return _lead_label(self.reflex, obs)

	def observe(self, act):
		self.reflex.observe(act)


class _CradleTeacher:
	"""Wraps agents/experiments/cradle/cradle_teacher.CradleTeacher (binary flippers). Its state
	machine advances on its own labels; the agent's actual actions are not fed back."""

	def __init__(self, catch_speed: float):
		sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "cradle"))
		from cradle_teacher import CradleTeacher

		self.t = CradleTeacher(catch_speed=catch_speed)

	def label(self, obs):
		return np.asarray(self.t.act(obs), dtype=np.int64)

	def observe(self, act):
		pass


def _make_teacher():
	kind = os.environ.get(TEACHER_ENV, "lead1")
	if kind == "lead1":
		return _Lead1Teacher()
	if kind == "cradle_s5":
		return _CradleTeacher(5.0)
	raise SystemExit(f"unknown teacher {kind!r}")


def _lead_label(reflex, obs):
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	view = np.array(obs, dtype=np.float32)
	if view[OBS_BALL_VY] > 0:
		view[OBS_BALL_Y] += view[OBS_BALL_VY] * DT_STEP
	a = reflex.label(view)
	a[2] = 0
	return a


def _game(task) -> dict:
	"""One full game. drive='teacher' or 'agent'. Returns score and, if collect, features + labels."""
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import PinballEnv

	weights, seed, drive, collect, max_steps = task
	agent = _agent()
	T.set_flat_params(agent.decoder, np.asarray(weights, dtype=np.float32))
	dec = agent.decoder
	env = PinballEnv(binary_path=_BINARY[0], headless=True, frame_skip=4, relaunch_at_rest=True)
	X, L = [], []
	score = 0.0
	presses = 0
	game_over = False
	try:
		obs, _ = env.reset(seed=seed)
		h = agent.initial_state(1)
		filt = T.FlipperObsFilter("delay1")
		teacher = _make_teacher()
		prev = np.zeros(2, dtype=bool)
		prev_drained = False
		with torch.no_grad():
			for _ in range(max_steps):
				label = teacher.label(obs)
				o = torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0)
				if drive == "teacher":
					h = agent.brain.step(h, agent._channel_drive(o))
					act = label
				else:
					a, h = agent.act(o, h, greedy=True)
					act = np.asarray(a[0], dtype=np.int64)
					act[2] = 0
				if collect:
					X.append(dec.norm(h[:, dec.idx])[0].numpy().astype(np.float32))
					L.append(label[:2].astype(np.float32))
				teacher.observe(act)
				pressed = act[:2].astype(bool)
				presses += int((pressed & ~prev).sum())
				prev = pressed
				obs, _r, term, _trunc, info = env.step(act)
				score += info["score_delta"]
				if info["drained"] and not prev_drained:
					h = agent.initial_state(1)
					filt = T.FlipperObsFilter("delay1")
					teacher = _make_teacher()
				prev_drained = bool(info["drained"])
				if term:
					game_over = True
					break
	finally:
		env.close()
	out = dict(seed=seed, score=float(score), presses=presses, game_over=game_over)
	if collect:
		out["X"] = np.stack(X)
		out["L"] = np.stack(L)
	return out


def _fit(dec, feats, labels, epochs: int) -> None:
	"""Refit projection + head on the aggregated dataset (same recipe as dagger_enc._variant)."""
	import torch

	pos = labels.sum(0)
	pw = ((labels.shape[0] - pos) / pos.clamp_min(1)).clamp(max=200.0)
	opt = torch.optim.Adam(list(dec.proj.parameters()) + list(dec.head.parameters()), lr=1e-2)
	for _ in range(epochs):
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
		dec.head.bias[2] = -10.0


def main() -> None:
	import torch

	import confirm
	from agents import train_pinball_circuit_cem as T

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--rounds", type=int, default=3)
	ap.add_argument("--games", type=int, default=24, help="full games per DAgger round")
	ap.add_argument("--train-seed-base", type=int, default=63000)
	ap.add_argument("--eval-seeds", default="62000:62064")
	ap.add_argument("--epochs", type=int, default=600)
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=30000)
	ap.add_argument("--max-samples", type=int, default=400000, help="cap on the aggregated dataset (random subsample)")
	ap.add_argument("--teacher", choices=["lead1", "cradle_s5"], default="lead1")
	ap.add_argument("--start", default=START, help="checkpoint to continue from (default: the lead brain)")
	ap.add_argument("--out", default=os.path.join(HERE, "dagger_full_brain.pt"))
	args = ap.parse_args()
	os.environ[TEACHER_ENV] = args.teacher
	torch.manual_seed(0)
	rng = np.random.default_rng(0)
	t0 = time.time()

	ck = torch.load(args.start, weights_only=False)
	w0 = np.asarray(ck["champion"], dtype=np.float32)
	main_agent = _agent()
	T.set_flat_params(main_agent.decoder, w0)
	dec = main_agent.decoder
	X_all, L_all = [], []
	w = w0.copy()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		for rnd in range(args.rounds + 1):
			drive = "teacher" if rnd == 0 else "agent"
			seeds = [args.train_seed_base + 1000 * rnd + i for i in range(args.games)]
			res = pool.map(_game, [(w, s, drive, True, args.max_steps) for s in seeds], chunksize=1)
			for r in res:
				X_all.append(r["X"])
				L_all.append(r["L"])
			lg = np.log1p([r["score"] for r in res])
			feats = np.concatenate(X_all)
			labels = np.concatenate(L_all)
			if len(feats) > args.max_samples:
				keep = rng.choice(len(feats), args.max_samples, replace=False)
				feats, labels = feats[keep], labels[keep]
			_fit(dec, torch.as_tensor(feats), torch.as_tensor(labels), args.epochs)
			w = T.get_flat_params(dec)
			print(f"round {rnd} ({drive}-driven): mean log1p {lg.mean():.2f}, median {np.median([r['score'] for r in res]):,.0f}, "
			      f"presses/game {np.mean([r['presses'] for r in res]):.0f}, dataset {len(feats)}  {time.time() - t0:.0f}s", flush=True)

		lo, hi = (int(v) for v in args.eval_seeds.split(":"))
		eval_seeds = list(range(lo, hi))
		tasks = [(wt, s, "agent", False, args.max_steps) for s in eval_seeds for wt in (w, w0)]
		res = pool.map(_game, tasks, chunksize=1)
	new = {r["seed"]: r for r in res[0::2]}
	old = {r["seed"]: r for r in res[1::2]}
	print(f"\npaired full games {lo}..{hi - 1} ({len(eval_seeds)})")
	for name, by in (("new (full-game DAgger)", new), ("old (lead brain)", old)):
		sc = np.array([by[s]["score"] for s in eval_seeds])
		print(f"  {name:<24} log1p {np.log1p(sc).mean():.3f} +- {np.log1p(sc).std(ddof=1) / math.sqrt(len(sc)):.3f}  "
		      f"median {np.median(sc):,.0f}  mean {sc.mean():,.0f}  presses/game {np.mean([by[s]['presses'] for s in eval_seeds]):.0f}  "
		      f"game over {np.mean([by[s]['game_over'] for s in eval_seeds]):.0%}")
	d = np.array([np.log1p(new[s]["score"]) - np.log1p(old[s]["score"]) for s in eval_seeds])
	se = d.std(ddof=1) / math.sqrt(len(d))
	print(f"  new - old log1p {d.mean():+.3f} +- {se:.3f}  t {d.mean() / se:+.2f}  p {confirm.wilcoxon_p(d):.3g}  "
	      f"wins {int((d > 0).sum())}/{int((d < 0).sum())}")
	out = dict(ck, mean=w, champion=w, champion_fitness=-1e18, generation=0,
	           seed_policy=f"full-game DAgger({args.rounds}) of {args.teacher} teacher from {os.path.basename(args.start)}")
	torch.save(out, args.out)
	print(f"saved {args.out}")


if __name__ == "__main__":
	main()
