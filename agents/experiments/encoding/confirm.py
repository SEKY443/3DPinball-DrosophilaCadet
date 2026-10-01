#!/usr/bin/env python3
"""Paired confirmation evaluation on the untouched seed set 5000-5095 (one life per seed, identical
seeds for every policy). Nothing here is tuned on these seeds.

Policies: popcode seed (start_popcode_seed.pt, popcode encoding, delay1 flipper obs), gen-280
champion (best_heldout_gen280.pt, broadcast encoding, raw flipper obs), the correct-side pulse reflex
(ReflexLabeler y>11.5, hold 1), and never-press.

Episode end = the trainer's _evaluate condition: terminated or truncated (TimeLimit --max-steps 3000)
or info["drained"]. Presses = physical flipper down->up edges read from obs, as the trainer counts
them; contacts = info["flipper_hit"]. Circuit policies act greedily with agent.act, as in training.

Paired statistics: mean difference of log1p(score) +- SE, t = mean / SE, and a two-sided Wilcoxon
signed-rank p (normal approximation with tie and zero corrections - scipy is not installed).
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

CONN = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")
POLICIES = {
	"popcode_seed": dict(kind="circuit", ckpt="agents/experiments/encoding/start_popcode_seed.pt", encoding="popcode", fobs="delay1"),
	"gen280": dict(kind="circuit", ckpt="agents/experiments/big_circuit/best_heldout_gen280.pt", encoding="broadcast", fobs="raw"),
	"reflex_y11.5": dict(kind="reflex"),
	"never_press": dict(kind="none"),
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
		spec = POLICIES[name]
		ck = torch.load(os.path.join(ROOT, spec["ckpt"]), weights_only=False)
		assert ck.get("encoding", "broadcast") == spec["encoding"], "checkpoint/encoding mismatch"
		agent = build_encoded_agent(CONN, int(ck["readout_dim"]), spec["encoding"])
		agent.calibrate(_ENV.observation_space)  # as _worker_init; norm params are then overwritten
		T.set_flat_params(agent.decoder, np.asarray(ck["champion"], dtype=np.float32))
		_AGENTS[name] = agent
	return _AGENTS[name]


def _run(task) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

	name, seed = task
	spec = POLICIES[name]
	obs, info = _ENV.reset(seed=seed)
	if spec["kind"] == "circuit":
		agent = _agent(name)
		h = agent.initial_state(1)
		filt = T.FlipperObsFilter(spec["fobs"])
	elif spec["kind"] == "reflex":
		reflex = T.ReflexLabeler(11.5, 1)
	prev_l, prev_r = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
	score = steps = contacts = presses = 0
	drained = done = False
	with torch.no_grad():
		while not done:
			if spec["kind"] == "circuit":
				a, h = agent.act(torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0), h, greedy=True)
				act = a[0]
			elif spec["kind"] == "reflex":
				act = reflex.label(obs)
				act[2] = 0
				reflex.observe(act)
			else:
				act = np.zeros(3, dtype=np.int64)
			obs, _r, term, trunc, info = _ENV.step(act)
			score += info["score_delta"]
			contacts += int(info["flipper_hit"])
			lu, ru = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
			presses += int(lu and not prev_l) + int(ru and not prev_r)
			prev_l, prev_r = lu, ru
			steps += 1
			drained = bool(info["drained"])
			done = term or trunc or drained
	return dict(policy=name, seed=seed, score=float(score), steps=steps, contacts=contacts, presses=presses,
	            drained=int(drained), truncated=int(bool(trunc)))


def wilcoxon_p(d: np.ndarray) -> float:
	d = d[d != 0]
	n = len(d)
	if n == 0:
		return 1.0
	ad = np.abs(d)
	order = np.argsort(ad)
	ranks = np.empty(n)
	sorted_ad = ad[order]
	i = 0
	tie_term = 0.0
	while i < n:  # average ranks for ties
		j = i
		while j + 1 < n and sorted_ad[j + 1] == sorted_ad[i]:
			j += 1
		ranks[order[i:j + 1]] = (i + j) / 2 + 1
		t = j - i + 1
		tie_term += t ** 3 - t
		i = j + 1
	w_plus = ranks[d > 0].sum()
	mu = n * (n + 1) / 4
	sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
	z = (w_plus - mu) / sd
	return math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="5000:5096")
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--press-cost", type=float, default=0.00088)
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/encoding/confirm_5000.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	tasks = [(p, s) for s in seeds for p in POLICIES]
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(_run, tasks, chunksize=1)
	by = {p: {r["seed"]: r for r in res if r["policy"] == p} for p in POLICIES}
	summary = {}
	print(f"confirmation lives {lo}..{hi - 1} ({len(seeds)} lives, paired), max_steps {args.max_steps}, {time.time() - t0:.0f}s")
	print(f"{'policy':<14} {'log1p':>6} {'se':>5} {'fit@pc':>6} {'median':>8} {'mean':>8} {'steps':>6} {'press':>6} {'cont':>5} {'drain%':>6} {'trunc%':>6}")
	for p in POLICIES:
		rs = [by[p][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		pr = np.array([r["presses"] for r in rs])
		row = dict(log1p=float(lg.mean()), se=float(lg.std(ddof=1) / math.sqrt(len(lg))),
		           fitness_with_press_cost=float((lg - args.press_cost * pr).mean()),
		           median_score=float(np.median([r["score"] for r in rs])), mean_score=float(np.mean([r["score"] for r in rs])),
		           steps=float(np.mean([r["steps"] for r in rs])), presses=float(pr.mean()),
		           contacts=float(np.mean([r["contacts"] for r in rs])),
		           drained=float(np.mean([r["drained"] for r in rs])), truncated=float(np.mean([r["truncated"] for r in rs])))
		summary[p] = row
		print(f"{p:<14} {row['log1p']:6.2f} {row['se']:5.2f} {row['fitness_with_press_cost']:6.2f} {row['median_score']:8.0f} "
		      f"{row['mean_score']:8.0f} {row['steps']:6.0f} {row['presses']:6.1f} {row['contacts']:5.1f} "
		      f"{100 * row['drained']:6.0f} {100 * row['truncated']:6.0f}")
	paired = {}
	for a, b in (("popcode_seed", "gen280"), ("popcode_seed", "reflex_y11.5"), ("popcode_seed", "never_press"),
	             ("reflex_y11.5", "gen280")):
		for label, pc in (("log1p", 0.0), ("with_press_cost", args.press_cost)):
			d = np.array([np.log1p(by[a][s]["score"]) - pc * by[a][s]["presses"]
			              - np.log1p(by[b][s]["score"]) + pc * by[b][s]["presses"] for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			paired[f"{a} - {b} [{label}]"] = dict(mean=float(d.mean()), se=float(se), t=float(d.mean() / se),
			                                      wilcoxon_p=wilcoxon_p(d), wins=int((d > 0).sum()), losses=int((d < 0).sum()))
	print("paired differences (same seeds): mean +- se, t, Wilcoxon p (normal approx), wins/losses")
	for k, v in paired.items():
		print(f"  {k:<44} {v['mean']:+6.3f} +- {v['se']:.3f}  t {v['t']:+5.2f}  p {v['wilcoxon_p']:.3g}  {v['wins']}/{v['losses']}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), summary=summary, paired=paired, lives=res), fh)


if __name__ == "__main__":
	main()
