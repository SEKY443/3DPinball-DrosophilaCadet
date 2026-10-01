#!/usr/bin/env python3
"""Imitation-seeds the circuit decoder from the correct-side reflex (train_pinball_circuit_cem.
ReflexLabeler) in several variants and scores each seeded decoder on HELD-OUT lives.

Variants = reflex (threshold, hold) x flipper-obs mode (raw / zero / delay1, see FlipperObsFilter,
applied at seeding AND evaluation). Seeding rollouts use --seed-seeds (default 7000-7023, the
tuning set); evaluation uses --eval-seed0.. (default 7024-7071), so nothing is tuned and scored on
the same lives. Reported per variant, closed loop (the seeded agent drives):
  log1p(score/life), score, steps, contacts, presses/life, P(L), P(R),
  agreement with a SHADOW reflex fed the same observations and the agent's own actions:
    agree = all-steps match on both flippers, recall = P(agent presses that side | reflex does),
    precision = P(reflex presses that side | agent does).
Also writes a resumable start checkpoint per variant (champion = mean = seeded weights, sigma
0.2, champion_fitness -1e18, generation 0) under --ckpt-dir.

Usage: python agents/experiments/step3/seed_eval.py --workers 2
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


def _seed_job(job: dict) -> np.ndarray:
	import torch

	from agents import train_pinball_circuit_cem as T

	torch.set_num_threads(1)
	return T.heuristic_seed(job["binary"], job["connectome"], 8, 4, episodes=len(job["seeds"]), max_steps=3000,
	                        epochs=job["epochs"], seed=0, seed_policy="reflex", reflex_y=job["y"],
	                        reflex_hold=job["hold"], flipper_obs=job["fobs"], episode_seeds=job["seeds"],
	                        end_at_drain=True, calibrate_bias=job["calib"])


def _init(cfg: dict) -> None:
	global _ENV
	import atexit
	import signal

	import gymnasium as gym
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_CFG.update(cfg)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=cfg["binary"], headless=True, frame_skip=4), max_episode_steps=3000)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)


def _eval(task: tuple) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T

	name, weights, y, hold, fobs, seed, rdim = task
	if rdim not in _AGENTS:
		_AGENTS[rdim] = T.build_agent("circuit", _CFG["connectome"], rdim)
		_AGENTS[rdim].calibrate(_ENV.observation_space)
	_AGENT = _AGENTS[rdim]
	T.set_flat_params(_AGENT.decoder, np.asarray(weights, dtype=np.float32))
	filt = T.FlipperObsFilter(fobs.split("@")[0], loom_tau0=float(fobs.split("@")[1]) if "@" in fobs else 2.0)
	shadow = T.ReflexLabeler(y, hold)
	obs, _ = _ENV.reset(seed=seed)
	h = _AGENT.initial_state(1)
	score = steps = contacts = 0
	agree = lab_press = agent_press = hit = 0
	acts = []
	done = False
	with torch.no_grad():
		while not done:
			label = shadow.label(obs)
			a, h = _AGENT.act(torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0), h, greedy=True)
			a = np.asarray(a[0], dtype=np.int64)
			shadow.observe(a)
			agree += int((a[:2] == label[:2]).all())
			lab_press += int(label[:2].sum())
			agent_press += int(a[:2].sum())
			hit += int((a[:2] & label[:2]).sum())
			acts.append(a[:2].copy())
			obs, _r, term, trunc, info = _ENV.step(a)
			score += info["score_delta"]
			contacts += int(info["flipper_hit"])
			steps += 1
			done = term or trunc or info["drained"]
	acts = np.array(acts)
	presses = int(((acts[1:] == 1) & (acts[:-1] == 0)).sum() + acts[0].sum())
	return dict(name=name, seed=seed, score=int(score), steps=steps, contacts=contacts, presses=presses,
	            p_left=float(acts[:, 0].mean()), p_right=float(acts[:, 1].mean()), agree=agree / steps,
	            lab_press=lab_press, agent_press=agent_press, hit=hit)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--connectome", default=os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"))
	ap.add_argument("--variants", default="11.5:1:raw,11.5:1:zero,11.5:1:delay1,10.5:3:raw,10.5:3:zero,10.5:3:delay1",
	                help="comma list of Y:HOLD:FLIPPER_OBS")
	ap.add_argument("--seed-seeds", default="7000:7024", help="lo:hi range of seeding lives")
	ap.add_argument("--eval-seed0", type=int, default=7024)
	ap.add_argument("--eval-episodes", type=int, default=48)
	ap.add_argument("--epochs", type=int, default=300)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--eval-ckpts", default=None, help="comma list of start checkpoints to score instead of seeding "
	                "(reflex Y/HOLD for the shadow agreement are taken from --shadow)")
	ap.add_argument("--shadow", default="11.5:1")
	ap.add_argument("--no-calibrate", action="store_true", help="skip heuristic_seed's head-bias rate calibration")
	ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "agents/experiments/step3/seeds"))
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/step3/seed_eval.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seed_seeds.split(":"))
	eval_seeds = list(range(args.eval_seed0, args.eval_seed0 + args.eval_episodes))
	assert not set(eval_seeds) & set(range(lo, hi)), "seeding and evaluation lives overlap"
	assert all(not (5000 <= s < 6000) for s in list(range(lo, hi)) + eval_seeds), "5000+ is the held-out set"
	variants = [v.split(":") for v in args.variants.split(",")]
	ctx = mp.get_context("spawn")
	t0 = time.time()

	if args.eval_ckpts:
		import torch

		sy, sh = args.shadow.split(":")
		names, weights, variants, rdims = [], [], [], []
		for path in args.eval_ckpts.split(","):
			ck = torch.load(path, weights_only=False)
			names.append(os.path.splitext(os.path.basename(path))[0])
			weights.append(np.asarray(ck["champion"], dtype=np.float32))
			fo = ck.get("flipper_obs", "raw")
			variants.append((sy, sh, fo + (f"@{ck['loom_tau0']}" if "loom" in fo else "")))
			rdims.append(int(ck["readout_dim"]))
		_score(args, names, weights, variants, rdims, eval_seeds, ctx, t0)
		return

	jobs = [dict(binary=args.binary, connectome=args.connectome, y=float(y), hold=int(hd), fobs=f,
	             seeds=list(range(lo, hi)), epochs=args.epochs, calib=not args.no_calibrate) for y, hd, f in variants]
	with ctx.Pool(args.workers) as pool:
		weights = pool.map(_seed_job, jobs, chunksize=1)
	print(f"seeded {len(jobs)} variants in {time.time() - t0:.0f}s", flush=True)

	import torch

	os.makedirs(args.ckpt_dir, exist_ok=True)
	names = []
	for (y, hd, f), w in zip(variants, weights):
		name = f"reflex_y{y}_h{hd}_{f}"
		names.append(name)
		torch.save({"mean": w.astype(np.float32), "sigma": np.full(w.size, 0.2, dtype=np.float32), "champion": w.astype(np.float32),
		            "champion_fitness": -1e18, "generation": 0, "curriculum_start_gen": 1, "readout_dim": 8, "agent": "circuit",
		            "objective": "life", "drill_bank": None, "press_cost": 0.0, "flipper_obs": f,
		            "seed_policy": f"reflex y>{y} hold {hd}", "seed_lives": [lo, hi]}, os.path.join(args.ckpt_dir, name + ".pt"))

	_score(args, names, weights, variants, [8] * len(names), eval_seeds, ctx, t0)


def _score(args, names, weights, variants, rdims, eval_seeds, ctx, t0) -> None:
	tasks = [(n, w.tolist(), float(y), int(hd), f, s, rd) for n, w, (y, hd, f), rd in zip(names, weights, variants, rdims)
	         for s in eval_seeds]
	with ctx.Pool(args.workers, initializer=_init, initargs=(dict(binary=args.binary, connectome=args.connectome),)) as pool:
		res = pool.map(_eval, tasks, chunksize=1)

	summary = {}
	print(f"eval lives {eval_seeds[0]}..{eval_seeds[-1]}  total {time.time() - t0:.0f}s")
	print(f"{'variant':<34} {'log1p':>6} {'se':>5} {'score':>8} {'steps':>6} {'cont':>5} {'press':>6} {'P(L)':>5} {'P(R)':>5} "
	      f"{'agree':>6} {'recall':>6} {'prec':>6}")
	for n in names:
		r = [x for x in res if x["name"] == n]
		lg = np.log1p([x["score"] for x in r])
		lab, ag, hit = (sum(x[k] for x in r) for k in ("lab_press", "agent_press", "hit"))
		row = dict(log1p=float(lg.mean()), se=float(lg.std(ddof=1) / np.sqrt(len(lg))), score=float(np.mean([x["score"] for x in r])),
		           steps=float(np.mean([x["steps"] for x in r])), contacts=float(np.mean([x["contacts"] for x in r])),
		           presses=float(np.mean([x["presses"] for x in r])), p_left=float(np.mean([x["p_left"] for x in r])),
		           p_right=float(np.mean([x["p_right"] for x in r])), agree=float(np.mean([x["agree"] for x in r])),
		           recall=hit / lab if lab else float("nan"), precision=hit / ag if ag else float("nan"))
		summary[n] = row
		print(f"{n:<34} {row['log1p']:6.2f} {row['se']:5.2f} {row['score']:8.0f} {row['steps']:6.0f} {row['contacts']:5.1f} "
		      f"{row['presses']:6.1f} {row['p_left']:5.3f} {row['p_right']:5.3f} {row['agree']:6.3f} {row['recall']:6.3f} {row['precision']:6.3f}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), summary=summary, lives=res), fh)


if __name__ == "__main__":
	main()
