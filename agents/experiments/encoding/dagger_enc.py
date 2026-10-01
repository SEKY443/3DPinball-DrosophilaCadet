#!/usr/bin/env python3
"""DAgger imitation of the correct-side pulse reflex through the fixed circuit under a chosen sensory
encoding (agents/experiments/encoding/encoding.py), then closed-loop scoring on held-out lives.

Same procedure as agents/experiments/step3/dagger_seed.py (round 0 = reflex rollouts, rounds 1..N =
agent rollouts relabelled by a shadow reflex fed the agent's own actions; refit proj + head with
pos-weighted BCE, then rate-calibrate the head biases), and the same held-out scoring as
agents/experiments/step3/seed_eval.py (log1p score per life, presses, shadow recall/precision).
Tuning lives 7000-7023, held-out lives 7024-7071. One worker process per variant (ENC:FOBS:RDIM).

Usage: python agents/experiments/encoding/dagger_enc.py --variants dspop+ttc:delay1:8,popcode:delay1:8
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


def _rollout(env, agent, T, fobs, seed, drive_with_reflex, X=None, L=None):
	import torch

	obs, _ = env.reset(seed=seed)
	h = agent.initial_state(1)
	filt = T.FlipperObsFilter(fobs)
	lab = T.ReflexLabeler(11.5, 1)
	dec = agent.decoder
	score = steps = contacts = agree = lab_press = agent_press = hit = 0
	acts, done = [], False
	while not done:
		label = lab.label(obs)
		label[2] = 0
		with torch.no_grad():
			o = torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0)
			if drive_with_reflex:
				h = agent.brain.step(h, agent._channel_drive(o))
				act = label
			else:
				a, h = agent.act(o, h, greedy=True)
				act = np.asarray(a[0], dtype=np.int64)
				act[2] = 0
			if X is not None:
				X.append(dec.norm(h[:, dec.idx])[0].clone())
				L.append(label[:2].copy())
		lab.observe(act)
		agree += int((act[:2] == label[:2]).all())
		lab_press += int(label[:2].sum()); agent_press += int(act[:2].sum()); hit += int((act[:2] & label[:2]).sum())
		acts.append(act[:2].copy())
		obs, _r, term, trunc, info = env.step(act)
		score += info["score_delta"]; contacts += int(info["flipper_hit"]); steps += 1
		done = term or trunc or info["drained"]
	acts = np.array(acts)
	presses = int(((acts[1:] == 1) & (acts[:-1] == 0)).sum() + acts[0].sum())
	return dict(seed=seed, score=int(score), steps=steps, contacts=contacts, presses=presses, agree=agree / steps,
	            lab_press=lab_press, agent_press=agent_press, hit=hit, p_left=float(acts[:, 0].mean()),
	            p_right=float(acts[:, 1].mean()))


def _variant(job: dict) -> dict:
	import gymnasium as gym
	import torch

	from agents import train_pinball_circuit_cem as T
	from agents.experiments.encoding.encoding import build_encoded_agent
	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	torch.manual_seed(0)
	enc, fobs, rdim = job["variant"]
	t0 = time.time()
	env = gym.wrappers.TimeLimit(PinballEnv(binary_path=job["binary"], headless=True, frame_skip=4), max_episode_steps=3000)
	agent = build_encoded_agent(job["connectome"], rdim, enc)
	agent.calibrate(env.observation_space)
	dec = agent.decoder
	X, L, log = [], [], []
	try:
		for rnd in range(job["rounds"] + 1):
			res = [_rollout(env, agent, T, fobs, s, rnd == 0, X, L) for s in job["seeds"]]
			rec = sum(r["hit"] for r in res) / max(1, sum(r["lab_press"] for r in res))
			log.append(f"round {rnd}: {'reflex' if rnd == 0 else 'agent'} rollouts mean log1p "
			           f"{np.mean(np.log1p([r['score'] for r in res])):.2f}, recall {rec:.2f}, dataset {len(L)}")
			feats = torch.stack(X)
			labels = torch.as_tensor(np.array(L), dtype=torch.float32)
			pos = labels.sum(0)
			pw = ((labels.shape[0] - pos) / pos.clamp_min(1)).clamp(max=200.0)
			opt = torch.optim.Adam(list(dec.proj.parameters()) + list(dec.head.parameters()), lr=1e-2)
			for _ in range(job["epochs"]):
				logits = dec.dist_inputs(dec.proj(feats))[:, :2]
				loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)
				opt.zero_grad(); loss.backward(); opt.step()
			with torch.no_grad():
				logits = dec.dist_inputs(dec.proj(feats))
				for k in range(2):
					rate = float(labels[:, k].mean())
					dec.head.bias[k] -= torch.quantile(logits[:, k], 1.0 - rate)
				dec.head.bias[2] = -10.0  # launch is automatic in PinballEnv; keep the bit off
		w = T.get_flat_params(dec)
		held = [_rollout(env, agent, T, fobs, s, False) for s in job["eval_seeds"]]
	finally:
		env.close()
	name = f"dagger_{enc}_{fobs}_r{rdim}"
	ck = {"mean": w, "sigma": np.full(w.size, 0.05, dtype=np.float32), "champion": w, "champion_fitness": -1e18,
	      "generation": 0, "curriculum_start_gen": 1, "readout_dim": rdim, "agent": "circuit", "objective": "life",
	      "drill_bank": None, "press_cost": 0.00088, "flipper_obs": fobs, "loom_tau0": 2.0, "encoding": enc,
	      "seed_policy": f"DAgger({job['rounds']}) of reflex y>11.5 hold 1 through encoding {enc}",
	      "seed_lives": [job["seeds"][0], job["seeds"][-1]]}
	torch.save(ck, os.path.join(job["ckpt_dir"], name + ".pt"))
	lg = np.log1p([r["score"] for r in held])
	lab, ag, hit = (sum(r[k] for r in held) for k in ("lab_press", "agent_press", "hit"))
	summary = dict(name=name, log1p=float(lg.mean()), se=float(lg.std(ddof=1) / np.sqrt(len(lg))),
	               score=float(np.mean([r["score"] for r in held])), steps=float(np.mean([r["steps"] for r in held])),
	               contacts=float(np.mean([r["contacts"] for r in held])), presses=float(np.mean([r["presses"] for r in held])),
	               p_left=float(np.mean([r["p_left"] for r in held])), p_right=float(np.mean([r["p_right"] for r in held])),
	               agree=float(np.mean([r["agree"] for r in held])), recall=hit / lab if lab else float("nan"),
	               precision=hit / ag if ag else float("nan"), log=log, seconds=time.time() - t0, lives=held)
	return summary


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--connectome", default=os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"))
	ap.add_argument("--variants", default="dspop+ttc:delay1:8,popcode:delay1:8")
	ap.add_argument("--seeds", default="7000:7024")
	ap.add_argument("--eval-seeds", default="7024:7072")
	ap.add_argument("--rounds", type=int, default=3)
	ap.add_argument("--epochs", type=int, default=600)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--ckpt-dir", default=os.path.join(ROOT, "agents/experiments/encoding/seeds"))
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/encoding/dagger_enc.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	elo, ehi = (int(v) for v in args.eval_seeds.split(":"))
	assert not set(range(lo, hi)) & set(range(elo, ehi)), "tuning and held-out lives overlap"
	os.makedirs(args.ckpt_dir, exist_ok=True)
	jobs = []
	for v in args.variants.split(","):
		enc, fobs, rdim = v.split(":")
		jobs.append(dict(variant=(enc, fobs, int(rdim)), binary=args.binary, connectome=args.connectome,
		                 seeds=list(range(lo, hi)), eval_seeds=list(range(elo, ehi)), rounds=args.rounds,
		                 epochs=args.epochs, ckpt_dir=args.ckpt_dir))
	with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
		results = pool.map(_variant, jobs, chunksize=1)
	print(f"held-out lives {elo}..{ehi - 1}")
	print(f"{'variant':<34} {'log1p':>6} {'se':>5} {'score':>8} {'steps':>6} {'cont':>5} {'press':>6} {'P(L)':>5} {'P(R)':>5} "
	      f"{'agree':>6} {'recall':>6} {'prec':>6}")
	for r in results:
		print(f"{r['name']:<34} {r['log1p']:6.2f} {r['se']:5.2f} {r['score']:8.0f} {r['steps']:6.0f} {r['contacts']:5.1f} "
		      f"{r['presses']:6.1f} {r['p_left']:5.3f} {r['p_right']:5.3f} {r['agree']:6.3f} {r['recall']:6.3f} {r['precision']:6.3f}")
		print("  " + "\n  ".join(r["log"]) + f"\n  ({r['seconds']:.0f}s)")
	prev = []
	if os.path.exists(args.out):
		with open(args.out, encoding="utf-8") as fh:
			prev = [p for p in json.load(fh) if p["name"] not in {r["name"] for r in results}]
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(prev + results, fh)


if __name__ == "__main__":
	main()
