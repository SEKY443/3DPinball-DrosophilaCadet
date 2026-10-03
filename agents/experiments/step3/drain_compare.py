#!/usr/bin/env python3
"""Drain detection at the TDrain hit vs. at the plunger re-feed, and per-policy hit profiles.

Plays each life once with the default ("refeed") drain detection, recording per step the score,
presses, flipper contacts and scoring-object hits. The trajectory up to the first TDrain hit
(table_map id 80) is exactly what PinballEnv(drain_detect="tdrain") would produce, so both
episode definitions are evaluated from the same run:
  refeed: life ends at info["drained"] (current training objective)
  tdrain: life ends at the first TDrain hit
Per policy: mean steps, dead steps removed, mean log1p(score) and press-cost fitness
(log1p(score) - 0.00088 * presses) under both, post-drain points share, and a check for TDrain
hits that did NOT end the life (multiball). Also the tdrain-life hit profile: upper-playfield
active share, multiplier bank a_targ7-9 hits/life, a_roll9, ramp.

Usage: python agents/experiments/step3/drain_compare.py --seed0 7024 --episodes 24 --workers 2
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
sys.path.insert(0, os.path.dirname(__file__))

PRESS_COST = 0.00088
DRAIN_ID = 80
ACTIVE_END_Y = 9.0
_ENV = None
_POL: dict = {}
_CFG: dict = {}


def _init(cfg):
	global _ENV
	import atexit
	import signal

	import gymnasium as gym

	from env_python.pinball_env import PinballEnv

	_CFG.update(cfg)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=cfg["binary"], headless=True, frame_skip=4), max_episode_steps=3000)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)


def _life(task):
	from policies import make_policy

	from agents.train_pinball_circuit_cem import load_upper_object_ids
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	spec, seed = task
	if spec not in _POL:
		_POL[spec] = make_policy(spec, _ENV.observation_space)
	pol = _POL[spec]
	upper = load_upper_object_ids(os.path.join(ROOT, "agents/table_map.json"))
	obs, _ = _ENV.reset(seed=seed)
	pol.reset()
	steps, score, presses, prev = 0, 0, 0, np.zeros(2, dtype=np.int64)
	td = None
	td_score = td_presses = 0
	phase = "launch"
	ev = []  # (step, id, base_points, phase)
	done = False
	while not done:
		a = pol.act(obs)
		presses += int(((a[:2] == 1) & (prev == 0)).sum())
		prev = a[:2].copy()
		obs, _r, term, trunc, info = _ENV.step(a)
		steps += 1
		score += info["score_delta"]
		if info["flipper_hit"]:
			phase = "active"
		for (oid, _p, _x, _y), b in zip(info["hit_objects"], info["hit_base_points"]):
			if td is None:
				ev.append((steps, int(oid), int(b), phase))
		if td is None and any(h[0] == DRAIN_ID for h in info["hit_objects"]):
			td, td_score, td_presses = steps, score, presses
		if phase == "active" and obs[OBS_BALL_Y] >= ACTIVE_END_Y and obs[OBS_BALL_VY] > 0:
			phase = "passive"
		done = term or trunc or info["drained"]
	if td is None:  # truncated at max steps without a drain
		td, td_score, td_presses = steps, score, presses
	up_act = sum(b for _s, oid, b, ph in ev if ph == "active" and oid in upper)
	up_all = sum(b for _s, oid, b, ph in ev if oid in upper)
	tot_base = sum(b for _s, _oid, b, _ph in ev)
	return dict(spec=spec, seed=seed, steps=steps, score=int(score), presses=presses,
	            td=td, td_score=int(td_score), td_presses=td_presses, refeed_drained=bool(info["drained"]),
	            ev_counts={str(k): sum(1 for e in ev if e[1] == k) for k in (60, 61, 62, 77, 27, 29, 30)},
	            up_act_base=up_act, up_all_base=up_all, tot_base=tot_base)


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--policies", default="reflex,ckpt:agents/experiments/big_circuit/best_heldout_gen280.pt,"
	                "popcode:agents/experiments/encoding/start_popcode_seed.pt")
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seed0", type=int, default=7024)
	ap.add_argument("--episodes", type=int, default=24)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/step3/drain_compare.json"))
	args = ap.parse_args()
	if 5000 <= args.seed0 < 6000:
		sys.exit("seeds 5000+ are the held-out confirmation set")
	specs = [s if ":" not in s else s.split(":", 1)[0] + ":" + os.path.join(ROOT, s.split(":", 1)[1]) for s in args.policies.split(",")]
	seeds = list(range(args.seed0, args.seed0 + args.episodes))
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(dict(binary=args.binary),)) as pool:
		res = pool.map(_life, [(s, sd) for s in specs for sd in seeds], chunksize=1)
	print(f"{len(seeds)} lives per policy, seeds {seeds[0]}..{seeds[-1]}, {time.time() - t0:.0f}s")
	summ = {}
	for s in specs:
		r = [x for x in res if x["spec"] == s]
		lr = np.log1p([x["score"] for x in r])
		lt = np.log1p([x["td_score"] for x in r])
		fr = lr - PRESS_COST * np.array([x["presses"] for x in r])
		ft = lt - PRESS_COST * np.array([x["td_presses"] for x in r])
		post = sum(x["score"] - x["td_score"] for x in r) / max(1, sum(x["score"] for x in r))
		multi = sum(1 for x in r if x["refeed_drained"] and x["steps"] - x["td"] > 300)
		row = dict(steps_refeed=float(np.mean([x["steps"] for x in r])), steps_tdrain=float(np.mean([x["td"] for x in r])),
		           dead_steps=float(np.mean([x["steps"] - x["td"] for x in r])),
		           log1p_refeed=float(lr.mean()), log1p_tdrain=float(lt.mean()), fit_refeed=float(fr.mean()), fit_tdrain=float(ft.mean()),
		           se_diff=float((lt - lr).std(ddof=1) / np.sqrt(len(r))), post_drain_points_share=post,
		           lives_postdrain_points=sum(1 for x in r if x["score"] > x["td_score"]), suspicious_tdrain=multi,
		           presses=float(np.mean([x["td_presses"] for x in r])),
		           upper_active_share=sum(x["up_act_base"] for x in r) / max(1, sum(x["tot_base"] for x in r)),
		           upper_share=sum(x["up_all_base"] for x in r) / max(1, sum(x["tot_base"] for x in r)),
		           targ7_9_per_life=float(np.mean([x["ev_counts"]["60"] + x["ev_counts"]["61"] + x["ev_counts"]["62"] for x in r])),
		           roll9_per_life=float(np.mean([x["ev_counts"]["77"] for x in r])),
		           ramp_per_life=float(np.mean([x["ev_counts"]["27"] for x in r])),
		           outlane_per_life=float(np.mean([x["ev_counts"]["29"] + x["ev_counts"]["30"] for x in r])))
		summ[s] = row
		name = os.path.basename(s) if ":" in s else s
		print(f"{name:<28} steps {row['steps_refeed']:6.0f} -> {row['steps_tdrain']:6.0f} (dead {row['dead_steps']:4.0f})  "
		      f"log1p {row['log1p_refeed']:.3f} -> {row['log1p_tdrain']:.3f} (diff SE {row['se_diff']:.3f})  "
		      f"fit(press cost) {row['fit_refeed']:.3f} -> {row['fit_tdrain']:.3f}  post-drain pts {post:.1%} "
		      f"(lives {row['lives_postdrain_points']}/{len(r)})  suspicious {multi}  presses {row['presses']:.0f}")
		print(f"{'':<28} base-point shares: upper {row['upper_share']:.1%}, upper&active {row['upper_active_share']:.1%}; per life: "
		      f"targ7-9 {row['targ7_9_per_life']:.2f}  a_roll9 {row['roll9_per_life']:.2f}  ramp {row['ramp_per_life']:.2f}  "
		      f"outlanes {row['outlane_per_life']:.2f}")
	for key in ("log1p", "fit"):
		for mode in ("refeed", "tdrain"):
			order = sorted(specs, key=lambda s: -summ[s][f"{key}_{mode}"])
			print(f"ranking by {key}_{mode}: " + " > ".join(os.path.basename(s) if ":" in s else s for s in order))
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(dict(args=vars(args), summary=summ, lives=res), f)


if __name__ == "__main__":
	main()
