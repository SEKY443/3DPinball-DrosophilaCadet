"""Paired check of whether a target-trained popcode champion hits the multiplier bank (a_targ7-9) more
often than the popcode seed. Counts per life: all bank-target hits, hits inside an active window
(after a real flipper contact, until the ball falls back - same rule as the trainer's bonus), and bank
completions (a hit paying the 1500-point completion award). Reuses confirm.py's env/agent setup.

Usage: python agents/experiments/encoding/target_eval.py --trained <ckpt> --seeds 7100:7148 --workers 4
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm  # noqa: E402

TRAINED_ENV = "PINBALL_TARGET_EVAL_TRAINED"
BANK = ("a_targ7", "a_targ8", "a_targ9")
BANK_COMPLETION_BASE = 1500
_trained = os.environ.get(TRAINED_ENV)
confirm.POLICIES.clear()
confirm.POLICIES["seed"] = dict(kind="circuit", ckpt="agents/experiments/encoding/start_popcode_seed.pt",
                                encoding="popcode", fobs="delay1")
if _trained:
	confirm.POLICIES["trained"] = dict(kind="circuit", ckpt=_trained, encoding="popcode", fobs="delay1")


def _bank_ids() -> frozenset:
	with open(os.path.join(confirm.ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		by_name = {o["name"]: o["id"] for o in json.load(f)["objects"]}
	return frozenset(by_name[n] for n in BANK)


def _run(task) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	name, seed = task
	spec = confirm.POLICIES[name]
	bank = _bank_ids()
	env = confirm._ENV
	obs, info = env.reset(seed=seed)
	agent = confirm._agent(name)
	h = agent.initial_state(1)
	filt = T.FlipperObsFilter(spec["fobs"])
	score = hits = active_hits = completions = contacts = 0
	active = done = False
	with torch.no_grad():
		while not done:
			a, h = agent.act(torch.as_tensor(np.asarray(filt(obs), dtype=np.float32)).unsqueeze(0), h, greedy=True)
			obs, _r, term, trunc, info = env.step(a[0])
			score += info["score_delta"]
			if info["flipper_hit"]:
				contacts += 1
				active = True
			for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
				if oid in bank:
					hits += 1
					active_hits += int(active)
					completions += int(base >= BANK_COMPLETION_BASE)
			if active and obs[OBS_BALL_Y] >= T.ACTIVE_END_Y and obs[OBS_BALL_VY] > 0:
				active = False
			done = term or trunc or bool(info["drained"])
	return dict(policy=name, seed=seed, score=float(score), hits=hits, active_hits=active_hits,
	            completions=completions, contacts=contacts)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--trained", required=True)
	ap.add_argument("--binary", default=os.path.join(confirm.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="7100:7148")
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--workers", type=int, default=4)
	ap.add_argument("--out", default=os.path.join(confirm.ROOT, "agents/experiments/target_run/target_eval.json"))
	args = ap.parse_args()
	if os.environ.get(TRAINED_ENV) != args.trained:
		os.environ[TRAINED_ENV] = args.trained
		os.execv(sys.executable, [sys.executable] + sys.argv)
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	tasks = [(p, s) for s in seeds for p in ("seed", "trained")]
	with mp.get_context("spawn").Pool(args.workers, initializer=confirm._init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(_run, tasks, chunksize=1)
	by = {p: {r["seed"]: r for r in res if r["policy"] == p} for p in ("seed", "trained")}
	print(f"target eval, lives {lo}..{hi - 1} ({len(seeds)} paired)")
	for p in ("seed", "trained"):
		rs = [by[p][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		print(f"{p:<8} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  "
		      + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.2f}" for k in ("hits", "active_hits", "completions", "contacts")))
	for k in ("hits", "active_hits", "completions"):
		d = np.array([by["trained"][s][k] - by["seed"][s][k] for s in seeds], dtype=float)
		se = d.std(ddof=1) / math.sqrt(len(d))
		print(f"  trained - seed {k:<12} {d.mean():+.2f} +- {se:.2f}  t {d.mean() / se if se else float('nan'):+.2f}  "
		      f"p {confirm.wilcoxon_p(d):.3g}")
	d = np.array([np.log1p(by["trained"][s]["score"]) - np.log1p(by["seed"][s]["score"]) for s in seeds])
	print(f"  trained - seed log1p        {d.mean():+.3f} +- {d.std(ddof=1) / math.sqrt(len(d)):.3f}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
