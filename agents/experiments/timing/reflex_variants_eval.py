"""Paired comparison of reflex teacher variants with a per-side timing shift.

Each variant is (k_left, k_right): when the ball is falling, the reflex sees y + k * vy * DT_STEP for
the side the ball is on (k > 0 presses earlier, k < 0 later, 0 = plain reflex). The timing oracle
(timing_oracle.npz) suggested the right flipper wants to press earlier and the left one later for
multiplier-bank hits. Use a TUNING seed range to pick a variant, then confirm the pick on fresh seeds
with --variants restricted to the pick and the baseline.

Usage: python agents/experiments/timing/reflex_variants_eval.py --seeds 21000:21096 --workers 12
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "encoding"))
import reflex_lead_eval as R  # noqa: E402  (reuses its env init, ROOT and bank ids)

DT_STEP = 0.0347
VARIANTS = {
	"plain": (0.0, 0.0),
	"lead1": (1.0, 1.0),
	"L-1_R+1": (-1.0, 1.0),
	"L0_R+1": (0.0, 1.0),
	"L0_R+2": (0.0, 2.0),
	"lead2": (2.0, 2.0),
}


def _run(task) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y, OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

	name, (k_left, k_right), seed = task
	env = R._ENV
	obs, _ = env.reset(seed=seed)
	reflex = ReflexLabeler(11.5, 1)
	score = presses = contacts = hits = completions = 0
	prev_l, prev_r = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
	done = False
	while not done:
		view = np.array(obs, dtype=np.float32)
		if view[OBS_BALL_VY] > 0:
			k = k_left if view[OBS_BALL_X] > 0 else k_right  # x > 0 is the LEFT flipper's side (mirrored table)
			view[OBS_BALL_Y] = view[OBS_BALL_Y] + k * view[OBS_BALL_VY] * DT_STEP
		a = reflex.label(view)
		a[2] = 0
		reflex.observe(a)
		obs, _r, term, trunc, info = env.step(a)
		score += info["score_delta"]
		contacts += int(info["flipper_hit"])
		for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
			if oid in R._BANK_IDS:
				hits += 1
				completions += int(base >= 1500)
		lu, ru = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
		presses += int(lu and not prev_l) + int(ru and not prev_r)
		prev_l, prev_r = lu, ru
		done = term or trunc or bool(info["drained"])
	return dict(policy=name, seed=seed, score=float(score), presses=presses, contacts=contacts, hits=hits,
	            completions=completions)


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(R.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="21000:21096")
	ap.add_argument("--workers", type=int, default=12)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated names from VARIANTS")
	ap.add_argument("--baseline", default="lead1")
	ap.add_argument("--out", default=os.path.join(HERE, "reflex_variants.json"))
	args = ap.parse_args()
	names = args.variants.split(",")
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	with mp.get_context("spawn").Pool(args.workers, initializer=R._init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(_run, [(n, VARIANTS[n], s) for s in seeds for n in names], chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["policy"] == n} for n in names}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)}), baseline {args.baseline}")
	for n in names:
		rs = [by[n][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		line = (f"{n:<9} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  "
		        + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.2f}" for k in ("hits", "completions", "contacts", "presses")))
		if n != args.baseline:
			for k in ("hits", "completions", "log1p"):
				if k == "log1p":
					d = np.array([np.log1p(by[n][s]["score"]) - np.log1p(by[args.baseline][s]["score"]) for s in seeds])
				else:
					d = np.array([by[n][s][k] - by[args.baseline][s][k] for s in seeds], dtype=float)
				se = d.std(ddof=1) / math.sqrt(len(d))
				line += f" | d{k} {d.mean():+.2f} (t {d.mean() / se if se else float('nan'):+.1f}, p {confirm.wilcoxon_p(d):.2g})"
		print(line)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
