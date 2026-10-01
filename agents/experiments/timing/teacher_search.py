"""CMA-ES search over the reflex TEACHER's few parameters (not the circuit's 667).

Lesson so far: CMA on the circuit's readout never moved behaviour, but a better teacher + DAgger did
(popcode seed, one-step-lead reflex). The teacher is a 5-parameter hand rule, a dimension where CMA
works well. Each generation evaluates every candidate on the SAME fresh set of lives (common random
numbers), so candidates are compared pairwise; seeds change every generation to avoid overfitting.

Teacher parameters (search space is a normalised vector z; see _decode):
  thr     trigger height: press when the (shifted) ball y > thr          default 11.5
  k       timing lead in decision steps (y + k * vy * DT_STEP when falling) default 1.0
  kv      extra lead for fast balls: k_eff = k + kv * (vy / VY_REF - 1)    default 0.0
  hold    decision steps the flipper is held                             default 1
  cool    released steps before the same side may fire again             default 2
Fitness per life: log1p(score) + BANK_BONUS * bank completions.

Usage: python agents/experiments/timing/teacher_search.py --generations 20 --popsize 8 --lives 32 --workers 16
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
sys.path.insert(0, HERE)
import reflex_lead_eval as R  # noqa: E402  (env init, ROOT, bank ids)

DT_STEP = 0.0347
VY_REF = 5.0
BANK_BONUS = 0.05
# name, default, scale (z = (value - default) / scale), lower, upper
PARAMS = (("thr", 11.5, 0.5, 10.0, 13.0), ("k", 1.0, 0.75, -1.0, 4.0), ("kv", 0.0, 0.75, -2.0, 3.0),
          ("hold", 1.0, 0.75, 1.0, 4.0), ("cool", 2.0, 0.75, 1.0, 5.0))


def _decode(z) -> dict:
	out = {}
	for (name, default, scale, lo, hi), v in zip(PARAMS, z):
		out[name] = float(np.clip(default + scale * v, lo, hi))
	out["hold"] = int(round(out["hold"]))
	out["cool"] = int(round(out["cool"]))
	return out


def run_teacher_life(env, bank_ids, p: dict, seed: int) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	obs, _ = env.reset(seed=seed)
	reflex = ReflexLabeler(p["thr"], p["hold"], p["cool"])
	score = completions = hits = presses = 0
	prev = np.zeros(2, dtype=bool)
	done = False
	while not done:
		view = np.array(obs, dtype=np.float32)
		vy = float(view[OBS_BALL_VY])
		if vy > 0:
			k_eff = p["k"] + p["kv"] * (vy / VY_REF - 1.0)
			view[OBS_BALL_Y] = view[OBS_BALL_Y] + k_eff * vy * DT_STEP
		a = reflex.label(view)
		a[2] = 0
		reflex.observe(a)
		pressed = a[:2].astype(bool)
		presses += int((pressed & ~prev).sum())
		prev = pressed
		obs, _r, term, trunc, info = env.step(a)
		score += info["score_delta"]
		for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
			if oid in bank_ids:
				hits += 1
				completions += int(base >= 1500)
		done = term or trunc or bool(info["drained"])
	return dict(score=float(score), completions=completions, hits=hits, presses=presses)


def _task(task) -> dict:
	cand, p, seed = task
	r = run_teacher_life(R._ENV, R._BANK_IDS, p, seed)
	r.update(cand=cand, seed=seed)
	return r


def main() -> None:
	import cma

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(R.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--generations", type=int, default=20)
	ap.add_argument("--popsize", type=int, default=8)
	ap.add_argument("--lives", type=int, default=32)
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--sigma0", type=float, default=0.6)
	ap.add_argument("--seed-base", type=int, default=30000, help="generation g uses seeds seed_base + 1000*g ...")
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(HERE, "teacher_search.json"))
	args = ap.parse_args()
	es = cma.CMAEvolutionStrategy(np.zeros(len(PARAMS)), args.sigma0,
	                              {"popsize": args.popsize, "seed": 1, "verbose": -9})
	history = []
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=R._init, initargs=(args.binary, args.max_steps)) as pool:
		for g in range(args.generations):
			zs = es.ask()
			# candidate index len(zs) = the current CMA mean, evaluated on the same lives for tracking
			cands = [_decode(z) for z in zs] + [_decode(es.mean)]
			seeds = [args.seed_base + 1000 * g + i for i in range(args.lives)]
			res = pool.map(_task, [(c, p, s) for c, p in enumerate(cands) for s in seeds], chunksize=1)
			fit = np.zeros(len(cands))
			stats = []
			for c in range(len(cands)):
				rs = [r for r in res if r["cand"] == c]
				lg = np.log1p([r["score"] for r in rs])
				comp = np.array([r["completions"] for r in rs])
				fit[c] = float(np.mean(lg + BANK_BONUS * comp))
				stats.append(dict(params=cands[c], fitness=fit[c], log1p=float(lg.mean()), completions=float(comp.mean()),
				                  hits=float(np.mean([r["hits"] for r in rs])), presses=float(np.mean([r["presses"] for r in rs]))))
			es.tell(zs, list(-fit[:-1]))
			m = stats[-1]
			best = int(np.argmax(fit[:-1]))
			print(f"gen {g:3d}  mean-teacher fit {m['fitness']:.3f} log1p {m['log1p']:.3f} comp {m['completions']:.2f} "
			      f"hits {m['hits']:.2f} presses {m['presses']:.1f}  params {json.dumps(m['params'])}  "
			      f"| best cand fit {fit[best]:.3f}  sigma {es.sigma:.3f}  {time.time() - t0:.0f}s", flush=True)
			history.append(dict(gen=g, seeds=[seeds[0], seeds[-1]], candidates=stats))
			with open(args.out, "w", encoding="utf-8") as fh:
				json.dump(dict(args=vars(args), history=history, final_mean=_decode(es.mean)), fh)
	print(f"final teacher (CMA mean): {json.dumps(_decode(es.mean))}")


if __name__ == "__main__":
	main()
