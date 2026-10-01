"""CMA-ES search over the reflex teacher's parameters with FULL-GAME fitness (all balls until game
over), so the search can favour whatever earns multipliers, ramp awards and missions across balls.

Same 5 teacher parameters and decoding as agents/experiments/timing/teacher_search.py; starts from
lead1. Every game runs on a fresh engine (the IPC reset doesn't restore the ball count after a game
over) with relaunch_at_rest=True so balls after the first are launched. Fitness per game:
log1p(total game score). All candidates of a generation play the same seeds (common random numbers).

Usage: python agents/experiments/fullgame/teacher_search_full.py --generations 25 --popsize 8 --games 24 --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "timing"))
import teacher_search as TS  # noqa: E402  (PARAMS, _decode, DT_STEP, VY_REF)

BANK = ("a_targ7", "a_targ8", "a_targ9")
_BINARY = [None]
_BANK_IDS: set = set()


def _init(binary: str) -> None:
	import signal

	_BINARY[0] = binary
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		_BANK_IDS.update(o["id"] for o in json.load(f)["objects"] if o["name"] in BANK)


def run_teacher_game(p: dict, seed: int, max_steps: int) -> dict:
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y, PinballEnv

	env = PinballEnv(binary_path=_BINARY[0], headless=True, frame_skip=4, relaunch_at_rest=True)
	try:
		obs, _ = env.reset(seed=seed)
		reflex = ReflexLabeler(p["thr"], p["hold"], p["cool"])
		score = completions = balls = 0
		prev_drained = game_over = False
		for _ in range(max_steps):
			view = np.array(obs, dtype=np.float32)
			vy = float(view[OBS_BALL_VY])
			if vy > 0:
				view[OBS_BALL_Y] += (p["k"] + p["kv"] * (vy / TS.VY_REF - 1.0)) * vy * TS.DT_STEP
			a = reflex.label(view)
			a[2] = 0
			reflex.observe(a)
			obs, _r, term, _trunc, info = env.step(a)
			score += info["score_delta"]
			for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
				if oid in _BANK_IDS and base >= 1500:
					completions += 1
			if info["drained"] and not prev_drained:
				balls += 1
				reflex = ReflexLabeler(p["thr"], p["hold"], p["cool"])
			prev_drained = bool(info["drained"])
			if term:
				game_over = True
				break
	finally:
		env.close()
	return dict(score=float(score), completions=completions, balls=balls, game_over=game_over)


def _task(task) -> dict:
	cand, p, seed, max_steps = task
	r = run_teacher_game(p, seed, max_steps)
	r.update(cand=cand, seed=seed)
	return r


def main() -> None:
	import cma

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--generations", type=int, default=25)
	ap.add_argument("--popsize", type=int, default=8)
	ap.add_argument("--games", type=int, default=24)
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--sigma0", type=float, default=0.6)
	ap.add_argument("--cma-seed", type=int, default=1, help="CMA-ES sampling seed (use a different one for an independent replicate run)")
	ap.add_argument("--seed-base", type=int, default=50000)
	ap.add_argument("--max-steps", type=int, default=30000)
	ap.add_argument("--out", default=os.path.join(HERE, "teacher_search_full.json"))
	args = ap.parse_args()
	es = cma.CMAEvolutionStrategy(np.zeros(len(TS.PARAMS)), args.sigma0, {"popsize": args.popsize, "seed": args.cma_seed, "verbose": -9})
	history = []
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		for g in range(args.generations):
			zs = es.ask()
			cands = [TS._decode(z) for z in zs] + [TS._decode(es.mean)]
			seeds = [args.seed_base + 1000 * g + i for i in range(args.games)]
			res = pool.map(_task, [(c, p, s, args.max_steps) for c, p in enumerate(cands) for s in seeds], chunksize=1)
			fit = np.zeros(len(cands))
			stats = []
			for c in range(len(cands)):
				rs = [r for r in res if r["cand"] == c]
				lg = np.log1p([r["score"] for r in rs])
				fit[c] = float(lg.mean())
				stats.append(dict(params=cands[c], log1p=fit[c], median=float(np.median([r["score"] for r in rs])),
				                  completions=float(np.mean([r["completions"] for r in rs])),
				                  game_over=float(np.mean([r["game_over"] for r in rs]))))
			es.tell(zs, list(-fit[:-1]))
			m = stats[-1]
			print(f"gen {g:3d}  mean-teacher log1p {m['log1p']:.3f} median {m['median']:,.0f} comp {m['completions']:.2f} "
			      f"over {m['game_over']:.0%}  params {json.dumps(m['params'])} | best cand {fit[:-1].max():.3f} "
			      f"sigma {es.sigma:.3f}  {time.time() - t0:.0f}s", flush=True)
			history.append(dict(gen=g, seeds=[seeds[0], seeds[-1]], candidates=stats))
			with open(args.out, "w", encoding="utf-8") as fh:
				json.dump(dict(args=vars(args), history=history, final_mean=TS._decode(es.mean)), fh)
	print(f"final teacher (CMA mean): {json.dumps(TS._decode(es.mean))}")


if __name__ == "__main__":
	main()
