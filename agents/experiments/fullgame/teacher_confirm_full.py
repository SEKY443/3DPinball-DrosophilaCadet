"""Paired FULL-GAME confirmation of a searched reflex teacher against lead1 on fresh seeds.

Reuses teacher_search_full.run_teacher_game (fresh engine per game, relaunch_at_rest) so the
numbers are comparable with the search log.

Usage: python agents/experiments/fullgame/teacher_confirm_full.py \
         --teacher '{"thr":11.45,"k":1.42,"kv":-0.14,"hold":1,"cool":2}' --seeds 60000:60064 --workers 16
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
import teacher_search_full as TSF  # noqa: E402

LEAD1 = {"thr": 11.5, "k": 1.0, "kv": 0.0, "hold": 1, "cool": 2}


def _task(task) -> dict:
	name, p, seed, max_steps = task
	r = TSF.run_teacher_game(p, seed, max_steps)
	r.update(name=name, seed=seed)
	return r


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--teacher", required=True, help="JSON dict with thr, k, kv, hold, cool")
	ap.add_argument("--binary", default=os.path.join(TSF.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="60000:60064")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=30000)
	ap.add_argument("--out", default=os.path.join(HERE, "teacher_confirm_full.json"))
	args = ap.parse_args()
	new = json.loads(args.teacher)
	new["hold"], new["cool"] = int(new["hold"]), int(new["cool"])
	teachers = {"searched": new, "lead1": LEAD1}
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	with mp.get_context("spawn").Pool(args.workers, initializer=TSF._init, initargs=(args.binary,)) as pool:
		res = pool.map(_task, [(n, p, s, args.max_steps) for s in seeds for n, p in teachers.items()], chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["name"] == n} for n in teachers}
	print(f"paired full games {lo}..{hi - 1} ({len(seeds)}); searched = {json.dumps(new)}")
	for n in teachers:
		rs = [by[n][s] for s in seeds]
		sc = np.array([r["score"] for r in rs])
		lg = np.log1p(sc)
		print(f"{n:<9} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median(sc):,.0f}  "
		      f"mean {sc.mean():,.0f}  completions {np.mean([r['completions'] for r in rs]):.2f}  "
		      f"game over {np.mean([r['game_over'] for r in rs]):.0%}")
	d = np.array([np.log1p(by["searched"][s]["score"]) - np.log1p(by["lead1"][s]["score"]) for s in seeds])
	se = d.std(ddof=1) / math.sqrt(len(d))
	print(f"  searched - lead1 log1p {d.mean():+.3f} +- {se:.3f}  t {d.mean() / se:+.2f}  p {confirm.wilcoxon_p(d):.3g}  "
	      f"wins {int((d > 0).sum())}/{int((d < 0).sum())}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), games=res), fh)


if __name__ == "__main__":
	main()
