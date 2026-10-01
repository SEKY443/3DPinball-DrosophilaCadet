"""Paired confirmation of a searched reflex teacher against lead1 on fresh seeds.

Usage: python agents/experiments/timing/teacher_confirm.py --teacher '{"thr":11.78,"k":1.30,"kv":-0.17,"hold":2,"cool":2}' \
         --seeds 9400:9496 --workers 16
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
import reflex_lead_eval as R  # noqa: E402
import teacher_search as TS  # noqa: E402

LEAD1 = {"thr": 11.5, "k": 1.0, "kv": 0.0, "hold": 1, "cool": 2}


def _task(task) -> dict:
	name, p, seed = task
	r = TS.run_teacher_life(R._ENV, R._BANK_IDS, p, seed)
	r.update(name=name, seed=seed)
	return r


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--teacher", required=True, help="JSON dict with thr, k, kv, hold, cool")
	ap.add_argument("--binary", default=os.path.join(R.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="9400:9496")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--out", default=os.path.join(HERE, "teacher_confirm.json"))
	args = ap.parse_args()
	new = json.loads(args.teacher)
	new["hold"], new["cool"] = int(new["hold"]), int(new["cool"])
	teachers = {"searched": new, "lead1": LEAD1}
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	with mp.get_context("spawn").Pool(args.workers, initializer=R._init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(_task, [(n, p, s) for s in seeds for n, p in teachers.items()], chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["name"] == n} for n in teachers}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)}); searched = {json.dumps(new)}")
	for n in teachers:
		rs = [by[n][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		print(f"{n:<9} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):8.0f}  "
		      + "  ".join(f"{k} {np.mean([r[k] for r in rs]):.2f}" for k in ("hits", "completions", "presses")))
	for k in ("log1p", "hits", "completions"):
		if k == "log1p":
			d = np.array([np.log1p(by["searched"][s]["score"]) - np.log1p(by["lead1"][s]["score"]) for s in seeds])
		else:
			d = np.array([by["searched"][s][k] - by["lead1"][s][k] for s in seeds], dtype=float)
		se = d.std(ddof=1) / math.sqrt(len(d))
		print(f"  searched - lead1 {k:<12} {d.mean():+.3f} +- {se:.3f}  t {d.mean() / se if se else float('nan'):+.2f}  "
		      f"p {confirm.wilcoxon_p(d):.3g}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), lives=res), fh)


if __name__ == "__main__":
	main()
