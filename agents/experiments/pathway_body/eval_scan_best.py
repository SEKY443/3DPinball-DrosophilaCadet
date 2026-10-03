"""Paired 96-life eval (seeds 77000-77095) of the best scanned cell per circuit plus the best g2=0 cell of real.
Diffs vs real best with t and Wilcoxon p (same helpers as eval_gf_split.py).

Usage: python eval_scan_best.py [--workers N]   (writes eval_scan_best.json)
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
import gf_body as G  # noqa: E402
import gf_split_body as S  # noqa: E402
from plot_scan import best_of  # noqa: E402

sys.path.insert(0, os.path.join(G.ROOT, "agents", "experiments", "encoding"))
import confirm  # noqa: E402


def _task(t):
	name, params, seed = t
	r = S.run_life_split(params, seed)
	r["policy"] = name
	return r


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--names", nargs="+", default=["real", "shuffled_1", "shuffled_2"])
	ap.add_argument("--out", default=os.path.join(G.HERE, "eval_scan_best.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	pol = {}
	for n in args.names:
		p = os.path.join(G.HERE, f"scan_gains_{n}.json")
		if not os.path.exists(p):
			print(f"skip {n}: no scan file")
			continue
		cells = json.load(open(p, encoding="utf-8"))["cells"]
		b = best_of(cells)
		if b:
			pol[f"{n}_best"] = b["params"]
		if n == "real":
			l4 = best_of([c for c in cells if c["i2"] == 0])
			if l4:
				pol["real_lc4only_best"] = l4["params"]
	assert "real_best" in pol, "real scan needed"
	policies = list(pol)
	for n, p in pol.items():
		print(n, json.dumps({a: b for a, b in p.items() if a != "circuit_json"}))
	tasks = [(n, pol[n], s) for s in seeds for n in policies]
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		res = pool.map(_task, tasks, chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["policy"] == n} for n in policies}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	for n in policies:
		rs = [by[n][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		line = (f"{n:<18} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):9.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):7.1f}  c/press {np.sum([r['contacts'] for r in rs]) / max(1, np.sum([r['presses'] for r in rs])):.3f}")
		if n != "real_best":
			d = np.array([np.log1p(by[n][s]["score"]) - np.log1p(by["real_best"][s]["score"]) for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			line += f" | {n} - real_best {d.mean():+.3f} +- {se:.3f} t {d.mean() / se if se else float('nan'):+.2f} Wilcoxon p {confirm.wilcoxon_p(d):.2g}"
		print(line)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), params=pol, lives=res), fh)


if __name__ == "__main__":
	main()
