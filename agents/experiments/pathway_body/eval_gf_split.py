"""Paired evaluation on seeds 77000-77095: real split body, real LC4-only / LPLC2-only ablations, real single-signal
GiantFiberBody (best_params_real_budget.json), lead1 reflex, split bodies on shuffled_1..6.

Usage: python eval_gf_split.py --workers 8   (writes eval_gf_split.json; stdout is the table)
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

sys.path.insert(0, os.path.join(G.ROOT, "agents", "experiments", "encoding"))
import confirm  # noqa: E402


def _task(t):
	name, kind, params, seed = t
	if kind == "split":
		r = S.run_life_split(params, seed)
	else:
		r = G.run_life(kind, params, seed)
	r["policy"] = name
	return r


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--out", default=os.path.join(G.HERE, "eval_gf_split.json"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	load = lambda f: json.load(open(os.path.join(G.HERE, f), encoding="utf-8"))["params"]
	pol = {"split_real": ("split", load("best_split_real.json")),
	       "split_lc4only": ("split", load("best_split_real_lc4only.json")),
	       "split_lplc2only": ("split", load("best_split_real_lplc2only.json")),
	       "single_real": ("gf", load("best_params_real_budget.json")),
	       "lead1": ("lead1", None)}
	for i in range(1, 7):
		pol[f"split_shuf{i}"] = ("split", load(f"best_split_shuffled_{i}.json"))
	policies = list(pol)
	for n, (k, p) in pol.items():
		print(n, json.dumps({a: b for a, b in (p or {}).items() if a != "circuit_json"}))
	tasks = [(n, pol[n][0], pol[n][1], s) for s in seeds for n in policies]
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		res = pool.map(_task, tasks, chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["policy"] == n} for n in policies}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")

	def diff(a, b):
		d = np.array([np.log1p(by[a][s]["score"]) - np.log1p(by[b][s]["score"]) for s in seeds])
		se = d.std(ddof=1) / math.sqrt(len(d))
		return d.mean(), se, (d.mean() / se if se else float("nan")), confirm.wilcoxon_p(d)

	for n in policies:
		rs = [by[n][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		line = (f"{n:<16} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):9.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):7.1f}  contacts {np.mean([r['contacts'] for r in rs]):6.1f}  "
		        f"c/press {np.sum([r['contacts'] for r in rs]) / max(1, np.sum([r['presses'] for r in rs])):.3f}")
		if n != "split_real":
			m, se, t, p = diff(n, "split_real")
			line += f" | {n} - split_real {m:+.3f} +- {se:.3f} t {t:+.2f} p {p:.2g}"
		print(line)
	for a, b in (("split_real", "single_real"), ("split_real", "lead1")):
		m, se, t, p = diff(a, b)
		print(f"KEY {a} - {b}: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), params={n: p for n, (k, p) in pol.items()}, lives=res), fh)


if __name__ == "__main__":
	main()
