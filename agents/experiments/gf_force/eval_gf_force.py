"""Paired evaluation (seeds 77000-77095, one life each): real constant-force, real rate-force,
shuffled_1..6 rate-force. Body params = budget params + tuned (f0, k) from best_force_<name>.json.

Usage: python agents/experiments/gf_force/eval_gf_force.py --seeds 77000:77096
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

sys.path.insert(0, os.path.join(G.ROOT, "agents", "experiments", "encoding"))
import confirm  # noqa: E402


def _task(t):
	name, params, seed = t
	r = G.run_life("gf", params, seed)
	r["policy"] = name
	return r


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
	ap.add_argument("--out", default=os.path.join(G.HERE, "eval_gf_force.json"))
	ap.add_argument("--shuffles", default="1,2,3,4,5,6")
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	shuf = [int(v) for v in args.shuffles.split(",") if v.strip()]
	params = {"real_const": dict(G.load_budget_params("real"), force_mode="const")}
	for name, tag in [("real_rate", "real")] + [(f"shuf{i}_rate", f"shuffled_{i}") for i in shuf]:
		with open(os.path.join(G.HERE, f"best_force_{tag}.json"), encoding="utf-8") as f:
			bf = json.load(f)
		params[name] = dict(G.load_budget_params(tag), force_mode="rate", f0=bf["f0"], k=bf["k"])
	policies = list(params)
	for n in policies:
		print(n, json.dumps({k: v for k, v in params[n].items() if k != "circuit_json"}))
	tasks = [(n, params[n], s) for s in seeds for n in policies]
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
		line = (f"{n:<12} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):9.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):7.1f}  contacts {np.mean([r['contacts'] for r in rs]):6.1f}  "
		        f"c/press {np.sum([r['contacts'] for r in rs]) / max(1, np.sum([r['presses'] for r in rs])):.3f}")
		if n != "real_rate":
			m, se, t, p = diff(n, "real_rate")
			line += f" | {n} - real_rate {m:+.3f} +- {se:.3f} t {t:+.2f} p {p:.2g}"
		print(line)
	m, se, t, p = diff("real_rate", "real_const")
	print(f"KEY real_rate - real_const {m:+.3f} +- {se:.3f} t {t:+.2f} p {p:.2g}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), params=params, lives=res), fh)


if __name__ == "__main__":
	main()
