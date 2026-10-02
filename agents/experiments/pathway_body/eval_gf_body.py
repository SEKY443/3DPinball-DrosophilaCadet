"""Paired evaluation on fresh seeds 77000-77095: real GF body, 3 degree-shuffled GF bodies (each with
its own tuned params from best_params_<name>.json), lead1 reflex teacher, never-press.

Usage: python agents/experiments/pathway_body/eval_gf_body.py --seeds 77000:77096 --workers 8
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

POLICIES = ["gf_real", "gf_shuf1", "gf_shuf2", "gf_shuf3", "lead1", "never_press"]


def _task(t):
	name, params, seed = t
	kind = "gf" if name.startswith("gf") else ("lead1" if name == "lead1" else "none")
	r = G.run_life(kind, params, seed)
	r["policy"] = name
	return r


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--suffix", default="")
	ap.add_argument("--out", default=None)
	ap.add_argument("--shuffles", default="1,2,3", help="comma-separated shuffled-circuit indices to include")
	args = ap.parse_args()
	shuf_ids = [int(v) for v in args.shuffles.split(",") if v.strip()]
	policies = ["gf_real"] + [f"gf_shuf{i}" for i in shuf_ids] + ["lead1", "never_press"]
	args.out = args.out or os.path.join(G.HERE, f"eval_gf_body{args.suffix}.json")
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	params = {}
	for name, tag in [("gf_real", "real")] + [(f"gf_shuf{i}", f"shuffled_{i}") for i in shuf_ids]:
		with open(os.path.join(G.HERE, f"best_params_{tag}{args.suffix}.json"), encoding="utf-8") as f:
			params[name] = json.load(f)["params"]
		print(name, json.dumps({k: v for k, v in params[name].items() if k != "circuit_json"}))
	tasks = [(n, params.get(n), s) for s in seeds for n in policies]
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		res = pool.map(_task, tasks, chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["policy"] == n} for n in policies}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	for n in policies:
		rs = [by[n][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		line = (f"{n:<12} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median([r['score'] for r in rs]):9.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):7.1f}  contacts {np.mean([r['contacts'] for r in rs]):6.1f}  "
		        f"c/press {np.sum([r['contacts'] for r in rs]) / max(1, np.sum([r['presses'] for r in rs])):.3f}")
		if n != "gf_real":
			d = np.array([np.log1p(by[n][s]["score"]) - np.log1p(by["gf_real"][s]["score"]) for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			line += f" | {n} - gf_real {d.mean():+.3f} +- {se:.3f} t {d.mean() / se if se else float('nan'):+.2f} p {confirm.wilcoxon_p(d):.2g}"
		print(line)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), params=params, lives=res), fh)


if __name__ == "__main__":
	main()
