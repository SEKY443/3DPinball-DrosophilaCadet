"""Random search of the analog-force parameters (f0, k) per circuit with the budget body params fixed.

force while pressed = clip(f0 + k * (a_side - theta_on), 0.2, 1.0). Objective = mean log1p(score per life)
on the dev lives (same dev seeds as tune_gf_body.py: 7000:7032), accepting only configs with mean
presses/life <= --max-presses. Config 0 is always (f0=1, k=0) = the constant-force reference.

Usage: python agents/experiments/gf_force/tune_gf_force.py --name real [--configs 40] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gf_body as G  # noqa: E402


def _task(t):
	return G.run_life("gf", t[0], t[1])


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--name", required=True, help="real or shuffled_<i>")
	ap.add_argument("--seeds", default="7000:7032")
	ap.add_argument("--configs", type=int, default=40)
	ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
	ap.add_argument("--search-seed", type=int, default=0)
	ap.add_argument("--max-presses", type=float, default=25.0)
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	base = G.load_budget_params(args.name)
	rng = np.random.default_rng(args.search_seed)
	fk = [(1.0, 0.0)] + [(float(rng.uniform(0.2, 1.0)), float(rng.uniform(0, 20))) for _ in range(args.configs - 1)]
	results = []
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		for i, (f0, k) in enumerate(fk):
			c = dict(base, force_mode="rate", f0=f0, k=k)
			res = pool.map(_task, [(c, s) for s in seeds], chunksize=1)
			lg = float(np.mean(np.log1p([r["score"] for r in res])))
			pr = float(np.mean([r["presses"] for r in res]))
			results.append(dict(f0=f0, k=k, log1p=lg, presses=pr))
			print(f"[{args.name}] cfg {i:2d} f0 {f0:.3f} k {k:6.2f} -> log1p {lg:.3f} presses {pr:.1f}", flush=True)
	ok = [r for r in results if r["presses"] <= args.max_presses]
	if not ok:
		raise SystemExit(f"[{args.name}] no config within the press budget")
	best = max(ok, key=lambda r: r["log1p"])
	print(f"[{args.name}] BEST {json.dumps(best)}", flush=True)
	with open(os.path.join(G.HERE, f"best_force_{args.name}.json"), "w", encoding="utf-8") as f:
		json.dump(dict(name=args.name, f0=best["f0"], k=best["k"], dev_log1p=best["log1p"], dev_presses=best["presses"],
		               const_ref=results[0], n_accepted=len(ok), all=results), f, indent=1)


if __name__ == "__main__":
	main()
