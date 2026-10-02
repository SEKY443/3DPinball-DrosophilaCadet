"""Random/grid search of the few GiantFiberBody constants on dev lives 7000-7031 for one circuit.

Searches gain, theta_on, tau0 (theta_off = 0.7 * theta_on, hold_max in {1,2,3}); objective = mean
log1p(score per life). Then logs DNp01 activity statistics (ball far vs y > 10) for the best config.

Usage: python agents/experiments/pathway_body/tune_gf_body.py --name real --circuit agents/experiments/pathway_body/gf_circuit.json
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
	return G.run_life("gf", t[0], t[1], stats=t[2])


def sample_configs(n: int, seed: int, theta_range=None) -> list[dict]:
	rng = np.random.default_rng(seed)
	cfgs = []
	for _ in range(n):
		cfgs.append(dict(gain=float(np.exp(rng.uniform(np.log(0.5), np.log(8.0)))),
		                 theta_on=float(rng.uniform(*theta_range) if theta_range else (rng.uniform(0.02, 0.7) if rng.random() < 0.5 else rng.uniform(0.5, 0.66))),
		                 tau0=float(np.exp(rng.uniform(np.log(0.7), np.log(6.0)))),
		                 hold_max=int(rng.choice([1, 2, 3]))))
	return cfgs


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--name", required=True)
	ap.add_argument("--circuit", required=True)
	ap.add_argument("--seeds", default="7000:7032")
	ap.add_argument("--configs", type=int, default=60)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--refractory", type=int, default=2)
	ap.add_argument("--search-seed", type=int, default=0)
	ap.add_argument("--rho", type=float, default=None, help="sub-critical dynamics: gain*rho target")
	ap.add_argument("--theta-range", default=None, help="lo:hi for theta_on")
	ap.add_argument("--suffix", default="")
	ap.add_argument("--max-presses", type=float, default=None, help="accept only configs with mean dev presses/life <= this")
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	circuit = os.path.abspath(args.circuit)
	tr = tuple(float(v) for v in args.theta_range.split(":")) if args.theta_range else None
	cfgs = [dict(c, circuit_json=circuit, refractory=args.refractory, rho_target=args.rho) for c in sample_configs(args.configs, args.search_seed, tr)]
	for c in cfgs:
		c["theta_off"] = 0.7 * c["theta_on"]
	results = []  # (log1p, cfg, presses)
	widened = False

	def evaluate(pool, cs, tag):
		for i, c in enumerate(cs):
			res = pool.map(_task, [(c, s, False) for s in seeds], chunksize=1)
			lg = float(np.mean(np.log1p([r["score"] for r in res])))
			pr = float(np.mean([r["presses"] for r in res]))
			results.append((lg, c, pr))
			print(f"[{args.name}] {tag} cfg {i:2d} gain {c['gain']:.2f} on {c['theta_on']:.3f} tau0 {c['tau0']:.2f} hold {c['hold_max']} "
			      f"-> log1p {lg:.3f} presses {pr:.1f}", flush=True)

	def accepted():
		return [r for r in results if args.max_presses is None or r[2] <= args.max_presses]

	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		evaluate(pool, cfgs, "a")
		if not accepted():
			widened = True
			print(f"[{args.name}] no config with presses <= {args.max_presses}; resampling once with theta_on 0.4:0.9", flush=True)
			cs = [dict(c, circuit_json=circuit, refractory=args.refractory, rho_target=args.rho) for c in sample_configs(args.configs, args.search_seed + 1, (0.4, 0.9))]
			for c in cs:
				c["theta_off"] = 0.7 * c["theta_on"]
			evaluate(pool, cs, "b")
		best_lg, best, best_pr = max(accepted(), key=lambda r: r[0])
		res = pool.map(_task, [(best, s, True) for s in seeds], chunksize=1)
	far = np.array([x for r in res for x in r["far"]]).reshape(-1, 2)
	zone = np.array([x for r in res for x in r["zone"]]).reshape(-1, 2)
	st = {}
	for nm, arr in (("far", far), ("zone_y>10", zone)):
		st[nm] = dict(n=int(len(arr)), mean_L=float(arr[:, 0].mean()), mean_R=float(arr[:, 1].mean()),
		              max_L=float(arr[:, 0].max()), max_R=float(arr[:, 1].max()),
		              frac_above_theta_on=float((arr.max(axis=1) > best["theta_on"]).mean()))
	print(f"[{args.name}] BEST log1p {best_lg:.3f} presses {best_pr:.1f} widened {widened} accepted {len(accepted())}: {json.dumps(best)}")
	print(f"[{args.name}] GF activity stats: {json.dumps(st)}")
	with open(os.path.join(G.HERE, f"best_params_{args.name}{args.suffix}.json"), "w", encoding="utf-8") as f:
		json.dump(dict(name=args.name, dev_log1p=best_lg, dev_presses=best_pr, widened=widened, n_accepted=len(accepted()), params=best, gf_stats=st,
		               all=[dict(log1p=l, presses=p, **{k: v for k, v in c.items() if k != "circuit_json"}) for l, c, p in results]), f, indent=1)


if __name__ == "__main__":
	main()
