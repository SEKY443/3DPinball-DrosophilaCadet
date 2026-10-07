"""Same protocol and compute budget as pathway_body/tune_gf_retina.py (theta_on bisection to ~21.5 presses/life, accept <= 25,
60 sampled + 5x10 locally refined configs, dev lives 7000-7031), for the TouchGiantFiberBody on one touch circuit.
The search space is the retina one plus the touch parameters gj (JO drive scale) and lam (proximity length).
The retina tuner's calibrate_batch is reused unchanged; only the param sampling / life runner are swapped in.

Names: real, shuffled_1..6 (-> gf_touch_circuit[_shuffled_k].json). Writes best_touch_<name><suffix>.json.
Usage: python tune_gf_touch.py --name real [--n-initial 60 --top 5 --perturb 10 --seeds 7000:7032]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gf_touch_body as TB  # noqa: E402  (also puts pathway_body on sys.path)
import gf_body as G  # noqa: E402
import tune_gf_retina as TR  # noqa: E402


def _task(t):
	return TB.run_life_touch("touch", t[0], t[1])


def sample_drive(rng):
	c = TR.sample_drive(rng, None)
	c["gj"] = float(np.exp(rng.uniform(np.log(0.5), np.log(8.0))))
	c["lam"] = float(np.exp(rng.uniform(np.log(0.2), np.log(3.0))))
	return c


def perturb(c, rng, sigma=0.25):
	n = TR.perturb(c, rng, None, sigma)
	n["gj"] = float(np.clip(c["gj"] * np.exp(rng.normal(0, sigma)), 0.5, 8.0))
	n["lam"] = float(np.clip(c["lam"] * np.exp(rng.normal(0, sigma)), 0.2, 3.0))
	return n


def to_params(c: dict, theta_on: float, circuit: str) -> dict:
	p = {k: v for k, v in c.items() if k not in ("theta_off_ratio", "theta_on")}
	p.update(circuit_json=circuit, rho_target=0.8, fields="sweep", theta_on=float(theta_on), theta_off=c["theta_off_ratio"] * float(theta_on))
	return p


TR._task, TR.to_params = _task, to_params


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--name", required=True)
	ap.add_argument("--seeds", default="7000:7032")
	ap.add_argument("--n-initial", type=int, default=60)
	ap.add_argument("--top", type=int, default=5)
	ap.add_argument("--perturb", type=int, default=10)
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--search-seed", type=int, default=0)
	ap.add_argument("--suffix", default="")
	args = ap.parse_args()
	circuit = os.path.join(HERE, "gf_touch_circuit.json" if args.name == "real" else f"gf_touch_circuit_{args.name}.json")
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	rng = np.random.default_rng(args.search_seed)
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		drives = [sample_drive(rng) for _ in range(args.n_initial)]
		acc = TR.calibrate_batch(pool, drives, circuit, seeds, f"touch {args.name} A")
		top = sorted(acc, key=lambda r: -r[0])[:args.top]
		pdrives = [perturb(t[3], rng) for t in top for _ in range(args.perturb)]
		acc_b = TR.calibrate_batch(pool, pdrives, circuit, seeds, f"touch {args.name} B") if pdrives else []
	allacc = acc + acc_b
	if not allacc:
		print(f"[{args.name}] no accepted config", flush=True)
		sys.exit(1)
	best_lg, best_pr, best, _ = max(allacc, key=lambda r: r[0])
	print(f"[{args.name}] configs sampled {len(drives)} + refined {len(pdrives)}; accepted {len(allacc)}", flush=True)
	print(f"[{args.name}] BEST log1p {best_lg:.3f} presses {best_pr:.1f}: {json.dumps(best)}", flush=True)
	with open(os.path.join(HERE, f"best_touch_{args.name}{args.suffix}.json"), "w", encoding="utf-8") as f:
		json.dump(dict(name=args.name, dev_log1p=best_lg, dev_presses=best_pr, n_sampled=len(drives), n_refined=len(pdrives), n_accepted=len(allacc), params=best,
		               all=[dict(log1p=l, presses=p, stage=s, **{k: v for k, v in q.items() if k != "circuit_json"})
		                    for s, grp in (("A", acc), ("B", acc_b)) for l, p, q, _ in grp]), f, indent=1)


if __name__ == "__main__":
	main()
