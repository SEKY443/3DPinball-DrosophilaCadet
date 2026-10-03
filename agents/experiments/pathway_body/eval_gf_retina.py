"""Paired evaluation on seeds 77000-77095: retinotopic body on the real circuit, the current web body (real_best from
eval_scan_best.json, single-point split body), the lead1 reflex, and retinotopic bodies on shuffled_1..6.

Reports per policy log1p mean +- SE, median score, presses/life, contacts/press and % unseen drains (no flipper press and
GF below theta_on in the last 20 steps before a drain while the ball was in flipper reach; see blind_spot.py), then paired
differences (t, Wilcoxon p), pooled shuffle-minus-real and real retina minus real current.

Usage: python eval_gf_retina.py --workers 8   (writes eval_gf_retina.json; stdout is the table)
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
import blind_spot as B  # noqa: E402
import gf_body as G  # noqa: E402

sys.path.insert(0, os.path.join(G.ROOT, "agents", "experiments", "encoding"))
import confirm  # noqa: E402


def _task(t):
	name, kind, params, seed = t
	r = B.run_life_traj(kind, params, seed)
	r["policy"] = name
	return r


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--suffix", default="", help="best_retina_<name><suffix>.json to evaluate, e.g. _sweep; adds the mid-layout real body as a reference")
	ap.add_argument("--out", default=None)
	args = ap.parse_args()
	args.out = args.out or os.path.join(G.HERE, f"eval_gf_retina{args.suffix}.json")
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	load = lambda f: json.load(open(os.path.join(G.HERE, f), encoding="utf-8"))["params"]
	pol = {"retina_real": ("retina", load(f"best_retina_real{args.suffix}.json")),
	       "split_current": ("split", json.load(open(os.path.join(G.HERE, "eval_scan_best.json"), encoding="utf-8"))["params"]["real_best"]),
	       "lead1": ("lead1", None)}
	for i in range(1, 7):
		pol[f"retina_shuf{i}"] = ("retina", load(f"best_retina_shuffled_{i}{args.suffix}.json"))
	if args.suffix:
		pol["retina_mid"] = ("retina", load("best_retina_real.json"))
	policies = list(pol)
	for n, (k, p) in pol.items():
		print(n, json.dumps({a: b for a, b in (p or {}).items() if a != "circuit_json"}))
	tasks = [(n, pol[n][0], pol[n][1], s) for s in seeds for n in policies]
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		res = pool.map(_task, tasks, chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["policy"] == n} for n in policies}
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	lg = {n: np.array([math.log1p(by[n][s]["score"]) for s in seeds]) for n in policies}

	def stat(d):
		se = d.std(ddof=1) / math.sqrt(len(d))
		return d.mean(), se, (d.mean() / se if se else float("nan")), confirm.wilcoxon_p(d)

	for n in policies:
		rs = [by[n][s] for s in seeds]
		dr = [r for r in rs if r["drained"]]
		un = sum(r["unseen"] for r in dr)
		line = (f"{n:<14} log1p {lg[n].mean():.3f} +- {lg[n].std(ddof=1) / math.sqrt(len(seeds)):.3f}  median {np.median([r['score'] for r in rs]):9.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):7.1f}  contacts/press {np.sum([r['contacts'] for r in rs]) / max(1, np.sum([r['presses'] for r in rs])):.3f}  "
		        f"unseen drains {100.0 * un / max(1, len(dr)):5.1f}% ({un}/{len(dr)})")
		if n != "retina_real":
			m, se, t, p = stat(lg[n] - lg["retina_real"])
			line += f" | vs retina_real {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}"
		print(line)
	shuf = np.mean([lg[f"retina_shuf{i}"] for i in range(1, 7)], axis=0)
	m, se, t, p = stat(shuf - lg["retina_real"])
	print(f"KEY pooled shuffle (mean of 6 per seed) - retina_real: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	pooled = np.concatenate([lg[f"retina_shuf{i}"] - lg["retina_real"] for i in range(1, 7)])
	print(f"KEY pooled shuffle (all {len(pooled)} paired lives) - retina_real: {pooled.mean():+.3f} (naive SE {pooled.std(ddof=1) / math.sqrt(len(pooled)):.3f})")
	m, se, t, p = stat(lg["retina_real"] - lg["split_current"])
	print(f"KEY retina_real - split_current: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	m, se, t, p = stat(lg["retina_real"] - lg["lead1"])
	print(f"KEY retina_real - lead1: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), params={n: p for n, (k, p) in pol.items()}, lives=res), fh)


if __name__ == "__main__":
	main()
