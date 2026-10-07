"""Paired evaluation on seeds 77000-77095: touch body on the real touch circuit, the current web body (sweep retina body,
best_retina_real_sweep.json on gf_circuit.json), the lead1 reflex, and touch bodies on shuffled touch circuits 1..6.

Per policy: log1p mean +- SE, median score, presses/life, contacts/press; paired differences vs touch_real (t, Wilcoxon),
pooled shuffle-minus-real, and touch real minus sweep real.

Usage: python eval_gf_touch.py --workers 8 [--suffix ""]   (reads best_touch_<name><suffix>.json; writes eval_gf_touch<suffix>.json)
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
import gf_touch_body as TB  # noqa: E402
import gf_body as G  # noqa: E402

sys.path.insert(0, os.path.join(G.ROOT, "agents", "experiments", "encoding"))
import confirm  # noqa: E402


def _task(t):
	name, kind, params, seed = t
	r = TB.run_life_touch(kind, params, seed)
	r["policy"] = name
	return r


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--suffix", default="")
	ap.add_argument("--out", default=None)
	args = ap.parse_args()
	args.out = args.out or os.path.join(HERE, f"eval_gf_touch{args.suffix}.json")
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	load = lambda f: json.load(open(f, encoding="utf-8"))["params"]
	pol = {"touch_real": ("touch", load(os.path.join(HERE, f"best_touch_real{args.suffix}.json"))),
	       "sweep_real": ("retina", load(os.path.join(G.HERE, "best_retina_real_sweep.json"))),
	       "lead1": ("lead1", None)}
	for i in range(1, 7):
		pol[f"touch_shuf{i}"] = ("touch", load(os.path.join(HERE, f"best_touch_shuffled_{i}{args.suffix}.json")))
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
		line = (f"{n:<12} log1p {lg[n].mean():.3f} +- {lg[n].std(ddof=1) / math.sqrt(len(seeds)):.3f}  median {np.median([r['score'] for r in rs]):9.0f}  "
		        f"presses {np.mean([r['presses'] for r in rs]):7.1f}  contacts/press {np.sum([r['contacts'] for r in rs]) / max(1, np.sum([r['presses'] for r in rs])):.3f}")
		if n != "touch_real":
			m, se, t, p = stat(lg[n] - lg["touch_real"])
			line += f" | vs touch_real {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}"
		print(line)
	shuf = np.mean([lg[f"touch_shuf{i}"] for i in range(1, 7)], axis=0)
	m, se, t, p = stat(shuf - lg["touch_real"])
	print(f"KEY pooled shuffle (mean of 6 per seed) - touch_real: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	pooled = np.concatenate([lg[f"touch_shuf{i}"] - lg["touch_real"] for i in range(1, 7)])
	print(f"KEY pooled shuffle (all {len(pooled)} paired lives) - touch_real: {pooled.mean():+.3f} (naive SE {pooled.std(ddof=1) / math.sqrt(len(pooled)):.3f})")
	m, se, t, p = stat(lg["touch_real"] - lg["sweep_real"])
	print(f"KEY touch_real - sweep_real: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	m, se, t, p = stat(lg["touch_real"] - lg["lead1"])
	print(f"KEY touch_real - lead1: {m:+.3f} +- {se:.3f} t {t:+.2f} Wilcoxon p {p:.2g}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), params={n: p for n, (k, p) in pol.items()}, lives=res), fh)


if __name__ == "__main__":
	main()
