"""Paired confirmation of a CMA-trained popcode champion against its seed on fresh seeds.

Reuses confirm.py's worker (_init/_run) and Wilcoxon helper without editing it. The POLICIES patch
below runs at import time, so spawned workers (which re-import this file as __mp_main__) see it too.

Usage: python agents/experiments/encoding/confirm_trained.py --trained <ckpt> --seeds 6000:6096 --workers 8
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import confirm  # noqa: E402

TRAINED_ENV = "PINBALL_CONFIRM_TRAINED"
_trained = os.environ.get(TRAINED_ENV)
confirm.POLICIES.clear()
confirm.POLICIES["popcode_seed"] = dict(kind="circuit", ckpt="agents/experiments/encoding/start_popcode_seed.pt",
                                        encoding="popcode", fobs="delay1")
if _trained:
	confirm.POLICIES["popcode_trained"] = dict(kind="circuit", ckpt=_trained, encoding="popcode", fobs="delay1")
confirm.POLICIES["reflex_y11.5"] = dict(kind="reflex")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--trained", required=True, help="trained popcode checkpoint (copy it first if a run is writing it)")
	ap.add_argument("--binary", default=os.path.join(confirm.ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="6000:6096")
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--press-cost", type=float, default=0.00088)
	ap.add_argument("--out", default=os.path.join(confirm.ROOT, "agents/experiments/encoding/confirm_6000.json"))
	args = ap.parse_args()
	if os.environ.get(TRAINED_ENV) != args.trained:
		# Re-exec so the env var is set before import in this process and in every spawned worker.
		os.environ[TRAINED_ENV] = args.trained
		os.execv(sys.executable, [sys.executable] + sys.argv)

	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	names = list(confirm.POLICIES)
	tasks = [(p, s) for s in seeds for p in names]
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=confirm._init, initargs=(args.binary, args.max_steps)) as pool:
		res = pool.map(confirm._run, tasks, chunksize=1)
	by = {p: {r["seed"]: r for r in res if r["policy"] == p} for p in names}
	print(f"confirmation lives {lo}..{hi - 1} ({len(seeds)} lives, paired), {time.time() - t0:.0f}s")
	summary = {}
	for p in names:
		rs = [by[p][s] for s in seeds]
		lg = np.log1p([r["score"] for r in rs])
		pr = np.array([r["presses"] for r in rs])
		summary[p] = dict(log1p=float(lg.mean()), se=float(lg.std(ddof=1) / math.sqrt(len(lg))),
		                  median_score=float(np.median([r["score"] for r in rs])), presses=float(pr.mean()),
		                  contacts=float(np.mean([r["contacts"] for r in rs])), steps=float(np.mean([r["steps"] for r in rs])))
		s = summary[p]
		print(f"{p:<16} log1p {s['log1p']:.3f} +- {s['se']:.3f}  median {s['median_score']:8.0f}  presses {s['presses']:6.1f}  "
		      f"contacts {s['contacts']:.1f}  steps {s['steps']:.0f}")
	paired = {}
	for a, b in (("popcode_trained", "popcode_seed"), ("popcode_trained", "reflex_y11.5"), ("popcode_seed", "reflex_y11.5")):
		for label, pc in (("log1p", 0.0), ("with_press_cost", args.press_cost)):
			d = np.array([np.log1p(by[a][s]["score"]) - pc * by[a][s]["presses"]
			              - np.log1p(by[b][s]["score"]) + pc * by[b][s]["presses"] for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			paired[f"{a} - {b} [{label}]"] = dict(mean=float(d.mean()), se=float(se), t=float(d.mean() / se),
			                                      wilcoxon_p=confirm.wilcoxon_p(d), wins=int((d > 0).sum()), losses=int((d < 0).sum()))
	for k, v in paired.items():
		print(f"  {k:<46} {v['mean']:+6.3f} +- {v['se']:.3f}  t {v['t']:+5.2f}  p {v['wilcoxon_p']:.3g}  {v['wins']}/{v['losses']}")
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), summary=summary, paired=paired, lives=res), fh)


if __name__ == "__main__":
	main()
