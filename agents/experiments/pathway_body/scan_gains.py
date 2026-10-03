"""2-D grid scan of the looming gains (g4 = LC4, g2 = LPLC2) of a tuned split giant-fiber body.

Loads best_split_<name>.json and keeps everything fixed except g4, g2 and theta_on. g4 grid: 8 log-spaced values over the
tune_gf_split.py range [0.5, 8]; g2 grid: {0} + 7 log-spaced values over [0.5, 8]. Each cell is calibrated exactly like
tune_gf_split.py (5-step log bisection of theta_on on the first 8 dev lives toward ~21.5 presses, accept only if the
calibrated subset error <= 6.5, one theta_on*1.1 nudge if presses > 25 on the full dev set, reject if still > 25 or if
< 15 presses at the minimum theta_on) and scored on dev lives 7000-7031. Rejected cells are NaN.

Usage: python scan_gains.py --name real [--workers N] [--seeds 7000:7032] [--max-cells K]
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
import tune_gf_split as T  # noqa: E402

N4, N2 = 8, 7
G_LO, G_HI = 0.5, 8.0


def _task(t):
	return T.S.run_life_split(t[0], t[1])


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--name", required=True)
	ap.add_argument("--seeds", default="7000:7032")
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--max-cells", type=int, default=0, help="smoke test: scan only K evenly spaced cells")
	ap.add_argument("--out", default=None)
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	sub = seeds[:T.SUB]
	best = json.load(open(os.path.join(G.HERE, f"best_split_{args.name}.json"), encoding="utf-8"))["params"]
	circuit = best["circuit_json"]
	ratio = best["theta_off"] / best["theta_on"]
	base = {k: best[k] for k in ("R", "v0", "s0", "hold_max", "refractory")}
	base["theta_off_ratio"] = ratio
	g4s = np.exp(np.linspace(np.log(G_LO), np.log(G_HI), N4))
	g2s = np.concatenate([[0.0], np.exp(np.linspace(np.log(G_LO), np.log(G_HI), N2))])
	cells = [(i, j) for i in range(N4) for j in range(len(g2s))]
	if args.max_cells:
		cells = [cells[k] for k in np.unique(np.linspace(0, len(cells) - 1, args.max_cells).astype(int))]
	drives = [dict(base, g4=float(g4s[i]), g2=float(g2s[j])) for i, j in cells]
	n = len(drives)

	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		def run(items, ss):  # items: list of (idx, theta); returns {idx: list of life results}
			res = pool.map(_task, [(T.to_params(drives[i], th, circuit), s) for i, th in items for s in ss], chunksize=1)
			return {i: res[k * len(ss):(k + 1) * len(ss)] for k, (i, _) in enumerate(items)}

		pm = lambda rs: float(np.mean([r["presses"] for r in rs]))
		p_lo = {i: pm(r) for i, r in run([(i, T.THETA_LO) for i in range(n)], sub).items()}
		alive = [i for i, p in p_lo.items() if p >= T.MIN_REACH]
		print(f"[{args.name}] {len(alive)}/{n} cells reach >= {T.MIN_REACH} presses at theta_on {T.THETA_LO}", flush=True)
		lo_, hi_ = {i: np.log(T.THETA_LO) for i in alive}, {i: np.log(T.THETA_HI) for i in alive}
		bst = {i: (abs(p_lo[i] - T.TARGET), T.THETA_LO) for i in alive}
		for _ in range(5):
			mids = {i: float(np.exp(0.5 * (lo_[i] + hi_[i]))) for i in alive}
			pmid = {i: pm(r) for i, r in run([(i, mids[i]) for i in alive], sub).items()}
			for i in alive:
				if abs(pmid[i] - T.TARGET) < bst[i][0]:
					bst[i] = (abs(pmid[i] - T.TARGET), mids[i])
				if pmid[i] > T.TARGET:
					lo_[i] = np.log(mids[i])
				else:
					hi_[i] = np.log(mids[i])
		thetas = {i: bst[i][1] for i in alive if bst[i][0] <= 6.5}
		print(f"[{args.name}] {len(thetas)}/{len(alive)} cells calibrated", flush=True)

		def summarize(items):
			out = {}
			for i, rs in run(items, seeds).items():
				out[i] = dict(log1p=float(np.mean(np.log1p([r["score"] for r in rs]))), presses=pm(rs),
				              contacts_per_press=float(np.sum([r["contacts"] for r in rs]) / max(1, np.sum([r["presses"] for r in rs]))))
			return out

		items = list(thetas.items())
		full = summarize(items)
		th = dict(thetas)
		over = [i for i in full if full[i]["presses"] > T.BAND[1]]
		if over:
			for i in over:
				th[i] = thetas[i] * 1.1
			full.update(summarize([(i, th[i]) for i in over]))

	out_cells = []
	for k, (i, j) in enumerate(cells):
		f = full.get(k)
		ok = f is not None and f["presses"] <= T.BAND[1]
		cell = dict(i4=int(i), i2=int(j), g4=float(g4s[i]), g2=float(g2s[j]), ok=bool(ok))
		if ok:
			cell.update(f, theta_on=th[k], params=T.to_params(drives[k], th[k], circuit))
		else:
			cell.update(log1p=float("nan"), presses=float("nan"), contacts_per_press=float("nan"), theta_on=float("nan"))
		out_cells.append(cell)
		print(f"[{args.name}] g4 {cell['g4']:.2f} g2 {cell['g2']:.2f} -> log1p {cell['log1p']:.3f} presses {cell['presses']:.1f} "
		      f"c/press {cell['contacts_per_press']:.3f} on {cell['theta_on']:.3f}" + ("" if ok else "  NaN"), flush=True)
	out = args.out or os.path.join(G.HERE, f"scan_gains_{args.name}.json")
	with open(out, "w", encoding="utf-8") as fh:
		json.dump(dict(name=args.name, g4_axis=g4s.tolist(), g2_axis=g2s.tolist(), seeds=args.seeds, base=base, cells=out_cells), fh, indent=1)


if __name__ == "__main__":
	main()
