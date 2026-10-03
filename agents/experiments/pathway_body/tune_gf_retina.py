"""Same protocol as tune_gf_split.py, but for the retinotopic RetinaGiantFiberBody (gf_retina_body.py).

Press-calibrated search of RetinaGiantFiberBody constants on dev lives 7000-7031 for one circuit.

Protocol: sample drive configs (R, g4, g2, v0, s0, hold_max, refractory, theta_off ratio); for each, CALIBRATE theta_on by
log-space bisection (5 steps on the first 8 dev lives) so mean presses/life ~ 21.5 (target band [18, 25]); score the config
on all 32 dev lives (objective mean log1p score/life; accepted only if presses <= 25, one nudge theta_on*1.1 if over).
Configs that cannot reach >= 15 presses at the minimum theta_on are discarded. Then local refinement: the top 5 calibrated
configs each get ~10 Gaussian perturbations (also calibrated). Same compute budget for every circuit.

Names: real, shuffled_1..6 (circuit auto-resolved), real_lc4only (g2=0), real_lplc2only (g4=0).
Usage: python tune_gf_retina.py --name real   (writes best_retina_<name>.json)
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
import gf_retina_body as S  # noqa: E402

THETA_LO, THETA_HI = 0.01, 0.8
TARGET, BAND = 21.5, (18.0, 25.0)
MIN_REACH = 15.0
SUB = 8


def _task(t):
	return S.run_life_retina(t[0], t[1])


def sample_drive(rng, ablate):
	lg = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
	c = dict(R=lg(0.3, 3.0), g4=lg(0.5, 8.0), g2=lg(0.5, 8.0), v0=lg(0.05, 10.0), s0=lg(0.05, 1.5),
	         theta_off_ratio=float(rng.uniform(0.5, 0.9)), hold_max=int(rng.choice([1, 2, 3])), refractory=int(rng.choice([1, 2, 3])))
	return fix_ablate(c, ablate)


def fix_ablate(c, ablate):
	if ablate == "lc4":
		c["g2"] = 0.0
	elif ablate == "lplc2":
		c["g4"] = 0.0
	return c


def perturb(c, rng, ablate, sigma=0.25):
	rng_l = lambda v, lo, hi: float(np.clip(v * np.exp(rng.normal(0, sigma)), lo, hi))
	n = dict(c)
	n["R"], n["v0"], n["s0"] = rng_l(c["R"], 0.3, 3.0), rng_l(c["v0"], 0.05, 10.0), rng_l(c["s0"], 0.05, 1.5)
	n["g4"], n["g2"] = (rng_l(c["g4"], 0.5, 8.0) if c["g4"] > 0 else 0.0), (rng_l(c["g2"], 0.5, 8.0) if c["g2"] > 0 else 0.0)
	n["theta_off_ratio"] = float(np.clip(c["theta_off_ratio"] + rng.normal(0, 0.05), 0.5, 0.9))
	if rng.random() < 0.2:
		n["hold_max"] = int(rng.choice([1, 2, 3]))
	if rng.random() < 0.2:
		n["refractory"] = int(rng.choice([1, 2, 3]))
	return fix_ablate(n, ablate)


FIELDS = "mid"  # set from --fields in main(); worker processes receive full params dicts, so no global needed there


def to_params(c: dict, theta_on: float, circuit: str) -> dict:
	p = {k: v for k, v in c.items() if k not in ("theta_off_ratio", "theta_on")}
	p.update(circuit_json=circuit, rho_target=0.8, retina=True, fields=FIELDS, theta_on=float(theta_on), theta_off=c["theta_off_ratio"] * float(theta_on))
	return p


def calibrate_batch(pool, drives, circuit, seeds, tag):
	"""Returns list of (log1p, presses, params) for accepted configs (presses <= 25) among `drives`."""
	sub = seeds[:SUB]

	def presses(items):  # items: list of (idx, theta); returns {idx: mean presses}
		tasks = [(to_params(drives[i], th, circuit), s) for i, th in items for s in sub]
		res = pool.map(_task, tasks, chunksize=1)
		return {i: float(np.mean([r["presses"] for r in res[k * len(sub):(k + 1) * len(sub)]])) for k, (i, _) in enumerate(items)}

	p_lo = presses([(i, THETA_LO) for i in range(len(drives))])
	alive = [i for i, p in p_lo.items() if p >= MIN_REACH]
	print(f"[{tag}] {len(alive)}/{len(drives)} configs reach >= {MIN_REACH} presses at theta_on {THETA_LO}", flush=True)
	lo = {i: np.log(THETA_LO) for i in alive}
	hi = {i: np.log(THETA_HI) for i in alive}
	best = {i: (abs(p_lo[i] - TARGET), THETA_LO) for i in alive}
	for _ in range(5):
		mids = {i: float(np.exp(0.5 * (lo[i] + hi[i]))) for i in alive}
		pm = presses([(i, mids[i]) for i in alive])
		for i in alive:
			if abs(pm[i] - TARGET) < best[i][0]:
				best[i] = (abs(pm[i] - TARGET), mids[i])
			if pm[i] > TARGET:
				lo[i] = np.log(mids[i])
			else:
				hi[i] = np.log(mids[i])
	thetas = {i: best[i][1] for i in alive if best[i][0] <= 6.5}  # else a cliff: no theta_on lands near the band
	print(f"[{tag}] {len(thetas)}/{len(alive)} calibrated into ~[15, 28] presses on the {SUB}-life subset", flush=True)

	def full(items):
		tasks = [(to_params(drives[i], th, circuit), s) for i, th in items for s in seeds]
		res = pool.map(_task, tasks, chunksize=1)
		out = {}
		for k, (i, th) in enumerate(items):
			rs = res[k * len(seeds):(k + 1) * len(seeds)]
			out[i] = (float(np.mean(np.log1p([r["score"] for r in rs]))), float(np.mean([r["presses"] for r in rs])), th)
		return out

	fl = full(list(thetas.items()))
	over = [i for i, (_, pr, _) in fl.items() if pr > BAND[1]]
	if over:
		fl.update(full([(i, fl[i][2] * 1.1) for i in over]))
	acc = []
	for i, (lgm, pr, th) in fl.items():
		ok = pr <= BAND[1]
		d = drives[i]
		print(f"[{tag}] R {d['R']:.2f} g4 {d['g4']:.2f} g2 {d['g2']:.2f} v0 {d['v0']:.2f} s0 {d['s0']:.2f} hold {d['hold_max']} ref {d['refractory']} "
		      f"on {th:.3f} -> log1p {lgm:.3f} presses {pr:.1f} {'OK' if ok else 'over'}", flush=True)
		if ok:
			acc.append((lgm, pr, to_params(d, th, circuit), d))
	return acc


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--name", required=True)
	ap.add_argument("--circuit", default=None)
	ap.add_argument("--seeds", default="7000:7032")
	ap.add_argument("--n-initial", type=int, default=60)
	ap.add_argument("--top", type=int, default=5)
	ap.add_argument("--perturb", type=int, default=10)
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--search-seed", type=int, default=0)
	ap.add_argument("--suffix", default="", help="appended to the output name, e.g. _sweep")
	ap.add_argument("--fields", choices=["mid", "sweep"], default="mid", help="receptive-field layout (sweep = blind-spot patch for flipper tips)")
	args = ap.parse_args()
	global FIELDS
	FIELDS = args.fields
	ablate = "lc4" if args.name.endswith("_lc4only") else ("lplc2" if args.name.endswith("_lplc2only") else None)
	base = args.name.replace("_lc4only", "").replace("_lplc2only", "")
	circuit = os.path.abspath(args.circuit or os.path.join(G.HERE, "gf_circuit.json" if base == "real" else f"gf_circuit_{base}.json"))
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	rng = np.random.default_rng(args.search_seed)
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		drives = [sample_drive(rng, ablate) for _ in range(args.n_initial)]
		acc = calibrate_batch(pool, drives, circuit, seeds, f"{args.name} A")
		n_eval_a = len(drives)
		top = sorted(acc, key=lambda r: -r[0])[:args.top]
		pdrives = [perturb(t[3], rng, ablate) for t in top for _ in range(args.perturb)]
		acc_b = calibrate_batch(pool, pdrives, circuit, seeds, f"{args.name} B") if pdrives else []
	allacc = acc + acc_b
	if not allacc:
		print(f"[{args.name}] no accepted config", flush=True)
		sys.exit(1)
	best_lg, best_pr, best, _ = max(allacc, key=lambda r: r[0])
	print(f"[{args.name}] configs sampled {n_eval_a} + refined {len(pdrives)} = {n_eval_a + len(pdrives)}; accepted {len(allacc)}", flush=True)
	print(f"[{args.name}] BEST log1p {best_lg:.3f} presses {best_pr:.1f}: {json.dumps(best)}", flush=True)
	with open(os.path.join(G.HERE, f"best_retina_{args.name}{args.suffix}.json"), "w", encoding="utf-8") as f:
		json.dump(dict(name=args.name, dev_log1p=best_lg, dev_presses=best_pr, n_sampled=n_eval_a, n_refined=len(pdrives), n_accepted=len(allacc), params=best,
		               all=[dict(log1p=l, presses=p, stage=s, **{k: v for k, v in q.items() if k != "circuit_json"})
		                    for s, grp in (("A", acc), ("B", acc_b)) for l, p, q, _ in grp]), f, indent=1)


if __name__ == "__main__":
	main()
