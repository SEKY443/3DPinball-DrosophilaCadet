"""Summarize online chains: learning curves, final mu per chain, final 96-life eval with paired diffs vs real.

Usage: python summarize_online.py DIR [--png DIR/online_summary.png]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "pathway_body"))
import gf_body as G  # noqa: E402

sys.path.insert(0, os.path.join(G.ROOT, "agents", "experiments", "encoding"))
import confirm  # noqa: E402

BLOCK = 50
PARAMS = ["R", "g4", "g2", "v0", "s0"]


def load(d: str) -> dict:
	chains: dict = {}
	for f in sorted(glob.glob(os.path.join(d, "*_c*.json"))):
		with open(f, encoding="utf-8") as fh:
			s = json.load(fh)
		chains.setdefault(s["circuit"], []).append(s)
	return chains


def curve(s: dict, key) -> np.ndarray:
	"""Per-block mean of key(history entry); incomplete trailing blocks are kept."""
	v = np.array([key(h) for h in s["history"]], dtype=float)
	return np.array([v[i:i + BLOCK].mean() for i in range(0, len(v), BLOCK)]) if len(v) else np.array([])


def avg_curves(ss: list, key):
	cs = [curve(s, key) for s in ss]
	n = min((len(c) for c in cs), default=0)
	a = np.array([c[:n] for c in cs]) if n else np.zeros((0, 0))
	return a.mean(0) if n else a, (a.std(0, ddof=1) / math.sqrt(len(cs)) if n and len(cs) > 1 else np.zeros(n))


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("dir")
	ap.add_argument("--png", default=None)
	args = ap.parse_args()
	chains = load(args.dir)
	order = sorted(chains, key=lambda c: (c != "real", c))
	rmean = lambda h: 0.5 * (h["r_plus"] + h["r_minus"])
	pmean = lambda h: 0.5 * (h["presses_plus"] + h["presses_minus"])
	curves = {}
	for c in order:
		ss = chains[c]
		m, se = avg_curves(ss, rmean)
		pm, _ = avg_curves(ss, pmean)
		tm, _ = avg_curves(ss, lambda h: h["theta_on"])
		curves[c] = (m, se)
		print(f"== {c}: {len(ss)} chains, updates done {[s['done_updates'] for s in ss]}")
		print(f"   mean reward per {BLOCK} updates: " + " ".join(f"{v:.2f}" for v in m))
		print(f"   mean presses per life:          " + " ".join(f"{v:.1f}" for v in pm))
		print(f"   theta_on:                       " + " ".join(f"{v:.3f}" for v in tm))
		for s in ss:
			print(f"   chain {s['chain']} final mu (exp): " + " ".join(f"{p} {math.exp(v):.3f}" for p, v in zip(PARAMS, s["mu"])) + f" theta_on {s['theta_on']:.3f}")
	# final eval
	ev = {}
	for c in order:
		per = [s for s in chains[c] if s.get("eval")]
		if not per:
			continue
		seeds = [e["seed"] for e in per[0]["eval"]]
		lg = np.array([[math.log1p(e["score"]) for e in s["eval"]] for s in per])  # chains x seeds
		pr = np.array([[e["presses"] for e in s["eval"]] for s in per])
		ev[c] = (seeds, lg)
		print(f"EVAL {c:<11} chains {len(per)}  log1p {lg.mean():.3f} (per chain {' '.join(f'{v:.3f}' for v in lg.mean(1))})  presses {pr.mean():.1f}")
	if "real" in ev:
		for c in order:
			if c == "real" or c not in ev or ev[c][0] != ev["real"][0]:
				continue
			d = ev[c][1].mean(0) - ev["real"][1].mean(0)  # per-seed, chain-averaged; paired by seed
			se = d.std(ddof=1) / math.sqrt(len(d))
			print(f"KEY {c} - real: {d.mean():+.3f} +- {se:.3f} t {d.mean() / se if se else float('nan'):+.2f} Wilcoxon p {confirm.wilcoxon_p(d):.2g} ({len(d)} paired lives)")
	png = args.png or os.path.join(args.dir, "online_summary.png")
	try:
		import matplotlib

		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(1, 2, figsize=(11, 4))
		for c in order:
			m, se = curves[c]
			x = (np.arange(len(m)) + 1) * BLOCK
			ax[0].plot(x, m, label=c)
			ax[0].fill_between(x, m - se, m + se, alpha=0.2)
		ax[0].set_xlabel("update")
		ax[0].set_ylabel(f"mean reward ({BLOCK}-update blocks)")
		ax[0].legend()
		names = [c for c in order if c in ev]
		ax[1].bar(names, [ev[c][1].mean() for c in names])
		ax[1].set_ylabel("final eval mean log1p score (96 lives)")
		fig.tight_layout()
		fig.savefig(png, dpi=120)
		print(f"wrote {png}")
	except ImportError:
		print("matplotlib not available; skipped PNG")


if __name__ == "__main__":
	main()
