"""Text/PNG summary of scan_gains_<name>.json. PNG heatmaps only if matplotlib is importable; always prints ASCII tables.

Usage: python plot_scan.py [--names real shuffled_1 shuffled_2]  (writes scan_table.txt)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gf_body as G  # noqa: E402


def load(name):
	d = json.load(open(os.path.join(G.HERE, f"scan_gains_{name}.json"), encoding="utf-8"))
	return d, {(c["i4"], c["i2"]): c for c in d["cells"]}


def best_of(cells):
	ok = [c for c in cells if c["ok"]]
	return max(ok, key=lambda c: c["log1p"]) if ok else None


def fmt(c):
	return f"g4 {c['g4']:.2f} g2 {c['g2']:.2f} log1p {c['log1p']:.3f} presses {c['presses']:.1f} c/press {c['contacts_per_press']:.3f} theta_on {c['theta_on']:.3f}"


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--names", nargs="+", default=["real", "shuffled_1", "shuffled_2"])
	args = ap.parse_args()
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		plt = None
	lines = []
	for name in args.names:
		p = os.path.join(G.HERE, f"scan_gains_{name}.json")
		if not os.path.exists(p):
			lines.append(f"== {name}: no scan file")
			continue
		d, cm = load(name)
		g4s, g2s = d["g4_axis"], d["g2_axis"]
		M = np.full((len(g4s), len(g2s)), np.nan)
		for (i, j), c in cm.items():
			M[i, j] = c["log1p"]
		b = best_of(d["cells"])
		lines.append(f"== {name}: mean log1p/life on dev lives (rows g4 = LC4, cols g2 = LPLC2; NaN = no theta_on in budget window; * = best)")
		lines.append("g4\\g2  " + " ".join(f"{g:7.2f}" for g in g2s))
		for i, g in enumerate(g4s):
			row = []
			for j in range(len(g2s)):
				v = M[i, j]
				s = "    NaN" if math.isnan(v) else f"{v:7.3f}"
				if b and b["i4"] == i and b["i2"] == j:
					s = s[:-1] + "*"
				row.append(s)
			lines.append(f"{g:5.2f}  " + " ".join(row))
		if b:
			lines.append(f"best cell:        {fmt(b)}")
		l4 = best_of([c for c in d["cells"] if c["i2"] == 0])
		lines.append(f"best g2=0 (LC4-only): {fmt(l4)}" if l4 else "best g2=0 (LC4-only): none")
		r0 = best_of([c for c in d["cells"] if c["i4"] == 0])
		lines.append(f"smallest-g4 row (g4={g4s[0]:.2f}, LPLC2-dominated) best: {fmt(r0)}" if r0 else "smallest-g4 row: all NaN")
		lines.append(f"cells valid: {sum(c['ok'] for c in d['cells'])}/{len(d['cells'])}")
		lines.append("")
		if plt is not None:
			fig, ax = plt.subplots(figsize=(7, 6))
			im = ax.imshow(M, origin="lower", aspect="auto")
			ax.set_xticks(range(len(g2s)), [f"{g:.2f}" for g in g2s])
			ax.set_yticks(range(len(g4s)), [f"{g:.2f}" for g in g4s])
			ax.set_xlabel("g2 (LPLC2)")
			ax.set_ylabel("g4 (LC4)")
			ax.set_title(f"{name}: mean log1p score (dev)")
			if b:
				ax.plot(b["i2"], b["i4"], "r*", ms=16)
			fig.colorbar(im)
			fig.savefig(os.path.join(G.HERE, f"scan_gains_{name}.png"), dpi=120, bbox_inches="tight")
			plt.close(fig)
	if plt is None:
		lines.append("(matplotlib not available: PNG skipped)")
	txt = "\n".join(lines)
	print(txt)
	with open(os.path.join(G.HERE, "scan_table.txt"), "w", encoding="utf-8") as fh:
		fh.write(txt + "\n")


if __name__ == "__main__":
	main()
