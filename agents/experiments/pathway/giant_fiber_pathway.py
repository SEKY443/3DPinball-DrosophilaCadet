"""Looming -> giant fiber pathway in the full MaleCNS v1.0 connectome (candidate flipper "body").

The fly's classic escape reflex: looming visual neurons (LC4, LPLC2) drive the giant fiber
descending neuron (DNp01), which triggers the jump. For pinball the analogue is "ball approaching
the flipper -> press". This script measures, from the real connectome:
  1. direct synapse counts LC4/LPLC2 (left / right eye) -> DNp01 (left / right), and their share of
     each giant fiber's total synaptic input;
  2. the top input cell types of each giant fiber (other usable pathways);
  3. two-hop LC4/LPLC2 -> X -> DNp01 contributions through intermediate cells;
  4. retinotopy: for LC4/LPLC2 cells with optic-lobe hex coordinates (assignedOlHex1/2), how direct
     synapses onto the giant fiber vary across the visual field.
Needs data/body-annotations.feather, data/body-neurotransmitters.feather (unused here) and
data/connectome-weights.feather (agents/download_malecns.sh).

Usage: python agents/experiments/pathway/giant_fiber_pathway.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.path.join(ROOT, "data")
LOOMING = ("LC4", "LPLC2")
GF = "DNp01"


def main() -> None:
	ann = pd.read_feather(os.path.join(DATA, "body-annotations.feather"))
	w = pd.read_feather(os.path.join(DATA, "connectome-weights.feather"))
	cols = {c.lower(): c for c in w.columns}
	pre_c = cols.get("body_pre", list(w.columns)[0])
	post_c = cols.get("body_post", list(w.columns)[1])
	wt_c = cols.get("weight", list(w.columns)[2])
	print(f"weights: {len(w):,} edges, columns {list(w.columns)}")

	ann = ann.set_index("bodyId")
	gf = ann[ann["type"] == GF]
	gf_ids = {row["somaSide"]: bid for bid, row in gf.iterrows()}
	print("giant fibers:", gf_ids)
	loom = ann[ann["type"].isin(LOOMING)]

	into_gf = w[w[post_c].isin(gf_ids.values())]
	total_in = into_gf.groupby(post_c)[wt_c].sum()
	report = {}

	print("\n== 1. direct looming -> giant fiber synapses")
	for t in LOOMING:
		for eye in ("L", "R"):
			cells = loom[(loom["type"] == t) & (loom["somaSide"] == eye)].index
			for side, g in gf_ids.items():
				s = into_gf[(into_gf[pre_c].isin(cells)) & (into_gf[post_c] == g)][wt_c].sum()
				share = s / total_in.get(g, 1)
				report[f"{t}_{eye}->GF_{side}"] = dict(synapses=int(s), share=float(share), cells=int(len(cells)))
				print(f"  {t:5s} eye {eye} ({len(cells):3d} cells) -> GF {side}: {int(s):6d} synapses ({share:.1%} of its input)")

	print("\n== 2. top input types of each giant fiber")
	tmap = ann["type"].to_dict()
	smap = ann["somaSide"].to_dict()
	for side, g in gf_ids.items():
		e = into_gf[into_gf[post_c] == g].copy()
		e["type"] = e[pre_c].map(tmap)
		top = e.groupby("type")[wt_c].sum().sort_values(ascending=False).head(10)
		print(f"  GF {side} (total input {int(total_in.get(g, 0))}): " + ", ".join(f"{k} {int(v)}" for k, v in top.items()))

	print("\n== 3. two-hop looming -> X -> giant fiber (top intermediates)")
	loom_ids = set(loom.index)
	from_loom = w[w[pre_c].isin(loom_ids)]
	mids = into_gf[~into_gf[pre_c].isin(loom_ids)][[pre_c, post_c, wt_c]].rename(columns={pre_c: "mid", wt_c: "w2"})
	two = from_loom.merge(mids, left_on=post_c, right_on="mid", suffixes=("", "_gf"))
	two["path"] = two[wt_c] * two["w2"]
	two["mid_type"] = two["mid"].map(tmap)
	two["gf_side"] = two[post_c + "_gf"].map(smap)
	top_mid = two.groupby(["mid_type", "gf_side"])["path"].sum().sort_values(ascending=False).head(10)
	for (mt, s), v in top_mid.items():
		print(f"  via {mt} -> GF {s}: path weight {int(v)}")

	print("\n== 4. retinotopy of direct looming -> giant fiber input")
	for t in LOOMING:
		sub = loom[loom["type"] == t]
		e = into_gf[into_gf[pre_c].isin(sub.index)]
		per_cell = e.groupby(pre_c)[wt_c].sum()
		hexed = sub[["assignedOlHex1", "assignedOlHex2", "somaSide"]].dropna()
		hexed = hexed.assign(gf=per_cell.reindex(hexed.index).fillna(0))
		print(f"  {t}: {len(hexed)}/{len(sub)} cells have hex coordinates; "
		      f"cells with any GF synapse: {(per_cell > 0).sum()}")
		if len(hexed) > 5:
			for eye in ("L", "R"):
				h = hexed[hexed["somaSide"] == eye]
				if len(h) > 5:
					c1 = np.corrcoef(h["assignedOlHex1"].astype(float), h["gf"])[0, 1]
					c2 = np.corrcoef(h["assignedOlHex2"].astype(float), h["gf"])[0, 1]
					print(f"    eye {eye}: corr(GF synapses, hex1) {c1:+.2f}, corr(hex2) {c2:+.2f}, "
					      f"hex1 range {h['assignedOlHex1'].min()}..{h['assignedOlHex1'].max()}, "
					      f"hex2 range {h['assignedOlHex2'].min()}..{h['assignedOlHex2'].max()}")
	with open(os.path.join(os.path.dirname(__file__), "giant_fiber_pathway.json"), "w", encoding="utf-8") as f:
		json.dump(report, f, indent=1)


if __name__ == "__main__":
	main()
