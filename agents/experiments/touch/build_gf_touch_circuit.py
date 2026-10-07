"""Build the touch-augmented giant-fiber circuit: gf_circuit.json (LC4/LPLC2 -> ... -> DNp01) PLUS Johnston's-organ (JO)
mechanosensory neurons that reach either DNp01 directly or through one intermediate, and the strongest of those intermediates.

Path weight is the connectome-normalised strength: direct = w(JO,GF) / in(GF); two-hop = sum_m w(JO,m)/in(m) * w(m,GF)/in(GF)
(in(.) = total incoming synapses of that cell; edges below MIN_SYN synapses are ignored). JO cells are chosen per rootSide
(antenna side) by descending path weight, intermediates by descending path weight from the chosen JO cells. Every edge among
all included nodes comes from the full weights table; signs from body-neurotransmitters.feather (as build_gf_circuit.py).

Channels: loom_L, loom_R (LC4/LPLC2 by soma side, unchanged) + JO_L, JO_R (JO cells by rootSide).
Writes gf_touch_circuit.json and gf_touch_circuit_shuffled_{1..6}.json (degree-preserving over the whole circuit).

Usage: python agents/experiments/touch/build_gf_touch_circuit.py [--n-jo-per-side 125] [--n-mid 40]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "shuffle"))
from make_shuffled import shuffle_edges  # noqa: E402

SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
MIN_SYN = 3


def build(n_jo_side: int, n_mid: int) -> dict:
	D = os.path.join(ROOT, "data")
	base = json.load(open(os.path.join(ROOT, "agents", "experiments", "pathway_body", "gf_circuit.json"), encoding="utf-8"))
	ann = pd.read_feather(os.path.join(D, "body-annotations.feather"), columns=["bodyId", "type", "somaSide", "rootSide", "superclass", "class"]).set_index("bodyId")
	ntt = pd.read_feather(os.path.join(D, "body-neurotransmitters.feather"), columns=["body", "consensus_nt"])
	nt = dict(zip(ntt["body"], ntt["consensus_nt"]))
	w = pd.read_feather(os.path.join(D, "connectome-weights.feather"))
	w = w[w["weight"] >= MIN_SYN]
	base_ids = [n["id"] for n in base["nodes"]]
	gf_ids = [base_ids[i] for i in base["outputs"]]  # [GF_L, GF_R]
	lc_ids = {n["id"] for n in base["nodes"] if n["role"] == "input"}
	jo_all = ann.index[ann["type"].fillna("").str.startswith("JO-") | (ann["type"].fillna("").isin(["JO-unclear"]))]
	jo_all = set(int(x) for x in jo_all)
	print("JO neurons total:", len(jo_all), dict(ann.loc[sorted(jo_all), "rootSide"].value_counts(dropna=False)))

	inn = w.groupby("body_post")["weight"].sum()
	into_gf = w[w["body_post"].isin(gf_ids)]
	# mid -> GF
	m2g = into_gf[~into_gf["body_pre"].isin(set(gf_ids) | lc_ids | jo_all)].copy()
	m2g["f2"] = m2g["weight"] / m2g["body_post"].map(inn)
	m2g = m2g.set_index("body_pre")["f2"].groupby(level=0).sum()
	m2g = m2g[m2g.index.isin(ann.index)]
	# JO -> GF direct
	dj = into_gf[into_gf["body_pre"].isin(jo_all)]
	direct = (dj["weight"] / dj["body_post"].map(inn)).groupby(dj["body_pre"]).sum()
	# JO -> mid
	j1 = w[w["body_pre"].isin(jo_all) & w["body_post"].isin(set(m2g.index))].copy()
	j1["f1"] = j1["weight"] / j1["body_post"].map(inn)
	j1["path"] = j1["f1"] * j1["body_post"].map(m2g)
	two = j1.groupby("body_pre")["path"].sum()
	score = direct.reindex(sorted(jo_all)).fillna(0).add(two.reindex(sorted(jo_all)).fillna(0), fill_value=0)
	score = score[score > 0]
	chosen = []
	for side in ("L", "R"):
		s = score[[i for i in score.index if ann.loc[i, "rootSide"] == side]].sort_values(ascending=False)
		print(f"JO rootSide {side}: {len(s)} with a path to GF; keeping {min(n_jo_side, len(s))}")
		chosen += [int(i) for i in s.index[:n_jo_side]]
	chosen_set = set(chosen)
	jj = j1[j1["body_pre"].isin(chosen_set)]
	mid_score = jj.groupby("body_post")["path"].sum().sort_values(ascending=False)
	mids = [int(i) for i in mid_score.index[:n_mid]]

	ids = sorted(set(base_ids) | chosen_set | set(mids))
	idx = {i: j for j, i in enumerate(ids)}
	sub = w[w["body_pre"].isin(ids) & w["body_post"].isin(ids) & (w["body_pre"] != w["body_post"])]
	edges = sorted([idx[int(a)], idx[int(b)], int(c)] for a, b, c in zip(sub["body_pre"], sub["body_post"], sub["weight"]))
	old = {n["id"]: n for n in base["nodes"]}
	nodes = []
	for i in ids:
		if i in old:
			nodes.append(dict(old[i]))
		else:
			a = ann.loc[i]
			ntv = nt.get(i)
			role = "input" if i in chosen_set else "interneuron"
			side = a["rootSide"] if role == "input" else a["somaSide"]
			nodes.append(dict(id=int(i), type=a["type"], side=None if pd.isna(side) else side, nt=ntv, sign=SIGN.get(ntv, 0), role=role))
	inputs = [[idx[i], 0 if old[i]["side"] == "L" else 1] for i in sorted(lc_ids) if old[i]["side"] in ("L", "R")]
	inputs += [[idx[i], 2 if ann.loc[i, "rootSide"] == "L" else 3] for i in sorted(chosen_set)]
	# keep inputs in the original LC order (same channel assignment as gf_circuit.json)
	lc_inputs = [[idx[base_ids[c]], ch] for c, ch in base["inputs"]]
	inputs = lc_inputs + inputs[len(lc_inputs):]
	return dict(version="malecns-gf-touch-circuit-v1", nodes=nodes, edges=edges, inputs=inputs,
	            outputs=[idx[gf_ids[0]], idx[gf_ids[1]]], channels=["loom_L", "loom_R", "JO_L", "JO_R"], output_sides=["L", "R"],
	            n_jo=len(chosen_set), n_mid=len(mids))


def report(name: str, g: dict) -> None:
	nodes = g["nodes"]
	print(f"[{name}] nodes {len(nodes)} edges {len(g['edges'])} synapses {sum(e[2] for e in g['edges'])} inputs {len(g['inputs'])}")
	ch = collections.Counter(c for _, c in g["inputs"])
	print(f"[{name}] input channels {dict(sorted(ch.items()))}; JO types {dict(collections.Counter(n['type'] for n in nodes if str(n['type']).startswith('JO')))}")
	chan_of = {i: c for i, c in g["inputs"]}
	for gi, gs in zip(g["outputs"], g["output_sides"]):
		tot = sum(e[2] for e in g["edges"] if e[1] == gi)
		jo = collections.Counter()
		for a, b, wt in g["edges"]:
			if b == gi and chan_of.get(a, -1) in (2, 3):
				jo["JO_L" if chan_of[a] == 2 else "JO_R"] += wt
		lc = sum(wt for a, b, wt in g["edges"] if b == gi and chan_of.get(a, -1) in (0, 1))
		print(f"[{name}] GF_{gs} in-synapses {tot}: JO_L {jo['JO_L']} ({jo['JO_L'] / tot:.1%}), JO_R {jo['JO_R']} ({jo['JO_R'] / tot:.1%}), LC4/LPLC2 {lc} ({lc / tot:.1%})")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--n-jo-per-side", type=int, default=125)
	ap.add_argument("--n-mid", type=int, default=40)
	a = ap.parse_args()
	g = build(a.n_jo_per_side, a.n_mid)
	json.dump(g, open(os.path.join(HERE, "gf_touch_circuit.json"), "w", encoding="utf-8"))
	report("real", g)
	print("new intermediates:", dict(collections.Counter(n["type"] for n in g["nodes"] if n["role"] == "interneuron").most_common(15)))
	for s in range(1, 7):
		edges, done, attempts = shuffle_edges(g["edges"], s)
		gs = dict(g, edges=edges, shuffled=dict(seed=s, swaps=done, attempts=attempts))
		json.dump(gs, open(os.path.join(HERE, f"gf_touch_circuit_shuffled_{s}.json"), "w", encoding="utf-8"))
		report(f"shuf{s}", gs)


if __name__ == "__main__":
	main()
