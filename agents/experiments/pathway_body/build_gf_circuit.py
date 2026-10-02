"""Build the giant-fiber (DNp01) circuit: all LC4 + LPLC2 cells, both DNp01 giant fibers, the top-N
two-hop intermediates (LC4/LPLC2 -> X -> GF) and the top-M other direct GF inputs.

Same JSON format as agents/build_pinball_connectome.py (nodes/edges/inputs/outputs/channels) plus a
top-level "output_sides" field. Also writes 3 degree-preserving shuffles (make_shuffled.shuffle_edges).

Usage: python agents/experiments/pathway_body/build_gf_circuit.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "shuffle"))
from make_shuffled import shuffle_edges  # noqa: E402

SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}  # same as agents/build_pinball_connectome.py
GF_L, GF_R = 10010, 10001
MIN_SYN = 3


def lc_gf_ipsi_check(g: dict) -> dict:
	"""Synapses from LC4/LPLC2 cells onto each GF, split by whether the cell's side matches the GF."""
	nodes = g["nodes"]
	side = {i: ("L" if g["channels"][c] == "loom_L" else "R") for i, c in g["inputs"]}
	out = collections.Counter()
	gf_side = {g["outputs"][0]: "L", g["outputs"][1]: "R"}
	for a, b, w in g["edges"]:
		if a in side and b in gf_side:
			out["ipsi" if side[a] == gf_side[b] else "contra"] += w
	return dict(out)


def build(data_dir: str, n_two_hop: int, n_direct: int) -> dict:
	ann = feather.read_table(os.path.join(data_dir, "body-annotations.feather"), columns=["bodyId", "type", "somaSide"]).to_pandas().set_index("bodyId")
	nt = dict(zip(*(feather.read_table(os.path.join(data_dir, "body-neurotransmitters.feather"), columns=["body", "consensus_nt"]).to_pydict()[k] for k in ("body", "consensus_nt"))))
	lc = ann.index[ann.type.isin(["LC4", "LPLC2"])].to_numpy()
	gfs = np.array([GF_L, GF_R])

	e = feather.read_table(os.path.join(data_dir, "connectome-weights.feather"))
	# Filter early: only edges that touch an LC cell or a GF.
	m = pc.or_(pc.or_(pc.is_in(e["body_pre"], value_set=__import__("pyarrow").array(lc)), pc.is_in(e["body_post"], value_set=__import__("pyarrow").array(lc))),
	           pc.or_(pc.is_in(e["body_post"], value_set=__import__("pyarrow").array(gfs)), pc.is_in(e["body_pre"], value_set=__import__("pyarrow").array(gfs))))
	# two-hop needs X -> GF and LC -> X; X -> GF covered by post in gfs, LC -> X by pre in lc.
	sub = e.filter(m)
	pre, post, w = sub["body_pre"].to_numpy(), sub["body_post"].to_numpy(), sub["weight"].to_numpy()
	keep = w >= MIN_SYN
	pre, post, w = pre[keep], post[keep], w[keep]

	lc_set, gf_set = set(int(x) for x in lc), {GF_L, GF_R}
	inc, out = collections.Counter(), collections.Counter()
	direct = collections.Counter()
	for a, b, ww in zip(pre.tolist(), post.tolist(), w.tolist()):
		if a in lc_set and b not in lc_set and b not in gf_set:
			inc[b] += ww
		if b in gf_set and a not in gf_set and a not in lc_set:
			out[a] += ww
			direct[a] += ww
	cand = [x for x in inc.keys() & out.keys() if x in ann.index]
	two_hop = sorted(cand, key=lambda x: (-min(inc[x], out[x]), x))[:n_two_hop]
	rest = sorted((x for x in direct if x not in set(two_hop) and x in ann.index), key=lambda x: (-direct[x], x))[:n_direct]
	ids = sorted(lc_set | gf_set | set(two_hop) | set(rest))
	idx = {i: j for j, i in enumerate(ids)}

	# Edges among the chosen cells: need the full edge list restricted to chosen ids. Intermediates
	# may connect to each other, so re-filter the whole table.
	idarr = __import__("pyarrow").array(np.array(ids))
	m2 = pc.and_(pc.is_in(e["body_pre"], value_set=idarr), pc.is_in(e["body_post"], value_set=idarr))
	s2 = e.filter(m2)
	p2, q2, w2 = s2["body_pre"].to_numpy(), s2["body_post"].to_numpy(), s2["weight"].to_numpy()
	edges = sorted([idx[int(a)], idx[int(b)], int(c)] for a, b, c in zip(p2, q2, w2) if c >= MIN_SYN and a != b)

	role = {}
	for i in ids:
		role[i] = "input" if i in lc_set else ("output" if i in gf_set else "interneuron")
	nodes = [dict(id=int(i), type=ann.loc[i, "type"], side=ann.loc[i, "somaSide"], nt=nt.get(i), sign=SIGN.get(nt.get(i), 0), role=role[i]) for i in ids]
	inputs = [[idx[int(i)], 0 if ann.loc[i, "somaSide"] == "L" else 1] for i in sorted(lc_set) if ann.loc[i, "somaSide"] in ("L", "R")]
	return dict(version="malecns-gf-circuit-v1", nodes=nodes, edges=edges, inputs=inputs,
	            outputs=[idx[GF_L], idx[GF_R]], channels=["loom_L", "loom_R"], output_sides=["L", "R"],
	            n_two_hop=len(two_hop), n_direct=len(rest))


def report(name: str, g: dict) -> None:
	cnt = collections.Counter((n["role"], n["type"] if n["role"] != "interneuron" else "interneuron", n.get("side")) for n in g["nodes"])
	print(f"[{name}] nodes {len(g['nodes'])} edges {len(g['edges'])} contacts {sum(e[2] for e in g['edges'])} inputs {len(g['inputs'])}")
	for k, v in sorted(cnt.items(), key=lambda kv: str(kv[0])):
		print("   ", k, v)
	print(f"[{name}] LC->GF synapses ipsi/contra: {lc_gf_ipsi_check(g)}")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
	ap.add_argument("--n-two-hop", type=int, default=40)
	ap.add_argument("--n-direct", type=int, default=20)
	args = ap.parse_args()
	g = build(args.data_dir, args.n_two_hop, args.n_direct)
	with open(os.path.join(HERE, "gf_circuit.json"), "w", encoding="utf-8") as f:
		json.dump(g, f)
	report("real", g)
	inter = collections.Counter(n["type"] for n in g["nodes"] if n["role"] == "interneuron")
	print("interneuron types:", dict(inter.most_common()))
	for s in (1, 2, 3):
		edges, done, attempts = shuffle_edges(g["edges"], s)
		gs = dict(g, edges=edges, shuffled=dict(seed=s, swaps=done, attempts=attempts))
		with open(os.path.join(HERE, f"gf_circuit_shuffled_{s}.json"), "w", encoding="utf-8") as f:
			json.dump(gs, f)
		report(f"shuf{s}", gs)


if __name__ == "__main__":
	main()
