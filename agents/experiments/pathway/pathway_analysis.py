"""Pathway analysis of the 444-cell pinball circuit: which motor-neuron outputs does each sensory
channel actually reach through the real wiring, and is any of that structure specific to the real
connectome (vs degree-preserving shuffles)?

Signed effective weights follow the model (agents/fixed_circuit_agent.py / web/circuit_policy.js):
w(pre->post) = contacts * sign(pre) / sum_pre' contacts(pre'->post). Multi-hop influence is the sum
of W^k for k = 1..K_HOPS (input cell -> output cell), averaged over the cells of each input channel.
Output motor neurons are annotated with soma/root side and neuromere from the MaleCNS v1.0 body
annotations (data/body-annotations.feather, see agents/download_malecns.sh).

Reports: (1) the subset's input/output/descending composition by side; (2) per channel, the mean
influence on LEFT-side vs RIGHT-side motor neurons and a laterality index; (3) the same laterality for
the shuffled connectomes - wiring-specific structure is what the real graph has and shuffles lack.

Usage: python agents/experiments/pathway/pathway_analysis.py
"""
from __future__ import annotations

import collections
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REAL = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")
SHUFFLED = [os.path.join(ROOT, f"agents/experiments/shuffle/connectome_shuffled_{i}.json") for i in (1, 2, 3)]
ANNOT = os.path.join(ROOT, "data/body-annotations.feather")
K_HOPS = 6


def influence(g: dict) -> np.ndarray:
	"""(n, n) summed multi-hop signed influence, rows = source cell, cols = target cell."""
	n = len(g["nodes"])
	sign = np.array([nd["sign"] for nd in g["nodes"]], dtype=np.float64)
	total = np.zeros(n)
	for pre, post, c in g["edges"]:
		total[post] += c * abs(sign[pre])
	W = np.zeros((n, n))
	for pre, post, c in g["edges"]:
		if total[post] > 0:
			W[pre, post] += c * sign[pre] / total[post]
	acc, P = np.zeros((n, n)), np.eye(n)
	for _ in range(K_HOPS):
		P = P @ W
		acc += P
	return acc


def side_of(row) -> str:
	for col in ("somaSide", "rootSide"):
		v = row.get(col)
		if isinstance(v, str) and v[:1] in ("L", "R"):
			return v[:1]
	return "?"


def channel_laterality(g: dict, M: np.ndarray, out_side: dict) -> dict:
	"""channel -> (mean influence on L outputs, on R outputs, laterality (L-R)/(|L|+|R|))."""
	by_ch = collections.defaultdict(list)
	for cell, ch in g["inputs"]:
		by_ch[g["channels"][ch]].append(cell)
	left = [o for o in g["outputs"] if out_side[o] == "L"]
	right = [o for o in g["outputs"] if out_side[o] == "R"]
	res = {}
	for ch, cells in by_ch.items():
		li = float(M[np.ix_(cells, left)].mean()) if left else 0.0
		ri = float(M[np.ix_(cells, right)].mean()) if right else 0.0
		res[ch] = (li, ri, (li - ri) / (abs(li) + abs(ri) + 1e-12))
	return res


def main() -> None:
	with open(REAL, encoding="utf-8") as f:
		g = json.load(f)
	ann = pd.read_feather(ANNOT).set_index("bodyId")
	nodes = g["nodes"]
	info = {}
	for i, nd in enumerate(nodes):
		row = ann.loc[nd["id"]].to_dict() if nd["id"] in ann.index else {}
		info[i] = dict(side=side_of(row), superclass=row.get("superclass"), neuromere=row.get("somaNeuromere"),
		               cls=row.get("class"), type=nd["type"])
	out_side = {o: info[o]["side"] for o in g["outputs"]}

	print("== composition")
	print("inputs by side:", dict(collections.Counter(info[c]["side"] for c, _ in g["inputs"])))
	print("outputs by side:", dict(collections.Counter(out_side.values())))
	print("outputs by neuromere x side:", dict(collections.Counter((info[o]["neuromere"], out_side[o]) for o in g["outputs"])))
	dns = [i for i in range(len(nodes)) if (info[i]["superclass"] or "").startswith("descending")]
	print("descending neurons:", [(info[i]["type"], info[i]["side"]) for i in dns])

	M = influence(g)
	real = channel_laterality(g, M, out_side)
	shuf = []
	for p in SHUFFLED:
		with open(p, encoding="utf-8") as f:
			gs = json.load(f)
		shuf.append(channel_laterality(gs, influence(gs), out_side))

	print(f"\n== channel -> motor-neuron influence ({K_HOPS} hops), laterality = (L-R)/(|L|+|R|)")
	print(f"{'channel':<14} {'infl L':>9} {'infl R':>9} {'lat real':>9} {'lat shuffled (3)':>24}")
	for ch in g["channels"]:
		if ch not in real:
			continue
		li, ri, lat = real[ch]
		sl = " ".join(f"{s[ch][2]:+.2f}" for s in shuf)
		print(f"{ch:<14} {li:+9.4f} {ri:+9.4f} {lat:+9.2f} {sl:>24}")

	# direct DN contribution: how much of the input->output influence passes through the descending neurons
	print("\n== descending-neuron routing")
	for d in dns:
		inflow = M[[c for c, _ in g["inputs"]], d].mean()
		outflow = M[d, g["outputs"]].mean()
		print(f"  {info[d]['type']} ({info[d]['side']}): mean influence from inputs {inflow:+.4f}, onto outputs {outflow:+.4f}")
	with open(os.path.join(os.path.dirname(__file__), "pathway_analysis.json"), "w", encoding="utf-8") as f:
		json.dump(dict(real=real, shuffled=shuf, out_side={str(k): v for k, v in out_side.items()},
		               info={str(k): v for k, v in info.items()}), f, default=str)


if __name__ == "__main__":
	main()
