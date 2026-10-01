"""Degree-preserving shuffle of the pinball connectome (control for "does the fly wiring matter?").

Keeps every node exactly as it is (id, cell type, neurotransmitter sign, role, input/output
assignment, channel mapping) and every edge's synapse count, but rewires WHO connects to WHOM with
directed double-edge swaps: (a->b, c->d) becomes (a->d, c->b), rejected if it would create a self
loop or a duplicate edge. Each swap preserves every node's out-degree and in-degree, so the shuffled
graph has the same degree sequence, the same excitatory/inhibitory balance per presynaptic cell and
the same total input per cell - only the specific wiring pattern is destroyed.

Usage: python agents/experiments/shuffle/make_shuffled.py --seed 1 --out agents/experiments/shuffle/connectome_shuffled_1.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REAL = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")


def shuffle_edges(edges, seed: int, swaps_per_edge: int = 10):
	rng = random.Random(seed)
	edges = [list(e) for e in edges]
	present = {(e[0], e[1]) for e in edges}
	n = len(edges)
	done = attempts = 0
	target = swaps_per_edge * n
	while done < target and attempts < 50 * target:
		attempts += 1
		i, j = rng.randrange(n), rng.randrange(n)
		if i == j:
			continue
		a, b, wab = edges[i]
		c, d, wcd = edges[j]
		if a == d or c == b or (a, d) in present or (c, b) in present:
			continue
		present.discard((a, b))
		present.discard((c, d))
		present.add((a, d))
		present.add((c, b))
		edges[i] = [a, d, wab]
		edges[j] = [c, b, wcd]
		done += 1
	return edges, done, attempts


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--src", default=REAL)
	ap.add_argument("--seed", type=int, required=True)
	ap.add_argument("--out", required=True)
	args = ap.parse_args()
	with open(args.src, encoding="utf-8") as f:
		g = json.load(f)
	edges, done, attempts = shuffle_edges(g["edges"], args.seed)
	before = {(e[0], e[1]) for e in g["edges"]}
	after = {(e[0], e[1]) for e in edges}
	out_deg = lambda es: sorted((e[0] for e in es))
	in_deg = lambda es: sorted((e[1] for e in es))
	assert out_deg(edges) == out_deg(g["edges"]) and in_deg(edges) == in_deg(g["edges"]), "degree sequence changed"
	assert len(after) == len(edges) and all(a != b for a, b in after), "duplicate or self loop"
	g = dict(g, edges=edges, shuffled=dict(seed=args.seed, swaps=done, attempts=attempts,
	                                        kept_original_edges=len(before & after), source=os.path.relpath(args.src, ROOT)))
	os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(g, f)
	print(f"seed {args.seed}: {done} swaps ({attempts} attempts), {len(before & after)}/{len(edges)} original edges kept -> {args.out}")


if __name__ == "__main__":
	main()
