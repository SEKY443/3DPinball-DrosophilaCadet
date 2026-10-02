"""Selectivity probe of the sub-critical GF circuits: step drive on one loom channel (10 steps) then 0
(20 steps), from a clean reset. Prints GF_L/GF_R at the end of the drive and 20 steps after removal.

Usage: python agents/experiments/pathway_body/probe_gf_subcrit.py [--rho 0.8]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gf_body as G  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rho", type=float, default=0.8)
args = ap.parse_args()
res = {}
for name, f in (("real", "gf_circuit.json"), ("shuf1", "gf_circuit_shuffled_1.json"), ("shuf2", "gf_circuit_shuffled_2.json"), ("shuf3", "gf_circuit_shuffled_3.json")):
	b = G.GiantFiberBody(os.path.join(G.HERE, f), 1.0, 1.0, 0.5, rho_target=args.rho)
	print(f"[{name}] rho {b.rho:.3f} dyn_gain {b.dyn_gain:.3f} (gain*rho {b.dyn_gain * b.rho:.3f})")
	for side, ch in (("L", 0), ("R", 1)):
		for A in (0.2, 0.5, 1.0):
			h = torch.zeros(1, b.brain.n)
			d = torch.zeros(1, 2)
			d[0, ch] = A
			for _ in range(10):
				h = b._step(h, d)
			end = [float(h[0, i]) for i in b.out_idx]
			for _ in range(20):
				h = b._step(h, torch.zeros(1, 2))
			aft = [float(h[0, i]) for i in b.out_idx]
			print(f"   loom_{side} A={A}: end-of-drive GF_L {end[0]:.3f} GF_R {end[1]:.3f} | +20 after GF_L {aft[0]:.3f} GF_R {aft[1]:.3f}")
			res[f"{name}_{side}_{A}"] = dict(end=end, after=aft)
json.dump(res, open(os.path.join(G.HERE, "probe_subcrit.json"), "w"), indent=1)
