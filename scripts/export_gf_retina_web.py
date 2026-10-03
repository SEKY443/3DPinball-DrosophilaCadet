"""Export the untrained retinotopic giant-fiber body for the static browser demo.

Writes web/gf_body_retina.json (graph + precomputed normalized weights + dynamics gain + per-cell receptive-field
points + params) and web/testdata/gf_body_retina_vectors.json (obs sequences from real engine lives with the Python
body's activities and actions, for web/test/verify_gf_retina.mjs).

Usage (repo root):  .venv/bin/python scripts/export_gf_retina_web.py [--seeds 77000 77001] [--max-steps 600]
Source of truth: agents/experiments/pathway_body/gf_retina_body.py (read-only) and best_retina_real_sweep.json["params"] (the "sweep" layout).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PB = os.path.join(ROOT, "agents", "experiments", "pathway_body")
for p in (ROOT, PB, os.path.dirname(__file__)):
	if p not in sys.path:
		sys.path.insert(0, p)


def main() -> None:
	from export_gf_web import load_params, run_export
	from gf_retina_body import RetinaGiantFiberBody

	# The web body is the "sweep" receptive-field layout (three lines over the whole flipper sweep, see gf_retina_body.py):
	# 96-life paired eval 11.235 vs 11.047 for the single-line "mid" layout (eval_gf_retina_sweep.log).
	params = load_params("best_retina_real_sweep.json", lambda d: d["params"])
	params.pop("retina", None)  # run_life_retina cache-key flag, not a constructor argument

	def inputs(g, br):
		# original channel of every input cell = its SIDE (gf_circuit.json channels loom_L=0 / loom_R=1); the browser
		# derives the display group (LC4/LPLC2 x side) from input_is_lc4
		orig = {int(c): int(ch) for c, ch in g["inputs"]}
		# per input cell: circuit cell index, side
		return [[int(c), orig[int(c)]] for c in br.input_cells.tolist()]

	def extras(body):
		ks = body.brain.input_channels.tolist()  # per-cell channel index in body.targets / body.is_lc4
		# is_lc4 and the receptive-field point (x, y) of every input cell
		return {
			"input_is_lc4": [bool(body.is_lc4[k]) for k in ks],
			"input_targets": [[float(body.targets[k][0]), float(body.targets[k][1])] for k in ks],
		}, None

	run_export(
		RetinaGiantFiberBody, params,
		"Retinotopic giant-fiber body: each LC4/LPLC2 cell watches its own spot along the flipper -> "
		"fixed MaleCNS circuit -> DNp01 (GF_L, GF_R). Untrained.",
		inputs, extras, default_max_steps=600,
		graph_name="gf_body_retina.json", vectors_name="gf_body_retina_vectors.json",
	)


if __name__ == "__main__":
	main()
