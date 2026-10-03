"""Export the untrained split giant-fiber body for the static browser demo.

Writes web/gf_body_split.json (graph + precomputed normalized weights + dynamics gain + params) and
web/testdata/gf_body_split_vectors.json (obs sequences from real engine lives with the Python body's
activities and actions, for web/test/verify_gf_body.mjs).

Usage (repo root):  .venv/bin/python scripts/export_gf_web.py [--seeds 77000 77001 77002 77003] [--max-steps 150]
Source of truth: agents/experiments/pathway_body/gf_split_body.py (read-only) and the "real_best" entry of
agents/experiments/pathway_body/eval_scan_best.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PB = os.path.join(ROOT, "agents", "experiments", "pathway_body")
for p in (ROOT, PB):
	if p not in sys.path:
		sys.path.insert(0, p)


H_EVERY = 25


def g9(x: float) -> float:
	return float("%.9g" % x)


def load_params(best_json: str, select) -> dict:
	"""Body constructor params: select(parsed best_json) -> params dict (copied) plus the repo-relative circuit path."""
	with open(os.path.join(PB, best_json), encoding="utf-8") as f:
		p = dict(select(json.load(f)))
	p["circuit_json"] = os.path.join(PB, "gf_circuit.json")  # repo-relative, not the machine-specific path
	return p


def export_graph(body, params: dict, path: str, description: str, inputs_fn, after_inputs: dict | None = None,
                 tail: dict | None = None) -> None:
	"""Write the browser graph JSON. Shared fields come from `body`; body-specific ones are passed in:
	inputs_fn(circuit_graph, brain) -> "inputs" list, `after_inputs` fields follow "inputs", `tail` fields end the
	file (key order is part of the output, so exports stay byte-identical)."""
	from agents.fixed_circuit_agent import DYNAMICS_ITERATIONS, DYNAMICS_LEAK

	with open(params["circuit_json"], encoding="utf-8") as f:
		g = json.load(f)
	# a few cells have NaN type/side in the source graph; map them to "" (JSON has no NaN)
	for n in g["nodes"]:
		for k in ("type", "side"):
			if not isinstance(n.get(k), str):
				n[k] = ""
	types = sorted({n["type"] for n in g["nodes"]})
	br = body.brain
	out = {
		"description": description,
		"channels": ["LC4_L", "LC4_R", "LPLC2_L", "LPLC2_R"],
		"types": types,
		"nodes": {
			"id": [n["id"] for n in g["nodes"]],
			"type": [types.index(n["type"]) for n in g["nodes"]],
			"side": [n.get("side", "") for n in g["nodes"]],
			"sign": [int(n["sign"]) for n in g["nodes"]],
		},
		"inputs": inputs_fn(g, br),
		**(after_inputs or {}),
		"outputs": [int(i) for i in body.out_idx],  # [GF_L, GF_R]
		"edges": {
			"pre": br.pre.tolist(),
			"post": br.post.tolist(),
			"w": [g9(x) for x in br.w.tolist()],  # normalized signed weights (float32)
		},
		"dynamics": {"leak": DYNAMICS_LEAK, "iterations": DYNAMICS_ITERATIONS, "gain": g9(body.dyn_gain),
		             "rho": g9(body.rho)},
		**(tail or {}),
		"params": {k: params[k] for k in ("R", "g4", "g2", "v0", "s0", "theta_on", "theta_off", "hold_max",
		                                  "refractory", "rho_target")},
	}
	with open(path, "w", encoding="utf-8") as f:
		json.dump(out, f, separators=(",", ":"))
	print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)")


def record_vectors(body_cls, params: dict, seeds: list[int], max_steps: int, path: str) -> None:
	import gf_body as G
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	G.init_worker(max_steps=3000)
	body = body_cls(**params)
	lives = []
	for seed in seeds:
		obs, _ = G._ENV.reset(seed=seed)
		body.reset()
		steps, done = [], False
		while not done and len(steps) < max_steps:
			act = body.act(obs)
			steps.append({
				"obs": [float(obs[i]) for i in (OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY)],
				"action": [int(act[0]), int(act[1])],
				"a": [body.last_a[0], body.last_a[1]],
			})
			if len(steps) % H_EVERY == 1:  # full activity vector only every H_EVERY-th step (keeps the file small)
				steps[-1]["h"] = [g9(x) for x in body.h[0].tolist()]
			obs, _r, term, trunc, info = G._ENV.step(act)
			done = term or trunc or bool(info["drained"])
		lives.append({"seed": seed, "steps": steps})
		pr = [sum(s["action"][k] for s in steps) for k in (0, 1)]
		print(f"seed {seed}: {len(steps)} steps, pressed-steps L/R = {pr}")
	with open(path, "w", encoding="utf-8") as f:
		json.dump({"lives": lives}, f, separators=(",", ":"))
	print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KiB)")


def run_export(body_cls, params: dict, description: str, inputs_fn, extras_fn, default_max_steps: int,
               graph_name: str, vectors_name: str) -> None:
	"""CLI skeleton shared by the exports: --seeds / --max-steps, then graph JSON and parity vectors under web/.
	extras_fn(body) -> (after_inputs, tail) body-specific graph fields."""
	ap = argparse.ArgumentParser()
	ap.add_argument("--seeds", type=int, nargs="+", default=[77000, 77001])
	ap.add_argument("--max-steps", type=int, default=default_max_steps)
	args = ap.parse_args()
	body = body_cls(**params)
	after_inputs, tail = extras_fn(body)
	export_graph(body, params, os.path.join(ROOT, "web", graph_name), description, inputs_fn, after_inputs, tail)
	record_vectors(body_cls, params, args.seeds, args.max_steps, os.path.join(ROOT, "web", "testdata", vectors_name))


def main() -> None:
	from agents.train_pinball_circuit_cem import LOOM_TARGET_LEFT, LOOM_TARGET_RIGHT
	from gf_split_body import SplitGiantFiberBody

	params = load_params("eval_scan_best.json", lambda d: d["params"]["real_best"])
	run_export(
		SplitGiantFiberBody, params,
		"Split giant-fiber body: LC4/LPLC2 -> fixed MaleCNS circuit -> DNp01 (GF_L, GF_R). Untrained.",
		# input cells with their remapped channel (0..3 = LC4_L, LC4_R, LPLC2_L, LPLC2_R)
		lambda g, br: [[int(c), int(ch)] for c, ch in zip(br.input_cells.tolist(), br.input_channels.tolist())],
		lambda body: ({}, {"loom_targets": {"left": list(LOOM_TARGET_LEFT), "right": list(LOOM_TARGET_RIGHT)}}),
		default_max_steps=900, graph_name="gf_body_split.json", vectors_name="gf_body_split_vectors.json",
	)


if __name__ == "__main__":
	main()
