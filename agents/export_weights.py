#!/usr/bin/env python3
"""Exports a trained DrosophilaController checkpoint to web/weights.json.

Not ONNX: ONNX has no clean opset for LIF leak/reset/threshold dynamics, so forcing it through
ONNX would mean re-approximating away the spiking behavior anyway. Instead this writes a small
hand-rolled JSON that web/inference.js re-implements the exact same euler-integration LIF step
against (see norse.torch.functional.lif.lif_feed_forward_step for the reference equations this
mirrors): i_new = i + input; v += dt*tau_mem_inv*(v_leak - v + i_new); i = i_new - dt*tau_syn_inv*i_new;
spike = v_decayed >= v_th; v = spike ? v_reset : v_decayed.

Usage:
    python agents/export_weights.py --checkpoint agents/checkpoints/drosophila_controller_final.pt --out web/weights.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.snn_model import DrosophilaController, OUTPUT_LABELS  # noqa: E402


def lif_params_to_dict(cell) -> dict:
	p = cell.p
	return {
		"tau_syn_inv": float(p.tau_syn_inv),
		"tau_mem_inv": float(p.tau_mem_inv),
		"v_leak": float(p.v_leak),
		"v_th": float(p.v_th),
		"v_reset": float(p.v_reset),
		"dt": float(cell.dt),
	}


def export(checkpoint_path: str, out_path: str) -> None:
	ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
	model = DrosophilaController(hidden_dim=ckpt["hidden_dim"], sim_steps=ckpt["sim_steps"])
	model.load_state_dict(ckpt["model_state_dict"])
	model.eval()

	state = model.state_dict()
	# Fold DrosophilaController.forward's obs/self.obs_scale normalization into the weight
	# matrix (Linear(obs/scale) == (weight/scale) @ obs + bias, dividing column-wise by scale)
	# so the exported graph takes raw observations directly, matching env_python's
	# _state_to_obs output and web/bridge.js's stateToObs - no separate scale field needed.
	scaled_input_weight = state["input_linear.weight"] / model.obs_scale.unsqueeze(0)
	payload = {
		"version": 1,
		"input_dim": model.input_linear.in_features,
		"hidden_dim": model.hidden_dim,
		"output_dim": len(OUTPUT_LABELS),
		"sim_steps": model.sim_steps,
		"input_linear": {
			"weight": scaled_input_weight.tolist(),  # (hidden_dim, input_dim)
			"bias": state["input_linear.bias"].tolist(),  # (hidden_dim,)
		},
		"hidden_lif": lif_params_to_dict(model.hidden_lif),
		"output_linear": {
			"weight": state["output_linear.weight"].tolist(),  # (output_dim, hidden_dim)
			"bias": state["output_linear.bias"].tolist(),  # (output_dim,)
		},
		"output_lif": lif_params_to_dict(model.output_lif),
		"output_labels": OUTPUT_LABELS,
	}

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(payload, f, indent=2)
	print(f"wrote {out_path}")

	_verify_bit_for_bit(model, payload)


def _verify_bit_for_bit(model: DrosophilaController, payload: dict) -> None:
	"""Runs a fixed batch of observations through both the live PyTorch model and a pure-Python
	re-implementation driven only by `payload`, and asserts they match closely. This is the
	spec web/inference.js's JS re-implementation must reproduce (see Phase 5 of the project plan)."""
	torch.manual_seed(0)
	obs = torch.randn(16, payload["input_dim"])

	with torch.no_grad():
		expected = model(obs).numpy()

	actual = _reference_forward(payload, obs.numpy())

	import numpy as np
	max_diff = float(np.max(np.abs(expected - actual)))
	print(f"reference re-implementation max abs diff vs PyTorch model: {max_diff:.3e}")
	assert max_diff < 1e-5, "reference forward pass does not match the trained model closely enough"


def _reference_forward(payload: dict, obs):
	"""Pure NumPy re-implementation of DrosophilaController.forward, using only fields from
	the exported JSON payload - this is what web/inference.js's JS port must match."""
	import numpy as np

	w_in = np.array(payload["input_linear"]["weight"])  # (hidden, input)
	b_in = np.array(payload["input_linear"]["bias"])  # (hidden,)
	w_out = np.array(payload["output_linear"]["weight"])  # (output, hidden)
	b_out = np.array(payload["output_linear"]["bias"])  # (output,)
	hp = payload["hidden_lif"]
	op = payload["output_lif"]
	batch = obs.shape[0]
	hidden_dim = payload["hidden_dim"]
	output_dim = payload["output_dim"]

	h_v = np.zeros((batch, hidden_dim))
	h_i = np.zeros((batch, hidden_dim))
	o_v = np.zeros((batch, output_dim))
	o_i = np.zeros((batch, output_dim))
	spike_sum = np.zeros((batch, output_dim))

	def lif_step(input_current, v, i, p):
		i_new = i + input_current
		dv = p["dt"] * p["tau_mem_inv"] * ((p["v_leak"] - v) + i_new)
		v_decayed = v + dv
		di = -p["dt"] * p["tau_syn_inv"] * i_new
		i_decayed = i_new + di
		spike = (v_decayed - p["v_th"] > 0.0).astype(v.dtype)  # norse.functional.heaviside: strict >, not >=
		v_new = (1 - spike) * v_decayed + spike * p["v_reset"]
		return spike, v_new, i_decayed

	for _ in range(payload["sim_steps"]):
		hidden_current = obs @ w_in.T + b_in
		h_spikes, h_v, h_i = lif_step(hidden_current, h_v, h_i, hp)
		output_current = h_spikes @ w_out.T + b_out
		o_spikes, o_v, o_i = lif_step(output_current, o_v, o_i, op)
		spike_sum = spike_sum + o_spikes

	return spike_sum / payload["sim_steps"]


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "web", "weights.json"))
	args = parser.parse_args()
	export(args.checkpoint, args.out)


if __name__ == "__main__":
	main()
