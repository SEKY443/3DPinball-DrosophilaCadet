#!/usr/bin/env python3
"""Exports a trained FixedCircuitAgent checkpoint to a circuit_readout JSON for web/circuit_policy.js.
The fixed circuit itself is not re-exported here since it never changes after
agents/build_pinball_connectome.py wrote the connectome JSON; web/circuit_policy.js loads both
files.

Supports two checkpoint formats:
  - PPO (agents/train_pinball_circuit.py): ckpt["agent"] is a state dict, loaded via
    agent.load_state_dict.
  - CMA/CEM (agents/train_pinball_circuit_cem.py): ckpt["champion"] is a flat float32 parameter
    vector for agent.decoder, loaded via set_flat_params. ckpt["agent"] here is a plain string
    ("circuit"/"mlp"), not a state dict - format is told apart by that difference, not a version
    field neither checkpoint writer stamps.

Usage:
    python agents/export_pinball_circuit.py --checkpoint agents/checkpoints/pinball_circuit.pt \\
        --connectome web/connectome.json --out web/circuit_readout.json \\
        --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
        --test-vectors web/testdata/circuit_vectors.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.fixed_circuit_agent import (  # noqa: E402
	DYNAMICS_GAIN,
	DYNAMICS_ITERATIONS,
	DYNAMICS_LEAK,
	OBS_SCALE,
	FixedCircuitAgent,
)
from agents.train_pinball_circuit_cem import set_flat_params  # noqa: E402

ACTION_LABELS = ["flipper_left", "flipper_right", "launch"]


def _build_agent_from_checkpoint(ckpt: dict, connectome_path: str) -> FixedCircuitAgent:
	"""Tells the two checkpoint formats apart by shape, not a version field: a PPO checkpoint's
	ckpt["agent"] is itself a state dict (has .items()/tensor values); a CMA/CEM checkpoint's
	ckpt["agent"] is a plain string ("circuit") and the real parameters live in ckpt["champion"]."""
	agent_field = ckpt.get("agent")
	if isinstance(agent_field, dict):
		readout_dim = agent_field["decoder.proj.weight"].shape[0]
		agent = FixedCircuitAgent.from_json(connectome_path, readout_dim=readout_dim)
		agent.load_state_dict(agent_field)
	elif "champion" in ckpt:
		readout_dim = ckpt["readout_dim"]
		agent = FixedCircuitAgent.from_json(connectome_path, readout_dim=readout_dim)
		set_flat_params(agent.decoder, np.asarray(ckpt["champion"], dtype=np.float32))
	else:
		raise ValueError(f"unrecognized checkpoint format: keys={sorted(ckpt.keys())}")
	agent.eval()
	return agent


def export(checkpoint_path: str, connectome_path: str, out_path: str,
           binary_path: str | None = None, test_vectors_path: str | None = None,
           test_steps: int = 200, frame_skip: int = 4) -> None:
	graph = json.loads(open(connectome_path).read())
	ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	agent = _build_agent_from_checkpoint(ckpt, connectome_path)

	dec = agent.decoder
	payload = {
		"version": 2,
		"n_channels": agent.brain.n_channels,
		"obs_scale": list(OBS_SCALE),
		"dynamics": {"iterations": DYNAMICS_ITERATIONS, "leak": DYNAMICS_LEAK, "gain": DYNAMICS_GAIN},
		"readout_norm": {
			"mean": dec.norm.mean.detach().tolist(),
			"log_scale": dec.norm.log_scale.detach().tolist(),
			"clip": dec.norm.clip,
		},
		"proj_weight": dec.proj.weight.detach().tolist(),  # (readout_dim, n_readout)
		"head": {
			"weight": dec.head.weight.detach().tolist(),  # (3, readout_dim)
			"bias": dec.head.bias.detach().tolist(),
		},
		"action_labels": ACTION_LABELS,
	}
	# proj has no bias for every checkpoint this project has actually produced (ActionDecoder's
	# calibrate() only installs a biased proj when fit against explicit targets, which this
	# training pipeline never does) - included when present anyway so the JSON schema doesn't
	# silently go stale if that ever changes; web/circuit_policy.js applies it when it exists.
	if getattr(dec.proj, "bias", None) is not None:
		payload["proj_bias"] = dec.proj.bias.detach().tolist()

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(payload, f, indent=2)
	print(f"wrote {out_path}")

	_verify_bit_for_bit(agent, graph, payload)

	if binary_path and test_vectors_path:
		_write_test_vectors(agent, binary_path, frame_skip, test_steps, test_vectors_path)


def _verify_bit_for_bit(agent: FixedCircuitAgent, graph: dict, payload: dict) -> None:
	"""Runs a fixed random rollout through both the live PyTorch agent and a pure-NumPy
	re-implementation driven only by graph (the connectome JSON) + payload (the readout JSON) -
	this is what web/circuit_policy.js must match."""
	torch.manual_seed(0)
	batch, steps = 8, 20
	obs_seq = torch.randn(steps, batch, len(payload["obs_scale"])) * torch.tensor(payload["obs_scale"]) * 0.5

	with torch.no_grad():
		h = agent.initial_state(batch)
		expected = []
		for t in range(steps):
			feats, h = agent.step(obs_seq[t], h)
			expected.append(agent.decoder.dist_inputs(feats).numpy())
	expected = np.stack(expected)

	actual = _reference_forward(graph, payload, obs_seq.numpy())

	max_diff = float(np.max(np.abs(expected - actual)))
	print(f"reference re-implementation max abs diff vs PyTorch agent (logits): {max_diff:.3e}")
	# This is a 20-step synthetic rollout stress test (random inputs, not real gameplay), so it
	# only needs to catch a real logic bug, not match float32 bit-for-bit - float32 accumulation
	# order (torch's index_add_ vs numpy's add.at) drifts more on the 444-node/7267-edge circuit
	# than the 108-node one after 60 compounded tanh iterations. The real <1e-5 bar lives in
	# web/test/verify_policy.mjs, which checks single real-engine steps (no compounding) against
	# circuit_policy.js.
	assert max_diff < 1e-3, "reference forward pass does not match the trained agent closely enough"


def _reference_forward(graph: dict, payload: dict, obs_seq: np.ndarray) -> np.ndarray:
	"""Pure NumPy re-implementation of FixedCircuitAgent.forward, using only fields from the
	connectome JSON and the readout JSON - what web/circuit_policy.js must match. Kept in
	float32 throughout (not float64): torch's own accumulation is float32, and on the 444-node/
	7267-edge circuit the two accumulation orders (index_add_ vs np.add.at) drift enough in
	float64-vs-float32 rounding alone to matter after 3 recurrent iterations x 20 steps."""
	n = len(graph["nodes"])
	signs = np.array([node["sign"] for node in graph["nodes"]], dtype=np.float32)
	pre = np.array([e[0] for e in graph["edges"]], dtype=np.int64)
	post = np.array([e[1] for e in graph["edges"]], dtype=np.int64)
	contacts = np.array([e[2] for e in graph["edges"]], dtype=np.float32)

	totals = np.zeros(n, dtype=np.float32)
	np.add.at(totals, post, contacts * np.abs(signs[pre]))
	w = np.where(totals[post] > 0, contacts * signs[pre] / np.maximum(totals[post], 1e-12), 0.0).astype(np.float32)

	input_cells = np.array([c for c, _ in graph["inputs"]], dtype=np.int64)
	input_channels = np.array([c for _, c in graph["inputs"]], dtype=np.int64)
	output_idx = np.array(graph["outputs"], dtype=np.int64)

	obs_scale = np.array(payload["obs_scale"], dtype=np.float32)
	dyn = payload["dynamics"]
	steps, batch, _ = obs_seq.shape

	activity = np.zeros((batch, n), dtype=np.float32)
	mean = np.array(payload["readout_norm"]["mean"], dtype=np.float32)
	log_scale = np.array(payload["readout_norm"]["log_scale"], dtype=np.float32)
	clip = payload["readout_norm"]["clip"]
	proj_w = np.array(payload["proj_weight"], dtype=np.float32)  # (readout_dim, n_readout)
	proj_b = np.array(payload["proj_bias"], dtype=np.float32) if "proj_bias" in payload else None
	head_w = np.array(payload["head"]["weight"], dtype=np.float32)  # (3, readout_dim)
	head_b = np.array(payload["head"]["bias"], dtype=np.float32)

	obs_seq = obs_seq.astype(np.float32)
	logits_seq = []
	for t in range(steps):
		channel_drive = np.clip(obs_seq[t] / obs_scale, -1.0, 1.0)
		drive = np.zeros((batch, n), dtype=np.float32)
		drive[:, input_cells] = channel_drive[:, input_channels]
		for _ in range(dyn["iterations"]):
			msg = np.zeros((batch, n), dtype=np.float32)
			np.add.at(msg.T, post, (activity[:, pre] * w).T)
			activity = (1 - dyn["leak"]) * activity + dyn["leak"] * np.tanh(drive + dyn["gain"] * msg)

		raw = activity[:, output_idx]
		normed = np.clip((raw - mean) * np.exp(-log_scale), -clip, clip)
		feats = normed @ proj_w.T
		if proj_b is not None:
			feats = feats + proj_b
		logits = feats @ head_w.T + head_b
		logits_seq.append(logits)

	return np.stack(logits_seq)


def _write_test_vectors(agent: FixedCircuitAgent, binary_path: str, frame_skip: int,
                         steps: int, out_path: str) -> None:
	"""Drives the real native engine for `steps` decisions (env_python/pinball_env.py's
	PinballEnv, matching training's frame_skip) and records, per decision: the raw StateFrame
	fields (for web/observation.js's port of _state_to_obs), the resulting 15-value observation,
	the circuit activity BEFORE this step's dynamics update (activity_before - lets a verifier
	teacher-force JS to the same starting state instead of compounding its own float32 rounding
	across hundreds of recurrent steps, which this project measured to diverge chaotically: two
	independently-summed float32 implementations of the identical forward pass, differing only in
	edge summation order, land within 1e-6 on a single step but up to 8e-3 apart by step 91 of a
	continuously-carried 200-step rollout - a real sensitivity of this gain=1.4 recurrent network,
	not a JS porting bug), and the trained agent's logits - what web/test/verify_policy.mjs
	replays through circuit_policy.js/observation.js to check both JS ports against this same
	live PyTorch agent."""
	from env_python.pinball_env import PinballEnv

	env = PinballEnv(binary_path=binary_path, headless=True, frame_skip=frame_skip)
	records = []
	try:
		obs, info = env.reset(seed=0)
		h = agent.initial_state(1)
		with torch.no_grad():
			for i in range(steps):
				state = env._pending_state  # the exact State that produced `obs` below
				h_before = h[0].tolist()  # circuit activity BEFORE this step's dynamics update
				obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
				feats, h = agent.step(obs_t, h)
				logits = agent.decoder.dist_inputs(feats)[0].tolist()
				records.append({
					"activity_before": h_before,
					"raw": {
						"tick": state.tick,
						"ball_x": state.ball_x, "ball_y": state.ball_y,
						"ball_vx": state.ball_vx, "ball_vy": state.ball_vy,
						"flipper_left": state.flipper_left, "flipper_right": state.flipper_right,
						"ball_in_play": state.ball_in_play, "tilted": state.tilted,
						"score_delta": state.score_delta, "done": state.done,
						"flipper_hit": state.flipper_hit, "relaunch_pending": state.relaunch_pending,
						"ball2_x": state.ball2_x, "ball2_y": state.ball2_y,
						"ball2_vx": state.ball2_vx, "ball2_vy": state.ball2_vy,
						"ball2_active": state.ball2_active,
					},
					"obs": [float(x) for x in obs],
					"logits": logits,
				})
				action = np.array([logits[0] > 0, logits[1] > 0, False], dtype=bool)
				obs, reward, terminated, truncated, info = env.step(action)
				if terminated:
					obs, info = env.reset(seed=i + 1)
					h = agent.initial_state(1)
	finally:
		env.close()

	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w") as f:
		json.dump({"frame_skip": frame_skip, "steps": records}, f)
	print(f"wrote {out_path} ({len(records)} real-engine decision steps)")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--connectome", default=os.path.join(os.path.dirname(__file__), "..", "web", "connectome.json"))
	parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "web", "circuit_readout.json"))
	parser.add_argument("--binary", default=None, help="native engine binary - when given (with --test-vectors), records a real-engine test-vector file for web/test/verify_policy.mjs")
	parser.add_argument("--test-vectors", default=None, help="output path for the real-engine test-vector JSON (see --binary)")
	parser.add_argument("--test-steps", type=int, default=200)
	parser.add_argument("--frame-skip", type=int, default=4, help="must match the frame_skip the checkpoint was trained with (agents/train_pinball_circuit_cem.py's default is 4)")
	args = parser.parse_args()
	export(args.checkpoint, args.connectome, args.out, args.binary, args.test_vectors, args.test_steps, args.frame_skip)


if __name__ == "__main__":
	main()
