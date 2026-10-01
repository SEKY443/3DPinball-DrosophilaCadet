#!/usr/bin/env python3
"""Exports an encoded-circuit checkpoint (agents/experiments/encoding/) to a circuit_readout JSON for
web/circuit_policy.js, plus real-engine test vectors for web/test/verify_policy.mjs.

The JSON has the same schema as agents/export_pinball_circuit.py's (version 2) plus two optional
fields that web/circuit_policy.js reads only when present, so old exports keep working unchanged:
  "encoding":    {"name": "popcode", "sigmoid_clip": 50.0, "channels": {"0": {...}, ...}} - the
                 per-cell tuning (bump centres/widths, speed thresholds/scales) from encoding.py;
                 omitted for the 'broadcast' encoding.
  "flipper_obs": "delay1" (or "raw") - the FlipperObsFilter mode the checkpoint was trained with.

Test vectors: the real native engine is driven for --test-steps decisions by the agent's greedy
action OR a random press (probability --random-press, so the flipper-state / delay1 path is
exercised). Each step records the raw StateFrame, the unfiltered obs, the delay1 state before the
step (prev_flippers, null right after a reset), the filtered obs, the per-cell input drive, the circuit activity before the
step, and the logits. Circuit and filter reset at every drain and every --life-steps decisions (one training episode =
one ball life, truncated at --max-steps).

Usage:
  python agents/experiments/encoding/export_enc.py --checkpoint agents/experiments/encoding/start_popcode_seed.pt \\
      --connectome web/connectome_444.json --out web/circuit_readout_444_popcode.json \\
      --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \\
      --test-vectors web/testdata/circuit_444_popcode_vectors.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

import torch  # noqa: E402

torch.set_num_threads(1)

from agents import train_pinball_circuit_cem as T  # noqa: E402
from agents.experiments.encoding import encoding as E  # noqa: E402
from agents.fixed_circuit_agent import DYNAMICS_GAIN, DYNAMICS_ITERATIONS, DYNAMICS_LEAK, OBS_SCALE  # noqa: E402

ACTION_LABELS = ["flipper_left", "flipper_right", "launch"]


def encoding_spec(name: str) -> dict | None:
	if name == "broadcast":
		return None
	if name != "popcode":
		raise SystemExit(f"only 'broadcast' and 'popcode' are ported to the browser, not {name!r}")
	return {
		"name": "popcode",
		"sigmoid_clip": 50.0,
		"channels": {
			str(E.CH_X): {"kind": "bumps", "centers": E.X_CENTERS.tolist(), "sigmas": [float(E.X_SIGMA)] * E.N_PER_CH},
			str(E.CH_Y): {"kind": "bumps", "centers": E.Y_CENTERS.tolist(), "sigmas": E.Y_SIGMA.tolist()},
			str(E.CH_VX): {"kind": "ds_speed", "thresholds": E.SPEED_THRESH.tolist(), "scales": E.SPEED_SCALE.tolist()},
			str(E.CH_VY): {"kind": "ds_speed", "thresholds": E.SPEED_THRESH.tolist(), "scales": E.SPEED_SCALE.tolist()},
		},
	}


def build(ckpt_path: str, connectome: str, encoding: str | None = None):
	ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
	if "encoding" in ck and encoding and ck["encoding"] != encoding:
		raise SystemExit(f"checkpoint says encoding {ck['encoding']!r}, --encoding says {encoding!r}")
	enc = ck.get("encoding") or encoding
	if enc is None:
		raise SystemExit("checkpoint has no 'encoding' field (not saved via train_enc.py) - pass --encoding explicitly")
	fobs = ck.get("flipper_obs", "raw")
	if fobs not in ("raw", "delay1"):
		raise SystemExit(f"flipper_obs {fobs!r} is not ported to the browser")
	agent = E.build_encoded_agent(connectome, int(ck["readout_dim"]), enc)
	with open(connectome) as fh:
		agent.graph_inputs = json.load(fh)["inputs"]
	T.set_flat_params(agent.decoder, np.asarray(ck["champion"], dtype=np.float32))
	agent.eval()
	return agent, enc, fobs


def payload_for(agent, enc: str, fobs: str) -> dict:
	dec = agent.decoder
	p = {
		"version": 2,
		"n_channels": len(OBS_SCALE),
		"obs_scale": list(OBS_SCALE),
		"dynamics": {"iterations": DYNAMICS_ITERATIONS, "leak": DYNAMICS_LEAK, "gain": DYNAMICS_GAIN},
		"readout_norm": {"mean": dec.norm.mean.detach().tolist(), "log_scale": dec.norm.log_scale.detach().tolist(),
		                 "clip": dec.norm.clip},
		"proj_weight": dec.proj.weight.detach().tolist(),
		"head": {"weight": dec.head.weight.detach().tolist(), "bias": dec.head.bias.detach().tolist()},
		"action_labels": ACTION_LABELS,
		"flipper_obs": fobs,
	}
	spec = encoding_spec(enc)
	if spec is not None:
		p["encoding"] = spec
	if getattr(dec.proj, "bias", None) is not None:
		p["proj_bias"] = dec.proj.bias.detach().tolist()
	return p


def write_vectors(agent, fobs: str, binary: str, steps: int, random_press: float, out_path: str, seed0: int,
                  life_steps: int = 0) -> None:
	from env_python.pinball_env import PinballEnv

	rng = np.random.default_rng(0)
	slot_of_cell = {int(c): k for k, c in enumerate(agent.brain.input_cells.tolist())}
	to_graph_order = np.array([slot_of_cell[int(c)] for c, _ in agent.graph_inputs])
	env = PinballEnv(binary_path=binary, headless=True, frame_skip=4)
	records, life, t_life = [], seed0, 0
	try:
		obs, _ = env.reset(seed=life)
		h = agent.initial_state(1)
		filt = T.FlipperObsFilter(fobs)
		with torch.no_grad():
			for _ in range(steps):
				state = env._pending_state
				prev = None if filt._prev is None else [float(v) for v in filt._prev]
				fo = np.asarray(filt(obs), dtype=np.float32)
				o = torch.as_tensor(fo).unsqueeze(0)
				# EncodedCircuitAgent orders its per-cell drive channel-major; re-order it to the
				# connectome's graph.inputs order, which is what web/circuit_policy.js uses
				drive = agent._channel_drive(o)[0].numpy()[to_graph_order].tolist()
				h_before = h[0].tolist()
				feats, h = agent.step(o, h)
				logits = agent.decoder.dist_inputs(feats)[0].tolist()
				records.append({
					"activity_before": h_before,
					"raw": {k: getattr(state, k) for k in (
						"tick", "ball_x", "ball_y", "ball_vx", "ball_vy", "flipper_left", "flipper_right",
						"ball_in_play", "tilted", "score_delta", "done", "flipper_hit", "relaunch_pending",
						"ball2_x", "ball2_y", "ball2_vx", "ball2_vy", "ball2_active")},
					"obs": [float(x) for x in obs],
					"prev_flippers": prev,
					"filtered_obs": [float(x) for x in fo],
					"input_drive": drive,
					"logits": logits,
				})
				act = np.array([logits[0] > 0 or rng.random() < random_press,
				                logits[1] > 0 or rng.random() < random_press, False], dtype=bool)
				obs, _r, term, trunc, info = env.step(act)
				t_life += 1
				if term or info["drained"] or (life_steps and t_life >= life_steps):
					life += 1
					t_life = 0
					obs, _ = env.reset(seed=life)
					h = agent.initial_state(1)
					filt.reset()
	finally:
		env.close()
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w") as f:
		json.dump({"frame_skip": 4, "flipper_obs": fobs, "steps": records}, f)
	n_prev = sum(1 for r in records if r["prev_flippers"] and any(r["prev_flippers"]))
	print(f"wrote {out_path}: {len(records)} steps over {life - seed0 + 1} lives, {n_prev} with a delayed flipper up")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--checkpoint", required=True)
	ap.add_argument("--connectome", default=os.path.join(ROOT, "web", "connectome_444.json"))
	ap.add_argument("--out", required=True)
	ap.add_argument("--encoding", default=None, help="only needed for checkpoints without an 'encoding' field")
	ap.add_argument("--binary", default=None)
	ap.add_argument("--test-vectors", default=None)
	ap.add_argument("--test-steps", type=int, default=600)
	ap.add_argument("--random-press", type=float, default=0.1)
	ap.add_argument("--seed0", type=int, default=0)
	ap.add_argument("--life-steps", type=int, default=250, help="end a life after this many decisions (like the "
	                "trainer's --max-steps truncation) so the vectors cover resets of the circuit and delay1 state; 0 = off")
	args = ap.parse_args()
	agent, enc, fobs = build(args.checkpoint, args.connectome, args.encoding)
	payload = payload_for(agent, enc, fobs)
	os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
	with open(args.out, "w") as f:
		json.dump(payload, f, indent=2)
	print(f"wrote {args.out} (encoding {enc}, flipper_obs {fobs})")
	if args.binary and args.test_vectors:
		write_vectors(agent, fobs, args.binary, args.test_steps, args.random_press, args.test_vectors, args.seed0,
		              args.life_steps)


if __name__ == "__main__":
	main()
