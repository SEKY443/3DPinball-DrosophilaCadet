#!/usr/bin/env python3
"""Export a PPO checkpoint (agents/train_pinball_circuit.py, nfly.rl.simple.common.save_checkpoint
format: {"agent": agent.state_dict(), "returns": [...], "config": {...}}) to the CMA-format
checkpoint agents/train_pinball_circuit_cem.py and its evaluation tools expect
({"champion": flat np.float32 vector, "mean", "sigma", "champion_fitness", "generation",
"readout_dim", "agent"}), so a PPO-trained decoder can be evaluated/compared with the CMA track's
own tools, or fed back into CMA-ES as a seed.

Only agent.decoder's parameters are exported - agent.value (PPO's critic) has no CEM/CMA-ES
counterpart, same reason train_pinball_circuit_cem.py's docstring gives for dropping it.

Usage:
	python agents/export_ppo_to_cma.py --ppo /tmp/ppo_smoke.pt --connectome web/connectome.json \\
		--readout-dim 32 --out /tmp/ppo_smoke_cma.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "nfly"))

from agents.fixed_circuit_agent import FixedCircuitAgent  # noqa: E402
from agents.train_pinball_circuit_cem import get_flat_params  # noqa: E402


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--ppo", required=True, help="PPO checkpoint saved by agents/train_pinball_circuit.py")
	parser.add_argument("--connectome", required=True)
	parser.add_argument("--readout-dim", type=int, required=True)
	parser.add_argument("--out", required=True)
	args = parser.parse_args()

	ckpt = torch.load(args.ppo, map_location="cpu", weights_only=False)
	agent = FixedCircuitAgent.from_json(args.connectome, readout_dim=args.readout_dim)
	agent.load_state_dict(ckpt["agent"])

	champion = get_flat_params(agent.decoder)
	torch.save({
		"champion": champion,
		"mean": champion,
		"sigma": np.zeros_like(champion),
		"champion_fitness": float("nan"),
		"generation": ckpt.get("update", 0),
		"readout_dim": args.readout_dim,
		"agent": "circuit",
	}, args.out)
	print(f"exported {len(champion)}-param champion from {args.ppo} -> {args.out}", flush=True)


if __name__ == "__main__":
	main()
