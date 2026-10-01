#!/usr/bin/env python3
"""Thin wrapper around agents/train_pinball_circuit_cem.py that runs the circuit agent under a sensory
encoding from agents/experiments/encoding/encoding.py. The trainer file itself is NOT modified.

Extra flags: --encoding NAME (default 'broadcast' = production behaviour) and --target-objects
NAME,NAME,... (restricts the trainer's --active-upper-weight bonus to those table_map objects, e.g. the
multiplier bank a_targ7,a_targ8,a_targ9). Everything else is passed through to the trainer unchanged. How it works: the encoding name is put in the environment
(PINBALL_ENCODING) so that spawned worker processes, which re-import this module as __mp_main__,
apply the same patch: `train_pinball_circuit_cem.build_agent` returns an EncodedCircuitAgent for
agent kind 'circuit'. Saved checkpoints get an extra "encoding" field; --resume refuses a
checkpoint whose "encoding" differs. Imitation seeding inside the trainer (heuristic_seed) builds a
plain FixedCircuitAgent, so with a non-broadcast encoding --resume (or --no-seed-heuristic) is
required.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

from agents import train_pinball_circuit_cem as T  # noqa: E402
from agents.experiments.encoding.encoding import Encoder, build_encoded_agent  # noqa: E402

ENV_KEY = "PINBALL_ENCODING"


def _patch(encoding: str) -> None:
	Encoder(encoding)  # validates the name
	original = T.build_agent

	def build_agent(kind: str, connectome_path: str, readout_dim: int):
		if kind == "circuit":
			return build_encoded_agent(connectome_path, readout_dim, encoding)
		return original(kind, connectome_path, readout_dim)

	T.build_agent = build_agent


def _patch_save(encoding: str) -> None:
	import torch

	original = torch.save

	def save(obj, f, *a, **kw):
		if isinstance(obj, dict) and "readout_dim" in obj and "champion" in obj:
			obj = dict(obj, encoding=encoding)
		return original(obj, f, *a, **kw)

	T.torch.save = save


TARGET_ENV_KEY = "PINBALL_TARGET_OBJECTS"


def _patch_targets(names: str) -> None:
	"""Restrict the trainer's --active-upper-weight bonus to the named table objects (comma-separated
	table_map names, e.g. the multiplier bank 'a_targ7,a_targ8,a_targ9') instead of every upper object."""
	wanted = [n.strip() for n in names.split(",") if n.strip()]

	def load_target_object_ids(table_map_path: str, upper_max_y: float = T.UPPER_MAX_Y) -> frozenset:
		import json

		with open(table_map_path, encoding="utf-8") as f:
			by_name = {o["name"]: o["id"] for o in json.load(f)["objects"]}
		missing = [n for n in wanted if n not in by_name]
		if missing:
			raise SystemExit(f"--target-objects: unknown table_map names {missing}")
		return frozenset(by_name[n] for n in wanted)

	T.load_upper_object_ids = load_target_object_ids


# Runs in the parent (after main() sets the variables) and in every spawned worker.
if os.environ.get(ENV_KEY):
	_patch(os.environ[ENV_KEY])
if os.environ.get(TARGET_ENV_KEY):
	_patch_targets(os.environ[TARGET_ENV_KEY])


def main() -> None:
	argv = sys.argv[1:]
	encoding = "broadcast"
	if "--encoding" in argv:
		i = argv.index("--encoding")
		encoding = argv[i + 1]
		del argv[i:i + 2]
	Encoder(encoding)
	if encoding != "broadcast" and "--resume" not in argv and "--no-seed-heuristic" not in argv:
		raise SystemExit("--encoding other than 'broadcast' needs --resume (or --no-seed-heuristic): the trainer's "
		                 "heuristic_seed builds an un-encoded FixedCircuitAgent")
	if "--resume" in argv:
		import torch

		ck = torch.load(argv[argv.index("--resume") + 1], weights_only=False)
		if ck.get("encoding", "broadcast") != encoding:
			raise SystemExit(f"checkpoint encoding {ck.get('encoding', 'broadcast')!r} != --encoding {encoding!r}")
	if "--target-objects" in argv:
		i = argv.index("--target-objects")
		targets = argv[i + 1]
		del argv[i:i + 2]
		if "--active-upper-weight" not in argv:
			raise SystemExit("--target-objects only affects the --active-upper-weight bonus; pass that too")
		os.environ[TARGET_ENV_KEY] = targets
		_patch_targets(targets)
		print(f"train_enc: active bonus restricted to {targets}", flush=True)
	os.environ[ENV_KEY] = encoding
	_patch(encoding)
	_patch_save(encoding)
	sys.argv = [sys.argv[0]] + argv
	print(f"train_enc: sensory encoding {encoding!r}", flush=True)
	T.main()


if __name__ == "__main__":
	main()
