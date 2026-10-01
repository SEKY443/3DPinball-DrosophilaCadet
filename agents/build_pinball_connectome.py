"""Build a bounded, measured MaleCNS circuit for PinballEnv - the same method as
cobanov/flyjump's build-connectome.py (a small, FIXED, anatomically-selected subgraph feeding a
small trained readout), pointed at pinball instead of Chrome Dino.

Why this exists alongside train_fly.py: nfly's ConnectomeRNN is the full ~139k-neuron
connectome with millions of LEARNED per-edge gains - built for offline research-scale training,
not for shipping to a browser (the exported weights alone would be tens to hundreds of MB, and
reproducing its chunked sparse dynamics bit-for-bit in JS is a real undertaking). flyjump's
approach instead hand-selects ~80 measured cells and keeps their connectome-derived weights
FIXED (only a ~200-parameter downstream readout is trained), which is genuinely small enough to
export as JSON and run as a plain forward pass in the browser. This script is that selection
step for our task.

Channel mapping (the input population and grouping) is an engineered choice, same as flyjump's
own disclaimer that their channel-to-cell-type mapping is engineered, not a biological receptive
field:
  - Most of PinballEnv's channels have no visual input, so flyjump's lobula visual-motion cell
	types (LC4, LPLC2, ...) have no pinball analog for THEM. We use `vnc_sensory` (ventral nerve
	cord sensory/mechanosensory afferents - leg/wing/haltere periphery) as the input population
	for those: this is a mechanosensory control task (press a button at the right moment), which
	sits closer to proprioception than vision.
  - UPDATE: four channels (`vel1`, `size1`, `vel2`, `size2`) DO have a real visual analog after
	all - a ball closing in on the flipper zone is "something approaching, faster as it nears" in
	exactly the shape LC4/LPLC2 looming detectors respond to (flyjump's own oncoming-cactus
	signal). Added specifically to fix a real gap: the engine can spawn a genuine second ball
	(multiball, see TPinballTable.cpp's MultiballFlag on real DEMO.DAT tables) that a purely
	mechanosensory, single-ball-position circuit had no way to perceive at all - see
	env_python/pinball_env.py's `_velocity_signal`/`_size_signal`. Split by real cell TYPE rather
	than pooled: per Ache et al. 2019 (Curr. Biol.), the real Giant Fiber's looming response sums
	a linear function of LC4's velocity signal and a Gaussian function of LPLC2's size signal -
	two anatomically and functionally distinct neuron types, not one interchangeable "loom"
	population - so `vel1`/`vel2` draw only from VEL_CELL_TYPES (LC4) and `size1`/`size2` draw
	only from SIZE_CELL_TYPES (LPLC2), both instead of vnc_sensory. There's no direct LC4/LPLC2
	->vnc_motor synapse in the data (confirmed empirically: 0 direct edges) - real fly anatomy
	routes visual projection neurons through descending neurons first - so these are ranked by
	weight onto `descending_neuron` instead, then connected to the same vnc_motor outputs via the
	existing bridge-finding step below (confirmed empirically: real 2-hop loom->X->motor paths
	exist in the data for the pooled LC4+LPLC set).
  - `vnc_motor` (leg/wing motor neurons) is the output population, rather than flyjump's
	`descending_neuron`: our 3 actions ARE motor commands (flipper press, launch), so reading
	out directly from measured motor neurons is a more literal anatomical fit than routing
	through the higher-level descending layer Dino's more abstract jump/duck decision needed.
  - We do not claim the vnc_sensory `type` codes (SNta.., SNch.., ...) have specific known
	functional identities - unlike flyjump's named, well-characterized visual cell types, these
	MaleCNS naming codes are not something this script asserts biological meaning for. Input
	cells are therefore grouped into channels by connectivity RANK TIER (channel 0 = strongest
	4 sensory cells by direct synaptic weight onto vnc_motor, channel 1 = next 4, ...), not by
	named subtype, so no unverified claim is embedded in the selection.

Requires the same 3 MaleCNS v1.0 files load_malecns() uses - see agents/download_malecns.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

OBS_LABELS = ["ball_x", "ball_y", "ball_vx", "ball_vy", "vel1_L", "vel1_C", "vel1_R", "size1_L", "size1_C", "size1_R", "vel2", "size2", "flipper_left", "flipper_right", "tilted"]
N_CHANNELS = len(OBS_LABELS)
CELLS_PER_CHANNEL = 4
MAX_OUTPUTS = 16
N_BRIDGES = 32
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}

# vel1/vel2 and size1/size2 (see env_python/pinball_env.py's _velocity_signal/_size_signal) ARE
# a pinball analog of flyjump's oncoming-cactus signal after all: a ball closing in on the
# flipper zone is "something approaching, faster as it nears", the exact shape LC4/LPLC-family
# visual projection neurons detect. Routed through real looming-detector cell types instead of
# vnc_sensory, unlike the other 7 channels - see the module docstring's original (now partly
# superseded) disclaimer. Split by TYPE rather than pooled, per Ache et al. 2019 (Current
# Biology): the real Giant Fiber looming response is a linear function of LC4's velocity signal
# plus a Gaussian function of LPLC2's size signal - two neurons with different tuning, not one
# interchangeable "loom" population - so vel1/vel2 draw only from LC4 and size1/size2 draw only
# from LPLC2, instead of both channels pooling all four LC/LPLC types together as before.
VEL_CHANNEL_LABELS = ("vel1_L", "vel1_C", "vel1_R", "vel2")
SIZE_CHANNEL_LABELS = ("size1_L", "size1_C", "size1_R", "size2")
VEL_CELL_TYPES = ("LC4",)
SIZE_CELL_TYPES = ("LPLC2",)


def sha256_of(path: Path) -> str:
	return hashlib.sha256(path.read_bytes()).hexdigest()


def build(data_dir: Path, out_dir: Path, cells_per_channel: int = CELLS_PER_CHANNEL,
          max_outputs: int = MAX_OUTPUTS, n_bridges: int = N_BRIDGES) -> dict:
	ann_rows = feather.read_table(data_dir / "body-annotations.feather", columns=["bodyId", "type", "superclass"]).to_pylist()
	ann = {r["bodyId"]: r for r in ann_rows}

	e = feather.read_table(data_dir / "connectome-weights.feather")
	pre, post, weight = e["body_pre"].to_numpy(), e["body_post"].to_numpy(), e["weight"].to_numpy()

	nt = {r["body"]: r["consensus_nt"] for r in feather.read_table(data_dir / "body-neurotransmitters.feather", columns=["body", "consensus_nt"]).to_pylist()}

	sensory = np.array([i for i, r in ann.items() if r["superclass"] == "vnc_sensory"])
	motor = np.array([i for i, r in ann.items() if r["superclass"] == "vnc_motor"])
	descending = np.array([i for i, r in ann.items() if r["superclass"] == "descending_neuron"])
	vel_cells = np.array([i for i, r in ann.items() if r["type"] in VEL_CELL_TYPES])
	size_cells = np.array([i for i, r in ann.items() if r["type"] in SIZE_CELL_TYPES])

	mech_tiers = [i for i, label in enumerate(OBS_LABELS) if label not in VEL_CHANNEL_LABELS + SIZE_CHANNEL_LABELS]
	vel_tiers = [i for i, label in enumerate(OBS_LABELS) if label in VEL_CHANNEL_LABELS]
	size_tiers = [i for i, label in enumerate(OBS_LABELS) if label in SIZE_CHANNEL_LABELS]

	# Mechanosensory channels: rank vnc_sensory cells by direct total synapse weight onto
	# vnc_motor, then split the top len(mech_tiers) * cells_per_channel into rank-tier groups,
	# one group per channel - same method as before this circuit had visual channels at all.
	mm = np.isin(post, motor)
	ix = np.flatnonzero(np.isin(pre, sensory) & mm)
	strength: dict[int, int] = {}
	for p, w in zip(pre[ix], weight[ix]):
		strength[int(p)] = strength.get(int(p), 0) + int(w)
	ranked_mech = sorted(strength, key=lambda i: (-strength[i], i))
	n_mech = len(mech_tiers) * cells_per_channel
	if len(ranked_mech) < n_mech:
		raise ValueError(f"only {len(ranked_mech)} vnc_sensory cells connect to vnc_motor, need {n_mech}")
	chosen_mech = ranked_mech[:n_mech]

	# Visual channels: rank VEL_CELL_TYPES (LC4) and SIZE_CELL_TYPES (LPLC2) cells SEPARATELY by
	# direct synapse weight onto descending_neuron - there are zero direct LC4/LPLC->vnc_motor
	# synapses in this data (confirmed empirically), matching real fly anatomy where visual
	# projection neurons route through descending neurons before reaching motor circuits. Split
	# by type rather than pooled (see VEL_CELL_TYPES/SIZE_CELL_TYPES comment above) - LC4 only
	# feeds vel1/vel2, LPLC2 only feeds size1/size2. The bridge-finding step below is what
	# actually connects these to the chosen vnc_motor outputs (confirmed empirically for the
	# pooled LC4+LPLC set: real 2-hop loom->X->motor paths exist in the data).
	dm = np.isin(post, descending)

	def _rank_by_weight_onto(cells: np.ndarray) -> list[int]:
		ix = np.flatnonzero(np.isin(pre, cells) & dm)
		s: dict[int, int] = {}
		for p, w in zip(pre[ix], weight[ix]):
			s[int(p)] = s.get(int(p), 0) + int(w)
		return sorted(s, key=lambda i: (-s[i], i))

	ranked_vel = _rank_by_weight_onto(vel_cells)
	n_vel = len(vel_tiers) * cells_per_channel
	if len(ranked_vel) < n_vel:
		raise ValueError(f"only {len(ranked_vel)} {VEL_CELL_TYPES} cells connect to descending_neuron, need {n_vel}")
	chosen_vel = ranked_vel[:n_vel]

	ranked_size = _rank_by_weight_onto(size_cells)
	n_size = len(size_tiers) * cells_per_channel
	if len(ranked_size) < n_size:
		raise ValueError(f"only {len(ranked_size)} {SIZE_CELL_TYPES} cells connect to descending_neuron, need {n_size}")
	chosen_size = ranked_size[:n_size]

	chosen_inputs = chosen_mech + chosen_vel + chosen_size
	inputs = (
		[(cell, tier) for k, tier in enumerate(mech_tiers) for cell in chosen_mech[k * cells_per_channel:(k + 1) * cells_per_channel]]
		+ [(cell, tier) for k, tier in enumerate(vel_tiers) for cell in chosen_vel[k * cells_per_channel:(k + 1) * cells_per_channel]]
		+ [(cell, tier) for k, tier in enumerate(size_tiers) for cell in chosen_size[k * cells_per_channel:(k + 1) * cells_per_channel]]
	)

	# Outputs: vnc_motor cells ranked by direct synaptic weight from the chosen inputs.
	ix = np.flatnonzero(np.isin(pre, chosen_inputs) & mm)
	out_strength: dict[int, int] = {}
	for p, w in zip(post[ix], weight[ix]):
		out_strength[int(p)] = out_strength.get(int(p), 0) + int(w)
	targets = sorted(out_strength, key=lambda i: (-out_strength[i], i))[:max_outputs]
	if not targets:
		raise ValueError("no vnc_motor cells receive direct input from the chosen sensory cells")

	# Bridges: strongest two-hop cells by min(incoming from inputs, outgoing to targets).
	im = np.isin(pre, chosen_inputs)
	om = np.isin(post, targets)
	u, inv = np.unique(post[im], return_inverse=True)
	incoming = dict(zip((int(x) for x in u), np.bincount(inv, weights=weight[im])))
	u, inv = np.unique(pre[om], return_inverse=True)
	outgoing = dict(zip((int(x) for x in u), np.bincount(inv, weights=weight[om])))
	excluded = set(chosen_inputs) | set(targets)
	bridges = sorted(
		(i for i in incoming.keys() & outgoing.keys() if i in ann and i not in excluded),
		key=lambda i: (-min(incoming[i], outgoing[i]), i),
	)[:n_bridges]

	ids = sorted(set(chosen_inputs) | set(targets) | set(bridges))
	idx = {i: j for j, i in enumerate(ids)}

	mask = np.isin(pre, ids) & np.isin(post, ids)
	edges = sorted([[idx[int(a)], idx[int(b)], int(w)] for a, b, w in zip(pre[mask], post[mask], weight[mask])])

	nodes = [
		{
			"id": i,
			"type": ann[i]["type"],
			"nt": nt.get(i),
			"sign": SIGN.get(nt.get(i), 0),
			"role": "input" if i in chosen_inputs else ("output" if i in targets else "interneuron"),
		}
		for i in ids
	]
	graph = {
		"version": "malecns-pinball-circuit-v1",
		"nodes": nodes,
		"edges": edges,
		"inputs": [[idx[i], c] for i, c in inputs],
		"outputs": [idx[i] for i in targets],
		"channels": OBS_LABELS,
	}

	out_dir.mkdir(parents=True, exist_ok=True)
	graph_path = out_dir / "connectome.json"
	graph_path.write_text(json.dumps(graph, separators=(",", ":")) + "\n")

	manifest = {
		"dataset": "FlyEM MaleCNS v1.0, min confidence 0.5",
		"license": "CC BY 4.0",
		"source": "https://male-cns.janelia.org/download/",
		"nodes": len(ids),
		"edges": len(edges),
		"synapticContacts": sum(e[2] for e in edges),
		"inputCells": len(inputs),
		"readoutCells": len(targets),
		"graphSha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
		"sources": {
			"body-annotations": sha256_of(data_dir / "body-annotations.feather"),
			"connectome-weights": sha256_of(data_dir / "connectome-weights.feather"),
			"body-neurotransmitters": sha256_of(data_dir / "body-neurotransmitters.feather"),
		},
		"selection": (
			f"{cells_per_channel} vnc_sensory cells per rank tier for the {len(mech_tiers)} "
			f"mechanosensory channels ({', '.join(OBS_LABELS[t] for t in mech_tiers)}), ranked "
			f"by direct synaptic weight onto vnc_motor; {cells_per_channel} {'/'.join(VEL_CELL_TYPES)} "
			f"cells per rank tier for the {len(vel_tiers)} velocity channels "
			f"({', '.join(OBS_LABELS[t] for t in vel_tiers)}) and {cells_per_channel} "
			f"{'/'.join(SIZE_CELL_TYPES)} cells per rank tier for the {len(size_tiers)} size "
			f"channels ({', '.join(OBS_LABELS[t] for t in size_tiers)}), ranked by direct synaptic "
			f"weight onto descending_neuron (no direct LC/LPLC->vnc_motor synapses exist in this "
			f"data - see Ache et al. 2019, Curr. Biol. for the linear-velocity + Gaussian-size "
			f"Giant Fiber looming model this split is based on); "
			f"top {max_outputs} vnc_motor cells by direct weight from the chosen mechanosensory "
			f"inputs; top {n_bridges} two-hop bridge cells (from either input population) by "
			"minimum incoming/outgoing contact strength. Ties by body ID. No game outcomes used."
		),
		"assumptions": (
			"Engineered channel injection (rank tier, not named cell subtype identity). "
			"Simplified signed, normalized, leaky tanh rate units. Acetylcholine +1, "
			"GABA/glutamate -1; unknown and modulatory transmitters 0. Not a physiological or "
			"whole-brain model."
		),
	}
	(out_dir / "connectome_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
	return manifest


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
	parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "web"))
	parser.add_argument("--cells-per-channel", type=int, default=CELLS_PER_CHANNEL, help="vnc_sensory input cells per observation channel")
	parser.add_argument("--max-outputs", type=int, default=MAX_OUTPUTS, help="vnc_motor readout cells")
	parser.add_argument("--n-bridges", type=int, default=N_BRIDGES, help="two-hop interneuron bridge cells")
	args = parser.parse_args()
	manifest = build(Path(args.data_dir), Path(args.out_dir), args.cells_per_channel, args.max_outputs, args.n_bridges)
	print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
	main()
