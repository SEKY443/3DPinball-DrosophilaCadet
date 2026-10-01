"""Build a small, FIXED, measured MaleCNS circuit for BreakoutEnv - same method and same MaleCNS
v1.0 data as agents/build_pinball_connectome.py (flyjump's approach: hand-select ~80-700 measured
cells, keep their connectome weights fixed, train only a small readout on top), adapted for
Breakout's much simpler channel set.

Channel mapping: ball_x/ball_y/ball_vx/ball_vy/paddle_x have no visual analog (this is the same
engineered-channel-injection choice pinball's builder makes) and route through `vnc_sensory`
(mechanosensory afferents), ranked by direct synaptic weight onto `vnc_motor`. `vel`/`size` DO
have a real visual analog - the ball closing in on the paddle's y-plane is a genuine looming
stimulus - so they route through LC4 (velocity, linear response) and LPLC2 (size, Gaussian
response) respectively, per the same Ache et al. 2019 (Curr. Biol.) Giant Fiber model
agents/build_pinball_connectome.py uses, ranked by weight onto `descending_neuron` (no direct
LC4/LPLC2->vnc_motor synapses in this data). Output is `vnc_motor` (paddle move-left/move-right
are motor commands).

Requires the same 3 MaleCNS v1.0 feather files agents/build_pinball_connectome.py uses - see
agents/download_malecns.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

OBS_LABELS = ["ball_x", "ball_y", "ball_vx", "ball_vy", "paddle_x", "has_laser", "vel", "size", "vel2", "size2"]
CELLS_PER_CHANNEL = 4
MAX_OUTPUTS = 16
N_BRIDGES = 32
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}

VEL_CHANNEL_LABELS = ("vel", "vel2")
SIZE_CHANNEL_LABELS = ("size", "size2")
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

    ix = np.flatnonzero(np.isin(pre, chosen_inputs) & mm)
    out_strength: dict[int, int] = {}
    for p, w in zip(post[ix], weight[ix]):
        out_strength[int(p)] = out_strength.get(int(p), 0) + int(w)
    targets = sorted(out_strength, key=lambda i: (-out_strength[i], i))[:max_outputs]
    if not targets:
        raise ValueError("no vnc_motor cells receive direct input from the chosen sensory cells")

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
        "version": "malecns-breakout-arcade-circuit-v1",
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
            f"channels ({', '.join(OBS_LABELS[t] for t in size_tiers)}), ranked by direct "
            f"synaptic weight onto descending_neuron "
            f"(no direct LC/LPLC->vnc_motor synapses exist in this data - see Ache et al. 2019, "
            f"Curr. Biol. for the linear-velocity + Gaussian-size Giant Fiber looming model this "
            f"split is based on); top {max_outputs} vnc_motor cells by direct weight from the "
            f"chosen mechanosensory inputs; top {n_bridges} two-hop bridge cells by minimum "
            "incoming/outgoing contact strength. Ties by body ID. No game outcomes used."
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
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "arcade"))
    parser.add_argument("--cells-per-channel", type=int, default=CELLS_PER_CHANNEL)
    parser.add_argument("--max-outputs", type=int, default=MAX_OUTPUTS)
    parser.add_argument("--n-bridges", type=int, default=N_BRIDGES)
    args = parser.parse_args()
    manifest = build(Path(args.data_dir), Path(args.out_dir), args.cells_per_channel, args.max_outputs, args.n_bridges)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
