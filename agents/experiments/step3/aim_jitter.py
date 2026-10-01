#!/usr/bin/env python3
"""Does the champion AIM at valuable objects, or hit them by accident? Counterfactual timing jitter.

For each life (seed) the champion is played once (closed loop) and every flipper press that
produced a flipper contact is found. For each such press the life is replayed from reset (the engine
is deterministic per seed + action sequence) with only that press's held segment shifted by
delta in --deltas decision steps; before the shifted segment the recorded actions are replayed,
after it the policy runs closed loop again (its circuit state is stepped throughout).

Outcome per (press, delta):
  - "no_contact": no flipper contact in the shifted press window - counted separately, NOT as 0,
  - otherwise the SHOT VALUE: multiplier-free points (hit_base_points) of upper-playfield objects
    (table_map center y < 6, see train_pinball_circuit_cem.load_upper_object_ids) from that contact
    until the ball falls back (y >= --fall-y moving down), a new contact, drain, or --max-window.
The real shot value uses the identical definition on the unperturbed run. Reported: fraction of
presses where real > median(jittered-with-contact), ties separately, with a Wilson 95% CI.

Usage:
	python agents/experiments/step3/aim_jitter.py --episodes 24 --seed0 7000 --workers 2
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

_CFG: dict = {}


def _init(cfg: dict) -> None:
	import torch

	from agents import train_pinball_circuit_cem as T

	_CFG.update(cfg)
	T._worker_init(cfg["binary"], cfg["connectome"], cfg["readout_dim"], cfg["frame_skip"], cfg["max_steps"],
	               "circuit", "life", None, 0.0)
	ckpt = torch.load(cfg["policy"], weights_only=False)
	T.set_flat_params(T._WORKER_AGENT.decoder, np.asarray(ckpt["champion"], dtype=np.float32))
	_CFG["upper"] = T.load_upper_object_ids(cfg["table_map"])


def _run(seed: int, override: dict | None, stop_after: int | None) -> list[dict]:
	"""One life. override maps step -> forced action (np array); steps not in it use the policy.
	Returns per-step records. Stops at drain/termination or after step `stop_after`."""
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	env, agent = T._WORKER_ENV, T._WORKER_AGENT
	obs, _ = env.reset(seed=seed)
	h = agent.initial_state(1)
	recs = []
	t = 0
	done = False
	with torch.no_grad():
		while not done:
			obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
			a, h = agent.act(obs_t, h, greedy=True)  # always stepped so circuit state stays consistent
			action = np.asarray(a[0]).copy()
			if override is not None and t in override:
				action = override[t].copy()
			obs, _r, term, trunc, info = env.step(action)
			up = sum(b for (oid, _p, _x, _y), b in zip(info["hit_objects"], info["hit_base_points"]) if oid in _CFG["upper"])
			recs.append(dict(a=action, hit=bool(info["flipper_hit"]), x=float(obs[OBS_BALL_X]), y=float(obs[OBS_BALL_Y]),
			                 vy=float(obs[OBS_BALL_VY]), up=int(up), drained=bool(info["drained"])))
			t += 1
			done = term or trunc or info["drained"] or (stop_after is not None and t > stop_after)
	return recs


def _shot_value(recs: list[dict], c: int) -> int:
	"""Upper base value from contact step c until fall-back / next contact / drain / max window."""
	v = recs[c]["up"]
	for t in range(c + 1, min(len(recs), c + 1 + _CFG["max_window"])):
		r = recs[t]
		if r["hit"] or r["drained"]:
			break
		v += r["up"]
		if r["y"] >= _CFG["fall_y"] and r["vy"] > 0:
			break
	return v


def _presses(recs: list[dict]) -> list[tuple[int, int, int, int]]:
	"""(side, press_start, release, first_contact) for each press segment containing a contact."""
	out, seen = [], set()
	for c, r in enumerate(recs):
		if not r["hit"]:
			continue
		# Action 0 (LEFT flipper, FlipperL) sits at POSITIVE table x: the table x axis is mirrored
		# relative to the screen (plunger lane, screen-right, is at x=-7.2; a_flip1 = left flipper
		# has its AABB centre at x=+1.78 in agents/table_map.json). Verified by placing the ball at
		# x=+2/-2 and pressing each flipper (step3 follow-up). An earlier version of this script
		# used the mirrored mapping and jittered the WRONG flipper.
		side = 0 if r["x"] > 0 else 1
		if not recs[c]["a"][side] and not (c > 0 and recs[c - 1]["a"][side]):
			continue  # no held press on that side - cannot attribute
		p = c
		while p > 0 and recs[p - 1]["a"][side]:
			p -= 1
		if not recs[p]["a"][side]:
			p += 1
		rel = p
		while rel < len(recs) and recs[rel]["a"][side]:
			rel += 1
		if (side, p) in seen:
			continue
		seen.add((side, p))
		out.append((side, p, rel, c))
	return out


def _life(seed: int) -> list[dict]:
	base = _run(seed, None, None)
	results = []
	for side, p, rel, c in _presses(base):
		real = _shot_value(base, c)
		jit = []
		for d in _CFG["deltas"]:
			lo, hi = min(p, p + d), max(rel, rel + d)
			if lo < 1:
				continue
			override = {t: base[t]["a"] for t in range(lo)}
			for t in range(lo, hi):
				a = base[t]["a"].copy() if t < len(base) else np.zeros(3, dtype=np.int64)
				a[side] = 1 if (p + d) <= t < (rel + d) else 0
				override[t] = a
			recs = _run(seed, override, hi + _CFG["max_window"] + 5)
			contact = next((t for t in range(lo, min(len(recs), hi + 3)) if recs[t]["hit"]), None)
			jit.append(dict(delta=d, contact=contact is not None,
			                value=_shot_value(recs, contact) if contact is not None else None))
		results.append(dict(seed=seed, side=side, press=p, release=rel, contact=c, real=real, jitter=jit))
	return results


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
	if n == 0:
		return (float("nan"), float("nan"))
	ph = k / n
	den = 1 + z * z / n
	centre = (ph + z * z / (2 * n)) / den
	half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
	return centre - half, centre + half


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--policy", default=os.path.join(ROOT, "agents/experiments/big_circuit/best_heldout_gen280.pt"))
	ap.add_argument("--connectome", default=os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"))
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--table-map", default=os.path.join(ROOT, "agents/table_map.json"))
	ap.add_argument("--readout-dim", type=int, default=8)
	ap.add_argument("--frame-skip", type=int, default=4)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--episodes", type=int, default=24)
	ap.add_argument("--seed0", type=int, default=7000)
	ap.add_argument("--workers", type=int, default=2)
	ap.add_argument("--deltas", default="-2,-1,1,2")
	ap.add_argument("--fall-y", type=float, default=8.0)
	ap.add_argument("--max-window", type=int, default=400)
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/step3/aim_jitter_7000.json"))
	args = ap.parse_args()
	if 5000 <= args.seed0 < 6000:
		sys.exit("seeds 5000+ are the held-out confirmation set - use 7000+")
	cfg = dict(binary=args.binary, connectome=args.connectome, readout_dim=args.readout_dim, frame_skip=args.frame_skip,
	           max_steps=args.max_steps, policy=args.policy, table_map=args.table_map,
	           deltas=[int(d) for d in args.deltas.split(",")], fall_y=args.fall_y, max_window=args.max_window)
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
		per_life = pool.map(_life, range(args.seed0, args.seed0 + args.episodes), chunksize=1)
	rows = [r for life in per_life for r in life]

	n_jit = sum(len(r["jitter"]) for r in rows)
	n_nocontact = sum(1 for r in rows for j in r["jitter"] if not j["contact"])
	wins = ties = losses = 0
	margins = []
	for r in rows:
		vals = [j["value"] for j in r["jitter"] if j["contact"]]
		if not vals:
			continue
		med = float(np.median(vals))
		margins.append(r["real"] - med)
		if r["real"] > med:
			wins += 1
		elif r["real"] == med:
			ties += 1
		else:
			losses += 1
	n = wins + ties + losses
	decided = wins + losses
	both_zero = sum(1 for r in rows if r["real"] == 0 and all((j["value"] or 0) == 0 for j in r["jitter"] if j["contact"])
	                and any(j["contact"] for j in r["jitter"]))
	summary = dict(
		presses=len(rows), jitter_runs=n_jit, jitter_no_contact=n_nocontact,
		comparable=n, real_gt_median=wins, ties=ties, real_lt_median=losses,
		frac_gt=wins / n if n else None, ci95_gt=_wilson(wins, n),
		frac_gt_excluding_ties=wins / decided if decided else None, ci95_excl_ties=_wilson(wins, decided),
		ties_all_zero=both_zero,
		mean_real=float(np.mean([r["real"] for r in rows])) if rows else None,
		mean_jitter=float(np.mean([j["value"] for r in rows for j in r["jitter"] if j["contact"]])) if n_jit > n_nocontact else None,
		median_margin=float(np.median(margins)) if margins else None,
		seconds=round(time.time() - t0),
	)
	print(json.dumps(summary, indent=1))
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(dict(args=vars(args), summary=summary, rows=rows), f, default=int)
	print(f"wrote {args.out}")


if __name__ == "__main__":
	main()
