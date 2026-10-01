#!/usr/bin/env python3
"""Scoring-object hit diagnostic + verification for the StateFrame hit fields (step 3).

Plays N one-ball lives (the trainer's `life` episode: ends at the first real drain or
--max-steps) with a policy and reports, per table object (agents/table_map.json):
  - how often the ball hit it and how many points were attributed to it,
  - how much of that happened during ACTIVE play vs passively,
and verifies the new wire fields:
  - every reported id is a valid table_map id with a known position,
  - per step, info["score_delta"] == sum(hit points) + info["unattributed_points"],
  - the ball is actually at the object when the hit is reported (distance ball -> object AABB and
    ball -> object centre; the AABB bounds the ball CENTRE at contact, see gen_table_map.py).

Phases: "launch" = before the first flipper contact of the life; "active" = from a flipper contact
until the ball falls back into the flipper zone (y >= --active-end-y while moving down, vy > 0);
"passive" = any other time after the first flipper contact.

Usage (policy "none" never presses; otherwise a CMA-format checkpoint):
	python agents/experiments/step3/hit_diagnostic.py --policy agents/experiments/big_circuit/best_heldout_gen280.pt \\
		--connectome agents/experiments/big_circuit/connectome.json --episodes 24 --seed0 7000 --workers 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

DEFAULT_BINARY = os.path.join(ROOT, "vendor", "SpaceCadetPinball", "bin", "SpaceCadetPinball")
DEFAULT_MAP = os.path.join(ROOT, "agents", "table_map.json")

_POLICY = None  # flat decoder weights, or None for "never press"
_ACTIVE_END_Y = 9.0


def _init(binary, connectome, readout_dim, frame_skip, max_steps, policy_path, active_end_y):
	global _POLICY, _ACTIVE_END_Y
	import torch

	from agents import train_pinball_circuit_cem as T

	T._worker_init(binary, connectome, readout_dim, frame_skip, max_steps, "circuit", "life", None, 0.0)
	_ACTIVE_END_Y = active_end_y
	if policy_path != "none":
		ckpt = torch.load(policy_path, weights_only=False)
		_POLICY = np.asarray(ckpt["champion"], dtype=np.float32)
		T.set_flat_params(T._WORKER_AGENT.decoder, _POLICY)


def _episode(seed: int) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

	env, agent = T._WORKER_ENV, T._WORKER_AGENT
	obs, info = env.reset(seed=seed)
	h = agent.initial_state(1)
	phase = "launch"
	events, sum_violations, steps, score, unattributed, contacts = [], 0, 0, 0, 0, 0
	done = False
	with torch.no_grad():
		while not done:
			if _POLICY is None:
				action = np.zeros(3, dtype=np.int64)
			else:
				obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
				a, h = agent.act(obs_t, h, greedy=True)
				action = a[0]
			obs, _r, terminated, truncated, info = env.step(action)
			steps += 1
			if info["flipper_hit"]:
				phase = "active"
				contacts += 1  # also the id of the active window this contact opens
			pts = sum(p for _, p, _, _ in info["hit_objects"])
			if pts + info["unattributed_points"] != info["score_delta"]:
				sum_violations += 1
			for (obj_id, points, bx, by), base in zip(info["hit_objects"], info["hit_base_points"]):
				events.append((steps, int(obj_id), int(points), float(bx), float(by), phase, int(base),
				               contacts if phase == "active" else 0))
			score += info["score_delta"]
			unattributed += info["unattributed_points"]
			if phase == "active" and obs[OBS_BALL_Y] >= _ACTIVE_END_Y and obs[OBS_BALL_VY] > 0:
				phase = "passive"
			done = terminated or truncated or info["drained"]
	return dict(seed=seed, steps=steps, score=int(score), unattributed=int(unattributed), contacts=contacts,
	            drained=bool(info["drained"]), sum_violations=sum_violations, events=events)


def _aabb_distance(aabb: dict, x: float, y: float) -> float:
	dx = max(aabb["x_min"] - x, 0.0, x - aabb["x_max"])
	dy = max(aabb["y_min"] - y, 0.0, y - aabb["y_max"])
	return float(np.hypot(dx, dy))


def summarize(results: list[dict], table_map: dict) -> dict:
	objs = {o["id"]: o for o in table_map["objects"]}
	per = defaultdict(lambda: dict(hits=0, points=0, active_hits=0, active_points=0, launch_hits=0,
	                               launch_points=0, d_aabb=[], d_center=[]))
	bad_ids = 0
	for r in results:
		for _step, oid, pts, bx, by, phase, *_rest in r["events"]:
			if oid not in objs:
				bad_ids += 1
				continue
			p = per[oid]
			p["hits"] += 1
			p["points"] += pts
			if phase == "active":
				p["active_hits"] += 1
				p["active_points"] += pts
			elif phase == "launch":
				p["launch_hits"] += 1
				p["launch_points"] += pts
			o = objs[oid]
			if o["aabb"] is not None:
				p["d_aabb"].append(_aabb_distance(o["aabb"], bx, by))
				p["d_center"].append(float(np.hypot(bx - o["center"][0], by - o["center"][1])))
	total_score = sum(r["score"] for r in results)
	rows = []
	for oid, p in sorted(per.items(), key=lambda kv: -kv[1]["points"]):
		o = objs[oid]
		rows.append(dict(
			id=oid, name=o["name"], group=o["group"], hits=p["hits"], points=p["points"],
			share=p["points"] / total_score if total_score else 0.0,
			launch_hits=p["launch_hits"], launch_points=p["launch_points"],
			active_hits=p["active_hits"], active_points=p["active_points"],
			passive_hits=p["hits"] - p["active_hits"] - p["launch_hits"],
			passive_points=p["points"] - p["active_points"] - p["launch_points"],
			median_d_aabb=float(np.median(p["d_aabb"])) if p["d_aabb"] else None,
			median_d_center=float(np.median(p["d_center"])) if p["d_center"] else None,
		))
	all_d_aabb = [d for p in per.values() for d in p["d_aabb"]]
	all_d_center = [d for p in per.values() for d in p["d_center"]]
	groups = defaultdict(lambda: dict(hits=0, points=0, active_hits=0, active_points=0))
	for row in rows:
		g = groups[row["group"]]
		for k in ("hits", "points", "active_hits", "active_points"):
			g[k] += row[k]
	return dict(
		episodes=len(results),
		total_score=int(total_score),
		mean_score=total_score / max(1, len(results)),
		mean_steps=float(np.mean([r["steps"] for r in results])),
		mean_contacts=float(np.mean([r["contacts"] for r in results])),
		unattributed_points=int(sum(r["unattributed"] for r in results)),
		sum_violations=int(sum(r["sum_violations"] for r in results)),
		bad_ids=bad_ids,
		n_hits=len(all_d_aabb),
		median_d_aabb=float(np.median(all_d_aabb)) if all_d_aabb else None,
		p90_d_aabb=float(np.percentile(all_d_aabb, 90)) if all_d_aabb else None,
		median_d_center=float(np.median(all_d_center)) if all_d_center else None,
		objects=rows,
		groups=dict(groups),
	)


def print_summary(s: dict) -> None:
	print(f"episodes {s['episodes']}  mean score/life {s['mean_score']:.0f}  mean steps {s['mean_steps']:.0f}  "
	      f"mean flipper contacts {s['mean_contacts']:.1f}")
	print(f"verification: bad ids {s['bad_ids']}  score-sum violations {s['sum_violations']}  "
	      f"unattributed points {s['unattributed_points']} ({s['unattributed_points'] / max(1, s['total_score']):.1%})  "
	      f"hits {s['n_hits']}  ball->AABB median {s['median_d_aabb']:.3f} (p90 {s['p90_d_aabb']:.3f})  "
	      f"ball->centre median {s['median_d_center']:.3f}")
	print(f"{'id':>3} {'name':<13} {'group':<10} {'hits':>5} {'points':>8} {'share':>6} "
	      f"{'act_h':>5} {'act_pts':>8} {'pas_h':>5} {'pas_pts':>8} {'lau_h':>5} {'lau_pts':>8} {'dAABB':>6} {'dCtr':>6}")
	for r in s["objects"]:
		print(f"{r['id']:>3} {r['name']:<13} {r['group']:<10} {r['hits']:>5} {r['points']:>8} {r['share']:>6.1%} "
		      f"{r['active_hits']:>5} {r['active_points']:>8} {r['passive_hits']:>5} {r['passive_points']:>8} "
		      f"{r['launch_hits']:>5} {r['launch_points']:>8} "
		      f"{(r['median_d_aabb'] if r['median_d_aabb'] is not None else float('nan')):>6.2f} "
		      f"{(r['median_d_center'] if r['median_d_center'] is not None else float('nan')):>6.2f}")
	print("by group:")
	for g, v in sorted(s["groups"].items(), key=lambda kv: -kv[1]["points"]):
		print(f"  {g:<10} hits {v['hits']:>5}  points {v['points']:>8}  active hits {v['active_hits']:>4}  "
		      f"active points {v['active_points']:>8}")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--policy", default="none")
	parser.add_argument("--binary", default=DEFAULT_BINARY)
	parser.add_argument("--connectome", default=os.path.join(ROOT, "agents", "experiments", "big_circuit", "connectome.json"))
	parser.add_argument("--readout-dim", type=int, default=8)
	parser.add_argument("--frame-skip", type=int, default=4)
	parser.add_argument("--max-steps", type=int, default=3000)
	parser.add_argument("--episodes", type=int, default=24)
	parser.add_argument("--seed0", type=int, default=7000)
	parser.add_argument("--workers", type=int, default=2)
	parser.add_argument("--active-end-y", type=float, default=9.0)
	parser.add_argument("--table-map", default=DEFAULT_MAP)
	parser.add_argument("--out", default=None, help="optional JSON output path")
	args = parser.parse_args()
	if args.seed0 >= 5000 and args.seed0 < 6000:
		sys.exit("seeds 5000+ are the held-out confirmation set - use 7000+ for diagnostics")

	with open(args.table_map, encoding="utf-8") as f:
		table_map = json.load(f)
	seeds = list(range(args.seed0, args.seed0 + args.episodes))
	t0 = time.time()
	ctx = mp.get_context("spawn")
	with ctx.Pool(args.workers, initializer=_init, initargs=(
			args.binary, args.connectome, args.readout_dim, args.frame_skip, args.max_steps, args.policy,
			args.active_end_y)) as pool:
		results = pool.map(_episode, seeds, chunksize=1)
	summary = summarize(results, table_map)
	print(f"policy {args.policy}  seeds {seeds[0]}..{seeds[-1]}  {time.time() - t0:.0f}s")
	print_summary(summary)
	if args.out:
		with open(args.out, "w", encoding="utf-8") as f:
			json.dump(dict(args=vars(args), summary=summary, episodes=results), f)
		print(f"wrote {args.out}")


if __name__ == "__main__":
	main()
