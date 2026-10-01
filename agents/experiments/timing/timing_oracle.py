"""Replay oracle for flipper TIMING: for every ball approach in a reflex-driven life, try each press
timing in a small window around the reflex's own press (and no press at all), then let the reflex
play on for a short horizon, and record the value of each option. Output is a labelled dataset
(observation before the window -> value of every timing option) used to test whether the best
timing is predictable from what the agent can see (timing_learnability.py), before any training.

The engine is deterministic given seed + action sequence, so each option is evaluated by resetting
to the same seed and replaying the recorded actions up to the window start.

Usage: python agents/experiments/timing/timing_oracle.py --seeds 20000:20120 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

OFFSETS = (-2, -1, 0, 1, 2, 3)          # press step relative to the reflex's natural press step t0
OPTIONS = OFFSETS + (None,)             # None = do not press in the window at all
WINDOW = (min(OFFSETS), max(OFFSETS))   # window covers t0-2 .. t0+3
BANK = ("a_targ7", "a_targ8", "a_targ9")
DRAIN_NAME = "drain"
BANK_VALUE = 1000.0                     # value units: base points; a bank hit counts as 1000
DRAIN_PENALTY = 20000.0                 # losing the ball within the horizon
MAX_EVENTS_PER_LIFE = 12

_ENV = None
_IDS: dict = {}


def _init(binary: str) -> None:
	global _ENV
	import atexit
	import signal

	import torch

	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_ENV = PinballEnv(binary_path=binary, headless=True, frame_skip=4)
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	atexit.register(_ENV.close)
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		objs = json.load(f)["objects"]
	_IDS["bank"] = frozenset(o["id"] for o in objs if o["name"] in BANK)
	_IDS["drain"] = frozenset(o["id"] for o in objs if o["name"] == DRAIN_NAME)
	_IDS["upper"] = frozenset(o["id"] for o in objs if o["center"] is not None and o["center"][1] < 6.0
	                          and o["group"] not in ("drain", "flipper", "plunger"))


def _reflex():
	from agents.train_pinball_circuit_cem import ReflexLabeler

	return ReflexLabeler(11.5, 1)


def _reflex_act(reflex, obs) -> np.ndarray:
	a = reflex.label(obs)
	a[2] = 0
	reflex.observe(a)
	return a


def _base_life(seed: int, max_steps: int):
	"""Reflex-driven life: returns (actions, observations-before-each-action, press events)."""
	obs, _ = _ENV.reset(seed=seed)
	reflex = _reflex()
	actions, observations, events = [], [], []
	for t in range(max_steps):
		a = _reflex_act(reflex, obs)
		observations.append(np.asarray(obs, dtype=np.float32))
		actions.append(a.copy())
		for side in (0, 1):
			if a[side] and (t == 0 or not actions[t - 1][side]):
				events.append((t, side))
		obs, _r, term, trunc, info = _ENV.step(a)
		if term or trunc or info["drained"]:
			break
	return actions, observations, events


def _option_value(seed: int, actions, t0: int, side: int, offset, horizon: int) -> dict:
	"""Replay to the window start, apply one timing option, then run the reflex for `horizon` steps."""
	ws, we = t0 + WINDOW[0], t0 + WINDOW[1]
	obs, _ = _ENV.reset(seed=seed)
	for t in range(ws):
		obs, _r, term, trunc, info = _ENV.step(actions[t])
		if term or trunc or info["drained"]:
			return dict(bank=0, upper=0.0, drained=1, invalid=1)
	bank = 0
	upper = 0.0
	drained = 0

	def account(info):
		nonlocal bank, upper, drained
		for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
			if oid in _IDS["bank"]:
				bank += 1
			if oid in _IDS["upper"]:
				upper += base
			if oid in _IDS["drain"]:
				drained = 1
		if info["drained"]:
			drained = 1

	for t in range(ws, we + 1):
		a = np.zeros(3, dtype=np.int64)
		if offset is not None and t == t0 + offset:
			a[side] = 1
		obs, _r, term, trunc, info = _ENV.step(a)
		account(info)
		if term or trunc or drained:
			return dict(bank=bank, upper=upper, drained=drained, invalid=0)
	reflex = _reflex()
	for _ in range(horizon):
		obs, _r, term, trunc, info = _ENV.step(_reflex_act(reflex, obs))
		account(info)
		if term or trunc or drained:
			break
	return dict(bank=bank, upper=upper, drained=drained, invalid=0)


def _life_task(task) -> dict:
	seed, max_steps, horizon = task
	actions, observations, events = _base_life(seed, max_steps)
	rows = []
	for t0, side in events[:MAX_EVENTS_PER_LIFE]:
		ws = t0 + WINDOW[0]
		if ws < 2:
			continue
		outcomes = [_option_value(seed, actions, t0, side, off, horizon) for off in OPTIONS]
		if any(o["invalid"] for o in outcomes):
			continue
		values = [BANK_VALUE * o["bank"] + o["upper"] - DRAIN_PENALTY * o["drained"] for o in outcomes]
		rows.append(dict(seed=seed, t0=t0, side=side,
		                 feat=np.concatenate([observations[ws], observations[ws - 1]]).tolist(),
		                 values=values, bank=[o["bank"] for o in outcomes], upper=[o["upper"] for o in outcomes],
		                 drained=[o["drained"] for o in outcomes]))
	return dict(seed=seed, life_steps=len(actions), n_events=len(events), rows=rows)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="20000:20120")
	ap.add_argument("--workers", type=int, default=8)
	ap.add_argument("--max-steps", type=int, default=3000)
	ap.add_argument("--horizon", type=int, default=75, help="decision steps the reflex plays after the window")
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/timing/timing_oracle.npz"))
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	t_start = time.time()
	rows = []
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		for i, res in enumerate(pool.imap_unordered(_life_task, [(s, args.max_steps, args.horizon) for s in range(lo, hi)])):
			rows.extend(res["rows"])
			print(f"[{i + 1}/{hi - lo}] seed {res['seed']} steps {res['life_steps']} events {res['n_events']} "
			      f"rows {len(res['rows'])}  total rows {len(rows)}  {time.time() - t_start:.0f}s", flush=True)
	np.savez(args.out, feat=np.array([r["feat"] for r in rows], dtype=np.float32),
	         values=np.array([r["values"] for r in rows], dtype=np.float32),
	         bank=np.array([r["bank"] for r in rows], dtype=np.int16),
	         upper=np.array([r["upper"] for r in rows], dtype=np.float32),
	         drained=np.array([r["drained"] for r in rows], dtype=np.int8),
	         seed=np.array([r["seed"] for r in rows]), t0=np.array([r["t0"] for r in rows]),
	         side=np.array([r["side"] for r in rows]), options=np.array([-99 if o is None else o for o in OPTIONS]))
	print(f"saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
	main()
