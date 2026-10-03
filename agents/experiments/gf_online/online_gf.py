"""Online-learning split giant-fiber body: one independent chain = one circuit + one RNG seed.

Learned (log space, mu): R, g4, g2, v0, s0. Fixed: hold_max 2, refractory 1, theta_off = 0.75*theta_on.
Each update samples eps ~ N(0, sigma^2 I), plays one life with mu+eps and one with mu-eps on the SAME game seed
(antithetic), reward r = log1p(score) - 0.05*max(0, presses-25), and moves mu by alpha*(r+ - r-)/(2*sigma)*eps
(step norm clipped). Homeostasis after every life: theta_on *= exp(0.05*(presses-22)/22), clipped to [0.01, 1].
Snapshots to OUT/<circuit>_c<K>.json every 100 updates (resumable); after the last update the final mu is
evaluated on the 96 paired lives 77000-77095 (mu frozen, homeostasis still active) and stored in the same file.

Usage: python online_gf.py --circuit real --chain 0 --updates 1500 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PB = os.path.join(HERE, "..", "pathway_body")
sys.path.insert(0, PB)
import gf_body as G  # noqa: E402
import gf_split_body as S  # noqa: E402

PARAMS = ["R", "g4", "g2", "v0", "s0"]
RANGES = dict(R=(0.3, 3.0), g4=(0.5, 8.0), g2=(0.5, 8.0), v0=(0.05, 10.0), s0=(0.05, 1.5))  # tune_gf_split.py ranges
CLIP_FACTOR = 3.0  # mu is kept within the tune range widened by this factor (guards against numerical runaway)
MU0 = np.array([0.5 * (np.log(RANGES[p][0]) + np.log(RANGES[p][1])) for p in PARAMS])  # geometric midpoints
MU_LO = np.array([np.log(RANGES[p][0]) - np.log(CLIP_FACTOR) for p in PARAMS])
MU_HI = np.array([np.log(RANGES[p][1]) + np.log(CLIP_FACTOR) for p in PARAMS])
THETA0, THETA_LO, THETA_HI = 0.2, 0.01, 1.0
SIGMA, ALPHA, STEP_CLIP = 0.15, 0.05, 0.3
PRESS_FREE, PRESS_COST = 25, 0.05
HOMEO_GAIN, HOMEO_TARGET = 0.05, 22.0
SNAP_EVERY = 100
EVAL_SEEDS = list(range(77000, 77096))
_BODY_KEY = json.dumps({"online": 1})


def circuit_path(name: str) -> str:
	return os.path.abspath(os.path.join(PB, "gf_circuit.json" if name == "real" else f"gf_circuit_{name}.json"))


def train_seed(chain: int, k: int) -> int:
	return 100000 + chain * 100000 + k


def make_body(circuit: str) -> S.SplitGiantFiberBody:
	body = S.SplitGiantFiberBody(circuit_path(circuit), R=1.0, g4=1.0, g2=1.0, v0=1.0, s0=1.0, theta_on=THETA0,
	                             hold_max=2, refractory=1, rho_target=0.8)
	G._BODIES[_BODY_KEY] = body  # run_life("gf", ...) picks the body up from the cache; params are mutated in place
	return body


def play(body, mu: np.ndarray, theta_on: float, seed: int) -> tuple[dict, float]:
	"""One life with params exp(mu) and theta_on; returns (life stats, theta_on after homeostasis)."""
	body.R, body.g4, body.g2, body.v0, body.s0 = (float(v) for v in np.exp(mu))
	body.theta_on = float(theta_on)
	body.theta_off = 0.75 * body.theta_on
	r = G.run_life("gf", {"online": 1}, seed)
	theta_on = float(np.clip(theta_on * np.exp(HOMEO_GAIN * (r["presses"] - HOMEO_TARGET) / HOMEO_TARGET), THETA_LO, THETA_HI))
	return r, theta_on


def reward(r: dict) -> float:
	return float(np.log1p(r["score"]) - PRESS_COST * max(0, r["presses"] - PRESS_FREE))


def _save(path: str, state: dict) -> None:
	tmp = path + ".tmp"
	with open(tmp, "w", encoding="utf-8") as f:
		json.dump(state, f)
	os.replace(tmp, path)


def run_chain(circuit: str, chain: int, updates: int, out: str) -> str:
	"""Runs (or resumes) one chain in this process; G.init_worker() must have been called. Returns the JSON path."""
	os.makedirs(out, exist_ok=True)
	path = os.path.join(out, f"{circuit}_c{chain}.json")
	tag = f"[{circuit} c{chain}]"
	rng = np.random.default_rng(1000 + chain)
	state = dict(circuit=circuit, chain=chain, updates=updates, mu=MU0.tolist(), theta_on=THETA0, done_updates=0, history=[],
	             rng_state=None, eval=None, params=PARAMS)
	if os.path.exists(path):
		with open(path, encoding="utf-8") as f:
			old = json.load(f)
		if old.get("circuit") == circuit and old.get("chain") == chain:
			state = old
			state["updates"] = updates
			if state["rng_state"] is not None:
				rng.bit_generator.state = state["rng_state"]
			print(f"{tag} resuming at update {state['done_updates']}", flush=True)
	body = make_body(circuit)
	mu, theta_on = np.array(state["mu"]), float(state["theta_on"])
	t0 = time.time()
	start = state["done_updates"]
	for k in range(start, updates):
		eps = rng.normal(0.0, SIGMA, size=len(PARAMS))
		seed = train_seed(chain, k)
		rp, theta_on = play(body, np.clip(mu + eps, MU_LO, MU_HI), theta_on, seed)
		rm, theta_on = play(body, np.clip(mu - eps, MU_LO, MU_HI), theta_on, seed)
		rwp, rwm = reward(rp), reward(rm)
		step = ALPHA * (rwp - rwm) / (2.0 * SIGMA) * eps
		n = float(np.linalg.norm(step))
		if n > STEP_CLIP:
			step *= STEP_CLIP / n
		mu = np.clip(mu + step, MU_LO, MU_HI)
		state["history"].append(dict(k=k, seed=seed, r_plus=rwp, r_minus=rwm, score_plus=rp["score"], score_minus=rm["score"],
		                             presses_plus=rp["presses"], presses_minus=rm["presses"], theta_on=theta_on, mu=mu.tolist()))
		state["mu"], state["theta_on"], state["done_updates"] = mu.tolist(), theta_on, k + 1
		state["rng_state"] = rng.bit_generator.state
		el = time.time() - t0
		print(f"{tag} upd {k + 1}/{updates} r+ {rwp:.2f} r- {rwm:.2f} presses {rp['presses']}/{rm['presses']} theta_on {theta_on:.3f} "
		      f"mu {' '.join(f'{v:.2f}' for v in mu)} ({(k + 1 - start) / max(el, 1e-9) * 3600:.0f} upd/h)", flush=True)
		if (k + 1) % SNAP_EVERY == 0 and k + 1 < updates:
			_save(path, state)
	if state["eval"] is None:
		ev, th = [], theta_on
		for s in EVAL_SEEDS:
			r, th = play(body, mu, th, s)
			ev.append(dict(seed=s, score=r["score"], presses=r["presses"], contacts=r["contacts"]))
		state["eval"] = ev
		print(f"{tag} final eval: log1p {np.mean([np.log1p(e['score']) for e in ev]):.3f} presses {np.mean([e['presses'] for e in ev]):.1f}", flush=True)
	_save(path, state)
	return path


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--circuit", required=True, choices=["real"] + [f"shuffled_{i}" for i in range(1, 7)])
	ap.add_argument("--chain", type=int, required=True)
	ap.add_argument("--updates", type=int, default=1500)
	ap.add_argument("--out", required=True)
	args = ap.parse_args()
	G.init_worker()
	run_chain(args.circuit, args.chain, args.updates, args.out)


if __name__ == "__main__":
	main()
