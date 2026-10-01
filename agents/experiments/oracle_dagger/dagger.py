#!/usr/bin/env python3
"""Oracle-labeled imitation (DAgger-style) for the fixed-circuit agent's readout.

The engine is deterministic per seed, so after the current policy loses a ball we can replay the
life and test whether pressing both flippers for a few steps shortly before the ball dropped past
the flippers would have saved it. Those presses become supervised labels; everywhere else the
policy's own actions are kept. The readout (agent.decoder) is retrained on the aggregated data
each iteration. The fly circuit itself is never modified.

Usage:
    python agents/experiments/oracle_dagger/dagger.py \\
        --init agents/experiments/life_objective_2026-09-26/best_heldout_gen125_champion.pt \\
        --iterations 5 --seeds-per-iter 32 --out-dir agents/experiments/oracle_dagger/run1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

BINARY = os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball")
CONNECTOME = os.path.join(ROOT, "web/connectome.json")
MAX_STEPS = 3000
PASS_Y = 13.9  # the engine registers a drain ~85 steps after the ball is already below this line
OFFSETS = list(range(-4, 41, 3))
PRESS_STEPS = 3
RELEASE_STEPS = 1  # env's MIN_FLIPPER_RELEASE_TICKS=1 native tick, frame_skip=4: one released step is enough
SAVE_WINDOW = 80
SAVE_Y = 5.0

_ENV = None
_AGENT = None


def _worker_init(readout_dim: int) -> None:
	global _ENV, _AGENT
	import atexit

	import gymnasium as gym

	from agents.train_pinball_circuit_cem import build_agent
	from env_python.pinball_env import PinballEnv

	torch.set_num_threads(1)
	_ENV = gym.wrappers.TimeLimit(PinballEnv(binary_path=BINARY, headless=True, frame_skip=4), max_episode_steps=MAX_STEPS)
	_AGENT = build_agent("circuit", CONNECTOME, readout_dim)
	_AGENT.calibrate(_ENV.observation_space)
	signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
	atexit.register(_ENV.close)


def _load(flat: np.ndarray) -> None:
	from agents.train_pinball_circuit_cem import set_flat_params
	set_flat_params(_AGENT.decoder, flat)


@torch.no_grad()
def _run_life(seed: int):
	"""Greedy life with the currently loaded decoder. Returns readouts, actions, ys, score, drained."""
	obs, _ = _ENV.reset(seed=seed)
	h = _AGENT.initial_state(1)
	dec = _AGENT.decoder
	readouts, actions, ys, observations = [], [], [], []
	score, drained = 0.0, False
	for _ in range(MAX_STEPS):
		h = _AGENT.brain.step(h, _AGENT._channel_drive(torch.as_tensor(obs).unsqueeze(0)))
		readouts.append(h[0, dec.idx].numpy().copy())
		observations.append(np.asarray(obs, dtype=np.float32).copy())
		logits = dec.dist_inputs(dec.features(h))[0]
		l, r = int(logits[0] > 0), int(logits[1] > 0)
		actions.append((l, r))
		obs, _, term, trunc, info = _ENV.step(np.array([l, r, 0]))
		ys.append(float(obs[1]))
		score += info["score_delta"]
		if info["drained"]:
			drained = True
			break
		if term or trunc:
			break
	return np.array(readouts, dtype=np.float32), actions, ys, score, drained, np.array(observations)


def _pass_step(ys, T: int) -> int:
	for t in range(T - 1, -1, -1):
		if ys[t] < PASS_Y:
			return t
	return 0


def _forced_sequence(intervention: str):
	if intervention == "press":
		return [(1, 1)] * PRESS_STEPS
	if intervention == "release_press":
		return [(0, 0)] * RELEASE_STEPS + [(1, 1)] * PRESS_STEPS
	raise ValueError(f"unknown intervention {intervention!r}")


def _saved_by_intervention(seed: int, actions, start: int, sequence) -> bool:
	_ENV.reset(seed=seed)
	for l, r in actions[:start]:
		_, _, term, trunc, info = _ENV.step(np.array([l, r, 0]))
		if info["drained"] or term or trunc:
			return False
	for i in range(SAVE_WINDOW):
		l, r = sequence[i] if i < len(sequence) else (0, 0)
		obs, _, term, trunc, info = _ENV.step(np.array([l, r, 0]))
		if obs[1] < SAVE_Y:
			return True
		if info["drained"] or term or trunc:
			return False
	return False


def collect_task(task):
	flat, seed, intervention = task
	_load(flat)
	sequence = _forced_sequence(intervention)
	readouts, actions, ys, score, drained, observations = _run_life(seed)
	corrective = None
	t_pass = None
	n_saving = 0
	if drained:
		T = len(actions) - 1
		t_pass = _pass_step(ys, T)
		for d in OFFSETS:  # ascending d = latest press first
			start = t_pass - d
			if 0 <= start and start + len(sequence) <= T and _saved_by_intervention(seed, actions, start, sequence):
				corrective = start
				break
	if corrective is not None:
		n_saving = 1
	return dict(seed=seed, readouts=readouts, observations=observations, actions=np.array(actions, dtype=np.float32),
	            score=score, life=len(actions), drained=drained, corrective=corrective, t_pass=t_pass, n_saving=n_saving)


def eval_task(task):
	flat, seed = task
	_load(flat)
	_, actions, _, score, drained, _ = _run_life(seed)
	return score, len(actions)


def _logits(decoder, R: torch.Tensor) -> torch.Tensor:
	return decoder.dist_inputs(decoder.proj(decoder.norm(R)))


def main():
	from agents.train_pinball_circuit_cem import build_agent, get_flat_params, set_flat_params
	import gymnasium as gym

	p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	p.add_argument("--init", required=True)
	p.add_argument("--iterations", type=int, default=5)
	p.add_argument("--seeds-per-iter", type=int, default=32)
	p.add_argument("--seed-base", type=int, default=10000)
	p.add_argument("--eval-seeds", type=int, default=32)
	p.add_argument("--eval-seed-base", type=int, default=20000)
	p.add_argument("--workers", type=int, default=8)
	p.add_argument("--label-weight", type=float, default=20.0)
	p.add_argument("--intervention", choices=["press", "release_press"], default="release_press")
	p.add_argument("--epochs", type=int, default=300)
	p.add_argument("--lr", type=float, default=1e-3)
	p.add_argument("--target-logit", type=float, default=4.0)
	p.add_argument("--out-dir", required=True)
	args = p.parse_args()
	assert args.seed_base >= 4096 or args.seed_base + args.iterations * args.seeds_per_iter <= 4000, "collection seeds overlap the held-out test set 4000-4095"
	os.makedirs(args.out_dir, exist_ok=True)
	log_path = os.path.join(args.out_dir, "log.txt")

	def log(msg):
		print(msg, flush=True)
		with open(log_path, "a") as f:
			f.write(msg + "\n")

	torch.set_num_threads(4)
	ck = torch.load(args.init, weights_only=False)
	readout_dim = ck["readout_dim"]
	agent = build_agent("circuit", CONNECTOME, readout_dim)
	agent.calibrate(gym.spaces.Box(-np.inf, np.inf, (15,), np.float32))
	set_flat_params(agent.decoder, ck["champion"])
	decoder = agent.decoder
	init_flat = get_flat_params(decoder).copy()

	eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.eval_seeds))
	ctx = mp.get_context("spawn")
	pool = ctx.Pool(args.workers, initializer=_worker_init, initargs=(readout_dim,))
	try:
		init_eval = np.array(pool.map(eval_task, [(init_flat, s) for s in eval_seeds]))
		init_log = np.log1p(np.maximum(init_eval[:, 0], 0))
		log(f"init eval: mean log1p {init_log.mean():.3f}  median score {np.median(init_eval[:, 0]):.0f}  mean life {init_eval[:, 1].mean():.0f}")
		log(f"intervention: {args.intervention}")

		sequence = _forced_sequence(args.intervention)
		data_R, data_y, data_w = [], [], []
		flat = init_flat
		for k in range(args.iterations):
			t0 = time.time()
			seeds = [args.seed_base + k * args.seeds_per_iter + i for i in range(args.seeds_per_iter)]
			results = pool.map(collect_task, [(flat, s, args.intervention) for s in seeds])
			n_drained = sum(r["drained"] for r in results)
			n_corr = 0
			for r in results:
				y = r["actions"].copy()
				w = np.ones(len(y), dtype=np.float32)
				if r["corrective"] is not None:
					s = r["corrective"]
					# the saving window is usually wider than the chosen intervention, so neighbouring steps are
					# ambiguous rather than "don't press": mask them first, then overwrite with the corrective labels
					lo = max(0, r["t_pass"] - max(OFFSETS) - 1)
					w[lo:r["t_pass"] - min(OFFSETS) + 1] = 0.0
					y[s:s + len(sequence)] = np.array(sequence, dtype=np.float32)
					w[s:s + len(sequence)] = args.label_weight
					n_corr += 1
				data_R.append(r["readouts"]); data_y.append(y); data_w.append(w)
			np.savez_compressed(os.path.join(args.out_dir, f"data_iter{k}.npz"),
			                    R=np.concatenate([r["readouts"] for r in results]),
			                    obs=np.concatenate([r["observations"] for r in results]),
			                    y=np.concatenate(data_y[-len(results):]), w=np.concatenate(data_w[-len(results):]))
			R = torch.as_tensor(np.concatenate(data_R))
			Y = torch.as_tensor(np.concatenate(data_y))
			W = torch.as_tensor(np.concatenate(data_w))
			log(f"iter {k}: lives {len(results)}  drained {n_drained}  savable (corrective labels) {n_corr}  "
			    f"mean life {np.mean([r['life'] for r in results]):.0f}  dataset {len(R)}  collect {time.time() - t0:.0f}s")

			if k == 0:
				with torch.no_grad():
					lg = _logits(decoder, R)[:, :2]
					own_acc = ((lg > 0).float() == torch.as_tensor(np.concatenate([r["actions"] for r in results]))).float().mean(0)
					log(f"  pre-train agreement with own actions: left {own_acc[0]:.4f} right {own_acc[1]:.4f}")
					med = lg.abs().median(0).values.clamp_min(1e-6)
					scale = args.target_logit / med
					decoder.head.weight[:2] *= scale[:, None]
					decoder.head.bias[:2] *= scale
					log(f"  head rescaled by {scale.tolist()} (median |logit| was {med.tolist()})")

			params = [q for q in decoder.parameters()]
			opt = torch.optim.Adam(params, lr=args.lr)
			for epoch in range(args.epochs):
				lg = _logits(decoder, R)[:, :2]
				loss = (torch.nn.functional.binary_cross_entropy_with_logits(lg, Y, reduction="none").mean(1) * W).sum() / W.sum()
				opt.zero_grad()
				loss.backward()
				decoder.head.weight.grad[2] = 0
				decoder.head.bias.grad[2] = 0
				opt.step()
				if epoch == 0:
					loss0 = loss.item()
			with torch.no_grad():
				lg = _logits(decoder, R)[:, :2]
				corr_mask = W > 1
				corr_acc = ((lg[corr_mask] > 0).float() == Y[corr_mask]).float().mean().item() if corr_mask.any() else float("nan")
			flat = get_flat_params(decoder).copy()

			ev = np.array(pool.map(eval_task, [(flat, s) for s in eval_seeds]))
			lg_ev = np.log1p(np.maximum(ev[:, 0], 0))
			d = lg_ev - init_log
			se = d.std(ddof=1) / np.sqrt(len(d))
			log(f"  train loss {loss0:.4f} -> {loss.item():.4f}  corrective-label acc {corr_acc:.3f}")
			log(f"  eval: mean log1p {lg_ev.mean():.3f}  median score {np.median(ev[:, 0]):.0f}  mean life {ev[:, 1].mean():.0f}  "
			    f"vs init diff {d.mean():+.3f} (SE {se:.3f}, t={d.mean() / se:.2f})")
			torch.save({"champion": flat, "mean": flat, "sigma": np.zeros_like(flat), "champion_fitness": float(lg_ev.mean()),
			            "generation": k + 1, "readout_dim": readout_dim, "agent": "circuit"},
			           os.path.join(args.out_dir, f"iter{k + 1}.pt"))
	finally:
		pool.terminate()
		pool.join()


if __name__ == "__main__":
	main()
