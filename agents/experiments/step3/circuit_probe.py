#!/usr/bin/env python3
"""Root-cause probes of the gen-280 circuit champion's press rhythm (step 3 follow-up).

1. Records --episodes champion lives (one process, one engine): obs, readout activity, actions.
2. H1 (rhythm intrinsic?): drives the circuit OFFLINE for 200 steps with
     a) zero observation, b) a frozen real observation (flipper channels frozen too),
     c) the same frozen ball observation but flipper channels = previous action (proprioceptive
        feedback only - obs[12:14] is the physical flipper state, which follows the previous
        decision within one 4-tick step),
     d) as c) but with flipper channels forced to 0,
   and reports the press period of each.
3. Channel ablation on the recorded lives (open-loop: same obs stream, some channels replaced):
   agreement of the greedy action with the original when ball channels (0-11) are replaced by
   their per-life mean, or flipper channels (12-13) by 0.
4. H3: ridge probe from readout activity (80 cells, and the 8-d bottleneck) to ball x, y, vy on
   steps with y > 10 (flipper zone), leave-one-life-out R^2.

Usage: python agents/experiments/step3/circuit_probe.py --episodes 6 --seed0 7000
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


def pattern(actions: np.ndarray) -> dict:
	a = np.asarray(actions)[:, :2]
	gaps = []
	for side in (0, 1):
		on = np.where(a[:, side] == 1)[0]
		starts = on[np.r_[True, np.diff(on) > 1]] if len(on) else on
		gaps.extend(np.diff(starts).tolist())
	vals, counts = np.unique(gaps, return_counts=True) if gaps else (np.array([]), np.array([]))
	top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:3]
	return dict(p_left=float(a[:, 0].mean()), p_right=float(a[:, 1].mean()), lr_equal=float((a[:, 0] == a[:, 1]).mean()),
	            top_gaps={int(v): int(c) for c, v in top})


def run_offline(agent, obs_fn, steps: int = 200) -> np.ndarray:
	h = agent.initial_state(1)
	prev = np.zeros(3, dtype=np.int64)
	acts = []
	with torch.no_grad():
		for t in range(steps):
			o = obs_fn(t, prev)
			a, h = agent.act(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), h, greedy=True)
			prev = np.asarray(a[0])
			acts.append(prev.copy())
	return np.array(acts)


def ridge_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, lam: float = 1.0) -> list[float]:
	r2 = []
	preds = np.zeros_like(Y)
	for g in np.unique(groups):
		tr, te = groups != g, groups == g
		mu_x, mu_y = X[tr].mean(0), Y[tr].mean(0)
		xs = X[tr] - mu_x
		w = np.linalg.solve(xs.T @ xs + lam * len(xs) * np.eye(X.shape[1]), xs.T @ (Y[tr] - mu_y))
		preds[te] = (X[te] - mu_x) @ w + mu_y
	for k in range(Y.shape[1]):
		ss = ((Y[:, k] - Y[:, k].mean()) ** 2).sum()
		r2.append(float(1 - ((Y[:, k] - preds[:, k]) ** 2).sum() / ss))
	return r2


def main() -> None:
	ap = argparse.ArgumentParser()
	ap.add_argument("--policy", default=os.path.join(ROOT, "agents/experiments/big_circuit/best_heldout_gen280.pt"))
	ap.add_argument("--connectome", default=os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"))
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--episodes", type=int, default=6)
	ap.add_argument("--seed0", type=int, default=7000)
	ap.add_argument("--out", default=os.path.join(ROOT, "agents/experiments/step3/circuit_probe_7000.json"))
	args = ap.parse_args()

	import gymnasium as gym

	from agents import train_pinball_circuit_cem as T
	from env_python.pinball_env import PinballEnv

	ckpt = torch.load(args.policy, weights_only=False)
	agent = T.build_agent("circuit", args.connectome, ckpt["readout_dim"])
	env = gym.wrappers.TimeLimit(PinballEnv(binary_path=args.binary, headless=True, frame_skip=4), max_episode_steps=3000)
	agent.calibrate(env.observation_space)
	T.set_flat_params(agent.decoder, np.asarray(ckpt["champion"], dtype=np.float32))
	dec = agent.decoder
	out: dict = {}

	# ---- record lives ----
	lives = []
	try:
		for seed in range(args.seed0, args.seed0 + args.episodes):
			obs, _ = env.reset(seed=seed)
			h = agent.initial_state(1)
			O, A, R, F = [], [], [], []
			done = False
			with torch.no_grad():
				while not done:
					o = np.asarray(obs, dtype=np.float32)
					a, h = agent.act(torch.as_tensor(o).unsqueeze(0), h, greedy=True)
					O.append(o)
					A.append(np.asarray(a[0]))
					R.append(dec.norm(h[:, dec.idx])[0].numpy())
					F.append(dec.features(h)[0].numpy())
					obs, _r, term, trunc, info = env.step(a[0])
					done = term or trunc or info["drained"]
			lives.append(dict(obs=np.array(O), act=np.array(A), readout=np.array(R), feat=np.array(F)))
	finally:
		env.close()
	out["recorded"] = [pattern(lv["act"]) for lv in lives]

	# ---- H1: offline drives ----
	all_obs = np.concatenate([lv["obs"] for lv in lives])
	falling = all_obs[(all_obs[:, 1] > 3) & (all_obs[:, 1] < 7) & (all_obs[:, 3] > 0)]
	frozen = falling[len(falling) // 2].copy()
	zero = np.zeros_like(frozen)
	offline = {
		"zero_obs": run_offline(agent, lambda t, p: zero),
		"frozen_real_obs": run_offline(agent, lambda t, p: frozen),
		"frozen_ball_flipper_feedback": run_offline(agent, lambda t, p: np.r_[frozen[:12], p[0], p[1], frozen[14]]),
		"frozen_ball_flippers_forced_0": run_offline(agent, lambda t, p: np.r_[frozen[:12], 0.0, 0.0, frozen[14]]),
		"zero_ball_flipper_feedback": run_offline(agent, lambda t, p: np.r_[np.zeros(12), p[0], p[1], 0.0]),
	}
	out["H1_offline"] = {k: pattern(v[20:]) for k, v in offline.items()}  # skip 20-step transient
	out["H1_frozen_obs"] = frozen.tolist()

	# ---- channel ablation, open loop on recorded obs streams ----
	def replay(obs_stream: np.ndarray) -> np.ndarray:
		h = agent.initial_state(1)
		acts = []
		with torch.no_grad():
			for o in obs_stream:
				a, h = agent.act(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0), h, greedy=True)
				acts.append(np.asarray(a[0]))
		return np.array(acts)

	abl = {"ball_channels_to_life_mean": [], "flipper_channels_to_0": [], "flipper_channels_to_prev_action": []}
	for lv in lives:
		o = lv["obs"].copy()
		o[:, :12] = o[:, :12].mean(0)
		abl["ball_channels_to_life_mean"].append(float((replay(o)[:, :2] == lv["act"][:, :2]).all(1).mean()))
		o = lv["obs"].copy()
		o[:, 12:14] = 0
		abl["flipper_channels_to_0"].append(float((replay(o)[:, :2] == lv["act"][:, :2]).all(1).mean()))
	out["ablation_action_agreement"] = {k: float(np.mean(v)) for k, v in abl.items() if v}

	# ---- H3: linear probes in the flipper zone ----
	X80 = np.concatenate([lv["readout"] for lv in lives])
	X8 = np.concatenate([lv["feat"] for lv in lives])
	Y = all_obs[:, [0, 1, 3]]
	g = np.concatenate([np.full(len(lv["obs"]), i) for i, lv in enumerate(lives)])
	zone = Y[:, 1] > 10
	# lagged readout (activity one step after the obs) too, since the circuit integrates over time
	out["H3_probe_r2_flipper_zone"] = {
		"n_steps": int(zone.sum()),
		"readout80_xyvy": ridge_r2(X80[zone], Y[zone], g[zone]),
		"bottleneck8_xyvy": ridge_r2(X8[zone], Y[zone], g[zone]),
		"readout80_xyvy_all_steps": ridge_r2(X80, Y, g),
	}
	# how much readout variance the obs explains vs. the circuit's own dynamics
	out["H3_obs_to_readout_r2_mean"] = float(np.mean(ridge_r2(all_obs, X80, g)))

	# H4: head geometry
	W = dec.head.weight.detach().numpy()
	cos = float(W[0] @ W[1] / (np.linalg.norm(W[0]) * np.linalg.norm(W[1])))
	out["H4_head_cosine_left_right"] = cos
	out["H4_head_bias"] = dec.head.bias.detach().numpy().tolist()
	logits = X8 @ W.T + dec.head.bias.detach().numpy()
	out["H4_logit_corr_left_right"] = float(np.corrcoef(logits[:, 0], logits[:, 1])[0, 1])

	print(json.dumps(out, indent=1))
	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(out, f, indent=1)


if __name__ == "__main__":
	main()
