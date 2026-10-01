#!/usr/bin/env python3
"""Offline probes for the per-flipper looming input (step 3, option b).

Drives the engine with the correct-side pulse reflex (ReflexLabeler y>11.5) on lives 7000-7023
(fit 7000-7011, test 7012-7023), then feeds the RECORDED observation stream through the fixed
circuit under several input variants (FlipperObsFilter modes, incl. 'loom' with several tau0) and
for each variant fits a per-side logistic readout on all 80 readout cells and reports:
  - held-out ROC-AUC and precision at the reflex's own press rate,
  - LAG profile: AUC of the readout score at step t+lag against the reflex press at step t,
    lag in -3..+3 (peak at lag > 0 = the circuit's evidence arrives after the right moment).
Also estimates the decision-step duration (seconds) from the recorded obs (dy / vy).
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
import torch  # noqa: E402

torch.set_num_threads(1)
import gymnasium as gym  # noqa: E402

from agents import train_pinball_circuit_cem as T  # noqa: E402
from env_python.pinball_env import PinballEnv  # noqa: E402

VARIANTS = [(m, float(t) if t != "-" else None) for m, t in
            (v.split(":") for v in (sys.argv[1] if len(sys.argv) > 1 else
             "raw:-,zero:-,delay1:-,loom:1,loom:2,loom:4").split(","))]


def auc(score, y):
	order = np.argsort(-score)
	ranks = np.empty(len(score))
	ranks[order] = np.arange(len(score))
	pos = y == 1
	npos, nneg = pos.sum(), (~pos).sum()
	if not npos or not nneg:
		return float("nan")
	return float(1 - (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg))


def main() -> None:
	env = gym.wrappers.TimeLimit(PinballEnv(binary_path=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"),
	                                        headless=True, frame_skip=4), max_episode_steps=3000)
	agent = T.build_agent("circuit", os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"), 8)
	agent.calibrate(env.observation_space)
	dec = agent.decoder
	lives = []
	try:
		for seed in range(7000, 7024):
			obs, _ = env.reset(seed=seed)
			lab = T.ReflexLabeler(11.5, 1)
			O, L = [], []
			done = False
			while not done:
				a = lab.label(obs)
				a[2] = 0
				lab.observe(a)
				O.append(np.asarray(obs, dtype=np.float32))
				L.append(a[:2].copy())
				obs, _r, term, trunc, info = env.step(a)
				done = term or trunc or info["drained"]
			lives.append((seed, np.array(O), np.array(L)))
	finally:
		env.close()

	allO = np.concatenate([o for _, o, _ in lives])
	dy = np.concatenate([np.diff(o[:, 1]) for _, o, _ in lives])
	vy = np.concatenate([o[1:, 3] for _, o, _ in lives])
	m = np.abs(vy) > 5
	print(f"decision-step duration estimate: median dy/vy = {np.median(dy[m] / vy[m]):.4f} s "
	      f"(T.LOOM_STEP_DT = {T.LOOM_STEP_DT})")

	LAB = np.concatenate([l for _, _, l in lives])
	LIFE = np.concatenate([np.full(len(l), s) for s, _, l in lives])
	tr, te = LIFE < 7012, LIFE >= 7012
	print(f"steps {len(LAB)}  press rate L {LAB[:, 0].mean():.4f} R {LAB[:, 1].mean():.4f}")
	for mode, tau0 in VARIANTS:
		feats = []
		for _, O, _ in lives:
			filt = T.FlipperObsFilter(mode, loom_tau0=tau0 or 2.0)
			h = agent.initial_state(1)
			with torch.no_grad():
				for o in O:
					h = agent.brain.step(h, agent._channel_drive(torch.as_tensor(filt(o)).unsqueeze(0)))
					feats.append(dec.norm(h[:, dec.idx])[0].numpy())
		X = torch.as_tensor(np.array(feats))
		mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
		X = (X - mu) / sd
		res = []
		for side in (0, 1):
			y = torch.as_tensor(LAB[:, side], dtype=torch.float32)
			model = torch.nn.Linear(X.shape[1], 1)
			opt = torch.optim.Adam(model.parameters(), lr=1e-2)
			pw = torch.tensor([(1 - y[tr].mean()) / y[tr].mean().clamp_min(1e-4)])
			for _ in range(800):
				loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X[tr]).squeeze(-1), y[tr], pos_weight=pw)
				opt.zero_grad()
				loss.backward()
				opt.step()
			with torch.no_grad():
				s_all = model(X).squeeze(-1).numpy()
			yt = LAB[te, side]
			st = s_all[te]
			k = max(1, int(yt.sum()))
			prec = float(yt[np.argsort(-st)[:k]].mean())
			# lag profile within test lives: score(t+lag) vs label(t)
			lags = {}
			for lag in range(-3, 4):
				sc, lb = [], []
				for seed in range(7012, 7024):
					idx = np.where(LIFE == seed)[0]
					s_l, y_l = s_all[idx], LAB[idx, side]
					if lag >= 0:
						sc.append(s_l[lag:]); lb.append(y_l[:len(y_l) - lag])
					else:
						sc.append(s_l[:lag]); lb.append(y_l[-lag:])
				lags[lag] = auc(np.concatenate(sc), np.concatenate(lb))
			res.append((auc(st, yt), prec, lags))
		name = mode if tau0 is None else f"{mode}(tau0={tau0})"
		print(f"{name:<22} L: AUC {res[0][0]:.3f} prec@rate {res[0][1]:.3f}   R: AUC {res[1][0]:.3f} prec@rate {res[1][1]:.3f}")
		for side, r in zip("LR", res):
			best = max(r[2], key=r[2].get)
			print(f"    lag profile {side} (score at t+lag vs press at t): "
			      + " ".join(f"{lag:+d}:{v:.3f}" for lag, v in r[2].items()) + f"   peak {best:+d}")


if __name__ == "__main__":
	main()
