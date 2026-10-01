#!/usr/bin/env python3
"""Can a linear readout of the fixed circuit represent the correct-side pulse reflex at all?

Drives the engine with ReflexLabeler (pulse y>11.5) on lives 7000-7011 (fit) and 7012-7023 (test),
records normalized readout activity (80 cells) with each FlipperObsFilter mode, the raw obs, and the
reflex labels, then fits per-side logistic regression (full 80-d, i.e. the upper bound of the
trainable proj+head) and reports held-out ROC-AUC and precision at the label's own press rate.
Raw-observation logistic regression and a small MLP on raw obs are given as references.
"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
import torch
torch.set_num_threads(1)
import gymnasium as gym
from agents import train_pinball_circuit_cem as T
from env_python.pinball_env import PinballEnv

MODES = ("raw", "zero", "delay1")
env = gym.wrappers.TimeLimit(PinballEnv(binary_path=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"),
                                        headless=True, frame_skip=4), max_episode_steps=3000)
agent = T.build_agent("circuit", os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json"), 8)
agent.calibrate(env.observation_space)
dec = agent.decoder
data = {m: [] for m in MODES}; OBS, LAB, LIFE = [], [], []
try:
    for seed in range(7000, 7024):
        obs, _ = env.reset(seed=seed); lab = T.ReflexLabeler(11.5, 1)
        hs = {m: agent.initial_state(1) for m in MODES}; filt = {m: T.FlipperObsFilter(m) for m in MODES}
        done = False
        while not done:
            a = lab.label(obs); a[2] = 0; lab.observe(a)
            with torch.no_grad():
                for m in MODES:
                    hs[m] = agent.brain.step(hs[m], agent._channel_drive(torch.as_tensor(np.asarray(filt[m](obs), dtype=np.float32)).unsqueeze(0)))
                    data[m].append(dec.norm(hs[m][:, dec.idx])[0].numpy())
            OBS.append(np.asarray(obs, dtype=np.float32)); LAB.append(a[:2].copy()); LIFE.append(seed)
            obs, _r, term, trunc, info = env.step(a)
            done = term or trunc or info["drained"]
finally:
    env.close()
LAB = np.array(LAB); LIFE = np.array(LIFE); OBS = np.array(OBS)
tr, te = LIFE < 7012, LIFE >= 7012

def fit_eval(X, name, hidden=0):
    X = torch.as_tensor(X, dtype=torch.float32); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; X = (X - mu) / sd
    out = []
    for side in (0, 1):
        y = torch.as_tensor(LAB[:, side], dtype=torch.float32)
        model = torch.nn.Linear(X.shape[1], 1) if not hidden else torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        pw = torch.tensor([(1 - y[tr].mean()) / y[tr].mean().clamp_min(1e-4)])
        for _ in range(800):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X[tr]).squeeze(-1), y[tr], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            s = model(X[te]).squeeze(-1).numpy()
        yt = LAB[te, side]
        order = np.argsort(-s); ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
        pos = yt == 1; npos, nneg = pos.sum(), (~pos).sum()
        auc = 1 - (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
        k = max(1, int(npos)); prec = yt[order[:k]].mean()  # precision at the label's press rate
        out.append((auc, prec, int(npos)))
    print(f"{name:<34} L: AUC {out[0][0]:.3f} prec@rate {out[0][1]:.3f} (n+ {out[0][2]})   R: AUC {out[1][0]:.3f} prec@rate {out[1][1]:.3f} (n+ {out[1][2]})")

print(f"steps {len(LAB)}  press rate L {LAB[:,0].mean():.4f} R {LAB[:,1].mean():.4f}")
for m in MODES:
    fit_eval(np.array(data[m]), f"circuit readout80 linear [{m}]")
fit_eval(OBS, "raw obs linear")
fit_eval(OBS, "raw obs MLP(32)", hidden=32)
