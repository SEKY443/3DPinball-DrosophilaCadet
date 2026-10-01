#!/usr/bin/env python3
"""Re-check step 3's flipper-zone probe (circuit_probe.ridge_r2: ridge lambda = 1.0 * n_train on
UNstandardized readout activity, leave-one-life-out) against a standardized ridge (lambda = 1) on the
same broadcast/delay1 readout of the reflex rollouts."""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
from agents.experiments.encoding import probe as P
from agents.experiments.step3.circuit_probe import ridge_r2 as step3_ridge
from agents.experiments.encoding.encoding import build_encoded_agent
import gymnasium as gym

d = dict(np.load(os.path.join(ROOT, "agents/experiments/encoding/reflex_rollouts_7000.npz")))
obs, life = d["obs"], d["life"]
agent = build_encoded_agent(P.CONN, 8, "broadcast")
agent.calibrate(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32))
H, _ = P.run_circuit(agent, P.filtered_obs(obs, life, "delay1"), life)
R = H[:, agent.decoder.idx.numpy()]
z = obs[:, 1] > 10
Y = obs[z][:, [0, 1, 3]]
print("step-3 ridge (lam=n, raw units)  x/y/vy:", np.round(step3_ridge(R[z], Y, life[z]), 2))
Rs = (R[z] - R[z].mean(0)) / (R[z].std(0) + 1e-6)
print("step-3 ridge on standardized     x/y/vy:", np.round(step3_ridge(Rs, Y, life[z], lam=1e-4), 2))
print("readout activity std in zone: median %.2e" % np.median(R[z].std(0)))
