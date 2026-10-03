"""Uniform stateful policy wrappers for the step-3 diagnostics (nudge sandbox).

Spec strings:
  reflex[:Y]          correct-side pulse reflex (train_pinball_circuit_cem.ReflexLabeler, default y > 11.5)
  ckpt:PATH           fixed-circuit checkpoint on the plain observation (e.g. gen-280 champion)
  popcode:PATH        EncodedCircuitAgent (agents/experiments/encoding/encoding.py, copied read-only
                      from the step3 copy) with the checkpoint's `encoding` and `flipper_obs` filter

Each policy: reset(); act(obs) -> np.int64 array (3,) [left, right, launch=0].
If the action actually executed differs from act()'s output (jitter/oracle overrides), call
executed(action) so stateful policies (the reflex's cooldown) follow the real history; circuit
policies only see the environment's observations, so they need nothing.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
CONNECTOME = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")


class ReflexPolicy:
	def __init__(self, y: float = 11.5):
		from agents.train_pinball_circuit_cem import ReflexLabeler

		self.lab = ReflexLabeler(y, 1)
		self._last = None

	def reset(self) -> None:
		self.lab.reset()

	def act(self, obs) -> np.ndarray:
		a = self.lab.label(obs)
		a[2] = 0
		self.lab.observe(a)
		return a

	def executed(self, action) -> None:
		pass  # observe() already ran on the proposed action; overrides are rare 1-step pulses


class CircuitPolicy:
	def __init__(self, path: str, observation_space, encoded: bool):
		import torch

		from agents import train_pinball_circuit_cem as T

		torch.set_num_threads(1)
		ck = torch.load(path, weights_only=False)
		if encoded:
			from agents.experiments.encoding.encoding import build_encoded_agent

			self.agent = build_encoded_agent(CONNECTOME, ck["readout_dim"], ck["encoding"])
		else:
			self.agent = T.build_agent("circuit", CONNECTOME, ck["readout_dim"])
		self.agent.calibrate(observation_space)
		T.set_flat_params(self.agent.decoder, np.asarray(ck["champion"], dtype=np.float32))
		self.filt = T.FlipperObsFilter(ck.get("flipper_obs", "raw"), loom_tau0=ck.get("loom_tau0", 2.0))
		self.h = None

	def reset(self) -> None:
		self.h = self.agent.initial_state(1)
		self.filt.reset()

	def act(self, obs) -> np.ndarray:
		import torch

		with torch.no_grad():
			o = torch.as_tensor(np.asarray(self.filt(obs), dtype=np.float32)).unsqueeze(0)
			a, self.h = self.agent.act(o, self.h, greedy=True)
		a = np.asarray(a[0], dtype=np.int64).copy()
		a[2] = 0
		return a

	def executed(self, action) -> None:
		pass


def make_policy(spec: str, observation_space):
	kind, _, arg = spec.partition(":")
	if kind == "reflex":
		return ReflexPolicy(float(arg) if arg else 11.5)
	if kind == "ckpt":
		return CircuitPolicy(arg, observation_space, encoded=False)
	if kind == "popcode":
		return CircuitPolicy(arg, observation_space, encoded=True)
	raise ValueError(f"unknown policy spec {spec!r}")
