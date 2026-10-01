"""DAgger-seed the popcode circuit from the one-step-lead reflex instead of the plain reflex.

Wraps agents/experiments/encoding/dagger_enc.py without editing it: at import time (so also in every
spawned worker, which re-imports this file as __mp_main__) the trainer's ReflexLabeler is replaced by
LeadReflexLabeler, which feeds the reflex the ball's predicted y one decision step ahead
(y + vy * DT_STEP when falling). DT_STEP comes from reflex_lead_eval.py's estimate.

Usage: python agents/experiments/timing/dagger_lead.py --variants popcode:delay1:8 --workers 1 \
         --ckpt-dir agents/experiments/timing/seeds --out agents/experiments/timing/dagger_lead.json
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))

from agents import train_pinball_circuit_cem as T  # noqa: E402
from agents.experiments.encoding import dagger_enc  # noqa: E402

DT_STEP = 0.0347  # y-units per vy-unit per decision step (reflex_lead_9000.log)
_BaseReflex = T.ReflexLabeler


class LeadReflexLabeler(_BaseReflex):
	def label(self, obs, ball_in_play: bool = True):
		from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y

		view = np.array(obs, dtype=np.float32)
		if view[OBS_BALL_VY] > 0:
			view[OBS_BALL_Y] = view[OBS_BALL_Y] + view[OBS_BALL_VY] * DT_STEP
		return super().label(view, ball_in_play)


T.ReflexLabeler = LeadReflexLabeler

if __name__ == "__main__":
	dagger_enc.main()
