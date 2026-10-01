"""Brain + cradle reflex module, evaluated in FULL games.

The fly brain could not imitate the stateful cradle routine (dagger_cradle.log), so the routine runs
as a fixed reflex module instead (like a local reflex arc that needs no brain): the brain plays every
step, and only while the module is in its cradle routine (catch -> hold -> settle -> release -> timed
shot, agents/experiments/cradle/cradle_teacher.CradleTeacher.override) does the module's action
replace the brain's. The brain's circuit keeps running on every step either way.

Policies on the same seeds: brain (lead brain alone), hybrid (lead brain + cradle module),
cradle_s5 (the all-reflex cradle teacher, reference).

Usage: python agents/experiments/fullgame/hybrid_eval.py --seeds 67000:67064 --workers 16
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "encoding"))
sys.path.insert(0, os.path.join(ROOT, "agents", "experiments", "cradle"))

BRAIN = os.path.join(ROOT, "agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt")
CONN = os.path.join(ROOT, "agents/experiments/big_circuit/connectome.json")
CATCH_SPEED = 5.0
POST_SHOT = int(os.environ.get("HYBRID_POST_SHOT", "8"))  # module keeps flippers down this long after its shot
# hybrid variants: name -> (catch speed, catch y above, catch |x| below)
# optional 4th element: shot forces (left, right) for analog flippers (ACTF frames)
VARIANTS = {"hybrid": (5.0, 9.0, 5.0), "hybrid_s7": (7.0, 9.0, 5.0), "hybrid_wide": (5.0, 8.0, 6.0),
            "hybrid_s7wide": (7.0, 8.0, 6.0), "hybrid_late": (7.0, 10.0, 6.0),
            "hybrid_force": (7.0, 8.0, 6.0, (1.0, 0.75)), "hybrid_late_force": (7.0, 10.0, 6.0, (1.0, 0.75)),
            "hybrid_pass": (7.0, 10.0, 6.0, (1.0, 0.75), (18, 0.33)),
            "hybrid_pass120": (7.0, 10.0, 6.0, (1.0, 0.75), (18, 0.33), 120),
            "hybrid_pass160": (7.0, 10.0, 6.0, (1.0, 0.75), (18, 0.33), 160)}
_BINARY = [None]
_AGENT = [None]
_IDS: dict = {}


def _init(binary: str) -> None:
	import signal

	import torch

	torch.set_num_threads(1)
	_BINARY[0] = binary
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
	with open(os.path.join(ROOT, "agents/table_map.json"), encoding="utf-8") as f:
		objs = json.load(f)["objects"]
	_IDS["ramp"] = frozenset(o["id"] for o in objs if o["name"] == "ramp")
	_IDS["bank"] = frozenset(o["id"] for o in objs if o["name"] in ("a_targ7", "a_targ8", "a_targ9"))


def _brain():
	if _AGENT[0] is None:
		import gymnasium as gym
		import torch

		from agents import train_pinball_circuit_cem as T
		from agents.experiments.encoding.encoding import build_encoded_agent
		from env_python.pinball_env import OBS_DIM

		ck = torch.load(BRAIN, weights_only=False)
		agent = build_encoded_agent(CONN, int(ck["readout_dim"]), ck["encoding"])
		agent.calibrate(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32))
		T.set_flat_params(agent.decoder, np.asarray(ck["champion"], dtype=np.float32))
		_AGENT[0] = agent
	return _AGENT[0]


def _game(task) -> dict:
	import torch

	from agents import train_pinball_circuit_cem as T
	from cradle_teacher import CradleTeacher
	from env_python.pinball_env import PinballEnv

	name, seed, max_steps = task
	env = PinballEnv(binary_path=_BINARY[0], headless=True, frame_skip=4, relaunch_at_rest=True)
	score = 0.0
	ramp = bank = completions = override_steps = steps = 0
	catches = cradles = eligible = 0
	extra = {"abort_bounce": 0, "abort_timeout": 0, "min_speed_sum": 0.0}
	game_over = False

	def fresh():
		if name.startswith("hybrid"):
			spec = VARIANTS.get(name, (CATCH_SPEED, 9.0, 5.0))
			speed, cy, cx = spec[:3]
			forces = spec[3] if len(spec) > 3 else None
			left_pass = spec[4] if len(spec) > 4 else None
			pass_timeout = spec[5] if len(spec) > 5 else 80
			module = CradleTeacher(catch_speed=speed, post_shot=POST_SHOT, catch_y=cy, catch_x_max=cx, forces=forces,
			                       left_pass=left_pass, pass_hold_timeout=pass_timeout)
		else:
			module = CradleTeacher(catch_speed=CATCH_SPEED)
		st = dict(module=module)
		if name == "brain" or name.startswith("hybrid"):
			st["h"] = _brain().initial_state(1)
			st["filt"] = T.FlipperObsFilter("delay1")
		return st

	try:
		obs, _ = env.reset(seed=seed)
		st = fresh()
		prev_drained = False
		with torch.no_grad():
			for _ in range(max_steps):
				module_act = st["module"].act(obs)
				if name == "cradle_s5":
					act = module_act
				else:
					o = torch.as_tensor(np.asarray(st["filt"](obs), dtype=np.float32)).unsqueeze(0)
					a, st["h"] = _brain().act(o, st["h"], greedy=True)
					act = np.asarray(a[0], dtype=np.int64)
					act[2] = 0
					if name.startswith("hybrid") and st["module"].override:
						act = module_act
						override_steps += 1
				obs, _r, term, _trunc, info = env.step(act)
				steps += 1
				score += info["score_delta"]
				for (oid, _p, _x, _y), base in zip(info.get("hit_objects", ()), info.get("hit_base_points", ())):
					ramp += int(oid in _IDS["ramp"])
					bank += int(oid in _IDS["bank"])
					completions += int(oid in _IDS["bank"] and base >= 1500)
				if info["drained"] and not prev_drained:
					m = st["module"]
					catches, cradles, eligible = catches + m.catches, cradles + m.cradles, eligible + m.eligible
					for k in extra:
						extra[k] += getattr(m, k)
					st = fresh()
				prev_drained = bool(info["drained"])
				if term:
					game_over = True
					break
		m = st["module"]
		catches, cradles, eligible = catches + m.catches, cradles + m.cradles, eligible + m.eligible
		for k in extra:
			extra[k] += getattr(m, k)
	finally:
		env.close()
	return dict(name=name, seed=seed, score=float(score), ramp=ramp, bank=bank, completions=completions,
	            override_share=override_steps / max(1, steps), game_over=game_over,
	            catches=catches, cradles=cradles, eligible=eligible, **extra)


def main() -> None:
	import confirm

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--binary", default=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"))
	ap.add_argument("--seeds", default="67000:67064")
	ap.add_argument("--workers", type=int, default=16)
	ap.add_argument("--max-steps", type=int, default=30000)
	ap.add_argument("--out", default=os.path.join(HERE, "hybrid_eval.json"))
	args = ap.parse_args()
	ap_names = os.environ.get("HYBRID_POLICIES", "brain,hybrid,cradle_s5")
	names = tuple(ap_names.split(","))
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	t0 = time.time()
	with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.binary,)) as pool:
		res = pool.map(_game, [(n, s, args.max_steps) for s in seeds for n in names], chunksize=1)
	by = {n: {r["seed"]: r for r in res if r["name"] == n} for n in names}
	print(f"paired full games {lo}..{hi - 1} ({len(seeds)})  {time.time() - t0:.0f}s")
	for n in names:
		rs = [by[n][s] for s in seeds]
		sc = np.array([r["score"] for r in rs])
		lg = np.log1p(sc)
		line = (f"{n:<10} log1p {lg.mean():.3f} +- {lg.std(ddof=1) / math.sqrt(len(lg)):.3f}  median {np.median(sc):,.0f}  "
		        f"mean {sc.mean():,.0f}  ramp {np.mean([r['ramp'] for r in rs]):.1f}  bank {np.mean([r['bank'] for r in rs]):.1f}  "
		        f"compl {np.mean([r['completions'] for r in rs]):.2f}  module {np.mean([r['override_share'] for r in rs]):.0%}  "
		        f"eligible {np.mean([r['eligible'] for r in rs]):.0f}  catches {np.mean([r['catches'] for r in rs]):.1f}  "
		        f"cradles {np.mean([r['cradles'] for r in rs]):.1f}  bounce {np.mean([r['abort_bounce'] for r in rs]):.1f}  "
		        f"timeout {np.mean([r['abort_timeout'] for r in rs]):.1f}  "
		        f"minspd {sum(r['min_speed_sum'] for r in rs) / max(1, sum(r['abort_bounce'] + r['abort_timeout'] for r in rs)):.2f}  "
		        f"over {np.mean([r['game_over'] for r in rs]):.0%}")
		if n != "brain":
			d = np.array([np.log1p(by[n][s]["score"]) - np.log1p(by["brain"][s]["score"]) for s in seeds])
			se = d.std(ddof=1) / math.sqrt(len(d))
			line += f" | vs brain {d.mean():+.3f} +- {se:.3f} t {d.mean() / se:+.2f} p {confirm.wilcoxon_p(d):.2g} wins {int((d > 0).sum())}/{int((d < 0).sum())}"
		print(line)
	with open(args.out, "w", encoding="utf-8") as fh:
		json.dump(dict(args=vars(args), games=res), fh)


if __name__ == "__main__":
	main()
