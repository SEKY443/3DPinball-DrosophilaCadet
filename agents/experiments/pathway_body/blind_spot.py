"""Blind-spot diagnosis of the giant-fiber pinball bodies (split single-point body vs retinotopic body).

(a) Static map: ball states (x, y, vx, vy) sampled from real recorded lives (lower table, heading down) are pushed through
    the eye model; a state is "blind" when both LC4 (tanh(theta_dot / v0)) and LPLC2 (tanh(theta / s0)) normalised drives,
    maximised over the body's receptive-field points, stay below --thr. Prints an ASCII heatmap of the blind fraction per
    (x, y) bin plus a synthetic table of the max normalised drive for a ball falling straight down.
(b) Behavioural: over --lives lives, a drain is "unseen" when the ball was in flipper reach (y > 10.5, |x| < 3.1) in the last
    N steps before it but no flipper was pressed (and the GF output stayed below theta_on on both sides) in those N steps.
    Reports % unseen, where the drains happen (x at drain), and the share of flipper-reach passes with no press.

Usage: python blind_spot.py --which current|retina [--params best_retina_real.json] [--lives 64]
Shared with eval_gf_retina.py: run_life_traj, drain_summary.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gf_body as G  # noqa: E402
import gf_retina_body as RB  # noqa: E402
import gf_split_body as S  # noqa: E402

REACH_Y, REACH_X = 10.5, 3.1  # flipper AABB (agents/table_map.json): y_min 10.51, |x| up to 3.10
LAST_N = 20
SPLIT_POINTS = np.array([[1.7839, 12.0688], [-1.7839, 12.0688]])


def _body(kind: str, params: dict):
	key = kind + json.dumps(params, sort_keys=True)
	if key not in G._BODIES:
		G._BODIES[key] = RB.RetinaGiantFiberBody(**params) if kind == "retina" else S.SplitGiantFiberBody(**params)
	b = G._BODIES[key]
	b.reset()
	return b


def run_life_traj(kind: str, params: dict | None, seed: int, keep_traj: bool = False) -> dict:
	"""kind: 'split' | 'retina' | 'lead1'. Same life as gf_body.run_life plus a per-step trace and drain summary."""
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y, OBS_FLIPPER_LEFT, OBS_FLIPPER_RIGHT

	env = G._ENV
	obs, _ = env.reset(seed=seed)
	body = reflex = None
	if kind == "lead1":
		reflex = ReflexLabeler(11.5, 1)
	else:
		body = _body(kind, params)
	prev_l, prev_r = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
	score = steps = contacts = presses = 0
	rows = []
	done, drained = False, False
	while not done:
		x, y, vx, vy = (float(obs[i]) for i in (OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY))
		if body is not None:
			act = body.act(obs)
			a = body.last_a
		else:
			view = np.array(obs, dtype=np.float32)
			if view[OBS_BALL_VY] > 0:
				view[OBS_BALL_Y] = view[OBS_BALL_Y] + view[OBS_BALL_VY] * G._DT
			act = reflex.label(view)
			act[2] = 0
			reflex.observe(act)
			a = (0.0, 0.0)
		rows.append((x, y, vx, vy, a[0], a[1], float(act[0]), float(act[1])))
		obs, _r, term, trunc, info = env.step(act)
		score += info["score_delta"]
		contacts += int(info["flipper_hit"])
		lu, ru = obs[OBS_FLIPPER_LEFT] > 0.5, obs[OBS_FLIPPER_RIGHT] > 0.5
		presses += int(lu and not prev_l) + int(ru and not prev_r)
		prev_l, prev_r = lu, ru
		steps += 1
		drained = bool(info["drained"]) or (bool(term) and not trunc)  # life over without hitting the step cap
		done = term or trunc or drained
	tr = np.asarray(rows, dtype=np.float32)
	out = dict(kind=kind, seed=seed, score=float(score), steps=steps, contacts=contacts, presses=presses)
	out.update(drain_summary(tr, drained, getattr(body, "theta_on", None)))
	if keep_traj:
		out["traj"] = tr
	return out


def in_reach(tr: np.ndarray) -> np.ndarray:
	return (tr[:, 1] > REACH_Y) & (np.abs(tr[:, 0]) < REACH_X)


def drain_summary(tr: np.ndarray, drained: bool, theta_on: float | None, n: int = LAST_N) -> dict:
	"""Drain-level unseen flag plus pass-level (contiguous in-reach run) press statistics."""
	reach = in_reach(tr)
	pressed = (tr[:, 6] > 0.5) | (tr[:, 7] > 0.5)
	# passes: contiguous in-reach runs while the ball moves down at some point of the run
	passes = missed = 0
	i = 0
	while i < len(tr):
		if reach[i]:
			j = i
			while j + 1 < len(tr) and reach[j + 1]:
				j += 1
			if tr[i:j + 1, 3].max() > 0:
				passes += 1
				missed += int(not pressed[i:j + 1].any())
			i = j + 1
		else:
			i += 1
	out = dict(drained=bool(drained), passes=passes, missed_passes=missed, drain_x=float("nan"), drain_reach=False, unseen=False)
	if drained and len(tr):
		last = slice(max(0, len(tr) - n), len(tr))
		low_gf = True if theta_on is None else bool(max(tr[last, 4].max(), tr[last, 5].max()) < theta_on)
		out["drain_x"] = float(tr[-1, 0])
		out["drain_reach"] = bool(reach[last].any())
		out["unseen"] = bool(out["drain_reach"] and not pressed[last].any() and low_gf)
	return out


def eye_visibility(obs_xyvv: np.ndarray, kind: str, params: dict) -> tuple[float, float]:
	"""(max LC4, max LPLC2) normalised drive of a ball state over the body's receptive-field points."""
	if kind == "retina":
		b = _body("retina", params)
		tg4, tg2 = b.targets[b.is_lc4], b.targets[~b.is_lc4]
	else:
		tg4 = tg2 = SPLIT_POINTS
	obs = np.zeros(15)
	obs[:4] = obs_xyvv
	_, td = RB.cell_signals(obs, tg4, params["R"])
	th, _ = RB.cell_signals(obs, tg2, params["R"])
	return float(np.tanh(td.max() / params["v0"])), float(np.tanh(th.max() / params["s0"]))


def _char(frac: float | None) -> str:
	if frac is None:
		return " "
	for lim, c in ((0.02, "."), (0.15, ":"), (0.4, "o"), (0.7, "O")):
		if frac < lim:
			return c
	return "#"


def static_report(trajs: list[np.ndarray], kind: str, params: dict, thr: float) -> dict:
	pts = np.concatenate([t[:, :4] for t in trajs])
	pts = pts[(pts[:, 1] > 8.0) & (pts[:, 3] > 0.5)]
	vis = np.array([eye_visibility(p, kind, params) for p in pts])
	blind = vis.max(axis=1) < thr
	xe = np.arange(-5.0, 5.01, 0.5)
	ye = np.arange(8.0, 14.51, 0.5)
	print(f"  static map: {len(pts)} recorded down-moving states (y > 8, vy > 0.5); blind = max(LC4, LPLC2) normalised drive < {thr}")
	print("  blind fraction per bin (rows y 8.0 -> 14.5, cols x -5 -> +5, width 0.5):  ' ' none  . <2%  : <15%  o <40%  O <70%  # >=70%")
	hdr = "".join("|" if abs(xv) < 1e-9 else ("+" if abs(abs(xv) - 2.5) < 1e-9 else " ") for xv in xe[:-1] + 0.25)
	print("        " + hdr + "   (| x=0 axis, + pivots at +-2.5)")
	for j in range(len(ye) - 1):
		line = ""
		for i in range(len(xe) - 1):
			m = (pts[:, 0] >= xe[i]) & (pts[:, 0] < xe[i + 1]) & (pts[:, 1] >= ye[j]) & (pts[:, 1] < ye[j + 1])
			line += _char(float(blind[m].mean()) if m.sum() >= 3 else None)
		print(f"  y{ye[j]:5.1f} {line}")
	zone = (pts[:, 1] > 10.5) & (np.abs(pts[:, 0]) < REACH_X)
	cen = zone & (np.abs(pts[:, 0]) < 0.96)
	res = dict(n=len(pts), blind_frac_all=float(blind.mean()), blind_frac_zone=float(blind[zone].mean()) if zone.any() else float("nan"),
	           blind_frac_centre=float(blind[cen].mean()) if cen.any() else float("nan"), n_zone=int(zone.sum()), n_centre=int(cen.sum()))
	print(f"  blind fraction: all {res['blind_frac_all']:.3f} | flipper zone (y>10.5,|x|<3.1; n={res['n_zone']}) {res['blind_frac_zone']:.3f} "
	      f"| centre gap (|x|<0.96; n={res['n_centre']}) {res['blind_frac_centre']:.3f}")
	print("  synthetic: max normalised drive x10 (0-9) for a ball falling straight down (vx=0, vy=8); rows y, cols x -4..4 step 0.5")
	for yv in (8.0, 10.0, 11.0, 12.0, 13.0):
		cells = "".join(str(min(9, int(10 * max(eye_visibility(np.array([xv, yv, 0.0, 8.0]), kind, params))))) for xv in np.arange(-4, 4.01, 0.5))
		print(f"  y{yv:5.1f} {cells}")
	return res


def behavioural_report(res: list[dict]) -> dict:
	dr = [r for r in res if r["drained"]]
	reach = [r for r in dr if r["drain_reach"]]
	un = [r for r in dr if r["unseen"]]
	passes, missed = sum(r["passes"] for r in res), sum(r["missed_passes"] for r in res)
	out = dict(lives=len(res), drains=len(dr), drains_reach=len(reach), unseen=len(un),
	           pct_unseen=100.0 * len(un) / max(1, len(dr)), passes=passes, missed_passes=missed, pct_missed_passes=100.0 * missed / max(1, passes))
	print(f"  lives {len(res)}, drains {len(dr)} (ball in flipper reach in last {LAST_N} steps: {len(reach)}); "
	      f"UNSEEN drains {len(un)} = {out['pct_unseen']:.1f}% of drains")
	print(f"  flipper-reach passes {passes}, with no press during the pass {missed} = {out['pct_missed_passes']:.1f}%")
	for name, lo, hi in (("centre gap |x|<0.96", 0.0, 0.96), ("flipper span 0.96-2.49", 0.96, 2.489), ("outlane >2.49", 2.489, 99.0)):
		sel = [r for r in dr if lo <= abs(r["drain_x"]) < hi]
		su = [r for r in sel if r["unseen"]]
		out[f"drains_{name.split()[0]}"] = (len(sel), len(su))
		print(f"    drains in {name:<24} {len(sel):3d}   unseen {len(su):3d} ({100.0 * len(su) / max(1, len(sel)):.0f}%)")
	xe = np.arange(-5.0, 5.01, 1.0)
	print("    drain x histogram (bin: drains/unseen) " + " ".join(
		f"[{xe[i]:+.0f},{xe[i + 1]:+.0f}):{sum(1 for r in dr if xe[i] <= r['drain_x'] < xe[i + 1])}/"
		f"{sum(1 for r in un if xe[i] <= r['drain_x'] < xe[i + 1])}" for i in range(len(xe) - 1)))
	return out


def _task(t):
	kind, params, seed = t
	return run_life_traj(kind, params, seed, keep_traj=True)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--which", choices=["current", "retina"], required=True)
	ap.add_argument("--params", default=None, help="json with a 'params' dict (default: eval_scan_best.json real_best / best_retina_real.json)")
	ap.add_argument("--lives", type=int, default=64)
	ap.add_argument("--seed0", type=int, default=9000)
	ap.add_argument("--thr", type=float, default=0.25)
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--out", default=None)
	args = ap.parse_args()
	if args.which == "current":
		kind = "split"
		params = json.load(open(args.params or os.path.join(G.HERE, "eval_scan_best.json")))["params"]
		params = params["real_best"] if "real_best" in params else params
	else:
		kind = "retina"
		params = json.load(open(args.params or os.path.join(G.HERE, "best_retina_real.json")))["params"]
	print(f"== blind-spot diagnosis: {args.which} body ({kind}); params " + json.dumps({k: v for k, v in params.items() if k != "circuit_json"}))
	tasks = [(kind, params, args.seed0 + i) for i in range(args.lives)]
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		res = pool.map(_task, tasks, chunksize=1)
	print("(a) static")
	st = static_report([r["traj"] for r in res], kind, params, args.thr)
	print("(b) behavioural")
	be = behavioural_report(res)
	if args.out:
		json.dump(dict(which=args.which, params=params, static=st, behavioural=be), open(args.out, "w"), indent=1)


if __name__ == "__main__":
	main()
