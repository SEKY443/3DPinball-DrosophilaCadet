"""Where does the untrained giant-fiber body lose score against the hand-written lead1 reflex?

Both play the same paired lives (seeds 77000-77095, one ball life each). Per decision step we record the ball state, both
flipper actions, flipper_hit, score_delta and the NEXT ball velocity, then compare:
  * the score decomposition  score ~ contacts x score-per-contact (log difference split into the two factors),
  * press timing: how long the flipper had been pressed when it touched the ball, "empty" presses (no contact soon after),
  * where on the flipper the ball is hit (|x| near the tip vs the pivot), ball speed at contact,
  * what the hit does: the ball's outgoing vertical velocity (negative = up the table) and speed,
  * how lives end: x of the drain and whether the last contact was long ago.
Usage: python where_lost.py [--seeds 77000:77096] [--workers 8]   (writes where_lost.json; stdout is the report)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PB = os.path.join(ROOT, "agents", "experiments", "pathway_body")
for p in (ROOT, PB):
	if p not in sys.path:
		sys.path.insert(0, p)

import blind_spot as B  # noqa: E402
import gf_body as G  # noqa: E402

EMPTY_WINDOW = 8  # a press with no flipper contact within this many steps after its onset is "empty"


def _life(task):
	kind, params, seed = task
	from agents.train_pinball_circuit_cem import ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VX, OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y

	env = G._ENV
	obs, _ = env.reset(seed=seed)
	body = B._body(kind, params) if kind != "lead1" else None
	if body is not None:
		body.reset()
	reflex = ReflexLabeler(11.5, 1) if kind == "lead1" else None
	rows, done, drained = [], False, False
	while not done:
		x, y, vx, vy = (float(obs[i]) for i in (OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY))
		if body is not None:
			act = body.act(obs)
		else:
			view = np.array(obs, dtype=np.float32)
			if view[OBS_BALL_VY] > 0:
				view[OBS_BALL_Y] = view[OBS_BALL_Y] + view[OBS_BALL_VY] * G._DT
			act = reflex.label(view)
			act[2] = 0
			reflex.observe(act)
		obs, _r, term, trunc, info = env.step(act)
		rows.append((x, y, vx, vy, int(act[0]), int(act[1]), int(bool(info["flipper_hit"])), float(info["score_delta"]),
		             float(obs[OBS_BALL_VX]), float(obs[OBS_BALL_VY]),
		             float(body.last_a[0]) if body is not None else 0.0, float(body.last_a[1]) if body is not None else 0.0))
		drained = bool(info["drained"]) or (bool(term) and not trunc)
		done = term or trunc or drained
	return dict(kind=kind, seed=seed, drained=drained, rows=np.asarray(rows, dtype=np.float32))


def analyse(lives):
	score = np.array([l["rows"][:, 7].sum() for l in lives])
	contacts = np.array([l["rows"][:, 6].sum() for l in lives])
	steps = np.array([len(l["rows"]) for l in lives])
	presses = empty = 0
	ages, xs, spd, vy_out, spd_out, held_long = [], [], [], [], [], 0
	n_contact = 0
	last_gap, drain_x = [], []
	for l in lives:
		r = l["rows"]
		hit = np.nonzero(r[:, 6] > 0.5)[0]
		for side, col in ((0, 4), (1, 5)):
			a = r[:, col]
			onsets = np.nonzero((a[1:] > 0.5) & (a[:-1] < 0.5))[0] + 1
			if len(a) and a[0] > 0.5:
				onsets = np.concatenate([[0], onsets])
			presses += len(onsets)
			for o in onsets:
				if not np.any((hit >= o) & (hit <= o + EMPTY_WINDOW)):
					empty += 1
		for t in hit:
			side = 0 if r[t, 0] > 0 else 1
			col = 4 if side == 0 else 5
			age = 0
			k = t
			while k >= 0 and r[k, col] > 0.5:
				age += 1
				k -= 1
			n_contact += 1
			ages.append(age)
			xs.append(abs(r[t, 0]))
			spd.append(math.hypot(r[t, 2], r[t, 3]))
			vy_out.append(r[t, 9])
			spd_out.append(math.hypot(r[t, 8], r[t, 9]))
			held_long += int(age >= 4)
		if l["drained"]:
			drain_x.append(abs(float(r[-1, 0])))
			last_gap.append(len(r) - 1 - (hit[-1] if len(hit) else -1))
	# How drains happen: dwell in the strike zone (10.6 <= y <= 13.4, |x| < 2.6) during the last 150 steps, how slow the ball was
	# there, and whether the flipper on the ball's side was pressed while it dwelled. Split by centre-gap drains (|x| < 0.96).
	dw = {"centre": [], "other": []}
	for l in lives:
		if not l["drained"]:
			continue
		r = l["rows"][-150:]
		z = (r[:, 1] >= 10.6) & (r[:, 1] <= 13.4) & (np.abs(r[:, 0]) < 2.6)
		sp = np.hypot(r[:, 2], r[:, 3])
		side_pressed = np.where(r[:, 0] > 0, r[:, 4], r[:, 5]) > 0.5
		key = "centre" if abs(float(l["rows"][-1, 0])) < 0.96 else "other"
		dw[key].append((int(z.sum()), float(np.mean(sp[z] < 5)) if z.any() else 0.0, float(np.median(sp[z])) if z.any() else float("nan"),
		                int((z & side_pressed).sum()), bool(np.any(r[:, 6] > 0.5))))
	def summ(v):
		if not v:
			return dict(n=0)
		v = np.array(v, dtype=float)
		return dict(n=len(v), dwell_steps=float(v[:, 0].mean()), slow_frac=float(v[:, 1].mean()), median_speed=float(np.nanmedian(v[:, 2])),
		            pressed_steps=float(v[:, 3].mean()), any_press_frac=float(np.mean(v[:, 3] > 0)), contact_in_last150=float(v[:, 4].mean()))
	drains = {k: summ(v) for k, v in dw.items()}
	# Final approach: the 12 steps before the ball first crosses the resting-tip line (y > 13.1) in the last 200 steps.
	ap_ = {"centre": [], "other": []}
	for l in lives:
		if not l["drained"]:
			continue
		r = l["rows"][-200:]
		cross = np.nonzero(r[:, 1] > 13.1)[0]
		i = int(cross[0]) if len(cross) else len(r) - 1
		w = r[max(0, i - 12):i + 1]
		if len(w) < 4:
			continue
		near = (w[:, 0] > 0).astype(int)   # side of the ball: x > 0 -> left flipper (action 0)
		act = np.where(near == 0, w[:, 4], w[:, 5])
		gf = np.where(near == 0, w[:, 10], w[:, 11])
		key = "centre" if abs(float(l["rows"][-1, 0])) < 0.96 else "other"
		ap_[key].append((abs(float(r[i, 0])), float(np.median(w[:, 3])), float(np.median(np.hypot(w[:, 2], w[:, 3]))), float(gf.max()),
		                 float(act.max()), float((w[:, 3] > 0).mean())))
	approach = {}
	for k, v in ap_.items():
		v = np.array(v, dtype=float)
		approach[k] = dict(n=len(v)) if not len(v) else dict(n=len(v), absx_at_tip_line=float(np.median(v[:, 0])), median_vy=float(np.median(v[:, 1])),
		                                                   median_speed=float(np.median(v[:, 2])), gf_max=float(np.median(v[:, 3])),
		                                                   pressed_frac=float(v[:, 4].mean()), moving_down_frac=float(v[:, 5].mean()))
	lg = np.log1p(score)
	spc = np.where(contacts > 0, score / np.maximum(contacts, 1), np.nan)
	out = dict(n=len(lives), log1p=float(lg.mean()), log1p_se=float(lg.std(ddof=1) / math.sqrt(len(lg))), median_score=float(np.median(score)),
	           mean_score=float(score.mean()), steps=float(steps.mean()), contacts=float(contacts.mean()), zero_contact_lives=int((contacts == 0).sum()),
	           score_per_contact=float(np.nansum(score) / max(1.0, contacts.sum())), presses=presses / len(lives),
	           empty_press_frac=empty / max(1, presses), contacts_per_press=n_contact / max(1, presses),
	           contact_age_median=float(np.median(ages)), contact_age_mean=float(np.mean(ages)), held_ge4_frac=held_long / max(1, n_contact),
	           contact_absx_median=float(np.median(xs)), tip_frac=float(np.mean(np.array(xs) < 1.3)), pivot_frac=float(np.mean(np.array(xs) > 2.0)),
	           contact_speed_median=float(np.median(spd)), out_vy_median=float(np.median(vy_out)), out_up_frac=float(np.mean(np.array(vy_out) < -5)),
	           out_speed_median=float(np.median(spd_out)),
	           drain_absx_median=float(np.median(drain_x)) if drain_x else float("nan"),
	           drain_centre_frac=float(np.mean(np.array(drain_x) < 0.96)) if drain_x else float("nan"),
	           last_contact_gap_median=float(np.median(last_gap)) if last_gap else float("nan"), drains=drains, approach=approach)
	return out, score, contacts


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--seeds", default="77000:77096")
	ap.add_argument("--workers", type=int, default=os.cpu_count())
	ap.add_argument("--save", default=None, help="write the raw trajectories to this .npz (keep it outside the repo)")
	args = ap.parse_args()
	lo, hi = (int(v) for v in args.seeds.split(":"))
	seeds = list(range(lo, hi))
	sweep = json.load(open(os.path.join(PB, "best_retina_real_sweep.json")))["params"]
	tasks = [(k, p, s) for s in seeds for k, p in (("retina", sweep), ("lead1", None))]
	with mp.get_context("spawn").Pool(args.workers, initializer=G.init_worker) as pool:
		res = pool.map(_life, tasks, chunksize=1)
	by = {k: sorted([r for r in res if r["kind"] == k], key=lambda r: r["seed"]) for k in ("retina", "lead1")}
	if args.save:  # raw per-step rows for offline cuts: cols x,y,vx,vy,actL,actR,flipper_hit,score_delta,next_vx,next_vy,gfL,gfR
		np.savez_compressed(args.save, **{f"{k}_{l['seed']}": l["rows"] for k in by for l in by[k]},
		                    **{f"{k}_{l['seed']}_drained": np.array(l["drained"]) for k in by for l in by[k]})
	stats, sc, ct = {}, {}, {}
	for k in by:
		stats[k], sc[k], ct[k] = analyse(by[k])
	names = {"retina": "fly (sweep body)", "lead1": "lead1 reflex"}
	keys = [("log1p", "log1p score"), ("median_score", "median score"), ("mean_score", "mean score"), ("steps", "steps/life"),
	        ("contacts", "flipper contacts/life"), ("zero_contact_lives", "lives with 0 contacts"), ("score_per_contact", "score per contact"),
	        ("presses", "presses/life"), ("empty_press_frac", "empty presses (no contact within 8 steps)"), ("contacts_per_press", "contacts per press"),
	        ("contact_age_median", "flipper held (steps) at contact, median"), ("held_ge4_frac", "contacts after >=4 steps held"),
	        ("contact_absx_median", "|x| of contact, median"), ("tip_frac", "contacts near the tip (|x|<1.3)"), ("pivot_frac", "contacts near the pivot (|x|>2.0)"),
	        ("contact_speed_median", "ball speed at contact, median"), ("out_vy_median", "outgoing vy, median (neg = up)"),
	        ("out_up_frac", "strong upward hits (vy<-5)"), ("out_speed_median", "outgoing speed, median"),
	        ("drain_absx_median", "drain |x|, median"), ("drain_centre_frac", "drains in the centre gap (|x|<0.96)"),
	        ("last_contact_gap_median", "steps from last contact to drain, median")]
	print(f"paired lives {lo}..{hi - 1} ({len(seeds)})")
	print(f"{'':54s} {names['retina']:>18s} {names['lead1']:>14s}")
	for key, label in keys:
		a, b = stats["retina"][key], stats["lead1"][key]
		print(f"{label:54s} {a:18.3f} {b:14.3f}")
	d = np.log1p(sc["lead1"]) - np.log1p(sc["retina"])
	se = d.std(ddof=1) / math.sqrt(len(d))
	print("\ndrains by type, last 150 steps before the drain (zone = strike zone near the flippers):")
	for k in ("retina", "lead1"):
		for typ in ("centre", "other"):
			d_ = stats[k]["drains"][typ]
			if d_["n"]:
				print(f"  {names[k]:18s} {typ:6s} drains n={d_['n']:3d}: dwell {d_['dwell_steps']:5.1f} steps in zone, slow(<5) {100 * d_['slow_frac']:3.0f}%, "
				      f"median speed {d_['median_speed']:5.1f}, flipper pressed on {d_['pressed_steps']:4.1f} of them, any press {100 * d_['any_press_frac']:3.0f}%, "
				      f"a contact in last 150 steps {100 * d_['contact_in_last150']:3.0f}%")
	print("\nfinal approach (12 steps before the ball crosses y=13.1), medians over drains:")
	for k in ("retina", "lead1"):
		for typ in ("centre", "other"):
			d_ = stats[k]["approach"][typ]
			if d_["n"]:
				print(f"  {names[k]:18s} {typ:6s} n={d_['n']:3d}: |x| at the line {d_['absx_at_tip_line']:4.2f}, vy {d_['median_vy']:5.1f}, speed {d_['median_speed']:5.1f}, "
				      f"near-side GF max {d_['gf_max']:5.2f}, flipper pressed in window {100 * d_['pressed_frac']:3.0f}%, steps moving down {100 * d_['moving_down_frac']:3.0f}%")
	print(f"\nlead1 - fly log1p: {d.mean():+.3f} +- {se:.3f} (t {d.mean() / se:+.2f})")
	c_ratio = math.log(stats["lead1"]["contacts"] / stats["retina"]["contacts"])
	p_ratio = math.log(stats["lead1"]["score_per_contact"] / stats["retina"]["score_per_contact"])
	print(f"mean-score ratio lead1/fly: contacts factor {math.exp(c_ratio):.2f}x, score-per-contact factor {math.exp(p_ratio):.2f}x")
	json.dump(dict(stats=stats), open(os.path.join(HERE, "where_lost.json"), "w"), indent=1)


if __name__ == "__main__":
	main()
