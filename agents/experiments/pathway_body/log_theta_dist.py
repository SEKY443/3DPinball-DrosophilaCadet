"""Log distribution of d, closing, theta and theta_dot (R=1 and R given) over a few lead1 lives (closing > 0 only)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gf_body as G
import gf_split_body as S

def main():
	from agents.train_pinball_circuit_cem import LOOM_TARGET_LEFT, LOOM_TARGET_RIGHT, ReflexLabeler
	from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_Y
	G.init_worker()
	rows = []
	for seed in range(7000, 7006):
		obs, _ = G._ENV.reset(seed=seed)
		rf = ReflexLabeler(11.5, 1)
		done = False
		while not done:
			for tg in (LOOM_TARGET_LEFT, LOOM_TARGET_RIGHT):
				th, td = S.eye_signals(obs, tg, 1.0)
				if th > 0 and td > 0:
					rows.append((th, td, obs[OBS_BALL_Y]))
			v = np.array(obs, dtype=np.float32)
			if v[OBS_BALL_VY] > 0: v[OBS_BALL_Y] += v[OBS_BALL_VY] * G._DT
			a = rf.label(v); a[2] = 0; rf.observe(a)
			obs, _r, te, tr, info = G._ENV.step(a)
			done = te or tr or bool(info["drained"])
	r = np.array(rows)
	qs = [1, 5, 25, 50, 75, 95, 99]
	print("n closing samples", len(r))
	print("theta (R=1) pct", qs, np.round(np.percentile(r[:, 0], qs), 4).tolist())
	print("theta_dot (R=1) pct", qs, np.round(np.percentile(r[:, 1], qs), 4).tolist())
	z = r[r[:, 2] > 10]
	print("y>10: n", len(z), "theta", np.round(np.percentile(z[:, 0], qs), 4).tolist(), "theta_dot", np.round(np.percentile(z[:, 1], qs), 4).tolist())
	G._ENV.close()

if __name__ == "__main__":
	main()
