"""Diagnose WHERE score comes from during a ball's life: launch phase vs later, table position
at scoring time, and drain position (center gap / flipper-reachable / outlane).

Usage: score_sources.py <policy> <n_episodes> <seed0> <out_json>
policy: "none" (never press) or "ckpt:<path>" (CMA-format checkpoint, greedy action).
"""
import json
import os
import sys

import numpy as np

ROOT = "/Users/seky/Developer/Git/3DPinball-DrosophilaCadet"
sys.path.insert(0, ROOT)
import gymnasium as gym  # noqa: E402

from env_python.pinball_env import OBS_BALL_VY, OBS_BALL_X, OBS_BALL_Y, PinballEnv  # noqa: E402

POLICY = sys.argv[1]
N_EP = int(sys.argv[2])
SEED0 = int(sys.argv[3])
OUT_JSON = sys.argv[4]

MAX_STEPS = 3000
LAUNCH_PHASE_STEPS = 100

# Score-position grid: 5 x-bins over [-8, 8], 6 y-bins over [-13, 15].
X_LO, X_HI, X_BINS = -8.0, 8.0, 5
Y_LO, Y_HI, Y_BINS = -13.0, 15.0, 6

# Drain-position classification thresholds (table coords; flipper pivots at x=+-2.489).
CENTER_GAP_X = 1.5
OUTLANE_X = 3.2

env = gym.wrappers.TimeLimit(
    PinballEnv(binary_path=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"),
               headless=True, frame_skip=4),
    max_episode_steps=MAX_STEPS)

AGENT = None
if POLICY.startswith("ckpt:"):
    import torch
    sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
    from agents.train_pinball_circuit_cem import build_agent, set_flat_params
    torch.set_num_threads(1)
    ck = torch.load(POLICY.split(":", 1)[1], weights_only=False)
    AGENT = build_agent(ck.get("agent", "circuit"), os.path.join(ROOT, "web/connectome.json"), ck["readout_dim"])
    AGENT.calibrate(env.observation_space)
    set_flat_params(AGENT.decoder, ck["champion"])
    H = [None]


def policy_action(obs, t):
    if AGENT is not None:
        import torch
        if t == 0:
            H[0] = AGENT.initial_state(1)
        a, H[0] = AGENT.act(torch.as_tensor(obs).unsqueeze(0), H[0], greedy=True)
        return int(a[0][0]), int(a[0][1])
    if POLICY == "none":
        return 0, 0
    raise ValueError(f"unsupported policy {POLICY!r}")


def x_bin(x):
    idx = int((x - X_LO) / (X_HI - X_LO) * X_BINS)
    return min(max(idx, 0), X_BINS - 1)


def y_bin(y):
    idx = int((y - Y_LO) / (Y_HI - Y_LO) * Y_BINS)
    return min(max(idx, 0), Y_BINS - 1)


def classify_drain(x):
    ax = abs(x)
    if ax < CENTER_GAP_X:
        return "center_gap"
    if ax < OUTLANE_X:
        return "flipper_reachable"
    return "outlane"


episodes = []
grid = np.zeros((X_BINS, Y_BINS), dtype=float)

for ep in range(N_EP):
    obs, info = env.reset(seed=SEED0 + ep)
    prev_obs = obs
    t = 0
    total_score = 0.0
    score_launch = 0.0
    score_after = 0.0
    flipper_hit_count = 0
    drained = False
    truncated = False
    drain_x = None
    drain_y = None
    done = False
    while not done:
        l, r = policy_action(obs, t)
        prev_obs = obs
        obs, _, term, trunc, info = env.step(np.array([l, r, 0]))
        sd = float(info["score_delta"])
        total_score += sd
        if t < LAUNCH_PHASE_STEPS:
            score_launch += sd
        else:
            score_after += sd
        if sd != 0.0:
            grid[x_bin(obs[OBS_BALL_X]), y_bin(obs[OBS_BALL_Y])] += sd
        if info["flipper_hit"]:
            flipper_hit_count += 1
        t += 1
        if info["drained"]:
            drained = True
            drain_x = float(prev_obs[OBS_BALL_X])
            drain_y = float(prev_obs[OBS_BALL_Y])
            done = True
        elif term or trunc:
            truncated = bool(trunc) and not term
            done = True

    episodes.append({
        "seed": SEED0 + ep,
        "score": total_score,
        "life_steps": t,
        "drained": drained,
        "truncated": truncated,
        "drain_x": drain_x,
        "drain_y": drain_y,
        "score_launch": score_launch,
        "score_after": score_after,
        "flipper_hit_count": flipper_hit_count,
    })
    print(f"  ep {ep:3d} seed {SEED0+ep} score {total_score:8.0f} life {t:5d} "
          f"drained {drained} trunc {truncated} drain_xy "
          f"{'' if drain_x is None else f'({drain_x:.2f},{drain_y:.2f})'}", flush=True)

env.close()

scores = np.array([e["score"] for e in episodes], dtype=float)
lives = np.array([e["life_steps"] for e in episodes], dtype=float)
drain_xs = np.array([e["drain_x"] for e in episodes if e["drain_x"] is not None], dtype=float)
n_drained = int(np.sum([e["drained"] for e in episodes]))
n_truncated = int(np.sum([e["truncated"] for e in episodes]))
total_score_sum = float(scores.sum())
launch_sum = float(sum(e["score_launch"] for e in episodes))
after_sum = float(sum(e["score_after"] for e in episodes))
launch_frac = launch_sum / total_score_sum if total_score_sum != 0 else float("nan")

if len(scores) > 1 and scores.std() > 0 and lives.std() > 0:
    corr = float(np.corrcoef(lives, scores)[0, 1])
else:
    corr = float("nan")

drain_classes = {"center_gap": 0, "flipper_reachable": 0, "outlane": 0}
for x in drain_xs:
    drain_classes[classify_drain(x)] += 1

grid_flat = []
for xi in range(X_BINS):
    for yi in range(Y_BINS):
        x_lo = X_LO + xi * (X_HI - X_LO) / X_BINS
        x_hi = X_LO + (xi + 1) * (X_HI - X_LO) / X_BINS
        y_lo = Y_LO + yi * (Y_HI - Y_LO) / Y_BINS
        y_hi = Y_LO + (yi + 1) * (Y_HI - Y_LO) / Y_BINS
        val = float(grid[xi, yi])
        grid_flat.append({
            "x_bin": xi, "y_bin": yi,
            "x_range": [x_lo, x_hi], "y_range": [y_lo, y_hi],
            "score": val,
            "share": val / total_score_sum if total_score_sum != 0 else float("nan"),
        })
grid_flat_sorted = sorted(grid_flat, key=lambda c: -c["score"])

aggregates = {
    "policy": POLICY,
    "n_episodes": N_EP,
    "seed0": SEED0,
    "score_mean": float(scores.mean()),
    "score_median": float(np.median(scores)),
    "life_mean": float(lives.mean()),
    "life_median": float(np.median(lives)),
    "n_drained": n_drained,
    "n_truncated": n_truncated,
    "total_score_sum": total_score_sum,
    "launch_phase_score_sum": launch_sum,
    "after_launch_score_sum": after_sum,
    "launch_phase_score_fraction": launch_frac,
    "life_score_correlation": corr,
    "drain_x_values": [float(x) for x in drain_xs],
    "drain_classification_counts": drain_classes,
    "score_grid": grid_flat_sorted,
}

with open(OUT_JSON, "w") as f:
    json.dump({"episodes": episodes, "aggregates": aggregates}, f, indent=1)

print("=" * 70)
print(f"policy={POLICY} n={N_EP} seed0={SEED0}")
print(f"score  mean={aggregates['score_mean']:.1f}  median={aggregates['score_median']:.1f}")
print(f"life   mean={aggregates['life_mean']:.1f}  median={aggregates['life_median']:.1f}  "
      f"drained={n_drained}/{N_EP}  truncated={n_truncated}/{N_EP}")
print(f"launch-phase (steps<{LAUNCH_PHASE_STEPS}) score fraction: {launch_frac:.3f} "
      f"(launch_sum={launch_sum:.0f} after_sum={after_sum:.0f} total={total_score_sum:.0f})")
print(f"life-vs-score correlation: {corr:.3f}")
print(f"drain x values (n={len(drain_xs)}): {np.array2string(drain_xs, precision=2, max_line_width=200)}")
print(f"drain classification (|x|<{CENTER_GAP_X}=center_gap, <{OUTLANE_X}=flipper_reachable, else outlane): "
      f"{drain_classes}")
print("top 5 score-grid cells by share of total score:")
for c in grid_flat_sorted[:5]:
    print(f"  x{c['x_range']} y{c['y_range']}  score={c['score']:.0f}  share={c['share']:.3f}")
