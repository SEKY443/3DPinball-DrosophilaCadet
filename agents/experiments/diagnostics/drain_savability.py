"""Diagnose how many ball drains could have been prevented by a single, perfectly-timed
flipper press. The engine is deterministic given env.reset(seed=s) and the action sequence, so
a base trajectory can be replayed exactly and branched at any step by forcing a press.

Usage: drain_savability.py <policy> <n_seeds> <seed0> <out_json>
policy: "none" (never press) or "ckpt:<path>" (CMA-format checkpoint, greedy action).

Reuses env setup / policy conventions from score_sources.py (same session, same directory).
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = "/Users/seky/Developer/Git/3DPinball-DrosophilaCadet"
sys.path.insert(0, ROOT)

POLICY = sys.argv[1]
N_SEEDS = int(sys.argv[2])
SEED0 = int(sys.argv[3])
OUT_JSON = sys.argv[4]

MAX_STEPS = 3000
OFFSETS = list(range(-5, 41))
PASS_Y = 13.9
PRESS_TYPES = ("left", "right", "both")
PRESS_DURATION = 3
SAVE_WINDOW = 80
SAVE_Y_THRESH = 5.0
N_WORKERS = 8

CENTER_GAP_X = 1.5
OUTLANE_X = 3.2

PRESS_ACTIONS = {
    "left": (1, 0),
    "right": (0, 1),
    "both": (1, 1),
}

_WORKER_ENV = None
_WORKER_AGENT = None
_WORKER_H = None


def _make_env():
    import gymnasium as gym
    from env_python.pinball_env import PinballEnv
    return gym.wrappers.TimeLimit(
        PinballEnv(binary_path=os.path.join(ROOT, "vendor/SpaceCadetPinball/bin/SpaceCadetPinball"),
                   headless=True, frame_skip=4),
        max_episode_steps=MAX_STEPS)


def _worker_init(policy):
    global _WORKER_ENV, _WORKER_AGENT
    import atexit
    import signal
    import torch

    torch.set_num_threads(1)
    _WORKER_ENV = _make_env()

    if policy.startswith("ckpt:"):
        sys.path.insert(0, os.path.join(ROOT, "vendor", "nfly"))
        from agents.train_pinball_circuit_cem import build_agent, set_flat_params
        ck = torch.load(policy.split(":", 1)[1], weights_only=False)
        _WORKER_AGENT = build_agent(ck.get("agent", "circuit"), os.path.join(ROOT, "web/connectome.json"),
                                     ck["readout_dim"])
        _WORKER_AGENT.calibrate(_WORKER_ENV.observation_space)
        set_flat_params(_WORKER_AGENT.decoder, ck["champion"])
    elif policy == "none":
        _WORKER_AGENT = None
    else:
        raise ValueError(f"unsupported policy {policy!r}")

    def _on_terminate(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_terminate)
    atexit.register(_WORKER_ENV.close)


def _policy_action(obs, t, h_box):
    if _WORKER_AGENT is not None:
        import torch
        if t == 0:
            h_box[0] = _WORKER_AGENT.initial_state(1)
        a, h_box[0] = _WORKER_AGENT.act(torch.as_tensor(obs).unsqueeze(0), h_box[0], greedy=True)
        return int(a[0][0]), int(a[0][1])
    return 0, 0


def _run_base_life(env):
    """Run the base policy for one life. Returns (actions, T, drained, drain_x) where actions
    is the list of (l, r) taken at each step 0..len-1, T is the drain step index (len(actions)-1
    style: the step at which info['drained'] became True), or None if it never drained within
    MAX_STEPS (truncated)."""
    obs, info = env.reset(seed=CUR_SEED[0])
    actions = []
    trace = []
    h_box = [None]
    t = 0
    prev_obs = obs
    while t < MAX_STEPS:
        l, r = _policy_action(obs, t, h_box)
        actions.append((l, r))
        prev_obs = obs
        obs, _, term, trunc, info = env.step(np.array([l, r, 0]))
        trace.append((float(obs[0]), float(obs[1])))
        t += 1
        if info["drained"]:
            return actions, t - 1, True, trace
        if term or trunc:
            return actions, None, False, None
    return actions, None, False, None


def _pass_step(trace, T):
    """Last step before the drain at which the ball was still above the flipper line. The engine
    registers a drain only after the ball has rested below the flippers for a long time, so the
    ball is already lost well before T."""
    for t in range(T - 1, -1, -1):
        if trace[t][1] < PASS_Y:
            return t
    return 0


CUR_SEED = [None]


def _replay(env, seed, actions, upto, forced=None):
    """Reset with `seed`, replay `actions[0:upto]` verbatim, then if `forced` is given
    (press_l, press_r, n_steps) apply that action for n_steps, then (0, 0) thereafter, until
    either drain, termination, truncation, or (if `forced` given) SAVE_WINDOW steps past the
    start of the forced press have elapsed. Returns a dict with drained (bool), saved (bool: y
    fell below SAVE_Y_THRESH before draining, only meaningful when forced is given), and
    final_x (x at the step before drain, if drained)."""
    from env_python.pinball_env import OBS_BALL_X, OBS_BALL_Y

    obs, info = env.reset(seed=seed)
    prev_obs = obs
    t = 0
    for (l, r) in actions[:upto]:
        prev_obs = obs
        obs, _, term, trunc, info = env.step(np.array([l, r, 0]))
        t += 1
        if info["drained"]:
            return {"drained": True, "saved": False, "final_x": float(prev_obs[OBS_BALL_X]), "drain_step": t - 1}
        if term or trunc:
            return {"drained": False, "saved": False, "final_x": None, "drain_step": None}

    if forced is None:
        # sanity-check path: continue with recorded actions to the end (used for determinism check)
        for (l, r) in actions[upto:]:
            prev_obs = obs
            obs, _, term, trunc, info = env.step(np.array([l, r, 0]))
            t += 1
            if info["drained"]:
                return {"drained": True, "saved": False, "final_x": float(prev_obs[OBS_BALL_X]), "drain_step": t - 1}
            if term or trunc:
                return {"drained": False, "saved": False, "final_x": None, "drain_step": None}
        return {"drained": False, "saved": False, "final_x": None, "drain_step": None}

    press_l, press_r, n_press = forced
    saved = False
    for i in range(SAVE_WINDOW):
        if i < n_press:
            l, r = press_l, press_r
        else:
            l, r = 0, 0
        prev_obs = obs
        obs, _, term, trunc, info = env.step(np.array([l, r, 0]))
        t += 1
        if obs[OBS_BALL_Y] < SAVE_Y_THRESH:
            saved = True
        if info["drained"]:
            return {"drained": True, "saved": saved, "final_x": float(prev_obs[OBS_BALL_X]), "drain_step": t - 1}
        if saved:
            return {"drained": False, "saved": True, "final_x": None, "drain_step": None}
        if term or trunc:
            return {"drained": False, "saved": saved, "final_x": None, "drain_step": None}
    return {"drained": False, "saved": saved, "final_x": None, "drain_step": None}


def classify_drain(x):
    ax = abs(x)
    if ax < CENTER_GAP_X:
        return "center_gap"
    if ax < OUTLANE_X:
        return "flipper_reachable"
    return "outlane"


def _process_seed(seed):
    global CUR_SEED
    CUR_SEED[0] = seed
    env = _WORKER_ENV

    actions, T, drained, trace = _run_base_life(env)
    result = {"seed": seed, "drained": drained}
    if not drained:
        return result

    T_pass = _pass_step(trace, T)
    drain_x = trace[T_pass][0]
    result["T"] = T
    result["T_pass"] = T_pass
    result["dead_steps"] = T - T_pass
    result["drain_x"] = drain_x
    result["drain_class"] = classify_drain(drain_x)

    # determinism / sanity check: replaying all recorded actions unchanged must drain at
    # exactly step T.
    sanity = _replay(env, seed, actions, upto=T + 1, forced=None)
    result["deterministic"] = bool(sanity["drained"] and sanity["drain_step"] == T)

    saving = []
    for d in OFFSETS:
        start = T_pass - d
        if start < 0 or start >= T:
            continue
        for ptype in PRESS_TYPES:
            pl, pr = PRESS_ACTIONS[ptype]
            outcome = _replay(env, seed, actions, upto=start, forced=(pl, pr, PRESS_DURATION))
            if outcome["saved"]:
                saving.append({"d": d, "press": ptype})

    result["saving_events"] = saving
    result["any_saved"] = len(saving) > 0
    if saving:
        ds = [e["d"] for e in saving]
        # d = steps before T that the press starts (larger d = earlier in time before drain).
        result["earliest_saving_offset_d"] = max(ds)  # earliest-in-time press that still saves
        result["latest_saving_offset_d"] = min(ds)    # latest-in-time press that still saves
        result["window_width"] = len(set(ds))
        result["saved_by_type"] = {pt: any(e["press"] == pt for e in saving) for pt in PRESS_TYPES}
    else:
        result["earliest_saving_offset_d"] = None
        result["latest_saving_offset_d"] = None
        result["window_width"] = 0
        result["saved_by_type"] = {pt: False for pt in PRESS_TYPES}

    return result


def _pool_worker(seed):
    return _process_seed(seed)


def main():
    seeds = list(range(SEED0, SEED0 + N_SEEDS))
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(processes=N_WORKERS, initializer=_worker_init, initargs=(POLICY,)) as pool:
        for res in pool.imap(_pool_worker, seeds):
            print(f"  seed {res['seed']:6d} drained={res.get('drained')} "
                  f"T={res.get('T')} drain_x={res.get('drain_x')} "
                  f"det={res.get('deterministic')} any_saved={res.get('any_saved')}", flush=True)
            results.append(res)
        pool.close()
        pool.join()

    drained_results = [r for r in results if r["drained"]]
    n_drained = len(drained_results)
    n_total = len(results)
    n_det_checked = sum(1 for r in drained_results if "deterministic" in r)
    n_det_ok = sum(1 for r in drained_results if r.get("deterministic"))

    n_any_saved = sum(1 for r in drained_results if r["any_saved"])
    frac_any_saved = n_any_saved / n_drained if n_drained else float("nan")

    frac_by_type = {}
    for pt in PRESS_TYPES:
        n = sum(1 for r in drained_results if r["saved_by_type"][pt])
        frac_by_type[pt] = n / n_drained if n_drained else float("nan")

    window_widths = [r["window_width"] for r in drained_results if r["any_saved"]]
    width_hist = {}
    for w in window_widths:
        width_hist[str(w)] = width_hist.get(w, 0) + 1

    class_counts = {"center_gap": 0, "flipper_reachable": 0, "outlane": 0}
    class_saved = {"center_gap": 0, "flipper_reachable": 0, "outlane": 0}
    for r in drained_results:
        c = r["drain_class"]
        class_counts[c] += 1
        if r["any_saved"]:
            class_saved[c] += 1
    class_frac = {c: (class_saved[c] / class_counts[c] if class_counts[c] else float("nan"))
                  for c in class_counts}

    aggregates = {
        "policy": POLICY,
        "n_seeds": N_SEEDS,
        "seed0": SEED0,
        "n_drained": n_drained,
        "n_total": n_total,
        "n_determinism_checked": n_det_checked,
        "n_determinism_ok": n_det_ok,
        "fraction_any_press_saved": frac_any_saved,
        "fraction_saved_by_press_type": frac_by_type,
        "window_width_histogram": width_hist,
        "drain_class_counts": class_counts,
        "drain_class_saved_counts": class_saved,
        "drain_class_savable_fraction": class_frac,
    }

    with open(OUT_JSON, "w") as f:
        json.dump({"per_seed": results, "aggregates": aggregates}, f, indent=1)

    print("=" * 70)
    print(f"policy={POLICY} n_seeds={N_SEEDS} seed0={SEED0}")
    print(f"drained {n_drained}/{n_total}; determinism ok {n_det_ok}/{n_det_checked}")
    print(f"fraction of drains savable by ANY single press: {frac_any_saved:.3f}")
    print(f"fraction savable per press type: "
          f"{ {k: round(v, 3) for k, v in frac_by_type.items()} }")
    print(f"timing-window width histogram (n offsets that save, among savable drains): {width_hist}")
    print(f"drain class counts: {class_counts}")
    print(f"drain class savable fraction: { {k: (round(v,3) if v==v else v) for k,v in class_frac.items()} }")


if __name__ == "__main__":
    main()
