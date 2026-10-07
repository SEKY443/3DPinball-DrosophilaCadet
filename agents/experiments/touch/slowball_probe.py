"""Does a giant-fiber body respond to SLOW balls in a flipper's strike zone? Records per-step ball state, GF activity and
presses over real lives (seeds 77000-77047), then reports, for the ball inside a flipper strike zone and moving toward the drain
(vy > 0), the fraction of steps that side's GF is above theta_on, binned by ball speed. Also drains through a strike zone.

Adapted from /private/tmp/claude-501/slowball/probe.py with the body parametrised.
Usage: python slowball_probe.py --body sweep|touch [--params best.json] [--lives 48] [--workers 8]
  sweep: best_retina_real_sweep.json on gf_circuit.json (RetinaGiantFiberBody); touch: best_touch_real.json (TouchGiantFiberBody).
"""
import argparse, json, math, multiprocessing as mp, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
PB = os.path.join(HERE, "..", "pathway_body")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path[:0] = [ROOT, PB, HERE]
import numpy as np

BODY = PARAMS = None

def _init(body, params):
    global BODY, PARAMS
    BODY, PARAMS = body, params
    import gf_body as G
    G.init_worker()

def _life(seed):
    import gf_body as G, gf_retina_body as R, gf_touch_body as T
    from env_python.pinball_env import OBS_BALL_X, OBS_BALL_Y, OBS_BALL_VX, OBS_BALL_VY
    body = (T.TouchGiantFiberBody if BODY == "touch" else R.RetinaGiantFiberBody)(**PARAMS)
    env = G._ENV
    obs, _ = env.reset(seed=seed); body.reset()
    rows = []; done = False; drained = False
    while not done:
        a = body.act(obs)
        rows.append((float(obs[OBS_BALL_X]), float(obs[OBS_BALL_Y]), float(obs[OBS_BALL_VX]), float(obs[OBS_BALL_VY]),
                     body.last_a[0], body.last_a[1], int(a[0]), int(a[1])))
        obs, r, term, trunc, info = env.step(a)
        drained = bool(info.get("drained")); done = term or trunc or drained
    return dict(seed=seed, rows=rows, drained=drained, theta_on=body.theta_on)

def zone(x, y):
    # strike zone of each flipper (mirrored table: left flipper = action 0 at +x): x between tip and pivot side, y 10.6-13.3
    if 10.6 <= y <= 13.3:
        if 0.6 <= x <= 2.6: return 0
        if -2.6 <= x <= -0.6: return 1
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", choices=["sweep", "touch"], required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--lives", type=int, default=48)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pf = args.params or (os.path.join(PB, "best_retina_real_sweep.json") if args.body == "sweep" else os.path.join(HERE, "best_touch_real.json"))
    params = json.load(open(pf))["params"]
    if args.body == "sweep":
        params["circuit_json"] = os.path.join(PB, "gf_circuit.json")
    seeds = list(range(77000, 77000 + args.lives))
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.body, params)) as pool:
        lives = pool.map(_life, seeds, chunksize=1)
    th = lives[0]["theta_on"]
    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 999)]
    stat = {b: [0, 0] for b in bins}   # [steps in zone moving down, steps with that side's GF above theta_on]
    for L in lives:
        for x, y, vx, vy, aL, aR, pL, pR in L["rows"]:
            z = zone(x, y)
            if z is None or vy <= 0: continue
            sp = math.hypot(vx, vy)
            for b in bins:
                if b[0] <= sp < b[1]:
                    stat[b][0] += 1; stat[b][1] += int((aL if z == 0 else aR) > th)
    print(f"body {args.body}: lives {len(lives)}, theta_on {th:.3f}, params {pf}")
    print("ball in a flipper strike zone, moving toward the drain: fraction of steps that side's giant fiber is above threshold")
    for b in bins:
        n, k = stat[b]; print(f"  speed {b[0]:>3}-{b[1] if b[1] < 999 else 'inf':>3}: {k:5d}/{n:5d} = {100.0 * k / max(1, n):5.1f}%")
    slow = fast = slow_unanswered = fast_unanswered = 0
    for L in lives:
        if not L["drained"]: continue
        tail = [r for r in L["rows"][-25:] if zone(r[0], r[1]) is not None]
        if not tail: continue
        sp = np.median([math.hypot(r[2], r[3]) for r in tail])
        answered = any((r[6] if zone(r[0], r[1]) == 0 else r[7]) for r in tail)
        if sp < 5: slow += 1; slow_unanswered += int(not answered)
        else: fast += 1; fast_unanswered += int(not answered)
    print(f"drains passing through a strike zone: slow (<5) {slow}, of which no press on that side {slow_unanswered}; "
          f"fast {fast}, of which no press {fast_unanswered}")
