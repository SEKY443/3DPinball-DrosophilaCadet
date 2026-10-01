# Breakout practice project

A small side project applying the same method as the pinball project (`../agents/`,
`../env_python/`) - a fixed, measured MaleCNS connectome feeding a small trained readout,
evolved with CEM - to a simpler game, to test whether the approach generalizes and to get a
faster, cleaner result than pinball's multi-day saga allowed.

## Files

- `env.py` - pure Python/numpy Breakout physics (ball, paddle, 8x4 brick grid), no native engine
  or IPC needed since it's simple enough to simulate in-process. Same observation-channel
  philosophy as pinball: raw kinematics (`ball_x/y/vx/vy`, `paddle_x`) plus a looming pair
  (`vel`/`size`) for the ball closing in on the paddle's y-plane, mirroring
  `env_python/pinball_env.py`'s ball2 loom channels almost unchanged.
- `build_connectome.py` - same MaleCNS v1.0 data and selection method as
  `agents/build_pinball_connectome.py` (LC4/LPLC2 for the looming pair, `vnc_sensory` for the rest,
  `vnc_motor` output), adapted for this game's 7-channel observation. Produces `connectome.json`
  (76 nodes, 419 edges - the same scale as the pinball project's original circuit).
- `train.py` - CEM neuroevolution, reusing `agents.fixed_circuit_agent.FixedCircuitAgent` and
  `agents.train_pinball_circuit_cem.set_flat_params` directly. Much simpler than the pinball
  trainer: no `ResilientPool`, no native-engine crash class of bugs, no log-bloat problem - a
  plain `multiprocessing.Pool` of pure-Python workers is enough, and each generation takes
  ~1-2 seconds instead of pinball's ~15-30s (no native binary IPC round-trips).
- `eval.py` - real-episode eval, mirrors `agents/eval_pinball_circuit_cem.py`.

## Result (2026-09-19, first run)

500 generations, population=24, workers=4, ~8 minutes total wall time. champion_fitness climbed
56 -> 82.25 (by gen14) -> 368.25 (by gen500), no plateau observed within this run.

Real 10-episode eval: **mean 25.9/32 bricks broken (81%)**, mean 18.3 paddle_hits/episode, one
episode (#5) fully cleared the board (0 bricks left). A genuinely competent, real-connectome
learned policy from a single ~8-minute run - no flipper-focus-style fitness isolation, curriculum,
or novelty search needed, unlike pinball. Likely because Breakout's score is inherently
skill-gated (you cannot score without the paddle keeping the ball alive), unlike pinball's score
being dominated by passive bumper bounces the policy didn't control.

## Next steps (not yet tried)

- Train longer / larger population to see if it can approach 100% clearance.
- Try `--optimizer cma` (not implemented in this train.py yet - would need porting
  `agents/train_pinball_circuit_cem.py`'s `_run_cma` the way that file does, if the CEM path
  plateaus in a future run).
- A JS/browser port (`web/`-style) was not attempted - pinball's browser deployment work was not
  duplicated here since this is a practice/side project, not the main deliverable.
