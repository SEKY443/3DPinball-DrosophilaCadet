# Step 3 report: which table objects the ball hits, and what they are worth

Completed 2026-09-29 22:56 CST. All work is in `/Users/seky/Developer/Git/3DPinball-step3`. The production repo was not modified, nothing was committed, and no training run was started.

**Headline:** The engine now reports every object the ball hits and the points each hit earns, and the numbers check out. The gen-280 champion does not aim, though. It presses both flippers on a fixed rhythm that ignores the ball.

## What changed
- **Engine hook** (`src_cpp/patches/0007-score-object-attribution-hook.patch`, 2 lines of logic):
  - `control::handler()` is the single place where every table object's game logic runs, and `TPinballTable::AddScore()` is where points are added. I bracket each `handler()` call and log each `AddScore()`.
  - When a ball-contact handler (`ControlCollision` or `ControlBallCaptured`) is running, every point added inside it is credited to that object. With nested handlers, the outermost ball-contact handler gets the credit, so mission and jackpot awards go to the hit that triggered them.
  - The logic lives in `src_cpp/state_export.{h,cpp}`: `BeginControl`, `EndControl`, `OnScoreAdded` and `DumpTableMap`.
- **IPC frame** (`src_cpp/ipc_protocol.h`): the frame grows from 53 to 94 bytes and every old field keeps its offset.
  - New fields: `hit_count`, `hit_ids[4]`, `hit_points[4]`, `unattributed_points`, and `hit_base_points[4]` (the same points before the score multiplier).
  - Invariant: `score_delta == sum(hit_points) + unattributed_points`.
- **Python side:** `env_python/ipc_client.py` parses the new fields (`State.hits`, `base_hits`). `env_python/pinball_env.py` adds `info["hit_objects"]` as (id, points, x, y), plus `info["hit_base_points"]` and `info["unattributed_points"]`.
- **Table map:**
  - `agents/table_map.json` lists all 88 scoring objects: stable id, name, class, group, `center` and `aabb` in ball coordinates, the score array, `base_points`, a `value_note` on what game state changes the payout, and `control.cpp:line` citations.
  - Regenerate it with `scripts/gen_table_map.py`, which merges an engine dump (`PINBALL_DUMP_TABLE_MAP`) with a parse of `control.cpp`.
  - The line citations point to the patched `control.cpp`, which is 3 lines longer than upstream after the include.
- **Tests:** `scripts/smoke_ipc.py` was already broken before this change because its frame format was out of date. I fixed it and it now also checks the hit fields.
- **Diagnostics:** `agents/experiments/step3/hit_diagnostic.py` (per-object hits and points) and `aim_jitter.py` (whether hits are aimed).
- **Trainer** (`agents/train_pinball_circuit_cem.py`): new flags `--active-upper-weight W` (default 0) and `--table-map`.

**Build:** `./scripts/apply_patches.sh`, then `cmake -S vendor/SpaceCadetPinball -B build-native …` (same flags as `scripts/build_native.sh`), then `cmake --build build-native -j 2`.

## Verification
- **Wire format:** the smoke test passes (20/20 round trips). All 7 patches apply cleanly to a fresh clone of the submodule.
- **Browser demo:** `node web/test/verify_policy.mjs` reports all checks passed. `bridge.js` reads fields by offset and the old offsets are unchanged, so it is unaffected. The WASM build was **not** rebuilt.
- **Gen-280 champion, 24 lives, seeds 7000–7023:**
  - 0 invalid ids and 0 steps violating the score-sum invariant; 0% of points unattributed.
  - 1561 hits. Distance from the ball to the object's bounding box at hit time: median 0.000, p90 0.30. Distance to the object's centre: median 0.57.
- **No-press baseline, 24 lives:** same result (0 bad ids, 0 violations).
- **`W=0` changes nothing:** I ran the old code with the old binary and the new code with the new binary on seeds 7100–7103. Fitness, score, ticks and presses are identical to the last digit.

## Diagnostic: gen-280 champion (mean 130k points per life, 8.6 flipper contacts per life)
"Upper" means the object sits above y=6 on the table. "Active" means between a flipper contact and the ball falling back.

| Zone / phase | Share of points |
|---|---|
| Upper playfield, active | 34.6% |
| Upper playfield, launch | 9.6% |
| Upper playfield, passive | 9.2% |
| Lower lanes (y≈8), active | 25.0% |
| Lower lanes, passive or launch | 15.5% |
| Drain | 6.1% |

- **Top objects:**
  - Return lanes `a_roll6` and `a_roll7`: 300k and 250k.
  - Spinner `a_flag2`: 287k; 172 of its 212 hits are active.
  - Outlanes `a_roll4` and `a_roll8`: 220k each. These are the ball on its way to the drain.
  - Bonus lane `a_roll5`: 190k.
- **Payouts depend on game state:**
  - Lit return lanes pay 25k instead of 5k. `a_roll9` pays 0 itself but lights them, so its value only shows up later.
  - Points are multiplied by the table multiplier (x2–x10); for example an outlane hit paid 60k and a return lane 125k.
  - Completing a target bank or a mission pays an extra award.
  - This is why the map records the base value plus a note, and why `hit_base_points` exists.
- **Active shots are mostly empty:** 59% of active windows (n=206) hit no upper object at all. The mean shot is worth 3.4k base points.
- **The champion does not aim:**
  - It presses **both flippers on a fixed 3-decision-step rhythm**. Left and right are identical on 99% of steps, and 98% of gaps between presses are exactly 3 steps.
  - Jitter test (8 lives, n=76 presses, shifts of ±1 and ±2 steps): the real shot beat the median of the shifted shots in only 1/76 cases (1.3%, 95% CI 0.2–7.1%). 71 were ties, 4 were worse, and 6 of 304 shifted runs produced no contact.
  - This test is weak for this policy, because shifting one press barely changes anything when the neighbouring presses keep the flipper moving. The rhythm itself is the stronger evidence that hits of valuable objects are accidental.

## Proposal (implemented behind a flag)
- **Search fitness:** `log1p(score + W·active_upper_base)`, where `active_upper_base` is the multiplier-free points from upper objects (by the object's table_map position, y<6) hit between a real flipper **contact** and the ball falling back.
  - Pressing without contact earns nothing, so spamming presses can't farm it.
  - Promotion (`_paired_validation`) and the check that decides whether to run it both use the plain `log1p(score)`.
- **Strength of W=1:** at gen 280 the bonus would be 18.3% of the shaped total (10.1% at W=0.5), well under the 50% concern, so W=1 is reasonable for a first run.
- **Smoke test:** 3 generations with CMA, 2 workers, population 6. It runs and validation fires. Log: `agents/experiments/step3/smoke_active_upper.log`.

## Risks and follow-ups
- **The reward may not be enough on its own.** A metronome can't aim, so a real run will probably also need press cost or a reason to press in response to the ball state.
- **Warm start:** `--resume` needs a checkpoint that contains `mean` and `sigma`, like `start_gen215_centered.pt`. `best_heldout_gen280.pt` only holds the champion weights.
- **Ignored hits:** hits beyond 4 per tick are folded into `unattributed_points`; this has not been observed.
- **Browser build:** the WASM build was not rebuilt. After a rebuild the frame is 94 bytes and `bridge.js` still works, because it reads fields by offset.

## Follow-up: why the champion behaves like a metronome
Everything below uses dev seeds 7000–7023 (24 lives per policy) and at most 2 workers.

Scripts:
- `agents/experiments/step3/policy_compare.py` runs trained and hand-written policies side by side.
- `agents/experiments/step3/circuit_probe.py` probes the circuit offline and on recorded lives.

### The rhythm comes from the flipper-state inputs, not from the circuit or the ball
- **Offline, the circuit alone does not oscillate.** Fed a zero observation or a frozen real observation, it either never presses or holds the right flipper continuously.
- **The rhythm needs the flipper-state feedback.** Observation channels 12–13 report whether each flipper is up, which is effectively the previous action. With that feedback connected, period 3 appears at once, with left and right identical, even when every ball input is zero.
- **Ablation on recorded lives (open loop):**
  - Replacing all 12 ball channels with their per-life mean leaves 98.4% of actions unchanged.
  - Setting the flipper channels to 0 leaves only 15% unchanged.
- **Conclusion:** the champion is a press/release oscillator that runs off its own flipper-state inputs and is essentially blind to the ball.

### The objective does not reward reacting: a blind rhythm is near-optimal
Fitness here is mean log1p(score per life) with no press cost. SE is about 0.2 for every row.

| Policy | log1p | Score/life | Steps | Contacts | Presses/life |
|---|---|---|---|---|---|
| gen-280 circuit | 11.34 | 130k | 1314 | 8.6 | 862 |
| MLP ceiling | 11.36 | 118k | 1472 | 12.5 | 381 |
| Open-loop metronome, period 3, phase 0 / 1 / 2 | 11.05 / 10.99 / 11.66 | 86k / 83k / 163k | | | ~800 |
| Metronome, period 4 / 5 | 11.17 / 11.21 | | | | 618 / 518 |
| **Correct-side pulse reflex, y > 11.5** | **11.40** | 147k | 1317 | 7.7 | **13** |
| Correct-side pulse reflex, y > 9.5 | 11.31 | | | | 52 |
| Never press | 9.68 | 22k | 445 | 0 | 0 |

- **Metronome phase is a lottery.** Phase 2's advantage on these seeds is luck: on seeds 7024+ it drops to 10.85.
- **A position gate removes most presses for free.** On seeds 7024+, pressing only while the ball is below y=8 scores 10.88 with 187 presses per life, against 10.85 and 795 presses for the plain metronome.
- **A one-threshold reflex ties the champion with 1–2% of its presses.** At press cost 0.00088 that is a difference of about 0.75 in log units. CMA has not found this policy.

### Other hypotheses
- **Ball state near the flippers is poorly encoded (partly supported).** Leave-one-life-out R² for a linear probe from readout activity, restricted to y > 10: x 0.76, y −0.20, vy 0.20. Over all steps, y reaches 0.87. So "the ball is low" is readable, but fine timing information near the flipper is not.
- **Left and right are not tied by the architecture.** Each flipper has its own head row (cosine similarity 0.49 between them). But the two logits correlate at 0.995, because both are driven by the same flipper-feedback feature.
- **MLP:** it is not a metronome (left and right equal on 81% of steps, 3-step gaps 61%), yet it ties the circuit on fitness. The limit is the objective and the search, not the connectome.

### Flipper side mapping is mirrored in the trainer
- **Action 0 (left flipper) is at positive table x.** The table's x axis is mirrored relative to the screen.
- **Evidence:**
  - `a_flip1`, the left flipper, has its centre at x=+1.78 in the table map.
  - The plunger lane, on the right side of the screen, is at x=−7.2.
  - Placing the ball at x=±2 and pressing each flipper confirms it.
  - A reflex with the mirrored mapping gets 0.4 contacts per life; the correct mapping gets 24.5.
- **Affected code:**
  - `_heuristic_action` in `agents/train_pinball_circuit_cem.py` (lines ~544–552) uses `flip_left = x <= 0`. `heuristic_seed` therefore teaches the decoder to press the wrong flipper.
  - The same assumption ("left flipper at x=−2.489") appears in `agents/experiments/reward_redesign_2026-09-25.py` and the round3 trainer.
- **Not fixed here, for review.** The one-line fix is to swap the two conditions.

## Follow-up 2: seeding the circuit from the reflex (the "cheap check")
Seeding lives were 7000–7023 and scoring lives 7024–7071 (48, held out), with at most 2 workers.

### Trainer changes (`agents/train_pinball_circuit_cem.py`)
- **Side fix:** `_heuristic_action` now uses `x > 0 → left`.
- **New classes:**
  - `ReflexLabeler(threshold, hold, cooldown)`: the correct-side reflex.
  - `FlipperObsFilter(raw|zero|delay1)`: controls what the agent sees of observation channels 12–13 (flipper state). It applies both at seeding and in `_evaluate`.
- **New CLI flags:** `--seed-policy {escape,reflex}`, `--seed-reflex-y`, `--seed-reflex-hold`, `--seed-episodes`, `--flipper-obs`.
- **`heuristic_seed` bug fixes:**
  - It used to store already-projected features, so imitation only trained the 8→3 head on a random fixed 8-d projection. It now trains the projection (`proj`) and the head together.
  - It gains a head-bias rate calibration, which the reflex policy uses. Without it, the class-weighted loss over-predicts rare presses.
  - These changes alter what `escape` seeding produces for any new, non-resumed run. Runs resumed from a checkpoint are unaffected.
- **Side-convention comments:** the comment in `env_python/pinball_env.py` now states the mirrored convention.
- **Archived scripts left as historical records:** `agents/experiments/reward_redesign_2026-09-25.py` and `agents/experiments/round3_validation_2026-09-25/train_pinball_circuit_cem_round3.py` still use the mirrored mapping.

### Reflex tuning on seeds 7000–7023
| Reflex variant | log1p | Presses/life |
|---|---|---|
| Pulse, y > 11.5 | 11.40 | 13 |
| Hold 3 steps, y > 10.5 | 11.37 | 38 |
| Hold 2 steps, y > 11.5 | 11.18 | |
| Pulse, y > 12.5 | 10.42 | |

### Held-out results, seeds 7024–7071 (48 lives)
Agreement, recall and precision compare the seeded agent against a shadow reflex run on the same observations. Recall is the fraction of reflex presses the agent also makes; precision is the fraction of the agent's presses the reflex would make.

| Policy | log1p ± se | Presses/life | P(L)/P(R) | Agreement | Recall | Precision |
|---|---|---|---|---|---|---|
| Reflex, pulse y > 11.5 (reference) | **11.45 ± 0.12** | 14 | | | | |
| Reflex, hold 3, y > 10.5 (reference) | 11.00 ± 0.13 | 27 | | | | |
| Behaviour cloning, pulse, flipper obs raw | 10.11 | 20 | .018/.020 | .939 | .13 | .11 |
| Behaviour cloning, pulse, flipper obs zeroed | 10.11 | 2 | .004/.004 | .965 | .06 | .21 |
| Behaviour cloning, pulse, flipper obs delayed 1 step | 10.28 | 9 | .006/.015 | .956 | .13 | .21 |
| Behaviour cloning, hold 3, raw | 9.94 | 2 | .056/.030 | .869 | .16 | .04 |
| **DAgger (3 rounds), pulse, delayed, readout_dim 8** | **10.52 ± 0.11** | 10 | .012/.009 | .963 | .19 | .23 |
| DAgger (3 rounds), pulse, delayed, readout_dim 32 | 10.46 ± 0.11 | 10 | .014/.009 | .963 | .20 | .26 |

For comparison, the first behaviour-cloning attempt, which only trained the head, scored 9.75–10.18 (`seed_eval_v1_headonly.log`).

### What this shows
- **The circuit cannot reproduce the reflex's timing closely enough.**
  - Offline, a linear readout of all 80 readout cells separates the reflex's press steps well (ROC-AUC 0.97–0.98). But at the reflex's own press rate, precision is only 0.27–0.52 (`reflex_separability.py`).
  - Closed loop, the seeded agent catches only about 20% of the reflex's presses, and the ball drains.
  - Widening the bottleneck from 8 to 32 does not help. Holding the flipper for 3 steps instead of pulsing imitates worse (recall 0.16, precision 0.04).
- **The best seed is about 0.9 log units below the reflex it imitates.** On the plain objective it is also below the gen-280 champion.
- **Under press cost 0.00088 it is roughly level with the champion.** The champion's 862 presses cost about 0.76, putting it near 10.6; the seed's 10 presses cost about 0.01, putting it at about 10.5. The champion figure is estimated from dev seeds.
- **The seed is fragile.** In a 2-generation CMA smoke test at `--cma-sigma0 0.2`, perturbed candidates held the flippers up (P(L) up to 0.68). Use `--cma-sigma0 0.05` or lower.

### Start checkpoint and command (only if a circuit run is still wanted)
The start checkpoint is `agents/experiments/step3/start_reflex_seed.pt`. It is the DAgger seed with delayed flipper obs and readout_dim 8, stored with `mean`, `sigma`, `champion`, `champion_fitness=-1e18`, `generation=0`, `readout_dim=8` and `flipper_obs=delay1`. Resuming from it was smoke-tested.

```
python agents/train_pinball_circuit_cem.py --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \
  --connectome agents/experiments/big_circuit/connectome.json --readout-dim 8 --flipper-obs delay1 \
  --optimizer cma --cma-sigma0 0.05 --workers 8 --population 16 --fresh-fraction 0 \
  --ipop-patience 80 --ipop-growth 2.0 --ipop-max-population 48 --generations 300 \
  --courses-per-candidate 8 --validation-episodes 32 --promotion-t 2.0 --max-steps 3000 \
  --press-cost 0.00088 --resume agents/experiments/step3/start_reflex_seed.pt \
  --out agents/experiments/reflex_seed/reflex_seed.pt --save-every 5
```

## Follow-up 3: looming input, lookahead oracle, nudge, missions, drain anatomy (2026-09-30)

### Looming input (option b): fails the 11.2 bar
All numbers are held-out, seeds 7024–7071, 48 lives.

| Variant | log1p | Presses/life | Recall | Precision |
|---|---|---|---|---|
| DAgger, flipper channels 12–13 carry per-flipper looming (`loom12`, tau0 = 4 steps) | **10.59** | 5.5 | 0.09 | 0.11 |
| DAgger, looming routed through the LPLC2 size channels (`loom`, as originally framed) | 10.39 | | | |
| Pulse reflex (reference) | 11.45 | | | |

Recall is the share of the reflex's presses the seeded agent also makes; precision is the share of the agent's presses the reflex would make.

- **The visual pathway barely reaches the readout** (`loom_probe*.log`).
  - One-step influence on the readout from a +0.5 input: LC4/LPLC2 channels 4–11 give 1e-5 to 2e-3; mechanosensory channels 0–3 and 12–14 give about 2.
  - Driving only the visual channels gives press-classification ROC-AUC of 0.50–0.77.
  - Graph distance is not the cause: visual input cells are 2 hops from the readout and reach all 64 readout cells. Under real mechanosensory drive the shared intermediate cells saturate and mask the visual signal.
- **Timing lag is not the problem.** The circuit's press score peaks at lag 0 relative to the reflex press. But it is already high 1–3 steps earlier (AUC 0.94–0.97 at lags −3 to −1). The circuit cannot pin down the exact step, so it is a sharpness and base-rate problem.

### Greedy lookahead oracle (8 lives, K = 5 press timings per approach, `lookahead_oracle.py`)
- Base reflex scores 11.74; the oracle scores 11.63, a paired difference of **−0.10 (SE 0.33)**.
- Choosing timing one approach ahead, to maximise survival and then points, does not raise the one-life score. This gives no evidence of timing-only headroom above the reflex.

### A. Nudge (separate copy `/Users/seky/Developer/Git/3DPinball-nudge`)
**How the engine nudges** (`nudge.cpp`, `pb.cpp:324-346`, `pb.cpp:572-635`):
- There are three table bumps, and each has a held-key semantic:
  - LeftTableBump calls `nudge_right`, adding (+1.0, +0.5) units/s to the ball's velocity.
  - RightTableBump calls `nudge_left`, adding (−1.0, +0.5).
  - BottomTableBump calls `nudge_up`, adding (0, +0.5).
- The engine reverses the impulse after 0.4 s or on key release, whichever comes first. It only affects balls not currently in contact with a component.
- **Tilt:** while any nudge is held, `nudge_count` grows by 4·dt; otherwise it decays by dt. Above 0.5 a warning shows; above 1.0 the table tilts.
- **Measured** (one decision step ≈ 0.033 s):
  - A 2-step nudge moves the ball 0.066 units sideways. The maximum is about 0.2 units at 6 steps; holding for 8 steps tilts.
  - One-step taps every 2–3 steps tilt after 10–14 taps. Taps every 5 steps or more never tilt.

**IPC extension, implemented in the nudge copy only:**
- New `ActionNudgeFrame` with magic "ACTN" (20 bytes = ActionFrame + held nudge code + padding). The server accepts both ACT! and ACTN.
- `Action(nudge=0)` still sends the byte-identical 16-byte ACT! frame, so existing clients are unaffected.
- `PinballEnv.step()` reads an optional 4th action element as the nudge code.
- Built with `cmake --build build-native -j 2` in the nudge copy.

**Drain anatomy under the pulse reflex** (48 lives, seeds 7024–7071): 30 outlane, 11 center, 3 side, 4 alive at the step cap.
- 26 of 30 outlane drains and 9 of 11 center drains had no flipper contact in the final approach.
- The env's `drained` flag fires a median **84 decision steps** (about 2.8 s) after the ball actually hits the drain, because it waits for the ball to be re-fed to the plunger.

**Nudge oracle.** For each outlane death I replayed the life identically, then held one nudge at k ∈ {30, 20, 12, 6, 3} steps before the outlane hit, for each of the 3 codes and 3 hold lengths (45 branches). A save means no drain until the original drained-flag step + 200 steps, with no tilt.
- **25 of 30 deaths had at least one saving branch** (83%, Wilson 95% CI 66–93%). None of the branches tilted.
- **Saves fall steeply with lead time:**

| Nudge lead k (steps before outlane hit) | Share of branches that save | Deaths with a save at k or less |
|---|---|---|
| 30 | 37% | 25/30 |
| 20 | 20% | 13/30 |
| 12 | 8% | 8/30 |
| 6 | 2.6% | 3/30 |
| 3 | 0.7% | 2/30 |

- **All three directions save at similar rates:** +x 14%, −x 17%, and the bottom bump (no sideways component) 10%.
- **Reading:** early saves come from chaotic sensitivity to any small perturbation, not from steering the ball away from the outlane. A reactive, near-the-outlane nudge (≤6 steps) saves only about 10% of outlane deaths.
- **No sham control was run,** so I have not measured how often a random early nudge would turn a surviving ball into a drain.

### B. Missions (`control.cpp`)
**How a mission starts:**
1. The launch passes `s_onewy10`, which puts the table into "select mission" (`WaitingDeploymentController`, `control.cpp` about line 4586).
2. A mission-spot target selects a mission by rank (`SelectMissionController`). At the starting rank (1 light in `middle_circle`):
   - `a_targ13` selects Launch Training.
   - `a_targ14` selects Re-entry Training.
   - `a_targ15` selects Practice.
   - Completing all three targets selects Science.
3. A ramp hit then starts the mission while its light `lite56` is on and the fuel gauge (`fuel_bargraph`) is above 0. This pays `mission_select_scores`: 10k at rank 1.
4. The fuel countdown is the time limit. The fuel rollovers `a_roll179`–`a_roll184` and targets `a_targ10`–`12` refuel it.

**Rank-1 missions, each paying 500,000 (`SpecialAddScore(500000, true)`):**
| Mission | Requirement after start | Source |
|---|---|---|
| Practice | 8 hits on bumpers 1–4 | line 3715 |
| Launch Training | 3 more ramp hits | line 3377 |
| Re-entry Training | 3 hits on the top lanes `a_roll1`–`3` | line 3827 |

Higher-rank missions pay 0.75M–5M and need longer sequences.

**Current reach** (gen-280 champion log, 24 lives):
- A selection target was hit in 17 of 24 lives, but a real mission start (ramp paying 10k instead of the 5k base) happened in only 3 of 24. No mission completed.
- Per active shot window (n = 206): ramp 4.4%, each select target about 3–5%, any bumper 1–4 about 56%, top lanes about 18%.
- Unaimed, Practice needs about 1/0.05 + 1/0.044 + about 4 shots, roughly 45 shots. A life has about 8 shots, so this is roughly 5 lives' worth.
- Even if aiming doubled the select and ramp rates, it would still take about 3 lives. **Under the one-ball-life objective, missions are out of reach.**

**Cheap high-value targets:**
- **Multiplier bank** `a_targ7`–`9`: completing it raises every later point ×2 to ×10. The observed 125k inlane hit was 25k × 5.
- **Space-warp rollover** `a_roll9`: lights the return lanes (5k → 25k).
- **`a_kout3` (black hole, 20k):** hit 4 times in 24 lives.
- **`a_kout1` (gravity well, 50k at x=0, y=6):** never hit.
These are value targets for an aiming policy, but no measurement yet shows any policy can aim.

### Recommendation
**Sensory encoding first, then the full-game objective.** Reasoning:
1. **Sensory encoding has the only measured, certain headroom:** the reflex (a 3-parameter hand rule) scores 11.45 and the best seeded circuit 10.59, a gap of **0.86 log units**. The measurements above point at the input side, not the readout: the visual pathway is masked, and the mechanosensory code can't localise the press step.
2. **Nudge:** outlanes are 68% of reflex drains, but reactive nudges save about 10% of them (3 of 30). The early 37% save rate is chaos: it is direction-agnostic, and a policy can't anticipate which perturbation helps 1 s ahead. Expected value is at most about 0.1 × 0.68 of drains, a small increase in life length, for an action-space change plus tilt risk. **Not worth it now.**
3. **Missions:** +500k is worth about +1.5–2 log units per life, but completing one takes about 45 unaided shots, roughly 5 lives. That is unreachable under the one-ball-life objective without strong aim, which we don't have. Missions become relevant only if the objective spans a full game (3 balls, with mission state kept across balls), **and** after aim exists.
4. **Value-weighted aiming:** picking the best of 5 timings per contact doubles upper-playfield value per shot. But the greedy lookahead gives −0.10 ± 0.33 on total score, and neither the circuit nor the reflex aims. It is stage 3, after encoding.
5. **Cheap engineering fixes worth doing regardless:**
   - Detect drains at the TDrain hit (id 80) instead of the re-feed. That removes about 84 useless steps per life, and the trainer currently counts the post-drain awards as part of the life.
   - The side-mapping fix from Follow-up 2.

## Follow-up 4: drain detection at the TDrain hit, popcode seed profile and aim (nudge sandbox, 2026-09-30)
All code for this follow-up is in `/Users/seky/Developer/Git/3DPinball-nudge`. Nothing in step3 was edited except this report.
- `env_python/pinball_env.py` gains `PinballEnv(drain_detect="refeed"|"tdrain")`. The default, `refeed`, keeps the current behaviour. `tdrain` ends the life on the step the engine reports a hit on the drain object (TDrain, id 80).
- `agents/experiments/step3/policies.py`: a common wrapper for the reflex, the plain circuit and the popcode circuit.
- `drain_compare.py` and the generalised `aim_jitter.py`.
- `encoding.py` and `start_popcode_seed.pt`, copied unchanged from step3.

**1. Drain at the TDrain hit** (seeds 7024–7047, 24 lives per policy, one run per life scored both ways):

| Policy | Steps: refeed → tdrain | Dead steps removed | log1p (both modes) | Press-cost fitness: refeed → tdrain |
|---|---|---|---|---|
| Reflex | 1376 → 1307 | 70 | 11.552 | 11.540 → 11.540 |
| gen-280 champion | 1229 → 1156 | 73 | 10.834 | 10.126 → 10.169 |
| Popcode seed | 1186 → 1102 | 84 | 11.426 | 11.417 → 11.417 |

- **Post-drain points are 0% for all three policies.** The drain's own award is paid on the TDrain hit step itself, so it counts in both definitions.
- **The only fitness change** is the gen-280 champion's metronome presses during the dead steps, which cost it 0.043 under the old definition.
- **Ranking is unchanged** in every column: reflex > popcode > gen-280.
- **No false drains:** no life ended at a TDrain hit while play continued.
- **The env option was verified** on 3 lives: `tdrain` mode ends on the same step with the same score as the offline truncation.
- **Verdict: a pure speedup of about 6%** (70–84 of about 1200 steps per life). Caveat: in multiball, a drain of one of two balls would also end the life; this was not seen here.

**2. Popcode seed: hit profile** (same 24 lives, lives cut at the TDrain hit). Shares are of multiplier-free (base) points.

| Policy | Upper share | Upper & active share | Targets 7–9 per life | `a_roll9` per life | Ramp per life | Outlane hits per life |
|---|---|---|---|---|---|---|
| Reflex | 63.1% | 48.8% | 2.58 | 0.96 | 0.29 | 0.71 |
| gen-280 champion | 64.0% | 35.5% | 2.17 | 0.75 | 0.42 | 0.29 |
| Popcode seed | 57.5% | 39.1% | 2.71 | 1.08 | 0.46 | 0.75 |

**Aim-jitter test on the popcode seed** (16 lives, seeds 7024–7039; 105 presses that made contact; shifts of ±1 and ±2 steps):
- **No aim:**
  - The real shot beat the median of the shifted shots in 46 cases, lost in 46, and tied in 13. That is 50% of the 92 decided cases (95% CI 40–60%).
  - The mean real shot value (4.9k base points) is above the mean shifted median (2.8k), with a paired difference of +2.1k ± 0.9k. But the sign test says chance, so the mean gap comes from a few large shots.
- **Timing sensitivity:** 38 of 420 shifted runs (9%) lost the contact entirely.
- **Value headroom:** picking the best of the 5 timings per contact gives **12.4k** base points per shot, against 4.9k for the real shot (2.5×). The popcode seed makes about 6.6 contacts per life, so that is up to roughly +50k base points per life if it could be realised.
- **Counterweight:** the greedy lookahead oracle on the reflex, which picks timing one approach ahead by survival and then points, gave −0.10 ± 0.33 on the whole-life score. Per-shot value picked greedily did not turn into life score there.

**Opinion on running the `--active-upper-weight` stage:**
- **Worth one moderate run** (W = 0.5–1) from the popcode champion, because:
  - the policy is no longer a metronome, so its shots are real timing decisions;
  - per-shot value headroom is large (2.5×);
  - the flag is already built so that promotion is judged only on plain `log1p(score)`, which makes a failed run cheap.
- **Expect little.** There is currently zero aim, and the greedy oracle suggests value picked per shot does not translate into life score.
- **Check first** that `--active-upper-weight` works together with `--encoding popcode` and `--flipper-obs delay1` in the train_enc path, and use `tdrain` episodes for the speedup.
