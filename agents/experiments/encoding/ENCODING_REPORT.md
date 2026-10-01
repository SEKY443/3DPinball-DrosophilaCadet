# Encoding study: why the circuit can't time the ball, and a fix

Completed 2026-09-30 08:50 CST. All work is in `/Users/seky/Developer/Git/3DPinball-step3/agents/experiments/encoding/`. No shared file was edited (trainer, `env_python/*`, `src_cpp/*` and the binary are untouched), nothing was committed, and no training run was launched. At most 2 worker processes were used.

**Headline:** Ball state near the flippers is not lost. The circuit gets each ball variable as one scalar, broadcast identically to 12 cells, so its readout is low-dimensional and nearly linear in the observation. A linear readout of such a code cannot form the sharp conjunction the reflex needs (y > 11.5 AND falling AND ball on that side). A range-fractionated population code on the same 48 mechanosensory input cells fixes this:
- **Held-out imitation score:** 11.41 ± 0.12, against 10.52 for the current broadcast encoding in the same pipeline.
- **Against the reflex:** paired difference −0.04 ± 0.11 relative to the reflex's 11.45, so the imitation is statistically as good as the reflex.

## 1. Diagnosis (broadcast encoding, delay1 flipper obs, 31,525 reflex steps on lives 7000–7023)

All numbers come from `probe.py` (`probe_v2.log`). The flipper zone is y > 10.

| Hypothesis | Measurement | Verdict |
|---|---|---|
| Saturation of injected currents | Cells with \|activity\| > 0.95 in the zone: 0.00 for input, all and readout cells. `ReadoutNorm` ±10 clip: 0.000 | Refuted |
| Recurrent dynamics swamp the input | At the ball_y input cells, median \|1.4·msg\| / \|drive\| = 0.38. The drive dominates | Refuted |
| Time constants smear the 4-tick steps | Prior state is retained at 0.3³ ≈ 3% per step (3 iterations, leak 0.7). The step-3 lag profile peaks at 0 | Refuted |
| y/vy information is lost in the zone | Ridge probe on the 64 standardized readout cells, active zone: **R² y 0.98, vy 0.99** (x 1.00) | Refuted, see note |
| Few channels, few effective dimensions, flipper feedback dominates | See the four points below the table | **Supported** |

Why the last hypothesis holds:
- **Ball y is a single, compressed scalar.** Each ball variable reaches the circuit as one number (obs/scale), identical on all 12 cells of its channel. ball_y/20 puts the whole flipper band (y 10.2–14) into drive 0.51–0.70. One decision step of fall (median 0.1 units) is a drive change of 0.005.
- **The visual looming channels are silent in the zone.** vel1_*/size1_* are exactly 0 for every step with y > 9: `LOOM_ZONE_Y = 9` in `pinball_env.py`, and the distance term goes negative past it.
- **The readout is low-dimensional and dominated by flipper feedback.** Participation ratio is 6.0 in the zone and 4.2 overall. A 0.1 y step moves the readout by d′ = 0.56, measured against the readout variability that the current observation does not explain linearly. Toggling the flipper-state channel moves it by d′ = 49, about 90× more.
- **So the readout can find where the ball is but can't cut sharply.** A linear readout of this code places the ball well but cannot draw the sharp AND-boundary the reflex's label needs:
  - raw-observation linear readout: precision@rate 0.08 / 0.14 (left / right);
  - circuit: 0.27 / 0.52;
  - an MLP on the raw observation: 0.51 / 0.68.

**Note: the step-3 zone R² of y −0.20 was a probe artifact.** `circuit_probe.ridge_r2` uses ridge λ = 1·n_train on unstandardized activity, and the readout's standard deviation is only about 0.08. On the same data, that probe gives y ≈ 0.00, while a standardized ridge gives y 1.00 and vy 0.97 (`r2_recheck.log`).

**The precision metric has a ceiling below 1.** 37% of the steps with y > 11.5, falling, on the correct side fall inside the reflex's 2-step cooldown, so the observation alone cannot label them. A reader with no memory tops out at precision 0.68 (left) / 0.59 (right). Knowing the current flipper state raises that to about 0.82 / 0.76.

## 2. Encodings (all in `encoding.py`)

Every encoding uses the same 180 input cells, the same dynamics, the same 64 readout cells and the same 667 decoder parameters. Only the drive each input cell receives changes.

- **`gain`:** re-centres and re-scales the broadcast inputs, y → (y−10)/3 and vy → vy/10. This is a normalization fix, not a population code.
- **`popcode` (recommended):**
  - The 12 ball_y cells get Gaussian bumps at y = −10, −5, 0, 4, 7, 9, then 10.2 to 13.4 in steps of about 0.6.
  - The 12 ball_x cells tile x from −7 to 7.
  - The ball_vx and ball_vy cells become two sets of 6 direction-selective sigmoid units, one set per direction, with graded speed thresholds (0.5 to 35 units/s).
  - Tuning centres come from table geometry, not from the reflex's threshold.
  - Biological analogy: fly leg proprioceptors fractionate their range. Femoral chordotonal claw neurons tile joint position, and hook neurons are direction-selective for movement (Mamiya et al., 2018, *Nat. Neurosci.*). The ball channels already enter through VNC mechanosensory (SN) cells. Assigning cell subtypes to channels remains an engineering choice.
- **`dspop`:** the vy cells become falling-only position bumps, bump(y − c_k)·σ(vy). This is T4/T5-like direction selectivity at retinotopic positions; T4/T5 are retinotopic and direction-selective. It hands the circuit a feature very close to the label.
- **`retino`:** 2-D ball-position bumps over the 96 LC4/LPLC2 cells, with an approach gate on LC4. LC4/LPLC2 are retinotopic looming detectors. In this connectome they measurably barely reach the readout.
- **`+ttc`:** the 12 tilt cells carry per-flipper time-to-contact populations with log-Gaussian preferences at 0.5 to 7 steps. This is an LPLC2 / giant-fibre-like τ code. Putting it on tilt mechanoreceptors is **not** anatomically plausible.

**Offline separability with delay1 flipper obs.** Precision is at the label's own press rate; the value after "/" is the swapped fold. The bar to beat is 0.27–0.52 at AUC 0.97.

| Encoding | AUC (L / R) | Precision, left | Precision, right | Readout PR (zone) | d′ for a 0.1 y step |
|---|---|---|---|---|---|
| broadcast | 0.975 / 0.983 | 0.28 / 0.27 | 0.52 / 0.53 | 6.0 | 0.56 |
| retino | 0.976 / 0.983 | 0.27 / 0.27 | 0.52 / 0.53 | 5.9 | 0.56 |
| broadcast+ttc | 0.988 / 0.992 | 0.36 / 0.36 | 0.57 / 0.50 | 7.0 | 0.44 |
| gain | 0.996 / 0.997 | 0.49 / 0.19 | 0.54 / 0.58 | 4.6 | 0.89 |
| **popcode** | **0.999 / 0.999** | **0.69 / 0.68** | **0.66 / 0.71** | 8.3 | 1.48 |
| dspop | 0.999 / 0.998 | 0.66 / 0.68 | 0.71 / 0.73 | 7.7 | 1.63 |
| popcode+ttc | 0.999 / 0.999 | 0.81 / 0.66 | 0.82 / 0.68 | 8.7 | 1.27 |
| dspop+ttc | 0.999 / 0.999 | 0.83 / 0.73 | 0.85 / 0.76 | 8.4 | 1.51 |

Raw and zero flipper modes are in `probe_v2.log`. With raw flipper obs, popcode reaches 0.68 / 0.85.

What the table shows:
- **Retinotopy on the visual cells does nothing.** That pathway does not reach the readout in this circuit.
- **Population codes on the mechanosensory cells roughly double precision** and reach about the observation-only ceiling. TTC adds a little more offline.

## 3. DAgger imitation plus held-out evaluation (lives 7024–7071, 48 lives)

All rows use `dagger_enc.py`: 3 DAgger rounds on 7000–7023 and readout_dim 8. Recall and precision are measured against a shadow reflex.

| Variant | log1p ± se | Paired Δ vs broadcast | Paired Δ vs reflex | Presses/life | Recall | Precision |
|---|---|---|---|---|---|---|
| broadcast, delay1 (control) | 10.52 ± 0.11 | 0 | −0.93 ± 0.16 | 10.1 | 0.19 | 0.22 |
| **popcode, delay1** | **11.41 ± 0.12** | **+0.89 ± 0.17** | **−0.04 ± 0.11** | 11.0 | 0.79 | 0.79 |
| popcode, raw | 11.28 ± 0.14 | +0.77 ± 0.15 | −0.17 ± 0.15 | 12.2 | 0.89 | 0.89 |
| dspop+ttc, delay1 | 11.15 ± 0.12 | +0.64 ± 0.13 | −0.30 ± 0.15 | 10.3 | 0.71 | 0.73 |

- **The control reproduces step 3 exactly** (10.52), so the gain comes from the encoding and not from a pipeline change.
- **Better offline separability did not give a better closed-loop score.** dspop+ttc beat popcode offline but scores lower closed loop, which shows the limits of offline separability as a proxy. The simplest and most defensible encoding, popcode, is best.
- **Success criterion (≥ 11.2) is met** by popcode with delay1 and with raw flipper obs.

## 4. Recommendation, start checkpoint and command (NOT launched)

Use the `popcode` encoding with `--flipper-obs delay1`.

The start checkpoint is `agents/experiments/encoding/start_popcode_seed.pt`. It holds:
- `mean` = `champion` = the DAgger weights (667 values);
- `sigma` = 0.05, `champion_fitness` = −1e18, `generation` = 0, `readout_dim` = 8;
- `flipper_obs` = 'delay1', `encoding` = 'popcode', `press_cost` = 0.00088, `objective` = 'life'.

```
python agents/experiments/encoding/train_enc.py --encoding popcode \
  --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball \
  --connectome agents/experiments/big_circuit/connectome.json --readout-dim 8 --flipper-obs delay1 \
  --optimizer cma --cma-sigma0 0.05 --workers 8 --population 16 --fresh-fraction 0 \
  --ipop-patience 80 --ipop-growth 2.0 --ipop-max-population 48 --generations 300 \
  --courses-per-candidate 8 --validation-episodes 32 --promotion-t 2.0 --max-steps 3000 \
  --press-cost 0.00088 --resume agents/experiments/encoding/start_popcode_seed.pt \
  --out agents/experiments/encoding_run/popcode_seed.pt --save-every 5
```

`train_enc.py` is a wrapper; the trainer file is not modified. It does four things:
- It patches `train_pinball_circuit_cem.build_agent` so that the agent kind `circuit` returns an `EncodedCircuitAgent`.
- It sets `PINBALL_ENCODING`, so the spawned workers, which re-import it as `__mp_main__`, apply the same patch.
- It adds `encoding` to every saved checkpoint and refuses to resume from a checkpoint with a different encoding.
- It requires `--resume` or `--no-seed-heuristic` for any encoding other than broadcast, because the trainer's `heuristic_seed` builds an unencoded agent.

Verification:
- **Wrapper smoke test:** 1 generation with 1 worker and max-steps 150 (`smoke_train_enc.log`). It resumed, validated and saved a checkpoint with `encoding = popcode`.
- **Worker patching:** a spawn check confirmed the workers build an `EncodedCircuitAgent('popcode')`.
- **No-op check:** the `broadcast` encoding is bit-identical to `FixedCircuitAgent` (max abs diff 0 over 300 steps).

## Files (all new, under `agents/experiments/encoding/`)

| File | What it is |
|---|---|
| `encoding.py` | The encodings and `EncodedCircuitAgent` |
| `collect.py` | Records the reflex rollouts → `reflex_rollouts_7000.npz` |
| `probe.py` | Diagnosis and separability → `probe_v1.log`, `probe_v2.log`/`.json` |
| `r2_recheck.py` | Re-check of the step-3 R² probe → `r2_recheck.log` |
| `dagger_enc.py` | DAgger plus held-out scoring → `dagger_enc_v1.log`, `dagger_enc_v2.log`, `dagger_enc.json`, `seeds/*.pt` |
| `train_enc.py` | Trainer wrapper |
| `start_popcode_seed.pt` | Start checkpoint |
| `smoke_train_enc.log` | Wrapper smoke-test log |

## Risks and follow-ups

- **CMA's gain is untested.** The seed matches the reflex, but it is not known whether CMA improves on it under press cost. At σ0 0.2, step 3 showed seeds are fragile; keep σ0 at 0.05.
- **The browser runtime does not support the new encoding.** `web/connectome.js` and the export assume broadcast channels, so the per-cell encoding must be ported before deployment.
- **Norm calibration uses unrealistic observations.** The readout norm is calibrated from random Box samples, as in production. With popcode, many cells have near-zero spread in those samples. No clipping was observed (0.000), but under CMA the norm parameters are searchable and could drift.
- **The +ttc variant is not anatomically plausible**, because it reuses the tilt cells. It is not recommended.
- **Two metrics in `probe.py` only apply to broadcast.** For the population codes, the `rec/drv` metric is meaningless because most cells have near-zero drive, and `ycell |corr|` is not comparable.
- **Sampling noise is sizeable.** The ±1-step tolerant precision and the swapped fold vary by about 0.05–0.15 across folds, because there are only about 80 positive steps per fold.

## 5. Browser port (2026-09-30 08:56 CST)

- **Readout JSON schema:** `export_enc.py` writes the standard version-2 readout JSON plus two optional fields. `"encoding"` holds the name, `sigmoid_clip` and the per-channel `bumps` centres/sigmas or `ds_speed` thresholds/scales. `"flipper_obs"` is `raw` or `delay1`. JSON without these fields behaves exactly as before, so old exports are unaffected.
- **`web/circuit_policy.js`:** `PinballConnectomeJS.step` takes an optional per-input-cell drive.
  - `PinballCircuitPolicyJS` gains `filterObs` (delay1, reset in `reset()`) and `encodeInputs` (popcode).
  - Encoding math runs in float64 on float32-rounded observations and is rounded to float32 once, like NumPy.
  - The sigmoid argument is clipped to ±50, as in `encoding.py`. Gaussian bumps only take `exp` of non-positive numbers, so they cannot overflow.
- **`web/bridge.js`:**
  - Adds `?brain=popcode`.
  - Resets the policy at a detected drain, using the same rule as `pinball_env.py`. Previously it reset only on the `ball_in_play` false→true edge, and that edge never fires after a drain.
  - The drain reset applies to every brain.
- **Tests:** `web/test/verify_policy.mjs` also checks the popcode vectors, including the delay1 state, the filtered obs and the per-cell drive. On 600 real-engine steps over 3 lives with 108 delayed-flipper-up steps:
  - obs diff 8e-7, flipper filter diff 0, input drive diff 0 (bit-exact);
  - single-step logit relative diff 2.9e-6, action mismatches 0;
  - free-running logit diff 3e-5.
  - The old 444 and 108 models still pass.
  - Negative controls both fail as they should: a wrong `flipper_obs` and one y centre shifted by 0.3.
- **Build:** no WASM or dist rebuild is needed, because no engine code changed.
- **Export command for a trained checkpoint** (copy the checkpoint first if the run is still writing it):
  ```
  python agents/experiments/encoding/export_enc.py --checkpoint agents/experiments/encoding_run/popcode_seed.pt \
    --connectome web/connectome_444.json --out web/circuit_readout_444_popcode.json \
    --binary vendor/SpaceCadetPinball/bin/SpaceCadetPinball --test-vectors web/testdata/circuit_444_popcode_vectors.json
  node web/test/verify_policy.mjs
  ```

## 6. Paired confirmation on untouched seeds 5000–5095 (2026-09-30 09:09 CST)

This run used `confirm.py` → `confirm_5000.json` / `.log`. Each of the 96 seeds is played as one life, and every policy gets the same seeds. The episode ends the way the trainer's does: terminated, truncated at 3000 steps, or `info["drained"]`.

| Policy | log1p ± SE | With press cost 0.00088 | Median score | Mean score | Steps | Presses/life | Contacts/life |
|---|---|---|---|---|---|---|---|
| popcode seed | 11.30 ± 0.08 | 11.29 | 82,250 | 106,776 | 1149 | 10.9 | 6.1 |
| gen-280 | 11.06 ± 0.09 | 10.38 | 68,125 | 88,130 | 1193 | 780.6 | 7.4 |
| reflex y>11.5 | 11.30 ± 0.08 | 11.28 | 72,500 | 112,047 | 1173 | 12.1 | 6.7 |
| never press | 9.76 ± 0.08 | 9.76 | 20,250 | 23,078 | 476 | 0 | 0 |

Paired differences. The Wilcoxon signed-rank test uses a normal approximation because scipy is not installed. "Wins/losses" counts lives where the first policy scored higher or lower.

| Comparison | Mean Δ ± SE | t | Wilcoxon p | Wins/losses |
|---|---|---|---|---|
| popcode − gen280 | +0.235 ± 0.111 | 2.10 | 0.078 | 50/38 |
| popcode − gen280, with press cost | +0.912 ± 0.093 | 9.82 | 3e-14 | 86/10 |
| popcode − reflex | +0.002 ± 0.081 | 0.02 | 0.92 | 34/31 |
| popcode − never press | +1.535 ± 0.106 | 14.5 | 9e-16 | 85/1 |
