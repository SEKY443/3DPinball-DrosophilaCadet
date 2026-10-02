# Morning report - 2026-10-02

Overnight work on "does the fly wiring matter, and can the fly's own escape pathway be the flipper body?"

## 1. Shuffled-wiring control (with a trained readout)
Three degree-preserving shuffles of the 444-cell circuit (node identities, signs, roles and each
cell's in/out degree kept; ~6.5% of edges left in place), trained with exactly the same lead1 DAgger
pipeline as the best brain. Fresh 96 paired lives (seeds 76000-76095):

| Wiring | log1p ± SE | median | diff vs real |
|---|---|---|---|
| real | 11.19 ± 0.11 | 78,750 | - |
| shuffle 1 | 11.21 ± 0.09 | 72,500 | +0.02 (t 0.25) |
| shuffle 2 | 11.24 ± 0.10 | 78,875 | +0.05 (t 0.54) |
| shuffle 3 | 11.31 ± 0.10 | 82,875 | +0.13 (t 1.38) |

Verdict: with a trained readout the specific fly wiring gives no measurable advantage - the readout
and imitation do the work (flybench would call this task "non-diagnostic").

## 2. Pathway analysis
- 444-cell circuit: real wiring has lateralized sensory->motor structure that shuffles lack, but the
  visual pathway barely reaches the motor outputs and the DNp11/DNp03 layer is a bottleneck.
- Full MaleCNS v1.0 (re-downloaded to `data/`, MD5 verified): LC4/LPLC2 looming neurons project
  strictly ipsilaterally onto the giant fibers DNp01 (left eye -> left GF only; ~26% of each GF's
  input), every LC4/LPLC2 cell contacts its GF. No retinotopic coordinates in the v1.0 annotations.

## 3. Giant-fiber body (no trained readout) - in progress
Built by an implementer subagent in `agents/experiments/pathway_body/`: 373-cell circuit (all
LC4/LPLC2, both GFs, 60 intermediates); ball looming toward a flipper drives that side's eye; GF
activity above a threshold presses that side's flipper; only a few constants tuned per circuit.

First run (default dynamics gain 1.4): the circuit LATCHES - any drive switches both GFs on for the
rest of the life, so the body just spams the flippers (~620 presses/life). Real vs shuffles: no
significant difference (+0.05 / +0.12 / +0.26 for real over shuffles, only shuffle 3 near p 0.06).
Cause: spectral radius of the real circuit is 0.996 (shuffles 0.78-0.82), so gain*rho > 1.
Side finding: the real fly wiring is markedly more recurrent than its degree-matched shuffles.

Second run (each circuit set to gain*rho = 0.8, sub-critical):
- The latch is gone; activity returns to 0 after a stimulus. The real circuit is now a selective
  looming detector: driving one eye raises the same-side GF 25-40x more than the other side
  (e.g. loom_L A=0.5 -> GF_L 0.316, GF_R 0.008). The shuffles are essentially non-lateralized.
- Best real config: GF quiet when the ball is far (2.8% of steps above threshold) and active near the
  flippers (19.5%).
- 96 paired lives (seeds 77000-77095):

| Policy | log1p | presses/life | contacts/life | contacts per press | diff vs real |
|---|---|---|---|---|---|
| GF body, real wiring | 11.22 | 19.3 | 9.4 | 0.49 | - |
| GF body, shuffle 1 | 11.09 | 47.3 | 6.1 | 0.13 | -0.13 (n.s.) |
| GF body, shuffle 2 | 11.39 | 90.3 | 12.1 | 0.13 | +0.17 (t 1.45, p 0.04 Wilcoxon) |
| GF body, shuffle 3 | 11.23 | 83.3 | 11.0 | 0.13 | +0.01 (n.s.) |
| lead1 reflex | 11.48 | 16.5 | 8.9 | 0.54 | +0.26 (t 2.5) |
| never press | 9.85 | 0 | 0 | - | -1.37 |

The real wiring's signature is PRESS EFFICIENCY (about 4x more ball contacts per press, close to the
hand-written reflex) - not score, because the game does not penalize extra presses and the shuffles
compensate by pressing 2.5-5x more.

Third run - SAME PRESS BUDGET (every circuit re-tuned with <= 25 presses/life on dev lives; same 96
paired lives; numbers re-checked independently from eval_gf_body_budget.json):

| Policy | log1p ± SE | median | presses/life | contacts per press | diff vs real GF | t | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| **GF body, real wiring** | **11.22 ± 0.09** | 73,500 | 19.3 | **0.49** | - | | |
| GF body, shuffle 1 | 10.80 ± 0.09 | 52,750 | 20.5 | 0.31 | **-0.43** | -4.45 | 0.00008 |
| GF body, shuffle 2 | 10.97 ± 0.10 | 54,000 | 25.4 | 0.31 | **-0.26** | -2.38 | 0.0095 |
| GF body, shuffle 3 | 10.99 ± 0.10 | 63,375 | 21.6 | 0.35 | **-0.24** | -2.26 | 0.033 |
| lead1 reflex (hand-written) | 11.48 ± 0.09 | 97,375 | 16.4 | 0.54 | +0.26 | +2.47 | 0.007 |
| never press | 9.85 | 23,375 | 0 | - | -1.37 | | |

**Headline: with no trained readout and the same press budget, the real fly wiring plays
significantly better than every degree-matched shuffle (+0.24 to +0.43 log1p).** This is the first
result in the project where the fly's actual wiring - the strictly ipsilateral LC4/LPLC2 -> giant
fiber escape pathway - is what makes the difference. It is still 0.26 below the hand-written reflex.
(Caveats: 3 shuffles, 60 random configs x 32 dev lives per circuit; shuffle 3's p would not survive a
Bonferroni correction across the three comparisons, shuffles 1 and 2 would.)

## Six-shuffle check (added 04:45)
Three more degree-preserving shuffles (4-6, make_shuffled.py on the GF circuit) tuned under the same
<= 25 presses/life budget; all six evaluated with the real GF body on the same 96 paired lives
(seeds 77000-77095), numbers recomputed from eval_gf_body_budget6.json:

| Shuffle | log1p | presses/life | contacts/press | diff vs real | t |
|---|---|---|---|---|---|
| (real GF body) | 11.22 | 19.3 | 0.49 | - | - |
| 1 | 10.80 | 20.5 | 0.31 | -0.43 | -4.45 |
| 2 | 10.97 | 25.4 | 0.31 | -0.25 | -2.37 |
| 3 | 10.99 | 21.6 | 0.35 | -0.24 | -2.26 |
| 4 | 10.94 | 28.0 | 0.25 | -0.28 | -2.55 |
| 5 | 10.77 | 20.9 | 0.30 | -0.45 | -4.49 |
| 6 | 10.84 | 20.2 | 0.36 | -0.38 | -3.38 |

The real wiring beats all six shuffles; all six differences are significant at p < 0.05 (Wilcoxon
p 2.6e-6 to 0.033); three of six (shuffles 1, 5, 6) survive a Bonferroni correction (0.05/6 = 0.0083),
and the pooled test below is the stronger summary. Pooled over the six
shuffles: shuffle - real = -0.34 +/- 0.08 log1p (t -4.25). Every shuffle also has lower press
efficiency (0.25-0.36 contacts per press vs 0.49).

## What exists now (all uncommitted, new files only)
- `agents/experiments/shuffle/` - shuffled-connectome control for the trained brain (+ eval).
- `agents/experiments/pathway/` - 444-cell pathway analysis, full-MaleCNS giant-fiber analysis.
- `agents/experiments/pathway_body/` - 373-cell giant-fiber circuit (+3 shuffles), `GiantFiberBody`
  controller (looming drive -> GF rate -> thresholded flipper press, sub-critical dynamics
  gain*rho = 0.8), probe, tuning (press budget) and paired evaluation scripts with logs/JSON.
- `.github/workflows/deploy.yml` updated for GitHub Pages (game data question still open).
- `data/` holds the full MaleCNS v1.0 download (gitignored, ~1.1 GB).

## Decisions for you
1. **Which result is the headline?** The trained "lead brain" scores higher (~11.2-11.3 vs 11.22,
   on different seed sets) but its wiring does not matter; the giant-fiber body scores about the same
   with no training and its real wiring DOES matter. I suggest presenting the GF body as the
   "fly brain plays" result and the lead brain as the best score.
2. **Put the GF body on the web demo?** Needs a small JS controller + the 373-cell circuit JSON;
   the dashboard can show the two giant fibers and the looming input live.
3. **Commit tonight's work?** (shuffle control, pathway analyses, GF body, deploy.yml). I did not
   commit anything while you slept.
4. **Next research step for the GF body** (options): add LC4-velocity vs LPLC2-size tuning per
   Ache et al. 2019 instead of one looming signal; let GF rate set analog flipper force; add the
   mechanosensory inputs; more shuffles / more tuning to firm up the statistics.
5. **GitHub Pages game data** (still open from before): visitor-supplied DAT vs publishing DEMO.DAT.
