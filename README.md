# 3DPinball-DrosophilaCadet

An open-source Reinforcement Learning (RL) environment wrapping the classic Windows XP *3D Pinball: Space Cadet* game engine, designed for training biological-inspired neural networks and virtual fruit fly models (*Drosophila melanogaster*).

## 🚀 Overview
This project decompiles and refactors the native C++/SDL2 game engine (`k4zmu2a/SpaceCadetPinball`) to expose internal game states (ball vectors, flipper physics, rewards) directly to a Python-based `gymnasium` interface. This allows AI agents—ranging from traditional PPO/DQN models to neuromorphic Spiking Neural Networks (SNN)—to interact with the game with zero screen-capture overhead.

## 🪰 Main result: the fly's own escape pathway plays pinball
The flippers are driven by the fly's looming-escape circuit taken straight from the MaleCNS v1.0
connectome: 373 cells, all LC4/LPLC2 looming neurons -> the two giant fibers (DNp01) plus 60
intermediates, with real signs and synapse counts. A ball approaching a flipper "looms" on that
side's eye; when that side's giant fiber becomes active, the flipper is pressed. There is **no trained readout**.
Only a few body constants (looming gain and time constant, press threshold) are tuned per circuit,
and the dynamics are kept sub-critical (gain x spectral radius = 0.8).

- **The real wiring matters.** Under the same press budget (<= 25 presses/life) the real circuit
  beats all six degree-preserving shuffles on 96 paired lives (log1p score per life 11.22 vs
  10.77-10.99; pooled shuffle - real = -0.34 +/- 0.08, t -4.25) and is about 1.5x more press-efficient
  (0.49 vs 0.25-0.36 ball contacts per press). It is still 0.26 below the hand-written reflex.
- Why it works: LC4/LPLC2 project strictly ipsilaterally onto the giant fibers (~26% of each GF's
  input), so the real circuit is a lateralized looming detector; the shuffles are not.
- Code, circuits, tuning and evaluation: `agents/experiments/pathway_body/`; pathway analysis:
  `agents/experiments/pathway/`; details in `docs/MORNING_REPORT_2026-10-02.md`.

```sh
# Paired evaluation: real giant-fiber body vs six shuffles, lead reflex and never-press
python3 agents/experiments/pathway_body/eval_gf_body.py --seeds 77000:77096 --suffix _budget --shuffles 1,2,3,4,5,6
```

## 🏆 Best score: the trained fly brain
A fixed 444-neuron subnetwork of the MaleCNS *Drosophila* connectome (real cell types, signs and
synapse counts; the wiring is never trained) plays the game. Only a small readout (667 parameters)
is trained, by imitation (DAgger) of a hand-written reflex teacher.

- **Sensory encoding ("popcode")**: ball position/velocity enter the connectome's mechanosensory
  input cells as range-fractionated bumps and direction-selective speed units (like leg
  proprioceptor claw/hook neurons) - see `docs/ENCODING_REPORT.md`.
- **Teacher**: a correct-side flipper reflex that presses one decision step early
  (`agents/experiments/timing/`), which hits the multiplier targets more often.
- **Best brain**: `agents/experiments/timing/seeds/dagger_popcode_delay1_r8.pt`
  (web export `web/circuit_readout_444_popcode.json`). It presses ~11 times per ball (the earlier
  CMA champion: ~780) and scores on par with its teacher on held-out seeds.
- **Caveat**: with this trained readout, degree-preserving shuffles of the wiring score the same
  (`agents/experiments/shuffle/`), so the readout, not the wiring, does the work here.
- **Findings and negative results** (timing oracle, aiming, cradling, full-game training):
  `docs/STEP3_REPORT.md`; full experiment log in the local, untracked changelog.

```sh
# Browser demo (serves web/; the page only runs the best brain)
cd web && python3 -m http.server 8765 --bind 127.0.0.1    # open http://127.0.0.1:8765/

# Full-game evaluation of the best brain (fresh engine per game, auto-relaunch after drains)
python3 agents/experiments/fullgame/full_game_eval.py --policies brain --seeds 40000:40032

# Re-train the brain by imitation of the lead reflex
python3 agents/experiments/timing/dagger_lead.py --variants popcode:delay1:8 --workers 1 \
  --ckpt-dir agents/experiments/timing/seeds --out agents/experiments/timing/dagger_lead.json
```

## 🛠️ Tech Stack
- **Game Core:** C++11 / SDL2 (Cross-platform compatibility)
- **RL Framework:** Python 3 / Gymnasium
- **Agent Options:** PyTorch / Stable-Baselines3 / Spiking Neural Networks (SNN)

## 📂 Repository Structure
- `/vendor/SpaceCadetPinball`: Git submodule, unmodified upstream `k4zmu2a/SpaceCadetPinball`.
- `/src_cpp`: State-extraction + IPC/Wasm bridge overlay, applied onto the submodule via the
  patches in `src_cpp/patches/` rather than forking it.
- `/env_python`: `ipc_client.py` (wire-protocol client) and `pinball_env.py` (Gymnasium env).
- `/agents`: `snn_model.py` (Norse SNN controller), `train.py` (REINFORCE loop),
  `export_weights.py` (`weights.json` export for the browser).
- `/web`: Browser demo - `index.html`, `bridge.js`, `inference.js` (JS port of the SNN forward
  pass), `weights.json` (committed export output), `dist/` (gitignored Emscripten build output).
- `/scripts`: `apply_patches.sh`, `build_native.sh`, `build_wasm.sh`, `smoke_ipc.py`.
- `/.github/workflows`: `ci.yml` (native build + IPC smoke test), `deploy.yml` (Wasm build +
  GitHub Pages deploy).

## Getting started (macOS local dev)

**You need your own `CADET.DAT` or `PINBALL.DAT`.** This project does not include or fetch the
original game's data files - they're copyrighted (from Windows XP or Full Tilt! Pinball).
Extract one from a copy you legitimately own and place it next to the built binary
(`vendor/SpaceCadetPinball/bin/`) or in the working directory you run it from.

```sh
git submodule update --init --recursive   # first time only
./scripts/build_native.sh                  # applies src_cpp/patches, builds via Homebrew SDL2
```

Requires Xcode Command Line Tools with the license accepted (`sudo xcodebuild -license accept`)
and `brew install sdl2 sdl2_mixer` first.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r agents/requirements.txt
python3 scripts/smoke_ipc.py               # protocol/client test, no DAT file needed

PINBALL_BINARY=vendor/SpaceCadetPinball/bin/SpaceCadetPinball \
  python3 agents/train.py --binary "$PINBALL_BINARY" --episodes 500
python3 agents/export_weights.py --checkpoint agents/checkpoints/drosophila_controller_final.pt

./scripts/build_wasm.sh                    # requires Emscripten; untested in this repo so far
```

`agents/train.py --dummy` runs the same training loop against a synthetic stand-in
environment (`agents/dummy_env.py`) for verifying the training plumbing without a DAT file -
it is not a pinball simulator and learning to solve it says nothing about real gameplay.
