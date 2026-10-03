#!/bin/zsh
# After the sweep tuning finishes: blind-spot check (mid and sweep layouts) and the paired 96-life evaluation.
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
while [ ! -f run_retina_sweep.done ]; do sleep 60; done
$PY blind_spot.py --which retina --params best_retina_real.json --out blind_spot_retina_mid.json > blind_spot_retina_mid.log 2> blind_spot_retina_mid.err
$PY blind_spot.py --which retina --params best_retina_real_sweep.json --out blind_spot_retina_sweep.json > blind_spot_retina_sweep.log 2> blind_spot_retina_sweep.err
$PY eval_gf_retina.py --suffix _sweep > eval_gf_retina_sweep.log 2> eval_gf_retina_sweep.err
echo ALLDONE > run_retina_sweep_eval.done
