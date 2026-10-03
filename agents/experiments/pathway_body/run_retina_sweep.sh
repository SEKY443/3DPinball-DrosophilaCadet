#!/bin/zsh
# Blind-spot patch for flipper tips: tune the "sweep" receptive-field layout for the real circuit and six shuffles.
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
for n in real shuffled_1 shuffled_2 shuffled_3 shuffled_4 shuffled_5 shuffled_6; do
  $PY tune_gf_retina.py --name $n --fields sweep --suffix _sweep > tune_retina_sweep_$n.log 2>&1
done
echo ALLDONE > run_retina_sweep.done
