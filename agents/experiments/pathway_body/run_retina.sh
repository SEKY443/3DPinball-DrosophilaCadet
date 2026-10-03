#!/bin/zsh
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
for n in real shuffled_1 shuffled_2 shuffled_3 shuffled_4 shuffled_5 shuffled_6; do
  $PY tune_gf_retina.py --name $n > tune_retina_$n.log 2>&1
done
$PY blind_spot.py --which retina --out blind_spot_retina.json > blind_spot_retina.log 2> blind_spot_retina.err
$PY eval_gf_retina.py > eval_gf_retina.log 2> eval_gf_retina.err
echo ALLDONE > run_retina.done
