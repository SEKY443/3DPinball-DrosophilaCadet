#!/bin/zsh
cd /Users/seky/Developer/Git/3DPinball-step3/agents/experiments/gf_force
PY=/Users/seky/Developer/Git/3DPinball-DrosophilaCadet/.venv/bin/python
for n in real shuffled_1 shuffled_2 shuffled_3 shuffled_4 shuffled_5 shuffled_6; do
  $PY tune_gf_force.py --name $n > tune_force_$n.log 2>&1
done
$PY eval_gf_force.py > eval_gf_force.log 2>&1
echo done > run_all.done
