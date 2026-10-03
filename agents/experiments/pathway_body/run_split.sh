#!/bin/zsh
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
for n in real real_lc4only real_lplc2only shuffled_1 shuffled_2 shuffled_3 shuffled_4 shuffled_5 shuffled_6; do
  $PY tune_gf_split.py --name $n > tune_split_$n.log 2>&1
done
$PY eval_gf_split.py > eval_gf_split.log 2>&1
