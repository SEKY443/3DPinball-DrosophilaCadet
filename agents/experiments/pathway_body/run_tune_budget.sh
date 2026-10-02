#!/bin/zsh
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
A=(--rho 0.8 --theta-range 0.03:0.5 --max-presses 25 --suffix _budget)
$PY tune_gf_body.py --name real --circuit gf_circuit.json "${A[@]}" > tune_real_budget.log 2>&1
for i in 1 2 3; do $PY tune_gf_body.py --name shuffled_$i --circuit gf_circuit_shuffled_$i.json "${A[@]}" > tune_shuffled_${i}_budget.log 2>&1; done
$PY eval_gf_body.py --suffix _budget --workers 8 > eval_gf_body_budget.log 2>&1
