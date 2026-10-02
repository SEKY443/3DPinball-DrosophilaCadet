#!/bin/zsh
# Three more degree-preserving shuffles (4-6) under the same press budget, then a paired evaluation
# of the real GF body against all six shuffles on seeds 77000-77095.
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
A=(--rho 0.8 --theta-range 0.03:0.5 --max-presses 25 --suffix _budget)
for i in 4 5 6; do $PY tune_gf_body.py --name shuffled_$i --circuit gf_circuit_shuffled_$i.json "${A[@]}" > tune_shuffled_${i}_budget.log 2>&1; done
$PY eval_gf_body.py --suffix _budget --shuffles 1,2,3,4,5,6 --workers 8 --out eval_gf_body_budget6.json > eval_gf_body_budget6.log 2>&1
