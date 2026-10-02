#!/bin/zsh
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
$PY tune_gf_body.py --name real --circuit gf_circuit.json > tune_real.log 2>&1
for i in 1 2 3; do $PY tune_gf_body.py --name shuffled_$i --circuit gf_circuit_shuffled_$i.json > tune_shuffled_$i.log 2>&1; done
