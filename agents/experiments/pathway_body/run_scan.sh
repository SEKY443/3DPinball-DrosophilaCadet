#!/bin/zsh
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/pathway_body
PY=../../../.venv/bin/python
while ps -p 76320 > /dev/null 2>&1; do sleep 60; done
for n in real shuffled_1 shuffled_2; do
  $PY scan_gains.py --name $n > scan_$n.log 2>&1
done
$PY plot_scan.py > scan_plot.log 2>&1
$PY eval_scan_best.py > scan_eval.log 2>&1
