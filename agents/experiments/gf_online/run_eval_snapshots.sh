#!/bin/zsh
# Waits for the local retina tuning/eval run to finish (CPU), then evaluates the Colab snapshots at update 300.
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet
while [ ! -f agents/experiments/pathway_body/run_retina.done ] && ps -p 78276 > /dev/null 2>&1; do sleep 60; done
.venv/bin/python agents/experiments/gf_online/eval_snapshots.py --at 300 > agents/experiments/gf_online/eval_snapshots.log 2>&1
.venv/bin/python agents/experiments/gf_online/summarize_online.py agents/experiments/gf_online/results_eval300 > agents/experiments/gf_online/summary_eval300.log 2>&1
echo DONE > agents/experiments/gf_online/eval_snapshots.done
