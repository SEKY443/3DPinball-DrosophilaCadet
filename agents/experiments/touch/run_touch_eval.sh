#!/bin/zsh
# After touch tuning: slow-ball probe (sweep vs touch), then the paired 96-life evaluation.
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet/agents/experiments/touch
PY=../../../.venv/bin/python
$PY slowball_probe.py --body sweep > slowball_sweep.log 2> slowball_sweep.err
$PY slowball_probe.py --body touch > slowball_touch.log 2> slowball_touch.err
$PY eval_gf_touch.py --workers 8 > eval_gf_touch.log 2> eval_gf_touch.err
echo ALLDONE > run_touch_eval.done
