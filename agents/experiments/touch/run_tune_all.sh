#!/bin/sh
cd /Users/seky/Developer/Git/3DPinball-DrosophilaCadet
for n in real shuffled_1 shuffled_2 shuffled_3 shuffled_4 shuffled_5 shuffled_6; do
  .venv/bin/python agents/experiments/touch/tune_gf_touch.py --name $n > agents/experiments/touch/tune_touch_$n.log 2> agents/experiments/touch/tune_touch_$n.err
done
