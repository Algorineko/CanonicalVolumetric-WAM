#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
PY=/home/zhangjialiang/miniconda3/envs/openwam/bin/python
export CUDA_VISIBLE_DEVICES=0
for seed in 42 43 44; do
  $PY scripts/train_stage4a.py --variant a1 --seed $seed --no-fm-loss \
    --out runs/stage4a/a1nofm_s${seed} > runs/stage4a/a1nofm_s${seed}.log 2>&1
  $PY scripts/eval_stage4a.py --ckpt runs/stage4a/a1nofm_s${seed}/best.pt \
    > runs/stage4a/metrics/eval_a1nofm_s${seed}.console 2>&1
  echo "[done] a1nofm s$seed"
done
echo "A1NOFM_DONE"
