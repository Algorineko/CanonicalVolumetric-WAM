#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
LOCK=/tmp/cvwam_variants.lock
if mkdir "$LOCK" 2>/dev/null; then
  trap 'rmdir "$LOCK"' EXIT
else
  echo "[guard] lock held; exiting"
  exit 1
fi
PY=/home/zhangjialiang/miniconda3/envs/openwam/bin/python
export CUDA_VISIBLE_DEVICES=1
for seed in 42 43 44; do
  for v in a0 a1 a2 a3 a3mask; do
    if [ -f runs/stage4a/${v}_s${seed}/meta.json ]; then echo "[skip] $v s$seed"; continue; fi
    $PY scripts/train_stage4a.py --variant $v --seed $seed --out runs/stage4a/${v}_s${seed} \
      > runs/stage4a/${v}_s${seed}.log 2>&1
    if [ -f runs/stage4a/${v}_s${seed}/meta.json ]; then
      $PY scripts/eval_stage4a.py --ckpt runs/stage4a/${v}_s${seed}/best.pt \
        > runs/stage4a/metrics/eval_${v}_s${seed}.console 2>&1
      echo "[done] $v s$seed"
    else
      echo "[FAILED] $v s$seed"
    fi
  done
done
echo "ALL VARIANTS DONE"
