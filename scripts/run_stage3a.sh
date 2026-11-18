#!/bin/bash
# Stage 3a full sequence: overfit checks -> split trainings -> evals + ablations.
# GPU 0 only; sequential. Requires episode voxel caches to be complete.
set -x
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
PY=/home/zhangjialiang/miniconda3/envs/openwam/bin/python
export CUDA_VISIBLE_DEVICES=0

# --- overfit single episode (learning-signal check, 500 steps each) ---
$PY scripts/train_tiny.py --variant p1    --mode overfit --steps 500  2>&1 | tee runs/stage3a/metrics/overfit_p1.log
$PY scripts/train_tiny.py --variant p1_r  --mode overfit --steps 500  2>&1 | tee runs/stage3a/metrics/overfit_p1_r.log
$PY scripts/train_tiny.py --variant p1_v  --mode overfit --steps 500  2>&1 | tee runs/stage3a/metrics/overfit_p1_v.log

# --- split trainings (train clean 10/25/40, val randomized 0) ---
$PY scripts/train_tiny.py --variant p1       --mode split --steps 3000 2>&1 | tee runs/stage3a/metrics/train_p1.log
$PY scripts/train_tiny.py --variant p1_r     --mode split --steps 3000 2>&1 | tee runs/stage3a/metrics/train_p1_r.log
$PY scripts/train_tiny.py --variant p1_v     --mode split --steps 3000 2>&1 | tee runs/stage3a/metrics/train_p1_v.log
$PY scripts/train_tiny.py --variant act_only --mode split --steps 3000 2>&1 | tee runs/stage3a/metrics/train_act_only.log

# --- evaluation: val + test episodes ---
EP="demo_randomized_balanced:0 demo_randomized_balanced:1"
$PY scripts/eval_tiny.py --ckpt p0 --variant p0 --episodes $EP --tag p0 2>&1 | tee runs/stage3a/metrics/eval_p0.log
for v in p1 p1_r p1_v act_only; do
  $PY scripts/eval_tiny.py --ckpt runs/stage3a/${v}_split/best.pt --episodes $EP --tag $v 2>&1 | tee runs/stage3a/metrics/eval_$v.log
done

# --- action-shuffle ablation ---
for v in p1 p1_v; do
  $PY scripts/eval_tiny.py --ckpt runs/stage3a/${v}_split/best.pt --episodes $EP --tag $v --shuffle-actions 2>&1 | tee runs/stage3a/metrics/eval_${v}_shuffle.log
done

# --- camera-dropout ablation (inputs rebuilt from subsets; targets = full 3-view GT) ---
for sub in head_only head_left head_right; do
  $PY scripts/eval_tiny.py --ckpt p0 --variant p0 --episodes $EP --tag p0 --input-dropout $sub 2>&1 | tee runs/stage3a/metrics/eval_p0_drop_$sub.log
  $PY scripts/eval_tiny.py --ckpt runs/stage3a/p1_split/best.pt --episodes $EP --tag p1 --input-dropout $sub 2>&1 | tee runs/stage3a/metrics/eval_p1_drop_$sub.log
  $PY scripts/eval_tiny.py --ckpt runs/stage3a/p1_v_split/best.pt --episodes $EP --tag p1_v --input-dropout $sub 2>&1 | tee runs/stage3a/metrics/eval_p1_v_drop_$sub.log
done
echo "STAGE3A_SEQUENCE_DONE"
