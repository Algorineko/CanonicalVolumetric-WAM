#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
PY=/home/zhangjialiang/miniconda3/envs/openwam/bin/python
export CUDA_VISIBLE_DEVICES=1
$PY scripts/train_repr.py --variant p2e --seed 43 --out runs/stage3c/p2e_s43 > runs/stage3c/p2e_s43.log 2>&1
$PY scripts/train_repr.py --variant p2e --seed 44 --out runs/stage3c/p2e_s44 > runs/stage3c/p2e_s44.log 2>&1
$PY scripts/eval_repr.py --checkpoints "p2e_s43=runs/stage3c/p2e_s43/best.pt" "p2e_s44=runs/stage3c/p2e_s44/best.pt" > runs/stage3c/eval_p2e_seeds.log 2>&1
echo "p2e seeds done"
