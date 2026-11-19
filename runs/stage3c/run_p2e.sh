#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
export CUDA_VISIBLE_DEVICES=1
# wait for p3 (chain) to finish to avoid GPU contention
while ps aux | grep "train_repr.py --variant p3" | grep -v grep > /dev/null; do sleep 20; done
/home/zhangjialiang/miniconda3/envs/openwam/bin/python scripts/train_repr.py --variant p2e --out runs/stage3c/p2e_s42 > runs/stage3c/p2e_s42.log 2>&1
echo "p2e done"
