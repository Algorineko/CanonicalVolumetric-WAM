#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
PY=/home/zhangjialiang/miniconda3/envs/openwam/bin/python
export CUDA_VISIBLE_DEVICES=1
# wait until all 26 point caches are DONE (stage3c_point_cache.py finishes)
while true; do
  n=$(ls /data/canonical-volumetric-wam/cache/point_ep/*/episode_*/DONE 2>/dev/null | wc -l)
  if [ "$n" -ge 26 ]; then break; fi
  echo "waiting for point caches: $n/26"; sleep 30
done
$PY scripts/train_repr.py --variant p2  --out runs/stage3c/p2_s42  > runs/stage3c/p2_s42.log 2>&1
$PY scripts/train_repr.py --variant p2e --out runs/stage3c/p2e_s42 > runs/stage3c/p2e_s42.log 2>&1
$PY scripts/train_repr.py --variant p3  --out runs/stage3c/p3_s42  > runs/stage3c/p3_s42.log 2>&1
echo "stage3c training chain done"
