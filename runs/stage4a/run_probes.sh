#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
PY=/home/zhangjialiang/miniconda3/envs/robottwin/bin/python
$PY scripts/replay_episode_gt.py --variant demo_clean_balanced --episode 10 \
  --head-cam-shift-cm 7.5 --head-cam-rot-deg 7.5 \
  --out /data/canonical-volumetric-wam/cache/replay_gt/probe_ep10_mid \
  > runs/stage4a/probe_mid_replay.log 2>&1
$PY scripts/replay_episode_gt.py --variant demo_clean_balanced --episode 10 \
  --head-cam-shift-cm 15 --head-cam-rot-deg 20 \
  --out /data/canonical-volumetric-wam/cache/replay_gt/probe_ep10_strong \
  > runs/stage4a/probe_strong_replay.log 2>&1
OW=/home/zhangjialiang/miniconda3/envs/openwam/bin/python
$OW scripts/build_replay_geometry.py \
  --replay-dir /data/canonical-volumetric-wam/cache/replay_gt/probe_ep10_mid \
  --variant demo_clean_balanced --episode 10 --tag probe_mid \
  > runs/stage4a/probe_mid_geom.log 2>&1
$OW scripts/build_replay_geometry.py \
  --replay-dir /data/canonical-volumetric-wam/cache/replay_gt/probe_ep10_strong \
  --variant demo_clean_balanced --episode 10 --tag probe_strong \
  > runs/stage4a/probe_strong_geom.log 2>&1
echo "PROBES_DONE"
