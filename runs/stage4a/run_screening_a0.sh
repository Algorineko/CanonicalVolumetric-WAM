#!/bin/bash
cd /home/zhangjialiang/code/RoboTwin
PY=/home/zhangjialiang/miniconda3/envs/robottwin/bin/python
$PY scripts/eval_policy_xpolicylab.py \
  --task_name open_microwave --policy_name cvwam_fm --port 8849 --protocol ws \
  --task_config demo_clean_balanced --seed 0 --test_num 20 --instruction_type seen \
  --device_id 0 > /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/rollout_screen_clean_a0.log 2>&1
$PY scripts/eval_policy_xpolicylab.py \
  --task_name open_microwave --policy_name cvwam_fm --port 8849 --protocol ws \
  --task_config demo_randomized_balanced --seed 0 --test_num 20 --instruction_type seen \
  --device_id 0 > /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/rollout_screen_rand_a0.log 2>&1
echo "SCREENING_A0_DONE"
