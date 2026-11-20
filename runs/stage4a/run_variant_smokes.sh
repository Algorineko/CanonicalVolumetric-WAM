#!/bin/bash
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
LOCK=/tmp/cvwam_smokes.lock
if mkdir "$LOCK" 2>/dev/null; then trap 'rmdir "$LOCK"' EXIT; else echo "[guard] lock held"; exit 1; fi
DEPLOY=/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/deploy.yml
SRV=/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/setup_eval_policy_server.sh
PY=/home/zhangjialiang/miniconda3/envs/robottwin/bin/python
for v in a1 a2 a3 a3mask; do
  sed -i "s|ckpt_path:.*|ckpt_path: /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/${v}_s42/best.pt|" $DEPLOY
  setsid nohup bash $SRV RoboTwin open_microwave runs/stage4a/${v}_s42/best.pt \
    demo_clean_balanced joint 0 0 openwam 8849 localhost \
    > runs/stage4a/policy_server_${v}.log 2>&1
  until ss -tln 2>/dev/null | grep -q 8849; do sleep 2; done
  cd /home/zhangjialiang/code/RoboTwin
  $PY scripts/eval_policy_xpolicylab.py \
    --task_name open_microwave --policy_name cvwam_fm --port 8849 --protocol ws \
    --task_config demo_clean_balanced --seed 0 --test_num 5 --instruction_type seen \
    --device_id 0 > /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/rollout_smoke_${v}.log 2>&1
  cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
  for p in $(ps aux | grep setup_policy_server | grep -v grep | awk '{print $2}'); do kill -9 $p; done
  sleep 2
  echo "[smoke-done] $v"
done
echo "ALL_SMOKES_DONE"
