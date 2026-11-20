#!/bin/bash
# v2: server launched in BACKGROUND (v1 ran it in foreground and blocked after a1).
# Per-variant: fresh server on a verified-free port, wait for LISTEN, 5-ep smoke seed 0 Clean, kill by PID.
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
LOCK=/tmp/cvwam_smokes.lock
if mkdir "$LOCK" 2>/dev/null; then trap 'rmdir "$LOCK"' EXIT; else echo "[guard] lock held"; exit 1; fi
DEPLOY=/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/deploy.yml
SRV=/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/setup_eval_policy_server.sh
PY=/home/zhangjialiang/miniconda3/envs/robottwin/bin/python
GPU=${GPU:-1}
VARIANTS=${VARIANTS:-"a1 a2 a3 a3mask"}
for v in $VARIANTS; do
  if ss -tln 2>/dev/null | grep -q ':8849 '; then echo "[abort] port 8849 already in use before $v"; exit 2; fi
  sed -i "s|ckpt_path:.*|ckpt_path: /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/${v}_s42/best.pt|" $DEPLOY
  grep -n "ckpt_path" $DEPLOY > runs/stage4a/smoke_ckpt_${v}.txt
  setsid nohup bash $SRV RoboTwin open_microwave runs/stage4a/${v}_s42/best.pt \
    demo_clean_balanced joint 0 $GPU openwam 8849 localhost \
    > runs/stage4a/policy_server_${v}.log 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 90); do
    if ss -tln 2>/dev/null | grep -q ':8849 '; then ok=1; break; fi
    if ! kill -0 $SPID 2>/dev/null; then break; fi
    sleep 2
  done
  if [ $ok -ne 1 ]; then echo "[server-fail] $v"; pkill -P $SPID 2>/dev/null; kill -9 $SPID 2>/dev/null; continue; fi
  cd /home/zhangjialiang/code/RoboTwin
  $PY scripts/eval_policy_xpolicylab.py \
    --task_name open_microwave --policy_name cvwam_fm --port 8849 --protocol ws \
    --task_config demo_clean_balanced --seed 0 --test_num 5 --instruction_type seen \
    --device_id $GPU > /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/rollout_smoke_${v}.log 2>&1
  cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
  pkill -9 -P $SPID 2>/dev/null; kill -9 $SPID 2>/dev/null
  for p in $(pgrep -f "setup_policy_server.py"); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 15); do ss -tln 2>/dev/null | grep -q ':8849 ' || break; sleep 1; done
  echo "[smoke-done] $v"
done
echo "ALL_SMOKES_DONE"
