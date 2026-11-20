#!/bin/bash
# Closed-loop sanity after FM integration sign fix (2026-11-20).
# Usage: VARIANTS="a0 a1" bash runs/stage4a/run_sanity_sign_fix.sh
#   then: VARIANTS="a2 a3 a3mask" bash runs/stage4a/run_sanity_sign_fix.sh
# Logs suffixed _signfix; results in rollout_results/${v}_smoke5_rates_signfix.txt.
# Expect after fix: a0 ~5/5 (matches valid a0 smoke5), a1 settles redo validity
# (a1 redo 5/5 ran 16:06-16:18, before the 16:24 world-branch edit; if a1 ~5/5
# under correct sign, the act '+=' flip happened at 16:24 and redo was valid).
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM || exit 1
LOCK=/tmp/cvwam_sanity_signfix.lock
if mkdir "$LOCK" 2>/dev/null; then trap 'rmdir "$LOCK"' EXIT; else echo "[guard] lock held"; exit 1; fi
DEPLOY=/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/deploy.yml
SRV=/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/setup_eval_policy_server.sh
PY=/home/zhangjialiang/miniconda3/envs/robottwin/bin/python
GPU=${GPU:-1}
PORT=${PORT:-8849}
SAMPLER=${SAMPLER:-euler_minus}
VARIANTS=${VARIANTS:-"a0 a1"}
for v in $VARIANTS; do
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then echo "[abort] port $PORT in use before $v"; exit 2; fi
  sed -i "s|ckpt_path:.*|ckpt_path: /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/${v}_s42/best.pt|" $DEPLOY
  sed -i "/^sampler:/d" $DEPLOY
  sed -i "s|^n_euler: 8|n_euler: 8\nsampler: ${SAMPLER}|" $DEPLOY
  grep -n "ckpt_path\|^sampler" $DEPLOY > runs/stage4a/smoke_ckpt_${v}_${SAMPLER}.txt
  setsid nohup bash $SRV RoboTwin open_microwave runs/stage4a/${v}_s42/best.pt \
    demo_clean_balanced joint 0 $GPU openwam $PORT localhost \
    > runs/stage4a/policy_server_${v}_${SAMPLER}.log 2>&1 &
  SPID=$!
  ok=0
  for i in $(seq 1 90); do
    if ss -tln 2>/dev/null | grep -q ":$PORT "; then ok=1; break; fi
    if ! kill -0 $SPID 2>/dev/null; then break; fi
    sleep 2
  done
  if [ $ok -ne 1 ]; then echo "[server-fail] $v"; pkill -P $SPID 2>/dev/null; kill -9 $SPID 2>/dev/null; continue; fi
  cd /home/zhangjialiang/code/RoboTwin
  $PY scripts/eval_policy_xpolicylab.py \
    --task_name open_microwave --policy_name cvwam_fm --port $PORT --protocol ws \
    --task_config demo_clean_balanced --seed 0 --test_num 5 --instruction_type seen \
    --device_id $GPU > /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/rollout_smoke_${v}_${SAMPLER}.log 2>&1
  cd /home/zhangjialiang/code/CanonicalVolumetric-WAM
  pkill -9 -P $SPID 2>/dev/null; kill -9 $SPID 2>/dev/null
  for p in $(pgrep -f "setup_policy_server.py"); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 15); do ss -tln 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done
  grep -a "Final success rate" runs/stage4a/rollout_smoke_${v}_${SAMPLER}.log \
    | tail -1 > runs/stage4a/rollout_results/${v}_smoke5_rates_${SAMPLER}.txt
  echo "[sanity-done] $v: $(cat runs/stage4a/rollout_results/${v}_smoke5_rates_${SAMPLER}.txt)"
done
echo "SANITY_ALL_DONE"
