#!/bin/bash
# CV-WAM stage4b P0: parameterized stress-test chain (one condition per call).
# Usage: run_stress_chain.sh <COND> <PORT> <GPU> <TRIALS> [SEED]
#   COND ∈ baseline|cam_t005|cam_t015|cam_t030|cam_r7.5|cam_r20|
#          drop1of3|drop_head|occ|lo
# Parallel-safe: per-chain deploy yml (runs/stage4b/deploy_p$PORT.yml),
# own port, cleanup kills ONLY its own server process group (no global pkill).
# Server-key conditions (drop*/occ) abort unless the server log confirms the
# stress config ([cvwam-stress-server] line) — no silent no-ops.
set -u
cd /home/zhangjialiang/code/CanonicalVolumetric-WAM || exit 1
COND=${1:?COND}; PORT=${2:?PORT}; GPU=${3:?GPU}; TRIALS=${4:?TRIALS}
SEED=${5:-0}
TASK=open_microwave
CLIENT_ENV=()
SRV_KEYS=""

case "$COND" in
  baseline)  ;;
  cam_t005)  CLIENT_ENV=("CVWAM_HEAD_CAM_SHIFT_M=0.05") ;;
  cam_t015)  CLIENT_ENV=("CVWAM_HEAD_CAM_SHIFT_M=0.15") ;;
  cam_t030)  CLIENT_ENV=("CVWAM_HEAD_CAM_SHIFT_M=0.30") ;;
  cam_r7.5)  CLIENT_ENV=("CVWAM_HEAD_CAM_ROT_DEG=7.5") ;;
  cam_r20)   CLIENT_ENV=("CVWAM_HEAD_CAM_ROT_DEG=20") ;;
  drop1of3)  SRV_KEYS=$'\nstress:\n  drop_cam: one_of_three' ;;
  drop_head) SRV_KEYS=$'\nstress:\n  drop_cam: cam_head' ;;
  occ)       SRV_KEYS=$'\nstress:\n  occ_box:\n    b_min: [0.38, -0.08, 1.03]\n    b_max: [0.54, 0.08, 1.19]\n    cams: [cam_head]' ;;
  lo)        TASK=open_microwave_lo ;;
  *) echo "[abort] unknown COND $COND"; exit 2 ;;
esac

LOCK=/tmp/cvwam_stress_${PORT}.lock
if mkdir "$LOCK" 2>/dev/null; then trap 'rmdir "$LOCK"' EXIT; else echo "[guard] port $PORT lock held"; exit 1; fi

if ss -tln 2>/dev/null | grep -q ":$PORT "; then echo "[abort] port $PORT in use"; exit 2; fi

DEPLOY=runs/stage4b/deploy_p${PORT}.yml
cat > "$DEPLOY" <<EOF
policy_name: cvwam_fm
protocol: ws
host: localhost
port: null
bench_name: null
task_name: $TASK
ckpt_name: runs/stage4a/a0_s42/best.pt
env_cfg_type: demo_clean_balanced
seed: 0
action_type: joint
gpu_id: 0
eval_batch: false
ckpt_path: /home/zhangjialiang/code/CanonicalVolumetric-WAM/runs/stage4a/a0_s42/best.pt
exec_horizon: 4
n_euler: 8${SRV_KEYS}
EOF

LOG=runs/stage4b/policy_server_${COND}_p${PORT}.log
source /home/zhangjialiang/miniconda3/etc/profile.d/conda.sh
conda activate openwam
setsid nohup env CUDA_VISIBLE_DEVICES=$GPU PYTHONWARNINGS=ignore::UserWarning \
  python /home/zhangjialiang/code/RoboTwin/XPolicyLab/setup_policy_server.py \
    --config_path "$(pwd)/$DEPLOY" \
    --overrides port=$PORT host=localhost bench_name=RoboTwin task_name=$TASK \
      ckpt_name=runs/stage4a/a0_s42/best.pt env_cfg_type=demo_clean_balanced \
      seed=$SEED policy_name=cvwam_fm action_type=joint > "$LOG" 2>&1 &
SPID=$!
PGID=$(ps -o pgid= -p $SPID 2>/dev/null | tr -d ' ')

ok=0
for i in $(seq 1 90); do
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then ok=1; break; fi
  if ! kill -0 $SPID 2>/dev/null; then break; fi
  sleep 2
done
if [ $ok -ne 1 ] || [ -z "$PGID" ]; then
  echo "[server-fail] $COND port $PORT"; kill -9 -$PGID 2>/dev/null; exit 3
fi
# ckpt verification (port-incident lesson): server must print the expected ckpt
sleep 1
CKPT_LINE=$(grep -m1 "ckpt=" "$LOG" || true)
case "$CKPT_LINE" in
  *a0_s42/best.pt*) ;;
  *) echo "[abort] ckpt mismatch: '$CKPT_LINE'"; kill -9 -$PGID 2>/dev/null; exit 4 ;;
esac
if [ -n "$SRV_KEYS" ] && ! grep -q "\[cvwam-stress-server\]" "$LOG"; then
  echo "[abort] server ignored stress keys (wrapper lacks support?)"; kill -9 -$PGID 2>/dev/null; exit 5
fi
echo "[chain] $COND port=$PORT trials=$TRIALS seed=$SEED | $CKPT_LINE"

ROBOTWIN_PY=/home/zhangjialiang/miniconda3/envs/robottwin/bin/python
env "${CLIENT_ENV[@]}" "$ROBOTWIN_PY" /home/zhangjialiang/code/RoboTwin/scripts/eval_policy_xpolicylab.py \
  --task_name $TASK --policy_name cvwam_fm --port $PORT --protocol ws \
  --task_config demo_clean_balanced --seed $SEED --test_num $TRIALS \
  --instruction_type seen --device_id 0 \
  > runs/stage4b/rollout_${COND}_p${PORT}.log 2>&1

kill -TERM -$PGID 2>/dev/null; sleep 3; kill -KILL -$PGID 2>/dev/null
for i in $(seq 1 15); do ss -tln 2>/dev/null | grep -q ":$PORT " || break; sleep 1; done

grep -a "Final success rate" runs/stage4b/rollout_${COND}_p${PORT}.log | tail -1 \
  > runs/stage4b/metrics/stress/${COND}_rates.txt
grep -ao "current seed: [0-9]*" runs/stage4b/rollout_${COND}_p${PORT}.log \
  | awk '{print $3}' > runs/stage4b/metrics/stress/${COND}_seeds.txt
cp runs/stage4b/deploy_p${PORT}.yml runs/stage4b/metrics/stress/${COND}_deploy_used.yml
echo "[chain-done] $COND: $(cat runs/stage4b/metrics/stress/${COND}_rates.txt)"
