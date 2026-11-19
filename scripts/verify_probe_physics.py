"""Observer probe verification: does --head-cam-dis change ONLY the camera?

Compares probe replay (probe_ep10_camdis05) against the original replay
(clean_10): robot qpos, door angle, microwave pose must match (physics
determinism); head camera extrinsics must DIFFER (observer intervention).
"""

import json
import sys
from pathlib import Path

import numpy as np

A = Path("/data/canonical-volumetric-wam/cache/replay_gt/clean_10/replay_gt.json")
B = Path("/data/canonical-volumetric-wam/cache/replay_gt/probe_ep10_camdis05/replay_gt.json")
OUT = Path("runs/stage3b/metrics/probe_verification.json")

ra = json.loads(A.read_text())
rb = json.loads(B.read_text())
n = min(len(ra), len(rb))
print(f"frames: orig {len(ra)}, probe {len(rb)}")

qpos_d, door_d, mic_d, head_d = [], [], [], []
for i in range(n):
    a, b = ra[i], rb[i]
    qa = np.asarray(a["robot_qpos_actual"])
    qb = np.asarray(b["robot_qpos_actual"])
    qpos_d.append(float(np.abs(qa - qb).max()))
    door_d.append(abs(float(a["door_angle"]) - float(b["door_angle"])))
    mic_d.append(float(np.linalg.norm(np.asarray(a["mic_p"]) - np.asarray(b["mic_p"]))))
    Ta = np.asarray(a["T_world_cam_cv_head_camera"])
    Tb = np.asarray(b["T_world_cam_cv_head_camera"])
    head_d.append(float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])))

res = dict(
    frames_compared=n,
    qpos_max_diff=np.max(qpos_d), qpos_mean_diff=float(np.mean(qpos_d)),
    door_angle_max_diff=float(np.max(door_d)),
    mic_pos_max_diff_m=float(np.max(mic_d)),
    head_cam_center_diff_mean_m=float(np.mean(head_d)),
    head_cam_center_diff_min_m=float(np.min(head_d)),
    verdict_physics=("IDENTICAL" if np.max(qpos_d) < 1e-6 and np.max(mic_d) < 1e-6
                     else "DIVERGED"),
    verdict_camera=("MOVED" if np.min(head_d) > 0.01 else "UNCHANGED"),
)
print(json.dumps(res, indent=2))
OUT.write_text(json.dumps(res, indent=2))
