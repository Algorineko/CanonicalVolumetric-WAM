"""Stage 3c §3: representation-comparison cache.

Stores per frame: decoded 3-view depth (uint16 mm, 0=invalid) + per-camera
T_base_cam + K + qpos + action chunk. P2 (fused 4096-pt cloud), P2-E (+free
space ray samples) and P3 (per-view pointmap) are BUILT AT TRAIN/EVAL TIME
from this cache — no representation-specific redundancy on disk.

Output: <heavy_root>/cache/point_ep/<variant>/episode_N/frame_XXXXXX.npz
Train episodes stride 1; val/test stride 2 (matches pareto test stride).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base  # noqa: E402


def geo_T(T_bw, cam2world_gl):
    """cam2world_gl -> world_cam_cv (verified chain, same as voxel cache)."""
    from cvwam import geometry as geo
    return geo.cam2world_gl_to_world_cam_cv(cam2world_gl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sm = json.loads(Path(args.split_manifest).read_text())
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    out_root = Path(cfg["cache_dir"]) / "point_ep"

    jobs = []
    for name, eps, stride in (("train", sm["splits"]["clean_train"], 1),
                              ("eval", sm["splits"]["clean_val"] +
                               sm["splits"]["rand_val"] +
                               sm["splits"]["clean_test"] +
                               sm["splits"]["rand_test"], 2)):
        for spec in eps:
            jobs.append((spec, stride))
    print(f"{len(jobs)} episodes", flush=True)

    for spec, stride in jobs:
        variant, ep = spec.split(":")
        ep = int(ep)
        out_dir = out_root / variant / f"episode_{ep:07d}"
        done_flag = out_dir / "DONE"
        if done_flag.exists():
            print(f"[skip] {spec}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = rta.locate_episode(cfg["data_root"], variant, ep,
                                   task=cfg.get("task", "open_microwave"))
        t0 = time.perf_counter()
        with h5py.File(paths.hdf5, "r") as f:
            T_total = int(f["state/left_arm_joint_states"].shape[0])
            K = np.asarray(f["vision/cam_head/intrinsic_matrix"], dtype=np.float64)
            E = {cam: np.asarray(f[f"vision/{cam}/extrinsics_matrix"], dtype=np.float64)
                 for cam in rta.CAMERAS}
            state_arm = {a: np.asarray(f[f"state/{a}_arm_joint_states"], dtype=np.float32)
                         for a in ("left", "right")}
            state_grip = {a: np.asarray(f[f"state/{a}_ee_joint_states"], dtype=np.float32)
                          for a in ("left", "right")}
            action_arm = {a: np.asarray(f[f"action/{a}_arm_joint_states"], dtype=np.float32)
                          for a in ("left", "right")}
            action_grip = {a: np.asarray(f[f"action/{a}_ee_joint_states"], dtype=np.float32)
                           for a in ("left", "right")}
        n = 0
        for t in range(0, T_total - 17, stride):
            depth_mm = {cam: rta.decode_ffv1_depth_window(p, t, 1)[0]
                        for cam, p in paths.depth_ffv1.items()}
            Tbc = {}
            for cam in rta.CAMERAS:
                # verified chain: T_base_cam = T_bw @ world_cam_cv(cam2world_gl)
                Tbc[cam] = T_bw @ geo_T(T_bw, E[cam][t].astype(np.float64))
            act_end = min(t + 16, T_total - 1)
            chunk = np.concatenate([
                action_arm["left"][t:act_end], action_arm["right"][t:act_end],
                action_grip["left"][t:act_end], action_grip["right"][t:act_end]],
                axis=1).astype(np.float32)
            qpos = np.concatenate([state_arm["left"][t], state_arm["right"][t],
                                   state_grip["left"][t], state_grip["right"][t]])
            np.savez_compressed(
                out_dir / f"frame_{t:06d}.npz",
                depth=np.stack([depth_mm[c] for c in rta.CAMERAS]).astype(np.uint16),
                cams=list(rta.CAMERAS),
                T_base_cam=np.stack([Tbc[c][:3, :4] for c in rta.CAMERAS]),
                K=np.asarray(K[0], dtype=np.float64),
                qpos=qpos.astype(np.float32),
                action_chunk=chunk, action_valid=len(chunk))
            n += 1
        done_flag.write_text(json.dumps({"spec": spec, "stride": stride,
                                         "frames": n, "sec": round(time.perf_counter() - t0, 1)}))
        print(f"[done] {spec}: {n} frames ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
