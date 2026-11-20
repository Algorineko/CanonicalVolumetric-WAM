"""Stage 2c: cache episode-level voxel data (20mm global; 10mm optional subset).

Avoids repeated CPU ray casting at training time. For every frame of an
episode: decode 3-view depth sequentially, unproject to base frame, voxelize
with per-view ray casting (hit/free/observed/unknown/view_count/conflict),
and store qpos + action chunks + instruction in a per-episode manifest.

Storage: <heavy_root>/cache/voxels_ep/<variant>/<episode>/<pitch>mm/frame_XXXX.npz
         + manifest.jsonl per episode (frame index -> file, qpos, action_seq).
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
from cvwam import voxel as vx  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base  # noqa: E402


def cache_episode(cfg, variant: str, ep_no: int, pitch: float, b_min, b_max,
                  action_horizon: int = 16):
    paths = rta.locate_episode(cfg["data_root"], variant, ep_no,
                               task=cfg.get("task", "open_microwave"))
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    out_dir = Path(cfg["cache_dir"]) / "voxels_ep" / variant / \
        f"episode_{ep_no:07d}" / f"{int(pitch * 1000):02d}mm"
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = None
    with h5py.File(paths.hdf5, "r") as f:
        T_total = int(f["state/left_arm_joint_states"].shape[0])
        freq = int(f["additional_info/frequency"][()])
        instr = json.loads(f["instructions"][()].decode())[0]
        # pre-read all per-frame camera params (small)
        K = np.asarray(f["vision/cam_head/intrinsic_matrix"], dtype=np.float64)
        E = {cam: np.asarray(f[f"vision/{cam}/extrinsics_matrix"], dtype=np.float64)
             for cam in rta.CAMERAS}
        state_arm = {arm: np.asarray(f[f"state/{arm}_arm_joint_states"], dtype=np.float32)
                     for arm in ("left", "right")}
        state_grip = {arm: np.asarray(f[f"state/{arm}_ee_joint_states"], dtype=np.float32)
                      for arm in ("left", "right")}
        action_arm = {arm: np.asarray(f[f"action/{arm}_arm_joint_states"], dtype=np.float32)
                      for arm in ("left", "right")}
        action_grip = {arm: np.asarray(f[f"action/{arm}_ee_joint_states"], dtype=np.float32)
                       for arm in ("left", "right")}

    # per-episode table height: robust lower envelope of frame-0 head view
    dmm0 = rta.decode_ffv1_depth_at(paths.depth_ffv1["cam_head"], 0).astype(np.float32)
    val0 = dmm0 > 0
    p0, _ = depth_to_base(np.where(val0, dmm0 / 1000.0, 0.0),
                          np.asarray(K[0], dtype=np.float64),
                          T_bw @ geo_T(T_bw, E["cam_head"][0]), valid=val0)
    z0 = p0[:, 2]
    table_z_ep = float(np.median(z0[z0 < np.quantile(z0, 0.25)]))
    print(f"  per-episode table z (base) = {table_z_ep:.4f}", flush=True)

    manifest = []
    t_all = time.perf_counter()
    for t in range(T_total):
        t0 = time.perf_counter()
        depth_mm = {cam: rta.decode_ffv1_depth_window(p, t, 1)[0]
                    for cam, p in paths.depth_ffv1.items()}
        cams = []
        pts_list, valid_list = [], []
        for vi, cam in enumerate(rta.CAMERAS):
            dmm = depth_mm[cam].astype(np.float32)
            valid = dmm > 0
            depth_m = np.where(valid, dmm / 1000.0, 0.0)
            T_base_cam = T_bw @ geo_T(T_bw, E[cam][t])
            K_t = np.asarray(K[t], dtype=np.float64)  # same K all cams/frames
            p, msk = depth_to_base(depth_m, K_t, T_base_cam, valid=valid)
            cams.append(T_base_cam[:3, 3])
            pts_list.append(p)                # depth_to_base already filtered
            valid_list.append(np.ones(len(p), dtype=bool))
        if shape is None:
            _, _, shape = vx.grid_config(b_min, b_max, pitch)
        g = vx.voxelize_rays(np.stack(cams), pts_list, valid_list,
                             b_min, b_max, pitch, shape)
        qpos = np.concatenate([state_arm["left"][t], state_arm["right"][t],
                               state_grip["left"][t], state_grip["right"][t]])
        # action chunk [t .. t+H): absolute joint targets from the action group
        act_end = min(t + action_horizon, T_total)
        action_chunk = np.concatenate([
            action_arm["left"][t:act_end], action_arm["right"][t:act_end],
            action_grip["left"][t:act_end], action_grip["right"][t:act_end],
        ], axis=1)
        npz = out_dir / f"frame_{t:07d}.npz"
        np.savez_compressed(
            npz, hit=g["hit"], free=g["free"], observed=g["observed"],
            unknown=g["unknown"], view_count=g["view_count"],
            conflict=g["conflict"], qpos=qpos, action_chunk=action_chunk,
            action_valid=len(action_chunk), b_min=b_min.astype(np.float32),
            pitch=np.float32(pitch), shape=np.asarray(shape))
        manifest.append({"t": t, "npz": str(npz),
                         "ms": round((time.perf_counter() - t0) * 1000)})
        if t % 100 == 0:
            print(f"  [{variant} ep{ep_no:07d} {int(pitch*1000)}mm] {t}/{T_total} "
                  f"({(time.perf_counter()-t0)*1000:.0f} ms/frame)", flush=True)
    mf = out_dir / "manifest.json"
    mf.write_text(json.dumps({
        "variant": variant, "episode": ep_no, "pitch_m": pitch,
        "T_total": T_total, "frequency_hz": freq, "language": instr,
        "b_min": b_min.tolist(), "b_max": b_max.tolist(),
        "table_z_base": table_z_ep,
        "action_horizon": action_horizon, "frames": manifest,
        "total_s": round(time.perf_counter() - t_all, 1)}, indent=1))
    print(f"[done] {variant} ep{ep_no:07d} {int(pitch*1000)}mm: {T_total} frames "
          f"in {time.perf_counter()-t_all:.0f}s -> {out_dir}", flush=True)


def geo_T(T_bw, cam2world_gl):
    from cvwam import geometry as geo
    return geo.cam2world_gl_to_world_cam_cv(cam2world_gl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="variant:episode_no, e.g. demo_clean_balanced:10")
    ap.add_argument("--pitches", nargs="+", type=float, default=[0.02])
    ap.add_argument("--action-horizon", type=int, default=16)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bbox = {ax: np.asarray(v, dtype=np.float64)
            for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])
    for spec in args.episodes:
        variant, ep = spec.split(":")
        for pitch in args.pitches:
            cache_episode(cfg, variant, int(ep), pitch, b_min, b_max,
                          args.action_horizon)


if __name__ == "__main__":
    main()
