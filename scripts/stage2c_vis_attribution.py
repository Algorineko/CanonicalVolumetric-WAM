"""Stage 2c: attribute newly-observed (hit) voxels to camera motion vs scene change.

For sampled (t, t+k) pairs, per camera: hit-set at both times (voxelized),
newly-hit = hit_{t+k} \ hit_t, removed-hit = hit_t \ hit_{t+k}. Attribution:
if the camera center moved > 1cm between t and t+k the new hits are tagged
camera-motion-driven, else scene-change-driven (e.g. the microwave door
swinging open reveals its cavity to the STATIC head camera).

Outputs stage2c/metrics/vis_attribution.csv. This is the quantitative core of
the decoupling claim: visibility change has causes that occupancy change does
not (and vice versa).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import voxel as vx  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base  # noqa: E402


def hit_set_for_camera(paths, f5, t, cam, b_min, pitch, shape, T_bw):
    dmm = rta.decode_ffv1_depth_at(paths.depth_ffv1[cam], t).astype(np.float32)
    valid = dmm > 0
    dm = np.where(valid, dmm / 1000.0, 0.0)
    K = np.asarray(f5["vision/cam_head/intrinsic_matrix"][t], dtype=np.float64)
    T_bc = T_bw @ (rta_geo_T(np.asarray(f5[f"vision/{cam}/extrinsics_matrix"][t],
                                        dtype=np.float64)))
    p, _ = depth_to_base(dm, K, T_bc, valid=valid)
    idx, inb = vx.points_to_voxel_idx(p, b_min, pitch, shape)
    idx = idx[inb]
    return set(map(tuple, idx.tolist())), T_bc[:3, 3]


def rta_geo_T(cam2world_gl):
    from cvwam import geometry as geo
    return geo.cam2world_gl_to_world_cam_cv(cam2world_gl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", nargs="+",
                    default=["demo_clean_balanced:10", "demo_clean_balanced:25",
                             "demo_clean_balanced:40", "demo_randomized_balanced:0",
                             "demo_randomized_balanced:1"])
    ap.add_argument("--n-pairs", type=int, default=6)
    ap.add_argument("--offsets", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--pitch", type=float, default=0.02)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bbox = {ax: np.asarray(v, dtype=np.float64)
            for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    _, _, shape = vx.grid_config(b_min, b_max, args.pitch)

    rows = []
    for spec in args.episodes:
        variant, ep = spec.split(":")
        ep = int(ep)
        paths = rta.locate_episode(cfg["data_root"], variant, ep,
                                   task=cfg.get("task", "open_microwave"))
        with h5py.File(paths.hdf5, "r") as f:
            T = int(f["state/left_arm_joint_states"].shape[0])
            ts = np.linspace(0, T - 1 - max(args.offsets), args.n_pairs).astype(int)
            E = {cam: np.asarray(f[f"vision/{cam}/extrinsics_matrix"]) for cam in rta.CAMERAS}
        for k in args.offsets:
            for t in ts:
                for cam in rta.CAMERAS:
                    with h5py.File(paths.hdf5, "r") as f5:
                        h_t, c_t = hit_set_for_camera(paths, f5, t, cam, b_min,
                                                      args.pitch, shape, T_bw)
                        h_k, c_k = hit_set_for_camera(paths, f5, t + k, cam, b_min,
                                                      args.pitch, shape, T_bw)
                    disp = float(np.linalg.norm(c_k - c_t))
                    new = len(h_k - h_t)
                    gone = len(h_t - h_k)
                    rows.append({
                        "episode": spec, "t": int(t), "offset": k, "camera": cam,
                        "cam_disp_m": round(disp, 4),
                        "cam_disp_gt_1cm": int(disp > 0.01),
                        "newly_hit_voxels": new, "removed_hit_voxels": gone,
                        "attribution": "camera_motion" if disp > 0.01 else "scene_change",
                    })
                if t == ts[0]:
                    print(f"{spec} k={k} t={t} done", flush=True)
    outdir = Path("runs/stage2c/metrics")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "vis_attribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # summary
    import collections
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["camera"], r["attribution"])
        agg[key][0] += r["newly_hit_voxels"]
        agg[key][1] += 1
    summary = {f"{c}/{a}": {"newly_hit_total": v[0], "pairs": v[1]}
               for (c, a), v in sorted(agg.items())}
    (outdir / "vis_attribution_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
