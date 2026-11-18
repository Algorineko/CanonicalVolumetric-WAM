"""Stage 2c: workspace-bbox coverage check.

For sampled frames of each episode: unproject GT depth WITHOUT cropping and
measure (a) fraction of points inside the canonical bbox, (b) the extent of
excluded points, (c) whether the excluded mass is background (far/table
underside) vs task-relevant (microwave / grippers). Verdict text for the
report: does the 20mm bbox crop any key manipulation region?
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import geometry as geo  # noqa: E402
from cvwam.geometry import T_base_world, depth_to_base  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", nargs="+",
                    default=["demo_clean_balanced:10", "demo_clean_balanced:25",
                             "demo_clean_balanced:40", "demo_randomized_balanced:0",
                             "demo_randomized_balanced:1"])
    ap.add_argument("--n-frames", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bbox = {ax: np.asarray(v, dtype=np.float64)
            for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))

    rows = []
    for spec in args.episodes:
        variant, ep = spec.split(":")
        paths = rta.locate_episode(cfg["data_root"], variant, int(ep))
        import h5py
        with h5py.File(paths.hdf5, "r") as f:
            T = int(f["state/left_arm_joint_states"].shape[0])
            K = np.asarray(f["vision/cam_head/intrinsic_matrix"][0], dtype=np.float64)
            E = {cam: np.asarray(f[f"vision/{cam}/extrinsics_matrix"]) for cam in rta.CAMERAS}
        ts = np.linspace(0, T - 2, args.n_frames).astype(int)
        in_frac, excl_all, excl_near = [], [], []
        for t in ts:
            pts_all, inside = [], []
            for cam in rta.CAMERAS:
                dmm = rta.decode_ffv1_depth_at(paths.depth_ffv1[cam], t).astype(np.float32)
                valid = dmm > 0
                dm = np.where(valid, dmm / 1000.0, 0.0)
                T_bc = T_bw @ geo.cam2world_gl_to_world_cam_cv(E[cam][t])
                p, _ = depth_to_base(dm, K, T_bc, valid=valid)
                pts_all.append(p)
                inside.append(((p >= b_min) & (p <= b_max)).all(1))
            P = np.concatenate(pts_all)
            In = np.concatenate(inside)
            in_frac.append(float(In.mean()))
            excl = P[~In]
            if len(excl):
                # near-field excluded points = potentially task-relevant
                    # task-relevant heuristic: excluded points within 15cm OUTSIDE bbox
                d_out = np.maximum(b_min - excl, 0).max(1) + \
                    np.maximum(excl - b_max, 0).max(1)
                excl_near.append(float((d_out < 0.15).mean()))
                excl_all.append(excl)
        row = {
            "episode": spec, "frames": len(ts),
            "in_bbox_frac_mean": round(float(np.mean(in_frac)), 4),
            "excl_within_15cm_frac_mean": round(float(np.mean(excl_near)), 4)
            if excl_near else None,
        }
        if excl_all:
            Epts = np.concatenate(excl_all)
            row["excluded_xyz_p05_p995"] = np.percentile(Epts, [5, 95], axis=0).round(3).tolist()
        rows.append(row)
        print(json.dumps(row))
    outdir = Path("runs/stage2c/metrics")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "workspace_coverage.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"-> {outdir}/workspace_coverage.json")


if __name__ == "__main__":
    main()
