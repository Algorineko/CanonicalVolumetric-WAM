"""Stage 2c/3a: build camera-dropout input voxels for eval frames.

Subsets: head_only, head_left, head_right (the full 3-view "all" is the main
cache). Targets stay the full-3-view GT from the main episode cache — hidden
viewpoints are never used as INPUT for their own hidden-view experiments; the
full observation only defines the shared supervision/target region.
"""

import argparse
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

SUBSETS = {
    "head_only": ["cam_head"],
    "head_left": ["cam_head", "cam_left_wrist"],
    "head_right": ["cam_head", "cam_right_wrist"],
}


def build_subset(cfg, variant, ep, t, subset_name, pitch, b_min, b_max, T_bw):
    paths = rta.locate_episode(cfg["data_root"], variant, ep,
                               task=cfg.get("task", "open_microwave"))
    cams = SUBSETS[subset_name]
    with h5py.File(paths.hdf5, "r") as f:
        K = np.asarray(f["vision/cam_head/intrinsic_matrix"][t], dtype=np.float64)
        E = {cam: np.asarray(f[f"vision/{cam}/extrinsics_matrix"][t],
                             dtype=np.float64) for cam in cams}
    centers, pts_list, valid_list = [], [], []
    for cam in cams:
        dmm = rta.decode_ffv1_depth_at(paths.depth_ffv1[cam], t).astype(np.float32)
        valid = dmm > 0
        dm = np.where(valid, dmm / 1000.0, 0.0)
        T_bc = T_bw @ (rta_geo_T(E[cam]))
        p, _ = depth_to_base(dm, K, T_bc, valid=valid)
        centers.append(T_bc[:3, 3])
        pts_list.append(p)
        valid_list.append(np.ones(len(p), dtype=bool))
    _, _, shape = vx.grid_config(b_min, b_max, pitch)
    g = vx.voxelize_rays(np.stack(centers), pts_list, valid_list,
                         b_min, b_max, pitch, shape)
    return g, shape


def rta_geo_T(cam2world_gl):
    from cvwam import geometry as geo
    return geo.cam2world_gl_to_world_cam_cv(cam2world_gl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", nargs="+",
                    default=["demo_randomized_balanced:0", "demo_randomized_balanced:1"])
    ap.add_argument("--t-stride", type=int, default=8)
    ap.add_argument("--pitch", type=float, default=0.02)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    bbox = {ax: np.asarray(v, dtype=np.float64)
            for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    for spec in args.episodes:
        variant, ep = spec.split(":")
        ep = int(ep)
        mf = json.loads((Path(cfg["cache_dir"]) / "voxels_ep" / variant /
                         f"episode_{ep:07d}" / "20mm" / "manifest.json").read_text())
        ts = list(range(0, mf["T_total"] - 17, args.t_stride))
        for sub in SUBSETS:
            outdir = Path(cfg["cache_dir"]) / "voxels_ep_dropout" / sub / variant / \
                f"episode_{ep:07d}" / "20mm"
            outdir.mkdir(parents=True, exist_ok=True)
            for t in ts:
                g, shape = build_subset(cfg, variant, ep, t, sub, args.pitch,
                                        b_min, b_max, T_bw)
                np.savez_compressed(outdir / f"frame_{t:07d}.npz",
                                    hit=g["hit"], free=g["free"],
                                    observed=g["observed"], unknown=g["unknown"],
                                    view_count=g["view_count"], b_min=b_min.astype(np.float32),
                                    pitch=np.float32(args.pitch), shape=np.asarray(shape))
            print(f"{spec} {sub}: {len(ts)} frames -> {outdir}")


if __name__ == "__main__":
    main()
