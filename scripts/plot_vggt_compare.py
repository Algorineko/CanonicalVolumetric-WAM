"""Render GT vs VGGT (deployable/oracle aligned) side-by-side point clouds."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam import viz  # noqa: E402
from cvwam import align as al  # noqa: E402
from eval_geometry import load_gt_frame, load_pred  # noqa: E402
import cv2  # noqa: E402

cfg = yaml.safe_load(Path("configs/smoke.yaml").read_text())
sid = sys.argv[1] if len(sys.argv) > 1 else "demo_clean_balanced/episode_0000010/t=212"
vc = "head+left_wrist+right_wrist"

gt = load_gt_frame(cfg, sid)
variant, ep, t = sid.split("/")
t = int(t.split("=")[1])

# GT cloud at native res (bbox)
pc = np.load(Path(cfg["cache_dir"]) / "pointclouds" / "gt" / variant / ep /
             f"frame_{t:07d}.npz")
G = pc["xyz_bbox"]
Gc = pc["rgb_bbox"]

for model in ["vggt_official", "vggt_omnivggt"]:
    pred, meta = load_pred(Path(cfg["heavy_root"]) / "cache" / model, sid, vc)
    wp = pred["world_points"]
    extr = pred["extrinsic"]
    views = vc.split("+")
    centers_v = np.stack([al.camera_centers_from_extrinsic(extr[i]) for i in range(len(views))])
    centers_g = np.stack([al.camera_centers_from_extrinsic(np.linalg.inv(gt["T_base_cam"][i]))
                          for i in range(len(views))])
    P = wp.reshape(-1, 3)
    fin = np.isfinite(P).all(1)
    P = P[fin]
    # oracle alignment (same as eval_geometry)
    H, W = gt["depth_m"].shape[1:]
    Hp, Wp = pred["pre_shape"]
    sx, sy = Wp / W, Hp / H
    Kgt = gt["K"].copy()
    for i in range(len(views)):
        Kgt[i, 0, 0] *= sx; Kgt[i, 0, 2] *= sx; Kgt[i, 1, 1] *= sy; Kgt[i, 1, 2] *= sy
    gv = np.stack([cv2.resize(gt["depth_valid"][i].astype(np.uint8), (Wp, Hp),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
                   for i in range(len(views))])
    gd = np.stack([cv2.resize(gt["depth_m"][i], (Wp, Hp),
                              interpolation=cv2.INTER_NEAREST) for i in range(len(views))])
    src, dst = [], []
    for i in range(len(views)):
        m = gv[i] & np.isfinite(wp[i]).all(-1)
        v, u = np.indices((Hp, Wp))
        pts = (np.stack([u[m], v[m], np.ones(m.sum())], 1)
               @ np.linalg.inv(Kgt[i]).T) * gd[i][m, None]
        dst.append(pts @ gt["T_base_cam"][i][:3, :3].T + gt["T_base_cam"][i][:3, 3])
        src.append(wp[i][m].reshape(-1, 3))
    s, R, tt = al.umeyama_sim3(np.concatenate(src), np.concatenate(dst))
    Po = (s * R @ P.T).T + tt
    rgb_flat = np.concatenate([np.repeat(i + 1, (Hp * Wp)) for i in range(len(views))])[fin]
    colors = np.stack([rgb_flat * 90, 255 - rgb_flat * 90, np.full(len(P), 120)], axis=1)
    viz.render_two_views(G, Gc, Po, colors,
                         f"runs/smoke/plots/compare_{model}.png",
                         title_a=f"GT fused {sid}",
                         title_b=f"{model} oracle-aligned (view-id colored)", elev=-60)
    print(f"saved compare_{model}.png")
print("done")
