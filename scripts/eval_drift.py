"""Static-background drift of VGGT reconstructions across timesteps (§6.5).

For one episode, take consecutive sampled timesteps; for each, reconstruct,
align (deployable and/or oracle), select the static table region via GT depth
(z within [table_z, table_z+1.5cm] in base), and measure the NN drift between
consecutive reconstructions restricted to that region.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import align as al  # noqa: E402


def aligned_points(cfg, sid, vc, model_cache, mode):
    variant, ep, t = sid.split("/")
    t = int(t.split("=")[1])
    gt = np.load(Path(cfg["cache_dir"]) / "frames" / variant / ep / f"frame_{t:07d}.npz")
    pr = np.load(model_cache / variant / ep / f"t={t}" / vc.replace("+", "_") / "geometry.npz")
    import cv2
    H, W = gt["depth_m"].shape[1:]
    Hp, Wp = pr["pre_shape"]
    wp = pr["world_points"]
    extr = pr["extrinsic"]
    views = vc.split("+")
    centers_v = np.stack([al.camera_centers_from_extrinsic(extr[i]) for i in range(len(views))])
    centers_g = np.stack([al.camera_centers_from_extrinsic(gt["T_base_cam"][i]) for i in range(len(views))])
    if mode == "oracle":
        # pixel correspondences against GT base points (coarse: use hi-res GT)
        sx, sy = Wp / W, Hp / H
        Kgt = gt["K"].copy()
        for i in range(len(views)):
            Kgt[i, 0, 0] *= sx; Kgt[i, 0, 2] *= sx; Kgt[i, 1, 1] *= sy; Kgt[i, 1, 2] *= sy
        src, dst = [], []
        for i in range(len(views)):
            gv = cv2.resize(gt["depth_valid"][i].astype(np.uint8), (Wp, Hp),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            gd = cv2.resize(gt["depth_m"][i], (Wp, Hp), interpolation=cv2.INTER_NEAREST)
            m = gv & np.isfinite(wp[i]).all(-1)
            v, u = np.indices((Hp, Wp))
            pts = (np.stack([u[m], v[m], np.ones(m.sum())], 1)
                   @ np.linalg.inv(Kgt[i]).T) * gd[m, None]
            dst.append(pts @ gt["T_base_cam"][i][:3, :3].T + gt["T_base_cam"][i][:3, 3])
            src.append(wp[i][m].reshape(-1, 3))
        s, R, tt = al.umeyama_sim3(np.concatenate(src), np.concatenate(dst))
        return {i: (s * R @ wp[i].reshape(-1, 3).T).T + tt for i in range(len(views))}
    elif mode == "head_anchor":
        # deployable, no wrist centers: fix the (static) head camera to its
        # known GT pose, scale about the camera center using the known
        # camera-to-table height via the robust lower envelope of z.
        T_wc0 = np.eye(4)
        T_wc0[:3, :3] = extr[0][:3, :3].T          # cam0 -> VGGT world (cam-from-world)
        T_wc0[:3, 3] = -extr[0][:3, :3].T @ extr[0][:3, 3]
        R = gt["T_base_cam"][0][:3, :3] @ T_wc0[:3, :3]
        c_true = gt["T_base_cam"][0][:3, 3]
        P0 = wp[0].reshape(-1, 3)
        P0 = P0[np.isfinite(P0).all(-1)]
        z_camrel = (P0 @ R.T)[:, 2]
        bottom = z_camrel[z_camrel < np.quantile(z_camrel, 0.25)]
        dist_anch = abs(float(np.median(bottom)))   # camera-to-table, VGGT units
        s = abs(1.35 - 0.7401) / dist_anch
        return {i: c_true + s * (wp[i].reshape(-1, 3) @ R.T) for i in range(len(views))}
    else:
        s, R, tt, deg = al.fit_similarity_on_centers(centers_v, centers_g)
        assert s is not None and not deg, "centers degenerate"
        return {i: (s * R @ wp[i].reshape(-1, 3).T).T + tt for i in range(len(views))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--model", default="vggt_official")
    ap.add_argument("--vc", default="head+left_wrist+right_wrist")
    ap.add_argument("--episodes", nargs="+",
                    default=["demo_clean_balanced/episode_0000010"])
    ap.add_argument("--times", nargs="+", type=int, required=True,
                    help="sorted timesteps of the same episode")
    ap.add_argument("--align", default="deployable", choices=["deployable", "oracle", "head_anchor"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_cache = Path(cfg["heavy_root"]) / "cache" / args.model
    table_z = cfg["table_z_base"]
    rows = []
    for ep in args.episodes:
        prev = None
        for t in args.times:
            sid = f"{ep}/t={t}"
            pts = aligned_points(cfg, sid, args.vc, model_cache, args.align)
            variant, e, _ = sid.split("/")
            gt = np.load(Path(cfg["cache_dir"]) / "frames" / variant / e / f"frame_{t:07d}.npz")
            # static table region: GT depth says the pixel hits near-table height
            P = np.concatenate([pts[i].reshape(-1, 3) for i in range(len(pts))])
            # restrict to the static table band (VGGT points near known table height)
            Pb = P[(P[:, 2] > table_z - 0.01) & (P[:, 2] < table_z + 0.02)]
            if prev is not None and len(Pb) and len(prev):
                d, _ = cKDTree(prev).query(Pb[:100_000], k=1, workers=-1)
                rows.append({"episode": ep, "t": t, "align": args.align,
                             "drift_median_mm": round(float(np.median(d) * 1000), 2),
                             "drift_mean_mm": round(float(d.mean() * 1000), 2),
                             "drift_p95_mm": round(float(np.percentile(d, 95) * 1000), 2),
                             "n_table_pts": int(len(Pb))})
                print(rows[-1])
            prev = Pb

    out = Path("runs/smoke/metrics/static_drift.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
