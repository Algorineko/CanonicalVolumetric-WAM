"""Evaluate VGGT-family reconstructions against GT geometry (plan.md §6.4-6.5).

Alignment modes, strictly separated:
- oracle      : Umeyama Sim(3) on pixel-aligned GT-depth correspondences
                (diagnostic upper bound only, never a deployment claim)
- deployable  : G2: similarity fit on known camera centers (from calibration)
                G1: anchor VGGT world to the known GT camera-0 pose, scale from
                the known camera-height/table-distance constant via the
                dominant plane (RANSAC). No GT depth used.
- g3          : VGGT per-view depth rescaled per deployable anchor, unprojected
                with GT K/T (isolates depth error from camera-estimation error)

Outputs geometry_metrics.csv rows per (sample, model, view_config, align_mode,
route) + PNG comparisons.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import align as al  # noqa: E402

VIEW_ORDER = ["head", "left_wrist", "right_wrist"]


def load_gt_frame(cfg, sid):
    variant, ep, t = sid.split("/")
    frames = Path(cfg["cache_dir"]) / "frames" / variant / ep
    d = np.load(frames / f"frame_{t.split('=')[1].zfill(7)}.npz")
    return d


def load_pred(model_cache: Path, sid, vc):
    variant, ep, t = sid.split("/")
    p = model_cache / variant / ep / f"t={t.split('=')[1]}" / vc.replace("+", "_")
    d = np.load(p / "geometry.npz")
    meta = json.loads((p / "meta.json").read_text())
    return d, meta


def scaled_K(K: np.ndarray, s_x: float, s_y: float) -> np.ndarray:
    Ks = K.copy()
    Ks[0, 0] *= s_x
    Ks[0, 2] *= s_x
    Ks[1, 1] *= s_y
    Ks[1, 2] *= s_y
    return Ks


def unproject(depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray):
    """depth (H,W) + K + camera->base transform -> base-frame points + valid.

    T_base_cam maps camera->base (p_base = R p_cam + t), so
    p_base = p_cam @ R.T + t in row-vector form. Pass np.eye(4) to keep
    camera-frame points.
    """
    h, w = depth.shape
    v, u = np.indices((h, w), dtype=np.float64)
    m = np.isfinite(depth) & (depth > 0)
    uv1 = np.stack([u[m], v[m], np.ones(m.sum())], axis=1)
    p_cam = (uv1 @ np.linalg.inv(K).T) * depth[m, None]
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    p_base = p_cam @ R.T + t
    return p_base, m


def rot_err_deg(R_a, R_b):
    c = (np.trace(R_a @ R_b.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def evaluate_one(cfg, sid, vc, model_cache, model_tag, table_z=0.7401,
                 cam_h=1.35, max_pairs=200_000):
    gt = load_gt_frame(cfg, sid)
    pred, meta = load_pred(model_cache, sid, vc)
    views = vc.split("+")
    vidx = [VIEW_ORDER.index(v) for v in views]

    H, W = gt["depth_m"].shape[1:]          # 240x320
    Hp, Wp = pred["pre_shape"]              # e.g. 392x518
    s_x, s_y = Wp / W, Hp / H
    depth_v = pred["depth"][..., 0]         # (S,Hp,Wp)
    dconf = pred["depth_conf"]
    extr = pred["extrinsic"]                # (S,3,4) cam-from-VGGT-world
    intr = pred["intrinsic"]
    wp = pred["world_points"]               # (S,Hp,Wp,3)
    wconf = pred.get("world_points_conf")
    if wconf is None:
        wconf = dconf

    K_gt = gt["K"][vidx]                    # (S,3,3) at 240x320
    T_bc = gt["T_base_cam"][vidx]           # (S,4,4)
    gt_depth_lo = gt["depth_m"][vidx]       # (S,240,320)
    gt_valid_lo = gt["depth_valid"][vidx]
    bbox_min = np.asarray([cfg["workspace_bbox_base"][k][0] for k in "xyz"])
    bbox_max = np.asarray([cfg["workspace_bbox_base"][k][1] for k in "xyz"])

    # GT depth/points at the preprocessed resolution (nearest keeps validity)
    gt_depth_hi = np.stack([np.asarray(
        __import__("cv2").resize(gt_depth_lo[i], (Wp, Hp),
                                 interpolation=cv2_NEAREST)) for i in range(len(views))])
    gt_valid_hi = np.stack([np.asarray(
        __import__("cv2").resize(gt_valid_lo[i].astype(np.uint8), (Wp, Hp),
                                 interpolation=cv2_NEAREST)).astype(bool)
        for i in range(len(views))])
    K_gt_hi = np.stack([scaled_K(K_gt[i], s_x, s_y) for i in range(len(views))])
    T_cb = np.stack([np.linalg.inv(T_bc[i]) for i in range(len(views))])  # base->cam
    gt_pts_hi = []
    for i in range(len(views)):
        p, m = unproject(gt_depth_hi[i], K_gt_hi[i], T_bc[i])
        pts = np.zeros((Hp, Wp, 3))
        pts[m] = p
        pts[~gt_valid_hi[i]] = np.nan
        gt_pts_hi.append(pts)
    gt_pts_hi = np.stack(gt_pts_hi)         # (S,Hp,Wp,3)

    rows = []

    def add_row(route, align_mode, pts_base, conf, extra=None):
        """Compute metrics for one aligned point map (S,Hp,Wp,3)."""
        finite = np.isfinite(pts_base).all(-1)
        mask = finite & (conf > 0)
        P = pts_base[mask]
        if len(P) < 100:
            row = dict(route=route, align_mode=align_mode, n_points=int(mask.sum()),
                       error="too_few_points")
            rows.append({**base_row, **row})
            return
        inb = ((P >= bbox_min) & (P <= bbox_max)).all(1)
        Pw = P[inb]
        # GT reference cloud for NN: all views' valid points, in-bbox
        gmask = gt_valid_hi & np.isfinite(gt_pts_hi).all(-1)
        G = gt_pts_hi[gmask]
        Gin = ((G >= bbox_min) & (G <= bbox_max)).all(1)
        G = G[Gin]
        if len(G) == 0 or len(Pw) == 0:
            rows.append({**base_row, **dict(route=route, align_mode=align_mode,
                                            n_points=int(mask.sum()),
                                            n_in_bbox=int(len(Pw)),
                                            n_gt_in_bbox=int(len(G)),
                                            error="empty_point_set")})
            return
        # depth comparison per view: transform aligned points into each cam
        depth_rows = []
        for i, vname in enumerate(views):
            pc = (Pw - T_bc[i][:3, 3]) @ T_bc[i][:3, :3]
            z = pc[:, 2]
            # map points back to pixels for per-pixel depth error
            uv = (K_gt_hi[i] @ pc.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            ui, vi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
            in_img = (ui >= 0) & (ui < Wp) & (vi >= 0) & (vi < Hp)
            okm = np.zeros(len(Pw), dtype=bool)
            okm[in_img] = gt_valid_hi[i][vi[in_img], ui[in_img]]
            pz = np.full(gt_depth_hi[i].shape, np.nan)
            pz[vi[okm], ui[okm]] = z[okm]
            de = al.depth_err_stats(pz, gt_depth_hi[i], gt_valid_hi[i])
            depth_rows.append({f"depth_{vname}_{k}": v for k, v in de.items()})
        gt_tree = cKDTree(G)
        acc = al.nn_metrics(Pw, gt_tree)
        comp = al.nn_metrics(G, cKDTree(Pw))
        row = dict(
            route=route, align_mode=align_mode, n_points=int(mask.sum()),
            n_in_bbox=int(len(Pw)),
            **{f"acc_{k}": v for k, v in acc.items()},
            **{f"compl_{k}": v for k, v in comp.items()},
            **{k: v for dr in depth_rows for k, v in dr.items()},
        )
        # camera centers + rotation errors (needs alignment transform)
        if extra and "centers_err_mm" in extra:
            row.update(extra["centers_err_mm"])
        if extra and "rot_err_deg" in extra:
            row.update(extra["rot_err_deg"])
        # confidence quantile coverage (accuracy of top-q% by conf)
        cq = {}
        cflat = conf[mask][inb]
        if len(cflat) and len(G):
            for q in (0.5, 0.75, 0.9):
                thr = np.quantile(cflat, 1 - q)
                sel = cflat >= thr
                m = al.nn_metrics(Pw[sel], gt_tree)
                cq[f"acc_top{int(q * 100)}_median_mm"] = m["median_mm"]
        row.update(cq)
        rows.append({**base_row, **row})

    base_row = dict(sample_id=sid, model=model_tag, view_config=vc)

    # ---------- oracle Sim(3) on GT-depth pixel correspondences ----------
    src, dst = [], []
    for i in range(len(views)):
        m = gt_valid_hi[i] & np.isfinite(wp[i]).all(-1) & np.isfinite(gt_pts_hi[i]).all(-1)
        a = wp[i][m].reshape(-1, 3)
        b = gt_pts_hi[i][m].reshape(-1, 3)
        if len(a) > max_pairs:
            sel = np.random.default_rng(0).choice(len(a), max_pairs, replace=False)
            a, b = a[sel], b[sel]
        src.append(a)
        dst.append(b)
    src, dst = np.concatenate(src), np.concatenate(dst)
    s_o, R_o, t_o = al.umeyama_sim3(src, dst)
    wp_base_o = np.stack([al.apply_sim3(wp[i].reshape(-1, 3), s_o, R_o, t_o)
                          .reshape(Hp, Wp, 3) for i in range(len(views))])
    # camera pose errors under oracle alignment
    centers_vggt = np.stack([al.camera_centers_from_extrinsic(extr[i]) for i in range(len(views))])
    centers_gt = np.stack([al.camera_centers_from_extrinsic(T_cb[i]) for i in range(len(views))])
    centers_aligned = al.apply_sim3(centers_vggt, s_o, R_o, t_o)
    cerr = np.linalg.norm(centers_aligned - centers_gt, axis=1) * 1000
    rerr = []
    for i in range(len(views)):
        R_wc_vggt = extr[i][:3, :3].T           # world->cam -> cam->world
        R_wc_gt = T_bc[i][:3, :3]
        rerr.append(rot_err_deg(R_o @ R_wc_vggt, R_wc_gt))
    add_row("world_points", "oracle", wp_base_o, wconf, extra=dict(
        centers_err_mm={f"cam_center_err_{views[i]}_mm": float(cerr[i])
                        for i in range(len(views))},
        rot_err_deg={f"cam_rot_err_{views[i]}_deg": float(rerr[i])
                     for i in range(len(views))}))
    # route B: VGGT depth unprojection (official-recommended)
    ptsB = []
    for i in range(len(views)):
        p, m = unproject(depth_v[i], intr[i], np.eye(4))
        pts = np.zeros((Hp, Wp, 3))
        pts[m] = p
        pts[~m] = np.nan
        ptsB.append(pts)
    ptsB = np.stack(ptsB)
    ptsB_base = np.stack([al.apply_sim3(ptsB[i].reshape(-1, 3), s_o, R_o, t_o)
                          .reshape(Hp, Wp, 3) for i in range(len(views))])
    add_row("depth_unproject", "oracle", ptsB_base, dconf)

    # ---------- deployable ----------
    if len(views) >= 3:
        s_d, R_d, t_d, degen = al.fit_similarity_on_centers(centers_vggt, centers_gt)
        if s_d is not None and not degen:
            wp_base_d = np.stack([al.apply_sim3(wp[i].reshape(-1, 3), s_d, R_d, t_d)
                                  .reshape(Hp, Wp, 3) for i in range(len(views))])
            cad = al.apply_sim3(centers_vggt, s_d, R_d, t_d)
            cderr = np.linalg.norm(cad - centers_gt, axis=1) * 1000
            add_row("world_points", "deployable", wp_base_d, wconf, extra=dict(
                centers_err_mm={f"cam_center_err_{views[i]}_mm": float(cderr[i])
                                for i in range(len(views))},
                rot_err_deg={f"cam_rot_err_{views[i]}_deg":
                             rot_err_deg(R_d @ extr[i][:3, :3].T, T_bc[i][:3, :3])
                             for i in range(len(views))},
                similarity_scale=round(float(s_d), 6),
                centers_degenerate=False))
        else:
            rows.append({**base_row, **dict(route="world_points", align_mode="deployable",
                                            error="centers_degenerate_or_insufficient",
                                            n_centers=len(views))})
    else:
        # Single view: VGGT world == camera-0 frame (extr ~ identity).
        # Deployable similarity with scale about the CAMERA CENTER (a scale
        # about the origin would move the anchored camera off its true pose):
        #   p_base = c_true + s * (R p_vggt),  R = T_base_cam0[:3,:3]
        # s comes from the known camera-to-table distance: estimate the table
        # as the robust lower envelope of the camera-relative z (bottom 25%),
        # robust to VGGT's monocular tilt/distortion. No GT depth used.
        R_d = T_bc[0][:3, :3].copy()
        c_true = al.camera_centers_from_extrinsic(T_cb[0])  # camera center in base
        P0 = wp[0].reshape(-1, 3)
        P0 = P0[np.isfinite(P0).all(-1)]
        z_camrel = (P0 @ R_d.T)[:, 2]           # camera-relative z
        bottom = z_camrel[z_camrel < np.quantile(z_camrel, 0.25)]
        c_tab_rel = float(np.median(bottom))    # table in camera-relative z
        h_known = abs(cam_h - table_z)
        dist_anch = abs(-c_tab_rel)             # camera-to-table distance
        s_1 = h_known / dist_anch if dist_anch > 1e-6 else float("nan")
        t_d = c_true                            # camera stays at true pose
        wp_base_d = np.stack([c_true + s_1 * (wp[i].reshape(-1, 3) @ R_d.T)
                              .reshape(Hp, Wp, 3) for i in range(len(views))])
        add_row("world_points", "deployable", wp_base_d, wconf, extra=dict(
            similarity_scale=round(float(s_1), 6),
            table_cam_rel_z=round(c_tab_rel, 4),
            n_points_used=int(len(P0))))

        # ---------- G3 single view: VGGT depth + GT calibration, same scale ----------
        p_cam, m = unproject(depth_v[0], K_gt_hi[0], np.eye(4))
        p_base = (p_cam * s_1) @ T_bc[0][:3, :3].T + T_bc[0][:3, 3]
        pts = np.zeros((Hp, Wp, 3))
        pts[m] = p_base
        pts[~m] = np.nan
        add_row("depth_gt_calib(g3)", "deployable", pts[None], dconf,
                extra=dict(similarity_scale=round(float(s_1), 6)))

    # ---------- G3 (multi view): VGGT depth + GT calibration, deployable scale ----------
    # scale in the CAMERA frame (natural origin) then transform to base
    if len(views) >= 3 and s_d is not None and not degen:
        s_g3 = s_d
        pts3 = []
        for i in range(len(views)):
            p_cam, m = unproject(depth_v[i], K_gt_hi[i], np.eye(4))
            p_base = (p_cam * s_g3) @ T_bc[i][:3, :3].T + T_bc[i][:3, 3]
            pts = np.zeros((Hp, Wp, 3))
            pts[m] = p_base
            pts[~m] = np.nan
            pts3.append(pts)
        add_row("depth_gt_calib(g3)", "deployable", np.stack(pts3), dconf,
                extra=dict(similarity_scale=round(float(s_g3), 6)))

    return rows


cv2_NEAREST = 0  # cv2.INTER_NEAREST


def main():
    import cv2
    globals()["cv2_NEAREST"] = cv2.INTER_NEAREST

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--models", nargs="+",
                    default=["vggt_official", "vggt_omnivggt"])
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--view-configs", nargs="+",
                    default=["head", "head+left_wrist+right_wrist"])
    ap.add_argument("--out", default="runs/smoke/metrics/geometry_metrics.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache_names = {"vggt_official": "vggt_official", "vggt_omnivggt": "vggt_omnivggt"}
    all_rows = []
    t0 = time.perf_counter()
    for model in args.models:
        model_cache = Path(cfg["heavy_root"]) / "cache" / cache_names[model]
        for sid in args.frames:
            for vc in args.view_configs:
                try:
                    rows = evaluate_one(cfg, sid, vc, model_cache, model)
                    all_rows.extend(rows)
                    print(f"[{model}] {sid} {vc}: {len(rows)} metric rows", flush=True)
                except FileNotFoundError as e:
                    print(f"[{model}] {sid} {vc}: MISSING ({e})", flush=True)
                    all_rows.append(dict(sample_id=sid, model=model, view_config=vc,
                                         route="all", align_mode="all", error="missing_pred"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"rows={len(all_rows)} -> {out}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
