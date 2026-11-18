"""Build GT-depth robot-base point clouds from exported canonical samples.

Per docs/plan.md Stage 1 §5.4 / Gate G1:
- per-view and fused PLY (view_id + rgb), camera centers/axes PLY,
- multi-view nearest-neighbor consistency (mm, both directions),
- workspace coverage vs the configured bbox,
- per-sample npz point-cloud cache for Stage 2b voxelization.
"""

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import geometry as geo  # noqa: E402

VIEW_NAMES = ("head", "left_wrist", "right_wrist")


def nn_stats(src: np.ndarray, kdtree: cKDTree):
    """Distances (m) from every src point to its nearest neighbor in the tree."""
    d, _ = kdtree.query(src, k=1, workers=-1)
    return d


def summarize(dist: np.ndarray, prefix: str):
    return {
        f"{prefix}_mean_mm": float(dist.mean() * 1000),
        f"{prefix}_median_mm": float(np.median(dist) * 1000),
        f"{prefix}_p95_mm": float(np.percentile(dist, 95) * 1000),
        f"{prefix}_max_mm": float(dist.max() * 1000),
    }


def load_sample(npz_path: Path):
    d = np.load(npz_path, allow_pickle=False)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--source", default="gt", choices=["gt"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--ply-frames", default="0",
                    help="comma list of manifest indices (0-based) that also get PLY dumps")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache = Path(cfg["cache_dir"])
    frames_dir = cache / "frames"
    pc_dir = cache / "pointclouds" / args.source
    pc_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out or "runs/smoke")
    ply_idx = {int(x) for x in args.ply_frames.split(",")}

    bbox = {ax: np.asarray(v, dtype=np.float64) for ax, v in cfg["workspace_bbox_base"].items()}
    b_min = np.array([bbox["x"][0], bbox["y"][0], bbox["z"][0]])
    b_max = np.array([bbox["x"][1], bbox["y"][1], bbox["z"][1]])

    rows = []
    manifest = [json.loads(l) for l in (frames_dir / "manifest.jsonl").read_text().splitlines()]
    for i, m in enumerate(manifest):
        npz_path = Path(m["npz"])
        s = load_sample(npz_path)
        rgb, depth, valid = s["rgb"], s["depth_m"], s["depth_valid"]
        K, T_base_cam = s["K"], s["T_base_cam"]
        t0 = time.perf_counter()

        per_view, fused, view_id, colors = [], [], [], []
        for v in range(3):
            p, msk = geo.depth_to_base(depth[v], K[v], T_base_cam[v], valid=valid[v])
            per_view.append((p, msk))
            fused.append(p)
            view_id.append(np.full(len(p), v, dtype=np.int16))
            h, w = depth[v].shape
            vv, uu = np.where(msk)
            colors.append(rgb[v][vv, uu])
        fused = np.concatenate(fused)
        view_id = np.concatenate(view_id)
        colors = np.concatenate(colors)
        in_bbox = ((fused >= b_min) & (fused <= b_max)).all(axis=1)

        # multi-view consistency on in-bbox points.
        # The NN-distance distribution is bimodal: a sharp registration peak
        # (<2mm) for truly co-observed surfaces, plus a broad tail of points
        # only one camera sees (occlusion/close range). Report the registration
        # cluster explicitly instead of a single mask median.
        subtrees = {}
        for v in range(3):
            sel = fused[view_id == v][in_bbox[view_id == v]]
            subtrees[v] = (sel, cKDTree(sel))
        nn_out = {}
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                src, _ = subtrees[a]
                d = nn_stats(src, subtrees[b][1])
                tag = f"nn_{VIEW_NAMES[a][:4]}_to_{VIEW_NAMES[b][:4]}"
                cl = d[d < 0.01]  # registration cluster (<10mm)
                nn_out[tag + "_cluster_median_mm"] = (
                    float(np.median(cl) * 1000) if len(cl) >= 100 else float("nan"))
                nn_out[tag + "_frac_lt_2mm"] = float((d < 0.002).mean())
                nn_out[tag + "_frac_lt_10mm"] = float((d < 0.01).mean())
                nn_out[tag + "_full_median_mm"] = float(np.median(d) * 1000)

        dt = time.perf_counter() - t0
        row = {
            "sample_id": m["sample_id"],
            "points_total": int(len(fused)),
            "crop_retention_frac": round(float(in_bbox.mean()), 4),
            "points_in_bbox": int(in_bbox.sum()),
            "per_view_points": [int(len(p)) for p, _ in per_view],
            "build_time_s": round(dt, 3),
            "bbox_min_m": b_min.tolist(),
            "bbox_max_m": b_max.tolist(),
            **nn_out,
        }
        # extent stats (for workspace-boundary refinement)
        q = np.percentile(fused, [0.5, 99.5], axis=0)
        row["cloud_p05_p995"] = q.round(3).tolist()

        npz_out = pc_dir / Path(*Path(m["sample_id"]).parts[:2]) / \
            f"frame_{Path(m['npz']).stem.split('_')[1]}.npz"
        npz_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_out, xyz=fused.astype(np.float32), view_id=view_id,
            rgb_u8=colors.astype(np.uint8),
            xyz_bbox=fused[in_bbox].astype(np.float32),
            view_id_bbox=view_id[in_bbox],
            rgb_bbox=colors[in_bbox].astype(np.uint8),
            b_min=b_min.astype(np.float32), b_max=b_max.astype(np.float32),
            T_base_cam=T_base_cam, K=K)
        row["pc_npz"] = str(npz_out)

        if i in ply_idx:
            ply_dir = Path(cfg["heavy_root"]) / "cache" / "ply" / args.source / \
                Path(m["sample_id"]).parts[0] / Path(m["sample_id"]).parts[1]
            ply_dir.mkdir(parents=True, exist_ok=True)
            tag = Path(m["npz"]).stem.split("_")[1]
            for v in range(3):
                p, msk = per_view[v]
                geo.write_ply(ply_dir / f"view_{VIEW_NAMES[v]}_{tag}.ply", p,
                              colors=rgb[v][np.where(msk)[0], np.where(msk)[1]])
            geo.write_ply(ply_dir / f"fused_{tag}.ply", fused[in_bbox],
                          colors=colors[in_bbox], extra={"view_id": view_id[in_bbox]})
            # camera centers + axes
            centers, axes = [], []
            for v in range(3):
                centers.append(geo.camera_center_in_base(T_base_cam[v]))
                axes.append(geo.camera_axes_ply_axes(T_base_cam[v]).reshape(6, 3))
            geo.write_ply(ply_dir / f"cameras_{tag}.ply", np.asarray(centers))
            geo.write_ply(ply_dir / f"cam_axes_{tag}.ply", np.concatenate(axes))
            row["ply_dir"] = str(ply_dir)
        rows.append(row)
        print(f"[{i:2d}] {m['sample_id']}: pts={len(fused)} crop={row['crop_retention_frac']:.3f} "
              f"L-H cluster={row['nn_left_to_head_cluster_median_mm']:.2f}mm "
              f"(<2mm {row['nn_left_to_head_frac_lt_2mm']:.3f})", flush=True)

    csv_path = out_root / "metrics" / "gt_geometry.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"csv -> {csv_path}")

    # aggregate registration-cluster stats over all samples
    summary = {}
    for a in VIEW_NAMES:
        for b in VIEW_NAMES:
            if a == b:
                continue
            cm = np.array([r[f"nn_{a[:4]}_to_{b[:4]}_cluster_median_mm"] for r in rows],
                          dtype=float)
            f2 = np.array([r[f"nn_{a[:4]}_to_{b[:4]}_frac_lt_2mm"] for r in rows])
            f10 = np.array([r[f"nn_{a[:4]}_to_{b[:4]}_frac_lt_10mm"] for r in rows])
            ok = np.isfinite(cm)
            summary[f"{a[:4]}->{b[:4]}"] = {
                "cluster_median_mm": {
                    "mean": round(float(np.nanmean(cm)), 3),
                    "max": round(float(np.nanmax(cm)), 3),
                    "n_valid": int(ok.sum())},
                "frac_lt_2mm": {"mean": round(float(f2.mean()), 3),
                                 "min": round(float(f2.min()), 3)},
                "frac_lt_10mm": {"mean": round(float(f10.mean()), 3)},
            }
    (out_root / "metrics" / "gt_geometry_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
