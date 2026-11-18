"""Stage 2b: build canonical voxel caches from GT (and optionally VGGT) clouds.

Per docs/plan.md §7.2-7.4: hit/free/observed/unknown/view_count/conflict
channels, pitches 20mm and 10mm, orthogonal slice + 3D renders,
geometry_stats.csv. Storage contract §7.3 under <heavy_root>/cache/voxels.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import voxel as vx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--source", default="gt", help="pointcloud cache source dir name")
    ap.add_argument("--sources", nargs="+", default=["gt"],
                    help="e.g. gt vggt_official")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cache = Path(cfg["cache_dir"])
    out_root = Path(args.out or "runs/smoke")
    vox_root = cache / "voxels"
    rows = []

    for source in args.sources:
        pc_dir = cache / "pointclouds" / source
        files = sorted(pc_dir.glob("**/frame_*.npz"))
        if not files:
            print(f"no point clouds for source {source}; skip")
            continue
        for pitch in cfg["voxel"]["pitches_m"]:
            t_src = time.perf_counter()
            for f in files:
                t0 = time.perf_counter()
                rel = f.relative_to(pc_dir)
                for point_only, tag in [(False, "ray"), (True, "pointonly")]:
                    g = vx.voxelize_sample(f, pitch, point_only=point_only)
                    st = vx.grid_stats(g)
                    dt = time.perf_counter() - t0
                    out_npz = vox_root / source / f"pitch{int(pitch * 1000):02d}mm" / \
                        rel.parent / f"{f.stem}_{tag}.npz"
                    out_npz.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        out_npz,
                        hit=g["hit"], free=g["free"], observed=g["observed"],
                        unknown=g["unknown"], view_count=g["view_count"],
                        conflict=g["conflict"], b_min=g["b_min"],
                        pitch=np.asarray(g["pitch"]), shape=g["shape"])
                    sz = out_npz.stat().st_size / 2**20
                    rows.append({"source": source, "pitch_m": pitch,
                                 "point_only": point_only, "sample_id":
                                 f"{source}/{str(rel.parent)}/{f.stem}",
                                 "preprocess_time_s": round(dt, 2),
                                 "npz_size_mib": round(sz, 2), **st})
                print(f"[{source} p={pitch}] {rel}: {rows[-2]['grid_shape']} "
                      f"hit={rows[-2]['hit_frac']} free={rows[-2]['free_frac']} "
                      f"unk={rows[-2]['unknown_frac']} conf={rows[-2]['conflict_frac']} "
                      f"({dt:.1f}s, {sz:.1f}MiB)", flush=True)

    csv_path = out_root / "metrics" / "voxel_stats.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"csv -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
