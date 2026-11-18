"""Stage 2c: Hit/Free/Unknown 3x3 transition statistics over cached episodes.

For offsets k in {1,4,8,16} and all t: classify each voxel's state at t and
t+k (hit / free / unknown; observed-free vs hit vs unknown), accumulate 3x3
transition counts, and decompose by diagnostic region:
- all (whole workspace bbox)
- table (static background band z in [table, table+3cm])
- robot (FK link spheres, if URDF FK available)
- object (simulator object poses, if available; else NOT_AVAILABLE)
Also reports visibility-change vs physical-change decomposition:
- physical change: hit<->free flips (either direction)
- visibility change: observed<->unknown flips with no hit/free flip
Outputs stage2c/metrics/transition_stats.json + per-episode CSV rows.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OFFSETS = (1, 4, 8, 16)
STATE_IDX = {"hit": 0, "free": 1, "unknown": 2}


def state_code(g):
    """hit=0, free=1, unknown=2 (hit wins over free; observed=hit|free)."""
    s = np.full(g["hit"].shape, 2, dtype=np.uint8)
    s[g["free"]] = 1
    s[g["hit"]] = 0
    return s


def transition_matrix(s_a, s_b):
    m = np.zeros((3, 3), dtype=np.int64)
    np.add.at(m, (s_a.ravel(), s_b.ravel()), 1)
    return m


def region_masks(g, cfg, table_z, robot_fn=None):
    b_min = g["b_min"].astype(np.float64)
    pitch = float(g["pitch"])
    shape = g["shape"]
    i, j, k = np.indices(shape)
    centers = b_min + np.stack([i, j, k], axis=-1) * pitch + pitch / 2
    z = centers[..., 2]
    regions = {"all": np.ones(shape, bool),
               "table": (z >= table_z) & (z <= table_z + 0.03)}
    if robot_fn is not None:
        regions["robot"] = robot_fn(centers)
    else:
        regions["robot"] = np.zeros(shape, bool)
    return regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--pitch", default="20mm")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    # per-episode table height lives in each manifest (randomized tables differ)
    rob_cache = root / "_robot_region"

    out_rows = []
    summary = {}
    for spec in args.episodes:
        variant, ep = spec.split(":")
        edir = root / variant / f"episode_{int(ep):07d}" / args.pitch
        manifest = json.loads((edir / "manifest.json").read_text())
        frames = manifest["frames"]
        T = manifest["T_total"]
        table_z = float(manifest.get("table_z_base", cfg["table_z_base"]))
        robot_available = (rob_cache / variant /
                           f"episode_{int(ep):07d}" / "meta.json").exists()
        if not robot_available:
            print(f"  [{spec}] robot region NOT_AVAILABLE (meta.json missing);"
                  f" robot rows will be zero", flush=True)
        for k in OFFSETS:
            mats = {r: np.zeros((3, 3), np.int64) for r in
                    ("all", "table", "robot", "object")}
            n_pairs = 0
            for t in range(0, T - k):
                a = np.load(frames[t]["npz"])
                b = np.load(frames[t + k]["npz"])
                sa, sb = state_code(a), state_code(b)
                regs = region_masks(a, cfg, table_z)
                rm = None
                if robot_available:
                    rm = np.load(rob_cache / variant /
                                 f"episode_{int(ep):07d}" / f"frame_{t:07d}.npz")
                if rm is not None and tuple(rm["robot"].shape) == tuple(sa.shape):
                    regs["robot"] = rm["robot"]
                elif robot_available and t == 0:
                    print(f"  [{spec}] robot region grid mismatch; robot rows zero",
                          flush=True)
                for rname, rmask in regs.items():
                    m = transition_matrix(sa[rmask], sb[rmask])
                    mats[rname] += m
                n_pairs += 1
            row = {"episode": spec, "offset": k, "pairs": n_pairs}
            for rname, m in mats.items():
                tot = m.sum()
                row[f"tm_{rname}"] = m.tolist()
                # headline stats: physical flip rate + visibility flip rate
                hh, hf, hu = m[0]
                fh, ff, fu = m[1]
                uh, uf, uu = m[2]
                row[f"{rname}_physical_flip_rate"] = round(
                    float((hf + fh) / tot), 5)
                row[f"{rname}_vis_flip_rate"] = round(
                    float((hu + uh + uf + fu) / tot), 5)
                row[f"{rname}_hit_free_to_unknown"] = round(
                    float((hu + uf) / tot), 5)
                row[f"{rname}_unknown_to_observed"] = round(
                    float((uh + uf) / tot), 5)
            out_rows.append(row)
            print(json.dumps({kk: row[kk] for kk in
                              ("episode", "offset", "pairs",
                               "all_physical_flip_rate", "all_vis_flip_rate")}))
    outdir = Path("runs/stage2c/metrics")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "transition_rows.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    # compact matrices per (episode, offset) for the report
    (outdir / "transition_stats.json").write_text(json.dumps(out_rows, indent=1))
    print(f"-> {outdir}/transition_stats.json ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
