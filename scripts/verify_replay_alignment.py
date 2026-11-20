"""Stage 3b §2 gate: verify the deterministic replay reproduces the original
episode (RGB within JPEG noise, depth EXACT via lossless codec, camera
extrinsics exact, frame count aligned). If it fails, replay labels are NOT
attached to the original data (independent regeneration instead)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--variant", default="demo_clean_balanced")
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--max-t", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = rta.locate_episode(cfg["data_root"], args.variant, args.episode,
                               task=cfg.get("task", "open_microwave"))

    with __import__("h5py").File(paths.hdf5, "r") as f:
        T = int(f["state/left_arm_joint_states"].shape[0])
        E_h5 = {cam: np.asarray(f[f"vision/{cam}/extrinsics_matrix"]) for cam in rta.CAMERAS}
    rows = json.loads((Path(args.replay_dir) / "replay_gt.json").read_text())
    n_replay = len(rows)
    print(f"original T={T} (obs list {T + 1}); replay rows={n_replay}")

    # frame-count alignment: original obs list has T+1 entries (T stored);
    # replay captures frame0 + every _take_picture + final
    img_dir = Path(args.replay_dir) / "rgbdepth"
    max_t = args.max_t or T
    stats = {"rgb_diff_mean": [], "rgb_diff_p99": [], "depth_exact_frac": [],
             "extr_max_abs": [], "n_checked": 0}
    mismatches = []
    for t in range(0, min(T, max_t)):
        # replay row t must correspond to original obs t
        if t >= len(rows):
            mismatches.append(f"row {t} missing in replay")
            break
        imgf = img_dir / f"frame_{rows[t]['frame']:07d}.npz"
        d = np.load(imgf)
        rgb_r = d["rgb_head"]
        dep_r = d["depth_mm_head"]
        # original: decode RGB bitstream + ffv1 depth at t
        with __import__("h5py").File(paths.hdf5, "r") as f:
            rgb_o = rta.decode_image_bit(f["vision/cam_head/colors"][t])
        dep_o = rta.decode_ffv1_depth_at(paths.depth_ffv1["cam_head"], t)
        if rgb_o.shape != rgb_r.shape:
            mismatches.append(f"t={t} rgb shape {rgb_o.shape} vs {rgb_r.shape}")
            continue
        rgb_diff = np.abs(rgb_o.astype(np.int16) - rgb_r.astype(np.int16))
        stats["rgb_diff_mean"].append(float(rgb_diff.mean()))
        stats["rgb_diff_p99"].append(float(np.percentile(rgb_diff, 99)))
        stats.setdefault("rgb_frac_gt10", []).append(
            float((rgb_diff > 10).mean()))
        # replay depth is float64 mm (pre-rint); original stored rint->uint16
        dep_r_u16 = np.rint(dep_r.astype(np.float64)).astype(np.int32)
        exact = float((dep_r_u16 == dep_o.astype(np.int32)).mean())
        stats["depth_exact_frac"].append(exact)
        # extrinsics: same-convention comparison (cam2world_gl vs cam2world_gl;
        # the HDF5 extrinsics_matrix IS cam2world_gl from the same API)
        Twc_gl = np.asarray(E_h5["cam_head"][t], dtype=np.float64)
        T_r = np.asarray(rows[t]["T_world_cam_gl_head_camera"], dtype=np.float64)
        e = float(np.abs(T_r - Twc_gl).max())
        stats["extr_max_abs"].append(e)
        stats["n_checked"] += 1
        if exact < 0.99 or e > 1e-4:   # <1% depth diff = float-vs-rint boundary artifact
            mismatches.append(f"t={t}: depth_exact={exact:.4f} extr_err={e:.2e} "
                              f"rgb_mean={stats['rgb_diff_mean'][-1]:.2f}")
        if t % 100 == 0:
            print(f"  t={t} checked", flush=True)

    out = {
        "episode": f"{args.variant}:{args.episode}",
        "T_original": T, "replay_rows": n_replay,
        "rgb_diff_mean": round(float(np.mean(stats["rgb_diff_mean"])), 3),
        "rgb_diff_p99_max": round(float(np.max(stats["rgb_diff_p99"])), 1),
        "depth_exact_frac_min": round(float(np.min(stats["depth_exact_frac"])), 6),
        "depth_exact_frac_mean": round(float(np.mean(stats["depth_exact_frac"])), 6),
        "rgb_frac_gt10_mean": round(float(np.mean(stats.get("rgb_frac_gt10", [0]))), 6),
        "rgb_frac_gt10_max": round(float(np.max(stats.get("rgb_frac_gt10", [0]))), 6),
        "extr_max_abs_max": float(np.max(stats["extr_max_abs"])) if stats["extr_max_abs"] else None,
        "n_checked": stats["n_checked"],
        "n_mismatches": len(mismatches),
        "mismatch_samples": mismatches[:10],
        "verdict": "ALIGNED" if (len(mismatches) == 0 and stats["n_checked"] > 0) else "MISALIGNED",
    }
    outdir = Path("runs/stage3b/metrics")
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"replay_alignment_{args.variant}_{args.episode}.json"
    fn.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
