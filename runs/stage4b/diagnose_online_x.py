"""Diagnose the euler_minus closed-loop mystery: online-built voxel vs cached
training voxel distribution.

Reads the wrapper's dumped online x (from DEBUG_DUMP) and compares channel
statistics against cached voxels_ep training frames of the same episode
(seed 100000 = demo_clean_balanced episode 0 in the export numbering — look
up via the cache manifest).

Run (openwam env): python runs/stage4b/diagnose_online_x.py <dump.npz>
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


def chan_stats(x):
    # x (1,4,H,W,D) float
    x = x[0]
    return {
        "hit": float(x[0].sum()),
        "free": float(x[1].sum()),
        "observed": float(x[2].sum()),
        "view_count_sum": float(x[3].sum()),
        "shape": list(x.shape[1:]),
        "hit_frac_of_observed": float((x[0] > 0).sum() / max((x[2] > 0).sum(), 1)),
    }


def main():
    dump = np.load(sys.argv[1])
    x_online = dump["x"]
    print("=== online (serving) x ===")
    for k, v in chan_stats(x_online).items():
        print(f"  {k}: {v}")
    print(f"  qpos: {dump['qpos'][:6]}")

    cfg = yaml.safe_load((REPO / "configs/smoke.yaml").read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    # episode dir for demo_clean_balanced episode 0
    ep_dirs = sorted(root.glob("demo_clean_balanced/episode_*"))
    if not ep_dirs:
        print("no cached episodes found under", root)
        return
    mf = json.loads((ep_dirs[0] / "20mm" / "manifest.json").read_text())
    print(f"=== cached ({ep_dirs[0].name}/20mm) ===")
    print(f"  bbox b_min={mf['b_min']} b_max={mf['b_max']} pitch={mf['pitch_m']}")
    frames = sorted((ep_dirs[0] / "20mm").glob("frame_*.npz"))
    mid = frames[len(frames) // 2]
    fr = np.load(mid)
    print(f"  frame {mid.name} keys: {fr.files}")
    x_c = None
    for key in ("voxel", "x", "occ"):
        if key in fr.files:
            x_c = fr[key]
            break
    if x_c is None:
        x_c = fr[fr.files[0]]
    if x_c.ndim == 2:  # compact sparse layout? show raw keys instead
        print(f"  raw arrays: {[ (k, fr[k].shape, str(fr[k].dtype)) for k in fr.files ]}")
    else:
        for k, v in chan_stats(x_c[None]).items():
            print(f"  {k}: {v}")
    # bbox comparison: where do online points live? Rebuild from depth+E
    from cvwam.geometry import cam2world_gl_to_world_cam_cv, depth_to_base
    import cvwam.robotwin_adapter as rta
    T_bw = rta.T_world_base_from_config(cfg)
    from cvwam.geometry import T_base_world
    T_bw = T_base_world(T_bw)
    pts_all = []
    for c in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        dmm = dump[f"{c}_depth"]
        K = dump[f"{c}_K"]
        gl = dump[f"{c}_E"]
        if gl.shape == (3, 4):
            gl = np.vstack([gl, [0.0, 0.0, 0.0, 1.0]])
        T_base_cam = T_bw @ cam2world_gl_to_world_cam_cv(gl)
        valid = dmm > 0
        p, _ = depth_to_base(np.where(valid, dmm / 1000.0, 0.0), K, T_base_cam,
                             valid=valid)
        v = valid.ravel()
        pts_all.append(p)
        print(f"  {c}: cam_center_base={T_base_cam[:3, 3].round(3)} "
              f"n_valid={int(valid.sum())} p_shape={p.shape} "
              f"p_range={p[v].min(0).round(2)}..{p[v].max(0).round(2)}")
    mf_np = {k: np.asarray(mf[k]) for k in ("b_min", "b_max")}
    inbox = [float((((p >= mf_np["b_min"]) & (p < mf_np["b_max"])).all(-1)).mean())
             for p in pts_all]
    print(f"  frac of valid points INSIDE cached bbox: "
          f"head={inbox[0]:.3f} lwrist={inbox[1]:.3f} rwrist={inbox[2]:.3f}")


if __name__ == "__main__":
    main()
