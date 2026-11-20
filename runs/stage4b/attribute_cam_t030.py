"""P0 Failure Attribution: nominal vs cam_t030 (0.30 m head shift) paired
analysis on the SAME deterministic replay (planning verdict §3).

Inputs: two replay_episode_gt.py outputs of the SAME episode/seed (nominal +
shifted camera; determinism verified by verify_probe_physics), the a0 ckpt.
Per paired frame: canonical voxel from each condition's 3-cam depth, coverage,
door-region observability, cross-condition observed/hit IoU, then SAME-NOISE
x0jump action difference + FK EE end-point difference.

Run (openwam env):
  python runs/stage4b/attribute_cam_t030.py \
    --nominal /data/.../replay_gt/clean_10 \
    --shifted /data/.../replay_gt/probe_ep10_camshift030_seed1000 \
    --ckpt runs/stage4a/a0_s42/best.pt --stride 10
Outputs: metrics/attribution/att_frames.csv + att_summary.json (console too).
NOTE: qpos is held at the serving first-call convention (zeros + grip=1) for
BOTH conditions so the paired action difference isolates the voxel change.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import yaml  # noqa: E402
from cvwam.geometry import (T_base_world, cam2world_gl_to_world_cam_cv,  # noqa: E402
                            depth_to_base)
import cvwam.robotwin_adapter as rta  # noqa: E402
from voxelize_gpu import voxelize_rays_gpu  # noqa: E402
from train_stage4a import normalize_act  # noqa: E402


def build_voxel(depth_mm, Ks, Es, T_bw, b_min, b_max, pitch, shape, device):
    cams_c, pts, val = [], [], []
    for c in ("head", "left", "right"):
        dmm = np.asarray(depth_mm[c], dtype=np.float64)
        gl = np.asarray(Es[c], dtype=np.float64)
        if gl.shape == (3, 4):
            gl = np.vstack([gl, [0.0, 0.0, 0.0, 1.0]])
        T_base_cam = T_bw @ cam2world_gl_to_world_cam_cv(gl)
        valid = dmm > 0
        p, _ = depth_to_base(np.where(valid, dmm / 1000.0, 0.0),
                             np.asarray(Ks[c], dtype=np.float64), T_base_cam,
                             valid=valid)
        v = valid.ravel()
        cams_c.append(T_base_cam[:3, 3])
        pts.append(p)
        val.append(v)
    return voxelize_rays_gpu(np.stack(cams_c), pts, val, b_min, b_max, pitch,
                             shape, device=device)


def voxel_to_np(g, shape):
    x = np.stack([g["hit"].float().cpu().numpy(), g["free"].float().cpu().numpy(),
                  g["observed"].float().cpu().numpy(),
                  (g["view_count"].float() / 3.0).cpu().numpy()])
    return x[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--shifted", required=True)
    ap.add_argument("--ckpt", default="runs/stage4a/a0_s42/best.pt")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--geom-dir",
                    default="/data/canonical-volumetric-wam/cache/voxels_ep_replay/"
                            "demo_clean_balanced_10_replay/20mm",
                    help="build_replay_geometry output with segmentation-driven "
                         "door_region per frame (overrides the fixed box)")
    args = ap.parse_args()
    device = args.device
    geom_dir = args.geom_dir or None

    nom = json.loads((Path(args.nominal) / "replay_gt.json").read_text())
    shf = json.loads((Path(args.shifted) / "replay_gt.json").read_text())
    n = min(len(nom), len(shf))
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"paired frames: nominal={len(nom)} shifted={len(shf)} -> {n}")

    cfg = yaml.safe_load((REPO / "configs/smoke.yaml").read_text())
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    mf = json.loads((Path("/data/canonical-volumetric-wam/cache/voxels_ep/"
                          "demo_clean_balanced/episode_0000010/20mm")
                     / "manifest.json").read_text())
    b_min = np.asarray(mf["b_min"]); b_max = np.asarray(mf["b_max"])
    pitch = float(mf["pitch_m"])
    from cvwam import voxel as vx
    _, _, shape = vx.grid_config(b_min, b_max, pitch)

    # door-region box (base frame): link names are generic (base/link_0..3) and
    # per-frame movement detection failed (all-zero ptp); use the measured door
    # front location in base frame (stage4b design notes: door link base
    # ≈ (0.72,-0.03,0.88)); document as an approximation box
    door_center = np.asarray([0.72, -0.03, 0.88])
    door_box = (door_center - 0.10, door_center + 0.10)
    print(f"door box (base frame approximation): {door_box[0].round(2)}..{door_box[1].round(2)}")

    ck = torch.load(REPO / args.ckpt, map_location=device, weights_only=False)
    from cvwam.models_stage4a import Stage4AModel
    model = Stage4AModel(variant=ck["variant"],
                         bottleneck_ch=ck.get("bottleneck_ch", 128)).to(device).eval()
    model.load_state_dict(ck["model"])
    stats = ck["stats"]

    # FK setup (same mapping for both conditions — systematic errors cancel in
    # the paired difference)
    from build_robot_region import load_chain, fk_links
    chain, joints = load_chain("left")
    jnames = [j for j, _ in chain][:6]

    rows = []
    idx = list(range(0, n, args.stride))
    for i in idx:
        rn, rs = nom[i], shf[i]
        img_n = np.load(Path(args.nominal) / "rgbdepth" / f"frame_{i:07d}.npz")
        img_s = np.load(Path(args.shifted) / "rgbdepth" / f"frame_{i:07d}.npz")
        def cams(img, row):
            ren = {"head_camera": "head", "left_camera": "left",
                   "right_camera": "right"}
            Ks = {ren[c]: row[f"K_{c}"] for c in ren}
            Es = {ren[c]: row[f"T_world_cam_gl_{c}"] for c in ren}
            d = {t: img[f"depth_mm_{t}"] for t in ("head", "left", "right")}
            return d, Ks, Es
        d_n, K_n, E_n = cams(img_n, rn)
        d_s, K_s, E_s = cams(img_s, rs)
        gn = build_voxel(d_n, K_n, E_n, T_bw, b_min, b_max, pitch, shape, device)
        gs = build_voxel(d_s, K_s, E_s, T_bw, b_min, b_max, pitch, shape, device)
        xn, xs = voxel_to_np(gn, shape), voxel_to_np(gs, shape)
        obs_n = float((xn[0, 2] > 0).sum()); obs_s = float((xs[0, 2] > 0).sum())
        hit_n = float((xn[0, 0] > 0).sum()); hit_s = float((xs[0, 0] > 0).sum())
        free_n = float((xn[0, 1] > 0).sum()); free_s = float((xs[0, 1] > 0).sum())
        oiou = float(((xn[0, 2] > 0) & (xs[0, 2] > 0)).sum()) / max(
            float(((xn[0, 2] > 0) | (xs[0, 2] > 0)).sum()), 1.0)
        hiou = float(((xn[0, 0] > 0) & (xs[0, 0] > 0)).sum()) / max(
            float(((xn[0, 0] > 0) | (xs[0, 0] > 0)).sum()), 1.0)
        door = float("nan")
        if geom_dir is not None:
            gf = Path(args.geom_dir) / f"frame_{i:07d}.npz"
            if gf.exists():
                dr = np.load(gf)["door_region"]
                dr_t = torch.as_tensor(dr, device=device)
                door = float((gn["observed"][dr_t]).sum()) / max(int(dr_t.sum()), 1)
        elif door_box is not None:
            lo = torch.as_tensor(np.asarray(door_box[0]), device=device)
            hi = torch.as_tensor(np.asarray(door_box[1]), device=device)
            bmin_t = torch.as_tensor(np.asarray(b_min), device=device)
            cells = torch.stack(torch.meshgrid(
                *[torch.arange(s, device=device) for s in shape], indexing="ij"), -1)
            centers = bmin_t + (cells + 0.5) * pitch
            inbox = ((centers >= lo) & (centers <= hi)).all(-1)
            door = float((gn["observed"][inbox]).sum()) / max(float(inbox.sum()), 1.0)
        # same-noise x0jump action difference
        torch.manual_seed(0)
        eps = torch.randn(1, 16, 14, device=device)
        with torch.no_grad():
            qpos = torch.zeros(1, 14, device=device); qpos[0, 12] = 1; qpos[0, 13] = 1
            xt_n = torch.from_numpy(xn.astype(np.float32)).to(device)
            xt_s = torch.from_numpy(xs.astype(np.float32)).to(device)
            o_n = model(xt_n, eps, torch.tensor([1.0], device=device), qpos,
                        torch.zeros(1, dtype=torch.long, device=device),
                        noisy_world=None, cam72=None)
            o_s = model(xt_s, eps, torch.tensor([1.0], device=device), qpos,
                        torch.zeros(1, dtype=torch.long, device=device),
                        noisy_world=None, cam72=None)
            a_n = (eps - o_n["v_a"]).clamp(0, 1)
            a_s = (eps - o_s["v_a"]).clamp(0, 1)
        act_diff = float(((a_n - a_s) ** 2).mean())
        lo_s = np.asarray(stats["lo"], dtype=np.float32)
        sc_s = np.asarray(stats["scale"], dtype=np.float32)
        an = lo_s + a_n[0].cpu().numpy() * sc_s
        asf = lo_s + a_s[0].cpu().numpy() * sc_s
        qn = {jn: float(v) for jn, v in zip(jnames, an[0][0:6])}
        qs = {jn: float(v) for jn, v in zip(jnames, asf[0][0:6])}
        pn = fk_links(chain, joints, qn)["fl_link6"][1]
        ps = fk_links(chain, joints, qs)["fl_link6"][1]
        ee_diff = float(np.linalg.norm(pn - ps))
        rows.append(dict(frame=i, obs_nom=obs_n, obs_shift=obs_s,
                         hit_nom=hit_n, hit_shift=hit_s,
                         free_nom=free_n, free_shift=free_s,
                         obs_iou=oiou, hit_iou=hiou, door_obs_shift=door,
                         act_mse_norm=act_diff, ee_diff_m=ee_diff))
        print(f"frame {i:3d}: obs {obs_n:.0f}->{obs_s:.0f} hitIoU {hiou:.3f} "
              f"doorObs(shift) {door:.3f} actMSE {act_diff:.4f} eeDiff {ee_diff*100:.1f}cm")

    import csv
    out = REPO / "runs/stage4b/metrics/attribution"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "att_frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {
        "episode_frame_paired": len(rows),
        "obs_ratio_shift_over_nom": float(np.mean([r["obs_shift"] / max(r["obs_nom"], 1) for r in rows])),
        "hit_iou_mean": float(np.mean([r["hit_iou"] for r in rows])),
        "obs_iou_mean": float(np.mean([r["obs_iou"] for r in rows])),
        "door_obs_shift_mean": float(np.nanmean([r["door_obs_shift"] for r in rows])),
        "act_mse_norm_mean": float(np.mean([r["act_mse_norm"] for r in rows])),
        "ee_diff_m_mean": float(np.mean([r["ee_diff_m"] for r in rows])),
        "ee_diff_m_max": float(np.max([r["ee_diff_m"] for r in rows])),
    }
    (out / "att_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
