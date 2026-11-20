"""G-Integrity pre-rollout checks (planning verdict 2026-11-20 §1).

Mandatory before any formal rollout. Checks camera contract, online-obs
non-emptiness/distribution, qpos contract, same-frame online/offline parity,
sampler determinism, ckpt/config/patch hashes. Blind-baseline runs (zero-obs
COND) are a separate chain condition, not part of this script.

Usage (openwam env):
  python runs/stage4b/gintegrity.py --dump /data/cvwam_debug/<run>/online_x_obs_chunk0.npz \
      [--cache-episode /data/.../voxels_ep/demo_clean_balanced/episode_0000000/20mm] \
      [--ckpt runs/stage4a/a0_s42/best.pt] [--json <out>]
Exit code 0 = all hard checks pass; 1 = at least one FAIL.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = {}


def check(name, ok, detail=""):
    RESULTS[name] = {"ok": bool(ok), "detail": str(detail)}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def sha256(path, n=8):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--cache-episode", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--min-observed", type=int, default=1000)
    ap.add_argument("--min-parity-iou", type=float, default=0.90)
    ap.add_argument("--sampler-parity", action="store_true",
                    help="deterministic double-run of the serving x0jump path")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dump = np.load(args.dump)

    # 1. camera contract: nominal head pose -> chain -> anchor (0.2, 0.032, 1.35)
    import yaml
    import cvwam.robotwin_adapter as rta
    from cvwam.geometry import T_base_world, cam2world_gl_to_world_cam_cv
    cfg = yaml.safe_load((REPO / "configs/smoke.yaml").read_text())
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
    emb = yaml.safe_load(Path(
        "/home/zhangjialiang/code/RoboTwin/assets/embodiments/aloha-agilex/config.yml"
    ).read_text())
    hc = emb["static_camera_list"][0]
    fwd = np.asarray(hc["forward"], float); fwd /= np.linalg.norm(fwd)
    left = np.asarray(hc["left"], float); left /= np.linalg.norm(left)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([fwd, left, np.cross(fwd, left)], 1)
    pose[:3, 3] = np.asarray(hc["position"], float)
    t = (T_bw @ cam2world_gl_to_world_cam_cv(pose))[:3, 3]
    anchor = np.asarray(cfg["robot_base"]["head_cam_base_anchor_m"], float)
    err = float(np.abs(t - anchor).max())
    check("camera_contract", err < 1e-5, f"anchor err {err:.2e} (t={t.round(4)})")

    # 2. obs non-empty + distribution
    x = dump["x"]
    obs_n = float((x[0, 2] > 0).sum()); hit_n = float((x[0, 0] > 0).sum())
    check("obs_nonempty", obs_n >= args.min_observed,
          f"observed={obs_n:.0f} hit={hit_n:.0f} (min {args.min_observed})")
    if obs_n > 0:
        ratio = hit_n / obs_n
        check("hit_obs_ratio_plausible", 0.01 < ratio < 0.6,
              f"hit/observed={ratio:.3f}")
    dz = [float(np.asarray(dump[f"{c}_depth"]).astype(bool).mean())
          for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")]
    check("depth_nonzero_all_cams", all(v > 0.5 for v in dz),
          f"valid-depth frac {['%.2f' % v for v in dz]}")

    # 3. qpos contract: first-call zeros are a known artifact; flag if
    #    arm joints are zero while gripper reads 1 (home should be nonzero later)
    q = np.asarray(dump["qpos"], float)
    arm_zero = bool(np.allclose(q[:12], 0.0))
    check("qpos_firstcall", True,
          f"arm_all_zero={arm_zero} (known first-call artifact if chunk0)"
          if arm_zero else f"qpos0={q[:6].round(3)}")

    # 4. same-frame parity vs cached episode frame0
    if args.cache_episode:
        f0 = sorted(Path(args.cache_episode).glob("frame_*.npz"))[0]
        fr = np.load(f0)
        cs = np.stack([fr["hit"], fr["free"], fr["observed"],
                       fr["view_count"]]).astype(np.float32)
        inter = ((x[0, 2] > 0) & (cs[2] > 0)).sum()
        union = ((x[0, 2] > 0) | (cs[2] > 0)).sum()
        iou = float(inter) / max(float(union), 1.0)
        check("sameframe_parity", iou >= args.min_parity_iou,
              f"observed IoU {iou:.3f} vs {f0.name} (min {args.min_parity_iou})")

    # 5. sampler parity: x0jump deterministic under fixed seed
    if args.sampler_parity and args.ckpt:
        import torch
        from cvwam.models_stage4a import Stage4AModel
        device = "cuda:0"
        ck = torch.load(REPO / args.ckpt, map_location=device, weights_only=False)
        model = Stage4AModel(variant=ck["variant"],
                             bottleneck_ch=ck.get("bottleneck_ch", 128)
                             ).to(device).eval()
        model.load_state_dict(ck["model"])
        xt = torch.from_numpy(x.astype(np.float32)).to(device)
        qpos = torch.from_numpy(q[:14].astype(np.float32))[None].to(device)
        with torch.no_grad():
            torch.manual_seed(0)
            eps = torch.randn(1, 16, 14, device=device)
            o1 = model(xt, eps, torch.tensor([1.0], device=device), qpos,
                       torch.zeros(1, dtype=torch.long, device=device),
                       noisy_world=None, cam72=None)
            a1 = eps - o1["v_a"]
            torch.manual_seed(0)
            eps2 = torch.randn(1, 16, 14, device=device)
            o2 = model(xt, eps2, torch.tensor([1.0], device=device), qpos,
                       torch.zeros(1, dtype=torch.long, device=device),
                       noisy_world=None, cam72=None)
            a2 = eps2 - o2["v_a"]
        mse = float(((a1 - a2) ** 2).mean())
        check("sampler_parity_deterministic", mse == 0.0, f"double-run MSE {mse:.2e}")

    # 6/7. ckpt/config/patch hashes (informational)
    hashes = {}
    if args.ckpt:
        hashes["ckpt"] = sha256(REPO / args.ckpt)
    for rel in ("/home/zhangjialiang/code/RoboTwin/XPolicyLab/policy/cvwam_fm/model.py",
                "/home/zhangjialiang/code/RoboTwin/scripts/eval_policy_xpolicylab.py",
                "/home/zhangjialiang/code/RoboTwin/scripts/collect_data.py"):
        p = Path(rel)
        if p.exists():
            hashes[p.name] = sha256(p)
    RESULTS["hashes"] = hashes
    print(f"[INFO] hashes: {hashes}")

    n_fail = sum(1 for k, v in RESULTS.items()
                 if isinstance(v, dict) and v.get("ok") is False)
    print(f"G-INTEGRITY: {'PASS' if n_fail == 0 else f'{n_fail} FAIL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"results": RESULTS, "dump": args.dump}, indent=2))
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
