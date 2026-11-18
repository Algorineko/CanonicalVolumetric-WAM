"""Stage 3b probe: metric mono-depth (MoGe-2) as a candidate RGB-only geometry
source, evaluated with the SAME protocol as the VGGT smoke (plan.md §6.5).

The user opened the door to mono-depth research (LingBot-Depth / MoGe); this
is a cheap feasibility probe on the already-evaluated frames — NOT a reopen of
the VGGT line. LingBot-Depth weights are not local; MoGe-2 ViT-B is
(/data/models/moge-2-vitb-normal)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, "/home/zhangjialiang/code/lingbot-vla-v2/lingbotvla/models/vla/vision_models/MoGe")
from cvwam import geometry as geo  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--frames", nargs="+", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = "cuda:0"

    from moge.model.v2 import MoGeModel
    t0 = time.perf_counter()
    model = MoGeModel.from_pretrained("/data/models/moge-2-vitb-normal/model.pt").to(device).eval()
    print(f"MoGe-2 loaded in {time.perf_counter() - t0:.1f}s")

    rows = []
    for sid in args.frames:
        variant, ep, t = sid.split("/")
        t = int(t.split("=")[1])
        frames = Path(cfg["cache_dir"]) / "frames" / variant / ep
        gt = np.load(frames / f"frame_{t:07d}.npz")
        img = gt["rgb"][0]  # head view uint8 HxWx3
        K = gt["K"][0]
        H, W = img.shape[:2]
        fov_x = float(np.degrees(2 * np.arctan(W / (2 * K[0, 0]))))
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        torch.cuda.reset_peak_memory_stats()
        tt0 = time.perf_counter()
        with torch.inference_mode():
            out = model.infer(x, fov_x=fov_x, use_fp16=True)
        lat = time.perf_counter() - tt0
        depth = out["depth"].float().cpu().numpy()   # metric, at input res
        intr = out["intrinsics"].float().cpu().numpy()
        mem = torch.cuda.max_memory_allocated() / 2**30
        # depth comparison at native res (valid GT pixels)
        gt_d = gt["depth_m"][0]
        valid = gt["depth_valid"][0] & (gt_d > 0)
        p = depth[valid]
        g = gt_d[valid]
        mae = float(np.abs(p - g).mean() * 1000)
        rmse = float(np.sqrt(((p - g) ** 2).mean()) * 1000)
        absrel = float((np.abs(p - g) / g).mean())
        # point metrics: unproject MoGe depth with GT K/T -> base, NN vs GT cloud
        T_bc = gt["T_base_cam"][0]
        h, w = gt_d.shape
        vv, uu = np.where(valid)
        pcam = (np.stack([uu, vv, np.ones(len(uu))], 1) @ np.linalg.inv(K).T) * depth[vv, uu, None]
        pbase = pcam @ T_bc[:3, :3].T + T_bc[:3, 3]
        inb_min = np.asarray([cfg["workspace_bbox_base"][k][0] for k in "xyz"])
        inb_max = np.asarray([cfg["workspace_bbox_base"][k][1] for k in "xyz"])
        inb = ((pbase >= inb_min) & (pbase <= inb_max)).all(1)
        pc_dir = Path(cfg["cache_dir"]) / "pointclouds" / "gt" / variant / ep
        pc = np.load(pc_dir / f"frame_{t:07d}.npz")
        G = pc["xyz_bbox"]
        tree = cKDTree(G)
        d_nn, _ = tree.query(pbase[inb], k=1, workers=-1)
        rows.append(dict(sample_id=sid, depth_mae_mm=round(mae, 1),
                         depth_rmse_mm=round(rmse, 1), absrel=round(absrel, 4),
                         acc_median_mm=round(float(np.median(d_nn) * 1000), 1),
                         acc_mean_mm=round(float(d_nn.mean() * 1000), 1),
                         n_in_bbox=int(inb.sum()), latency_s=round(lat, 3),
                         peak_gib=round(mem, 2),
                         intra_fov_err=round(float(np.abs(intr - K).max()), 2)))
        print(json.dumps(rows[-1]))
    outdir = Path("runs/stage3b/metrics")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "monodepth_probe_moge2.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {outdir / 'monodepth_probe_moge2.json'}")
    print("reference (VGGT-1B G1 head-only, oracle-aligned): depth MAE 75.0 mm, acc median 28.4 mm")


if __name__ == "__main__":
    main()
