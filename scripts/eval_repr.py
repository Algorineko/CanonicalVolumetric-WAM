"""Stage 3c unified evaluation: P2/P2E/P3 in the SAME voxel metric space as
stage3b pareto (val-thresholded -> test, falseOcc budgets). Outputs
runs/stage3c/metrics/pareto_{variant}.csv with the same schema as
runs/stage3b/metrics/pareto_p1_v_*.csv -> directly comparable.

Reuses pareto_eval's calibrate/evaluate/batch machinery via an adapter that
replaces infer_batch with the representation-specific forward.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.models_repr import PointDynamicsNet, PointmapDynamicsNet  # noqa: E402
from cvwam.repr_dataset import ReprDataset  # noqa: E402
import train_repr as tr  # noqa: E402
import pareto_eval as pe  # noqa: E402


def build_model(variant, base, grid):
    if variant in ("p2", "p2e"):
        return PointDynamicsNet(base=base, cond_dim=128, n_offsets=4,
                                pt_in=4 if variant == "p2e" else 3, grid=grid)
    return PointmapDynamicsNet(base=base, cond_dim=128, n_offsets=4, grid=grid)


def make_infer(grid):
    def infer_repr(model_pack, variant, batch, device):
        variant_ck, model = model_pack
        oi = tr.offset_index(batch, device)
        qpos = batch["qpos"].to(device)
        act = batch["action"].to(device)
        gm = batch["grid_meta"].to(device)
        b_min, pitch = gm[:, :3], gm[:, 3]
        if variant_ck in ("p2", "p2e"):
            pts = batch["points"].to(device)
            out = model(pts, qpos, act, oi, b_min, pitch)
        else:
            pm = batch["pointmap"].float().to(device)
            camT = batch["camT"].to(device)
            K = batch["K"].to(device)
            K4 = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], -1)
            K4 = K4[:, None, :].expand(-1, 3, -1).contiguous()
            out = model(pm, camT, K4, qpos, act, oi, b_min, pitch)
        blocks = []
        C = out.shape[1] // 4
        for k in range(4):
            blocks.append({"occ_logit": out[:, k * C], "vis_logit": out[:, k * C + 1]})
        occ = torch.empty_like(batch["voxel"].to(device)[:, 0])
        for ki in range(4):
            sel = (oi == ki)
            if sel.sum() == 0:
                continue
            occ[sel] = blocks[ki]["occ_logit"][sel].float()
        return occ.float().cpu()
    return infer_repr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path/best.pt")
    ap.add_argument("--out-dir", default="runs/stage3c/metrics")
    ap.add_argument("--stride-test", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    vox_root = Path(cfg["cache_dir"]) / "voxels_ep"
    pt_root = Path(cfg["cache_dir"]) / "point_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    test_eps = sm["splits"]["clean_test"] + sm["splits"]["rand_test"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g0 = np.load(vox_root / "demo_clean_balanced" / "episode_0000010" / "20mm" /
                 "frame_0000000.npz")
    grid = tuple(int(v) for v in g0["shape"])

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        csv_path = out_dir / f"pareto_{name}.csv"
        if csv_path.exists():
            print(f"[skip] {name}", flush=True)
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        variant = ck["variant"]
        model = build_model(variant, ck.get("base", 32), grid).to(device).eval()
        model.load_state_dict(ck["model"])
        val_ds = ReprDataset(vox_root, pt_root, val_eps, stride_t=2, repr=variant)
        test_ds = ReprDataset(vox_root, pt_root, test_eps,
                              stride_t=args.stride_test, repr=variant)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=8, pin_memory=True)
        pe.infer_batch = make_infer(grid)
        print(f"[start] {name} ({variant})", flush=True)
        thr_tables = pe.calibrate(val_loader, (variant, model), variant, device)
        rows = pe.evaluate(val_loader, (variant, model), variant, thr_tables,
                           device, "val")
        rows += pe.evaluate(test_loader, (variant, model), variant, thr_tables,
                            device, "test")
        import csv
        for r in rows:
            r["model"] = name
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({c for r in rows for c in r}))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[done] {name}: {len(rows)} rows -> {csv_path}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
