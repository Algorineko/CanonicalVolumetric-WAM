"""Threshold-sweep frontier: evaluate models at FIXED thresholds (no val
calibration) to measure the test-side (falseOcc, dynIoU) frontier directly —
resolves whether strict-budget collapse is calibration transfer or capacity.

Sweep: thr in {-4,-2,-1,-0.5,0,0.5,1,2,4} applied to TEST only; reports
per-threshold mean falseOcc and dynIoU per test group.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.dynamics_dataset import DynamicsDataset  # noqa: E402
from cvwam.repr_dataset import ReprDataset  # noqa: E402
from cvwam.models_tiny import TinyDynamicsNet  # noqa: E402
import train_tiny as tt  # noqa: E402
import pareto_eval as pe  # noqa: E402

THRS = (-4, -2, -1, -0.5, 0.0, 0.5, 1, 2, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default="runs/stage3c/metrics/thr_sweep.csv")
    args = ap.parse_args()
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    pt_root = Path(cfg["cache_dir"]) / "point_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    test_eps = sm["splits"]["clean_test"] + sm["splits"]["rand_test"]
    REPR = ("p2", "p2e", "p3")
    rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        ck = torch.load(path, map_location=device, weights_only=False)
        if ck["variant"] in REPR:
            import eval_repr as er
            g0 = np.load(root / "demo_clean_balanced" / "episode_0000010" / "20mm" /
                         "frame_0000000.npz")
            grid = tuple(int(v) for v in g0["shape"])
            model = er.build_model(ck["variant"], ck.get("base", 32), grid).to(device).eval()
            model.load_state_dict(ck["model"])
            ds = ReprDataset(root, pt_root, test_eps, stride_t=2, repr=ck["variant"])
            infer = er.make_infer(grid)
        else:
            model = TinyDynamicsNet(variant=tt.MODEL_DIR[ck["variant"]], base=ck["base"],
                                    n_offsets=4).to(device).eval()
            model.load_state_dict(ck["model"])
            ds = DynamicsDataset(root, test_eps, stride_t=2)
            infer = pe.infer_batch
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
        acc = {t: collections_() for t in THRS}
        for batch in dl:
            occ = infer((ck["variant"], model) if ck["variant"] in REPR else model,
                        ck["variant"], batch, device)
            gt = pe.batch_gt(batch)
            oi = tt.offset_index(batch, torch.device("cpu"))
            obs = gt["obs_f"]
            gt_hit = gt["state_f"] == 0
            gt_free = gt["state_f"] == 1
            both = obs & (gt["state_t"] != 2)
            dyn = ((gt["state_t"] == 0) ^ gt_hit) & both
            eps = gt["episodes"]
            for ki in range(4):
                sel = (oi == ki)
                if sel.sum() == 0:
                    continue
                idx = torch.nonzero(sel).flatten()
                for j in idx.tolist():
                    for t in THRS:
                        pred = occ[j] > t
                        inter = (pred & dyn[j]).sum().item()
                        uni = ((pred & dyn[j]) | (gt_hit[j] & dyn[j])).sum().item()
                        fo_n = (pred & gt_free[j]).sum().item()
                        fo_d = max(int(gt_free[j].sum()), 1)
                        acc[t]["dyn_i"].append(inter)
                        acc[t]["dyn_u"].append(uni)
                        acc[t]["fo_n"].append(fo_n)
                        acc[t]["fo_d"].append(fo_d)
                        acc[t]["grp"].append("rand" if "randomized" in eps[j] else "clean")
        for t in THRS:
            for grp in ("clean", "rand"):
                gi = [i for i, g in enumerate(acc[t]["grp"]) if g == grp]
                if not gi:
                    continue
                # per-sample IoU then mean (matches pareto protocol)
                ious = [acc[t]["dyn_i"][i] / acc[t]["dyn_u"][i]
                        for i in gi if acc[t]["dyn_u"][i] > 0]
                fos = [acc[t]["fo_n"][i] / acc[t]["fo_d"][i] for i in gi]
                diou = float(np.mean(ious)) if ious else float("nan")
                fo = float(np.mean(fos))
                rows.append(dict(model=name, thr=t, group=grp + "_test",
                                 false_occ=round(fo, 5), dyn_iou=round(diou, 5)))
        print(f"[done] {name}", flush=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({c for r in rows for c in r}))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"-> {args.out}", flush=True)


def collections_():
    return dict(dyn_i=[], dyn_u=[], fo_n=[], fo_d=[], grp=[])


if __name__ == "__main__":
    main()
