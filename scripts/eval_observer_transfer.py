"""Observer strong-intervention zero-shot transfer eval (Stage 4a §6).

Evaluates the trained tiny Observation Model (runs/stage3b/observer_proto/
last.pt, trained ONLY on camera config A) on:
  - in-config holdout (original replay frames 340+, the training-side control)
  - probe_mid    (7.5cm / 7.5deg, zero-shot)
  - probe_strong (15cm / 20deg, zero-shot)
  - persistence baseline (visibility at t == visibility at t+k)
per intervention level. Reports visIoU, visibility-CHANGE F1 (on cells that
flip between t and t+k), reveal/occlude accuracy, and persistence baseline.
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
from train_observer import ObsDataset, ObsNet, cam_params  # noqa: E402

GT_ROOT = Path("/data/canonical-volumetric-wam/cache/replay_gt")
REP_ROOT = Path("/data/canonical-volumetric-wam/cache/voxels_ep_replay")

SETS = {
    "inconf_holdout": ("demo_clean_balanced_10_replay", "clean_10", 340),
    "probe_mid": ("demo_clean_balanced_10_probe_mid", "probe_ep10_mid", 0),
    "probe_strong": ("demo_clean_balanced_10_probe_strong", "probe_ep10_strong", 0),
}


@torch.no_grad()
def eval_set(model, edir, rows, cfg, ts, device, offsets=(4, 8, 12, 16)):
    ds = ObsDataset(edir, rows, cfg, ts, offsets=offsets)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    vis_iou, chg_iou, reveal_acc, occl_acc, persist_iou = [], [], [], [], []
    for batch in dl:
        vox = batch["voxel"].to(device)
        cond = torch.cat([batch["qpos"].to(device),
                          batch["action"].to(device).flatten(1),
                          batch["cam"].to(device)], 1)
        oi = torch.tensor([offsets.index(k) for k in batch["offset"]], device=device)
        out = model(vox, cond, oi)
        for j in range(vox.shape[0]):
            ki = offsets.index(batch["offset"][j])
            vis_logit = out[j, ki]
            tgt = batch["vis_tgt"][j].bool()
            pred = (vis_logit > 0).cpu()
            cur_obs = vox[j, 2] > 0.5
            vis_iou.append(_iou(pred, tgt))
            persist_iou.append(_iou(cur_obs, tgt))
            flip = cur_obs ^ tgt                     # visibility change cells
            if flip.any():
                chg_iou.append(_iou(pred & flip, tgt & flip) if (pred & flip).any()
                               else 0.0)
                reveal = (~cur_obs) & tgt            # unseen -> seen
                occl = cur_obs & (~tgt)              # seen -> unseen
                if reveal.any():
                    reveal_acc.append(float((pred & reveal).sum() / reveal.sum()))
                if occl.any():
                    occl_acc.append(float((~pred & occl).sum() / occl.sum()))
    return dict(
        visIoU=round(float(np.nanmean(vis_iou)), 4),
        persistence_visIoU=round(float(np.nanmean(persist_iou)), 4),
        changeIoU=round(float(np.nanmean(chg_iou)), 4) if chg_iou else None,
        reveal_acc=round(float(np.nanmean(reveal_acc)), 4) if reveal_acc else None,
        occlude_acc=round(float(np.nanmean(occl_acc)), 4) if occl_acc else None,
        n_pairs=len(vis_iou))


def _iou(a, b):
    u = (a | b).sum().item()
    return (a & b).sum().item() / u if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage3b/observer_proto/last.pt")
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default="runs/stage4a/metrics/observer_transfer.json")
    args = ap.parse_args()
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = ObsNet().to(device).eval()
    model.load_state_dict(ck["model"])

    res = {}
    for name, (rep_name, gt_name, ts_start) in SETS.items():
        gt_p = GT_ROOT / gt_name / "replay_gt.json"
        if not gt_p.exists():
            res[name] = "NOT_RUN (missing " + str(gt_p) + ")"
            continue
        rows = json.loads(gt_p.read_text())
        edir = REP_ROOT / rep_name / "20mm"
        n = len(list(edir.glob("frame_*.npz")))
        rows = rows[:n]
        T = len(rows)
        ts = list(range(ts_start, T - max((4, 8, 12, 16)) - 1, 2))
        res[name] = eval_set(model, edir, rows, cfg, ts, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
