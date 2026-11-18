"""P1-V occupancy threshold calibration (fair-decision-rule analysis).

The P1-V occupancy head is trained with pos_weight ~5; evaluating at the
0.5-probability threshold is one arbitrary decision rule. Sweep thresholds on
VAL (randomized 0), pick the best by observed-occupancy IoU, then report
TEST metrics at that threshold. P1's decision rule (argmax) is its own
calibration; this script also sweeps P1's hit margin for completeness
(argmax is equivalent to threshold 1/3 on softmax — report sensitivity).
Outputs stage3a/metrics/p1v_threshold_sweep.json.
"""

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
from cvwam.dynamics_dataset import DynamicsDataset, OFFSETS  # noqa: E402
from cvwam.models_tiny import TinyDynamicsNet, split_outputs  # noqa: E402
import train_tiny as tt  # noqa: E402
import eval_tiny as ev  # noqa: E402


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u > 0 else float("nan")


def collect_logits(model, ckpt_variant, ds, device):
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                    pin_memory=True)
    recs = []
    for batch in dl:
        vox = batch["voxel"].to(device)
        act = batch["action"].to(device)
        qpos = batch["qpos"].to(device)
        oi = tt.offset_index(batch, device)
        with torch.no_grad():
            o = model(vox, qpos, act, oi)
        blocks = split_outputs(o, tt.MODEL_DIR[ckpt_variant], len(OFFSETS))
        for ki, k in enumerate(OFFSETS):
            sel = oi == ki
            if sel.sum() == 0:
                continue
            state_f = batch["state_future"].to(device)[sel]
            obs_f = batch["observed_future"].to(device).bool()[sel]
            if tt.MODEL_DIR[ckpt_variant] == "decoupled":
                logits = blocks[ki]["occ_logit"][sel].float()
            else:
                probs = torch.softmax(blocks[ki]["state_logits"][sel], dim=1)
                logits = torch.log(probs[:, 0].clamp_min(1e-6) /
                                   (1 - probs[:, 0].clamp_max(1 - 1e-6)))
            state_t = batch["state_t"].to(device)[sel]
            recs.append((k, logits.cpu(), obs_f.cpu(), state_f.cpu(), state_t.cpu()))
    return recs


def main():
    device = "cuda:0"
    cfg = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    val_eps = ["demo_randomized_balanced:0"]
    test_eps = ["demo_randomized_balanced:1"]
    out = {}
    for name, ckpt in [("p1_v", "runs/stage3a/p1_v_split/best.pt"),
                       ("p1", "runs/stage3a/p1_split/best.pt")]:
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model = TinyDynamicsNet(variant=tt.MODEL_DIR[ck["variant"]],
                                base=ck["base"], n_offsets=len(OFFSETS)).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        val_recs = collect_logits(model, ck["variant"],
                                  DynamicsDataset(root, val_eps, stride_t=2), device)
        test_recs = collect_logits(model, ck["variant"],
                                   DynamicsDataset(root, test_eps), device)
        # sweep on VAL per offset
        thrs = np.linspace(-4, 4, 33)
        best_thr = {}
        for k in OFFSETS:
            vl = [(lg, ob, st, stt) for (kk, lg, ob, st, stt) in val_recs if kk == k]
            if not vl:
                continue
            scores = []
            for thr in thrs:
                ious = [iou(lg > thr, st == 0) for lg, ob, st, stt in vl]
                scores.append(float(np.nanmean(ious)))
            best_thr[k] = float(thrs[int(np.argmax(scores))])
        # test metrics at calibrated thresholds
        rows = []
        for k, lg, ob, st, stt in test_recs:
            thr = best_thr.get(k, 0.0)
            pred = lg > thr
            obs = ob
            gt_hit = st == 0
            gt_free = st == 1
            occ_iou = iou(pred & obs, gt_hit & obs)
            false_occ = float((pred & gt_free).sum()) / float(max(gt_free.sum(), 1))
            both = obs & (stt != 2)
            dyn = ((stt == 0) ^ gt_hit) & both
            dyn_iou = iou(pred & dyn, gt_hit & dyn)
            rows.append({"offset": k, "thr": round(thr, 2),
                         "observed_occupancy_iou": round(occ_iou, 4),
                         "knownfree_false_occ": round(false_occ, 4),
                         "dynamic_region_iou": round(dyn_iou, 4)})
        out[name] = {"best_thr_val": {str(k): round(v, 2) for k, v in best_thr.items()},
                     "test_calibrated": rows}
        print(name, "best_thr:", out[name]["best_thr_val"])
        for k in OFFSETS:
            r = [x for x in rows if x["offset"] == k]
            if r:
                print(f"  k={k} occIoU@cal={np.nanmean([x['observed_occupancy_iou'] for x in r]):.3f} "
                      f"falseOcc={np.nanmean([x['knownfree_false_occ'] for x in r]):.4f} "
                      f"dynIoU={np.nanmean([x['dynamic_region_iou'] for x in r]):.3f}")
    json.dump(out, open("runs/stage3a/metrics/p1v_threshold_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
