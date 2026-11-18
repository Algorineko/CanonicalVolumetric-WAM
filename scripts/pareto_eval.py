"""Stage 3b §5: Pareto evaluation — fixed falseOcc budgets on VAL, applied
unchanged to TEST.

For each model and each falseOcc budget b ∈ {0.2%, 0.5%, 1%, 2%}: sweep the
occupancy decision threshold on VAL (per offset, computed on the pooled val
pairs) to hit b; apply that threshold unchanged to TEST; report dynIoU,
change precision/recall/F1, observed occupancy IoU, reveal metrics at each
budget. Results saved per (model, seed, split, episode, offset).

Statistical note: frames within an episode are correlated (15 Hz); metrics
are aggregated per episode first, and dispersion is reported over episodes —
adjacent frames are never treated as independent samples.
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
from cvwam.dynamics_dataset import DynamicsDataset, OFFSETS  # noqa: E402
from cvwam.models_tiny import TinyDynamicsNet, split_outputs  # noqa: E402
import train_tiny as tt  # noqa: E402

BUDGETS = (0.002, 0.005, 0.01, 0.02)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u > 0 else float("nan")


def collect(model, variant, ds, device):
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    recs = []
    for batch in dl:
        vox = batch["voxel"].to(device)
        act = batch["action"].to(device)
        qpos = batch["qpos"].to(device)
        oi = tt.offset_index(batch, device)
        with torch.no_grad():
            o = model(vox, qpos, act, oi) if model is not None else None
        blocks = split_outputs(o, tt.MODEL_DIR[variant], len(OFFSETS)) if model is not None else None
        for ki, k in enumerate(OFFSETS):
            sel = oi == ki
            if sel.sum() == 0:
                continue
            state_t = batch["state_t"].to(device)[sel]
            state_f = batch["state_future"].to(device)[sel]
            obs_f = batch["observed_future"].to(device).bool()[sel]
            cur_hit = (vox[:, 0] > 0.5)[sel]
            gt_hit = state_f == 0
            gt_free = state_f == 1
            if model is None:  # P0
                occ_logit = torch.full_like(cur_hit, 0.5, dtype=torch.float32) * \
                    (cur_hit.float() * 2 - 1) * 1e6   # +inf where current hit
                occ_logit = torch.where(cur_hit, torch.ones_like(cur_hit, dtype=torch.float32) * 1e6,
                                        -torch.ones_like(cur_hit, dtype=torch.float32) * 1e6)
            elif tt.MODEL_DIR[variant] == "decoupled":
                occ_logit = blocks[ki]["occ_logit"][sel].float()
            else:
                probs = torch.softmax(blocks[ki]["state_logits"][sel], dim=1)
                p_hit = probs[:, 0]
                occ_logit = torch.log(p_hit.clamp_min(1e-6) /
                                      (1 - p_hit).clamp_min(1e-6))
            recs.append(dict(offset=k, state_t=state_t.cpu(), obs=obs_f.cpu(),
                             gt_hit=gt_hit.cpu(), gt_free=gt_free.cpu(),
                             cur_hit=cur_hit.cpu(), occ_logit=occ_logit.cpu(),
                             episodes=batch["episode"]))
    return recs


def metrics_at(recs, thr_by_offset):
    """Per-rec metrics at given thresholds; returns rows with per-episode split."""
    rows = []
    for r in recs:
        thr = thr_by_offset[r["offset"]]
        pred_hit = r["occ_logit"] > thr
        obs = r["obs"]
        gt_hit, gt_free, cur_hit = r["gt_hit"], r["gt_free"], r["cur_hit"]
        occ_iou = iou(pred_hit & obs, gt_hit & obs)
        false_occ = float((pred_hit & gt_free).sum()) / float(max(gt_free.sum(), 1))
        both = obs & (r["state_t"] != 2)
        dyn = ((r["state_t"] == 0) ^ gt_hit) & both
        pred_flip = pred_hit ^ cur_hit
        # change P/R/F1 (exact match)
        tp = float((pred_flip & dyn).sum())
        prec = tp / float(max((pred_flip & both).sum(), 1))
        rec = tp / float(max(dyn.sum(), 1))
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        dyn_iou = iou(pred_hit & dyn, gt_hit & dyn)
        reveal = (r["state_t"] == 2) & obs
        rev_acc = float(((pred_hit == gt_hit) & reveal).sum()) / float(max(reveal.sum(), 1))
        ep = sorted(set(r["episodes"]))[0] if len(set(r["episodes"])) == 1 else "mixed"
        rows.append(dict(offset=r["offset"], episode=ep, occ_iou=occ_iou,
                         false_occ=false_occ, dyn_iou=dyn_iou,
                         change_p=prec, change_r=rec, change_f1=f1,
                         reveal_acc=rev_acc))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name=path pairs, e.g. p1=runs/stage3b/ckpt/p1_s42/best.pt; use p0 for persistence")
    ap.add_argument("--out", default="runs/stage3b/metrics/pareto.csv")
    args = ap.parse_args()
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    test_eps = sm["splits"]["clean_test"] + sm["splits"]["rand_test"]

    all_rows = []
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        variant = "p0" if path == "p0" else None
        model = None
        if path != "p0":
            ck = torch.load(path, map_location=device, weights_only=False)
            variant = ck["variant"]
            model = TinyDynamicsNet(variant=tt.MODEL_DIR[variant], base=ck["base"],
                                    n_offsets=len(OFFSETS)).to(device).eval()
            model.load_state_dict(ck["model"])
        val_recs = collect(model, variant, DynamicsDataset(root, val_eps), device)
        # calibrate on VAL per offset per budget
        thr_tables = {b: {} for b in BUDGETS}
        for k in OFFSETS:
            vr = [r for r in val_recs if r["offset"] == k]
            if not vr:
                continue
            gt_free_frac = float(np.mean([float(r["gt_free"][r["obs"]].float().mean())
                                          for r in vr]))
            for b in BUDGETS:
                # threshold such that VAL false_occ == b (b = frac of GT-free voxels
                # predicted occupied); sweep over quantiles of occ_logit within
                # GT-free & observed region
                thr_list = np.quantile(np.concatenate(
                    [r["occ_logit"][r["gt_free"] & r["obs"]].numpy().ravel()
                     for r in vr]), np.linspace(0, 1, 81))
                best_thr, best_err = thr_list[0], 1e9
                for thr in thr_list:
                    fo = np.mean([float(((r["occ_logit"] > thr) & r["gt_free"]).sum()) /
                                  max(float(r["gt_free"].sum()), 1) for r in vr])
                    err = abs(fo - b)
                    if err < best_err:
                        best_err, best_thr = err, float(thr)
                thr_tables[b][k] = best_thr
        test_recs = collect(model, variant, DynamicsDataset(root, test_eps), device)
        for b in BUDGETS:
            rows = metrics_at(test_recs, thr_tables[b])
            for r in rows:
                r.update({"model": name, "budget": b})
            all_rows.extend(rows)
        # also record the raw default-threshold operating point
        rows = metrics_at(test_recs, {k: 0.0 for k in OFFSETS})
        for r in rows:
            r.update({"model": name, "budget": "default"})
        all_rows.extend(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in all_rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"-> {out} ({len(all_rows)} rows)")
    # summary print: dynIoU at each budget
    import collections
    agg = collections.defaultdict(list)
    for r in all_rows:
        agg[(r["model"], r["budget"])].append(r)
    print(f"{'model':12s} {'budget':>8s} {'occIoU':>7s} {'dynIoU':>7s} {'chgF1':>7s} {'falseOcc':>8s}")
    for (m, b), rs in sorted(agg.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        print(f"{m:12s} {str(b):>8s} "
              f"{np.nanmean([r['occ_iou'] for r in rs]):7.3f} "
              f"{np.nanmean([r['dyn_iou'] for r in rs]):7.3f} "
              f"{np.nanmean([r['change_f1'] for r in rs]):7.3f} "
              f"{np.nanmean([r['false_occ'] for r in rs]):8.4f}")


if __name__ == "__main__":
    main()
