"""Stage 3b §5: Pareto evaluation — streaming rewrite.

v2 (memory-safe): v1 accumulated per-sample FULL-GRID logits across all val
pairs (~36 GB RAM) and was OOM-killed silently. v2 streams batches and:
  pass A (val): infer -> accumulate FREE-region logits only (~200MB/offset)
                + per-sample free counts -> pick thresholds per falseOcc budget
  pass B (val): infer again -> per-batch metrics at the chosen thresholds
  pass C (test): infer -> metrics at FIXED thresholds (never re-tuned)
Per (model, budget, offset, episode) mean rows; per-model CSV, resumable.
Clean/Rand test reported separately via test_group.
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


def dilate1(g):
    out = g.clone()
    for d in (1, 2, 3):
        for s in (-1, 1):
            out |= torch.roll(g, s, d)
    return out


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = TinyDynamicsNet(variant=tt.MODEL_DIR[ck["variant"]], base=ck["base"],
                        n_offsets=len(OFFSETS)).to(device).eval()
    m.load_state_dict(ck["model"])
    return ck["variant"], m


def infer_batch(model, variant, batch, device):
    """Returns occ_logit (B, grid) on CPU for one batch (any offset mix)."""
    vox = batch["voxel"].to(device, non_blocking=True)
    if model is None:  # P0 persistence
        return torch.where(vox[:, 0] > 0.5, torch.full_like(vox[:, 0], 1e6),
                           torch.full_like(vox[:, 0], -1e6)).float().cpu()
    act = batch["action"].to(device, non_blocking=True)
    qpos = batch["qpos"].to(device, non_blocking=True)
    oi = tt.offset_index(batch, device)
    with torch.no_grad():
        o = model(torch.zeros_like(vox) if variant == "act_only" else vox,
                  qpos, act, oi)
    blocks = split_outputs(o, tt.MODEL_DIR[variant], len(OFFSETS))
    occ = torch.empty_like(vox[:, 0])  # (B, grid) per-voxel occ logits
    for ki in range(len(OFFSETS)):
        sel = (oi == ki)
        if sel.sum() == 0:
            continue
        if tt.MODEL_DIR[variant] == "decoupled":
            occ_k = blocks[ki]["occ_logit"][sel].float()
        else:
            probs = torch.softmax(blocks[ki]["state_logits"][sel], dim=1)
            p_hit = probs[:, 0]
            occ_k = torch.log(p_hit.clamp_min(1e-6) / (1 - p_hit).clamp_min(1e-6))
        occ[sel] = occ_k
    return occ.float().cpu()


def batch_gt(batch):
    return dict(
        state_t=batch["state_t"], state_f=batch["state_future"],
        obs_f=batch["observed_future"].bool(),
        cur_hit=batch["voxel"][:, 0] > 0.5,
        episodes=batch["episode"])


def batch_metrics(occ, gt, thrs):
    """Per-sample metrics at each threshold; returns rows keyed (thr, sample)."""
    obs = gt["obs_f"]
    gt_hit = gt["state_f"] == 0
    gt_free = gt["state_f"] == 1
    both = obs & (gt["state_t"] != 2)
    dyn = ((gt["state_t"] == 0) ^ gt_hit) & both
    reveal = (gt["state_t"] == 2) & obs
    # threshold-independent parts of surf F1 (dilated GT)
    gt_hit_d = dilate1(gt_hit)
    out = []
    flat = lambda x: x.flatten(1)
    obs_f = flat(obs)
    gt_f = flat(gt_hit).float()
    gt_d_f = flat(gt_hit_d).float()
    gt_free_f = flat(gt_free).float()
    dyn_f = flat(dyn)
    reveal_f = flat(reveal)
    n_free_s = flat(gt_free).sum(1)
    for thr in thrs:
        pred = occ > thr
        pred_obs = flat(pred & obs)
        # occ IoU: pred vs gt_hit restricted to obs
        p_o = pred_obs.float()
        inter = (p_o * gt_f).sum(1)
        uni = ((p_o + gt_f) > 0).float().sum(1)
        occ_iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
        # surf F1 @1 (matches f1_per_sample: pred undilated ∩ dilated GT)
        pred_d = flat(dilate1(pred & obs)).float()
        tp = (p_o * gt_d_f).sum(1)
        prec = tp / p_o.sum(1).clamp_min(1)
        rec = (gt_f * pred_d).sum(1) / gt_f.sum(1).clamp_min(1)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-9)
        valid = (p_o.sum(1) > 0) & (gt_f.sum(1) > 0)
        surf_f1 = torch.where(valid, f1, float("nan"))
        pf = pred_obs.bool()
        false_occ = (pf & gt_free_f.bool()).sum(1).float() / n_free_s.clamp_min(1).float()
        # dyn IoU = IoU(pred_hit & dyn, gt_hit & dyn)  (original semantics)
        a_dyn = pf & dyn_f
        b_dyn = dyn_f & gt_f.bool()
        dinter = (a_dyn & b_dyn).sum(1).float()
        duni = (a_dyn | b_dyn).sum(1).float()
        dyn_iou = torch.where(duni > 0, dinter / duni.clamp_min(1), float("nan"))
        flip_f = flat(pred ^ gt["cur_hit"].bool())
        chg_prec = (flip_f & dyn_f).sum(1).float() / flip_f.sum(1).clamp_min(1).float()
        chg_rec = (flip_f & dyn_f).sum(1).float() / dyn_f.sum(1).clamp_min(1).float()
        chg_f1 = 2 * chg_prec * chg_rec / (chg_prec + chg_rec).clamp_min(1e-9)
        rev_f = flat((pred == gt_hit) & reveal)
        rev_acc = rev_f.sum(1).float() / reveal_f.sum(1).clamp_min(1).float()
        out.append(dict(occ_iou=occ_iou, surf_f1=surf_f1, false_occ=false_occ,
                        dyn_iou=dyn_iou, chg_prec=chg_prec, chg_rec=chg_rec,
                        chg_f1=chg_f1, rev_acc=rev_acc))
    return out


def calibrate(val_loader, model, variant, device):
    """Pass A: free-logit quantiles per offset -> thresholds per budget."""
    free_logits = {k: [] for k in OFFSETS}
    free_counts = {k: [] for k in OFFSETS}
    for bi, batch in enumerate(val_loader):
        occ = infer_batch(model, variant, batch, device)
        gt = batch_gt(batch)
        is_free = (gt["state_f"] == 1) & gt["obs_f"]
        oi = tt.offset_index(batch, torch.device("cpu"))
        for ki, k in enumerate(OFFSETS):
            sel = (oi == ki)
            if sel.sum() == 0:
                continue
            free_logits[k].append(occ[sel][is_free[sel]].numpy().ravel())
            free_counts[k].append(is_free[sel].flatten(1).sum(1).numpy())
        if bi % 50 == 0:
            print(f"  [cal {bi}/{len(val_loader)}]", flush=True)
    thr_tables = {b: {} for b in BUDGETS}
    thr_tables["default"] = {k: 0.0 for k in OFFSETS}
    for k in OFFSETS:
        if not free_counts[k]:
            continue
        lg = np.concatenate(free_logits[k])
        seg = np.concatenate([np.full(int(c.sum()), i, dtype=np.int32)
                              for i, c in enumerate(free_counts[k])])
        nf = np.concatenate(free_counts[k]).astype(np.float64)
        # dense grid: 201 quantiles + midpoints between consecutive distinct
        # values (v1's 41-quantile grid skipped the budget-crossing interval
        # and degenerated to fo=0 operating points — archived in
        # failures/pareto_v2_sparse_grid/)
        qs = np.quantile(lg, np.linspace(0, 1, 201))
        qs = np.unique(qs)
        mids = (qs[1:] + qs[:-1]) / 2
        grid = np.unique(np.concatenate([qs, mids, [0.0]]))
        fos = []
        for thr in grid:
            cnt = np.bincount(seg, weights=(lg > thr).astype(np.float64),
                              minlength=len(nf))
            fos.append(np.mean(cnt / np.maximum(nf, 1)))
        fos = np.asarray(fos)
        for b in BUDGETS:
            feasible = np.nonzero(fos <= b)[0]
            if len(feasible):
                # recall-max under budget: largest fo still <= b
                i = feasible[np.argmax(fos[feasible])]
            else:  # nothing meets the budget -> most conservative (min fo)
                i = int(np.argmin(fos))
            thr_tables[b][k] = float(grid[i])
            print(f"  [thr] k={k} budget={b} thr={grid[i]:.3f} "
                  f"(fo={fos[i]:.4f})", flush=True)
        free_logits[k] = None
    return thr_tables


def evaluate(loader, model, variant, thr_tables, device, split_name):
    """Pass B/C: per (budget, offset, episode) mean of per-sample metrics."""
    acc = {}  # (budget, offset, episode) -> list of metric dicts
    blist = list(BUDGETS) + ["default"]
    for bi, batch in enumerate(loader):
        occ = infer_batch(model, variant, batch, device)
        gt = batch_gt(batch)
        oi = tt.offset_index(batch, torch.device("cpu"))
        for ki, k in enumerate(OFFSETS):
            sel = (oi == ki)
            if sel.sum() == 0:
                continue
            idx = torch.nonzero(sel).flatten()
            gt_k = {}
            for key, v in gt.items():
                if torch.is_tensor(v):
                    gt_k[key] = v[idx]
                else:
                    gt_k[key] = [v[i] for i in idx.tolist()]
            occ_k = occ[idx]
            thrs_k = [thr_tables[b][k] for b in blist]
            met = batch_metrics(occ_k, gt_k, thrs_k)
            eps_k = gt_k["episodes"]
            for ti, b in enumerate(blist):
                for j in range(len(eps_k)):
                    key = (str(b), k, eps_k[j])
                    acc.setdefault(key, []).append(
                        {m: float(v[j]) for m, v in met[ti].items()})
        if bi % 50 == 0:
            print(f"  [{split_name} {bi}/{len(loader)}]", flush=True)
    rows = []
    for (b, k, ep), ms in acc.items():
        tg = ("rand_" if "randomized" in ep else "clean_") + split_name
        row = dict(budget=b, split=split_name, test_group=tg if
                   split_name == "test" else "", episode=ep, offset=k, n=len(ms))
        for m in ("occ_iou", "surf_f1", "false_occ", "dyn_iou", "chg_prec",
                  "chg_rec", "chg_f1", "rev_acc"):
            vals = [x[m] for x in ms if not np.isnan(x[m])]
            row[m] = round(float(np.mean(vals)), 5) if vals else ""
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out-dir", default="runs/stage3b/metrics/pareto")
    ap.add_argument("--stride-test", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    test_eps = sm["splits"]["clean_test"] + sm["splits"]["rand_test"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_ds = DynamicsDataset(root, val_eps)
    test_ds = DynamicsDataset(root, test_eps, stride_t=args.stride_test)
    print(f"[data] val {len(val_ds)} pairs, test {len(test_ds)} pairs", flush=True)

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        csv_path = out_dir / f"pareto_{name}.csv"
        if csv_path.exists():
            print(f"[skip] {name} (resume)", flush=True)
            continue
        print(f"[start] {name}", flush=True)
        variant, model = (None, None) if path == "p0" else load_model(path, device)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=8, pin_memory=True)
        thr_tables = calibrate(val_loader, model, variant, device)
        rows = evaluate(val_loader, model, variant, thr_tables, device, "val")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=8, pin_memory=True)
        rows += evaluate(test_loader, model, variant, thr_tables, device, "test")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({c for r in rows for c in r}))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[done] {name}: {len(rows)} rows -> {csv_path}", flush=True)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
