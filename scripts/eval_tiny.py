"""Stage 3a evaluation harness (plan.md §10.3 geometry metrics + user list).

Metrics per (model, episode, offset k):
  observed_occupancy_iou : IoU(pred occupied, GT hit) within future-observed voxels
  surface_f1_tol1        : F1 of hit voxels, +-1 voxel dilation tolerance
  dynamic_region_iou     : IoU(pred occupied, GT occupied) within GT dynamic
                           region (hit/free flips between t and t+k, both obs)
  change_f1              : F1 of predicted physical flip mask (exact match)
  reveal_occ_iou         : IoU(pred hit, GT hit) within reveal region
                           (unknown_t -> observed_{t+k})
  reveal_occ_acc         : occupancy accuracy (hit vs free) in reveal region
  knownfree_false_occ    : pred occupied inside GT free_{t+k} / |GT free|
  (+ action-shuffle variant of the above, camera-dropout variant, latency, VRAM)

P0 persistence evaluated with the same code (prediction = current state).
Camera-dropout inputs are rebuilt on the fly from raw depth for a fixed
eval-frame subset; targets stay the full 3-view GT (hidden-view experiments
never receive the hidden cameras' observations as input).
"""

import argparse
import csv
import json
import sys
import time
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


def iou(a, b):
    inter = (a & b).sum()
    uni = (a | b).sum()
    return float(inter) / float(uni) if uni > 0 else float("nan")


def f1(pred, gt, tol=0):
    if tol:
        shifts = [torch.roll(torch.roll(gt, sx, 1), 0, 1) for sx in ()]
        gt_d = gt.clone()
        pred_d = pred.clone()
        for d in (1, 2, 3):
            for s in (-tol, tol):
                gt_d |= torch.roll(gt, s, d)
                pred_d |= torch.roll(pred, s, d)
    else:
        gt_d, pred_d = gt, pred
    tp = (pred & gt_d).sum()
    prec = float(tp) / float(pred.sum()) if pred.sum() > 0 else float("nan")
    rec = float((gt & pred_d).sum()) / float(gt.sum()) if gt.sum() > 0 else float("nan")
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        return float("nan")
    return 2 * prec * rec / (prec + rec)


def preds_for(batch, model, variant, device, shuffle_actions=False):
    """Returns per-offset dict: pred_hit (B,X,Y,Z) bool, pred_phys (B,..) bool."""
    vox = batch["voxel"].to(device)
    act = batch["action"].to(device)
    # action permutation is handled at the DATASET level (_ShuffleActionsWrapper):
    # in-batch permutation is a no-op because eval batches hold adjacent frames
    # with nearly identical chunks at 15Hz.
    _ = shuffle_actions
    qpos = batch["qpos"].to(device)
    oi = tt.offset_index(batch, device)
    out = {}
    if variant == "p0":
        hit_t = vox[:, 0] > 0.5
        for ki, k in enumerate(OFFSETS):
            sel = oi == ki
            out[k] = {"pred_hit": hit_t[sel],
                      "pred_phys_hit_changed": torch.zeros_like(hit_t[sel]),
                      "cur_hit": hit_t[sel], "cur_free": (vox[:, 1] > 0.5)[sel]}
        return out
    vox_in = torch.zeros_like(vox) if variant == "act_only" else vox
    with torch.no_grad():
        o = model(vox_in, qpos, act, oi)
    blocks = split_outputs(o, tt.MODEL_DIR[variant], len(OFFSETS))
    for ki, k in enumerate(OFFSETS):
        sel = oi == ki
        pred_vis = None
        if tt.MODEL_DIR[variant] == "decoupled":
            occ = blocks[ki]["occ_logit"][sel]
            vis = blocks[ki]["vis_logit"][sel]
            pred_hit = occ > 0
            pred_vis = vis > 0
        else:
            arg = blocks[ki]["state_logits"][sel].argmax(1)
            pred_hit = arg == 0
        cur_hit = (vox[:, 0] > 0.5)[sel]
        cur_free = (vox[:, 1] > 0.5)[sel]
        pred_flip = pred_hit ^ cur_hit          # hit assignment changed
        out[k] = {"pred_hit": pred_hit, "pred_phys_hit_changed": pred_flip,
                  "cur_hit": cur_hit, "cur_free": cur_free,
                  "pred_vis": pred_vis}
    return out


def metric_row(pred_hit, pred_flip, cur_hit, cur_free, state_t, state_f,
               obs_f, tol=1, pred_vis=None):
    """All boolean tensors shaped (N, X,Y,Z) flattened-batch on GPU."""
    gt_hit = state_f == 0
    gt_free = state_f == 1
    obs = obs_f.bool()
    # 1 observed occupancy IoU
    m1 = iou(pred_hit & obs, gt_hit & obs)
    # 2 surface F1 (tolerance) within observed region
    ph = pred_hit & obs
    gh = gt_hit & obs
    m2 = f1(ph, gh, tol=tol)
    # 3 dynamic region: hit/free flips between t and future among doubly-observed
    both_obs = obs & (state_t != 2)
    phys_t = state_t == 0                     # occupied at t
    phys_f = gt_hit
    dyn = (phys_t ^ phys_f) & both_obs
    m3 = iou(pred_hit & dyn, gt_hit & dyn)
    # 4 change F1 (exact)
    gt_flip = (phys_t ^ phys_f) & both_obs
    m4 = f1(pred_flip & both_obs, gt_flip, tol=0)
    # 5/6 reveal region: unknown_t -> observed_f
    reveal = (state_t == 2) & obs
    m5 = iou(pred_hit & reveal, gt_hit & reveal)
    acc_den = reveal.sum()
    if acc_den > 0:
        acc = float(((pred_hit == gt_hit) & reveal).sum()) / float(acc_den)
    else:
        acc = float("nan")
    # 7 known-free false occupancy
    kf = gt_free
    m7 = float((pred_hit & kf).sum()) / float(kf.sum()) if kf.sum() > 0 else float("nan")
    out = dict(observed_occupancy_iou=m1, surface_f1_tol1=m2,
               dynamic_region_iou=m3, change_f1=m4, reveal_occ_iou=m5,
               reveal_occ_acc=acc, knownfree_false_occ=m7)
    if pred_vis is not None:
        out["vis_iou"] = iou(pred_vis, obs)   # visibility head vs future observed
    return out


class _ShuffleActionsWrapper:
    """Replace each sample's action chunk with a FOREIGN one (cyclic shift by
    half the dataset; for ~82% of samples the donor is from a DIFFERENT
    episode, for the rest a distant phase of the same episode). This is a
    deliberately STRONGER control than same-episode resampling: the actions
    are wrong for the scene state, which is exactly what an action-sensitivity
    check wants. Adjacent-frame permutation inside an eval batch would be a
    no-op at 15Hz (consecutive chunks nearly identical), hence the large
    shift."""

    def __init__(self, base_ds):
        self.base = base_ds
        self.items = base_ds.items
        n = len(base_ds)
        shift = max(n // 2, 1)
        self.perm = [(i + shift) % n for i in range(n)]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        own = self.base[idx]
        foreign = self.base[self.perm[idx]]
        own["action"] = foreign["action"]     # foreign action chunk
        return own


class _DropoutWrapper:
    """DynamicsDataset variant that swaps the INPUT voxel for one built from a
    camera subset (voxels_ep_dropout). Targets stay the full 3-view GT."""

    def __init__(self, base_ds, subset_root, subset):
        self.base = base_ds
        self.root = Path(subset_root) / subset
        self.subset = subset
        # keep only timesteps that have a dropout-voxel file; index the base
        # dataset by ORIGINAL position (its __getitem__ is position-based)
        keep = []
        for i, (edir, t, k, spec) in enumerate(base_ds.items):
            variant, ep = spec.split(":")
            p = self.root / variant / f"episode_{int(ep):07d}" / "20mm" / \
                f"frame_{t:07d}.npz"
            if p.exists():
                keep.append(i)
        self.keep = keep
        self.items = [base_ds.items[i] for i in keep]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx):
        i = self.keep[idx]
        item = self.base[i]
        edir, t, k, spec = self.base.items[i]
        variant, ep = spec.split(":")
        p = self.root / variant / f"episode_{int(ep):07d}" / "20mm" / \
            f"frame_{t:07d}.npz"
        d = np.load(p)
        x = np.stack([d["hit"].astype(np.float32), d["free"].astype(np.float32),
                      d["observed"].astype(np.float32),
                      d["view_count"].astype(np.float32) / 3.0])
        item["voxel"] = torch.from_numpy(x)
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint path or 'p0'")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shuffle-actions", action="store_true",
                    help="swap action chunks across distant motion phases")
    ap.add_argument("--input-dropout", default=None,
                    choices=["head_only", "head_left", "head_right"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"

    variant = args.variant
    model = None
    if args.ckpt != "p0":
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        variant = variant or ck["variant"]
        model = TinyDynamicsNet(variant=tt.MODEL_DIR[variant], base=ck["base"],
                                n_offsets=len(OFFSETS)).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
    assert variant, "variant required for p0"

    ds = DynamicsDataset(root, args.episodes)
    if args.shuffle_actions:
        ds = _ShuffleActionsWrapper(ds)
    if args.input_dropout:
        ds = _DropoutWrapper(ds, Path(cfg["cache_dir"]) / "voxels_ep_dropout",
                             args.input_dropout)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4,
                    pin_memory=True)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n_batches = 0
    for batch in dl:
        P = preds_for(batch, model, variant, device,
                      shuffle_actions=args.shuffle_actions)
        state_t = batch["state_t"].to(device)
        state_f = batch["state_future"].to(device)
        obs_f = batch["observed_future"].to(device)
        for ki, k in enumerate(OFFSETS):
            sel = (torch.tensor([OFFSETS.index(x) for x in batch["offset"]],
                                device=device) == ki)
            if sel.sum() == 0:
                continue
            r = metric_row(P[k]["pred_hit"], P[k].get("pred_phys_hit_changed",
                                                      P[k]["pred_hit"]),
                           P[k]["cur_hit"], P[k]["cur_free"],
                           state_t[sel], state_f[sel], obs_f[sel],
                           pred_vis=P[k].get("pred_vis"))
            r.update({"offset": k, "n": int(sel.sum()),
                      "episodes": ",".join(sorted(set(batch["episode"])))})
            rows.append(r)
        n_batches += 1
    latency = (time.perf_counter() - t0) / max(n_batches, 1)
    mem = torch.cuda.max_memory_allocated() / 2**30
    tag = args.tag or variant
    outdir = Path("runs/stage3a/metrics")
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = args.out_suffix
    if not suffix:
        if args.shuffle_actions:
            suffix = "_shuffle"
        elif args.input_dropout:
            suffix = f"_drop_{args.input_dropout}"
    csv_path = outdir / f"eval_{tag}{suffix}.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"eval {tag}{suffix}: {len(rows)} rows, latency {latency*1000:.0f} ms/batch, "
          f"peak {mem:.2f} GiB -> {csv_path}")


if __name__ == "__main__":
    main()
