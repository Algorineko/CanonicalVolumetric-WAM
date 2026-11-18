"""Stage 3a training: P0 (persistence, no training) / P1 / P1-R / P1-V / act-only.

Protocol (plan.md §8.5 + user instructions):
  1) overfit mode: single episode (ep10), 500 steps — learning-signal check.
  2) split mode: train = clean 10/25/40, val = randomized 0, test = randomized 1
     (BY EPISODE; no adjacent-frame leakage possible).
GPU 0 only. fp32 (bf16 autocast was planned but NOT implemented — recorded
as a deviation in the report). Logs every step batch loss; eval every N steps
on a fixed val subset; checkpoints best-by-val.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam.dynamics_dataset import DynamicsDataset, OFFSETS, class_weights_from  # noqa: E402
from cvwam.models_tiny import TinyDynamicsNet, split_outputs, count_params  # noqa: E402

VARIANTS = {
    "p1": "state_ce",
    "p1_r": "residual",
    "p1_v": "decoupled",
    "act_only": "state_ce",       # same as p1 but voxel input zeroed
}
MODEL_DIR = {"p1": "state_ce", "p1_r": "residual", "p1_v": "decoupled",
             "act_only": "state_ce"}


def make_model(variant, base, cond_dim, device):
    m = TinyDynamicsNet(variant=MODEL_DIR[variant], base=base,
                        cond_dim=cond_dim, n_offsets=len(OFFSETS))
    return m.to(device)


def offset_index(batch, device):
    return torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)


def losses(batch, model, variant, w_state, pos_weight_occ, device):
    """Returns dict of scalar losses + aux tensors for metrics."""
    vox = batch["voxel"].to(device, non_blocking=True)
    if variant == "act_only":
        vox = torch.zeros_like(vox)
    qpos = batch["qpos"].to(device)
    act = batch["action"].to(device)
    oi = offset_index(batch, device)
    out = model(vox, qpos, act, oi)
    blocks = split_outputs(out, MODEL_DIR[variant], len(OFFSETS))
    state_f = batch["state_future"].to(device)      # (B,X,Y,Z) long 0/1/2
    obs_f = batch["observed_future"].to(device).bool()  # (B,X,Y,Z) observed mask
    L = {}
    # group samples by their offset so each head sees its own targets
    for k_idx, k in enumerate(OFFSETS):
        sel = (oi == k_idx)
        if sel.sum() == 0:
            continue
        tgt = state_f[sel]
        m_obs = obs_f[sel]
        if MODEL_DIR[variant] == "decoupled":
            occ_logit = blocks[k_idx]["occ_logit"][sel]
            vis_logit = blocks[k_idx]["vis_logit"][sel]
            # occupancy BCE only on future-observed voxels (unknown excluded)
            occ_tgt = (tgt == 0).float()          # hit=1, free=0
            occ_loss = F.binary_cross_entropy_with_logits(
                occ_logit[m_obs], occ_tgt[m_obs],
                pos_weight=pos_weight_occ)
            vis_tgt = m_obs.float()               # observed=1, unknown=0
            vis_loss = F.binary_cross_entropy_with_logits(vis_logit, vis_tgt)
            L[f"loss_k{k}"] = occ_loss + vis_loss
        else:
            logits = blocks[k_idx]["state_logits"][sel]   # (B',3,X,Y,Z)
            L[f"loss_k{k}"] = F.cross_entropy(logits, tgt, weight=w_state)
    L["loss"] = torch.stack(list(L.values())).mean()
    return L


def evaluate_quick(model, loader, variant, device, max_batches=6):
    """Cheap val proxy: mean masked-CE loss (state variants) or occ/vis BCE."""
    model.eval()
    vals = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            L = losses(batch, model, variant, None,
                       torch.tensor(4.0, device=device), device)
            vals.append(L["loss"].item())
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--variant", required=True, choices=list(VARIANTS) + ["p0"])
    ap.add_argument("--mode", default="split", choices=["overfit", "split"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride-t", type=int, default=1)
    args = ap.parse_args()
    if args.variant == "p0":
        print("P0 persistence needs no training."); return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    train_eps = ["demo_clean_balanced:10", "demo_clean_balanced:25",
                 "demo_clean_balanced:40"]
    val_eps = ["demo_randomized_balanced:0"]
    overfit_eps = ["demo_clean_balanced:10"]
    eps = overfit_eps if args.mode == "overfit" else train_eps

    w_state, counts = class_weights_from(root, eps)
    w_state = w_state.to(device)
    # occupancy pos_weight from train data: free/hit ratio within observed
    # occupancy BCE pos_weight up-weights the positive (hit) class by the
    # free/hit ratio (~8.15); VERIFICATION FIX: was w_state[1]/w_state[0]=0.123
    # (inverted), which crippled P1-V's occupancy head (see failures/)
    pos_w = (w_state[0] / w_state[1]).detach() if MODEL_DIR[args.variant] == "decoupled" \
        else torch.tensor(4.0, device=device)
    print(f"state counts (sampled): {counts}, CE weights: {w_state.tolist()}")

    ds = DynamicsDataset(root, eps, stride_t=args.stride_t)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True,
                    persistent_workers=True)
    val_ds = DynamicsDataset(root, val_eps) if args.mode == "split" else ds
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=2, pin_memory=True)

    model = make_model(args.variant, args.base, 128, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_par = count_params(model)
    print(f"variant={args.variant} mode={args.mode} params={n_par/1e6:.2f}M "
          f"train_pairs={len(ds)} val_pairs={len(val_ds)}")

    run_dir = Path(args.out or f"runs/stage3a/{args.variant}_{args.mode}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "train.log", "w")
    best_val = float("inf")
    t0 = time.perf_counter()
    step = 0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            L = losses(batch, model, args.variant, w_state, pos_w, device)
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 2**30
                msg = (f"step {step} loss {L['loss'].item():.4f} "
                       f"({time.perf_counter()-t0:.0f}s, peak {mem:.2f}GiB)")
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if step % 250 == 0 or step == args.steps:
                v = evaluate_quick(model, val_dl, args.variant, device)
                msg = f"step {step} val_loss {v:.4f}"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                if v < best_val:
                    best_val = v
                    torch.save({"model": model.state_dict(),
                                "variant": args.variant, "step": step,
                                "base": args.base, "best_val": v},
                               run_dir / "best.pt")
    torch.save({"model": model.state_dict(), "variant": args.variant,
                "step": args.steps,
                "base": args.base, "best_val": best_val},
               run_dir / "last.pt")
    meta = dict(variant=args.variant, mode=args.mode, steps=args.steps,
                batch=args.batch, lr=args.lr, base=args.base, params=n_par,
                train_pairs=len(ds), val_pairs=len(val_ds),
                best_val=best_val, wall_s=round(time.perf_counter() - t0, 1),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                torch=torch.__version__, seed=args.seed,
                train_eps=eps, val_eps=val_eps if args.mode == "split" else eps)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
