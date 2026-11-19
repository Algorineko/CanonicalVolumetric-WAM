"""Stage 3c training: P2 / P2-E / P3 with P1-V's decoupled supervision.

Same protocol as train_tiny.py (AdamW, zero-init head, best-by-val-loss
checkpoint) but the loss is FIXED to P1-V's formula (mean_all BCE_vis +
pos_weight * mean_obs BCE_occ, pos_weight=free/hit from train stats) so the
comparison isolates the INPUT REPRESENTATION.
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
from cvwam.dynamics_dataset import OFFSETS, class_weights_from  # noqa: E402
from cvwam.models_repr import PointDynamicsNet, PointmapDynamicsNet, count_params  # noqa: E402
from cvwam.repr_dataset import ReprDataset  # noqa: E402


def build_model(variant, base, cond_dim, grid):
    if variant in ("p2", "p2e"):
        return PointDynamicsNet(base=base, cond_dim=cond_dim, n_offsets=len(OFFSETS),
                                pt_in=4 if variant == "p2e" else 3,
                                grid=grid)
    return PointmapDynamicsNet(base=base, cond_dim=cond_dim,
                               n_offsets=len(OFFSETS), grid=grid)


def offset_index(batch, device):
    return torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)


def losses(batch, model, variant, pos_weight, device, grid):
    oi = offset_index(batch, device)
    qpos = batch["qpos"].to(device)
    act = batch["action"].to(device)
    state_f = batch["state_future"].to(device)
    obs_f = batch["observed_future"].to(device).bool()
    if variant in ("p2", "p2e"):
        pts = batch["points"].to(device)
        gm = batch["grid_meta"].to(device)
        b_min, pitch = gm[:, :3], gm[:, 3]
        out = model(pts, qpos, act, oi, b_min, pitch)
    else:
        pm = batch["pointmap"].float().to(device)
        camT = batch["camT"].to(device)
        K = batch["K"].to(device)                       # (B,3,3)
        K4 = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], -1)
        K4 = K4[:, None, :].expand(-1, 3, -1)           # (B,3,4)
        gm = batch["grid_meta"].to(device)
        b_min, pitch = gm[:, :3], gm[:, 3]
        out = model(pm, camT, K4, qpos, act, oi, b_min, pitch)
    blocks = []
    C = out.shape[1] // len(OFFSETS)
    for k in range(len(OFFSETS)):
        blocks.append({"occ_logit": out[:, k * C], "vis_logit": out[:, k * C + 1]})
    L = {}
    for k_idx, k in enumerate(OFFSETS):
        sel = (oi == k_idx)
        if sel.sum() == 0:
            continue
        tgt = state_f[sel]
        m_obs = obs_f[sel]
        occ_logit = blocks[k_idx]["occ_logit"][sel]
        vis_logit = blocks[k_idx]["vis_logit"][sel]
        occ_tgt = (tgt == 0).float()
        vis_tgt = m_obs.float()
        occ_loss = F.binary_cross_entropy_with_logits(
            occ_logit[m_obs], occ_tgt[m_obs], pos_weight=pos_weight)
        vis_loss = F.binary_cross_entropy_with_logits(vis_logit, vis_tgt)
        L[f"loss_k{k}"] = occ_loss + vis_loss
    L["loss"] = torch.stack(list(L.values())).mean()
    return L


@torch.no_grad()
def evaluate_quick(model, loader, variant, device, grid, max_batches=6):
    model.eval()
    vals = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        try:
            L = losses(batch, model, variant, torch.tensor(4.0, device=device),
                       device, grid)
        except FileNotFoundError:
            continue
        vals.append(L["loss"].item())
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--variant", required=True, choices=["p2", "p2e", "p3"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    vox_root = Path(cfg["cache_dir"]) / "voxels_ep"
    pt_root = Path(cfg["cache_dir"]) / "point_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    train_eps = sm["splits"]["clean_train"]
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]

    # grid meta + pos_weight from the voxel cache (train stats)
    w_state, counts = class_weights_from(vox_root, train_eps)
    pos_w = (w_state[0] / w_state[1]).detach().to(device)
    g0 = np.load(vox_root / "demo_clean_balanced" / "episode_0000010" / "20mm" /
                 "frame_0000000.npz")
    grid = tuple(int(v) for v in g0["shape"])
    print(f"grid={grid} pos_weight={float(pos_w):.2f}")

    ds = ReprDataset(vox_root, pt_root, train_eps, stride_t=1, repr=args.variant)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=6,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    val_ds = ReprDataset(vox_root, pt_root, val_eps, stride_t=2, repr=args.variant)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=3, pin_memory=True)

    model = build_model(args.variant, args.base, 128, grid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_par = count_params(model)
    print(f"variant={args.variant} params={n_par/1e6:.2f}M "
          f"train={len(ds)} val={len(val_ds)}")

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "train.log", "w")
    best_val = float("inf")
    t0 = time.perf_counter()
    step = 0
    while step < args.steps:
        for batch in dl:
            if step >= args.steps:
                break
            L = losses(batch, model, args.variant, pos_w, device, grid)
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
                v = evaluate_quick(model, val_dl, args.variant, device, grid)
                msg = f"step {step} val_loss {v:.4f}"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
                if v < best_val:
                    best_val = v
                    torch.save({"model": model.state_dict(), "variant": args.variant,
                                "step": step, "base": args.base, "best_val": v,
                                "grid": grid, "params": n_par},
                               run_dir / "best.pt")
    torch.save({"model": model.state_dict(), "variant": args.variant,
                "step": args.steps, "base": args.base, "best_val": best_val,
                "grid": grid, "params": n_par}, run_dir / "last.pt")
    meta = dict(variant=args.variant, steps=args.steps, batch=args.batch, lr=args.lr,
                base=args.base, params=n_par, train_pairs=len(ds),
                val_pairs=len(val_ds), best_val=best_val,
                wall_s=round(time.perf_counter() - t0, 1),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                seed=args.seed, train_eps=train_eps, val_eps=val_eps,
                loss="p1v_decoupled_posweight")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
