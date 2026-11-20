"""Stage 4a Stage A: tokenizer autoencoder + pre-registered validation gate.

enc1-3 (from models_tiny.TinyDynamicsNet) + SKIP-FREE LatentDecoder
(128->64->64->4ch; skips would bypass the 10x17x12 bottleneck and make the
reconstruction validation vacuous). Input = current voxel only (no pairs, no
futures — no leakage).

Loss: BCE_hit(observed) + BCE_free(observed) + BCE_obs(all) + MSE_vc(observed)
Gate (val, pre-registered): vis recon IoU >= 0.85 AND occ recon IoU >= 0.70.
Pass -> freeze a copy as E_frozen; all Stage 4a variants AE-initialize.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam.dynamics_dataset import DynamicsDataset  # noqa: E402
from cvwam.models_tiny import count_params  # noqa: E402

OUT = Path("runs/stage4a/tokenizer_ae")


class LatentDecoder(nn.Module):
    """Skip-free decoder from the enc3 latent back to voxel channels."""

    def __init__(self, in_ch=128, out_ch=4, width=64, cond_dim=0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, width, 3, padding=1)
        self.conv2 = nn.Conv3d(width, width, 3, padding=1)
        self.head = nn.Conv3d(width, out_ch, 1)
        self.cond = None
        if cond_dim:
            self.cond = nn.Linear(cond_dim, width)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z, cond=None, out_shape=None):
        h = torch.nn.functional.silu(self.conv1(z))
        h = torch.nn.functional.interpolate(h, scale_factor=2, mode="trilinear",
                                            align_corners=False)
        h = torch.nn.functional.silu(self.conv2(h))
        if out_shape is not None:
            h = torch.nn.functional.interpolate(h, size=out_shape, mode="trilinear",
                                                align_corners=False)
        else:
            h = torch.nn.functional.interpolate(h, scale_factor=2, mode="trilinear",
                                                align_corners=False)
        if self.cond is not None and cond is not None:
            h = h + self.cond(cond).view(-1, h.shape[1], 1, 1, 1)
        return self.head(h)


def recon_ious(recon, gt):
    """recon: (4,H,W,D) logits; gt: (4,H,W,D) float channels [hit,free,obs,vc/3]."""
    hit_p = recon[0] > 0
    free_p = recon[1] > 0
    obs_p = recon[2] > 0
    hit_g = gt[0] > 0.5
    free_g = gt[1] > 0.5
    obs_g = gt[2] > 0.5

    def iou(a, b):
        u = (a | b).sum().item()
        return (a & b).sum().item() / u if u else float("nan")

    return dict(vis_iou=iou(obs_p, obs_g), occ_iou=iou(hit_p, hit_g),
                free_iou=iou(free_p, free_g))


@torch.no_grad()
def evaluate(model, dl, device, max_batches=40):
    model.eval()
    agg = {}
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        x = batch["voxel"].to(device)
        recon = model["dec"](model["enc"](x),
                             out_shape=tuple(x.shape[2:]))
        for j in range(x.shape[0]):
            for k, v in recon_ious(recon[j], x[j]).items():
                if not np.isnan(v):
                    agg.setdefault(k, []).append(v)
    model.train()
    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--bottleneck-ch", type=int, default=128,
                    help="tokenizer-v2 fallback uses 256")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    train_eps = sm["splits"]["clean_train"]
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]

    train_ds = DynamicsDataset(root, train_eps)
    val_ds = DynamicsDataset(root, val_eps, stride_t=2)
    tr_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                       num_workers=6, drop_last=True, pin_memory=True)
    va_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                       num_workers=3, pin_memory=True)

    OUT.mkdir(parents=True, exist_ok=True)
    model = nn.ModuleDict({
        "enc": _Encoder(),
        "dec": LatentDecoder(in_ch=args.bottleneck_ch),
    }).to(device)
    # if bottleneck widened, enc3 output must match: enc3 is 128-ch; the
    # tokenizer-v2 variant inserts a 1x1 Conv(128->bottleneck) after enc3.
    if args.bottleneck_ch != 128:
        model["neck"] = nn.Conv3d(128, args.bottleneck_ch, 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n_par = count_params(model)
    print(f"tokenizer AE params {n_par/1e6:.3f}M bottleneck={args.bottleneck_ch}")

    log = open(OUT / "train.log", "w")
    t0 = time.perf_counter()
    step = 0
    while step < args.steps:
        for batch in tr_dl:
            if step >= args.steps:
                break
            x = batch["voxel"].to(device)
            hit, free, obs = x[:, 0], x[:, 1], x[:, 2]
            vc = x[:, 3]
            z = model["enc"](x)
            if "neck" in model:
                z = model["neck"](z)
            recon = model["dec"](z, out_shape=tuple(x.shape[2:]))
            obs_b = obs > 0.5
            # occupancy pos_weight = free/hit within observed (project lesson L5:
            # unweighted BCE collapses sparse hits — same failure class as the
            # stage3a pos_weight bug)
            hit_frac = (hit[obs_b] > 0.5).float().mean().clamp(min=1e-4)
            pw = (1.0 - hit_frac) / hit_frac
            l_hit = nn.functional.binary_cross_entropy_with_logits(
                recon[:, 0][obs_b], hit[obs_b], pos_weight=pw)
            l_free = nn.functional.binary_cross_entropy_with_logits(
                recon[:, 1][obs_b], free[obs_b])
            l_obs = nn.functional.binary_cross_entropy_with_logits(
                recon[:, 2], obs)
            l_vc = nn.functional.mse_loss(recon[:, 3][obs_b], vc[obs_b])
            loss = l_hit + l_free + l_obs + l_vc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0 or step == 1:
                msg = f"step {step} loss {loss.item():.4f} ({time.perf_counter()-t0:.0f}s)"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()

    res = evaluate(model, va_dl, device)
    gate = dict(**res, bottleneck_ch=args.bottleneck_ch, params=n_par,
                vis_pass=bool(res["vis_iou"] >= 0.85),
                occ_pass=bool(res["occ_iou"] >= 0.70),
                gate_pass=bool(res["vis_iou"] >= 0.85 and res["occ_iou"] >= 0.70))
    (OUT / "meta.json").write_text(json.dumps(gate, indent=2))
    torch.save({"enc": model["enc"].state_dict(),
                "dec": model["dec"].state_dict(),
                "neck": model["neck"].state_dict() if "neck" in model else None,
                "bottleneck_ch": args.bottleneck_ch, "gate": gate},
               OUT / "best.pt")
    print(json.dumps(gate, indent=2))


class _Encoder(nn.Module):
    """enc1-3 verbatim from TinyDynamicsNet (in_ch=4, base=32)."""

    def __init__(self):
        super().__init__()
        c1, c2, c3 = 32, 64, 128
        self.enc1 = nn.Sequential(nn.Conv3d(4, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c3, c3, 3, padding=1), nn.SiLU())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e3


if __name__ == "__main__":
    main()
