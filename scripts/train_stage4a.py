"""Stage 4a training: A0/A1/A2/A3/A3-mask Action/World FM variants.

Protocol (AdamW 3e-4 wd 1e-4, grad-clip 1.0, fp32). Batch 32, 3000 steps
(extend to 4000 iff TRAIN loss improved >=2% over last 500 steps — the
original "val-ish" wording was inaccurate; AUDIT-S6). Every 250 steps a fixed
val subset gets a full noise-to-action x0jump evaluation; best-by-val is saved
as best_val.pt (AUDIT-S6: the old code saved the FINAL weights as best.pt with
no val selection — historical best.pt files are final checkpoints, re-labeled
by the Model Integrity Audit). Final weights: last.pt (+ best.pt legacy alias
for serving compatibility). Encoder AE-initialized from Stage A.

World FM target = pooled_occupancy(x_future) (pooled canvas, v4 protocol).
Diagnostics (not trained on): decoded w0 masked dynIoU (formula fixed under
AUDIT-S7); action x0-MSE @ t in {0.5,0.9}; learned gates (a3mask).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.dynamics_dataset import OFFSETS  # noqa: E402
from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from cvwam.models_tiny import count_params  # noqa: E402

VARIANTS = ("a0", "a1", "a2", "a3", "a3mask")




def pooled_occupancy(x_fut, out_shape=(10, 17, 12)):
    """(B,4,40,65,45) -> (B,2,10,17,12): [hit_frac, obs_frac] via adaptive pool.

    Bounded [0,1] world-FM target — physically meaningful (fraction of each
    latent cell occupied/observed), no AE-latent scale pathology.
    """
    hit = x_fut[:, 0:1]
    obs = x_fut[:, 2:3]
    h = torch.nn.functional.adaptive_avg_pool3d(hit, out_shape)
    o = torch.nn.functional.adaptive_avg_pool3d(obs, out_shape)
    return torch.cat([h, o], 1)


def unpool_soft(w0, out_shape=(40, 65, 45)):
    """(B,2,10,17,12) soft canvas -> (B,2,40,65,45) [hit,obs] logits-ish maps."""
    return torch.nn.functional.interpolate(w0, size=out_shape, mode="trilinear",
                                           align_corners=False)


def soft_masked_dyn_iou(vox_soft, batch, thr=0.5):
    """vox_soft: (B,2,40,65,45) [hit,obs] soft maps in [0,1]."""
    obs_f = batch["observed_future"].to(vox_soft.device).bool()
    gt_hit = batch["state_future"].to(vox_soft.device) == 0
    st = batch["state_t"].to(vox_soft.device)
    pred_hit = vox_soft[:, 0] > thr
    both = obs_f & (st != 2)
    dyn = ((st == 0) ^ gt_hit) & both
    inter = (pred_hit & dyn).sum((1, 2, 3)).float()
    uni = ((pred_hit & dyn) | (gt_hit & dyn)).sum((1, 2, 3)).float()
    iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
    ok = ~torch.isnan(iou)
    return float(iou[ok].mean()) if ok.any() else float("nan")



def act_normalization(root, train_eps, stats_path="runs/stage4a/act_stats.json",
                      binding=None):
    """Per-dim affine stats over ALL train frame chunks -> <stats_path>.

    AUDIT-S6: new files embed a binding dict {task, train_eps}; loading with a
    mismatched binding aborts (silently reusing another task's stats was a
    real correctness bug — open_laptop prep, 2026-11-20). Legacy files without
    a binding are accepted with a warning.
    """
    stats_path = Path(stats_path)
    if stats_path.exists():
        st = json.loads(stats_path.read_text())
        b = st.get("binding")
        if b is not None and binding is not None and b != binding:
            raise SystemExit(f"[AUDIT-S6] act_stats binding mismatch: "
                             f"{stats_path} bound to {b}, current {binding}")
        if b is None:
            print(f"[AUDIT-S6] warning: {stats_path} has no binding record "
                  f"(legacy file); verify it matches this task/split manually")
        return st
    lo = np.full(14, np.inf)
    hi = np.full(14, -np.inf)
    from cvwam.dynamics_dataset import load_episode
    for spec in train_eps:
        variant, ep = spec.split(":")
        edir, mf = load_episode(Path(root), variant, int(ep))
        for t in range(mf["T_total"]):
            a = np.load(edir / f"frame_{t:07d}.npz")["action_chunk"]
            lo = np.minimum(lo, a.min(0))
            hi = np.maximum(hi, a.max(0))
    scale = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    stats = dict(lo=lo.tolist(), hi=hi.tolist(), scale=scale.tolist(),
                 binding=binding)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def normalize_act(a, stats):
    lo = np.asarray(stats["lo"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    return np.clip((a - lo) / scale, 0.0, 1.0)


@torch.no_grad()
def val_x0jump_mse(model, fixed_batches, stats, device):
    """AUDIT-S6: full noise-to-action x0jump (serving protocol) on fixed val
    batches with a fixed noise seed — deterministic selection metric for
    best_val.pt."""
    model.eval()
    errs = []
    for batch in fixed_batches:
        x = batch["voxel"].to(device)
        qpos = batch["qpos"].to(device)
        act = torch.from_numpy(normalize_act(
            batch["action"].numpy(), stats)).to(device)
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)
        torch.manual_seed(1234)
        eps = torch.randn_like(act)
        world = None
        if model.variant in ("a2", "a3", "a3mask"):
            torch.manual_seed(1234)
            world = torch.randn(batch["voxel"].shape[0], 2, 10, 17, 12, device=device)
            o = model(x, eps, torch.tensor([1.0] * x.shape[0], device=device),
                      qpos, oi, noisy_world=world, cam72=None)
            world = world - o["v_w"]  # w0 estimate, then re-call (serving)
        a = eps.clone()
        o = model(x, a, torch.tensor([1.0] * x.shape[0], device=device),
                  qpos, oi, noisy_world=world, cam72=None)
        a = (eps - o["v_a"]).clamp(0, 1)
        errs.append(float(((a - act) ** 2).mean()))
    model.train()
    return float(np.mean(errs)) if errs else float("nan")


def denormalize_act(a_norm, stats):
    lo = np.asarray(stats["lo"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    return lo + a_norm * scale


def sample_time(B, device):
    return 0.001 + 0.998 * torch.distributions.Beta(
        torch.tensor([1.5]), torch.tensor([1.0])).sample((B,)).to(device).squeeze(-1)


def fm_loss(v_hat, x_t, target):
    return nn.functional.mse_loss(v_hat, target)


@torch.no_grad()
def masked_dyn_iou(world_pred_voxel, batch, thr=0.5):
    """world_pred_voxel: (B,4,40,65,45) logits-ish [hit,free,obs,vc]."""
    obs_f = batch["observed_future"].to(world_pred_voxel.device).bool()
    gt_hit = batch["state_future"].to(world_pred_voxel.device) == 0
    st = batch["state_t"].to(world_pred_voxel.device)
    pred_hit = world_pred_voxel[:, 0] > thr
    both = obs_f & (st != 2)
    dyn = ((st == 0) ^ gt_hit) & both
    inter = (pred_hit & dyn).sum((1, 2, 3)).float()
    uni = ((pred_hit & dyn) | (gt_hit & dyn)).sum((1, 2, 3)).float()
    iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
    ok = ~torch.isnan(iou)
    return float(iou[ok].mean()) if ok.any() else float("nan")


@torch.no_grad()
def diagnostics(model, batch, stats, ae_dec, device):
    """w0 decode dynIoU + action x0-MSE @ t 0.5/0.9 + observer-on-GT (a3)."""
    out = {}
    x = batch["voxel"].to(device)
    qpos = batch["qpos"].to(device)
    act = batch["action"].to(device)
    B = x.shape[0]
    x_fut = batch["voxel_future"].to(device)
    for t_val in (0.5, 0.9):
        t = torch.full((B,), t_val, device=device)
        eps_a = torch.randn_like(act)
        x_t_a = (1 - t)[:, None, None] * act + t[:, None, None] * eps_a
        noisy_w = None
        if model.variant in ("a2", "a3", "a3mask"):
            tgt_w = pooled_occupancy(x_fut)
            eps_w = torch.randn_like(tgt_w)
            noisy_w = (1 - t)[:, None, None, None, None] * tgt_w \
                + t[:, None, None, None, None] * eps_w
        o = model(x, x_t_a, t, qpos,
                  torch.zeros(B, dtype=torch.long, device=device),
                  noisy_world=noisy_w)
        x0 = x_t_a - t[:, None, None] * o["v_a"]
        out[f"act_x0_mse_t{t_val}"] = float(nn.functional.mse_loss(
            x0, act, reduction="mean"))
    if model.variant in ("a2", "a3", "a3mask"):
        tgt_w = pooled_occupancy(x_fut)
        t = torch.full((B,), 0.5, device=device)
        eps_w = torch.randn_like(tgt_w)
        x_t_w = (1 - t)[:, None, None, None, None] * tgt_w + t[:, None, None, None, None] * eps_w
        cam72 = batch.get("cam72")
        cam72 = cam72.to(device) if cam72 is not None else None
        o = model(x, x_t_a, t, qpos,
                  torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device),
                  noisy_world=x_t_w, cam72=cam72)
        w0 = x_t_w - t[:, None, None, None, None] * o["v_w"]
        vox = unpool_soft(w0, x.shape[2:])           # soft (2,40,65,45): hit,obs
        out["w0_soft_dynIoU_t0.5"] = round(soft_masked_dyn_iou(vox, batch), 4)
        if "obs_logits" in o and o["obs_logits"] is not None:
            vis_p = o["obs_logits"][:, OFFSETS.index(batch["offset"][0])][0] > 0
            vis_g = batch["observed_future"][0].bool()
            out["observer_iou_sample"] = round(float((vis_p & vis_g).sum() /
                                                     max((vis_p | vis_g).sum(), 1)), 4)
    return out


def encode_frozen(model, x, latent_stats=None):
    with torch.no_grad():
        _, _, z = model.enc(x)
        if model.neck is not None:
            z = model.neck(z)
    z = z.detach()
    if latent_stats is not None:
        z = (z - latent_stats["mu"]) / latent_stats["sd"]
    return z


def decode_frozen(model, z_norm, ae_dec, latent_stats=None):
    if latent_stats is not None:
        z = z_norm * latent_stats["sd"] + latent_stats["mu"]
    else:
        z = z_norm
    return ae_dec(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lam-aux", type=float, default=1.0)
    ap.add_argument("--lam-world", type=float, default=1.0)
    ap.add_argument("--lam-obs", type=float, default=1.0)
    ap.add_argument("--no-fm-loss", action="store_true",
                    help="§3 matched ablation: drop the action FM loss (aux-only); "
                         "same architecture/data/steps as a1 — isolates the FM "
                         "co-training effect on the aux world prediction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--act-stats", default="runs/stage4a/act_stats.json",
                    help="task-specific action normalization stats json "
                         "(MUST be per-task; reusing another task's file "
                         "silently is a correctness bug)")
    ap.add_argument("--ae", default="runs/stage4a/tokenizer_ae/best.pt",
                    help="tokenizer AE checkpoint for encoder init")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    train_eps = sm["splits"]["clean_train"]
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    stats = act_normalization(root, train_eps, stats_path=args.act_stats,
                              binding={"task": cfg.get("task", "open_microwave"),
                                       "train_eps": train_eps})

    from cvwam.dynamics_dataset import DynamicsDataset
    train_ds = DynamicsDataset(root, train_eps)
    val_ds = DynamicsDataset(root, val_eps, stride_t=2)
    tr_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                       num_workers=6, drop_last=True, pin_memory=True)
    va_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                       num_workers=3, pin_memory=True)
    # AUDIT-S6: fixed val subset (first 4 batches, deterministic order) for
    # best_val selection; _VAL_STATS shared with val_x0jump_mse
    va_fixed = [b for i, b in enumerate(va_dl) if i < 4]

    ae = torch.load(args.ae, map_location="cpu",
                    weights_only=False)
    bottleneck = ae.get("bottleneck_ch", 128)
    model = Stage4AModel(variant=args.variant, bottleneck_ch=bottleneck).to(device)
    # AE-init the encoder (+neck) from Stage A weights
    missing, unexpected = model.enc.load_state_dict(ae["enc"], strict=False)
    print(f"enc AE-init: missing={len(missing)} unexpected={len(unexpected)}")
    if model.neck is not None and ae.get("neck"):
        model.neck.load_state_dict(ae["neck"])
    # AUDIT-S6 cleanup: E_frozen/latent normalization belonged to the retired
    # latent-space world FM (v3); the pooled-canvas pipeline (v4) never uses
    # them. Removed — see runs/model_integrity/source_review.md.
    # frozen AE decoder for diagnostics
    from train_tokenizer_ae import LatentDecoder
    ae_dec = LatentDecoder(in_ch=bottleneck, out_ch=4).to(device)
    ae_dec.load_state_dict(ae["dec"])
    ae_dec.eval()
    for p in ae_dec.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_par = count_params(model)
    frozen_par = 0  # E_frozen removed (AUDIT-S6: v3 dead code)
    print(f"variant={args.variant} params={n_par/1e6:.2f}M frozen={frozen_par/1e6:.2f}M "
          f"train={len(train_ds)} val={len(val_ds)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train.log", "w")
    best_val = float("inf")
    t0 = time.perf_counter()
    step = 0
    pos_weight = torch.tensor(8.15, device=device)
    steps_target = args.steps
    hist = []

    def run_epoch(dl, train=True):
        nonlocal step, steps_target
        if train:
            model.train()
        else:
            model.eval()
        while True:
            for batch in dl:
                if step >= steps_target:
                    return
                x = batch["voxel"].to(device)
                x_fut = batch["voxel_future"].to(device)
                qpos = batch["qpos"].to(device)
                act = torch.from_numpy(normalize_act(
                    batch["action"].numpy(), stats)).to(device)
                oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]],
                                  device=device)
                B = x.shape[0]
                t = sample_time(B, device)
                eps_a = torch.randn_like(act)
                x_t_a = (1 - t)[:, None, None] * act + t[:, None, None] * eps_a
                noisy_w, cam72 = None, None
                L = 0.0
                if args.variant in ("a2", "a3", "a3mask"):
                    tgt_w = pooled_occupancy(x_fut)
                    eps_w = torch.randn_like(tgt_w)
                    noisy_w = (1 - t)[:, None, None, None, None] * tgt_w \
                        + t[:, None, None, None, None] * eps_w
                with torch.no_grad():
                    cam72 = None  # camera params not in voxels_ep cache (a3 obs aux off)
                o = model(x, x_t_a, t, qpos, oi,
                          noisy_world=noisy_w, cam72=cam72)
                v_target_a = eps_a - act
                if not args.no_fm_loss:
                    L = L + fm_loss(o["v_a"], x_t_a, v_target_a)
                if o.get("v_w") is not None:
                    v_target_w = eps_w - tgt_w
                    L = L + args.lam_world * fm_loss(o["v_w"], noisy_w, v_target_w)
                if o.get("aux_logits") is not None:
                    tgt = batch["state_future"].to(device)
                    m_obs = batch["observed_future"].to(device).bool()
                    logits = o["aux_logits"]
                    L_aux = 0.0
                    for ki, k in enumerate(OFFSETS):
                        sel = (oi == ki)
                        if sel.sum() == 0:
                            continue
                        occ_l = logits[sel, 2 * ki]      # P1-V convention [occ, vis]
                        vis_l = logits[sel, 2 * ki + 1]
                        L_aux = L_aux + torch.nn.functional.binary_cross_entropy_with_logits(
                            vis_l, m_obs[sel].float()) \
                            + args.lam_aux * torch.nn.functional.binary_cross_entropy_with_logits(
                                occ_l[m_obs[sel]], (tgt[sel] == 0).float()[m_obs[sel]],
                                pos_weight=pos_weight)
                    L = L + L_aux / len(OFFSETS)
                if train:
                    opt.zero_grad(set_to_none=True)
                    L.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                step += 1
                if step % 50 == 0 or step == 1:
                    mem = torch.cuda.max_memory_allocated() / 2**30
                    msg = (f"step {step} loss {L.item():.4f} "
                           f"({time.perf_counter()-t0:.0f}s, peak {mem:.2f}GiB)")
                    print(msg, flush=True); log.write(msg + "\n"); log.flush()
                if step % 250 == 0 or step == steps_target:
                    hist.append((step, float(L.item())))
                    if args.variant in ("a2", "a3", "a3mask"):
                        # separate world-term logging (the a2 explosion was invisible)
                        msg2 = (f"  [world] step {step} loss {L.item():.4f}")
                        print(msg2, flush=True); log.write(msg2 + "\n"); log.flush()
                    if len(hist) >= 3:
                        (s0, l0), _, (s2, l2) = hist[-3:]
                        if s2 >= 2000 and l0 > 0 and l2 <= l0 * 0.98 and steps_target < 4000:
                            steps_target = 4000
                            print(f"[extend] val-ish loss improving; steps -> 4000", flush=True)
                    msg = f"step {step} train_loss {L.item():.4f}"
                    print(msg, flush=True); log.write(msg + "\n"); log.flush()
                    torch.save({"model": model.state_dict(), "variant": args.variant,
                                "step": step, "bottleneck_ch": bottleneck,
                                "stats": stats}, out_dir / "last.pt")
                    # AUDIT-S6: real val selection — fixed val batches, serving
                    # x0jump protocol (deterministic seed), MSE vs GT
                    vm = val_x0jump_mse(model, va_fixed, stats, device)
                    msg_v = f"  [val-x0jump] step {step} mse {vm:.5f}"
                    print(msg_v, flush=True); log.write(msg_v + "\n"); log.flush()
                    if vm < best_val:
                        best_val = vm
                        torch.save({"model": model.state_dict(),
                                    "variant": args.variant, "step": step,
                                    "bottleneck_ch": bottleneck, "stats": stats,
                                    "val_x0jump_mse": vm},
                                   out_dir / "best_val.pt")

    run_epoch(tr_dl, train=True)
    # final diagnostics on one val batch
    batch = next(iter(va_dl))
    diag = diagnostics(model, batch, stats, ae_dec, device)
    meta = dict(variant=args.variant, steps=steps_target, batch=args.batch, lr=args.lr,
                params=n_par, frozen_params=frozen_par, seed=args.seed,
                lam=dict(aux=args.lam_aux, world=args.lam_world, obs=args.lam_obs),
                train_pairs=len(train_ds), wall_s=round(time.perf_counter() - t0, 1),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                diagnostics=diag, train_eps=train_eps, val_eps=val_eps)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save({"model": model.state_dict(), "variant": args.variant,
                "bottleneck_ch": bottleneck, "stats": stats,
                "step": steps_target}, out_dir / "last.pt")
    # legacy alias: serving chains historically load best.pt; it is the FINAL
    # checkpoint (NOT val-best) — best_val.pt is the val-selected one (AUDIT-S6)
    torch.save({"model": model.state_dict(), "variant": args.variant,
                "bottleneck_ch": bottleneck, "stats": stats,
                "step": steps_target}, out_dir / "best.pt")
    meta["best_val_x0jump_mse"] = best_val
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
