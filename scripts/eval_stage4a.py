"""Stage 4a offline eval: per-variant diagnostics over the full val set.

Reports: action x0-MSE @ t 0.5/0.9 (denormalized, rad), world w0 masked dynIoU
per offset (a2/a3/a3mask via frozen AE decoder), aux-head masked dynIoU (a1),
observer visIoU + observer-on-GT-geometry upper bound (a3/a3mask), learned
gates (a3mask). Writes runs/stage4a/metrics/eval_{variant}_s{seed}.json.
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
from cvwam.dynamics_dataset import OFFSETS, DynamicsDataset  # noqa: E402
from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from train_stage4a import (act_normalization, encode_frozen,  # noqa: E402
                           masked_dyn_iou, normalize_act, sample_time)
from train_tokenizer_ae import LatentDecoder  # noqa: E402


def iou(pred, gt):
    u = (pred | gt).sum().item()
    return (pred & gt).sum().item() / u if u else float("nan")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--max-batches", type=int, default=40)
    args = ap.parse_args()
    device = "cuda:0"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    ds = DynamicsDataset(root, val_eps, stride_t=2)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = Stage4AModel(variant=ck["variant"],
                         bottleneck_ch=ck.get("bottleneck_ch", 128)).to(device).eval()
    model.load_state_dict(ck["model"])
    stats = ck["stats"]
    ae = torch.load("runs/stage4a/tokenizer_ae/best.pt", map_location="cpu",
                    weights_only=False)
    ae_dec = LatentDecoder(in_ch=ck.get("bottleneck_ch", 128), out_ch=4).to(device)
    ae_dec.load_state_dict(ae["dec"])
    ae_dec.eval()

    agg = dict(act_mse=[], world_dyn_iou=[], aux_dyn_iou=[], obs_iou=[],
               obs_iou_gt_geom=[], gates=None)
    for bi, batch in enumerate(dl):
        if bi >= args.max_batches:
            break
        x = batch["voxel"].to(device)
        x_fut = batch["voxel_future"].to(device)
        qpos = batch["qpos"].to(device)
        act_raw = batch["action"].numpy()
        act = torch.from_numpy(normalize_act(act_raw, stats)).to(device)
        B = x.shape[0]
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)
        t = sample_time(B, device)
        eps_a = torch.randn_like(act)
        x_t_a = (1 - t)[:, None, None] * act + t[:, None, None] * eps_a
        noisy_w = None
        if model.variant in ("a2", "a3", "a3mask"):
            z_fut = encode_frozen(model, x_fut)
            eps_w = torch.randn_like(z_fut)
            noisy_w = (1 - t)[:, None, None, None, None] * z_fut \
                + t[:, None, None, None, None] * eps_w
        o = model(x, x_t_a, t, qpos, oi, noisy_world=noisy_w, cam72=None)
        # action x0 reconstruction (denormalized error)
        x0 = x_t_a - t[:, None, None] * o["v_a"]
        err = (x0 - act).cpu().numpy() * np.asarray(stats["scale"])[None, None, :]
        agg["act_mse"].append(float((err ** 2).mean()))
        # world decode dynIoU
        if noisy_w is not None:
            w0 = noisy_w - t[:, None, None, None, None] * o["v_w"]
            vox = ae_dec(w0)
            agg["world_dyn_iou"].append(masked_dyn_iou(vox, batch))
        # aux head dynIoU (a1): occ logit per offset, threshold 0
        if o.get("aux_logits") is not None:
            logits = o["aux_logits"]
            pred_hit = torch.zeros_like(batch["state_future"].to(device) == 0)
            for ki, k in enumerate(OFFSETS):
                sel = (oi == ki)
                pred_hit[sel] = logits[sel, 2 * ki] > 0
            agg["aux_dyn_iou"].append(masked_dyn_iou(
                torch.stack([pred_hit.float()], 1), batch) if False else
                masked_dyn_iou_aux(pred_hit, batch))
        # observer (a3/a3mask) — note: cam72 unavailable in voxels_ep cache,
        # observer eval requires the replay pipeline; here use zero cam (proxy)
        if o.get("obs_logits") is not None:
            vis_p = o["obs_logits"][0] > 0
            vis_g = batch["observed_future"][0].bool()
            agg["obs_iou"].append(iou(vis_p, vis_g))
            # upper bound: observer fed GT future geometry (frozen-enc latent)
            z_gt = encode_frozen(model, x_fut)
            wf_gt = model.gtok(z_gt) + model.mod(0)
            obs_gt = model.observer(wf_gt, torch.zeros(B, 72, device=device),
                                    oi)
            vis_p2 = obs_gt[0] > 0
            agg["obs_iou_gt_geom"].append(iou(vis_p2, vis_g))
    if model.variant == "a3mask" and hasattr(model, "gates"):
        agg["gates"] = [round(float(g), 3) for g in model.gates.detach().cpu()]

    res = dict(ckpt=args.ckpt, variant=ck["variant"],
               act_x0_mse_rad=round(float(np.mean(agg["act_mse"])), 5),
               world_masked_dynIoU=(round(float(np.mean(agg["world_dyn_iou"])), 4)
                                    if agg["world_dyn_iou"] else None),
               aux_masked_dynIoU=(round(float(np.mean(agg["aux_dyn_iou"])), 4)
                                  if agg["aux_dyn_iou"] else None),
               observer_visIoU=(round(float(np.mean(agg["obs_iou"])), 4)
                                if agg["obs_iou"] else None),
               observer_visIoU_gt_geom=(round(float(np.mean(agg["obs_iou_gt_geom"])), 4)
                                        if agg["obs_iou_gt_geom"] else None),
               gates=agg["gates"], n_batches=min(args.max_batches, len(dl)))
    out = Path("runs/stage4a/metrics") / (
        f"eval_{ck['variant']}_{Path(args.ckpt).parent.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def masked_dyn_iou_aux(pred_hit, batch):
    device = pred_hit.device
    obs_f = batch["observed_future"].to(device).bool()
    gt_hit = batch["state_future"].to(device) == 0
    st = batch["state_t"].to(device)
    both = obs_f & (st != 2)
    dyn = ((st == 0) ^ gt_hit) & both
    inter = (pred_hit & dyn).sum((1, 2, 3)).float()
    uni = ((pred_hit & dyn) | (gt_hit & dyn)).sum((1, 2, 3)).float()
    iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
    ok = ~torch.isnan(iou)
    return float(iou[ok].mean()) if ok.any() else float("nan")


if __name__ == "__main__":
    main()
