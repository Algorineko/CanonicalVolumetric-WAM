"""Model Integrity Audit S7 recompute: old-vs-new masked dyn-IoU on stage4a evals.

Re-forwards the stage4a val set (same protocol as eval_stage4a.py: batch 32,
shuffle off, stride_t=2, n_batches from the original eval json) and scores the
SAME forward draws with BOTH the verbatim pre-fix (buggy) formula and the fixed
formula, so the old/new delta isolates the formula change from the noise redraw
(the original eval runs were unseeded, so their exact t/eps draws are not
reproducible). act_x0_mse (metric-independent) is carried along as a
noise-magnitude probe against the original json.

Per checkpoint writes <orig_eval_name>.recomputed.json into --out-dir:
  {file, metric, old, new, buggy_same_draws, new_raw, buggy_raw,
   act_x0_mse_rad_old, act_x0_mse_rad_new, ckpt, variant, seed, n_batches, note}
old          = value stored in the original eval json
new          = fixed metric on the recompute draws (round 4, same convention)
buggy_same_draws = pre-fix formula on the identical draws (formula-only delta)

Run (single GPU process, sequential over checkpoints):
  <openwam python> scripts/recompute_stage4a_dyniou.py \
      --ckpts runs/stage4a/a1_s42/best.pt ... --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from cvwam.dynamics_dataset import OFFSETS, DynamicsDataset  # noqa: E402
from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from train_stage4a import (pooled_occupancy, soft_masked_dyn_iou,  # noqa: E402
                           unpool_soft, normalize_act, sample_time)
from train_tokenizer_ae import LatentDecoder  # noqa: E402
from eval_stage4a import masked_dyn_iou_aux  # AUDIT-S7 fixed  # noqa: E402
from test_dynamic_metrics import (_buggy_masked_dyn_iou_aux,  # noqa: E402
                                  _buggy_soft_masked_dyn_iou)


def _mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


@torch.no_grad()
def recompute(ckpt_path, args):
    device = args.device
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    ae = torch.load(args.ae, map_location="cpu", weights_only=False)
    ae_dec = LatentDecoder(in_ch=ck.get("bottleneck_ch", 128), out_ch=4).to(device)
    ae_dec.load_state_dict(ae["dec"])
    ae_dec.eval()

    # original eval json (provides old values + the batch count to replicate)
    metrics_dir = Path(args.metrics_dir)
    orig_path = metrics_dir / f"eval_{ck['variant']}_{Path(ckpt_path).parent.name}.json"
    orig = json.loads(orig_path.read_text())
    n_batches = int(orig["n_batches"])
    if args.max_batches is not None:                 # smoke/debug override
        n_batches = min(n_batches, args.max_batches)
    metric_field = ("world_masked_dynIoU"
                    if orig.get("world_masked_dynIoU") is not None
                    else "aux_masked_dynIoU")
    if orig.get(metric_field) is None:
        return None  # a0-style: metric never invoked, not affected

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agg = dict(act_mse=[], buggy=[], fixed=[])
    for bi, batch in enumerate(dl):
        if bi >= n_batches:
            break
        x = batch["voxel"].to(device)
        x_fut = batch["voxel_future"].to(device)
        qpos = batch["qpos"].to(device)
        act_raw = batch["action"].numpy()
        act = torch.from_numpy(normalize_act(act_raw, stats)).to(device)
        B = x.shape[0]
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)
        t = sample_time(B, device)                       # RNG order == eval_stage4a
        eps_a = torch.randn_like(act)
        x_t_a = (1 - t)[:, None, None] * act + t[:, None, None] * eps_a
        noisy_w = None
        if model.variant in ("a2", "a3", "a3mask"):
            tgt_w = pooled_occupancy(x_fut)
            eps_w = torch.randn_like(tgt_w)
            noisy_w = (1 - t)[:, None, None, None, None] * tgt_w \
                + t[:, None, None, None, None] * eps_w
        o = model(x, x_t_a, t, qpos, oi, noisy_world=noisy_w, cam72=None)
        x0 = x_t_a - t[:, None, None] * o["v_a"]
        err = (x0 - act).cpu().numpy() * np.asarray(stats["scale"])[None, None, :]
        agg["act_mse"].append(float((err ** 2).mean()))
        if noisy_w is not None:
            w0 = noisy_w - t[:, None, None, None, None] * o["v_w"]
            vox = unpool_soft(w0, x.shape[2:])
            agg["buggy"].append(_buggy_soft_masked_dyn_iou(vox, batch))
            agg["fixed"].append(soft_masked_dyn_iou(vox, batch))
        if o.get("aux_logits") is not None:
            logits = o["aux_logits"]
            pred_hit = torch.zeros_like(batch["state_future"].to(device) == 0)
            for ki, k in enumerate(OFFSETS):
                sel = (oi == ki)
                pred_hit[sel] = logits[sel, 2 * ki] > 0
            agg["buggy"].append(_buggy_masked_dyn_iou_aux(pred_hit, batch))
            agg["fixed"].append(masked_dyn_iou_aux(pred_hit, batch))
        # observer branch skipped: consumes no RNG and its outputs are not
        # part of the affected metric fields

    buggy_raw, fixed_raw = _mean(agg["buggy"]), _mean(agg["fixed"])
    res = dict(file=orig_path.name, metric=metric_field,
               old=orig[metric_field], new=round(fixed_raw, 4),
               buggy_same_draws=round(buggy_raw, 4),
               new_raw=fixed_raw, buggy_raw=buggy_raw,
               act_x0_mse_rad_old=orig["act_x0_mse_rad"],
               act_x0_mse_rad_new=round(float(np.mean(agg["act_mse"])), 5),
               ckpt=ckpt_path, variant=ck["variant"], seed=args.seed,
               n_batches=n_batches,
               note=("old = original eval json (buggy formula, unseeded draws); "
                     "new = fixed AUDIT-S7 formula on this run's draws (seed "
                     f"{args.seed}); buggy_same_draws = pre-fix formula on the "
                     "identical draws (isolates formula effect from redraw noise)"))
    return res, orig_path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--ae", default="runs/stage4a/tokenizer_ae/best.pt")
    ap.add_argument("--metrics-dir", default="runs/stage4a/metrics")
    ap.add_argument("--out-dir", default="runs/model_integrity/metric_recompute")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="override the batch count taken from the eval json "
                         "(smoke/debug only; real runs keep the json value)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ckpt in args.ckpts:
        out = recompute(ckpt, args)
        if out is None:
            print(f"[skip] {ckpt}: eval json has no affected metric field (null)")
            continue
        res, orig_name = out
        dest = out_dir / f"{orig_name}.recomputed.json"
        if dest.exists():
            raise SystemExit(f"[refuse] {dest} exists (no-overwrite policy)")
        dest.write_text(json.dumps(res, indent=2))
        print(f"[done] {res['file']}: {res['metric']} old={res['old']} "
              f"new={res['new']} buggy_same_draws={res['buggy_same_draws']} "
              f"(act_mse old={res['act_x0_mse_rad_old']} "
              f"new={res['act_x0_mse_rad_new']})")


if __name__ == "__main__":
    main()
