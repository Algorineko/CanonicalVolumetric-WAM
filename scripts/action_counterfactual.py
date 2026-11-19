"""Model-side action counterfactual sensitivity (Stage 3b §4 hold-probe,
model-side half): same obs_t + qpos, compare predicted occupancy under the
REAL recorded action chunk vs a HOLD chunk (current qpos repeated 16x).

Primary metric: number of cells whose predicted occupancy CHANGES between the
two action chunks (threshold-free action-conditioning measure), and the
fraction of those that coincide with GT-flip cells.
"""

import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.dynamics_dataset import DynamicsDataset, OFFSETS  # noqa: E402
import train_tiny as tt  # noqa: E402

OUT = Path("runs/stage3b/metrics/action_counterfactual.json")
MAX_BATCHES = 12
VARIANTS = (("p1", "runs/stage3b/ckpt/p1_s42/best.pt"),
            ("p1_v", "runs/stage3b/ckpt/p1_v_s42/best.pt"))


def occ_logits_grid(model, variant, vox, qpos, act, oi):
    """-> (B, n_off, grid) cpu float occ logits; nan for samples of other offsets."""
    out = model(vox, qpos, act, oi)
    blocks = tt.split_outputs(out, tt.MODEL_DIR[variant], len(OFFSETS))
    B = vox.shape[0]
    grids = []
    for ki in range(len(OFFSETS)):
        sel = (oi == ki)
        idx = torch.nonzero(sel).flatten()
        if len(idx) == 0:
            continue
        if tt.MODEL_DIR[variant] == "decoupled":
            occ = blocks[ki]["occ_logit"][sel]
        else:
            pr = torch.softmax(blocks[ki]["state_logits"][sel], dim=1)[:, 0]
            occ = torch.log(pr.clamp_min(1e-6) / (1 - pr).clamp_min(1e-6))
        full = torch.full((B, *occ.shape[1:]), float("nan"), device=occ.device)
        full[idx] = occ
        grids.append(full.float().cpu())
    return torch.stack(grids, 1)  # (B,4,grid)


def main():
    device = "cuda:0"
    cfg = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path("runs/stage3b/split_manifest.json").read_text())
    ds = DynamicsDataset(root, sm["splits"]["clean_test"], stride_t=2)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    out = {}
    for name, path in VARIANTS:
        ck = torch.load(path, map_location=device, weights_only=False)
        variant = ck["variant"]
        m = tt.make_model(name, ck["base"], 128, device)
        m.load_state_dict(ck["model"])
        m.eval()
        changed = 0
        changed_gt = 0
        gt_cells = 0
        n = 0
        dlogit_gt_sum = 0.0
        dlogit_gt_n = 0
        dlogit_all_sum = 0.0
        dlogit_all_n = 0
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                if bi >= MAX_BATCHES:
                    break
                vox = batch["voxel"].to(device)
                qpos = batch["qpos"].to(device)
                act_real = batch["action"].to(device)
                act_hold = qpos.unsqueeze(1).expand(-1, 16, -1).contiguous()
                oi = tt.offset_index(batch, device)
                occ_r = occ_logits_grid(m, variant, vox, qpos, act_real, oi)
                occ_h = occ_logits_grid(m, variant, vox, qpos, act_hold, oi)
                valid = ~torch.isnan(occ_r[:, 0])          # (B,)
                diff = ((occ_r > 0) != (occ_h > 0)) & valid[:, None]
                gt_flip = ((batch["state_t"] == 0) ^
                           (batch["state_future"] == 0)).bool()
                gf = []
                for ki in range(len(OFFSETS)):
                    sel = (oi == ki).cpu()
                    idx = torch.nonzero(sel).flatten()
                    full = torch.zeros((vox.shape[0], *gt_flip.shape[1:]),
                                       dtype=torch.bool)
                    full[idx] = gt_flip[sel]
                    gf.append(full)
                gf = torch.stack(gf, 1) & valid[:, None]
                dlogit = (occ_r - occ_h).abs()
                ok = ~torch.isnan(dlogit)
                dlogit_gt_sum += float(dlogit[gf & ok].sum())
                dlogit_gt_n += int((gf & ok).sum())
                dlogit_all_sum += float(dlogit[ok].sum())
                dlogit_all_n += int(ok.sum())
                changed += int(diff.sum())
                changed_gt += int((diff & gf).sum())
                gt_cells += int(gf.sum())
                n += int(valid.sum())
        out[name] = dict(
            changed_cells_per_sample=round(changed / max(n, 1), 2),
            changed_at_gt_flip=round(changed_gt / max(gt_cells, 1), 4),
            mean_abs_dlogit_gt_flip=round(dlogit_gt_sum / max(dlogit_gt_n, 1), 4),
            mean_abs_dlogit_all=round(dlogit_all_sum / max(dlogit_all_n, 1), 4),
            note="changed_at_gt_flip = fraction of GT-flip cells whose binary "
                 "prediction depends on the action; dlogit = |occ_real-occ_hold| "
                 "mean over cells (threshold-free conditioning strength)")
        print(name, out[name], flush=True)
    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
