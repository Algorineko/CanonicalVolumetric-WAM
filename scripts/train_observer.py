"""Tiny Observation Model prototype (next_model_design.md §Tiny Prototype).

L4 transfer test: visibility (observed mask at t+k) as a function of
(current voxel field, action chunk, FUTURE camera parameters). Trained ONLY
on the original replay (camera config A); evaluated on
  (a) in-config holdout: original replay frames t>=340 (same camera, unseen t)
  (b) cross-config:      probe replay (same physics bit-exact, head camera
                          +2.5mm offset => different visibility targets)

If the Observation Model factorization is real, visIoU should transfer across
configs (b) with only a small drop from (a); if the model memorizes config A's
visibility, (b) collapses.

Camera params: per-frame per-camera T_base_cam (3x4, 12 numbers x3 cams, at
t+k) from replay_gt.json via the verified chain — camera poses are
exogenous/proprioceptive (head camera mounted on robot; offset is a config),
NOT future scene observations. Documented assumption.
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import geometry as geo  # noqa: E402
from cvwam.geometry import T_base_world  # noqa: E402
from cvwam.models_tiny import apply_film, film  # noqa: E402

REP_ROOT = Path("/data/canonical-volumetric-wam/cache/voxels_ep_replay")
GT_ROOT = Path("/data/canonical-volumetric-wam/cache/replay_gt")
OFFSETS = (4, 8, 12, 16)


def cam_params(replay_gt_rows, cfg, idx):
    """-> (n_cams*24,) flattened [T_base_cam(t+k) | T_base_cam(t)] per camera."""
    T_wb = rta.T_world_base_from_config(cfg)
    T_bw = geo.T_base_world(T_wb)
    vec = []
    for i in idx:
        r = replay_gt_rows[i]
        row = []
        for name in ("head_camera", "left_camera", "right_camera"):
            for key in ("T_world_cam_cv_head_camera", "T_world_cam_cv_left_camera",
                        "T_world_cam_cv_right_camera"):
                if key == f"T_world_cam_cv_{name}":
                    T_wc = np.asarray(r[key], dtype=np.float64)
                    if T_wc.shape == (3, 4):  # stored without homogeneous row
                        T_wc = np.vstack([T_wc, [0.0, 0.0, 0.0, 1.0]])
                    T_bc = T_bw @ T_wc                      # base <- cam_cv
                    row += T_bc[:3, :4].reshape(-1).tolist()
        vec.append(row)
    return np.asarray(vec, dtype=np.float32)  # (n, 72)


class ObsDataset(Dataset):
    def __init__(self, rep_dir, rows, cfg, ts, offsets=OFFSETS):
        self.edir = Path(rep_dir) / "20mm"
        self.rows = rows
        self.cfg = cfg
        self.pairs = [(t, k) for t in ts for k in offsets
                      if t + k < len(rows)]
        self._cam_cache = {}

    def __len__(self):
        return len(self.pairs)

    def _cam(self, idx):
        if idx not in self._cam_cache:
            self._cam_cache[idx] = cam_params(self.rows, self.cfg, [idx])[0]
        return self._cam_cache[idx]

    def __getitem__(self, i):
        t, k = self.pairs[i]
        da = np.load(self.edir / f"frame_{t:07d}.npz")
        db = np.load(self.edir / f"frame_{t + k:07d}.npz")
        hit = da["hit"].astype(np.float32)
        free = da["free"].astype(np.float32)
        obs = da["observed"].astype(np.float32)
        vc = da["view_count"].astype(np.float32) / 3.0
        x = np.stack([hit, free, obs, vc])
        act = np.zeros((16, 14), dtype=np.float32)
        n = min(int(da["action_valid"]), 16)
        act[:n] = da["action_chunk"][:n]
        vis_tgt = db["observed"].astype(np.float32)
        cam = np.concatenate([self._cam(t + k), self._cam(t)])  # (144,)
        return dict(voxel=torch.from_numpy(x), action=torch.from_numpy(act),
                    qpos=torch.from_numpy(da["qpos"].astype(np.float32)),
                    vis_tgt=torch.from_numpy(vis_tgt),
                    cam=torch.from_numpy(cam), t=t, offset=k)


class ObsNet(nn.Module):
    """Tiny voxel CNN + (qpos, action, camera) FiLM -> visibility logits."""

    def __init__(self, base=32, cond_dim=128, n_offsets=4):
        super().__init__()
        self.n_offsets = n_offsets
        self.cond_mlp = nn.Sequential(
            nn.Linear(14 + 16 * 14 + 72, 256), nn.SiLU(),
            nn.Linear(256, cond_dim), nn.SiLU())
        self.offset_emb = nn.Embedding(n_offsets, cond_dim)
        c1, c2, c3 = base, base * 2, base * 4
        self.enc1 = nn.Sequential(nn.Conv3d(4, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c3, c3, 3, padding=1), nn.SiLU())
        self.film3 = film(cond_dim, c3)
        self.film2 = film(cond_dim, c2)
        self.dec2 = nn.Sequential(nn.Conv3d(c3 + c2, c2, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.dec1 = nn.Sequential(nn.Conv3d(c2 + c1, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
        self.head = nn.Conv3d(c1, n_offsets, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, voxel, cond_vec, offset_idx):
        cond = self.cond_mlp(cond_vec) + self.offset_emb(offset_idx)
        e1 = self.enc1(voxel)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = apply_film(e3, self.film3(cond))
        d2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = apply_film(d2, self.film2(cond))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)  # (B, n_offsets, grid)


def iou(pred, gt):
    inter = (pred & gt).sum().item()
    uni = (pred | gt).sum().item()
    return inter / uni if uni else float("nan")


@torch.no_grad()
def evaluate(model, dl, device, thr=0.0):
    model.eval()
    vals, reveal_acc = [], []
    for batch in dl:
        vox = batch["voxel"].to(device)
        cond = torch.cat([batch["qpos"].to(device),
                          batch["action"].to(device).flatten(1),
                          batch["cam"].to(device)], 1)
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)
        out = model(vox, cond, oi)
        for j in range(vox.shape[0]):
            ki = OFFSETS.index(batch["offset"][j])
            vis_logit = out[j, ki]
            tgt = batch["vis_tgt"][j].bool()
            pred = (vis_logit > thr).cpu()
            vals.append(iou(pred, tgt))
    model.train()
    return float(np.nanmean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/stage3b/observer_proto")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0"
    cfg = yaml.safe_load(Path(args.config).read_text())
    rows = json.loads((GT_ROOT / "clean_10" / "replay_gt.json").read_text())
    # keep only frames that exist in the voxel cache
    n_frames = len(list((REP_ROOT / "demo_clean_balanced_10_replay" / "20mm").glob("frame_*.npz")))
    rows = rows[:n_frames]
    T = len(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_ts = list(range(0, 340))
    inconf_ts = list(range(340, T - max(OFFSETS) - 1))
    probe_rows = json.loads((GT_ROOT / "probe_ep10_camdis05" /
                             "replay_gt.json").read_text())[:n_frames]
    probe_ts = list(range(0, T - max(OFFSETS) - 1))

    tr_ds = ObsDataset(REP_ROOT / "demo_clean_balanced_10_replay", rows, cfg, train_ts)
    in_ds = ObsDataset(REP_ROOT / "demo_clean_balanced_10_replay", rows, cfg, inconf_ts)
    pr_ds = ObsDataset(REP_ROOT / "demo_clean_balanced_10_probe_camdis05", probe_rows,
                       cfg, probe_ts)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=6,
                       drop_last=True)
    in_dl = DataLoader(in_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    pr_dl = DataLoader(pr_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    print(f"train {len(tr_ds)} inconf {len(in_ds)} probe {len(pr_ds)} pairs", flush=True)

    model = ObsNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    log = open(out / "train.log", "w")
    t0 = time.perf_counter()
    step = 0
    while step < args.steps:
        for batch in tr_dl:
            if step >= args.steps:
                break
            vox = batch["voxel"].to(device)
            cond = torch.cat([batch["qpos"].to(device),
                              batch["action"].to(device).flatten(1),
                              batch["cam"].to(device)], 1)
            oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)
            tgt = batch["vis_tgt"].to(device)
            # per-offset blocks
            outm = model(vox, cond, oi)
            loss = 0.0
            for ki in range(len(OFFSETS)):
                sel = (oi == ki)
                if sel.sum() == 0:
                    continue
                loss = loss + F.binary_cross_entropy_with_logits(
                    outm[sel, ki], tgt[sel])
            loss = loss / len(OFFSETS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0 or step == 1:
                msg = f"step {step} loss {loss.item():.4f} ({time.perf_counter()-t0:.0f}s)"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
    v_in = evaluate(model, in_dl, device)
    v_probe = evaluate(model, pr_dl, device)
    res = dict(steps=args.steps, visIoU_inconf_holdout=round(v_in, 4),
               visIoU_cross_config_probe=round(v_probe, 4),
               transfer_drop=round(v_in - v_probe, 4),
               verdict=("TRANSFERS" if v_probe > 0.9 * max(v_in, 1e-9) else
                        "PARTIAL" if v_probe > 0.75 * max(v_in, 1e-9) else "FAILS"))
    (out / "result.json").write_text(json.dumps(res, indent=2))
    torch.save({"model": model.state_dict()}, out / "last.pt")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
