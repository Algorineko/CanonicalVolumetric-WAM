"""Stage 3b §1.3: door-region / robot-region / object-level evaluation on the
independent replay episode (427 frames, real qpos, door GT, actions attached).

Regions (evaluation-only; future GT is allowed for REGION DEFINITION, never
for prediction input):
  door_sweep  = door_region_t | door_region_{t+k}
  robot_sweep = robot_region_t | robot_region_{t+k}   (FK from real qpos)
Metrics per (t, k): door_dynIoU, door_surfF1@1, door_new_recall/prec (door
moved into new cells), robot_dynIoU, nonrobot_dynIoU (excl door), static
falseOcc, overall occ/dyn IoU. P0 persistence = the door-persistence baseline.

Outputs per-model CSV (resumable) + door_summary.csv.
"""

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.dynamics_dataset import OFFSETS  # noqa: E402
from cvwam.models_tiny import TinyDynamicsNet, split_outputs  # noqa: E402
import train_tiny as tt  # noqa: E402
import pareto_eval as pe  # noqa: E402

REP_ROOT = Path("/data/canonical-volumetric-wam/cache/voxels_ep_replay")
REP_NAME = "demo_clean_balanced_10_replay"
METRICS = ("door_dynIoU", "door_surfF1", "door_new_recall", "door_new_prec",
           "robot_dynIoU", "nonrobot_dynIoU", "static_falseOcc",
           "occ_iou", "dyn_iou")


def frame_arrays(npz_path: Path):
    d = np.load(npz_path)
    hit = d["hit"].astype(np.float32)
    free = d["free"].astype(np.float32)
    obs = d["observed"].astype(np.float32)
    vc = d["view_count"].astype(np.float32) / 3.0
    state = np.full(hit.shape, 2, dtype=np.int64)
    state[free > 0.5] = 1
    state[hit > 0.5] = 0
    act = d["action_chunk"] if "action_chunk" in d.files else \
        np.zeros((0, 14), dtype=np.float32)
    return dict(voxel=np.stack([hit, free, obs, vc]), state=state,
                obs=obs.astype(bool), qpos=d["qpos"].astype(np.float32),
                action=act,
                avalid=int(d["action_valid"]) if "action_valid" in d.files else 0,
                door=d["door_region"].astype(bool), hit=hit.astype(bool))


class ReplayDataset(Dataset):
    def __init__(self, edir, offsets=OFFSETS, stride_t=1, action_horizon=16):
        mf = json.loads((edir / "manifest.json").read_text())
        self.T = len(list(edir.glob("frame_*.npz")))
        self.edir = edir
        self.pairs = [(t, k) for t in range(0, self.T - max(offsets) - 1, stride_t)
                      for k in offsets]
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        t, k = self.pairs[idx]
        a = frame_arrays(self.edir / f"frame_{t:07d}.npz")
        b = frame_arrays(self.edir / f"frame_{t + k:07d}.npz")
        act = np.zeros((self.action_horizon, 14), dtype=np.float32)
        n = min(a["avalid"], self.action_horizon)
        act[:n] = a["action"][:n]
        return dict(
            voxel=torch.from_numpy(a["voxel"]),
            state_t=torch.from_numpy(a["state"]),
            cur_hit=torch.from_numpy(a["hit"]),
            door_t=torch.from_numpy(a["door"]),
            qpos=torch.from_numpy(a["qpos"]),
            action=torch.from_numpy(act),
            state_f=torch.from_numpy(b["state"]),
            obs_f=torch.from_numpy(b["obs"]),
            hit_f=torch.from_numpy(b["hit"]),
            door_f=torch.from_numpy(b["door"]),
            offset=k, t=t)


def robot_region_for(edir: Path, cache_root: Path):
    """FK-from-real-qpos robot occupancy per frame; cached npz per frame.

    qpos dim mapping (correlation-verified vs drive targets, r>=0.998; FK
    link6 vs original ee_poses: mean 2.34mm, p95 18.7mm = known PhysX contact
    divergence): LEFT arm joints at qpos_full dims [6,14,18,22,26,30]; LEFT
    gripper physical travel (meters, [0,0.04765]) at dim 34; dims 0-5 = right
    arm (frozen at 0 for this episode; if a right-arm episode is evaluated,
    re-derive the mapping with the same correlation probe).
    """
    L_ARM_DIMS = (6, 14, 18, 22, 26, 30)
    cache = cache_root / REP_NAME
    cache.mkdir(parents=True, exist_ok=True)
    T = len(list(edir.glob("frame_*.npz")))  # manifest T_total may overcount
    need = [t for t in range(T) if not (cache / f"frame_{t:07d}.npz").exists()]
    if need:
        import build_robot_region as brr
        chains = {s: brr.load_chain(s) for s in ("left", "right")}
        spheres = {s: brr.load_spheres(s) for s in ("left", "right")}
        d0 = np.load(edir / "frame_0000000.npz")
        b_min = d0["b_min"].astype(np.float64)
        pitch = float(d0["pitch"])
        shape = tuple(int(v) for v in d0["shape"])
        for t in need:
            q = np.load(edir / f"frame_{t:07d}.npz")["qpos_full"].astype(np.float64)
            robot = np.zeros(shape, dtype=bool)
            for side in ("left", "right"):
                p = "fl" if side == "left" else "fr"
                if side == "left":
                    qd = {f"{p}_joint{i + 1}": q[L_ARM_DIMS[i]] for i in range(6)}
                    g = float(np.clip(q[34], 0, 0.04765))
                    qd[f"{p}_joint7"] = g
                    qd[f"{p}_joint8"] = g
                else:  # right arm parked for this episode (all-zero verified)
                    qd = {f"{p}_joint{i + 1}": 0.0 for i in range(6)}
                    qd[f"{p}_joint7"] = qd[f"{p}_joint8"] = 0.0
                poses = brr.fk_links(chains[side][0], chains[side][1], qd)
                for link, sph_list in spheres[side].items():
                    if link not in poses:
                        continue
                    R, tl = poses[link]
                    for c_local, r in sph_list:
                        c = R @ np.asarray(c_local) + tl
                        idx = np.floor((c - b_min) / pitch).astype(int)
                        rad = int(np.ceil(r / pitch))
                        lo = np.maximum(idx - rad, 0)
                        hi = np.minimum(idx + rad + 1, np.asarray(shape))
                        if (idx < 0).any() or (idx >= np.asarray(shape)).any():
                            continue
                        gi, gj, gk = np.indices((hi[0] - lo[0], hi[1] - lo[1],
                                                 hi[2] - lo[2]))
                        centers = b_min + np.stack([gi + lo[0], gj + lo[1],
                                                    gk + lo[2]], axis=-1) * pitch + pitch / 2
                        robot[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= \
                            np.linalg.norm(centers - c, axis=-1) <= r
            np.savez_compressed(cache / f"frame_{t:07d}.npz", robot=robot)
        print(f"[robot] built {len(need)} frames -> {cache}", flush=True)
    return cache, T


def nanmean(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    ok = ~np.isnan(x)
    return float(x[ok].mean()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--stride-t", type=int, default=2)
    ap.add_argument("--out-dir", default="runs/stage3b/metrics/door")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = "cuda:0"
    edir = REP_ROOT / REP_NAME / "20mm"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[robot] ensuring FK robot regions (cached)", flush=True)
    rob_cache, T = robot_region_for(edir, REP_ROOT / "_robot_region")
    shape = tuple(int(v) for v in np.load(edir / "frame_0000000.npz")["shape"])
    robots = np.stack([np.load(rob_cache / f"frame_{t:07d}.npz")["robot"]
                       for t in range(T)])

    ds = ReplayDataset(edir, stride_t=args.stride_t)
    print(f"[data] {len(ds)} (t,k) pairs", flush=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=6, pin_memory=True)

    # pass 1+2 per model: infer occ_logits on GPU, metrics on CPU
    model_cache = {}
    raws = [np.load(edir / f"frame_{t:07d}.npz") for t in range(T)]
    st_t = np.stack([np.where(d["free"] > 0.5, 1,
                              np.where(d["hit"] > 0.5, 0, 2)).astype(np.int64)
                     for d in raws])
    obs_all = np.stack([d["observed"].astype(bool) for d in raws])
    hits = np.stack([d["hit"].astype(bool) for d in raws])
    doors = np.stack([d["door_region"].astype(bool) for d in raws])
    del raws

    def get_model(path):
        if path == "p0":
            return None, None
        if path not in model_cache:
            ck = torch.load(path, map_location=device, weights_only=False)
            m = TinyDynamicsNet(variant=tt.MODEL_DIR[ck["variant"]], base=ck["base"],
                                n_offsets=len(OFFSETS)).to(device).eval()
            m.load_state_dict(ck["model"])
            model_cache[path] = (ck["variant"], m)
        return model_cache[path]

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        csv_path = out / f"door_{name}.csv"
        if csv_path.exists():
            print(f"[skip] {name} (resume)", flush=True)
            continue
        variant, model = get_model(path)
        g0 = np.load(edir / "frame_0000000.npz")
        grid = tuple(int(v) for v in g0["shape"])
        logits = np.zeros((len(ds),) + grid, dtype=np.float32)
        i = 0
        for batch in dl:
            occ = pe.infer_batch(model, variant, batch, device)  # (B,grid)
            B = occ.shape[0]
            logits[i:i + B] = occ.numpy()
            i += B
        assert i == len(ds)
        print(f"[infer] {name} done", flush=True)

        rows = []
        for j, (t, k) in enumerate(ds.pairs):
            pred_hit = logits[j] > 0.0
            gt_hit = hits[t + k]
            obs_f = obs_all[t + k]
            cur_hit = hits[t]
            door_sweep = doors[t] | doors[t + k]
            robot_sweep = robots[t] | robots[t + k]
            both = obs_f & (st_t[t] != 2)
            dyn = ((st_t[t] == 0) ^ gt_hit) & both
            row = dict(model=name, t=t, offset=k)
            sel = door_sweep & obs_f
            row["door_dynIoU"] = nanmean(iou_np(pred_hit & sel, gt_hit & sel))
            row["door_surfF1"] = nanmean(f1_np(pred_hit & sel, gt_hit & sel, tol=1))
            new = gt_hit & ~cur_hit & door_sweep & obs_f
            n_new = int(new.sum())
            row["door_new_recall"] = (float((pred_hit & new).sum()) / n_new
                                      if n_new else float("nan"))
            pred_new = pred_hit & ~cur_hit & door_sweep & obs_f
            n_pn = int(pred_new.sum())
            row["door_new_prec"] = (float((pred_new & gt_hit).sum()) / n_pn
                                    if n_pn else float("nan"))
            rsel = robot_sweep & obs_f
            row["robot_dynIoU"] = nanmean(iou_np(pred_hit & rsel, gt_hit & rsel))
            ndyn = dyn & ~(robot_sweep | door_sweep)
            row["nonrobot_dynIoU"] = nanmean(iou_np(pred_hit & ndyn, gt_hit & ndyn))
            static = gt_hit & cur_hit & ~(robot_sweep | door_sweep) & obs_f
            n_st = int(static.sum())
            row["static_falseOcc"] = (float((pred_hit & static).sum()) / n_st
                                      if n_st else float("nan"))
            row["occ_iou"] = nanmean(iou_np(pred_hit & obs_f, gt_hit & obs_f))
            row["dyn_iou"] = nanmean(iou_np(pred_hit & dyn, gt_hit & dyn))
            rows.append(row)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({c for r in rows for c in r}))
            w.writeheader()
            for r in rows:
                w.writerow({c: (round(v, 5) if isinstance(v, float) else v)
                            for c, v in r.items()})
        print(f"[done] {name}: {len(rows)} rows -> {csv_path}", flush=True)

    # summary: aggregate per model per offset
    summary = []
    for spec in args.checkpoints:
        name, _ = spec.split("=", 1)
        csv_path = out / f"door_{name}.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(open(csv_path)))
        for k in OFFSETS:
            rk = [r for r in rows if int(r["offset"]) == k]
            agg = dict(model=name, offset=k, n=len(rk))
            for met in METRICS:
                vals = [float(r[met]) for r in rk
                        if r[met] not in ("", "nan")] if rk else []
                vals = [v for v in vals if not np.isnan(v)]
                agg[met] = round(float(np.mean(vals)), 4) if vals else None
                agg[f"n_{met}"] = len(vals)
            summary.append(agg)
    with open(out / "door_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({c for r in summary for c in r}))
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print("[all done] -> door_summary.csv", flush=True)


def iou_np(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / uni if uni > 0 else float("nan")


def f1_np(pred, gt, tol=1):
    if tol:
        pred = _dilate1(pred)
        gt = _dilate1(gt)
    tp = np.logical_and(pred, gt).sum()
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(gt.sum()), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9) if (pred.sum() and gt.sum()) \
        else float("nan")


def _dilate1(g):
    out = g.copy()
    for d in (0, 1, 2):
        for s in (-1, 1):
            out |= np.roll(g, s, d)
    return out


if __name__ == "__main__":
    main()
