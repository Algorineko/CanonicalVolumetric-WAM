"""Stage 3c: representation-comparison dataset.

Wraps DynamicsDataset (voxel cache = TARGETS in the unified voxel eval space)
and builds representation-specific INPUTS on the fly from the point cache
(scripts/stage3c_point_cache.py: decoded 3-view depth + camera params):

- p2  : fused surface point cloud, 4096 pts (xyz).
- p2e : p2 + free-space ray samples (subsampled endpoints, fixed fractions
        along each camera->endpoint ray, bbox-clipped). Points carry a type
        channel (1=surface, 0=free) -> (N,4).
- p3  : per-view pointmap (240x320 xyz + validity) + per-camera K/T.

Seeded per (episode, t) so downsampling is deterministic.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cvwam.dynamics_dataset import DynamicsDataset, OFFSETS  # noqa: E402
from cvwam.geometry import depth_to_base  # noqa: E402

FREE_FRACTIONS = (0.25, 0.5, 0.75, 0.9)


class ReprDataset(Dataset):
    def __init__(self, voxel_root, point_root, episodes, offsets=OFFSETS,
                 stride_t=1, repr="p2", n_pts=4096, n_free_per_cam=341,
                 n_free_fixed=4096, action_horizon=16):
        self.base = DynamicsDataset(voxel_root, episodes, offsets=offsets,
                                    stride_t=stride_t,
                                    action_horizon=action_horizon)
        self.point_root = Path(point_root)
        self.repr = repr
        self.n_pts = n_pts
        self.n_free_per_cam = n_free_per_cam
        self.n_free_fixed = n_free_fixed
        # per-episode grid metadata from the voxel cache manifest
        self.meta = {}
        for spec in episodes:
            variant, ep = spec.split(":")
            mf = json.loads((Path(voxel_root) / variant /
                             f"episode_{int(ep):07d}" / "20mm" /
                             "manifest.json").read_text())
            self.meta[spec] = (np.asarray(mf["b_min"], dtype=np.float32),
                               float(mf["pitch_m"]),
                               tuple(int(v) for v in
                                     (np.load(Path(voxel_root) / variant /
                                              f"episode_{int(ep):07d}" / "20mm" /
                                              "frame_0000000.npz")["shape"])))

    def __len__(self):
        return len(self.base)

    def _unproject(self, pd):
        """depth3 (3,240,320) uint16 + params -> per-cam base points + grids."""
        depth3 = pd["depth"].astype(np.float32)
        K = pd["K"].astype(np.float64)
        Tbc = pd["T_base_cam"].astype(np.float64)  # (3,3,4) cam->base
        surf, grids = [], []
        for vi in range(3):
            dmm = depth3[vi]
            valid = dmm > 0
            dm = np.where(valid, dmm / 1000.0, 0.0)
            p, msk = depth_to_base(dm, K, Tbc[vi], valid=valid)
            surf.append(p)
            # full pointmap grid in base frame (row-vector: p_base = p_cam@R^T + t)
            H, W = dmm.shape
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            z = dm
            x = (u - K[0, 2]) / K[0, 0] * z
            y = (v - K[1, 2]) / K[1, 1] * z
            pc = np.stack([x, y, z], axis=-1)          # cam frame (cv)
            pb = pc @ Tbc[vi][:3, :3].T + Tbc[vi][:3, 3]   # cam->base
            grids.append(np.concatenate([pb, valid[..., None]], -1))
        return surf, grids, Tbc

    def __getitem__(self, idx):
        item = self.base[idx]
        spec, t = item["episode"], int(item["t"])
        variant, ep = spec.split(":")
        pd_path = self.point_root / variant / f"episode_{int(ep):07d}" / \
            f"frame_{t:06d}.npz"
        pd = np.load(pd_path)
        b_min, pitch, shape = self.meta[spec]
        rng = np.random.default_rng(abs(hash(spec)) % (2**32) + t)
        surf, grids, Tbc = self._unproject(pd)
        fused = np.concatenate(surf, 0)
        if self.repr in ("p2", "p2e"):
            # FIXED-SIZE outputs (DataLoader collate requires equal shapes):
            # surface part always self.n_pts rows (resampled with replacement
            # if fewer), free part (P2-E) always self.n_free_fixed rows.
            sel = rng.choice(len(fused), self.n_pts,
                             replace=len(fused) < self.n_pts)
            pts = fused[sel].astype(np.float32)
            if self.repr == "p2":
                return {**item, "points": pts,
                        "grid_meta": np.concatenate([b_min, [pitch]]).astype(np.float32)}
            # P2-E: free-space samples along rays
            free = []
            for vi in range(3):
                o = Tbc[vi][:3, 3]
                e = surf[vi]
                if len(e) == 0:
                    continue
                sub = rng.choice(len(e), min(self.n_free_per_cam, len(e)),
                                 replace=False)
                fr = np.asarray(FREE_FRACTIONS)[None, :, None]
                smp = o[None, None, :] + fr * (e[sub][:, None, :] - o[None, None, :])
                smp = smp.reshape(-1, 3)
                inside = np.all((smp >= b_min) & (smp < b_min + np.asarray(shape) * pitch), 1)
                free.append(smp[inside])
            free = np.concatenate(free, 0) if free else np.zeros((0, 3))
            nf = self.n_free_fixed
            fsel = rng.choice(len(free), nf, replace=len(free) < nf) if len(free) \
                else np.zeros(nf, int)
            allp = np.concatenate([pts, free[fsel]], 0)
            typ = np.concatenate([np.ones(self.n_pts), np.zeros(nf)])
            pts4 = np.zeros((len(allp), 4), dtype=np.float32)
            pts4[:, :3] = allp
            pts4[:, 3] = typ
            return {**item, "points": pts4,
                    "grid_meta": np.concatenate([b_min, [pitch]]).astype(np.float32)}
        # P3: per-view pointmap (3,4,240,320) float16 CHW + cam params
        pm = np.stack(grids, 0).transpose(0, 3, 1, 2).astype(np.float16)
        camT = np.zeros((3, 3, 4), dtype=np.float32)
        camT[:, :, :3] = Tbc[:, :, :3]      # R (cam->base)
        camT[:, :, 3] = Tbc[:, :, 3]        # t
        return {**item, "pointmap": pm, "camT": camT, "K": pd["K"].astype(np.float32),
                "grid_meta": np.concatenate([b_min, [pitch]]).astype(np.float32)}
