"""Stage 3a dataset over cached episode voxels.

Pairs (t -> t+k) with k in OFFSETS from the episode voxel cache
(<heavy_root>/cache/voxels_ep/...). Splits are BY EPISODE — no adjacent-frame
or intra-episode test leakage is possible by construction.

Input  : voxel_t channels [hit, free, observed, view_count/3] (4ch float32)
         qpos_t (14), action_chunk (H,14) — actions are the absolute joint
         targets the robot commands AFTER state t (action[i] -> state[i+1],
         verified from RoboTwin pkl2hdf5).
Target : voxel_{t+k} state (hit/free/unknown code) + observed mask.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

OFFSETS = (4, 8, 12, 16)
INPUT_CHANNELS = ("hit", "free", "observed", "view_count")


def load_episode(root: Path, variant: str, ep: int, pitch="20mm"):
    edir = root / variant / f"episode_{ep:07d}" / pitch
    manifest = json.loads((edir / "manifest.json").read_text())
    return edir, manifest


def frame_to_tensors(npz_path: Path):
    d = np.load(npz_path)
    hit = d["hit"].astype(np.float32)
    free = d["free"].astype(np.float32)
    obs = d["observed"].astype(np.float32)
    vc = d["view_count"].astype(np.float32) / 3.0
    x = np.stack([hit, free, obs, vc])          # (4,H,W,D) axis order ZYX-like
    state = np.full(hit.shape, 2, dtype=np.int64)
    state[free > 0.5] = 1
    state[hit > 0.5] = 0
    return x, state, obs, d["qpos"].astype(np.float32), d["action_chunk"], \
        int(d["action_valid"])


class DynamicsDataset(Dataset):
    """(t -> t+k) pairs. max_t limits t to keep the action chunk full."""

    def __init__(self, root, episodes, offsets=OFFSETS, action_horizon=16,
                 pitch="20mm", stride_t=1):
        self.items = []
        self.root = Path(root)
        for spec in episodes:
            variant, ep = spec.split(":")
            edir, manifest = load_episode(self.root, variant, int(ep), pitch)
            T = manifest["T_total"]
            for t in range(0, T - max(offsets) - 1, stride_t):
                if manifest["frames"][t]["npz"] is None:
                    continue
                for k in offsets:
                    self.items.append((edir, t, k, spec))
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        edir, t, k, spec = self.items[idx]
        x, state_t, obs_t, qpos, action_chunk, avalid = frame_to_tensors(
            edir / f"frame_{t:07d}.npz")
        xk, state_k, obs_k, *_ = frame_to_tensors(edir / f"frame_{t+k:07d}.npz")
        act = np.zeros((self.action_horizon, 14), dtype=np.float32)
        n = min(avalid, self.action_horizon)
        act[:n] = action_chunk[:n]
        return {
            "voxel": torch.from_numpy(x),
            "state_t": torch.from_numpy(state_t),
            "qpos": torch.from_numpy(qpos),
            "action": torch.from_numpy(act),
            "state_future": torch.from_numpy(state_k),
            "observed_future": torch.from_numpy(obs_k),
            "offset": k,
            "episode": spec,
            "t": t,
        }


def class_weights_from(root, episodes, pitch="20mm", max_frames=200):
    """Inverse-frequency weights for the 3-state CE, from train episodes only."""
    counts = np.zeros(3, dtype=np.int64)
    for spec in episodes:
        variant, ep = spec.split(":")
        edir, manifest = load_episode(Path(root), variant, int(ep), pitch)
        for t in np.linspace(0, manifest["T_total"] - 1, max_frames).astype(int):
            _, state, *_ = frame_to_tensors(edir / f"frame_{t:07d}.npz")
            counts += np.bincount(state.ravel(), minlength=3)
    w = counts.sum() / np.maximum(counts, 1) / 3.0
    return torch.tensor(w, dtype=torch.float32), counts.tolist()
