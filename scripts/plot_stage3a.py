"""Stage 3a plots: future-occupancy predictions vs GT for one test sample
(P0 / P1 / P1-R / P1-V side by side), an orthogonal slice through the change."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.dynamics_dataset import DynamicsDataset, OFFSETS  # noqa: E402
from cvwam.models_tiny import TinyDynamicsNet, split_outputs  # noqa: E402
import train_tiny as tt  # noqa: E402

cfg = yaml.safe_load(Path("configs/smoke.yaml").read_text())
root = Path(cfg["cache_dir"]) / "voxels_ep"
EP = "demo_randomized_balanced:1"
K = 16
T_SAMPLE = 400

ds = DynamicsDataset(root, [EP])
idx = next(i for i, (edir, t, k, s) in enumerate(ds.items) if t == T_SAMPLE and k == K)
item = ds[idx]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items()}

fig_rows = []
with torch.no_grad():
    for name, ck in [("GT", None), ("P0", None), ("P1", "p1"), ("P1-R", "p1_r"),
                     ("P1-V", "p1_v")]:
        if ck:
            c = torch.load(f"runs/stage3a/{ck}_split/best.pt", map_location=device,
                           weights_only=False)
            m = TinyDynamicsNet(variant=tt.MODEL_DIR[c["variant"]], base=c["base"],
                                n_offsets=len(OFFSETS)).to(device).eval()
            m.load_state_dict(c["model"])
            P = __import__("eval_tiny").preds_for(batch, m, c["variant"], device)
            pred = P[K]["pred_hit"][0].cpu().numpy()
        else:
            pred = (item["state_future"] == 0).numpy() if name == "GT" \
                else (item["voxel"][0] > 0.5).numpy()
        fig_rows.append((name, pred))

# slice through the GT change: pick the slice with most change voxels
gt_hit = (item["state_future"] == 0).numpy()
cur_hit = item["voxel"][0].numpy()[0] > 0.5
change = gt_hit ^ cur_hit
if change.sum() > 0:
    sums = [change.sum(axis=(1, 2)).max(), change.sum(axis=(0, 2)).max(),
            change.sum(axis=(0, 1)).max()]
    axis = int(np.argmax(sums))
    slice_idx = int(np.argmax(change.sum(axis=tuple(i for i in range(3) if i != axis))))
else:
    axis, slice_idx = 1, 30
print(f"slice axis {axis} idx {slice_idx}, change voxels {change.sum()}")

fig, axs = plt.subplots(1, len(fig_rows), figsize=(3.4 * len(fig_rows), 3.6))
for ax, (name, pred) in zip(axs, fig_rows):
    img = np.zeros(pred.shape[1 - 0], dtype=np.int64) if False else None
    # slice: hit=red, gt-future-hit-outline=blue? keep simple: pred hit red, GT hit outline cyan
    p = np.take(pred, slice_idx, axis=axis)
    g = np.take(gt_hit, slice_idx, axis=axis)
    img = np.zeros(p.shape + (3,), dtype=np.float32)
    img[..., 0] = p * 0.85
    img[..., 1] = g * 0.6
    img[..., 2] = g * 0.75
    ax.imshow(np.swapaxes(img, 0, 1), origin="lower", interpolation="nearest")
    ax.set_title(f"{name} (k={K})", fontsize=9)
    ax.set_xlabel("red=pred hit, cyan=GT hit")
fig.suptitle(f"future occupancy @ t+{K}, {EP} t={T_SAMPLE}", fontsize=10)
fig.tight_layout()
fig.savefig("runs/stage3a/plots/future_occ_comparison.png", dpi=130)
print("saved runs/stage3a/plots/future_occ_comparison.png")
