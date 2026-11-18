"""Stage 2c plots: 3x3 transition heatmaps per offset/region + flip decomposition."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

rows = [json.loads(l) for l in
        Path("runs/stage2c/metrics/transition_rows.jsonl").read_text().splitlines()]
outdir = Path("runs/stage2c/plots")
outdir.mkdir(parents=True, exist_ok=True)

STATES = ["hit", "free", "unknown"]
# 1) heatmaps for region=all across offsets (one episode)
ep = rows[0]["episode"]
rows_ep = [r for r in rows if r["episode"] == ep]
fig, axs = plt.subplots(1, len(rows_ep), figsize=(3.4 * len(rows_ep), 3.4))
for ax, r in zip(axs, rows_ep):
    m = np.asarray(r["tm_all"], dtype=float)
    m = m / max(m.sum(), 1)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="red" if i != j else "black")
    ax.set_xticks(range(3), STATES, fontsize=8)
    ax.set_yticks(range(3), STATES, fontsize=8)
    ax.set_xlabel(f"t+{r['offset']}")
    ax.set_title(f"k={r['offset']} (all region)", fontsize=9)
fig.suptitle(f"Hit/Free/Unknown transition, {ep}", fontsize=10)
fig.tight_layout()
fig.savefig(outdir / "transition_heatmaps_all.png", dpi=130)
plt.close(fig)

# 2) physical vs visibility flip decomposition per offset (mean over episodes)
offs = sorted({r["offset"] for r in rows})
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, region in zip(axs, ["all", "table"]):
    phys, vis, unk_reveal = [], [], []
    for k in offs:
        rs = [r for r in rows if r["offset"] == k]
        phys.append(np.mean([r[f"{region}_physical_flip_rate"] for r in rs]))
        vis.append(np.mean([r[f"{region}_vis_flip_rate"] for r in rs]))
        unk_reveal.append(np.mean([r[f"{region}_unknown_to_observed"] for r in rs]))
    x = np.arange(len(offs))
    ax.bar(x - 0.2, phys, 0.4, label="physical flip (hit<->free)")
    ax.bar(x + 0.2, vis, 0.4, label="visibility flip (obs<->unknown)")
    ax.plot(x, unk_reveal, "k.--", label="unknown->observed (reveal)")
    ax.set_xticks(x, [f"t+{k}" for k in offs])
    ax.set_ylabel("voxel fraction")
    ax.set_title(f"region={region}", fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("Decoupling evidence: physical vs visibility change rates", fontsize=10)
fig.tight_layout()
fig.savefig(outdir / "physical_vs_visibility_flips.png", dpi=130)
plt.close(fig)
print("plots saved:", list(p.name for p in outdir.iterdir()))
