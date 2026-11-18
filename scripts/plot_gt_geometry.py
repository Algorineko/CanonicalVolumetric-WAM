"""Render GT fused point clouds + RGB rows for visual G1 checks."""

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import viz  # noqa: E402

cfg = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke.yaml").read_text())
cache = Path(cfg["cache_dir"])
out = Path("runs/smoke/plots")
out.mkdir(parents=True, exist_ok=True)

manifest = [l.split('"sample_id": "')[1].split('"')[0]
            for l in (cache / "frames" / "manifest.jsonl").read_text().splitlines()]

for i in [0, 5, 10, 20, 24]:
    sid = manifest[i]
    variant, ep, t = sid.split("/")
    npz = cache / "pointclouds" / "gt" / variant / ep / f"frame_{t.split('=')[1].zfill(7)}.npz"
    d = np.load(npz)
    f_npz = cache / "frames" / variant / ep / f"frame_{t.split('=')[1].zfill(7)}.npz"
    s = np.load(f_npz)
    tag = f"{variant}/{ep}/t={t}"
    viz.save_rgb_row(s["rgb"], ["head", "left_wrist", "right_wrist"],
                     out / f"rgb_{i:02d}.png", title=tag)
    viz.render_two_views(
        d["xyz_bbox"], d["rgb_bbox"], d["xyz_bbox"], d["view_id_bbox"],
        out / f"gt_cloud_{i:02d}.png",
        title_a=f"GT fused (RGB) {tag}", title_b="view_id: 0=head 1=left 2=right")
    print(f"rendered {tag} -> gt_cloud_{i:02d}.png")
print("done")
