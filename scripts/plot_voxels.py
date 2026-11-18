"""Render Stage 2b voxel visualizations: orthogonal slices + 3D hit cells."""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import viz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--pitches", nargs="+", default=["20mm", "10mm"])
    ap.add_argument("--sources", nargs="+", default=["gt"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    vox_root = Path(cfg["cache_dir"]) / "voxels"
    out = Path("runs/smoke/plots")
    out.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        for pitch in args.pitches:
            files = sorted((vox_root / source / f"pitch{pitch}" / "_".join([]) if False
                            else vox_root / source / f"pitch{pitch}").glob("**/*_ray.npz"))
            if not files:
                print(f"none for {source}/{pitch}")
                continue
            for f in files[:3]:  # first 3 samples
                d = np.load(f)
                sid = str(f.parent.relative_to(vox_root / source / f"pitch{pitch}"))
                tag = f"{source} {pitch} {sid}"
                viz.voxel_slices(d["hit"], d["free"], d["unknown"],
                                 d["b_min"], float(d["pitch"]),
                                 out / f"voxel_slices_{source}_{pitch}_"
                                 f"{sid.replace('/', '_')}.png", frame_tag=tag)
                viz.render_voxels_3d(d["hit"],
                                     out / f"voxel3d_{source}_{pitch}_"
                                     f"{sid.replace('/', '_')}.png", title=tag)
                print(f"rendered {tag}")
    print("done")


if __name__ == "__main__":
    main()
