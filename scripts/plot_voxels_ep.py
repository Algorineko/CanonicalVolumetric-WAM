"""User-facing canonical voxel visualizations (metric base-frame space).

Panels per requested frame:
  A) 3D overview: hit(red) / free(cyan) / unknown(dark) + door_region(green)
     + table plane outline + robot-base marker.
  B) Orthogonal slices through the densest occupied plane (door region tinted).
  C) Dynamics t -> t+k: kept(gray) / new(orange) / cleared(blue) vs door
     sweep union (green).
Outputs PNGs under runs/stage3b/viz/. matplotlib only, no GPU.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_HIT, C_FREE, C_UNK = "#d94f43", "#59c2c9", "#2f2f3a"
C_DOOR, C_NEW, C_CLEAR, C_KEEP = "#3fbf6f", "#f28e2b", "#4e79a7", "#8a8a93"


def load_frame(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def to_metric(idx, b_min, pitch):
    return b_min + (idx + 0.5) * pitch


def _table_plane(ax, b_min, b_max, table_z):
    xs = [b_min[0], b_max[0], b_max[0], b_min[0], b_min[0]]
    ys = [b_min[1], b_min[1], b_max[1], b_max[1], b_min[1]]
    ax.plot(xs, ys, [table_z] * 5, color="#666666", ls="--", lw=0.8)


def panel_overview(vox, b_min, b_max, pitch, table_z, title, path,
                   max_free=25000, max_unk=15000):
    hit_idx = np.argwhere(vox["hit"])
    free_idx = np.argwhere(vox["free"] & ~vox["hit"])
    unk_idx = np.argwhere(vox["unknown"])
    if len(free_idx) > max_free:
        free_idx = free_idx[:: len(free_idx) // max_free + 1]
    if len(unk_idx) > max_unk:
        unk_idx = unk_idx[:: len(unk_idx) // max_unk + 1]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection="3d")
    xyz = to_metric(unk_idx, b_min, pitch)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=C_UNK, s=0.6, linewidths=0,
               label=f"unknown ({int(vox['unknown'].sum())})")
    xyz = to_metric(free_idx, b_min, pitch)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=C_FREE, s=0.8, linewidths=0,
               label=f"free observed ({int((vox['free'] & ~vox['hit']).sum())})")
    xyz = to_metric(hit_idx, b_min, pitch)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=C_HIT, s=2.2, linewidths=0,
               label=f"hit occupied ({int(vox['hit'].sum())})")
    if "door_region" in vox:
        xyz = to_metric(np.argwhere(vox["door_region"]), b_min, pitch)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=C_DOOR, s=3.0,
                   linewidths=0, label="door_region (GT)")
    if "obj_region" in vox:
        xyz = to_metric(np.argwhere(vox["obj_region"]), b_min, pitch)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="#ffffff", s=1.0,
                   linewidths=0, alpha=0.5, label="obj_region (microwave body)")
    _table_plane(ax, b_min, b_max, table_z)
    ax.scatter([0], [0], [0], c="k", marker="X", s=80, label="robot base")
    ax.set_xlabel("x base fwd (m)"); ax.set_ylabel("y base (m)"); ax.set_zlabel("z up (m)")
    ax.set_xlim(b_min[0], b_max[0]); ax.set_ylim(b_min[1], b_max[1])
    ax.set_zlim(b_min[2], b_max[2])
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def panel_slices(vox, b_min, pitch, title, path):
    """y-z slice at the x-plane with most door/hit mass + center x/y slices."""
    hit, door = vox["hit"], vox.get("door_region")
    xs = hit.sum(axis=(1, 2))
    ci = int(np.argmax(xs)) if door is None else int(
        np.argmax(hit.sum(axis=(1, 2)) + 3 * door.sum(axis=(1, 2))))
    cj, ck = vox["hit"].shape[1] // 2, vox["hit"].shape[2] // 2
    zs = hit.sum(axis=(0, 1))
    nz = np.nonzero(zs)[0]
    ck = int(nz.mean()) if len(nz) else ck

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#101018", "#59c2c9", "#d94f43", "#3fbf6f"])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    views = [
        (hit[ci], vox["free"][ci], vox["unknown"][ci], door[ci] if door is not None else None,
         f"x-slice i={ci} (y-z)"),
        (hit[:, cj], vox["free"][:, cj], vox["unknown"][:, cj], door[:, cj] if door is not None else None,
         f"y-slice j={cj} (x-z)"),
        (hit[:, :, ck], vox["free"][:, :, ck], vox["unknown"][:, :, ck],
         door[:, :, ck] if door is not None else None, f"z-slice k={ck} (x-y)"),
    ]
    sh = vox["hit"].shape
    for ax, (h, f, u, dr, tt) in zip(axs, views):
        img = np.zeros(h.shape, dtype=np.int64)
        img[u] = 0
        img[f] = 1
        img[h] = 2
        if dr is not None:
            img[dr] = 3
        extent = [0, h.shape[0] * pitch, 0, h.shape[1] * pitch]
        ax.imshow(img.T, origin="lower", cmap=cmap, vmin=0, vmax=3,
                  extent=extent, interpolation="nearest")
        ax.set_title(f"{tt}", fontsize=9)
        ax.set_xlabel("meters")
    fig.suptitle(title + "   dark=unknown cyan=free red=hit green=door_region",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def panel_dynamics(vox_a, vox_b, b_min, b_max, pitch, table_z, k, title, path):
    a, b = vox_a["hit"], vox_b["hit"]
    kept = a & b
    new = b & ~a
    cleared = a & ~b
    sweep = vox_b.get("door_region") | vox_a.get("door_region", np.zeros_like(a))
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection="3d")
    for m, c, s, lb in ((kept, C_KEEP, 1.4, f"kept ({int(kept.sum())})"),
                        (cleared, C_CLEAR, 3.0, f"cleared ({int(cleared.sum())})"),
                        (new, C_NEW, 3.0, f"new ({int(new.sum())})")):
        xyz = to_metric(np.argwhere(m), b_min, pitch)
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=s, linewidths=0, label=lb)
    if "door_region" in vox_b:
        xyz = to_metric(np.argwhere(sweep), b_min, pitch)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=C_DOOR, s=2.0,
                   linewidths=0, alpha=0.35, label="door sweep (t..t+k GT)")
    _table_plane(ax, b_min, b_max, table_z)
    ax.scatter([0], [0], [0], c="k", marker="X", s=80)
    ax.set_xlabel("x base fwd (m)"); ax.set_ylabel("y base (m)"); ax.set_zlabel("z up (m)")
    ax.set_xlim(b_min[0], b_max[0]); ax.set_ylim(b_min[1], b_max[1])
    ax.set_zlim(b_min[2], b_max[2])
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pick_most_dynamic(ep_dir, k=16, field="door_region"):
    frames = sorted(ep_dir.glob("frame_*.npz"))
    best_t, best_v = 0, -1.0
    for i, f in enumerate(frames):
        if i + k >= len(frames):
            break
        if i % 8:  # stride scan
            continue
        da = load_frame(f)
        db = load_frame(frames[i + k])
        v = float((da[field].astype(bool) ^ db[field].astype(bool)).sum())
        if v > best_v:
            best_v, best_t = v, i
    return frames[best_t], frames[best_t + k], best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep-dir", required=True, help="path to episode 20mm dir")
    ap.add_argument("--frames", nargs="*", type=int, default=None,
                    help="frame indices; default = auto most-dynamic + one static")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out-dir", default="runs/stage3b/viz")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ep = Path(args.ep_dir)
    frames = sorted(ep.glob("frame_*.npz"))
    mf_path = ep / "manifest.json"
    import json
    mf = json.loads(mf_path.read_text()) if mf_path.exists() else {}
    b_min = np.asarray(mf.get("b_min", [0.3, -0.65, 0.71]), dtype=np.float64)
    b_max = np.asarray(mf.get("b_max", [1.1, 0.65, 1.6]), dtype=np.float64)
    pitch = float(mf.get("pitch_m", 0.02))
    table_z = float(mf.get("table_z_base", 0.7401))
    tag = mf.get("variant", "?") + f"_ep{mf.get('episode', '?')}"

    if args.frames is None:
        f_dyn, f_fut, score = pick_most_dynamic(ep, k=args.k)
        t_dyn = int(f_dyn.stem.split("_")[1])
        t_sta = len(frames) // 2
        picks = [("dyn", t_dyn), ("mid", t_sta)]
        print(f"auto-picked dyn t={t_dyn} (door change {score:.0f} voxels), mid t={t_sta}")
    else:
        picks = [("f", t) for t in args.frames]

    for label, t in picks:
        vox = load_frame(frames[t])
        name = f"{tag}_{label}_t{t:04d}"
        panel_overview(vox, b_min, b_max, pitch, table_z,
                       f"{name}  door_angle={float(vox.get('door_angle', np.nan)):.2f} rad",
                       out / f"{name}_overview.png")
        panel_slices(vox, b_min, pitch, name, out / f"{name}_slices.png")
        if t + args.k < len(frames):
            vox_b = load_frame(frames[t + args.k])
            panel_dynamics(vox, vox_b, b_min, b_max, pitch, table_z, args.k,
                           f"{name}  t->{t+args.k} (k={args.k})",
                           out / f"{name}_dynamics_k{args.k}.png")
        print(f"[done] {name}")
    print("all done")


if __name__ == "__main__":
    main()
