"""Visualization helpers: RGB rows, point-cloud renders, voxel slices.

Uses matplotlib Agg; runs under the PhysAdaptWAM .venv-wam python or any env
with matplotlib. No GPU required.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def save_rgb_row(rgb: np.ndarray, names, path, title=""):
    """rgb [V,H,W,3] uint8 -> one row of labeled images."""
    v = rgb.shape[0]
    fig, axs = plt.subplots(1, v, figsize=(3.2 * v, 2.8))
    if v == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        ax.imshow(rgb[i])
        ax.set_title(names[i], fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_points(xyz: np.ndarray, colors: np.ndarray, path, title="",
                  elev=-72, azim=-90, max_pts=60000, equal_aspect=True):
    """Scatter render of a point cloud viewed from above (-z up view)."""
    if len(xyz) > max_pts:
        sel = np.random.default_rng(0).choice(len(xyz), max_pts, replace=False)
        xyz, colors = xyz[sel], colors[sel]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    c = np.asarray(colors, dtype=np.float64) / 255.0
    if c.ndim == 1 or c.shape[1] != 3:
        c = np.repeat(np.asarray(c).reshape(-1, 1), 3, axis=1) if c.ndim == 1 or c.shape[1] == 1 else c
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.clip(c, 0, 1), s=0.3, linewidths=0)
    ax.set_xlabel("x base (fwd)")
    ax.set_ylabel("y base")
    ax.set_zlabel("z base (up)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.7))
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_two_views(xyz_a, col_a, xyz_b, col_b, path, title_a="", title_b="",
                     elev=-72, azim=-90):
    fig = plt.figure(figsize=(13, 5.5))
    for j, (xyz, col, tt) in enumerate([(xyz_a, col_a, title_a), (xyz_b, col_b, title_b)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        if len(xyz) > 60000:
            sel = np.random.default_rng(0).choice(len(xyz), 60000, replace=False)
            xyz, col = xyz[sel], col[sel]
        c = np.asarray(col, dtype=np.float64)
        if c.max() > 1.5:
            c = c / 255.0
        if c.ndim == 1:
            c = np.repeat(c.reshape(-1, 1), 3, axis=1)
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=np.clip(c, 0, 1), s=0.3, linewidths=0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.7))
        ax.set_title(tt, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def voxel_slices(hit, free, unknown, b_min, pitch, path, frame_tag=""):
    """3 orthogonal slices through the grid center; hit=red, free=cyan, unknown=dark."""
    shape = np.asarray(hit.shape)
    ci, cj, ck = shape // 2
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    views = [
        (hit[ci, :, :], free[ci, :, :], unknown[ci, :, :], f"x-slice i={ci} (y-z)"),
        (hit[:, cj, :], free[:, cj, :], unknown[:, cj, :], f"y-slice j={cj} (x-z)"),
        (hit[:, :, ck], free[:, :, ck], unknown[:, :, ck], f"z-slice k={ck} (x-y)"),
    ]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#101018", "#59c2c9", "#d94f43"])  # unknown/free/hit
    for ax, (h, f, u, tt) in zip(axs, views):
        img = np.zeros(h.shape, dtype=np.int64)
        img[u] = 0
        img[f] = 1
        img[h] = 2
        ax.imshow(img.T, origin="lower", cmap=cmap, vmin=0, vmax=2,
                  extent=[0, h.shape[0], 0, h.shape[1]], interpolation="nearest")
        ax.set_title(f"{tt}  {frame_tag}", fontsize=9)
        ax.set_xlabel("voxels")
    fig.suptitle("red=hit  cyan=free  dark=unknown", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_voxels_3d(hit, path, max_cells=20000, title=""):
    """3D scatter of hit voxels (red) — matplotlib, no external renderer needed."""
    idx = np.argwhere(hit)
    if len(idx) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "no hit voxels", ha="center")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return
    if len(idx) > max_cells:
        sel = np.random.default_rng(0).choice(len(idx), max_cells, replace=False)
        idx = idx[sel]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(idx[:, 0], idx[:, 1], idx[:, 2], c="#d94f43", s=1.5, linewidths=0)
    ax.set_xlabel("i (x)")
    ax.set_ylabel("j (y)")
    ax.set_zlabel("k (z)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
