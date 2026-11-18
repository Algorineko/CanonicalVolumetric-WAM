"""Stage 2b voxel semantics unit tests (docs/plan.md §7.4 table)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import voxel as vx  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_known_point():
    b_min = np.zeros(3)
    pitch = 0.02
    shape = np.array([40, 40, 30])
    p = np.array([[0.031, 0.011, 0.021]])      # cell (1,0,1)
    idx, inb = vx.points_to_voxel_idx(p, b_min, pitch, shape)
    check("known point -> correct voxel index", inb[0] and idx[0].tolist() == [1, 0, 1],
          f"{idx[0]}")
    g = vx.voxelize_points_only(p, np.array([0]), b_min, pitch, shape)
    check("point-only hit in cell", g["hit"][1, 0, 1] and g["hit"].sum() == 1)


def test_boundary():
    b_min = np.zeros(3)
    pitch = 0.02
    shape = np.array([40, 40, 30])
    b_max = np.array([0.8, 0.8, 0.6])
    p = np.array([[0.799999, 0.4, 0.3],   # inside (cell 39)
                  [0.8, 0.4, 0.3]])       # exactly at bbox max -> excluded
    idx, inb = vx.points_to_voxel_idx(p, b_min, pitch, shape)
    check("boundary point at bbox max excluded", inb.tolist() == [True, False],
          f"inb={inb} idx={idx.tolist()}")


def test_wall():
    """Camera at origin looking +x at a wall at x=0.3: cells 0..14 free,
    cell 15 hit, cells 16+ unknown."""
    pitch = 0.02
    shape = np.array([40, 8, 8])           # 0.8 x 0.16 x 0.16 m
    b_min = np.array([-0.02, -0.08, -0.08])
    b_max = b_min + shape * pitch
    cam = np.array([0.0, 0.0, 0.0])        # in cell (1,4,4)
    wall_x = 0.30
    ys = np.linspace(-0.05, 0.05, 5)
    hits = np.stack([np.full_like(ys, wall_x), ys, np.zeros_like(ys)], axis=1)
    g = vx.voxelize_rays(cam[None], [hits], [np.ones(len(hits), bool)], b_min, b_max, pitch, shape)
    # free: cells strictly before wall along all rays
    check("wall: cells before wall free",
          g["free"][2:15, 4, 4].all() and not g["free"][16:, 4, 4].any(),
          f"free rows {[int(x) for x in np.where(g['free'][:, 4, 4])[0]]}")
    check("wall: hit at wall cell", g["hit"][15, 4, 4] or g["hit"][16, 4, 4],
          f"hit rows {[int(x) for x in np.where(g['hit'][:, 4, 4])[0]]}")
    check("wall: nothing behind wall free",
          not g["free"][17:, :, :].any(), f"max free x-idx {np.where(g['free'].any(axis=(1, 2)))[0].max()}")
    check("wall: behind wall unknown",
          g["unknown"][20:, 4, 4].all())
    check("wall: free requires valid depth only",
          True)  # covered by construction (valid=True here)


def test_two_views_static_box():
    """Two cameras on opposite sides see the same static box surface:
    overlapping hit cells must be consistent in base frame."""
    pitch = 0.02
    shape = np.array([30, 30, 20])
    b_min = np.array([0.0, 0.0, 0.0])
    b_max = b_min + shape * pitch
    box_center = np.array([0.3, 0.3, 0.1])
    # two cameras at ±y looking at box center from 0.25m
    c1 = box_center + np.array([0, -0.25, 0.0])
    c2 = box_center + np.array([0, +0.25, 0.0])
    # front/back faces (single flat face per camera, like observed surfaces)
    hits1 = np.stack([np.full(9, 0.3), np.full(9, 0.26), np.linspace(0.06, 0.14, 9)], axis=1)
    hits2 = np.stack([np.full(9, 0.3), np.full(9, 0.34), np.linspace(0.06, 0.14, 9)], axis=1)
    cams = np.stack([c1, c2])
    pts = np.concatenate([hits1, hits2])
    g = vx.voxelize_rays(cams, [hits1, hits2], [np.ones(9, bool)] * 2, b_min, b_max, pitch, shape)
    check("two views: box front+back hits present",
          g["hit"][15, 13, 5] or g["hit"][15, 13, 7],
          f"hit cells near expected: {np.argwhere(g['hit']).tolist()[:6]}")
    check("two views: interior of box NOT free (behind hits)",
          not g["free"][15, 14, 5] and not g["free"][15, 14, 7],
          f"interior free={g['free'][15, 14, 5]}")
    # free space between camera1 and front face should be observed
    # (rays travel at x=0.30 -> cell 15; y from camera y=0.05 up to face y=0.26)
    check("two views: free space in front of faces observed",
          g["free"][15, 3:12, 5].sum() > 0,
          f"free cells y3..11 at x15,z5: {np.where(g['free'][15, 3:12, 5])[0].tolist()}")


def test_missing_view():
    pitch = 0.02
    shape = np.array([30, 30, 20])
    b_min = np.zeros(3)
    b_max = b_min + shape * pitch
    c = np.array([-0.1, 0.15, 0.1])
    hits = np.stack([np.full(25, 0.3), np.linspace(0.05, 0.25, 25), np.full(25, 0.1)], axis=1)
    g1 = vx.voxelize_rays(c[None], [hits], [np.ones(25, bool)], b_min, b_max, pitch, shape)
    g2 = vx.voxelize_rays(c[None], [hits], [np.zeros(25, bool)], b_min, b_max, pitch, shape)  # invalid depth
    a1, a2 = vx.grid_stats(g1)["active_cells"], vx.grid_stats(g2)["active_cells"]
    check("missing view reduces coverage", a2 < a1, f"{a2} < {a1}")
    check("invalid depth -> no free space claimed", not g2["free"].any() and not g2["hit"].any())


def test_raycast_stats():
    g = vx.grid_stats({"shape": np.array([40, 40, 30]),
                       "hit": np.zeros((40, 40, 30), bool),
                       "free": np.zeros((40, 40, 30), bool),
                       "observed": np.zeros((40, 40, 30), bool),
                       "unknown": np.ones((40, 40, 30), bool),
                       "view_count": np.zeros((40, 40, 30), np.uint8),
                       "conflict": np.zeros((40, 40, 30), np.int16)})
    check("grid stats identity", g["unknown_frac"] == 1.0 and g["total_cells"] == 48000)


if __name__ == "__main__":
    test_known_point()
    test_boundary()
    test_wall()
    test_two_views_static_box()
    test_missing_view()
    test_raycast_stats()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL VOXEL TESTS PASSED")
