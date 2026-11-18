"""Canonical voxelization with visibility semantics (plan.md §7.2).

Channels per cell:
  hit        observed surface endpoint lies in the cell
  free       a valid depth ray traversed the cell strictly BEFORE its hit
  observed   hit OR free
  unknown    NOT observed
  view_count number of distinct cameras contributing evidence
  conflict   cells with contradictory free/hit evidence (recorded; hit wins:
             final free excludes hit cells)

Implementation: ray marching at pitch/2 from the camera center to the hit
point, segment clipped to the workspace bbox (slab method) since cameras sit
outside the bbox (e.g. head cam x=0.2 < 0.30). Ray marching at half-pitch is a
documented approximation of exact 3D DDA; the §7.4 unit tests use
axis-aligned cases where it is exact. Missing depth is never evidence of free
space. Point-only mode (hit without rays) is a comparison baseline, not
ray-based occupancy fusion.
"""

import numpy as np


def grid_config(b_min, b_max, pitch):
    b_min = np.asarray(b_min, dtype=np.float64)
    b_max = np.asarray(b_max, dtype=np.float64)
    shape = np.ceil((b_max - b_min) / pitch - 1e-9).astype(np.int64)
    return b_min, b_max, shape


def points_to_voxel_idx(points: np.ndarray, b_min: np.ndarray, pitch: float,
                        shape: np.ndarray):
    """plan.md §7.2: idx = floor((p - b_min)/pitch); points at bbox max excluded."""
    idx = np.floor((np.asarray(points, dtype=np.float64) - b_min) / pitch).astype(np.int64)
    in_bounds = ((idx >= 0) & (idx < shape)).all(axis=-1)
    return idx, in_bounds


def voxelize_points_only(xyz, view_id, b_min, pitch, shape):
    """Point-only hit grid (no free space) — comparison baseline."""
    shape = tuple(int(v) for v in shape)
    hit = np.zeros(shape, dtype=bool)
    idx, inb = points_to_voxel_idx(xyz, b_min, pitch, shape)
    hit[idx[inb, 0], idx[inb, 1], idx[inb, 2]] = True
    vc = np.zeros(shape, dtype=np.uint8)
    for v in np.unique(view_id[inb]):
        m = inb & (view_id == v)
        np.add.at(vc, (idx[m, 0], idx[m, 1], idx[m, 2]), 1)
    return {"hit": hit, "free": np.zeros(shape, bool),
            "observed": hit.copy(), "unknown": ~hit,
            "view_count": vc, "conflict": np.zeros(shape, np.int16)}


def clip_segment_to_bbox(p0: np.ndarray, p1: np.ndarray, b_min: np.ndarray,
                         b_max: np.ndarray):
    """Slab clip of segments [p0,p1] to the axis-aligned bbox.

    Returns (t0, t1) arrays (N,) in [0,1]; t0>=t1 means no intersection.
    """
    p0 = np.atleast_2d(p0)
    p1 = np.atleast_2d(p1)
    d = p1 - p0
    t0 = np.zeros(len(d))
    t1 = np.ones(len(d))
    for ax in range(3):
        da = d[:, ax]
        lo, hi = b_min[ax], b_max[ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo - p0[:, ax]) / np.where(da != 0, da, 1)
            tb = (hi - p0[:, ax]) / np.where(da != 0, da, 1)
        ta = np.where(da != 0, ta, np.where(p0[:, ax] < lo, -np.inf, np.where(
            p0[:, ax] > hi, np.inf, -np.inf)))
        tb = np.where(da != 0, tb, np.where(p0[:, ax] < lo, -np.inf, np.where(
            p0[:, ax] > hi, np.inf, np.inf)))
        tmin = np.minimum(ta, tb)
        tmax = np.maximum(ta, tb)
        t0 = np.maximum(t0, tmin)
        t1 = np.minimum(t1, tmax)
    return t0, t1


def voxelize_rays(cam_centers, hit_pts_list, valid_list, b_min, b_max, pitch, shape):
    """Fuse multi-view rays into hit/free grids (vectorized ray marching).

    cam_centers: (S,3) in base frame; hit_pts_list[k]: (N_k,3) points observed
    by camera k ONLY (a camera never casts rays through other cameras' points);
    valid_list[k]: (N_k,) bool. Rays connect each camera to its own valid hits.
    """
    shape = tuple(int(v) for v in shape)
    hit = np.zeros(shape, dtype=bool)
    free = np.zeros(shape, dtype=bool)
    conflict = np.zeros(shape, dtype=np.int16)
    view_count = np.zeros(shape, dtype=np.uint8)
    hit_seen_by = np.zeros(shape, dtype=np.uint8)

    for s, c in enumerate(cam_centers):
        ends = np.asarray(hit_pts_list[s], dtype=np.float64)
        ok_v = np.asarray(valid_list[s], dtype=bool)
        h_idx, hin = points_to_voxel_idx(ends, b_min, pitch, shape)
        seen_hit = np.zeros(shape, dtype=bool)
        e_ok = h_idx[hin & ok_v]
        seen_hit[e_ok[:, 0], e_ok[:, 1], e_ok[:, 2]] = True
        hit |= seen_hit
        hit_seen_by += seen_hit.astype(np.uint8)

        ends = ends[hin & ok_v]
        if len(ends) == 0:
            continue
        ev = np.repeat(c[None], len(ends), axis=0)
        t0, t1 = clip_segment_to_bbox(ev, ends, b_min, b_max)
        ok = (t0 < t1) & (t1 > 0)
        if not ok.any():
            continue
        # stop half a pitch (in distance) short of the hit so marching never
        # marks the hit cell itself free
        ray_len = np.linalg.norm(ends[ok] - ev[ok], axis=1)
        t_march = t1[ok] - (pitch / 2) / np.maximum(ray_len, 1e-12)
        t_march = np.minimum(t_march, t1[ok])
        starts = ev[ok] + (t0[ok][:, None]) * (ends[ok] - ev[ok])
        stops = ev[ok] + (t_march[:, None]) * (ends[ok] - ev[ok])
        seg = stops - starts
        n_steps = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) / (pitch / 2)).astype(int), 1)
        sf = np.zeros(shape, dtype=bool)
        # free-cell marking in chunks to bound memory
        chunk = 200_000
        i = 0
        while i < len(starts):
            j = min(i + chunk, len(starts))
            # per-ray step counts; sample uniformly at pitch/2 or finer
            ns = n_steps[i:j]
            maxn = int(ns.max())
            tt = (np.arange(maxn)[None, :] + 0.5) / np.maximum(ns[:, None], 1)
            m = tt < 1.0
            P = starts[i:j][:, None, :] + tt[..., None] * seg[i:j][:, None, :]
            idx = np.floor((P - b_min) / pitch).astype(np.int64)
            inb = ((idx >= 0) & (idx < np.asarray(shape))).all(-1) & m
            ii, jj, kk = idx[inb, 0], idx[inb, 1], idx[inb, 2]
            sf[ii, jj, kk] = True
            i = j
        conflict += (sf & seen_hit).astype(np.int16)
        free |= sf & ~seen_hit
        view_count += (sf | seen_hit).astype(np.uint8)

    conflict += ((hit_seen_by > 0) & free).astype(np.int16)
    free = free & ~hit  # hit wins in the hard-grid implementation
    observed = hit | free
    return {"hit": hit, "free": free, "observed": observed,
            "unknown": ~observed, "view_count": view_count,
            "conflict": conflict, "shape": np.asarray(shape)}


def voxelize_sample(pointcloud_npz_path, pitch, point_only=False):
    """Build voxel grids for one exported point-cloud npz (GT or VGGT)."""
    d = np.load(pointcloud_npz_path)
    b_min = d["b_min"].astype(np.float64)
    b_max = d["b_max"].astype(np.float64)
    _, _, shape = grid_config(b_min, b_max, pitch)
    xyz = d["xyz"]
    view_id = d["view_id"]
    valid = ((xyz >= b_min) & (xyz <= b_max)).all(1)
    if point_only:
        g = voxelize_points_only(xyz[valid], view_id[valid], b_min, pitch, shape)
    else:
        T = d["T_base_cam"]
        centers = np.stack([T[v][:3, 3] for v in range(T.shape[0])])
        pts_list = [xyz[(view_id == v) & valid] for v in range(T.shape[0])]
        g = voxelize_rays(centers, pts_list,
                          [np.ones(len(p), bool) for p in pts_list],
                          b_min, b_max, pitch, shape)
    g["b_min"] = b_min
    g["pitch"] = pitch
    g["shape"] = np.asarray(shape)
    return g


def grid_stats(g):
    n = float(np.prod(g["shape"]))
    return {
        "grid_shape": list(map(int, g["shape"])),
        "hit_frac": round(float(g["hit"].mean()), 5),
        "free_frac": round(float(g["free"].mean()), 5),
        "unknown_frac": round(float(g["unknown"].mean()), 5),
        "active_frac": round(float(g["observed"].mean()), 5),
        "conflict_frac": round(float((g["conflict"] > 0).mean()), 5),
        "active_cells": int(g["observed"].sum()),
        "total_cells": int(n),
    }
