"""GPU (torch) port of cvwam.voxel.voxelize_rays — same semantics, ~50ms for
3 cameras vs ~1.5s numpy. Used by the Stage 4a policy adapter for online
voxel construction during closed-loop rollouts (numpy version stays the
offline/training reference).

Semantics preserved exactly (validated against numpy on cached frames):
  per-camera rays only; hit wins over free; pitch/2 marching stops half a
  cell short of the hit; conflicts counted; invalid points produce no rays.
"""

import numpy as np
import torch


def voxelize_rays_gpu(cam_centers, hit_pts_list, valid_list, b_min, b_max,
                      pitch, shape, device="cuda:0"):
    shape_t = tuple(int(v) for v in shape)
    hit = torch.zeros(shape_t, dtype=torch.bool, device=device)
    free = torch.zeros(shape_t, dtype=torch.bool, device=device)
    conflict = torch.zeros(shape_t, dtype=torch.int16, device=device)
    view_count = torch.zeros(shape_t, dtype=torch.uint8, device=device)
    hit_seen_by = torch.zeros(shape_t, dtype=torch.uint8, device=device)

    b_min_t = torch.as_tensor(b_min, dtype=torch.float64, device=device)
    pitch_t = float(pitch)
    inv_pitch = 1.0 / pitch_t

    def to_idx(p):  # p (N,3) float64 torch
        return torch.floor((p - b_min_t) * inv_pitch).long()

    for s, c in enumerate(cam_centers):
        ends = torch.as_tensor(np.asarray(hit_pts_list[s], dtype=np.float64),
                               device=device)
        ok_v = torch.as_tensor(np.asarray(valid_list[s], dtype=bool), device=device)
        cam = torch.as_tensor(np.asarray(c, dtype=np.float64), device=device)
        hin = ((ends >= b_min_t) & (ends < b_min_t + pitch_t *
               torch.tensor(shape_t, device=device, dtype=torch.float64))).all(-1)
        h_idx = to_idx(ends)
        sel = hin & ok_v
        e_ok = h_idx[sel]
        seen_hit = torch.zeros(shape_t, dtype=torch.bool, device=device)
        seen_hit[e_ok[:, 0], e_ok[:, 1], e_ok[:, 2]] = True
        hit |= seen_hit
        hit_seen_by += seen_hit.to(torch.uint8)

        ends_v = ends[sel]
        if len(ends_v) == 0:
            continue
        ev = cam[None, :].expand(len(ends_v), 3)
        t0, t1 = clip_segment_to_bbox_gpu(ev, ends_v, b_min_t,
                                          torch.as_tensor(b_max, dtype=torch.float64,
                                                          device=device))
        ok = (t0 < t1) & (t1 > 0)
        if not ok.any():
            continue
        ev, ends_ok, t0, t1 = ev[ok], ends_v[ok], t0[ok], t1[ok]
        ray_len = torch.linalg.norm(ends_ok - ev, dim=1)
        t_march = t1 - (pitch_t / 2) / torch.clamp(ray_len, min=1e-12)
        t_march = torch.minimum(t_march, t1)
        starts = ev + t0[:, None] * (ends_ok - ev)
        stops = ev + t_march[:, None] * (ends_ok - ev)
        seg = stops - starts
        n_steps = torch.clamp((torch.linalg.norm(seg, dim=1) / (pitch_t / 2))
                              .ceil().long(), min=1)
        sf = torch.zeros(shape_t, dtype=torch.bool, device=device)
        chunk = 400_000
        i = 0
        while i < len(starts):
            j = min(i + chunk, len(starts))
            ns = n_steps[i:j]
            maxn = int(ns.max())
            tt = (torch.arange(maxn, device=device, dtype=torch.float64)[None, :]
                  + 0.5) / torch.clamp(ns[:, None], min=1)
            m = tt < 1.0
            P = starts[i:j][:, None, :] + tt[..., None] * seg[i:j][:, None, :]
            idx = torch.floor((P - b_min_t) * inv_pitch).long()
            inb = ((idx >= 0) & (idx < torch.tensor(shape_t, device=device))).all(-1) & m
            sel_idx = idx[inb]                      # (M,3)
            ii, jj, kk = sel_idx[:, 0], sel_idx[:, 1], sel_idx[:, 2]
            sf[ii, jj, kk] = True
            i = j
        conflict += (sf & seen_hit).to(torch.int16)
        free |= sf & ~seen_hit
        view_count += (sf | seen_hit).to(torch.uint8)

    conflict += ((hit_seen_by > 0) & free).to(torch.int16)
    free = free & ~hit
    observed = hit | free
    return dict(hit=hit, free=free, observed=observed, unknown=~observed,
                view_count=view_count, conflict=conflict,
                shape=np.asarray(shape_t))


def clip_segment_to_bbox_gpu(starts, ends, b_min, b_max):
    """Slab clip for segments start->end (torch port of voxel.clip_segment_to_bbox)."""
    d = ends - starts
    t0 = torch.zeros(len(starts), dtype=torch.float64, device=ends.device)
    t1 = torch.ones(len(starts), dtype=torch.float64, device=ends.device)
    for k in range(3):
        lo, hi = b_min[k], b_max[k]
        s, e, dd = starts[:, k], ends[:, k], d[:, k]
        parallel = dd.abs() < 1e-12
        with torch.no_grad():
            t_near = torch.where(dd != 0, (lo - s) / torch.where(dd == 0, 1.0, dd),
                                 torch.full_like(s, -1e18))
            t_far = torch.where(dd != 0, (hi - s) / torch.where(dd == 0, 1.0, dd),
                                torch.full_like(s, 1e18))
        tn = torch.minimum(t_near, t_far)
        tf = torch.maximum(t_near, t_far)
        inside_p = (s >= lo) & (s <= hi)
        t_near = torch.where(parallel, torch.where(inside_p, t0, torch.full_like(s, 1e18)), tn)
        t_far = torch.where(parallel, torch.where(inside_p, t1, torch.full_like(s, -1e18)), tf)
        t0 = torch.maximum(t0, t_near)
        t1 = torch.minimum(t1, t_far)
    return t0, t1
