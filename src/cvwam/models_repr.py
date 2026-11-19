"""Stage 3c representation-comparison models (plan §3).

All three share TinyDynamicsNet's conditioning (qpos+action FiLM) and its
3D conv decoder family, and emit the SAME decoupled head semantics as P1-V
(per offset: occupancy logit + visibility logit on the canonical voxel grid)
so evaluation stays in one unified 3D space. Differences are ONLY the encoder:

- PointDynamicsNet  (P2 / P2-E): per-point MLP -> scatter into voxel cells
  (mean feature + log-count channel) -> conv encoder -> shared decoder.
  P2-E additionally receives free-space ray samples (type channel 0).
- PointmapDynamicsNet (P3): shared 2D CNN over per-view pointmaps (xyz+valid),
  per-view camera FiLM (T_base_cam, K), voxel-center re-projection with
  bilinear feature sampling -> 1x1 mix -> shared decoder.

No voxel-grid input is given to any of these models — the voxel cache is only
used for TARGETS (and by the P1-V baseline).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvwam.models_tiny import apply_film, film  # noqa: E402


def apply_film2d(x, film_out):
    scale, shift = film_out.chunk(2, dim=1)
    shape = (-1, x.shape[1], 1, 1)
    return x * scale.view(shape) + shift.view(shape)


class _DecoderMixin(nn.Module):
    """enc2/enc3 + FiLM decoder + decoupled head, mirrors TinyDynamicsNet."""

    def _build_decoder(self, base, cond_dim, n_offsets, c1):
        c2, c3 = base * 2, base * 4
        self.enc2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c3, c3, 3, padding=1), nn.SiLU())
        self.film3 = film(cond_dim, c3)
        self.film2 = film(cond_dim, c2)
        self.dec2 = nn.Sequential(nn.Conv3d(c3 + c2, c2, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.dec1 = nn.Sequential(nn.Conv3d(c2 + c1, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
        self.head = nn.Conv3d(c1, 2 * n_offsets, 1)   # occ_logit + vis_logit
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def decode(self, e1, cond):
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = apply_film(e3, self.film3(cond))
        d2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = apply_film(d2, self.film2(cond))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class PointDynamicsNet(_DecoderMixin, nn.Module):
    """P2 (pt_in=3) / P2-E (pt_in=4 with type flag)."""

    def __init__(self, base=32, cond_dim=128, n_offsets=4, pt_in=4, n_feat=16,
                 grid=(40, 65, 45)):
        super().__init__()
        self.n_offsets = n_offsets
        self.grid_shape = tuple(grid)
        self.cond_mlp = nn.Sequential(
            nn.Linear(14 + 16 * 14, 256), nn.SiLU(),
            nn.Linear(256, cond_dim), nn.SiLU())
        self.offset_emb = nn.Embedding(n_offsets, cond_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(pt_in, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
            nn.Linear(32, n_feat), nn.SiLU())
        self._build_decoder(base, cond_dim, n_offsets, c1=n_feat + 1)

    def forward(self, points, qpos, action, offset_idx, b_min, pitch):
        """points (B,N,pt_in) xyz(+type); b_min (B,3); pitch (B,)."""
        B, N, _ = points.shape
        Fc = self.point_mlp[-2].out_features
        cond = self.cond_mlp(torch.cat([qpos, action.flatten(1)], dim=1))
        cond = cond + self.offset_emb(offset_idx)
        feat = self.point_mlp(points)                      # (B,N,F)
        xyz = points[..., :3]
        if not torch.is_tensor(pitch) or pitch.dim() == 0:
            pitch = torch.as_tensor(pitch, device=xyz.device).expand(B)
        idx = ((xyz - b_min[:, None, :]) / pitch[:, None, None]).long()
        sh = self.grid_shape
        for d in range(3):
            idx[..., d] = idx[..., d].clamp(0, sh[d] - 1)
        inside = ((xyz >= b_min[:, None, :]) &
                  (xyz < b_min[:, None, :] + pitch[:, None, None] *
                   torch.tensor(sh, device=xyz.device, dtype=xyz.dtype)[None, :])).all(-1)
        flat = (idx[..., 0] * sh[1] + idx[..., 1]) * sh[2] + idx[..., 2]
        V = sh[0] * sh[1] * sh[2]
        vol = points.new_zeros(B, Fc + 1, V)
        ones = torch.ones(B, 1, N, device=points.device, dtype=points.dtype)
        for b in range(B):
            m = inside[b]
            if m.any():
                vol[b, :Fc].index_add_(1, flat[b][m],
                                       feat[b][m].transpose(0, 1))
                vol[b, Fc:].index_add_(1, flat[b][m],
                                       torch.ones(1, int(m.sum()), device=points.device,
                                                  dtype=points.dtype))
        cnt = vol[:, Fc:].clamp_min(1.0)
        e1 = torch.cat([vol[:, :Fc] / cnt, torch.log1p(vol[:, Fc:])], 1)
        e1 = e1.view(B, Fc + 1, *sh)
        return self.decode(e1, cond)


class PointmapDynamicsNet(_DecoderMixin, nn.Module):
    """P3: per-view 2D CNN + camera FiLM -> re-projected 3D volume."""

    def __init__(self, base=32, cond_dim=128, n_offsets=4, grid=(40, 65, 45),
                 pm_hw=(240, 320)):
        super().__init__()
        self.n_offsets = n_offsets
        self.grid_shape = tuple(grid)
        self.pm_hw = tuple(pm_hw)
        self.cond_mlp = nn.Sequential(
            nn.Linear(14 + 16 * 14, 256), nn.SiLU(),
            nn.Linear(256, cond_dim), nn.SiLU())
        self.offset_emb = nn.Embedding(n_offsets, cond_dim)
        # shared 2D CNN over (xyz, valid) pointmap
        self.img = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU())
        self.filmv = nn.Linear(16, 2 * 128)     # per-view cam FiLM (R|t|K4)
        self.proj2d = nn.Conv2d(128, 32, 1)     # cheap per-view reduction pre-projection
        self.mix3d = nn.Conv3d(3 * 32, 32, 1)
        self._vox_centers = None
        self._build_decoder(base, cond_dim, n_offsets, c1=32)

    def _voxel_centers(self, device):
        sh = self.grid_shape
        if self._vox_centers is None or self._vox_centers.device != device:
            i, j, k = torch.meshgrid(torch.arange(sh[0]), torch.arange(sh[1]),
                                     torch.arange(sh[2]), indexing="ij")
            c = torch.stack([i, j, k], -1).reshape(-1, 3).float() + 0.5
            self._vox_centers = c.to(device)            # (V,3) index space
        return self._vox_centers

    def forward(self, pointmap, camT, K4, qpos, action, offset_idx, b_min, pitch):
        """pointmap (B,3,4,H,W) xyz+valid in BASE frame; camT (B,3,3,4) cam->base;
        K4 (B,3,4) [fx,fy,cx,cy]; b_min (B,3); pitch (B,) or float."""
        B, V = pointmap.shape[0], pointmap.shape[1]
        cond = self.cond_mlp(torch.cat([qpos, action.flatten(1)], dim=1))
        cond = cond + self.offset_emb(offset_idx)
        if not torch.is_tensor(pitch) or pitch.dim() == 0:
            pitch = torch.as_tensor(pitch, device=pointmap.device).expand(B).float()
        f = self.img(pointmap.flatten(0, 1))           # (B*3,128,h,w)
        fv = self.filmv(torch.cat([camT.reshape(B, V, 12),
                                   K4], -1).flatten(0, 1))
        f = apply_film2d(f, fv).view(B, V, 128, *f.shape[-2:])
        centers = self._voxel_centers(pointmap.device)[None] * pitch[:, None, None] \
            + b_min[:, None, :]                        # (B,Vc,3) metric
        H, W = self.pm_hw
        sampled = []
        for vi in range(V):
            R = camT[:, vi, :, :3]                      # cam->base rotation
            t = camT[:, vi, :, 3]
            # base->cam, row-vector: p_cam = (p_base - t) @ R
            pc = (centers - t[:, None, :]) @ R
            z = pc[..., 2]
            u = K4[:, vi, 0][:, None] * pc[..., 0] / z.clamp_min(1e-6) + K4[:, vi, 2][:, None]
            v = K4[:, vi, 1][:, None] * pc[..., 1] / z.clamp_min(1e-6) + K4[:, vi, 3][:, None]
            valid = (z > 0.05) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            g = torch.stack([u / W * 2 - 1, v / H * 2 - 1], -1)[:, None]  # (B,1,Vc,2)
            s = F.grid_sample(self.proj2d(f[:, vi]), g, mode="bilinear",
                              align_corners=False).squeeze(2)   # (B,32,Vc)
            sampled.append(s * valid.float()[:, None])
        vol = torch.cat(sampled, 1).view(B, 3 * 32, *self.grid_shape)
        e1 = self.mix3d(vol)
        return self.decode(e1, cond)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
