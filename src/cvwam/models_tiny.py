"""Stage 3a tiny dynamics predictors (plan.md §8.2, P0/P1 + P1-R/P1-V).

All models share a small 3D conv encoder over the current voxel and a
conditioning MLP over (qpos, action chunk). Differences:

- P1  (state_ce)   : predicts future 3-state logits (hit/free/unknown) per
                     offset head. The "uniform Hit/Free/Unknown" baseline.
- P1-R (residual)  : same output space, but the head predicts a residual
                     added to a logit prior built from the CURRENT frame
                     state (identity-friendly initialization).
- P1-V (decoupled) : two heads — occupancy probability p_occ (BCE on
                     future-OBSERVED voxels only: hit=1, free=0; unknown
                     future voxels excluded from the occupancy loss) and
                     future visibility m_vis (BCE everywhere: observed=1).
                     Encodes research decision 4: physical occupancy change
                     and visibility change are decoupled.
- ACT-ONLY control : P1 architecture with the voxel input zeroed — tests
                     whether performance comes from geometry or from a
                     single-task time-progress shortcut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def film(cond_dim, ch):
    return nn.Linear(cond_dim, ch * 2)


def apply_film(x, film_out):
    scale, shift = film_out.chunk(2, dim=1)
    shape = (-1, x.shape[1], 1, 1, 1)
    return x * scale.view(shape) + shift.view(shape)


class TinyDynamicsNet(nn.Module):
    def __init__(self, in_ch=4, base=32, cond_dim=128, n_offsets=4,
                 variant="state_ce", dropout=0.0):
        super().__init__()
        self.variant = variant
        self.n_offsets = n_offsets
        self.cond_mlp = nn.Sequential(
            nn.Linear(14 + 16 * 14, 256), nn.SiLU(),
            nn.Linear(256, cond_dim), nn.SiLU())
        self.offset_emb = nn.Embedding(n_offsets, cond_dim)
        c1, c2, c3 = base, base * 2, base * 4
        self.enc1 = nn.Sequential(nn.Conv3d(in_ch, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
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
        if variant == "state_ce":
            out_ch = 3 * n_offsets
        elif variant == "decoupled":
            out_ch = 2 * n_offsets   # occupancy logit + visibility logit
        else:
            out_ch = 3 * n_offsets   # residual: same output space
        self.head = nn.Conv3d(c1, out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)   # zero-init head: persistence start
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, voxel, qpos, action, offset_idx):
        """voxel (B,4,X,Y,Z); qpos (B,14); action (B,H,14); offset_idx (B,) int
        index into the model's offset list (or None for all heads)."""
        B = voxel.shape[0]
        cond = self.cond_mlp(torch.cat([qpos, action.flatten(1)], dim=1))
        cond = cond + self.offset_emb(offset_idx)
        e1 = self.dropout(self.enc1(voxel))
        e2 = self.dropout(self.enc2(e1))
        e3 = self.dropout(self.enc3(e2))
        b = apply_film(e3, self.film3(cond))
        d2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = apply_film(d2, self.film2(cond))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.head(d1)                       # (B, 3*n_offsets or 2*n_offsets, ...)
        if self.variant == "residual":
            # logit prior from the current frame's state one-hot, added to
            # every offset block
            prior = F.one_hot(offset_state_prior(voxel),
                              num_classes=3).permute(0, 4, 1, 2, 3).float()
            prior = (prior - 1.0 / 3) * 4.0       # one-hot -> confident logit
            out = out.view(B, self.n_offsets, 3, *out.shape[2:]) + prior.unsqueeze(1)
            out = out.reshape(B, self.n_offsets * 3, *out.shape[3:])
        return out


def offset_state_prior(voxel):
    """Current state code per voxel: 0 hit, 1 free, 2 unknown.
    voxel channels: [hit, free, observed, vc/3]."""
    hit = (voxel[:, 0] > 0.5).long()
    free = (voxel[:, 1] > 0.5).long()
    return 2 - 2 * hit - free        # hit->0, free->1, else->2


def split_outputs(out, variant, n_offsets):
    """-> list per offset of dicts with logits/prob tensors."""
    per = []
    C = out.shape[1] // n_offsets
    for k in range(n_offsets):
        block = out[:, k * C:(k + 1) * C]
        if variant == "decoupled":
            per.append({"occ_logit": block[:, 0], "vis_logit": block[:, 1]})
        else:
            per.append({"state_logits": block})
    return per


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
