"""Stage 4a: Observer-Factorized Joint WAM variants (A0/A1/A2/A3/A3-mask).

Design per Stage 4a spec (design-agent spec, 2026-11-20):
- shared enc1-3 (AE-initialized, trainable); latent = (128,10,17,12)
- geometry tokens G_tok = avg_pool2 over latent -> (270,128); resolution FIXED
  across variants so A0-vs-A2/A3 attributes geometry SOURCE, not token count
- FM recipe shared: t ~ 0.001+0.998*Beta(1.5,1) shared across streams;
  per-stream time embeddings; v-prediction, target eps - a_norm
- A0: action expert cross-attn <- current G_tok only (1.41M)
- A1: A0 + P1-V aux decoder WITH skips (decodes current-frame features; the
  one place skips are legitimate) (2.11M)
- A2: single joint stack over [G_tok 270 | w_tok 270 | act 16 | qpos 1];
  world FM target = E_frozen(x_{t+k}) (frozen AE encoder copy) (1.34M+0.86M)
- A3: two stacks + bidirectional cross-attn + HARD gates:
    action stream reads generated world tokens + qpos (NO current-geometry
    path — the attribution point); observer reads world-stream features +
    camera params only (no current-voxel skips). Observer output is
    AUXILIARY ONLY (never consumed by the action stream). (2.22M+0.86M)
- A3-mask: A3 + sigmoid gate on every cross-attn path + extra action<-current
  path with near-closed init (gamma=-4); learned gates are a reported finding.

Action dims: cache order [Larm6, Rarm6, Lgrip, Rgrip]; per-dim affine
normalization to [0,1] with zero-variance guard (constant dims -> 0 exactly).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvwam.models_tiny import apply_film, film  # noqa: E402


# ---------------------------------------------------------------- utilities

def sinusoidal_emb(t, dim):
    """t (B,) float -> (B, dim) sinusoidal time embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class AttnBlock(nn.Module):
    """Pre-LN transformer block: self-attn (+ optional cross-attn) + MLP."""

    def __init__(self, d=128, heads=4, mlp_ratio=4, cross=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.ln_cross = nn.LayerNorm(d)
            self.ln_kv = nn.LayerNorm(d)
            self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.SiLU(),
                                 nn.Linear(d * mlp_ratio, d))

    def forward(self, x, kv=None, gate=None):
        h = self.ln1(x)
        x = x + self.self_attn(h, h, h)[0]
        if self.cross and kv is not None:
            q = self.ln_cross(x)
            k = v = self.ln_kv(kv)
            attn_out = self.cross_attn(q, k, v)[0]
            if gate is not None:
                attn_out = torch.sigmoid(gate) * attn_out
            x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class ActionTokenizer(nn.Module):
    """(B,16,14) + qpos(B,14) -> tokens (B,17,128): act tokens + qpos token."""

    def __init__(self, d=128, horizon=16):
        super().__init__()
        self.proj = nn.Linear(14, d)
        self.pos = nn.Parameter(torch.zeros(horizon, d))
        self.qpos_proj = nn.Linear(14, d)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, act, qpos):
        tok = self.proj(act) + self.pos[None]
        qtok = self.qpos_proj(qpos)[:, None]
        return torch.cat([tok, qtok], 1)


class WorldTokenizer(nn.Module):
    """(B,C,10,17,12) -> tokens (B,270,d) via avg_pool2 + linear."""

    def __init__(self, in_ch, d=128):
        super().__init__()
        self.proj = nn.Linear(in_ch, d)

    def forward(self, z):
        pooled = F.avg_pool3d(z, 2, ceil_mode=True)        # (B,C,5,9,6)
        B, C = pooled.shape[:2]
        return self.proj(pooled.flatten(2).transpose(1, 2))  # (B,270,d)


def unpool_tokens(tok, out_shape=(10, 17, 12)):
    """(B,270,d) -> (B,d,10,17,12) — inverse of WorldTokenizer pooling."""
    B, N, d = tok.shape
    return tok.transpose(1, 2).reshape(B, d, 5, 9, 6).contiguous()


class FMTimeEmb(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        return self.mlp(sinusoidal_emb(t, 128))


class ModalityEmb(nn.Module):
    def __init__(self, n_streams, d=128):
        super().__init__()
        self.emb = nn.Parameter(torch.zeros(n_streams, d))
        nn.init.normal_(self.emb, std=0.02)

    def forward(self, i):
        return self.emb[i]


class VActionHead(nn.Module):
    def __init__(self, d=128, out=14):
        super().__init__()
        self.head = nn.Linear(d, out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tok):          # (B,16,d) -> (B,16,14)
        return self.head(tok)


class VWorldHead(nn.Module):
    """world tokens -> pooled future-occupancy velocity (B,2,10,17,12).

    v4 redesign: the world FM target is the POOLED FUTURE OCCUPANCY CANVAS
    (hit_frac, obs_frac at 10x17x12, values in [0,1]) — physically meaningful
    and bounded, replacing the AE-latent target whose raw scale (absmax 497,
    per-position zero-inflation) made any normalization pathological.
    """

    def __init__(self, d=128, out_ch=2):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.dw = nn.Conv3d(d, d, 3, padding=1, groups=d)
        self.head = nn.Conv3d(d, out_ch, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, tok, out_shape=(10, 17, 12)):
        h = self.lin(tok)
        v = unpool_tokens(h, out_shape)
        v = F.interpolate(v, size=out_shape, mode="trilinear", align_corners=False)
        v = F.silu(self.dw(v))
        return self.head(v)


# ---------------------------------------------------------------- variants

class _Encoder(nn.Module):
    """enc1-3 verbatim from TinyDynamicsNet (in_ch=4, base=32)."""

    def __init__(self):
        super().__init__()
        c1, c2, c3 = 32, 64, 128
        self.enc1 = nn.Sequential(nn.Conv3d(4, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.SiLU(),
                                  nn.Conv3d(c3, c3, 3, padding=1), nn.SiLU())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        return e1, e2, e3


class P1VAuxHead(nn.Module):
    """A1 auxiliary decoder WITH skips (decodes current-frame features).

    P1-V decoupled semantics: per offset channels [occ_logit, vis_logit]
    (models_tiny split_outputs convention). Same structure as
    TinyDynamicsNet's film3/dec2/film2/dec1.
    """

    def __init__(self, cond_dim=128, n_offsets=4):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Linear(14 + 16 * 14, 256), nn.SiLU(),
            nn.Linear(256, cond_dim), nn.SiLU())
        self.offset_emb = nn.Embedding(n_offsets, cond_dim)
        c1, c2, c3 = 32, 64, 128
        self.film3 = film(cond_dim, c3)
        self.film2 = film(cond_dim, c2)
        self.dec2 = nn.Sequential(nn.Conv3d(c3 + c2, c2, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c2, c2, 3, padding=1), nn.SiLU())
        self.dec1 = nn.Sequential(nn.Conv3d(c2 + c1, c1, 3, padding=1), nn.SiLU(),
                                  nn.Conv3d(c1, c1, 3, padding=1), nn.SiLU())
        self.head = nn.Conv3d(c1, 2 * n_offsets, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, e1, e2, e3, qpos, act, offset_idx):
        cond = self.cond_mlp(torch.cat([qpos, act.flatten(1)], dim=1))
        cond = cond + self.offset_emb(offset_idx)
        b = apply_film(e3, self.film3(cond))
        d2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = apply_film(d2, self.film2(cond))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class ObserverHead(nn.Module):
    """World-stream features + camera params -> visibility logits (aux only).

    Skip-free 2-stage upsample decoder to FULL voxel resolution (40,65,45);
    cam FiLM. NO current-voxel skip (else it could answer from the current
    observed mask — a strong persistence prior).
    """

    def __init__(self, d=128, cond_dim=72, n_offsets=4):
        super().__init__()
        self.conv_in = nn.Conv3d(d, 64, 3, padding=1)
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, 64), nn.SiLU(),
                                      nn.Linear(64, 64))
        self.offset_emb = nn.Embedding(n_offsets, 64)
        self.conv_mid = nn.Conv3d(64, 64, 3, padding=1)
        self.head = nn.Conv3d(64, n_offsets, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, world_feat, cam72, offset_idx, out_shape=(40, 65, 45),
                latent_shape=(10, 17, 12)):
        """world_feat: (B,270,d) world-stream hidden tokens."""
        h = unpool_tokens(world_feat)                       # (B,128,5,9,6)
        h = F.interpolate(h, size=latent_shape, mode="trilinear", align_corners=False)
        h = F.silu(self.conv_in(h))
        cond = self.cond_mlp(cam72) + self.offset_emb(offset_idx)
        h = h + cond.view(-1, 64, 1, 1, 1)
        h = F.interpolate(h, scale_factor=2, mode="trilinear", align_corners=False)
        h = F.silu(self.conv_mid(h))
        h = F.interpolate(h, size=out_shape, mode="trilinear", align_corners=False)
        return self.head(h)                                 # (B,n_off,40,65,45)


class Stage4AModel(nn.Module):
    def __init__(self, variant="a0", d=128, n_offsets=4, horizon=16,
                 bottleneck_ch=128):
        super().__init__()
        assert variant in ("a0", "a1", "a2", "a3", "a3mask")
        self.variant = variant
        self.n_offsets = n_offsets
        self.enc = _Encoder()
        if bottleneck_ch != 128:
            self.neck = nn.Conv3d(128, bottleneck_ch, 1)
        else:
            self.neck = None
        self.gtok = WorldTokenizer(bottleneck_ch, d)
        self.act_tok = ActionTokenizer(d, horizon)
        self.t_act = FMTimeEmb(d)
        self.mod = ModalityEmb(3, d)  # 0=current/world, 1=action, 2=qpos-fused
        self.v_a = VActionHead(d)

        if variant in ("a2",):
            # single joint stack over [G_tok | w_tok | act | qpos]
            self.w_tok = WorldTokenizer(2, d)
            self.t_world = FMTimeEmb(d)
            self.w_off = nn.Embedding(n_offsets, d)
            self.stack = nn.ModuleList([AttnBlock(d), AttnBlock(d)])
            self.v_w = VWorldHead(d, out_ch=2)
        elif variant in ("a3", "a3mask"):
            self.w_tok = WorldTokenizer(2, d)
            self.t_world = FMTimeEmb(d)
            self.w_off = nn.Embedding(n_offsets, d)
            self.world_stack = nn.ModuleList([AttnBlock(d, cross=True),
                                              AttnBlock(d, cross=True)])
            self.action_stack = nn.ModuleList([AttnBlock(d, cross=True),
                                               AttnBlock(d, cross=True)])
            self.v_w = VWorldHead(d, out_ch=2)
            self.observer = ObserverHead(d, cond_dim=72, n_offsets=n_offsets)
            if variant == "a3mask":
                # gates on every cross-attn path + extra action<-current path
                self.gates = nn.Parameter(torch.tensor(
                    [0.0, 0.0,   # world<-act (2 blocks)
                     0.0, 0.0,   # act<-world (2 blocks)
                     -4.0]))     # act<-current (near-closed init)
                self.current_cross = AttnBlock(d, cross=True)
        if variant in ("a0", "a1"):
            self.a0_blocks = nn.ModuleList([AttnBlock(d, cross=True),
                                            AttnBlock(d, cross=True)])
        if variant == "a1":
            self.aux = P1VAuxHead(n_offsets=n_offsets)

    def forward(self, x_t, noisy_act, t, qpos, offset_idx,
                noisy_world=None, x_current=None, cam72=None):
        """Returns dict:
        v_a (B,16,14); v_w (B,128,10,17,12) or None;
        aux_logits (B,3*n_off,40,65,45) or None (a1);
        obs_logits (B,n_off,40,65,45) or None (a3/a3mask);
        world_feat (B,270,d) or None (a3, for eval).
        """
        B = x_t.shape[0]
        e1, e2, e3 = self.enc(x_t)
        if self.neck is not None:
            e3 = self.neck(e3)
        G_tok = self.gtok(e3 if self.neck is not None else e3)  # (B,270,d)
        G_tok = G_tok + self.mod(0)
        act_tok = self.act_tok(noisy_act, qpos) + self.mod(1)
        t_emb_a = self.t_act(t)

        out = dict(v_w=None, aux_logits=None, obs_logits=None, world_feat=None)
        if self.variant == "a0":
            h = act_tok + t_emb_a[:, None]
            for blk in self.a0_blocks:
                h = blk(h, G_tok)
            out["v_a"] = self.v_a(h[:, :16])
            return out
        if self.variant == "a1":
            h = act_tok + t_emb_a[:, None]
            for blk in self.a0_blocks:
                h = blk(h, G_tok)
            out["v_a"] = self.v_a(h[:, :16])
            out["aux_logits"] = self.aux(e1, e2, e3, qpos, noisy_act, offset_idx)
            return out
        if self.variant == "a2":
            w_tok = self.w_tok(noisy_world) + self.t_world(t)[:, None] \
                + self.w_off(offset_idx)[:, None]
            tokens = torch.cat([G_tok, w_tok, act_tok], 1)      # (B,557,d)
            for blk in self.stack:
                tokens = blk(tokens)
            out["v_a"] = self.v_a(tokens[:, 540:556])
            out["v_w"] = self.v_w(tokens[:, 270:540])
            return out
        # a3 / a3mask
        w_tok = self.w_tok(noisy_world) + self.t_world(t)[:, None] \
            + self.w_off(offset_idx)[:, None]
        wf = w_tok
        gates = getattr(self, "gates", None)
        for bi, blk in enumerate(self.world_stack):
            g = gates[bi] if gates is not None else None      # world <- act
            wf = blk(wf, kv=act_tok, gate=g)
        act_h = act_tok
        for bi, blk in enumerate(self.action_stack):
            g = gates[2 + bi] if gates is not None else None  # act <- world
            act_h = blk(act_h, kv=wf, gate=g)
            if self.variant == "a3mask":
                act_h = self.current_cross(act_h, kv=G_tok, gate=self.gates[4])
        out["v_a"] = self.v_a(act_h[:, :16])
        out["v_w"] = self.v_w(wf)
        out["world_feat"] = wf
        if cam72 is not None:
            out["obs_logits"] = self.observer(wf, cam72, offset_idx)
        return out
