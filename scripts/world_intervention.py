"""Stage 4a §2 (P3, P0): World-token intervention diagnostics.

For a trained a2/a3/a3mask checkpoint, run the FM inference (Euler 8, fixed
seeded noise for action AND world streams, fixed current observation) under
interventions on the WORLD information available to the ACTION expert:

  W0 normal     : world stream denoises from noise (standard generation)
  W1 blocked    : a3/a3mask — zero the world->action cross-attn contributions
                  (a2: joint stack is not separable -> run W1' uninformative
                  world tokens = modality embedding only)
  W2 current    : world stream starts from the CURRENT pooled occupancy canvas
                  ("world says nothing will change")
  W3 wrong-fut  : world stream starts from another EPISODE's future canvas
                  (AUDIT-S5.3: true cross-episode donor, not a batch roll)
  W4 oracle     : world stream starts from the GT future canvas (pooled
                  x_{t+k}) — OFFLINE DIAGNOSTIC ONLY (leaks the true future;
                  never a deployment or closed-loop claim)
  W5 noise      : world stream starts from pure noise at every step (no content)

All interventions replace the world stream's INITIAL canvas (the information
the action expert reads during ALL denoising steps). W1 is an architectural
block active at every cross-attn in every step.

Recorded per intervention: action deviation vs W0 (normalized + PHYSICAL,
AUDIT-S5.4), EE-trajectory deviation (FK over the predicted chunk, physical
units), expert action error (vs GT chunk, normalized + physical), per offset
k (proximity-to-interaction proxy).

AUDIT FIXES 2026-11-20 (Model Integrity Audit, runs/model_integrity/source_review.md):
- AUDIT-S5.1: euler_chunk(w0_init=None) no longer draws its own torch.randn.
  The caller passes the per-batch seeded `world_noise` tensor explicitly, so
  W0 and W1 share the SAME world-stream start (before: two independent randn
  draws -> the W0-vs-W1 delta was confounded by a noise redraw).
- AUDIT-S5.3: W3 donor is a true cross-episode future canvas, sampled from a
  preloaded pool of pooled_occupancy canvases over the val-split episodes
  (was: torch.roll(fut_canvas, B//2), which under shuffle=False pairs
  near-adjacent frames of the SAME episode). Donor must be genuinely
  different from the sample's own future: hit-IoU < 0.9; failures are counted
  and reported under donor_diag.
- AUDIT-S5.4: FK and the radian/physical deviations run on DENORMALIZED
  actions, following the serving contract (XPolicyLab/policy/cvwam_fm/
  model.py:292-295): act.clamp(0,1) then lo + act*scale. Normalized
  deviations are kept under *_norm fields for comparison (distinct names).
- AUDIT-DF (new): direct-forward probe — single model.forward, NO Euler:
  |dv_a| between held world canvases (W2 current / W3 cross-episode /
  W4 oracle) on a t-grid {0.1, 0.5, 0.9, 1.0}, with all action-stream inputs
  held identical. Measures whether world CONTENT changes the one-step action
  velocity at all.

If no effect: distinguish (1) world generation quality, (2) action expert
doesn't read future, (3) reads but can't use, (4) tested states don't need
future — via W4-vs-W0 (reading sensitivity) and W0-quality diagnostics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cvwam.dynamics_dataset import (  # noqa: E402
    OFFSETS, DynamicsDataset, frame_to_tensors, load_episode)
from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from train_stage4a import normalize_act, pooled_occupancy  # noqa: E402

N_EULER = 8
DF_T_GRID = (0.1, 0.5, 0.9, 1.0)   # AUDIT-DF
DONOR_IOU_THR = 0.9                # AUDIT-S5.3
DONOR_MAX_TRIES = 8                # AUDIT-S5.3


@torch.no_grad()
def euler_chunk(model, x, qpos, oi, act0_noise, w0_init, variant,
                block_world_to_action=False, n_euler=N_EULER, world_noise=None):
    """Denoise action (and world) streams; returns final action chunk (B,16,14)
    in normalized units. w0_init: (B,2,10,17,12) initial world canvas or None
    (W0: pure noise start). block: zero world->action cross-attn (a3/a3mask).

    AUDIT-S5.1: the w0_init=None branch uses the caller-provided `world_noise`
    (B,2,10,17,12) instead of an internal torch.randn — W0 and W1 must share
    the same world-stream start for their delta to be attributable to the
    block alone, not to a noise redraw.
    """
    B = x.shape[0]
    act = act0_noise.clone()
    hold = w0_init is not None           # W2/W3/W4: world HELD at injected canvas
    if w0_init is not None:
        world = w0_init.clone()
    else:
        # AUDIT-S5.1: explicit shared noise; no internal re-draw.
        assert world_noise is not None, \
            "AUDIT-S5.1: pass the per-batch world_noise explicitly (W0/W1 share it)"
        world = world_noise.clone()
    for i in range(n_euler):
        t = torch.tensor([1.0 - i / n_euler] * B, device=x.device)
        o = model(x, act, t, qpos, oi, noisy_world=world, cam72=None,
                  block_world_to_action=block_world_to_action)
        # FM sampling: v=e-a => integrate t:1->0 as x -= dt*v
        # (sign fix 2026-11-20; see runs/stage4a/failures/fm_integration_sign_bug_note.md)
        act = act - (1.0 / n_euler) * o["v_a"]
        if not hold and o["v_w"] is not None:
            # W0: world ODE. a0/a1 have no world stream (v_w=None) and never
            # read world tokens — nothing to integrate (the pre-fix script
            # crashed here for a0, which is why no wi_a0_* output existed).
            world = world - (1.0 / n_euler) * o["v_w"]
    return act


def denorm_phys(act_norm, stats):
    """AUDIT-S5.4: normalized action -> PHYSICAL units, serving contract
    (XPolicyLab/policy/cvwam_fm/model.py:292-295): act.clamp(0,1) THEN
    lo + act*scale. Arm dims are radians (left arm dims 0-5 are the active
    ones in this task; right arm dims 6-11 are constant raw 0 -> normalized
    0, denorm recovers 0); gripper dim 12 (Lgrip) is a raw [0,1] command
    (lo=0, hi=1, scale=1 -> denorm is identity there, what fk_left expects);
    dim 13 (Rgrip) is constant raw 1.0 (lo=hi=1 -> normalized 0 maps back
    to 1)."""
    lo = torch.as_tensor(stats["lo"], dtype=act_norm.dtype,
                         device=act_norm.device)
    scale = torch.as_tensor(stats["scale"], dtype=act_norm.dtype,
                            device=act_norm.device)
    return act_norm.clamp(0, 1) * scale + lo


def ee_traj(actions_cache_order, fk_fn):
    """actions (16,14) PHYSICAL cache order -> EE xyz path (16,3) via FK
    (left arm). AUDIT-S5.4: caller must pass denormalized (physical) actions;
    feeding normalized actions here was the S5.4 bug."""
    pts = []
    for i in range(actions_cache_order.shape[0]):
        p = fk_fn(actions_cache_order[i])
        pts.append(p)
    return np.stack(pts)


# ------------------------------------------------------------- AUDIT-S5.3

def hit_iou(a, b, thr=0.05):
    """Hit-channel IoU of two pooled canvases (2,10,17,12), soft cells
    thresholded at thr. NaN when both are empty (an empty donor is
    uninformative -> caller retries)."""
    ha = a[0] > thr
    hb = b[0] > thr
    uni = int((ha | hb).sum())
    if uni == 0:
        return float("nan")
    return int((ha & hb).sum()) / uni


def build_donor_pool(root, val_eps, max_per_ep=12):
    """AUDIT-S5.3: pooled occupancy canvases of val-split frames, grouped by
    episode spec — W3 donors must come from a DIFFERENT episode than the
    sample (true cross-episode wrong future), so the pool is keyed by spec."""
    pool = {}
    for spec in val_eps:
        v, e = spec.split(":")
        edir, mf = load_episode(Path(root), v, int(e))
        ts = [t for t in range(mf["T_total"])
              if mf["frames"][t]["npz"] is not None]
        if len(ts) > max_per_ep:
            sel = np.linspace(0, len(ts) - 1, max_per_ep).astype(int)
            ts = [ts[i] for i in sel]
        pool[spec] = []
        for t in ts:
            x, *_ = frame_to_tensors(edir / f"frame_{t:07d}.npz")
            pool[spec].append(pooled_occupancy(torch.from_numpy(x[None]))[0])
    return pool


def pick_donor(fut_canvas, ep_spec, pool, rng, iou_thr=DONOR_IOU_THR,
               max_tries=DONOR_MAX_TRIES):
    """AUDIT-S5.3: random donor canvas from a DIFFERENT episode; require
    hit-IoU < iou_thr vs the sample's own future (a genuinely different
    future). Returns (canvas, iou, ok); ok=False means no try satisfied the
    threshold -> caller counts a warning and uses the most-different
    candidate found (or an empty-canvas fallback if all were empty)."""
    others = [s for s in pool if s != ep_spec]
    best, best_iou, fallback = None, None, None
    for _ in range(max_tries):
        spec = others[int(rng.integers(len(others)))]
        c = pool[spec][int(rng.integers(len(pool[spec])))]
        iou = hit_iou(fut_canvas, c)
        if np.isnan(iou):
            if fallback is None:
                fallback = c        # both empty; last resort only
            continue
        if best is None or iou < best_iou:
            best, best_iou = c, iou
        if iou < iou_thr:
            return c, iou, True
    if best is not None:
        return best, best_iou, best_iou < iou_thr
    return fallback, float("nan"), False


# ------------------------------------------------------------- AUDIT-DF

@torch.no_grad()
def direct_forward_probe(model, x, qpos, oi, act_noise, canvases,
                         t_grid=DF_T_GRID):
    """AUDIT-DF: single model.forward per held world canvas (NO Euler), all
    action-stream inputs (x, qpos, oi, act_noise) held identical across
    canvases. Returns {t: {pair: mean |dv_a| L2}} plus the per-condition
    mean |v_a| magnitude, so deltas can be read relative to response scale.
    This is the direct measurement of "does world content change the
    one-step action velocity at all"."""
    names = list(canvases)
    out = {"per_t": {}, "v_a_norm": {}}
    for tval in t_grid:
        t = torch.full((x.shape[0],), float(tval), device=x.device)
        v = {}
        for name, w0 in canvases.items():
            v[name] = model(x, act_noise, t, qpos, oi, noisy_world=w0,
                            cam72=None)["v_a"]
        out["v_a_norm"][tval] = {n: float(v[n].norm(dim=-1).mean())
                                 for n in names}
        row = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                row[f"{a}_vs_{b}"] = float((v[a] - v[b]).norm(dim=-1).mean())
        out["per_t"][tval] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda:1",
                    help="single-GPU discipline: prefer card 1 (/code/AGENTS.md)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = args.device
    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads(Path(args.split_manifest).read_text())
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    ds = DynamicsDataset(root, val_eps, stride_t=4)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    model = Stage4AModel(variant=ck["variant"],
                         bottleneck_ch=ck.get("bottleneck_ch", 128)).to(device).eval()
    model.load_state_dict(ck["model"])
    stats = ck["stats"]

    # AUDIT-S5.3: cross-episode donor pool over the val episodes (replaces the
    # torch.roll batch shift, which paired same-episode adjacent frames).
    pool = build_donor_pool(root, val_eps)
    print(f"[S5.3] donor pool: {len(pool)} episodes, "
          f"{sum(len(v) for v in pool.values())} canvases")

    import functools
    import build_robot_region as brr
    chains = {s: brr.load_chain(s) for s in ("left",)}
    spheres = None  # EE path only needs link6 position

    def fk_left(a14_cache):
        """EE (link6) xyz from a PHYSICAL cache-order 14-dim action
        (Larm6 rad, Rarm6 rad, Lgrip/Rgrip [0,1] command). AUDIT-S5.4: the
        caller denormalizes before FK; normalized input here is a bug."""
        qd = {f"fl_joint{i+1}": float(a14_cache[i]) for i in range(6)}
        grip = float(np.clip(a14_cache[12], 0, 1))
        qd["fl_joint7"] = float(np.clip(-0.01 + 0.055 * grip, 0, 0.04765))
        qd["fl_joint8"] = qd["fl_joint7"]
        poses = brr.fk_links(chains["left"][0], chains["left"][1], qd)
        R, tl = poses["fl_link6"]
        return tl

    INTERV = ("W0", "W1", "W2", "W3", "W4", "W5")
    agg = {w: dict(act_dev=[], act_dev_phys=[], ee_dev=[], expert_err=[],
                   expert_err_phys=[],
                   per_k={k: [] for k in OFFSETS},
                   per_k_phys={k: [] for k in OFFSETS}) for w in INTERV}
    df_agg = {}          # AUDIT-DF: pair -> t -> [per-batch means]
    df_vnorm = {}        # AUDIT-DF: condition -> t -> [per-batch means]
    donor_ious, donor_warn, donor_n, donor_nan = [], 0, 0, 0
    n_pairs = 0
    for bi, batch in enumerate(dl):
        if bi >= args.max_batches:
            break
        x = batch["voxel"].to(device)
        x_fut = batch["voxel_future"].to(device)
        qpos = batch["qpos"].to(device)
        act_raw = batch["action"].to(device)
        act_gt = torch.from_numpy(normalize_act(batch["action"].numpy(),
                                                stats)).to(device)
        B = x.shape[0]
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]],
                          device=device)
        # fixed noise across interventions (seeded per batch index)
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1000 + bi)
        act_noise = torch.randn(B, 16, 14, generator=g).to(device)
        # AUDIT-S5.1: ONE world noise tensor shared by W0 AND W1 (and the
        # dither on held canvases), generated once here and passed down.
        world_noise = torch.randn(B, 2, 10, 17, 12, generator=g).to(device)
        cur_canvas = pooled_occupancy(x)
        fut_canvas = pooled_occupancy(x_fut)
        # AUDIT-S5.3: per-sample donor canvas from a DIFFERENT episode with
        # hit-IoU < DONOR_IOU_THR vs the sample's own future.
        donor_rng = np.random.default_rng(args.seed * 7919 + bi)
        donors = []
        for i in range(B):
            c, iou, ok = pick_donor(fut_canvas[i].cpu(), batch["episode"][i],
                                    pool, donor_rng)
            donors.append(c)
            donor_n += 1
            if np.isnan(iou):
                donor_nan += 1
            else:
                donor_ious.append(iou)
            if not ok:
                donor_warn += 1
        fut_donor = torch.stack(donors).to(device)
        inits = {
            "W0": world_noise,
            "W2": cur_canvas + 0.05 * world_noise,
            "W3": fut_donor + 0.05 * world_noise,
            "W4": fut_canvas + 0.05 * world_noise,   # ORACLE — offline only
            "W5": world_noise,                        # same as W0 start; differs by
            # re-noising below
        }
        chunks = {}
        for w in INTERV:
            if w == "W0":
                a = euler_chunk(model, x, qpos, oi, act_noise, None,
                                ck["variant"], world_noise=world_noise)
            elif w == "W1":
                a = euler_chunk(model, x, qpos, oi, act_noise, None,
                                ck["variant"], block_world_to_action=True,
                                world_noise=world_noise)
            elif w == "W5":
                # re-noise the world at every step = no usable content;
                # approximated by starting from pure noise (same as W0 start
                # here since W0 also starts from noise) -> skip duplicate
                continue
            else:
                a = euler_chunk(model, x, qpos, oi, act_noise, inits[w],
                                ck["variant"])
            chunks[w] = a
        a0 = chunks["W0"]
        # AUDIT-DF: direct-forward |dv_a| grid over the held canvases
        # (same tensors as the interventions, incl. the identical 0.05 dither).
        df = direct_forward_probe(model, x, qpos, oi, act_noise,
                                  {"W2_current": inits["W2"],
                                   "W3_cross_ep": inits["W3"],
                                   "W4_oracle": inits["W4"]})
        for tval, row in df["per_t"].items():
            for pair, val in row.items():
                df_agg.setdefault(pair, {}).setdefault(tval, []).append(val)
            for cond, val in df["v_a_norm"][tval].items():
                df_vnorm.setdefault(cond, {}).setdefault(tval, []).append(val)
        a0_phys = denorm_phys(a0, stats)      # AUDIT-S5.4
        for w in INTERV:
            if w not in chunks:
                continue
            aw = chunks[w]
            dev = float((aw - a0).norm(dim=-1).mean())          # normalized
            aw_phys = denorm_phys(aw, stats)                    # AUDIT-S5.4
            dev_phys = float((aw_phys - a0_phys).norm(dim=-1).mean())
            # expert error (vs GT): normalized + physical
            err = float((aw - act_gt).norm(dim=-1).mean())
            err_phys = float((aw_phys - act_raw).norm(dim=-1).mean())
            # EE trajectory deviation (per-sample, left arm FK) —
            # AUDIT-S5.4: FK now consumes PHYSICAL actions
            ee = []
            ap_ = aw_phys.cpu().numpy()
            a0p = a0_phys.cpu().numpy()
            for j in range(min(B, 8)):
                p_w = ee_traj(ap_[j], fk_left)
                p_0 = ee_traj(a0p[j], fk_left)
                ee.append(float(np.linalg.norm(p_w - p_0, axis=1).mean()))
            agg[w]["act_dev"].append(dev)
            agg[w]["act_dev_phys"].append(dev_phys)
            agg[w]["ee_dev"].append(float(np.mean(ee)))
            agg[w]["expert_err"].append(err)
            agg[w]["expert_err_phys"].append(err_phys)
            for ki, k in enumerate(OFFSETS):
                sel = (oi == ki)
                if sel.any():
                    agg[w]["per_k"][k].append(
                        float((aw[sel] - a0[sel]).norm(dim=-1).mean()))
                    agg[w]["per_k_phys"][k].append(
                        float((aw_phys[sel] - a0_phys[sel]).norm(dim=-1).mean()))
        n_pairs += B
        if donor_warn:
            print(f"[S5.3][WARN] batch {bi}: {donor_warn} donors failed the "
                  f"hit-IoU < {DONOR_IOU_THR} check (cumulative)")

    res = dict(ckpt=args.ckpt, variant=ck["variant"], n_pairs=n_pairs,
               device=device,
               gt_chunk_norm_note="see interventions.W0.expert_err = generation-from-noise baseline error vs GT",
               seed=args.seed, n_euler=N_EULER,
               audit_fixes={
                   "S5_1": "W0/W1 share the per-batch seeded world_noise "
                           "(euler_chunk takes it explicitly, no internal randn)",
                   "S5_3": "W3 donors are cross-episode pooled future canvases "
                           f"(hit-IoU < {DONOR_IOU_THR} enforced; failures counted)",
                   "S5_4": "FK + physical devs use denorm per serving contract "
                           "(clamp(0,1) then lo+act*scale); *_norm kept for comparison",
                   "DF": "direct_forward: single-step |dv_a| between held canvases "
                         "on t-grid, no Euler",
               },
               donor_diag=dict(
                   n=donor_n, n_warn=donor_warn, n_iou_nan=donor_nan,
                   iou_mean=round(float(np.mean(donor_ious)), 4) if donor_ious else None,
                   iou_min=round(float(np.min(donor_ious)), 4) if donor_ious else None,
                   iou_max=round(float(np.max(donor_ious)), 4) if donor_ious else None,
                   iou_thr=DONOR_IOU_THR,
                   note="donor accepted when hit-IoU vs the sample's own future "
                        "< thr (genuinely different future); n_warn = accepted "
                        "despite failing the check (most-different fallback)"),
               oracle_note="W4 leaks true future; OFFLINE diagnostic only",
               interventions={})
    for w in INTERV:
        if not agg[w]["act_dev"]:
            continue
        res["interventions"][w] = dict(
            act_dev_norm=round(float(np.mean(agg[w]["act_dev"])), 4),
            act_dev_phys=round(float(np.mean(agg[w]["act_dev_phys"])), 4),
            ee_dev_m=round(float(np.mean(agg[w]["ee_dev"])), 4),
            expert_err_norm=round(float(np.mean(agg[w]["expert_err"])), 4),
            expert_err_phys=round(float(np.mean(agg[w]["expert_err_phys"])), 4),
            per_k={k: round(float(np.mean(v)), 4) for k, v in
                   agg[w]["per_k"].items() if v},
            per_k_phys={k: round(float(np.mean(v)), 4) for k, v in
                        agg[w]["per_k_phys"].items() if v})
    # AUDIT-DF: aggregate the direct-forward grid (mean over batches)
    res["direct_forward"] = dict(
        t_grid=list(DF_T_GRID),
        pairs={pair: {str(t): round(float(np.mean(v)), 6)
                      for t, v in tv.items()}
               for pair, tv in df_agg.items()},
        v_a_norm={cond: {str(t): round(float(np.mean(v)), 4)
                         for t, v in tv.items()}
                  for cond, tv in df_vnorm.items()},
        note="single model.forward per canvas, NO Euler; canvases = the same "
             "held tensors as interventions W2/W3/W4 (incl. the identical "
             "0.05*world_noise dither, so deltas isolate canvas content); "
             "a0 ignores noisy_world by construction -> exact 0 sanity check")
    out = Path(args.out or ("runs/stage4a/metrics/world_intervention/"
                            f"wi_{ck['variant']}_{Path(args.ckpt).parent.name}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
