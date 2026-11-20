"""Synthetic unit tests for the masked dynamic-IoU metric (Model Integrity Audit S7).

S7 bug (docs-local/review/review.md #9, confirmed in runs/model_integrity/source_review.md):
the intersection term of the masked dyn-IoU was

    inter = (pred_hit & dyn)                      # BUGGY

but inside ``dyn`` a cell with ``gt_hit == True`` is a NEWLY OCCUPIED cell and a
cell with ``gt_hit == False`` is a FREED cell; a prediction on a freed cell must
count against the metric (union only), never as intersection. Correct:

    inter = (pred_hit & gt_hit & dyn)             # FIXED

Union stays ``(pred_hit & dyn) | (gt_hit & dyn)``; per-batch IoU = inter/uni with
uni == 0 -> NaN (excluded from the mean).

This file contains:
  * verbatim pre-fix copies (_buggy_*) of the three implementations, for a
    permanent regression anchor of the erroneous values. Provenance: captured
    from the working tree on 2026-11-20 before the fix; ``git diff`` verified
    the metric bodies were identical to git HEAD (train_stage4a.py blob 535ba88)
    and scripts/eval_stage4a.py was unmodified.
    - _buggy_soft_masked_dyn_iou  <- scripts/train_stage4a.py::soft_masked_dyn_iou
    - _buggy_masked_dyn_iou       <- scripts/train_stage4a.py::masked_dyn_iou
    - _buggy_masked_dyn_iou_aux   <- scripts/eval_stage4a.py::masked_dyn_iou_aux
  * live imports of the same functions (post-fix) that are asserted against
    hand-derived correct values.

Hand derivation of every expected value (see _CASES comments for the layout):
grid (B=1, H=2, W=3, D=3) = 18 cells; flat idx = h*9 + w*3 + d.

  idx 0,1,2        NEW         st=1 (free@t)  fut=0 (hit)  obs_f=1  -> dyn (new occupation)
  idx 3,4          FREED       st=0 (hit@t)   fut=1 (free) obs_f=1  -> dyn (freed)
  idx 5            STATIC_OCC  st=0           fut=0        obs_f=1  -> not dyn (no change)
  idx 9,10         STATIC_FREE st=1           fut=1        obs_f=1  -> not dyn (no change)
  idx 11           UNOBS_FUT   st=0           fut=0        obs_f=0  -> not dyn (not observed at future)
  idx 6,7,8,12..17 UNKNOWN_T   st=2           fut=1        obs_f=1  -> not dyn (st == 2)

dyn = ((st == 0) ^ gt_hit) & obs_f & (st != 2):
  |dyn| = 5 (3 new + 2 freed), |gt_hit & dyn| = |new| = 3, |freed| = 2.

For a prediction P (restricted to the scored set dyn):
  inter_buggy = |P & dyn|
  inter_fixed = |P & gt_hit & dyn|
  uni         = |(P & dyn) | (gt_hit & dyn)|

  1 PREDICTED_PERFECT  P = gt_hit = {NEW, STATIC_OCC, UNOBS_FUT}
      P&dyn = NEW(3):  inter 3, uni 3  -> buggy 1.0, fixed 1.0
      (a perfect prediction is insensitive to the bug)
  2 ALL_HIT            P = all 18 cells
      P&dyn = dyn(5):  buggy inter 5, uni 5 -> 1.0   <-- degenerate perfect score
                       fixed inter 3, uni 5 -> 0.6
  3 ALL_FREE           P = {}
      P&dyn = {}:      inter 0, uni 3     -> 0.0 both (empty pred not inflated)
  4 PERSISTENCE        P = {cells occupied at t} = {FREED, STATIC_OCC, UNOBS_FUT}
      P&dyn = FREED(2): buggy inter 2, uni 2|3 = 5 -> 0.4  <-- stale-occupancy
                        fixed inter 0 (FREED has gt_hit False), uni 5 -> 0.0
                        baseline rewarded by the bug
  5 INVERTED           P = ~gt_hit = {FREED, STATIC_FREE, UNKNOWN_T}
      P&dyn = FREED(2): buggy 2/5 = 0.4, fixed 0/5 = 0.0
      (on the dyn-masked scored set, persistence and anti-oracle coincide --
      both hit exactly the freed cells; the metric cannot distinguish them.
      Two independent probes that the fixed metric gives zero credit to
      stale/anti-physical hits.)
  6 PERFECT_SOFT_BOUNDARY (soft only, thr=0.5, strict >): NEW soft = [0.9, 0.9, 0.5]
      -> third NEW cell is NOT a hit (0.5 > 0.5 is False):
      P&dyn = {NEW0, NEW1}(2): inter 2, uni 2|3 = 3 -> 2/3 both versions
  7 EMPTY_DYN edge: same grid but NEW -> static occ, FREED -> static free
      => |dyn| = 0: uni = 0 -> per-batch IoU NaN -> excluded from mean;
      single batch returns NaN, [EmptyDyn, Perfect] mixed returns 1.0
      (NaN masking identical in both versions).

Run: /home/zhangjialiang/miniconda3/envs/openwam/bin/python tests/test_dynamic_metrics.py
(also pytest-compatible). Pre-fix run records the live==buggy values into
runs/model_integrity/test_dynamic_metrics_prefix.log; post-fix run must PASS.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_stage4a import masked_dyn_iou, soft_masked_dyn_iou  # noqa: E402
from eval_stage4a import masked_dyn_iou_aux  # noqa: E402


# ---------------------------------------------------------------------------
# Verbatim pre-fix copies (AUDIT-S7 buggy formula) -- provenance in module doc.
# Do not "fix" these; they anchor the erroneous values.
# ---------------------------------------------------------------------------
def _buggy_soft_masked_dyn_iou(vox_soft, batch, thr=0.5):
    """vox_soft: (B,2,40,65,45) [hit,obs] soft maps in [0,1]."""
    obs_f = batch["observed_future"].to(vox_soft.device).bool()
    gt_hit = batch["state_future"].to(vox_soft.device) == 0
    st = batch["state_t"].to(vox_soft.device)
    pred_hit = vox_soft[:, 0] > thr
    both = obs_f & (st != 2)
    dyn = ((st == 0) ^ gt_hit) & both
    inter = (pred_hit & dyn).sum((1, 2, 3)).float()
    uni = ((pred_hit & dyn) | (gt_hit & dyn)).sum((1, 2, 3)).float()
    iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
    ok = ~torch.isnan(iou)
    return float(iou[ok].mean()) if ok.any() else float("nan")


@torch.no_grad()
def _buggy_masked_dyn_iou(world_pred_voxel, batch, thr=0.5):
    """world_pred_voxel: (B,4,40,65,45) logits-ish [hit,free,obs,vc]."""
    obs_f = batch["observed_future"].to(world_pred_voxel.device).bool()
    gt_hit = batch["state_future"].to(world_pred_voxel.device) == 0
    st = batch["state_t"].to(world_pred_voxel.device)
    pred_hit = world_pred_voxel[:, 0] > thr
    both = obs_f & (st != 2)
    dyn = ((st == 0) ^ gt_hit) & both
    inter = (pred_hit & dyn).sum((1, 2, 3)).float()
    uni = ((pred_hit & dyn) | (gt_hit & dyn)).sum((1, 2, 3)).float()
    iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
    ok = ~torch.isnan(iou)
    return float(iou[ok].mean()) if ok.any() else float("nan")


def _buggy_masked_dyn_iou_aux(pred_hit, batch):
    device = pred_hit.device
    obs_f = batch["observed_future"].to(device).bool()
    gt_hit = batch["state_future"].to(device) == 0
    st = batch["state_t"].to(device)
    both = obs_f & (st != 2)
    dyn = ((st == 0) ^ gt_hit) & both
    inter = (pred_hit & dyn).sum((1, 2, 3)).float()
    uni = ((pred_hit & dyn) | (gt_hit & dyn)).sum((1, 2, 3)).float()
    iou = torch.where(uni > 0, inter / uni.clamp_min(1), float("nan"))
    ok = ~torch.isnan(iou)
    return float(iou[ok].mean()) if ok.any() else float("nan")


# ---------------------------------------------------------------------------
# Synthetic grid construction
# ---------------------------------------------------------------------------
H, W, D = 2, 3, 3


def _idx(h, w, d):
    return h * W * D + w * D + d


def _flat(cells):
    m = np.zeros((H, W, D), dtype=bool)
    for (h, w, d) in cells:
        m[h, w, d] = True
    return m


# cell groups, see module docstring for semantics
NEW = [(0, 0, 0), (0, 1, 0), (0, 2, 0)]              # idx 0,1,2   (3 cells)
FREED = [(0, 0, 1), (0, 1, 1)]                       # idx 3,4     (2 cells)
STATIC_OCC = [(0, 2, 1)]                             # idx 5       (1 cell)
STATIC_FREE = [(1, 0, 0), (1, 1, 0)]                 # idx 9,10    (2 cells)
UNOBS_FUT = [(1, 2, 0)]                              # idx 11      (1 cell)

ALL_CELLS = [(h, w, d) for h in range(H) for w in range(W) for d in range(D)]
OTHER = [c for c in ALL_CELLS
         if c not in NEW + FREED + STATIC_OCC + STATIC_FREE + UNOBS_FUT]


def make_batch(empty_dyn=False):
    """Build a fake DynamicsDataset-style batch dict on CPU (B=1)."""
    st = np.full((H, W, D), 2, dtype=np.int64)           # default: unknown at t
    fut = np.full((H, W, D), 1, dtype=np.int64)          # default: free in future
    obs = np.zeros((H, W, D), dtype=np.float32)
    for c in NEW:
        st[c] = 1
        fut[c] = 0
        obs[c] = 1.0
    for c in FREED:
        st[c] = 0
        fut[c] = 1
        obs[c] = 1.0
    for c in STATIC_OCC:
        st[c] = 0
        fut[c] = 0
        obs[c] = 1.0
    for c in STATIC_FREE:
        st[c] = 1
        fut[c] = 1
        obs[c] = 1.0
    for c in UNOBS_FUT:
        st[c] = 0
        fut[c] = 0
        obs[c] = 0.0
    for c in OTHER:                                      # UNKNOWN_T
        st[c] = 2
        fut[c] = 1
        obs[c] = 1.0
    if empty_dyn:
        # turn every dynamic cell static: NEW -> occupied both frames,
        # FREED -> free both frames => |dyn| = 0
        for c in NEW:
            st[c] = 0
        for c in FREED:
            fut[c] = 0
    return {
        "state_t": torch.from_numpy(st)[None],           # (1,H,W,D)
        "state_future": torch.from_numpy(fut)[None],     # (1,H,W,D)
        "observed_future": torch.from_numpy(obs)[None],  # (1,H,W,D)
    }


def _pred_pattern(name, empty_dyn=False):
    """Bool prediction grid per case (hand-derived in module docstring)."""
    p = np.zeros((H, W, D), dtype=bool)
    if name == "PREDICTED_PERFECT":
        for c in NEW + STATIC_OCC + UNOBS_FUT:
            p[c] = True
    elif name == "ALL_HIT":
        for c in ALL_CELLS:
            p[c] = True
    elif name == "ALL_FREE":
        pass
    elif name == "PERSISTENCE":
        for c in FREED + STATIC_OCC + UNOBS_FUT:
            p[c] = True
    elif name == "INVERTED":
        gt = np.zeros((H, W, D), dtype=bool)
        for c in NEW + STATIC_OCC + UNOBS_FUT:
            gt[c] = True
        p = ~gt
    else:
        raise ValueError(name)
    return p


def _soft_from_pattern(p, boundary_idx=None):
    """(1,2,H,W,D) soft canvas: 0.9 hit / 0.1 miss; optional 0.5 boundary cell."""
    a = np.where(p, 0.9, 0.1).astype(np.float32)
    if boundary_idx is not None:
        a[boundary_idx] = 0.5
    return torch.from_numpy(a)[None, None]           # (1,1,H,W,D) -> ch0


def _logits_from_pattern(p):
    """(1,4,H,W,D) hit logits: +10 hit / -10 miss (thr 0.5)."""
    a = np.where(p, 10.0, -10.0).astype(np.float32)
    x = np.zeros((1, 4, H, W, D), dtype=np.float32)
    x[0, 0] = a
    return torch.from_numpy(x)


CASES = ["PREDICTED_PERFECT", "ALL_HIT", "ALL_FREE", "PERSISTENCE", "INVERTED"]

# hand-derived anchors (module docstring derivations)
BUGGY_ANCHOR = {"PREDICTED_PERFECT": 1.0, "ALL_HIT": 1.0, "ALL_FREE": 0.0,
                "PERSISTENCE": 0.4, "INVERTED": 0.4}
FIXED_ANCHOR = {"PREDICTED_PERFECT": 1.0, "ALL_HIT": 0.6, "ALL_FREE": 0.0,
                "PERSISTENCE": 0.0, "INVERTED": 0.0}
BOUNDARY_VAL = 2.0 / 3.0


def _close(a, b):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    # 1e-6: the metric accumulates in float32 (`.float()` on int counts);
    # 2/3 arrives as 0.66666668..., so exact-rational 1e-9 is too tight.
    return abs(a - b) < 1e-6


def run_all():
    """Return {case: {impl: (buggy_val, live_val)}} for all cases + edge cases."""
    out = {}
    for name in CASES:
        batch = make_batch()
        pred = _pred_pattern(name)
        soft = torch.cat([_soft_from_pattern(pred),
                          torch.zeros(1, 1, H, W, D)], 1)          # (1,2,H,W,D)
        logits = _logits_from_pattern(pred)
        pred_hit = torch.from_numpy(pred)[None]                    # (1,H,W,D) bool
        out[name] = {
            "soft": (_buggy_soft_masked_dyn_iou(soft, batch),
                     soft_masked_dyn_iou(soft, batch)),
            "masked4ch": (_buggy_masked_dyn_iou(logits, batch),
                          masked_dyn_iou(logits, batch)),
            "aux": (_buggy_masked_dyn_iou_aux(pred_hit, batch),
                    masked_dyn_iou_aux(pred_hit, batch)),
        }
    # case 6: soft threshold boundary (third NEW cell at exactly thr)
    batch = make_batch()
    p = _pred_pattern("PREDICTED_PERFECT")
    p[NEW[2]] = False                                              # 0.5 is NOT a hit
    soft = torch.cat([_soft_from_pattern(
        _pred_pattern("PREDICTED_PERFECT"), boundary_idx=NEW[2]),
        torch.zeros(1, 1, H, W, D)], 1)
    logits = _logits_from_pattern(p)
    pred_hit = torch.from_numpy(p)[None]
    out["PERFECT_SOFT_BOUNDARY"] = {
        "soft": (_buggy_soft_masked_dyn_iou(soft, batch),
                 soft_masked_dyn_iou(soft, batch)),
        "masked4ch": (_buggy_masked_dyn_iou(logits, batch),
                      masked_dyn_iou(logits, batch)),
        "aux": (_buggy_masked_dyn_iou_aux(pred_hit, batch),
                masked_dyn_iou_aux(pred_hit, batch)),
    }
    # case 7: empty-dyn NaN handling
    b_empty = make_batch(empty_dyn=True)
    b_perf = make_batch()
    p_all = _pred_pattern("ALL_HIT")
    p_perf = _pred_pattern("PREDICTED_PERFECT")
    # single empty-dyn batch
    soft_e = torch.cat([_soft_from_pattern(p_all), torch.zeros(1, 1, H, W, D)], 1)
    out["EMPTY_DYN_single"] = {
        "soft": (_buggy_soft_masked_dyn_iou(soft_e, b_empty),
                 soft_masked_dyn_iou(soft_e, b_empty)),
    }
    # mixed batch [empty-dyn, perfect]: NaN excluded from mean -> 1.0
    b_mix = {"state_t": torch.cat([b_empty["state_t"], b_perf["state_t"]], 0),
             "state_future": torch.cat([b_empty["state_future"],
                                        b_perf["state_future"]], 0),
             "observed_future": torch.cat([b_empty["observed_future"],
                                           b_perf["observed_future"]], 0)}
    pred2 = torch.stack([torch.from_numpy(p_all), torch.from_numpy(p_perf)], 0)
    soft2 = torch.stack([_soft_from_pattern(p_all)[0],
                         _soft_from_pattern(p_perf)[0]], 0)
    soft2 = torch.cat([soft2, torch.zeros(2, 1, H, W, D)], 1)
    out["EMPTY_DYN_mixed"] = {
        "soft": (_buggy_soft_masked_dyn_iou(soft2, b_mix),
                 soft_masked_dyn_iou(soft2, b_mix)),
        "aux": (_buggy_masked_dyn_iou_aux(pred2, b_mix),
                masked_dyn_iou_aux(pred2, b_mix)),
    }
    return out


def _fmt(v):
    return "NaN" if (isinstance(v, float) and math.isnan(v)) else f"{v:.6g}"


def main():
    res = run_all()
    print(f"{'case':22s} {'impl':9s} {'buggy-copy':>10s} {'live':>10s} "
          f"{'buggy-anchor':>12s} {'fixed-anchor':>12s} verdict(live)")
    all_ok = True
    for case, impls in res.items():
        for impl, (buggy_v, live_v) in impls.items():
            if case == "EMPTY_DYN_single":
                exp = float("nan")
            elif case == "EMPTY_DYN_mixed":
                exp = 1.0                      # NaN batch excluded from mean
            elif case == "PERFECT_SOFT_BOUNDARY":
                exp = BOUNDARY_VAL
            else:
                exp = FIXED_ANCHOR[case]
            ok = _close(live_v, exp)
            all_ok &= ok
            print(f"{case:22s} {impl:9s} {_fmt(buggy_v):>10s} {_fmt(live_v):>10s} "
                  f"{'-':>12s} {_fmt(exp):>12s} {'OK' if ok else 'FAIL'}")
    print("\nverdict:",
          "live implementations match FIXED hand-derived anchors"
          if all_ok else
          "live implementations do NOT match fixed anchors (pre-fix buggy state "
          "expected for the first run; record values, then apply AUDIT-S7 fix)")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
def test_buggy_formula_anchors():
    """The verbatim pre-fix copies must still reproduce the hand-derived
    BUGGY values (permanent anchor of what the bug did)."""
    res = run_all()
    for case in CASES:
        for impl, (buggy_v, _) in res[case].items():
            assert _close(buggy_v, BUGGY_ANCHOR[case]), \
                f"{case}/{impl}: buggy copy {buggy_v} != anchor {BUGGY_ANCHOR[case]}"


def test_fixed_formula_cases():
    """Live implementations must match the hand-derived FIXED values."""
    res = run_all()
    for case in CASES:
        for impl, (_, live_v) in res[case].items():
            assert _close(live_v, FIXED_ANCHOR[case]), \
                f"{case}/{impl}: live {live_v} != fixed anchor {FIXED_ANCHOR[case]}"
    for impl, (_, live_v) in res["PERFECT_SOFT_BOUNDARY"].items():
        assert _close(live_v, BOUNDARY_VAL), \
            f"boundary/{impl}: live {live_v} != 2/3"
    assert math.isnan(res["EMPTY_DYN_single"]["soft"][1]), \
        "empty-dyn single batch must return NaN"
    for impl in ("soft", "aux"):
        assert _close(res["EMPTY_DYN_mixed"][impl][1], 1.0), \
            f"empty-dyn mixed/{impl}: NaN must be excluded from mean (want 1.0)"


def test_impl_consistency():
    """All three implementations must agree on the same prediction pattern."""
    res = run_all()
    for case in CASES + ["PERFECT_SOFT_BOUNDARY"]:
        vals = [v[1] for v in res[case].values()]
        assert all(_close(v, vals[0]) for v in vals), \
            f"{case}: implementations disagree: {vals}"


if __name__ == "__main__":
    sys.exit(main())
