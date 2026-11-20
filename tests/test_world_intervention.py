"""Model Integrity Audit: regression tests for world_intervention.py fixes.

Covers S5.1 (W0/W1 shared world noise), S5.3 (cross-episode donor pool),
S5.4 (physical-unit FK / serving denorm contract) and the AUDIT-DF
direct-forward probe plumbing. Uses small RANDOM models on CPU — read-only
w.r.t. checkpoints and runs/ artifacts; results go to a NEW file
runs/model_integrity/l9_rerun/test_results_world_intervention.json.

Run (openwam env): /home/zhangjialiang/miniconda3/envs/openwam/bin/python \
    tests/test_world_intervention.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from world_intervention import (  # noqa: E402
    DF_T_GRID, denorm_phys, direct_forward_probe, euler_chunk, hit_iou,
    pick_donor)

OUT = REPO / "runs/model_integrity/l9_rerun/test_results_world_intervention.json"
RESULTS = {}


def check(name, ok, detail):
    RESULTS[name] = dict(ok=bool(ok), detail=detail)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def _unzero_action_head(m):
    """Random-init Stage4AModel has a ZERO-INIT v_a head (v_a == 0 for every
    input), which would make propagation tests pass trivially. Give it a
    small nonzero head so world->action influence is observable. This tests
    the PLUMBING (harness), not trained behavior — the trained behavior is
    measured by the actual runs on checkpoints."""
    torch.nn.init.normal_(m.v_a.head.weight, std=0.05)
    torch.nn.init.normal_(m.v_a.head.bias, std=0.05)
    return m


def small_inputs(B=2, seed=0):
    torch.manual_seed(seed)
    x = (torch.rand(B, 4, 40, 65, 45) > 0.9).float()
    qpos = torch.rand(B, 14)
    oi = torch.zeros(B, dtype=torch.long)
    act = torch.randn(B, 16, 14)
    return x, qpos, oi, act


# ------------------------------------------------------------------ S5.1

def test_s5_1_shared_world_noise():
    """w0_init=None + explicit world_noise: same noise -> identical chunks
    (pre-fix, euler_chunk re-drew torch.randn internally so two identical
    calls differed); different noise -> different chunks."""
    m = _unzero_action_head(Stage4AModel(variant="a3")).eval()
    x, qpos, oi, act = small_inputs()
    wn = torch.randn(2, 2, 10, 17, 12)
    a1 = euler_chunk(m, x, qpos, oi, act, None, "a3", world_noise=wn)
    a2 = euler_chunk(m, x, qpos, oi, act, None, "a3", world_noise=wn)
    ok1 = check("S5.1_same_world_noise_deterministic", torch.equal(a1, a2),
                f"max|diff|={float((a1 - a2).abs().max()):.3e}")
    wn2 = torch.randn_like(wn)
    a3 = euler_chunk(m, x, qpos, oi, act, None, "a3", world_noise=wn2)
    ok2 = check("S5.1_world_noise_change_propagates",
                not torch.equal(a1, a3),
                f"|a(wn)-a(wn')|={float((a1 - a3).norm(dim=-1).mean()):.4f}")
    # the fix must refuse the old implicit behavior: no world_noise -> assert
    try:
        euler_chunk(m, x, qpos, oi, act, None, "a3", world_noise=None)
        ok3 = check("S5.1_missing_noise_rejected", False, "no assert raised")
    except AssertionError:
        ok3 = check("S5.1_missing_noise_rejected", True,
                    "AssertionError raised without explicit world_noise")
    return ok1 and ok2 and ok3


# ------------------------------------------------------------------ S5.3

def test_s5_3_donor():
    """Donor comes from another episode and must differ (hit-IoU < 0.9);
    identical-canvas-only pools fail the check (warn path)."""
    # synthetic canvases: A hits top-left block, B hits disjoint block
    cA = torch.zeros(2, 10, 17, 12)
    cA[0, :5, :8, :6] = 1.0
    cB = torch.zeros(2, 10, 17, 12)
    cB[0, 5:, 9:, 6:] = 1.0
    cA_clone = cA.clone()
    pool = {"demo_clean_balanced:1": [cA, cA.clone()],
            "demo_randomized_balanced:0": [cB, cB.clone()],
            "demo_clean_balanced:2": [cA_clone]}
    check("S5.3_hit_iou_identical", hit_iou(cA, cA_clone) == 1.0, "IoU=1")
    check("S5.3_hit_iou_disjoint", hit_iou(cA, cB) == 0.0, "IoU=0")
    check("S5.3_hit_iou_both_empty", np.isnan(hit_iou(torch.zeros_like(cA),
                                                      torch.zeros_like(cA))),
          "NaN for empty union")
    rng = np.random.default_rng(0)
    c, iou, ok = pick_donor(cA, "demo_clean_balanced:1", pool, rng)
    ok1 = check("S5.3_donor_differs_from_own_future", ok and iou < 0.9,
                f"accepted donor hit-IoU={iou:.3f} (< 0.9)")
    # pool where every cross-episode donor is IDENTICAL to the own future:
    # all tries fail -> ok=False (warning path), not a silent pass
    pool_bad = {"demo_randomized_balanced:0": [cA_clone]}
    rng = np.random.default_rng(0)
    _, iou_bad, ok_bad = pick_donor(cA, "demo_clean_balanced:1", pool_bad, rng)
    ok2 = check("S5.3_identical_donor_rejected",
                (not ok_bad) and iou_bad >= 0.9,
                f"ok={ok_bad}, IoU={iou_bad:.3f} (>= 0.9 -> warn counted)")
    # all-empty pool with an all-empty OWN future: hit-IoU is NaN (empty
    # union) -> donor unusable, all tries fail -> ok=False, no crash
    pool_empty = {"demo_randomized_balanced:0": [torch.zeros_like(cA)]}
    rng = np.random.default_rng(0)
    _, iou_e, ok_e = pick_donor(torch.zeros_like(cA), "demo_clean_balanced:1",
                                pool_empty, rng)
    ok3 = check("S5.3_empty_donor_nan_handled",
                (not ok_e) and np.isnan(iou_e),
                f"ok={ok_e}, IoU=NaN handled without crash (empty union)")
    # an empty donor vs a NON-empty own future is a genuinely different
    # (degenerate) future: IoU=0 < thr -> accepted by design
    rng = np.random.default_rng(0)
    _, iou_d, ok_d = pick_donor(cA, "demo_clean_balanced:1", pool_empty, rng)
    check("S5.3_empty_donor_vs_nonempty_own_is_different",
          ok_d and iou_d == 0.0,
          f"ok={ok_d}, IoU={iou_d:.3f} (all-empty canvas is 'nothing will "
          "be hit anywhere' — a wrong future by construction)")
    return ok1 and ok2 and ok3


# ------------------------------------------------------------------ S5.4

def test_s5_4_denorm():
    """denorm_phys follows the serving contract exactly (cvwam_fm/model.py:
    act.clamp(0,1) THEN lo + act*scale) and leaves [0,1]-range gripper dims
    (lo=0, hi=1, scale=1 in act_stats) unchanged."""
    stats = json.loads((REPO / "runs/stage4a/act_stats.json").read_text())
    torch.manual_seed(1)
    an = torch.rand(3, 16, 14) * 1.6 - 0.3   # values outside [0,1] on purpose
    p = denorm_phys(an, stats)
    lo = torch.tensor(stats["lo"])
    sc = torch.tensor(stats["scale"])
    p_ref = an.clamp(0, 1) * sc + lo
    ok1 = check("S5.4_denorm_matches_serving_contract",
                torch.allclose(p, p_ref, atol=1e-6),
                "clamp(0,1) applied BEFORE lo+act*scale")
    below = an < 0
    ok2 = check("S5.4_out_of_range_clamped",
                torch.allclose(p[below], lo.expand_as(p)[below], atol=1e-6),
                f"{int(below.sum())} sub-zero values map to lo")
    ok3 = check("S5.4_zero_normalized_maps_to_lo",
                torch.allclose(denorm_phys(torch.zeros(1, 1, 14), stats)[0, 0],
                               lo, atol=1e-7),
                "normalized 0 -> raw lo exactly (constant dims recover their "
                "raw constant: right arm 0, Rgrip 1.0; Lgrip 0)")
    # physical vs normalized deviation are distinct quantities now (the old
    # code reported the normalized number in both fields)
    a0 = torch.rand(3, 16, 14)
    a1 = a0 + 0.1
    dev_norm = float((a1 - a0).norm(dim=-1).mean())
    dev_phys = float((denorm_phys(a1, stats) - denorm_phys(a0, stats))
                     .norm(dim=-1).mean())
    ok4 = check("S5.4_norm_and_phys_devs_distinct",
                abs(dev_norm - dev_phys) > 1e-3,
                f"norm={dev_norm:.4f} vs phys={dev_phys:.4f} (arm scale>1 "
                "amplifies physical deviation)")
    return ok1 and ok2 and ok3 and ok4


# ------------------------------------------------------------------ DF

def test_df_probe():
    """direct_forward_probe plumbing: a0 ignores noisy_world -> |dv_a| exactly
    0 (sanity); a3 responds to canvas content -> |dv_a| > 0."""
    x, qpos, oi, act = small_inputs(seed=3)
    wA = torch.zeros(2, 2, 10, 17, 12)
    wB = torch.zeros(2, 2, 10, 17, 12)
    wB[0, 0, 0, 0, 0] = 1.0
    canv = {"W2_current": wA, "W4_oracle": wB}
    m0 = Stage4AModel(variant="a0").eval()
    r0 = direct_forward_probe(m0, x, qpos, oi, act, canv, t_grid=(0.5,))
    d0 = max(abs(v) for row in r0["per_t"].values() for v in row.values())
    ok1 = check("DF_a0_exact_zero_sanity", d0 == 0.0,
                f"max|dv_a|={d0:.3e} (a0 has no world path)")
    m3 = _unzero_action_head(Stage4AModel(variant="a3")).eval()
    r3 = direct_forward_probe(m3, x, qpos, oi, act, canv,
                              t_grid=DF_T_GRID)
    d3 = max(v for row in r3["per_t"].values() for v in row.values())
    ok2 = check("DF_a3_content_changes_one_step_v_a", d3 > 0.0,
                f"max|dv_a| over t-grid={d3:.4f}")
    return ok1 and ok2


if __name__ == "__main__":
    torch.manual_seed(42)
    test_s5_1_shared_world_noise()
    test_s5_3_donor()
    test_s5_4_denorm()
    test_df_probe()
    n_ok = sum(r["ok"] for r in RESULTS.values())
    summary = dict(n_tests=len(RESULTS), n_pass=n_ok, results=RESULTS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n{n_ok}/{len(RESULTS)} passed -> {OUT}")
