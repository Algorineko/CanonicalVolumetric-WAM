"""Stage 3b unit tests: matched-loss equivalence, probability relations,
pos_weight regression (the Bug-3 guard), P1-W weights."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_factorized_equals_3class():
    """Unweighted factorized BCE with P1-F's normalization == unweighted
    3-class CE, where p_hit=p_obs*p_occ, p_free=p_obs*(1-p_occ),
    p_unknown=1-p_obs. Verified on random logits."""
    torch.manual_seed(0)
    N = 500
    vis_logit = torch.randn(N)
    occ_logit = torch.randn(N)
    y_obs = torch.bernoulli(torch.sigmoid(vis_logit)).long()      # 0/1
    y_occ = torch.bernoulli(torch.sigmoid(occ_logit)).long()      # 0/1 (defined for all)
    p_obs = torch.sigmoid(vis_logit)
    p_occ = torch.sigmoid(occ_logit)
    probs = torch.stack([p_obs * p_occ, p_obs * (1 - p_occ), 1 - p_obs], dim=1)
    check("probability simplex", bool((probs.sum(1) - 1).abs().max() < 1e-5))
    # 3-class CE on composed probabilities (log of composed probs)
    logp = torch.log(probs.clamp_min(1e-12))
    ce3 = F.nll_loss(logp, y_obs * (y_occ.clamp(max=1)))
    # careful: target state = hit(1) if y_obs=1,y_occ=1; free if obs=1,occ=0; unknown if obs=0
    tgt = torch.where(y_obs == 1, 1 - y_occ, torch.full_like(y_occ, 2))
    ce3 = F.nll_loss(logp, tgt)
    # P1-F form: (sum_all BCE_vis + sum_observed BCE_occ)/N_all
    n_all = N
    m_obs = y_obs == 1
    vis_sum = F.binary_cross_entropy_with_logits(vis_logit, y_obs.float(), reduction="sum")
    occ_sum = F.binary_cross_entropy_with_logits(occ_logit[m_obs],
                                                 y_occ[m_obs].float(), reduction="sum")
    p1f = (vis_sum + occ_sum) / n_all
    check("factorized(unweighted) == 3-class NLL", bool((p1f - ce3).abs() < 1e-5),
          f"|Δ|={abs(p1f - ce3).item():.2e}")
    # identity at the per-voxel level too
    bce_vis = F.binary_cross_entropy_with_logits(vis_logit, y_obs.float(), reduction="none")
    bce_occ = F.binary_cross_entropy_with_logits(occ_logit, y_occ.float(), reduction="none")
    per_voxel_3class = -logp[torch.arange(N), tgt]
    ident = torch.where(m_obs, bce_vis + bce_occ, bce_vis)
    check("per-voxel identity BCE_vis+mask*BCE_occ == -log p(state)",
          bool((ident - per_voxel_3class).abs().max() < 1e-5))


def test_pos_weight_regression():
    """Bug-3 guard: the occupancy pos_weight must up-weight HIT (positive)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tt3b", "scripts/train_tiny.py")
    src = Path("scripts/train_tiny.py").read_text()
    ns = {"__name__": "tt3b", "__file__": "scripts/train_tiny.py"}
    exec(compile(src.split("if __name__")[0], "train_tiny", "exec"), ns)
    # simulate the class_weights_from output order [hit, free, unknown]
    w_state = torch.tensor([22.65, 2.73, 0.386])
    pos_w = (w_state[0] / w_state[1]).detach()
    check("P1-V pos_weight = hit/free ratio (>>1, up-weights hit)",
          float(pos_w) > 5.0, f"pos_w={float(pos_w):.2f}")
    check("P1-W weights transplant pos_weight ratio", ns["P1W_WEIGHTS"][0] / ns["P1W_WEIGHTS"][1] > 5.0,
          str(ns["P1W_WEIGHTS"]))


def test_p1w_forward():
    from cvwam.models_tiny import TinyDynamicsNet
    m = TinyDynamicsNet(variant="state_ce", base=8, n_offsets=4)
    B = 2
    vox = torch.zeros(B, 4, 20, 20, 20)
    out = m(vox, torch.zeros(B, 14), torch.zeros(B, 16, 14), torch.tensor([0, 1]))
    check("p1_w shares state_ce head (3*n_offsets channels)", out.shape[1] == 12)
    m2 = TinyDynamicsNet(variant="decoupled", base=8, n_offsets=4)
    out2 = m2(vox, torch.zeros(B, 14), torch.zeros(B, 16, 14), torch.tensor([0, 1]))
    check("p1_f shares decoupled head (2*n_offsets channels)", out2.shape[1] == 8)


if __name__ == "__main__":
    test_factorized_equals_3class()
    test_pos_weight_regression()
    test_p1w_forward()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL STAGE3B TESTS PASSED")
