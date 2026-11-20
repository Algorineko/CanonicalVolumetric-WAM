"""Model Integrity Audit: computation-graph conditioning tests (S1-S4).

Run (openwam env, GPU): python tests/test_model_conditioning.py
Each test prints CONFIRMED/NOT_REPRODUCED for the review findings and writes
runs/model_integrity/test_results.json. Read-only w.r.t. checkpoints.
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

DEVICE = "cuda:1"
OUT = REPO / "runs/model_integrity/test_results.json"
RESULTS = {}


def load(variant, seed=42):
    ck = torch.load(REPO / f"runs/stage4a/{variant}_s{seed}/best.pt",
                    map_location=DEVICE, weights_only=False)
    m = Stage4AModel(variant=ck["variant"],
                     bottleneck_ch=ck.get("bottleneck_ch", 128)).to(DEVICE).eval()
    m.load_state_dict(ck["model"])
    return m, ck


def fixed_inputs(variant, B=2):
    torch.manual_seed(7)
    x = (torch.rand(B, 4, 40, 65, 45, device=DEVICE) > 0.9).float()
    act = torch.randn(B, 16, 14, device=DEVICE)
    qpos = torch.rand(B, 14, device=DEVICE)
    oi = torch.zeros(B, dtype=torch.long, device=DEVICE)
    world = torch.randn(B, 2, 10, 17, 12, device=DEVICE) \
        if variant in ("a2", "a3", "a3mask") else None
    return x, act, qpos, oi, world


@torch.no_grad()
def t1_s1_current_conditioning():
    """S1: does swapping x_t change v_a/v_w with everything else fixed?"""
    out = {}
    for variant in ("a0", "a2", "a3", "a3mask"):
        m, _ = load(variant)
        x, act, qpos, oi, w = fixed_inputs(variant)
        t = torch.full((2,), 0.5, device=DEVICE)
        o1 = m(x, act, t, qpos, oi, noisy_world=w, cam72=None)
        o2 = m(torch.zeros_like(x), act, t, qpos, oi, noisy_world=w, cam72=None)
        d_va = float((o1["v_a"] - o2["v_a"]).abs().mean())
        d_vw = float((o1["v_w"] - o2["v_w"]).abs().mean()) if w is not None else None
        # a0/a2: must respond; a3: predicted ~0 (S1); a3mask: tiny via gate 0.019
        out[variant] = {"d_v_a": d_va, "d_v_w": d_vw}
        print(f"T1[{variant}]: d_v_a={d_va:.3e} d_v_w={d_vw}")
    verdict = {
        "a0_responds": out["a0"]["d_v_a"] > 1e-4,
        "a2_responds": out["a2"]["d_v_a"] > 1e-4,
        "a3_zero_dependency": out["a3"]["d_v_a"] < 1e-6 and out["a3"]["d_v_w"] < 1e-6,
        "a3mask_near_closed_gate": 0 < out["a3mask"]["d_v_a"] < 0.05,
    }
    RESULTS["S1_current_conditioning"] = {"outputs": out, "verdict": verdict,
                                          "confirmed": verdict["a3_zero_dependency"]}
    print(f"T1 verdict: {verdict}")


@torch.no_grad()
def t1b_encoder_init_displacement():
    """Review P1: did the a3 encoder move during stage4a training? Its INIT
    source is the tokenizer AE (train_stage4a.py loads ae['enc']), so compare
    against THAT, not random init."""
    m, _ = load("a3")
    ae = torch.load(REPO / "runs/stage4a/tokenizer_ae/best.pt",
                    map_location=DEVICE, weights_only=False)
    ae_enc = ae["enc"]
    sd = m.state_dict()
    rel = []
    for k2, ref in ae_enc.items():
        k = f"enc.{k2}"
        if k in sd and sd[k].dtype.is_floating_point:
            rel.append(float((sd[k] - ref.to(sd[k].dtype)).norm()
                             / (ref.norm() + 1e-9)))
    RESULTS["S1_encoder_displacement_vs_AE_init"] = {
        "n_tensors": len(rel),
        "rel_displacement_mean": float(np.mean(rel)),
        "rel_displacement_max": float(np.max(rel)),
        "interpretation": "~0 => encoder frozen during stage4a (expected: "
                          "nothing consumes G_tok in a3)",
    }
    print(f"T1b enc displacement vs AE init: mean {np.mean(rel):.4f} "
          f"max {np.max(rel):.4f} (~0 => never trained in stage4a)")


def t2_s2_observer():
    """S2: observer inactive at cam72=None and untrained (zero grad)."""
    m, _ = load("a3mask")
    x, act, qpos, oi, w = fixed_inputs("a3mask")
    t = torch.full((2,), 0.5, device=DEVICE)
    with torch.no_grad():
        o = m(x, act, t, qpos, oi, noisy_world=w, cam72=None)
        obs_inactive = o.get("obs_logits") is None
        # with cam72 present, does observer produce output, and is it trained?
        cam72 = torch.randn(2, 72, device=DEVICE)
        o2 = m(x, act, t, qpos, oi, noisy_world=w, cam72=cam72)
        has_logits = o2.get("obs_logits") is not None
        # weight displacement vs fresh init (trained-in-stage4a? -> nonzero)
        torch.manual_seed(0)
        fresh = Stage4AModel(variant="a3mask", bottleneck_ch=128).to(DEVICE)
        d = sum(float((a - b).norm()) for a, b in
                zip(m.observer.state_dict().values(),
                    fresh.observer.state_dict().values()))
    grad_ok = True
    try:
        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.observer.parameters():
            p.requires_grad_(True)
        o3 = m(x, act, t, qpos, oi, noisy_world=w, cam72=cam72)
        m.zero_grad(set_to_none=True)
        o3["obs_logits"].abs().sum().backward()
        gnorm = sum(float(p.grad.norm()) for p in m.observer.parameters()
                    if p.grad is not None)
    except RuntimeError as e:
        grad_ok = False
        gnorm = -1
    RESULTS["S2_observer"] = {"inactive_at_none": obs_inactive,
                              "produces_with_cam72": has_logits,
                              "grad_flows_to_observer": grad_ok,
                              "grad_norm_one_backward": gnorm,
                              "observer_weight_disp_vs_fresh_init": d}
    print(f"T2: inactive@None={obs_inactive} produces={has_logits} "
          f"weight_disp_vs_fresh={d:.4f} (≈0 vs init-scale => never trained)")
    # an untrained-from-scratch observer that received gradient flow in training
    # would have nonzero weight displacement vs init; here grad check is about
    # the serving ckpt: compare observer weights vs fresh init
    torch.manual_seed(0)
    fresh = Stage4AModel(variant="a3mask", bottleneck_ch=128).to(DEVICE)
    d = sum(float((a - b).norm()) for (n, a), (n2, b) in
            zip(m.observer.state_dict().items(),
                fresh.observer.state_dict().items()))
    RESULTS["S2_observer"] = {"inactive_at_none": obs_inactive,
                              "produces_with_cam72": has_logits,
                              "fresh_grad_norm_one_backward": gnorm,
                              "observer_weight_disp_vs_init": d}
    print(f"T2: inactive@None={obs_inactive} produces={has_logits} "
          f"weight_disp_vs_init={d:.4f}")


@torch.no_grad()
def t3_s3_spatial_permutation():
    """S3: world tokens carry no positional info. PURE position test: permute
    the TOKEN ORDER after w_tok (same token multiset, permuted KV order) —
    attention over KV is permutation-equivariant, so if no positional info
    exists the action output must be IDENTICAL. Control: shuffle token VALUES
    (changes the multiset -> output must change)."""
    m, _ = load("a2")
    x, act, qpos, oi, w = fixed_inputs("a2")
    t = torch.full((2,), 0.5, device=DEVICE)
    orig_wtok = m.w_tok

    def make_patched(mode):
        class P(torch.nn.Module):
            def forward(self, z):
                tok = orig_wtok(z)  # (B,270,d)
                if mode == "identity":
                    return tok
                perm = torch.randperm(tok.shape[1], device=tok.device)
                return tok[:, perm] if mode == "permute" else \
                    tok.flip(1)  # fallback
        return P()

    m.w_tok = make_patched("identity")
    o1 = m(x, act, t, qpos, oi, noisy_world=w, cam72=None)
    m.w_tok = make_patched("permute")
    o2 = m(x, act, t, qpos, oi, noisy_world=w, cam72=None)
    d_perm = float((o1["v_a"] - o2["v_a"]).abs().mean())
    # value control: shuffle canvas values (changes token contents)
    m.w_tok = orig_wtok
    w_shuf = w.flatten(1)[:, torch.randperm(w[0].numel(), device=DEVICE)].reshape(w.shape)
    o3 = m(x, act, t, qpos, oi, noisy_world=w_shuf, cam72=None)
    d_val = float((o1["v_a"] - o3["v_a"]).abs().mean())
    RESULTS["S3_spatial_permutation"] = {
        "v_a_diff_token_order_permute": d_perm,
        "v_a_diff_value_shuffle": d_val,
        "note": "token-order permute diff ~0 => no positional info in world "
                "tokens (attention permutation-invariance); value shuffle is "
                "the content control",
    }
    print(f"T3: token-order permute d={d_perm:.3e} value-shuffle d={d_val:.3e}")


@torch.no_grad()
def t4_s4_time_embedding():
    """S4: does the action stream respond to t at all (a0 direct vs a2/a3
    indirect-through-world)?"""
    out = {}
    for variant in ("a0", "a2", "a3"):
        m, _ = load(variant)
        x, act, qpos, oi, w = fixed_inputs(variant)
        o1 = m(x, act, torch.full((2,), 0.1, device=DEVICE), qpos, oi,
               noisy_world=w, cam72=None)
        o2 = m(x, act, torch.full((2,), 0.9, device=DEVICE), qpos, oi,
               noisy_world=w, cam72=None)
        out[variant] = float((o1["v_a"] - o2["v_a"]).abs().mean())
    RESULTS["S4_time_response_v_a"] = out
    print(f"T4 v_a |t=0.1 vs 0.9|: {out}")


def main():
    t1_s1_current_conditioning()
    t1b_encoder_init_displacement()
    t2_s2_observer()
    t3_s3_spatial_permutation()
    t4_s4_time_embedding()
    OUT.write_text(json.dumps(RESULTS, indent=2))
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
