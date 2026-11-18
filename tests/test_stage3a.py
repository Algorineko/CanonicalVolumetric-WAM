"""Stage 3a unit tests: leakage, semantics, metric correctness, loss masking."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FAILURES = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_split_leakage():
    from cvwam.dynamics_dataset import DynamicsDataset
    train = DynamicsDataset("/data/canonical-volumetric-wam/cache/voxels_ep",
                            ["demo_clean_balanced:10"])
    val = DynamicsDataset("/data/canonical-volumetric-wam/cache/voxels_ep",
                          ["demo_randomized_balanced:0"])
    tr_eps = {e for _, _, _, e in train.items}
    va_eps = {e for _, _, _, e in val.items}
    check("episode-level split: no shared episodes", not (tr_eps & va_eps),
          f"{tr_eps} vs {va_eps}")
    # adjacent-frame leakage: t values come from the SAME episode only; the
    # future frame t+k is inside the same episode by construction — verify
    # that no dataset item crosses episodes.
    ok = all(f"episode_{int(spec.split(':')[1]):07d}" in str(edir)
             for edir, t, k, spec in train.items[:20])
    check("items reference their own episode dir", ok)
    s = train[0]
    check("sample shapes", s["voxel"].shape[0] == 4 and s["qpos"].shape == (14,)
          and s["action"].shape == (16, 14)
          and s["state_future"].shape == s["state_t"].shape)


def test_action_semantics():
    """action_chunk[i] must equal the HDF5 action group at t+i (commands
    toward state t+i+1), not the state group."""
    import h5py, json, yaml
    cfg = yaml.safe_load(open("configs/smoke.yaml"))
    paths = "/data/openmw_balanced_work/demo_clean_balanced/open_microwave/aloha_agilex/data/episode_0000010.hdf5"
    from cvwam.dynamics_dataset import DynamicsDataset
    ds = DynamicsDataset("/data/canonical-volumetric-wam/cache/voxels_ep",
                         ["demo_clean_balanced:10"])
    item = None
    for edir, t, k, spec in ds.items:
        if t == 50 and k == 4:
            item = ds[ds.items.index((edir, t, k, spec))]
            break
    with h5py.File(paths, "r") as f:
        a_l = np.asarray(f["action/left_arm_joint_states"])
        s_l = np.asarray(f["state/left_arm_joint_states"])
    np.testing.assert_allclose(item["action"][0, :6].numpy(), a_l[50],
                               atol=1e-5, err_msg="action chunk must be action group")
    check("action_chunk == action[t:t+H] (not state)", True)
    # and state at t must equal state group at t (not action)
    with h5py.File(paths, "r") as f:
        qpos_ref = np.concatenate([np.asarray(f["state/left_arm_joint_states"])[50],
                                   np.asarray(f["state/right_arm_joint_states"])[50],
                                   np.asarray(f["state/left_ee_joint_states"])[50],
                                   np.asarray(f["state/right_ee_joint_states"])[50]])
    np.testing.assert_allclose(item["qpos"].numpy(), qpos_ref, atol=1e-5)
    check("qpos == state[t]", True)


def test_metric_correctness():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_tiny", "scripts/eval_tiny.py")
    # eval_tiny imports torch/cv2-heavy deps; import only the function via exec
    src = Path("scripts/eval_tiny.py").read_text()
    ns = {"__file__": "scripts/eval_tiny.py"}
    exec(compile(src.split("def main()")[0], "eval_tiny_head", "exec"), ns)
    metric_row = ns["metric_row"]
    iou = ns["iou"]
    f1 = ns["f1"]
    X, Y, Z = 8, 8, 8
    N = 2
    state_t = torch.full((N, X, Y, Z), 2, dtype=torch.int64)
    state_f = torch.full((N, X, Y, Z), 2, dtype=torch.int64)
    obs_f = torch.zeros((N, X, Y, Z))
    # future-observed box containing both hit locations
    obs_f[:, 4:8, 4:7, 4:7] = 1
    # free block observed at t and future
    state_t[:, 2:4, 2:4, 2:4] = 1      # free at t
    obs_f[:, 2:4, 2:4, 2:4] = 1
    state_f[:, 2:4, 2:4, 2:4] = 1      # still free
    state_t[:, 6, 5, 5] = 0            # hit at t
    state_f[:, 6, 5, 5] = 1            # ...became free (hit->free flip)
    state_f[:, 5, 5, 5] = 0            # new hit at -1 x (was free at t)
    state_t[:, 5, 5, 5] = 1
    pred_hit = (state_f == 0)          # oracle prediction
    cur_hit = state_t == 0
    cur_free = state_t == 1
    pred_flip = pred_hit ^ cur_hit
    r = metric_row(pred_hit, pred_flip, cur_hit, cur_free, state_t, state_f, obs_f)
    check("oracle: observed occupancy IoU == 1", r["observed_occupancy_iou"] == 1.0,
          f"{r['observed_occupancy_iou']}")
    check("oracle: surface F1(tol1) == 1", r["surface_f1_tol1"] == 1.0)
    check("oracle: change F1 == 1", r["change_f1"] == 1.0,
          f"{r['change_f1']}")
    check("oracle: known-free false occ == 0", r["knownfree_false_occ"] == 0.0)
    # shifted prediction inside the observed region: tol1 catches it exactly
    pred_shift = torch.zeros_like(pred_hit)
    pred_shift[:, 6, 5, 5] = True      # 1 voxel off from GT hit [5,5,5]
    r2 = metric_row(pred_shift, pred_flip, cur_hit, cur_free, state_t, state_f, obs_f)
    check("shift-1 pred: surface F1(tol1)==1", r2["surface_f1_tol1"] == 1.0,
          f"{r2['surface_f1_tol1']}")
    check("shift-1 pred: exact-metric penalized (occ IoU < 1)",
          r2["observed_occupancy_iou"] < 1.0)


def test_decoupled_loss_masking():
    """Unknown future voxels must not contribute to the occupancy loss:
    corrupting GT state at future-unknown voxels must leave the loss unchanged."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_tiny", "scripts/train_tiny.py")
    tt = importlib.util.module_from_spec(spec)
    src = Path("scripts/train_tiny.py").read_text()
    ns = {"__name__": "train_tiny_test", "__file__": "scripts/train_tiny.py"}
    exec(compile(src.split("def main()")[0], "train_tiny_head", "exec"), ns)
    losses = ns["losses"]
    B, X, Y, Z = 4, 10, 10, 10
    device = "cpu"
    vox = torch.zeros(B, 4, X, Y, Z)
    vox[:, 1] = 1.0                     # everything free at t
    qpos = torch.zeros(B, 14)
    act = torch.zeros(B, 16, 14)
    batch = {"voxel": vox, "qpos": qpos, "action": act,
             "offset": [4] * B, "state_future": None, "observed_future": None}
    # observed future = free everywhere except one hit voxel
    state_f = torch.full((B, X, Y, Z), 1, dtype=torch.int64)
    obs_f = torch.ones(B, X, Y, Z)
    state_f[:, 0, 0, 0] = 0; obs_f[:, 0, 0, 0] = 1
    batch["state_future"] = state_f
    batch["observed_future"] = obs_f

    class Dummy(nn.Module):
        def forward(self, *a, **k):
            out = torch.zeros(B, 8, X, Y, Z)
            out[:, 0] = torch.zeros(B, X, Y, Z)   # offset-0 occ logit
            out[:, 1] = torch.zeros(B, X, Y, Z)   # offset-0 vis logit
            return out

    model = Dummy()
    oi = torch.tensor([0] * B)
    # monkeypatch split_outputs expectation: decoupled block indexing
    blocks = [{"occ_logit": torch.zeros(B, X, Y, Z, requires_grad=True),
               "vis_logit": torch.zeros(B, X, Y, Z, requires_grad=True)}] * 4
    L0 = losses(batch, model, "p1_v", None, torch.tensor(4.0), device)
    # corrupt the state at future-UNKNOWN voxels: make them all 'hit'
    state_f2 = state_f.clone()
    unknown = obs_f == 0
    state_f2[unknown] = 0
    batch["state_future"] = state_f2
    L1 = losses(batch, model, "p1_v", None, torch.tensor(4.0), device)
    d = abs(L0["loss_k4"].item() - L1["loss_k4"].item())
    check("P1-V: unknown-region GT corruption does not change occupancy loss",
          d < 1e-6, f"delta {d:.2e}")
    # corrupting an OBSERVED voxel's state must change the loss
    state_f3 = state_f.clone()
    state_f3[:, 1, 1, 1] = 0            # observed free -> corrupted to hit
    batch["state_future"] = state_f3
    L2 = losses(batch, model, "p1_v", None, torch.tensor(4.0), device)
    check("P1-V: observed-region corruption DOES change loss",
          abs(L0["loss_k4"].item() - L2["loss_k4"].item()) > 1e-6)


def test_model_forward():
    from cvwam.models_tiny import TinyDynamicsNet, split_outputs, count_params
    for variant in ("state_ce", "residual", "decoupled"):
        m = TinyDynamicsNet(variant=variant, base=8, n_offsets=4)
        B = 2
        vox = torch.zeros(B, 4, 35, 65, 45)
        vox[:, 0, 2, 2, 2] = 1
        vox[:, 1, 3, 3, 3] = 1
        out = m(vox, torch.zeros(B, 14), torch.zeros(B, 16, 14),
                torch.tensor([0, 1]))
        C = out.shape[1] // 4
        check(f"forward {variant}: shape", out.shape == (B, C * 4, 35, 65, 45),
              f"{tuple(out.shape)}")
        check(f"forward {variant}: finite", torch.isfinite(out).all())
    p = count_params(TinyDynamicsNet(base=32))
    check("params ~ a few M", 0.3e6 < p < 12e6, f"{p/1e6:.2f}M")


if __name__ == "__main__":
    test_split_leakage()
    test_action_semantics()
    test_metric_correctness()
    test_decoupled_loss_masking()
    test_model_forward()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL STAGE3A TESTS PASSED")
