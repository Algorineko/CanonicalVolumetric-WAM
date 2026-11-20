"""Offline sanity: FM integration sign in the serving adapter.

Reproduces runs/stage4a/failures/fm_integration_sign_bug_note.md:
load a0_s42, first 6 val batches, 8-step Euler from N(0,I) at t=1 -> t=0.
Expected (normalized units, per-dim MSE vs GT action):
  act += dt*v  ->  ~8.99   (reverse integration, garbage)
  act -= dt*v  ->  ~0.0024 (correct; train protocol x_t=(1-t)a+t*e, v=e-a)
Both directions share the same initial noise (seeded) for a paired comparison.

Run (openwam env, repo root): python runs/stage4a/failures/sanity_fm_sign.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from cvwam.dynamics_dataset import OFFSETS, DynamicsDataset  # noqa: E402
from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from train_stage4a import normalize_act  # noqa: E402


@torch.no_grad()
def integrate(model, x, qpos, oi, act_gt, v_field, sign, n_euler, noise):
    act = noise.clone()
    for i in range(n_euler):
        t = torch.tensor([1.0 - i / n_euler] * x.shape[0], device=x.device)
        o = model(x, act, t, qpos, oi, noisy_world=None, cam72=None)
        act = act + sign * (1.0 / n_euler) * o[v_field]
    return ((act - act_gt) ** 2).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/stage4a/a0_s42/best.pt")
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--max-batches", type=int, default=6)
    ap.add_argument("--n-euler", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    torch.manual_seed(0)

    device = args.device
    ck = torch.load(REPO / args.ckpt, map_location=device, weights_only=False)
    cfg = yaml.safe_load((REPO / args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep"
    sm = json.loads((REPO / args.split_manifest).read_text())
    val_eps = sm["splits"]["clean_val"] + sm["splits"]["rand_val"]
    ds = DynamicsDataset(root, val_eps, stride_t=2)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = Stage4AModel(variant=ck["variant"],
                         bottleneck_ch=ck.get("bottleneck_ch", 128)).to(device).eval()
    model.load_state_dict(ck["model"])
    stats = ck["stats"]

    mse = {"plus": [], "minus": []}
    for bi, batch in enumerate(dl):
        if bi >= args.max_batches:
            break
        x = batch["voxel"].to(device)
        qpos = batch["qpos"].to(device)
        act_gt = torch.from_numpy(
            normalize_act(batch["action"].numpy(), stats)).to(device)
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]], device=device)
        noise = torch.randn_like(act_gt)  # same start noise for both signs
        mse["plus"].append(integrate(model, x, qpos, oi, act_gt, "v_a", +1.0,
                                     args.n_euler, noise))
        mse["minus"].append(integrate(model, x, qpos, oi, act_gt, "v_a", -1.0,
                                      args.n_euler, noise))
        print(f"batch {bi}: += {mse['plus'][-1]:.4f}  -= {mse['minus'][-1]:.4f}")

    out = {k: float(np.mean(v)) for k, v in mse.items()}
    print(f"MEAN over {len(mse['plus'])} batches: "
          f"act += dt*v -> {out['plus']:.4f}   act -= dt*v -> {out['minus']:.4f}")
    ok = out["minus"] < 0.05 and out["plus"] > 1.0
    print("SANITY:", "PASS (minus=correct, plus=garbage)" if ok else "UNEXPECTED")
    (Path(__file__).parent / "sanity_fm_sign_result.json").write_text(
        json.dumps({"result": out, "n_batches": len(mse["plus"]),
                    "n_euler": args.n_euler, "ckpt": args.ckpt}, indent=2))


if __name__ == "__main__":
    main()
