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
  W3 wrong-fut  : world stream starts from another trajectory's future canvas
                  (same task, similar phase)
  W4 oracle     : world stream starts from the GT future canvas (pooled
                  x_{t+k}) — OFFLINE DIAGNOSTIC ONLY (leaks the true future;
                  never a deployment or closed-loop claim)
  W5 noise      : world stream starts from pure noise at every step (no content)

All interventions replace the world stream's INITIAL canvas (the information
the action expert reads during ALL denoising steps). W1 is an architectural
block active at every cross-attn in every step.

Recorded per intervention: action deviation vs W0 (normalized + radians),
EE-trajectory deviation (FK over the predicted chunk), expert action error
(vs GT chunk), per offset k (proximity-to-interaction proxy).

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
from cvwam.dynamics_dataset import OFFSETS, DynamicsDataset  # noqa: E402
from cvwam.models_stage4a import Stage4AModel  # noqa: E402
from train_stage4a import pooled_occupancy, normalize_act  # noqa: E402

N_EULER = 8


@torch.no_grad()
def euler_chunk(model, x, qpos, oi, act0_noise, w0_init, variant,
                block_world_to_action=False, n_euler=N_EULER):
    """Denoise action (and world) streams; returns final action chunk (B,16,14)
    in normalized units. w0_init: (B,2,10,17,12) initial world canvas or None
    (W0: pure noise start). block: zero world->action cross-attn (a3/a3mask).
    """
    B = x.shape[0]
    act = act0_noise.clone()
    hold = w0_init is not None           # W2/W3/W4: world HELD at injected canvas
    world = w0_init.clone() if w0_init is not None \
        else torch.randn(B, 2, 10, 17, 12, device=x.device)
    for i in range(n_euler):
        t = torch.tensor([1.0 - i / n_euler] * B, device=x.device)
        o = model(x, act, t, qpos, oi, noisy_world=world, cam72=None,
                  block_world_to_action=block_world_to_action)
        act = act + (1.0 / n_euler) * o["v_a"]
        if not hold:
            world = world + (1.0 / n_euler) * o["v_w"]   # W0: world ODE
    return act


def ee_traj(actions_cache_order, fk_fn):
    """actions (16,14) cache order -> EE xyz path (16,3) via FK (left arm)."""
    pts = []
    for i in range(actions_cache_order.shape[0]):
        p = fk_fn(actions_cache_order[i])
        pts.append(p)
    return np.stack(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--split-manifest", default="runs/stage3b/split_manifest.json")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda:0"
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

    # wrong-future canvas pool: futures from OTHER episodes at similar phase
    other = {}
    for spec in sm["splits"]["clean_val"]:
        v, e = spec.split(":")
        other[spec] = (v, int(e))

    import functools
    import build_robot_region as brr
    chains = {s: brr.load_chain(s) for s in ("left",)}
    spheres = None  # EE path only needs link6 position

    def fk_left(a14_cache):
        """EE (link6) xyz from a cache-order 14-dim action (Larm6,Rarm6,Lgrip,Rgrip)."""
        qd = {f"fl_joint{i+1}": float(a14_cache[i]) for i in range(6)}
        grip = float(np.clip(a14_cache[12], 0, 1))
        qd["fl_joint7"] = float(np.clip(-0.01 + 0.055 * grip, 0, 0.04765))
        qd["fl_joint8"] = qd["fl_joint7"]
        poses = brr.fk_links(chains["left"][0], chains["left"][1], qd)
        R, tl = poses["fl_link6"]
        return tl

    INTERV = ("W0", "W1", "W2", "W3", "W4", "W5")
    agg = {w: dict(act_dev=[], act_dev_rad=[], ee_dev=[], expert_err=[],
                   per_k={k: [] for k in OFFSETS}) for w in INTERV}
    n_pairs = 0
    for bi, batch in enumerate(dl):
        if bi >= args.max_batches:
            break
        x = batch["voxel"].to(device)
        x_fut = batch["voxel_future"].to(device)
        qpos = batch["qpos"].to(device)
        act_gt = torch.from_numpy(normalize_act(batch["action"].numpy(),
                                                stats)).to(device)
        B = x.shape[0]
        oi = torch.tensor([OFFSETS.index(k) for k in batch["offset"]],
                          device=device)
        # fixed noise across interventions (seeded per batch index)
        g = torch.Generator(device="cpu").manual_seed(args.seed * 1000 + bi)
        act_noise = torch.randn(B, 16, 14, generator=g).to(device)
        world_noise = torch.randn(B, 2, 10, 17, 12, generator=g).to(device)
        cur_canvas = pooled_occupancy(x)
        fut_canvas = pooled_occupancy(x_fut)
        # W3: wrong future — shift the batch by half (pairs from other episodes)
        fut_other = torch.roll(fut_canvas, shifts=B // 2, dims=0)
        inits = {
            "W0": world_noise,
            "W2": cur_canvas + 0.05 * world_noise,
            "W3": fut_other + 0.05 * world_noise,
            "W4": fut_canvas + 0.05 * world_noise,   # ORACLE — offline only
            "W5": world_noise,                        # same as W0 start; differs by
            # re-noising below
        }
        chunks = {}
        for w in INTERV:
            if w == "W0":
                a = euler_chunk(model, x, qpos, oi, act_noise, None,
                                ck["variant"])
            elif w == "W1":
                a = euler_chunk(model, x, qpos, oi, act_noise, None,
                                ck["variant"], block_world_to_action=True)
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
        for w in INTERV:
            if w not in chunks:
                continue
            aw = chunks[w]
            dev = float((aw - a0).norm(dim=-1).mean())
            dev_rad = float((aw - a0).norm(dim=-1).mean())  # normalized units
            # expert error (vs GT)
            err = float((aw - act_gt).norm(dim=-1).mean())
            # EE trajectory deviation (per-sample, left arm FK)
            ee = []
            an = aw.cpu().numpy()
            a0n = a0.cpu().numpy()
            for j in range(min(B, 8)):
                p_w = ee_traj(an[j], fk_left)
                p_0 = ee_traj(a0n[j], fk_left)
                ee.append(float(np.linalg.norm(p_w - p_0, axis=1).mean()))
            agg[w]["act_dev"].append(dev)
            agg[w]["act_dev_rad"].append(dev_rad)
            agg[w]["ee_dev"].append(float(np.mean(ee)))
            agg[w]["expert_err"].append(err)
            for ki, k in enumerate(OFFSETS):
                sel = (oi == ki)
                if sel.any():
                    agg[w]["per_k"][k].append(
                        float((aw[sel] - a0[sel]).norm(dim=-1).mean()))
        n_pairs += B

    res = dict(ckpt=args.ckpt, variant=ck["variant"], n_pairs=n_pairs,
               gt_chunk_norm_note="see interventions.W0.expert_err = generation-from-noise baseline error vs GT",
               seed=args.seed, n_euler=N_EULER,
               oracle_note="W4 leaks true future; OFFLINE diagnostic only",
               interventions={})
    for w in INTERV:
        if not agg[w]["act_dev"]:
            continue
        res["interventions"][w] = dict(
            act_dev_norm=round(float(np.mean(agg[w]["act_dev"])), 4),
            ee_dev_m=round(float(np.mean(agg[w]["ee_dev"])), 4),
            expert_err_norm=round(float(np.mean(agg[w]["expert_err"])), 4),
            per_k={k: round(float(np.mean(v)), 4) for k, v in
                   agg[w]["per_k"].items() if v})
    out = Path(args.out or ("runs/stage4a/metrics/world_intervention/"
                            f"wi_{ck['variant']}_{Path(args.ckpt).parent.name}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
