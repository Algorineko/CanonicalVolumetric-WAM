"""OmniVGGT (VGGT-derivative, CVPR 2026) RGB-only smoke on the same frames.

Runs under PhysAdaptWAM/.venv-wam python (proven dependency stack). Loading
recipe replicated from PhysAdaptWAM/archive/20261113/scripts/
audit_spatial_teacher.py: local DINOv2 architecture-only init + strict
safetensors load. Clearly labeled as a derivative variant, NOT the official
VGGT-1B baseline.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VIEW_FILES = {"head": "_head.png", "left_wrist": "_left_wrist.png",
              "right_wrist": "_right_wrist.png"}


def frame_png(rgb_dir: Path, t: int, view: str) -> str:
    return str(rgb_dir / f"frame_{t:07d}{VIEW_FILES[view]}")


def load_model(cfg):
    upstream = cfg["omnivggt"]["code"]
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    from omnivggt.models.omnivggt import OmniVGGT
    from safetensors.torch import load_file

    weights = Path(cfg["omnivggt"]["weights"])
    verified = json.loads(weights.with_name("verified.json").read_text())
    assert verified["bytes"] == weights.stat().st_size, "weight size mismatch"

    original_hub_load = torch.hub.load

    def architecture_only(repo, name, *args, **kwargs):
        assert repo == "facebookresearch/dinov2" and name == "dinov2_vitl14_reg"
        return original_hub_load(
            "/home/zhangjialiang/.cache/torch/hub/facebookresearch_dinov2_main",
            name, source="local", pretrained=False)

    torch.hub.load = architecture_only
    t0 = time.perf_counter()
    try:
        # RGB-only configuration: drop_prob=1.0 guarantees the aggregator
        # selects NO GT camera poses and NO GT sparse depth (deterministic),
        # so the model runs purely from images.
        model = OmniVGGT(cam_drop_prob=1.0, depth_drop_prob=1.0)
    finally:
        torch.hub.load = original_hub_load
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval().requires_grad_(False).cuda()
    model.load_time_s = time.perf_counter() - t0
    return model


def run_once(model, images, dtype):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    B, S = 1, images.shape[0]
    zero_depth = torch.zeros(B, S, *images.shape[-2:], 1, device=images.device)
    zero_mask = torch.zeros(B, S, *images.shape[-2:], device=images.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        pred = model(images, depth=zero_depth, mask=zero_mask)
    torch.cuda.synchronize()
    lat = time.perf_counter() - t0
    mem = dict(peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return pred, lat, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--view-configs", nargs="+", default=["head", "head+left_wrist+right_wrist"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    frames_root = Path(cfg["cache_dir"]) / "frames"
    out_root = Path(cfg["heavy_root"]) / "cache" / "vggt_omnivggt"
    out_root.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    _code = cfg["omnivggt"]["code"]
    if str(_code) not in sys.path:
        sys.path.insert(0, str(_code))
    from omnivggt.utils.load_fn import load_and_preprocess_images

    model = load_model(cfg)
    print(f"OmniVGGT loaded in {model.load_time_s:.1f}s (weights {cfg['omnivggt']['weights']})")

    warmed = False
    for sid in args.frames:
        variant, ep, t = sid.split("/")
        t = int(t.split("=")[1])
        rgb_dir = frames_root / variant / ep / "rgb"
        for vc in args.view_configs:
            views = vc.split("+")
            paths = [frame_png(rgb_dir, t, v) for v in views]
            images = load_and_preprocess_images(paths, mode="crop").cuda()
            if not warmed:
                run_once(model, images, dtype)
                warmed = True
            reps = []
            for _ in range(args.repeats):
                pred, lat, mem = run_once(model, images, dtype)
                reps.append((lat, mem))
            lat = float(np.median([r[0] for r in reps]))
            mem = reps[-1][1]

            def npf(x):
                return x.detach().float().cpu().numpy()

            depth = npf(pred["depth"]).squeeze(0)
            dconf = npf(pred["depth_conf"]).squeeze(0)
            world_points = npf(pred["world_points"]).squeeze(0)
            wconf = npf(pred["world_points_conf"]).squeeze(0)
            pre_shape = tuple(images.shape[-2:])
            # pose via camera head (same encoding as official)
            from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
            ext, K = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
            extrinsic = npf(ext).squeeze(0)
            intrinsic = npf(K).squeeze(0)

            run_dir = out_root / variant / ep / f"t={t}" / vc.replace("+", "_")
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "geometry.npz",
                                depth=depth, depth_conf=dconf, extrinsic=extrinsic,
                                intrinsic=intrinsic, world_points=world_points,
                                world_points_conf=wconf, pre_shape=np.asarray(pre_shape))
            meta = dict(sample_id=sid, view_config=vc, views=views,
                        image_pre_shape=[int(x) for x in pre_shape],
                        depth_shape=list(depth.shape), points_shape=list(world_points.shape),
                        finite_points=int(np.isfinite(world_points).all(-1).sum()),
                        total_px=int(np.prod(depth.shape[:3])),
                        latency_s_median=round(lat, 3),
                        latency_all_s=[round(r[0], 3) for r in reps],
                        model_load_s=round(model.load_time_s, 2),
                        **{k: round(v, 3) for k, v in mem.items()},
                        torch=torch.__version__, dtype=str(dtype),
                        note="OmniVGGT derivative variant, not official VGGT-1B")
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(json.dumps(meta))
            del pred
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
