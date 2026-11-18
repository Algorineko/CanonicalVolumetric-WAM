"""Official VGGT-1B reconstruction smoke test on RoboTwin frames (plan.md §6.2).

Runs G1 (single head view) / G2 (three synchronized views) on exported frames,
saves raw geometry per run, and measures latency + peak VRAM (warmup first).
Alignment and GT comparison are done by eval_geometry.py, not here.
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
VIEW_ORDER = ["head", "left_wrist", "right_wrist"]


def frame_png(rgb_dir: Path, t: int, view: str) -> str:
    return str(rgb_dir / f"frame_{t:07d}{VIEW_FILES[view]}")


def load_model(weights: Path, device):
    """Load official VGGT-1B. The HF repo stores model.pt (plain state_dict),
    which PyTorchModelHubMixin.from_pretrained does not resolve (it expects
    model.safetensors), so load the state dict directly."""
    from vggt.models.vggt import VGGT
    t0 = time.perf_counter()
    model = VGGT()
    sd = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model = model.eval().to(device)
    model.load_time_s = time.perf_counter() - t0
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_once(model, images, device, dtype):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        pred = model(images)
        ext, K = None, None
        if "pose_enc" in pred:
            ext, K = {}, {}
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            e, k = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
            ext, K = e, k
    torch.cuda.synchronize()
    lat = time.perf_counter() - t0
    mem = dict(
        peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    return pred, ext, K, lat, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--frames", nargs="+", required=True,
                    help="sample ids like demo_clean_balanced/episode_0000010/t=212")
    ap.add_argument("--view-configs", nargs="+", default=["head", "head+left_wrist+right_wrist"],
                    help="e.g. head, head+left_wrist, head+left_wrist+right_wrist")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    _repo = cfg["vggt"]["official_repo"]
    if str(_repo) not in sys.path:
        sys.path.insert(0, str(_repo))
    frames_root = Path(cfg["cache_dir"]) / "frames"
    out_root = Path(cfg["heavy_root"]) / "cache" / "vggt_official"
    out_root.mkdir(parents=True, exist_ok=True)
    device = cfg["vggt"]["device"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    import vggt  # noqa: F401  (adds package to path via third_party checkout)
    from vggt.utils.load_fn import load_and_preprocess_images

    model = load_model(Path(cfg["vggt"]["weights"]), device)
    print(f"model loaded in {model.load_time_s:.1f}s; device={device}")

    warmed = False
    for sid in args.frames:
        variant, ep, t = sid.split("/")
        t = int(t.split("=")[1])
        rgb_dir = frames_root / variant / ep / "rgb"
        for vc in args.view_configs:
            views = vc.split("+")
            paths = [frame_png(rgb_dir, t, v) for v in views]
            images = load_and_preprocess_images(paths, mode="crop").to(device)
            if not warmed:  # cudnn/autotune warmup, untimed
                run_once(model, images, device, dtype)
                warmed = True
            reps = []
            for _ in range(args.repeats):
                pred, ext, K, lat, mem = run_once(model, images, device, dtype)
                reps.append((lat, mem))
            lat = float(np.median([r[0] for r in reps]))
            mem = reps[-1][1]

            def npf(x):
                return x.detach().float().cpu().numpy()

            depth = npf(pred["depth"]).squeeze(0)          # [S,H,W,1]
            dconf = npf(pred["depth_conf"]).squeeze(0)     # [S,H,W]
            extrinsic = npf(ext).squeeze(0)                # camera-from-VGGT-world (S,3,4)
            intrinsic = npf(K).squeeze(0)
            world_points = npf(pred["world_points"]).squeeze(0)  # [S,H,W,3]
            wconf = npf(pred["world_points_conf"]).squeeze(0) if "world_points_conf" in pred else None
            pre_shape = tuple(images.shape[-2:])

            run_dir = out_root / variant / ep / f"t={t}" / vc.replace("+", "_")
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run_dir / "geometry.npz",
                                depth=depth, depth_conf=dconf, extrinsic=extrinsic,
                                intrinsic=intrinsic, world_points=world_points,
                                world_points_conf=wconf, pre_shape=np.asarray(pre_shape))
            finite = int(np.isfinite(world_points).all(-1).sum())
            meta = dict(sample_id=sid, view_config=vc, views=views,
                        input_paths=paths, image_pre_shape=[int(x) for x in pre_shape],
                        depth_shape=list(depth.shape), points_shape=list(world_points.shape),
                        finite_points=finite, total_px=int(np.prod(depth.shape[:3])),
                        latency_s_median=round(lat, 3),
                        latency_all_s=[round(r[0], 3) for r in reps],
                        model_load_s=round(model.load_time_s, 2),
                        **{k: round(v, 3) for k, v in mem.items()},
                        torch=torch.__version__, dtype=str(dtype))
            (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
            print(json.dumps(meta))
            # free VRAM between runs
            del pred
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
