"""Export synchronized multi-view RGB-D samples from RoboTwin native data.

Per docs/plan.md Stage 1: N episodes x 5 timesteps, 3 cameras (head + both
wrists), canonical sample contract (§5.3). Verifies ffv1 depth stream length
against the HDF5 frame count and cross-checks the x264rgb backup codec on one
frame. Outputs NPZ + PNG under <heavy_root>/cache/frames/ and a manifest.jsonl.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam.geometry import T_base_world  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--out", default=None, help="override cache dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(args.out or cfg["cache_dir"]) / "frames"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    report = []

    for variant, eps in cfg["episodes"].items():
        for ep_no in eps:
            paths = rta.locate_episode(cfg["data_root"], variant, ep_no,
                                       task=cfg.get("task", "open_microwave"))
            t0 = time.perf_counter()
            idx = rta.load_episode_index(paths)
            T = idx["T"]
            timesteps = sorted({0, T // 4, T // 2, (3 * T) // 4, T - 2})
            print(f"[{variant} ep{ep_no:07d}] T={T} freq={idx['frequency_hz']}Hz "
                  f"timesteps={timesteps}", flush=True)

            # full-stream length check (ffv1 containers carry no count metadata)
            counts = {cam: rta.count_ffv1_frames(p) for cam, p in paths.depth_ffv1.items()}
            for cam, n in counts.items():
                if n != T:
                    raise RuntimeError(f"ffv1 {cam} has {n} frames != HDF5 T={T} "
                                       f"({paths.hdf5})")
            print(f"  ffv1 frame counts == T: {counts}")

            ep_dir = out_root / variant / f"episode_{ep_no:07d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            T_base_world_ = T_base_world(rta.T_world_base_from_config(cfg))
            for t in timesteps:
                sample = rta.load_canonical_sample(paths, cfg, t, T_base_world_)
                npz = ep_dir / f"frame_{t:07d}.npz"
                rgb = sample["rgb"]
                state_parts = sample.pop("state_parts")
                action_parts = sample.pop("action_parts")
                np.savez_compressed(
                    npz,
                    **{k: v for k, v in sample.items()
                       if isinstance(v, (np.ndarray, int, str, float))},
                    state_left_arm=state_parts["left_arm_joint_states"],
                    state_right_arm=state_parts["right_arm_joint_states"],
                    state_left_grip=state_parts["left_ee_joint_states"],
                    state_right_grip=state_parts["right_ee_joint_states"],
                    action_left_arm=action_parts["left_arm_joint_states"],
                    action_right_arm=action_parts["right_arm_joint_states"],
                    action_left_grip=action_parts["left_ee_joint_states"],
                    action_right_grip=action_parts["right_ee_joint_states"],
                    state_left_ee_pose=state_parts["left_ee_poses"],
                    state_right_ee_pose=state_parts["right_ee_poses"],
                )
                for vi, cam in enumerate(rta.STATE_NAMES):
                    rgb_dir = ep_dir / "rgb"
                    rgb_dir.mkdir(exist_ok=True)
                    from PIL import Image
                    Image.fromarray(rgb[vi]).save(
                        rgb_dir / f"frame_{t:07d}_{cam}.png")
                report.append({
                    "sample_id": f"{variant}/episode_{ep_no:07d}/t={t}",
                    "variant": variant, "episode": ep_no, "t": t, "T": T,
                    "npz": str(npz), "language": sample["language"],
                    "qpos": sample["qpos"].tolist(),
                    "depth_valid_frac": sample["depth_valid"].mean(axis=(1, 2)).round(4).tolist(),
                    "hit_candidates_head": int((sample["depth_m"][0] > 0).sum()),
                })
                print(f"  t={t:3d} saved {npz.name} "
                      f"depth_valid={[round(v, 3) for v in report[-1]['depth_valid_frac']]}")

            # one-frame x264rgb vs ffv1 cross-check on this episode
            t_chk = timesteps[len(timesteps) // 2]
            for cam in rta.CAMERAS:
                a = rta.decode_ffv1_depth_at(paths.depth_ffv1[cam], t_chk)
                b = rta.decode_x264rgb_depth_at(paths.depth_x264rgb[cam], t_chk)
                diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
                frac_eq = float((diff == 0).mean())
                if frac_eq < 0.999:
                    raise RuntimeError(f"x264rgb/ffv1 mismatch {variant} ep{ep_no} "
                                       f"{cam} t={t_chk}: equal frac {frac_eq:.6f} "
                                       f"maxdiff {diff.max()}")
                print(f"  codec cross-check {cam} t={t_chk}: identical ({frac_eq:.6f})")

            report[-len(timesteps):]  # keep order
            print(f"  episode done in {time.perf_counter() - t0:.1f}s")

    with open(manifest_path, "w") as f:
        for row in report:
            f.write(json.dumps(row) + "\n")
    print(f"manifest -> {manifest_path} ({len(report)} samples)")


if __name__ == "__main__":
    main()
