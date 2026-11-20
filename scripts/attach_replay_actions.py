"""Attach original-episode action chunks to replay/probe episode frames (eval
needs actions; the replay was DRIVEN by those same recorded joint paths, so
the actions are the replay's own commands too).

Usage: attach_replay_actions.py --variant demo_clean_balanced --episode 10
                                --tag replay            (-> ..._replay)
                                --tag probe_camdis05    (-> ..._probe_camdis05)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="demo_clean_balanced")
    ap.add_argument("--episode", type=int, default=10)
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--config", default="configs/smoke.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep_replay"
    rep_dir = root / f"{args.variant}_{args.episode}_{args.tag}" / "20mm"
    mf = json.loads((rep_dir / "manifest.json").read_text())
    src = rta.locate_episode(cfg["data_root"], args.variant, args.episode,
                             task=cfg.get("task", "open_microwave"))
    with __import__("h5py").File(src.hdf5, "r") as f:
        act_l = np.asarray(f["action/left_arm_joint_states"], dtype=np.float32)
        act_r = np.asarray(f["action/right_arm_joint_states"], dtype=np.float32)
        gl = np.asarray(f["action/left_ee_joint_states"], dtype=np.float32)
        gr = np.asarray(f["action/right_ee_joint_states"], dtype=np.float32)
        T_total = int(act_l.shape[0])
    n_attached = 0
    for fr in mf["frames"]:
        t = fr["t"]
        if t >= T_total:
            break
        act_end = min(t + 16, T_total)
        chunk = np.concatenate([act_l[t:act_end], act_r[t:act_end],
                                gl[t:act_end], gr[t:act_end]], axis=1)
        d = np.load(fr["npz"])
        if "action_chunk" in d.files:
            n_attached += 1  # idempotent: already attached
            continue
        np.savez_compressed(fr["npz"], **{k: d[k] for k in d.files},
                            action_chunk=chunk, action_valid=len(chunk))
        n_attached += 1
    print(f"attached actions to {n_attached}/{len(mf['frames'])} frames "
          f"({args.variant}_{args.episode}_{args.tag})")


if __name__ == "__main__":
    main()
