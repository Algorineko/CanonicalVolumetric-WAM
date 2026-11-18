"""Attach original-episode action chunks to replay episode frames (eval needs
actions; the replay was DRIVEN by those same recorded joint paths, so the
actions are the replay's own commands too)."""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cvwam import robotwin_adapter as rta  # noqa: E402
from cvwam import geometry as geo  # noqa: E402
from cvwam.geometry import T_base_world  # noqa: E402


def main():
    cfg = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    sm = json.loads(Path("runs/stage3b/split_manifest.json").read_text())
    root = Path(cfg["cache_dir"]) / "voxels_ep_replay"
    for spec in sm["splits"]["clean_train"][:1]:  # replay ep10 = clean:10
        pass
    variant, ep = "demo_clean_balanced", 10
    rep_dir = root / f"{variant}_{ep}_replay" / "20mm"
    mf = json.loads((rep_dir / "manifest.json").read_text())
    src = rta.locate_episode(cfg["data_root"], variant, ep)
    T_bw = T_base_world(rta.T_world_base_from_config(cfg))
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
        np.savez_compressed(fr["npz"], **{k: d[k] for k in d.files},
                            action_chunk=chunk, action_valid=len(chunk))
        n_attached += 1
    print(f"attached actions to {n_attached}/{len(mf['frames'])} frames")


if __name__ == "__main__":
    main()
