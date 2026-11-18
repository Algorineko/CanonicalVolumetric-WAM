"""RoboTwin open_microwave native-data adapter for CV-WAM Stage 1.

Reads /data/openmw_balanced_work native collection HDF5 + ffv1 depth videos and
produces canonical samples per docs/plan.md §5.3. All conversions are verified
against anchors in configs/smoke.yaml; any mismatch raises instead of passing.
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import av
import h5py
import numpy as np

import cvwam.geometry as geo

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_NAMES = ("head", "left_wrist", "right_wrist")


def _load_decode_image_bit():
    """Import the official RoboTwin/XPolicyLab JPEG-bitstream decoder (read-only
    reuse of /home/zhangjialiang/code/RoboTwin/data/decode_image_bit.py)."""
    path = Path("/home/zhangjialiang/code/RoboTwin/data/decode_image_bit.py")
    spec = importlib.util.spec_from_file_location("robottwin_decode_image_bit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


decode_image_bit = _load_decode_image_bit().decode_image_bit


@dataclass
class EpisodePaths:
    hdf5: Path
    depth_ffv1: dict   # cam name -> mkv path
    depth_x264rgb: dict


def locate_episode(data_root: str, variant: str, episode_no: int) -> EpisodePaths:
    root = Path(data_root) / variant / "open_microwave" / "aloha_agilex"
    hdf5 = root / "data" / f"episode_{episode_no:07d}.hdf5"
    if not hdf5.exists():
        raise FileNotFoundError(f"episode hdf5 not found: {hdf5}")
    ep = root / "depth_videos" / f"episode_{episode_no:07d}"
    return EpisodePaths(
        hdf5=hdf5,
        depth_ffv1={c: ep / "ffv1" / f"{c}.mkv" for c in CAMERAS},
        depth_x264rgb={c: ep / "x264rgb" / f"{c}.mp4" for c in CAMERAS},
    )


def count_ffv1_frames(path) -> int:
    """ffv1 mkv containers carry no frame-count metadata; count by decoding."""
    count = 0
    with av.open(str(path)) as c:
        for _ in c.decode(video=0):
            count += 1
    return count


def decode_ffv1_depth_at(path, t: int) -> np.ndarray:
    """Decode one ffv1 gray16le frame -> uint16 millimeter depth map."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(video=0)):
            if i == t:
                arr = frame.to_ndarray()   # gray16le -> uint16
                if arr.dtype != np.uint16:
                    arr = arr.astype(np.uint16)
                return arr
    raise IndexError(f"frame {t} out of range in {path}")


def decode_ffv1_depth_window(path, t_start: int, n: int) -> np.ndarray:
    """Decode n consecutive frames starting at t_start -> (n,H,W) uint16."""
    out = []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(video=0)):
            if i >= t_start and i < t_start + n:
                arr = frame.to_ndarray()
                out.append(arr if arr.dtype == np.uint16 else arr.astype(np.uint16))
                if len(out) == n:
                    break
    if len(out) != n:
        raise IndexError(f"requested [{t_start},{t_start + n}) beyond stream {path}")
    return np.stack(out)


def decode_x264rgb_depth_at(path, t: int) -> np.ndarray:
    """Decode one x264rgb qp0 frame -> uint16 mm via (R<<8)|G (B carries nothing)."""
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(video=0)):
            if i == t:
                rgb = frame.to_ndarray()      # (H,W,3) uint8
                return (rgb[..., 0].astype(np.uint16) << 8) | rgb[..., 1].astype(np.uint16)
    raise IndexError(f"frame {t} out of range in {path}")


def T_world_base_from_config(cfg: dict) -> np.ndarray:
    rb = cfg["robot_base"]
    return geo.T_world_base_from_pose(rb["pos_m"], rb["quat_wxyz"])


def load_canonical_sample(paths: EpisodePaths, cfg: dict, t: int,
                          T_base_world_: np.ndarray) -> dict:
    """Build one canonical sample (plan.md §5.3 contract) at timestep t."""
    t = int(t)
    with h5py.File(paths.hdf5, "r") as f:
        T_total = f["state/left_arm_joint_states"].shape[0]
        if not (0 <= t < T_total):
            raise IndexError(f"t={t} outside episode length {T_total}")
        if t != 0:
            # extrinsics/intrinsics are per-frame; assert the wrist camera actually
            # moves between frames (guards against a frozen extrinsics bug).
            e0 = f["vision/cam_left_wrist/extrinsics_matrix"][0]
            et = f["vision/cam_left_wrist/extrinsics_matrix"][t]
            if np.allclose(e0, et, atol=1e-6):
                raise RuntimeError(
                    f"wrist extrinsics unchanged between frame 0 and {t}: "
                    f"adapter assumption violated ({paths.hdf5.name})")

        rgb_v, depth_m_v, depth_valid_v, K_v, T_base_cam_v = [], [], [], [], []
        for cam in CAMERAS:
            g = f[f"vision/{cam}"]
            K = np.asarray(g["intrinsic_matrix"][t], dtype=np.float64)
            T_world_cam_cv = geo.cam2world_gl_to_world_cam_cv(
                np.asarray(g["extrinsics_matrix"][t], dtype=np.float64))
            T_base_cam = T_base_world_ @ T_world_cam_cv
            depth_mm = decode_ffv1_depth_at(paths.depth_ffv1[cam], t).astype(np.float32)
            valid = depth_mm > 0
            depth_m = np.where(valid, depth_mm / 1000.0, 0.0).astype(np.float32)
            rgb = np.asarray(decode_image_bit(g["colors"][t]), dtype=np.uint8)
            shape = np.asarray(g["shape"])
            if rgb.shape != (shape[0], shape[1], 3) or depth_mm.shape != (shape[0], shape[1]):
                raise RuntimeError(f"shape mismatch for {cam}: rgb {rgb.shape} "
                                   f"depth {depth_mm.shape} shape {shape}")
            rgb_v.append(rgb)
            depth_m_v.append(depth_m)
            depth_valid_v.append(valid)
            K_v.append(K)
            T_base_cam_v.append(T_base_cam)

        state = {
            "left_arm_joint_states": f["state/left_arm_joint_states"][t],
            "left_ee_joint_states": f["state/left_ee_joint_states"][t],
            "right_arm_joint_states": f["state/right_arm_joint_states"][t],
            "right_ee_joint_states": f["state/right_ee_joint_states"][t],
            "left_ee_poses": f["state/left_ee_poses"][t],
            "right_ee_poses": f["state/right_ee_poses"][t],
        }
        action = {
            "left_arm_joint_states": f["action/left_arm_joint_states"][t],
            "left_ee_joint_states": f["action/left_ee_joint_states"][t],
            "right_arm_joint_states": f["action/right_arm_joint_states"][t],
            "right_ee_joint_states": f["action/right_ee_joint_states"][t],
        }
        instr = json.loads(f["instructions"][()].decode())  # 100 paraphrases
        freq = int(f["additional_info/frequency"][()])

    sample = {
        "episode_id": paths.hdf5.stem,
        "task_id": "open_microwave",
        "t": t,
        "T": T_total,
        "frequency_hz": freq,
        "rgb": np.stack(rgb_v),                       # [3,H,W,3] uint8 RGB
        "depth_m": np.stack(depth_m_v),               # [3,H,W] float32, meters, z-depth
        "depth_valid": np.stack(depth_valid_v),       # [3,H,W] bool
        "K": np.stack(K_v),                           # [3,3,3] float64
        "T_base_cam": np.stack(T_base_cam_v),         # [3,4,4] float64 (OpenCV cam axes)
        "qpos": np.concatenate([
            state["left_arm_joint_states"], state["right_arm_joint_states"],
            state["left_ee_joint_states"], state["right_ee_joint_states"],
        ]),                                           # 14-dim: L arm(6) R arm(6) L grip R grip
        "state_parts": state,
        "action_parts": action,
        "language": instr[0],
        "instruction_id": 0,
    }
    _verify_anchors(sample, cfg)
    return sample


def _verify_anchors(sample: dict, cfg: dict) -> None:
    """Numeric anchors; a failure here means the adapter, not the data, is wrong."""
    rb = cfg["robot_base"]
    # 1) static head camera center must match the embodiment config anchor
    c = geo.camera_center_in_base(sample["T_base_cam"][0])
    anchor = np.asarray(rb["head_cam_base_anchor_m"], dtype=np.float64)
    err = np.linalg.norm(c - anchor)
    if err > rb["anchor_tol_m"]:
        raise RuntimeError(f"head camera center {c} != anchor {anchor} (err {err:.6f} m)")
    # 2) K must equal the D435 fovy-37 320x240 intrinsics
    K0 = sample["K"][0]
    expected = np.array([[358.6421813964844, 0, 160], [0, 358.6421813964844, 120],
                         [0, 0, 1]])
    if not np.allclose(K0, expected, atol=1e-4):
        raise RuntimeError(f"K mismatch: got {K0}")
    # 3) depth must be metric-plausible (no mm-vs-m confusion): head camera sits
    #    1.35 m above the table plane (table z=0.7401 world/base).
    d0 = sample["depth_m"][0]
    med = float(np.median(d0[d0 > 0]))
    if not (0.2 < med < 3.0):
        raise RuntimeError(f"head depth median {med:.3f} m implausible; unit bug?")


def build_index(cfg: dict):
    """Iterate over (variant, episode_no) pairs declared in the smoke config."""
    for variant, eps in cfg["episodes"].items():
        for ep in eps:
            yield variant, ep, locate_episode(cfg["data_root"], variant, ep)


def load_episode_index(paths: EpisodePaths) -> dict:
    with h5py.File(paths.hdf5, "r") as f:
        T = int(f["state/left_arm_joint_states"].shape[0])
        freq = int(f["additional_info/frequency"][()])
        instr = json.loads(f["instructions"][()].decode())[0]
    return {"T": T, "frequency_hz": freq, "language": instr,
            "hdf5": str(paths.hdf5)}
