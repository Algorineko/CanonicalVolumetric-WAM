"""Geometry utilities for CV-WAM Stage 1: camera conventions, depth unprojection
into the robot-base frame, and PLY IO.

Conventions (verified against RoboTwin 2.0 data on 2026-11-18):
- OpenCV camera axes: x right, y down, z forward (optical axis).
- RoboTwin HDF5 `extrinsics_matrix` stores cam2world_gl: camera -> world with
  OpenGL axes (x right, y up, z backward).
- Relation: T_world_cam_cv = cam2world_gl @ diag(1,-1,-1), hence
  extrinsic_cv (world->cam_cv) = diag(1,-1,-1) @ inv(cam2world_gl).
- Robot base frame: SAPIEN world frame transformed by inv(T_world_base), where
  T_world_base comes from aloha-agilex `robot_pose` = [x,y,z, qw,qx,qy,qz]
  (envs/robot/robot.py:63-64, sapien.Pose(p, q[-4:]) with wxyz order).
"""

import numpy as np

GL_TO_CV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                    dtype=np.float64)


def make_tf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform from rotation R (3x3) and t (3,)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quat_wxyz_to_R(q_wxyz) -> np.ndarray:
    """Rotation matrix from a unit quaternion in [w, x, y, z] order."""
    w, x, y, z = [float(v) for v in q_wxyz]
    n = w * w + x * x + y * y + z * z
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def T_world_base_from_pose(pos, quat_wxyz) -> np.ndarray:
    """T_world_base from RoboTwin robot_pose (position + wxyz quaternion)."""
    return make_tf(quat_wxyz_to_R(quat_wxyz), pos)


def T_base_world(T_world_base: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid 4x4 transform."""
    R = T_world_base[:3, :3]
    t = T_world_base[:3, 3]
    return make_tf(R.T, -R.T @ t)


def cam2world_gl_to_world_cam_cv(cam2world_gl: np.ndarray) -> np.ndarray:
    """cam2world_gl (cam_gl -> world) -> T_world_cam_cv (cam_cv -> world).

    p_world = R_gl @ p_cam_gl + t with p_cam_gl = diag(1,-1,-1) @ p_cam_cv.
    """
    T = np.asarray(cam2world_gl, dtype=np.float64).copy()
    R = T[:3, :3] @ GL_TO_CV[:3, :3]
    t = T[:3, 3]
    return make_tf(R, t)


def T_base_cam_cv_from_cam2world_gl(cam2world_gl: np.ndarray,
                                    T_base_world_: np.ndarray) -> np.ndarray:
    """Full chain to the robot-base frame (OpenCV camera axes)."""
    return T_base_world_ @ cam2world_gl_to_world_cam_cv(cam2world_gl)


def depth_to_base(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                  valid: np.ndarray | None = None, dtype=np.float32):
    """Unproject an optical-axis depth map into the robot-base frame.

    Reference implementation from docs/plan.md §5.4. K is the (3,3) OpenCV
    intrinsic at the actual image resolution; T_base_cam maps camera-frame
    points (OpenCV axes) into the base frame. Computation stays in float64;
    only the output is cast (float32 per the reference, or float64 for strict
    round-trip validation).

    Returns (p_base [N,3] dtype, mask [H,W] bool).
    """
    depth_m = np.asarray(depth_m, dtype=np.float64)
    h, w = depth_m.shape
    v, u = np.indices((h, w), dtype=np.float64)
    mask = np.isfinite(depth_m) & (depth_m > 0)
    if valid is not None:
        mask &= np.asarray(valid).astype(bool)
    uv1 = np.stack([u[mask], v[mask], np.ones(mask.sum())], axis=1)
    p_cam = (uv1 @ np.linalg.inv(np.asarray(K, dtype=np.float64)).T) * depth_m[mask, None]
    T = np.asarray(T_base_cam, dtype=np.float64)
    p_base = p_cam @ T[:3, :3].T + T[:3, 3]
    return p_base.astype(dtype), mask


def base_to_cam(p_base: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """base-frame points -> camera-frame points (OpenCV axes), float64."""
    p_base = np.asarray(p_base, dtype=np.float64)
    T = np.asarray(T_base_cam, dtype=np.float64)
    return (p_base - T[:3, 3]) @ T[:3, :3]


def project_base_to_px(p_base: np.ndarray, K: np.ndarray,
                       T_base_cam: np.ndarray) -> np.ndarray:
    """Project base-frame points into pixel coordinates [N,2] (u,v)."""
    p_cam = base_to_cam(p_base, T_base_cam)
    px = (np.asarray(K, dtype=np.float64) @ p_cam.T).T
    return (px[:, :2] / px[:, 2:3])[:, :2]


def camera_center_in_base(T_base_cam: np.ndarray) -> np.ndarray:
    """Camera optical center expressed in the base frame.

    T_base_cam maps camera -> base (p_base = R p_cam + t), so the camera
    center (p_cam = 0) is simply the translation column t. (-R^T t would be
    the base-frame origin expressed in the camera frame.)
    """
    T = np.asarray(T_base_cam, dtype=np.float64)
    return T[:3, 3].copy()


def camera_axes_ply_axes(T_base_cam: np.ndarray, length: float = 0.05):
    """Axis segment endpoints [3,2,3] in base frame: origin->+x, origin->+y, origin->+z_cam."""
    T = np.asarray(T_base_cam, dtype=np.float64)
    c = camera_center_in_base(T)
    R = T[:3, :3]
    return np.stack([c, c + R @ np.array([length, 0, 0]),
                     c, c + R @ np.array([0, length, 0]),
                     c, c + R @ np.array([0, 0, length])], axis=0).reshape(3, 2, 3)


def write_ply(path, points: np.ndarray, colors: np.ndarray | None = None,
              extra: dict | None = None) -> None:
    """Write an ASCII PLY (per-vertex xyz [+ rgb u8] [+ extra float channels])."""
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    has_color = colors is not None
    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if has_color:
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    if extra:
        for name in extra:
            lines += [f"property float {name}"]
    lines += ["end_header"]
    header = "\n".join(lines) + "\n"
    cols = [points]
    if has_color:
        cols.append(np.clip(np.asarray(colors, dtype=np.int64), 0, 255).astype(np.int64))
    if extra:
        for v in extra.values():
            cols.append(np.asarray(v, dtype=np.float32))
    body = np.concatenate([c.reshape(n, -1) for c in cols], axis=1)
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, body, fmt=" ".join(
            ["%.4f"] * 3 + (["%d"] * 3 if has_color else []) +
            (["%.3f"] * len(extra) if extra else [])))
