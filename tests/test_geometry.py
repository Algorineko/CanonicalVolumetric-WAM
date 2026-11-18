"""Stage 1 geometry unit tests (docs/plan.md §5.4 / Gate G1).

Run:  <openwam python> tests/test_geometry.py   (also pytest-compatible)
All errors must stay below 1e-5 m (floating-point level).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cvwam.geometry as geo  # noqa: E402

TOL_M = 1e-5
FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append((name, detail))


def test_project_unproject(depth, T_base_cam, K, name):
    """Constant-depth plane at every integer pixel; unproject -> project back
    must reproduce (u, v) and the original optical-axis depth exactly."""
    h, w = depth.shape
    p_base, mask = geo.depth_to_base(depth, K, T_base_cam, dtype=np.float64)
    check(f"[{name}] unproject count", mask.sum() == h * w, f"{mask.sum()}/{h * w}")
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    proj = geo.project_base_to_px(p_base, K, T_base_cam)
    err_px = np.abs(proj - np.stack([u.ravel(), v.ravel()], axis=1)).max()
    check(f"[{name}] project(unproject(p)) == (u,v)", err_px < 1e-6,
          f"max px err {err_px:.2e}")
    z_rt = geo.base_to_cam(p_base, T_base_cam)[:, 2]
    err_m = np.abs(z_rt - depth.ravel()).max()
    check(f"[{name}] z-depth roundtrip < 1e-5 m", err_m < TOL_M, f"{err_m:.2e} m")
    return p_base


def test_nontrivial_poses():
    from scipy.spatial.transform import Rotation

    K = np.array([[358.6421813964844, 0, 160], [0, 358.6421813964844, 120], [0, 0, 1]])
    h, w = 240, 320
    test_project_unproject(np.full((h, w), 0.85), _cam_Rz90(), K, "head cam pose")
    # non-trivial rotation (xyz Euler 30/20/10 deg) + translation
    q_xyzw = Rotation.from_euler("xyz", [30, 20, 10], degrees=True).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    test_project_unproject(np.full((h, w), 1.2),
                           geo.make_tf(geo.quat_wxyz_to_R(q_wxyz), [0.4, 0.2, 1.0]),
                           K, "nontrivial R")


def _cam_Rz90():
    T_world_base = geo.T_world_base_from_pose(
        [0, -0.65, 0], [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    # head camera from embodiment vectors (see test_camera_construction)
    cam2world_gl = _head_cam2world_gl()
    return geo.T_base_cam_cv_from_cam2world_gl(cam2world_gl, geo.T_base_world(T_world_base))


def _head_cam2world_gl():
    """Build cam2world_gl for the aloha head camera from embodiment vectors.

    config.yml head_camera: position [-0.032,-0.45,1.35], forward [0,0.6,-0.8],
    left [-1,0,0]. OpenGL camera axes: +x right, +y up, +z backward.
    """
    pos = np.array([-0.032, -0.45, 1.35])
    right = -np.array([-1.0, 0, 0])          # config 'left' = camera -x
    forward = np.array([0, 0.6, -0.8])
    z_gl = -forward
    up = np.cross(z_gl, right)               # y = z x x (right-handed)
    R = np.column_stack([right, up, z_gl])
    return geo.make_tf(R, pos)


def test_camera_construction():
    # Verified ground truth: extrinsics_matrix[0] of cam_head in
    # /data/openmw_balanced_work/demo_clean_balanced/.../episode_0000010.hdf5
    expected = np.array([[1, 0, 0, -0.032],
                         [0, 0.8, -0.6, -0.45],
                         [0, 0.6, 0.8, 1.35],
                         [0, 0, 0, 1]], dtype=np.float64)
    err = np.abs(_head_cam2world_gl() - expected).max()
    check("cam2world_gl from config vectors == HDF5-verified matrix", err < 1e-9,
          f"{err:.2e}")

    T_world_base = geo.T_world_base_from_pose(
        [0, -0.65, 0], [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2])
    T_bw = geo.T_base_world(T_world_base)
    check("T_base_world @ T_world_base == I",
          np.allclose(T_bw @ T_world_base, np.eye(4), atol=1e-12))

    T_base_cam = geo.T_base_cam_cv_from_cam2world_gl(_head_cam2world_gl(), T_bw)
    c = geo.camera_center_in_base(T_base_cam)
    err = np.linalg.norm(c - np.array([0.2, 0.032, 1.35]))
    check("head cam base-frame anchor [0.2,0.032,1.35]", err < TOL_M, f"err {err:.2e} m")

    # optical axis (OpenCV +z) must point forward/down toward the table
    opt_world = T_bw.T[:3, :3].T @ (T_base_cam[:3, :3] @ np.array([0, 0, 1.0])) \
        + 0  # z_cam in base -> world
    opt_world = _head_cam2world_gl()[:3, :3] @ np.array([0, 0, -1.0])
    check("head optical axis points forward/down", opt_world[2] < -0.5,
          f"opt dir {opt_world}")

    # table point lands at expected base coords: world (0,0,0.7401) -> base
    p_base_tbl = np.array([0.0, 0.0, 0.7401]) @ T_bw[:3, :3].T + T_bw[:3, 3]
    check("table point base coords", np.allclose(p_base_tbl, [0.65, 0.0, 0.7401],
                                                 atol=1e-9), f"{p_base_tbl}")

    # a table point must be in front of the head camera (positive optical z)
    p_cam = (p_base_tbl - T_base_cam[:3, 3]) @ T_base_cam[:3, :3]
    check("table point in camera frustum", p_cam[2] > 0, f"z_cam {p_cam[2]:.3f}")

    # cam2world_gl @ diag(1,-1,-1) convention consistency (random points)
    rng = np.random.default_rng(1)
    gl = _head_cam2world_gl()
    cv = geo.cam2world_gl_to_world_cam_cv(gl)
    p_gl = rng.uniform(-1, 1, size=(5, 3))
    w1 = p_gl @ gl[:3, :3].T + gl[:3, 3]
    w2 = (p_gl * np.array([1, -1, -1])) @ cv[:3, :3].T + cv[:3, 3]
    check("gl/cv axis-flip consistency", np.abs(w1 - w2).max() < 1e-9)


def test_ply(tmpdir="/tmp/cvwam_ply_test"):
    from cvwam.geometry import write_ply
    Path(tmpdir).mkdir(exist_ok=True)
    pts = np.random.default_rng(2).uniform(size=(10, 3)).astype(np.float32)
    col = np.full((10, 3), 128)
    p = Path(tmpdir) / "t.ply"
    write_ply(p, pts, col, extra={"view_id": np.arange(10)})
    text = p.read_text()
    check("ply header ok", text.startswith("ply") and "element vertex 10" in text
          and "property float view_id" in text)
    body = text.split("end_header\n")[1].strip().splitlines()
    ok = len(body) == 10 and float(body[0].split()[6]) == 0.0
    check("ply body ok", ok)


if __name__ == "__main__":
    test_nontrivial_poses()
    test_camera_construction()
    test_ply()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED")
        for n, d in FAILURES:
            print(f"  - {n}: {d}")
        sys.exit(1)
    print("ALL TESTS PASSED")
