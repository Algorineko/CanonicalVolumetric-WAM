"""Alignment utilities: Umeyama Sim(3), camera-center similarity, plane-based
scale anchoring, and NN point-cloud metrics.

Strictly separates oracle diagnostics (uses GT-depth correspondences) from
deployable alignment (uses only known calibration / physical constants),
per docs/plan.md §6.4.
"""

import numpy as np
from scipy.spatial import cKDTree


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale=True):
    """Least-squares similarity src->dst: dst ~ s R src + t. Returns (s, R, t)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = dst_c.T @ src_c / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (src_c ** 2).sum() / len(src)
    if with_scale:
        s = np.trace(np.diag(D) @ S) / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * R @ np.asarray(points, dtype=np.float64).T).T + t


def fit_similarity_on_centers(centers_src, centers_dst):
    """Deployable: similarity from known camera-center correspondences.
    Returns (s, R, t, degeneracy_flag)."""
    A = np.asarray(centers_src, dtype=np.float64)
    B = np.asarray(centers_dst, dtype=np.float64)
    if len(A) < 2:
        return None, None, None, True
    s, R, t = umeyama_sim3(A, B, with_scale=True)
    # degeneracy: centers nearly collinear -> rotation about the line undefined
    A_c = A - A.mean(0)
    if len(A) == 2:
        collinear = True
    else:
        sv = np.linalg.svd(A_c, compute_uv=False)
        collinear = sv[1] < 1e-3 * max(sv[0], 1e-9)
    return s, R, t, bool(collinear)


def fit_dominant_plane(points: np.ndarray, iters: int = 500, thresh: float = 0.01,
                       seed: int = 0, min_frac: float = 0.15):
    """RANSAC dominant plane. Returns (normal, d, inlier_frac) with n·p = d."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=np.float64)
    best = (None, None, 0.0)
    n = len(pts)
    if n < 10:
        return None, None, 0.0
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i]
        nvec = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nvec)
        if norm < 1e-9:
            continue
        nvec = nvec / norm
        dd = nvec @ p0
        dist = np.abs(pts @ nvec - dd)
        frac = float((dist < thresh).mean())
        if frac > best[2]:
            best = (nvec, float(dd), frac)
    if best[2] < min_frac:
        return None, None, best[2]
    # refine with inliers
    nvec, dd, _ = best
    inl = np.abs(pts @ nvec - dd) < thresh
    if inl.sum() >= 3:
        c = pts[inl].mean(0)
        Q = pts[inl] - c
        _, _, Vt = np.linalg.svd(Q, full_matrices=False)
        nvec = Vt[2]
        dd = float(nvec @ c)
    return nvec, dd, best[2]


def camera_centers_from_extrinsic(extr: np.ndarray) -> np.ndarray:
    """extr (3,4) or (4,4) maps world->camera; center in world = -R^T t."""
    extr = np.asarray(extr, dtype=np.float64)
    R = extr[:3, :3]
    t = extr[:3, 3]
    return -R.T @ t


def nn_metrics(src: np.ndarray, dst_tree: cKDTree, thresholds_m=(0.01, 0.02)):
    """Accuracy-style: NN distances of src against dst. NaN-safe on empty src."""
    if len(src) == 0:
        out = {"mean_mm": float("nan"), "median_mm": float("nan"),
               "p95_mm": float("nan")}
        for th in thresholds_m:
            out[f"fscore_{int(th * 1000)}mm"] = float("nan")
        return out
    d, _ = dst_tree.query(src, k=1, workers=-1)
    out = {"mean_mm": float(d.mean() * 1000), "median_mm": float(np.median(d) * 1000),
           "p95_mm": float(np.percentile(d, 95) * 1000)}
    for th in thresholds_m:
        out[f"fscore_{int(th * 1000)}mm"] = float((d < th).mean())
    return out


def depth_err_stats(pred_z: np.ndarray, gt_z: np.ndarray, valid: np.ndarray):
    """Depth metrics on valid pixels only; arrays same shape, meters."""
    p = pred_z[valid]
    g = gt_z[valid]
    ok = np.isfinite(p) & np.isfinite(g) & (g > 0)
    p, g = p[ok], g[ok]
    if len(p) == 0:
        return {"mae_mm": float("nan"), "rmse_mm": float("nan"),
                "absrel": float("nan"), "n": 0}
    return {"mae_mm": float(np.abs(p - g).mean() * 1000),
            "rmse_mm": float(np.sqrt(((p - g) ** 2).mean()) * 1000),
            "absrel": float((np.abs(p - g) / g).mean()),
            "n": int(ok.sum())}
