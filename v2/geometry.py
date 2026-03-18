"""
Pure-geometry helpers — no YOLO, no model dependencies.

All functions take/return numpy arrays.  Camera intrinsics are passed
explicitly (no module-level globals).
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import open3d as o3d

from .types import BBox3D, CameraIntrinsics


# ---------------------------------------------------------------------------
# Unprojection: depth + mask → 3-D points (camera frame)
# ---------------------------------------------------------------------------

# Type alias for custom unproject functions
# Signature: (us, vs, depths, intrinsics) → (N, 3) float32
UnprojectFn = Callable[[np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics], np.ndarray]


def _default_unproject(
    us: np.ndarray, vs: np.ndarray, depths: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Standard pinhole unprojection (camera frame)."""
    X = (us.astype(np.float32) - intrinsics.cx) * depths / intrinsics.fx
    Y = (vs.astype(np.float32) - intrinsics.cy) * depths / intrinsics.fy
    return np.stack([X, Y, depths], axis=1).astype(np.float32)


def extract_points_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_points: int = 2000,
    sample_ratio: float = 0.5,
    o3_nb_neighbors: int = 50,
    o3_std_ratio: float = 0.1,
    cluster_eps: float = 0.1,
    cluster_min_samples: int = 10,
    cluster_min_fraction: float = 0.6,
    seed: int = 0,
    unproject_fn: Optional[UnprojectFn] = None,
) -> np.ndarray:
    """Depth + binary mask → cleaned (M,3) points.

    Uses *unproject_fn* to convert (u, v, depth) → 3-D points.
    Defaults to standard pinhole unprojection (camera frame).

    Returns an empty (0,3) array when no valid points are found.
    """
    rng = np.random.default_rng(seed)

    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    mask_bool = np.squeeze(mask_bool)
    valid = mask_bool & (depth_m > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    vs, us = np.nonzero(valid)
    zs = depth_m[vs, us].astype(np.float32)

    # Sub-sample
    M = zs.shape[0]
    n_keep = min(max_points, max(1, int(M * sample_ratio)))
    if M > n_keep:
        idx = rng.choice(M, size=n_keep, replace=False)
        us, vs, zs = us[idx], vs[idx], zs[idx]

    # Unproject (pluggable — THUD uses custom formula, others use pinhole)
    _unproject = unproject_fn or _default_unproject
    pts = _unproject(us, vs, zs, intrinsics)

    if pts.shape[0] < 4:
        return pts

    # Statistical outlier removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=o3_nb_neighbors, std_ratio=o3_std_ratio,
    )
    pts = np.asarray(pcd.select_by_index(ind).points, dtype=np.float32)

    # Keep largest DBSCAN cluster
    if pts.shape[0] > cluster_min_samples:
        pts = _keep_largest_cluster(
            pts, eps=cluster_eps,
            min_samples=cluster_min_samples,
            min_fraction=cluster_min_fraction,
        )

    return pts


def _keep_largest_cluster(
    pts: np.ndarray, eps: float, min_samples: int, min_fraction: float,
) -> np.ndarray:
    """DBSCAN → keep only the largest cluster if it dominates."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_samples, print_progress=False))
    if labels.max() < 0:
        return pts
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) == 0:
        return pts
    biggest = unique[np.argmax(counts)]
    biggest_mask = labels == biggest
    if biggest_mask.sum() / max(1, len(labels)) >= min_fraction:
        return pts[biggest_mask]
    return pts


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def cam_to_world(points_cam: np.ndarray, T_w_c: np.ndarray) -> np.ndarray:
    """Apply 4×4 camera→world to (N,3) points."""
    if points_cam is None or points_cam.size == 0:
        return points_cam
    R = T_w_c[:3, :3]
    t = T_w_c[:3, 3]
    return (points_cam @ R.T) + t


# ---------------------------------------------------------------------------
# 3-D bounding box
# ---------------------------------------------------------------------------

def compute_bbox(points: np.ndarray, fast: bool = True) -> Optional[BBox3D]:
    """Compute AABB (+ fast OBB) from (N,3) points."""
    if points is None or points.shape[0] == 0:
        return None

    mn = np.min(points, axis=0)
    mx = np.max(points, axis=0)
    center = (mn + mx) / 2.0
    extent = mx - mn

    if fast:
        return BBox3D(
            aabb_min=mn, aabb_max=mx,
            obb_center=center, obb_extent=extent,
        )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    return BBox3D(
        aabb_min=mn, aabb_max=mx,
        obb_center=np.asarray(obb.center),
        obb_extent=np.asarray(obb.extent),
        obb_R=np.asarray(obb.R),
    )


# ---------------------------------------------------------------------------
# Scene-height estimation
# ---------------------------------------------------------------------------

def estimate_scene_height(
    depth_m: np.ndarray,
    T_w_c: np.ndarray,
    intrinsics: CameraIntrinsics,
    stride: int = 4,
) -> float:
    """Quick estimate of scene Z-extent in world frame."""
    if depth_m is None or T_w_c is None:
        return 0.0

    H, W = depth_m.shape[:2]
    vv = np.arange(0, H, stride, dtype=np.int32)
    uu = np.arange(0, W, stride, dtype=np.int32)
    V, U = np.meshgrid(vv, uu, indexing="ij")
    D = depth_m[V, U].astype(np.float32)
    valid = D > 0
    if not np.any(valid):
        return 0.0

    us = U[valid].astype(np.float32)
    vs = V[valid].astype(np.float32)
    zs = D[valid]

    X = (us - intrinsics.cx) * zs / intrinsics.fx
    Y = (vs - intrinsics.cy) * zs / intrinsics.fy
    pts_cam = np.stack([X, Y, zs], axis=1)
    pts_w = cam_to_world(pts_cam, T_w_c)
    if pts_w is None or pts_w.size == 0:
        return 0.0

    return float(np.ptp(pts_w[:, 2]))


# ---------------------------------------------------------------------------
# 3-D IoU (axis-aligned)
# ---------------------------------------------------------------------------

def aabb_iou(a: BBox3D, b: BBox3D) -> float:
    inter_min = np.maximum(a.aabb_min, b.aabb_min)
    inter_max = np.minimum(a.aabb_max, b.aabb_max)
    if np.any(inter_min >= inter_max):
        return 0.0
    inter_vol = float(np.prod(inter_max - inter_min))
    vol_a = float(np.prod(np.maximum(a.aabb_max - a.aabb_min, 1e-9)))
    vol_b = float(np.prod(np.maximum(b.aabb_max - b.aabb_min, 1e-9)))
    return inter_vol / (vol_a + vol_b - inter_vol + 1e-12)


def aabb_containment(a: BBox3D, b: BBox3D) -> float:
    """max(intersection/vol_a, intersection/vol_b)."""
    inter_min = np.maximum(a.aabb_min, b.aabb_min)
    inter_max = np.minimum(a.aabb_max, b.aabb_max)
    if np.any(inter_min >= inter_max):
        return 0.0
    inter_vol = float(np.prod(inter_max - inter_min))
    vol_a = float(np.prod(np.maximum(a.aabb_max - a.aabb_min, 1e-9)))
    vol_b = float(np.prod(np.maximum(b.aabb_max - b.aabb_min, 1e-9)))
    return max(inter_vol / vol_a, inter_vol / vol_b)


# ---------------------------------------------------------------------------
# Reprojection visibility
# ---------------------------------------------------------------------------

def bbox_visibility_fraction(
    bbox: BBox3D,
    T_w_c: np.ndarray,
    intrinsics: CameraIntrinsics,
    max_depth: float = 10.0,
) -> float:
    """Fraction of a 3-D bbox visible when projected into the camera."""
    if bbox is None or T_w_c is None:
        return 0.0

    # 8 AABB corners
    mn, mx = bbox.aabb_min, bbox.aabb_max
    corners = np.array([
        [mn[0], mn[1], mn[2]], [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]], [mn[0], mx[1], mx[2]],
        [mx[0], mn[1], mn[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mn[2]], [mx[0], mx[1], mx[2]],
    ])

    T_c_w = np.linalg.inv(T_w_c)
    R, t = T_c_w[:3, :3], T_c_w[:3, 3]
    corners_cam = corners @ R.T + t

    # Must be in front of camera
    z = corners_cam[:, 2]
    if np.all(z <= 0):
        return 0.0
    ok = z > 0.01
    if not np.any(ok):
        return 0.0
    cc = corners_cam[ok]

    u = intrinsics.fx * cc[:, 0] / cc[:, 2] + intrinsics.cx
    v = intrinsics.fy * cc[:, 1] / cc[:, 2] + intrinsics.cy

    proj_area = max(0.0, np.ptp(u)) * max(0.0, np.ptp(v))
    if proj_area <= 0:
        return 0.0

    u_clip = np.clip(u, 0, intrinsics.width)
    v_clip = np.clip(v, 0, intrinsics.height)
    vis_area = max(0.0, np.ptp(u_clip)) * max(0.0, np.ptp(v_clip))

    if np.min(cc[:, 2]) > max_depth:
        return 0.0

    return vis_area / proj_area
