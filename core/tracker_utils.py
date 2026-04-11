"""Utilities for :mod:`core.new_tracker` — overlap scoring, GPU memory
probing, and the 4-level global matching steps.

These helpers are kept separate so the top-level tracking generator in
``new_tracker.py`` reads as a short orchestration loop.
"""
from __future__ import annotations

from typing import Optional, Set, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover — torch is optional for CPU-only tooling
    torch = None  # type: ignore[assignment]

from core.geometry import aabb_containment, aabb_iou
from core.types import BBox3D


# ---------------------------------------------------------------------------
# GPU memory probing
# ---------------------------------------------------------------------------

def gpu_mem_mb(cuda_available: bool) -> Optional[float]:
    """Return current allocated CUDA memory in MB (or None if CUDA is off)."""
    if not cuda_available:
        return None
    torch.cuda.synchronize()
    return float(torch.cuda.memory_allocated() / (1024 ** 2))


# ---------------------------------------------------------------------------
# Overlap scoring
# ---------------------------------------------------------------------------

def compute_overlap_score(
    det_bbox: Optional[BBox3D],
    reg_bbox: Optional[BBox3D],
    distance_threshold: float,
    overlap_threshold: float,
) -> Tuple[float, float, bool]:
    """Compute matching score between a detection and a registry object.

    Returns
    -------
    effective_score : float
        ``max(aabb_iou, aabb_containment)`` — 0.0 when rejected.
    centroid_distance : float
        Euclidean distance between bbox centres.
    is_candidate : bool
        True when the score passes the overlap or containment threshold.
    """
    if det_bbox is None or reg_bbox is None:
        return 0.0, float("inf"), False

    dist = float(np.linalg.norm(det_bbox.center - reg_bbox.center))
    if dist > distance_threshold:
        return 0.0, dist, False

    # Containment computed early — needed to handle partial re-observation
    # (e.g. partial view of a sofa matching the full sofa in the registry).
    iou = aabb_iou(det_bbox, reg_bbox)
    containment = aabb_containment(det_bbox, reg_bbox)

    # Size-similarity check — reject if ≥2 extents wildly differ. Skip
    # when one bbox is largely contained in the other (partial re-obs).
    if containment < 0.3:
        ext_a = det_bbox.obb_extent
        ext_b = reg_bbox.obb_extent
        ratios = np.minimum(ext_a, ext_b) / (np.maximum(ext_a, ext_b) + 1e-6)
        if np.sum(ratios < 0.15) >= 2:
            return 0.0, dist, False

    effective_score = max(iou, containment)
    is_candidate = iou >= overlap_threshold or containment >= 0.3
    return effective_score, dist, is_candidate


# ---------------------------------------------------------------------------
# 4-level global matching — each function returns a global id or None.
# ---------------------------------------------------------------------------

def match_yolo_track_id(
    bbox: Optional[BBox3D],
    tid: int,
    object_registry,
    matched_gids: Set[int],
    overlap_th: float,
    dist_th: float,
) -> Optional[int]:
    """LEVEL 1 — YOLO track-id mapping, verified spatially.

    Accept the mapped gid if the overlap check passes, OR if the
    centroids are well within half of ``dist_th`` (fallback when a
    bbox just shifted across frames).
    """
    if tid < 0 or tid not in object_registry.yolo_to_global:
        return None
    candidate = object_registry.yolo_to_global[tid]
    if candidate in matched_gids or candidate not in object_registry.objects:
        return None

    existing = object_registry.objects[candidate]
    _, cdist, is_match = compute_overlap_score(
        bbox, existing.get("bbox_3d"), dist_th, overlap_th,
    )
    if is_match or cdist < dist_th * 0.5:
        return candidate
    return None


def match_prev_frame(
    bbox: Optional[BBox3D],
    object_registry,
    matched_gids: Set[int],
    overlap_th: float,
    dist_th: float,
) -> Optional[int]:
    """LEVEL 2 — best-overlap match against the previous frame."""
    best_gid, best_score = None, 0.0
    for cand_gid, prev_obj in object_registry.prev_frame.items():
        if cand_gid in matched_gids:
            continue
        score, _, is_cand = compute_overlap_score(
            bbox, prev_obj.get("bbox_3d"), dist_th, overlap_th,
        )
        if is_cand and score > best_score:
            best_score = score
            best_gid = cand_gid
    return best_gid


def match_registry_reobservation(
    bbox: Optional[BBox3D],
    object_registry,
    matched_gids: Set[int],
    overlap_th: float,
    dist_th: float,
) -> Optional[int]:
    """LEVEL 3 — global registry lookup (re-observation after occlusion).

    Uses a relaxed distance threshold proportional to the existing object
    size so partial views of large objects (sofa, table) whose centroid
    is offset can still match.
    """
    best_gid, best_score = None, 0.0
    for cand_gid, obj in object_registry.objects.items():
        if cand_gid in matched_gids:
            continue
        existing_bbox = obj.get("bbox_3d")
        level3_dist = dist_th
        if existing_bbox is not None:
            max_extent = float(np.max(existing_bbox.obb_extent))
            level3_dist = max(dist_th, max_extent * 0.75)
        score, _, is_cand = compute_overlap_score(
            bbox, existing_bbox, level3_dist, overlap_th,
        )
        if is_cand and score > best_score:
            best_score = score
            best_gid = cand_gid
    return best_gid
