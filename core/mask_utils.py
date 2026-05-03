"""
Mask preprocessing and class filtering.

Standalone — no model dependencies, no global state.
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Morphological cleanup
# ---------------------------------------------------------------------------

def morph_mask(
    mask: np.ndarray,
    method: str = "erode",
    kernel_size: int = 9,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological operation to a uint8 (0/255) mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    ops = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
    }
    return cv2.morphologyEx(mask, ops.get(method, cv2.MORPH_ERODE), k, iterations=iterations)


def _masks_to_binary(masks, orig_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Convert ultralytics Masks -> list of (H,W) uint8 masks (0/255).

    Uses ``masks.xy`` (pixel-coordinate polygons) — same source as the
    proven old pipeline — to avoid rounding differences from normalised
    coords.
    """
    H, W = orig_shape
    result: List[np.ndarray] = []

    # Prefer polygon-based (pixel coords, matching old masks_to_binary_by_polygons)
    xy = getattr(masks, "xy", None)
    if xy is not None and len(xy) > 0:
        for poly in xy:
            if poly is None or len(poly) == 0:
                result.append(np.zeros((H, W), dtype=np.uint8))
                continue
            # poly may be a single (N,2) array or a list of arrays
            all_polys = poly if isinstance(poly, (list, tuple)) else [poly]
            m = np.zeros((H, W), dtype=np.uint8)
            for p in all_polys:
                pts = np.asarray(p, dtype=np.int32)
                if pts.ndim == 2 and pts.shape[0] >= 3:
                    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                    cv2.fillPoly(m, [pts], 255)
            result.append(m)
        return result

    # Fallback: tensor data
    data = getattr(masks, "data", None)
    if data is not None:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        for i in range(data.shape[0]):
            m = data[i]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            result.append(((m > 0.5) * 255).astype(np.uint8))
        return result

    return []


def _remove_overlaps(masks: List[np.ndarray]) -> List[np.ndarray]:
    """Zero-out pixels that belong to more than one mask."""
    if len(masks) <= 1:
        return masks
    stack = np.stack([(m > 0).astype(np.uint8) for m in masks], axis=0)
    overlap = stack.sum(axis=0) > 1
    return [np.where(overlap, np.uint8(0), m) for m in masks]


# ---------------------------------------------------------------------------
# High-level: YOLO result -> clean masks
# ---------------------------------------------------------------------------

def preprocess_masks(
    yolo_result,
    kernel_size: int = 9,
) -> List[np.ndarray]:
    """Extract and clean masks from a single YOLO result.

    Returns a list of (H,W) uint8 binary masks (0/255), one per detection.
    """
    masks_attr = getattr(yolo_result, "masks", None)
    if masks_attr is None or getattr(masks_attr, "data", None) is None:
        N = 0
        if hasattr(yolo_result, "boxes") and yolo_result.boxes is not None:
            try:
                N = len(yolo_result.boxes.xyxy)
            except Exception:
                pass
        return [np.zeros(yolo_result.orig_shape, dtype=np.uint8)] * N

    bin_masks = _masks_to_binary(masks_attr, yolo_result.orig_shape)
    cleaned = [morph_mask(m, method="erode", kernel_size=kernel_size) for m in bin_masks]
    return _remove_overlaps(cleaned)


# ---------------------------------------------------------------------------
# YOLO result -> track IDs + class names
# ---------------------------------------------------------------------------

def extract_ids_and_classes(
    yolo_result, n_masks: int,
) -> Tuple[np.ndarray, Optional[List[str]]]:
    """Pull track IDs and class names from a YOLO result."""
    track_ids = None
    class_names = None
    boxes = getattr(yolo_result, "boxes", None)

    if boxes is not None:
        ids = getattr(boxes, "id", None)
        if ids is not None:
            try:
                track_ids = ids.detach().cpu().numpy().astype(np.int64)
            except Exception:
                pass
        cls_t = getattr(boxes, "cls", None)
        if cls_t is not None and hasattr(yolo_result, "names"):
            try:
                cls_ids = cls_t.detach().cpu().numpy().astype(np.int64)
                class_names = [yolo_result.names[int(c)] for c in cls_ids]
            except Exception:
                pass

    if track_ids is None:
        track_ids = np.arange(n_masks, dtype=np.int64)
    return track_ids, class_names


# ---------------------------------------------------------------------------
# Class filtering
# ---------------------------------------------------------------------------

def should_skip(name: Optional[str], exact: Set[str], substrings: Set[str]) -> bool:
    if name is None:
        return False
    low = name.lower()
    if low in exact:
        return True
    return any(kw in low for kw in substrings)


def filter_by_class(
    masks: List[np.ndarray],
    track_ids: np.ndarray,
    class_names: Optional[List[str]],
    skip_exact: Set[str],
    skip_substring: Set[str],
) -> Tuple[List[np.ndarray], np.ndarray, Optional[List[str]]]:
    """Remove detections whose class is in the skip sets."""
    if not skip_exact or class_names is None:
        return masks, track_ids, class_names

    keep = [i for i, c in enumerate(class_names)
            if not should_skip(c, skip_exact, skip_substring)]
    if len(keep) == len(class_names):
        return masks, track_ids, class_names

    masks = [masks[i] for i in keep] if masks else []
    track_ids = track_ids[keep] if track_ids is not None else np.array([], dtype=np.int64)
    class_names = [class_names[i] for i in keep]
    return masks, track_ids, class_names
