"""
Shared helpers used by both ``run.py`` and ``bench.py``.

These were previously duplicated across ``yolo_ssg.py``,
``benchmark/benchmark_tracking.py`` and
``benchmark/benchmark_tracking_thud.py``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Skip-class filtering
# ---------------------------------------------------------------------------

def should_skip_class(name: str, skip_set: Set[str], substring_skip: Set[str]) -> bool:
    """Return *True* if *name* matches any skip pattern."""
    if name is None:
        return False
    low = name.lower()
    if low in skip_set:
        return True
    for kw in substring_skip:
        if kw in low:
            return True
    return False


# ---------------------------------------------------------------------------
# YOLO result extraction
# ---------------------------------------------------------------------------

def extract_yolo_ids(yolo_res, masks_clean):
    """Pull track IDs and class names from a YOLO result object.

    Returns
    -------
    track_ids : np.ndarray[int64]
    class_names : list[str] | None
    """
    track_ids = None
    class_names = None

    if hasattr(yolo_res, "boxes") and yolo_res.boxes is not None:
        if getattr(yolo_res.boxes, "id", None) is not None:
            try:
                track_ids = yolo_res.boxes.id.detach().cpu().numpy().astype(np.int64)
            except Exception:
                pass
        if getattr(yolo_res.boxes, "cls", None) is not None and hasattr(yolo_res, "names"):
            try:
                cls_ids = yolo_res.boxes.cls.detach().cpu().numpy().astype(np.int64)
                class_names = [yolo_res.names[int(c)] for c in cls_ids]
            except Exception:
                pass

    n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
    if track_ids is None:
        track_ids = np.arange(n, dtype=np.int64)

    return track_ids, class_names


# ---------------------------------------------------------------------------
# Class filter
# ---------------------------------------------------------------------------

def apply_class_filter(masks_clean, track_ids, class_names, skip_set, _substring_skip):
    """Remove detections whose class name is in *skip_set*."""
    if not skip_set or class_names is None:
        return masks_clean, track_ids, class_names

    keep = [i for i, c in enumerate(class_names) if not should_skip_class(c, skip_set, _substring_skip)]
    if len(keep) == len(class_names):
        return masks_clean, track_ids, class_names

    masks_clean = [masks_clean[i] for i in keep] if masks_clean else []
    track_ids = track_ids[keep] if track_ids is not None else None
    class_names = [class_names[i] for i in keep]
    return masks_clean, track_ids, class_names


# ---------------------------------------------------------------------------
# Build PredInstance list (for benchmarking)
# ---------------------------------------------------------------------------

def build_pred_instances(frame_objs, track_ids, masks_clean):
    """Convert pipeline output dicts → ``PredInstance`` list.

    Imports ``PredInstance`` lazily to keep the helper module lightweight.
    """
    from metrics.tracking_metrics import PredInstance

    preds: List[PredInstance] = []
    for obj in frame_objs:
        mask = None
        yolo_tid = obj.get("yolo_track_id", -1)
        if yolo_tid >= 0 and track_ids is not None:
            idxs = np.where(track_ids == yolo_tid)[0]
            if len(idxs) > 0 and masks_clean and idxs[0] < len(masks_clean):
                mask = masks_clean[idxs[0]]
        preds.append(
            PredInstance(
                pred_id=obj["global_id"],
                class_name=obj.get("class_name"),
                mask=mask,
            )
        )
    return preds
