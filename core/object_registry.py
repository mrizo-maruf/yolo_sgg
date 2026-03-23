"""
Global Object Registry — persistent cross-frame object storage.

Matching logic lives in ``core.new_tracker.run_tracking`` (inline in the
per-detection loop).  This module handles only creation, update, cleanup,
and reprojection-visibility queries.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from core.geometry import bbox_visibility_fraction, compute_bbox
from core.types import BBox3D, CameraIntrinsics, TrackedObject


class GlobalObjectRegistry:
    """Central storage for all tracked objects with PCD accumulation.

    Matching priority (implemented in new_tracker):
        1. YOLO track-ID mapping (verified spatially)
        2. Previous-frame temporal continuity
        3. Global registry lookup (re-observation after occlusion)
        4. New object
    """

    __slots__ = (
        "overlap_threshold", "distance_threshold", "max_points",
        "inactive_limit", "volume_ratio_threshold", "visibility_threshold",
        "objects", "_next_id", "prev_frame", "yolo_to_global",
        "current_frame", "_visible_this_frame",
    )

    def __init__(
        self,
        overlap_threshold: float = 0.1,
        distance_threshold: float = 1.0,
        max_points: int = 10_000,
        inactive_limit: int = 0,
        volume_ratio_threshold: float = 0.1,
        visibility_threshold: float = 0.2,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.max_points = max_points
        self.inactive_limit = inactive_limit
        self.volume_ratio_threshold = volume_ratio_threshold
        self.visibility_threshold = visibility_threshold

        self.objects: Dict[int, dict] = {}
        self._next_id = 1
        self.prev_frame: Dict[int, dict] = {}
        self.yolo_to_global: Dict[int, int] = {}
        self.current_frame = -1
        self._visible_this_frame: Set[int] = set()

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self, frame_idx: int) -> None:
        """Reset per-frame state before processing detections."""
        self.current_frame = frame_idx
        self._visible_this_frame.clear()

    def end_frame(self, matched_gids: Set[int]) -> None:
        """Store previous-frame snapshot and optionally cleanup."""
        self.prev_frame = {
            gid: self.objects[gid]
            for gid in matched_gids
            if gid in self.objects
        }
        if self.inactive_limit > 0:
            self._cleanup(self.current_frame)

    # ------------------------------------------------------------------
    # Object creation / update
    # ------------------------------------------------------------------

    def new_id(self) -> int:
        gid = self._next_id
        self._next_id += 1
        return gid

    def register_new(
        self,
        gid: int,
        pts: np.ndarray,
        bbox: Optional[BBox3D],
        cls: Optional[str],
        mask: Optional[np.ndarray],
        yolo_id: int,
        frame_idx: int,
    ) -> None:
        """Create a brand-new tracked object."""
        self.objects[gid] = {
            "points_accumulated": pts.copy() if pts is not None else np.zeros((0, 3), np.float32),
            "bbox_3d": bbox,
            "class_name": cls,
            "first_seen_frame": frame_idx,
            "last_seen_frame": frame_idx,
            "observation_count": 1,
            "last_mask": mask,
        }
        if yolo_id >= 0:
            self.yolo_to_global[yolo_id] = gid
        self._visible_this_frame.add(gid)

    def update_object(
        self,
        gid: int,
        pts: np.ndarray,
        bbox: Optional[BBox3D],
        cls: Optional[str],
        mask: Optional[np.ndarray],
        yolo_id: int,
        frame_idx: int,
    ) -> None:
        """Merge new points into an existing object and update metadata."""
        obj = self.objects[gid]

        # Merge point clouds
        existing = obj.get("points_accumulated")
        if existing is not None and existing.shape[0] > 0 and pts is not None and pts.shape[0] > 0:
            merged = np.vstack([existing, pts])
            if merged.shape[0] > self.max_points:
                rng = np.random.default_rng(frame_idx)
                idx = rng.choice(merged.shape[0], self.max_points, replace=False)
                merged = merged[idx]
        elif pts is not None and pts.shape[0] > 0:
            merged = pts
        else:
            merged = existing

        obj["points_accumulated"] = merged

        # Recompute bbox from accumulated points
        if merged is not None and merged.shape[0] > 0:
            obj["bbox_3d"] = compute_bbox(merged, fast=True)

        obj["last_seen_frame"] = frame_idx
        obj["observation_count"] = obj.get("observation_count", 0) + 1
        if cls and not obj.get("class_name"):
            obj["class_name"] = cls
        if mask is not None:
            obj["last_mask"] = mask

        # Update YOLO mapping
        if yolo_id >= 0:
            self.yolo_to_global[yolo_id] = gid

        self._visible_this_frame.add(gid)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_reprojection_visible(
        self,
        T_w_c: Optional[np.ndarray],
        intrinsics: CameraIntrinsics,
        excluded_ids: Set[int],
        max_depth: float = 10.0,
    ) -> List[TrackedObject]:
        """Return objects not detected but visible via reprojection."""
        if T_w_c is None:
            return []
        extras: List[TrackedObject] = []
        for gid, obj in self.objects.items():
            if gid in excluded_ids:
                continue
            bbox = obj.get("bbox_3d")
            if bbox is None:
                continue
            frac = bbox_visibility_fraction(bbox, T_w_c, intrinsics, max_depth)
            if frac >= self.visibility_threshold:
                extras.append(TrackedObject(
                    global_id=gid,
                    yolo_id=-1,
                    class_name=obj.get("class_name"),
                    bbox_3d=bbox,
                    mask=obj.get("last_mask"),
                    first_seen=obj["first_seen_frame"],
                    last_seen=obj.get("last_seen_frame", 0),
                    observation_count=obj.get("observation_count", 0),
                ))
        return extras

    def get_all_objects(self) -> Dict[int, dict]:
        return self.objects

    def get_visible_objects(self, frame_idx: int) -> List[TrackedObject]:
        """Return TrackedObjects visible in the given frame."""
        visible = []
        for gid in self._visible_this_frame:
            obj = self.objects.get(gid)
            if obj is None:
                continue
            visible.append(TrackedObject(
                global_id=gid,
                yolo_id=-1,
                class_name=obj.get("class_name"),
                bbox_3d=obj.get("bbox_3d"),
                mask=obj.get("last_mask"),
                first_seen=obj["first_seen_frame"],
                last_seen=obj.get("last_seen_frame", 0),
                observation_count=obj.get("observation_count", 0),
            ))
        return visible

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup(self, frame_idx: int) -> None:
        to_remove = [
            gid for gid, obj in self.objects.items()
            if frame_idx - obj.get("last_seen_frame", frame_idx) > self.inactive_limit
        ]
        for gid in to_remove:
            del self.objects[gid]
            self.yolo_to_global = {
                k: v for k, v in self.yolo_to_global.items() if v != gid
            }
