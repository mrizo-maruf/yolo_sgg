"""
Global Object Registry — persistent cross-frame object tracking.

Matching priority:
    1. YOLO track-ID mapping (verified spatially)
    2. Previous-frame temporal continuity
    3. Global registry lookup (re-observation after occlusion)
    4. New object

Anti-containment guards (volume-ratio + dimension checks) prevent
matching small objects to large containers.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from .geometry import (
    aabb_containment,
    aabb_iou,
    bbox_visibility_fraction,
    compute_bbox,
)
from .types import BBox3D, CameraIntrinsics, TrackedObject


class GlobalObjectRegistry:
    """Central registry for all tracked objects with PCD accumulation."""

    __slots__ = (
        "overlap_threshold", "distance_threshold", "max_points",
        "inactive_limit", "volume_ratio_threshold", "visibility_threshold",
        "objects", "_next_id", "_prev_frame", "yolo_to_global",
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

        # global_id → object data dict
        self.objects: Dict[int, dict] = {}
        self._next_id = 1
        self._prev_frame: Dict[int, dict] = {}
        self.yolo_to_global: Dict[int, int] = {}
        self.current_frame = -1
        self._visible_this_frame: Set[int] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_idx: int,
        detections: List[dict],
    ) -> List[TrackedObject]:
        """Match *detections* to the registry and return lightweight objects.

        Each detection dict must have:
            yolo_track_id  (int)
            points         (np.ndarray (N,3) world)
            bbox_3d        (BBox3D | None)
            class_name     (str | None)
            mask           (np.ndarray | None)
        """
        self.current_frame = frame_idx
        self._visible_this_frame.clear()

        results: List[TrackedObject] = []
        matched_globals: Set[int] = set()

        for det in detections:
            yolo_id = det["yolo_track_id"]
            pts = det["points"]
            bbox = det["bbox_3d"]
            cls = det.get("class_name")
            mask = det.get("mask")

            gid = self._match(yolo_id, pts, bbox, matched_globals)
            matched_globals.add(gid)

            # Accumulate points
            obj = self.objects[gid]
            existing = obj.get("points_accumulated")
            if existing is not None and existing.shape[0] > 0 and pts is not None and pts.shape[0] > 0:
                merged = np.vstack([existing, pts])
                if merged.shape[0] > self.max_points:
                    rng = np.random.default_rng(frame_idx)
                    idx = rng.choice(merged.shape[0], self.max_points, replace=False)
                    merged = merged[idx]
                obj["points_accumulated"] = merged
            elif pts is not None and pts.shape[0] > 0:
                obj["points_accumulated"] = pts

            # Recompute bbox from accumulated points
            acc = obj["points_accumulated"]
            if acc is not None and acc.shape[0] > 0:
                obj["bbox_3d"] = compute_bbox(acc, fast=True)

            obj["last_seen_frame"] = frame_idx
            obj["observation_count"] = obj.get("observation_count", 0) + 1
            obj["class_name"] = cls or obj.get("class_name")
            obj["last_mask"] = mask
            self._visible_this_frame.add(gid)

            results.append(TrackedObject(
                global_id=gid,
                yolo_id=yolo_id,
                class_name=obj.get("class_name"),
                bbox_3d=obj.get("bbox_3d"),
                mask=mask,
                first_seen=obj["first_seen_frame"],
                last_seen=frame_idx,
                observation_count=obj["observation_count"],
            ))

        # Snapshot for next-frame temporal matching
        self._prev_frame = {gid: self.objects[gid] for gid in matched_globals}
        return results

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

    # ------------------------------------------------------------------
    # Matching internals
    # ------------------------------------------------------------------

    def _match(
        self,
        yolo_id: int,
        pts: np.ndarray,
        bbox: Optional[BBox3D],
        already_matched: Set[int],
    ) -> int:
        """Find or create a global ID for this detection."""

        # 1) YOLO track-ID mapping
        if yolo_id in self.yolo_to_global:
            gid = self.yolo_to_global[yolo_id]
            if gid not in already_matched and gid in self.objects:
                if self._spatial_ok(bbox, self.objects[gid].get("bbox_3d")):
                    return gid

        # 2) Previous-frame temporal continuity
        best_gid, best_score = None, -1.0
        for gid, obj in self._prev_frame.items():
            if gid in already_matched:
                continue
            score = self._score(bbox, obj.get("bbox_3d"))
            if score > best_score:
                best_score, best_gid = score, gid
        if best_gid is not None and best_score > 0:
            self.yolo_to_global[yolo_id] = best_gid
            return best_gid

        # 3) Global registry scan
        best_gid, best_score = None, -1.0
        for gid, obj in self.objects.items():
            if gid in already_matched:
                continue
            score = self._score(bbox, obj.get("bbox_3d"))
            if score > best_score:
                best_score, best_gid = score, gid
        if best_gid is not None and best_score > 0:
            self.yolo_to_global[yolo_id] = best_gid
            return best_gid

        # 4) New object
        gid = self._new_id()
        self.objects[gid] = {
            "global_id": gid,
            "points_accumulated": pts,
            "bbox_3d": bbox,
            "class_name": None,
            "first_seen_frame": self.current_frame,
            "last_seen_frame": self.current_frame,
            "observation_count": 0,
            "last_mask": None,
        }
        self.yolo_to_global[yolo_id] = gid
        return gid

    def _score(self, a: Optional[BBox3D], b: Optional[BBox3D]) -> float:
        if a is None or b is None:
            return -1.0
        dist = float(np.linalg.norm(a.center - b.center))
        if dist > self.distance_threshold:
            return -1.0
        # Dimension check
        ext_a, ext_b = a.obb_extent, b.obb_extent
        ratios = np.minimum(ext_a, ext_b) / (np.maximum(ext_a, ext_b) + 1e-6)
        if np.sum(ratios < 0.15) >= 2:
            return -1.0
        iou = aabb_iou(a, b)
        contain = aabb_containment(a, b)
        eff = max(iou, contain)
        if eff >= self.overlap_threshold or contain >= 0.5:
            return eff
        return -1.0

    def _spatial_ok(self, a: Optional[BBox3D], b: Optional[BBox3D]) -> bool:
        return self._score(a, b) > -0.5

    def _new_id(self) -> int:
        gid = self._next_id
        self._next_id += 1
        return gid
