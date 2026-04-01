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
        "merge_iou_threshold", "merge_containment_threshold",
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
        merge_iou_threshold: float = 0.5,
        merge_containment_threshold: float = 0.7,
    ) -> None:
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.max_points = max_points
        self.inactive_limit = inactive_limit
        self.volume_ratio_threshold = volume_ratio_threshold
        self.visibility_threshold = visibility_threshold
        self.merge_iou_threshold = merge_iou_threshold
        self.merge_containment_threshold = merge_containment_threshold

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

    def end_frame(self, visible_gids: Set[int]) -> None:
        """Store previous-frame snapshot and optionally cleanup.

        ``visible_gids`` should include both:
        - objects updated from current YOLO detections
        - objects considered visible via reprojection fallback
        """
        self.prev_frame = {
            gid: self.objects[gid]
            for gid in visible_gids
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
            "last_seen_yolo_frame": frame_idx,
            "last_seen_camera_view_frame": frame_idx,
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
        obj["last_seen_yolo_frame"] = frame_idx
        obj["last_seen_camera_view_frame"] = frame_idx
        obj["observation_count"] = obj.get("observation_count", 0) + 1
        if cls and not obj.get("class_name"):
            obj["class_name"] = cls
        if mask is not None:
            obj["last_mask"] = mask

        # Update YOLO mapping
        if yolo_id >= 0:
            self.yolo_to_global[yolo_id] = gid

        self._visible_this_frame.add(gid)

    def mark_visible_in_camera(self, gid: int, frame_idx: int) -> None:
        """Mark an existing object as visible in the current camera view.

        This is used when YOLO misses an object, but reprojection indicates
        that its 3D bbox is still visible in the image.
        """
        obj = self.objects.get(gid)
        if obj is None:
            return
        obj["last_seen_camera_view_frame"] = frame_idx
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
                self.mark_visible_in_camera(gid, self.current_frame)
                extras.append(TrackedObject(
                    global_id=gid,
                    yolo_id=-1,
                    class_name=obj.get("class_name"),
                    bbox_3d=bbox,
                    mask=obj.get("last_mask"),
                    first_seen=obj["first_seen_frame"],
                    last_seen=self.current_frame,
                    observation_count=obj.get("observation_count", 0),
                ))
        return extras

    def get_all_objects(self) -> Dict[int, dict]:
        return self.objects

    def get_all_pcds_for_visualization(self) -> List[dict]:
        """Return per-object dicts suitable for RerunVisualizer."""
        result = []
        for gid, obj in self.objects.items():
            bbox = obj.get("bbox_3d")
            bbox_dict = bbox.to_dict() if bbox is not None else None
            result.append({
                "global_id": gid,
                "points": obj.get("points_accumulated"),
                "class_name": obj.get("class_name"),
                "bbox_3d": bbox_dict,
                "visible_current_frame": gid in self._visible_this_frame,
                "observation_count": obj.get("observation_count", 0),
            })
        return result

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
    # Object merging — collapse duplicates that grew to overlap
    # ------------------------------------------------------------------

    def merge_overlapping_objects(self) -> List[tuple]:
        """Detect and merge registry objects whose bboxes significantly overlap.

        Called after ``end_frame`` to clean up duplicates that arose when a
        partial re-observation was incorrectly registered as a new object
        and later grew to overlap the original.

        Thresholds are read from ``self.merge_iou_threshold`` and
        ``self.merge_containment_threshold`` (set via config).

        Returns list of ``(absorbed_gid, survivor_gid)`` pairs.
        """
        from core.geometry import aabb_iou, aabb_containment

        if len(self.objects) < 2:
            return []

        iou_th = self.merge_iou_threshold
        cont_th = self.merge_containment_threshold

        gids = list(self.objects.keys())
        merged_pairs: List[tuple] = []
        absorbed: Set[int] = set()

        for i in range(len(gids)):
            if gids[i] in absorbed:
                continue
            for j in range(i + 1, len(gids)):
                if gids[j] in absorbed:
                    continue
                # gids[i] may have been absorbed during a previous j iteration
                # (when it had fewer observations than gids[j'])
                if gids[i] in absorbed:
                    break

                obj_a = self.objects[gids[i]]
                obj_b = self.objects[gids[j]]
                bbox_a = obj_a.get("bbox_3d")
                bbox_b = obj_b.get("bbox_3d")

                if bbox_a is None or bbox_b is None:
                    continue

                iou = aabb_iou(bbox_a, bbox_b)
                containment = aabb_containment(bbox_a, bbox_b)

                if iou >= iou_th or containment >= cont_th:
                    # Survivor = the object seen in more frames (tie → older)
                    cnt_a = obj_a.get("observation_count", 0)
                    cnt_b = obj_b.get("observation_count", 0)
                    if cnt_a >= cnt_b:
                        survivor, absorbed_id = gids[i], gids[j]
                    else:
                        survivor, absorbed_id = gids[j], gids[i]

                    self._merge_into(survivor, absorbed_id)
                    absorbed.add(absorbed_id)
                    merged_pairs.append((absorbed_id, survivor))
                    # If the outer-loop object was absorbed, stop the inner loop
                    if absorbed_id == gids[i]:
                        break

        return merged_pairs

    def _merge_into(self, survivor_gid: int, absorbed_gid: int) -> None:
        """Absorb one object into another and delete the absorbed entry."""
        surv = self.objects[survivor_gid]
        abso = self.objects[absorbed_gid]

        # Merge point clouds
        pts_s = surv.get("points_accumulated")
        pts_a = abso.get("points_accumulated")
        if (pts_s is not None and pts_s.shape[0] > 0
                and pts_a is not None and pts_a.shape[0] > 0):
            merged = np.vstack([pts_s, pts_a])
            if merged.shape[0] > self.max_points:
                rng = np.random.default_rng(42)
                idx = rng.choice(merged.shape[0], self.max_points, replace=False)
                merged = merged[idx]
            surv["points_accumulated"] = merged
        elif pts_a is not None and pts_a.shape[0] > 0:
            surv["points_accumulated"] = pts_a

        # Recompute bbox from merged points
        pts = surv.get("points_accumulated")
        if pts is not None and pts.shape[0] > 0:
            surv["bbox_3d"] = compute_bbox(pts, fast=True)

        # Merge metadata
        surv["first_seen_frame"] = min(
            surv.get("first_seen_frame", 0),
            abso.get("first_seen_frame", 0),
        )
        surv["last_seen_frame"] = max(
            surv.get("last_seen_frame", 0),
            abso.get("last_seen_frame", 0),
        )
        surv["observation_count"] = (
            surv.get("observation_count", 0) + abso.get("observation_count", 0)
        )
        if not surv.get("class_name") and abso.get("class_name"):
            surv["class_name"] = abso["class_name"]

        # Remap YOLO track-ID mappings: absorbed → survivor
        for yid in list(self.yolo_to_global):
            if self.yolo_to_global[yid] == absorbed_gid:
                self.yolo_to_global[yid] = survivor_gid

        # Update prev_frame
        if absorbed_gid in self.prev_frame:
            self.prev_frame[survivor_gid] = surv
            del self.prev_frame[absorbed_gid]

        # Transfer visibility
        if absorbed_gid in self._visible_this_frame:
            self._visible_this_frame.add(survivor_gid)
            self._visible_this_frame.discard(absorbed_gid)

        # Remove absorbed object
        del self.objects[absorbed_gid]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cleanup(self, frame_idx: int) -> None:
        to_remove = [
            gid for gid, obj in self.objects.items()
            if frame_idx
            - max(
                obj.get("last_seen_frame", -1),
                obj.get("last_seen_camera_view_frame", -1),
            )
            > self.inactive_limit
        ]
        for gid in to_remove:
            del self.objects[gid]
            self.yolo_to_global = {
                k: v for k, v in self.yolo_to_global.items() if v != gid
            }
