"""
THUD Real Scenes dataset loader adapter.

Wraps ``thud_utils.real_scene_loader.RealSceneDataLoader``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import YOLOE.utils as yutils
from thud_utils.real_scene_loader import (
    RealSceneDataLoader,
    Object2D as RealObject2D,
    Object3D as RealObject3D,
    discover_real_scenes,
)
from metrics.tracking_metrics import GTInstance

from .base import DatasetLoader


_DEFAULT_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background",
}


def _load_thud_depth_as_meters(
    depth_path: str,
    scale: float = 1000.0,
    max_m: float = 100.0,
) -> np.ndarray:
    """Load a THUD uint16 depth PNG → float32 metres."""
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    depth_m = d.astype(np.float32) / scale
    depth_m[depth_m < 0.01] = 0.0
    depth_m[depth_m > max_m] = 0.0
    return depth_m


class THUDRealDatasetLoader(DatasetLoader):
    """Adapter for THUD Real Scenes."""

    def __init__(
        self,
        scene_dir: str,
        skip_labels: Optional[Set[str]] = None,
        depth_scale: float = 1000.0,
        depth_max_m: float = 100.0,
        tracking_distance: float = 0.3,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS
        self._depth_scale = depth_scale
        self._depth_max_m = depth_max_m

        self._loader = RealSceneDataLoader(str(self._scene_dir), verbose=True)
        self._loader.assign_tracking_ids(distance_threshold=tracking_distance)

    # -- metadata ----------------------------------------------------------

    @property
    def scene_label(self) -> str:
        try:
            parts = self._scene_dir.parts
            idx = next(i for i, p in enumerate(parts) if p == "Real_Scenes")
            return "/".join(parts[idx:])
        except StopIteration:
            return self._scene_dir.name

    def get_frame_indices(self) -> List[int]:
        return self._loader.get_frame_indices()

    def get_class_names(self) -> List[str]:
        return self._loader.get_class_names()

    # -- paths -------------------------------------------------------------

    def get_rgb_paths(self) -> List[str]:
        paths = []
        for fidx in self._loader.get_frame_indices():
            rp = self._loader.get_rgb_path(fidx)
            if Path(rp).exists():
                paths.append(rp)
        return paths

    def get_depth_paths(self) -> List[str]:
        paths = []
        for fidx in self._loader.get_frame_indices():
            dp = self._loader.get_depth_path(fidx)
            if Path(dp).exists():
                paths.append(dp)
        return paths

    # -- depth -------------------------------------------------------------

    def load_depth(self, path: str) -> np.ndarray:
        return _load_thud_depth_as_meters(path, self._depth_scale, self._depth_max_m)

    # -- camera ------------------------------------------------------------

    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        K = self._loader.get_intrinsics_matrix()
        if K is not None:
            return K, self._loader.image_height, self._loader.image_width
        return None

    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        cam_pose = self._loader.load_camera_pose(frame_idx)
        if cam_pose is not None:
            return cam_pose.transform_matrix
        return None

    # -- ground truth ------------------------------------------------------

    def get_gt_instances(self, frame_idx: int):
        gt_2d = self._loader.get_tracked_2d_objects(frame_idx)
        gt_masks = self._loader.get_instance_masks(frame_idx, objects_2d=gt_2d)
        instances = []
        for obj, mask in zip(gt_2d, gt_masks):
            bbox = None
            if hasattr(obj, "bbox") and obj.bbox is not None:
                b = obj.bbox
                bbox = (b.get("x_min", 0), b.get("y_min", 0),
                        b.get("x_max", 0), b.get("y_max", 0))
            instances.append(
                GTInstance(
                    track_id=getattr(obj, "track_id", getattr(obj, "object_id", -1)),
                    class_name=getattr(obj, "class_name", getattr(obj, "label", "unknown")),
                    mask=mask,
                    bbox_xyxy=bbox,
                )
            )
        return instances

    # -- multi-scene -------------------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        return discover_real_scenes(root)
