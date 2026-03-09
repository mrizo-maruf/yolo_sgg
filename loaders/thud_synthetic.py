"""
THUD Synthetic dataset loader adapter.

Wraps ``thud_utils.thud_synthetic_loader.THUDSyntheticLoader``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

import YOLOE.utils as yutils
from thud_utils.thud_synthetic_loader import (
    THUDSyntheticLoader,
    GTObject,
    discover_thud_synthetic_scenes,
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


class THUDSyntheticDatasetLoader(DatasetLoader):
    """Adapter for THUD Synthetic Scenes."""

    def __init__(
        self,
        scene_dir: str,
        skip_labels: Optional[Set[str]] = None,
        depth_scale: float = 1000.0,
        depth_max_m: float = 100.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS
        self._depth_scale = depth_scale
        self._depth_max_m = depth_max_m

        self._loader = THUDSyntheticLoader(
            str(self._scene_dir),
            load_rgb=True,
            load_depth=True,
            skip_labels=self._skip_labels,
            verbose=False,
        )

        # Read intrinsics from first frame
        first_fd = self._loader.get_frame_data(self._loader.frame_indices[0])
        self._K = first_fd.camera_intrinsic
        if first_fd.rgb is not None:
            self._img_h, self._img_w = first_fd.rgb.shape[:2]
        else:
            self._img_h, self._img_w = 530, 730

    # -- metadata ----------------------------------------------------------

    @property
    def scene_label(self) -> str:
        try:
            parts = self._scene_dir.parts
            idx = next(
                i for i, p in enumerate(parts)
                if p in ("Gym", "House", "Office", "Supermarket_1", "Supermarket_2")
            )
            return "/".join(parts[idx:])
        except StopIteration:
            return self._scene_dir.name

    def get_frame_indices(self) -> List[int]:
        return list(self._loader.frame_indices)

    # -- paths -------------------------------------------------------------

    def get_rgb_paths(self) -> List[str]:
        paths = []
        for fidx in self._loader.frame_indices:
            rp = self._loader.rgb_dir / f"rgb_{fidx}.png"
            if rp.exists():
                paths.append(str(rp))
        return paths

    def get_depth_paths(self) -> List[str]:
        paths = []
        for fidx in self._loader.frame_indices:
            dp = self._loader.depth_dir / f"depth_{fidx}.png"
            if dp.exists():
                paths.append(str(dp))
        return paths

    # -- depth -------------------------------------------------------------

    def load_depth(self, path: str) -> np.ndarray:
        return _load_thud_depth_as_meters(path, self._depth_scale, self._depth_max_m)

    # -- camera ------------------------------------------------------------

    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        if self._K is not None:
            return self._K, self._img_h, self._img_w
        return None

    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        fd = self._loader.get_frame_data(frame_idx)
        return fd.cam_transform_4x4

    def get_all_poses(self) -> Optional[List[np.ndarray]]:
        poses = []
        for fidx in self._loader.frame_indices:
            fd = self._loader.get_frame_data(fidx)
            poses.append(fd.cam_transform_4x4)
        if all(p is None for p in poses):
            return None
        return poses

    # -- ground truth ------------------------------------------------------

    def get_gt_instances(self, frame_idx: int):
        fd = self._loader.get_frame_data(frame_idx)
        return [
            GTInstance(
                track_id=g.track_id,
                class_name=g.class_name,
                mask=g.mask,
                bbox_xyxy=g.bbox2d_xyxy,
            )
            for g in fd.gt_objects
        ]

    # -- multi-scene -------------------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        scene_type = kwargs.get("scene_type", "static")
        return discover_thud_synthetic_scenes(root, scene_type=scene_type)
