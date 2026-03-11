"""
IsaacSim dataset loader adapter.

Wraps ``isaacsim_utils.isaac_sim_loader.IsaacSimSceneLoader``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import YOLOE.utils as yutils
from isaacsim_utils.isaac_sim_loader import IsaacSimSceneLoader, FrameData, GTObject
from metrics.tracking_metrics import GTInstance

from .base import DatasetLoader


_DEFAULT_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background",
}


class IsaacSimLoader(DatasetLoader):
    """Adapter for IsaacSim scenes."""

    # Default IsaacSim camera parameters (overridden if cfg values change)
    _IMAGE_WIDTH = 1280
    _IMAGE_HEIGHT = 720
    _FOCAL_LENGTH = 50
    _HORIZONTAL_APERTURE = 80
    _VERTICAL_APERTURE = 45

    def __init__(
        self,
        scene_dir: str,
        load_rgb: bool = False,
        load_depth: bool = False,
        skip_labels: Optional[Set[str]] = None,
        pi3: bool = False,  # Whether to apply load pi3 depth predictions
        dav3: bool = False,  # Whether to apply dav3 depth predictions
        require_all_ids: bool = True,  # Only load objects with all IDs >= 0
        image_width: int = 1280,
        image_height: int = 720,
        focal_length: float = 50,
        horizontal_aperture: float = 80,
        vertical_aperture: float = 45,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS

        # Camera params (can come from cfg via caller)
        self._image_width = image_width
        self._image_height = image_height
        self._focal_length = focal_length
        self._horizontal_aperture = horizontal_aperture
        self._vertical_aperture = vertical_aperture

        self._loader = IsaacSimSceneLoader(
            str(self._scene_dir),
            load_rgb=load_rgb,
            load_depth=load_depth,
            skip_labels=self._skip_labels,
            pi3=pi3,
            dav3=dav3,
            require_all_ids=require_all_ids,
        )
        # Pre-load poses from traj.txt
        traj_path = self._scene_dir / "traj.txt"
        self._poses = (
            yutils.load_camera_poses(str(traj_path)) if traj_path.exists() else None
        )

    # -- metadata ----------------------------------------------------------

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_frame_indices(self) -> List[int]:
        return list(self._loader.frame_indices)

    # -- paths -------------------------------------------------------------

    def get_rgb_paths(self) -> List[str]:
        return sorted(
            str(p)
            for p in self._loader.rgb_dir.glob("*.jpg")
        )

    def get_depth_paths(self) -> List[str]:
        return yutils.list_png_paths(str(self._loader.depth_dir))

    # -- depth -------------------------------------------------------------

    def load_depth(self, path: str) -> np.ndarray:
        return yutils.load_depth_as_meters(path)

    # -- camera ------------------------------------------------------------

    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        w = self._image_width
        h = self._image_height
        _fx = self._focal_length / self._horizontal_aperture * w
        _fy = self._focal_length / self._vertical_aperture * h
        _cx = w / 2.0
        _cy = h / 2.0
        K = np.array([
            [_fx,  0.0, _cx],
            [0.0, _fy, _cy],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64)
        return K, h, w

    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._poses is None:
            return None
        idx = self.get_frame_indices().index(frame_idx) if frame_idx in self.get_frame_indices() else 0
        return self._poses[min(idx, len(self._poses) - 1)]

    def get_all_poses(self) -> Optional[List[np.ndarray]]:
        return self._poses

    # -- ground truth ------------------------------------------------------

    def get_gt_instances(self, frame_idx: int):
        fd: FrameData = self._loader.get_frame_data(frame_idx)
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
        root_p = Path(root)
        return sorted(
            str(d)
            for d in root_p.iterdir()
            if d.is_dir() and (d / "rgb").exists() and (d / "depth").exists()
        )
