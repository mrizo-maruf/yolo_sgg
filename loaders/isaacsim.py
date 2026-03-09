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

    def __init__(
        self,
        scene_dir: str,
        skip_labels: Optional[Set[str]] = None,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS
        self._loader = IsaacSimSceneLoader(
            str(self._scene_dir),
            load_rgb=True,
            skip_labels=self._skip_labels,
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
        # IsaacSim uses the global defaults in YOLOE.utils
        return None

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
