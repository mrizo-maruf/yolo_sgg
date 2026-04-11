"""ScanNet++ GT depth provider."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..base import DepthProvider
from ..pose_utils import load_poses_txt, lookup_pose


class ScanNetPPDepthProvider(DepthProvider):
    """ScanNet++ GT depth: uint16 PNG where ``depth_metres = pixel / scale``."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "frame_{frame_idx:06d}.png",
        depth_scale: float = 1000.0,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
        pose_lookup: str = "index",
        depth_files: Optional[List[int]] = None,
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._filename_pattern = filename_pattern
        self._depth_scale = depth_scale
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._depth_files = sorted(self._depth_dir.glob("*.png"))
        self._pose_lookup = pose_lookup
        self._poses = load_poses_txt(pose_path)

    def _depth_path(self, frame_idx: int) -> Path:
        return self._depth_files[frame_idx]

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self._depth_path(frame_idx)
        if not path.exists():
            return None
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # uint16, mm
        if arr is None:
            return None
        dm = arr.astype(np.float32) / self._depth_scale
        dm[dm < self._min_depth] = 0.0
        dm[dm > self._max_depth] = 0.0
        return dm

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return lookup_pose(self._poses, frame_idx, self._pose_lookup)
