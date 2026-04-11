"""Generic metric PNG depth provider."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..base import DepthProvider
from ..pose_utils import load_poses_txt, lookup_pose


class MetricPngDepthProvider(DepthProvider):
    """Metric depth from 16-bit PNG: ``depth_m = (png / png_max) * max_depth``."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str,
        png_max_value: int = 65535,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
        pose_lookup: str = "frame_number",
    ) -> None:
        if png_max_value <= 0:
            raise ValueError("png_max_value must be > 0")
        self._depth_dir = Path(depth_dir)
        self._filename_pattern = filename_pattern
        self._png_max = float(png_max_value)
        self._max_depth = float(max_depth)
        self._min_depth = float(min_depth)
        self._pose_lookup = pose_lookup
        self._poses = load_poses_txt(pose_path)

    def _depth_path(self, frame_idx: int) -> Path:
        filename = self._filename_pattern.format(
            frame_idx=frame_idx,
            frame_number=frame_idx,
        )
        return self._depth_dir / filename

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self._depth_path(frame_idx)
        if not path.exists():
            return None
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        dm = (arr.astype(np.float32) / self._png_max) * self._max_depth
        dm[dm < self._min_depth] = 0.0
        dm[dm > self._max_depth] = 0.0
        return dm.astype(np.float32)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return lookup_pose(self._poses, frame_idx, self._pose_lookup)
