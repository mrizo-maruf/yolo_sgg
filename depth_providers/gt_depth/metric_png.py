"""Generic metric PNG depth provider."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..base import DepthProvider
from ..pose_utils import load_poses_txt, lookup_pose
from ..sequence_sync import OrderedIndexMap, sorted_files_with_ids


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
        self._depth_files, self._depth_ids = sorted_files_with_ids(self._depth_dir, "depth*.png")
        self._sync = OrderedIndexMap(self._depth_ids)

    def _depth_path(self, frame_idx: int) -> Path:
        filename = self._filename_pattern.format(
            frame_idx=frame_idx,
            frame_number=frame_idx,
        )
        direct = self._depth_dir / filename
        if direct.exists():
            return direct

        if self._pose_lookup == "frame_number":
            ord_idx = self._sync.resolve_frame_number_index(int(frame_idx))
        else:
            ord_idx = self._sync.resolve_index(int(frame_idx))
        if ord_idx is not None and 0 <= ord_idx < len(self._depth_files):
            return self._depth_files[ord_idx]
        return direct

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
        if self._pose_lookup == "frame_number":
            ord_idx = self._sync.resolve_frame_number_index(int(frame_idx))
        else:
            ord_idx = self._sync.resolve_index(int(frame_idx))
        if self._poses is not None and ord_idx is not None and 0 <= ord_idx < len(self._poses):
            return self._poses[ord_idx]
        return lookup_pose(self._poses, frame_idx, self._pose_lookup)

    def get_sync_debug(self, frame_idx: int) -> dict:
        if self._pose_lookup == "frame_number":
            ord_idx = self._sync.resolve_frame_number_index(int(frame_idx))
        else:
            ord_idx = self._sync.resolve_index(int(frame_idx))
        depth_path = str(self._depth_path(frame_idx))
        pose_index = (
            int(ord_idx)
            if (ord_idx is not None and self._poses is not None
                and 0 <= ord_idx < len(self._poses))
            else None
        )
        return {
            "frame_key": int(frame_idx),
            "depth_path": depth_path,
            "pose_index": pose_index,
        }
