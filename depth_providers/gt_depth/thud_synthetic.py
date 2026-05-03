"""THUD Synthetic GT depth provider."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..base import DepthProvider
from ..pose_utils import load_poses_txt, lookup_pose


class THUDSyntheticDepthProvider(DepthProvider):
    """THUD Synthetic GT depth — raw uint16 with custom unprojection."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "depth_{frame_idx}.png",
        pose_path: Optional[str] = None,
        pose_lookup: str = "frame_number",
        scale: Optional[float] = None,
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._filename_pattern = filename_pattern
        self._pose_lookup = pose_lookup
        self._poses = load_poses_txt(pose_path)
        self._scale = scale

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
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return d.astype(np.float32)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return lookup_pose(self._poses, frame_idx, self._pose_lookup)

    def unproject(
        self,
        us: np.ndarray,
        vs: np.ndarray,
        depths: np.ndarray,
        intrinsics,
    ) -> np.ndarray:
        """THUD-specific depth-to-3D conversion."""
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy
        H = intrinsics.height

        us_f = us.astype(np.float32)
        vs_flip = (H - 1 - vs).astype(np.float32)

        Xr = (us_f - cx) * depths / fx
        Zr = (vs_flip - cy) * depths / fy

        X = Xr / 2.5 / 1000.0
        Y = (depths / 6.5 + 200.0) / 1000.0
        Z = (Zr / 2.0 + 300.0) / 1000.0

        return np.stack([X, Y, Z], axis=1).astype(np.float32)
