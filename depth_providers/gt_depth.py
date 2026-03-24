"""Ground-truth / pre-computed depth providers."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .base import DepthProvider


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _load_poses_txt(pose_path: Optional[str]) -> Optional[list[np.ndarray]]:
    if pose_path is None:
        return None
    path = Path(pose_path)
    if not path.exists():
        return None

    poses: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 16:
                continue
            poses.append(
                np.array(list(map(float, vals)), dtype=np.float32).reshape(4, 4)
            )
    return poses or None


def _lookup_pose(
    poses: Optional[list[np.ndarray]],
    frame_idx: int,
    mode: str,
) -> Optional[np.ndarray]:
    if poses is None:
        return None
    if mode == "index":
        idx = frame_idx
    elif mode == "frame_number":
        idx = frame_idx - 1
    else:
        raise ValueError(f"Unknown pose lookup mode: {mode}")
    if 0 <= idx < len(poses):
        return poses[idx]
    return None


# ---------------------------------------------------------------------------
# Generic metric PNG depth
# ---------------------------------------------------------------------------

class MetricPngDepthProvider(DepthProvider):
    """Metric depth from 16-bit PNG: depth_m = (png / png_max) * max_depth."""

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
        self._poses = _load_poses_txt(pose_path)

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
        return _lookup_pose(self._poses, frame_idx, self._pose_lookup)


# ---------------------------------------------------------------------------
# Dataset-specific GT providers
# ---------------------------------------------------------------------------

class IsaacSimDepthProvider(MetricPngDepthProvider):
    """IsaacSim GT depth (16-bit PNG -> metres)."""

    def __init__(
        self,
        depth_dir: str,
        png_max_value: int = 65535,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
        pose_lookup: str = "frame_number",
    ) -> None:
        super().__init__(
            depth_dir=depth_dir,
            filename_pattern="depth{frame_idx:06d}.png",
            png_max_value=png_max_value,
            max_depth=max_depth,
            min_depth=min_depth,
            pose_path=pose_path,
            pose_lookup=pose_lookup,
        )


class ScanNetPPDepthProvider(DepthProvider):
    """ScanNet++ GT depth: uint16 PNG where depth_metres = pixel / scale."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "frame_{frame_idx:06d}.png",
        depth_scale: float = 1000.0,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
        pose_lookup: str = "index",
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._filename_pattern = filename_pattern
        self._depth_scale = depth_scale
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._pose_lookup = pose_lookup
        self._poses = _load_poses_txt(pose_path)

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
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # uint16, mm
        if arr is None:
            return None
        dm = arr.astype(np.float32) / self._depth_scale     # -> metres
        dm[dm < self._min_depth] = 0.0
        dm[dm > self._max_depth] = 0.0
        return dm

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return _lookup_pose(self._poses, frame_idx, self._pose_lookup)


class CODaDepthProvider(MetricPngDepthProvider):
    """CODa GT depth."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "depth{frame_idx:06d}.png",
        png_max_value: int = 65535,
        max_depth: float = 80.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
        pose_lookup: str = "frame_number",
    ) -> None:
        super().__init__(
            depth_dir=depth_dir,
            filename_pattern=filename_pattern,
            png_max_value=png_max_value,
            max_depth=max_depth,
            min_depth=min_depth,
            pose_path=pose_path,
            pose_lookup=pose_lookup,
        )


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
        self._poses = _load_poses_txt(pose_path)
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
        return _lookup_pose(self._poses, frame_idx, self._pose_lookup)

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
