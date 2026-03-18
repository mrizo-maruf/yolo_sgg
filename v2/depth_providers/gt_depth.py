"""
Ground-truth / pre-computed depth providers.

Lazy loading — paths are constructed from known naming conventions,
never globbed.  Only loaded when ``get_depth`` is called.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from .base import DepthProvider


class IsaacSimDepthProvider(DepthProvider):
    """16-bit PNG depth → metres (IsaacSim convention).

    File naming: ``depth{frame_number:06d}.png`` (1-based frame numbers).

    depth_metres = (uint16_value / png_max_value) * max_depth
    """

    def __init__(
        self,
        depth_dir: str,
        png_max_value: int = 65535,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._png_max = float(png_max_value)
        self._max_depth = max_depth
        self._min_depth = min_depth

    def get_depth(self, frame_number: int) -> Optional[np.ndarray]:
        """Load depth for *frame_number* (the actual file number, 1-based)."""

        path = self._depth_dir / f"depth{frame_number:06d}.png"
        if not path.exists():
            return None
        dm = self._load(str(path))
        return dm

    def _load(self, path: str) -> np.ndarray:
        d = np.array(Image.open(path))
        if d.dtype != np.uint16:
            d = d.astype(np.uint16)
        dm = (d.astype(np.float32) / self._png_max) * self._max_depth
        dm[dm < self._min_depth] = 0.0
        dm[dm > self._max_depth] = 0.0
        return dm

    def get_depth_pcd_from_masks(
        self,
        frame_idx: int,
        intrinsics,
        masks: List[np.ndarray],
        max_points: int = 0,
        seed: int = 0,
    ) -> Optional[List[np.ndarray]]:
        """IsaacSim: pinhole unproject once, extract per-mask pcds."""
        depth = self.get_depth(frame_idx)
        if depth is None:
            return None

        H, W = depth.shape
        us_all, vs_all = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        valid = depth > 0
        X = (us_all - intrinsics.cx) * depth / intrinsics.fx
        Y = (vs_all - intrinsics.cy) * depth / intrinsics.fy
        xyz = np.stack([X, Y, depth], axis=-1).astype(np.float32)  # (H,W,3)

        rng = np.random.default_rng(seed)
        results: List[np.ndarray] = []
        for mask in masks:
            m = valid & np.squeeze(mask).astype(bool)
            pts = xyz[m]  # (N, 3)
            if max_points > 0 and len(pts) > max_points:
                idx = rng.choice(len(pts), size=max_points, replace=False)
                pts = pts[idx]
            results.append(pts)
        return results


class THUDSyntheticDepthProvider(DepthProvider):
    """THUD Synthetic depth provider.

    File naming: ``depth_{frame_number}.png``

    Raw uint16 depth values are returned as-is (float32).  The custom
    THUD unprojection formula is applied in :meth:`unproject`, matching
    the original ``Depth_to_pointcloud.py`` + ``ExportPointCloud.py``
    pipeline so that resulting 3-D points live in the same
    **(x-right, y-forward, z-up)** frame as the 3-D bounding boxes.
    """

    def __init__(
        self,
        depth_dir: str,
    ) -> None:
        self._depth_dir = Path(depth_dir)

    def get_depth(self, frame_number: int) -> Optional[np.ndarray]:
        """Load raw depth for *frame_number* (matches ``depth_{N}.png``).

        Returns uint16 values as float32 — no metric conversion.
        """
        path = self._depth_dir / f"depth_{frame_number}.png"
        if not path.exists():
            return None
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return d.astype(np.float32)

    def unproject(
        self,
        us: np.ndarray,
        vs: np.ndarray,
        depths: np.ndarray,
        intrinsics,
    ) -> np.ndarray:
        """THUD-specific depth-to-3D.

        Replicates the THUD ``Depth_to_pointcloud.py`` +
        ``ExportPointCloud.py`` formula.  The v-axis is flipped and
        custom scale/offset constants convert raw depth values into
        world-frame metres.
        """
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy
        H = intrinsics.height

        us_f = us.astype(np.float32)
        # Flip v (top-to-bottom) as in the THUD pipeline
        vs_flip = (H - 1 - vs).astype(np.float32)

        Xr = (us_f - cx) * depths / fx
        Zr = (vs_flip - cy) * depths / fy

        # Apply THUD scale/offset then mm → m  (ExportPointCloud / 1000)
        X = Xr / 2.5 / 1000.0
        Y = (depths / 6.5 + 200.0) / 1000.0   # forward (depth direction)
        Z = (Zr / 2.0 + 300.0) / 1000.0       # up (flipped-v direction)

        return np.stack([X, Y, Z], axis=1).astype(np.float32)

    def get_depth_pcd_from_masks(
        self,
        frame_idx: int,
        intrinsics,
        masks: List[np.ndarray],
        max_points: int = 0,
        seed: int = 0,
    ) -> Optional[List[np.ndarray]]:
        """THUD: custom unproject once, extract per-mask pcds."""
        depth = self.get_depth(frame_idx)
        if depth is None:
            return None

        H, W = depth.shape
        us_all, vs_all = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
        )
        valid = depth > 0

        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.cx, intrinsics.cy
        vs_flip = (H - 1 - vs_all).astype(np.float32)

        Xr = (us_all - cx) * depth / fx
        Zr = (vs_flip - cy) * depth / fy

        X = Xr / 2.5 / 1000.0
        Y = (depth / 6.5 + 200.0) / 1000.0
        Z = (Zr / 2.0 + 300.0) / 1000.0
        xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)  # (H,W,3)

        rng = np.random.default_rng(seed)
        results: List[np.ndarray] = []
        for mask in masks:
            m = valid & np.squeeze(mask).astype(bool)
            pts = xyz[m]  # (N, 3)
            if max_points > 0 and len(pts) > max_points:
                idx = rng.choice(len(pts), size=max_points, replace=False)
                pts = pts[idx]
            results.append(pts)
        return results
