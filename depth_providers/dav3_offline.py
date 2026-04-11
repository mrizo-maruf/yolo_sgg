"""Offline DepthAnythingV3 depth provider."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import DepthProvider
from .pose_utils import load_poses_txt, lookup_pose

_DEPTH_SCALE_RE = re.compile(r"png_depth_scale:\s*([0-9eE+.\-]+)")


class DAv3OfflineDepthProvider(DepthProvider):
    """Load offline DAv3 depth PNGs (metric) with optional poses."""

    def __init__(
        self,
        depth_dir: str,
        *,
        filename_pattern: str = "depth{frame_idx:06d}.png",
        pose_path: Optional[str] = None,
        png_depth_scale: Optional[float] = None,
        min_depth: float = 0.01,
        max_depth: float = 100.0,
        pose_lookup: str = "frame_number",
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._filename_pattern = filename_pattern
        self._pose_lookup = pose_lookup
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._poses = load_poses_txt(pose_path)
        self._depth_files = sorted(self._depth_dir.glob("depth*.png"))

        if png_depth_scale is None:
            self._png_depth_scale = self._read_png_depth_scale_from_meta()
        else:
            self._png_depth_scale = float(png_depth_scale)
        if self._png_depth_scale <= 0.0:
            raise ValueError(f"png_depth_scale must be > 0, got {self._png_depth_scale}")

    def _read_png_depth_scale_from_meta(self) -> float:
        """Read scale from metadata; fallback to 1 mm/unit."""
        for name in ("dav3_depth_meta.txt", "depth_scale.txt", "meta.txt"):
            path = self._depth_dir / name
            if not path.exists():
                continue
            try:
                txt = path.read_text(encoding="utf-8")
            except Exception:
                continue
            m = _DEPTH_SCALE_RE.search(txt)
            if m:
                try:
                    value = float(m.group(1))
                    if value > 0:
                        return value
                except ValueError:
                    continue
        return 0.001

    def _depth_path(self, frame_idx: int) -> Path:
        filename = self._filename_pattern.format(
            frame_idx=frame_idx,
            frame_number=frame_idx,
        )
        direct = self._depth_dir / filename
        if direct.exists():
            return direct

        candidates: list[int] = []
        if frame_idx > 0:
            candidates.append(frame_idx - 1)
        candidates.append(frame_idx)
        for idx in candidates:
            if idx < 0:
                continue
            alt = self._depth_dir / self._filename_pattern.format(
                frame_idx=idx,
                frame_number=idx,
            )
            if alt.exists():
                return alt

        if self._depth_files:
            ord_idx = frame_idx - 1 if self._pose_lookup == "frame_number" else frame_idx
            if 0 <= ord_idx < len(self._depth_files):
                return self._depth_files[ord_idx]

        return direct

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        path = self._depth_path(frame_idx)
        if not path.exists():
            return None

        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]

        dm = arr.astype(np.float32) * self._png_depth_scale
        dm[~np.isfinite(dm)] = 0.0
        dm[dm < self._min_depth] = 0.0
        if self._max_depth > 0.0:
            dm[dm > self._max_depth] = 0.0
        return dm.astype(np.float32)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return lookup_pose(self._poses, frame_idx, self._pose_lookup)

    @property
    def png_depth_scale(self) -> float:
        return self._png_depth_scale
