"""ScanNet++ dataset loader.

ScanNet++ directory layout (typical)::

    scene/
        color/  (or rgb/)
            frame000000.jpg  ...
        depth/
            frame000000.png  ...
        pose/
            frame000000.txt  ...   (4x4 matrix, space-separated)
        intrinsic/
            intrinsic_color.txt    (4x4 matrix)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from core.types import CameraIntrinsics
from depth_providers.base import DepthProvider

from .base import DatasetLoader

_FRAME_RE = re.compile(r"^frame(\d+)\.(jpg|png)$")


class ScanNetPPLoader(DatasetLoader):
    """Loader for ScanNet++ scenes.

    Supports the typical ScanNet++ DSLR export layout with per-frame
    pose files and an intrinsic matrix file.
    """

    def __init__(
        self,
        scene_dir: str,
        depth_provider: DepthProvider,
        skip_labels: Optional[Set[str]] = None,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        png_max_value: int = 65535,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels

        # Discover directories
        self._rgb_dir = self._find_rgb_dir()
        self._depth_dir = self._scene_dir / "depth"
        self._pose_dir = self._scene_dir / "pose"

        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)
        self._intrinsics = self._load_intrinsics()

        self._depth_provider = depth_provider

    def _find_rgb_dir(self) -> Path:
        for name in ("color", "rgb", "images"):
            d = self._scene_dir / name
            if d.exists():
                return d
        return self._scene_dir / "color"

    def _load_intrinsics(self) -> CameraIntrinsics:
        # Try intrinsic_color.txt
        intr_dir = self._scene_dir / "intrinsic"
        for fname in ("intrinsic_color.txt", "intrinsic_depth.txt"):
            p = intr_dir / fname
            if p.exists():
                K = _load_4x4_or_3x3(p)
                if K is not None:
                    # Get image size from first frame
                    w, h = self._probe_image_size()
                    return CameraIntrinsics.from_K(K[:3, :3], w, h)

        # Fallback: reasonable defaults
        w, h = self._probe_image_size()
        fx = fy = max(w, h) * 0.8
        return CameraIntrinsics(fx=fx, fy=fy, cx=w / 2.0, cy=h / 2.0,
                                width=w, height=h)

    def _probe_image_size(self) -> Tuple[int, int]:
        if self._frame_numbers:
            fnum = self._frame_numbers[0]
            for ext in ("jpg", "png"):
                p = self._rgb_dir / f"frame{fnum:06d}.{ext}"
                if p.exists():
                    img = cv2.imread(str(p))
                    if img is not None:
                        return img.shape[1], img.shape[0]
        return 1280, 720

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        fnum = self._frame_number(frame_idx)
        for ext in ("jpg", "png"):
            p = self._rgb_dir / f"frame{fnum:06d}.{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is not None and hasattr(self._depth_provider, 'feed_frame'):
                    self._depth_provider.feed_frame(fnum, img)
                return img, str(p)
        return None, ""

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_number(frame_idx)
        return self._depth_provider.get_depth(fnum)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_number(frame_idx)

        # Prefer depth-provider pose
        pred_pose = self._depth_provider.get_pose(fnum)
        if pred_pose is not None:
            return pred_pose

        # Per-frame pose file
        pose_file = self._pose_dir / f"frame{fnum:06d}.txt"
        if pose_file.exists():
            return _load_4x4_or_3x3(pose_file)
        return None

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        scenes = []
        for d in sorted(root_p.iterdir()):
            if d.is_dir() and ((d / "color").exists() or (d / "rgb").exists()):
                scenes.append(str(d))
        return scenes

    def _frame_number(self, frame_idx: int) -> int:
        if 0 <= frame_idx < len(self._frame_numbers):
            return self._frame_numbers[frame_idx]
        return frame_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_frame_numbers(rgb_dir: Path) -> List[int]:
    if not rgb_dir.exists():
        return []
    nums: List[int] = []
    for name in os.listdir(str(rgb_dir)):
        m = _FRAME_RE.match(name)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    return nums


def _load_4x4_or_3x3(path: Path) -> Optional[np.ndarray]:
    """Load a space-separated matrix file (3x3 or 4x4)."""
    try:
        vals = []
        with path.open("r") as f:
            for line in f:
                vals.extend(float(v) for v in line.strip().split())
        if len(vals) == 16:
            return np.array(vals, dtype=np.float64).reshape(4, 4)
        if len(vals) == 9:
            return np.array(vals, dtype=np.float64).reshape(3, 3)
    except Exception:
        pass
    return None
