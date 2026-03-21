"""IsaacSim dataset loader (v2)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from v2.depth_providers.base import DepthProvider, OnlineDepthProvider
from v2.depth_providers.gt_depth import IsaacSimDepthProvider
from v2.types import CameraIntrinsics

from .base import DatasetLoader

_FRAME_RE = re.compile(r"^frame(\d+)\.(jpg|png)$")


class IsaacSimLoader(DatasetLoader):
    """Adapter for IsaacSim scenes with a pluggable depth provider.

    Directory layout::

        scene/
            rgb/frame000001.jpg  ...
            depth/depth000001.png  ...
            traj.txt
    """

    def __init__(
        self,
        scene_dir: str,
        depth_provider: Optional[DepthProvider] = None,
        depth_provider_type: int = 0,
        skip_labels: Optional[Set[str]] = None,
        image_width: int = 1280,
        image_height: int = 720,
        focal_length: float = 50.0,
        horizontal_aperture: float = 80.0,
        vertical_aperture: float = 45.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels

        self._intrinsics = CameraIntrinsics.from_physical(
            focal_length,
            horizontal_aperture,
            vertical_aperture,
            image_width,
            image_height,
        )

        self._rgb_dir = self._scene_dir / "rgb"
        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)

        if depth_provider is not None:
            self._depth_provider = depth_provider
        else:
            depth_dir = self._scene_dir / "depth"
            if not depth_dir.exists():
                raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
            if depth_provider_type != 0:
                raise ValueError(
                    "Only depth_provider_type=0 (GT) is supported when depth_provider "
                    "is not provided."
                )
            self._depth_provider = IsaacSimDepthProvider(str(depth_dir))

        traj_path = self._scene_dir / "traj.txt"
        self._poses = _load_poses(traj_path) if traj_path.exists() else None

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        frame_number = self._frame_number(frame_idx)

        jpg_path = self._rgb_dir / f"frame{frame_number:06d}.jpg"
        png_path = self._rgb_dir / f"frame{frame_number:06d}.png"
        path = jpg_path if jpg_path.exists() else png_path

        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is not None and isinstance(self._depth_provider, OnlineDepthProvider):
            self._depth_provider.feed_frame(frame_number, img)

        return img, str(path)

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._depth_provider is None:
            return None
        frame_number = self._frame_number(frame_idx)
        return self._depth_provider.get_depth(frame_number)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        frame_number = self._frame_number(frame_idx)

        if self._depth_provider is not None:
            pred_pose = self._depth_provider.get_pose(frame_number)
            if pred_pose is not None:
                return pred_pose

        if self._poses is None:
            return None

        if 0 <= frame_idx < len(self._poses):
            return self._poses[frame_idx]

        if 1 <= frame_number <= len(self._poses):
            return self._poses[frame_number - 1]

        return self._poses[-1] if self._poses else None

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        return sorted(
            str(d)
            for d in root_p.iterdir()
            if d.is_dir() and (d / "rgb").exists()
        )

    def _frame_number(self, frame_idx: int) -> int:
        if 0 <= frame_idx < len(self._frame_numbers):
            return self._frame_numbers[frame_idx]
        return frame_idx


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


def _load_poses(traj_path: Path) -> Optional[List[np.ndarray]]:
    poses = []
    with traj_path.open("r", encoding="utf-8") as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 16:
                continue
            poses.append(np.array(list(map(float, vals)), dtype=np.float32).reshape(4, 4))
    return poses or None
