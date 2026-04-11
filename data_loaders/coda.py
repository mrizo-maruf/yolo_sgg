"""CODa dataset loader."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from core.types import CameraIntrinsics
from depth_providers.base import DepthProvider
from depth_providers.pose_utils import load_poses_txt

from .base import DatasetLoader

_FRAME_RE = re.compile(r"^(?:frame|rgb)(\d+)\.(jpg|png)$")


class CODaLoader(DatasetLoader):
    """Loader for CODa scenes.

    Directory layout::

        scene/
            rgb/frame000001.png  (or rgb000001.png)
            depth/depth000001.png
            traj.txt  (optional)
    """

    def __init__(
        self,
        scene_dir: str,
        depth_provider: DepthProvider,
        skip_labels: Optional[Set[str]] = None,
        image_width: int = 1280,
        image_height: int = 720,
        fx: float = 600.0,
        fy: float = 600.0,
        cx: float = 640.0,
        cy: float = 360.0,
        max_depth: float = 80.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels

        self._intrinsics = CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=image_width, height=image_height,
        )

        self._rgb_dir = self._scene_dir / "rgb"
        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)

        self._depth_provider = depth_provider

        traj_path = self._scene_dir / "traj.txt"
        self._poses = load_poses_txt(str(traj_path)) if traj_path.exists() else None

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        fnum = self._frame_number(frame_idx)
        # Try multiple naming patterns
        for pat in [f"frame{fnum:06d}.png", f"frame{fnum:06d}.jpg",
                    f"rgb{fnum:06d}.png", f"rgb{fnum:06d}.jpg"]:
            p = self._rgb_dir / pat
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
        pred_pose = self._depth_provider.get_pose(fnum)
        if pred_pose is not None:
            return pred_pose
        if self._poses is not None and 0 <= frame_idx < len(self._poses):
            return self._poses[frame_idx]
        return None

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        return sorted(
            str(d) for d in root_p.iterdir()
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
