"""
IsaacSim dataset loader (v2).

Lazy — paths are constructed from the known naming convention, never
globbed.  Frame numbers are discovered once at init (lightweight
``os.listdir`` + int parse).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from v2.depth_providers.base import DepthProvider
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
        depth_provider_type: int,
        depth_provider: Optional[DepthProvider] = None,
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
            focal_length, horizontal_aperture, vertical_aperture,
            image_width, image_height,
        )

        # Discover frame numbers from rgb/ filenames (ints only, no Path objects)
        self._rgb_dir = self._scene_dir / "rgb"
        self._frame_numbers: List[int] = _discover_scene_lenth(self._rgb_dir)

        # Depth provider: default to GT if not supplied
        if depth_provider is not None:
            self._depth_provider = depth_provider
        else:
            depth_dir = self._scene_dir / "depth"
            if depth_dir.exists():
                if depth_provider_type == 0:
                    self._depth_provider = IsaacSimDepthProvider(str(depth_dir))
                elif depth_provider_type == 1:
                    pass
                    # TODO: add support for depth provider type 1 (e.g., a learned depth model) offline
                elif depth_provider_type == 2:
                    pass
                    # TODO: add support for depth provider type 2 (e.g., a learned depth model) online
            else:
                raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
                # self._depth_provider = None

        # Poses from traj.txt (one per line, maps 1-to-1 with sorted frame numbers)
        traj_path = self._scene_dir / "traj.txt"
        self._poses = _load_poses(traj_path) if traj_path.exists() else None

    # -- DatasetLoader interface -------------------------------------------

    @property
    def scene_label(self) -> str:
        return self._scene_dir.name

    def get_num_frames(self) -> int:
        return self._frame_numbers

    def get_rgb(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        path = str(self._rgb_dir / f"frame{frame_idx:06d}.jpg")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, path

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._depth_provider is None:
            raise ValueError("Depth provider not initialized for this dataset.")
            return None
        return self._depth_provider.get_depth(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._poses is None:
            return None
        return self._poses[min(frame_idx, len(self._poses) - 1)]

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    # -- multi-scene -------------------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        return sorted(
            str(d) for d in root_p.iterdir()
            if d.is_dir() and (d / "rgb").exists()
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _discover_scene_lenth(rgb_dir: Path) -> int:
    """Scan filenames in *rgb_dir* and return the number of frames."""
    if not rgb_dir.exists():
        return 0
    nums: List[int] = []
    for name in os.listdir(str(rgb_dir)):
        m = _FRAME_RE.match(name)
        if m:
            nums.append(int(m.group(1)))
    return len(nums)


def _load_poses(traj_path: Path) -> Optional[List[np.ndarray]]:
    poses = []
    with open(traj_path) as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 16:
                continue
            poses.append(np.array(list(map(float, vals)), dtype=np.float64).reshape(4, 4))
    return poses or None
