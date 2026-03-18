"""
THUD Synthetic dataset loader (v2).

Lazy — paths are constructed from the known naming convention.
Camera intrinsics and poses are parsed from ``Label/captures_*.json``
(matching the original THUDSyntheticLoader).

Directory layout::

    scene/
        RGB/rgb_2.png  rgb_3.png  ...
        Depth/depth_2.png  depth_3.png  ...
        Label/
            captures_000.json  captures_001.json  ...
            Instance/Instance_2.png  ...
            Semantic/segmentation_2.png  ...
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from v2.depth_providers.base import DepthProvider
from v2.depth_providers.gt_depth import THUDSyntheticDepthProvider
from v2.types import CameraIntrinsics

from .base import DatasetLoader


_DEFAULT_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background",
}

_RGB_RE = re.compile(r"^rgb_(\d+)\.png$")


class THUDSyntheticLoader(DatasetLoader):
    """Adapter for THUD Synthetic scenes with pluggable depth provider.

    Frame indexing
    --------------
    RGB/depth files use integer ``N`` (``rgb_N.png``).  JSON annotations
    use ``step`` where ``step = N - 2``.
    """

    def __init__(
        self,
        scene_dir: str,
        depth_provider: Optional[DepthProvider] = None,
        skip_labels: Optional[Set[str]] = None,
        depth_scale: float = 1000.0,
    ) -> None:
        self._scene_dir = Path(scene_dir)
        self._skip_labels = skip_labels or _DEFAULT_SKIP_LABELS

        # Canonical folder paths (uppercase, matching the THUD layout)
        self._rgb_dir = self._scene_dir / "RGB"
        self._depth_dir = self._scene_dir / "Depth"
        self._label_dir = self._scene_dir / "Label"

        # Discover frame numbers from RGB/ filenames (ints only)
        self._frame_numbers: List[int] = _discover_frame_numbers(self._rgb_dir)

        # Camera data from Label/captures_*.json  (intrinsics + per-step poses)
        # _cam_data[step] = {"intrinsic": 3x3|None, "translation": (x,y,z)|None,
        #                    "rotation": (qx,qy,qz,qw)|None}
        self._cam_data: Dict[int, Dict[str, Any]] = _load_camera_data(self._label_dir)
        self._intrinsics = self._resolve_intrinsics()

        # Depth provider
        if depth_provider is not None:
            self._depth_provider = depth_provider
        elif self._depth_dir.exists():
            self._depth_provider = THUDSyntheticDepthProvider(
                str(self._depth_dir), scale=depth_scale,
            )
        else:
            self._depth_provider = None

    # -- DatasetLoader interface -------------------------------------------

    @property
    def scene_label(self) -> str:
        parts = self._scene_dir.parts
        try:
            idx = next(
                i for i, p in enumerate(parts)
                if p in ("Gym", "House", "Office", "Supermarket_1", "Supermarket_2")
            )
            return "/".join(parts[idx:])
        except StopIteration:
            return self._scene_dir.name

    def get_num_frames(self) -> int:
        return len(self._frame_numbers)

    def get_rgb(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        fnum = self._frame_numbers[frame_idx]
        path = str(self._rgb_dir / f"rgb_{fnum}.png")
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, path

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._depth_provider is None:
            return None
        fnum = self._frame_numbers[frame_idx]
        return self._depth_provider.get_depth(fnum)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        fnum = self._frame_numbers[frame_idx]
        step = fnum - 2
        cam = self._cam_data.get(step)
        if cam is None:
            return None
        t = cam.get("translation")
        r = cam.get("rotation")
        if t is None or r is None:
            return None
        return _compose_transform_4x4(t, r)

    def get_intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    # -- internals ---------------------------------------------------------

    def _resolve_intrinsics(self) -> CameraIntrinsics:
        """Pick intrinsics from the first annotated step that has them."""
        for step in sorted(self._cam_data):
            K = self._cam_data[step].get("intrinsic")
            if K is not None:
                # Detect image size from a single file
                if self._frame_numbers:
                    fnum = self._frame_numbers[0]
                    p = self._rgb_dir / f"rgb_{fnum}.png"
                    img = cv2.imread(str(p))
                    if img is not None:
                        h, w = img.shape[:2]
                        return CameraIntrinsics.from_K(K, w, h)
                return CameraIntrinsics.from_K(K, 730, 530)
        # Fallback
        return CameraIntrinsics(fx=500.0, fy=500.0, cx=365.0, cy=265.0,
                                width=730, height=530)

    # -- multi-scene -------------------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        root_p = Path(root)
        scenes = []
        for cap_dir in sorted(root_p.rglob("Capture_*")):
            if (cap_dir / "RGB").exists():
                scenes.append(str(cap_dir))
        return scenes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _discover_frame_numbers(rgb_dir: Path) -> List[int]:
    """Fast scan of ``RGB/`` for ``rgb_N.png`` → sorted integer list."""
    if not rgb_dir.exists():
        return []
    nums: List[int] = []
    for name in os.listdir(str(rgb_dir)):
        m = _RGB_RE.match(name)
        if m:
            nums.append(int(m.group(1)))
    nums.sort()
    return nums


def _load_camera_data(label_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Parse ``captures_*.json`` and extract only camera info per step."""
    cam: Dict[int, Dict[str, Any]] = {}
    if not label_dir.exists():
        return cam
    for jf in sorted(label_dir.glob("captures_*.json")):
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for capture in data.get("captures", []):
            step = int(capture.get("step", -1))
            if step < 0:
                continue
            d = cam.setdefault(step, {})
            sensor = capture.get("sensor", {})
            if "intrinsic" not in d:
                intr = sensor.get("camera_intrinsic")
                if intr:
                    try:
                        arr = np.array(intr, dtype=np.float64)
                        if arr.shape == (3, 3):
                            d["intrinsic"] = arr
                    except Exception:
                        pass
            if "translation" not in d:
                t = sensor.get("translation")
                r = sensor.get("rotation")
                if t and len(t) == 3:
                    d["translation"] = tuple(float(v) for v in t)
                if r and len(r) == 4:
                    d["rotation"] = tuple(float(v) for v in r)
    return cam


def _quat_to_rotation_matrix(
    qx: float, qy: float, qz: float, qw: float,
) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def _compose_transform_4x4(
    translation: Tuple[float, float, float],
    rotation_xyzw: Tuple[float, float, float, float],
) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _quat_to_rotation_matrix(*rotation_xyzw)
    T[:3, 3] = translation
    return T
