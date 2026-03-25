"""Offline Pi3 depth providers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import DepthProvider
from .gt_depth import _load_poses_txt, _lookup_pose


_DEPTH_SCALE_RE = re.compile(r"png_depth_scale:\s*([0-9eE+.\-]+)")


class IsaacSimOfflinePi3DepthProvider(DepthProvider):
    """Load offline Pi3 depth and align poses to GT world via Sim(3).

    Depth format (from Pi3 export):
        depth_m = depth_png_uint16 * png_depth_scale

    Pose alignment:
        T_world_cam_aligned = T_sim3_pi3_to_world @ T_pi3_world_cam
    """

    def __init__(
        self,
        depth_dir: str,
        pose_path: Optional[str] = None,
        transform_path: Optional[str] = None,
        png_depth_scale: Optional[float] = None,
        min_depth: float = 0.01,
        max_depth: float = 100.0,
        pose_lookup: str = "frame_number",
        require_transform: bool = True,
    ) -> None:
        self._depth_dir = Path(depth_dir)
        self._pose_lookup = pose_lookup
        self._min_depth = float(min_depth)
        self._max_depth = float(max_depth)
        self._poses = _load_poses_txt(pose_path)
        self._depth_files = sorted(self._depth_dir.glob("depth*.png"))

        if png_depth_scale is None:
            self._png_depth_scale = self._read_png_depth_scale_from_meta()
        else:
            self._png_depth_scale = float(png_depth_scale)
        if self._png_depth_scale <= 0.0:
            raise ValueError(f"png_depth_scale must be > 0, got {self._png_depth_scale}")

        if transform_path is None:
            if require_transform:
                raise FileNotFoundError(
                    "transform_path is required for IsaacSimOfflinePi3DepthProvider."
                )
            self._sim3 = np.eye(4, dtype=np.float32)
        else:
            self._sim3 = self._load_sim3_matrix(transform_path, require_transform)

    def _read_png_depth_scale_from_meta(self) -> float:
        """Read scale from Pi3 metadata; fallback to 1 mm/unit."""
        for name in ("pi3_depth_meta.txt", "depth_scale.txt", "meta.txt"):
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
        # Default matches Pi3 exporter default (1 unit = 1 mm).
        return 0.001

    @staticmethod
    def _load_sim3_matrix(path_str: str, require: bool) -> np.ndarray:
        path = Path(path_str)
        if not path.exists():
            if require:
                raise FileNotFoundError(f"Pi3 alignment transform not found: {path}")
            return np.eye(4, dtype=np.float32)

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if "sim3_matrix_4x4" in payload:
            sim3 = np.asarray(payload["sim3_matrix_4x4"], dtype=np.float64)
            if sim3.shape != (4, 4):
                raise ValueError(
                    f"sim3_matrix_4x4 must be 4x4, got shape {sim3.shape}"
                )
        else:
            scale = float(payload["scale"])
            rotation = np.asarray(payload["rotation"], dtype=np.float64)
            translation = np.asarray(payload["translation"], dtype=np.float64)
            if rotation.shape != (3, 3):
                raise ValueError(f"rotation must be 3x3, got shape {rotation.shape}")
            if translation.shape != (3,):
                raise ValueError(
                    f"translation must be shape (3,), got shape {translation.shape}"
                )
            sim3 = np.eye(4, dtype=np.float64)
            sim3[:3, :3] = scale * rotation
            sim3[:3, 3] = translation

        if not np.isfinite(sim3).all():
            raise ValueError(f"Invalid values in Sim(3) transform: {path}")

        return sim3.astype(np.float32)

    def _depth_path(self, frame_idx: int) -> Path:
        # Pi3 offline export uses depth000000.png for the first RGB frame,
        # while IsaacSim frame numbers usually start from 1.
        candidates: list[int] = []
        if frame_idx > 0:
            candidates.append(frame_idx - 1)
        candidates.append(frame_idx)

        for idx in candidates:
            if idx < 0:
                continue
            p = self._depth_dir / f"depth{idx:06d}.png"
            if p.exists():
                return p

        # Fallback: index by sorted file order.
        if self._depth_files:
            ord_idx = frame_idx - 1 if self._pose_lookup == "frame_number" else frame_idx
            if 0 <= ord_idx < len(self._depth_files):
                return self._depth_files[ord_idx]

        return self._depth_dir / f"depth{frame_idx:06d}.png"

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
        pose = _lookup_pose(self._poses, frame_idx, self._pose_lookup)
        if pose is None:
            return None
        pose = pose.astype(np.float32)
        return (self._sim3 @ pose).astype(np.float32)

    def get_sim3_matrix(self) -> np.ndarray:
        return self._sim3.copy()

    @property
    def png_depth_scale(self) -> float:
        return self._png_depth_scale
