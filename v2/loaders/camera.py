"""Live camera loader (v2)."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from v2.depth_providers.base import DepthProvider, OnlineDepthProvider
from v2.types import CameraIntrinsics

from .base import DatasetLoader


class CameraLoader(DatasetLoader):
    """Streams frames from a live camera with an online depth provider."""

    def __init__(
        self,
        camera_id: int = 0,
        intrinsics: Optional[CameraIntrinsics] = None,
        depth_provider: Optional[DepthProvider] = None,
        max_frames: int = 1000,
    ) -> None:
        self._camera_id = camera_id
        self._max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None

        self._intrinsics = intrinsics
        self._depth_provider = depth_provider

        self._rgb_cache: dict[int, np.ndarray] = {}
        self._frame_count = 0

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._camera_id)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self._camera_id}")
            if self._intrinsics is None:
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fx = fy = w * 0.8
                self._intrinsics = CameraIntrinsics(
                    fx=fx,
                    fy=fy,
                    cx=w / 2.0,
                    cy=h / 2.0,
                    width=w,
                    height=h,
                )
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._depth_provider is not None:
            self._depth_provider.close()

    @property
    def scene_label(self) -> str:
        return f"camera_{self._camera_id}"

    def get_num_frames(self) -> int:
        return self._max_frames

    def get_rgb(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        if frame_idx in self._rgb_cache:
            return self._rgb_cache[frame_idx], f"camera:{frame_idx}"

        cap = self._ensure_open()
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to capture frame {frame_idx}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._rgb_cache[frame_idx] = rgb
        self._frame_count = max(self._frame_count, frame_idx + 1)

        if isinstance(self._depth_provider, OnlineDepthProvider):
            self._depth_provider.feed_frame(frame_idx, rgb)

        return rgb, f"camera:{frame_idx}"

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        if self._depth_provider is not None:
            pose = self._depth_provider.get_pose(frame_idx)
            if pose is not None:
                return pose
        return None

    def get_intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            self._ensure_open()
        assert self._intrinsics is not None
        return self._intrinsics
