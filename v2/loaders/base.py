"""
Abstract base for v2 dataset loaders.

Each loader owns a :class:`DepthProvider` and exposes a uniform
interface for the tracking pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from v2.depth_providers.base import DepthProvider
from v2.types import CameraIntrinsics


class DatasetLoader(ABC):
    """Protocol every dataset loader implements."""

    # -- metadata ----------------------------------------------------------

    @property
    @abstractmethod
    def scene_label(self) -> str: ...

    @abstractmethod
    def get_num_frames(self) -> int: ...

    # -- per-frame access --------------------------------------------------

    @abstractmethod
    def get_rgb(self, frame_idx: int) -> Tuple[np.ndarray, str]:
        """Return (H,W,3) uint8 RGB array and its file path."""
        ...

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return (H,W) float32 depth in metres via the depth provider."""
        if self.depth_provider is None:
            return None
        return self.depth_provider.get_depth(frame_idx)

    @abstractmethod
    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """4×4 camera-to-world transform."""
        ...

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics: ...

    # -- depth provider (set by constructor) --------------------------------

    @property
    def depth_provider(self) -> Optional[DepthProvider]:
        return getattr(self, "_depth_provider", None)

    # -- convenience -------------------------------------------------------

    def get_gt_instances(self, frame_idx: int):
        return None

    # -- multi-scene discovery ----------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        return []
