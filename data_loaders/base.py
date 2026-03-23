"""
Abstract base class for dataset loaders.

Each loader owns a DepthProvider and exposes a uniform interface
for the tracking pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from core.types import CameraIntrinsics
from depth_providers.base import DepthProvider


class DatasetLoader(ABC):
    """Protocol every dataset loader implements."""

    # -- metadata ----------------------------------------------------------

    @property
    @abstractmethod
    def scene_label(self) -> str:
        """Human-readable label for the scene."""
        ...

    @abstractmethod
    def get_num_frames(self) -> int:
        """Total number of frames."""
        ...

    def get_scene_name(self) -> str:
        """Short scene name (used for per-scene class-name lookup)."""
        return self.scene_label

    # -- per-frame access --------------------------------------------------

    @abstractmethod
    def get_rgb(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        """Return (H,W,3) uint8 RGB array and its file path."""
        ...

    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return (H,W) float32 depth in metres via the depth provider."""
        return self.depth_provider.get_depth(frame_idx)

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """4x4 camera-to-world transform.

        Default: delegate to depth provider (for providers that predict
        poses, e.g. Pi3). Subclasses with GT poses should override.
        """
        return self.depth_provider.get_pose(frame_idx)

    @abstractmethod
    def get_intrinsics(self) -> CameraIntrinsics:
        """Camera intrinsics."""
        ...

    # -- masked PCDs (main 3-D API) ----------------------------------------

    def get_masked_pcds(
        self,
        frame_idx: int,
        masks: List[np.ndarray],
        max_points: int = 2000,
        sample_ratio: float = 0.5,
    ) -> List[np.ndarray]:
        """Per-mask depth → raw world-frame point clouds.

        Combines the loader's own pose / intrinsics with the depth
        provider's preprocessing.  Callers only need to pass the frame
        index and binary masks.

        Returns one (M, 3) float32 array per mask (world frame).
        Empty (0, 3) when mask has no valid pixels.
        """
        T_w_c = self.get_pose(frame_idx)
        intrinsics = self.get_intrinsics()
        return self.depth_provider.get_masked_pcds(
            frame_idx, masks, T_w_c, intrinsics,
            max_points=max_points, sample_ratio=sample_ratio,
        )

    # -- depth provider ----------------------------------------------------

    @property
    def depth_provider(self) -> DepthProvider:
        """The depth provider for this loader (always set)."""
        return self._depth_provider

    # -- ground truth (benchmarking) ----------------------------------------

    def get_gt_instances(self, frame_idx: int):
        return None

    # -- multi-scene discovery ---------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        return []
