"""
Abstract base class for dataset loaders.

Every dataset (IsaacSim, THUD Synthetic, THUD Real, …) implements
this interface.  ``run.py`` and ``bench.py`` programme against the
base class so that adding a new dataset only requires writing a new
adapter — no changes to the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np


class DatasetLoader(ABC):
    """Protocol that every dataset loader must implement."""

    # ------------------------------------------------------------------
    # Scene metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def scene_label(self) -> str:
        """Human-readable label for the scene (used in logs / plots)."""
        ...

    @abstractmethod
    def get_frame_indices(self) -> List[int]:
        """Ordered list of frame indices."""
        ...

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @abstractmethod
    def get_rgb_paths(self) -> List[str]:
        """Absolute paths to RGB images, one per frame index."""
        ...

    @abstractmethod
    def get_depth_paths(self) -> List[str]:
        """Absolute paths to depth images, one per frame index."""
        ...

    # ------------------------------------------------------------------
    # Depth loading
    # ------------------------------------------------------------------

    @abstractmethod
    def load_depth(self, path: str) -> np.ndarray:
        """Load a single depth image and return float32 metres (H, W)."""
        ...

    def build_depth_cache(self) -> Dict[str, np.ndarray]:
        """Pre-load all depth maps into a dict keyed by path."""
        return {p: self.load_depth(p) for p in self.get_depth_paths()}

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    @abstractmethod
    def get_camera_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """Return ``(K_3x3, img_height, img_width)`` or *None*."""
        ...

    @abstractmethod
    def get_camera_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return the 4×4 camera-to-world transform for *frame_idx*."""
        ...

    def get_all_poses(self) -> Optional[List[np.ndarray]]:
        """Return all poses in frame-index order (or *None*)."""
        indices = self.get_frame_indices()
        poses = [self.get_camera_pose(i) for i in indices]
        if all(p is None for p in poses):
            return None
        return poses

    # ------------------------------------------------------------------
    # Ground truth (for benchmarking — optional)
    # ------------------------------------------------------------------

    def get_gt_instances(self, frame_idx: int):
        """Return a list of ``GTInstance`` for *frame_idx*, or *None*.

        Subclasses that support benchmarking should override this.
        The return type is ``list[metrics.tracking_metrics.GTInstance]``.
        """
        return None

    # ------------------------------------------------------------------
    # Multi-scene discovery (class method)
    # ------------------------------------------------------------------

    @classmethod
    def discover_scenes(cls, root: str, **kwargs) -> List[str]:
        """Return paths to all valid scene directories under *root*.

        Default returns an empty list.  Multi-scene datasets override this.
        """
        return []
