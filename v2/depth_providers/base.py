"""
Abstract base class for depth providers.

A depth provider converts a frame index (or path) into a float32
depth map in **metres**, shape (H, W).

Two families:
    • Offline: depth is pre-computed or from GT.  ``get_depth(idx)``
      returns immediately.
    • Online / streaming: depth is computed on-the-fly (e.g. Pi3X,
      DepthAnything V3).  The provider may buffer a window of frames
      for temporal consistency.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class DepthProvider(ABC):
    """Base depth provider."""

    @abstractmethod
    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return (H, W) float32 depth map, or None.

        For datasets with standard pinhole cameras (e.g. IsaacSim) this
        should be metric depth in metres.  For datasets with custom
        unprojection (e.g. THUD Synthetic) this can be the raw depth
        values — the matching :meth:`unproject` handles the conversion.
        """
        ...

    def unproject(
        self,
        us: np.ndarray,
        vs: np.ndarray,
        depths: np.ndarray,
        intrinsics,
    ) -> np.ndarray:
        """(u, v, depth) pixel arrays → (N, 3) 3-D points.

        Default: standard pinhole unprojection (camera frame).
        Override for datasets with custom depth-to-3D formulas.
        """
        X = (us.astype(np.float32) - intrinsics.cx) * depths / intrinsics.fx
        Y = (vs.astype(np.float32) - intrinsics.cy) * depths / intrinsics.fy
        return np.stack([X, Y, depths], axis=1).astype(np.float32)

    @abstractmethod
    def get_depth_pcd_from_masks(
        self,
        frame_idx: int,
        intrinsics,
        masks: list[np.ndarray],
        max_points: int = 0,
        seed: int = 0,
    ) -> Optional[list[np.ndarray]]:
        """Unproject depth once, then extract per-mask point clouds.

        Parameters
        ----------
        frame_idx : int
            Frame number to load depth for.
        intrinsics : CameraIntrinsics
            Camera intrinsics.
        masks : list[np.ndarray]
            List of (H, W) bool/uint8 masks — one per detection.
        max_points : int
            If > 0, randomly sub-sample each cloud to at most this many.
        seed : int
            RNG seed for reproducible sub-sampling.

        Returns
        -------
        list[np.ndarray] | None
            One (N_i, 3) float32 array per mask, or None if depth
            cannot be loaded.
        """
        ...

    def warmup(self) -> None:
        """Optional pre-computation (e.g. load model, pre-process batch)."""

    def close(self) -> None:
        """Release resources (GPU memory, file handles)."""


class OnlineDepthProvider(DepthProvider):
    """Base for streaming / real-time depth estimators.

    Subclasses must implement ``feed_frame`` (push RGB) and
    ``get_depth`` (pull depth for a given index).
    """

    @abstractmethod
    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        """Push an RGB frame into the internal buffer / window."""
        ...
