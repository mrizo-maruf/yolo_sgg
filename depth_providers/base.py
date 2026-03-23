"""Abstract base class for depth providers.

Each depth provider owns the full ``depth image → world-frame point cloud``
pipeline because different depth sources (GT PNG, Pi3x predictions,
DAV3 streaming, …) need different preprocessing / unprojection math.

The common API is :meth:`get_masked_pcds` which takes a frame index, a
list of binary masks from YOLO, the camera-to-world transform, and
intrinsics, and returns one raw (uncleaned) world-frame PCD per mask.
PCD cleaning (outlier removal, DBSCAN) is intentionally **not** done
here — use ``core.geometry.clean_pcd`` on each returned chunk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class DepthProvider(ABC):
    """Base depth provider.

    Subclasses **must** implement :meth:`get_depth`.
    They *may* override :meth:`unproject` (for non-pinhole depth) or
    the full :meth:`get_masked_pcds` (for providers where the depth map
    concept doesn't apply, e.g. a model that outputs PCDs directly).
    """

    # ------------------------------------------------------------------
    # Core: depth map
    # ------------------------------------------------------------------

    @abstractmethod
    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return (H, W) float32 depth map in metres, or None."""
        ...

    # ------------------------------------------------------------------
    # Optional: pose
    # ------------------------------------------------------------------

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return 4×4 camera-to-world pose, or None.

        Only providers that also predict poses (e.g. Pi3) need to
        override this.
        """
        return None

    # ------------------------------------------------------------------
    # Unprojection (override for non-standard depth→3D, e.g. THUD)
    # ------------------------------------------------------------------

    def unproject(
        self,
        us: np.ndarray,
        vs: np.ndarray,
        depths: np.ndarray,
        intrinsics,
    ) -> np.ndarray:
        """(u, v, depth) pixel arrays → (N, 3) 3-D points in **camera frame**.

        Default: standard pinhole model.  Override for datasets with
        non-standard depth encoding (e.g. THUD synthetic).
        """
        X = (us.astype(np.float32) - intrinsics.cx) * depths / intrinsics.fx
        Y = (vs.astype(np.float32) - intrinsics.cy) * depths / intrinsics.fy
        return np.stack([X, Y, depths], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # Main API: depth + masks → world-frame PCDs
    # ------------------------------------------------------------------

    def get_masked_pcds(
        self,
        frame_idx: int,
        masks: List[np.ndarray],
        T_w_c: Optional[np.ndarray],
        intrinsics,
        max_points: int = 2000,
        sample_ratio: float = 0.5,
    ) -> List[np.ndarray]:
        """Per-mask depth → raw world-frame point clouds.

        Parameters
        ----------
        frame_idx : int
            Which frame to load depth for.
        masks : list[np.ndarray]
            Binary masks (H, W) from YOLO, one per detection.
        T_w_c : (4, 4) ndarray or None
            Camera-to-world transform.  If None, points stay in camera
            frame.
        intrinsics : CameraIntrinsics
            Camera calibration.
        max_points : int
            Max points to keep per mask (sub-sampled randomly).
        sample_ratio : float
            Fraction of valid pixels to keep before capping at
            *max_points*.

        Returns
        -------
        list[np.ndarray]
            One (M, 3) float32 array per mask, in **world frame**.
            Empty (0, 3) when depth is missing or mask has no valid
            pixels.

        Notes
        -----
        The default implementation calls ``get_depth`` + ``unproject``
        + a simple cam→world rotation.  Providers with exotic depth
        formats (or models that output PCDs directly) should override
        this method entirely.
        """
        depth_m = self.get_depth(frame_idx)
        empty = np.zeros((0, 3), dtype=np.float32)

        if depth_m is None:
            return [empty for _ in masks]

        results: List[np.ndarray] = []
        for mask in masks:
            pts = self._mask_to_world_pcd(
                depth_m, mask, T_w_c, intrinsics,
                max_points, sample_ratio,
            )
            results.append(pts)
        return results

    # ------------------------------------------------------------------
    # Internal: single-mask helper (shared by default get_masked_pcds)
    # ------------------------------------------------------------------

    def _mask_to_world_pcd(
        self,
        depth_m: np.ndarray,
        mask: np.ndarray,
        T_w_c: Optional[np.ndarray],
        intrinsics,
        max_points: int,
        sample_ratio: float,
    ) -> np.ndarray:
        """Single mask → subsampled world-frame PCD (no cleaning)."""
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        mask_bool = np.squeeze(mask_bool)
        valid = mask_bool & (depth_m > 0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        vs, us = np.nonzero(valid)
        zs = depth_m[vs, us].astype(np.float32)

        # Sub-sample
        M = zs.shape[0]
        n_keep = min(max_points, max(1, int(M * sample_ratio)))
        if M > n_keep:
            rng = np.random.default_rng(0)
            idx = rng.choice(M, size=n_keep, replace=False)
            us, vs, zs = us[idx], vs[idx], zs[idx]

        pts_cam = self.unproject(us, vs, zs, intrinsics)

        if T_w_c is not None and pts_cam.shape[0] > 0:
            R = T_w_c[:3, :3]
            t = T_w_c[:3, 3]
            pts_cam = (pts_cam @ R.T) + t

        return pts_cam.astype(np.float32)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Optional pre-computation (e.g. load model)."""

    def close(self) -> None:
        """Release resources."""


class OnlineDepthProvider(DepthProvider):
    """Base for streaming / real-time depth estimators.

    Must also implement ``feed_frame`` to push RGB frames.
    """

    @abstractmethod
    def feed_frame(self, frame_idx: int, rgb: np.ndarray) -> None:
        """Push an RGB frame into the internal buffer."""
        ...
