"""Pi3 offline depth providers.

These providers read precomputed Pi3 depth maps from disk and optionally
load model-predicted poses from a trajectory text file.
"""
from __future__ import annotations

from typing import Optional

from .gt_depth import MetricPngDepthProvider


class Pi3OfflineDepthProvider(MetricPngDepthProvider):
    """Generic Pi3 offline provider for metric PNG depth."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "depth{frame_idx:06d}.png",
        png_max_value: int = 65535,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
        pose_lookup: str = "frame_number",
    ) -> None:
        super().__init__(
            depth_dir=depth_dir,
            filename_pattern=filename_pattern,
            png_max_value=png_max_value,
            max_depth=max_depth,
            min_depth=min_depth,
            pose_path=pose_path,
            pose_lookup=pose_lookup,
        )


class Pi3OfflineIsaacSimDepthProvider(Pi3OfflineDepthProvider):
    """Pi3 offline depth for IsaacSim scenes."""

    def __init__(
        self,
        depth_dir: str,
        png_max_value: int = 65535,
        max_depth: float = 10.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            depth_dir=depth_dir,
            filename_pattern="depth{frame_idx:06d}.png",
            png_max_value=png_max_value,
            max_depth=max_depth,
            min_depth=min_depth,
            pose_path=pose_path,
            pose_lookup="frame_number",
        )


class Pi3OfflineTHUDSyntheticDepthProvider(Pi3OfflineDepthProvider):
    """Pi3 offline depth for THUD Synthetic scenes."""

    def __init__(
        self,
        depth_dir: str,
        png_max_value: int = 65535,
        max_depth: float = 100.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            depth_dir=depth_dir,
            filename_pattern="depth_{frame_idx}.png",
            png_max_value=png_max_value,
            max_depth=max_depth,
            min_depth=min_depth,
            pose_path=pose_path,
            pose_lookup="frame_number",
        )


class Pi3OfflineCODaDepthProvider(Pi3OfflineDepthProvider):
    """Pi3 offline depth for CODa scenes."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "depth{frame_idx:06d}.png",
        png_max_value: int = 65535,
        max_depth: float = 80.0,
        min_depth: float = 0.01,
        pose_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            depth_dir=depth_dir,
            filename_pattern=filename_pattern,
            png_max_value=png_max_value,
            max_depth=max_depth,
            min_depth=min_depth,
            pose_path=pose_path,
            pose_lookup="frame_number",
        )
