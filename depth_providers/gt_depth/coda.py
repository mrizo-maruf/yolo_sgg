"""CODa GT depth provider."""
from __future__ import annotations

from typing import Optional

from .metric_png import MetricPngDepthProvider


class CODaDepthProvider(MetricPngDepthProvider):
    """CODa GT depth."""

    def __init__(
        self,
        depth_dir: str,
        filename_pattern: str = "depth{frame_idx:06d}.png",
        png_max_value: int = 65535,
        max_depth: float = 80.0,
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
