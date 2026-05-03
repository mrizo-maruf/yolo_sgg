"""Ground-truth / pre-computed depth providers (package).

Each dataset-specific provider lives in its own module. This ``__init__``
re-exports them so existing imports like
``from depth_providers.gt_depth import IsaacSimDepthProvider`` keep working.
"""
from __future__ import annotations

from .coda import CODaDepthProvider
from .isaacsim import IsaacSimDepthProvider
from .metric_png import MetricPngDepthProvider
from .scanetpp import ScanNetPPDepthProvider
from .thud_synthetic import THUDSyntheticDepthProvider

__all__ = [
    "MetricPngDepthProvider",
    "IsaacSimDepthProvider",
    "ScanNetPPDepthProvider",
    "CODaDepthProvider",
    "THUDSyntheticDepthProvider",
]
