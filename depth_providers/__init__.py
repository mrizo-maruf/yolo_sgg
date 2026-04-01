"""Depth providers — uniform interface for depth (and optional pose)."""

from .base import DepthProvider, OnlineDepthProvider
from .factory import PROVIDER_CHOICES, build_depth_provider
from .gt_depth import (
    MetricPngDepthProvider,
    IsaacSimDepthProvider,
    THUDSyntheticDepthProvider,
    CODaDepthProvider,
)
from .pi3_offline import IsaacSimOfflinePi3DepthProvider
from .pi3_online import Pi3OnlineDepthProvider

__all__ = [
    "DepthProvider",
    "OnlineDepthProvider",
    "MetricPngDepthProvider",
    "IsaacSimDepthProvider",
    "IsaacSimOfflinePi3DepthProvider",
    "Pi3OnlineDepthProvider",
    "THUDSyntheticDepthProvider",
    "CODaDepthProvider",
    "build_depth_provider",
    "PROVIDER_CHOICES",
]
