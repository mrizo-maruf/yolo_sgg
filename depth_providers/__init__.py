"""Depth providers — uniform interface for depth (and optional pose)."""

from .base import DepthProvider, OnlineDepthProvider
from .factory import PROVIDER_CHOICES, build_depth_provider
from .gt_depth import (
    MetricPngDepthProvider,
    IsaacSimDepthProvider,
    THUDSyntheticDepthProvider,
    CODaDepthProvider,
)

__all__ = [
    "DepthProvider",
    "OnlineDepthProvider",
    "MetricPngDepthProvider",
    "IsaacSimDepthProvider",
    "THUDSyntheticDepthProvider",
    "CODaDepthProvider",
    "build_depth_provider",
    "PROVIDER_CHOICES",
]
