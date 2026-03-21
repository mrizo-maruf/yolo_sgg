"""Depth providers - uniform interfaces for depth (and optional pose)."""

from .base import DepthProvider, OnlineDepthProvider
from .dav3_offline import (
    DAv3OfflineCODaDepthProvider,
    DAv3OfflineDepthProvider,
    DAv3OfflineIsaacSimDepthProvider,
    DAv3OfflineTHUDSyntheticDepthProvider,
)
from .dav3_online import DAv3StreamingDepthProvider
from .gt_depth import (
    CODaDepthProvider,
    IsaacSimDepthProvider,
    MetricPngDepthProvider,
    THUDSyntheticDepthProvider,
)
from .pi3_offline import (
    Pi3OfflineCODaDepthProvider,
    Pi3OfflineDepthProvider,
    Pi3OfflineIsaacSimDepthProvider,
    Pi3OfflineTHUDSyntheticDepthProvider,
)
from .pi3_online import Pi3OnlineDepthProvider

__all__ = [
    "DepthProvider",
    "OnlineDepthProvider",
    "MetricPngDepthProvider",
    "IsaacSimDepthProvider",
    "THUDSyntheticDepthProvider",
    "CODaDepthProvider",
    "Pi3OfflineDepthProvider",
    "Pi3OfflineIsaacSimDepthProvider",
    "Pi3OfflineTHUDSyntheticDepthProvider",
    "Pi3OfflineCODaDepthProvider",
    "DAv3OfflineDepthProvider",
    "DAv3OfflineIsaacSimDepthProvider",
    "DAv3OfflineTHUDSyntheticDepthProvider",
    "DAv3OfflineCODaDepthProvider",
    "Pi3OnlineDepthProvider",
    "DAv3StreamingDepthProvider",
]
