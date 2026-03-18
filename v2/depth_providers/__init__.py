"""Depth providers — uniform interface for getting depth per frame."""

from .base import DepthProvider
from .gt_depth import IsaacSimDepthProvider, THUDSyntheticDepthProvider
from .pi3_offline import Pi3OfflineDepthProvider

__all__ = [
    "DepthProvider",
    "IsaacSimDepthProvider",
    "THUDSyntheticDepthProvider",
    "Pi3OfflineDepthProvider",
]
