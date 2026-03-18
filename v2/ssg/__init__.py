"""Scene-graph generation sub-package."""

from .edges import predict_edges
from .graph_merge import merge_local_to_global

__all__ = ["predict_edges", "merge_local_to_global"]
