"""Edge predictors for scene-graph generation."""
from .base import EdgePredictor
from .basic import BasicEdgePredictor
from .baseline import BaselineEdgePredictor
from .vlsat import VLSATEdgePredictor

__all__ = [
    "EdgePredictor",
    "BasicEdgePredictor",
    "BaselineEdgePredictor",
    "VLSATEdgePredictor",
]
