"""Scene-graph generation package.

``SceneGraph`` builds per-frame local graphs from tracked objects and
accumulates them into a persistent global graph. Edge prediction is
pluggable — multiple predictors (basic, baseline, VL-SAT) run in
parallel.

Companion subpackages live here too:

* ``scene_graph.ssg``           — SceneVerse-style geometric edge rules
  used by :class:`BasicEdgePredictor`.
* ``scene_graph.vl_sat_model``  — VL-SAT (MMGNet) neural relationship
  predictor used by :class:`VLSATEdgePredictor`.
"""
from .graph import SceneGraph
from .predictors import (
    BaselineEdgePredictor,
    BasicEdgePredictor,
    EdgePredictor,
    VLSATEdgePredictor,
)

__all__ = [
    "SceneGraph",
    "EdgePredictor",
    "BasicEdgePredictor",
    "BaselineEdgePredictor",
    "VLSATEdgePredictor",
]
