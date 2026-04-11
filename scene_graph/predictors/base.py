"""Edge predictor base class and shared node helpers."""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

from core.types import BBox3D, CameraIntrinsics


class EdgePredictor:
    """Base class for scene-graph edge predictors.

    Subclasses add edges directly to ``graph`` in-place.
    """

    def predict(
        self,
        graph: nx.MultiDiGraph,
        object_registry=None,
        T_w_c: Optional[np.ndarray] = None,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> None:
        raise NotImplementedError


def node_center(ndata: dict) -> Optional[np.ndarray]:
    """Extract the 3-D centre of an object from a graph node's data dict.

    Handles both ``BBox3D`` dataclasses and plain-dict serializations.
    """
    d = ndata.get("data", {})
    bbox = d.get("bbox_3d")
    if bbox is None:
        return None
    if isinstance(bbox, BBox3D):
        return np.asarray(bbox.obb_center, dtype=np.float64)
    obb = bbox.get("obb")
    if obb and obb.get("center") is not None:
        return np.asarray(obb["center"], dtype=np.float64)
    aabb = bbox.get("aabb")
    if aabb and aabb.get("min") is not None and aabb.get("max") is not None:
        return (np.asarray(aabb["min"], dtype=np.float64)
                + np.asarray(aabb["max"], dtype=np.float64)) / 2.0
    return None
