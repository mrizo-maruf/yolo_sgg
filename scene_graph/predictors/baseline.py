"""Baseline (egocentric camera-frame) edge predictor."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import numpy as np

from .base import EdgePredictor, node_center


class BaselineEdgePredictor(EdgePredictor):
    """Egocentric spatial relations (left/right/front/back/above/below).

    Works entirely from bbox centres projected into camera frame.
    """

    def predict(self, graph, object_registry=None, T_w_c=None,
                depth_m=None, intrinsics=None) -> None:
        if T_w_c is None:
            return
        if graph.number_of_nodes() < 2:
            return

        T_c_w = np.linalg.inv(T_w_c)
        R, t = T_c_w[:3, :3], T_c_w[:3, 3]

        cam_centres: Dict[int, np.ndarray] = {}
        for nid, ndata in graph.nodes(data=True):
            center = node_center(ndata)
            if center is not None:
                cam_centres[nid] = (R @ center) + t

        for (id_a, pa), (id_b, pb) in combinations(cam_centres.items(), 2):
            for rel in _camera_relations(pa, pb):
                graph.add_edge(id_a, id_b, label=rel, label_class="baseline")
            for rel in _camera_relations(pb, pa):
                graph.add_edge(id_b, id_a, label=rel, label_class="baseline")


def _camera_relations(src: np.ndarray, tgt: np.ndarray) -> List[str]:
    """Which of {left, right, front, back, above, below} ``tgt`` satisfies
    relative to ``src`` in camera frame."""
    rels: List[str] = []
    if tgt[0] < src[0]:
        rels.append("left")
    if tgt[0] > src[0]:
        rels.append("right")
    if tgt[2] > src[2]:
        rels.append("front")
    if tgt[2] < src[2]:
        rels.append("back")
    if tgt[1] < src[1]:
        rels.append("above")
    if tgt[1] > src[1]:
        rels.append("below")
    return rels
