"""Basic (geometric / SceneVerse-style) edge predictor."""
from __future__ import annotations

from .base import EdgePredictor


class BasicEdgePredictor(EdgePredictor):
    """Support, hanging, proximity, aligned, middle-of edges.

    Delegates to ``scene_graph.ssg.ssg_main.edges`` which adds edges
    directly to the graph.
    """

    def predict(self, graph, object_registry=None, T_w_c=None,
                depth_m=None, intrinsics=None) -> None:
        if T_w_c is None or depth_m is None:
            return
        if graph.number_of_nodes() < 2:
            return

        from ..ssg.ssg_main import edges as _ssg_edges

        frame_objs = [data.get("data", {}) for _, data in graph.nodes(data=True)]
        _ssg_edges(graph, frame_objs, T_w_c, depth_m)
