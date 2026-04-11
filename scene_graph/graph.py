"""SceneGraph class — builds per-frame local graphs and accumulates them."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from omegaconf import DictConfig

from core.types import CameraIntrinsics, TrackedObject

from .graph_utils import are_conflicting, bbox3d_to_dict, save_graph_json
from .predictors import (
    BaselineEdgePredictor,
    BasicEdgePredictor,
    EdgePredictor,
    VLSATEdgePredictor,
)

_EGO_SUBCLASSES = frozenset({"proximity", "middle_furniture"})
_ALLO_SUBCLASSES = frozenset({
    "support", "embedded", "hanging", "oppo_support", "aligned_furniture",
})


class SceneGraph:
    """Per-run scene-graph manager.

    Parameters
    ----------
    cfg : DictConfig
        The ``ssg`` sub-config. Expected keys (all optional):

        * ``basic_edges``         — bool, enable geometric edges
        * ``baseline_edges``      — bool, enable egocentric baseline
        * ``vlsat_edges``         — bool, enable VL-SAT neural edges
        * ``save_local_graph``    — bool
        * ``save_global_graph``   — bool
        * ``save_graph_dir``      — str
        * ``vis_graph``           — bool
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.global_graph = nx.MultiDiGraph()
        # Track aligned_furniture edge keys so we can clear stale ones fast.
        self._aligned_edge_refs: Set[Tuple[int, int, int]] = set()
        self._last_profile: Dict[str, float] = {}

        self._edge_predictors: List[EdgePredictor] = []

        if cfg.get("basic_edges", True):
            self._edge_predictors.append(BasicEdgePredictor())
            print("[SceneGraph] BasicEdgePredictor enabled.")
        if cfg.get("baseline_edges", True):
            self._edge_predictors.append(BaselineEdgePredictor())
            print("[SceneGraph] BaselineEdgePredictor enabled.")
        if cfg.get("vlsat_edges", False):
            self._edge_predictors.append(VLSATEdgePredictor(cfg))
            print("[SceneGraph] VLSATEdgePredictor enabled.")

        self._save_dir = Path(cfg.get("save_graph_dir", "results/scene_graphs"))
        self._last_local: Optional[nx.MultiDiGraph] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate_graph(
        self,
        frame_objects: List[TrackedObject],
        object_registry,
        T_w_c: Optional[np.ndarray] = None,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> nx.MultiDiGraph:
        """Build a local graph for one frame and merge into the global graph."""
        t_graph_start = time.perf_counter()
        t0 = time.perf_counter()
        local_graph = self._build_node_graph(frame_objects, object_registry)
        build_nodes_ms = (time.perf_counter() - t0) * 1000.0

        predictor_ms: Dict[str, float] = {}
        for pred in self._edge_predictors:
            t_pred = time.perf_counter()
            pred.predict(
                local_graph,
                object_registry=object_registry,
                T_w_c=T_w_c,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )
            predictor_ms[f"predict_{pred.__class__.__name__}_ms"] = (
                time.perf_counter() - t_pred
            ) * 1000.0

        t0 = time.perf_counter()
        self._merge_local_to_global(local_graph)
        merge_ms = (time.perf_counter() - t0) * 1000.0

        self._last_local = local_graph
        self._last_profile = {
            "build_nodes_ms": build_nodes_ms,
            "merge_ms": merge_ms,
            "local_nodes": float(local_graph.number_of_nodes()),
            "local_edges": float(local_graph.number_of_edges()),
            "global_nodes": float(self.global_graph.number_of_nodes()),
            "global_edges": float(self.global_graph.number_of_edges()),
            "total_ms": (time.perf_counter() - t_graph_start) * 1000.0,
        }
        self._last_profile.update(predictor_ms)
        return local_graph

    def get_last_profile(self) -> Dict[str, float]:
        """Return profiling stats from the last ``generate_graph`` call."""
        return dict(self._last_profile)

    def save_local_graph(self, scene_name: str = "scene") -> None:
        """Save the most recent local graph as JSON."""
        if self._last_local is None:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        out = self._save_dir / f"{scene_name}_local.json"
        save_graph_json(self._last_local, out)
        print(f"[SceneGraph] Local graph saved → {out}")

    def save_global_graph(self, scene_name: str = "scene") -> None:
        """Save the accumulated global graph as JSON."""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        out = self._save_dir / f"{scene_name}_global.json"
        save_graph_json(self.global_graph, out)
        print(f"[SceneGraph] Global graph saved → {out}")

    # ------------------------------------------------------------------
    # Internals — node graph
    # ------------------------------------------------------------------

    @staticmethod
    def _build_node_graph(
        frame_objects: List[TrackedObject],
        object_registry,
    ) -> nx.MultiDiGraph:
        """Create a graph with one node per visible object (no edges yet)."""
        g = nx.MultiDiGraph()
        all_objs = object_registry.get_all_objects()
        for obj in frame_objects:
            gid = obj.global_id
            _ = all_objs.get(gid, {})
            g.add_node(gid, data={
                "global_id": gid,
                "track_id": gid,
                "class_name": obj.class_name,
                "bbox_3d": bbox3d_to_dict(obj.bbox_3d),
                "visible_current_frame": True,
            })
        return g

    # ------------------------------------------------------------------
    # Internals — merge
    # ------------------------------------------------------------------

    def _merge_local_to_global(self, local_graph: nx.MultiDiGraph) -> None:
        """Merge local graph into global, handling ego/allo edges."""
        # global_ids are stable, so the node map is an identity.
        node_map: Dict[int, int] = {n: n for n in local_graph.nodes()}

        for lnode, gnode in node_map.items():
            if gnode in self.global_graph:
                self.global_graph.nodes[gnode].update(local_graph.nodes[lnode])
            else:
                self.global_graph.add_node(gnode, **local_graph.nodes[lnode])

        current_nodes = set(node_map.values())
        self._remove_dynamic_edges_for_current_nodes(current_nodes)
        self._clear_stale_aligned_edges()

        for u, v, _, data in local_graph.edges(keys=True, data=True):
            self._merge_edge(node_map[u], node_map[v], data)

    def _merge_edge(self, u_g: int, v_g: int, data: dict) -> None:
        """Apply a single local edge to the global graph."""
        label_class = data.get("label_class", "")
        label_subclass = data.get("label_subclass", "")

        # Non-basic edges (baseline, vlsat): upsert by semantic signature
        # to avoid unbounded duplicate growth across frames.
        if label_class != "basic":
            self._upsert_non_basic_edge(u_g, v_g, data)
            return

        # Egocentric basic edges — simply add from current frame
        if label_subclass in _EGO_SUBCLASSES:
            self._add_edge(u_g, v_g, data)
            return

        # Non-allocentric basic edges — no conflict checks needed
        if label_subclass not in _ALLO_SUBCLASSES:
            self._add_edge(u_g, v_g, data)
            return

        if not self._reconcile_allocentric_edge(u_g, v_g, label_subclass, data):
            self._add_edge(u_g, v_g, data)

    def _reconcile_allocentric_edge(
        self,
        u_g: int,
        v_g: int,
        label_subclass: str,
        data: dict,
    ) -> bool:
        """Reconcile an incoming allocentric basic edge against existing
        edges on (u_g, v_g). Returns True iff the edge was handled
        (updated or conflict-resolved) — False means the caller should add
        it as a fresh edge.
        """
        if not self.global_graph.has_edge(u_g, v_g):
            return False

        for ek in list(self.global_graph[u_g][v_g].keys()):
            ed = self.global_graph[u_g][v_g][ek]
            pc = ed.get("label_subclass", "")
            if ed.get("label_class") != "basic":
                continue

            if pc == label_subclass:
                if ed.get("label") == data.get("label"):
                    self.global_graph[u_g][v_g][ek].update(data)
                else:
                    self._remove_edge(u_g, v_g, ek)
                    self._add_edge(u_g, v_g, data)
                return True

            if pc in _ALLO_SUBCLASSES and are_conflicting(pc, label_subclass):
                self._remove_edge(u_g, v_g, ek)
                if pc == "support":
                    self._remove_reverse_oppo_support(v_g, u_g)
                return False

        return False

    def _remove_reverse_oppo_support(self, src: int, dst: int) -> None:
        """Drop the reverse ``oppo_support`` edge paired with a removed support."""
        if not self.global_graph.has_edge(src, dst):
            return
        for ek2 in list(self.global_graph[src][dst].keys()):
            ed2 = self.global_graph[src][dst][ek2]
            if (
                ed2.get("label_class") == "basic"
                and ed2.get("label_subclass") == "oppo_support"
            ):
                self._remove_edge(src, dst, ek2)
                return

    def _upsert_non_basic_edge(self, u: int, v: int, data: dict) -> None:
        """Upsert non-basic edges by semantic signature."""
        label_class = data.get("label_class", "")
        label = data.get("label", "")
        label_subclass = data.get("label_subclass", "")
        if self.global_graph.has_edge(u, v):
            for ek in list(self.global_graph[u][v].keys()):
                ed = self.global_graph[u][v][ek]
                if (
                    ed.get("label_class", "") == label_class
                    and ed.get("label", "") == label
                    and ed.get("label_subclass", "") == label_subclass
                ):
                    self.global_graph[u][v][ek].update(data)
                    return
        self._add_edge(u, v, data)

    # ------------------------------------------------------------------
    # Internals — dynamic edge maintenance
    # ------------------------------------------------------------------

    @staticmethod
    def _is_dynamic_edge(data: dict) -> bool:
        """True for camera-dependent relations that should be refreshed."""
        label_class = data.get("label_class", "")
        label_subclass = data.get("label_subclass", "")
        if label_class == "baseline":
            return True
        return label_class == "basic" and label_subclass in _EGO_SUBCLASSES

    def _remove_dynamic_edges_for_current_nodes(self, current_nodes: Set[int]) -> None:
        """Remove camera-dependent edges only for currently visible nodes.

        Refresh only edges where *both* endpoints are visible — keeps
        last-known edges for invisible objects in the global view.
        """
        to_remove: Set[Tuple[int, int, int]] = set()
        for u in current_nodes:
            if u not in self.global_graph:
                continue
            for _, v, k, d in self.global_graph.out_edges(u, keys=True, data=True):
                if self._is_dynamic_edge(d) and v in current_nodes:
                    to_remove.add((u, v, k))
            for v, _, k, d in self.global_graph.in_edges(u, keys=True, data=True):
                if self._is_dynamic_edge(d) and v in current_nodes:
                    to_remove.add((v, u, k))
        for u, v, k in to_remove:
            self._remove_edge(u, v, k)

    def _clear_stale_aligned_edges(self) -> None:
        """Remove aligned_furniture edges from the previous frame."""
        if not self._aligned_edge_refs:
            return
        stale = list(self._aligned_edge_refs)
        self._aligned_edge_refs.clear()
        for u, v, k in stale:
            if self.global_graph.has_edge(u, v, k):
                self.global_graph.remove_edge(u, v, k)

    def _add_edge(self, u: int, v: int, data: dict) -> int:
        """Add an edge and keep the aligned-edge index in sync."""
        k = self.global_graph.add_edge(u, v, **data)
        if (
            data.get("label_class") == "basic"
            and data.get("label_subclass") == "aligned_furniture"
        ):
            self._aligned_edge_refs.add((u, v, int(k)))
        return int(k)

    def _remove_edge(self, u: int, v: int, k: int) -> None:
        """Remove an edge and keep the aligned-edge index in sync."""
        self.global_graph.remove_edge(u, v, k)
        self._aligned_edge_refs.discard((u, v, int(k)))
