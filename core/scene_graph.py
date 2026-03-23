"""
Scene graph generation and management.

SceneGraph builds per-frame local graphs from tracked objects and
accumulates them into a persistent global graph.  Edge prediction
is pluggable — multiple predictors can run in parallel:

    * **basic**  — geometric / allocentric (support, proximity, hanging,
      aligned, middle) via ``ssg.ssg_main.edges``
    * **baseline** — egocentric camera-frame relations (left/right/…)
    * **vlsat** — VL-SAT neural edge predictor
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import numpy as np
from omegaconf import DictConfig

from core.types import BBox3D, CameraIntrinsics, TrackedObject


# ═══════════════════════════════════════════════════════════════════════════
# SceneGraph
# ═══════════════════════════════════════════════════════════════════════════

class SceneGraph:
    """Per-run scene-graph manager.

    Parameters
    ----------
    cfg : DictConfig
        The ``ssg`` sub-config.  Expected keys (all optional):

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

        self._edge_predictors: List[_EdgePredictor] = []

        if cfg.get("basic_edges", True):
            self._edge_predictors.append(BasicEdgePredictor())
        if cfg.get("baseline_edges", True):
            self._edge_predictors.append(BaselineEdgePredictor())
        if cfg.get("vlsat_edges", False):
            self._edge_predictors.append(VLSATEdgePredictor())

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
        """Build a local graph for one frame, merge into the global graph.

        Parameters
        ----------
        frame_objects : list[TrackedObject]
            Objects visible in this frame.
        object_registry : GlobalObjectRegistry
            Used by VL-SAT to grab accumulated point clouds.
        T_w_c : (4,4) ndarray | None
            Camera-to-world pose (needed by baseline & basic edges).
        depth_m : (H,W) ndarray | None
            Depth map (needed by basic edges for scene-height estimate).
        intrinsics : CameraIntrinsics | None
            Camera calibration (needed for scene-height estimate).

        Returns
        -------
        nx.MultiDiGraph — the local graph for this frame.
        """
        local_graph = self._build_node_graph(frame_objects, object_registry)

        # Run each edge predictor
        for pred in self._edge_predictors:
            pred.predict(
                local_graph,
                object_registry=object_registry,
                T_w_c=T_w_c,
                depth_m=depth_m,
                intrinsics=intrinsics,
            )

        # Merge into global
        self._merge_local_to_global(local_graph)
        self._last_local = local_graph
        return local_graph

    def save_local_graph(self, scene_name: str = "scene") -> None:
        """Save the most recent local graph as JSON."""
        if self._last_local is None:
            return
        self._save_dir.mkdir(parents=True, exist_ok=True)
        out = self._save_dir / f"{scene_name}_local.json"
        _save_graph_json(self._last_local, out)
        print(f"[SceneGraph] Local graph saved → {out}")

    def save_global_graph(self, scene_name: str = "scene") -> None:
        """Save the accumulated global graph as JSON."""
        self._save_dir.mkdir(parents=True, exist_ok=True)
        out = self._save_dir / f"{scene_name}_global.json"
        _save_graph_json(self.global_graph, out)
        print(f"[SceneGraph] Global graph saved → {out}")

    # ------------------------------------------------------------------
    # Internals
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
            reg = all_objs.get(gid, {})
            g.add_node(gid, data={
                "global_id": gid,
                "track_id": gid,
                "class_name": obj.class_name,
                "bbox_3d": _bbox3d_to_dict(obj.bbox_3d),
                "visible_current_frame": True,
            })
        return g

    def _merge_local_to_global(self, local_graph: nx.MultiDiGraph) -> None:
        """Merge local graph into global, handling ego/allo edges."""
        # Match nodes by track_id (1-to-1 since we use global_id)
        node_map: Dict[int, int] = {}
        for node in local_graph.nodes():
            node_map[node] = node  # global_ids are stable

        # Update node attributes
        for lnode, gnode in node_map.items():
            if gnode in self.global_graph:
                self.global_graph.nodes[gnode].update(local_graph.nodes[lnode])
            else:
                self.global_graph.add_node(gnode, **local_graph.nodes[lnode])

        current_nodes = set(node_map.values())

        # Egocentric edges: replace for visible nodes
        ego_classes = {"proximity", "middle_furniture"}
        for u, v, key, data in list(self.global_graph.edges(keys=True, data=True)):
            if data.get("label_class") in ego_classes:
                if u in current_nodes and v in current_nodes:
                    self.global_graph.remove_edge(u, v, key)

        # Remove stale aligned_furniture (recomputed each frame)
        for u, v, key, data in list(self.global_graph.edges(keys=True, data=True)):
            if data.get("label_class") == "aligned_furniture":
                self.global_graph.remove_edge(u, v, key)

        # Allocentric edges: merge carefully
        allo_classes = {"support", "embedded", "hanging",
                        "oppo_support", "aligned_furniture"}

        for u, v, key, data in local_graph.edges(keys=True, data=True):
            u_g = node_map[u]
            v_g = node_map[v]
            label_class = data.get("label_class", "")

            if label_class in ego_classes:
                # Simply add egocentric edges from current frame
                self.global_graph.add_edge(u_g, v_g, **data)
                continue

            if label_class not in allo_classes:
                # Baseline / vlsat edges — always add fresh
                self.global_graph.add_edge(u_g, v_g, **data)
                continue

            # Allocentric: check for conflicts
            edge_found = False
            if self.global_graph.has_edge(u_g, v_g):
                for ek in list(self.global_graph[u_g][v_g].keys()):
                    ed = self.global_graph[u_g][v_g][ek]
                    pc = ed.get("label_class")
                    if pc == label_class:
                        if ed.get("label") == data.get("label"):
                            self.global_graph[u_g][v_g][ek].update(data)
                        else:
                            self.global_graph.remove_edge(u_g, v_g, ek)
                            self.global_graph.add_edge(u_g, v_g, **data)
                        edge_found = True
                        break
                    elif pc in allo_classes and _are_conflicting(pc, label_class):
                        self.global_graph.remove_edge(u_g, v_g, ek)
                        if pc == "support" and self.global_graph.has_edge(v_g, u_g):
                            for ek2 in list(self.global_graph[v_g][u_g].keys()):
                                if self.global_graph[v_g][u_g][ek2].get("label_class") == "oppo_support":
                                    self.global_graph.remove_edge(v_g, u_g, ek2)
                                    break
                        break

            if not edge_found:
                self.global_graph.add_edge(u_g, v_g, **data)


# ═══════════════════════════════════════════════════════════════════════════
# Edge predictor interface
# ═══════════════════════════════════════════════════════════════════════════

class _EdgePredictor:
    """Base class for edge predictors."""

    def predict(
        self,
        graph: nx.MultiDiGraph,
        object_registry=None,
        T_w_c: Optional[np.ndarray] = None,
        depth_m: Optional[np.ndarray] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> None:
        """Add edges to *graph* in-place."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════
# Basic (geometric / SceneVerse-style) edges
# ═══════════════════════════════════════════════════════════════════════════

class BasicEdgePredictor(_EdgePredictor):
    """Support, hanging, proximity, aligned, middle-of edges.

    Delegates to the existing ``ssg.ssg_main.edges`` function which adds
    edges directly to the graph.
    """

    def predict(self, graph, object_registry=None, T_w_c=None,
                depth_m=None, intrinsics=None) -> None:
        if T_w_c is None or depth_m is None:
            return
        if graph.number_of_nodes() < 2:
            return

        from ssg.ssg_main import edges as _ssg_edges

        # Build frame_objs list expected by edges()
        frame_objs = []
        for node, data in graph.nodes(data=True):
            frame_objs.append(data.get("data", {}))

        _ssg_edges(graph, frame_objs, T_w_c, depth_m)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline (egocentric camera-frame) edges
# ═══════════════════════════════════════════════════════════════════════════

class BaselineEdgePredictor(_EdgePredictor):
    """Egocentric spatial relations (left/right/front/back/above/below).

    Works entirely from bbox centres projected into camera frame.
    """

    def predict(self, graph, object_registry=None, T_w_c=None,
                depth_m=None, intrinsics=None) -> None:
        if T_w_c is None:
            return
        if graph.number_of_nodes() < 2:
            return

        # Collect centres in camera frame
        T_c_w = np.linalg.inv(T_w_c)
        R, t = T_c_w[:3, :3], T_c_w[:3, 3]

        cam_centres: Dict[int, np.ndarray] = {}
        for nid, ndata in graph.nodes(data=True):
            center = _node_center(ndata)
            if center is not None:
                cam_centres[nid] = (R @ center) + t

        for (id_a, pa), (id_b, pb) in combinations(cam_centres.items(), 2):
            for rel in _camera_relations(pa, pb):
                graph.add_edge(id_a, id_b,
                               label=rel, label_class="baseline")
            for rel in _camera_relations(pb, pa):
                graph.add_edge(id_b, id_a,
                               label=rel, label_class="baseline")


# ═══════════════════════════════════════════════════════════════════════════
# VL-SAT neural edges
# ═══════════════════════════════════════════════════════════════════════════

class VLSATEdgePredictor(_EdgePredictor):
    """VL-SAT (MMGNet) neural relationship predictor.

    Lazily initialises the model on first call.
    """

    def __init__(self) -> None:
        self._predictor = None

    def predict(self, graph, object_registry=None, T_w_c=None,
                depth_m=None, intrinsics=None) -> None:
        if object_registry is None:
            return
        all_objs = object_registry.get_all_objects()
        if len(all_objs) < 2:
            return

        # Collect objects with enough accumulated points
        valid_ids: List[int] = []
        point_clouds: List[np.ndarray] = []
        for gid, obj in all_objs.items():
            if gid not in graph.nodes:
                continue
            pts = obj.get("points_accumulated")
            if pts is not None and len(pts) >= 10:
                valid_ids.append(gid)
                # Axis convention: world frame (Z-up) → 3RScan/VL-SAT (Y-up)
                pts_conv = pts[:, [0, 2, 1]].copy()
                pts_conv[:, 2] = -pts_conv[:, 2]
                point_clouds.append(pts_conv)

        if len(valid_ids) < 2:
            return

        if self._predictor is None:
            self._predictor = _init_vlsat()

        import torch

        num_points = self._predictor.config.dataset.num_points
        obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids = \
            self._predictor.preprocess_poinclouds(point_clouds, num_points)
        predicted = self._predictor.predict_relations(
            obj_points, obj_2d_feats, edge_indices, descriptor, batch_ids,
        )

        for k in range(predicted.shape[0]):
            src_idx = edge_indices[0][k].item()
            tgt_idx = edge_indices[1][k].item()
            rel_id = torch.argmax(predicted[k]).item()
            rel_name = self._predictor.rel_id_to_rel_name.get(rel_id, "none")
            if rel_name == "none":
                continue
            src_gid = valid_ids[src_idx]
            tgt_gid = valid_ids[tgt_idx]
            graph.add_edge(src_gid, tgt_gid,
                           label=rel_name, label_class="vlsat")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _are_conflicting(cls1: str, cls2: str) -> bool:
    conflicts = {
        "support": {"embedded", "hanging"},
        "embedded": {"support", "hanging"},
        "hanging": {"support", "embedded"},
    }
    return cls2 in conflicts.get(cls1, set())


def _node_center(ndata: dict) -> Optional[np.ndarray]:
    """Extract 3-D centre from a graph node's data dict."""
    d = ndata.get("data", {})
    bbox = d.get("bbox_3d")
    if bbox is None:
        return None
    # BBox3D dataclass vs dict
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


def _camera_relations(src: np.ndarray, tgt: np.ndarray) -> List[str]:
    """Which of {left, right, front, back, above, below} does *tgt*
    satisfy relative to *src* in camera frame?"""
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


def _init_vlsat():
    """Lazily import and init VL-SAT EdgePredictor."""
    vlsat_dir = str(Path(__file__).resolve().parent.parent / "vl_sat_model")
    while vlsat_dir in sys.path:
        sys.path.remove(vlsat_dir)
    sys.path.insert(0, vlsat_dir)
    if "utils" in sys.modules:
        del sys.modules["utils"]

    from vl_sat_model.vl_sat_interface import EdgePredictor

    cfg_dir = Path(vlsat_dir)
    predictor = EdgePredictor(
        config_path=str(cfg_dir / "config" / "mmgnet.json"),
        ckpt_path=str(cfg_dir / "3dssg_best_ckpt"),
        relationships_list=str(cfg_dir / "config" / "relationships.txt"),
    )
    return predictor


def _bbox3d_to_dict(bbox) -> Optional[Dict]:
    """Convert BBox3D dataclass (or None) to a plain dict."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return bbox
    return bbox.to_dict()


def _save_graph_json(graph: nx.MultiDiGraph, path: Path) -> None:
    """Serialize a MultiDiGraph to JSON with nodes + edges."""
    nodes = {}
    for nid, ndata in graph.nodes(data=True):
        d = ndata.get("data", {})
        edges_out = []
        for _, tgt, _, edata in graph.out_edges(nid, keys=True, data=True):
            edges_out.append({
                "target_id": int(tgt),
                "label": edata.get("label", ""),
                "label_class": edata.get("label_class", ""),
            })
        nodes[int(nid)] = {
            "track_id": int(nid),
            "class_name": d.get("class_name"),
            "bbox_3d": d.get("bbox_3d"),
            "edges": edges_out,
        }

    output = {
        "num_objects": len(nodes),
        "num_edges": graph.number_of_edges(),
        "nodes": nodes,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)