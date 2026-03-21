"""
Edge prediction for the local scene graph.

Computes spatial relationships (support, embedded, hanging, proximity,
aligned, in-the-middle) between objects in a single frame using their
3-D bounding boxes.

This is a cleaned-up, standalone version of ``ssg.ssg_utils.edges()``
from v1 — no global state, all inputs explicit.
"""
from __future__ import annotations

import itertools as it
import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from v2.geometry import cam_to_world, estimate_scene_height
from v2.types import BBox3D, CameraIntrinsics, TrackedObject


# ═══════════════════════════════════════════════════════════════════════════
# Lightweight ObjNode used by relationship predictors
# ═══════════════════════════════════════════════════════════════════════════

class ObjNode:
    """Minimal object representation for edge prediction."""
    __slots__ = ("id", "label", "position", "x_min", "x_max",
                 "y_min", "y_max", "z_min", "z_max")

    def __init__(
        self, *, id: int, label: str, position: np.ndarray,
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        z_min: float, z_max: float,
    ):
        self.id = id
        self.label = label
        self.position = np.asarray(position, dtype=np.float64)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max


def _obj_node_from_tracked(obj: TrackedObject) -> ObjNode:
    bbox = obj.bbox_3d
    if bbox is None:
        return ObjNode(
            id=obj.global_id, label=obj.class_name or "NONE",
            position=np.zeros(3),
            x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0,
        )
    mn, mx = bbox.aabb_min, bbox.aabb_max
    return ObjNode(
        id=obj.global_id,
        label=obj.class_name or "NONE",
        position=bbox.center,
        x_min=float(mn[0]), x_max=float(mx[0]),
        y_min=float(mn[1]), y_max=float(mx[1]),
        z_min=float(mn[2]), z_max=float(mx[2]),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Camera helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_theta(x: np.ndarray, y: np.ndarray) -> float:
    """Angle in degrees between two vectors."""
    lx = np.linalg.norm(x)
    ly = np.linalg.norm(y)
    if lx < 1e-9 or ly < 1e-9:
        return 0.0
    cos_ = np.clip(np.dot(x, y) / (lx * ly), -1, 1)
    return float(np.degrees(np.arccos(cos_)))


def _camera_angle_from_pose(T_w_c: np.ndarray) -> float:
    """Planar camera angle relative to +Y axis (sign by X)."""
    R = T_w_c[:3, :3]
    forward_cam = np.array([0.0, 0.0, -1.0])
    view = R @ forward_cam
    v_xy = view.copy()
    v_xy[2] = 0.0
    n = np.linalg.norm(v_xy)
    if n > 0:
        v_xy /= n
    else:
        v_xy = np.array([0.0, 1.0, 0.0])
    angle = _get_theta(v_xy, np.array([0.0, 1.0, 0.0]))
    return -angle if v_xy[0] < 0 else angle


# ═══════════════════════════════════════════════════════════════════════════
# Relationship predicates (geometric heuristics)
# ═══════════════════════════════════════════════════════════════════════════

def _overlap_xy(a: ObjNode, b: ObjNode) -> bool:
    """Check if projections overlap in the XY plane."""
    return (a.x_min < b.x_max and a.x_max > b.x_min and
            a.y_min < b.y_max and a.y_max > b.y_min)


def _support_relations(
    nodes: Dict[int, ObjNode],
    z_gap_threshold: float = 0.15,
) -> Tuple[List[list], List[list], Dict[int, bool]]:
    """Detect support / embedded relationships.

    Returns (support_rels, embedded_rels, hanging_candidates)
    """
    support, embedded = [], []
    hanging_candidates: Dict[int, bool] = {}
    ids = list(nodes.keys())

    for i, j in it.combinations(ids, 2):
        a, b = nodes[i], nodes[j]
        if not _overlap_xy(a, b):
            continue

        # Who is on top?
        if a.z_min >= b.z_max - z_gap_threshold:
            # a sits on b
            support.append([j, i, "support"])
        elif b.z_min >= a.z_max - z_gap_threshold:
            support.append([i, j, "support"])
        else:
            # Significant Z overlap → embedded
            z_overlap = min(a.z_max, b.z_max) - max(a.z_min, b.z_min)
            h_a = a.z_max - a.z_min
            h_b = b.z_max - b.z_min
            if h_a > 0 and h_b > 0:
                ratio = z_overlap / min(h_a, h_b)
                if ratio > 0.5:
                    smaller, larger = (i, j) if h_a < h_b else (j, i)
                    embedded.append([larger, smaller, "embedded"])

    return support, embedded, hanging_candidates


def _hanging_relations(
    nodes: Dict[int, ObjNode],
    scene_height: float,
    z_fraction: float = 0.6,
) -> List[list]:
    """Objects near the top of the scene → hanging."""
    if scene_height <= 0:
        return []
    rels = []
    threshold_z = scene_height * z_fraction
    for nid, node in nodes.items():
        if node.z_min > threshold_z:
            rels.append([nid, -1, "hanging"])
    return rels


def _proximity_relations(
    nodes: Dict[int, ObjNode],
    max_dist: float = 2.0,
) -> List[list]:
    """Pairwise proximity for nearby objects."""
    ids = list(nodes.keys())
    rels = []
    for i, j in it.combinations(ids, 2):
        d = float(np.linalg.norm(nodes[i].position - nodes[j].position))
        if 0 < d <= max_dist:
            if d < max_dist * 0.33:
                label = "close to"
            elif d < max_dist * 0.66:
                label = "near"
            else:
                label = "far from"
            rels.append([i, j, label])
    return rels


def _aligned_objects(
    nodes: Dict[int, ObjNode],
    angle_threshold: float = 0.065,
) -> List[List[int]]:
    """Find groups of objects that are aligned along one axis."""
    ids = list(nodes.keys())
    if len(ids) < 2:
        return []

    groups: List[List[int]] = []
    for i, j in it.combinations(ids, 2):
        diff = nodes[i].position - nodes[j].position
        d = np.linalg.norm(diff)
        if d < 1e-6:
            continue
        # Check alignment with principal axes
        for axis in [np.array([1, 0, 0]), np.array([0, 1, 0])]:
            cos_angle = abs(np.dot(diff / d, axis))
            if cos_angle > 1.0 - angle_threshold:
                # Merge into existing group or start new
                merged = False
                for g in groups:
                    if i in g or j in g:
                        g.extend([i, j])
                        merged = True
                        break
                if not merged:
                    groups.append([i, j])
                break

    # Deduplicate within groups
    return [list(set(g)) for g in groups if len(set(g)) >= 2]


def _middle_relations(
    proximity_rels: List[list],
    nodes: Dict[int, ObjNode],
) -> List[list]:
    """Find objects in the middle of two others."""
    rels = []
    pairs = [(r[0], r[1]) for r in proximity_rels]
    ids = list(nodes.keys())

    for candidate in ids:
        # Find two other objects that are both near to candidate
        neighbours = [p[1] if p[0] == candidate else p[0]
                      for p in pairs if candidate in p[:2]]
        for a, b in it.combinations(neighbours, 2):
            if a == candidate or b == candidate:
                continue
            pos_c = nodes[candidate].position
            pos_a = nodes[a].position
            pos_b = nodes[b].position
            midpoint = (pos_a + pos_b) / 2.0
            dist_to_mid = np.linalg.norm(pos_c - midpoint)
            dist_ab = np.linalg.norm(pos_a - pos_b)
            if dist_ab > 0 and dist_to_mid / dist_ab < 0.3:
                rels.append([candidate, a, b, "in the middle of"])
    return rels


# ═══════════════════════════════════════════════════════════════════════════
# Main edge prediction entry point
# ═══════════════════════════════════════════════════════════════════════════

def predict_edges(
    graph: nx.MultiDiGraph,
    objects: List[TrackedObject],
    T_w_c: Optional[np.ndarray],
    depth_m: Optional[np.ndarray],
    intrinsics: CameraIntrinsics,
) -> nx.MultiDiGraph:
    """Populate *graph* with relationship edges.

    Mutates and returns *graph*.
    """
    # Build ObjNode dict
    node_dict: Dict[int, ObjNode] = {}
    for obj in objects:
        node_dict[obj.global_id] = _obj_node_from_tracked(obj)

    # Camera angle
    camera_angle = 0.0
    if T_w_c is not None:
        camera_angle = _camera_angle_from_pose(T_w_c)

    # Scene height
    scene_h = 0.0
    if depth_m is not None and T_w_c is not None:
        scene_h = estimate_scene_height(depth_m, T_w_c, intrinsics)

    # --- Relationships ---
    support_rels, embedded_rels, hanging_cands = _support_relations(node_dict)
    hanging_rels = _hanging_relations(node_dict, scene_h)
    proximity_rels = _proximity_relations(node_dict)
    aligned_groups = _aligned_objects(node_dict)
    middle_rels = _middle_relations(proximity_rels, node_dict)

    # Add edges
    for src, tgt, label in support_rels:
        graph.add_edge(src, tgt, label=label, label_class="support")
        graph.add_edge(tgt, src, label="supported by", label_class="oppo_support")

    for src, tgt, label in embedded_rels:
        graph.add_edge(src, tgt, label=label, label_class="embedded")

    for src, tgt, label in hanging_rels:
        graph.add_edge(src, tgt, label=label, label_class="hanging")

    for src, tgt, label in proximity_rels:
        graph.add_edge(src, tgt, label=label, label_class="proximity")

    for group in aligned_groups:
        tag = "aligned:" + ",".join(str(x) for x in group)
        for a, b in it.combinations(group, 2):
            graph.add_edge(a, b, label=tag, label_class="aligned_furniture")
            graph.add_edge(b, a, label=tag, label_class="aligned_furniture")

    for rel in middle_rels:
        src, a, b, label = rel
        graph.add_edge(src, a, label=f"in mid of {a} {b}", label_class="middle_furniture")
        graph.add_edge(src, b, label=f"in mid of {a} {b}", label_class="middle_furniture")

    return graph
