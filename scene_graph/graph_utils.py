"""Internal helpers for SceneGraph — conflict rules, bbox conversion, JSON IO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np


def are_conflicting(cls1: str, cls2: str) -> bool:
    """Two allocentric basic subclasses that cannot coexist on the same edge."""
    conflicts = {
        "support": {"embedded", "hanging"},
        "embedded": {"support", "hanging"},
        "hanging": {"support", "embedded"},
    }
    return cls2 in conflicts.get(cls1, set())


def bbox3d_to_dict(bbox) -> Optional[Dict]:
    """Convert a ``BBox3D`` dataclass (or None/dict) to a plain dict."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        return bbox
    return bbox.to_dict()


def save_graph_json(graph: nx.MultiDiGraph, path: Path) -> None:
    """Serialize a MultiDiGraph to JSON with nodes + outgoing edges."""
    nodes = {}
    for nid, ndata in graph.nodes(data=True):
        d = ndata.get("data", {})
        edges_out = []
        for _, tgt, _, edata in graph.out_edges(nid, keys=True, data=True):
            edge_entry = {
                "target_id": int(tgt),
                "label": edata.get("label", ""),
                "label_class": edata.get("label_class", ""),
            }
            if edata.get("label_subclass"):
                edge_entry["label_subclass"] = edata["label_subclass"]
            edges_out.append(edge_entry)
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
    """JSON serializer for numpy scalars / arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
