"""
Merge a local (per-frame) scene graph into the persistent global graph.

Strategy:
    • Nodes are keyed by global_id — adding twice is idempotent.
    • Edges accumulate across frames.  Duplicate (src, tgt, label)
      triples are collapsed; a ``weight`` attribute counts occurrences.
"""
from __future__ import annotations

from typing import List

import networkx as nx

from v2.types import TrackedObject


def merge_local_to_global(
    global_graph: nx.MultiDiGraph,
    local_graph: nx.MultiDiGraph,
    objects: List[TrackedObject],
    frame_idx: int,
) -> nx.MultiDiGraph:
    """Merge *local_graph* edges into *global_graph* (mutates in-place).

    Node attributes are always updated to the latest observation.
    Edge weights are incremented for repeated relationships.
    """
    # --- Nodes ---
    for obj in objects:
        global_graph.add_node(
            obj.global_id,
            class_name=obj.class_name,
            last_seen=frame_idx,
            bbox_3d=obj.bbox_3d.to_dict() if obj.bbox_3d else None,
        )

    # --- Edges ---
    for u, v, data in local_graph.edges(data=True):
        label = data.get("label", "")
        label_class = data.get("label_class", "")

        # Check if an identical edge already exists
        exists = False
        if global_graph.has_node(u) and global_graph.has_node(v):
            for _, _, edata in global_graph.edges(u, data=True):
                if edata.get("label") == label and edata.get("label_class") == label_class:
                    edata["weight"] = edata.get("weight", 1) + 1
                    edata["last_frame"] = frame_idx
                    exists = True
                    break

        if not exists:
            global_graph.add_edge(
                u, v,
                label=label,
                label_class=label_class,
                weight=1,
                first_frame=frame_idx,
                last_frame=frame_idx,
            )

    return global_graph
