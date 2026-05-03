#!/usr/bin/env python3
"""
Visualize scene-graph edges in 3D world frame, one source object at a time.

Usage examples
--------------
  python visualize_scene_graph_edges_3d.py results/scene_graphs/scene_0_2.json --edge-type sv
  python visualize_scene_graph_edges_3d.py results/scene_graphs/scene_0_2.json --edge-type edges_vlsat

Controls
--------
  q       : next source object
  esc     : exit
  close   : next source object
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


_BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3D world-frame scene-graph visualizer (step by source object)."
    )
    p.add_argument("scene_graph_json", type=str, help="Path to scene graph JSON.")
    p.add_argument(
        "--edge-type",
        type=str,
        default="sv",
        help=(
            "Which edges to visualize: sv | bs | vlsat or full key "
            "(edges_sv | edges_bs | edges_vlsat)."
        ),
    )
    p.add_argument(
        "--only-with-edges",
        action="store_true",
        help="Skip source objects that have zero outgoing edges for selected edge type.",
    )
    return p.parse_args()


def _edge_key_from_cli(raw: str) -> str:
    r = raw.strip().lower()
    mapping = {
        "sv": "edges_sv",
        "sceneverse": "edges_sv",
        "scenverse": "edges_sv",
        "bs": "edges_bs",
        "baseline": "edges_bs",
        "vlsat": "edges_vlsat",
        "edges_sv": "edges_sv",
        "edges_bs": "edges_bs",
        "edges_vlsat": "edges_vlsat",
    }
    if r not in mapping:
        allowed = "sv, bs, vlsat, edges_sv, edges_bs, edges_vlsat"
        raise ValueError(f"Unknown edge type '{raw}'. Allowed: {allowed}")
    return mapping[r]


def _as_np(v: Optional[Iterable[float]]) -> Optional[np.ndarray]:
    if v is None:
        return None
    arr = np.asarray(list(v), dtype=float)
    if arr.shape != (3,):
        return None
    return arr


def _node_center(node: dict) -> Optional[np.ndarray]:
    bbox = node.get("bbox_3d") or {}
    obb = bbox.get("obb") if isinstance(bbox, dict) else None
    if isinstance(obb, dict):
        c = _as_np(obb.get("center"))
        if c is not None:
            return c

    aabb = bbox.get("aabb") if isinstance(bbox, dict) else None
    if isinstance(aabb, dict):
        mn = _as_np(aabb.get("min"))
        mx = _as_np(aabb.get("max"))
        if mn is not None and mx is not None:
            return (mn + mx) * 0.5
    return None


def _bbox_corners(node: dict) -> Optional[np.ndarray]:
    bbox = node.get("bbox_3d") or {}
    if not isinstance(bbox, dict):
        return None

    aabb = bbox.get("aabb")
    if isinstance(aabb, dict):
        mn = _as_np(aabb.get("min"))
        mx = _as_np(aabb.get("max"))
        if mn is not None and mx is not None:
            return _corners_from_minmax(mn, mx)

    obb = bbox.get("obb")
    if isinstance(obb, dict):
        c = _as_np(obb.get("center"))
        e = _as_np(obb.get("extent"))
        if c is not None and e is not None:
            half = 0.5 * e
            mn = c - half
            mx = c + half
            return _corners_from_minmax(mn, mx)
    return None


def _corners_from_minmax(mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    x0, y0, z0 = mn.tolist()
    x1, y1, z1 = mx.tolist()
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )


def _set_axes_equal_3d(ax, points: np.ndarray) -> None:
    if points.size == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = 0.5 * (mn + mx)
    span = np.max(mx - mn)
    if span <= 1e-6:
        span = 1.0
    half = 0.6 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))


def _draw_world_axes(ax, axis_len: float) -> None:
    axis_len = float(max(axis_len, 0.5))
    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    ax.quiver(*origin, axis_len, 0, 0, color="r", linewidth=2)
    ax.quiver(*origin, 0, axis_len, 0, color="g", linewidth=2)
    ax.quiver(*origin, 0, 0, axis_len, color="b", linewidth=2)
    ax.text(axis_len, 0, 0, "Xw", color="r", fontsize=10)
    ax.text(0, axis_len, 0, "Yw", color="g", fontsize=10)
    ax.text(0, 0, axis_len, "Zw", color="b", fontsize=10)


def _draw_bbox(ax, corners: np.ndarray, color: str = "gray", lw: float = 1.0, alpha: float = 0.8) -> None:
    for i, j in _BOX_EDGES:
        p = corners[i]
        q = corners[j]
        ax.plot(
            [p[0], q[0]],
            [p[1], q[1]],
            [p[2], q[2]],
            color=color,
            linewidth=lw,
            alpha=alpha,
        )


def _sorted_node_ids(nodes: Dict[str, dict]) -> List[int]:
    out: List[int] = []
    for k in nodes.keys():
        try:
            out.append(int(k))
        except Exception:
            continue
    return sorted(out)


def _node_lookup(nodes: Dict[str, dict], nid: int) -> Optional[dict]:
    if str(nid) in nodes:
        return nodes[str(nid)]
    if nid in nodes:
        return nodes[nid]
    return None


def _collect_bounds(nodes: Dict[str, dict], node_ids: List[int]) -> np.ndarray:
    pts: List[np.ndarray] = []
    for nid in node_ids:
        node = _node_lookup(nodes, nid)
        if node is None:
            continue
        c = _node_center(node)
        if c is not None:
            pts.append(c)
        corners = _bbox_corners(node)
        if corners is not None:
            pts.extend(list(corners))
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def _render_source_object(
    nodes: Dict[str, dict],
    source_id: int,
    edge_key: str,
    world_points: np.ndarray,
) -> Tuple[bool, bool]:
    import matplotlib.pyplot as plt

    source_node = _node_lookup(nodes, source_id)
    if source_node is None:
        return True, False

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    spread = np.ptp(world_points, axis=0).max() if world_points.size else 1.0
    _draw_world_axes(ax, axis_len=0.18 * float(max(spread, 1.0)))

    # Draw all object bboxes
    node_ids = _sorted_node_ids(nodes)
    for nid in node_ids:
        node = _node_lookup(nodes, nid)
        if node is None:
            continue
        corners = _bbox_corners(node)
        if corners is None:
            continue
        color = "tab:red" if nid == source_id else "lightgray"
        lw = 2.0 if nid == source_id else 1.0
        _draw_bbox(ax, corners, color=color, lw=lw, alpha=0.9)

        center = _node_center(node)
        if center is not None:
            label = f"{nid}:{node.get('class_name', 'unknown')}"
            ax.text(center[0], center[1], center[2], label, fontsize=8, color="black")

    src_center = _node_center(source_node)
    outgoing = source_node.get(edge_key, []) if isinstance(source_node, dict) else []
    if src_center is not None and isinstance(outgoing, list):
        for edge_idx, e in enumerate(outgoing):
            tgt_id = int(e.get("target_id", -1))
            rel = str(e.get("relation_type", ""))
            tgt_node = _node_lookup(nodes, tgt_id)
            if tgt_node is None:
                continue
            tgt_center = _node_center(tgt_node)
            if tgt_center is None:
                continue

            ax.plot(
                [src_center[0], tgt_center[0]],
                [src_center[1], tgt_center[1]],
                [src_center[2], tgt_center[2]],
                color="tab:orange",
                linewidth=2.2,
                alpha=0.95,
            )
            mid = 0.5 * (src_center + tgt_center)
            # Small deterministic z-offset helps avoid fully-overlapping labels.
            mid[2] += 0.01 * (edge_idx % 5)
            ax.text(
                mid[0], mid[1], mid[2], rel,
                color="tab:blue", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
            )

    _set_axes_equal_3d(ax, world_points)
    ax.set_xlabel("X world")
    ax.set_ylabel("Y world")
    ax.set_zlabel("Z world")
    ax.set_title(
        f"Source object {source_id} ({source_node.get('class_name', 'unknown')}) | "
        f"edge key: {edge_key}\n"
        "Controls: q = next object, esc = exit, close window = next object"
    )

    state = {"next": False, "abort": False}

    def _on_key(event):
        if event.key == "q":
            state["next"] = True
            plt.close(fig)
        elif event.key == "escape":
            state["abort"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.tight_layout()
    plt.show()

    # Window closed without explicit key => move to next.
    if not state["next"] and not state["abort"]:
        state["next"] = True
    return state["next"], state["abort"]


def main() -> int:
    args = _parse_args()
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required for this visualizer. "
            "Install it with: pip install matplotlib"
        ) from exc

    json_path = Path(args.scene_graph_json).expanduser().resolve()
    if not json_path.exists():
        raise SystemExit(f"JSON not found: {json_path}")

    try:
        edge_key = _edge_key_from_cli(args.edge_type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with open(json_path, "r") as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict) or not nodes:
        raise SystemExit("No nodes found in scene graph JSON.")

    node_ids = _sorted_node_ids(nodes)
    if not node_ids:
        raise SystemExit("Could not parse node IDs from JSON.")

    if args.only_with_edges:
        filtered: List[int] = []
        for nid in node_ids:
            node = _node_lookup(nodes, nid)
            if node is None:
                continue
            if isinstance(node.get(edge_key, []), list) and len(node.get(edge_key, [])) > 0:
                filtered.append(nid)
        node_ids = filtered
        if not node_ids:
            raise SystemExit(f"No nodes have outgoing edges for '{edge_key}'.")

    world_points = _collect_bounds(nodes, node_ids)

    print(f"[viz] file: {json_path}")
    print(f"[viz] edge key: {edge_key}")
    print(f"[viz] objects to show: {len(node_ids)}")

    for idx, source_id in enumerate(node_ids):
        node = _node_lookup(nodes, source_id) or {}
        out_edges = node.get(edge_key, [])
        n_out = len(out_edges) if isinstance(out_edges, list) else 0
        print(f"[viz] ({idx + 1}/{len(node_ids)}) source={source_id}, outgoing={n_out}")
        _, abort = _render_source_object(nodes, source_id, edge_key, world_points)
        if abort:
            print("[viz] stopped by user.")
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
