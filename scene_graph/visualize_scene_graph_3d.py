#!/usr/bin/env python3
"""Visualize 3D scene graph: bounding boxes, relation edges, and relation labels.

Usage:
    python visualize_scene_graph_3d.py <path_to_json> [--no-labels]

Press 'q' to close current window and move to next source object.
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


def create_bbox_lineset(aabb_min, aabb_max, color):
    """Create an Open3D LineSet for a wireframe AABB box."""
    x0, y0, z0 = aabb_min
    x1, y1, z1 = aabb_max
    points = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def create_edge_line(p1, p2, color):
    """Create an Open3D LineSet for a single edge between two points."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector([p1, p2])
    ls.lines = o3d.utility.Vector2iVector([[0, 1]])
    ls.colors = o3d.utility.Vector3dVector([color])
    return ls


def get_center(node):
    aabb = node["bbox_3d"]["aabb"]
    return np.array([(aabb["min"][i] + aabb["max"][i]) / 2 for i in range(3)])


def create_world_frame(size=1.0):
    """Create XYZ axis lines at the origin (R=X, G=Y, B=Z)."""
    points = [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R=X, G=Y, B=Z
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


class SceneGraphViewer:
    def __init__(self, json_path, show_labels=True, edge_type="bbq"):
        with open(json_path) as f:
            self.data = json.load(f)
        self.nodes = self.data["nodes"]
        self.edge_key = f"edges_{edge_type}"
        self.src_ids = [
            sid for sid in sorted(self.nodes.keys(), key=int)
            if self.nodes[sid].get(self.edge_key)
        ]
        self.current_idx = 0
        self.show_labels = show_labels

        # Compute scene center (where the virtual camera sits)
        all_centers = [get_center(node) for node in self.nodes.values()]
        self.scene_center = np.mean(all_centers, axis=0)

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Scene Graph Viewer", 1400, 900
        )

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1, 1, 1, 1])
        self.window.add_child(self.scene)

        # Info label at top
        self.info = gui.Label("")
        self.window.add_child(self.info)

        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)

        self._geom_counter = 0
        self._geom_names = []
        self._label_widgets = []

        self._show_current()
        gui.Application.instance.run()

    def _on_layout(self, ctx):
        r = self.window.content_rect
        self.info.frame = gui.Rect(r.x, r.y, r.width, 30)
        self.scene.frame = gui.Rect(r.x, r.y + 30, r.width, r.height - 30)

    def _on_key(self, event):
        if event.key == gui.KeyName.Q and event.type == gui.KeyEvent.DOWN:
            self.current_idx += 1
            if self.current_idx >= len(self.src_ids):
                print("Done — all objects shown.")
                self.window.close()
                return True
            self._show_current()
            return True
        return False

    def _clear_scene(self):
        for name in self._geom_names:
            self.scene.scene.remove_geometry(name)
        self._geom_names.clear()
        for lbl in self._label_widgets:
            self.scene.remove_3d_label(lbl)
        self._label_widgets.clear()

    def _add_geometry(self, geom, color):
        name = f"geom_{self._geom_counter}"
        self._geom_counter += 1
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 3.0
        mat.base_color = [color[0], color[1], color[2], 1.0]
        self.scene.scene.add_geometry(name, geom, mat)
        self._geom_names.append(name)

    def _show_current(self):
        self._clear_scene()

        src_id = self.src_ids[self.current_idx]
        src_node = self.nodes[src_id]
        edges = src_node.get(self.edge_key, [])

        edges_by_target = defaultdict(list)
        for edge in edges:
            edges_by_target[str(edge["target_id"])].append(edge["relation_type"])

        title = (f"[{self.current_idx+1}/{len(self.src_ids)}] Object {src_id}: "
                 f"{src_node['class_name']} — {src_node.get('description', '')}  "
                 f"(press Q for next)")
        self.info.text = title
        print(f"\n--- {title} ---")
        print(f"    {len(edges)} edges to {len(edges_by_target)} targets\n")

        # World frame at origin
        self._add_geometry(create_world_frame(size=0.5), [1, 1, 1])
        # Axis labels
        for axis_label, pos, clr in [("X", [0.55, 0, 0], gui.Color(1, 0, 0)),
                                      ("Y", [0, 0.55, 0], gui.Color(0, 0.7, 0)),
                                      ("Z", [0, 0, 0.55], gui.Color(0, 0, 1))]:
            lbl = self.scene.add_3d_label(pos, axis_label)
            lbl.color = clr
            self._label_widgets.append(lbl)

        # Virtual camera marker at scene center
        sc = self.scene_center
        cam_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
        cam_sphere.translate(sc)
        cam_sphere.paint_uniform_color([1.0, 0.5, 0.0])  # orange
        cam_name = f"geom_{self._geom_counter}"
        self._geom_counter += 1
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 0.5, 0.0, 1.0]
        self.scene.scene.add_geometry(cam_name, cam_sphere, mat)
        self._geom_names.append(cam_name)
        lbl = self.scene.add_3d_label(sc + np.array([0, 0, 0.12]), "Virtual Camera (scene center)")
        lbl.color = gui.Color(1.0, 0.5, 0.0)
        self._label_widgets.append(lbl)

        # Source bbox (red)
        src_aabb = src_node["bbox_3d"]["aabb"]
        self._add_geometry(
            create_bbox_lineset(src_aabb["min"], src_aabb["max"], [1, 0, 0]),
            [1, 0, 0],
        )
        src_center = get_center(src_node)
        lbl = self.scene.add_3d_label(src_center, f"{src_node['class_name']} ({src_id})")
        lbl.color = gui.Color(1, 0, 0)
        self._label_widgets.append(lbl)

        all_points = [src_aabb["min"], src_aabb["max"]]

        for target_id, relations in edges_by_target.items():
            if target_id not in self.nodes:
                continue
            target_node = self.nodes[target_id]
            tgt_aabb = target_node["bbox_3d"]["aabb"]

            # Target bbox (green)
            self._add_geometry(
                create_bbox_lineset(tgt_aabb["min"], tgt_aabb["max"], [0, 0.7, 0]),
                [0, 0.7, 0],
            )
            tgt_center = get_center(target_node)
            lbl = self.scene.add_3d_label(
                tgt_center, f"{target_node['class_name']} ({target_id})"
            )
            lbl.color = gui.Color(0, 0.5, 0)
            self._label_widgets.append(lbl)

            all_points.extend([tgt_aabb["min"], tgt_aabb["max"]])

            # Edge line (blue)
            self._add_geometry(
                create_edge_line(src_center, tgt_center, [0, 0, 1]),
                [0, 0, 1],
            )

            # Relation text at midpoint (black)
            if self.show_labels:
                mid = (src_center + tgt_center) / 2
                label_text = ", ".join(relations)
                lbl = self.scene.add_3d_label(mid, label_text)
                lbl.color = gui.Color(0, 0, 0)
                self._label_widgets.append(lbl)

        # Fit camera to scene bounds
        all_points = np.array(all_points)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            all_points.min(axis=0) - 0.5, all_points.max(axis=0) + 0.5
        )
        self.scene.setup_camera(60, bbox, bbox.get_center())


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D scene graph bounding boxes and relations."
    )
    parser.add_argument("json_path", help="Path to the scene graph JSON file.")
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Hide relation text labels on edges.",
    )
    parser.add_argument(
        "--edge-type", type=str, default="bbq",
        choices=["sv", "bbq", "vlsat"],
        help="Which edge predictor to visualize (default: bbq).",
    )
    args = parser.parse_args()
    SceneGraphViewer(args.json_path, show_labels=not args.no_labels,
                     edge_type=args.edge_type)


if __name__ == "__main__":
    main()
