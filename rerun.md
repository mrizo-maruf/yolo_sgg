# Rerun Visualization

Interactive 3D/2D visualization of the YOLO tracking + scene-graph pipeline
using [Rerun](https://www.rerun.io/).

## Quick Start

```bash
# Install the SDK (one-time)
pip install rerun-sdk

# Run with Rerun enabled
python new_run.py --dataset isaacsim --scene_path /path/to/scene --rerun

# Also show scene-graph edges in 3D
python new_run.py --dataset isaacsim --scene_path /path/to/scene --rerun --vis_edge
```

When `--rerun` is passed, a Rerun Viewer window spawns automatically and
streams data in real time as frames are processed.

---

## Layout

The viewer opens with a **three-panel** layout:

```
┌─────────────────────────────┬───────────────────┐
│                             │  Semantic          │
│  3D Tracking & Scene Graph  │  Segmentation      │
│                             ├───────────────────┤
│                             │  RGB + 2D Boxes    │
└─────────────────────────────┴───────────────────┘
```

| Panel | Entity root | Description |
|-------|-------------|-------------|
| **3D Tracking & Scene Graph** | `world3d/` | Object point clouds, 3D bounding boxes, camera trajectory, graph edges |
| **Semantic Segmentation** | `seg_view/` | RGB image with coloured instance mask overlay |
| **RGB + 2D Boxes** | `rgb_view/` | RGB image with reprojected 3D bounding boxes drawn as 2D rectangles |

---

## What is Visualized

### Panel 1 — 3D Tracking & Scene Graph

| Element | Entity path | Details |
|---------|-------------|---------|
| **World axes** | `world3d/origin` | Static RGB arrows (X=red, Y=green, Z=blue), length 0.3 m |
| **Object point clouds** | `world3d/objects/points` | Accumulated 3D points for every tracked object, each colored by a deterministic hue derived from its `track_id` |
| **3D bounding boxes** | `world3d/objects/boxes` | Axis-aligned bounding boxes (AABBs) drawn as wireframe edges. **Green** = visible in the current frame, **Red** = not visible |
| **Camera pose** | `world3d/camera` | Current camera-to-world transform |
| **Camera frustum + RGB** | `world3d/camera/image` | Pinhole frustum pyramid with a small RGB thumbnail attached |
| **Camera trajectory** | `world3d/camera_trajectory` | Yellow polyline connecting all past camera positions |
| **Scene-graph edges** | `world3d/graph_edges/edge_*` | Orange arrows between object bbox centres (only when `--vis_edge` is set). Labels at midpoint of each edge show the relation type |

### Panel 2 — Semantic Segmentation

A copy of the RGB frame with instance masks painted on top.  Each mask is
filled with the same deterministic colour used for the 3D point cloud of that
object.

### Panel 3 — RGB + Reprojected 2D Boxes

The original RGB frame with 2D rectangles obtained by projecting each visible
object's 3D AABB corners into the current camera.  A label above each box
shows `<class_name> T:<track_id>`.

---

## Colour Scheme

Every track ID is mapped to a colour using the **golden-ratio hue** method:

```
hue = (track_id * 0.618033988749895) % 1.0
rgb = hsv_to_rgb(hue, 0.85, 0.95)
```

This ensures colours are maximally spread across the hue wheel and
remain consistent for the same object across all panels and frames.

---

## Architecture

```
new_run.py
  │
  ├── --rerun flag  →  creates RerunVisualizer
  │                     calls .init() with camera intrinsics
  │
  └── tracking loop
        │
        └── rerun_vis.log_frame(frame_idx, object_registry,
                                 global_graph, T_w_c, rgb_path,
                                 masks, track_ids, class_names,
                                 vis_edges)
              │
              ├── _log_3d()          → Panel 1
              ├── _log_segmentation()→ Panel 2
              └── _log_rgb_with_boxes()→ Panel 3
```

### Key classes / files

| File | Role |
|------|------|
| `rerun_utils.py` | `RerunVisualizer` class — all Rerun logging |
| `new_run.py` | Pipeline entry point; creates and drives the visualizer |
| `core/object_registry.py` | `get_all_pcds_for_visualization()` provides per-object points, bbox, visibility |
| `core/scene_graph.py` | `SceneGraph.global_graph` (networkx MultiDiGraph) used for edge visualization |

---

## CLI Flags

| Flag | Effect |
|------|--------|
| `--rerun` | Enable the Rerun viewer |
| `--vis_edge` | Draw scene-graph edges as 3D arrows in the 3D panel |

Both can be combined with any dataset:

```bash
python new_run.py --dataset thud_synthetic \
       --scene_path /data/Office/static/Capture_1 \
       --rerun --vis_edge
```

---

## Config

The `ssg` section in `configs/core_tracking.yaml` also controls Rerun:

```yaml
ssg:
  rerun: false       # set true to enable (or use --rerun CLI flag)
  vis_edge: false    # set true to show graph edges (or use --vis_edge)
```

---

## Adding Rerun to a New Dataset

No dataset-specific code is needed.  As long as your dataset loader
implements the standard `DatasetLoader` interface (returns RGB paths, poses,
depth, intrinsics), Rerun visualization works automatically through
`new_run.py --rerun`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: rerun` | `pip install rerun-sdk` |
| Viewer doesn't open | Check that a display is available; on headless servers use `rr.save("out.rrd")` instead of `rr.init(..., spawn=True)` |
| Point clouds are empty | The object registry accumulates points over frames — wait a few frames |
| Bounding boxes flicker | This is expected: only objects whose AABB has valid min/max are drawn |
