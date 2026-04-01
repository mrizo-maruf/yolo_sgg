# Edge Documentation

This document explains how scene-graph edges are stored and how direction labels
(`left`, `right`, `front`, `behind/back`, `above`, `below`) are defined in each edge predictor.

## 1) How Edges Are Stored In JSON

In saved scene-graph JSON files, each node usually has one or more edge lists:

- `edges_sv`    : SceneVerse/basic edges
- `edges_bs`    : baseline geometric edges
- `edges_vlsat` : VL-SAT neural edges

Each list item is:

```json
{
  "target_id": <int>,
  "relation_type": <string>
}
```

Interpretation:

- If this entry is inside node `A`, then it is a directed edge `A -> target_id`.
- `relation_type` is the relation predicted for that directed pair.

## 2) SceneVerse / Basic Edges (`edges_sv`)

Implementation path:

- `ssg/ssg_main.py` -> `edges(...)`
- proximity logic: `ssg/relationships/proximity.py`
- hanging/vertical logic: `ssg/relationships/hanging.py`
- relation text generation: `ssg/ssg_utils.py::generate_relation(...)`

### Direction reference frame

SceneVerse directional proximity is **camera-yaw-relative**:

1. Object XY centers are rotated by `camera_angle` (`cw_rotate`), where
   `camera_angle` comes from camera pose (`view_from_pose`).
2. A 2D direction bucket is computed.
3. It is mapped to natural-language direction:
   - `to the left of`
   - `to the right of`
   - `in front of`
   - `behind`

So this is not a fixed world-axis threshold rule. It depends on camera heading.

### Above / below in SceneVerse

Vertical relations are created in `hanging.py` by Z comparisons (`src_min > tgt_max` etc.)
and `generate_relation(..., 'high')`, which yields both:

- `src -> tgt : above`
- `tgt -> src : below`

### Note about wording

Some relation text is dictionary-driven (for example support/inside/beside/on), so labels
are not all axis words.

## 3) Baseline Edges In `kg_nav_run.py` (`edges_bs`)

Implementation path:

- `kg_nav_run.py` -> `edges_bs(...)`
- world center projected to camera frame with `_egoview_project(...)`

Camera-frame axes used there:

- `X`: camera-right
- `Y`: camera-up
- `Z`: camera-forward (as documented in code)

Rules for `src -> tgt`:

- `tgt.x < src.x` => `left`
- `tgt.x > src.x` => `right`
- `tgt.z < src.z` => `front`
- `tgt.z > src.z` => `back`
- `tgt.y < src.y` => `above`
- `tgt.y > src.y` => `below`

So here, direction is explicit thresholding in camera frame.

## 4) Baseline Edges In `new_run.py` Path (core SceneGraph)

Implementation path:

- `core/scene_graph.py` -> `BaselineEdgePredictor`
- `_camera_relations(src, tgt)`

This implementation is also camera-frame threshold based, but currently uses:

- `tgt.z > src.z` => `front`
- `tgt.z < src.z` => `back`

while left/right and above/below follow the same sign pattern as above.

Important: this means `front/back` sign is currently opposite between:

- `kg_nav_run.py::edges_bs`
- `core/scene_graph.py::_camera_relations`

## 5) VL-SAT Edges (`edges_vlsat`)

Implementation paths:

- `kg_nav_run.py::edges_vlsat(...)`
- `core/scene_graph.py::VLSATEdgePredictor`
- model wrapper: `vl_sat_model/vl_sat_interface.py`
- label list: `vl_sat_model/config/relationships.txt`

### Direction reference frame

VL-SAT is **learned**, not hard-threshold axis rules.

- Input object point clouds are converted from pipeline world convention to VL-SAT convention:
  - `(x, y, z)_ours -> (x, z, -y)_vlsat`
- For each ordered pair `(src, tgt)`, the model predicts relation class(es).
- Directional labels (`left/right/front/behind/higher than/lower than`) come from learned classes.

So for VL-SAT, direction meaning is defined by training data + pair order, not by direct
hand-coded `if x/y/z` thresholds.

## 6) Practical Reading Guide

If you inspect one edge entry under node `A`:

- It always means **directed relation for pair `A -> target_id`**.
- For `edges_bs`: relation is from camera-frame threshold rules.
- For `edges_sv`: relation is SceneVerse logic (camera-yaw-relative for proximity, Z-based for above/below).
- For `edges_vlsat`: relation is neural prediction for ordered pair.
