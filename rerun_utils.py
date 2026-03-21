"""
rerun_utils.py — Rerun visualization for the YOLO-SSG tracking + scene-graph pipeline.

Three panels:
  1. **3D Reconstruction & Tracking**
     - Object point clouds colored by track_id
     - 3D AABBs: green = visible, red = not visible
     - Camera trajectory (yellow) + camera frame axes
     - Persistent scene-graph edges (lines between bbox centres with relation labels)
  2. **Semantic Segmentation** — clean masks overlay
  3. **RGB + Reprojected 2D Boxes** — class name & track_id annotations
"""
from __future__ import annotations

import colorsys
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import rerun as rr
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'rerun' package is required.  Install with: pip install rerun-sdk"
    ) from exc

import YOLOE.utils as yutils

# ───────────────────────────────────────────────────────────────────────────
# Colour helpers
# ───────────────────────────────────────────────────────────────────────────

def _track_color_u8(track_id: int) -> np.ndarray:
    """Deterministic colour for a track_id (uint8 RGB)."""
    hue = (int(track_id) * 0.618033988749895) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (np.array(rgb) * 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────────────────
# AABB helpers
# ───────────────────────────────────────────────────────────────────────────

_BOX_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)


def _aabb_corners(aabb: dict) -> Optional[np.ndarray]:
    """Return (8, 3) corners from an aabb dict with 'min' and 'max'."""
    mn = aabb.get("min")
    mx = aabb.get("max")
    if mn is None or mx is None:
        return None
    mn = np.asarray(mn, dtype=np.float32)
    mx = np.asarray(mx, dtype=np.float32)
    return np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)


def _aabb_center(aabb: dict) -> Optional[np.ndarray]:
    mn = aabb.get("min")
    mx = aabb.get("max")
    if mn is None or mx is None:
        return None
    return (np.asarray(mn, dtype=np.float32) + np.asarray(mx, dtype=np.float32)) / 2.0


# ───────────────────────────────────────────────────────────────────────────
# Projection helpers  (world 3D → image 2D)
# ───────────────────────────────────────────────────────────────────────────

def _project_points(pts_world: np.ndarray, T_w_c: np.ndarray,
                    fx: float, fy: float, cx: float, cy: float,
                    img_w: int, img_h: int) -> Optional[np.ndarray]:
    """Project (N, 3) world points → (N, 2) pixel coords.  Returns None on failure."""
    if pts_world.shape[0] == 0:
        return None
    T_c_w = np.linalg.inv(T_w_c)
    pts_cam = (T_c_w[:3, :3] @ pts_world.T).T + T_c_w[:3, 3]
    valid = pts_cam[:, 2] > 0
    if not np.any(valid):
        return None
    pts_cam = pts_cam[valid]
    u = (pts_cam[:, 0] / pts_cam[:, 2]) * fx + cx
    v = (pts_cam[:, 1] / pts_cam[:, 2]) * fy + cy
    return np.stack([u, v], axis=1).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

class RerunVisualizer:
    """Manages per-frame Rerun logging for the YOLO-SSG pipeline."""

    def __init__(self, recording_id: str = "yolo_ssg"):
        self._camera_positions: List[np.ndarray] = []
        self._recording_id = recording_id
        self._initialized = False

    # ------------------------------------------------------------------
    # Init  (call once before the tracking loop)
    # ------------------------------------------------------------------
    def init(self, img_w: int, img_h: int,
             fx: float, fy: float, cx: float, cy: float,
             spawn: bool = True):
        """Initialise rerun recording and static blueprints."""
        self._img_w = img_w
        self._img_h = img_h
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

        rr.init(self._recording_id, spawn=spawn)

        # ── Blueprint: 3 columns ──
        from rerun.blueprint import (
            Blueprint,
            Horizontal,
            Spatial3DView,
            Spatial2DView,
            Vertical,
        )

        blueprint = Blueprint(
            Horizontal(
                Spatial3DView(
                    name="3D Tracking & Scene Graph",
                    origin="world3d",
                ),
                Vertical(
                    Spatial2DView(
                        name="Semantic Segmentation",
                        origin="seg_view",
                    ),
                    Spatial2DView(
                        name="RGB + 2D Boxes",
                        origin="rgb_view",
                    ),
                ),
                column_shares=[2, 1],
            ),
        )
        rr.send_blueprint(blueprint)

        # Static: world coordinate axes
        rr.log(
            "world3d/origin",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * 0.3,
                colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            ),
            static=True,
        )

        self._initialized = True

    # ------------------------------------------------------------------
    # Per-frame update  (call inside the tracking loop)
    # ------------------------------------------------------------------
    def log_frame(
        self,
        frame_idx: int,
        object_registry,
        persistent_graph,
        T_w_c: Optional[np.ndarray],
        rgb_path: str,
        masks_clean: List[np.ndarray],
        track_ids: np.ndarray,
        class_names: Optional[List[str]],
        vis_edges: bool = False,
    ):
        """Log all visualisation data for a single frame."""
        if not self._initialized:
            raise RuntimeError("Call RerunVisualizer.init() before log_frame()")

        rr.set_time(timeline="frame", sequence=int(frame_idx))

        rgb_bgr = cv2.imread(rgb_path)
        if rgb_bgr is None:
            return
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = rgb.shape[:2]

        # === Panel 1: 3-D Reconstruction & Tracking ========================
        self._log_3d(frame_idx, object_registry, persistent_graph, T_w_c,
                     rgb, vis_edges)

        # === Panel 2: Semantic Segmentation =================================
        self._log_segmentation(rgb, masks_clean, track_ids)

        # === Panel 3: RGB + Reprojected 2-D Boxes ===========================
        self._log_rgb_with_boxes(rgb, object_registry, T_w_c, img_w, img_h)

    # ------------------------------------------------------------------
    # Panel 1 – 3-D
    # ------------------------------------------------------------------
    def _log_3d(self, frame_idx: int, object_registry, persistent_graph,
                T_w_c, rgb: np.ndarray, vis_edges: bool):
        all_vis = object_registry.get_all_pcds_for_visualization()

        # ── Camera pose, frustum with RGB thumbnail & trajectory ──
        if T_w_c is not None:
            rr.log(
                "world3d/camera",
                rr.Transform3D(
                    mat3x3=T_w_c[:3, :3].astype(np.float32),
                    translation=T_w_c[:3, 3].astype(np.float32),
                ),
            )
            # Pinhole frustum (shows as a pyramid in the 3-D view)
            rr.log(
                "world3d/camera/image",
                rr.Pinhole(
                    resolution=[self._img_w, self._img_h],
                    focal_length=[self._fx, self._fy],
                    principal_point=[self._cx, self._cy],
                    image_plane_distance=0.3,
                ),
            )
            # Small RGB attached to the frustum
            small_rgb = cv2.resize(rgb, (self._img_w, self._img_h),
                                   interpolation=cv2.INTER_AREA)
            rr.log(
                "world3d/camera/image/rgb",
                rr.Image(small_rgb, color_model=rr.ColorModel.RGB),
            )
            self._camera_positions.append(T_w_c[:3, 3].astype(np.float32))
            if len(self._camera_positions) >= 2:
                rr.log(
                    "world3d/camera_trajectory",
                    rr.LineStrips3D(
                        strips=[np.array(self._camera_positions, dtype=np.float32)],
                        colors=[[255, 255, 0]],  # yellow
                    ),
                )

        # ── Object point clouds ──
        all_points = []
        all_colors = []
        for obj in all_vis:
            pts = obj.get("points")
            if pts is None or len(pts) == 0:
                continue
            color = _track_color_u8(obj["global_id"])
            all_points.append(pts.astype(np.float32))
            all_colors.append(np.tile(color, (len(pts), 1)))

        if all_points:
            rr.log(
                "world3d/objects/points",
                rr.Points3D(
                    np.concatenate(all_points, axis=0),
                    colors=np.concatenate(all_colors, axis=0),
                    radii=np.full(sum(len(p) for p in all_points), 0.008, dtype=np.float32),
                ),
            )

        # ── 3-D bounding boxes (green=visible, red=not visible) ──
        strips = []
        strip_colors = []
        for obj in all_vis:
            bbox = obj.get("bbox_3d")
            if bbox is None:
                continue
            aabb = bbox.get("aabb")
            if aabb is None:
                continue
            corners = _aabb_corners(aabb)
            if corners is None:
                continue
            is_vis = obj.get("visible_current_frame", False)
            color = [0, 255, 0] if is_vis else [255, 0, 0]  # green / red
            edge_pts = corners[_BOX_EDGES]  # (12, 2, 3)
            strips.extend(edge_pts)
            strip_colors.extend([color] * len(edge_pts))

        if strips:
            rr.log(
                "world3d/objects/boxes",
                rr.LineStrips3D(
                    strips=np.array(strips, dtype=np.float32),
                    colors=np.array(strip_colors, dtype=np.uint8),
                ),
            )

        # ── Scene-graph edges (persistent graph) ──
        if vis_edges:
            self._log_graph_edges(persistent_graph, object_registry)
        else:
            # Clear any leftover edges from a previous toggle
            if hasattr(rr, "Clear"):
                rr.log("world3d/graph_edges", rr.Clear(recursive=True))

    # ------------------------------------------------------------------
    # Scene-graph edges in 3-D
    # ------------------------------------------------------------------
    def _log_graph_edges(self, graph, object_registry):
        """Draw edges of the persistent graph as 3D arrows with relation labels.

        Each edge is logged as a separate entity so that Rerun always
        renders every label (batch labels can be culled at distance).
        """
        # Clear previous per-edge entities
        if hasattr(rr, "Clear"):
            rr.log("world3d/graph_edges", rr.Clear(recursive=True))

        if graph is None or graph.number_of_edges() == 0:
            return

        all_objs = object_registry.get_all_objects()

        # Pre-compute centres for each node
        centres: Dict[int, np.ndarray] = {}
        for nid in graph.nodes:
            obj = all_objs.get(nid, {})
            bbox = obj.get("bbox_3d")
            if bbox and bbox.get("aabb"):
                c = _aabb_center(bbox["aabb"])
                if c is not None:
                    centres[nid] = c

        edge_color = np.array([255, 180, 50], dtype=np.uint8)   # orange
        label_color = np.array([255, 255, 255], dtype=np.uint8)  # white

        idx = 0
        for u, v, _key, data in graph.edges(keys=True, data=True):
            cu = centres.get(u)
            cv = centres.get(v)
            if cu is None or cv is None:
                continue
            label = data.get("label", data.get("label_class", ""))

            # Arrow from source to target (90 % of the way to avoid overlap)
            direction = cv - cu
            rr.log(
                f"world3d/graph_edges/edge_{idx}",
                rr.Arrows3D(
                    origins=[cu.astype(np.float32)],
                    vectors=[(direction * 0.9).astype(np.float32)],
                    colors=[edge_color],
                ),
            )

            # Label at midpoint
            mid = ((cu + cv) / 2.0).astype(np.float32)
            rr.log(
                f"world3d/graph_edges/lbl_{idx}",
                rr.Points3D(
                    [mid],
                    labels=[label],
                    radii=[0.025],
                    colors=[label_color],
                ),
            )
            idx += 1

    # ------------------------------------------------------------------
    # Panel 2 – Semantic Segmentation
    # ------------------------------------------------------------------
    def _log_segmentation(self, rgb: np.ndarray,
                          masks_clean: List[np.ndarray],
                          track_ids: np.ndarray):
        """Overlay coloured instance masks on the RGB image."""
        h, w = rgb.shape[:2]
        overlay = rgb.copy()

        for mask, tid in zip(masks_clean, track_ids):
            if mask is None:
                continue
            # Ensure mask is binary uint8 matching image dims
            m = mask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = m > 127 if m.max() > 1 else m > 0
            color = _track_color_u8(int(tid))
            for c in range(3):
                overlay[:, :, c] = np.where(binary, color[c], overlay[:, :, c])

        rr.log(
            "seg_view/segmentation",
            rr.Image(overlay, color_model=rr.ColorModel.RGB),
        )

    # ------------------------------------------------------------------
    # Panel 3 – RGB with reprojected 2-D boxes
    # ------------------------------------------------------------------
    def _log_rgb_with_boxes(self, rgb: np.ndarray, object_registry,
                            T_w_c: Optional[np.ndarray],
                            img_w: int, img_h: int):
        """Draw reprojected 3-D AABBs as 2-D rectangles on the RGB image."""
        canvas = rgb.copy()

        if T_w_c is not None:
            all_vis = object_registry.get_all_pcds_for_visualization()
            for obj in all_vis:
                if not obj.get("visible_current_frame", False):
                    continue
                bbox = obj.get("bbox_3d")
                if bbox is None or bbox.get("aabb") is None:
                    continue
                corners_3d = _aabb_corners(bbox["aabb"])
                if corners_3d is None:
                    continue

                px = _project_points(
                    corners_3d, T_w_c,
                    self._fx, self._fy, self._cx, self._cy,
                    img_w, img_h,
                )
                if px is None or len(px) < 2:
                    continue

                x1 = max(0, int(np.min(px[:, 0])))
                y1 = max(0, int(np.min(px[:, 1])))
                x2 = min(img_w - 1, int(np.max(px[:, 0])))
                y2 = min(img_h - 1, int(np.max(px[:, 1])))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                gid = obj["global_id"]
                color_bgr = tuple(int(c) for c in _track_color_u8(gid)[::-1])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, 2)

                class_name = obj.get("class_name", "")
                label = f"{class_name} T:{gid}" if class_name else f"T:{gid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
                cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        rr.log(
            "rgb_view/image",
            rr.Image(canvas, color_model=rr.ColorModel.RGB),
        )
