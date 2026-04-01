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

# No external dependencies beyond rerun, cv2, numpy, colorsys

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


def _rigidize_camera_pose(T_w_c: np.ndarray) -> Optional[np.ndarray]:
    """Convert a possibly scaled Sim(3) pose into a rigid pose for rendering."""
    if T_w_c is None:
        return None

    T = np.asarray(T_w_c, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return None

    A = T[:3, :3]
    col_norms = np.linalg.norm(A, axis=0)
    scale = float(np.mean(col_norms))
    if not np.isfinite(scale) or scale < 1e-8:
        return None

    R_approx = A / scale
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt

    T_vis = np.eye(4, dtype=np.float32)
    T_vis[:3, :3] = R.astype(np.float32)
    T_vis[:3, 3] = T[:3, 3].astype(np.float32)
    return T_vis


def _build_axis_remap_matrix(
    swap_yz: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    """Build linear axis remap matrix A where p' = A @ p."""
    A = np.eye(3, dtype=np.float32)
    if swap_yz:
        A = A[[0, 2, 1], :]
    if flip_x:
        A[0, :] *= -1.0
    if flip_y:
        A[1, :] *= -1.0
    if flip_z:
        A[2, :] *= -1.0
    return A


def _apply_axis_remap_points(points: np.ndarray, axis_remap: np.ndarray) -> np.ndarray:
    """Apply axis remap to (N,3) points."""
    if points is None or points.size == 0:
        return points
    return (np.asarray(points, dtype=np.float32) @ axis_remap.T).astype(np.float32)


def _apply_axis_remap_transform(transform_4x4: np.ndarray, axis_remap: np.ndarray) -> np.ndarray:
    """Apply axis remap to a 4x4 world transform."""
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = axis_remap @ np.asarray(transform_4x4[:3, :3], dtype=np.float32)
    out[:3, 3] = axis_remap @ np.asarray(transform_4x4[:3, 3], dtype=np.float32)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

class RerunVisualizer:
    """Manages per-frame Rerun logging for the YOLO-SSG pipeline."""

    def __init__(self, recording_id: str = "yolo_ssg", axis_remap: Optional[np.ndarray] = None):
        self._camera_positions: List[np.ndarray] = []
        self._recording_id = recording_id
        self._initialized = False
        self._warned_invalid_camera_pose = False
        if axis_remap is None:
            self._axis_remap = np.eye(3, dtype=np.float32)
        else:
            A = np.asarray(axis_remap, dtype=np.float32)
            if A.shape != (3, 3):
                raise ValueError(f"axis_remap must be 3x3, got shape {A.shape}")
            self._axis_remap = A

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

        rr.log("world3d", rr.ViewCoordinates.RDF, static=True)
        rr.log("world3d/camera", rr.ViewCoordinates.RDF, static=True)

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

        rr.log(
            "world3d/camera/image",
            rr.Pinhole(
                resolution=[self._img_w, self._img_h],
                focal_length=[self._fx, self._fy],
                principal_point=[self._cx, self._cy],
                image_plane_distance=0.3,
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
        self._log_segmentation(rgb, masks_clean, track_ids, class_names)

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
            T_vis = _rigidize_camera_pose(T_w_c)
            if T_vis is None:
                if not self._warned_invalid_camera_pose:
                    print(
                        "[Rerun] Invalid/non-finite camera pose for frustum rendering. "
                        "Frustum will be hidden until poses become valid."
                    )
                    self._warned_invalid_camera_pose = True
            else:
                T_vis = _apply_axis_remap_transform(T_vis, self._axis_remap)
                rr.log(
                    "world3d/camera",
                    rr.Transform3D(
                        mat3x3=T_vis[:3, :3].astype(np.float32),
                        translation=T_vis[:3, 3].astype(np.float32),
                    ),
                )
                # Small RGB attached to the frustum
                small_rgb = cv2.resize(rgb, (self._img_w, self._img_h),
                                       interpolation=cv2.INTER_AREA)
                rr.log(
                    "world3d/camera/image/rgb",
                    rr.Image(small_rgb, color_model=rr.ColorModel.RGB),
                )
                self._camera_positions.append(T_vis[:3, 3].astype(np.float32))
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
            all_points = [_apply_axis_remap_points(p, self._axis_remap) for p in all_points]
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
            corners = _apply_axis_remap_points(corners, self._axis_remap)
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
            if bbox is None:
                continue
            # bbox may be a BBox3D dataclass or a dict
            if hasattr(bbox, "aabb_min"):
                c = (bbox.aabb_min + bbox.aabb_max) / 2.0
                centres[nid] = (self._axis_remap @ np.asarray(c, dtype=np.float32)).astype(np.float32)
            elif isinstance(bbox, dict) and bbox.get("aabb"):
                c = _aabb_center(bbox["aabb"])
                if c is not None:
                    centres[nid] = (self._axis_remap @ np.asarray(c, dtype=np.float32)).astype(np.float32)

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
                          track_ids: np.ndarray,
                          class_names: Optional[List[str]] = None):
        """Overlay coloured instance masks on the RGB image."""
        h, w = rgb.shape[:2]
        overlay = rgb.copy()

        for i, (mask, tid) in enumerate(zip(masks_clean, track_ids)):
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

            # Draw class name label at mask center
            cls = None
            if class_names and i < len(class_names):
                cls = class_names[i]
            if cls:
                ys, xs = np.where(binary)
                if len(ys) > 0:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.5
                    thickness = 1
                    (tw, th), baseline = cv2.getTextSize(cls, font, scale, thickness)
                    # Black background rectangle
                    pad = 3
                    cv2.rectangle(
                        overlay,
                        (cx - tw // 2 - pad, cy - th // 2 - pad),
                        (cx + tw // 2 + pad, cy + th // 2 + pad + baseline),
                        (0, 0, 0), cv2.FILLED,
                    )
                    # White text
                    cv2.putText(
                        overlay, cls,
                        (cx - tw // 2, cy + th // 2),
                        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
                    )

        rr.log(
            "seg_view/segmentation",
            rr.Image(overlay, color_model=rr.ColorModel.RGB),
        )

    # ------------------------------------------------------------------
    # Panel 3 – RGB with reprojected 3-D box wireframes
    # ------------------------------------------------------------------
    def _log_rgb_with_boxes(self, rgb: np.ndarray, object_registry,
                            T_w_c: Optional[np.ndarray],
                            img_w: int, img_h: int):
        """Draw reprojected 3-D AABB wireframes on the RGB image."""
        canvas = rgb.copy()

        if T_w_c is not None:
            T_c_w = np.linalg.inv(T_w_c)
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

                # Project all 8 corners into camera frame
                pts_cam = (T_c_w[:3, :3] @ corners_3d.T).T + T_c_w[:3, 3]
                # Per-corner projection (only draw edges where both endpoints are in front)
                px = np.full((8, 2), np.nan, dtype=np.float32)
                for i in range(8):
                    if pts_cam[i, 2] > 0:
                        px[i, 0] = (pts_cam[i, 0] / pts_cam[i, 2]) * self._fx + self._cx
                        px[i, 1] = (pts_cam[i, 1] / pts_cam[i, 2]) * self._fy + self._cy

                gid = obj["global_id"]
                color_bgr = tuple(int(c) for c in _track_color_u8(gid)[::-1])

                # Draw the 12 edges of the 3D box
                drawn = False
                for i0, i1 in _BOX_EDGES:
                    if np.isnan(px[i0, 0]) or np.isnan(px[i1, 0]):
                        continue
                    p0 = (int(round(px[i0, 0])), int(round(px[i0, 1])))
                    p1 = (int(round(px[i1, 0])), int(round(px[i1, 1])))
                    cv2.line(canvas, p0, p1, color_bgr, 2, cv2.LINE_AA)
                    drawn = True

                if not drawn:
                    continue

                # Label at the top-most visible corner
                valid = ~np.isnan(px[:, 0])
                if not np.any(valid):
                    continue
                top_idx = np.argmin(px[valid, 1])
                lx = int(round(px[valid][top_idx, 0]))
                ly = int(round(px[valid][top_idx, 1]))

                class_name = obj.get("class_name", "")
                label = f"{class_name} T:{gid}" if class_name else f"T:{gid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (lx, ly - th - 6), (lx + tw + 4, ly), color_bgr, -1)
                cv2.putText(canvas, label, (lx + 2, ly - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        rr.log(
            "rgb_view/image",
            rr.Image(canvas, color_model=rr.ColorModel.RGB),
        )
