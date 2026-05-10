"""Rerun debug visualisation for the benchmark — single-scene only.

Logs four 2-D views per frame so a human can eyeball where tracking and
matching disagree with ground truth:

  1. **gt_view**     – RGB + reprojected GT 3-D boxes + 2-D masks + labels.
                       Box colour: green = visible in current frame, red = not.
  2. **pred_view**   – RGB + reprojected predicted 3-D boxes + masks.
                       Mask colour follows track_id; box colour = visible
                       (matched / freshly observed) vs invisible (reprojection).
  3. **yolo_view**   – RGB + raw YOLO 2-D boxes + masks + labels (per-track
                       colour). The ``upstream`` view, before 3-D matching.
  4. **match_view**  – RGB overlaid with both predicted and GT projected boxes,
                       coloured by match status:
                         orange = matched prediction
                         red    = unmatched prediction
                         blue   = matched GT (only)

Designed to be cheap to wire in: ``log_frame`` is the single per-frame call;
all 2-D projection and per-track-colouring is done internally.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover — only required when --rerun is passed
    rr = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_GREEN  = (0, 200, 0)
_RED    = (220, 0, 0)
_ORANGE = (255, 140, 0)
_BLUE   = (40, 100, 255)


def _track_id_color(tid: int) -> Tuple[int, int, int]:
    """Deterministic, well-separated RGB colour from a track id."""
    if tid is None or tid < 0:
        return (180, 180, 180)
    # Golden-ratio spacing in hue space; saturation/value fixed for vibrancy.
    import colorsys
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# 3-D bbox projection
# ---------------------------------------------------------------------------

# 12 edges of a cuboid (indices into the 8-corner array)
_BBOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _aabb_corners(xmin: float, ymin: float, zmin: float,
                  xmax: float, ymax: float, zmax: float) -> np.ndarray:
    """Return 8 corners of an axis-aligned 3-D box, in world frame."""
    return np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin],
        [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax],
        [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float64)


def _project_corners(
    corners_world: np.ndarray,
    T_w_c: np.ndarray,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points to (N,2) pixels and return their camera-Z.

    Negative Z means behind the camera; caller decides what to do with those.
    """
    T_c_w = np.linalg.inv(T_w_c)
    homo = np.concatenate([corners_world, np.ones((corners_world.shape[0], 1))], axis=1)
    cam = (T_c_w @ homo.T).T[:, :3]
    Z = cam[:, 2]
    eps = 1e-6
    Zc = np.where(np.abs(Z) < eps, eps, Z)
    u = K[0, 0] * cam[:, 0] / Zc + K[0, 2]
    v = K[1, 1] * cam[:, 1] / Zc + K[1, 2]
    return np.stack([u, v], axis=1), Z


def _bbox_edges_2d(
    corners_world: np.ndarray,
    T_w_c: np.ndarray,
    K: np.ndarray,
) -> Optional[List[np.ndarray]]:
    """Return a list of (2,2) edge segments in pixel space, or None if no
    edge has both endpoints in front of the camera.
    """
    pix, Z = _project_corners(corners_world, T_w_c, K)
    strips: List[np.ndarray] = []
    for a, b in _BBOX_EDGES:
        if Z[a] > 0 and Z[b] > 0:
            strips.append(np.array([pix[a], pix[b]], dtype=np.float32))
    return strips if strips else None


def _is_in_view(
    corners_world: np.ndarray,
    T_w_c: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
) -> bool:
    """A bbox is 'in view' if at least one corner is in front of the camera
    and projects inside the image rectangle (with a small margin)."""
    pix, Z = _project_corners(corners_world, T_w_c, K)
    in_front = Z > 0
    if not np.any(in_front):
        return False
    margin = 0.0
    inside = (
        (pix[:, 0] >= -margin) & (pix[:, 0] <= img_w + margin) &
        (pix[:, 1] >= -margin) & (pix[:, 1] <= img_h + margin)
    )
    return bool(np.any(in_front & inside))


def _bbox3d_world_corners(bbox_3d) -> Optional[np.ndarray]:
    """Get 8 world-frame corners from a registry-style ``BBox3D`` (AABB)."""
    if bbox_3d is None:
        return None
    mn = np.asarray(bbox_3d.aabb_min, dtype=np.float64)
    mx = np.asarray(bbox_3d.aabb_max, dtype=np.float64)
    if mn.shape != (3,) or mx.shape != (3,):
        return None
    return _aabb_corners(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2])


def _xyzxyz_world_corners(bbox_xyzxyz) -> Optional[np.ndarray]:
    """Get 8 world-frame corners from a ``(xmin,…,zmax)`` GT tuple."""
    if bbox_xyzxyz is None or len(bbox_xyzxyz) != 6:
        return None
    return _aabb_corners(*[float(v) for v in bbox_xyzxyz])


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """Tight (x1,y1,x2,y2) bounding rect from a non-empty boolean mask."""
    if mask is None:
        return None
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _build_segmentation_image(
    masks: Sequence[np.ndarray],
    ids: Sequence[int],
    img_h: int,
    img_w: int,
) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int]]]:
    """Compose a uint16 instance-segmentation image where pixel value == id.

    Returns the image and a {id: rgb} colour map suitable for an
    ``rr.AnnotationContext``.
    """
    seg = np.zeros((img_h, img_w), dtype=np.uint16)
    palette: Dict[int, Tuple[int, int, int]] = {}
    for mask, mid in zip(masks, ids):
        if mask is None or mid is None or mid < 0:
            continue
        m = np.asarray(mask).astype(bool)
        if m.shape[0] != img_h or m.shape[1] != img_w:
            # Defensive: nearest-resize. Should not happen in practice.
            import cv2
            m = cv2.resize(
                m.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        seg[m] = int(mid)
        palette[int(mid)] = _track_id_color(int(mid))
    return seg, palette


# ---------------------------------------------------------------------------
# Public visualizer
# ---------------------------------------------------------------------------

class BenchmarkDebugVisualizer:
    """Single-scene benchmark debug viewer (Rerun).

    Usage::

        vis = BenchmarkDebugVisualizer(recording_id="bench")
        vis.init(img_w, img_h, fx, fy, cx, cy)
        for tf in run_tracking(...):
            ...
            vis.log_frame(tf, gt_instances, pred_instances, mapping,
                          matched_gids=matched_gids)
    """

    GT_VIEW   = "gt_view"
    PRED_VIEW = "pred_view"
    YOLO_VIEW = "yolo_view"
    MATCH_VIEW = "match_view"

    def __init__(self, recording_id: str = "bench_debug") -> None:
        if rr is None:
            raise ImportError(
                "rerun-sdk is not installed; install with `pip install rerun-sdk` "
                "or run without --rerun."
            )
        self._recording_id = recording_id
        self._initialized = False
        self._img_w = 0
        self._img_h = 0
        self._K: Optional[np.ndarray] = None

    # -- setup ---------------------------------------------------------------

    def init(
        self,
        img_w: int,
        img_h: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        spawn: bool = True,
    ) -> None:
        self._img_w = int(img_w)
        self._img_h = int(img_h)
        self._K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        rr.init(self._recording_id, spawn=spawn)

        from rerun.blueprint import Blueprint, Grid, Spatial2DView

        blueprint = Blueprint(
            Grid(
                Spatial2DView(name="GT",          origin=self.GT_VIEW),
                Spatial2DView(name="Predictions", origin=self.PRED_VIEW),
                Spatial2DView(name="YOLO 2D",     origin=self.YOLO_VIEW),
                Spatial2DView(name="Matching",    origin=self.MATCH_VIEW),
            ),
        )
        rr.send_blueprint(blueprint)
        self._initialized = True

    # -- per-frame logging ---------------------------------------------------

    def log_frame(
        self,
        *,
        frame_idx: int,
        rgb: np.ndarray,
        T_w_c: Optional[np.ndarray],
        gt_instances: List,
        pred_objects: List,
        gt_to_pred: Dict[int, int],
        matched_gids: Optional[set] = None,
    ) -> None:
        """Log one frame's four debug views.

        Parameters
        ----------
        gt_instances
            ``List[GTInstance]`` for the current frame.
        pred_objects
            ``List[TrackedObject]`` from ``TrackedFrame.objects``.
        gt_to_pred
            ``{gt_track_id: pred_id}`` from the matcher (``mapping``).
        matched_gids
            Set of predicted ``global_id``s that were observed (matched
            against YOLO) in the current frame. Used to colour
            "currently-visible" vs "reprojection-only" predictions.
        """
        if not self._initialized:
            return
        # rr.set_time_sequence was removed in rerun ≥ 0.16; use rr.set_time instead.
        try:
            rr.set_time(timeline="frame", sequence=int(frame_idx))
        except AttributeError:
            rr.set_time_sequence("frame", int(frame_idx))

        # All four views share the RGB background.
        rgb_u8 = self._coerce_rgb(rgb)
        for view in (self.GT_VIEW, self.PRED_VIEW, self.YOLO_VIEW, self.MATCH_VIEW):
            rr.log(f"{view}/image", rr.Image(rgb_u8))

        # Sanity-guard: most overlays need a pose + intrinsics.
        if T_w_c is not None and self._K is not None:
            self._log_gt_view(gt_instances, T_w_c)
            self._log_pred_view(pred_objects, T_w_c, matched_gids or set())
            self._log_match_view(
                gt_instances, pred_objects, T_w_c, gt_to_pred,
            )

        # YOLO view is image-space only; works without pose.
        self._log_yolo_view(pred_objects)

    # ====================================================================
    # Per-view loggers
    # ====================================================================

    def _log_gt_view(self, gt_instances, T_w_c: np.ndarray) -> None:
        view = self.GT_VIEW
        edges_visible: List[np.ndarray] = []
        edges_invisible: List[np.ndarray] = []
        labels_visible: List[Tuple[Tuple[float, float], str]] = []
        labels_invisible: List[Tuple[Tuple[float, float], str]] = []
        masks: List[np.ndarray] = []
        ids: List[int] = []

        for gi in gt_instances:
            corners = _xyzxyz_world_corners(getattr(gi, "bbox_xyzxyz", None))
            visible_in_view = False
            if corners is not None:
                visible_in_view = _is_in_view(
                    corners, T_w_c, self._K, self._img_w, self._img_h,
                )
                strips = _bbox_edges_2d(corners, T_w_c, self._K)
                if strips:
                    bucket_e = edges_visible if visible_in_view else edges_invisible
                    bucket_l = labels_visible if visible_in_view else labels_invisible
                    bucket_e.extend(strips)
                    # Place label at the highest-on-screen corner
                    pix, Z = _project_corners(corners, T_w_c, self._K)
                    in_front = pix[Z > 0]
                    if in_front.size:
                        anchor = in_front[np.argmin(in_front[:, 1])]
                        bucket_l.append(
                            ((float(anchor[0]), float(anchor[1])),
                             f"{gi.class_name}#{gi.track_id}"),
                        )
            mask = getattr(gi, "mask", None)
            if mask is not None:
                masks.append(mask)
                ids.append(int(gi.track_id))

        self._log_linestrips(f"{view}/bbox_visible",   edges_visible,   _GREEN)
        self._log_linestrips(f"{view}/bbox_invisible", edges_invisible, _RED)
        self._log_text_labels(f"{view}/labels_visible",   labels_visible,   _GREEN)
        self._log_text_labels(f"{view}/labels_invisible", labels_invisible, _RED)
        self._log_seg(f"{view}/mask", masks, ids)

    def _log_pred_view(
        self,
        pred_objects,
        T_w_c: np.ndarray,
        matched_gids: set,
    ) -> None:
        view = self.PRED_VIEW
        edges_visible: List[np.ndarray] = []
        edges_invisible: List[np.ndarray] = []
        labels_visible: List[Tuple[Tuple[float, float], str]] = []
        labels_invisible: List[Tuple[Tuple[float, float], str]] = []
        masks: List[np.ndarray] = []
        ids: List[int] = []

        for obj in pred_objects:
            corners = _bbox3d_world_corners(getattr(obj, "bbox_3d", None))
            is_observed = obj.global_id in matched_gids
            if corners is not None:
                strips = _bbox_edges_2d(corners, T_w_c, self._K)
                if strips:
                    bucket_e = edges_visible if is_observed else edges_invisible
                    bucket_l = labels_visible if is_observed else labels_invisible
                    bucket_e.extend(strips)
                    pix, Z = _project_corners(corners, T_w_c, self._K)
                    in_front = pix[Z > 0]
                    if in_front.size:
                        anchor = in_front[np.argmin(in_front[:, 1])]
                        cls = obj.class_name or "obj"
                        bucket_l.append(
                            ((float(anchor[0]), float(anchor[1])),
                             f"{cls}#{obj.global_id}"),
                        )
            if obj.mask is not None:
                masks.append(obj.mask)
                ids.append(int(obj.global_id))

        self._log_linestrips(f"{view}/bbox_visible",   edges_visible,   _GREEN)
        self._log_linestrips(f"{view}/bbox_invisible", edges_invisible, _RED)
        self._log_text_labels(f"{view}/labels_visible",   labels_visible,   _GREEN)
        self._log_text_labels(f"{view}/labels_invisible", labels_invisible, _RED)
        self._log_seg(f"{view}/mask", masks, ids)

    def _log_yolo_view(self, pred_objects) -> None:
        """RGB + 2D bbox (from mask) + mask + per-track-id colour."""
        view = self.YOLO_VIEW
        bboxes_xyxy: List[Tuple[float, float, float, float]] = []
        labels: List[str] = []
        colors: List[Tuple[int, int, int]] = []
        masks: List[np.ndarray] = []
        ids: List[int] = []

        for obj in pred_objects:
            if obj.mask is None:
                continue
            xyxy = _bbox_from_mask(obj.mask)
            if xyxy is None:
                continue
            bboxes_xyxy.append(xyxy)
            labels.append(f"{obj.class_name or 'obj'}#{obj.yolo_id}")
            colors.append(_track_id_color(int(obj.yolo_id)))
            masks.append(obj.mask)
            ids.append(int(obj.yolo_id))

        if bboxes_xyxy:
            arr = np.asarray(bboxes_xyxy, dtype=np.float32)
            rr.log(
                f"{view}/bbox",
                rr.Boxes2D(
                    array=arr, array_format=rr.Box2DFormat.XYXY,
                    labels=labels, colors=np.asarray(colors, dtype=np.uint8),
                ),
            )
        else:
            rr.log(f"{view}/bbox", rr.Clear(recursive=False))

        self._log_seg(f"{view}/mask", masks, ids)

    def _log_match_view(
        self,
        gt_instances,
        pred_objects,
        T_w_c: np.ndarray,
        gt_to_pred: Dict[int, int],
    ) -> None:
        """RGB + matched/unmatched predictions + matched-GT boxes."""
        view = self.MATCH_VIEW
        # invert the mapping for fast lookup
        matched_pred_ids = set(gt_to_pred.values())
        matched_gt_track_ids = set(gt_to_pred.keys())

        edges_match:    List[np.ndarray] = []
        edges_unmatched: List[np.ndarray] = []
        edges_gt:       List[np.ndarray] = []
        labels_match:    List[Tuple[Tuple[float, float], str]] = []
        labels_unmatched: List[Tuple[Tuple[float, float], str]] = []

        for obj in pred_objects:
            corners = _bbox3d_world_corners(getattr(obj, "bbox_3d", None))
            if corners is None:
                continue
            strips = _bbox_edges_2d(corners, T_w_c, self._K)
            if not strips:
                continue
            is_match = int(obj.global_id) in matched_pred_ids
            (edges_match if is_match else edges_unmatched).extend(strips)
            pix, Z = _project_corners(corners, T_w_c, self._K)
            in_front = pix[Z > 0]
            if in_front.size:
                anchor = in_front[np.argmin(in_front[:, 1])]
                tag = f"{obj.class_name or 'obj'}#{obj.global_id}"
                (labels_match if is_match else labels_unmatched).append(
                    ((float(anchor[0]), float(anchor[1])), tag),
                )

        for gi in gt_instances:
            if gi.track_id not in matched_gt_track_ids:
                continue
            corners = _xyzxyz_world_corners(getattr(gi, "bbox_xyzxyz", None))
            if corners is None:
                continue
            strips = _bbox_edges_2d(corners, T_w_c, self._K)
            if strips:
                edges_gt.extend(strips)

        self._log_linestrips(f"{view}/pred_match",   edges_match,    _ORANGE)
        self._log_linestrips(f"{view}/pred_unmatch", edges_unmatched, _RED)
        self._log_linestrips(f"{view}/gt_match",     edges_gt,       _BLUE)
        self._log_text_labels(f"{view}/labels_match",    labels_match,    _ORANGE)
        self._log_text_labels(f"{view}/labels_unmatch", labels_unmatched, _RED)

    # ====================================================================
    # Low-level rerun helpers
    # ====================================================================

    def _log_linestrips(self, path: str, strips: List[np.ndarray], color) -> None:
        if strips:
            rr.log(
                path,
                rr.LineStrips2D(strips=strips,
                                 colors=np.asarray(color, dtype=np.uint8),
                                 radii=1.5),
            )
        else:
            rr.log(path, rr.Clear(recursive=False))

    def _log_text_labels(
        self,
        path: str,
        items: List[Tuple[Tuple[float, float], str]],
        color,
    ) -> None:
        if not items:
            rr.log(path, rr.Clear(recursive=False))
            return
        positions = np.asarray([p for p, _ in items], dtype=np.float32)
        labels = [t for _, t in items]
        rr.log(
            path,
            rr.Points2D(
                positions=positions, labels=labels,
                colors=np.asarray(color, dtype=np.uint8),
                radii=2.0,
            ),
        )

    def _log_seg(
        self,
        path: str,
        masks: Sequence[np.ndarray],
        ids: Sequence[int],
    ) -> None:
        if not masks:
            rr.log(path, rr.Clear(recursive=False))
            return
        seg, palette = _build_segmentation_image(
            masks, ids, self._img_h, self._img_w,
        )
        ann_classes = [
            rr.AnnotationInfo(id=int(k), color=v) for k, v in palette.items()
        ]
        rr.log(
            path,
            [
                rr.AnnotationContext(ann_classes),
                rr.SegmentationImage(seg),
            ],
        )

    @staticmethod
    def _coerce_rgb(rgb: np.ndarray) -> np.ndarray:
        """Ensure (H,W,3) uint8 RGB suitable for ``rr.Image``."""
        a = np.asarray(rgb)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.shape[2] == 4:
            a = a[..., :3]
        return a
