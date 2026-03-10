#!/usr/bin/env python3
"""
THUD Tracking Benchmark Runner
===============================

Adapted from ``benchmark_tracking.py`` for THUD synthetic scenes.
Uses ``THUDSyntheticLoader`` for ground-truth loading and handles
THUD-specific depth encoding, camera intrinsics, PNG-based RGB
images, and scene discovery.

Modes
-----
1. **Single-scene**  (default)

       python benchmark_tracking_thud.py \\
           /data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1 --vis

2. **Multi-scene benchmark**

       python benchmark_tracking_thud.py \\
           /data/THUD_Robot --multi

   Iterates over all ``Capture_*`` directories under
   ``Synthetic_Scenes/<scene>/<type>/`` and produces per-scene +
   aggregated metrics.

Metrics (dataset-agnostic, computed in ``metrics.tracking_metrics``)
--------------------------------------------------------------------
T-mIoU, T-SR, ID Switches, MOTA, MOTP, per-class breakdown.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure working directory (project root) is on sys.path so that
# top-level packages (YOLOE, thud_utils, metrics, …) are importable
# regardless of how the script is invoked.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False

# -- project imports -----------------------------------------------------------
import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry
from Pi3.utils import process_depth_model

from thud_utils.thud_synthetic_loader import (
    THUDSyntheticLoader,
    GTObject,
    discover_thud_synthetic_scenes,
)
from thud_utils.real_scene_loader import (
    RealSceneDataLoader,
    Object2D as RealObject2D,
    Object3D as RealObject3D,
    discover_real_scenes,
)

# -- decoupled metrics & vis ---------------------------------------------------
from metrics.tracking_metrics import (
    FrameRecord,
    GTInstance,
    MetricsAccumulator,
    PredInstance,
    match_greedy,
    print_summary,
    save_metrics,
)
from benchmark.visualization import (
    draw_masks_with_labels,
    plot_results,
    visualize_matching,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Default configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    # --- YOLO ---
    "yolo_model": "yoloe-11l-seg.pt",
    "depth_model": "yyfz233/Pi3X",  # set None if you want to use original depth
    "conf": 0.25,
    "iou": 0.5,
    # --- Mask pre-processing ---
    "kernel_size": 11,
    "alpha": 0.7,
    "fast_mask": True,
    # --- Point cloud ---
    "max_points_per_obj": 2000,
    "max_accumulated_points": 10000,
    "o3_nb_neighbors": 50,
    "o3std_ratio": 0.1,
    # --- Tracking registry ---
    "tracking_overlap_threshold": 0.3,
    "tracking_distance_threshold": 0.5,
    "tracking_inactive_limit": 0,
    "tracking_volume_ratio_threshold": 0.1,
    "reprojection_visibility_threshold": 0.2,
    # --- Matching ---
    "iou_threshold": 0.3,
    # --- Real_Scenes GT tracking ---
    "real_tracking_distance": 0.3,
    # --- THUD-specific ---
    "thud_root": "/home/yehia/rizo/THUD_Robot",
    "scene_type": "static",          # "static" or "dynamic"
    "depth_scale": 1000.0,           # raw uint16 → metres  (raw / depth_scale)
    "depth_max_m": 100.0,            # clamp ceiling (metres)
    # --- Classes to ignore (large structural / background) ---
    "skip_classes": [
        "wall", "floor", "ceiling", "roof", "carpet", "mat", "ground",
        "workspace", "workplace",
        "stairway", "stairs", "elevator",
        "room", "kitchen", "bathroom", "bedroom", "living room",
        "dining room", "office", "hallway", "corridor", "lobby",
        "garage", "basement", "attic",
        "sky", "ground", "grass", "field", "lawn",
        "building", "house", "warehouse",
        "road", "street", "sidewalk", "parking",
    ],
    # --- Visualisation ---
    "visualization": {
        "enabled": False,
        "interval": 10,
        "show_matching": True,
        "show_2d": True,
        "show_3d": False,
        "show_3d_bg": False,
        "show_windows": True,
        "save_dir": None,
        "max_bg_points_3d": 50000,
        "max_obj_points_3d": 3000,
    },
}

THUD_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background",
}


# ═══════════════════════════════════════════════════════════════════════════════
# THUD-specific helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _override_yutils_intrinsics(K: np.ndarray, img_h: int, img_w: int) -> None:
    """Monkey-patch the global camera intrinsics in ``YOLOE.utils``
    so that ``extract_points_from_mask`` uses the correct values for the
    current THUD scene.
    """
    yutils.fx = float(K[0, 0])
    yutils.fy = float(K[1, 1])
    yutils.cx = float(K[0, 2])
    yutils.cy = float(K[1, 2])
    yutils.IMAGE_WIDTH = img_w
    yutils.IMAGE_HEIGHT = img_h


def _load_thud_depth_raw(depth_path: str) -> np.ndarray:
    """Load a THUD depth PNG as raw float32 (no unit conversion).

    The THUD synthetic pipeline uses non-standard depth encoding that
    requires special scale / offset formulas (see
    ``THUDSyntheticLoader.depth_to_pointcloud``).  We therefore keep the
    raw uint16 values and let the custom point-extractor handle the
    conversion to 3-D points.

    Returns
    -------
    np.ndarray
        (H, W) float32, zero marks invalid pixels.
    """
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return np.zeros((1, 1), dtype=np.float32)
    return d.astype(np.float32)


def _make_thud_point_extractor(camera_intrinsic: np.ndarray):
    """Return a closure that extracts per-mask 3-D points using the THUD
    depth-to-pointcloud formulas.

    The returned function replaces the default
    ``extract_points_from_mask`` + ``cam_to_world`` pipeline inside
    ``create_3d_objects_with_tracking``.  Points are produced directly
    in the THUD *scene* coordinate frame (x-right, y-forward, z-up) so
    no additional camera-to-world transform is needed.

    Parameters
    ----------
    camera_intrinsic : np.ndarray
        3 × 3 intrinsic matrix from the THUD JSON annotation.

    Returns
    -------
    callable
        ``extractor(depth_raw, mask, frame_idx, max_points,
        o3_nb_neighbors, o3std_ratio, track_id) -> np.ndarray (N, 3)``
    """
    _fx = float(camera_intrinsic[0, 0])
    _fy = float(camera_intrinsic[1, 1])
    _cx = float(camera_intrinsic[0, 2])
    _cy = float(camera_intrinsic[1, 2])

    def _extractor(
        depth_raw: np.ndarray,
        mask: np.ndarray,
        frame_idx: int,
        max_points: int,
        o3_nb_neighbors: int,
        o3std_ratio: float,
        track_id: int,
    ) -> np.ndarray:
        H, W = depth_raw.shape[:2]
        depth_f = depth_raw.astype(np.float32)

        # Normalise mask to boolean
        if mask.dtype != bool:
            mask_bool = mask.squeeze() > 0
        else:
            mask_bool = mask.squeeze()

        valid = mask_bool & (depth_f > 0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        # Pixel grids – v is flipped (top→bottom) as in the THUD pipeline
        u_grid = np.arange(W, dtype=np.float32)
        v_flip  = np.arange(H - 1, -1, -1, dtype=np.float32)
        u_grid, v_flip = np.meshgrid(u_grid, v_flip)  # (H, W) each

        # Raw unprojection (THUD Depth_to_pointcloud.py formula)
        Xr = (u_grid - _cx) * depth_f / _fx
        Zr = (v_flip  - _cy) * depth_f / _fy

        # THUD-specific scale / offset + mm → m  (ExportPointCloud / 1000)
        X = Xr / 2.5 / 1000.0
        Y = (depth_f / 6.5 + 200.0) / 1000.0   # forward (depth direction)
        Z = (Zr / 2.0 + 300.0) / 1000.0         # up (flipped-v direction)

        pts = np.stack([X[valid], Y[valid], Z[valid]], axis=-1).astype(np.float32)

        # Sub-sample
        rng = np.random.default_rng(track_id)
        n = len(pts)
        if max_points and n > max_points:
            idx = rng.choice(n, size=max_points, replace=False)
            pts = pts[idx]
        elif n > 6:
            idx = rng.choice(n, size=max(1, int(n * 0.5)), replace=False)
            pts = pts[idx]

        # Statistical outlier removal (same as standard pipeline)
        if pts.shape[0] > 3:
            try:
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                _, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=o3_nb_neighbors, std_ratio=o3std_ratio,
                )
                pcd = pcd.select_by_index(ind)
                pts = np.asarray(pcd.points).astype(np.float32)
            except Exception:
                pass

        return pts

    return _extractor


def _load_real_depth_as_meters(
    depth_path: str,
    scale: float = 1000.0,
    max_m: float = 100.0,
) -> np.ndarray:
    """Load a depth PNG and convert to float32 metres (standard pipeline).

    Used for *real* THUD scenes where the depth is stored as raw uint16
    millimetres.  NOT suitable for THUD *synthetic* scenes (use
    ``_make_thud_point_extractor`` instead).
    """
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if d is None:
        return np.zeros((1, 1), dtype=np.float32)
    depth_m = d.astype(np.float32) / scale
    depth_m[depth_m < 0.01] = 0.0
    depth_m[depth_m > max_m] = 0.0
    return depth_m


def _track_objects_thud(
    rgb_paths: List[str],
    depth_paths: List[str],
    model_path: str = "yoloe-11l-seg-pf.pt",
    conf: float = 0.3,
    iou: float = 0.5,
):
    """Generator that runs YOLO tracking on THUD PNG images.

    Mirrors ``yutils.track_objects_in_video_stream`` but works with
    PNG RGB files instead of JPG.
    """
    from ultralytics import YOLOE as YOLOEModel

    model = YOLOEModel(model_path)
    office_class_names_synth = ["chair", "sofa", "bookshelf", "desk",
                         "table", "plants", "bag", "curtain", "laptop", "comuter",
                         "clock", "TV", "people", "cup", "ceramics", "book", "pot", "folder", "light", "PlantRack", "airconditioner"]
    house_class_names_synth = [
        "Chair",
        "Bench",
        "Table",
        "Sofa",
        "bookshelf",
        "Fridge",
        "Desk",
        "Bed"
    ]
    
    gym_class_names_synth = [
        "Radiator",
        "Bench",
        "Ball",
        "Rowing Machine",
        "Treadmill"
    ]

    super_market_1_class_names_synth = [
        "Shopping Cart",
        "Shelf",
        "Freezer",
        "Stall",
        "Checkout",
        "Fridge",
        "Cabinet",
        "Table",
        "door"
    ]
    
    
    model.set_classes(office_class_names_synth) if office_class_names_synth else None

    for ip, rgb_p in enumerate(rgb_paths):
        rgb = cv2.imread(rgb_p)
        if rgb is None:
            print(f"[WARN] Could not read image: {rgb_p}")
            continue
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        out = model.track(
            source=[rgb_input],
            tracker=yutils.TRACKER_CFG,
            device=yutils.DEVICE,
            conf=conf,
            verbose=False,
            persist=True,
            agnostic_nms=True,
        )
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        dp = depth_paths[ip] if ip < len(depth_paths) else ""
        yield res, rgb_p, dp


def _should_skip_class(name: str, skip_set: Set[str]) -> bool:
    if name is None:
        return False
    low = name.lower()
    if low in skip_set:
        return True
    for kw in ("room", "shot", "carpet", "yard", "floor", "mat", "resort"):
        if kw in low:
            return True
    return False


def _gt_objects_to_instances(gt_objects: List[GTObject]) -> List[GTInstance]:
    """Convert THUD GT objects to dataset-agnostic ``GTInstance``."""
    return [
        GTInstance(
            track_id=g.track_id,
            class_name=g.class_name,
            mask=g.mask,
            bbox_xyxy=g.bbox2d_xyxy,
        )
        for g in gt_objects
    ]


def _color_rgb01_from_id(obj_id: int) -> np.ndarray:
    rng = np.random.RandomState(int(obj_id) * 97 + 17)
    return np.array(
        [
            rng.uniform(0.2, 1.0),
            rng.uniform(0.2, 1.0),
            rng.uniform(0.2, 1.0),
        ],
        dtype=np.float64,
    )


def _lineset_from_bbox3d(bbox_3d: Optional[Dict], color_rgb01: np.ndarray):
    if not HAS_OPEN3D or not bbox_3d:
        return None

    obb = bbox_3d.get("obb") if isinstance(bbox_3d, dict) else None
    if obb and obb.get("center") is not None and obb.get("extent") is not None:
        center = np.asarray(obb.get("center"), dtype=np.float64)
        extent = np.asarray(obb.get("extent"), dtype=np.float64)
        R = np.asarray(obb.get("R", np.eye(3)), dtype=np.float64)
        try:
            obb_geom = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
            ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb_geom)
            ls.paint_uniform_color(color_rgb01.tolist())
            return ls
        except Exception:
            pass

    aabb = bbox_3d.get("aabb") if isinstance(bbox_3d, dict) else None
    if aabb and aabb.get("min") is not None and aabb.get("max") is not None:
        mn = np.asarray(aabb.get("min"), dtype=np.float64)
        mx = np.asarray(aabb.get("max"), dtype=np.float64)
        try:
            aabb_geom = o3d.geometry.AxisAlignedBoundingBox(min_bound=mn, max_bound=mx)
            ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb_geom)
            ls.paint_uniform_color(color_rgb01.tolist())
            return ls
        except Exception:
            return None
    return None


def _visualize_tracking_3d(
    frame_idx: int,
    frame_data,
    frame_objs: List[Dict],
    object_registry: GlobalObjectRegistry,
    max_bg_points: int = 50000,
    max_obj_points: int = 3000,
    include_background_cloud: bool = False,
) -> None:
    """Render tracked objects in 3D with point clouds and bounding boxes.

    Point clouds are fetched from *object_registry* (the lightweight
    ``frame_objs`` returned by ``process_frame`` do not carry points).
    """
    if not HAS_OPEN3D:
        print("[WARN] open3d not installed; skipping 3D visualization")
        return

    geoms = []

    # Optional background scene cloud from THUD depth (+ RGB colors)
    if include_background_cloud and (
        frame_data is not None
        and frame_data.depth is not None
        and frame_data.camera_intrinsic is not None
    ):
        pts = THUDSyntheticLoader.depth_to_pointcloud(
            depth=frame_data.depth,
            camera_intrinsic=frame_data.camera_intrinsic,
            rgb=frame_data.rgb,
            num_sample=max_bg_points,
        )
        xyz = pts[:, :3].astype(np.float64)

        # Align with world-frame tracked objects when camera pose is available
        if frame_data.cam_transform_4x4 is not None:
            T_w_c = np.asarray(frame_data.cam_transform_4x4, dtype=np.float64)
            ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
            xyz_h = np.concatenate([xyz, ones], axis=1)
            xyz = (T_w_c @ xyz_h.T).T[:, :3]

        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(xyz)
        if pts.shape[1] >= 6:
            pcd_bg.colors = o3d.utility.Vector3dVector(
                np.clip(pts[:, 3:6], 0.0, 1.0).astype(np.float64)
            )
        else:
            pcd_bg.paint_uniform_color([0.6, 0.6, 0.6])
        geoms.append(pcd_bg)

    # Fetch full object data (with accumulated PCDs) from the registry
    registry_objects = object_registry.get_all_objects()  # gid -> obj dict

    # Build set of visible (current-frame) global IDs for highlighting
    visible_gids = {int(obj.get("global_id", -1)) for obj in frame_objs}
    # Map gid -> frame_obj for current-frame metadata (yolo_track_id etc.)
    frame_obj_by_gid = {int(o.get("global_id", -1)): o for o in frame_objs}

    n_visible = len(visible_gids - {-1})
    n_inactive = len(registry_objects) - n_visible
    print(
        f"\n[3D TRACKING] Frame {frame_idx} | "
        f"visible={n_visible} | inactive={n_inactive} | "
        f"total_registry={len(registry_objects)}"
    )

    # Iterate over ALL registry objects so inactive ones are also shown
    for gid, reg_entry in registry_objects.items():
        gid = int(gid)
        is_visible = gid in visible_gids
        fo = frame_obj_by_gid.get(gid)

        yid = -1
        cname = reg_entry.get("class_name") or "unknown"
        if fo is not None:
            yid = int(fo.get("yolo_track_id", -1)) if fo.get("yolo_track_id") is not None else -1
            cname = fo.get("class_name") or cname

        points = reg_entry.get("points_accumulated")
        bbox_3d = (fo.get("bbox_3d") if fo else None) or reg_entry.get("bbox_3d")

        color = _color_rgb01_from_id(gid)
        # Dim inactive objects (blend toward gray)
        if not is_visible:
            color = color * 0.4 + np.array([0.3, 0.3, 0.3]) * 0.6

        npts = 0
        if isinstance(points, np.ndarray) and points.size > 0:
            p = points.astype(np.float64)
            npts = p.shape[0]
            if npts > max_obj_points:
                sel = np.random.choice(npts, size=max_obj_points, replace=False)
                p = p[sel]
            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(p)
            pcd_obj.paint_uniform_color(color.tolist())
            geoms.append(pcd_obj)

        lines = _lineset_from_bbox3d(bbox_3d, color)
        if lines is not None:
            geoms.append(lines)

        center_txt = "n/a"
        if isinstance(bbox_3d, dict):
            obb = bbox_3d.get("obb")
            if isinstance(obb, dict) and obb.get("center") is not None:
                c = np.asarray(obb.get("center"), dtype=np.float64)
                center_txt = f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"

        status = "VISIBLE" if is_visible else "inactive"
        print(
            f"  gid={gid:>3d} yolo_id={yid:>3d} class={cname:<16s} "
            f"points={npts:>5d} center={center_txt}  [{status}]"
        )

    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4))
    n_total = len(registry_objects)
    o3d.visualization.draw_geometries(
        geoms,
        window_name=(
            f"THUD Tracking 3D | frame {frame_idx} | {n_visible} visible + {n_inactive} inactive = {n_total}"
        ),
        width=1280,
        height=800,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Single-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_scene(scene_path: str, cfg: OmegaConf) -> Dict:
    """Run tracking + evaluation on one THUD capture directory."""

    scene_dir = Path(scene_path)
    # Build a human-readable scene label from the path
    # e.g.  .../Gym/static/Capture_1 → Gym/static/Capture_1
    try:
        parts = scene_dir.parts
        idx = next(i for i, p in enumerate(parts) if p in (
            "Gym", "House", "Office", "Supermarket_1", "Supermarket_2",
        ))
        scene_label = "/".join(parts[idx:])
    except StopIteration:
        scene_label = scene_dir.name

    print(f"\n{'=' * 60}")
    print(f"  THUD BENCHMARK – {scene_label}")
    print(f"{'=' * 60}")

    # --- load GT data via THUDSyntheticLoader --------------------------------
    loader = THUDSyntheticLoader(
        str(scene_dir),
        load_rgb=True,
        load_depth=True,
        skip_labels=THUD_SKIP_LABELS,
        verbose=False,
    )
    frame_indices = loader.frame_indices
    n_frames = len(frame_indices)
    print(f"Frames: {n_frames}")
    if n_frames == 0:
        print("[WARN] No frames discovered by THUDSyntheticLoader. Skipping scene.")
        return {}

    # Build loader-synchronized (frame_idx, rgb_path, depth_path) tuples.
    # This guarantees GT, RGB, and depth remain aligned in the benchmark loop.
    valid_entries: List[tuple[int, str, str]] = []
    for fidx in frame_indices:
        rp = loader.rgb_dir / f"rgb_{fidx}.png"
        dp = loader.depth_dir / f"depth_{fidx}.png"
        if rp.exists() and dp.exists():
            valid_entries.append((fidx, str(rp), str(dp)))

    if not valid_entries:
        print("[WARN] No frames with both RGB and depth found. Skipping scene.")
        return {}

    if len(valid_entries) != n_frames:
        print(
            f"[WARN] Using {len(valid_entries)}/{n_frames} frames with both RGB+depth "
            f"(loader remains source of truth for frame order)."
        )

    frame_indices_eval = [e[0] for e in valid_entries]
    rgb_paths = [e[1] for e in valid_entries]
    depth_paths = [e[2] for e in valid_entries]
    n_eval_frames = len(frame_indices_eval)
    
    # --- Process depth with Pi3X model if configured ----------------------------
    rgb_dir = str(loader.rgb_dir)
    depth_dir = str(loader.depth_dir)
    
    temp_cfg = OmegaConf.create({
        'rgb_dir': rgb_dir,
        'depth_dir': depth_dir,
        'traj_path': str(scene_dir / "camera_poses.txt"),  # Placeholder
        'depth_model': cfg.get('depth_model', None)
    })
    temp_cfg = process_depth_model(temp_cfg)
    
    # Update depth paths if depth_dir was changed
    if temp_cfg.depth_dir != depth_dir:
        depth_dir = temp_cfg.depth_dir
        print(f"Using processed depth from: {depth_dir}")
        # Rebuild depth_paths with new directory
        depth_paths = []
        for fidx in frame_indices_eval:
            dp = Path(depth_dir) / f"depth_{fidx:04d}.png"
            if dp.exists():
                depth_paths.append(str(dp))
            else:
                # Try without zero-padding
                dp_alt = Path(depth_dir) / f"depth_{fidx}.png"
                if dp_alt.exists():
                    depth_paths.append(str(dp_alt))
                else:
                    print(f"[WARN] Depth file not found: {dp} or {dp_alt}")
                    depth_paths.append("")  # Empty string as placeholder

    # --- Override global intrinsics in YOLOE.utils ---------------------------
    # Read intrinsics from the first frame
    first_fd = loader.get_frame_data(frame_indices_eval[0])
    K = first_fd.camera_intrinsic
    if K is not None:
        img_h, img_w = first_fd.rgb.shape[:2] if first_fd.rgb is not None else (530, 730)
        _override_yutils_intrinsics(K, img_h, img_w)
        print(f"Intrinsics: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
              f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  "
              f"image={img_w}x{img_h}")
    else:
        print("[WARN] No camera intrinsics found – using YOLOE defaults!")

    # --- Cache raw depth maps (no unit conversion) --------------------------
    # THUD synthetic uses non-standard depth encoding; the custom
    # point-extractor handles the scale / offset formulas internally.
    depth_cache: Dict[str, np.ndarray] = {}
    for dp in depth_paths:
        depth_cache[dp] = _load_thud_depth_raw(dp)

    # --- THUD point extractor (replaces standard depth pipeline) -------------
    thud_point_extractor = None
    if K is not None:
        thud_point_extractor = _make_thud_point_extractor(K)
        print("[INFO] Using THUD-specific depth→PCD pipeline")
    else:
        print("[WARN] No intrinsics – falling back to standard depth pipeline")

    # --- Initialise tracking pipeline ----------------------------------------
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points_per_object=int(cfg.max_accumulated_points),
        inactive_frames_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    # --- YOLO tracking stream (uses PNGs) ------------------------------------
    results_stream = _track_objects_thud(
        rgb_paths,
        depth_paths,
        model_path=cfg.yolo_model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
    )

    # --- Vis setup -----------------------------------------------------------
    vis_cfg = OmegaConf.to_container(cfg.visualization, resolve=True)
    vis_on = vis_cfg.get("enabled", False)
    vis_interval = vis_cfg.get("interval", 10)
    vis_save = vis_cfg.get("save_dir")
    if vis_save:
        os.makedirs(vis_save, exist_ok=True)

    skip_set = set(c.lower() for c in cfg.get("skip_classes", []))

    # --- Metrics accumulator -------------------------------------------------
    acc = MetricsAccumulator()

    # --- Main loop -----------------------------------------------------------
    frame_counter = 0
    for yolo_res, rgb_path, depth_path in tqdm(
        results_stream, total=n_eval_frames, desc=f"[{scene_label}]"
    ):
        # GT for this frame
        fidx = (
            frame_indices_eval[frame_counter]
            if frame_counter < n_eval_frames
            else frame_indices_eval[-1]
        )
        fd = loader.get_frame_data(fidx)
        gt_instances = _gt_objects_to_instances(fd.gt_objects)

        # Depth (pre-cached, raw uint16 values)
        depth_raw = depth_cache.get(depth_path)
        if depth_raw is None:
            frame_counter += 1
            continue

        # Camera pose from loader (JSON annotations)
        T_w_c = fd.cam_transform_4x4  # may be None

        # YOLO masks
        _, masks_clean = yutils.preprocess_mask(
            yolo_res=yolo_res,
            index=frame_counter,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            fast=True,
        )

        # Track IDs & class names
        track_ids, class_names = _extract_yolo_ids(yolo_res, masks_clean)

        # Filter skip classes
        masks_clean, track_ids, class_names = _apply_class_filter(
            masks_clean, track_ids, class_names, skip_set,
        )

        # Build 3-D objects with global tracking
        # Uses THUD-specific depth→PCD pipeline when available
        frame_objs, _ = yutils.create_3d_objects_with_tracking(
            track_ids,
            masks_clean,
            int(cfg.max_points_per_obj),
            depth_raw,
            T_w_c,
            frame_counter,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            object_registry=object_registry,
            class_names=class_names,
            point_extractor=thud_point_extractor,
        )

        # Build PredInstance list
        pred_instances = _build_pred_instances(frame_objs, track_ids, masks_clean)

        # Match GT ↔ pred
        mapping, ious = match_greedy(
            gt_instances,
            pred_instances,
            iou_threshold=float(cfg.iou_threshold),
        )

        # Feed accumulator
        rec = FrameRecord(
            frame_idx=int(fidx),
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        )
        acc.add_frame(rec)

        # --- optional visualisation ------------------------------------------
        if vis_on and (frame_counter % vis_interval == 0 or frame_counter == 0):
            _visualize_frame(
                rgb_path,
                gt_instances,
                pred_instances,
                mapping,
                ious,
                frame_counter,
                vis_cfg,
                vis_save,
            )

        if vis_cfg.get("show_3d", False) and (frame_counter % vis_interval == 0 or frame_counter == 0):
            _visualize_tracking_3d(
                frame_idx=int(fidx),
                frame_data=fd,
                frame_objs=frame_objs,
                object_registry=object_registry,
                max_bg_points=int(vis_cfg.get("max_bg_points_3d", 50000)),
                max_obj_points=int(vis_cfg.get("max_obj_points_3d", 3000)),
                include_background_cloud=bool(vis_cfg.get("show_3d_bg", False)),
            )

        frame_counter += 1

    # --- Final metrics -------------------------------------------------------
    metrics = acc.compute()
    print_summary(metrics, title=f"THUD BENCHMARK – {scene_label}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-scene benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_dataset(
    thud_root: str,
    cfg: OmegaConf,
    output_dir: Optional[str] = None,
) -> Dict:
    """Iterate over all THUD synthetic captures and aggregate metrics."""

    scene_type = cfg.get("scene_type", "static")
    scenes = discover_thud_synthetic_scenes(thud_root, scene_type=scene_type)

    if not scenes:
        print(f"No THUD synthetic scenes found under {thud_root} (type={scene_type})")
        return {}

    print(f"Found {len(scenes)} captures under {thud_root}  (type={scene_type})")

    all_results: Dict[str, Dict] = {}
    agg_keys = ["T_mIoU", "T_SR", "ID_consistency", "MOTA", "MOTP"]
    agg: Dict[str, List[float]] = defaultdict(list)

    for scene_dir in scenes:
        sd = Path(scene_dir)
        # Build a unique key from relative path
        try:
            rel = sd.relative_to(Path(thud_root) / "Synthetic_Scenes")
            key = str(rel)
        except ValueError:
            key = sd.name

        try:
            res = benchmark_scene(scene_dir, cfg)
            all_results[key] = res
            for k in agg_keys:
                if k in res:
                    agg[k].append(res[k])
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {key}: {exc}")
            traceback.print_exc()

    # Aggregate
    overall: Dict[str, Dict] = {}
    for k in agg_keys:
        vals = agg.get(k, [])
        if vals:
            overall[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    _print_aggregate(overall, agg_keys)

    out = Path(output_dir) if output_dir else Path(thud_root)
    combined = {
        "per_scene": all_results,
        "overall": overall,
        "num_scenes": len(all_results),
    }
    save_metrics(combined, out, scene_name="thud_all_scenes_aggregate")
    for name, res in all_results.items():
        save_metrics(res, out, scene_name=name.replace("/", "_"))
        scene_plot_dir = out / name.replace("/", "_") / "benchmark_plots"
        plot_results(res, output_dir=str(scene_plot_dir))

    if len(all_results) > 1:
        _plot_cross_scene(all_results, agg_keys, out)

    return combined


def _print_aggregate(overall: Dict, keys: List[str]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  THUD AGGREGATE  (all captures)")
    print(sep)
    for k in keys:
        if k in overall:
            m = overall[k]
            print(
                f"  {k:25s}  {m['mean']:.4f} ± {m['std']:.4f}  "
                f"[{m['min']:.4f} – {m['max']:.4f}]"
            )
    print(sep)


def _plot_cross_scene(
    all_results: Dict[str, Dict],
    agg_keys: List[str],
    out: Path,
) -> None:
    """Grouped bar chart comparing all captures side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_names = list(all_results.keys())
    n_scenes = len(scene_names)
    n_metrics = len(agg_keys)

    fig, ax = plt.subplots(figsize=(max(10, n_scenes * 2), 6))
    x = np.arange(n_scenes)
    width = 0.8 / n_metrics

    for i, key in enumerate(agg_keys):
        vals = [all_results[s].get(key, 0.0) for s in scene_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=key, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )
        avg = float(np.mean(vals))
        color = bars[0].get_facecolor()
        ax.axhline(avg, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            n_scenes - 0.5,
            avg + 0.02,
            f"avg {key}: {avg:.2f}",
            fontsize=7,
            color=color,
            ha="right",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scene_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("THUD Cross-Capture Metrics Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(min(0, ax.get_ylim()[0] - 0.05), 1.15)
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    plot_dir = out / "benchmark_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        plot_dir / "thud_cross_capture_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    print(f"[plots] Saved {plot_dir / 'thud_cross_capture_comparison.png'}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Real-scene helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _real_gt_to_instances(
    objects_2d: List[RealObject2D],
    masks: List[np.ndarray],
) -> List[GTInstance]:
    """Convert RealSceneDataLoader 2D objects + masks → ``GTInstance``."""
    instances: List[GTInstance] = []
    for obj, mask in zip(objects_2d, masks):
        if obj.track_id is None:
            continue
        instances.append(
            GTInstance(
                track_id=obj.track_id,
                class_name=obj.class_name,
                mask=mask,
                bbox_xyxy=tuple(obj.bbox_xyxy),
            )
        )
    return instances


def _track_objects_real(
    rgb_paths: List[str],
    depth_paths: List[str],
    class_names_to_track: Optional[List[str]] = None,
    model_path: str = "yoloe-11l-seg-pf.pt",
    conf: float = 0.3,
    iou: float = 0.5,
):
    """Generator that runs YOLO tracking on Real_Scenes PNG images."""
    from ultralytics import YOLOE as YOLOEModel

    model = YOLOEModel(model_path)
    class_names_to_track = ["door", "chair", "table", "stool", "people", "shelf", "robot"]
    model.set_classes(class_names_to_track) if class_names_to_track else None

    if class_names_to_track:
        print(f"Tracking only classes: {class_names_to_track}")
    else:
        print("Tracking all classes available in the model.")
    for ip, rgb_p in enumerate(rgb_paths):
        rgb = cv2.imread(rgb_p)
        if rgb is None:
            print(f"[WARN] Could not read image: {rgb_p}")
            continue
        rgb_input = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        out = model.track(
            source=[rgb_input],
            tracker=yutils.TRACKER_CFG,
            device=yutils.DEVICE,
            conf=conf,
            verbose=False,
            persist=True,
            agnostic_nms=True,
        )
        res = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
        dp = depth_paths[ip] if ip < len(depth_paths) else ""
        yield res, rgb_p, dp


# ═══════════════════════════════════════════════════════════════════════════════
# Real-scene single benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_real_scene(scene_path: str, cfg: OmegaConf) -> Dict:
    """Run tracking + evaluation on one THUD Real_Scenes capture directory."""

    scene_dir = Path(scene_path)
    # Build a human-readable label
    try:
        parts = scene_dir.parts
        idx = next(i for i, p in enumerate(parts) if p == "Real_Scenes")
        scene_label = "/".join(parts[idx:])
    except StopIteration:
        scene_label = scene_dir.name

    print(f"\n{'=' * 60}")
    print(f"  THUD REAL-SCENE BENCHMARK – {scene_label}")
    print(f"{'=' * 60}")

    # --- Load GT via RealSceneDataLoader ------------------------------------
    loader = RealSceneDataLoader(str(scene_dir), verbose=True)

    # Assign persistent tracking IDs across all frames
    tracking_dist = float(cfg.get("real_tracking_distance", 0.3))
    loader.assign_tracking_ids(distance_threshold=tracking_dist)
    print(f"  Unique GT tracks: {loader.get_num_tracks()}")

    frame_indices = loader.get_frame_indices()
    n_frames = len(frame_indices)
    print(f"  Frames: {n_frames}")

    # --- Override global intrinsics in YOLOE.utils --------------------------
    K = loader.get_intrinsics_matrix()
    if K is not None:
        img_h, img_w = loader.image_height, loader.image_width
        _override_yutils_intrinsics(K, img_h, img_w)
        print(f"  Intrinsics: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  "
              f"cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  "
              f"image={img_w}x{img_h}")
    else:
        print("  [WARN] No camera intrinsics – using YOLOE defaults!")

    # --- Collect RGB / depth paths in frame order ---------------------------
    rgb_paths: List[str] = []
    depth_paths: List[str] = []
    for fidx in frame_indices:
        rp = loader.get_rgb_path(fidx)
        dp = loader.get_depth_path(fidx)
        if Path(rp).exists():
            rgb_paths.append(rp)
        if Path(dp).exists():
            depth_paths.append(dp)

    if len(rgb_paths) != n_frames:
        print(f"  [WARN] RGB count ({len(rgb_paths)}) != frame count ({n_frames})")
    if len(depth_paths) != n_frames:
        print(f"  [WARN] Depth count ({len(depth_paths)}) != frame count ({n_frames})")
    
    # --- Process depth with Pi3X model if configured -------------------------
    rgb_dir = str(loader.rgb_dir)
    depth_dir = str(loader.depth_dir)
    
    temp_cfg = OmegaConf.create({
        'rgb_dir': rgb_dir,
        'depth_dir': depth_dir,
        'traj_path': str(scene_dir / "camera_poses.txt"),  # Placeholder
        'depth_model': cfg.get('depth_model', None)
    })
    temp_cfg = process_depth_model(temp_cfg)
    
    # Update depth paths if depth_dir was changed
    if temp_cfg.depth_dir != depth_dir:
        depth_dir = temp_cfg.depth_dir
        print(f"  Using processed depth from: {depth_dir}")
        # Use new depth directory paths
        depth_paths = sorted(Path(depth_dir).glob("depth_*.png"))
        depth_paths = [str(p) for p in depth_paths[:n_frames]]

    # --- Cache depth maps ---------------------------------------------------
    depth_scale = float(cfg.get("depth_scale", 1000.0))
    depth_max_m = float(cfg.get("depth_max_m", 100.0))
    depth_cache: Dict[str, np.ndarray] = {}
    for dp in depth_paths:
        depth_cache[dp] = _load_real_depth_as_meters(dp, depth_scale, depth_max_m)

    # --- Initialise tracking pipeline ---------------------------------------
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.tracking_overlap_threshold),
        distance_threshold=float(cfg.tracking_distance_threshold),
        max_points_per_object=int(cfg.max_accumulated_points),
        inactive_frames_limit=int(cfg.tracking_inactive_limit),
        volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
        reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
    )

    class_names = loader.get_class_names()
    # --- YOLO tracking stream -----------------------------------------------
    results_stream = _track_objects_real(
        rgb_paths,
        depth_paths,
        model_path=cfg.yolo_model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
    )

    # --- Vis setup ----------------------------------------------------------
    vis_cfg = OmegaConf.to_container(cfg.visualization, resolve=True)
    vis_on = vis_cfg.get("enabled", False)
    vis_interval = vis_cfg.get("interval", 10)
    vis_save = vis_cfg.get("save_dir")
    if vis_save:
        os.makedirs(vis_save, exist_ok=True)

    skip_set = set(c.lower() for c in cfg.get("skip_classes", []))

    # --- Metrics accumulator ------------------------------------------------
    acc = MetricsAccumulator()

    # --- Main loop ----------------------------------------------------------
    frame_counter = 0
    for yolo_res, rgb_path, depth_path in tqdm(
        results_stream, total=n_frames, desc=f"[{scene_label}]"
    ):
        fidx = frame_indices[frame_counter] if frame_counter < n_frames else frame_indices[-1]

        # GT: tracked 2D objects + instance masks
        gt_2d = loader.get_tracked_2d_objects(fidx)
        gt_masks = loader.get_instance_masks(fidx, objects_2d=gt_2d)
        gt_instances = _real_gt_to_instances(gt_2d, gt_masks)

        # Depth (pre-cached)
        depth_m = depth_cache.get(depth_path)
        if depth_m is None:
            frame_counter += 1
            continue

        # Camera pose
        cam_pose = loader.load_camera_pose(fidx)
        T_w_c = cam_pose.transform_matrix if cam_pose is not None else None

        # YOLO masks
        _, masks_clean = yutils.preprocess_mask(
            yolo_res=yolo_res,
            index=frame_counter,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            fast=True,
        )

        # Track IDs & class names
        track_ids, class_names = _extract_yolo_ids(yolo_res, masks_clean)

        # Filter skip classes
        masks_clean, track_ids, class_names = _apply_class_filter(
            masks_clean, track_ids, class_names, skip_set,
        )

        # Build 3-D objects with global tracking
        frame_objs, _ = yutils.create_3d_objects_with_tracking(
            track_ids,
            masks_clean,
            int(cfg.max_points_per_obj),
            depth_m,
            T_w_c,
            frame_counter,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            object_registry=object_registry,
            class_names=class_names,
        )

        # Build PredInstance list
        pred_instances = _build_pred_instances(frame_objs, track_ids, masks_clean)

        # Match GT ↔ pred
        mapping, ious = match_greedy(
            gt_instances,
            pred_instances,
            iou_threshold=float(cfg.iou_threshold),
        )

        # Feed accumulator
        rec = FrameRecord(
            frame_idx=frame_counter,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        )
        acc.add_frame(rec)

        # --- optional visualisation -----------------------------------------
        if vis_on and (frame_counter % vis_interval == 0 or frame_counter == 0):
            _visualize_frame(
                rgb_path,
                gt_instances,
                pred_instances,
                mapping,
                ious,
                frame_counter,
                vis_cfg,
                vis_save,
            )

        frame_counter += 1

    # --- Final metrics ------------------------------------------------------
    metrics = acc.compute()
    print_summary(metrics, title=f"THUD REAL-SCENE BENCHMARK – {scene_label}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Real-scene multi benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_real_dataset(
    thud_root: str,
    cfg: OmegaConf,
    output_dir: Optional[str] = None,
) -> Dict:
    """Iterate over all THUD Real_Scenes captures and aggregate metrics."""

    scenes = discover_real_scenes(thud_root)
    if not scenes:
        print(f"No THUD Real_Scenes found under {thud_root}")
        return {}

    print(f"Found {len(scenes)} Real_Scenes captures under {thud_root}")

    all_results: Dict[str, Dict] = {}
    agg_keys = ["T_mIoU", "T_SR", "ID_consistency", "MOTA", "MOTP"]
    agg: Dict[str, List[float]] = defaultdict(list)

    for scene_dir in scenes:
        sd = Path(scene_dir)
        try:
            rel = sd.relative_to(Path(thud_root) / "Real_Scenes")
            key = str(rel)
        except ValueError:
            key = sd.name

        try:
            res = benchmark_real_scene(scene_dir, cfg)
            all_results[key] = res
            for k in agg_keys:
                if k in res:
                    agg[k].append(res[k])
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {key}: {exc}")
            traceback.print_exc()

    # Aggregate
    overall: Dict[str, Dict] = {}
    for k in agg_keys:
        vals = agg.get(k, [])
        if vals:
            overall[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    _print_aggregate_real(overall, agg_keys)

    out = Path(output_dir) if output_dir else Path(thud_root)
    combined = {
        "per_scene": all_results,
        "overall": overall,
        "num_scenes": len(all_results),
    }
    save_metrics(combined, out, scene_name="thud_real_scenes_aggregate")
    for name, res in all_results.items():
        save_metrics(res, out, scene_name=name.replace("/", "_"))
        scene_plot_dir = out / name.replace("/", "_") / "benchmark_plots"
        plot_results(res, output_dir=str(scene_plot_dir))

    if len(all_results) > 1:
        _plot_cross_scene(all_results, agg_keys, out)

    return combined


def _print_aggregate_real(overall: Dict, keys: List[str]) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  THUD REAL-SCENES AGGREGATE  (all captures)")
    print(sep)
    for k in keys:
        if k in overall:
            m = overall[k]
            print(
                f"  {k:25s}  {m['mean']:.4f} ± {m['std']:.4f}  "
                f"[{m['min']:.4f} – {m['max']:.4f}]"
            )
    print(sep)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers  (mirrored from benchmark_tracking.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_yolo_ids(yolo_res, masks_clean):
    """Pull track IDs and class names from a YOLO result object."""
    track_ids = None
    class_names = None
    if hasattr(yolo_res, "boxes") and yolo_res.boxes is not None:
        if getattr(yolo_res.boxes, "id", None) is not None:
            try:
                track_ids = (
                    yolo_res.boxes.id.detach().cpu().numpy().astype(np.int64)
                )
            except Exception:
                pass
        if (
            getattr(yolo_res.boxes, "cls", None) is not None
            and hasattr(yolo_res, "names")
        ):
            try:
                cls_ids = (
                    yolo_res.boxes.cls.detach().cpu().numpy().astype(np.int64)
                )
                class_names = [yolo_res.names[int(c)] for c in cls_ids]
            except Exception:
                pass
    n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
    if track_ids is None:
        track_ids = np.arange(n, dtype=np.int64)
    return track_ids, class_names


def _apply_class_filter(masks_clean, track_ids, class_names, skip_set):
    """Remove detections whose class name is in *skip_set*."""
    if not skip_set or class_names is None:
        return masks_clean, track_ids, class_names
    keep = [
        i
        for i, c in enumerate(class_names)
        if not _should_skip_class(c, skip_set)
    ]
    if len(keep) == len(class_names):
        return masks_clean, track_ids, class_names
    masks_clean = [masks_clean[i] for i in keep] if masks_clean else []
    track_ids = track_ids[keep] if track_ids is not None else None
    class_names = [class_names[i] for i in keep]
    return masks_clean, track_ids, class_names


def _build_pred_instances(
    frame_objs, track_ids, masks_clean,
) -> List[PredInstance]:
    """Convert pipeline output dicts → ``PredInstance`` list."""
    preds: List[PredInstance] = []
    for obj in frame_objs:
        mask = None
        yolo_tid = obj.get("yolo_track_id", -1)
        if yolo_tid >= 0 and track_ids is not None:
            idxs = np.where(track_ids == yolo_tid)[0]
            if len(idxs) > 0 and masks_clean and idxs[0] < len(masks_clean):
                mask = masks_clean[idxs[0]]
        preds.append(
            PredInstance(
                pred_id=obj["global_id"],
                class_name=obj.get("class_name"),
                mask=mask,
            )
        )
    return preds


def _visualize_frame(
    rgb_path,
    gt_instances,
    pred_instances,
    mapping,
    ious,
    frame_idx,
    vis_cfg,
    vis_save,
):
    """Optionally render 2-D tracking + matching panels."""
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return

    save_match = None
    save_2d = None
    if vis_save:
        save_match = os.path.join(vis_save, f"matching_{frame_idx:06d}.png")
        save_2d = os.path.join(vis_save, f"tracking_2d_{frame_idx:06d}.png")

    show = vis_cfg.get("show_windows", True)

    if vis_cfg.get("show_matching", True):
        visualize_matching(
            rgb=rgb,
            gt_masks=[g.mask for g in gt_instances],
            gt_ids=[g.track_id for g in gt_instances],
            gt_labels=[g.class_name for g in gt_instances],
            pred_masks=[p.mask for p in pred_instances],
            pred_ids=[p.pred_id for p in pred_instances],
            pred_labels=[p.class_name or "" for p in pred_instances],
            mapping=mapping,
            ious=ious,
            frame_idx=frame_idx,
            save_path=save_match,
            show=show,
        )

    if vis_cfg.get("show_2d", True):
        labels = [
            f"G:{p.pred_id} {p.class_name or ''}" for p in pred_instances
        ]
        overlay = draw_masks_with_labels(
            rgb,
            [p.mask for p in pred_instances],
            [p.pred_id for p in pred_instances],
            labels,
            title=f"Frame {frame_idx}",
        )
        if save_2d:
            cv2.imwrite(save_2d, overlay)
        if show:
            cv2.imshow("Tracking 2D", overlay)
            cv2.waitKey(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="THUD Tracking Benchmark Runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples
--------
  # Single capture with visualisation every 5 frames
  python benchmark_tracking_thud.py \\
      /data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1 \\
      --vis --vis-interval 5

  # All static captures across all scenes
  python benchmark_tracking_thud.py /data/THUD_Robot --multi

  # All dynamic captures
  python benchmark_tracking_thud.py /data/THUD_Robot --multi --scene-type dynamic

  # Save visualisations without displaying
  python benchmark_tracking_thud.py \\
      /data/THUD_Robot/Synthetic_Scenes/Gym/static/Capture_1 \\
      --vis --vis-save ./debug_vis --no-show

  # Override depth scale
  python benchmark_tracking_thud.py \\
      /data/THUD_Robot/Synthetic_Scenes/Office/static/Capture_1 \\
      --depth-scale 1000.0

  # Single Real_Scenes capture
  python benchmark_tracking_thud.py \\
      /data/THUD_Robot/Real_Scenes/10L/static/Capture_1 --real

  # All Real_Scenes captures
  python benchmark_tracking_thud.py /data/THUD_Robot --real --multi
""",
    )
    p.add_argument(
        "path",
        help=(
            "Path to a single THUD capture dir "
            "(e.g. .../Gym/static/Capture_1) "
            "or the THUD dataset root (with --multi)."
        ),
    )
    p.add_argument(
        "--real",
        action="store_true",
        help="Use Real_Scenes loader instead of Synthetic_Scenes.",
    )
    p.add_argument(
        "--multi",
        action="store_true",
        help="Benchmark all captures under PATH/Synthetic_Scenes/ (or Real_Scenes/ with --real).",
    )
    p.add_argument(
        "--scene-type",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Which sub-directory to scan in multi mode (default: static).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for metrics JSON + plots.",
    )
    # THUD-specific
    p.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Raw depth ÷ this = metres (default: 1000).",
    )
    p.add_argument(
        "--depth-max",
        type=float,
        default=None,
        help="Clamp depth above this many metres (default: 100).",
    )
    # Visualisation flags
    p.add_argument(
        "--vis", action="store_true", help="Enable debug visualisation."
    )
    p.add_argument(
        "--vis-3d", action="store_true", help="Enable Open3D 3D tracking visualisation."
    )
    p.add_argument(
        "--vis-3d-bg",
        action="store_true",
        help="Also show background depth point cloud in 3D view (may misalign if frame conventions differ).",
    )
    p.add_argument(
        "--vis-interval",
        type=int,
        default=10,
        help="Visualise every N frames.",
    )
    p.add_argument(
        "--vis-save",
        type=str,
        default=None,
        help="Dir to save vis PNGs.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Don't pop up windows (only save).",
    )
    p.add_argument(
        "--vis-3d-max-bg",
        type=int,
        default=50000,
        help="Max background depth points in 3D view.",
    )
    p.add_argument(
        "--vis-3d-max-obj",
        type=int,
        default=3000,
        help="Max per-object points in 3D view.",
    )
    # YOLO overrides
    p.add_argument(
        "--model", type=str, default=None, help="YOLO model path."
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO confidence threshold.",
    )
    p.add_argument(
        "--iou-thresh",
        type=float,
        default=None,
        help="Matching IoU threshold.",
    )
    # Depth model
    p.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help="Depth model (e.g., 'yyfz233/Pi3X' or 'none' for original).",
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = OmegaConf.create(DEFAULT_CFG)

    # Apply CLI overrides
    if args.vis:
        cfg.visualization.enabled = True
    if args.vis_interval:
        cfg.visualization.interval = args.vis_interval
    if args.vis_save:
        cfg.visualization.save_dir = args.vis_save
    if args.no_show:
        cfg.visualization.show_windows = False
    if args.vis_3d:
        cfg.visualization.enabled = True
        cfg.visualization.show_3d = True
    if args.vis_3d_bg:
        cfg.visualization.show_3d_bg = True
    if args.vis_3d_max_bg is not None:
        cfg.visualization.max_bg_points_3d = args.vis_3d_max_bg
    if args.vis_3d_max_obj is not None:
        cfg.visualization.max_obj_points_3d = args.vis_3d_max_obj
    if args.model:
        cfg.yolo_model = args.model
    if args.conf is not None:
        cfg.conf = args.conf
    if args.iou_thresh is not None:
        cfg.iou_threshold = args.iou_thresh
    if args.depth_scale is not None:
        cfg.depth_scale = args.depth_scale
    if args.depth_max is not None:
        cfg.depth_max_m = args.depth_max
    if args.scene_type:
        cfg.scene_type = args.scene_type
    if args.depth_model is not None:
        if args.depth_model.lower() == 'none':
            cfg.depth_model = None
        else:
            cfg.depth_model = args.depth_model

    use_real = getattr(args, "real", False)

    path = Path(args.path)
    if not path.exists():
        print(f"Path does not exist: {path}")
        return 1

    if use_real:
        # ---- Real_Scenes mode ----
        if args.multi:
            results = benchmark_real_dataset(str(path), cfg, output_dir=args.output)
        else:
            results = benchmark_real_scene(str(path), cfg)
            out_dir = args.output or str(path)
            save_metrics(results, out_dir, scene_name=path.name)
            plot_results(
                results,
                output_dir=os.path.join(out_dir, "benchmark_plots"),
            )
    else:
        # ---- Synthetic_Scenes mode (original) ----
        if args.multi:
            results = benchmark_dataset(str(path), cfg, output_dir=args.output)
        else:
            results = benchmark_scene(str(path), cfg)
            out_dir = args.output or str(path)
            save_metrics(results, out_dir, scene_name=path.name)
            plot_results(
                results,
                output_dir=os.path.join(out_dir, "benchmark_plots"),
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
