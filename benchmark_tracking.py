"""
Benchmarking script for YOLO-SSG tracking capability.

Metrics Computed:
- T-mIoU: (1/T) * Σ IoU(M̂_t^o, M_t^o) - Temporal mean IoU per object
- T-SR: 1[∀t: |M̂_t^o| > 0] - Temporal Success Rate (object tracked in all frames)

Additional metrics:
- ID Switches: Number of times an object's predicted ID changes
- MOTA-like: Multi-Object Tracking Accuracy approximation
- Per-class tracking performance
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] Open3D not available. 3D visualization disabled.")

# Import YOLO-SSG components
import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry

# Import Isaac Sim data loader
from isaac_sim_loader import IsaacSimDataLoader, GTObject


# ============================================================================
# DEBUG VISUALIZATION FUNCTIONS
# ============================================================================

def generate_color_from_id(obj_id: int) -> Tuple[int, int, int]:
    """Generate deterministic BGR color from object ID."""
    if obj_id < 0:
        return (128, 128, 128)  # Gray for invalid
    random.seed(int(obj_id) * 7 + 13)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def visualize_yolo_tracking_2d(
    rgb_image: np.ndarray,
    yolo_res,
    masks_clean: List[np.ndarray],
    track_ids: np.ndarray,
    class_names: List[str],
    frame_idx: int,
    global_ids: List[int] = None,
    match_sources: List[str] = None,
    alpha: float = 0.5,
    show: bool = True,
    save_path: str = None
) -> np.ndarray:
    """
    Visualize YOLO tracking results with masks, class names, and IDs.
    
    Shows:
    - RGB image with mask overlays
    - Bounding boxes with labels: "class_name | YOLO:X | Global:Y"
    - Match source indicator (yolo_map, prev_frame, global_match, new)
    
    Args:
        rgb_image: RGB image (H, W, 3)
        yolo_res: YOLO result object
        masks_clean: List of preprocessed binary masks
        track_ids: YOLO track IDs
        class_names: List of class names
        frame_idx: Current frame index
        global_ids: List of global tracking IDs (from registry)
        match_sources: List of match source strings
        alpha: Mask overlay transparency
        show: Whether to display with matplotlib
        save_path: Optional path to save visualization
    
    Returns:
        Annotated image (H, W, 3) BGR
    """
    # Ensure RGB format and copy
    if rgb_image.shape[2] == 3:
        vis_img = rgb_image.copy()
    else:
        vis_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
    
    H, W = vis_img.shape[:2]
    
    # Create mask overlay
    overlay = vis_img.copy()
    
    n_objects = len(masks_clean) if masks_clean else 0
    
    for i in range(n_objects):
        if masks_clean[i] is None:
            continue
        
        # Get IDs
        yolo_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1
        global_id = int(global_ids[i]) if global_ids is not None and i < len(global_ids) else -1
        class_name = class_names[i] if class_names is not None and i < len(class_names) else "unknown"
        match_src = match_sources[i] if match_sources is not None and i < len(match_sources) else ""
        
        # Generate color based on global_id (for consistency)
        color_id = global_id if global_id >= 0 else yolo_id
        color = generate_color_from_id(color_id)
        
        # Apply mask overlay
        mask = masks_clean[i]
        if mask.dtype != bool:
            mask = mask > 0
        overlay[mask] = (
            overlay[mask] * (1 - alpha) + 
            np.array(color[::-1]) * alpha  # BGR to RGB
        ).astype(np.uint8)
        
        # Get bounding box from mask
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color[::-1], 2)
            
            # Build label
            label_parts = [class_name]
            label_parts.append(f"Y:{yolo_id}")
            if global_id >= 0:
                label_parts.append(f"G:{global_id}")
            if match_src:
                # Shorten match source
                src_short = match_src[:3] if len(match_src) > 3 else match_src
                label_parts.append(f"[{src_short}]")
            label = " | ".join(label_parts)
            
            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color[::-1], -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add frame info
    info_text = f"Frame {frame_idx} | Objects: {n_objects}"
    cv2.putText(overlay, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    if show:
        plt.figure(figsize=(14, 8))
        plt.imshow(overlay)
        plt.title(f"YOLO Tracking - Frame {frame_idx}")
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    if save_path and not show:
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    return overlay


def visualize_tracking_3d(
    object_registry,
    frame_objs: List[Dict],
    frame_idx: int,
    show_all_tracked: bool = True,
    point_size: float = 2.0,
    width: int = 1280,
    height: int = 720,
    non_blocking: bool = False
):
    """
    Visualize 3D tracking state using Open3D.
    
    Color coding:
    - BLUE bbox: Currently visible objects (detected this frame)
    - RED bbox: Not visible / last seen / reprojection-based objects
    - Points colored by global_id for consistency
    
    Args:
        object_registry: GlobalObjectRegistry instance
        frame_objs: List of objects detected/visible in current frame
        frame_idx: Current frame index
        show_all_tracked: If True, show all tracked objects; if False, only current frame
        point_size: Size of points in visualization
        width, height: Window dimensions
        non_blocking: If True, don't block execution (update existing window)
    """
    if not HAS_OPEN3D:
        print("[WARN] Open3D not available. Skipping 3D visualization.")
        return
    
    # Get all objects from registry
    all_objects = object_registry.get_all_objects()
    
    # Get currently visible IDs
    visible_ids = set()
    for obj in frame_objs:
        gid = obj.get('global_id', -1)
        if gid >= 0:
            visible_ids.add(gid)
    
    # Compute global bounds
    gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    
    for gid, obj_data in all_objects.items():
        pts = obj_data.get('points_accumulated')
        if pts is not None and len(pts) > 0:
            gmin = np.minimum(gmin, pts.min(axis=0))
            gmax = np.maximum(gmax, pts.max(axis=0))
    
    if not np.all(np.isfinite(gmin)):
        print("[WARN] No valid points for 3D visualization.")
        return
    
    diag = float(np.linalg.norm(gmax - gmin)) if np.all(np.isfinite(gmax - gmin)) else 1.0
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    summary = object_registry.get_tracking_summary()
    title = f"3D Tracking - Frame {frame_idx} | Visible: {len(visible_ids)}/{summary['total_objects']}"
    vis.create_window(window_name=title, width=width, height=height, visible=True)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.1])  # Dark blue background
    opt.point_size = float(point_size)
    
    # Add coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1 * diag, origin=[0, 0, 0])
    vis.add_geometry(coord)
    
    # Add all objects
    for gid, obj_data in all_objects.items():
        pts = obj_data.get('points_accumulated')
        bbox_3d = obj_data.get('bbox_3d')
        class_name = obj_data.get('class_name', '')
        
        if pts is None or len(pts) == 0:
            continue
        
        is_visible = gid in visible_ids
        
        # Generate point color based on global_id
        random.seed(int(gid) * 7 + 13)
        point_color = np.array([random.random(), random.random(), random.random()])
        
        # Reduce brightness for non-visible objects
        if not is_visible:
            point_color = point_color * 0.4
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(point_color, (len(pts), 1)))
        vis.add_geometry(pcd)
        
        # Add bounding box
        if bbox_3d and bbox_3d.get('aabb'):
            aabb_data = bbox_3d['aabb']
            mn = np.asarray(aabb_data.get('min'), dtype=np.float64)
            mx = np.asarray(aabb_data.get('max'), dtype=np.float64)
            
            if mn is not None and mx is not None:
                # Create AABB lineset
                corners = np.array([
                    [mn[0], mn[1], mn[2]],
                    [mx[0], mn[1], mn[2]],
                    [mx[0], mx[1], mn[2]],
                    [mn[0], mx[1], mn[2]],
                    [mn[0], mn[1], mx[2]],
                    [mx[0], mn[1], mx[2]],
                    [mx[0], mx[1], mx[2]],
                    [mn[0], mx[1], mx[2]],
                ], dtype=np.float64)
                lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
                
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners)
                ls.lines = o3d.utility.Vector2iVector(lines)
                
                # BLUE for visible, RED for not visible
                if is_visible:
                    bbox_color = [0.0, 0.5, 1.0]  # Blue
                else:
                    bbox_color = [1.0, 0.2, 0.2]  # Red
                
                ls.colors = o3d.utility.Vector3dVector([bbox_color] * len(lines))
                vis.add_geometry(ls)
    
    # Fit view
    gbox = o3d.geometry.AxisAlignedBoundingBox(gmin, gmax)
    vis.add_geometry(gbox)
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    ctr.set_lookat(gbox.get_center())
    ctr.set_front([-0.5, -0.5, -0.8])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.8)
    vis.remove_geometry(gbox, reset_bounding_box=False)
    
    # Legend info
    print(f"\n[3D Visualization] Frame {frame_idx}")
    print(f"  BLUE bbox  = Currently visible ({len(visible_ids)} objects)")
    print(f"  RED bbox   = Not visible / last seen ({summary['total_objects'] - len(visible_ids)} objects)")
    print(f"  Controls: mouse rotate, wheel zoom, right-button pan, Q quit\n")
    
    vis.run()
    vis.destroy_window()


def visualize_tracking_comparison(
    rgb_image: np.ndarray,
    gt_objects: List,
    pred_objects: List,
    gt_to_pred: Dict[int, int],
    frame_idx: int,
    alpha: float = 0.4,
    show: bool = True,
    save_path: str = None
) -> np.ndarray:
    """
    Visualize GT vs Predicted masks side by side for debugging matching.
    
    Args:
        rgb_image: RGB image
        gt_objects: List of GTObject
        pred_objects: List of PredObject  
        gt_to_pred: Mapping from GT track_id to pred global_id
        frame_idx: Current frame index
        alpha: Mask overlay transparency
        show: Whether to display
        save_path: Optional save path
    """
    H, W = rgb_image.shape[:2]
    
    # Create two panels: GT and Predictions
    gt_panel = rgb_image.copy()
    pred_panel = rgb_image.copy()
    
    # Draw GT masks (green tint)
    for gt_obj in gt_objects:
        if gt_obj.mask is None:
            continue
        mask = gt_obj.mask > 0 if gt_obj.mask.dtype != bool else gt_obj.mask
        color = generate_color_from_id(gt_obj.track_id)
        gt_panel[mask] = (gt_panel[mask] * (1 - alpha) + np.array(color[::-1]) * alpha).astype(np.uint8)
        
        # Add label
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            matched = "✓" if gt_obj.track_id in gt_to_pred else "✗"
            label = f"GT:{gt_obj.track_id} {gt_obj.class_name} {matched}"
            cv2.putText(gt_panel, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw Pred masks
    matched_pred_ids = set(gt_to_pred.values())
    for pred_obj in pred_objects:
        if pred_obj.mask is None:
            continue
        mask = pred_obj.mask > 0 if pred_obj.mask.dtype != bool else pred_obj.mask
        color = generate_color_from_id(pred_obj.global_id)
        pred_panel[mask] = (pred_panel[mask] * (1 - alpha) + np.array(color[::-1]) * alpha).astype(np.uint8)
        
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            matched = "✓" if pred_obj.global_id in matched_pred_ids else "FP"
            label = f"G:{pred_obj.global_id} {pred_obj.class_name or ''} {matched}"
            cv2.putText(pred_panel, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # Combine panels
    combined = np.hstack([gt_panel, pred_panel])
    
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].imshow(gt_panel)
        axes[0].set_title(f"Ground Truth - Frame {frame_idx}\n({len(gt_objects)} objects)")
        axes[0].axis('off')
        axes[1].imshow(pred_panel)
        axes[1].set_title(f"Predictions - Frame {frame_idx}\n({len(pred_objects)} objects, {len(gt_to_pred)} matched)")
        axes[1].axis('off')
        plt.suptitle(f"GT vs Predictions Comparison - Frame {frame_idx}", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    return combined


@dataclass
class PredObject:
    """Predicted object for a single frame."""
    global_id: int
    yolo_track_id: int
    class_name: Optional[str] = None
    mask: Optional[np.ndarray] = None  # Binary mask (H, W)
    bbox_3d: Optional[Dict] = None
    match_source: str = ""


@dataclass
class TrackingMetrics:
    """Metrics for tracking evaluation."""
    # Per-object metrics
    per_object_iou: Dict[int, List[float]] = field(default_factory=dict)  # gt_track_id -> [IoU per frame]
    per_object_tracked: Dict[int, List[bool]] = field(default_factory=dict)  # gt_track_id -> [tracked per frame]
    per_object_pred_ids: Dict[int, List[int]] = field(default_factory=dict)  # gt_track_id -> [pred_id per frame]
    per_object_class: Dict[int, str] = field(default_factory=dict)  # gt_track_id -> class_name
    
    # Per-class metrics
    per_class_iou: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    per_class_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Global metrics
    total_gt_objects: int = 0
    total_pred_objects: int = 0
    total_matches: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    
    # Frame-level
    frames_processed: int = 0


# Use IsaacSimDataLoader from separate module for GT data loading
# The loader provides: get_frame_count(), get_poses(), get_gt_objects(), load_depth()
# GT objects have: track_id, class_name, mask, bbox_2d, bbox_3d_aabb, visibility


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    if mask1 is None or mask2 is None:
        return 0.0
    
    # Ensure same shape
    if mask1.shape != mask2.shape:
        # Resize mask2 to match mask1
        mask2 = cv2.resize(mask2.astype(np.uint8), (mask1.shape[1], mask1.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    # Ensure binary
    m1 = (mask1 > 0).astype(bool)
    m2 = (mask2 > 0).astype(bool)
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def compute_bbox_iou_2d(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute 2D IoU between two bounding boxes [x1, y1, x2, y2]."""
    if bbox1 is None or bbox2 is None:
        return 0.0
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def match_predictions_to_gt(gt_objects: List[GTObject], 
                            pred_objects: List[PredObject],
                            iou_threshold: float = 0.3) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Match predicted objects to ground truth using greedy matching.
    
    Returns:
        mapping: Dict[gt_track_id, pred_global_id]
        ious: Dict[gt_track_id, iou_score]
    """
    if not gt_objects or not pred_objects:
        return {}, {}
    
    n_gt = len(gt_objects)
    n_pred = len(pred_objects)
    
    # Build IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    
    for i, gt_obj in enumerate(gt_objects):
        for j, pred_obj in enumerate(pred_objects):
            iou = compute_mask_iou(gt_obj.mask, pred_obj.mask)
            iou_matrix[i, j] = iou
    
    # Greedy matching (sort by IoU descending)
    mapping = {}
    ious = {}
    used_pred = set()
    used_gt = set()
    
    # Get all (gt_idx, pred_idx, iou) tuples sorted by IoU
    matches = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j, iou_matrix[i, j]))
    
    # Sort by IoU descending
    matches.sort(key=lambda x: -x[2])
    
    for gt_idx, pred_idx, iou in matches:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        
        gt_obj = gt_objects[gt_idx]
        pred_obj = pred_objects[pred_idx]
        
        mapping[gt_obj.track_id] = pred_obj.global_id
        ious[gt_obj.track_id] = iou
        
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
    
    return mapping, ious


class TrackingBenchmark:
    """Main benchmarking class for YOLO-SSG tracking."""
    
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.metrics = TrackingMetrics()
        
        # Track GT object appearances across frames
        self.gt_appearances = defaultdict(list)  # gt_track_id -> [frame_indices]
        self.gt_to_pred_history = defaultdict(list)  # gt_track_id -> [(frame_idx, pred_id, iou)]
    
    def run_benchmark(self, scene_path: str) -> Dict:
        """
        Run tracking benchmark on a scene.
        
        Args:
            scene_path: Path to scene directory
            
        Returns:
            metrics_dict: Dictionary with all computed metrics
        """
        print(f"\n{'='*60}")
        print(f"TRACKING BENCHMARK")
        print(f"Scene: {scene_path}")
        print(f"{'='*60}\n")
        
        # Load GT dataset using IsaacSimDataLoader
        gt_loader = IsaacSimDataLoader(scene_path)
        gt_loader.print_info()
        
        n_frames = gt_loader.get_frame_count()
        poses = gt_loader.get_poses()
        
        print(f"Total frames: {n_frames}")
        print(f"Poses loaded: {len(poses)}")
        
        # Reset metrics
        self.metrics = TrackingMetrics()
        self.gt_appearances = defaultdict(list)
        self.gt_to_pred_history = defaultdict(list)
        
        # Initialize object registry
        object_registry = GlobalObjectRegistry(
            overlap_threshold=float(self.cfg.get('tracking_overlap_threshold', 0.3)),
            distance_threshold=float(self.cfg.get('tracking_distance_threshold', 0.5)),
            max_points_per_object=int(self.cfg.get('max_accumulated_points', 10000)),
            inactive_frames_limit=int(self.cfg.get('tracking_inactive_limit', 0)),
            volume_ratio_threshold=float(self.cfg.get('tracking_volume_ratio_threshold', 0.1)),
            reprojection_visibility_threshold=float(self.cfg.get('reprojection_visibility_threshold', 0.2))
        )
        
        # Prepare paths
        rgb_dir = str(gt_loader.rgb_dir)
        depth_paths = yutils.list_png_paths(str(gt_loader.depth_dir))
        
        # Cache depths
        print("Caching depth images...")
        depth_cache = {}
        for idx, dp in enumerate(depth_paths):
            depth_cache[dp] = gt_loader.load_depth(idx)
        
        # Initialize YOLO tracker
        print("Initializing YOLO tracker...")
        results_stream = yutils.track_objects_in_video_stream(
            rgb_dir,
            depth_paths,
            model_path=self.cfg.yolo_model,
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
        )
        
        # Process frames
        print("Processing frames...")
        frame_idx = 0
        for yl_res, rgb_cur_path, depth_cur_path in tqdm(results_stream, total=n_frames, desc="Benchmarking"):
            # Load GT for this frame (using IsaacSimDataLoader)
            gt_objects = gt_loader.get_gt_objects(frame_idx)
            
            # Load depth
            depth_m = depth_cache.get(depth_cur_path)
            if depth_m is None:
                frame_idx += 1
                continue
            
            # Get pose
            T_w_c = poses[min(frame_idx, len(poses)-1)] if poses else None
            
            # Preprocess masks
            _, masks_clean = yutils.preprocess_mask(
                yolo_res=yl_res,
                index=frame_idx,
                KERNEL_SIZE=int(self.cfg.kernel_size),
                alpha=float(self.cfg.alpha),
                fast=True,
            )
            
            # Get track IDs and class names from YOLO
            track_ids = None
            class_names = None
            if hasattr(yl_res, 'boxes') and yl_res.boxes is not None:
                if getattr(yl_res.boxes, 'id', None) is not None:
                    try:
                        track_ids = yl_res.boxes.id.detach().cpu().numpy().astype(np.int64)
                    except Exception:
                        pass
                
                if getattr(yl_res.boxes, 'cls', None) is not None and hasattr(yl_res, 'names'):
                    try:
                        cls_ids = yl_res.boxes.cls.detach().cpu().numpy().astype(np.int64)
                        class_names = [yl_res.names[int(c)] for c in cls_ids]
                    except Exception:
                        pass
            
            if track_ids is None:
                n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
                track_ids = np.arange(n, dtype=np.int64)
            
            # === FILTER OUT UNWANTED CLASSES ===
            skip_classes = set(c.lower() for c in self.cfg.get('skip_classes', []))
            if skip_classes and class_names is not None:
                # Build keep mask
                keep_indices = []
                for i, cls_name in enumerate(class_names):
                    if cls_name is None:
                        keep_indices.append(i)  # Keep if no class name
                    elif cls_name.lower() not in skip_classes:
                        keep_indices.append(i)
                    else:
                        # Debug: optionally log filtered detections
                        if self.cfg.get('debug_skip_classes', False):
                            print(f"  [Frame {frame_idx}] Skipping class: {cls_name}")
                
                # Apply filter
                if len(keep_indices) < len(class_names):
                    filtered_count = len(class_names) - len(keep_indices)
                    if self.cfg.get('debug_skip_classes', False):
                        print(f"  [Frame {frame_idx}] Filtered {filtered_count} detections")
                    
                    # Filter all arrays
                    masks_clean = [masks_clean[i] for i in keep_indices] if masks_clean else []
                    track_ids = track_ids[keep_indices] if track_ids is not None else None
                    class_names = [class_names[i] for i in keep_indices]
            
            # Build 3D objects with tracking
            frame_objs, _ = yutils.create_3d_objects_with_tracking(
                track_ids,
                masks_clean,
                int(self.cfg.max_points_per_obj),
                depth_m,
                T_w_c,
                frame_idx,
                o3_nb_neighbors=self.cfg.o3_nb_neighbors,
                o3std_ratio=self.cfg.o3std_ratio,
                object_registry=object_registry,
                class_names=class_names
            )
            
            # Build predicted objects with masks
            pred_objects = []
            for idx, obj in enumerate(frame_objs):
                # Get mask for this prediction
                yolo_idx = None
                if obj['yolo_track_id'] >= 0 and track_ids is not None:
                    matches = np.where(track_ids == obj['yolo_track_id'])[0]
                    if len(matches) > 0:
                        yolo_idx = matches[0]
                
                mask = None
                if yolo_idx is not None and masks_clean is not None and yolo_idx < len(masks_clean):
                    mask = masks_clean[yolo_idx]
                
                pred_obj = PredObject(
                    global_id=obj['global_id'],
                    yolo_track_id=obj['yolo_track_id'],
                    class_name=obj.get('class_name'),
                    mask=mask,
                    bbox_3d=obj.get('bbox_3d'),
                    match_source=obj.get('match_source', '')
                )
                pred_objects.append(pred_obj)
            
            # Match predictions to GT
            gt_to_pred, gt_ious = match_predictions_to_gt(gt_objects, pred_objects, iou_threshold=0.3)
            
            # Update metrics
            self._update_frame_metrics(frame_idx, gt_objects, pred_objects, gt_to_pred, gt_ious)
            
            # === DEBUG VISUALIZATION ===
            vis_cfg = self.cfg.get('visualization', {})
            vis_enabled = vis_cfg.get('enabled', False)
            vis_interval = vis_cfg.get('interval', 10)  # Visualize every N frames
            vis_save_dir = vis_cfg.get('save_dir', None)
            
            if vis_enabled and (frame_idx % vis_interval == 0 or frame_idx == 0):
                # Load RGB for visualization
                rgb_image = cv2.imread(rgb_cur_path)
                if rgb_image is not None:
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    
                    # Prepare save paths
                    save_2d_path = None
                    save_3d_flag = False
                    save_cmp_path = None
                    if vis_save_dir:
                        save_dir = Path(vis_save_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_2d_path = str(save_dir / f"tracking_2d_frame{frame_idx:06d}.png")
                        save_cmp_path = str(save_dir / f"gt_vs_pred_frame{frame_idx:06d}.png")
                    
                    # 2D YOLO Tracking Visualization
                    if vis_cfg.get('show_2d', True):
                        global_ids = [obj['global_id'] for obj in frame_objs]
                        match_sources = [obj.get('match_source', '') for obj in frame_objs]
                        
                        visualize_yolo_tracking_2d(
                            rgb_image=rgb_image,
                            yolo_res=yl_res,
                            masks_clean=masks_clean,
                            track_ids=track_ids,
                            class_names=class_names,
                            frame_idx=frame_idx,
                            global_ids=global_ids,
                            match_sources=match_sources,
                            alpha=0.5,
                            show=vis_cfg.get('show_windows', True),
                            save_path=save_2d_path
                        )
                    
                    # 3D Open3D Tracking Visualization
                    if vis_cfg.get('show_3d', True) and HAS_OPEN3D:
                        visualize_tracking_3d(
                            object_registry=object_registry,
                            frame_objs=frame_objs,
                            frame_idx=frame_idx,
                            show_all_tracked=True,
                            point_size=vis_cfg.get('point_size', 2.0)
                        )
                    
                    # GT vs Predictions Comparison
                    if vis_cfg.get('show_comparison', False):
                        visualize_tracking_comparison(
                            rgb_image=rgb_image,
                            gt_objects=gt_objects,
                            pred_objects=pred_objects,
                            gt_to_pred=gt_to_pred,
                            frame_idx=frame_idx,
                            show=vis_cfg.get('show_windows', True),
                            save_path=save_cmp_path
                        )
            
            frame_idx += 1
        
        # Compute final metrics
        metrics_dict = self._compute_final_metrics()
        
        # Print results
        self._print_results(metrics_dict)
        
        return metrics_dict
    
    def _update_frame_metrics(self, frame_idx: int, 
                               gt_objects: List[GTObject],
                               pred_objects: List[PredObject],
                               gt_to_pred: Dict[int, int],
                               gt_ious: Dict[int, float]):
        """Update metrics for a single frame."""
        
        self.metrics.frames_processed += 1
        self.metrics.total_gt_objects += len(gt_objects)
        self.metrics.total_pred_objects += len(pred_objects)
        self.metrics.total_matches += len(gt_to_pred)
        
        # Track which predictions were matched
        matched_pred_ids = set(gt_to_pred.values())
        
        # False positives: predictions without GT match
        self.metrics.total_false_positives += len(pred_objects) - len(matched_pred_ids)
        
        # Process GT objects
        for gt_obj in gt_objects:
            gt_id = gt_obj.track_id
            
            # Record appearance
            self.gt_appearances[gt_id].append(frame_idx)
            
            # Store class name
            if gt_id not in self.metrics.per_object_class:
                self.metrics.per_object_class[gt_id] = gt_obj.class_name
            
            # Initialize tracking for this GT object
            if gt_id not in self.metrics.per_object_iou:
                self.metrics.per_object_iou[gt_id] = []
                self.metrics.per_object_tracked[gt_id] = []
                self.metrics.per_object_pred_ids[gt_id] = []
            
            # Check if matched
            if gt_id in gt_to_pred:
                pred_id = gt_to_pred[gt_id]
                iou = gt_ious.get(gt_id, 0.0)
                
                self.metrics.per_object_iou[gt_id].append(iou)
                self.metrics.per_object_tracked[gt_id].append(True)
                self.metrics.per_object_pred_ids[gt_id].append(pred_id)
                
                # Track history
                self.gt_to_pred_history[gt_id].append((frame_idx, pred_id, iou))
                
                # Per-class metrics
                cls_name = gt_obj.class_name
                self.metrics.per_class_iou[cls_name].append(iou)
                self.metrics.per_class_count[cls_name] += 1
            else:
                # False negative
                self.metrics.per_object_iou[gt_id].append(0.0)
                self.metrics.per_object_tracked[gt_id].append(False)
                self.metrics.per_object_pred_ids[gt_id].append(-1)
                self.metrics.total_false_negatives += 1
    
    def _compute_final_metrics(self) -> Dict:
        """Compute final aggregated metrics."""
        results = {
            'frames_processed': self.metrics.frames_processed,
            'total_gt_instances': self.metrics.total_gt_objects,
            'total_pred_instances': self.metrics.total_pred_objects,
            'total_matches': self.metrics.total_matches,
            'total_false_positives': self.metrics.total_false_positives,
            'total_false_negatives': self.metrics.total_false_negatives,
            'unique_gt_objects': len(self.metrics.per_object_iou),
        }
        
        # ========== T-mIoU: Temporal mean IoU per object, then average across objects ==========
        t_miou_per_object = {}
        for gt_id, ious in self.metrics.per_object_iou.items():
            if len(ious) > 0:
                t_miou_per_object[gt_id] = np.mean(ious)
        
        if t_miou_per_object:
            results['T_mIoU'] = np.mean(list(t_miou_per_object.values()))
            results['T_mIoU_std'] = np.std(list(t_miou_per_object.values()))
            results['T_mIoU_per_object'] = t_miou_per_object
        else:
            results['T_mIoU'] = 0.0
            results['T_mIoU_std'] = 0.0
            results['T_mIoU_per_object'] = {}
        
        # ========== T-SR: Temporal Success Rate (object tracked in ALL frames it appears) ==========
        t_sr_per_object = {}
        for gt_id, tracked_list in self.metrics.per_object_tracked.items():
            if len(tracked_list) > 0:
                # T-SR = 1 if tracked in ALL frames, else 0
                t_sr_per_object[gt_id] = 1.0 if all(tracked_list) else 0.0
        
        if t_sr_per_object:
            results['T_SR'] = np.mean(list(t_sr_per_object.values()))
            results['T_SR_per_object'] = t_sr_per_object
        else:
            results['T_SR'] = 0.0
            results['T_SR_per_object'] = {}
        
        # ========== ID Switches: Count how many times predicted ID changes for a GT object ==========
        id_switches = {}
        total_id_switches = 0
        for gt_id, pred_ids in self.metrics.per_object_pred_ids.items():
            # Filter out -1 (not tracked)
            valid_ids = [pid for pid in pred_ids if pid >= 0]
            if len(valid_ids) < 2:
                id_switches[gt_id] = 0
                continue
            
            switches = sum(1 for i in range(1, len(valid_ids)) if valid_ids[i] != valid_ids[i-1])
            id_switches[gt_id] = switches
            total_id_switches += switches
        
        results['ID_switches_total'] = total_id_switches
        results['ID_switches_per_object'] = id_switches
        
        # ========== ID Consistency: Ratio of frames where GT object kept same predicted ID ==========
        id_consistency = {}
        for gt_id, pred_ids in self.metrics.per_object_pred_ids.items():
            valid_ids = [pid for pid in pred_ids if pid >= 0]
            if len(valid_ids) == 0:
                id_consistency[gt_id] = 0.0
                continue
            
            # Most common ID
            most_common_id = Counter(valid_ids).most_common(1)[0][0]
            consistency = sum(1 for pid in valid_ids if pid == most_common_id) / len(valid_ids)
            id_consistency[gt_id] = consistency
        
        if id_consistency:
            results['ID_consistency'] = np.mean(list(id_consistency.values()))
            results['ID_consistency_per_object'] = id_consistency
        else:
            results['ID_consistency'] = 0.0
            results['ID_consistency_per_object'] = {}
        
        # ========== Detection rate: Fraction of GT appearances that were detected ==========
        detection_rates = {}
        for gt_id, tracked_list in self.metrics.per_object_tracked.items():
            if len(tracked_list) > 0:
                detection_rates[gt_id] = sum(tracked_list) / len(tracked_list)
        
        if detection_rates:
            results['detection_rate'] = np.mean(list(detection_rates.values()))
            results['detection_rate_per_object'] = detection_rates
        else:
            results['detection_rate'] = 0.0
            results['detection_rate_per_object'] = {}
        
        # ========== Per-class T-mIoU ==========
        per_class_t_miou = {}
        for cls_name, ious in self.metrics.per_class_iou.items():
            if len(ious) > 0:
                per_class_t_miou[cls_name] = {
                    'T_mIoU': float(np.mean(ious)),
                    'T_mIoU_std': float(np.std(ious)),
                    'count': self.metrics.per_class_count[cls_name]
                }
        results['per_class_metrics'] = per_class_t_miou
        
        # ========== MOTA-like metric ==========
        # MOTA = 1 - (FN + FP + ID_switches) / total_gt
        if self.metrics.total_gt_objects > 0:
            mota = 1.0 - (self.metrics.total_false_negatives + 
                          self.metrics.total_false_positives + 
                          total_id_switches) / self.metrics.total_gt_objects
            results['MOTA'] = max(-1.0, mota)  # MOTA can be negative
        else:
            results['MOTA'] = 0.0
        
        # ========== MOTP: Mean IoU over matched pairs ==========
        if self.metrics.total_matches > 0:
            all_ious = []
            for ious in self.metrics.per_object_iou.values():
                all_ious.extend([iou for iou in ious if iou > 0])
            results['MOTP'] = float(np.mean(all_ious)) if all_ious else 0.0
        else:
            results['MOTP'] = 0.0
        
        return results
    
    def _print_results(self, metrics: Dict):
        """Print benchmark results."""
        print("\n" + "="*60)
        print("TRACKING BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\n{'='*40}")
        print("MAIN METRICS")
        print(f"{'='*40}")
        print(f"  T-mIoU (Temporal Mean IoU):     {metrics['T_mIoU']:.4f} ± {metrics.get('T_mIoU_std', 0):.4f}")
        print(f"  T-SR (Temporal Success Rate):  {metrics['T_SR']:.4f}")
        print(f"  ID Consistency:                {metrics['ID_consistency']:.4f}")
        print(f"  Detection Rate:                {metrics['detection_rate']:.4f}")
        print(f"  MOTA (Multi-Object Tracking):  {metrics['MOTA']:.4f}")
        print(f"  MOTP (Mean IoU of matches):    {metrics['MOTP']:.4f}")
        
        print(f"\n{'='*40}")
        print("COUNTING STATISTICS")
        print(f"{'='*40}")
        print(f"  Frames processed:       {metrics['frames_processed']}")
        print(f"  Unique GT objects:      {metrics['unique_gt_objects']}")
        print(f"  Total GT instances:     {metrics['total_gt_instances']}")
        print(f"  Total predictions:      {metrics['total_pred_instances']}")
        print(f"  Total matches:          {metrics['total_matches']}")
        print(f"  False positives:        {metrics['total_false_positives']}")
        print(f"  False negatives:        {metrics['total_false_negatives']}")
        print(f"  ID switches:            {metrics['ID_switches_total']}")
        
        if metrics.get('per_class_metrics'):
            print(f"\n{'='*40}")
            print("PER-CLASS T-mIoU")
            print(f"{'='*40}")
            for cls_name, cls_metrics in sorted(metrics['per_class_metrics'].items(), 
                                                key=lambda x: -x[1]['T_mIoU']):
                print(f"  {cls_name:20s}: {cls_metrics['T_mIoU']:.4f} ± {cls_metrics['T_mIoU_std']:.4f} (n={cls_metrics['count']})")
        
        print("\n" + "="*60)


def run_multi_scene_benchmark(scenes_root: str, cfg: OmegaConf) -> Dict:
    """
    Run benchmark on multiple scenes and aggregate results.
    
    Args:
        scenes_root: Path to directory containing scene folders
        cfg: Configuration
        
    Returns:
        aggregated_metrics: Dict with per-scene and overall metrics
    """
    scenes_root = Path(scenes_root)
    scene_dirs = sorted([d for d in scenes_root.iterdir() if d.is_dir()])
    
    print(f"\nFound {len(scene_dirs)} potential scenes to benchmark")
    
    all_results = {}
    aggregated = {
        'T_mIoU': [],
        'T_SR': [],
        'ID_consistency': [],
        'detection_rate': [],
        'MOTA': [],
        'MOTP': [],
    }
    
    for scene_dir in scene_dirs:
        # Check if scene has required folders
        if not (scene_dir / "rgb").exists():
            print(f"Skipping {scene_dir.name}: missing rgb folder")
            continue
        if not (scene_dir / "depth").exists():
            print(f"Skipping {scene_dir.name}: missing depth folder")
            continue
        
        print(f"\n{'#'*60}")
        print(f"Benchmarking scene: {scene_dir.name}")
        print(f"{'#'*60}")
        
        benchmark = TrackingBenchmark(cfg)
        
        try:
            results = benchmark.run_benchmark(str(scene_dir))
            all_results[scene_dir.name] = results
            
            # Aggregate
            for key in aggregated.keys():
                if key in results and results[key] is not None:
                    aggregated[key].append(results[key])
        except Exception as e:
            print(f"Error processing {scene_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Compute overall metrics
    print("\n" + "="*60)
    print("OVERALL BENCHMARK RESULTS (Across All Scenes)")
    print("="*60)
    
    overall = {}
    for key, values in aggregated.items():
        if values:
            overall[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
            print(f"  {key:20s}: {overall[key]['mean']:.4f} ± {overall[key]['std']:.4f} "
                  f"[{overall[key]['min']:.4f} - {overall[key]['max']:.4f}]")
    
    return {
        'per_scene': all_results,
        'overall': overall,
        'num_scenes': len(all_results),
    }


def save_results(results: Dict, output_path: str):
    """Save benchmark results to JSON file."""
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def plot_tracking_results(results: Dict, output_dir: str = None):
    """Generate plots for tracking benchmark results."""
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Per-object T-mIoU distribution
    if 'T_mIoU_per_object' in results and results['T_mIoU_per_object']:
        fig, ax = plt.subplots(figsize=(10, 6))
        values = list(results['T_mIoU_per_object'].values())
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(results['T_mIoU'], color='r', linestyle='--', label=f"Mean: {results['T_mIoU']:.3f}")
        ax.set_xlabel('T-mIoU')
        ax.set_ylabel('Number of Objects')
        ax.set_title('Per-Object Temporal Mean IoU Distribution')
        ax.legend()
        
        if output_dir:
            plt.savefig(output_dir / 'per_object_tmiou.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Plot 2: Per-class T-mIoU bar chart
    if 'per_class_metrics' in results and results['per_class_metrics']:
        fig, ax = plt.subplots(figsize=(12, 6))
        classes = list(results['per_class_metrics'].keys())
        tmiou_values = [results['per_class_metrics'][c]['T_mIoU'] for c in classes]
        tmiou_std = [results['per_class_metrics'][c]['T_mIoU_std'] for c in classes]
        counts = [results['per_class_metrics'][c]['count'] for c in classes]
        
        # Sort by T-mIoU
        sorted_idx = np.argsort(tmiou_values)[::-1]
        classes = [classes[i] for i in sorted_idx]
        tmiou_values = [tmiou_values[i] for i in sorted_idx]
        tmiou_std = [tmiou_std[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]
        
        bars = ax.bar(range(len(classes)), tmiou_values, yerr=tmiou_std, capsize=3, alpha=0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('T-mIoU')
        ax.set_title('Per-Class Temporal Mean IoU')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'per_class_tmiou.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Plot 3: Summary metrics radar chart
    metric_names = ['T-mIoU', 'T-SR', 'ID Consistency', 'Detection Rate', 'MOTA', 'MOTP']
    metric_keys = ['T_mIoU', 'T_SR', 'ID_consistency', 'detection_rate', 'MOTA', 'MOTP']
    
    values = []
    for key in metric_keys:
        val = results.get(key, 0)
        # Clamp to [0, 1] for visualization
        values.append(max(0, min(1, val)))
    
    # Close the radar chart
    values.append(values[0])
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles.append(angles[0])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, markersize=8)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
    ax.set_ylim(0, 1)
    ax.set_title('Tracking Metrics Summary', y=1.08)
    
    # Add value labels
    for angle, value, name in zip(angles[:-1], values[:-1], metric_names):
        ax.annotate(f'{value:.2f}', xy=(angle, value), xytext=(angle, value + 0.1),
                   ha='center', fontsize=9)
    
    if output_dir:
        plt.savefig(output_dir / 'metrics_radar.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Configuration
    cfg = OmegaConf.create({
        # YOLO settings
        'yolo_model': 'yoloe-11l-seg-pf.pt',  # Update with your model path
        'conf': 0.25,
        'iou': 0.5,
        
        # Mask preprocessing
        'kernel_size': 17,
        'alpha': 0.7,
        'fast_mask': True,
        
        # Point cloud settings
        'max_points_per_obj': 2000,
        'max_accumulated_points': 10000,
        'o3_nb_neighbors': 50,
        'o3std_ratio': 0.1,
        
        # Tracking settings
        'tracking_overlap_threshold': 0.3,
        'tracking_distance_threshold': 0.5,
        'tracking_inactive_limit': 0,
        'tracking_volume_ratio_threshold': 0.1,
        'reprojection_visibility_threshold': 0.2,
        
        # ============================================================
        # CLASSES TO SKIP/FILTER OUT
        # ============================================================
        # These are typically large structural elements, rooms, or spaces
        # that shouldn't be tracked as individual objects
        'skip_classes': [
            # Structural elements
            'wall', 'floor', 'ceiling', 'roof', 'kitchen floor', 'carpet'
            'stairway', 'stairs', 'stair', 'escalator', 'elevator',
            
            # Rooms and spaces
            'room', 'kitchen', 'bathroom', 'bedroom', 'living room',
            'dining room', 'office', 'hallway', 'corridor', 'lobby',
            'garage', 'basement', 'attic',
            
            # Large venues/areas
            'basketball court', 'tennis court', 'football field',
            'soccer field', 'baseball field', 'stadium', 'arena',
            'mall', 'store', 'shop', 'market', 'supermarket',
            'restaurant', 'cafe', 'bar', 'gym', 'pool',
            'parking lot', 'parking', 'road', 'street', 'sidewalk',
            'highway', 'bridge', 'tunnel', 'court', 'courtyard',
            
            # Outdoor elements
            'sky', 'ground', 'grass', 'field', 'lawn',
            'mountain', 'hill', 'river', 'lake', 'ocean', 'sea',
            'beach', 'forest', 'tree', 'trees',
            
            # Building types
            'building', 'house', 'apartment', 'skyscraper',
            'warehouse', 'factory', 'school', 'hospital',
            'church', 'temple', 'mosque',
            
            # Other large elements
            'platform', 'stage', 'runway', 'track',
        ],
        
        # ============================================================
        # DEBUG VISUALIZATION SETTINGS
        # ============================================================
        'visualization': {
            'enabled': False,  # Master switch for all visualization
            
            # What to visualize
            'show_2d': True,      # YOLO masks with class IDs & global IDs
            'show_3d': True,      # Open3D: all objects with BLUE=visible, RED=not visible
            'show_comparison': False,  # GT vs Predictions side-by-side
            
            # Visualization frequency
            'interval': 10,       # Visualize every N frames (1 = every frame)
            
            # Display options
            'show_windows': True,  # Show matplotlib/Open3D windows (set False for headless)
            'point_size': 2.0,     # Point size for 3D visualization
            
            # Save options (optional)
            'save_dir': None,      # Directory to save visualizations (None = don't save)
            # Example: 'save_dir': './debug_vis'
        }
    })
    
    # Update paths for your system
    import sys
    
    # Parse command line arguments
    scene_path = None
    multi_mode = False
    vis_enabled = False
    
    for arg in sys.argv[1:]:
        if arg == '--multi':
            multi_mode = True
        elif arg == '--vis' or arg == '--visualize':
            vis_enabled = True
        elif arg == '--vis-3d':
            vis_enabled = True
            cfg.visualization.show_2d = False
            cfg.visualization.show_3d = True
        elif arg == '--vis-2d':
            vis_enabled = True
            cfg.visualization.show_2d = True
            cfg.visualization.show_3d = False
        elif arg.startswith('--vis-interval='):
            cfg.visualization.interval = int(arg.split('=')[1])
        elif arg.startswith('--vis-save='):
            cfg.visualization.save_dir = arg.split('=')[1]
        elif not arg.startswith('--'):
            scene_path = arg
    
    # Enable visualization if flag was passed
    if vis_enabled:
        cfg.visualization.enabled = True
    
    if scene_path is None:
        # Default scene path - update this!
        scene_path = "/home/maribjonov_mr/IsaacSim_bench/scene_3"
        print(f"Usage: python benchmark_tracking.py <scene_path> [options]")
        print(f"       python benchmark_tracking.py <scenes_root> --multi")
        print(f"\nOptions:")
        print(f"  --vis, --visualize    Enable debug visualization")
        print(f"  --vis-2d              Enable only 2D visualization (YOLO masks)")
        print(f"  --vis-3d              Enable only 3D visualization (Open3D)")
        print(f"  --vis-interval=N      Visualize every N frames (default: 10)")
        print(f"  --vis-save=DIR        Save visualizations to directory")
        print(f"  --multi               Run multi-scene benchmark")
        print(f"\nUsing default: {scene_path}")
    
    # Check for multi-scene mode
    if multi_mode:
        # Multi-scene benchmark
        results = run_multi_scene_benchmark(scene_path, cfg)
        output_path = Path(scene_path) / "benchmark_results_all.json"
        save_results(results, str(output_path))
    else:
        # Single scene benchmark
        benchmark = TrackingBenchmark(cfg)
        results = benchmark.run_benchmark(scene_path)
        
        # Save results
        output_path = Path(scene_path) / "tracking_benchmark_results.json"
        save_results(results, str(output_path))
        
        # Generate plots
        plot_tracking_results(results, output_dir=Path(scene_path) / "benchmark_plots")
