"""
Benchmarking script for YOLO-SSG tracking on THUD dataset.

Metrics Computed:
- T-mIoU: (1/T) * Σ IoU(M̂_t^o, M_t^o) - Temporal mean IoU per object
- T-SR: 1[∀t: |M̂_t^o| > 0] - Temporal Success Rate (object tracked in all frames) 
- ID Switches: Number of times an object's predicted ID changes
- MOTA: Multi-Object Tracking Accuracy
- MOTP: Multi-Object Tracking Precision
- Per-class tracking performance

Usage:
    python benchmark_thud.py --scene thud/Synthetic/Gym/Static/Capture_1
    python benchmark_thud.py --multi  # Run on all discovered THUD scenes
"""

import os
import sys
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from omegaconf import OmegaConf

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[WARN] Open3D not available. 3D visualization disabled.")

# Import THUD data loader
from thud_loader import THUDDataLoader, GTObject, discover_thud_scenes, get_scene_info

# Import YOLO-SSG components
import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry


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
    """Visualize YOLO tracking results with masks, class names, and IDs."""
    if rgb_image.shape[2] == 3:
        vis_img = rgb_image.copy()
    else:
        vis_img = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2RGB)
    
    H, W = vis_img.shape[:2]
    overlay = vis_img.copy()
    n_objects = len(masks_clean) if masks_clean else 0
    
    for i in range(n_objects):
        if masks_clean[i] is None:
            continue
        
        yolo_id = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1
        global_id = int(global_ids[i]) if global_ids is not None and i < len(global_ids) else -1
        class_name = class_names[i] if class_names is not None and i < len(class_names) else "unknown"
        match_src = match_sources[i] if match_sources is not None and i < len(match_sources) else ""
        
        color_id = global_id if global_id >= 0 else yolo_id
        color = generate_color_from_id(color_id)
        
        mask = masks_clean[i]
        if mask.dtype != bool:
            mask = mask > 0
        overlay[mask] = (
            overlay[mask] * (1 - alpha) + 
            np.array(color[::-1]) * alpha
        ).astype(np.uint8)
        
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color[::-1], 2)
            
            label_parts = [class_name]
            label_parts.append(f"Y:{yolo_id}")
            if global_id >= 0:
                label_parts.append(f"G:{global_id}")
            if match_src:
                src_short = match_src[:3] if len(match_src) > 3 else match_src
                label_parts.append(f"[{src_short}]")
            label = " | ".join(label_parts)
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color[::-1], -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
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
    """Visualize GT vs Predicted masks side by side."""
    H, W = rgb_image.shape[:2]
    gt_panel = rgb_image.copy()
    pred_panel = rgb_image.copy()
    
    # Draw GT masks
    for gt_obj in gt_objects:
        if gt_obj.mask is None:
            continue
        mask = gt_obj.mask > 0 if gt_obj.mask.dtype != bool else gt_obj.mask
        color = generate_color_from_id(gt_obj.track_id)
        gt_panel[mask] = (gt_panel[mask] * (1 - alpha) + np.array(color[::-1]) * alpha).astype(np.uint8)
        
        ys, xs = np.where(mask)
        if len(xs) > 0:
            x1, y1 = int(xs.min()), int(ys.min())
            label = f"{gt_obj.class_name}[{gt_obj.track_id}]"
            cv2.putText(gt_panel, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 1)
    
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
            is_matched = pred_obj.global_id in matched_pred_ids
            status = "✓" if is_matched else "✗"
            label = f"{pred_obj.class_name}[{pred_obj.global_id}]{status}"
            cv2.putText(pred_panel, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 1)
    
    combined = np.hstack([gt_panel, pred_panel])
    
    if show:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        axes[0].imshow(gt_panel)
        axes[0].set_title(f"Ground Truth - Frame {frame_idx}\n({len(gt_objects)} objects)")
        axes[0].axis('off')
        axes[1].imshow(pred_panel)
        axes[1].set_title(f"Predictions - Frame {frame_idx}\n({len(pred_objects)} objects, {len(gt_to_pred)} matched)")
        axes[1].axis('off')
        plt.suptitle(f"GT vs Predictions - Frame {frame_idx}", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    return combined


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PredObject:
    """Predicted object for a single frame."""
    global_id: int
    yolo_track_id: int
    class_name: Optional[str] = None
    mask: Optional[np.ndarray] = None
    bbox_3d: Optional[Dict] = None
    match_source: str = ""


@dataclass
class TrackingMetrics:
    """Metrics for tracking evaluation."""
    per_object_iou: Dict[int, List[float]] = field(default_factory=dict)
    per_object_tracked: Dict[int, List[bool]] = field(default_factory=dict)
    per_object_pred_ids: Dict[int, List[int]] = field(default_factory=dict)
    per_object_class: Dict[int, str] = field(default_factory=dict)
    
    per_class_iou: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    per_class_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    total_gt_objects: int = 0
    total_pred_objects: int = 0
    total_matches: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    
    frames_processed: int = 0
    
    gt_to_registry_match: Dict[int, int] = field(default_factory=dict)
    gt_to_registry_iou: Dict[int, float] = field(default_factory=dict)


# ============================================================================
# METRIC COMPUTATION FUNCTIONS
# ============================================================================

def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    if mask1 is None or mask2 is None:
        return 0.0
    
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2.astype(np.uint8), (mask1.shape[1], mask1.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
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
                            iou_threshold: float = 0.3,
                            rgb_image: np.ndarray = None,
                            frame_idx: int = 0,
                            visualize: bool = False,
                            save_path: str = None) -> Tuple[Dict[int, int], Dict[int, float]]:
    """
    Match predicted objects to ground truth using greedy IoU matching.
    
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
            iou_matrix[i, j] = compute_mask_iou(gt_obj.mask, pred_obj.mask)
    
    # Greedy matching
    mapping = {}
    ious = {}
    used_pred = set()
    used_gt = set()
    
    matches = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((i, j, iou_matrix[i, j]))
    
    matches.sort(key=lambda x: -x[2])
    
    for gt_idx, pred_idx, iou in matches:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        
        gt_id = gt_objects[gt_idx].track_id
        pred_id = pred_objects[pred_idx].global_id
        
        mapping[gt_id] = pred_id
        ious[gt_id] = iou
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
    
    if visualize and rgb_image is not None:
        visualize_tracking_comparison(
            rgb_image=rgb_image,
            gt_objects=gt_objects,
            pred_objects=pred_objects,
            gt_to_pred=mapping,
            frame_idx=frame_idx,
            show=True,
            save_path=save_path
        )
    
    return mapping, ious


# ============================================================================
# THUD TRACKING BENCHMARK CLASS
# ============================================================================

class THUDBenchmark:
    """Main benchmarking class for YOLO-SSG tracking on THUD dataset."""
    
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.metrics = TrackingMetrics()
        
        self.gt_appearances = defaultdict(list)
        self.gt_to_pred_history = defaultdict(list)
        self.gt_last_masks = {}
        self.object_registry = None
    
    def run_benchmark(self, scene_path: str) -> Dict:
        """
        Run tracking benchmark on a THUD scene.
        
        Args:
            scene_path: Path to THUD scene directory
            
        Returns:
            metrics_dict: Dictionary with all computed metrics
        """
        print(f"\n{'='*60}")
        print(f"THUD TRACKING BENCHMARK")
        print(f"Scene: {scene_path}")
        print(f"{'='*60}\n")
        
        # Get skip classes from config
        skip_classes = set(self.cfg.get('skip_classes', []))
        print(f"Using {len(skip_classes)} skip classes for filtering")
        
        # Load GT dataset using THUDDataLoader
        gt_loader = THUDDataLoader(scene_path, verbose=True)
        
        frame_indices = gt_loader.get_frame_indices()
        n_frames = len(frame_indices)
        
        print(f"Total frames: {n_frames}")
        
        # Reset metrics
        self.metrics = TrackingMetrics()
        self.gt_appearances = defaultdict(list)
        self.gt_to_pred_history = defaultdict(list)
        self.gt_last_masks = {}
        
        # Initialize object registry
        object_registry = GlobalObjectRegistry(
            overlap_threshold=float(self.cfg.get('tracking_overlap_threshold', 0.3)),
            distance_threshold=float(self.cfg.get('tracking_distance_threshold', 0.5)),
            max_points_per_object=int(self.cfg.get('max_accumulated_points', 10000)),
            inactive_frames_limit=int(self.cfg.get('tracking_inactive_limit', 0)),
            volume_ratio_threshold=float(self.cfg.get('tracking_volume_ratio_threshold', 0.1)),
            reprojection_visibility_threshold=float(self.cfg.get('reprojection_visibility_threshold', 0.2))
        )
        
        self.object_registry = object_registry
        
        # Prepare paths for YOLO
        rgb_dir = str(gt_loader.rgb_dir)
        depth_dir = str(gt_loader.depth_dir)
        
        # Get sorted RGB and depth paths
        rgb_paths = sorted(gt_loader.rgb_dir.glob("rgb_*.png"), 
                          key=lambda p: int(p.stem.split('_')[1]))
        depth_paths = sorted(gt_loader.depth_dir.glob("depth_*.png"),
                            key=lambda p: int(p.stem.split('_')[1]))
        
        print(f"RGB images found: {len(rgb_paths)}")
        print(f"Depth images found: {len(depth_paths)}")
        
        # Cache depths
        print("Caching depth images...")
        depth_cache = {}
        for dp in depth_paths:
            frame_num = int(dp.stem.split('_')[1])
            depth_img = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                depth_cache[frame_num] = depth_img.astype(np.float32) / 1000.0
        
        # Get camera intrinsics from first frame
        camera_pose = gt_loader.get_camera_pose(frame_indices[0]) if frame_indices else None
        if camera_pose is not None:
            fx = camera_pose.intrinsic[0, 0]
            fy = camera_pose.intrinsic[1, 1] 
            cx = camera_pose.intrinsic[0, 2]
            cy = camera_pose.intrinsic[1, 2]
            print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # Initialize YOLO tracker
        print("Initializing YOLO tracker...")
        
        # Convert paths for YOLO tracker
        rgb_path_strs = [str(p) for p in rgb_paths]
        depth_path_strs = [str(p) for p in depth_paths]
        
        # Map frame indices to paths
        frame_to_rgb = {int(p.stem.split('_')[1]): str(p) for p in rgb_paths}
        frame_to_depth = {int(p.stem.split('_')[1]): str(p) for p in depth_paths}
        
        # Visualization config
        vis_cfg = self.cfg.get('visualization', {})
        vis_enabled = vis_cfg.get('enabled', False)
        vis_interval = vis_cfg.get('interval', 10)
        vis_save_dir = vis_cfg.get('save_dir', None)
        
        # Process frames
        print("Processing frames...")
        
        # Use YOLO to track objects
        results_stream = yutils.track_objects_in_video_stream(
            rgb_dir,
            depth_path_strs[:len(rgb_path_strs)],  # Match count
            model_path=self.cfg.yolo_model,
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
        )
        
        frame_counter = 0
        for yl_res, rgb_cur_path, depth_cur_path in tqdm(results_stream, total=len(rgb_path_strs), desc="Benchmarking"):
            # Get frame index from path
            rgb_name = Path(rgb_cur_path).stem
            frame_idx = int(rgb_name.split('_')[1])
            
            # Skip if no GT data for this frame
            if frame_idx not in gt_loader.frame_data:
                frame_counter += 1
                continue
            
            # Load GT for this frame
            gt_objects_raw = gt_loader.get_gt_objects(frame_idx)
            
            # Filter GT by skip classes
            gt_objects = [obj for obj in gt_objects_raw 
                         if obj.class_name.lower() not in skip_classes]
            
            # Load depth
            depth_m = depth_cache.get(frame_idx)
            if depth_m is None:
                frame_counter += 1
                continue
            
            # Get camera pose
            pose = gt_loader.get_camera_pose(frame_idx)
            pose_matrix = np.eye(4)
            if pose is not None:
                # Build pose matrix from translation and rotation quaternion
                pose_matrix[:3, 3] = pose.translation
                # Convert quaternion to rotation matrix (simplified)
                # Note: For full quaternion support, use scipy.spatial.transform.Rotation
            
            # Process YOLO results
            masks_raw = yl_res.masks
            track_ids = np.array([], dtype=np.int64)
            class_names = []
            masks_clean = []
            
            if masks_raw is not None and yl_res.boxes is not None:
                if hasattr(yl_res.boxes, 'id') and yl_res.boxes.id is not None:
                    track_ids = yl_res.boxes.id.cpu().numpy().astype(np.int64)
                else:
                    track_ids = np.arange(len(masks_raw.data))
                
                masks_np = masks_raw.data.cpu().numpy()
                
                # Get class names from YOLO
                if hasattr(yl_res, 'names') and yl_res.boxes.cls is not None:
                    cls_indices = yl_res.boxes.cls.cpu().numpy().astype(int)
                    class_names = [yl_res.names.get(c, 'unknown') for c in cls_indices]
                else:
                    class_names = ['unknown'] * len(masks_np)
                
                # Preprocess masks
                for mask in masks_np:
                    mask_resized = cv2.resize(
                        (mask > 0.5).astype(np.uint8),
                        (gt_loader.image_width, gt_loader.image_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                    masks_clean.append(mask_resized)
            
            # Filter by skip classes
            if skip_classes and class_names:
                masks_clean, track_ids, class_names = yutils.filter_detections_by_class(
                    masks_clean, track_ids, class_names, skip_classes
                )
            
            # Update object registry
            intrinsic = pose.intrinsic if pose else np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]])
            
            frame_objs = object_registry.update_frame(
                masks=masks_clean,
                yolo_track_ids=track_ids,
                depth_im=depth_m,
                pose=pose_matrix,
                intrinsic=intrinsic,
                class_names=class_names,
                max_points_per_mask=int(self.cfg.get('max_points_per_obj', 2000)),
            )
            
            # Build PredObject list
            pred_objects = []
            for obj in frame_objs:
                pred_objects.append(PredObject(
                    global_id=obj['global_id'],
                    yolo_track_id=obj.get('yolo_track_id', -1),
                    class_name=obj.get('class_name', 'unknown'),
                    mask=obj.get('mask'),
                    bbox_3d=obj.get('bbox_3d'),
                    match_source=obj.get('match_source', '')
                ))
            
            # Match predictions to GT
            gt_to_pred, gt_ious = match_predictions_to_gt(
                gt_objects, pred_objects,
                iou_threshold=float(self.cfg.get('tracking_overlap_threshold', 0.3))
            )
            
            # Update metrics
            self._update_frame_metrics(
                frame_idx, gt_objects, pred_objects, gt_to_pred, gt_ious
            )
            
            # Visualization
            if vis_enabled and frame_counter % vis_interval == 0:
                rgb_image = cv2.imread(rgb_cur_path)
                if rgb_image is not None:
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    
                    save_2d_path = None
                    if vis_save_dir:
                        save_dir = Path(vis_save_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_2d_path = str(save_dir / f"thud_tracking_frame{frame_idx:06d}.png")
                    
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
            
            frame_counter += 1
        
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
        
        matched_pred_ids = set(gt_to_pred.values())
        
        self.metrics.total_false_positives += len(pred_objects) - len(matched_pred_ids)
        
        for gt_obj in gt_objects:
            gt_id = gt_obj.track_id
            
            self.gt_appearances[gt_id].append(frame_idx)
            
            if gt_id not in self.metrics.per_object_class:
                self.metrics.per_object_class[gt_id] = gt_obj.class_name
            
            if gt_obj.mask is not None:
                self.gt_last_masks[gt_id] = (gt_obj.mask.copy(), frame_idx, gt_obj.class_name)
            
            if gt_id not in self.metrics.per_object_iou:
                self.metrics.per_object_iou[gt_id] = []
                self.metrics.per_object_tracked[gt_id] = []
                self.metrics.per_object_pred_ids[gt_id] = []
            
            if gt_id in gt_to_pred:
                pred_id = gt_to_pred[gt_id]
                iou = gt_ious.get(gt_id, 0.0)
                
                self.metrics.per_object_iou[gt_id].append(iou)
                self.metrics.per_object_tracked[gt_id].append(True)
                self.metrics.per_object_pred_ids[gt_id].append(pred_id)
                
                self.gt_to_pred_history[gt_id].append((frame_idx, pred_id, iou))
                
                cls_name = gt_obj.class_name
                self.metrics.per_class_iou[cls_name].append(iou)
                self.metrics.per_class_count[cls_name] += 1
            else:
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
        
        # T-mIoU: Temporal mean IoU per object, then average
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
        
        # T-SR: Success rate (object tracked in all its appearances)
        t_sr_per_object = {}
        for gt_id, tracked_list in self.metrics.per_object_tracked.items():
            if len(tracked_list) > 0:
                t_sr_per_object[gt_id] = 1.0 if all(tracked_list) else 0.0
        
        if t_sr_per_object:
            results['T_SR'] = np.mean(list(t_sr_per_object.values()))
            results['T_SR_per_object'] = t_sr_per_object
        else:
            results['T_SR'] = 0.0
            results['T_SR_per_object'] = {}
        
        # ID consistency: Check if same pred_id used throughout
        id_consistency_per_object = {}
        for gt_id, pred_ids in self.metrics.per_object_pred_ids.items():
            valid_ids = [p for p in pred_ids if p >= 0]
            if len(valid_ids) > 0:
                most_common = Counter(valid_ids).most_common(1)[0][1]
                id_consistency_per_object[gt_id] = most_common / len(valid_ids)
            else:
                id_consistency_per_object[gt_id] = 0.0
        
        if id_consistency_per_object:
            results['ID_consistency'] = np.mean(list(id_consistency_per_object.values()))
            results['ID_consistency_per_object'] = id_consistency_per_object
        else:
            results['ID_consistency'] = 0.0
            results['ID_consistency_per_object'] = {}
        
        # ID switches: Count changes in pred_id
        total_switches = 0
        for gt_id, pred_ids in self.metrics.per_object_pred_ids.items():
            valid_ids = [p for p in pred_ids if p >= 0]
            for i in range(1, len(valid_ids)):
                if valid_ids[i] != valid_ids[i-1]:
                    total_switches += 1
        results['ID_switches'] = total_switches
        
        # Detection rate
        if results['total_gt_instances'] > 0:
            results['detection_rate'] = results['total_matches'] / results['total_gt_instances']
        else:
            results['detection_rate'] = 0.0
        
        # MOTA: Multi-Object Tracking Accuracy
        # MOTA = 1 - (FN + FP + ID_switches) / total_gt
        if results['total_gt_instances'] > 0:
            mota = 1.0 - (self.metrics.total_false_negatives + 
                         self.metrics.total_false_positives + 
                         total_switches) / results['total_gt_instances']
            results['MOTA'] = max(0.0, mota)
        else:
            results['MOTA'] = 0.0
        
        # MOTP: Multi-Object Tracking Precision (mean IoU of matched pairs)
        all_ious = []
        for gt_id, ious in self.metrics.per_object_iou.items():
            all_ious.extend([iou for iou in ious if iou > 0])
        
        if all_ious:
            results['MOTP'] = np.mean(all_ious)
        else:
            results['MOTP'] = 0.0
        
        # Per-class metrics
        per_class_metrics = {}
        for cls_name, ious in self.metrics.per_class_iou.items():
            if ious:
                per_class_metrics[cls_name] = {
                    'mean_iou': np.mean(ious),
                    'count': self.metrics.per_class_count[cls_name],
                    'std_iou': np.std(ious),
                }
        results['per_class_metrics'] = per_class_metrics
        
        return results
    
    def _print_results(self, metrics: Dict):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*60)
        print("THUD BENCHMARK RESULTS")
        print("="*60)
        
        print(f"\n{'='*40}")
        print("FRAME-LEVEL STATISTICS")
        print(f"{'='*40}")
        print(f"Frames processed:      {metrics['frames_processed']}")
        print(f"Total GT instances:    {metrics['total_gt_instances']}")
        print(f"Total Pred instances:  {metrics['total_pred_instances']}")
        print(f"Total matches:         {metrics['total_matches']}")
        print(f"False positives:       {metrics['total_false_positives']}")
        print(f"False negatives:       {metrics['total_false_negatives']}")
        print(f"Unique GT objects:     {metrics['unique_gt_objects']}")
        
        print(f"\n{'='*40}")
        print("TRACKING METRICS")
        print(f"{'='*40}")
        print(f"T-mIoU:           {metrics.get('T_mIoU', 0):.4f} ± {metrics.get('T_mIoU_std', 0):.4f}")
        print(f"T-SR:             {metrics.get('T_SR', 0):.4f}")
        print(f"ID consistency:   {metrics.get('ID_consistency', 0):.4f}")
        print(f"ID switches:      {metrics.get('ID_switches', 0)}")
        print(f"Detection rate:   {metrics.get('detection_rate', 0):.4f}")
        print(f"MOTA:             {metrics.get('MOTA', 0):.4f}")
        print(f"MOTP:             {metrics.get('MOTP', 0):.4f}")
        
        # Per-class results
        if 'per_class_metrics' in metrics and metrics['per_class_metrics']:
            print(f"\n{'='*40}")
            print("PER-CLASS METRICS")
            print(f"{'='*40}")
            print(f"{'Class':<20} {'Count':<10} {'Mean IoU':<12} {'Std IoU':<10}")
            print("-" * 52)
            
            for cls_name, cls_metrics in sorted(metrics['per_class_metrics'].items()):
                print(f"{cls_name:<20} {cls_metrics['count']:<10} "
                      f"{cls_metrics['mean_iou']:.4f}       {cls_metrics['std_iou']:.4f}")


# ============================================================================
# MULTI-SCENE BENCHMARK
# ============================================================================

def run_multi_scene_benchmark(scenes: list, cfg: OmegaConf, output_dir: str = None) -> Dict:
    """Run benchmark on multiple THUD scenes and aggregate results."""
    
    scene_dirs = [Path(s) for s in scenes]
    
    valid_scenes = []
    for scene_dir in scene_dirs:
        if scene_dir.exists() and (scene_dir / "RGB").exists():
            valid_scenes.append(scene_dir)
        else:
            print(f"[WARN] Skipping invalid scene: {scene_dir}")
    
    print(f"\nFound {len(valid_scenes)} valid THUD scenes to benchmark")
    
    all_results = {}
    aggregated = {
        'T_mIoU': [],
        'T_SR': [],
        'ID_consistency': [],
        'detection_rate': [],
        'MOTA': [],
        'MOTP': [],
    }
    
    for scene_dir in valid_scenes:
        scene_name = f"{scene_dir.parent.parent.parent.name}/{scene_dir.parent.parent.name}/{scene_dir.name}"
        print(f"\n{'='*60}")
        print(f"Processing: {scene_name}")
        print(f"{'='*60}")
        
        try:
            benchmark = THUDBenchmark(cfg)
            scene_results = benchmark.run_benchmark(str(scene_dir))
            all_results[scene_name] = scene_results
            
            for key in aggregated.keys():
                if key in scene_results:
                    aggregated[key].append(scene_results[key])
        
        except Exception as e:
            print(f"[ERROR] Failed to benchmark {scene_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compute overall metrics
    print("\n" + "="*60)
    print("OVERALL THUD BENCHMARK RESULTS")
    print("="*60)
    
    overall = {}
    for key, values in aggregated.items():
        if values:
            overall[key] = np.mean(values)
            overall[f'{key}_std'] = np.std(values)
            print(f"{key:20}: {overall[key]:.4f} ± {overall[f'{key}_std']:.4f}")
    
    return {
        'per_scene': all_results,
        'overall': overall,
        'num_scenes': len(all_results),
        'output_dir': output_dir or '.',
    }


def save_results(results: Dict, output_path: str):
    """Save benchmark results to JSON file."""
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    cfg = OmegaConf.create({
        'yolo_model': 'yoloe-11l-seg-pf.pt',
        'conf': 0.25,
        'iou': 0.5,
        'kernel_size': 11,
        'alpha': 0.7,
        'fast_mask': True,
        'max_points_per_obj': 2000,
        'max_accumulated_points': 10000,
        'o3_nb_neighbors': 50,
        'o3std_ratio': 0.1,
        'tracking_overlap_threshold': 0.3,
        'tracking_distance_threshold': 0.5,
        'tracking_inactive_limit': 0,
        'tracking_volume_ratio_threshold': 0.1,
        'reprojection_visibility_threshold': 0.2,
        'skip_classes': [
            'wall', 'floor', 'ceiling', 'roof', 'window',
            'stairway', 'stairs', 'escalator', 'elevator',
            'room', 'kitchen', 'bathroom', 'bedroom', 'living room',
            'sky', 'ground', 'grass', 'building', 'house',
        ],
        'visualization': {
            'enabled': False,
            'show_2d': True,
            'show_3d': False,
            'show_comparison': False,
            'interval': 20,
            'show_windows': True,
            'point_size': 2.0,
            'save_dir': None,
        }
    })
    
    # Parse command line arguments
    scene_path = None
    multi_mode = False
    vis_enabled = False
    output_dir = None
    
    for arg in sys.argv[1:]:
        if arg.startswith('--scene='):
            scene_path = arg.split('=', 1)[1]
        elif arg == '--multi':
            multi_mode = True
        elif arg == '--vis':
            vis_enabled = True
        elif arg.startswith('--output='):
            output_dir = arg.split('=', 1)[1]
    
    if vis_enabled:
        cfg.visualization.enabled = True
    
    if scene_path is None and not multi_mode:
        # Default THUD path
        default_path = "thud/Synthetic/Gym/Static/Capture_1"
        if Path(default_path).exists():
            scene_path = default_path
        else:
            print("Usage: python benchmark_thud.py --scene=<path_to_thud_scene>")
            print("       python benchmark_thud.py --multi  # Benchmark all THUD scenes")
            print("\nOptions:")
            print("  --vis        Enable visualization")
            print("  --output=DIR Save results to directory")
            sys.exit(1)
    
    if multi_mode:
        # Discover all THUD scenes
        thud_root = "thud"
        scenes = discover_thud_scenes(thud_root)
        
        if not scenes:
            print(f"No THUD scenes found in {thud_root}")
            sys.exit(1)
        
        print(f"Discovered {len(scenes)} THUD scenes:")
        for s in scenes:
            info = get_scene_info(s)
            print(f"  - {info['parent']}/{info['name']}: {info['rgb_count']} frames")
        
        results = run_multi_scene_benchmark(scenes, cfg, output_dir)
        
        if output_dir:
            save_results(results, f"{output_dir}/thud_benchmark_results.json")
    else:
        # Single scene benchmark
        print(f"Running THUD benchmark on: {scene_path}")
        
        benchmark = THUDBenchmark(cfg)
        results = benchmark.run_benchmark(scene_path)
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_results(results, f"{output_dir}/thud_benchmark_results.json")
