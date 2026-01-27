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

# Import YOLO-SSG components
import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry


# Camera intrinsics (from your dataset)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FOCAL_LENGTH = 50.0
HORIZONTAL_APERTURE = 80.0
VERTICAL_APERTURE = 45.0

fx = FOCAL_LENGTH / HORIZONTAL_APERTURE * IMAGE_WIDTH   # 800.0
fy = FOCAL_LENGTH / VERTICAL_APERTURE * IMAGE_HEIGHT    # 800.0
cx = IMAGE_WIDTH / 2.0  # 640.0
cy = IMAGE_HEIGHT / 2.0  # 360.0

PNG_DEPTH_SCALE = 0.00015244  # From your calibration
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0


@dataclass
class GTObject:
    """Ground truth object for a single frame."""
    track_id: int
    class_name: str
    mask: Optional[np.ndarray] = None  # Binary mask (H, W)
    bbox_2d: Optional[List[float]] = None  # [x1, y1, x2, y2]
    bbox_3d: Optional[Dict] = None  # 3D bbox info
    prim_path: str = ""
    visibility: float = 1.0


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


class GTDatasetLoader:
    """Load ground truth data from Isaac Sim benchmark dataset."""
    
    def __init__(self, scene_path: str):
        self.scene_path = Path(scene_path)
        self.rgb_dir = self.scene_path / "rgb"
        self.depth_dir = self.scene_path / "depth"
        self.seg_dir = self.scene_path / "seg"
        self.bbox_dir = self.scene_path / "bbox"
        self.traj_path = self.scene_path / "traj.txt"
        
        # Cache
        self._frame_count = None
        self._poses = None
        
    def get_frame_count(self) -> int:
        """Get number of frames in the scene."""
        if self._frame_count is None:
            rgb_files = list(self.rgb_dir.glob("frame*.jpg"))
            self._frame_count = len(rgb_files)
        return self._frame_count
    
    def get_poses(self) -> List[np.ndarray]:
        """Load camera poses."""
        if self._poses is None:
            self._poses = yutils.load_camera_poses(str(self.traj_path))
        return self._poses
    
    def get_frame_paths(self, frame_idx: int) -> Dict[str, Path]:
        """Get all paths for a specific frame."""
        frame_num = frame_idx + 1  # 1-indexed in filenames
        return {
            'rgb': self.rgb_dir / f"frame{frame_num:06d}.jpg",
            'depth': self.depth_dir / f"depth{frame_num:06d}.png",
            'seg_png': self.seg_dir / f"semantic{frame_num:06d}.png",
            'seg_json': self.seg_dir / f"semantic{frame_num:06d}_info.json",
            'bbox_json': self.bbox_dir / f"bbox{frame_num:06d}.json",
        }
    
    def load_depth(self, frame_idx: int) -> np.ndarray:
        """Load depth map in meters."""
        paths = self.get_frame_paths(frame_idx)
        depth_path = paths['depth']
        
        if not depth_path.exists():
            return None
        
        # Load 16-bit PNG
        depth_raw = np.array(Image.open(depth_path))
        
        # Convert to meters using the calibrated scale
        depth_m = depth_raw.astype(np.float32) * PNG_DEPTH_SCALE
        
        # Clamp invalid values
        depth_m[depth_m < MIN_DEPTH] = 0.0
        depth_m[depth_m > MAX_DEPTH] = 0.0
        
        return depth_m
    
    def load_segmentation(self, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Load segmentation map and info.
        
        Returns:
            seg_map: (H, W, 3) BGR color-coded segmentation
            seg_info: Dict mapping semantic_id -> {class, color_bgr}
        """
        paths = self.get_frame_paths(frame_idx)
        
        # Load colored segmentation
        seg_png_path = paths['seg_png']
        if not seg_png_path.exists():
            return None, {}
        
        seg_map = cv2.imread(str(seg_png_path), cv2.IMREAD_COLOR)  # BGR
        
        # Load segmentation info
        seg_json_path = paths['seg_json']
        seg_info = {}
        if seg_json_path.exists():
            with open(seg_json_path, 'r') as f:
                raw_info = json.load(f)
                for sem_id_str, info in raw_info.items():
                    sem_id = int(sem_id_str)
                    seg_info[sem_id] = {
                        'class': info['label']['class'],
                        'color_bgr': tuple(info['color_bgr'])
                    }
        
        return seg_map, seg_info
    
    def load_bboxes(self, frame_idx: int) -> Dict:
        """Load 2D and 3D bounding boxes."""
        paths = self.get_frame_paths(frame_idx)
        bbox_path = paths['bbox_json']
        
        if not bbox_path.exists():
            return {'bbox_2d': [], 'bbox_3d': []}
        
        with open(bbox_path, 'r') as f:
            data = json.load(f)
        
        result = {
            'bbox_2d': [],
            'bbox_3d': []
        }
        
        # Parse 2D bboxes
        if 'bboxes' in data and 'bbox_2d_tight' in data['bboxes']:
            for box in data['bboxes']['bbox_2d_tight']['boxes']:
                result['bbox_2d'].append({
                    'track_id': box.get('bbox_id', -1),
                    'semantic_id': box.get('semantic_id', -1),
                    'prim_path': box.get('prim_path', ''),
                    'label': self._extract_label(box.get('label', {})),
                    'xyxy': box.get('xyxy', [0, 0, 0, 0]),
                    'visibility': 1.0 - box.get('visibility_or_occlusion', 0.0)
                })
        
        # Parse 3D bboxes
        if 'bboxes' in data and 'bbox_3d' in data['bboxes']:
            for box in data['bboxes']['bbox_3d']['boxes']:
                result['bbox_3d'].append({
                    'track_id': box.get('track_id', box.get('bbox_id', -1)),
                    'semantic_id': box.get('semantic_id', -1),
                    'prim_path': box.get('prim_path', ''),
                    'label': box.get('label', 'unknown'),
                    'aabb': box.get('aabb_xyzmin_xyzmax', []),
                    'transform': box.get('transform_4x4', []),
                    'occlusion': box.get('occlusion_ratio', 0.0)
                })
        
        return result
    
    def _extract_label(self, label_dict: Dict) -> str:
        """Extract class label from label dict."""
        if isinstance(label_dict, str):
            return label_dict
        if isinstance(label_dict, dict):
            # Take first value
            for k, v in label_dict.items():
                return str(v) if v else str(k)
        return 'unknown'
    
    def get_gt_objects(self, frame_idx: int) -> List[GTObject]:
        """
        Get all ground truth objects for a frame.
        
        Returns list of GTObject with masks extracted from segmentation.
        """
        # Load segmentation
        seg_map, seg_info = self.load_segmentation(frame_idx)
        if seg_map is None:
            return []
        
        # Load bboxes for track_ids and labels
        bboxes = self.load_bboxes(frame_idx)
        
        # Build color -> bbox mapping
        # First, create mapping from prim_path to bbox info
        prim_to_bbox_2d = {}
        prim_to_bbox_3d = {}
        
        for box in bboxes['bbox_2d']:
            prim_to_bbox_2d[box['prim_path']] = box
        
        for box in bboxes['bbox_3d']:
            prim_to_bbox_3d[box['prim_path']] = box
        
        gt_objects = []
        
        # Extract objects from 3D bboxes (more reliable track_ids)
        for box_3d in bboxes['bbox_3d']:
            track_id = box_3d['track_id']
            label = box_3d['label']
            prim_path = box_3d['prim_path']
            
            # Skip background/unlabelled/walls/floor/ceiling
            if label.lower() in ['background', 'unlabelled', 'wall', 'floor', 'ceiling']:
                continue
            
            # Find corresponding 2D bbox
            box_2d = prim_to_bbox_2d.get(prim_path, {})
            
            # Extract mask from segmentation using 2D bbox color matching
            mask = self._extract_mask_for_object(seg_map, seg_info, box_2d, box_3d)
            
            gt_obj = GTObject(
                track_id=track_id,
                class_name=label,
                mask=mask,
                bbox_2d=box_2d.get('xyxy'),
                bbox_3d={
                    'aabb': box_3d['aabb'],
                    'transform': box_3d['transform']
                },
                prim_path=prim_path,
                visibility=box_2d.get('visibility', 1.0)
            )
            gt_objects.append(gt_obj)
        
        return gt_objects
    
    def _extract_mask_for_object(self, seg_map: np.ndarray, seg_info: Dict,
                                  box_2d: Dict, box_3d: Dict) -> Optional[np.ndarray]:
        """
        Extract binary mask for an object from segmentation map.
        
        Strategy:
        1. Use 2D bbox to crop region
        2. Find dominant non-background color in that region
        3. Create mask from that color
        """
        if seg_map is None:
            return None
        
        H, W = seg_map.shape[:2]
        
        # Get 2D bbox
        xyxy = box_2d.get('xyxy', [0, 0, W, H])
        if xyxy is None:
            xyxy = [0, 0, W, H]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Crop region
        crop = seg_map[y1:y2, x1:x2]
        
        # Find unique colors in crop (excluding background black)
        colors = crop.reshape(-1, 3)
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
        
        # Filter out background (black) and find dominant color
        best_color = None
        best_count = 0
        
        for color, count in zip(unique_colors, counts):
            # Skip black (background)
            if np.all(color == 0):
                continue
            if count > best_count:
                best_count = count
                best_color = color
        
        if best_color is None:
            return None
        
        # Create full mask from this color
        mask = np.all(seg_map == best_color, axis=2).astype(np.uint8) * 255
        
        return mask


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
        
        # Load GT dataset
        gt_loader = GTDatasetLoader(scene_path)
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
            # Load GT for this frame
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
    })
    
    # Update paths for your system
    import sys
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
    else:
        # Default scene path - update this!
        scene_path = "./test_scene"
        print(f"Usage: python benchmark_tracking.py <scene_path>")
        print(f"       python benchmark_tracking.py <scenes_root> --multi")
        print(f"\nUsing default: {scene_path}")
    
    # Check for multi-scene mode
    if len(sys.argv) > 2 and sys.argv[2] == '--multi':
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
