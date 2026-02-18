"""
Isaac Sim Replicator Dataset Loader (v2 - Preloaded).

Loads ALL data at initialization for fast indexed access during benchmarking.
Supports RGB, depth, segmentation, 2D/3D bounding boxes, and camera poses.

Usage:
    loader = IsaacSimDataLoader(scene_path)
    
    # Access preloaded data
    n_frames = len(loader)
    rgb = loader.get_rgb(frame_idx)
    depth = loader.get_depth(frame_idx)
    gt_objects = loader.get_gt_objects(frame_idx)
    pose = loader.get_pose(frame_idx)
    
    # Iterate
    for frame_idx in range(len(loader)):
        gt_objs = loader.get_gt_objects(frame_idx)
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import cv2
from tqdm import tqdm

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Default camera intrinsics
# ============================================================================
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720
DEFAULT_FOCAL_LENGTH = 50.0
DEFAULT_HORIZONTAL_APERTURE = 80.0
DEFAULT_VERTICAL_APERTURE = 45.0
DEFAULT_MIN_DEPTH = 0.01
DEFAULT_MAX_DEPTH = 10.0
DEFAULT_PNG_MAX_VALUE = 65535

# Default labels to skip
DEFAULT_SKIP_LABELS = [
    'background', 'unlabelled', 'unlabeled', 'unknown',
    'wall', 'floor', 'ground', 'roof'
]


@dataclass
class GTObject:
    """Ground truth object representation."""
    track_id: int
    class_name: str
    semantic_id: int = -1
    mask: Optional[np.ndarray] = None  # Binary mask (H, W), 0/255
    bbox_2d: Optional[List[float]] = None  # [x1, y1, x2, y2]
    bbox_3d_aabb: Optional[List[float]] = None  # [xmin, ymin, zmin, xmax, ymax, zmax]
    bbox_3d_transform: Optional[np.ndarray] = None  # 4x4 transform matrix
    prim_path: str = ""
    visibility: float = 1.0
    occlusion: float = 0.0
    color_bgr: Optional[Tuple[int, int, int]] = None


class IsaacSimDataLoader:
    """
    Data loader for Isaac Sim Replicator datasets.
    
    PRELOADS all data at initialization for fast indexed access.
    
    Expected directory structure:
        scene_path/
            rgb/frame000001.jpg, frame000002.jpg, ...
            depth/depth000001.png, depth000002.png, ...
            seg/semantic000001.png, semantic000001_info.json, ...
            bbox/bboxes000001_info.json, ...
            traj.txt  (optional camera poses)
    """
    
    def __init__(
        self,
        scene_path: str,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        focal_length: float = DEFAULT_FOCAL_LENGTH,
        horizontal_aperture: float = DEFAULT_HORIZONTAL_APERTURE,
        vertical_aperture: float = DEFAULT_VERTICAL_APERTURE,
        min_depth: float = DEFAULT_MIN_DEPTH,
        max_depth: float = DEFAULT_MAX_DEPTH,
        skip_labels: List[str] = None,
        preload_rgb: bool = True,
        preload_depth: bool = True,
        preload_masks: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize and preload all data.
        
        Args:
            scene_path: Path to scene directory
            skip_labels: Labels to filter out (e.g., ['wall', 'floor'])
            preload_rgb: Whether to preload RGB images into memory
            preload_depth: Whether to preload depth maps into memory
            preload_masks: Whether to preload/compute masks
            verbose: Print loading progress
        """
        self.scene_path = Path(scene_path)
        self.verbose = verbose
        
        # Camera parameters
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.horizontal_aperture = horizontal_aperture
        self.vertical_aperture = vertical_aperture
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # Compute intrinsics
        self.fx = focal_length / horizontal_aperture * image_width
        self.fy = focal_length / vertical_aperture * image_height
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0
        
        # Depth scale
        self.png_max_value = DEFAULT_PNG_MAX_VALUE
        self.png_depth_scale = (max_depth - min_depth) / float(self.png_max_value)
        
        # Skip labels
        self.skip_labels = set(
            l.lower() for l in (skip_labels if skip_labels is not None else DEFAULT_SKIP_LABELS)
        )
        
        # Directory paths
        self.rgb_dir = self.scene_path / "rgb"
        self.depth_dir = self.scene_path / "depth"
        self.seg_dir = self.scene_path / "seg"
        self.bbox_dir = self.scene_path / "bbox"
        self.traj_path = self.scene_path / "traj.txt"
        
        # Detect file naming convention
        self._detect_file_format()
        
        # Count frames
        self._frame_count = self._count_frames()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Loading Isaac Sim Dataset: {self.scene_path.name}")
            print(f"{'='*60}")
            print(f"  Frames: {self._frame_count}")
        
        # ============================
        # PRELOAD ALL DATA
        # ============================
        
        # 1. Load trajectory (poses)
        self._poses = self._load_all_poses()
        if self.verbose:
            print(f"  Poses loaded: {len(self._poses)}")
        
        # 2. Load all bboxes and segmentation info (lightweight)
        self._bboxes_2d: List[List[Dict]] = []  # [frame_idx][obj_idx]
        self._bboxes_3d: List[List[Dict]] = []
        self._seg_info: List[Dict] = []  # [frame_idx] -> {semantic_id: {class, color_bgr}}
        self._load_all_annotations()
        
        # 3. Precompute GT objects per frame (with masks)
        self._gt_objects: List[List[GTObject]] = []
        self._precompute_gt_objects(preload_masks)
        
        # 4. Optionally preload images
        self._rgb_cache: Dict[int, np.ndarray] = {}
        self._depth_cache: Dict[int, np.ndarray] = {}
        self._seg_cache: Dict[int, np.ndarray] = {}
        
        if preload_rgb:
            self._preload_rgb()
        if preload_depth:
            self._preload_depth()
        
        if self.verbose:
            total_objects = sum(len(objs) for objs in self._gt_objects)
            print(f"  Total GT objects across all frames: {total_objects}")
            print(f"  Skip labels: {sorted(self.skip_labels)}")
            print(f"{'='*60}\n")
    
    def _detect_file_format(self):
        """Detect file naming convention."""
        self._rgb_pattern = "frame{:06d}.jpg"
        self._depth_pattern = "depth{:06d}.png"
        self._seg_pattern = "semantic{:06d}.png"
        self._seg_info_pattern = "semantic{:06d}_info.json"
        self._bbox_pattern = "bboxes{:06d}_info.json"
        
        # Check alternative bbox pattern
        if self.bbox_dir.exists():
            sample_files = list(self.bbox_dir.glob("*.json"))
            if sample_files:
                sample_name = sample_files[0].name
                if sample_name.startswith("bbox") and not sample_name.startswith("bboxes"):
                    self._bbox_pattern = "bbox{:06d}.json"
    
    def _count_frames(self) -> int:
        """Count frames in RGB directory."""
        if self.rgb_dir.exists():
            return len(list(self.rgb_dir.glob("frame*.jpg")))
        return 0
    
    def _get_frame_paths(self, frame_idx: int) -> Dict[str, Path]:
        """Get file paths for a frame."""
        frame_num = frame_idx + 1  # 1-indexed filenames
        return {
            'rgb': self.rgb_dir / self._rgb_pattern.format(frame_num),
            'depth': self.depth_dir / self._depth_pattern.format(frame_num),
            'seg_png': self.seg_dir / self._seg_pattern.format(frame_num),
            'seg_json': self.seg_dir / self._seg_info_pattern.format(frame_num),
            'bbox_json': self.bbox_dir / self._bbox_pattern.format(frame_num),
        }
    
    # =========================================================================
    # LOADING METHODS (called at init)
    # =========================================================================
    
    def _load_all_poses(self) -> List[np.ndarray]:
        """Load all camera poses from traj.txt."""
        poses = []
        if self.traj_path.exists():
            with open(self.traj_path, 'r') as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) == 16:
                        T = np.array([float(v) for v in vals]).reshape(4, 4)
                        poses.append(T)
        return poses
    
    def _load_all_annotations(self):
        """Load all bbox and segmentation info."""
        if self.verbose:
            print("  Loading annotations...")
        
        iterator = range(self._frame_count)
        if self.verbose:
            iterator = tqdm(iterator, desc="  Annotations", leave=False)
        
        for frame_idx in iterator:
            paths = self._get_frame_paths(frame_idx)
            
            # Load bbox JSON
            bboxes_2d = []
            bboxes_3d = []
            
            if paths['bbox_json'].exists():
                with open(paths['bbox_json'], 'r') as f:
                    data = json.load(f)
                
                # Parse 2D bboxes
                if 'bboxes' in data:
                    for key in ['bbox_2d_tight', 'bbox_2d_loose', 'bbox_2d']:
                        if key in data['bboxes']:
                            for box in data['bboxes'][key].get('boxes', []):
                                bboxes_2d.append({
                                    'track_id': box.get('bbox_id', box.get('track_id', -1)),
                                    'semantic_id': box.get('semantic_id', -1),
                                    'prim_path': box.get('prim_path', ''),
                                    'label': self._extract_label(box.get('label', {})),
                                    'xyxy': box.get('xyxy', [0, 0, 0, 0]),
                                    'visibility': 1.0 - box.get('visibility_or_occlusion', 
                                                                box.get('occlusion', 0.0))
                                })
                            break
                    
                    # Parse 3D bboxes
                    if 'bbox_3d' in data['bboxes']:
                        for box in data['bboxes']['bbox_3d'].get('boxes', []):
                            transform = box.get('transform_4x4')
                            if transform is not None:
                                transform = np.array(transform, dtype=np.float64).reshape(4, 4)
                            
                            bboxes_3d.append({
                                'track_id': box.get('track_id', box.get('bbox_id', -1)),
                                'semantic_id': box.get('semantic_id', -1),
                                'prim_path': box.get('prim_path', ''),
                                'label': box.get('label', 'unknown'),
                                'aabb': box.get('aabb_xyzmin_xyzmax', []),
                                'transform': transform,
                                'occlusion': box.get('occlusion_ratio', 0.0)
                            })
            
            self._bboxes_2d.append(bboxes_2d)
            self._bboxes_3d.append(bboxes_3d)
            
            # Load seg info JSON
            seg_info = {}
            if paths['seg_json'].exists():
                with open(paths['seg_json'], 'r') as f:
                    raw_info = json.load(f)
                    for sem_id_str, info in raw_info.items():
                        try:
                            sem_id = int(sem_id_str)
                        except ValueError:
                            continue
                        
                        # Handle label format: {"wall": "wall"} or string
                        label_data = info.get('label', {})
                        if isinstance(label_data, str):
                            cls_name = label_data
                        elif isinstance(label_data, dict):
                            # Take first value or key
                            cls_name = ''
                            for k, v in label_data.items():
                                cls_name = str(v) if v else str(k)
                                break
                        else:
                            cls_name = str(label_data) if label_data else 'unknown'
                        
                        seg_info[sem_id] = {
                            'class': cls_name,
                            'color_bgr': tuple(info.get('color_bgr', [0, 0, 0]))
                        }
            
            self._seg_info.append(seg_info)
    
    def _extract_label(self, label_dict) -> str:
        """Extract label from various formats."""
        if isinstance(label_dict, str):
            return label_dict
        if isinstance(label_dict, dict):
            for k, v in label_dict.items():
                return str(v) if v else str(k)
        return 'unknown'
    
    def _precompute_gt_objects(self, include_masks: bool):
        """Precompute GT objects for all frames."""
        if self.verbose:
            print("  Computing GT objects...")
        
        iterator = range(self._frame_count)
        if self.verbose:
            iterator = tqdm(iterator, desc="  GT Objects", leave=False)
        
        for frame_idx in iterator:
            gt_objects = self._compute_gt_objects_for_frame(frame_idx, include_masks)
            self._gt_objects.append(gt_objects)
    
    def _compute_gt_objects_for_frame(self, frame_idx: int, include_masks: bool) -> List[GTObject]:
        """
        Compute GT objects for a single frame.
        
        Uses 3D bboxes as source of truth, filters by skip_labels,
        and extracts masks using semantic_id from seg_info.
        """
        bboxes_3d = self._bboxes_3d[frame_idx]
        bboxes_2d = self._bboxes_2d[frame_idx]
        seg_info = self._seg_info[frame_idx]
        
        if not bboxes_3d:
            return []
        
        # Build prim_path -> 2D bbox mapping
        prim_to_bbox_2d = {box['prim_path']: box for box in bboxes_2d}
        
        # Load segmentation map if needed
        seg_map = None
        if include_masks:
            paths = self._get_frame_paths(frame_idx)
            if paths['seg_png'].exists():
                seg_map = cv2.imread(str(paths['seg_png']), cv2.IMREAD_COLOR)
        
        gt_objects = []
        
        for box_3d in bboxes_3d:
            track_id = box_3d['track_id']
            label = box_3d['label'] or 'unknown'
            prim_path = box_3d['prim_path']
            semantic_id = box_3d.get('semantic_id', -1)
            aabb = box_3d.get('aabb')
            
            # Skip if invalid 3D bbox
            if aabb is None or len(aabb) != 6:
                continue
            
            # Apply skip_labels filter
            should_skip = False
            for skip_label in self.skip_labels:
                if skip_label in label.lower():
                    should_skip = True
                    break
            if should_skip:
                continue
            
            # Find 2D bbox by prim_path
            box_2d = prim_to_bbox_2d.get(prim_path, {})
            
            # Extract mask using semantic_id
            mask = None
            color_bgr = None
            if include_masks and seg_map is not None and semantic_id in seg_info:
                color_bgr = seg_info[semantic_id].get('color_bgr')
                if color_bgr and color_bgr != (0, 0, 0):
                    color_array = np.array(color_bgr, dtype=np.uint8)
                    mask = np.all(seg_map == color_array, axis=2).astype(np.uint8) * 255
                    if mask.sum() == 0:
                        mask = None  # No pixels found
            
            gt_obj = GTObject(
                track_id=track_id,
                class_name=label,
                semantic_id=semantic_id,
                mask=mask,
                bbox_2d=box_2d.get('xyxy') if box_2d else None,
                bbox_3d_aabb=aabb,
                bbox_3d_transform=box_3d.get('transform'),
                prim_path=prim_path,
                visibility=box_2d.get('visibility', 1.0) if box_2d else 1.0,
                occlusion=box_3d.get('occlusion', 0.0),
                color_bgr=color_bgr,
            )
            gt_objects.append(gt_obj)
        
        return gt_objects
    
    def _preload_rgb(self):
        """Preload all RGB images."""
        if self.verbose:
            print("  Preloading RGB images...")
        
        iterator = range(self._frame_count)
        if self.verbose:
            iterator = tqdm(iterator, desc="  RGB", leave=False)
        
        for frame_idx in iterator:
            paths = self._get_frame_paths(frame_idx)
            if paths['rgb'].exists():
                self._rgb_cache[frame_idx] = cv2.imread(str(paths['rgb']), cv2.IMREAD_COLOR)
    
    def _preload_depth(self):
        """Preload all depth maps."""
        if self.verbose:
            print("  Preloading depth maps...")
        
        iterator = range(self._frame_count)
        if self.verbose:
            iterator = tqdm(iterator, desc="  Depth", leave=False)
        
        for frame_idx in iterator:
            paths = self._get_frame_paths(frame_idx)
            if paths['depth'].exists():
                depth_raw = np.array(Image.open(paths['depth']))
                depth_m = depth_raw.astype(np.float32) * self.png_depth_scale + self.min_depth
                depth_m[depth_m < self.min_depth] = 0.0
                depth_m[depth_m > self.max_depth] = 0.0
                self._depth_cache[frame_idx] = depth_m
    
    # =========================================================================
    # PUBLIC ACCESS METHODS (fast indexed access)
    # =========================================================================
    
    def __len__(self) -> int:
        """Number of frames."""
        return self._frame_count
    
    def get_frame_count(self) -> int:
        """Get number of frames."""
        return self._frame_count
    
    def get_gt_objects(self, frame_idx: int) -> List[GTObject]:
        """
        Get precomputed GT objects for a frame.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            List of GTObject instances
        """
        if 0 <= frame_idx < self._frame_count:
            return self._gt_objects[frame_idx]
        return []
    
    def get_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get RGB image (BGR format)."""
        if frame_idx in self._rgb_cache:
            return self._rgb_cache[frame_idx]
        # Load on demand if not preloaded
        paths = self._get_frame_paths(frame_idx)
        if paths['rgb'].exists():
            return cv2.imread(str(paths['rgb']), cv2.IMREAD_COLOR)
        return None
    
    def load_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        """Alias for get_rgb (backward compatibility)."""
        return self.get_rgb(frame_idx)
    
    def get_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get depth map in meters."""
        if frame_idx in self._depth_cache:
            return self._depth_cache[frame_idx]
        # Load on demand if not preloaded
        paths = self._get_frame_paths(frame_idx)
        if paths['depth'].exists():
            depth_raw = np.array(Image.open(paths['depth']))
            depth_m = depth_raw.astype(np.float32) * self.png_depth_scale + self.min_depth
            depth_m[depth_m < self.min_depth] = 0.0
            depth_m[depth_m > self.max_depth] = 0.0
            return depth_m
        return None
    
    def load_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """Alias for get_depth (backward compatibility)."""
        return self.get_depth(frame_idx)
    
    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get camera pose (4x4 camera-to-world transform)."""
        if 0 <= frame_idx < len(self._poses):
            return self._poses[frame_idx]
        return None
    
    def get_poses(self) -> List[np.ndarray]:
        """Get all camera poses."""
        return self._poses
    
    def get_segmentation(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Dict]:
        """Get segmentation map and info."""
        seg_info = self._seg_info[frame_idx] if frame_idx < len(self._seg_info) else {}
        
        if frame_idx in self._seg_cache:
            return self._seg_cache[frame_idx], seg_info
        
        # Load on demand
        paths = self._get_frame_paths(frame_idx)
        if paths['seg_png'].exists():
            seg_map = cv2.imread(str(paths['seg_png']), cv2.IMREAD_COLOR)
            return seg_map, seg_info
        return None, seg_info
    
    def load_segmentation(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Dict]:
        """Alias for get_segmentation (backward compatibility)."""
        return self.get_segmentation(frame_idx)
    
    def get_bboxes(self, frame_idx: int) -> Dict[str, List[Dict]]:
        """Get raw bbox data for a frame."""
        return {
            'bbox_2d': self._bboxes_2d[frame_idx] if frame_idx < len(self._bboxes_2d) else [],
            'bbox_3d': self._bboxes_3d[frame_idx] if frame_idx < len(self._bboxes_3d) else [],
        }
    
    def load_bboxes(self, frame_idx: int) -> Dict[str, List[Dict]]:
        """Alias for get_bboxes (backward compatibility)."""
        return self.get_bboxes(frame_idx)
    
    @property
    def intrinsics(self) -> Dict[str, float]:
        """Camera intrinsics."""
        return {
            'fx': self.fx, 'fy': self.fy,
            'cx': self.cx, 'cy': self.cy,
            'width': self.image_width,
            'height': self.image_height,
        }
    
    @property
    def rgb_dir(self) -> Path:
        """RGB directory path."""
        return self._rgb_dir
    
    @rgb_dir.setter
    def rgb_dir(self, value):
        self._rgb_dir = value
    
    @property
    def depth_dir(self) -> Path:
        """Depth directory path."""
        return self._depth_dir
    
    @depth_dir.setter
    def depth_dir(self, value):
        self._depth_dir = value
    
    # =========================================================================
    # STATISTICS AND INFO
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_objects = sum(len(objs) for objs in self._gt_objects)
        objects_per_frame = [len(objs) for objs in self._gt_objects]
        
        # Class distribution
        class_counts = {}
        for objs in self._gt_objects:
            for obj in objs:
                cls = obj.class_name
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Unique track IDs
        all_track_ids = set()
        for objs in self._gt_objects:
            for obj in objs:
                all_track_ids.add(obj.track_id)
        
        return {
            'scene_path': str(self.scene_path),
            'num_frames': self._frame_count,
            'num_poses': len(self._poses),
            'total_gt_objects': total_objects,
            'unique_track_ids': len(all_track_ids),
            'avg_objects_per_frame': total_objects / max(1, self._frame_count),
            'min_objects_per_frame': min(objects_per_frame) if objects_per_frame else 0,
            'max_objects_per_frame': max(objects_per_frame) if objects_per_frame else 0,
            'class_distribution': class_counts,
            'skip_labels': list(self.skip_labels),
        }
    
    def print_info(self):
        """Print dataset summary."""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print(f"Isaac Sim Dataset: {self.scene_path.name}")
        print(f"{'='*60}")
        print(f"  Frames: {stats['num_frames']}")
        print(f"  Poses: {stats['num_poses']}")
        print(f"  Total GT objects: {stats['total_gt_objects']}")
        print(f"  Unique track IDs: {stats['unique_track_ids']}")
        print(f"  Objects/frame: {stats['avg_objects_per_frame']:.1f} avg "
              f"({stats['min_objects_per_frame']}-{stats['max_objects_per_frame']} range)")
        
        print(f"\n  Class distribution:")
        for cls, count in sorted(stats['class_distribution'].items(), key=lambda x: -x[1]):
            print(f"    {cls}: {count}")
        
        print(f"\n  Skip labels: {stats['skip_labels']}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize_frame(
        self,
        frame_idx: int,
        show_rgb: bool = True,
        show_masks: bool = True,
        show_bboxes: bool = True,
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (20, 8),
        save_path: Optional[str] = None,
    ):
        """Visualize a frame with GT objects."""
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return
        
        rgb = self.get_rgb(frame_idx)
        gt_objects = self.get_gt_objects(frame_idx)
        
        if rgb is None:
            print(f"No RGB for frame {frame_idx}")
            return
        
        # Create panels
        n_panels = 1 + int(show_masks)
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]
        
        # Panel 1: RGB with bboxes
        img = cv2.cvtColor(rgb.copy(), cv2.COLOR_BGR2RGB)
        overlay = img.copy()
        
        for obj in gt_objects:
            # Generate color from track_id
            np.random.seed(obj.track_id * 7 + 13)
            color = tuple(int(c) for c in np.random.randint(50, 255, 3))
            
            # Draw mask overlay
            if show_masks and obj.mask is not None:
                mask = obj.mask > 0
                overlay[mask] = (overlay[mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
            
            # Draw bbox
            if show_bboxes and obj.bbox_2d is not None:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{obj.class_name} (ID:{obj.track_id})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        axes[0].imshow(overlay)
        axes[0].set_title(f"Frame {frame_idx} - {len(gt_objects)} GT Objects")
        axes[0].axis('off')
        
        # Panel 2: Pure masks
        if show_masks and n_panels > 1:
            mask_vis = np.zeros_like(img)
            for obj in gt_objects:
                if obj.mask is not None:
                    np.random.seed(obj.track_id * 7 + 13)
                    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
                    mask = obj.mask > 0
                    mask_vis[mask] = color
            
            axes[1].imshow(mask_vis)
            axes[1].set_title("Segmentation Masks")
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()


# ============================================================================
# Convenience function
# ============================================================================

def load_scene(scene_path: str, **kwargs) -> IsaacSimDataLoader:
    """Load a scene dataset."""
    return IsaacSimDataLoader(scene_path, **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python isaac_sim_loader_v2.py <scene_path> [--info] [--vis <frame>]")
        sys.exit(1)
    
    scene_path = sys.argv[1]
    
    # Load dataset
    loader = IsaacSimDataLoader(scene_path, verbose=True)
    
    # Parse args
    if '--info' in sys.argv:
        loader.print_info()
    
    if '--vis' in sys.argv:
        idx = sys.argv.index('--vis')
        frame = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 0
        loader.visualize_frame(frame)
    
    # Default: print info
    if len(sys.argv) == 2:
        loader.print_info()
        if len(loader) > 0:
            print("First frame GT objects:")
            for obj in loader.get_gt_objects(0):
                print(f"  - {obj.track_id}: {obj.class_name} "
                      f"(mask: {'yes' if obj.mask is not None else 'no'})")
