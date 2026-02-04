"""
Isaac Sim Replicator Dataset Loader.

A reusable data loader for Isaac Sim benchmark datasets with optional visualization.
Supports loading RGB, depth, segmentation, 2D/3D bounding boxes, and camera poses.

Usage:
    loader = IsaacSimDataLoader(scene_path)
    
    # Get frame count
    n_frames = loader.get_frame_count()
    
    # Load individual components
    rgb = loader.load_rgb(frame_idx)
    depth = loader.load_depth(frame_idx)
    seg_map, seg_info = loader.load_segmentation(frame_idx)
    bboxes = loader.load_bboxes(frame_idx)
    poses = loader.get_poses()
    
    # Get all GT objects for a frame
    gt_objects = loader.get_gt_objects(frame_idx)
    
    # Visualization (optional)
    loader.visualize_frame(frame_idx, show_rgb=True, show_seg=True, show_bbox_2d=True)
    loader.visualize_3d(frame_idx)  # 3D point cloud with bboxes
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import cv2

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
# Default camera intrinsics (can be overridden in constructor)
# ============================================================================
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 720
DEFAULT_FOCAL_LENGTH = 50.0
DEFAULT_HORIZONTAL_APERTURE = 80.0
DEFAULT_VERTICAL_APERTURE = 45.0
DEFAULT_MIN_DEPTH = 0.01
DEFAULT_MAX_DEPTH = 10.0
DEFAULT_PNG_MAX_VALUE = 65535


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
    color_bgr: Optional[Tuple[int, int, int]] = None  # Segmentation color


@dataclass
class FrameData:
    """Container for all data from a single frame."""
    frame_idx: int
    rgb: Optional[np.ndarray] = None  # (H, W, 3) BGR
    depth: Optional[np.ndarray] = None  # (H, W) float32 in meters
    segmentation: Optional[np.ndarray] = None  # (H, W, 3) BGR color-coded
    seg_info: Dict = field(default_factory=dict)  # semantic_id -> {class, color_bgr}
    gt_objects: List[GTObject] = field(default_factory=list)
    pose: Optional[np.ndarray] = None  # 4x4 camera-to-world transform


class IsaacSimDataLoader:
    """
    Data loader for Isaac Sim Replicator benchmark datasets.
    
    Expected directory structure:
        scene_path/
            rgb/frame000001.jpg, frame000002.jpg, ...
            depth/depth000001.png, depth000002.png, ...
            seg/semantic000001.png, semantic000001_info.json, ...
            bbox/bboxes000001_info.json, ...  (or bbox000001.json)
            traj.txt  (optional camera poses)
    
    Args:
        scene_path: Path to scene directory
        image_width: Image width (default 1280)
        image_height: Image height (default 720)
        focal_length: Camera focal length (default 50.0)
        horizontal_aperture: Camera horizontal aperture (default 80.0)
        vertical_aperture: Camera vertical aperture (default 45.0)
        min_depth: Minimum valid depth in meters (default 0.01)
        max_depth: Maximum valid depth in meters (default 10.0)
        skip_labels: List of labels to skip (e.g., ['background', 'wall', 'floor'])
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
    ):
        self.scene_path = Path(scene_path)
        
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
        
        # Skip labels - these should match the pipeline's skip_classes for fair comparison
        DEFAULT_SKIP_LABELS = [
            # Background/unlabeled
            'background', 'unlabelled', 'unlabeled', 'unknown',
            
            # Structural elements
            'wall', 'floor', 'ground', 'roof'
        ]
        self.skip_labels = set(l.lower() for l in (skip_labels if skip_labels is not None else DEFAULT_SKIP_LABELS))
        
        # Directory paths
        self.rgb_dir = self.scene_path / "rgb"
        self.depth_dir = self.scene_path / "depth"
        self.seg_dir = self.scene_path / "seg"
        self.bbox_dir = self.scene_path / "bbox"
        self.traj_path = self.scene_path / "traj.txt"
        
        # Cache
        self._frame_count = None
        self._poses = None
        self._rgb_files = None
        self._depth_files = None
        
        # Detect file naming convention
        self._detect_file_format()
    
    def _detect_file_format(self):
        """Detect the file naming convention used in the dataset."""
        # Check RGB naming
        self._rgb_pattern = "frame{:06d}.jpg"
        self._depth_pattern = "depth{:06d}.png"
        self._seg_pattern = "semantic{:06d}.png"
        self._seg_info_pattern = "semantic{:06d}_info.json"
        self._bbox_pattern = "bboxes{:06d}_info.json"  # Default pattern
        
        # Check if alternative bbox pattern exists
        if self.bbox_dir.exists():
            sample_files = list(self.bbox_dir.glob("*.json"))
            if sample_files:
                sample_name = sample_files[0].name
                if sample_name.startswith("bbox") and not sample_name.startswith("bboxes"):
                    self._bbox_pattern = "bbox{:06d}.json"
    
    @property
    def intrinsics(self) -> Dict[str, float]:
        """Get camera intrinsics as dictionary."""
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'width': self.image_width,
            'height': self.image_height,
        }
    
    def get_frame_count(self) -> int:
        """Get number of frames in the scene."""
        if self._frame_count is None:
            if self.rgb_dir.exists():
                self._rgb_files = sorted(list(self.rgb_dir.glob("frame*.jpg")))
                self._frame_count = len(self._rgb_files)
            else:
                self._frame_count = 0
        return self._frame_count
    
    def get_poses(self) -> List[np.ndarray]:
        """
        Load camera poses from traj.txt.
        
        Returns:
            List of 4x4 camera-to-world transformation matrices
        """
        if self._poses is None:
            self._poses = []
            if self.traj_path.exists():
                with open(self.traj_path, 'r') as f:
                    for line in f:
                        vals = line.strip().split()
                        if len(vals) == 16:
                            T = np.array([float(v) for v in vals]).reshape(4, 4)
                            self._poses.append(T)
        return self._poses
    
    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get camera pose for a specific frame."""
        poses = self.get_poses()
        if poses and 0 <= frame_idx < len(poses):
            return poses[frame_idx]
        return None
    
    def get_frame_paths(self, frame_idx: int) -> Dict[str, Path]:
        """Get all file paths for a specific frame."""
        frame_num = frame_idx + 1  # 1-indexed in filenames
        return {
            'rgb': self.rgb_dir / self._rgb_pattern.format(frame_num),
            'depth': self.depth_dir / self._depth_pattern.format(frame_num),
            'seg_png': self.seg_dir / self._seg_pattern.format(frame_num),
            'seg_json': self.seg_dir / self._seg_info_pattern.format(frame_num),
            'bbox_json': self.bbox_dir / self._bbox_pattern.format(frame_num),
        }
    
    def load_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load RGB image for a frame.
        
        Returns:
            (H, W, 3) BGR image or None if not found
        """
        paths = self.get_frame_paths(frame_idx)
        rgb_path = paths['rgb']
        
        if not rgb_path.exists():
            return None
        
        return cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    
    def load_depth(self, frame_idx: int, as_meters: bool = True) -> Optional[np.ndarray]:
        """
        Load depth map for a frame.
        
        Args:
            frame_idx: Frame index
            as_meters: If True, convert to meters; if False, return raw uint16
            
        Returns:
            (H, W) depth map (float32 in meters or uint16 raw)
        """
        paths = self.get_frame_paths(frame_idx)
        depth_path = paths['depth']
        
        if not depth_path.exists():
            return None
        
        depth_raw = np.array(Image.open(depth_path))
        
        if not as_meters:
            return depth_raw
        
        # Convert to meters
        depth_m = depth_raw.astype(np.float32) * self.png_depth_scale + self.min_depth
        
        # Clamp invalid values
        depth_m[depth_m < self.min_depth] = 0.0
        depth_m[depth_m > self.max_depth] = 0.0
        
        return depth_m
    
    def load_segmentation(self, frame_idx: int) -> Tuple[Optional[np.ndarray], Dict]:
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
        
        seg_map = cv2.imread(str(seg_png_path), cv2.IMREAD_COLOR)
        
        # Load segmentation info
        seg_json_path = paths['seg_json']
        seg_info = {}
        
        if seg_json_path.exists():
            with open(seg_json_path, 'r') as f:
                raw_info = json.load(f)
                for sem_id_str, info in raw_info.items():
                    try:
                        sem_id = int(sem_id_str)
                    except ValueError:
                        continue
                    
                    # Handle different label formats
                    label_data = info.get('label', {})
                    if isinstance(label_data, str):
                        cls_name = label_data
                    elif isinstance(label_data, dict):
                        cls_name = label_data.get('class', label_data.get('name', ''))
                        if not cls_name:
                            for k, v in label_data.items():
                                cls_name = str(v) if v else str(k)
                                break
                    else:
                        cls_name = str(label_data) if label_data else 'unknown'
                    
                    seg_info[sem_id] = {
                        'class': cls_name,
                        'color_bgr': tuple(info.get('color_bgr', [0, 0, 0]))
                    }
        
        return seg_map, seg_info
    
    def load_bboxes(self, frame_idx: int) -> Dict[str, List[Dict]]:
        """
        Load 2D and 3D bounding boxes.
        
        Returns:
            Dict with 'bbox_2d' and 'bbox_3d' lists
        """
        paths = self.get_frame_paths(frame_idx)
        bbox_path = paths['bbox_json']
        
        if not bbox_path.exists():
            return {'bbox_2d': [], 'bbox_3d': []}
        
        with open(bbox_path, 'r') as f:
            data = json.load(f)
        
        result = {'bbox_2d': [], 'bbox_3d': []}
        
        # Parse 2D bboxes
        bbox_2d_key = None
        if 'bboxes' in data:
            for key in ['bbox_2d_tight', 'bbox_2d_loose', 'bbox_2d']:
                if key in data['bboxes']:
                    bbox_2d_key = key
                    break
        
        if bbox_2d_key and 'bboxes' in data:
            for box in data['bboxes'][bbox_2d_key].get('boxes', []):
                result['bbox_2d'].append({
                    'track_id': box.get('bbox_id', box.get('track_id', -1)),
                    'semantic_id': box.get('semantic_id', -1),
                    'prim_path': box.get('prim_path', ''),
                    'label': self._extract_label(box.get('label', {})),
                    'xyxy': box.get('xyxy', [0, 0, 0, 0]),
                    'visibility': 1.0 - box.get('visibility_or_occlusion', 
                                                box.get('occlusion', 0.0))
                })
        
        # Parse 3D bboxes
        if 'bboxes' in data and 'bbox_3d' in data['bboxes']:
            for box in data['bboxes']['bbox_3d'].get('boxes', []):
                transform = box.get('transform_4x4', None)
                if transform is not None:
                    transform = np.array(transform, dtype=np.float64).reshape(4, 4)
                
                result['bbox_3d'].append({
                    'track_id': box.get('track_id', box.get('bbox_id', -1)),
                    'semantic_id': box.get('semantic_id', -1),
                    'prim_path': box.get('prim_path', ''),
                    'label': box.get('label', 'unknown'),
                    'aabb': box.get('aabb_xyzmin_xyzmax', []),
                    'transform': transform,
                    'occlusion': box.get('occlusion_ratio', 0.0)
                })
        
        return result
    
    def _extract_label(self, label_dict) -> str:
        """Extract class label from various label formats."""
        if isinstance(label_dict, str):
            return label_dict
        if isinstance(label_dict, dict):
            for k, v in label_dict.items():
                return str(v) if v else str(k)
        return 'unknown'
    
    def get_gt_objects(self, frame_idx: int, 
                       include_masks: bool = True,
                       apply_filter: bool = True,
                       debug_visualize: bool = False) -> List[GTObject]:
        """
        Get all ground truth objects for a frame.
        
        Args:
            frame_idx: Frame index
            include_masks: Whether to extract masks from segmentation
            apply_filter: Whether to filter out skip_labels (default True)
            debug_visualize: Whether to show debug visualization (RGB + masks + class names)
            
        Returns:
            List of GTObject instances
        """
        # Load bboxes
        bboxes = self.load_bboxes(frame_idx)
        
        # Load RGB for debug visualization
        rgb_debug = None
        if debug_visualize:
            rgb_debug = self.load_rgb(frame_idx)
        
        # Load segmentation if needed
        seg_map, seg_info = None, {}
        if include_masks:
            seg_map, seg_info = self.load_segmentation(frame_idx)
        
        # Build mappings
        prim_to_bbox_2d = {box['prim_path']: box for box in bboxes['bbox_2d']}
        
        gt_objects = []
        skipped_labels = []
        
        # Extract objects from 3D bboxes (more reliable track_ids)
        for box_3d in bboxes['bbox_3d']:
            track_id = box_3d['track_id']
            label = box_3d['label']
            prim_path = box_3d['prim_path']
            
            # Skip unwanted labels if filtering is enabled
            if apply_filter:
                to_continue = False
                for l in self.skip_labels:
                    if l in label.lower():
                        print(f"DEBUG[IsaacSimDataLoader.get_gt_objects] fr={frame_idx} SKIPPING object with label '{label}' (track_id={track_id})")
                        print(f"DEBUG[IsaacSimDataLoader.get_gt_objects] fr={frame_idx} l({l}) in label({label.lower()})")
                        skipped_labels.append(label)
                        to_continue = True
                        continue
                if to_continue:
                    continue
                print(f"DEBUG[IsaacSimDataLoader.get_gt_objects] fr={frame_idx} NOT SKIPPING object with label '{label}' (track_id={track_id})")

            # Find corresponding 2D bbox
            box_2d = prim_to_bbox_2d.get(prim_path, {})
            
            # Extract mask
            mask = None
            color_bgr = None
            if include_masks and seg_map is not None:
                mask, color_bgr = self._extract_mask_for_object(
                    seg_map, seg_info, box_2d, box_3d
                )
            
            gt_obj = GTObject(
                track_id=track_id,
                class_name=label,
                semantic_id=box_3d.get('semantic_id', -1),
                mask=mask,
                bbox_2d=box_2d.get('xyxy'),
                bbox_3d_aabb=box_3d.get('aabb'),
                bbox_3d_transform=box_3d.get('transform'),
                prim_path=prim_path,
                visibility=box_2d.get('visibility', 1.0),
                occlusion=box_3d.get('occlusion', 0.0),
                color_bgr=color_bgr,
            )
            gt_objects.append(gt_obj)
        print(f"DEBUG[IsaacSimDataLoader.get_gt_objects] returning gt_object length {len(gt_objects)}")
        
        # Debug visualization: RGB + semantic masks + class names
        if debug_visualize and rgb_debug is not None:
            self._visualize_gt_objects_debug(rgb_debug, gt_objects, frame_idx)
        
        return gt_objects
    
    def _visualize_gt_objects_debug(
        self,
        rgb: np.ndarray,
        gt_objects: List[GTObject],
        frame_idx: int,
        alpha: float = 0.5,
        window_name: str = "GT Objects Debug"
    ) -> None:
        """
        Debug visualization showing RGB + semantic masks + class names.
        
        Args:
            rgb: BGR image (H, W, 3)
            gt_objects: List of GTObject instances
            frame_idx: Frame index for title
            alpha: Mask overlay alpha (0-1)
            window_name: OpenCV window name
        """
        if rgb is None:
            print("DEBUG: No RGB image available for visualization")
            return
        
        vis_img = rgb.copy()
        mask_overlay = np.zeros_like(rgb)
        
        for obj in gt_objects:
            # Generate consistent color based on track_id
            np.random.seed(obj.track_id)
            color = tuple(int(c) for c in np.random.randint(50, 255, 3))
            
            # Draw mask if available
            if obj.mask is not None:
                mask_bool = obj.mask > 0
                mask_overlay[mask_bool] = color
            
            # Draw 2D bbox if available
            if obj.bbox_2d is not None:
                x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw class name label
                label_text = f"{obj.class_name} (ID:{obj.track_id})"
                (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Background rectangle for text
                cv2.rectangle(vis_img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(vis_img, label_text, (x1 + 2, y1 - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            elif obj.mask is not None:
                # If no bbox, place label at mask centroid
                mask_points = np.where(obj.mask > 0)
                if len(mask_points[0]) > 0:
                    cy, cx = int(np.mean(mask_points[0])), int(np.mean(mask_points[1]))
                    label_text = f"{obj.class_name} (ID:{obj.track_id})"
                    cv2.putText(vis_img, label_text, (cx - 30, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Blend mask overlay with original image
        vis_img = cv2.addWeighted(vis_img, 1.0, mask_overlay, alpha, 0)
        
        # Add frame info
        info_text = f"Frame {frame_idx} | {len(gt_objects)} objects"
        cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show image
        cv2.imshow(window_name, vis_img)
        print(f"DEBUG[Visualization] Press any key to continue, 'q' to quit visualization...")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
    
    def _extract_mask_for_object(
        self, 
        seg_map: np.ndarray, 
        seg_info: Dict,
        box_2d: Dict, 
        box_3d: Dict
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int]]]:
        """
        Extract binary mask for an object from segmentation map.
        
        Returns:
            mask: Binary mask (H, W) with values 0/255
            color_bgr: The segmentation color used
        """
        if seg_map is None:
            return None, None
        
        H, W = seg_map.shape[:2]
        
        # Get 2D bbox
        xyxy = box_2d.get('xyxy', [0, 0, W, H])
        if xyxy is None:
            xyxy = [0, 0, W, H]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None
        
        # Crop region
        crop = seg_map[y1:y2, x1:x2]
        
        # Find unique colors in crop (excluding background black)
        colors = crop.reshape(-1, 3)
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
        
        # Filter out background (black) and find dominant color
        best_color = None
        best_count = 0
        
        for color, count in zip(unique_colors, counts):
            if np.all(color == 0):
                continue
            if count > best_count:
                best_count = count
                best_color = color
        
        if best_color is None:
            return None, None
        
        # Create full mask from this color
        mask = np.all(seg_map == best_color, axis=2).astype(np.uint8) * 255
        
        return mask, tuple(best_color.tolist())
    
    def get_frame_data(self, frame_idx: int, 
                       load_rgb: bool = True,
                       load_depth: bool = True,
                       load_seg: bool = True,
                       load_objects: bool = True) -> FrameData:
        """
        Load all data for a single frame.
        
        Args:
            frame_idx: Frame index
            load_rgb: Whether to load RGB image
            load_depth: Whether to load depth map
            load_seg: Whether to load segmentation
            load_objects: Whether to extract GT objects
            
        Returns:
            FrameData container with requested data
        """
        data = FrameData(frame_idx=frame_idx)
        
        if load_rgb:
            data.rgb = self.load_rgb(frame_idx)
        
        if load_depth:
            data.depth = self.load_depth(frame_idx)
        
        if load_seg:
            data.segmentation, data.seg_info = self.load_segmentation(frame_idx)
        
        if load_objects:
            data.gt_objects = self.get_gt_objects(frame_idx, include_masks=load_seg)
        
        data.pose = self.get_pose(frame_idx)
        
        return data
    
    def __len__(self) -> int:
        """Return number of frames."""
        return self.get_frame_count()
    
    def __iter__(self):
        """Iterate over frames."""
        for i in range(len(self)):
            yield self.get_frame_data(i)
    
    # =========================================================================
    # Visualization Methods
    # =========================================================================
    
    def visualize_frame(
        self,
        frame_idx: int,
        show_rgb: bool = True,
        show_seg: bool = True,
        show_bbox_2d: bool = True,
        show_depth: bool = False,
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize 2D data for a frame.
        
        Args:
            frame_idx: Frame index
            show_rgb: Show RGB image
            show_seg: Show segmentation overlay
            show_bbox_2d: Show 2D bounding boxes
            show_depth: Show depth map
            alpha: Segmentation overlay alpha
            figsize: Figure size
            save_path: If provided, save figure to this path
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available for visualization")
            return
        
        data = self.get_frame_data(frame_idx)
        
        # Count panels
        n_panels = sum([show_rgb, show_seg, show_depth])
        if n_panels == 0:
            print("No visualization options selected")
            return
        
        fig, axes = plt.subplots(1, n_panels, figsize=figsize)
        if n_panels == 1:
            axes = [axes]
        
        ax_idx = 0
        
        # RGB with bboxes
        if show_rgb and data.rgb is not None:
            ax = axes[ax_idx]
            img = cv2.cvtColor(data.rgb.copy(), cv2.COLOR_BGR2RGB)
            
            if show_bbox_2d:
                for obj in data.gt_objects:
                    if obj.bbox_2d is None:
                        continue
                    
                    x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                    
                    # Random color based on track_id
                    np.random.seed(obj.track_id)
                    color = tuple(int(c) for c in np.random.randint(0, 255, 3))
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label
                    text = f"{obj.class_name} (ID:{obj.track_id})"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                    cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
            
            ax.imshow(img)
            ax.set_title(f'RGB + 2D BBoxes - Frame {frame_idx} ({len(data.gt_objects)} objects)')
            ax.axis('off')
            ax_idx += 1
        
        # Segmentation overlay
        if show_seg and data.rgb is not None and data.segmentation is not None:
            ax = axes[ax_idx]
            rgb = cv2.cvtColor(data.rgb, cv2.COLOR_BGR2RGB)
            seg = cv2.cvtColor(data.segmentation, cv2.COLOR_BGR2RGB)
            
            # Blend
            blended = cv2.addWeighted(rgb, 1 - alpha, seg, alpha, 0)
            
            # Add annotations
            for obj in data.gt_objects:
                if obj.bbox_2d is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                text = f"{obj.class_name}"
                cv2.putText(blended, text, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            ax.imshow(blended)
            ax.set_title(f'RGB + Segmentation - Frame {frame_idx}')
            ax.axis('off')
            ax_idx += 1
        
        # Depth
        if show_depth and data.depth is not None:
            ax = axes[ax_idx]
            depth_vis = data.depth.copy()
            depth_vis[depth_vis <= 0] = np.nan
            
            im = ax.imshow(depth_vis, cmap='viridis')
            ax.set_title(f'Depth - Frame {frame_idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Depth (m)', fraction=0.046)
            ax_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
    
    def visualize_3d(
        self,
        frame_idx: int,
        show_point_cloud: bool = True,
        show_bbox_3d: bool = True,
        use_pose: bool = True,
        point_size: float = 2.0,
        max_box_edge: float = 20.0,
        window_name: Optional[str] = None,
    ) -> None:
        """
        Visualize 3D data for a frame using Open3D.
        
        Args:
            frame_idx: Frame index
            show_point_cloud: Show reconstructed point cloud
            show_bbox_3d: Show 3D bounding boxes
            use_pose: Transform point cloud to world frame
            point_size: Point size for visualization
            max_box_edge: Skip boxes larger than this (meters)
            window_name: Window title
        """
        if not HAS_OPEN3D:
            print("Open3D not available for 3D visualization")
            return
        
        data = self.get_frame_data(frame_idx)
        
        if data.rgb is None or data.depth is None:
            print(f"Missing RGB or depth for frame {frame_idx}")
            return
        
        # Create RGBD image
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(data.rgb, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(data.depth.astype(np.float32))
        
        # Create intrinsics
        intrinsics = o3d.camera.PinholeCameraIntrinsic()
        intrinsics.set_intrinsics(
            self.image_width, self.image_height,
            self.fx, self.fy, self.cx, self.cy
        )
        
        # Create RGBD and point cloud
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0,
            depth_trunc=self.max_depth,
            convert_rgb_to_intensity=False,
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        
        # Transform to world frame
        if use_pose and data.pose is not None:
            pcd.transform(data.pose)
        
        # Prepare geometries
        geometries = []
        
        if show_point_cloud:
            geometries.append(pcd)
        
        # Coordinate frame
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        geometries.append(coord)
        
        # 3D bounding boxes
        if show_bbox_3d:
            for obj in data.gt_objects:
                if obj.bbox_3d_aabb is None or len(obj.bbox_3d_aabb) != 6:
                    continue
                
                aabb = obj.bbox_3d_aabb
                xmin, ymin, zmin, xmax, ymax, zmax = aabb
                
                # Skip very large boxes
                sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin
                if max(sx, sy, sz) > max_box_edge:
                    continue
                
                # Create line set for bbox
                corners = np.array([
                    [xmin, ymin, zmin],
                    [xmax, ymin, zmin],
                    [xmax, ymax, zmin],
                    [xmin, ymax, zmin],
                    [xmin, ymin, zmax],
                    [xmax, ymin, zmax],
                    [xmax, ymax, zmax],
                    [xmin, ymax, zmax],
                ], dtype=np.float64)
                
                lines = np.array([
                    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
                    [4, 5], [5, 6], [6, 7], [7, 4],  # top
                    [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
                ], dtype=np.int32)
                
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(corners)
                ls.lines = o3d.utility.Vector2iVector(lines)
                
                # Color based on track_id
                np.random.seed(obj.track_id)
                color = np.random.rand(3).tolist()
                ls.paint_uniform_color(color)
                
                geometries.append(ls)
        
        # Visualize
        window_title = window_name or f"3D Visualization - Frame {frame_idx}"
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_title,
            width=1280,
            height=720,
            point_show_normal=False,
        )
    
    def visualize_sequence(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        mode: str = '2d',
        **kwargs
    ) -> None:
        """
        Visualize a sequence of frames.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive)
            step: Frame step
            mode: '2d' for 2D visualization, '3d' for 3D
            **kwargs: Additional arguments for visualize_frame or visualize_3d
        """
        if end_frame is None:
            end_frame = len(self)
        
        for frame_idx in range(start_frame, end_frame, step):
            print(f"\nFrame {frame_idx}/{end_frame-1}")
            
            if mode == '2d':
                self.visualize_frame(frame_idx, **kwargs)
            elif mode == '3d':
                self.visualize_3d(frame_idx, **kwargs)
            else:
                raise ValueError(f"Unknown mode: {mode}")
    
    def print_info(self) -> None:
        """Print dataset information."""
        print(f"\n{'='*60}")
        print(f"Isaac Sim Dataset Info")
        print(f"{'='*60}")
        print(f"  Scene path: {self.scene_path}")
        print(f"  Frames: {self.get_frame_count()}")
        print(f"  Poses: {len(self.get_poses())}")
        print(f"\n  Camera intrinsics:")
        print(f"    Image: {self.image_width} x {self.image_height}")
        print(f"    fx, fy: {self.fx:.1f}, {self.fy:.1f}")
        print(f"    cx, cy: {self.cx:.1f}, {self.cy:.1f}")
        print(f"  Depth range: [{self.min_depth}, {self.max_depth}] m")
        
        # Check folders
        print(f"\n  Directories:")
        for name, path in [('RGB', self.rgb_dir), ('Depth', self.depth_dir),
                          ('Segmentation', self.seg_dir), ('Bboxes', self.bbox_dir)]:
            exists = path.exists()
            n_files = len(list(path.glob('*'))) if exists else 0
            status = f"✓ {n_files} files" if exists else "✗ not found"
            print(f"    {name:15s}: {status}")
        
        # Show skip labels info
        print(f"\n  Skip labels: {len(self.skip_labels)} patterns")
        sample_labels = sorted(list(self.skip_labels))[:8]
        print(f"    Examples: {', '.join(sample_labels)}...")
        
        # Sample first frame objects (show filtered vs unfiltered)
        if len(self) > 0:
            objects_all = self.get_gt_objects(0, include_masks=False, apply_filter=False)
            objects_filtered = self.get_gt_objects(0, include_masks=False, apply_filter=True)
            
            print(f"\n  Objects in first frame:")
            print(f"    Total (unfiltered): {len(objects_all)}")
            print(f"    After filtering:    {len(objects_filtered)}")
            print(f"    Filtered out:       {len(objects_all) - len(objects_filtered)}")
            
            # Show what was filtered
            filtered_classes = set()
            for obj in objects_all:
                if obj.class_name.lower() in self.skip_labels:
                    filtered_classes.add(obj.class_name)
            if filtered_classes:
                print(f"    Filtered classes:   {', '.join(sorted(filtered_classes))}")
            
            # Show remaining class distribution
            class_counts = {}
            for obj in objects_filtered:
                class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
            if class_counts:
                print(f"\n    Remaining classes:")
                for cls, count in sorted(class_counts.items()):
                    print(f"      {cls}: {count}")
        
        print(f"{'='*60}\n")


# ============================================================================
# Convenience functions
# ============================================================================

def load_scene(scene_path: str, **kwargs) -> IsaacSimDataLoader:
    """Convenience function to create a data loader."""
    return IsaacSimDataLoader(scene_path, **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python isaac_sim_loader.py <scene_path> [frame_idx]")
        print("       python isaac_sim_loader.py <scene_path> --info")
        print("       python isaac_sim_loader.py <scene_path> --3d [frame_idx]")
        sys.exit(1)
    
    scene_path = sys.argv[1]
    loader = IsaacSimDataLoader(scene_path)
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '--info':
            loader.print_info()
        elif sys.argv[2] == '--3d':
            frame_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
            loader.visualize_3d(frame_idx)
        else:
            frame_idx = int(sys.argv[2])
            loader.visualize_frame(frame_idx, show_rgb=True, show_seg=True, 
                                   show_bbox_2d=True, show_depth=True)
    else:
        loader.print_info()
        for i in range(0, 20, 5):
            loader.visualize_frame(i, show_rgb=True, show_seg=True, 
                              show_bbox_2d=True)
            loader.visualize_3d(i)
