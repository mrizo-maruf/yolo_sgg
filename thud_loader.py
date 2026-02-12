"""
THUD (Synthetic Human/Object Tracking Dataset) Data Loader.

A data loader for THUD benchmark datasets with scene graph annotations.
Supports loading RGB, depth, instance/semantic segmentation, 2D/3D bounding boxes, and camera poses.

Dataset Structure:
    scene_path/  (e.g., Synthetic/Gym/Static/Capture_1)
        RGB/rgb_<N>.png
        Depth/depth_<N>.png
        Label/
            Instance/Instance_<N>.png
            Semantic/segmentation_<N>.png
            captures_XXX.json  (annotation files)
            annotation_info/
                annotation_definitions.json

Usage:
    loader = THUDDataLoader(scene_path)
    
    # Get frame count
    n_frames = loader.get_frame_count()
    
    # Load frame data
    rgb = loader.load_rgb(frame_idx)
    depth = loader.load_depth(frame_idx)
    
    # Get GT objects for tracking benchmark
    gt_objects = loader.get_gt_objects(frame_idx)
"""

import os
import re
import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import cv2
from collections import defaultdict

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
# Data Classes
# ============================================================================

@dataclass
class GTObject:
    """Ground truth object representation for THUD dataset."""
    track_id: int  # instance_id serves as track_id (consistent across frames)
    class_name: str
    label_id: int = -1
    mask: Optional[np.ndarray] = None  # Binary mask (H, W), 0/255
    bbox_2d: Optional[List[float]] = None  # [x1, y1, x2, y2]
    bbox_3d_center: Optional[List[float]] = None  # [x, y, z] center
    bbox_3d_size: Optional[List[float]] = None  # [sx, sy, sz] dimensions
    bbox_3d_rotation: Optional[List[float]] = None  # [qx, qy, qz, qw] quaternion
    color_rgba: Optional[Tuple[int, int, int, int]] = None  # Instance segmentation color
    visibility: float = 1.0


@dataclass
class CameraPose:
    """Camera pose information."""
    translation: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [qx, qy, qz, qw] quaternion
    intrinsic: np.ndarray  # 3x3 camera intrinsic matrix


@dataclass 
class FrameData:
    """Container for all data from a single frame."""
    frame_idx: int
    rgb: Optional[np.ndarray] = None  # (H, W, 3) RGB
    depth: Optional[np.ndarray] = None  # (H, W) float32
    instance_seg: Optional[np.ndarray] = None  # (H, W, 3) or (H, W, 4) color-coded
    semantic_seg: Optional[np.ndarray] = None  # (H, W, 3) color-coded
    gt_objects: List[GTObject] = field(default_factory=list)
    camera_pose: Optional[CameraPose] = None


class THUDDataLoader:
    """
    Data loader for THUD benchmark datasets.
    
    Expected directory structure:
        scene_path/
            RGB/rgb_<N>.png
            Depth/depth_<N>.png
            Label/
                Instance/Instance_<N>.png
                Semantic/segmentation_<N>.png
                captures_XXX.json
                annotation_info/annotation_definitions.json
    
    Note: File indices may not be synchronized across RGB, Depth, and Instance.
    The loader uses JSON captures as the source of truth for file associations.
    """
    
    def __init__(self, scene_path: str, verbose: bool = True):
        """
        Initialize THUD data loader.
        
        Args:
            scene_path: Path to scene directory (e.g., thud/Synthetic/Gym/Static/Capture_1)
            verbose: Print loading information
        """
        self.scene_path = Path(scene_path)
        self.verbose = verbose
        
        # Directory paths
        self.rgb_dir = self.scene_path / "RGB"
        self.depth_dir = self.scene_path / "Depth"
        self.label_dir = self.scene_path / "Label"
        self.instance_dir = self.label_dir / "Instance"
        self.semantic_dir = self.label_dir / "Semantic"
        self.annotation_info_dir = self.label_dir / "annotation_info"
        
        # Validate paths
        self._validate_paths()
        
        # Load annotation definitions
        self.label_definitions = self._load_annotation_definitions()
        
        # Parse capture JSON files and build frame index
        # frame_data now stores: {frame_idx: {sensor, annotations, rgb_file, depth_file, instance_file, semantic_file}}
        self.frame_data = {}  # frame_idx -> capture data with file paths
        self.frame_indices = []  # sorted list of available frame indices
        self._parse_capture_files()
        
        # Get image dimensions from first RGB image
        self.image_height, self.image_width = self._get_image_dimensions()
        
        if self.verbose:
            print(f"[THUDDataLoader] Loaded scene: {self.scene_path.name}")
            if self.frame_indices:
                print(f"  Frames: {len(self.frame_indices)} (range: {min(self.frame_indices)}-{max(self.frame_indices)})")
            else:
                print(f"  Frames: 0 (no valid frames found)")
            print(f"  Image size: {self.image_width}x{self.image_height}")
            print(f"  Label classes: {len(self.label_definitions.get('bounding_box', {}))}")
    
    def _validate_paths(self):
        """Validate that required directories exist."""
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene path not found: {self.scene_path}")
        
        required_dirs = [self.rgb_dir, self.label_dir]
        for d in required_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Required directory not found: {d}")
    
    def _load_annotation_definitions(self) -> Dict:
        """Load annotation definitions from JSON file."""
        definitions = {
            'bounding_box': {},  # label_id -> label_name
            'bounding_box_3d': {},
            'instance_segmentation': {},
            'semantic_segmentation': {},  # label_name -> pixel_value
        }
        
        ann_def_path = self.annotation_info_dir / "annotation_definitions.json"
        if not ann_def_path.exists():
            if self.verbose:
                print(f"[WARN] annotation_definitions.json not found at {ann_def_path}")
            return definitions
        
        with open(ann_def_path, 'r') as f:
            data = json.load(f)
        
        for ann_def in data.get('annotation_definitions', []):
            ann_id = ann_def.get('id', '')
            spec = ann_def.get('spec', [])
            
            if ann_id == 'bounding box':
                for item in spec:
                    definitions['bounding_box'][item['label_id']] = item['label_name']
            elif ann_id == 'bounding box 3D':
                for item in spec:
                    definitions['bounding_box_3d'][item['label_id']] = item['label_name']
            elif ann_id == 'instance segmentation':
                for item in spec:
                    definitions['instance_segmentation'][item['label_id']] = item['label_name']
            elif ann_id == 'semantic segmentation':
                for item in spec:
                    definitions['semantic_segmentation'][item['label_name']] = item.get('pixel_value', {})
        
        return definitions
    
    def _discover_existing_frames(self) -> set:
        """Discover frame indices from actual files on disk.
        
        Returns:
            Set of frame indices that have actual RGB files on disk.
        """
        existing_frames = set()
        
        # Scan RGB directory for actual files
        if self.rgb_dir.exists():
            for rgb_file in self.rgb_dir.glob("rgb_*.png"):
                match = re.search(r'rgb_(\d+)\.png', rgb_file.name, re.IGNORECASE)
                if match:
                    existing_frames.add(int(match.group(1)))
        
        return existing_frames
    
    def _parse_capture_files(self):
        """Parse all captures_XXX.json files and build frame index.
        
        Note: Each frame may have multiple capture entries in the JSON (one for 3D bbox,
        one for 2D bbox + instance segmentation, etc.). We merge all annotations for each frame.
        Only frames that have actual files on disk are included.
        """
        capture_files = sorted(self.label_dir.glob("captures_*.json"))
        
        if not capture_files:
            raise FileNotFoundError(f"No capture files found in {self.label_dir}")
        
        # First, discover what frames actually exist on disk
        existing_frames = self._discover_existing_frames()
        
        if self.verbose:
            print(f"  Found {len(existing_frames)} actual RGB files on disk")
        
        # Parse JSON annotations
        json_frame_data = {}  # frame_idx -> annotations data
        
        for capture_file in capture_files:
            with open(capture_file, 'r') as f:
                data = json.load(f)
            
            for capture in data.get('captures', []):
                # Extract frame index from filename
                filename = capture.get('filename', '')
                frame_idx = self._extract_frame_index(filename)
                
                if frame_idx is not None:
                    if frame_idx not in json_frame_data:
                        # First capture for this frame - store base data
                        json_frame_data[frame_idx] = {
                            'sensor': capture.get('sensor', {}),
                            'annotations': []
                        }
                    
                    # Merge annotations from this capture
                    annotations = capture.get('annotations', [])
                    json_frame_data[frame_idx]['annotations'].extend(annotations)
        
        # Now, only keep frames that have actual files on disk
        # If no existing_frames found (empty RGB dir), fall back to JSON frame indices
        if existing_frames:
            for frame_idx in existing_frames:
                if frame_idx in json_frame_data:
                    self.frame_data[frame_idx] = json_frame_data[frame_idx]
                else:
                    # Frame exists on disk but no JSON annotation - create empty annotation
                    self.frame_data[frame_idx] = {
                        'sensor': {},
                        'annotations': []
                    }
        else:
            # Fallback: use JSON frame indices (original behavior)
            self.frame_data = json_frame_data
        
        self.frame_indices = sorted(self.frame_data.keys())
        
        if self.verbose:
            print(f"  Parsed {len(capture_files)} capture files")
            if existing_frames:
                json_only = len(json_frame_data) - len([f for f in json_frame_data if f in existing_frames])
                if json_only > 0:
                    print(f"  [INFO] {json_only} JSON frames have no matching files on disk")
    
    def _extract_frame_index(self, filename: str) -> Optional[int]:
        """Extract frame index from filename like 'RGBxxx/rgb_2.png' or 'InstanceSegmentation.../Instance_2.png'."""
        # Try rgb pattern first
        match = re.search(r'rgb_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Try instance pattern
        match = re.search(r'Instance_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Try depth pattern
        match = re.search(r'depth_(\d+)\.png', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None
    
    def _get_image_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions from first RGB image."""
        if not self.frame_indices:
            return 720, 1280  # Default
        
        first_frame = self.frame_indices[0]
        rgb_path = self.rgb_dir / f"rgb_{first_frame}.png"
        
        if rgb_path.exists():
            img = Image.open(rgb_path)
            return img.height, img.width
        
        return 720, 1280  # Default
    
    def get_frame_count(self) -> int:
        """Get total number of available frames."""
        return len(self.frame_indices)
    
    def get_frame_indices(self) -> List[int]:
        """Get list of available frame indices."""
        return self.frame_indices.copy()
    
    def load_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load RGB image for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            RGB image as numpy array (H, W, 3) or None if not found
        """
        rgb_path = self.rgb_dir / f"rgb_{frame_idx}.png"
        
        if not rgb_path.exists():
            if self.verbose:
                print(f"[WARN] RGB not found: {rgb_path}")
            return None
        
        img = Image.open(rgb_path)
        return np.array(img)[:, :, :3]  # Ensure RGB only
    
    def load_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load depth image for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Depth map as float32 numpy array (H, W) or None if not found
        """
        depth_path = self.depth_dir / f"depth_{frame_idx}.png"
        
        if not depth_path.exists():
            if self.verbose:
                print(f"[WARN] Depth not found: {depth_path}")
            return None
        
        # Load as 16-bit PNG
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            return None
        
        # Convert to float (assuming linear depth encoding)
        # Note: Actual depth scale may need calibration
        return depth_img.astype(np.float32) / 1000.0  # Convert mm to meters (adjust as needed)
    
    def load_instance_segmentation(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load instance segmentation image.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Instance segmentation image (H, W, 3) or (H, W, 4) or None
        """
        instance_path = self.instance_dir / f"Instance_{frame_idx}.png"
        
        if not instance_path.exists():
            if self.verbose:
                print(f"[WARN] Instance seg not found: {instance_path}")
            return None
        
        img = Image.open(instance_path)
        return np.array(img)
    
    def load_semantic_segmentation(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load semantic segmentation image.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Semantic segmentation image (H, W, 3) or None
        """
        semantic_path = self.semantic_dir / f"segmentation_{frame_idx}.png"
        
        if not semantic_path.exists():
            if self.verbose:
                print(f"[WARN] Semantic seg not found: {semantic_path}")
            return None
        
        img = Image.open(semantic_path)
        return np.array(img)
    
    def get_camera_pose(self, frame_idx: int) -> Optional[CameraPose]:
        """
        Get camera pose for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            CameraPose object or None
        """
        if frame_idx not in self.frame_data:
            return None
        
        capture = self.frame_data[frame_idx]
        sensor = capture.get('sensor', {})
        
        translation = np.array(sensor.get('translation', [0, 0, 0]), dtype=np.float32)
        rotation = np.array(sensor.get('rotation', [0, 0, 0, 1]), dtype=np.float32)
        
        # Parse camera intrinsic matrix
        intrinsic_data = sensor.get('camera_intrinsic', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        intrinsic = np.array(intrinsic_data, dtype=np.float32)
        
        return CameraPose(
            translation=translation,
            rotation=rotation,
            intrinsic=intrinsic
        )
    
    def get_gt_objects(self, frame_idx: int) -> List[GTObject]:
        """
        Get all ground truth objects for a frame.
        
        This method combines 2D bounding boxes, 3D bounding boxes, and instance
        segmentation masks for each object.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of GTObject instances
        """
        if frame_idx not in self.frame_data:
            if self.verbose:
                print(f"[WARN] No frame data for frame {frame_idx}")
            return []
        
        capture = self.frame_data[frame_idx]
        annotations = capture.get('annotations', [])
        
        # Build lookup dicts by instance_id
        bbox_2d_by_id = {}  # instance_id -> bbox_2d data
        bbox_3d_by_id = {}  # instance_id -> bbox_3d data
        instance_colors = {}  # instance_id -> RGBA color
        
        for ann in annotations:
            ann_id = ann.get('id', '')
            values = ann.get('values', [])
            
            if ann_id == 'bounding box':
                for v in values:
                    inst_id = v.get('instance_id')
                    if inst_id is not None:
                        bbox_2d_by_id[inst_id] = v
            
            elif ann_id == 'bounding box 3D':
                for v in values:
                    inst_id = v.get('instance_id')
                    if inst_id is not None:
                        bbox_3d_by_id[inst_id] = v
            
            elif ann_id == 'instance segmentation':
                for v in values:
                    inst_id = v.get('instance_id')
                    color = v.get('color', {})
                    if inst_id is not None:
                        instance_colors[inst_id] = (
                            color.get('r', 0),
                            color.get('g', 0),
                            color.get('b', 0),
                            color.get('a', 255)
                        )
        
        # Load instance segmentation image for mask extraction
        instance_img = self.load_instance_segmentation(frame_idx)
        
        # Build GT objects
        gt_objects = []
        processed_ids = set()
        
        # Process all instance IDs from 2D bboxes (primary source)
        for inst_id, bbox_data in bbox_2d_by_id.items():
            if inst_id in processed_ids:
                continue
            processed_ids.add(inst_id)
            
            # Get class name and label_id
            label_id = bbox_data.get('label_id', -1)
            class_name = bbox_data.get('label_name', 'unknown')
            
            # Convert bbox from (x, y, width, height) to (x1, y1, x2, y2)
            x = bbox_data.get('x', 0)
            y = bbox_data.get('y', 0)
            w = bbox_data.get('width', 0)
            h = bbox_data.get('height', 0)
            bbox_2d = [x, y, x + w, y + h]
            
            # Get 3D bbox if available
            bbox_3d_data = bbox_3d_by_id.get(inst_id, {})
            bbox_3d_center = None
            bbox_3d_size = None
            bbox_3d_rotation = None
            
            if bbox_3d_data:
                trans = bbox_3d_data.get('translation', {})
                bbox_3d_center = [trans.get('x', 0), trans.get('y', 0), trans.get('z', 0)]
                
                size = bbox_3d_data.get('size', {})
                bbox_3d_size = [size.get('x', 0), size.get('y', 0), size.get('z', 0)]
                
                rot = bbox_3d_data.get('rotation', {})
                bbox_3d_rotation = [rot.get('x', 0), rot.get('y', 0), rot.get('z', 0), rot.get('w', 1)]
            
            # Get instance color
            color = instance_colors.get(inst_id)
            
            # Extract mask from instance segmentation image
            mask = None
            if instance_img is not None and color is not None:
                mask = self._extract_instance_mask(instance_img, color)
            
            gt_obj = GTObject(
                track_id=inst_id,
                class_name=class_name,
                label_id=label_id,
                mask=mask,
                bbox_2d=bbox_2d,
                bbox_3d_center=bbox_3d_center,
                bbox_3d_size=bbox_3d_size,
                bbox_3d_rotation=bbox_3d_rotation,
                color_rgba=color,
                visibility=1.0
            )
            gt_objects.append(gt_obj)
        
        # Also process 3D bbox objects that don't have 2D bboxes
        for inst_id, bbox_3d_data in bbox_3d_by_id.items():
            if inst_id in processed_ids:
                continue
            processed_ids.add(inst_id)
            
            label_id = bbox_3d_data.get('label_id', -1)
            class_name = bbox_3d_data.get('label_name', 'unknown')
            
            trans = bbox_3d_data.get('translation', {})
            bbox_3d_center = [trans.get('x', 0), trans.get('y', 0), trans.get('z', 0)]
            
            size = bbox_3d_data.get('size', {})
            bbox_3d_size = [size.get('x', 0), size.get('y', 0), size.get('z', 0)]
            
            rot = bbox_3d_data.get('rotation', {})
            bbox_3d_rotation = [rot.get('x', 0), rot.get('y', 0), rot.get('z', 0), rot.get('w', 1)]
            
            # Get instance color (may also be available for 3D-only objects)
            color = instance_colors.get(inst_id)
            
            # Extract mask from instance segmentation image
            mask = None
            if instance_img is not None and color is not None:
                mask = self._extract_instance_mask(instance_img, color)
            
            gt_obj = GTObject(
                track_id=inst_id,
                class_name=class_name,
                label_id=label_id,
                mask=mask,
                bbox_2d=None,  # No 2D bbox for this object
                bbox_3d_center=bbox_3d_center,
                bbox_3d_size=bbox_3d_size,
                bbox_3d_rotation=bbox_3d_rotation,
                color_rgba=color,
                visibility=1.0
            )
            gt_objects.append(gt_obj)
        
        return gt_objects
    
    def _extract_instance_mask(self, instance_img: np.ndarray, 
                               color: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract binary mask for a specific instance color.
        
        Args:
            instance_img: Instance segmentation image (H, W, 3) or (H, W, 4)
            color: RGBA color tuple
            
        Returns:
            Binary mask (H, W) with values 0 or 255
        """
        r, g, b, a = color
        
        if instance_img.ndim == 2:
            # Grayscale - shouldn't happen but handle it
            return np.zeros(instance_img.shape, dtype=np.uint8)
        
        # Check if image has alpha channel
        if instance_img.shape[2] == 4:
            # RGBA image
            mask = (
                (instance_img[:, :, 0] == r) &
                (instance_img[:, :, 1] == g) &
                (instance_img[:, :, 2] == b) &
                (instance_img[:, :, 3] == a)
            )
        else:
            # RGB image
            mask = (
                (instance_img[:, :, 0] == r) &
                (instance_img[:, :, 1] == g) &
                (instance_img[:, :, 2] == b)
            )
        
        return (mask.astype(np.uint8) * 255)
    
    def get_all_instance_ids(self) -> set:
        """
        Get all unique instance IDs across all frames.
        
        Returns:
            Set of instance IDs
        """
        all_ids = set()
        
        for frame_idx, capture in self.frame_data.items():
            for ann in capture.get('annotations', []):
                ann_id = ann.get('id', '')
                # Check all annotation types that have instance_id
                if ann_id in ('bounding box', 'bounding box 3D', 'instance segmentation'):
                    for v in ann.get('values', []):
                        inst_id = v.get('instance_id')
                        if inst_id is not None:
                            all_ids.add(inst_id)
        
        return all_ids
    
    def get_class_names(self) -> List[str]:
        """Get list of all class names in the dataset."""
        classes = set()
        for label_map in [self.label_definitions['bounding_box'],
                         self.label_definitions['bounding_box_3d'],
                         self.label_definitions['instance_segmentation']]:
            classes.update(label_map.values())
        return sorted(list(classes))
    
    def load_frame(self, frame_idx: int, 
                   load_rgb: bool = True,
                   load_depth: bool = True,
                   load_instance: bool = True,
                   load_semantic: bool = False) -> FrameData:
        """
        Load all data for a single frame.
        
        Args:
            frame_idx: Frame index
            load_rgb: Whether to load RGB image
            load_depth: Whether to load depth map
            load_instance: Whether to load instance segmentation
            load_semantic: Whether to load semantic segmentation
            
        Returns:
            FrameData object containing all requested data
        """
        frame = FrameData(frame_idx=frame_idx)
        
        if load_rgb:
            frame.rgb = self.load_rgb(frame_idx)
        
        if load_depth:
            frame.depth = self.load_depth(frame_idx)
        
        if load_instance:
            frame.instance_seg = self.load_instance_segmentation(frame_idx)
        
        if load_semantic:
            frame.semantic_seg = self.load_semantic_segmentation(frame_idx)
        
        frame.gt_objects = self.get_gt_objects(frame_idx)
        frame.camera_pose = self.get_camera_pose(frame_idx)
        
        return frame
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def visualize_frame(self, frame_idx: int,
                       show_rgb: bool = True,
                       show_instance: bool = True,
                       show_bbox_2d: bool = True,
                       show_labels: bool = True,
                       alpha: float = 0.5,
                       figsize: Tuple[int, int] = (16, 8),
                       save_path: Optional[str] = None):
        """
        Visualize a frame with annotations.
        
        Args:
            frame_idx: Frame index to visualize
            show_rgb: Show RGB image
            show_instance: Show instance segmentation overlay
            show_bbox_2d: Draw 2D bounding boxes
            show_labels: Show class labels
            alpha: Transparency for segmentation overlay
            figsize: Figure size
            save_path: Optional path to save visualization
        """
        if not HAS_MATPLOTLIB:
            print("[WARN] Matplotlib not available for visualization")
            return
        
        rgb = self.load_rgb(frame_idx)
        if rgb is None:
            print(f"[WARN] Could not load RGB for frame {frame_idx}")
            return
        
        gt_objects = self.get_gt_objects(frame_idx)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left panel: RGB with bboxes
        vis_rgb = rgb.copy()
        
        if show_bbox_2d:
            for obj in gt_objects:
                if obj.bbox_2d is not None:
                    x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                    
                    # Generate color from track_id
                    np.random.seed(obj.track_id * 7 + 13)
                    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
                    
                    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), color, 2)
                    
                    if show_labels:
                        label = f"{obj.class_name} [{obj.track_id}]"
                        cv2.putText(vis_rgb, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[0].imshow(vis_rgb)
        axes[0].set_title(f"Frame {frame_idx} - RGB + Bboxes ({len(gt_objects)} objects)")
        axes[0].axis('off')
        
        # Right panel: Instance segmentation overlay
        if show_instance:
            instance_img = self.load_instance_segmentation(frame_idx)
            if instance_img is not None:
                # Blend RGB with instance segmentation
                if instance_img.shape[2] == 4:
                    instance_rgb = instance_img[:, :, :3]
                else:
                    instance_rgb = instance_img
                vis_instance = cv2.addWeighted(rgb, 1 - alpha, instance_rgb, alpha, 0)
                axes[1].imshow(vis_instance)
            else:
                axes[1].imshow(rgb)
        else:
            axes[1].imshow(rgb)
        
        axes[1].set_title(f"Frame {frame_idx} - Instance Segmentation")
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_gt_masks(self, frame_idx: int,
                          figsize: Tuple[int, int] = (16, 10),
                          max_objects: int = 20):
        """
        Visualize individual GT object masks.
        
        Args:
            frame_idx: Frame index
            figsize: Figure size
            max_objects: Maximum number of objects to show
        """
        if not HAS_MATPLOTLIB:
            print("[WARN] Matplotlib not available")
            return
        
        rgb = self.load_rgb(frame_idx)
        gt_objects = self.get_gt_objects(frame_idx)
        
        if not gt_objects:
            print(f"[WARN] No GT objects for frame {frame_idx}")
            return
        
        # Limit number of objects
        gt_objects = gt_objects[:max_objects]
        n_objects = len(gt_objects)
        
        # Determine grid size
        cols = min(5, n_objects)
        rows = (n_objects + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, obj in enumerate(gt_objects):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]
            
            if obj.mask is not None and rgb is not None:
                # Create masked RGB visualization
                vis = rgb.copy()
                mask_bool = obj.mask > 0
                
                # Highlight mask region
                np.random.seed(obj.track_id * 7 + 13)
                color = np.random.randint(50, 255, 3)
                vis[mask_bool] = (vis[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
                
                ax.imshow(vis)
            else:
                ax.imshow(rgb if rgb is not None else np.zeros((100, 100, 3)))
            
            ax.set_title(f"{obj.class_name}\nID:{obj.track_id}", fontsize=8)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n_objects, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"GT Object Masks - Frame {frame_idx}", fontsize=12)
        plt.tight_layout()
        plt.show()


# ============================================================================
# Utility Functions
# ============================================================================

def discover_thud_scenes(thud_root: str) -> List[str]:
    """
    Discover all available THUD scenes.
    
    Args:
        thud_root: Root path to THUD dataset
        
    Returns:
        List of scene paths
    """
    thud_root = Path(thud_root)
    scenes = []
    
    # Look for Capture_* directories
    for capture_dir in thud_root.rglob("Capture_*"):
        if capture_dir.is_dir():
            # Verify it has required subdirectories
            if (capture_dir / "RGB").exists() and (capture_dir / "Label").exists():
                scenes.append(str(capture_dir))
    
    return sorted(scenes)


def get_scene_info(scene_path: str) -> Dict:
    """
    Get basic information about a THUD scene without fully loading it.
    
    Args:
        scene_path: Path to scene directory
        
    Returns:
        Dictionary with scene information
    """
    scene_path = Path(scene_path)
    
    info = {
        'path': str(scene_path),
        'name': scene_path.name,
        'parent': scene_path.parent.name,
        'rgb_count': 0,
        'depth_count': 0,
        'instance_count': 0,
        'capture_files': 0,
    }
    
    # Count files
    rgb_dir = scene_path / "RGB"
    depth_dir = scene_path / "Depth"
    instance_dir = scene_path / "Label" / "Instance"
    label_dir = scene_path / "Label"
    
    if rgb_dir.exists():
        info['rgb_count'] = len(list(rgb_dir.glob("rgb_*.png")))
    
    if depth_dir.exists():
        info['depth_count'] = len(list(depth_dir.glob("depth_*.png")))
    
    if instance_dir.exists():
        info['instance_count'] = len(list(instance_dir.glob("Instance_*.png")))
    
    if label_dir.exists():
        info['capture_files'] = len(list(label_dir.glob("captures_*.json")))
    
    return info


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Default test path
    test_path = "thud/Synthetic/Gym/Static/Capture_1"
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    
    print(f"Testing THUDDataLoader with: {test_path}")
    print("=" * 60)
    
    try:
        # Initialize loader
        loader = THUDDataLoader(test_path, verbose=True)
        
        print(f"\nFrame count: {loader.get_frame_count()}")
        print(f"Frame indices (first 10): {loader.get_frame_indices()[:10]}")
        print(f"Class names: {loader.get_class_names()}")
        print(f"All instance IDs: {len(loader.get_all_instance_ids())} unique IDs")
        
        # Test loading a frame
        if loader.frame_indices:
            test_frame = loader.frame_indices[0]
            print(f"\nLoading frame {test_frame}:")
            
            rgb = loader.load_rgb(test_frame)
            print(f"  RGB shape: {rgb.shape if rgb is not None else None}")
            
            depth = loader.load_depth(test_frame)
            print(f"  Depth shape: {depth.shape if depth is not None else None}")
            
            gt_objects = loader.get_gt_objects(test_frame)
            print(f"  GT objects: {len(gt_objects)}")
            
            if gt_objects:
                obj = gt_objects[0]
                print(f"  First object: {obj.class_name} (ID: {obj.track_id})")
                print(f"    bbox_2d: {obj.bbox_2d}")
                print(f"    mask shape: {obj.mask.shape if obj.mask is not None else None}")
            
            camera_pose = loader.get_camera_pose(test_frame)
            if camera_pose:
                print(f"  Camera translation: {camera_pose.translation}")
            
            # Visualize if matplotlib available
            if HAS_MATPLOTLIB:
                print("\nShowing visualization...")
                loader.visualize_frame(test_frame)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
