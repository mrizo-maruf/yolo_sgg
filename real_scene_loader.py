"""
THUD Real_Scenes Data Loader.

A data loader for THUD Real_Scenes benchmark datasets.
Supports loading RGB, depth, semantic segmentation, 3D bounding boxes, and camera poses.

Dataset Structure:
    Real_Scenes/
        {scene_name}/  (e.g., 10L)
            static/
                Capture_X/
                    RGB/frame-XXXXXX.color.png
                    Depth/frame-XXXXXX.depth.png
                    Camera_intrinsics/camera-intrinsics.txt  (3x3 matrix)
                    Label/
                        3D_Object_Detection/frame-XXXXXX.txt  (KITTI-like format)
                        Pose/frame-XXXXXX.pose.txt  (4x4 transformation matrix)
                        Semantic/frame-XXXXXX.png
                    Pointcloud/  (optional)

Usage:
    loader = RealSceneDataLoader(scene_path)
    
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
class GTObject3D:
    """Ground truth 3D object representation for THUD Real_Scenes dataset."""
    track_id: int  # Object ID (derived from line index for consistency across frames)
    class_name: str
    truncation: float = 0.0  # Truncation level (0 to 1)
    occlusion: int = 0  # Occlusion level (0, 1, 2, 3)
    alpha: float = 0.0  # Observation angle
    bbox_2d: Optional[List[float]] = None  # [x1, y1, x2, y2] 2D bounding box
    bbox_3d_size: Optional[List[float]] = None  # [h, w, l] height, width, length
    bbox_3d_center: Optional[List[float]] = None  # [x, y, z] 3D center location
    rotation_y: float = 0.0  # Rotation around Y-axis in camera coordinates
    score: float = 1.0  # Detection score (for GT, usually 1.0)


@dataclass
class CameraPose:
    """Camera pose information."""
    transform_matrix: np.ndarray  # 4x4 transformation matrix
    translation: np.ndarray  # [x, y, z] extracted from matrix
    rotation_matrix: np.ndarray  # 3x3 rotation matrix


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    matrix: np.ndarray  # 3x3 intrinsic matrix
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass 
class FrameData:
    """Container for all data from a single frame."""
    frame_idx: int
    rgb: Optional[np.ndarray] = None  # (H, W, 3) RGB
    depth: Optional[np.ndarray] = None  # (H, W) float32
    semantic_seg: Optional[np.ndarray] = None  # (H, W, 3) color-coded
    gt_objects: List[GTObject3D] = field(default_factory=list)
    camera_pose: Optional[CameraPose] = None


class RealSceneDataLoader:
    """
    Data loader for THUD Real_Scenes benchmark datasets.
    
    Expected directory structure:
        scene_path/  (e.g., Real_Scenes/10L/static/Capture_1)
            RGB/frame-XXXXXX.color.png
            Depth/frame-XXXXXX.depth.png
            Camera_intrinsics/camera-intrinsics.txt
            Label/
                3D_Object_Detection/frame-XXXXXX.txt
                Pose/frame-XXXXXX.pose.txt
                Semantic/frame-XXXXXX.png
    """
    
    def __init__(self, scene_path: str, verbose: bool = True):
        """
        Initialize Real_Scenes data loader.
        
        Args:
            scene_path: Path to scene directory (e.g., Real_Scenes/10L/static/Capture_1)
            verbose: Print loading information
        """
        self.scene_path = Path(scene_path)
        self.verbose = verbose
        
        # Directory paths
        self.rgb_dir = self.scene_path / "RGB"
        self.depth_dir = self.scene_path / "Depth"
        self.label_dir = self.scene_path / "Label"
        self.detection_3d_dir = self.label_dir / "3D_Object_Detection"
        self.pose_dir = self.label_dir / "Pose"
        self.semantic_dir = self.label_dir / "Semantic"
        self.intrinsics_dir = self.scene_path / "Camera_intrinsics"
        self.pointcloud_dir = self.scene_path / "Pointcloud"
        
        # Validate paths
        self._validate_paths()
        
        # Load camera intrinsics (single file for all frames)
        self.camera_intrinsics = self._load_camera_intrinsics()
        
        # Build frame index from RGB files
        self.frame_indices = []  # sorted list of available frame indices
        self._discover_frames()
        
        # Get image dimensions from first RGB image
        self.image_height, self.image_width = self._get_image_dimensions()
        
        # Track unique object classes
        self._class_names = set()
        self._scan_class_names()
        
        if self.verbose:
            print(f"[RealSceneDataLoader] Loaded scene: {self.scene_path}")
            if self.frame_indices:
                print(f"  Frames: {len(self.frame_indices)} (range: {min(self.frame_indices)}-{max(self.frame_indices)})")
            else:
                print(f"  Frames: 0 (no valid frames found)")
            print(f"  Image size: {self.image_width}x{self.image_height}")
            print(f"  Classes: {sorted(self._class_names)}")
    
    def _validate_paths(self):
        """Validate that required directories exist."""
        if not self.scene_path.exists():
            raise FileNotFoundError(f"Scene path not found: {self.scene_path}")
        
        required_dirs = [self.rgb_dir]
        for d in required_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Required directory not found: {d}")
    
    def _load_camera_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Load camera intrinsics from txt file."""
        intrinsics_path = self.intrinsics_dir / "camera-intrinsics.txt"
        
        if not intrinsics_path.exists():
            if self.verbose:
                print(f"[WARN] Camera intrinsics not found: {intrinsics_path}")
            return None
        
        try:
            matrix = np.loadtxt(intrinsics_path)
            
            if matrix.shape != (3, 3):
                if self.verbose:
                    print(f"[WARN] Invalid intrinsics shape: {matrix.shape}")
                return None
            
            return CameraIntrinsics(
                matrix=matrix.astype(np.float32),
                fx=float(matrix[0, 0]),
                fy=float(matrix[1, 1]),
                cx=float(matrix[0, 2]),
                cy=float(matrix[1, 2])
            )
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to load intrinsics: {e}")
            return None
    
    def _discover_frames(self):
        """Discover frame indices from RGB files."""
        if not self.rgb_dir.exists():
            return
        
        frame_indices = set()
        
        # Pattern: frame-XXXXXX.color.png
        for rgb_file in self.rgb_dir.glob("frame-*.color.png"):
            match = re.search(r'frame-(\d{6})\.color\.png', rgb_file.name)
            if match:
                frame_indices.add(int(match.group(1)))
        
        self.frame_indices = sorted(frame_indices)
    
    def _get_image_dimensions(self) -> Tuple[int, int]:
        """Get image dimensions from first RGB image."""
        if not self.frame_indices:
            return 540, 960  # Default for Real_Scenes
        
        first_frame = self.frame_indices[0]
        rgb_path = self.rgb_dir / f"frame-{first_frame:06d}.color.png"
        
        if rgb_path.exists():
            img = Image.open(rgb_path)
            return img.height, img.width
        
        return 540, 960  # Default
    
    def _scan_class_names(self):
        """Scan 3D detection files to discover class names."""
        if not self.detection_3d_dir.exists():
            return
        
        # Sample a few files to find class names
        sample_count = min(10, len(self.frame_indices))
        for frame_idx in self.frame_indices[:sample_count]:
            det_path = self.detection_3d_dir / f"frame-{frame_idx:06d}.txt"
            if det_path.exists():
                try:
                    with open(det_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                self._class_names.add(parts[0])
                except Exception:
                    pass
    
    def _format_frame_idx(self, frame_idx: int) -> str:
        """Format frame index as 6-digit zero-padded string."""
        return f"{frame_idx:06d}"
    
    def get_frame_count(self) -> int:
        """Get total number of available frames."""
        return len(self.frame_indices)
    
    def get_frame_indices(self) -> List[int]:
        """Get list of available frame indices."""
        return self.frame_indices.copy()
    
    def get_class_names(self) -> List[str]:
        """Get list of all class names in the dataset."""
        return sorted(list(self._class_names))
    
    def load_rgb(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load RGB image for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            RGB image as numpy array (H, W, 3) or None if not found
        """
        rgb_path = self.rgb_dir / f"frame-{self._format_frame_idx(frame_idx)}.color.png"
        
        if not rgb_path.exists():
            if self.verbose:
                print(f"[WARN] RGB not found: {rgb_path}")
            return None
        
        img = Image.open(rgb_path)
        arr = np.array(img)
        
        # Ensure RGB (remove alpha if present)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        
        return arr
    
    def load_depth(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load depth image for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Depth map as float32 numpy array (H, W) or None if not found
        """
        depth_path = self.depth_dir / f"frame-{self._format_frame_idx(frame_idx)}.depth.png"
        
        if not depth_path.exists():
            if self.verbose:
                print(f"[WARN] Depth not found: {depth_path}")
            return None
        
        # Load as 16-bit PNG
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        if depth_img is None:
            return None
        
        # Convert to float (assuming linear depth encoding in mm)
        return depth_img.astype(np.float32) / 1000.0  # Convert mm to meters
    
    def load_semantic_segmentation(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load semantic segmentation image.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Semantic segmentation image (H, W, 3) or None
        """
        semantic_path = self.semantic_dir / f"frame-{self._format_frame_idx(frame_idx)}.png"
        
        if not semantic_path.exists():
            if self.verbose:
                print(f"[WARN] Semantic seg not found: {semantic_path}")
            return None
        
        img = Image.open(semantic_path)
        return np.array(img)
    
    def load_camera_pose(self, frame_idx: int) -> Optional[CameraPose]:
        """
        Load camera pose for a frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            CameraPose object or None
        """
        pose_path = self.pose_dir / f"frame-{self._format_frame_idx(frame_idx)}.pose.txt"
        
        if not pose_path.exists():
            if self.verbose:
                print(f"[WARN] Pose not found: {pose_path}")
            return None
        
        try:
            transform = np.loadtxt(pose_path)
            
            if transform.shape != (4, 4):
                if self.verbose:
                    print(f"[WARN] Invalid pose shape: {transform.shape}")
                return None
            
            return CameraPose(
                transform_matrix=transform.astype(np.float32),
                translation=transform[:3, 3].astype(np.float32),
                rotation_matrix=transform[:3, :3].astype(np.float32)
            )
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to load pose: {e}")
            return None
    
    def get_camera_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get camera intrinsics (same for all frames)."""
        return self.camera_intrinsics
    
    def get_gt_objects(self, frame_idx: int) -> List[GTObject3D]:
        """
        Get ground truth 3D objects for a frame.
        
        The 3D object detection file uses KITTI-like format:
        class_name truncation occlusion alpha x1 y1 x2 y2 h w l x y z ry [score]
        
        Args:
            frame_idx: Frame index
            
        Returns:
            List of GTObject3D instances
        """
        det_path = self.detection_3d_dir / f"frame-{self._format_frame_idx(frame_idx)}.txt"
        
        if not det_path.exists():
            # No annotations for this frame
            return []
        
        gt_objects = []
        
        try:
            with open(det_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue
                    
                    try:
                        class_name = parts[0]
                        truncation = float(parts[1])
                        occlusion = int(float(parts[2]))
                        alpha = float(parts[3])
                        
                        # 2D bounding box (x1, y1, x2, y2)
                        x1 = float(parts[4])
                        y1 = float(parts[5])
                        x2 = float(parts[6])
                        y2 = float(parts[7])
                        bbox_2d = [x1, y1, x2, y2]
                        
                        # 3D dimensions (h, w, l)
                        h = float(parts[8])
                        w = float(parts[9])
                        l = float(parts[10])
                        bbox_3d_size = [h, w, l]
                        
                        # 3D location (x, y, z)
                        x = float(parts[11])
                        y = float(parts[12])
                        z = float(parts[13])
                        bbox_3d_center = [x, y, z]
                        
                        # Rotation and score
                        rotation_y = float(parts[14])
                        score = float(parts[15]) if len(parts) > 15 else 1.0
                        
                        # Update class names set
                        self._class_names.add(class_name)
                        
                        gt_obj = GTObject3D(
                            track_id=line_idx,  # Use line index as track_id
                            class_name=class_name,
                            truncation=truncation,
                            occlusion=occlusion,
                            alpha=alpha,
                            bbox_2d=bbox_2d,
                            bbox_3d_size=bbox_3d_size,
                            bbox_3d_center=bbox_3d_center,
                            rotation_y=rotation_y,
                            score=score
                        )
                        gt_objects.append(gt_obj)
                        
                    except (ValueError, IndexError) as e:
                        if self.verbose:
                            print(f"[WARN] Failed to parse line {line_idx}: {e}")
                        continue
                        
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to load detections: {e}")
        
        return gt_objects
    
    def load_pointcloud(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load point cloud for a frame (if available).
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Point cloud as numpy array (N, 3) or None
        """
        if not self.pointcloud_dir.exists():
            return None
        
        # Try common point cloud formats
        for ext in ['.ply', '.pcd', '.bin', '.npy']:
            pc_path = self.pointcloud_dir / f"frame-{self._format_frame_idx(frame_idx)}{ext}"
            
            if not pc_path.exists():
                continue
            
            try:
                if ext == '.npy':
                    return np.load(pc_path)
                elif ext == '.bin':
                    # KITTI binary format
                    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
                    return points[:, :3]  # Return XYZ only
                elif HAS_OPEN3D and ext in ['.ply', '.pcd']:
                    pcd = o3d.io.read_point_cloud(str(pc_path))
                    return np.asarray(pcd.points)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to load point cloud: {e}")
        
        return None
    
    def load_frame(self, frame_idx: int, 
                   load_rgb: bool = True,
                   load_depth: bool = True,
                   load_semantic: bool = False) -> FrameData:
        """
        Load all data for a single frame.
        
        Args:
            frame_idx: Frame index
            load_rgb: Whether to load RGB image
            load_depth: Whether to load depth map
            load_semantic: Whether to load semantic segmentation
            
        Returns:
            FrameData object containing all requested data
        """
        frame = FrameData(frame_idx=frame_idx)
        
        if load_rgb:
            frame.rgb = self.load_rgb(frame_idx)
        
        if load_depth:
            frame.depth = self.load_depth(frame_idx)
        
        if load_semantic:
            frame.semantic_seg = self.load_semantic_segmentation(frame_idx)
        
        frame.gt_objects = self.get_gt_objects(frame_idx)
        frame.camera_pose = self.load_camera_pose(frame_idx)
        
        return frame
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def visualize_frame(self, frame_idx: int,
                       show_rgb: bool = True,
                       show_bbox_2d: bool = True,
                       show_labels: bool = True,
                       show_semantic: bool = True,
                       alpha: float = 0.5,
                       figsize: Tuple[int, int] = (16, 8),
                       save_path: Optional[str] = None):
        """
        Visualize a frame with annotations.
        
        Args:
            frame_idx: Frame index to visualize
            show_rgb: Show RGB image
            show_bbox_2d: Draw 2D bounding boxes
            show_labels: Show class labels
            show_semantic: Show semantic segmentation overlay
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
                    
                    # Skip invalid boxes
                    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                        continue
                    
                    # Generate color from class name
                    np.random.seed(hash(obj.class_name) % (2**32))
                    color = tuple(int(c) for c in np.random.randint(50, 255, 3))
                    
                    cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), color, 2)
                    
                    if show_labels:
                        label = f"{obj.class_name}"
                        cv2.putText(vis_rgb, label, (x1, max(y1 - 5, 10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[0].imshow(vis_rgb)
        axes[0].set_title(f"Frame {frame_idx} - RGB + 2D Boxes ({len(gt_objects)} objects)")
        axes[0].axis('off')
        
        # Right panel: Semantic segmentation or depth
        if show_semantic:
            semantic_img = self.load_semantic_segmentation(frame_idx)
            if semantic_img is not None:
                if semantic_img.shape[2] == 4:
                    semantic_rgb = semantic_img[:, :, :3]
                else:
                    semantic_rgb = semantic_img
                vis_semantic = cv2.addWeighted(rgb, 1 - alpha, semantic_rgb, alpha, 0)
                axes[1].imshow(vis_semantic)
                axes[1].set_title(f"Frame {frame_idx} - Semantic Segmentation")
            else:
                # Show depth if no semantic
                depth = self.load_depth(frame_idx)
                if depth is not None:
                    axes[1].imshow(depth, cmap='viridis')
                    axes[1].set_title(f"Frame {frame_idx} - Depth")
                else:
                    axes[1].imshow(rgb)
        else:
            axes[1].imshow(rgb)
        
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_3d_objects(self, frame_idx: int):
        """
        Print 3D object information for a frame.
        
        Args:
            frame_idx: Frame index
        """
        gt_objects = self.get_gt_objects(frame_idx)
        
        print(f"\n=== Frame {frame_idx} - 3D Objects ===")
        print(f"Total objects: {len(gt_objects)}")
        
        for i, obj in enumerate(gt_objects):
            print(f"\n[{i}] {obj.class_name}")
            print(f"    3D Center: ({obj.bbox_3d_center[0]:.3f}, {obj.bbox_3d_center[1]:.3f}, {obj.bbox_3d_center[2]:.3f})")
            print(f"    3D Size (H,W,L): ({obj.bbox_3d_size[0]:.3f}, {obj.bbox_3d_size[1]:.3f}, {obj.bbox_3d_size[2]:.3f})")
            print(f"    Rotation Y: {obj.rotation_y:.3f}")
            if obj.bbox_2d:
                print(f"    2D Box: ({obj.bbox_2d[0]:.1f}, {obj.bbox_2d[1]:.1f}) - ({obj.bbox_2d[2]:.1f}, {obj.bbox_2d[3]:.1f})")


# ============================================================================
# Utility Functions
# ============================================================================

def discover_real_scenes(thud_root: str) -> List[str]:
    """
    Discover all available Real_Scenes.
    
    Args:
        thud_root: Root path to THUD dataset (should contain Real_Scenes folder)
        
    Returns:
        List of scene paths
    """
    thud_root = Path(thud_root)
    real_scenes_root = thud_root / "Real_Scenes"
    
    if not real_scenes_root.exists():
        return []
    
    scenes = []
    
    # Look for Capture_* directories under Real_Scenes/{scene}/static/
    for capture_dir in real_scenes_root.rglob("Capture_*"):
        if capture_dir.is_dir():
            # Verify it has RGB directory
            if (capture_dir / "RGB").exists():
                scenes.append(str(capture_dir))
    
    return sorted(scenes)


def get_real_scene_info(scene_path: str) -> Dict:
    """
    Get basic information about a Real_Scene without fully loading it.
    
    Args:
        scene_path: Path to scene directory
        
    Returns:
        Dictionary with scene information
    """
    scene_path = Path(scene_path)
    
    info = {
        'path': str(scene_path),
        'name': scene_path.name,
        'scene': scene_path.parent.parent.name if scene_path.parent.name == 'static' else 'unknown',
        'rgb_count': 0,
        'depth_count': 0,
        'semantic_count': 0,
        'detection_count': 0,
        'pose_count': 0,
    }
    
    # Count files
    rgb_dir = scene_path / "RGB"
    depth_dir = scene_path / "Depth"
    semantic_dir = scene_path / "Label" / "Semantic"
    detection_dir = scene_path / "Label" / "3D_Object_Detection"
    pose_dir = scene_path / "Label" / "Pose"
    
    if rgb_dir.exists():
        info['rgb_count'] = len(list(rgb_dir.glob("frame-*.color.png")))
    
    if depth_dir.exists():
        info['depth_count'] = len(list(depth_dir.glob("frame-*.depth.png")))
    
    if semantic_dir.exists():
        info['semantic_count'] = len(list(semantic_dir.glob("frame-*.png")))
    
    if detection_dir.exists():
        info['detection_count'] = len(list(detection_dir.glob("frame-*.txt")))
    
    if pose_dir.exists():
        info['pose_count'] = len(list(pose_dir.glob("frame-*.pose.txt")))
    
    return info


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Default test path
    test_path = "thud/Real_Scenes/10L/static/Capture_1"
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    
    print(f"Testing RealSceneDataLoader with: {test_path}")
    print("=" * 60)
    
    try:
        # Initialize loader
        loader = RealSceneDataLoader(test_path, verbose=True)
        
        print(f"\nFrame count: {loader.get_frame_count()}")
        print(f"Frame indices (first 10): {loader.get_frame_indices()[:10]}")
        print(f"Class names: {loader.get_class_names()}")
        
        # Camera intrinsics
        intrinsics = loader.get_camera_intrinsics()
        if intrinsics:
            print(f"\nCamera intrinsics:")
            print(f"  fx={intrinsics.fx}, fy={intrinsics.fy}")
            print(f"  cx={intrinsics.cx}, cy={intrinsics.cy}")
        
        # Test loading a frame
        if loader.frame_indices:
            test_frame = loader.frame_indices[0]
            print(f"\nLoading frame {test_frame}:")
            
            rgb = loader.load_rgb(test_frame)
            print(f"  RGB shape: {rgb.shape if rgb is not None else None}")
            
            depth = loader.load_depth(test_frame)
            print(f"  Depth shape: {depth.shape if depth is not None else None}")
            
            semantic = loader.load_semantic_segmentation(test_frame)
            print(f"  Semantic shape: {semantic.shape if semantic is not None else None}")
            
            gt_objects = loader.get_gt_objects(test_frame)
            print(f"  GT objects: {len(gt_objects)}")
            
            if gt_objects:
                obj = gt_objects[0]
                print(f"  First object: {obj.class_name}")
                print(f"    3D center: {obj.bbox_3d_center}")
                print(f"    3D size (HWL): {obj.bbox_3d_size}")
            
            camera_pose = loader.load_camera_pose(test_frame)
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
