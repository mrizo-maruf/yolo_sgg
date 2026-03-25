"""
CODA (UT Campus Object Dataset) Data Loader.

The loader handles:
- RGB images (rectified cam0)
- Depth maps generated on-the-fly from ego-compensated LiDAR point clouds (3d_comp)
- 3D bounding box annotations (ground truth)
- Camera intrinsics and LiDAR-to-camera extrinsics

Usage:
    loader = CODALoader("/path/to/CODa_tiny", sequence_id="0")
    frame_indices = loader.get_frame_indices()
    for idx in frame_indices:
        data = loader.get_frame_data(idx)
        # data.rgb, data.depth, data.gt_objects, data.cam_transform_4x4, etc.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
import cv2
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures (compatible with THUD loader)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GTObject:
    """Ground-truth object for one frame (CODA)."""
    track_id: int                 # instanceId from annotation
    instance_seg_id: int          # same as track_id
    bbox_2d_id: int               # same as track_id
    bbox_3d_id: int               # same as track_id

    class_name: str
    prim_path: Optional[str] = None      # not used in CODA

    # 2-D bounding box (projected from 3D bbox)
    bbox2d_xyxy: Optional[Tuple[float, float, float, float]] = None

    # 3-D bounding box (axis-aligned in world coordinates)
    box_3d_aabb_xyzmin_xyzmax: Tuple[float, float, float, float, float, float] = (0.0,0.0,0.0,0.0,0.0,0.0)
    box_3d_transform_4x4: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))

    # Extra OBB parameters (center, size, rotation)
    box_3d_center_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_3d_size_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_3d_rotation_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    visibility: Optional[float] = None
    occlusion: Optional[float] = None

    mask: Optional[np.ndarray] = None      # not provided in CODA


@dataclass(frozen=True)
class FrameData:
    """Per-frame container (matches IsaacSimSceneLoader.FrameData)."""
    frame_idx: int
    gt_objects: List[GTObject]

    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    cam_transform_4x4: Optional[np.ndarray] = None
    camera_intrinsic: Optional[np.ndarray] = None
    seg: Optional[np.ndarray] = None       # not used


# ---------------------------------------------------------------------------
# CODA Loader
# ---------------------------------------------------------------------------

class CODALoader:
    """
    Data loader for CODA (UT Campus Object Dataset).

    Parameters
    ----------
    root_dir : str
        Path to the CODa root directory, e.g. '/kaggle/input/datasets/krivosheymax/coda-tiny-copy/CODa_tiny'
    sequence_id : str
        Sequence identifier (e.g., "0" for the tiny version).
    load_rgb : bool
        If True, load RGB images.
    load_depth : bool
        If True, compute depth maps from LiDAR point clouds.
    skip_labels : set[str] | None
        Class names (case‑insensitive) to exclude entirely.
    require_3d : bool
        If True, skip objects without a 3D bounding box.
    max_depth : float
        Maximum depth to keep when building depth maps (meters). Points farther are ignored.
    verbose : bool
        Print progress information.
    """

    def __init__(
        self,
        root_dir: str,
        sequence_id: str = "0",
        load_rgb: bool = True,
        load_depth: bool = True,
        skip_labels: Optional[Set[str]] = None,
        require_3d: bool = True,
        max_depth: float = 80.0,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root_dir)
        self.seq = sequence_id
        self.load_rgb = load_rgb
        self.load_depth = load_depth
        self.skip_labels = {s.lower() for s in (skip_labels or set())}
        self.require_3d = require_3d
        self.max_depth = max_depth
        self.verbose = verbose

        # Paths
        self.rgb_dir = self.root / "2d_rect" / "cam0" / self.seq
        self.lidar_dir = self.root / "3d_comp" / "os1" / self.seq
        self.bbox_dir = self.root / "3d_bbox" / "os1" / self.seq
        self.calib_dir = self.root / "calibrations" / self.seq

        # Validate directories
        for d in [self.rgb_dir, self.lidar_dir, self.bbox_dir, self.calib_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Missing required directory: {d}")

        # Load calibration files
        self._load_calibrations()

        # Discover available frame indices
        self.frame_indices: List[int] = self._discover_frames()
        if self.verbose:
            print(f"[CODALoader] Sequence {self.seq}: {len(self.frame_indices)} frames found")
            if len(self.frame_indices) > 0:
                print(f"  Frame numbers: {self.frame_indices[:10]} ...")

        # Pre‑compute depth maps if requested (cached)
        self._depth_cache: Dict[int, np.ndarray] = {} if load_depth else None

        if load_depth and self.verbose:
            print("[CODALoader] Pre‑computing depth maps (this may take a while)...")
            for fid in tqdm(self.frame_indices, desc="Building depth"):
                self._depth_cache[fid] = self._build_depth_map(fid)

    # ------------------------------------------------------------------
    # Calibration loading
    # ------------------------------------------------------------------

    def _load_calibrations(self) -> None:
        """Load camera intrinsics and LiDAR-to-camera extrinsics."""
        # Intrinsics
        intr_path = self.calib_dir / "calib_cam0_intrinsics.yaml"
        with open(intr_path, 'r') as f:
            cam_intr = yaml.safe_load(f)
        K_data = cam_intr['camera_matrix']['data']
        self.K = np.array(K_data).reshape(3, 3)

        # Extrinsics (LiDAR → camera)
        ext_path = self.calib_dir / "calib_os1_to_cam0.yaml"
        with open(ext_path, 'r') as f:
            ext = yaml.safe_load(f)
        T_data = ext['extrinsic_matrix']['data']
        self.T_lidar_to_cam = np.array(T_data).reshape(4, 4)

        # Also store image size (from K? not strictly needed)
        self.image_width = int(cam_intr.get('image_width', 1224))
        self.image_height = int(cam_intr.get('image_height', 1024))

        if self.verbose:
            print(f"[CODALoader] Calibration loaded. Image size: {self.image_width}x{self.image_height}")

    # ------------------------------------------------------------------
    # Frame discovery
    # ------------------------------------------------------------------

    def _discover_frames(self) -> List[int]:
        """Return sorted list of frame numbers that have both RGB and LiDAR files."""
        frame_set = set()
        # RGB files: 2d_rect_cam0_{seq}_{frame}.jpg
        for p in self.rgb_dir.glob("2d_rect_cam0_*.jpg"):
            # extract frame number from filename
            parts = p.stem.split('_')
            if len(parts) >= 4:
                frame_num = int(parts[-1])
                # Check if corresponding LiDAR file exists
                lidar_file = self.lidar_dir / f"3d_comp_os1_{self.seq}_{frame_num}.bin"
                if lidar_file.exists():
                    frame_set.add(frame_num)
        return sorted(frame_set)

    # ------------------------------------------------------------------
    # Depth map generation
    # ------------------------------------------------------------------

    def _build_depth_map(self, frame_num: int) -> np.ndarray:
        """
        Project the ego‑compensated LiDAR point cloud into the camera frame
        and return a depth map (meters) of size (H, W) with zeros where no data.
        """
        lidar_file = self.lidar_dir / f"3d_comp_os1_{self.seq}_{frame_num}.bin"
        if not lidar_file.exists():
            return np.zeros((self.image_height, self.image_width), dtype=np.float32)

        # Load LiDAR points (N × 4: x,y,z,intensity)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        xyz_lidar = points[:, :3]

        # Transform to camera coordinates
        ones = np.ones((len(xyz_lidar), 1))
        xyz_lidar_h = np.hstack([xyz_lidar, ones])
        xyz_cam_h = (self.T_lidar_to_cam @ xyz_lidar_h.T).T
        xyz_cam = xyz_cam_h[:, :3]

        # Project to image plane
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        z = xyz_cam[:, 2]
        u = (fx * xyz_cam[:, 0] / z + cx).astype(int)
        v = (fy * xyz_cam[:, 1] / z + cy).astype(int)

        # Keep points in front of camera, inside image, and within max_depth
        mask = (z > 0) & (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        if self.max_depth > 0:
            mask &= (z <= self.max_depth)
        u = u[mask]
        v = v[mask]
        z = z[mask]

        # Build depth map: keep closest point per pixel
        depth_map = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        # Sort by increasing depth so that closer points overwrite farther ones
        order = np.argsort(z)
        for i in order:
            depth_map[v[i], u[i]] = z[i]
        depth_map[np.isinf(depth_map)] = 0.0
        return depth_map

    # ------------------------------------------------------------------
    # Load annotations for a frame
    # ------------------------------------------------------------------

    def _load_annotations(self, frame_num: int) -> List[GTObject]:
        """Load and parse 3D bounding box annotations."""
        json_path = self.bbox_dir / f"3d_bbox_os1_{self.seq}_{frame_num}.json"
        if not json_path.exists():
            return []

        with open(json_path, 'r') as f:
            data = json.load(f)

        objects = data.get('3dbbox', data.get('root', {}).get('3dbbox', []))
        if not objects:
            return []

        gt_list = []
        for obj in objects:
            # Required fields
            if not all(k in obj for k in ('cX', 'cY', 'cZ', 'l', 'w', 'h', 'r', 'p', 'y', 'instanceId', 'classId')):
                continue

            class_name = obj['classId']
            if class_name.lower() in self.skip_labels:
                continue

            # Extract parameters (in LiDAR coordinate frame)
            cx, cy, cz = obj['cX'], obj['cY'], obj['cZ']
            length, width, height = obj['l'], obj['w'], obj['h']
            yaw = obj['y']          # rotation about Z axis

            # Build OBB in LiDAR frame (we will keep it as is, but for AABB we need world coordinates)
            # Since points are ego‑compensated, the LiDAR frame is already the world frame for the sequence.
            # For compatibility, we store center, size, rotation.
            center_xyz = (cx, cy, cz)
            size_xyz = (length, width, height)
            # Rotation quaternion from yaw (rotation about Z)
            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            rot_xyzw = (0.0, 0.0, qz, qw)

            # Compute axis-aligned bounding box (AABB) from OBB
            # Corners of unrotated box
            dx, dy, dz = length/2, width/2, height/2
            corners_local = np.array([
                [-dx, -dy, -dz], [ dx, -dy, -dz],
                [ dx,  dy, -dz], [-dx,  dy, -dz],
                [-dx, -dy,  dz], [ dx, -dy,  dz],
                [ dx,  dy,  dz], [-dx,  dy,  dz],
            ])
            # Rotate about Z
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw),  np.cos(yaw), 0],
                          [0, 0, 1]])
            corners_world = (R @ corners_local.T).T + np.array([cx, cy, cz])
            aabb_min = corners_world.min(axis=0)
            aabb_max = corners_world.max(axis=0)
            aabb = (float(aabb_min[0]), float(aabb_min[1]), float(aabb_min[2]),
                    float(aabb_max[0]), float(aabb_max[1]), float(aabb_max[2]))

            # 2D bounding box projection (using camera intrinsics and extrinsics)
            # We need to project the 3D corners into the image.
            # Transform points from LiDAR to camera.
            corners_lidar_h = np.hstack([corners_world, np.ones((8,1))])
            corners_cam_h = (self.T_lidar_to_cam @ corners_lidar_h.T).T
            corners_cam = corners_cam_h[:, :3]
            # Project to pixels
            fx, fy = self.K[0,0], self.K[1,1]
            cx_p, cy_p = self.K[0,2], self.K[1,2]
            z_c = corners_cam[:, 2]
            u = (fx * corners_cam[:, 0] / z_c + cx_p).astype(int)
            v = (fy * corners_cam[:, 1] / z_c + cy_p).astype(int)
            # Clip to image
            u = np.clip(u, 0, self.image_width-1)
            v = np.clip(v, 0, self.image_height-1)
            x1, y1 = u.min(), v.min()
            x2, y2 = u.max(), v.max()
            if x2 <= x1 or y2 <= y1:
                # skip if degenerate projection
                bbox2d = None
            else:
                bbox2d = (float(x1), float(y1), float(x2), float(y2))

            # Build transformation matrix for the object (for compatibility)
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = R
            transform[:3, 3] = [cx, cy, cz]

            gt_obj = GTObject(
                track_id=obj['instanceId'],
                instance_seg_id=obj['instanceId'],
                bbox_2d_id=obj['instanceId'],
                bbox_3d_id=obj['instanceId'],
                class_name=class_name,
                bbox2d_xyxy=bbox2d,
                box_3d_aabb_xyzmin_xyzmax=aabb,
                box_3d_transform_4x4=transform,
                box_3d_center_xyz=center_xyz,
                box_3d_size_xyz=size_xyz,
                box_3d_rotation_xyzw=rot_xyzw,
            )
            gt_list.append(gt_obj)

        return gt_list

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame_count(self) -> int:
        return len(self.frame_indices)

    def get_frame_indices(self) -> List[int]:
        """Return list of frame numbers (original CODA frame numbers)."""
        return self.frame_indices.copy()

    def get_frame_data(self, frame_num: int) -> FrameData:
        """
        Return FrameData for the given CODA frame number.

        The frame number must be one of those returned by get_frame_indices().
        """
        if frame_num not in self.frame_indices:
            raise ValueError(f"Frame number {frame_num} not available. Valid frames: {self.frame_indices}")

        rgb = None
        if self.load_rgb:
            rgb_path = self.rgb_dir / f"2d_rect_cam0_{self.seq}_{frame_num}.jpg"
            img = cv2.imread(str(rgb_path))
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = None
        if self.load_depth:
            depth = self._depth_cache.get(frame_num)
            if depth is None:
                depth = self._build_depth_map(frame_num)

        gt_objects = self._load_annotations(frame_num)

        # Camera pose (optional – not used by default)
        # We could load from poses/dense, but for CODA tiny they may be missing or mis‑scaled.
        # We'll leave it as None for now.
        cam_transform = None

        return FrameData(
            frame_idx=frame_num,
            gt_objects=gt_objects,
            rgb=rgb,
            depth=depth,
            cam_transform_4x4=cam_transform,
            camera_intrinsic=self.K,
            seg=None,
        )

    # ------------------------------------------------------------------
    # Visualization (similar to THUD loader)
    # ------------------------------------------------------------------

    def visualize_frame(
        self,
        frame_num: int,
        show_rgb: bool = True,
        show_depth: bool = True,
        show_bbox_2d: bool = True,
        show_labels: bool = True,
        alpha: float = 0.5,
        figsize: Tuple[int, int] = (16, 8),
        save_path: Optional[str] = None,
    ):
        """Display RGB and depth with bounding boxes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed; cannot visualize.")
            return

        fd = self.get_frame_data(frame_num)

        if fd.rgb is None and show_rgb:
            print(f"No RGB for frame {frame_num}")
            return

        # Determine number of subplots
        n_plots = (1 if show_rgb else 0) + (1 if show_depth else 0)
        if n_plots == 0:
            print("Nothing to show.")
            return

        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0
        if show_rgb and fd.rgb is not None:
            img_rgb = fd.rgb.copy()
            # Draw 2D bounding boxes (projected from 3D)
            if show_bbox_2d:
                for obj in fd.gt_objects:
                    if obj.bbox2d_xyxy is not None:
                        x1, y1, x2, y2 = map(int, obj.bbox2d_xyxy)
                        np.random.seed(obj.track_id * 7 + 13)
                        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                        if show_labels:
                            label = f"{obj.class_name} ({obj.track_id})"
                            cv2.putText(img_rgb, label, (x1, y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            axes[plot_idx].imshow(img_rgb)
            axes[plot_idx].set_title(f"RGB (frame {frame_num})")
            axes[plot_idx].axis('off')
            plot_idx += 1

        if show_depth and fd.depth is not None:
            depth = fd.depth
            # Normalize for visualization (clip to 95th percentile)
            non_zero = depth[depth > 0]
            if len(non_zero) > 0:
                vmax = np.percentile(non_zero, 95)
            else:
                vmax = depth.max()
            depth_vis = np.clip(depth, 0, vmax)
            depth_norm = depth_vis / vmax
            depth_colored = (plt.cm.plasma(depth_norm)[:, :, :3] * 255).astype(np.uint8)
            axes[plot_idx].imshow(depth_colored)
            axes[plot_idx].set_title(f"Depth (frame {frame_num})")
            axes[plot_idx].axis('off')
            plot_idx += 1

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"Saved visualization to {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Helper: discover all sequences
# ---------------------------------------------------------------------------

def discover_coda_sequences(root_dir: str) -> List[str]:
    """Return list of available sequence IDs (folders under 2d_rect/cam0)."""
    root = Path(root_dir)
    seq_dirs = sorted((root / "2d_rect" / "cam0").glob("[0-9]*"))
    return [d.name for d in seq_dirs if d.is_dir()]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python coda_loader.py <root_dir> [sequence_id]")
        sys.exit(1)

    root = sys.argv[1]
    seq = sys.argv[2] if len(sys.argv) > 2 else "0"

    loader = CODALoader(root, seq, load_depth=True, max_depth=80.0, verbose=True)
    print(f"\nLoaded sequence {seq} with {loader.get_frame_count()} frames.")
    print("First 5 frame numbers:", loader.get_frame_indices()[:5])

    if loader.get_frame_count() > 0:
        frame_num = loader.get_frame_indices()[0]
        fd = loader.get_frame_data(frame_num)
        print(f"\nFrame {frame_num}: {len(fd.gt_objects)} objects")
        if fd.rgb is not None:
            print(f"  RGB shape: {fd.rgb.shape}")
        if fd.depth is not None:
            print(f"  Depth shape: {fd.depth.shape}, non‑zero: {np.sum(fd.depth > 0)}")

        # Show visualization if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            loader.visualize_frame(frame_num, show_depth=True)
        except ImportError:
            print("Matplotlib not available, skipping visualization.")