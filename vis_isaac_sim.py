"""
Visualize Isaac Sim data using isaac_sim_loaderv3.py

This script loads a scene using IsaacSimSceneLoader and visualizes each frame:
- 2D: RGB with semantic masks overlay, 2D bboxes, and class labels
- 3D: Point cloud with RGB colors and 3D bounding boxes in Open3D

Usage:
    Set SCENE_DIR, camera intrinsics, and depth parameters at the top of the file.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import open3d as o3d
from pathlib import Path

from isaac_sim_loaderv3 import IsaacSimSceneLoader, FrameData, GTObject


# -------------------------
# Configuration
# -------------------------
SCENE_DIR = "/home/maribjonov_mr/IsaacSim_bench/scene_7"  # Relative to workspace or absolute path

# Camera intrinsics (must match your data capture)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FX = 800
FY = 800
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0

# Depth parameters (must match your data capture)
MIN_DEPTH = 0.01
MAX_DEPTH = 10.0
PNG_MAX_VALUE = 65535  # Depth is stored as uint16

# Visualization settings
MASK_ALPHA = 0.4  # Transparency for semantic masks
SWAP_YZ = False  # Swap Y-Z axes in 3D visualization if needed
MAX_BOX_EDGE = 20.0  # Filter out very large boxes (meters)

# Bbox interpretation: "world" if already in world coords, "camera" if in camera frame, "use_transform" to use object's transform_4x4
BBOX_FRAME = "world"  # Set to "world" if bboxes are already in world coords after data collection fix

# Skip labels for cleaner visualization
SKIP_LABELS = {'wall', 'floor', 'ground', 'ceiling', 'background'}

# -------------------------
# Helper functions
# -------------------------

def generate_color_for_id(id_val: int):
    """Generate a consistent color for a given ID using golden ratio."""
    golden_ratio = 0.618033988749895
    hue = (id_val * golden_ratio) % 1.0
    saturation = 0.7 + (id_val % 3) * 0.1
    value = 0.8 + (id_val % 2) * 0.1
    rgb = hsv_to_rgb([hue, saturation, value])
    return rgb


def make_intrinsics(width: int, height: int, fx: float, fy: float, cx: float, cy: float):
    """Create Open3D camera intrinsics."""
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(width, height, fx, fy, cx, cy)
    return intr


def convert_depth_to_meters(depth_uint16: np.ndarray, png_max_value: float, min_depth: float, max_depth: float) -> np.ndarray:
    """
    Convert uint16 depth image to meters.
    
    Args:
        depth_uint16: Depth image in uint16 format (0-65535)
        png_max_value: Maximum value in PNG (typically 65535)
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters
    
    Returns:
        Depth in meters as float32
    """
    if depth_uint16.ndim == 3:
        depth_uint16 = depth_uint16[:, :, 0]
    
    png_depth_scale = (max_depth - min_depth) / float(png_max_value)
    depth_meters = depth_uint16.astype(np.float32) * png_depth_scale + min_depth
    
    return depth_meters


def create_point_cloud(rgb_bgr: np.ndarray, depth: np.ndarray, intrinsics, max_depth: float):
    """
    Create Open3D point cloud from RGB and depth images.
    
    Args:
        rgb_bgr: RGB image in BGR format (H, W, 3)
        depth: Depth image in meters (H, W)
        intrinsics: Open3D camera intrinsics
        max_depth: Maximum depth for truncation
    
    Returns:
        Open3D point cloud
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    
    # Create Open3D images
    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=max_depth,
        convert_rgb_to_intensity=False,
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    
    return pcd


def create_bbox_lineset(bbox_corners: np.ndarray, color=[0, 1, 0]):
    """
    Create Open3D lineset from 8 bbox corners.
    
    Args:
        bbox_corners: (8, 3) array of corner positions
        color: RGB color for the box
    
    Returns:
        Open3D LineSet
    """
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ], dtype=np.int32)
    
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(bbox_corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color(color)
    
    return ls


def aabb_to_corners(aabb_xyzmin_xyzmax):
    """Convert AABB format to 8 corners."""
    xmin, ymin, zmin, xmax, ymax, zmax = aabb_xyzmin_xyzmax
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
    return corners


def transform_corners(corners: np.ndarray, transform_4x4: np.ndarray):
    """Transform corners using 4x4 matrix."""
    corners_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    corners_transformed = (transform_4x4 @ corners_h.T).T
    return corners_transformed[:, :3]


def apply_swap_yz(points: np.ndarray) -> np.ndarray:
    """Swap Y and Z axes: (x,y,z) -> (x,z,y)"""
    p = points.copy()
    p[:, [1, 2]] = p[:, [2, 1]]
    return p


def visualize_2d(frame_data: FrameData, mask_alpha: float = 0.4):
    """
    Visualize frame in 2D using matplotlib.
    
    Shows RGB with semantic masks overlay, 2D bboxes, and labels.
    """
    if frame_data.rgb is None:
        print("  [2D VIS] No RGB data available")
        return
    
    # Convert BGR to RGB for matplotlib
    rgb = cv2.cvtColor(frame_data.rgb, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax_rgb = axes[0]
    ax_overlay = axes[1]
    
    # Panel 1: Raw RGB
    ax_rgb.imshow(rgb)
    ax_rgb.set_title(f"RGB - Frame {frame_data.frame_idx}")
    ax_rgb.axis("off")
    
    # Panel 2: RGB + Masks + 2D Bboxes + Labels
    overlay_img = rgb.copy().astype(np.float32)
    mask_overlay = np.zeros((H, W, 3), dtype=np.float32)
    
    # Count objects
    valid_objects = frame_data.gt_objects  # Objects already filtered by loader
    
    print(f"  [2D VIS] Objects shown: {len(valid_objects)}")
    
    # Draw masks and bboxes
    for obj in valid_objects:
        # Generate color for this object
        color = generate_color_for_id(obj.track_id)
        
        # Add mask to overlay
        if obj.mask is not None:
            mask_overlay[obj.mask] = color
        
        # Draw 2D bbox
        if obj.bbox2d_xyxy is not None:
            x1, y1, x2, y2 = obj.bbox2d_xyxy
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax_overlay.add_patch(rect)
            
            # Draw label
            label_text = f"T:{obj.track_id} | S:{obj.semantic_id} | {obj.class_name}"
            ax_overlay.text(
                x1 + 2, y1 - 5,
                label_text,
                fontsize=8,
                color="white",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
            )
    
    # Blend mask with RGB
    overlay_img = overlay_img * (1 - mask_alpha) + mask_overlay * mask_alpha * 255
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
    
    ax_overlay.imshow(overlay_img)
    ax_overlay.set_title(f"Frame {frame_data.frame_idx} - RGB + Masks + 2D BBoxes ({len(valid_objects)} objects)")
    ax_overlay.axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize_3d(frame_data: FrameData, intrinsics, max_depth: float, 
                 swap_yz: bool = False, max_box_edge: float = 20.0,
                 bbox_frame: str = "world",
                 png_max_value: float = 65535, min_depth: float = 0.01):
    """
    Visualize frame in 3D using Open3D.
    
    Shows point cloud with RGB colors and 3D bounding boxes.
    
    Args:
        frame_data: FrameData object
        intrinsics: Open3D camera intrinsics
        max_depth: Maximum depth for truncation
        swap_yz: Whether to swap Y and Z axes
        max_box_edge: Maximum bbox edge size (filter out larger boxes)
        bbox_frame: "world", "camera", or "use_transform" - how to interpret bbox coords
        png_max_value: Max value for uint16 depth
        min_depth: Minimum depth in meters
    """
    if frame_data.rgb is None or frame_data.depth is None:
        print("  [3D VIS] No RGB or depth data available")
        return
    
    # Convert depth from uint16 to meters
    depth_meters = convert_depth_to_meters(frame_data.depth, png_max_value, min_depth, max_depth)
    
    # Create point cloud
    pcd = create_point_cloud(frame_data.rgb, depth_meters, intrinsics, max_depth)
    
    # Apply Y-Z swap if needed (do this BEFORE camera transform)
    if swap_yz:
        pts = np.asarray(pcd.points)
        pcd.points = o3d.utility.Vector3dVector(apply_swap_yz(pts))
    
    # Apply camera transform if available to move point cloud to world frame
    T_world_from_cam = frame_data.cam_transform_4x4
    if T_world_from_cam is not None:
        pcd.transform(T_world_from_cam)
    
    # Prepare geometries list
    geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)]
    
    # Count objects
    valid_objects = frame_data.gt_objects  # Objects already filtered by loader
    
    print(f"  [3D VIS] Objects shown: {len(valid_objects)}")
    
    # Add 3D bounding boxes
    for obj in valid_objects:
        # Check box size
        xmin, ymin, zmin, xmax, ymax, zmax = obj.box_3d_aabb_xyzmin_xyzmax
        sx, sy, sz = xmax - xmin, ymax - ymin, zmax - zmin
        
        if max(sx, sy, sz) > max_box_edge:
            print(f"    SKIP (too large): '{obj.class_name}' T:{obj.track_id} "
                  f"size=({sx:.2f}, {sy:.2f}, {sz:.2f})")
            continue
        
        # Create bbox lineset based on frame interpretation
        if bbox_frame == "world":
            # Bboxes are already in world coords - use AABB directly
            corners = aabb_to_corners(obj.box_3d_aabb_xyzmin_xyzmax)
            
            # Apply Y-Z swap if needed
            if swap_yz:
                corners = apply_swap_yz(corners)
            
            ls = create_bbox_lineset(corners, color=generate_color_for_id(obj.track_id))
            
        elif bbox_frame == "use_transform":
            # Use object's transform_4x4 to convert local AABB to world
            corners = aabb_to_corners(obj.box_3d_aabb_xyzmin_xyzmax)
            corners_world = transform_corners(corners, obj.box_3d_transform_4x4)
            
            # Apply Y-Z swap if needed
            if swap_yz:
                corners_world = apply_swap_yz(corners_world)
            
            ls = create_bbox_lineset(corners_world, color=generate_color_for_id(obj.track_id))
            
        elif bbox_frame == "camera":
            # Bboxes are in camera frame - transform using camera pose
            if T_world_from_cam is None:
                print(f"    SKIP: '{obj.class_name}' T:{obj.track_id} - no camera transform available")
                continue
            
            corners = aabb_to_corners(obj.box_3d_aabb_xyzmin_xyzmax)
            corners_world = transform_corners(corners, T_world_from_cam)
            
            # Apply Y-Z swap if needed
            if swap_yz:
                corners_world = apply_swap_yz(corners_world)
            
            ls = create_bbox_lineset(corners_world, color=generate_color_for_id(obj.track_id))
            
        else:
            print(f"    SKIP: Unknown bbox_frame: {bbox_frame}")
            continue
        
        geoms.append(ls)
        
        print(f"    Box: '{obj.class_name}' T:{obj.track_id} S:{obj.semantic_id} "
              f"center=({(xmin+xmax)/2:.2f}, {(ymin+ymax)/2:.2f}, {(zmin+zmax)/2:.2f}) "
              f"size=({sx:.2f}, {sy:.2f}, {sz:.2f})")
    
    # Visualize
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Frame {frame_data.frame_idx} - 3D Visualization"
    )


def print_frame_info(frame_data: FrameData):
    """Print information about the frame."""
    print(f"\n{'='*70}")
    print(f"Frame {frame_data.frame_idx}")
    print(f"{'='*70}")
    
    print(f"  Objects: {len(frame_data.gt_objects)}")
    print(f"  RGB available: {frame_data.rgb is not None}")
    print(f"  Depth available: {frame_data.depth is not None}")
    print(f"  Camera transform available: {frame_data.cam_transform_4x4 is not None}")
    
    if frame_data.cam_transform_4x4 is not None:
        cam_pos = frame_data.cam_transform_4x4[:3, 3]
        print(f"  Camera position: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")
    
    print(f"\n  Objects:")
    for i, obj in enumerate(frame_data.gt_objects, 1):
        bbox_info = "Yes" if obj.bbox2d_xyxy is not None else "No"
        vis_info = f"{obj.visibility:.2f}" if obj.visibility is not None else "N/A"
        occ_info = f"{obj.occlusion:.2f}" if obj.occlusion is not None else "N/A"
        mask_pixels = np.sum(obj.mask) if obj.mask is not None else 0
        
        print(f"    {i:2d}. Track:{obj.track_id:3d} Sem:{obj.semantic_id:3d} "
              f"'{obj.class_name:15s}' | 2D BBox:{bbox_info:3s} | "
              f"Vis:{vis_info:5s} Occ:{occ_info:5s} | Mask:{mask_pixels:6d}px")


# -------------------------
# Main
# -------------------------

def main():
    scene_dir = SCENE_DIR
    
    print(f"Loading scene from: {scene_dir}")
    
    # Create loader
    loader = IsaacSimSceneLoader(
        scene_dir=scene_dir,
        load_rgb=True,
        load_depth=True,
        skip_labels=SKIP_LABELS
    )
    
    print(f"Found {len(loader.frame_indices)} frames")
    print(f"Frame indices: {loader.frame_indices}")
    
    # Create camera intrinsics for 3D visualization
    intrinsics = make_intrinsics(IMAGE_WIDTH, IMAGE_HEIGHT, FX, FY, CX, CY)
    
    # Process each frame
    for frame_idx in loader.frame_indices:
        print(f"\n{'#'*70}")
        print(f"# Processing Frame {frame_idx}")
        print(f"{'#'*70}")
        
        # Load frame data
        frame_data = loader.get_frame_data(frame_idx)
        
        # Print frame information
        print_frame_info(frame_data)
        
        # 2D Visualization
        if frame_data.rgb is not None:
            visualize_2d(frame_data, mask_alpha=MASK_ALPHA)
        
        # 3D Visualization
        if frame_data.rgb is not None and frame_data.depth is not None:
            visualize_3d(
                frame_data,
                intrinsics=intrinsics,
                max_depth=MAX_DEPTH,
                swap_yz=SWAP_YZ,
                max_box_edge=MAX_BOX_EDGE,
                bbox_frame=BBOX_FRAME,
                png_max_value=PNG_MAX_VALUE,
                min_depth=MIN_DEPTH
            )
        
        print(f"\nFrame {frame_idx} complete. Close visualization windows to continue...")
    
    print(f"\n{'#'*70}")
    print(f"# All frames processed!")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
