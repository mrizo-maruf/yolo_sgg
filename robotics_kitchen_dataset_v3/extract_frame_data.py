#!/usr/bin/env python3
"""
Extract Frame Data Script

This script extracts point clouds, camera parameters, and camera positions 
for every frame from VGGT predictions and optionally GT depth maps.

Usage:
    python extract_frame_data.py --data_dir ./predictions --gt_depth_dir ./depth_maps --output_dir ./frame_data

Outputs per frame:
    - frame_XXXXXX_pointcloud.npz: Point cloud (N, 3) + colors (N, 3) if available
    - frame_XXXXXX_camera.json: Camera intrinsics, extrinsics, and position
    - summary.json: Overview of all frames
"""

import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path


def load_camera_params(json_path):
    """Load camera parameters from JSON file."""
    with open(json_path, 'r') as f:
        camera_params = json.load(f)
    return camera_params


def load_gt_depth_maps(depth_dir):
    """Load GT depth maps from directory (.npy files)."""
    if not os.path.exists(depth_dir):
        return None
    
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    gt_depths = {}
    
    for i, depth_file in enumerate(depth_files):
        depth_path = os.path.join(depth_dir, depth_file)
        depth = np.load(depth_path)
        gt_depths[i] = depth
    
    print(f"Loaded {len(gt_depths)} GT depth maps")
    return gt_depths


def load_rgb_images(rgb_dir, image_files_list=None):
    """Load RGB images for coloring."""
    if not os.path.exists(rgb_dir):
        return {}
    
    if image_files_list is None:
        # Auto-discover images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        import glob
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
        image_files.sort()
    else:
        image_files = [os.path.join(rgb_dir, f) for f in image_files_list]
    
    rgb_images = {}
    for i, img_path in enumerate(image_files):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_images[i] = img_rgb
    
    print(f"Loaded {len(rgb_images)} RGB images")
    return rgb_images


def unproject_depth_to_pointcloud(depth, K, Twc=None, rgb_image=None, stride=1):
    """
    Convert depth map to 3D point cloud.
    
    Args:
        depth: (H, W) depth map in meters
        K: (3, 3) camera intrinsic matrix
        Twc: (4, 4) world-from-camera transform (optional, for world coords)
        rgb_image: (H, W, 3) RGB image for coloring (optional)
        stride: downsample factor
    
    Returns:
        points: (N, 3) 3D points
        colors: (N, 3) RGB colors (if rgb_image provided)
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel grid
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    
    # Downsample depth
    depth_ds = depth[::stride, ::stride]
    
    # Back-project to camera frame
    z = depth_ds
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    
    # Stack into point cloud
    points_cam = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    
    # Filter valid points
    valid_mask = (z.ravel() > 0) & np.isfinite(points_cam).all(axis=1)
    points_cam = points_cam[valid_mask]
    
    # Transform to world frame if Twc provided
    if Twc is not None:
        # Convert to homogeneous coordinates
        ones = np.ones((points_cam.shape[0], 1))
        points_cam_h = np.hstack([points_cam, ones])
        
        # Transform to world
        points_world_h = (Twc @ points_cam_h.T).T
        points = points_world_h[:, :3]
    else:
        points = points_cam
    
    # Get colors if RGB image provided
    colors = None
    if rgb_image is not None:
        rgb_ds = rgb_image[::stride, ::stride]
        colors_flat = rgb_ds.reshape(-1, 3)[valid_mask].astype(np.float32) / 255.0
        colors = colors_flat
    
    return points, colors


def extract_frame_data(camera_params, vggt_point_clouds=None, gt_depths=None, 
                      rgb_images=None, use_gt_depth=False, stride=1):
    """
    Extract all data for each frame.
    
    Returns:
        frame_data: dict with frame info for each frame
    """
    frame_data = {}
    num_frames = len(camera_params)
    
    for frame_idx in range(num_frames):
        frame_key = str(frame_idx)
        
        if frame_key not in camera_params:
            continue
        
        # Get camera parameters
        cam_data = camera_params[frame_key]
        Twc = np.array(cam_data['Twc'])
        Tcw = np.array(cam_data['Tcw']) 
        K = np.array(cam_data['intrinsic'])
        
        # Camera position (translation part of Twc)
        camera_position = Twc[:3, 3]
        camera_rotation = Twc[:3, :3]
        
        # Get point cloud
        points = None
        colors = None
        pointcloud_source = "none"
        
        if use_gt_depth and gt_depths and frame_idx in gt_depths:
            # Use GT depth + VGGT camera poses
            gt_depth = gt_depths[frame_idx]
            rgb_img = rgb_images.get(frame_idx, None)
            points, colors = unproject_depth_to_pointcloud(
                gt_depth, K, Twc, rgb_img, stride=stride
            )
            pointcloud_source = "gt_depth_vggt_pose"
            
        elif vggt_point_clouds is not None and frame_idx < vggt_point_clouds.shape[0]:
            # Use VGGT predicted point clouds
            points_frame = vggt_point_clouds[frame_idx]  # (3, H, W) or (H, W, 3)
            
            # Handle shape
            if points_frame.shape[0] == 3:  # (3, H, W)
                points_frame = points_frame.transpose(1, 2, 0)  # (H, W, 3)
            
            # Downsample and flatten
            points_ds = points_frame[::stride, ::stride]
            points = points_ds.reshape(-1, 3)
            
            # Remove invalid points
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            
            # Get colors from RGB if available
            if rgb_images and frame_idx in rgb_images:
                rgb_img = rgb_images[frame_idx]
                h, w = points_ds.shape[:2]
                rgb_resized = cv2.resize(rgb_img, (w, h))
                colors_ds = rgb_resized.astype(np.float32) / 255.0
                colors = colors_ds.reshape(-1, 3)[valid_mask]
            
            pointcloud_source = "vggt_predicted"
        
        # Store frame data
        frame_info = {
            # Camera parameters
            "intrinsic": K.tolist(),
            "extrinsic_Twc": Twc.tolist(),  # World-from-camera
            "extrinsic_Tcw": Tcw.tolist(),  # Camera-from-world
            
            # Camera pose
            "camera_position": camera_position.tolist(),
            "camera_rotation": camera_rotation.tolist(),
            
            # Point cloud info
            "pointcloud_source": pointcloud_source,
            "num_points": len(points) if points is not None else 0,
            "has_colors": colors is not None
        }
        
        frame_data[frame_idx] = {
            "info": frame_info,
            "points": points,
            "colors": colors
        }
    
    return frame_data


def save_frame_data(frame_data, output_dir):
    """Save frame data to individual files."""
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {
        "num_frames": len(frame_data),
        "frames": {}
    }
    
    for frame_idx, data in frame_data.items():
        # Save point cloud
        if data["points"] is not None:
            pc_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_pointcloud.npz")
            save_dict = {"points": data["points"]}
            if data["colors"] is not None:
                save_dict["colors"] = data["colors"]
            np.savez_compressed(pc_path, **save_dict)
        
        # Save camera info
        camera_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_camera.json")
        with open(camera_path, 'w') as f:
            json.dump(data["info"], f, indent=2)
        
        # Update summary
        summary["frames"][str(frame_idx)] = data["info"]
        
        print(f"Frame {frame_idx}: {data['info']['num_points']} points, "
              f"camera at {data['info']['camera_position']}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved data for {len(frame_data)} frames to {output_dir}")
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract frame data from VGGT predictions")
    parser.add_argument("--data_dir", required=True, help="VGGT predictions directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for frame data")
    parser.add_argument("--gt_depth_dir", default=None, help="GT depth maps directory (.npy files)")
    parser.add_argument("--rgb_dir", default=None, help="RGB images directory")
    parser.add_argument("--use_gt_depth", action="store_true", help="Use GT depth instead of VGGT point clouds")
    parser.add_argument("--stride", type=int, default=2, help="Downsample factor for point clouds")
    
    args = parser.parse_args()
    
    # Load VGGT predictions
    camera_params_path = os.path.join(args.data_dir, "camera_params.json")
    point_clouds_path = os.path.join(args.data_dir, "point_clouds.npz")
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    
    if not os.path.exists(camera_params_path):
        raise FileNotFoundError(f"Camera params not found: {camera_params_path}")
    
    # Load camera parameters
    camera_params = load_camera_params(camera_params_path)
    
    # Load VGGT point clouds (optional)
    vggt_point_clouds = None
    if os.path.exists(point_clouds_path) and not args.use_gt_depth:
        point_clouds_data = np.load(point_clouds_path)
        vggt_point_clouds = point_clouds_data['point_clouds']
        print(f"Loaded VGGT point clouds: {vggt_point_clouds.shape}")
    
    # Load GT depth maps (optional)
    gt_depths = None
    if args.gt_depth_dir:
        gt_depths = load_gt_depth_maps(args.gt_depth_dir)
    
    # Load RGB images (optional)
    rgb_images = {}
    if args.rgb_dir:
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        image_files = metadata.get('image_files', None)
        rgb_images = load_rgb_images(args.rgb_dir, image_files)
    
    print(f"Processing {len(camera_params)} frames...")
    
    # Extract frame data
    frame_data = extract_frame_data(
        camera_params=camera_params,
        vggt_point_clouds=vggt_point_clouds,
        gt_depths=gt_depths,
        rgb_images=rgb_images,
        use_gt_depth=args.use_gt_depth,
        stride=args.stride
    )
    
    # Save frame data
    save_frame_data(frame_data, args.output_dir)
    
    print("Frame data extraction complete!")
    
    # Print usage examples
    print("\nUsage examples:")
    print("# Load a specific frame's data:")
    print(f"import numpy as np")
    print(f"import json")
    print(f"frame_idx = 0")
    print(f"pc_data = np.load('{args.output_dir}/frame_{{frame_idx:06d}}_pointcloud.npz')")
    print(f"points = pc_data['points']  # (N, 3)")
    print(f"colors = pc_data['colors']  # (N, 3) if available")
    print(f"with open('{args.output_dir}/frame_{{frame_idx:06d}}_camera.json') as f:")
    print(f"    camera_data = json.load(f)")
    print(f"camera_pos = camera_data['camera_position']  # [x, y, z]")
    print(f"intrinsic = np.array(camera_data['intrinsic'])  # (3, 3)")


if __name__ == "__main__":
    main()