#!/usr/bin/env python3
"""
VGGT Prediction and Save Script

This script loads an RGB image sequence, predicts camera intrinsics, extrinsics, 
and depth maps using VGGT, then saves them for later visualization.

Usage:
    python predict_and_save.py --rgb_dir /path/to/rgb/images --output_dir /path/to/output

Outputs:
    - camera_params.json: Camera intrinsics and extrinsics for each frame
    - depth_maps.npz: Predicted depth maps for each frame
    - point_clouds.npz: 3D point clouds in world frame for each frame
"""

import argparse
import os
import json
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def save_camera_params(extrinsic, intrinsic, output_path):
    """
    Save camera parameters as a dictionary with frame numbers as keys.
    
    Args:
        extrinsic: torch.Tensor of shape (B, N, 3, 4) - camera-from-world matrices
        intrinsic: torch.Tensor of shape (B, N, 3, 3) - intrinsic matrices  
        output_path: str - path to save the JSON file
    """
    camera_params = {}
    
    # Convert tensors to numpy for JSON serialization
    extrinsic_np = extrinsic.cpu().numpy() if torch.is_tensor(extrinsic) else extrinsic
    intrinsic_np = intrinsic.cpu().numpy() if torch.is_tensor(intrinsic) else intrinsic
    
    batch_size, num_frames = extrinsic_np.shape[:2]
    
    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            # Get Tcw (camera-from-world) from VGGT
            Tcw_3x4 = extrinsic_np[batch_idx, frame_idx]  # (3, 4)
            
            # Convert to 4x4 homogeneous matrix
            Tcw_4x4 = np.vstack([Tcw_3x4, np.array([[0, 0, 0, 1]])])
            
            # Compute Twc (world-from-camera) - inverse of Tcw
            Twc_4x4 = np.linalg.inv(Tcw_4x4)
            
            # Get intrinsic matrix
            K = intrinsic_np[batch_idx, frame_idx]  # (3, 3)
            
            # Store in dictionary
            frame_key = str(frame_idx)
            camera_params[frame_key] = {
                "Twc": Twc_4x4.tolist(),  # World-from-camera (4x4)
                "Tcw": Tcw_4x4.tolist(),  # Camera-from-world (4x4) 
                "intrinsic": K.tolist()   # Intrinsic matrix (3x3)
            }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(camera_params, f, indent=2)
    
    print(f"Camera parameters saved to {output_path}")
    print(f"Saved {len(camera_params)} frames")
    
    return camera_params


def load_rgb_images(rgb_dir, max_frames=None):
    """Load RGB images from directory."""
    
    # Support different image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(rgb_dir, ext)))
    
    image_files.sort()
    
    if max_frames is not None:
        image_files = image_files[:max_frames]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {rgb_dir}")
    
    print(f"Found {len(image_files)} images")
    return image_files


def main():
    parser = argparse.ArgumentParser(description="VGGT prediction and save script")
    parser.add_argument("--rgb_dir", required=True, help="Directory containing RGB images")
    parser.add_argument("--output_dir", required=True, help="Output directory for predictions")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--model_name", default="facebook/VGGT-1B", help="VGGT model name")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--stride", type=int, default=1, help="Downsample factor for point clouds")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load RGB images
    image_files = load_rgb_images(args.rgb_dir, args.max_frames)
    
    # Initialize VGGT model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    images = load_and_preprocess_images(image_files).to(device)
    
    print(f"Image tensor shape: {images.shape}")
    
    # Run VGGT prediction
    print("Running VGGT prediction...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
        
        print(f"Extrinsic shape: {extrinsic.shape}")  # (B, N, 3, 4)
        print(f"Intrinsic shape: {intrinsic.shape}")  # (B, N, 3, 3)
        
        # Predict Depth Maps
        with torch.cuda.amp.autocast(dtype=dtype):
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
        
        print(f"Depth map shape: {depth_map.shape}")  # (B, N, H, W)
        
        # Construct 3D Points from Depth Maps and Cameras
        point_map_by_unprojection = unproject_depth_map_to_point_map(
            depth_map.squeeze(0), 
            extrinsic.squeeze(0), 
            intrinsic.squeeze(0)
        )
        
        print(f"Point cloud shape: {point_map_by_unprojection.shape}")  # (N, 3, H, W)
    
    # Save camera parameters
    camera_params_path = os.path.join(args.output_dir, "camera_params.json")
    save_camera_params(extrinsic, intrinsic, camera_params_path)
    
    # Save depth maps
    depth_maps_path = os.path.join(args.output_dir, "depth_maps.npz")
    depth_maps_np = depth_map.squeeze(0).cpu().numpy()  # (N, H, W)
    np.savez_compressed(depth_maps_path, depth_maps=depth_maps_np)
    print(f"Depth maps saved to {depth_maps_path}")
    
    # Save point clouds (in world frame)
    point_clouds_path = os.path.join(args.output_dir, "point_clouds.npz") 
    point_clouds_np = point_map_by_unprojection  # (N, 3, H, W)
    np.savez_compressed(point_clouds_path, point_clouds=point_clouds_np)
    print(f"Point clouds saved to {point_clouds_path}")
    
    # Save metadata
    metadata = {
        "num_frames": len(image_files),
        "image_shape": list(images.shape[-2:]),
        "model_name": args.model_name,
        "image_files": [os.path.basename(f) for f in image_files]
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    print("Prediction complete!")


if __name__ == "__main__":
    main()