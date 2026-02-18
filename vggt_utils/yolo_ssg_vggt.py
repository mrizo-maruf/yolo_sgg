from omegaconf import OmegaConf
import YOLOE.utils as yutils
import numpy as np
from ssg.ssg_main import edges
import matplotlib.pyplot as plt
import time
import torch
import networkx as nx
import json
import os
import glob
import cv2
import open3d as o3d
def load_vggt_frame_data(frame_data_dir, frame_idx):
    """
    Load VGGT predicted camera parameters and point cloud for a specific frame.
    
    Args:
        frame_data_dir: Directory containing frame_XXXXXX_camera.json and frame_XXXXXX_pointcloud.npz files
        frame_idx: Frame index
        
    Returns:
        camera_params: Dict with intrinsic, extrinsic_Twc, camera_position, etc.
        points: (N, 3) point cloud in world frame
        colors: (N, 3) RGB colors for points (if available)
    """
    # Load camera parameters
    camera_file = os.path.join(frame_data_dir, f"frame_{frame_idx:06d}_camera.json")
    if not os.path.exists(camera_file):
        return None, None, None
    
    with open(camera_file, 'r') as f:
        camera_params = json.load(f)
    
    # Load point cloud
    pc_file = os.path.join(frame_data_dir, f"frame_{frame_idx:06d}_pointcloud.npz")
    if not os.path.exists(pc_file):
        return camera_params, None, None
    
    pc_data = np.load(pc_file)
    points_cam = pc_data['points']  # (N, 3) - These are in CAMERA frame!
    colors = pc_data.get('colors', None)  # (N, 3) RGB values [0,1]
    
    # DEBUG: Check coordinate frames and data
    # print(f"[DEBUG] Loaded {len(points_cam)} points from {pc_file}")
    # print(f"[DEBUG] Original (camera frame) range: X[{points_cam[:, 0].min():.3f}, {points_cam[:, 0].max():.3f}], Y[{points_cam[:, 1].min():.3f}, {points_cam[:, 1].max():.3f}], Z[{points_cam[:, 2].min():.3f}, {points_cam[:, 2].max():.3f}]")
    # print(f"[DEBUG] Camera position: {np.array(camera_params['camera_position'])}")
    
    # Transform VGGT point clouds from camera frame to world frame
    T_w_c = np.array(camera_params['extrinsic_Twc'])
    # print(f"[DEBUG] Applying Twc transformation to convert camera frame → world frame")
    
    # Convert to homogeneous coordinates
    ones = np.ones((len(points_cam), 1))
    points_cam_h = np.hstack([points_cam, ones])
    
    # Transform to world frame using Twc (world-from-camera)
    points_world_h = (T_w_c @ points_cam_h.T).T
    points = points_world_h[:, :3]
    
    # print(f"[DEBUG] Transformed (world frame) range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    # print(f"[DEBUG] Points now in WORLD frame (transformed in yolo_ssg_e.py)")
    
    # add open3d visualization here if needed
    # visualize the frame also

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # if colors is not None:
    #     pcd.colors = o3d.utility.Vector3dVector(colors)

    # o3d.visualization.draw_geometries([pcd])

    return camera_params, points, colors

def extract_points_from_vggt_pointcloud(vggt_points, vggt_colors, mask_2d, intrinsic, T_w_c, 
                                       max_points=None, random_state=None, o3_nb_neighbors=20, o3_std_ratio=2.0):
    """
    Extract points from VGGT point cloud using 2D mask by projecting to image plane.
    
    Args:
        vggt_points: (N, 3) VGGT predicted points in world frame
        vggt_colors: (N, 3) RGB colors for points
        mask_2d: (H, W) 2D mask from YOLO segmentation
        intrinsic: (3, 3) camera intrinsic matrix
        T_w_c: (4, 4) camera-to-world transform
        max_points: Optional limit on number of points
        random_state: Random seed for sampling
        o3_nb_neighbors: Number of neighbors for statistical outlier removal (default: 20)
        o3_std_ratio: Standard deviation ratio threshold for outlier removal (default: 2.0)
        
    Returns:
        selected_points: (M, 3) points in world frame that fall within the mask
        selected_colors: (M, 3) colors for selected points
    """
    # print(f"[DEBUG] extract_points_from_vggt_pointcloud called")
    # print(f"[DEBUG] Input vggt_points shape: {vggt_points.shape if vggt_points is not None else None}")
    # print(f"[DEBUG] Input mask_2d shape: {mask_2d.shape}")
    # print(f"[DEBUG] Mask non-zero pixels: {np.sum(mask_2d > 0)} / {mask_2d.size} ({100*np.sum(mask_2d > 0)/mask_2d.size:.1f}%)")
    
    # Debug: show mask
    # try:
    #     plt.figure(figsize=(10, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(mask_2d, cmap='gray')
    #     plt.title(f'Input Mask ({np.sum(mask_2d > 0)} pixels)')
    #     plt.colorbar()
        
    #     # Show mask statistics
    #     plt.subplot(1, 2, 2)
    #     plt.hist(mask_2d.flatten(), bins=50, alpha=0.7)
    #     plt.title('Mask Value Distribution')
    #     plt.xlabel('Pixel Value')
    #     plt.ylabel('Count')
    #     plt.show()
    # except Exception as e:
    #     print(f"[DEBUG] Failed to show mask: {e}")
    
    if vggt_points is None or len(vggt_points) == 0:
        # print(f"[DEBUG] No VGGT points available")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Convert mask to boolean
    if mask_2d.dtype == np.uint8:
        mask_bool = mask_2d > 127
    else:
        mask_bool = mask_2d.astype(bool)
    
    # print(f"[DEBUG] Boolean mask has {np.sum(mask_bool)} True pixels")
    
    # Transform points from world to camera frame
    T_c_w = np.linalg.inv(T_w_c)
    R_c_w = T_c_w[:3, :3]
    t_c_w = T_c_w[:3, 3]
    
    # print(f"[DEBUG] Camera position in world: {T_w_c[:3, 3]}")
    # print(f"[DEBUG] World points range: X[{vggt_points[:, 0].min():.3f}, {vggt_points[:, 0].max():.3f}], Y[{vggt_points[:, 1].min():.3f}, {vggt_points[:, 1].max():.3f}], Z[{vggt_points[:, 2].min():.3f}, {vggt_points[:, 2].max():.3f}]")
    
    # Transform to camera frame
    points_cam = (R_c_w @ vggt_points.T).T + t_c_w
    # print(f"[DEBUG] Camera frame points range: X[{points_cam[:, 0].min():.3f}, {points_cam[:, 0].max():.3f}], Y[{points_cam[:, 1].min():.3f}, {points_cam[:, 1].max():.3f}], Z[{points_cam[:, 2].min():.3f}, {points_cam[:, 2].max():.3f}]")
    
    # Filter points behind camera
    valid_depth = points_cam[:, 2] > 0.01
    # print(f"[DEBUG] Points in front of camera: {np.sum(valid_depth)} / {len(points_cam)} ({100*np.sum(valid_depth)/len(points_cam):.1f}%)")
    
    points_cam_valid = points_cam[valid_depth]
    colors_valid = vggt_colors[valid_depth] if vggt_colors is not None else None
    world_points_valid = vggt_points[valid_depth]
    
    if len(points_cam_valid) == 0:
        # print(f"[DEBUG] No points in front of camera!")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Project to image plane
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # print(f"[DEBUG] Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    x_proj = points_cam_valid[:, 0] * fx / points_cam_valid[:, 2] + cx
    y_proj = points_cam_valid[:, 1] * fy / points_cam_valid[:, 2] + cy
    
    # print(f"[DEBUG] Projected points range (original): X[{x_proj.min():.1f}, {x_proj.max():.1f}], Y[{y_proj.min():.1f}, {y_proj.max():.1f}]")
    
    # Detect resolution mismatch and scale projection accordingly
    H, W = mask_2d.shape
    # print(f"[DEBUG] Mask resolution: {W}×{H}")
    
    # Estimate original image size from projection bounds and intrinsics
    proj_width = x_proj.max() - x_proj.min() + 2*cx
    proj_height = y_proj.max() - y_proj.min() + 2*cy
    
    # Detect if we need to scale up the projection
    scale_x = W / (2*cx) if cx > 0 else W / proj_width
    scale_y = H / (2*cy) if cy > 0 else H / proj_height
    
    # Use the smaller scale to maintain aspect ratio, but only if it's significantly different
    if scale_x > 1.5 or scale_y > 1.5:
        scale = min(scale_x, scale_y)
        # print(f"[DEBUG] Detected resolution mismatch. Scaling projection by {scale:.2f}")
        # print(f"[DEBUG] Estimated original size: {2*cx:.0f}×{2*cy:.0f}, Target size: {W}×{H}")
        
        # Scale the projection
        x_proj = x_proj * scale
        y_proj = y_proj * scale
        
        # print(f"[DEBUG] Projected points range (scaled): X[{x_proj.min():.1f}, {x_proj.max():.1f}], Y[{y_proj.min():.1f}, {y_proj.max():.1f}]")
    else:
        # print(f"[DEBUG] No scaling needed (scale factors: {scale_x:.2f}, {scale_y:.2f})")
        pass
    # Convert to integer pixel coordinates
    x_pix = np.round(x_proj).astype(int)
    y_pix = np.round(y_proj).astype(int)
    
    # Filter points within image bounds
    # print(f"[DEBUG] Image bounds: H={H}, W={W}")
    in_bounds = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
    # print(f"[DEBUG] Points within image bounds: {np.sum(in_bounds)} / {len(x_pix)} ({100*np.sum(in_bounds)/len(x_pix):.1f}%)")
    
    x_pix_valid = x_pix[in_bounds]
    y_pix_valid = y_pix[in_bounds]
    world_points_in_bounds = world_points_valid[in_bounds]
    colors_in_bounds = colors_valid[in_bounds] if colors_valid is not None else None
    
    if len(x_pix_valid) == 0:
        # print(f"[DEBUG] No points within image bounds!")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Debug: visualize projection overlay
#     try:
#         plt.figure(figsize=(15, 10))
        
#         # Show mask with projected points (full resolution)
#         plt.subplot(2, 3, 1)
#         plt.imshow(mask_2d, cmap='gray', alpha=0.7)
#         plt.scatter(x_pix_valid, y_pix_valid, c='red', s=1, alpha=0.8)
#         plt.title(f'Full Mask + Projected Points\n({len(x_pix_valid)} points)')
#         plt.xlim(0, W)
#         plt.ylim(H, 0)  # Flip Y axis for image coordinates
        
#         # Show just the projected points (full resolution)
#         plt.subplot(2, 3, 2)
#         plt.scatter(x_pix_valid, y_pix_valid, c='blue', s=1, alpha=0.8)
#         plt.xlim(0, W)
#         plt.ylim(H, 0)
#         plt.title(f'Projected Points Only\n(Full {W}×{H})')
#         plt.grid(True, alpha=0.3)
        
#         # Zoomed view around projected points
#         plt.subplot(2, 3, 3)
#         if len(x_pix_valid) > 0:
#             x_min, x_max = x_pix_valid.min(), x_pix_valid.max()
#             y_min, y_max = y_pix_valid.min(), y_pix_valid.max()
            
#             # Add padding around the projection region
#             pad_x = max(50, (x_max - x_min) * 0.2)
#             pad_y = max(50, (y_max - y_min) * 0.2)
            
#             x_zoom_min = max(0, int(x_min - pad_x))
#             x_zoom_max = min(W, int(x_max + pad_x))
#             y_zoom_min = max(0, int(y_min - pad_y))
#             y_zoom_max = min(H, int(y_max + pad_y))
            
#             mask_crop = mask_2d[y_zoom_min:y_zoom_max, x_zoom_min:x_zoom_max]
#             plt.imshow(mask_crop, cmap='gray', alpha=0.7, extent=[x_zoom_min, x_zoom_max, y_zoom_max, y_zoom_min])
#             plt.scatter(x_pix_valid, y_pix_valid, c='red', s=2, alpha=0.8)
#             plt.xlim(x_zoom_min, x_zoom_max)
#             plt.ylim(y_zoom_max, y_zoom_min)
#             plt.title(f'Zoomed: Projection Region\n[{x_zoom_min}:{x_zoom_max}, {y_zoom_min}:{y_zoom_max}]')
#         else:
#             plt.text(0.5, 0.5, 'No points to zoom', ha='center', va='center', transform=plt.gca().transAxes)
#             plt.title('No Points Available')
        
#         # Show mask statistics by region
#         plt.subplot(2, 3, 4)
#         if len(x_pix_valid) > 0:
#             # Divide image into grid and show mask density
#             grid_h, grid_w = 10, 10
#             mask_density = np.zeros((grid_h, grid_w))
            
#             for i in range(grid_h):
#                 for j in range(grid_w):
#                     y_start = int(i * H / grid_h)
#                     y_end = int((i + 1) * H / grid_h)
#                     x_start = int(j * W / grid_w)
#                     x_end = int((j + 1) * W / grid_w)
                    
#                     region_mask = mask_2d[y_start:y_end, x_start:x_end]
#                     mask_density[i, j] = np.sum(region_mask > 0) / region_mask.size
            
#             plt.imshow(mask_density, cmap='hot', interpolation='nearest')
#             plt.colorbar(label='Mask Density')
#             plt.title('Mask Density by Region\n(10×10 grid)')
            
#             # Mark where projections land
#             proj_grid_x = x_pix_valid * grid_w / W
#             proj_grid_y = y_pix_valid * grid_h / H
#             plt.scatter(proj_grid_x, proj_grid_y, c='cyan', s=5, alpha=0.7, marker='x')
#         else:
#             plt.text(0.5, 0.5, 'No projection\ndata available', ha='center', va='center', transform=plt.gca().transAxes)
        
#         # Show histogram of mask values in projection region
#         plt.subplot(2, 3, 5)
#         if len(x_pix_valid) > 0:
#             mask_values_at_proj = mask_2d[y_pix_valid, x_pix_valid]
#             plt.hist(mask_values_at_proj, bins=50, alpha=0.7, color='red', label='At projections')
#             plt.hist(mask_2d.flatten(), bins=50, alpha=0.5, color='blue', label='Full mask')
#             plt.xlabel('Mask Value')
#             plt.ylabel('Count')
#             plt.title('Mask Values:\nAt Projections vs Full Mask')
#             plt.legend()
#         else:
#             plt.text(0.5, 0.5, 'No projection\nhistogram available', ha='center', va='center', transform=plt.gca().transAxes)
        
#         # Show overall statistics
#         plt.subplot(2, 3, 6)
#         stats_text = f"""Resolution Analysis:
# Mask: {W} × {H}
# Intrinsics: cx={cx:.1f}, cy={cy:.1f}
# Estimated orig: {2*cx:.0f} × {2*cy:.0f}

# Projection Stats:
# Total points: {len(vggt_points)}
# In front: {len(points_cam_valid)}
# In bounds: {len(x_pix_valid)}
# In mask: {np.sum(mask_values) if len(x_pix_valid) > 0 else 0}

# Scale factors:
# X: {W/(2*cx):.2f}
# Y: {H/(2*cy):.2f}"""
        
#         plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
#                 fontfamily='monospace', fontsize=9, verticalalignment='top')
#         plt.axis('off')
#         plt.title('Debug Statistics')
        
#         plt.tight_layout()
#         plt.show()
#     except Exception as e:
#         print(f"[DEBUG] Failed to show projection overlay: {e}")
    
    # Check which points fall within the mask
    mask_values = mask_bool[y_pix_valid, x_pix_valid]
    selected_indices = mask_values
    # print(f"[DEBUG] Points within mask: {np.sum(selected_indices)} / {len(mask_values)} ({100*np.sum(selected_indices)/len(mask_values):.1f}%)")
    
    selected_points = world_points_in_bounds[selected_indices]
    selected_colors = colors_in_bounds[selected_indices] if colors_in_bounds is not None else None
    
    # print(f"[DEBUG] Selected points before sampling: {len(selected_points)}")
    
    # Sample if too many points
    if max_points is not None and len(selected_points) > max_points:
        rng = np.random.default_rng(random_state)
        sample_indices = rng.choice(len(selected_points), max_points, replace=False)
        selected_points = selected_points[sample_indices]
        if selected_colors is not None:
            selected_colors = selected_colors[sample_indices]
        # print(f"[DEBUG] Sampled down to: {len(selected_points)} points")
    
    if selected_colors is None:
        selected_colors = np.array([]).reshape(0, 3)
    
    # print(f"[DEBUG] Final result: {len(selected_points)} points, {len(selected_colors)} colors")
    
    # Apply statistical outlier removal using Open3D
    if len(selected_points) > 0:
        try:
            import open3d as o3d
            
            # print(f"[DEBUG] Applying statistical outlier removal...")
            # print(f"[DEBUG] Parameters: nb_neighbors={o3_nb_neighbors}, std_ratio={o3_std_ratio}")
            
            # Create temporary point cloud for outlier removal
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(selected_points.astype(np.float64))
            
            # Apply statistical outlier removal
            cl, ind = temp_pcd.remove_statistical_outlier(nb_neighbors=o3_nb_neighbors, std_ratio=o3_std_ratio)
            
            # Filter points and colors based on inlier indices
            if len(ind) > 0:
                selected_points = selected_points[ind]
                if selected_colors is not None and len(selected_colors) > 0:
                    selected_colors = selected_colors[ind]
                
                # print(f"[DEBUG] After outlier removal: {len(selected_points)} points ({len(ind)}/{len(temp_pcd.points)} kept, {len(temp_pcd.points)-len(ind)} outliers removed)")
            else:
                pass
                # print(f"[DEBUG] All points removed as outliers! Keeping original selection.")
                # Keep original points if all are considered outliers
            
        except ImportError:
            pass
            # print(f"[DEBUG] Open3D not available for outlier removal, skipping")
        except Exception as e:
            pass
            # print(f"[DEBUG] Failed to apply outlier removal: {e}, keeping original points")
    else:
        pass
        # print(f"[DEBUG] No points for outlier removal")
    
    # print(f"[DEBUG] Final result after outlier removal: {len(selected_points)} points, {len(selected_colors)} colors")
    
    # Debug: visualize final selection
    # if len(selected_points) > 0:
    #     try:
    #         import open3d as o3d
            
    #         print(f"[DEBUG] Visualizing selected points...")
    #         pcd = o3d.geometry.PointCloud()
    #         pcd.points = o3d.utility.Vector3dVector(selected_points.astype(np.float64))
            
    #         if len(selected_colors) > 0:
    #             if selected_colors.max() > 1.0:
    #                 colors_norm = selected_colors / 255.0
    #             else:
    #                 colors_norm = selected_colors
    #             pcd.colors = o3d.utility.Vector3dVector(colors_norm.astype(np.float64))
    #         else:
    #             pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red for extracted points
            
    #         # Add camera frame for reference
    #         camera_pos = T_w_c[:3, 3]
    #         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    #         camera_frame.translate(camera_pos)
    #         R_w_c = T_w_c[:3, :3]
    #         camera_frame.rotate(R_w_c, center=camera_pos)
            
    #         o3d.visualization.draw_geometries(
    #             [pcd, camera_frame],
    #             window_name=f"Extracted Points ({len(selected_points)} points)",
    #             width=800,
    #             height=600
    #         )
            
    #     except Exception as e:
    #         print(f"[DEBUG] Failed to visualize selected points: {e}")
    # else:
    #     print(f"[DEBUG] No points to visualize!")
    
    return selected_points, selected_colors

def main(cfg):
    rgb_dir_path = cfg.rgb_dir
    frame_data_dir = cfg.vggt_frame_data_dir  # New: VGGT frame data directory
    max_points_per_obj = int(cfg.max_points_per_obj)

    persistent_graph = nx.MultiDiGraph()

    # Metrics accumulation
    timings = {'yolo': [], 'preprocess': [], 'create_3d': [], 'edges': [], 'merge': []}
    gpu_usage = {'yolo': [], 'edges': []}
    cuda_available = torch.cuda.is_available()

    # Get list of available RGB images for YOLO processing
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir_path, "*.png")) + 
                      glob.glob(os.path.join(rgb_dir_path, "*.jpg")))
    
    if len(rgb_files) == 0:
        raise ValueError(f"No RGB images found in {rgb_dir_path}")
    
    print(f"Found {len(rgb_files)} RGB images")

    # Initialize YOLO model
    from ultralytics import YOLOE
    model = YOLOE(cfg.yolo_model)
    
    frame_idx = 0
    for rgb_file in rgb_files:
        if frame_idx >= cfg.max_frames:  # Limit for testing
            break
            
        print(f"\n[INFO] Processing frame {frame_idx}: {os.path.basename(rgb_file)}")
        
        # Load VGGT frame data
        camera_params, vggt_points, vggt_colors = load_vggt_frame_data(frame_data_dir, frame_idx)
        
        if camera_params is None:
            print(f"[WARN] No VGGT data for frame {frame_idx}, skipping")
            frame_idx += 1
            continue
        
        if vggt_points is None or len(vggt_points) == 0:
            print(f"[WARN] No point cloud data for frame {frame_idx}, skipping")
            frame_idx += 1
            continue
        
        # Extract camera parameters
        intrinsic = np.array(camera_params['intrinsic'])
        T_w_c = np.array(camera_params['extrinsic_Twc'])  # World-from-camera
        
        print(f"[INFO] Frame {frame_idx}: {len(vggt_points)} VGGT points, camera at {camera_params['camera_position']}")
        
        # Run YOLO on RGB image
        t_yolo_start = time.perf_counter()
        rgb_file = cv2.imread(rgb_file)
        rgb_file = cv2.cvtColor(rgb_file, cv2.COLOR_BGR2RGB)
        results = model.track(rgb_file, conf=cfg.conf, iou=cfg.iou, persist=True)
        yl_res = results[0] if results else None
        t_yolo_end = time.perf_counter()
        
        if yl_res is None:
            print(f"[WARN] YOLO failed for frame {frame_idx}, skipping")
            frame_idx += 1
            continue
        
        timings['yolo'].append((t_yolo_end - t_yolo_start) * 1000)  # ms

        # GPU usage after YOLO (if available)
        if cuda_available:
            torch.cuda.synchronize()
            gpu_mem_yolo = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_usage['yolo'].append(gpu_mem_yolo)

        # plt.imshow(yl_res.orig_img)
        # plt.show()
        
        # Process masks with morphological operations
        t_preprocess_start = time.perf_counter()
        mask_org, masks_clean = yutils.preprocess_mask(
            yolo_res=yl_res,
            index=frame_idx,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            show=cfg.show_mask_processing,
            fast=cfg.fast_mask,
        )
        t_preprocess_end = time.perf_counter()
        timings['preprocess'].append((t_preprocess_end - t_preprocess_start) * 1000)

        # Get track IDs
        track_ids = None
        if hasattr(yl_res, 'boxes') and yl_res.boxes is not None and getattr(yl_res.boxes, 'id', None) is not None:
            try:
                track_ids = yl_res.boxes.id.detach().cpu().numpy().astype(np.int64)
            except Exception:
                pass
        if track_ids is None:
            # fallback: sequential ids up to number of masks
            n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
            track_ids = np.arange(n, dtype=np.int64)

        print(f"[INFO] Frame {frame_idx}: Found {len(track_ids)} objects with track IDs: {track_ids}")

        # Build 3D objects using VGGT point cloud
        t_create3d_start = time.perf_counter()
        frame_objs = []
        graph = nx.MultiDiGraph()
        
        obj_counter = 0
        for t_id, mask_clean in zip(track_ids, masks_clean):
            
            if 'wall' in yl_res.names[int(yl_res.boxes.cls[obj_counter])]:
                obj_counter += 1
                continue
            if 'counter' in yl_res.names[int(yl_res.boxes.cls[obj_counter])]:
                obj_counter += 1
                continue
            if 'floor' in yl_res.names[int(yl_res.boxes.cls[obj_counter])]:
                obj_counter += 1
                continue
            if 'ceiling' in yl_res.names[int(yl_res.boxes.cls[obj_counter])]:
                obj_counter += 1
                continue
            if 'restroom' in yl_res.names[int(yl_res.boxes.cls[obj_counter])]:
                obj_counter += 1
                continue
            if 'faucet' in yl_res.names[int(yl_res.boxes.cls[obj_counter])]:
                obj_counter += 1
                continue
        
            # Extract points from VGGT point cloud using 2D mask projection
            obj_points, obj_colors = extract_points_from_vggt_pointcloud(
                vggt_points, vggt_colors, mask_clean, intrinsic, T_w_c,
                max_points=max_points_per_obj, random_state=42,
                o3_nb_neighbors=cfg.o3_nb_neighbors, o3_std_ratio=cfg.o3_std_ratio
            )
            
            if len(obj_points) == 0:
                print(f"[WARN] No points found for track ID {t_id}")
                continue
            
            # Compute 3D bounding box
            bbox3d = yutils.compute_3d_bboxes(obj_points, fast_mode=cfg.fast_bbox)
            
            # Create object dictionary
            obj = {
                'track_id': int(t_id),
                'label': yl_res.names[int(yl_res.boxes.cls[obj_counter])],
                'points': obj_points,
                'colors': obj_colors,
                'bbox_3d': bbox3d,
                'num_points': len(obj_points)
            }
            
            print(f"[DEBUG] Object {t_id}: Label={obj['label']}, Num points={obj['num_points']}")
            
            # Add to graph and frame objects
            graph.add_node(int(t_id), data=obj)
            frame_objs.append(obj)
            
            obj_counter += 1
            # print(f"[INFO] Track ID {t_id}: {len(obj_points)} points extracted")
        
        current_graph = graph
        t_create3d_end = time.perf_counter()
        timings['create_3d'].append((t_create3d_end - t_create3d_start) * 1000)

        # Edge predictor SceneVerse
        t_edges_start = time.perf_counter()
        # Create a dummy depth map for compatibility with edges function
        # Since we're using VGGT point clouds, we don't need actual depth maps
        dummy_depth = np.zeros((480, 640), dtype=np.float32)  # Placeholder
        triplet_rels, middle_rels = edges(current_graph, frame_objs, T_w_c, dummy_depth)
        
        # save to json
        relations_json = []
        for src_id, tgt_id, rel_type in triplet_rels:
            relations_json.append({
                'source_id': int(src_id),
                'source_label': current_graph.nodes[int(src_id)]['data']['label'],
                'target_id': int(tgt_id),
                'target_label': current_graph.nodes[int(tgt_id)]['data']['label'],
                'relation': rel_type
            })

        # for src_id, tgt_id, rel_type in aligned_rels:
        #     relations_json.append({
        #         'source_id': int(src_id),
        #         'source_label': current_graph.nodes[int(src_id)]['data']['label'],
        #         'target_id': int(tgt_id),
        #         'target_label': current_graph.nodes[int(tgt_id)]['data']['label'],
        #         'relation': rel_type
        #     })

        if len(middle_rels) > 0:
            for [src, tg1, tg2], rel_mid in middle_rels:
                relations_json.append({
                    'source_id': int(src),
                    'source_label': current_graph.nodes[int(src)]['data']['label'],
                    'target1_id': int(tg1),
                'target1_label': current_graph.nodes[int(tg1)]['data']['label'],
                'target2_id': int(tg2),
                'target2_label': current_graph.nodes[int(tg2)]['data']['label'],
                'relation': rel_mid,
            })

        # save json
        with open(f"/home/rizo/mipt_ccm/yolo_ssg/json_kitchen/relations_frame_{frame_idx+30}.json", "w") as f:
            json.dump(relations_json, f, indent=4)

        t_edges_end = time.perf_counter()
        timings['edges'].append((t_edges_end - t_edges_start) * 1000)
        
        # GPU usage after edges (if available)
        if cuda_available:
            torch.cuda.synchronize()
            gpu_mem_edges = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_usage['edges'].append(gpu_mem_edges)

        if cfg.print_resource_usage:
            print(50*'=')
            print(f"[yolo_sgg] Frame {frame_idx}: Latency (ms) - preprocess: {timings['preprocess'][-1]:.2f}, create_3d: {timings['create_3d'][-1]:.2f}, edges: {timings['edges'][-1]:.2f}, yolo: {timings['yolo'][-1]:.2f}")
            if cuda_available:
                print(f"[yolo_ssg] Frame {frame_idx}: GPU mem (MB) - yolo: {gpu_usage['yolo'][-1]:.1f}, "
                    f"edges: {gpu_usage['edges'][-1]:.1f}")
            print(50*'=')
        
        if bool(cfg.show_pcds):
            # visualize (blocking window) - objects are in world coordinates
            # Pass camera transform to show actual camera position
            yutils.visualize_frame_objects_open3d_world(frame_objs, frame_idx, T_w_c)
        
        # Save rendered frame from camera perspective
        if bool(cfg.save_rendered_frames) and T_w_c is not None:
            yutils.render_and_save_frame_from_camera(
                frame_objs=frame_objs,
                frame_index=frame_idx,
                T_w_c=T_w_c,
                output_dir=cfg.render_output_dir,
                show_points=True,
                show_aabb=cfg.render_show_aabb,
                show_obb=cfg.render_show_obb,
                width=cfg.render_width,
                height=cfg.render_height,
                point_size=cfg.render_point_size,
                graph=current_graph,
                show_edges=cfg.render_show_edges,
            )

        t_merge_start = time.perf_counter()
        persistent_graph = yutils.merge_scene_graphs(persistent_graph, current_graph)
        t_merge_end = time.perf_counter()
        timings['merge'].append((t_merge_end - t_merge_start) * 1000)
        
        # Visualize scene graph
        if cfg.vis_graph:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # Persistent graph
            yutils.draw_labeled_multigraph(persistent_graph, ax=axes[0])
            axes[0].set_title(f"Persistent Graph frame: {frame_idx}")

            # Current graph
            yutils.draw_labeled_multigraph(current_graph, ax=axes[1])
            axes[1].set_title(f"Current Graph frame: {frame_idx}")

            plt.tight_layout()
            plt.show()

        frame_idx += 1
        
        print(f"[INFO] Completed frame {frame_idx-1}, found {len(frame_objs)} objects")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if timings['preprocess']:
        print(f"\nLatency Averages (ms):")
        print(f"  Preprocessing:    {np.mean(timings['preprocess']):.2f} ± {np.std(timings['preprocess']):.2f}")
        print(f"  Create 3D:        {np.mean(timings['create_3d']):.2f} ± {np.std(timings['create_3d']):.2f}")
        print(f"  Edge Prediction:  {np.mean(timings['edges']):.2f} ± {np.std(timings['edges']):.2f}")
        print(f"  YOLO:             {np.mean(timings['yolo']):.2f} ± {np.std(timings['yolo']):.2f}")
        print(f"  Merge:           {np.mean(timings['merge']):.2f} ± {np.std(timings['merge']):.2f}")
        total_avg = np.mean(timings['preprocess']) + np.mean(timings['create_3d']) + np.mean(timings['edges']) + np.mean(timings['yolo']) + np.mean(timings['merge'])
        print(f"  Total per frame:  {total_avg:.2f}")
    
    if cuda_available and gpu_usage['yolo']:
        print(f"\nGPU Memory Usage Averages (MB):")
        print(f"  After YOLO:       {np.mean(gpu_usage['yolo']):.1f} ± {np.std(gpu_usage['yolo']):.1f}")
        print(f"  After Edges:      {np.mean(gpu_usage['edges']):.1f} ± {np.std(gpu_usage['edges']):.1f}")
    elif cuda_available:
        print(f"\nGPU Memory: CUDA available but no measurements recorded")
    else:
        print(f"\nGPU Memory: CUDA not available")
    
    print(f"\nTotal frames processed: {frame_idx}")
    print("="*60)
    return 0


if __name__ == "__main__":
    cfg = OmegaConf.create({
        # Input paths
        'rgb_dir': "/home/rizo/mipt_ccm/yolo_ssg/robotics_kitchen_dataset_v3/rgb_last",  # RGB images for YOLO
        'vggt_frame_data_dir': "/home/rizo/mipt_ccm/yolo_ssg/robotics_kitchen_dataset_v3/frame_data_vggt_last",  # VGGT predictions
        
        # YOLO settings
        'yolo_model': 'yoloe-11l-seg-pf-new.pt',
        'conf': 0.3,
        'iou': 0.5,
        
        # Mask processing
        'kernel_size': 15,
        'alpha': 0.7,
        'fast_mask': True,
        'show_mask_processing': False,
        
        # 3D processing
        'max_points_per_obj': 2000,
        'fast_bbox': True,
        
        # Outlier removal parameters
        'o3_nb_neighbors': 20,    # Number of neighbors for statistical outlier removal
        'o3_std_ratio': 2.0,      # Standard deviation ratio threshold
        
        # Processing limits
        'max_frames': 50,  # Limit number of frames for testing
        
        # Visualization
        'show_pcds': False,  # Set to True to see 3D point clouds
        'vis_graph': False,   # Set to True to see scene graphs
        'print_resource_usage': True,
        
        # Rendering options
        'save_rendered_frames': False,
        'render_output_dir': '/home/rizo/mipt_ccm/yolo_ssg/rendered_frames_vggt',
        'render_show_aabb': True,
        'render_show_obb': False,
        'render_show_edges': True,
        'render_width': 1280,
        'render_height': 720,
        'render_point_size': 2.0,
    })
    main(cfg)