import os
import torch
import numpy as np
import cv2
from PIL import Image
import glob
import time
from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor


def process_depth_model(cfg):
    """
    Process depth and trajectory using specified depth model.
    
    Args:
        cfg: Config object with fields:
            - rgb_dir: Path to RGB images directory
            - depth_dir: Path to original depth directory
            - traj_path: Path to original trajectory file
            - depth_model: Depth model name (e.g., 'yyfz233/Pi3X') or None
    
    Returns:
        cfg: Updated config with new depth_dir and traj_path if model was used
    """
    depth_model_list = ['yyfz233/Pi3X']
    
    if cfg.depth_model is None:
        print('Use original depth and traj')
        return cfg
    else:
        if cfg.depth_model not in depth_model_list:
            raise ValueError(f"Cannot use {cfg.depth_model} depth model, you should choose one from {depth_model_list}")
        
    print("\n" + "="*60)
    print(f"DEPTH MODEL PROCESSING: {cfg.depth_model}")
    print("="*60)
    
    # Performance tracking
    timings = {}
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        # Reset peak memory stats to get accurate measurements for this run
        torch.cuda.reset_peak_memory_stats()
    
    # --- Setup Device and Model ---
    t_model_start = time.perf_counter()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"Loading model on {device}...")
    model = Pi3X.from_pretrained(cfg.depth_model).to(device).eval()
    t_model_end = time.perf_counter()
    timings['model_load'] = (t_model_end - t_model_start) * 1000  # ms
    
    if cuda_available:
        torch.cuda.synchronize()
        gpu_mem_after_load = torch.cuda.memory_allocated() / (1024**2)  # MB
        gpu_mem_peak_after_load = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        print(f"GPU memory after model load: {gpu_mem_after_load:.1f} MB (peak: {gpu_mem_peak_after_load:.1f} MB)")
    
    # --- Load Images ---
    t_load_start = time.perf_counter()
    print(f"Loading images from: {cfg.rgb_dir}")
    imgs = load_images_as_tensor(cfg.rgb_dir, interval=1).to(device)
    t_load_end = time.perf_counter()
    timings['image_load'] = (t_load_end - t_load_start) * 1000  # ms
    
    # --- Inference ---
    t_inference_start = time.perf_counter()
    print("Running model inference...")
    dtype = torch.bfloat16 if cuda_available and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            results = model(imgs[None])
    
    if cuda_available:
        torch.cuda.synchronize()
    t_inference_end = time.perf_counter()
    timings['inference'] = (t_inference_end - t_inference_start) * 1000  # ms
    
    if cuda_available:
        gpu_mem_after_inference = torch.cuda.memory_allocated() / (1024**2)  # MB
        gpu_mem_peak_after_inference = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        print(f"GPU memory after inference: {gpu_mem_after_inference:.1f} MB (peak: {gpu_mem_peak_after_inference:.1f} MB)")
    
    print("Reconstruction complete!")
    
    # --- Extract Depth Maps ---
    t_extract_start = time.perf_counter()
    depth_maps = results['local_points'][..., 2]  # Shape: (1, N, H, W)
    depth_maps = depth_maps[0]  # Remove batch dimension -> (N, H, W)
    print(f"Extracted depth maps with shape: {depth_maps.shape}")
    t_extract_end = time.perf_counter()
    timings['extract_depth'] = (t_extract_end - t_extract_start) * 1000  # ms
    
    # --- Create Output Directories ---
    t_create_dirs_start = time.perf_counter()
    # Create pi3_depth directory near original depth folder
    depth_parent_dir = os.path.dirname(cfg.depth_dir)
    new_depth_dir = os.path.join(depth_parent_dir, 'pi3_depth')
    os.makedirs(new_depth_dir, exist_ok=True)
    
    # Create pi3_traj path near original trajectory file
    traj_parent_dir = os.path.dirname(cfg.traj_path)
    traj_basename = os.path.basename(cfg.traj_path)
    traj_name, traj_ext = os.path.splitext(traj_basename)
    new_traj_path = os.path.join(traj_parent_dir, f'pi3_{traj_name}{traj_ext}')
    t_create_dirs_end = time.perf_counter()
    timings['create_dirs'] = (t_create_dirs_end - t_create_dirs_start) * 1000  # ms
    
    # --- Convert to CPU and Numpy ---
    t_convert_start = time.perf_counter()
    depth_maps_np = depth_maps.cpu().numpy()
    t_convert_end = time.perf_counter()
    timings['convert_cpu'] = (t_convert_end - t_convert_start) * 1000  # ms
    
    # --- Get Target Resolution from First RGB Image ---
    t_resize_start = time.perf_counter()
    # Load first image from rgb_dir to get original resolution
    rgb_image_files = sorted(glob.glob(os.path.join(cfg.rgb_dir, '*.[jp][pn][g]')))  # Matches .jpg, .jpeg, .png
    if len(rgb_image_files) == 0:
        raise ValueError(f"No images found in {cfg.rgb_dir}")
    
    first_rgb = cv2.imread(rgb_image_files[0])
    target_height, target_width = first_rgb.shape[:2]
    print(f"Target resolution from first RGB image: {target_height}x{target_width}")
    
    # --- Upscale Depth Maps to Match RGB Resolution ---
    upscaled_depth_maps = []
    for i in range(depth_maps_np.shape[0]):
        depth = depth_maps_np[i]
        depth_upscaled = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        upscaled_depth_maps.append(depth_upscaled)
    
    depth_maps_np = np.stack(upscaled_depth_maps, axis=0)
    t_resize_end = time.perf_counter()
    timings['resize'] = (t_resize_end - t_resize_start) * 1000  # ms
    print(f"Upscaled depth maps to shape: {depth_maps_np.shape}")
    
    # --- Compute Global Min/Max for Consistent Normalization ---
    t_normalize_start = time.perf_counter()
    valid_mask_all = depth_maps_np > 0
    if valid_mask_all.sum() > 0:
        global_depth_min = depth_maps_np[valid_mask_all].min()
        global_depth_max = depth_maps_np[valid_mask_all].max()
        print(f"Global depth range: [{global_depth_min:.3f}, {global_depth_max:.3f}]")
    else:
        global_depth_min = 0
        global_depth_max = 1
    t_normalize_end = time.perf_counter()
    timings['normalize_compute'] = (t_normalize_end - t_normalize_start) * 1000  # ms
    
    # --- Save Depth as PNG Images ---
    t_save_png_start = time.perf_counter()
    print(f"Saving depth images to: {new_depth_dir}")
    for i in range(depth_maps_np.shape[0]):
        depth = depth_maps_np[i]
        
        # Normalize depth to 0-255 range using global min/max
        valid_mask = depth > 0
        depth_normalized = np.zeros_like(depth)
        if valid_mask.sum() > 0:
            depth_normalized[valid_mask] = (depth[valid_mask] - global_depth_min) / (global_depth_max - global_depth_min) * 255
            depth_normalized = np.clip(depth_normalized, 0, 255)
        
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # Save as grayscale PNG
        img = Image.fromarray(depth_uint8, mode='L')
        output_path = os.path.join(new_depth_dir, f'depth{i:06d}.png')
        img.save(output_path)
    
    t_save_png_end = time.perf_counter()
    timings['save_png'] = (t_save_png_end - t_save_png_start) * 1000  # ms
    print(f"Saved {depth_maps_np.shape[0]} depth images")
    
    # --- Save Depth Video ---
    t_save_video_start = time.perf_counter()
    video_path = os.path.join(new_depth_dir, 'depth_video.mp4')
    H, W = depth_maps_np.shape[1], depth_maps_np.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (W, H), isColor=False)
    
    for i in range(depth_maps_np.shape[0]):
        depth = depth_maps_np[i]
        
        valid_mask = depth > 0
        depth_normalized = np.zeros_like(depth)
        if valid_mask.sum() > 0:
            depth_normalized[valid_mask] = (depth[valid_mask] - global_depth_min) / (global_depth_max - global_depth_min) * 255
            depth_normalized = np.clip(depth_normalized, 0, 255)
        
        depth_uint8 = depth_normalized.astype(np.uint8)
        video_writer.write(depth_uint8)
    
    video_writer.release()
    t_save_video_end = time.perf_counter()
    timings['save_video'] = (t_save_video_end - t_save_video_start) * 1000  # ms
    print(f"Depth video saved to: {video_path}")
    
    # --- Save Camera Poses (Trajectory) ---
    t_save_traj_start = time.perf_counter()
    camera_poses = results['camera_poses'][0]  # Remove batch dimension -> (N, 4, 4)
    
    with open(new_traj_path, 'w') as f:
        for i in range(camera_poses.shape[0]):
            pose = camera_poses[i].cpu().numpy()  # Shape: (4, 4)
            flattened = pose.flatten()  # Shape: (16,)
            f.write(' '.join(map(str, flattened)) + '\n')
    
    print(f"Camera poses saved to: {new_traj_path}")
    print(f"Format: {camera_poses.shape[0]} poses, each row contains 16 values (flattened 4x4 matrix)")
    t_save_traj_end = time.perf_counter()
    timings['save_traj'] = (t_save_traj_end - t_save_traj_start) * 1000  # ms
    
    # --- Update Config Paths ---
    cfg.depth_dir = new_depth_dir
    cfg.traj_path = new_traj_path
    
    print(f"\nConfig updated:")
    print(f"  depth_dir: {cfg.depth_dir}")
    print(f"  traj_path: {cfg.traj_path}")
    
    # --- Print Performance Statistics ---
    print("\n" + "="*60)
    print("DEPTH MODEL PERFORMANCE STATISTICS")
    print("="*60)
    
    print(f"\nLatency Breakdown (ms):")
    print(f"  Model Load:        {timings['model_load']:.2f}")
    print(f"  Image Load:        {timings['image_load']:.2f}")
    print(f"  Inference:         {timings['inference']:.2f}")
    # print(f"  Extract Depth:     {timings['extract_depth']:.2f}")
    # print(f"  Convert to CPU:    {timings['convert_cpu']:.2f}")
    # print(f"  Resize:            {timings['resize']:.2f}")
    # print(f"  Compute Norm:      {timings['normalize_compute']:.2f}")
    # print(f"  Save PNGs:         {timings['save_png']:.2f}")
    # print(f"  Save Video:        {timings['save_video']:.2f}")
    # print(f"  Save Trajectory:   {timings['save_traj']:.2f}")
    
    total_time = sum(timings.values())
    print(f"  Total Time:        {total_time:.2f}")
    
    if cuda_available:
        print(f"\nGPU Memory Usage (MB):")
        print(f"  After Model Load:  {gpu_mem_after_load:.1f} (current), {gpu_mem_peak_after_load:.1f} (peak)")
        print(f"  After Inference:   {gpu_mem_after_inference:.1f} (current), {gpu_mem_peak_after_inference:.1f} (peak)")
        print(f"  Overall Peak:      {gpu_mem_peak_after_inference:.1f}")
    else:
        print(f"\nGPU Memory: CUDA not available")
    
    print(f"\nProcessed {camera_poses.shape[0]} frames")
    print(f"Output resolution: {target_height}x{target_width}")
    print("="*60 + "\n")
    
    return cfg
