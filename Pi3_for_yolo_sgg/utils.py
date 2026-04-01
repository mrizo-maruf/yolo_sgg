import os
import torch
import numpy as np
import cv2
import glob
import time
from pi3.models.pi3x import Pi3X
from pi3.utils.basic import load_images_as_tensor
from pi3.pipe.pi3x_vo import Pi3XVO


def _list_rgb_files(rgb_dir):
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(rgb_dir, pat)))
    return sorted(files)


def _build_scaled_intrinsics_tensor(
    fx,
    fy,
    cx,
    cy,
    orig_h,
    orig_w,
    infer_h,
    infer_w,
    n_frames,
    device,
    apply_resize_scaling=True,
):
    if apply_resize_scaling:
        scale_x = float(infer_w) / float(orig_w)
        scale_y = float(infer_h) / float(orig_h)
    else:
        scale_x = 1.0
        scale_y = 1.0

    K = torch.tensor(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    K_seq = K.view(1, 1, 3, 3).repeat(1, n_frames, 1, 1)
    return K, K_seq


def _resolve_ckpt_path(ckpt):
    if ckpt is None:
        return None

    ckpt_path = os.path.abspath(os.path.expanduser(str(ckpt)))
    if os.path.isfile(ckpt_path):
        return ckpt_path

    if os.path.isdir(ckpt_path):
        candidates = (
            os.path.join(ckpt_path, "model.safetensors"),
            os.path.join(ckpt_path, "pytorch_model.bin"),
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No checkpoint file found in directory: {ckpt_path}. "
            f"Expected one of: model.safetensors, pytorch_model.bin"
        )

    raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")


def process_depth_model(cfg):
    """
    Process depth and trajectory using specified depth model.
    
    Args:
        cfg: Config object with fields:
            - rgb_dir: Path to RGB images directory
            - depth_dir: Path to original depth directory
            - traj_path: Path to original trajectory file
            - depth_model: Depth model name (e.g., 'yyfz233/Pi3X') or None
            - ckpt: Optional local checkpoint file/dir path
            - original_img: If True, keep original image resolution (padding only if needed)
            - fx, fy, cx, cy: Camera intrinsics in original RGB resolution
    
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

    ckpt_path = _resolve_ckpt_path(getattr(cfg, 'ckpt', None))
        
    print("\n" + "="*60)
    print(f"DEPTH MODEL PROCESSING: {cfg.depth_model}")
    if ckpt_path is not None:
        print(f"Using local checkpoint: {ckpt_path}")
    else:
        print("No local checkpoint provided. Falling back to Hugging Face download (can take a while).")
    print("="*60)

    # Default pinhole intrinsics if not provided in config
    fx = float(getattr(cfg, 'fx', 800.0))
    fy = float(getattr(cfg, 'fy', 800.0))
    cx = float(getattr(cfg, 'cx', 640.0))
    cy = float(getattr(cfg, 'cy', 360.0))
    original_img = bool(getattr(cfg, 'original_img', False))
    print(f"Using intrinsics (original image space): fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    print(f"Image sizing mode: {'original' if original_img else 'downscaled (default)'}")
    
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
    if ckpt_path is not None:
        print("  - Building Pi3X architecture...")
        model = Pi3X().to(device).eval()
        print("  - Loading checkpoint weights from disk...")
        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(ckpt_path)
        else:
            weight = torch.load(ckpt_path, map_location=device, weights_only=False)
        print("  - Applying state dict...")
        model.load_state_dict(weight, strict=False)
        print("  - Local checkpoint loaded.")
    else:
        print("  - Downloading/loading from Hugging Face...")
        model = Pi3X.from_pretrained(cfg.depth_model).to(device).eval()
        print("  - Hugging Face model loaded.")
    
    # Create Pi3XVO pipeline for efficient windowed inference
    pipe = Pi3XVO(model)
    
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
    imgs = load_images_as_tensor(
        cfg.rgb_dir,
        interval=1,
        use_original_size=original_img,
    ).to(device)

    rgb_image_files = _list_rgb_files(cfg.rgb_dir)
    if len(rgb_image_files) == 0:
        raise ValueError(f"No RGB images found in {cfg.rgb_dir}")

    first_rgb = cv2.imread(rgb_image_files[0], cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise ValueError(f"Could not read RGB image: {rgb_image_files[0]}")

    orig_h, orig_w = first_rgb.shape[:2]
    infer_h, infer_w = imgs.shape[-2:]
    n_frames = imgs.shape[0]

    K_scaled, intrinsics_seq = _build_scaled_intrinsics_tensor(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        orig_h=orig_h,
        orig_w=orig_w,
        infer_h=infer_h,
        infer_w=infer_w,
        n_frames=n_frames,
        device=device,
        apply_resize_scaling=not original_img,
    )
    if original_img:
        print(f"Intrinsics used by model (original/padded mode {infer_w}x{infer_h}):\n{K_scaled.detach().cpu().numpy()}")
    else:
        print(f"Scaled intrinsics used by model (resized {infer_w}x{infer_h}):\n{K_scaled.detach().cpu().numpy()}")

    t_load_end = time.perf_counter()
    timings['image_load'] = (t_load_end - t_load_start) * 1000  # ms
    
    # --- Inference with Pi3XVO Pipeline ---
    t_inference_start = time.perf_counter()
    print("Running model inference with Pi3XVO pipeline...")
    dtype = torch.bfloat16 if cuda_available and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # Pi3XVO parameters
    chunk_size = 16  # Number of frames per chunk
    overlap = 12     # Overlap between chunks for alignment
    
    print(f"Processing {imgs.shape[0]} frames with chunk_size={chunk_size}, overlap={overlap}")
    
    with torch.no_grad():
        results = pipe(
            imgs=imgs[None],  # Add batch dimension -> (1, N, 3, H, W)
            chunk_size=chunk_size,
            overlap=overlap,
            conf_thre=0.05,
            inject_condition=[],
            intrinsics=intrinsics_seq,
            dtype=dtype
        )
    
    if cuda_available:
        torch.cuda.synchronize()
    t_inference_end = time.perf_counter()
    timings['inference'] = (t_inference_end - t_inference_start) * 1000  # ms
    
    if cuda_available:
        gpu_mem_after_inference = torch.cuda.memory_allocated() / (1024**2)  # MB
        gpu_mem_peak_after_inference = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        print(f"GPU memory after inference: {gpu_mem_after_inference:.1f} MB (peak: {gpu_mem_peak_after_inference:.1f} MB)")
    
    print("Reconstruction complete!")
    
    # --- Extract Depth Maps and Camera Poses ---
    t_extract_start = time.perf_counter()
    
    # Extract local_points from points using camera poses (inverse transformation)
    # For depth maps, we need local depth (Z coordinate in camera space)
    points = results['points'][0]  # Shape: (N, H, W, 3) - global points
    camera_poses = results['camera_poses'][0]  # Shape: (N, 4, 4)
    
    # Convert global points back to local camera space to get depth
    N, H, W, _ = points.shape
    # Homogenize points
    points_homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)  # (N, H, W, 4)
    
    # Transform to local camera space
    camera_poses_inv = torch.inverse(camera_poses)  # (N, 4, 4)
    local_points = torch.einsum('nij,nhwj->nhwi', camera_poses_inv, points_homo)[..., :3]  # (N, H, W, 3)
    
    depth_maps = local_points[..., 2]  # Z coordinate is depth
    
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
    # Use first RGB image (already validated/loaded above) for target resolution
    target_height, target_width = first_rgb.shape[:2]
    print(f"Target resolution from first RGB image: {target_height}x{target_width}")
    
    # --- Upscale Depth Maps to Match RGB Resolution ---
    infer_depth_h, infer_depth_w = depth_maps_np.shape[1], depth_maps_np.shape[2]
    if original_img and infer_depth_h >= target_height and infer_depth_w >= target_width:
        depth_maps_np = depth_maps_np[:, :target_height, :target_width]
        print(f"Cropped depth maps to original image size: {depth_maps_np.shape}")
    else:
        upscaled_depth_maps = []
        for i in range(depth_maps_np.shape[0]):
            depth = depth_maps_np[i]
            depth_upscaled = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            upscaled_depth_maps.append(depth_upscaled)

        depth_maps_np = np.stack(upscaled_depth_maps, axis=0)
    t_resize_end = time.perf_counter()
    timings['resize'] = (t_resize_end - t_resize_start) * 1000  # ms
    print(f"Depth maps output shape: {depth_maps_np.shape}")
    
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
    
    # --- Save Depth as Metric PNG Images (uint16) ---
    t_save_png_start = time.perf_counter()
    depth_png_scale = float(getattr(cfg, 'pi3_png_depth_scale', 0.001))  # 1 unit = 1 mm
    if depth_png_scale <= 0:
        raise ValueError(f"pi3_png_depth_scale must be > 0, got {depth_png_scale}")

    print(f"Saving metric depth images to: {new_depth_dir}")
    print(f"Pi3 depth PNG encoding: uint16 with scale {depth_png_scale} m/unit")

    depth_u16_max = np.iinfo(np.uint16).max
    clipped_values = 0
    for i in range(depth_maps_np.shape[0]):
        depth = depth_maps_np[i]

        valid_mask = np.logical_and(depth > 0, np.isfinite(depth))
        depth_u16 = np.zeros_like(depth, dtype=np.uint16)
        if valid_mask.sum() > 0:
            q = np.round(depth[valid_mask] / depth_png_scale)
            clipped_values += int(np.count_nonzero(q > depth_u16_max))
            q = np.clip(q, 1, depth_u16_max).astype(np.uint16)
            depth_u16[valid_mask] = q

        output_path = os.path.join(new_depth_dir, f'depth{i:06d}.png')
        ok = cv2.imwrite(output_path, depth_u16)
        if not ok:
            raise IOError(f"Failed to save depth PNG: {output_path}")

    # Save metadata for deterministic decoding in visualization
    meta_path = os.path.join(new_depth_dir, 'pi3_depth_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"png_depth_scale: {depth_png_scale:.10f}\n")
        f.write("format: uint16\n")
        f.write("description: depth_in_meters = png_value * png_depth_scale\n")
        f.write(f"global_depth_min_m: {float(global_depth_min):.6f}\n")
        f.write(f"global_depth_max_m: {float(global_depth_max):.6f}\n")
    print(f"Saved Pi3 depth metadata: {meta_path}")
    if clipped_values > 0:
        print(f"Warning: clipped {clipped_values} depth values at uint16 max range")
    
    t_save_png_end = time.perf_counter()
    timings['save_png'] = (t_save_png_end - t_save_png_start) * 1000  # ms
    print(f"Saved {depth_maps_np.shape[0]} metric depth images")
    
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
    # camera_poses already extracted during inference -> (N, 4, 4)
    
    with open(new_traj_path, 'w') as f:
        for i in range(camera_poses.shape[0]):
            pose = camera_poses[i].cpu().numpy()  # Shape: (4, 4) - already on CPU
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
