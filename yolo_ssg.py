from omegaconf import OmegaConf
import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry
import numpy as np
from ssg.ssg_main import edges
import matplotlib.pyplot as plt
import time
import torch
import networkx as nx

def main(cfg):
    rgb_dir_path = cfg.rgb_dir
    depth_folder = cfg.depth_dir
    traj_path = cfg.traj_path
    max_points_per_obj = int(cfg.max_points_per_obj)

    persistent_graph = nx.MultiDiGraph()
    
    # Initialize Global Object Registry for consistent tracking and reconstruction
    # - Stores accumulated PCDs centrally (for reconstruction)
    # - Returns lightweight bbox objects (for edge prediction)
    # - Tracks visibility status for all objects
    object_registry = GlobalObjectRegistry(
        overlap_threshold=float(cfg.get('tracking_overlap_threshold', 0.5)),
        distance_threshold=float(cfg.get('tracking_distance_threshold', 0.5)),
        max_points_per_object=int(cfg.get('max_accumulated_points', 10000)),
        inactive_frames_limit=int(cfg.get('tracking_inactive_limit', 0))  # 0 = never remove
    )

    # Metrics accumulation
    timings = {'yolo': [], 'preprocess': [], 'create_3d': [], 'edges': [], 'merge': []}
    gpu_usage = {'yolo': [], 'edges': []}
    cuda_available = torch.cuda.is_available()

    # 1) cache the depths (also initializes YOLOE.utils global DEPTH_PATHS)
    depth_paths = yutils.list_png_paths(depth_folder)
    depth_cache = {}
    for dp in depth_paths:
        depth_cache[dp] = yutils.load_depth_as_meters(dp)

    # 2) Load camera poses (camera->world)
    poses = yutils.load_camera_poses(traj_path)

    # 3) track
    results_stream = yutils.track_objects_in_video_stream(
        rgb_dir_path,
        depth_paths,
        model_path=cfg.yolo_model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
    )
    
    frame_idx = 0
    for yl_res, rgb_cur_path, depth_cur_path in results_stream:
        # YOLO tracking is done in the generator, measure time would require refactor
        # For now we measure from when result arrives
        timings['yolo'].append(yl_res.speed['inference'])  # ms
        
        depth_m = depth_cache.get(depth_cur_path)
        if depth_m is None:
            print(f"[WARN][yolo_sgg] Missing depth for {depth_cur_path}; skipping frame {frame_idx}")
            frame_idx += 1
            continue

        # GPU usage after YOLO (if available)
        if cuda_available:
            torch.cuda.synchronize()
            gpu_mem_yolo = torch.cuda.memory_allocated() / (1024**2)  # MB
            gpu_usage['yolo'].append(gpu_mem_yolo)

        # masks
        t_preprocess_start = time.perf_counter()
        mask_org, masks_clean = yutils.preprocess_mask(
            yolo_res=yl_res,
            index=frame_idx,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            fast=cfg.fast_mask,
        )

        t_preprocess_end = time.perf_counter()
        timings['preprocess'].append((t_preprocess_end - t_preprocess_start) * 1000)  # ms

        # track ids and class names from YOLO
        track_ids = None
        class_names = None
        if hasattr(yl_res, 'boxes') and yl_res.boxes is not None:
            # Get track IDs
            if getattr(yl_res.boxes, 'id', None) is not None:
                try:
                    track_ids = yl_res.boxes.id.detach().cpu().numpy().astype(np.int64)
                except Exception:
                    pass
            
            # Get class names if available
            if getattr(yl_res.boxes, 'cls', None) is not None and hasattr(yl_res, 'names'):
                try:
                    cls_ids = yl_res.boxes.cls.detach().cpu().numpy().astype(np.int64)
                    class_names = [yl_res.names[int(c)] for c in cls_ids]
                except Exception:
                    pass
        
        if track_ids is None:
            # fallback: sequential ids up to number of masks
            n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
            track_ids = np.arange(n, dtype=np.int64)

        # pose for this frame
        T_w_c = poses[min(frame_idx, len(poses)-1)] if poses else None

        # Build 3D objects with enhanced tracking
        # Returns LIGHTWEIGHT objects (bbox only) for edge prediction
        # PCDs are stored in registry for reconstruction
        t_create3d_start = time.perf_counter()
        frame_objs, current_graph = yutils.create_3d_objects_with_tracking(
            track_ids, 
            masks_clean, 
            max_points_per_obj, 
            depth_m, 
            T_w_c, 
            frame_idx,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            object_registry=object_registry,
            class_names=class_names
        )
        t_create3d_end = time.perf_counter()
        timings['create_3d'].append((t_create3d_end - t_create3d_start) * 1000)  # ms

        # Print tracking info if enabled
        if cfg.get('print_tracking_info', False):
            summary = object_registry.get_tracking_summary()
            print(f"  [Track] Frame {frame_idx}: Visible={summary['visible_objects']}, "
                  f"Total={summary['total_objects']}, Invisible={summary['invisible_objects']}")
            for obj in frame_objs:
                class_str = f" ({obj['class_name']})" if obj.get('class_name') else ""
                print(f"    YOLO_ID={obj['yolo_track_id']} -> GLOBAL_ID={obj['global_id']}{class_str} "
                      f"(source: {obj['match_source']}, obs_count: {obj['observation_count']})")

        # Edge predictor SceneVerse (uses lightweight bbox objects)
        t_edges_start = time.perf_counter()
        edges(current_graph, frame_objs, T_w_c, depth_m)
        t_edges_end = time.perf_counter()
        timings['edges'].append((t_edges_end - t_edges_start) * 1000)  # ms
        
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
        
        # Visualize reconstruction (all accumulated PCDs with visibility status)
        if bool(cfg.show_pcds):
            yutils.visualize_reconstruction(
                object_registry=object_registry,
                frame_index=frame_idx,
                show_visible_only=False,  # Show all objects (visible + ghosted invisible)
                show_aabb=True
            )

        t_merge_start = time.perf_counter()
        persistent_graph = yutils.merge_scene_graphs(persistent_graph, current_graph)
        t_merge_end = time.perf_counter()
        timings['merge'].append((t_merge_end - t_merge_start) * 1000)  # ms
        
        # vis DiGraph
        if cfg.vis_graph:
            fig, axes = plt.subplots(1, 2, figsize=(18, 8))

            # First graph
            yutils.draw_labeled_multigraph(persistent_graph, ax=axes[0])
            axes[0].set_title(f"Persistent Graph frame: {frame_idx}")

            # Second graph (example: current_graph)
            yutils.draw_labeled_multigraph(current_graph, ax=axes[1])
            axes[1].set_title(f"Current Graph frame: {frame_idx}")

            plt.tight_layout()
            plt.show()

            
        frame_idx += 1
        
        # return 0

    # Print tracking summary
    print("\n" + "="*60)
    print("TRACKING & RECONSTRUCTION SUMMARY")
    print("="*60)
    all_objs = object_registry.get_all_objects()
    print(f"Total unique objects tracked: {len(all_objs)}")
    print(f"\nPer-object details:")
    for gid, obj in all_objs.items():
        class_str = f" ({obj.get('class_name')})" if obj.get('class_name') else ""
        visible_str = "VISIBLE" if obj.get('visible_current_frame') else "not visible"
        print(f"  Object {gid}{class_str}: {visible_str}, "
              f"seen in {obj['observation_count']} frames, "
              f"first: {obj['first_seen_frame']}, last: {obj['last_seen_frame']}, "
              f"points: {len(obj['points_accumulated'])}")
    print("="*60)

    # Print summary statistics
    print("\n" + "="*60)
    print("PERFORMANCE STATISTICS")
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
        'rgb_dir': "/home/maribjonov_mr/IsaacSim_bench/nk_scene_complex/rgb",
        'depth_dir': "/home/maribjonov_mr/IsaacSim_bench/nk_scene_complex/depth",
        'traj_path': "/home/maribjonov_mr/IsaacSim_bench/nk_scene_complex/traj.txt",
        'yolo_model': '/home/maribjonov_mr/yolo_bench/yoloe-11l-seg-pf.pt',
        'conf': 0.25,
        'iou': 0.5,
        'kernel_size': 17,
        'alpha': 0.7,
        'max_points_per_obj': 2000,
        'max_accumulated_points': 10000,
        'show_pcds': False,
        'fast_mask': True,
        'o3_nb_neighbors': 50,
        'o3std_ratio': 0.1,
        'vis_graph': False,
        'print_resource_usage': False,
        # Tracking parameters
        'tracking_overlap_threshold': 0.3,  # 3D IoU threshold for matching
        'tracking_distance_threshold': 0.5,  # Max centroid distance (meters)
        'tracking_inactive_limit': 0,  # Frames before removing unseen objects (0 = keep forever)
        'print_tracking_info': False,  # Print per-frame tracking details
    })
    main(cfg)