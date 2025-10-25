from omegaconf import OmegaConf
import YOLOE.utils as yutils
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

        # track ids
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

        # pose for this frame
        T_w_c = poses[min(frame_idx, len(poses)-1)] if poses else None

        # build 3D objects
        t_create3d_start = time.perf_counter()
        frame_objs, current_graph = yutils.create_3d_objects(track_ids, 
                                                             masks_clean, 
                                                             max_points_per_obj, 
                                                             depth_m, 
                                                             T_w_c, 
                                                             frame_idx,
                                                             o3_nb_neighbors=cfg.o3_nb_neighbors,
                                                             o3std_ratio=cfg.o3std_ratio
        )
        t_create3d_end = time.perf_counter()
        timings['create_3d'].append((t_create3d_end - t_create3d_start) * 1000)  # ms

        # Edge predictor SceneVerse
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
        
        
        if bool(cfg.show_pcds):
            # visualize (blocking window)
            yutils.visualize_frame_objects_open3d(frame_objs, frame_idx)

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
        'rgb_dir': "/home/rizo/mipt_ccm/yolo_ssg/UR5-Peg-In-Hole_02_straight/rgb",
        'depth_dir': "/home/rizo/mipt_ccm/yolo_ssg/UR5-Peg-In-Hole_02_straight/depth",
        'traj_path': "/home/rizo/mipt_ccm/yolo_ssg/UR5-Peg-In-Hole_02_straight/traj.txt",
        'yolo_model': 'yoloe-11l-seg-pf-old.pt',
        'conf': 0.3,
        'iou': 0.5,
        'kernel_size': 11,
        'alpha': 0.7,
        'max_points_per_obj': 2000,
        'show_pcds': False,
        'fast_mask': False,
        'o3_nb_neighbors': 50,
        'o3std_ratio': 0.1,
        'vis_graph': True,
        'print_resource_usage': False,
    })
    main(cfg)