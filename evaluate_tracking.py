"""
Integration script to evaluate 3D tracking metrics on your tracking pipeline.
"""

from pathlib import Path
from omegaconf import OmegaConf
import networkx as nx
from metrics.metrics_3d import evaluate_tracking, save_metrics


def collect_graphs_from_tracking(cfg):
    """
    Run tracking and collect graphs per frame.
    This is a modified version that stores graphs per frame.
    """
    import YOLOE.utils as yutils
    from ssg.ssg_main import edges
    import numpy as np
    
    rgb_dir_path = cfg.rgb_dir
    depth_folder = cfg.depth_dir
    traj_path = cfg.traj_path
    max_points_per_obj = int(cfg.max_points_per_obj)
    
    # Dictionary to accumulate point clouds per track_id across frames
    accumulated_points_dict = {}
    max_accumulated_points = int(cfg.get('max_accumulated_points', 10000))
    
    # Store graphs per frame for evaluation
    graph_per_frame = {}
    
    # 1) cache the depths
    depth_paths = yutils.list_png_paths(depth_folder)
    depth_cache = {}
    for dp in depth_paths:
        depth_cache[dp] = yutils.load_depth_as_meters(dp)
    
    # 2) Load camera poses
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
        depth_m = depth_cache.get(depth_cur_path)
        if depth_m is None:
            print(f"[WARN] Missing depth for {depth_cur_path}; skipping frame {frame_idx}")
            frame_idx += 1
            continue
        
        # masks
        mask_org, masks_clean = yutils.preprocess_mask(
            yolo_res=yl_res,
            index=frame_idx,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha),
            fast=cfg.fast_mask,
        )
        
        # track ids
        track_ids = None
        if hasattr(yl_res, 'boxes') and yl_res.boxes is not None and getattr(yl_res.boxes, 'id', None) is not None:
            try:
                track_ids = yl_res.boxes.id.detach().cpu().numpy().astype(np.int64)
            except Exception:
                pass
        if track_ids is None:
            n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
            track_ids = np.arange(n, dtype=np.int64)
        
        # pose for this frame
        T_w_c = poses[min(frame_idx, len(poses)-1)] if poses else None
        
        # build 3D objects
        frame_objs, current_graph = yutils.create_3d_objects(
            track_ids, 
            masks_clean, 
            max_points_per_obj, 
            depth_m, 
            T_w_c, 
            frame_idx,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            accumulated_points_dict=accumulated_points_dict,
            max_accumulated_points=max_accumulated_points
        )
        
        # Store graph for this frame
        graph_per_frame[frame_idx] = current_graph.copy()
        
        # Edge predictor (optional, not needed for metrics)
        if cfg.get('compute_edges', True):
            edges(current_graph, frame_objs, T_w_c, depth_m)
        
        frame_idx += 1
        
        print(f"Processed frame {frame_idx}/{len(depth_paths)}", end='\r')
    
    print(f"\nTracking complete. Processed {frame_idx} frames.")
    return graph_per_frame


def main():
    """Main evaluation script"""
    
    # Configuration
    cfg = OmegaConf.create({
        'rgb_dir': "/home/maribjonov_mr/IsaacSim_bench/cabinet_complex/rgb",
        'depth_dir': "/home/maribjonov_mr/IsaacSim_bench/cabinet_complex/depth",
        'traj_path': "/home/maribjonov_mr/IsaacSim_bench/cabinet_complex/traj.txt",
        'yolo_model': '/home/maribjonov_mr/yolo_bench/yoloe-11l-seg-pf.pt',
        'conf': 0.3,
        'iou': 0.5,
        'kernel_size': 11,
        'alpha': 0.7,
        'max_points_per_obj': 2000,
        'max_accumulated_points': 10000,
        'fast_mask': True,  # Fast mode for evaluation
        'o3_nb_neighbors': 50,
        'o3std_ratio': 0.1,
        'compute_edges': False,  # Skip edge computation for faster evaluation
        'visualize_frames': True,  # Enable visualization
        'viz_frames': [0, 5, 10, 15, 20],  # Which frames to visualize
    })
    
    # Path to scene with ground truth
    scene_path = Path("/home/maribjonov_mr/IsaacSim_bench/cabinet_complex")
    scene_name = scene_path.name
    
    print("="*60)
    print(f"EVALUATING 3D TRACKING: {scene_name}")
    print("="*60)
    
    # Step 1: Run tracking and collect graphs
    print("\n[1/3] Running tracking pipeline...")
    graph_per_frame = collect_graphs_from_tracking(cfg)
    
    # GT filtering settings (same as gt_vis.py)
    MAX_BOX_EDGE = 20.0  # meters; filter very large boxes
    IGNORE_PRIM_PREFIXES = []  # e.g., ["/World/env"] to ignore environment
    
    # Step 1.5: Optional visualization of sample frames
    if cfg.get('visualize_frames', False):
        from metrics.metrics_3d import load_gt_data, load_prediction_data, visualize_frame_comparison
        
        print("\n[1.5/3] Loading data for visualization...")
        gt_tracks = load_gt_data(
            scene_path,
            max_box_edge=MAX_BOX_EDGE,
            ignore_prim_prefixes=IGNORE_PRIM_PREFIXES
        )
        pred_tracks = load_prediction_data(graph_per_frame)
        
        # Visualize specified frames
        frames_to_viz = cfg.get('viz_frames', [0, 1, 2])
        for frame_id in frames_to_viz:
            if frame_id in graph_per_frame:
                visualize_frame_comparison(
                    frame_id=frame_id,
                    gt_tracks=gt_tracks,
                    pred_tracks=pred_tracks,
                    graph_per_frame=graph_per_frame,
                    show_points=False
                )
    
    # Step 2: Evaluate metrics
    print("\n[2/3] Computing tracking metrics...")
    try:
        results = evaluate_tracking(
            scene_path=scene_path,
            graph_per_frame=graph_per_frame,
            iou_threshold=0.5,
            max_box_edge=MAX_BOX_EDGE,
            ignore_prim_prefixes=IGNORE_PRIM_PREFIXES
        )
        
        # Step 3: Save results
        print("\n[3/3] Saving metrics...")
        output_path = Path("metrics_data")
        save_metrics(results, output_path, scene_name=scene_name)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure the scene folder contains a 'bbox' subfolder with ground truth JSON files.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
