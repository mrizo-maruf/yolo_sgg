from omegaconf import OmegaConf
import YOLOE.utils as yutils
import numpy as np
from ssg.ssg_main import predict_new_edges

def main(cfg):
    rgb_dir_path = cfg.rgb_dir
    depth_folder = cfg.depth_dir
    traj_path = cfg.traj_path
    max_points_per_obj = int(cfg.max_points_per_obj)

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
        depth_m = depth_cache.get(depth_cur_path)
        if depth_m is None:
            print(f"[WARN][yolo_sgg] Missing depth for {depth_cur_path}; skipping frame {frame_idx}")
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
            # fallback: sequential ids up to number of masks
            n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
            track_ids = np.arange(n, dtype=np.int64)

        # pose for this frame
        T_w_c = poses[min(frame_idx, len(poses)-1)] if poses else None

        # build 3D objects
        frame_objs, current_graph = yutils.create_3d_objects(track_ids, masks_clean, max_points_per_obj, depth_m, T_w_c, frame_idx)

        if bool(cfg.show_pcds):
            # visualize (blocking window)
            yutils.visualize_frame_objects_open3d(frame_objs, frame_idx)

        # Edge predictor SceneVerse
        predict_new_edges(current_graph, frame_objs)

        # update_graph(current_graph, frame_objs, persistent_graph)
        frame_idx += 1

    print("[yolo_sgg] Done")
    return 0


if __name__ == "__main__":
    cfg = OmegaConf.create({
        'rgb_dir': "/home/rizo/mipt_ccm/yle_sc_sg/UR5-Peg-In-Hole_02_HD/rgb",
        'depth_dir': "/home/rizo/mipt_ccm/yle_sc_sg/UR5-Peg-In-Hole_02_HD/depth",
        'traj_path': "/home/rizo/mipt_ccm/yle_sc_sg/UR5-Peg-In-Hole_02_HD/traj.txt",
        'yolo_model': 'yoloe-11l-seg-pf.pt',
        'conf': 0.3,
        'iou': 0.5,
        'kernel_size': 19,
        'alpha': 0.7,
        'max_points_per_obj': 2000,
        'show_pcds': True,
        'fast_mask': False,
    })
    main(cfg)