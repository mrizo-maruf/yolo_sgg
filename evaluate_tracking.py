#!/usr/bin/env python3
"""
Lightweight wrapper: runs the tracking pipeline on a single scene
and evaluates 3-D bounding-box metrics (MOTA, MOTP, IDF1, HOTA)
using ``metrics.metrics_3d``.

For 2-D mask-based metrics (T-mIoU, T-SR, …), use
``benchmark.benchmark_tracking`` instead.
"""

from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from metrics.metrics_3d import evaluate_tracking, save_metrics


# ───────────────────────────────────────────────────────────────────────────
# Tracking pipeline  →  per-frame graphs
# ───────────────────────────────────────────────────────────────────────────

def collect_graphs_from_tracking(cfg) -> dict:
    """Run YOLO tracking and return ``{frame_idx: nx.MultiDiGraph}``."""
    import YOLOE.utils as yutils

    depth_paths = yutils.list_png_paths(cfg.depth_dir)
    depth_cache = {dp: yutils.load_depth_as_meters(dp) for dp in depth_paths}
    poses = yutils.load_camera_poses(cfg.traj_path)

    accumulated_points: dict = {}
    graph_per_frame: dict = {}

    results_stream = yutils.track_objects_in_video_stream(
        cfg.rgb_dir, depth_paths,
        model_path=cfg.yolo_model,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
    )

    frame_idx = 0
    for yolo_res, _, depth_path in results_stream:
        depth_m = depth_cache.get(depth_path)
        if depth_m is None:
            frame_idx += 1
            continue

        _, masks_clean = yutils.preprocess_mask(
            yolo_res=yolo_res, index=frame_idx,
            KERNEL_SIZE=int(cfg.kernel_size),
            alpha=float(cfg.alpha), fast=cfg.fast_mask,
        )

        # track IDs
        track_ids = None
        if hasattr(yolo_res, "boxes") and yolo_res.boxes is not None:
            ids_attr = getattr(yolo_res.boxes, "id", None)
            if ids_attr is not None:
                try:
                    track_ids = ids_attr.detach().cpu().numpy().astype(np.int64)
                except Exception:
                    pass
        if track_ids is None:
            n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
            track_ids = np.arange(n, dtype=np.int64)

        T_w_c = poses[min(frame_idx, len(poses) - 1)] if poses else None

        _, current_graph = yutils.create_3d_objects(
            track_ids, masks_clean,
            int(cfg.max_points_per_obj), depth_m, T_w_c, frame_idx,
            o3_nb_neighbors=cfg.o3_nb_neighbors,
            o3std_ratio=cfg.o3std_ratio,
            accumulated_points_dict=accumulated_points,
            max_accumulated_points=int(cfg.get("max_accumulated_points", 10000)),
        )
        graph_per_frame[frame_idx] = current_graph.copy()
        frame_idx += 1
        print(f"\rProcessed {frame_idx}/{len(depth_paths)}", end="")

    print(f"\nTracking complete – {frame_idx} frames.")
    return graph_per_frame


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = OmegaConf.create({
        "rgb_dir": "/home/yehia/rizo/IsaacSim_Dataset/scene_1/rgb",
        "depth_dir": "/home/yehia/rizo/IsaacSim_Dataset/scene_1/depth",
        "traj_path": "/home/yehia/rizo/IsaacSim_Dataset/scene_1/traj.txt",
        "yolo_model": "yoloe-11l-seg-pf.pt",
        "conf": 0.3, "iou": 0.5,
        "kernel_size": 11, "alpha": 0.7, "fast_mask": True,
        "max_points_per_obj": 2000, "max_accumulated_points": 10000,
        "o3_nb_neighbors": 50, "o3std_ratio": 0.1,
    })

    scene_path = Path("/home/yehia/rizo/IsaacSim_Dataset/scene_1")

    print("=" * 60)
    print(f"  3-D TRACKING EVALUATION – {scene_path.name}")
    print("=" * 60)

    print("\n[1/2] Running tracking pipeline …")
    graph_per_frame = collect_graphs_from_tracking(cfg)

    print("\n[2/2] Computing 3-D metrics …")
    try:
        results = evaluate_tracking(
            scene_path=scene_path,
            graph_per_frame=graph_per_frame,
            iou_threshold=0.5,
            max_box_edge=20.0,
        )
        save_metrics(results, Path("metrics_data"), scene_name=scene_path.name)
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
