import time
from dataclasses import asdict

import networkx as nx
import numpy as np
import torch

import YOLOE.utils as yutils
from YOLOE.utils import GlobalObjectRegistry

from yolo_sgg.core.config import ExperimentConfig
from yolo_sgg.plugins.detection_tracking import YoloTrackingAdapter
from yolo_sgg.plugins.edge_predictors import SceneVerseEdgePredictor
from yolo_sgg.services.graph_merge import GraphMergeService


class TrackingSceneGraphPipeline:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.persistent_graph = nx.MultiDiGraph()
        self.object_registry = GlobalObjectRegistry(
            overlap_threshold=float(cfg.tracking_overlap_threshold),
            distance_threshold=float(cfg.tracking_distance_threshold),
            max_points_per_object=int(cfg.max_accumulated_points),
            inactive_frames_limit=int(cfg.tracking_inactive_limit),
            volume_ratio_threshold=float(cfg.tracking_volume_ratio_threshold),
            reprojection_visibility_threshold=float(cfg.reprojection_visibility_threshold),
        )
        self.tracker = YoloTrackingAdapter(
            model_path=cfg.yolo_model,
            conf=float(cfg.conf),
            iou=float(cfg.iou),
            tracker_cfg=cfg.tracker_cfg,
            device=cfg.device,
        )
        self.edge_predictor = SceneVerseEdgePredictor()
        self.timings = {'yolo': [], 'preprocess': [], 'create_3d': [], 'edges': [], 'merge': []}

    def run(self, collect_frame_graphs: bool = False):
        cfg = self.cfg
        depth_paths = yutils.list_png_paths(cfg.depth_dir)
        depth_cache = {dp: yutils.load_depth_as_meters(dp) for dp in depth_paths}
        poses = yutils.load_camera_poses(cfg.traj_path)
        cuda_available = torch.cuda.is_available()

        frame_idx = 0
        frame_graphs = {}
        for yl_res, _rgb_cur_path, depth_cur_path in self.tracker.stream(cfg.rgb_dir, depth_paths):
            self.timings['yolo'].append(yl_res.speed.get('inference', 0.0))
            depth_m = depth_cache.get(depth_cur_path)
            if depth_m is None:
                frame_idx += 1
                continue

            t_pre = time.perf_counter()
            _, masks_clean = yutils.preprocess_mask(
                yolo_res=yl_res,
                index=frame_idx,
                KERNEL_SIZE=int(cfg.kernel_size),
                alpha=float(cfg.alpha),
                fast=cfg.fast_mask,
            )
            self.timings['preprocess'].append((time.perf_counter() - t_pre) * 1000)

            track_ids, class_names = self._extract_track_data(yl_res, masks_clean)
            T_w_c = poses[min(frame_idx, len(poses) - 1)] if poses else None

            t_3d = time.perf_counter()
            frame_objs, current_graph = yutils.create_3d_objects_with_tracking(
                track_ids,
                masks_clean,
                int(cfg.max_points_per_obj),
                depth_m,
                T_w_c,
                frame_idx,
                o3_nb_neighbors=cfg.o3_nb_neighbors,
                o3std_ratio=cfg.o3std_ratio,
                object_registry=self.object_registry,
                class_names=class_names,
            )
            self.timings['create_3d'].append((time.perf_counter() - t_3d) * 1000)

            t_edges = time.perf_counter()
            self.edge_predictor.predict(current_graph, frame_objs, T_w_c, depth_m)
            self.timings['edges'].append((time.perf_counter() - t_edges) * 1000)

            t_merge = time.perf_counter()
            self.persistent_graph = GraphMergeService.merge(self.persistent_graph, current_graph)
            self.timings['merge'].append((time.perf_counter() - t_merge) * 1000)

            if collect_frame_graphs:
                frame_graphs[frame_idx] = current_graph.copy()

            if cfg.show_pcds:
                yutils.visualize_reconstruction(self.object_registry, frame_idx, show_visible_only=False, show_aabb=True)

            frame_idx += 1

        if cfg.print_resource_usage or cuda_available:
            self._print_summary(frame_idx, cuda_available)

        if collect_frame_graphs:
            return self.persistent_graph, frame_graphs
        return self.persistent_graph

    @staticmethod
    def _extract_track_data(yl_res, masks_clean):
        track_ids = None
        class_names = None
        if hasattr(yl_res, 'boxes') and yl_res.boxes is not None:
            if getattr(yl_res.boxes, 'id', None) is not None:
                try:
                    track_ids = yl_res.boxes.id.detach().cpu().numpy().astype(np.int64)
                except Exception:
                    track_ids = None
            if getattr(yl_res.boxes, 'cls', None) is not None and hasattr(yl_res, 'names'):
                try:
                    cls_ids = yl_res.boxes.cls.detach().cpu().numpy().astype(np.int64)
                    class_names = [yl_res.names[int(c)] for c in cls_ids]
                except Exception:
                    class_names = None

        if track_ids is None:
            n = len(masks_clean) if isinstance(masks_clean, (list, tuple)) else 0
            track_ids = np.arange(n, dtype=np.int64)

        return track_ids, class_names

    def _print_summary(self, frames, cuda_available):
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        if self.timings['preprocess']:
            print(f"Frames processed: {frames}")
            print(f"YOLO avg ms:        {np.mean(self.timings['yolo']):.2f}")
            print(f"Preprocess avg ms:  {np.mean(self.timings['preprocess']):.2f}")
            print(f"Create3D avg ms:    {np.mean(self.timings['create_3d']):.2f}")
            print(f"Edges avg ms:       {np.mean(self.timings['edges']):.2f}")
            print(f"Merge avg ms:       {np.mean(self.timings['merge']):.2f}")
        print(f"CUDA available: {cuda_available}")

    def config_dict(self):
        return asdict(self.cfg)
