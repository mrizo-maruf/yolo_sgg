import argparse
from pathlib import Path

from yolo_sgg.core.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark modular YOLO-SGG pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to OmegaConf yaml")
    parser.add_argument("--scene-path", type=str, required=True, help="Path to scene folder with bbox GT")
    parser.add_argument("--output", type=str, default="metrics_data", help="Metrics output folder")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    from yolo_sgg.core.pipeline import TrackingSceneGraphPipeline
    from yolo_sgg.services.metrics_service import UnifiedMetricsService

    cfg = load_config(args.config)
    pipeline = TrackingSceneGraphPipeline(cfg)

    _persistent_graph, graph_per_frame = pipeline.run(collect_frame_graphs=True)

    metrics = UnifiedMetricsService()
    scene_path = Path(args.scene_path)
    results = metrics.evaluate_scene(scene_path, graph_per_frame, iou_threshold=args.iou_threshold)
    metrics.save(results, Path(args.output), scene_path.name)
    print(f"Saved benchmark metrics for {scene_path.name} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
