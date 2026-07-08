from pathlib import Path

from metrics.metrics_3d import evaluate_tracking, save_metrics


class UnifiedMetricsService:
    def evaluate_scene(self, scene_path: Path, graph_per_frame: dict, iou_threshold: float = 0.5):
        return evaluate_tracking(
            scene_path=scene_path,
            graph_per_frame=graph_per_frame,
            iou_threshold=iou_threshold,
        )

    def save(self, results: dict, output_path: Path, scene_name: str):
        save_metrics(results, output_path, scene_name=scene_name)
