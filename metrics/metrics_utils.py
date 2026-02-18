"""
Helper utilities for integrating metrics evaluation into existing tracking code.
"""

from pathlib import Path
import networkx as nx
from typing import Dict
import pickle


class MetricsCollector:
    """
    Lightweight wrapper to collect tracking graphs per frame.
    Add this to your existing tracking pipeline.
    """
    
    def __init__(self, enable: bool = True):
        self.enable = enable
        self.graph_per_frame = {}
    
    def add_frame(self, frame_idx: int, graph: nx.MultiDiGraph):
        """Store a copy of the graph for a specific frame"""
        if self.enable:
            self.graph_per_frame[frame_idx] = graph.copy()
    
    def save(self, filepath: Path):
        """Save collected graphs to disk"""
        if self.enable:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph_per_frame, f)
            print(f"Saved {len(self.graph_per_frame)} frames to {filepath}")
    
    def load(self, filepath: Path):
        """Load collected graphs from disk"""
        with open(filepath, 'rb') as f:
            self.graph_per_frame = pickle.load(f)
        print(f"Loaded {len(self.graph_per_frame)} frames from {filepath}")
        return self.graph_per_frame
    
    def evaluate(self, scene_path: Path, iou_threshold: float = 0.5):
        """Evaluate metrics on collected data"""
        from metrics.metrics_3d import evaluate_tracking, save_metrics
        
        results = evaluate_tracking(
            scene_path=scene_path,
            graph_per_frame=self.graph_per_frame,
            iou_threshold=iou_threshold
        )
        
        # Save results
        output_path = Path("metrics_data")
        scene_name = scene_path.name
        save_metrics(results, output_path, scene_name=scene_name)
        
        return results


def add_metrics_to_existing_pipeline():
    """
    Example of how to modify your existing yolo_ssg.py to collect metrics.
    
    Add these lines to your main() function in yolo_ssg.py:
    
    ```python
    # At the top of main():
    from metrics_utils import MetricsCollector
    metrics_collector = MetricsCollector(enable=cfg.get('collect_metrics', False))
    
    # In your frame loop, after creating current_graph:
    metrics_collector.add_frame(frame_idx, current_graph)
    
    # At the end of main(), before return:
    if cfg.get('collect_metrics', False):
        metrics_collector.save(Path('metrics_data/tracked_graphs.pkl'))
        
        # Optionally evaluate immediately
        if cfg.get('evaluate_metrics', False):
            scene_path = Path(cfg.rgb_dir).parent  # Assuming scene structure
            metrics_collector.evaluate(scene_path, iou_threshold=0.5)
    ```
    
    Then in your config:
    ```python
    cfg = OmegaConf.create({
        # ... existing config ...
        'collect_metrics': True,
        'evaluate_metrics': True,
    })
    ```
    """
    pass


def batch_evaluate_scenes(scenes_root: Path, iou_threshold: float = 0.5):
    """
    Evaluate multiple scenes at once.
    
    Args:
        scenes_root: Path to folder containing multiple scene folders
        iou_threshold: IoU threshold for matching
    """
    from metrics.metrics_3d import evaluate_tracking, save_metrics
    import json
    
    scene_folders = [f for f in scenes_root.iterdir() if f.is_dir()]
    
    all_results = {}
    
    for scene_path in scene_folders:
        scene_name = scene_path.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {scene_name}")
        print('='*60)
        
        # Check if tracking data exists
        graph_file = Path('metrics_data') / f'{scene_name}_tracked_graphs.pkl'
        if not graph_file.exists():
            print(f"Warning: No tracking data found at {graph_file}")
            continue
        
        # Load tracking data
        collector = MetricsCollector()
        graph_per_frame = collector.load(graph_file)
        
        # Evaluate
        try:
            results = evaluate_tracking(
                scene_path=scene_path,
                graph_per_frame=graph_per_frame,
                iou_threshold=iou_threshold
            )
            
            output_path = Path("metrics_data")
            save_metrics(results, output_path, scene_name=scene_name)
            
            all_results[scene_name] = results
            
        except Exception as e:
            print(f"Error evaluating {scene_name}: {e}")
            continue
    
    # Save aggregate results
    if all_results:
        aggregate_path = Path("metrics_data") / "all_scenes_summary.json"
        with open(aggregate_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nSaved aggregate results to {aggregate_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print(f"{'Scene':<30} {'MOTA':<10} {'MOTP':<10} {'IDF1':<10} {'HOTA':<10}")
        print("="*80)
        for scene_name, results in all_results.items():
            print(f"{scene_name:<30} "
                  f"{results.get('MOTA', 0):>8.2f}% "
                  f"{results.get('MOTP', 0):>8.4f}  "
                  f"{results.get('IDF1', 0):>8.2f}% "
                  f"{results.get('HOTA', 0):>8.2f}%")
        print("="*80)


if __name__ == "__main__":
    print(__doc__)
    print(add_metrics_to_existing_pipeline.__doc__)
