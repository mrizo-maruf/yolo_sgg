"""
Standalone script to visualize GT vs Predictions side-by-side.
Useful for debugging coordinate system mismatches and scale issues.
"""

from pathlib import Path
from metrics_3d import load_gt_data, load_prediction_data, visualize_frame_comparison
from metrics_utils import MetricsCollector
import argparse


def main():
    parser = argparse.ArgumentParser(description="Visualize tracking results vs ground truth")
    parser.add_argument('--scene', type=str, required=True, help='Path to scene folder')
    parser.add_argument('--tracked-graphs', type=str, required=True, help='Path to tracked_graphs.pkl file')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 5, 10], help='Frame indices to visualize')
    parser.add_argument('--no-points', action='store_true', help='Hide point clouds, show only boxes')
    
    args = parser.parse_args()
    
    scene_path = Path(args.scene)
    graph_file = Path(args.tracked_graphs)
    
    print(f"Loading scene: {scene_path.name}")
    print(f"Loading tracked graphs: {graph_file}")
    
    # Load data
    collector = MetricsCollector()
    graph_per_frame = collector.load(graph_file)
    
    print("Loading ground truth...")
    gt_tracks = load_gt_data(scene_path)
    
    print("Loading predictions...")
    pred_tracks = load_prediction_data(graph_per_frame)
    
    print(f"\nGT frames: {len(gt_tracks)}, Pred frames: {len(pred_tracks)}")
    
    # Visualize requested frames
    for frame_id in args.frames:
        if frame_id not in graph_per_frame:
            print(f"Warning: Frame {frame_id} not found in tracking data")
            continue
        
        visualize_frame_comparison(
            frame_id=frame_id,
            gt_tracks=gt_tracks,
            pred_tracks=pred_tracks,
            graph_per_frame=graph_per_frame,
            show_points=not args.no_points
        )


if __name__ == "__main__":
    main()
