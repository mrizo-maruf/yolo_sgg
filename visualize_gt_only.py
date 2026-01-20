"""
Simple script to visualize ground truth 3D bounding boxes only.
"""

from pathlib import Path
import json
import open3d as o3d
import numpy as np
import argparse


def create_bbox_lineset(xmin, ymin, zmin, xmax, ymax, zmax, color=[1, 0, 0]):
    """Create Open3D LineSet for a bounding box"""
    # Define the 8 corners
    points = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ]
    
    # Define the 12 edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def visualize_gt_frame(scene_path, frame_id):
    """Visualize ground truth bboxes for a specific frame"""
    
    scene_path = Path(scene_path)
    bbox_folder = scene_path / 'bbox'
    
    if not bbox_folder.exists():
        print(f"Error: Bbox folder not found at {bbox_folder}")
        return
    
    # Find the bbox file for this frame
    # Frame ID in filename is 1-indexed
    bbox_file = bbox_folder / f"bboxes{frame_id+1:06d}_info.json"
    
    if not bbox_file.exists():
        print(f"Error: Bbox file not found: {bbox_file}")
        return
    
    # Load the JSON
    with open(bbox_file, 'r') as f:
        data = json.load(f)
    
    # Extract bboxes
    boxes_data = data.get('bboxes', {}).get('bbox_3d', {}).get('boxes', [])
    
    if len(boxes_data) == 0:
        print(f"No boxes found in frame {frame_id}")
        return
    
    print(f"\n{'='*60}")
    print(f"Frame {frame_id}: Found {len(boxes_data)} GT boxes")
    print(f"{'='*60}\n")
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Ground Truth Frame {frame_id}", width=1280, height=720)
    
    # Add each bbox
    for box_data in boxes_data:
        track_id = box_data.get('track_id')
        label = box_data.get('label', 'unknown')
        aabb = box_data.get('aabb_xyzmin_xyzmax')
        
        if aabb and len(aabb) == 6:
            xmin, ymin, zmin, xmax, ymax, zmax = aabb
            
            # Create random color based on track_id
            color = [
                (track_id * 0.37) % 1.0,
                (track_id * 0.61) % 1.0,
                (track_id * 0.83) % 1.0
            ]
            
            line_set = create_bbox_lineset(xmin, ymin, zmin, xmax, ymax, zmax, color=color)
            vis.add_geometry(line_set)
            
            # Print info
            cx, cy, cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
            sx, sy, sz = xmax-xmin, ymax-ymin, zmax-zmin
            
            print(f"Box {track_id} ({label}):")
            print(f"  Center: [{cx:.3f}, {cy:.3f}, {cz:.3f}]")
            print(f"  Size:   [{sx:.3f}, {sy:.3f}, {sz:.3f}]")
            print(f"  AABB:   [{xmin:.3f}, {ymin:.3f}, {zmin:.3f}] to [{xmax:.3f}, {ymax:.3f}, {zmax:.3f}]")
            print()
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    print(f"Press Q to close window")
    print(f"{'='*60}\n")
    
    # Run
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Visualize ground truth 3D bounding boxes")
    parser.add_argument('--scene', type=str, required=True, help='Path to scene folder')
    parser.add_argument('--frames', type=int, nargs='+', default=[0], help='Frame indices to visualize (0-indexed)')
    
    args = parser.parse_args()
    
    for frame_id in args.frames:
        visualize_gt_frame(args.scene, frame_id)


if __name__ == "__main__":
    # For quick testing, uncomment and modify:
    # visualize_gt_frame("/home/maribjonov_mr/IsaacSim_bench/cabinet_simple", 0)
    
    main()
