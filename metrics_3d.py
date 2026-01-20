"""
3D Object Tracking Metrics Evaluation
Calculates HOTA, MOTP, MOTA, IDF1 for 3D bounding box tracking.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import csv
import open3d as o3d

class BBox3D:
    """3D Bounding Box representation"""
    def __init__(self, track_id: int, aabb: List[float], frame_id: int):
        self.track_id = track_id
        self.frame_id = frame_id
        # aabb format: [xmin, ymin, zmin, xmax, ymax, zmax]
        self.xmin, self.ymin, self.zmin = aabb[0], aabb[1], aabb[2]
        self.xmax, self.ymax, self.zmax = aabb[3], aabb[4], aabb[5]
        self.center = np.array([
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2
        ])
    
    def compute_iou_3d(self, other: 'BBox3D') -> float:
        """Compute 3D IoU with another bounding box"""
        # Intersection
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_zmin = max(self.zmin, other.zmin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)
        inter_zmax = min(self.zmax, other.zmax)
        
        if inter_xmin >= inter_xmax or inter_ymin >= inter_ymax or inter_zmin >= inter_zmax:
            return 0.0
        
        inter_volume = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin) * (inter_zmax - inter_zmin)
        
        # Volumes
        self_volume = (self.xmax - self.xmin) * (self.ymax - self.ymin) * (self.zmax - self.zmin)
        other_volume = (other.xmax - other.xmin) * (other.ymax - other.ymin) * (other.zmax - other.zmin)
        
        union_volume = self_volume + other_volume - inter_volume
        
        return inter_volume / union_volume if union_volume > 0 else 0.0
    
    def distance_to(self, other: 'BBox3D') -> float:
        """Compute Euclidean distance between centers"""
        return np.linalg.norm(self.center - other.center)


class TrackingMetrics3D:
    """Calculate 3D tracking metrics"""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        
    def match_frames(self, gt_boxes: List[BBox3D], pred_boxes: List[BBox3D]) -> Tuple[Dict, List[int], List[int]]:
        """
        Match predictions to ground truth using Hungarian algorithm (greedy for now).
        
        Returns:
            matches: dict {pred_idx: gt_idx}
            unmatched_gt: list of unmatched GT indices
            unmatched_pred: list of unmatched prediction indices
        """
        if len(gt_boxes) == 0:
            return {}, [], list(range(len(pred_boxes)))
        if len(pred_boxes) == 0:
            return {}, list(range(len(gt_boxes))), []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j] = pred.compute_iou_3d(gt)
        
        # Debug: print max IoU
        if iou_matrix.size > 0:
            max_iou = np.max(iou_matrix)
            if max_iou < self.iou_threshold:
                # Print diagnostic info for first mismatch
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                pred_idx, gt_idx = max_idx
                pred_box = pred_boxes[pred_idx]
                gt_box = gt_boxes[gt_idx]
                print(f"[DEBUG] Frame {pred_box.frame_id}: Max IoU {max_iou:.4f} < threshold {self.iou_threshold}")
                print(f"[DEBUG]   Pred center: [{pred_box.center[0]:.3f}, {pred_box.center[1]:.3f}, {pred_box.center[2]:.3f}]")
                print(f"[DEBUG]   GT center:   [{gt_box.center[0]:.3f}, {gt_box.center[1]:.3f}, {gt_box.center[2]:.3f}]")
                print(f"[DEBUG]   Distance: {pred_box.distance_to(gt_box):.3f} meters")
        
        # Greedy matching (better: use Hungarian algorithm from scipy)
        matches = {}
        matched_gt = set()
        matched_pred = set()
        
        # Sort by IoU (highest first)
        iou_pairs = []
        for i in range(len(pred_boxes)):
            for j in range(len(gt_boxes)):
                if iou_matrix[i, j] >= self.iou_threshold:
                    iou_pairs.append((iou_matrix[i, j], i, j))
        
        iou_pairs.sort(reverse=True)
        
        for iou, pred_idx, gt_idx in iou_pairs:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matches[pred_idx] = gt_idx
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
        
        unmatched_gt = [j for j in range(len(gt_boxes)) if j not in matched_gt]
        unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
        
        return matches, unmatched_gt, unmatched_pred
    
    def compute_mota_motp(self, gt_tracks: Dict[int, List[BBox3D]], 
                          pred_tracks: Dict[int, List[BBox3D]]) -> Dict[str, float]:
        """
        Compute MOTA and MOTP.
        
        MOTA = 1 - (FN + FP + IDSW) / GT
        MOTP = sum(distances) / matches
        """
        total_gt = 0
        total_fn = 0  # False Negatives (missed detections)
        total_fp = 0  # False Positives
        total_idsw = 0  # ID Switches
        total_distance = 0.0
        total_matches = 0
        
        # Get all frames
        all_frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
        
        # Track ID mapping for detecting ID switches
        prev_matches = {}  # {pred_track_id: gt_track_id}
        
        for frame_id in all_frames:
            gt_boxes = gt_tracks.get(frame_id, [])
            pred_boxes = pred_tracks.get(frame_id, [])
            
            total_gt += len(gt_boxes)
            
            # Match boxes
            matches, unmatched_gt, unmatched_pred = self.match_frames(gt_boxes, pred_boxes)
            
            # Count errors
            total_fn += len(unmatched_gt)
            total_fp += len(unmatched_pred)
            
            # Compute distances and detect ID switches
            for pred_idx, gt_idx in matches.items():
                pred_box = pred_boxes[pred_idx]
                gt_box = gt_boxes[gt_idx]
                
                # MOTP: sum of distances
                total_distance += pred_box.distance_to(gt_box)
                total_matches += 1
                
                # ID Switch detection
                pred_id = pred_box.track_id
                gt_id = gt_box.track_id
                
                if pred_id in prev_matches:
                    if prev_matches[pred_id] != gt_id:
                        total_idsw += 1
                
                prev_matches[pred_id] = gt_id
        
        # Calculate metrics
        mota = 1 - (total_fn + total_fp + total_idsw) / total_gt if total_gt > 0 else 0.0
        motp = total_distance / total_matches if total_matches > 0 else float('inf')
        
        return {
            'MOTA': mota * 100,  # Convert to percentage
            'MOTP': motp,
            'FN': total_fn,
            'FP': total_fp,
            'IDSW': total_idsw,
            'GT': total_gt,
            'Matches': total_matches
        }
    
    def compute_idf1(self, gt_tracks: Dict[int, List[BBox3D]], 
                     pred_tracks: Dict[int, List[BBox3D]]) -> float:
        """
        Compute IDF1 (ID F1 Score).
        
        IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
        where IDTP, IDFP, IDFN are computed over the entire sequence.
        """
        # Build track-to-track associations
        all_frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
        
        # Count ID-based matches
        idtp = 0  # ID True Positives
        idfp = 0  # ID False Positives
        idfn = 0  # ID False Negatives
        
        # Track correspondence matrix: {(gt_id, pred_id): count}
        correspondence = defaultdict(int)
        
        for frame_id in all_frames:
            gt_boxes = gt_tracks.get(frame_id, [])
            pred_boxes = pred_tracks.get(frame_id, [])
            
            matches, unmatched_gt, unmatched_pred = self.match_frames(gt_boxes, pred_boxes)
            
            for pred_idx, gt_idx in matches.items():
                gt_id = gt_boxes[gt_idx].track_id
                pred_id = pred_boxes[pred_idx].track_id
                correspondence[(gt_id, pred_id)] += 1
        
        # Compute IDTP: best one-to-one mapping
        matched_gt = set()
        matched_pred = set()
        
        # Sort by correspondence count
        sorted_pairs = sorted(correspondence.items(), key=lambda x: x[1], reverse=True)
        
        for (gt_id, pred_id), count in sorted_pairs:
            if gt_id not in matched_gt and pred_id not in matched_pred:
                idtp += count
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)
        
        # Count total detections for IDFP and IDFN
        total_gt_dets = sum(len(boxes) for boxes in gt_tracks.values())
        total_pred_dets = sum(len(boxes) for boxes in pred_tracks.values())
        
        idfn = total_gt_dets - idtp
        idfp = total_pred_dets - idtp
        
        idf1 = 2 * idtp / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0.0
        
        return {
            'IDF1': idf1 * 100,  # Convert to percentage
            'IDTP': idtp,
            'IDFP': idfp,
            'IDFN': idfn
        }
    
    def compute_hota(self, gt_tracks: Dict[int, List[BBox3D]], 
                     pred_tracks: Dict[int, List[BBox3D]]) -> Dict[str, float]:
        """
        Compute HOTA (Higher Order Tracking Accuracy).
        Simplified implementation.
        """
        all_frames = sorted(set(list(gt_tracks.keys()) + list(pred_tracks.keys())))
        
        alpha = 0.5  # Weight between detection and association
        
        tpa = 0  # True Positive Association
        total_matches = 0
        
        # Track associations
        gt_id_to_pred = defaultdict(set)
        pred_id_to_gt = defaultdict(set)
        
        for frame_id in all_frames:
            gt_boxes = gt_tracks.get(frame_id, [])
            pred_boxes = pred_tracks.get(frame_id, [])
            
            matches, _, _ = self.match_frames(gt_boxes, pred_boxes)
            
            for pred_idx, gt_idx in matches.items():
                gt_id = gt_boxes[gt_idx].track_id
                pred_id = pred_boxes[pred_idx].track_id
                
                gt_id_to_pred[gt_id].add(pred_id)
                pred_id_to_gt[pred_id].add(gt_id)
                total_matches += 1
        
        # Calculate association accuracy
        for gt_id, pred_ids in gt_id_to_pred.items():
            if len(pred_ids) == 1:  # One-to-one mapping
                pred_id = list(pred_ids)[0]
                if len(pred_id_to_gt[pred_id]) == 1:
                    tpa += 1
        
        # Simplified HOTA calculation
        num_gt_ids = len(gt_id_to_pred)
        num_pred_ids = len(pred_id_to_gt)
        
        deta = total_matches / (num_gt_ids + num_pred_ids) if (num_gt_ids + num_pred_ids) > 0 else 0.0
        assa = tpa / max(num_gt_ids, num_pred_ids) if max(num_gt_ids, num_pred_ids) > 0 else 0.0
        
        hota = np.sqrt(deta * assa)
        
        return {
            'HOTA': hota * 100,
            'DetA': deta * 100,
            'AssA': assa * 100
        }


def load_gt_data(scene_path: Path, 
                 max_box_edge: float = 20.0,
                 ignore_prim_prefixes: List[str] = None) -> Dict[int, List[BBox3D]]:
    """
    Load ground truth data from bbox JSON files.
    
    The bboxes are expected to be in WORLD coordinates (already transformed during
    data collection using the annotator's row-major transform matrix).
    
    Args:
        scene_path: Path to scene folder containing 'bbox' subfolder
        max_box_edge: Maximum allowed edge length (filters large/invalid boxes)
        ignore_prim_prefixes: List of prim path prefixes to ignore (e.g., ["/World/env"])
    
    Returns:
        Dict mapping frame_id to list of BBox3D objects
    """
    if ignore_prim_prefixes is None:
        ignore_prim_prefixes = []
    
    bbox_folder = scene_path / 'bbox'
    if not bbox_folder.exists():
        raise FileNotFoundError(f"Bbox folder not found: {bbox_folder}")
    
    gt_tracks = defaultdict(list)
    
    # Find all bbox JSON files
    bbox_files = sorted(bbox_folder.glob('bboxes*.json'))
    
    skipped_large = 0
    skipped_prim = 0
    skipped_invalid = 0
    
    for bbox_file in bbox_files:
        # Extract frame number from filename (e.g., bboxes000001_info.json -> 1)
        frame_num_str = bbox_file.stem.replace('bboxes', '').replace('_info', '')
        try:
            frame_id = int(frame_num_str)
        except ValueError:
            print(f"Warning: Could not parse frame number from {bbox_file.name}")
            continue
        
        # Load JSON
        with open(bbox_file, 'r') as f:
            data = json.load(f)
        
        # Extract bboxes - same structure as gt_vis.py
        if 'bboxes' in data and 'bbox_3d' in data['bboxes']:
            boxes_data = data['bboxes']['bbox_3d'].get('boxes', [])
            
            for box_data in boxes_data:
                # Get prim path for filtering
                prim_path = box_data.get('prim_path', '')
                
                # Skip ignored prims (like background environment)
                should_skip = False
                for prefix in ignore_prim_prefixes:
                    if prim_path and prim_path.startswith(prefix):
                        should_skip = True
                        skipped_prim += 1
                        break
                if should_skip:
                    continue
                
                # Get track_id and aabb
                track_id = box_data.get('track_id', box_data.get('bbox_id'))
                aabb = box_data.get('aabb_xyzmin_xyzmax')
                
                if track_id is None or aabb is None:
                    skipped_invalid += 1
                    continue
                
                # Parse AABB - boxes are already in WORLD coordinates after data collection fix
                xmin, ymin, zmin, xmax, ymax, zmax = map(float, aabb)
                
                # Filter by box size (same as gt_vis.py)
                sx = xmax - xmin
                sy = ymax - ymin
                sz = zmax - zmin
                if max(sx, sy, sz) > max_box_edge:
                    skipped_large += 1
                    continue
                
                # Create BBox3D object
                bbox = BBox3D(track_id, aabb, frame_id)
                gt_tracks[frame_id].append(bbox)
    
    print(f"[GT Load] Loaded {len(gt_tracks)} frames, "
          f"total boxes: {sum(len(boxes) for boxes in gt_tracks.values())}")
    print(f"[GT Load] Skipped: {skipped_large} too large, {skipped_prim} ignored prims, {skipped_invalid} invalid")
    
    if gt_tracks:
        sample_frame = min(gt_tracks.keys())
        if gt_tracks[sample_frame]:
            sample_box = gt_tracks[sample_frame][0]
            print(f"[GT Load] Sample - Track ID: {sample_box.track_id}, Frame: {sample_frame}")
            print(f"[GT Load]   Center: [{sample_box.center[0]:.3f}, {sample_box.center[1]:.3f}, {sample_box.center[2]:.3f}]")
            print(f"[GT Load]   Size: [{sample_box.xmax-sample_box.xmin:.3f}, {sample_box.ymax-sample_box.ymin:.3f}, {sample_box.zmax-sample_box.zmin:.3f}]")
    
    return gt_tracks


def load_prediction_data(graph_per_frame: Dict[int, 'nx.MultiDiGraph']) -> Dict[int, List[BBox3D]]:
    """
    Load prediction data from NetworkX graphs.
    
    Args:
        graph_per_frame: Dict mapping frame_id to MultiDiGraph with tracking results
    
    Returns:
        Dict mapping frame_id to list of BBox3D objects
    """
    pred_tracks = defaultdict(list)
    
    # Debug: check first frame structure
    if graph_per_frame and len(graph_per_frame) > 0:
        first_frame_id = min(graph_per_frame.keys())
        first_graph = graph_per_frame[first_frame_id]
        print(f"\n[DEBUG] First frame ({first_frame_id}) has {first_graph.number_of_nodes()} nodes")
        if first_graph.number_of_nodes() > 0:
            sample_node = list(first_graph.nodes(data=True))[0]
            print(f"[DEBUG] Sample node: {sample_node[0]}")
            print(f"[DEBUG] Available attributes: {list(sample_node[1].keys())}")
            if 'aabb_xyzmin_xyzmax' in sample_node[1]:
                print(f"[DEBUG] Sample AABB: {sample_node[1]['aabb_xyzmin_xyzmax']}")
            else:
                print(f"[DEBUG] WARNING: 'aabb_xyzmin_xyzmax' not found in node attributes!")
    
    for frame_id, graph in graph_per_frame.items():
        for node_id, node_data in graph.nodes(data=True):
            data = node_data.get('data', {})
            
            track_id = data.get('track_id')
            bbox_3d = data.get('bbox_3d', {})
            aabb = bbox_3d.get('aabb')
            
            if track_id is not None and aabb is not None:
                # Convert AABB dict to list format
                aabb_list = [
                    aabb['min'][0], aabb['min'][1], aabb['min'][2],
                    aabb['max'][0], aabb['max'][1], aabb['max'][2]
                ]
                bbox = BBox3D(track_id, aabb_list, frame_id)
                pred_tracks[frame_id].append(bbox)
    
    print(f"[DEBUG] Loaded predictions: {len(pred_tracks)} frames, "
          f"total boxes: {sum(len(boxes) for boxes in pred_tracks.values())}")
    if pred_tracks:
        sample_frame = min(pred_tracks.keys())
        if pred_tracks[sample_frame]:
            sample_box = pred_tracks[sample_frame][0]
            print(f"[DEBUG] Sample prediction - Track ID: {sample_box.track_id}, Frame: {sample_frame}")
            print(f"[DEBUG]   Center: [{sample_box.center[0]:.3f}, {sample_box.center[1]:.3f}, {sample_box.center[2]:.3f}]")
            print(f"[DEBUG]   Size: [{sample_box.xmax-sample_box.xmin:.3f}, {sample_box.ymax-sample_box.ymin:.3f}, {sample_box.zmax-sample_box.zmin:.3f}]")
    
    return pred_tracks


def create_bbox_lineset(bbox: BBox3D, color=[1, 0, 0]):
    """Create Open3D LineSet for a bounding box"""
    # Define the 8 corners of the box
    points = [
        [bbox.xmin, bbox.ymin, bbox.zmin],
        [bbox.xmax, bbox.ymin, bbox.zmin],
        [bbox.xmax, bbox.ymax, bbox.zmin],
        [bbox.xmin, bbox.ymax, bbox.zmin],
        [bbox.xmin, bbox.ymin, bbox.zmax],
        [bbox.xmax, bbox.ymin, bbox.zmax],
        [bbox.xmax, bbox.ymax, bbox.zmax],
        [bbox.xmin, bbox.ymax, bbox.zmax],
    ]
    
    # Define the 12 edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set


def visualize_frame_comparison(frame_id: int, 
                               gt_tracks: Dict[int, List[BBox3D]],
                               pred_tracks: Dict[int, List[BBox3D]],
                               graph_per_frame: Dict[int, 'nx.MultiDiGraph'] = None,
                               show_points: bool = False):
    """
    Visualize GT (RED) and predictions (GREEN) in the same window.
    
    Args:
        frame_id: Frame number to visualize
        gt_tracks: Ground truth bounding boxes
        pred_tracks: Predicted bounding boxes
        graph_per_frame: Optional graph data (unused)
        show_points: Whether to show point clouds (unused)
    """
    print(f"\n{'='*60}")
    print(f"Frame {frame_id}")
    print(f"{'='*60}")
    
    gt_boxes = gt_tracks.get(frame_id, [])
    pred_boxes = pred_tracks.get(frame_id, [])
    
    print(f"GT boxes: {len(gt_boxes)}, Pred boxes: {len(pred_boxes)}")
    
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        print("No data to visualize for this frame.")
        return
    
    # Create single visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_id} - GT(RED) vs Pred(GREEN)", width=1280, height=720)
    
    # Add GT bounding boxes (RED)
    print("\nGround Truth (RED):")
    for bbox in gt_boxes:
        line_set = create_bbox_lineset(bbox, color=[1, 0, 0])  # RED for GT
        vis.add_geometry(line_set)
        print(f"  GT {bbox.track_id}: center=[{bbox.center[0]:.3f}, {bbox.center[1]:.3f}, {bbox.center[2]:.3f}], "
              f"size=[{bbox.xmax-bbox.xmin:.3f}, {bbox.ymax-bbox.ymin:.3f}, {bbox.zmax-bbox.zmin:.3f}]")
    
    # Add prediction bounding boxes (GREEN)
    print("\nPredictions (GREEN):")
    for bbox in pred_boxes:
        line_set = create_bbox_lineset(bbox, color=[0, 1, 0])  # GREEN for predictions
        vis.add_geometry(line_set)
        print(f"  Pred {bbox.track_id}: center=[{bbox.center[0]:.3f}, {bbox.center[1]:.3f}, {bbox.center[2]:.3f}], "
              f"size=[{bbox.xmax-bbox.xmin:.3f}, {bbox.ymax-bbox.ymin:.3f}, {bbox.zmax-bbox.zmin:.3f}]")
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coord_frame)
    
    # Compute and display IoU for all pairs
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        print("\nIoU Matrix:")
        max_iou = 0
        for pred in pred_boxes:
            for gt in gt_boxes:
                iou = pred.compute_iou_3d(gt)
                dist = pred.distance_to(gt)
                max_iou = max(max_iou, iou)
                if iou > 0.001 or dist < 2.0:
                    print(f"  Pred {pred.track_id} <-> GT {gt.track_id}: IoU={iou:.4f}, dist={dist:.3f}m")
        if max_iou < 0.001:
            print("  No overlapping boxes found!")
    
    print(f"\nPress Q to continue to next frame...")
    print(f"{'='*60}\n")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def evaluate_tracking(scene_path: Path, 
                     graph_per_frame: Dict[int, 'nx.MultiDiGraph'],
                     iou_threshold: float = 0.5,
                     max_box_edge: float = 20.0,
                     ignore_prim_prefixes: List[str] = None) -> Dict[str, any]:
    """
    Evaluate 3D tracking metrics.
    
    Args:
        scene_path: Path to scene folder containing bbox ground truth
        graph_per_frame: Dict of frame_id -> NetworkX graph with predictions
        iou_threshold: IoU threshold for matching
        max_box_edge: Maximum allowed edge length for GT boxes
        ignore_prim_prefixes: Prim path prefixes to ignore in GT
    
    Returns:
        Dict containing all metrics
    """
    # Load data with same filtering as gt_vis.py
    print(f"Loading ground truth from {scene_path}...")
    gt_tracks = load_gt_data(
        scene_path, 
        max_box_edge=max_box_edge,
        ignore_prim_prefixes=ignore_prim_prefixes
    )
    
    print(f"Loading predictions...")
    pred_tracks = load_prediction_data(graph_per_frame)
    
    print(f"\n[DEBUG] GT frames: {sorted(list(gt_tracks.keys())[:10])}...")
    print(f"[DEBUG] Pred frames: {sorted(list(pred_tracks.keys())[:10])}...")
    print(f"[DEBUG] Total GT frames: {len(gt_tracks)}, Pred frames: {len(pred_tracks)}")
    
    # Check for frame overlap
    gt_frame_set = set(gt_tracks.keys())
    pred_frame_set = set(pred_tracks.keys())
    common_frames = gt_frame_set & pred_frame_set
    print(f"[DEBUG] Common frames: {len(common_frames)}")
    if len(common_frames) == 0:
        print("[ERROR] No overlapping frames between GT and predictions!")
        print(f"[ERROR] This suggests a frame numbering mismatch.")
        print(f"[ERROR] GT frames start at: {min(gt_tracks.keys()) if gt_tracks else 'N/A'}")
        print(f"[ERROR] Pred frames start at: {min(pred_tracks.keys()) if pred_tracks else 'N/A'}")
    
    # Initialize metrics calculator
    metrics_calc = TrackingMetrics3D(iou_threshold=iou_threshold)
    
    # Compute metrics
    print("Computing MOTA/MOTP...")
    mota_motp = metrics_calc.compute_mota_motp(gt_tracks, pred_tracks)
    
    print("Computing IDF1...")
    idf1_results = metrics_calc.compute_idf1(gt_tracks, pred_tracks)
    
    print("Computing HOTA...")
    hota_results = metrics_calc.compute_hota(gt_tracks, pred_tracks)
    
    # Combine results
    results = {
        **mota_motp,
        **idf1_results,
        **hota_results,
        'IoU_Threshold': iou_threshold
    }
    
    return results


def save_metrics(results: Dict, output_path: Path, scene_name: str = "scene"):
    """Save metrics to CSV and JSON files"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_path / f"{scene_name}_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {json_path}")
    
    # Save as CSV
    csv_path = output_path / f"{scene_name}_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            writer.writerow([key, value])
    print(f"Saved metrics to {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"TRACKING METRICS SUMMARY - {scene_name}")
    print("="*60)
    print(f"MOTA:  {results.get('MOTA', 0):.2f}%")
    print(f"MOTP:  {results.get('MOTP', 0):.4f} meters")
    print(f"IDF1:  {results.get('IDF1', 0):.2f}%")
    print(f"HOTA:  {results.get('HOTA', 0):.2f}%")
    print(f"  DetA: {results.get('DetA', 0):.2f}%")
    print(f"  AssA: {results.get('AssA', 0):.2f}%")
    print(f"\nDetection Stats:")
    print(f"  GT Objects:    {results.get('GT', 0)}")
    print(f"  Matches:       {results.get('Matches', 0)}")
    print(f"  False Neg:     {results.get('FN', 0)}")
    print(f"  False Pos:     {results.get('FP', 0)}")
    print(f"  ID Switches:   {results.get('IDSW', 0)}")
    print("="*60)


if __name__ == "__main__":
    # Example usage
    scene_path = Path("/path/to/scene")
    
    # Dummy graph data for testing
    import networkx as nx
    graph_per_frame = {}
    
    # Run evaluation
    results = evaluate_tracking(scene_path, graph_per_frame, iou_threshold=0.5)
    
    # Save results
    output_path = Path("metrics_data")
    save_metrics(results, output_path, scene_name="test_scene")
