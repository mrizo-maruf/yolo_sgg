# 3D Object Tracking Metrics Evaluation

This module provides comprehensive evaluation of 3D object tracking performance using standard metrics: **HOTA**, **MOTP**, **MOTA**, and **IDF1**.

## 📁 Files Overview

- **`metrics_3d.py`**: Core metrics calculation (fully decoupled from tracking code)
- **`evaluate_tracking.py`**: Standalone evaluation script
- **`metrics_utils.py`**: Helper utilities for integration with existing pipeline

## 🚀 Quick Start

### Option 1: Standalone Evaluation

Run tracking and evaluation separately:

```bash
python evaluate_tracking.py
```

Edit the configuration in `evaluate_tracking.py` to point to your scene data.

### Option 2: Integration with Existing Pipeline

Add metrics collection to your `yolo_ssg.py`:

```python
from metrics_utils import MetricsCollector

def main(cfg):
    # ... existing code ...
    
    # Add this at the beginning
    metrics_collector = MetricsCollector(enable=cfg.get('collect_metrics', False))
    
    # In your frame loop, after creating current_graph:
    metrics_collector.add_frame(frame_idx, current_graph)
    
    # At the end
    if cfg.get('collect_metrics', False):
        metrics_collector.save(Path('metrics_data/tracked_graphs.pkl'))
        
        if cfg.get('evaluate_metrics', False):
            scene_path = Path(cfg.rgb_dir).parent
            metrics_collector.evaluate(scene_path, iou_threshold=0.5)
```

Update your config:
```python
cfg = OmegaConf.create({
    # ... existing config ...
    'collect_metrics': True,
    'evaluate_metrics': True,
})
```

### Option 3: Batch Evaluation of Multiple Scenes

```python
from metrics_utils import batch_evaluate_scenes
from pathlib import Path

scenes_root = Path("/path/to/scenes")
batch_evaluate_scenes(scenes_root, iou_threshold=0.5)
```

## 📊 Metrics Explained

### MOTA (Multiple Object Tracking Accuracy)
- **Range**: -∞ to 100%
- **Formula**: `MOTA = 1 - (FN + FP + IDSW) / GT`
- **Measures**: Overall tracking accuracy (detection + association)
- **Higher is better**

### MOTP (Multiple Object Tracking Precision)
- **Range**: 0 to ∞ meters
- **Formula**: `MOTP = Σ(distances) / matches`
- **Measures**: Average localization error of matched objects
- **Lower is better**

### IDF1 (ID F1 Score)
- **Range**: 0 to 100%
- **Formula**: `IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)`
- **Measures**: Identity preservation over time
- **Higher is better**

### HOTA (Higher Order Tracking Accuracy)
- **Range**: 0 to 100%
- **Formula**: `HOTA = √(DetA × AssA)`
- **Measures**: Balance of detection and association accuracy
- **Higher is better**

## 📂 Ground Truth Data Format

Your scene folder should have this structure:

```
scene_folder/
├── rgb/
├── depth/
├── bbox/
│   ├── bboxes000001_info.json
│   ├── bboxes000002_info.json
│   └── ...
└── traj.txt
```

Each `bboxes*.json` file should contain:

```json
{
  "bboxes": {
    "bbox_3d": {
      "boxes": [
        {
          "track_id": 0,
          "label": "box,gray",
          "aabb_xyzmin_xyzmax": [xmin, ymin, zmin, xmax, ymax, zmax],
          "transform_4x4": [...],
          "occlusion_ratio": 0.0
        }
      ]
    }
  }
}
```

## 📈 Output Format

Metrics are saved to `metrics_data/` folder:

- **`{scene_name}_metrics.json`**: JSON format with all metrics
- **`{scene_name}_metrics.csv`**: CSV format for easy import to Excel/pandas

Example output:
```
============================================================
TRACKING METRICS SUMMARY - UR5-Peg-In-Hole_02_straight
============================================================
MOTA:  85.32%
MOTP:  0.0234 meters
IDF1:  89.45%
HOTA:  78.91%
  DetA: 82.34%
  AssA: 75.67%

Detection Stats:
  GT Objects:    450
  Matches:       412
  False Neg:     28
  False Pos:     15
  ID Switches:   5
============================================================
```

## ⚙️ Configuration

Key parameters you can adjust:

- **`iou_threshold`**: IoU threshold for matching (default: 0.5)
  - Higher = stricter matching
  - Typical values: 0.3 - 0.7

## 🔧 Advanced Usage

### Using the Core Metrics Module Directly

```python
from metrics_3d import (
    load_gt_data, 
    load_prediction_data, 
    TrackingMetrics3D,
    save_metrics
)
from pathlib import Path

# Load data
scene_path = Path("/path/to/scene")
gt_tracks = load_gt_data(scene_path)
pred_tracks = load_prediction_data(graph_per_frame)

# Compute metrics
metrics = TrackingMetrics3D(iou_threshold=0.5)
mota_motp = metrics.compute_mota_motp(gt_tracks, pred_tracks)
idf1 = metrics.compute_idf1(gt_tracks, pred_tracks)
hota = metrics.compute_hota(gt_tracks, pred_tracks)

# Save
results = {**mota_motp, **idf1, **hota}
save_metrics(results, Path("metrics_data"), "my_scene")
```

### Custom Metrics

Extend `TrackingMetrics3D` class to add custom metrics:

```python
from metrics_3d import TrackingMetrics3D

class CustomMetrics(TrackingMetrics3D):
    def compute_custom_metric(self, gt_tracks, pred_tracks):
        # Your implementation
        return metric_value
```

## 📝 Notes

- **3D IoU**: Uses axis-aligned bounding box (AABB) overlap
- **Distance**: Euclidean distance between bbox centers
- **ID Switches**: Detected when a prediction ID switches to a different GT ID
- **Temporal Consistency**: Metrics account for tracking across entire sequence

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError: Bbox folder not found`
- **Solution**: Ensure your scene folder has a `bbox/` subfolder with JSON files

**Issue**: `No matches found / MOTP is inf`
- **Solution**: Lower the `iou_threshold` or check bbox coordinate systems

**Issue**: `Different number of GT and prediction frames`
- **Solution**: This is normal - metrics handle missing frames gracefully

## 📚 References

- HOTA: [A Higher Order Metric for Evaluating Multi-Object Tracking](https://arxiv.org/abs/2009.07736)
- MOTA/MOTP: [CLEAR MOT Metrics](https://link.springer.com/article/10.1155/2008/246309)
- IDF1: [Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking](https://arxiv.org/abs/1609.01775)
