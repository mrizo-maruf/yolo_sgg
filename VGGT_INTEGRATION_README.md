# VGGT Integration for Scene Graph Generation

1. in 521: ```python predict_and_save.py --rgb_dir ./rgb_last --output_d ir ./predictions_last```
2. local vggt: ``` python extract_frame_data.py --data_dir ./predictions --rgb_dir ./rgb_last --output_dir ./frame_data_vggt_last```
3. local yoloe scene graph gen: ```python yolo_ssg_e.py```



## Overview
I've successfully integrated VGGT (Visual Geometry Grounded Trajectory) predictions into your scene graph generation pipeline. This allows you to generate scene graphs from RGB images using VGGT-predicted camera parameters and point clouds, without needing pre-computed depth maps or camera trajectories.

## Key Changes Made

### 1. New Functions Added to `yolo_ssg_e.py`

#### `load_vggt_frame_data(frame_data_dir, frame_idx)`
- Loads VGGT predicted camera parameters and point clouds for a specific frame
- Returns: camera_params (dict), points (N,3), colors (N,3)
- Handles missing files gracefully

#### `extract_points_from_vggt_pointcloud(vggt_points, vggt_colors, mask_2d, intrinsic, T_w_c, max_points, random_state)`
- Projects VGGT world-frame point clouds to 2D image plane
- Filters points that fall within YOLO segmentation masks
- Returns selected points and colors in world frame
- Supports point sampling to limit computational cost

### 2. Updated Pipeline Architecture

**Old Pipeline:**
```
RGB → YOLO → Masks → Depth Maps → Unproject → 3D Points → Scene Graph
```

**New VGGT Pipeline:**
```
RGB → YOLO → Masks → VGGT Point Cloud → Project+Filter → 3D Points → Scene Graph
```

### 3. Modified Main Function
- Replaced depth cache loading with VGGT frame data loading
- Direct YOLO processing on RGB images instead of video stream
- Point cloud filtering using mask projection instead of depth unprojection
- Maintained compatibility with existing edge prediction and visualization

## Configuration

### Required Paths
```python
'rgb_dir': "/path/to/rgb/images"                    # RGB images for YOLO
'vggt_frame_data_dir': "/path/to/vggt/frame/data"   # VGGT predictions
```

### Expected VGGT Data Structure
```
frame_data_vggt/
├── frame_000000_camera.json      # Camera parameters
├── frame_000000_pointcloud.npz   # Point cloud data
├── frame_000001_camera.json
├── frame_000001_pointcloud.npz
└── ...
```

### Camera JSON Format
```json
{
  "intrinsic": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "extrinsic_Twc": [[R11, R12, R13, tx], ...],  // 4x4 world-from-camera
  "camera_position": [x, y, z],
  "num_points": 38073,
  "has_colors": true
}
```

### Point Cloud NPZ Format
```python
{
  'points': (N, 3),   # 3D points in world frame
  'colors': (N, 3)    # RGB colors [0,1]
}
```

## Key Features

### 1. Automatic Frame Processing
- Iterates through RGB images in sequence
- Loads corresponding VGGT data automatically
- Handles missing frames gracefully

### 2. Smart Point Filtering
- Projects 3D points to 2D using camera parameters
- Filters points within YOLO segmentation masks
- Maintains world-frame coordinates for scene graph generation

### 3. Performance Optimizations
- Point sampling limits computational cost
- Fast bounding box computation
- GPU memory monitoring

### 4. Full Compatibility
- Works with existing edge prediction algorithms
- Supports all visualization features (point clouds, graphs, rendered frames)
- Maintains timing and performance metrics

## Usage

### Basic Execution
```bash
cd /home/rizo/mipt_ccm/yolo_ssg
python yolo_ssg_e.py
```

### Key Configuration Options
```python
'max_frames': 30,              # Limit frames for testing
'show_pcds': True,             # Visualize 3D point clouds
'vis_graph': True,             # Show scene graphs
'save_rendered_frames': True,  # Render from camera view
'max_points_per_obj': 2000,    # Limit points per object
```

## Benefits

1. **No Depth Required**: Works with RGB-only input using VGGT predictions
2. **Better Accuracy**: VGGT provides more accurate camera parameters than traditional SLAM
3. **World Frame**: Point clouds are directly in world coordinates
4. **Robust Tracking**: YOLO tracking works on original RGB images
5. **Scalable**: Handles variable numbers of frames automatically

## Testing

Run the verification script to ensure VGGT data is properly formatted:
```bash
python test_vggt_simple.py
```

Expected output:
```
✅ VGGT frame data structure is correct:
   Camera intrinsic shape: (3, 3)
   Camera extrinsic shape: (4, 4)
   Camera position: [x, y, z]
   Point cloud shape: (N, 3)
   Colors shape: (N, 3)
   Available frames: 30
```

## Next Steps

1. **Run the Pipeline**: Execute `python yolo_ssg_e.py` to generate scene graphs
2. **Visualize Results**: Enable visualization options to see point clouds and graphs
3. **Optimize Performance**: Adjust `max_points_per_obj` based on your computational resources
4. **Scale Up**: Increase `max_frames` to process full sequences

The integration is complete and ready for scene graph generation using your VGGT predictions!