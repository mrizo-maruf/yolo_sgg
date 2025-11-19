# YOLO-Based Scene Graph Generation (SSG)

Automated 3D scene understanding and relationship detection using YOLO segmentation and geometric reasoning.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 🚀 Quick Links

- **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[Full Setup Instructions](SETUP.md)** - Detailed installation guide
- **[Requirements](requirements.txt)** - Python dependencies
- **[Conda Environment](environment.yml)** - Reproducible conda setup

---

## 📋 Overview

![pipeline](/assets/image.png)


This project generates **3D scene graphs** from RGB-D or monocular video sequences by:

1. **Object Detection & Segmentation** - Using YOLO for instance segmentation
2. **Preprocessing segmentation masks** - Using OpenCV erosion operation to adjust masks
2. **3D Reconstruction** - Converting 2D masks to 3D point clouds
3. **Relationship Detection** - Computing spatial relationships (support, proximity, etc.) using SceneVerse edge prediction algorithm
4. **Graph Construction** - Building multi-frame persistent scene graphs
---

## 📦 Installation

### Quick Setup

```bash
# Create conda environment
conda create -n yolo_ssg python=3.10 -y
conda activate yolo_ssg

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install dependencies
pip install -r requirements.txt
```

**👉 See [SETUP.md](SETUP.md) for detailed instructions and [QUICKSTART.md](QUICKSTART.md) for rapid setup.**

---

<!-- ## 🏃 How to Run

### VGGT-Based Processing (Recommended)

Process RGB images with VGGT-predicted camera parameters:

```bash
python yolo_ssg_e.py
``` -->

### Traditional RGBD Processing

Process RGB-D sequences with known camera poses:

```bash
python yolo_ssg.py
```

### Data Requirements

1. **Download YOLO model**
2. **Download sample data**: *UR5-Peg-In-Hole_02_straight* folder from same drive
  * or any RGB-D sequence with camera trajectory
    ```
    depth/
      - frame1.png
      - ...
    rgb/
      - frame2.jpg
      - ...
    traj.txt
      - 4x4 matric of camera postion
    ```
3. **Configure paths** inside `yolo_ssg.py`

### Configuration Options

Edit the config in the script's `__main__` section:

```python
cfg = OmegaConf.create({
    'rgb_dir': "/path/to/rgb",
    'yolo_model': 'yoloe-11l-seg-pf-old.pt',
    'conf': 0.3,                      # Detection confidence
    'iou': 0.5,                       # IOU threshold
    'max_points_per_obj': 2000,       # Points per object
    'show_pcds': True,                # 3D visualization
    'vis_graph': True,                # Graph visualization
    'fast_mask': False,               # Show mask processing
})
```

---

## 📊 Output

The system generates:

1. **Scene Graphs** - NetworkX MultiDiGraph with nodes (objects) and edges (relationships)
2. **3D Point Clouds** - Open3D format with colors and bounding boxes
3. **Visualizations** - Interactive 3D views and 2D graph plots
4. **Rendered Frames** - Camera-perspective renders with annotations (optional)

### Relationship Types

- **Egocentric (Camera-Relative)**: Clock-direction proximity, distance-based
- **Allocentric (Object-Relative)**: Support, hanging, embedded, alignment

--- 
### Latency

#### Latency with big erosion using `opencv2`
* small objects might be erased by erosion
* outliers in pcds
* `kernel_size: 19`

For complex scene:
```
Latency Averages (ms):
  Preprocessing:    28.48 ± 10.72
  Create 3D:        26.59 ± 8.07
  Edge Prediction:  7.17 ± 4.45
  YOLO:             29.34 ± 4.98
  Merge:           1.28 ± 0.26
  Total per frame:  92.85

GPU Memory Usage Averages (MB):
  After YOLO:       179.5 ± 5.9
  After Edges:      169.8 ± 3.3

Total frames processed: 30
```

For simple scene:
```
Latency Averages (ms):
  Preprocessing:    16.71 ± 4.21
  Create 3D:        18.32 ± 5.77
  Edge Prediction:  1.82 ± 0.40
  YOLO:             28.20 ± 8.47
  Merge:           0.33 ± 0.13
  Total per frame:  65.38

GPU Memory Usage Averages (MB):
  After YOLO:       168.9 ± 1.0
  After Edges:      164.4 ± 0.4

Total frames processed: 40
```

#### Latency with `open3d` statistical outlier removal
For complex scene:
```Latency Averages (ms):
  Preprocessing:    13.37 ± 6.04
  Create 3D:        67.03 ± 22.42
  Edge Prediction:  10.35 ± 5.85
  YOLO:             29.40 ± 5.09
  Merge:           1.57 ± 0.33
  Total per frame:  121.72

GPU Memory Usage Averages (MB):
  After YOLO:       179.5 ± 5.9
  After Edges:      169.8 ± 3.3

Total frames processed: 30
```

For simple scene:
```
Latency Averages (ms):
  Preprocessing:    5.89 ± 0.88
  Create 3D:        29.74 ± 4.44
  Edge Prediction:  2.39 ± 0.40
  YOLO:             28.61 ± 4.76
  Merge:           0.34 ± 0.05
  Total per frame:  66.97

GPU Memory Usage Averages (MB):
  After YOLO:       168.9 ± 1.0
  After Edges:      164.4 ± 0.4

Total frames processed: 40
```

### Scene Graph Merging Algorithm

The system builds a **persistent scene graph** across multiple frames by merging frame-by-frame observations. This allows tracking objects and their relationships over time as the camera moves through the scene.

#### How It Works

**1. Node Matching**
- First, the algorithm matches objects between the current frame and the persistent graph using **tracking IDs** from YOLO (most reliable method)
- For objects without reliable IDs, it falls back to **spatial matching** using 3D position and bounding box overlap
- Objects that can't be matched are added as new nodes

**2. Node Updates**
- Matched nodes get updated with the latest 3D position, point cloud, and bounding box information
- This keeps the graph synchronized with the most recent observations

**3. Edge Handling: Egocentric vs Allocentric**

The algorithm treats two types of spatial relationships differently:

- **Egocentric relationships** (camera-dependent): proximity, "to the left/right", directional relations
  - These change as the camera moves
  - Old egocentric edges are removed and replaced with current frame observations
  
- **Allocentric relationships** (camera-independent): support, embedded, hanging, aligned
  - These are physical relationships that don't depend on camera viewpoint
  - Once detected, they persist across frames unless contradicted by new evidence
  - Conflicting allocentric relationships (e.g., object can't be both ON and INSIDE another) are resolved by keeping the newest observation

**4. Conflict Resolution**
- If a new observation contradicts an existing allocentric relationship, the old edge is removed
- Special handling for support relationships: removing a support edge also removes the corresponding opposite-support edge

### TO-DO
- [x] multi-obj rel visualization
- [x] add time/GPU usage
- [x] graph update
- [x] 3D obj generation faster
- [ ] graph update logic
- [ ] video visualization
- [ ] camera relations
- [ ] improve SV edge predictor (faster)
- [ ] support of new yolo-seg with obj names
- [ ] try with prompt model
- [ ] VL-SAT edge predictor support
- [ ] visualization in 3d
- [ ] add `requirements.txt`
---

## 🗂️ Project Structure

```
yolo_ssg/
├── yolo_ssg.py              # Main: RGBD processing
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── SETUP.md                 # Setup instructions
├── QUICKSTART.md           # Quick start guide
├── README.md               # This file
│
├── YOLOE/
│   └── utils.py            # YOLO utilities
│
└── ssg/
    ├── ssg_main.py         # Scene graph generation
    ├── ssg_utils.py        # SSG utilities
    ├── relationships/      # Relationship detectors
    │   ├── support.py
    │   ├── proximity.py
    │   ├── hanging.py
    │   └── multi_objs.py
    └── ssg_data/           # Data structures
```
