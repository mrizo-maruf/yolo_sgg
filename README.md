### how to run
1. download yoloe-11l-seg-pf-old.pt from https://drive.google.com/drive/folders/1noXhCDYF7yvHvDLeYBeSUMVQD-88OtBQ?usp=sharing
2. Download *UR5-Peg-In-Hole_02_straight* folder also from same drive
3. pass corresponding paths inside `yolo_ssg.py` conf
2. open3d, networkx, yolo ... libs install
3. run `python3 yolo_ssg.py`
4. to vis pcds, ``` 'show_pcds': True``` inside `yolo_ssg.py` conf
5. to vis masks orig/cleaned, ``` 'fast_mask': False,``` inside `yolo_ssg.py` conf
6. 


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
* update based on std div, of rgbs, kdl
* generate in intervals
* every time stemp update
* track only moving objects, cam can move/stay still
