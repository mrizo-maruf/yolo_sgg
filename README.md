### how to run
1. download yoloe-11l-seg-pf-new.pt from https://drive.google.com/drive/folders/1noXhCDYF7yvHvDLeYBeSUMVQD-88OtBQ?usp=sharing
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
- [ ] 3D obj generation faster
- [ ] video visualization
- [ ] camera relations
- [ ] improve SV edge predictor (faster)
- [ ] support of new yolo-seg with obj names
- [ ] try with prompt model
- [ ] VL-SAT edge predictor support
- [ ] visualization in 3d
- [ ] add `requirements.txt`

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