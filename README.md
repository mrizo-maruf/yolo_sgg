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
[x] - multi-obj rel visualization
[ ] - add time/GPU usage
[ ] - camera relations
[ ] - visualization
[ ] - support of new yolo-seg with obj names
[ ] - graph update
[ ] - try with prompt model
[ ] - VL-SAT edge predictor support