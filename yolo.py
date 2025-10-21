from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf-new.pt")

results = model.track(source="/home/rizo/mipt_ccm/yolo_ssg/UR5-Peg-In-Hole_02/output.mp4", conf=0.3, iou=0.5, show=False)

for result in results:
    result.show()
    
    break