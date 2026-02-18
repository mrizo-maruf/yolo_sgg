from ultralytics import YOLOE
import cv2
import numpy as np
from pathlib import Path

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg-pf-new.pt")

# Input/output paths
input_video = "/home/rizo/mipt_ccm/yolo_ssg/output_video.mp4"
output_video = "/home/rizo/mipt_ccm/yolo_ssg/output_video_annotated_newm.mp4"

# Open input video to get properties
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"Error: Could not open video {input_video}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Input video: {input_video}")
print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
print(f"Output will be saved to: {output_video}")

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print(f"Error: Could not open video writer")
    exit(1)

# Track with YOLOE and save annotated frames
print("Processing video...")
results = model.track(
    source=input_video,
    conf=0.3,
    iou=0.5,
    show=False,  # Don't show window, we'll save to file
    stream=True,  # Stream results for memory efficiency
    verbose=False,
    persist=True,  # Persist tracker between frames
)

frame_idx = 0
for result in results:
    # Get the annotated frame with boxes, masks, and labels
    annotated_frame = result.plot(
        conf=True,  # Show confidence
        labels=True,  # Show labels
        boxes=True,  # Show bounding boxes
        masks=True,  # Show segmentation masks
        probs=False,
    )
    
    # Write frame to output video
    out.write(annotated_frame)
    
    frame_idx += 1
    if frame_idx % 10 == 0 or frame_idx == total_frames:
        print(f"  Progress: {frame_idx}/{total_frames} frames", end='\r')

print()  # New line after progress
out.release()

# Verify output
output_path = Path(output_video)
if output_path.exists():
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Annotated video saved: {output_video} ({size_mb:.2f} MB)")
else:
    print("Error: Output video was not created")

print("Done!")
