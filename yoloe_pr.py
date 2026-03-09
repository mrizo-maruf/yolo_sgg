"""
YOLOE instance-seg + tracking on a folder of RGB frames (e.g. frame_000001.jpg ...)
- Runs tracking with ByteTrack (default) or BoT-SORT
- Saves per-frame visualizations with colored masks + track IDs
- Also optionally writes an output video

Install deps (example):
  pip install ultralytics opencv-python

Notes:
- Tracking works best if your frames are in correct time order.
- You can switch tracker="bytetrack.yaml" to "botsort.yaml".
"""

from ultralytics import YOLOE
import os
import glob
import cv2
import numpy as np

# -----------------------
# Config
# -----------------------
FRAMES_DIR = "/home/yehia/rizo/THUD_Robot/Real_Scenes/10L/static/Capture_1/RGB"          # folder with images
OUT_DIR = "/home/yehia/rizo/THUD_Robot/Real_Scenes/10L/static/Capture_1/RGB/10LC1_output_pf"      # where annotated frames will be saved
OUT_VIDEO_PATH = "/home/yehia/rizo/THUD_Robot/Real_Scenes/10L/static/Capture_1/RGB/10LC1_output_pf/track_pr.mp4"  # set None to disable video writing

MODEL_WEIGHTS = "yoloe-11l-seg-pf.pt"          # or yoloe-26s/m-seg.pt
# CLASSES = ["white cabinet", "mac and cheese box", "blue mug", "blue can", "mustard bottle", "brown box", "banana"]                 # keep only these classes
# CLASSES = None
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

CONF = 0.25
IOU = 0.5
IMG_SIZE = 960                               # can be 640/960/1280 etc.
DEVICE = 0                                   # 0 for GPU:0, "cpu" for CPU
TRACKER = "botsort.yaml"                  # or "botsort.yaml"
MASK_ALPHA = 0.45                            # mask overlay transparency
LINE_THICKNESS = 2
FONT_SCALE = 0.6

# -----------------------
# Helpers
# -----------------------
def color_for_id(track_id: int) -> tuple[int, int, int]:
    """
    Deterministic BGR color for a track id.
    """
    # Simple hash -> color
    rng = np.random.default_rng(int(track_id) * 99991)
    c = rng.integers(0, 255, size=3, dtype=np.int32)
    return int(c[0]), int(c[1]), int(c[2])  # BGR for OpenCV

def draw_mask_overlay1(img_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float) -> None:
    """
    In-place alpha overlay of a single binary mask on img_bgr.
    mask: HxW boolean or {0,1} float/uint8
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    overlay = np.zeros_like(img_bgr, dtype=np.uint8)
    overlay[mask] = color_bgr

    # alpha blend only on mask area
    img_bgr[mask] = cv2.addWeighted(img_bgr[mask], 1 - alpha, overlay[mask], alpha, 0)

def draw_mask_overlay(img_bgr: np.ndarray, mask: np.ndarray, color_bgr: tuple, alpha: float) -> None:
    """
    In-place alpha overlay of a single binary mask on img_bgr.
    Auto-resizes mask to image size if needed.
    """
    h, w = img_bgr.shape[:2]

    # Ensure mask is 2D
    if mask.ndim == 3:
        # sometimes comes as (1,H,W) or (H,W,1)
        mask = mask.squeeze()

    # Resize mask to frame size if needed
    if mask.shape[:2] != (h, w):
        # mask could be float {0..1} or bool; resize as float then threshold
        mask_f = mask.astype(np.float32)
        mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask_f > 0.5
    else:
        if mask.dtype != np.bool_:
            mask = mask > 0.5

    if not mask.any():
        return

    overlay = np.zeros_like(img_bgr, dtype=np.uint8)
    overlay[mask] = color_bgr
    img_bgr[mask] = cv2.addWeighted(img_bgr[mask], 1 - alpha, overlay[mask], alpha, 0)

def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2

# -----------------------
# Load + export (optional)
# -----------------------
model = YOLOE(MODEL_WEIGHTS)
# model.set_classes(CLASSES)

# If you specifically want ONNX:
# export_model_path = model.export(format="onnx")
# model = YOLOE(export_model_path)

# -----------------------
# Collect frames
# -----------------------
paths = []
for ext in IMG_EXTS:
    paths.extend(glob.glob(os.path.join(FRAMES_DIR, ext)))
paths = sorted(paths)

if not paths:
    raise FileNotFoundError(f"No frames found in: {FRAMES_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)

# Prepare video writer if requested
writer = None
if OUT_VIDEO_PATH:
    first = cv2.imread(paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {paths[0]}")
    h0, w0 = first.shape[:2]
    fps = 30  # change if you know your true FPS
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (w0, h0))

# -----------------------
# Tracking loop (frame-by-frame, persist IDs)
# -----------------------
for i, p in enumerate(paths):
    img_bgr = cv2.imread(p)
    if img_bgr is None:
        print(f"[WARN] Skipping unreadable frame: {p}")
        continue

    # Ultralytics expects RGB for some ops, but we will feed path directly.
    # For per-frame control + visualization, call track on the image array:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = model.track(
        source=img_rgb,
        persist=True,           # keep track IDs across frames
        tracker=TRACKER,
        conf=CONF,
        iou=IOU,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False
    )

    r = results[0]
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # If no detections this frame, just save raw frame
    if r.boxes is None or len(r.boxes) == 0:
        out_path = os.path.join(OUT_DIR, f"{i:06d}.jpg")
        cv2.imwrite(out_path, vis)
        if writer:
            writer.write(vis)
        continue

    # Track IDs (can be None if tracker didn't assign for some reason)
    ids = r.boxes.id
    ids = ids.int().cpu().numpy() if ids is not None else np.array([-1] * len(r.boxes))

    # Boxes + classes
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.int().cpu().numpy()

    # Masks (if available)
    # r.masks.data: [N, H, W] float {0..1}
    masks = None
    if r.masks is not None and r.masks.data is not None:
        masks = r.masks.data.cpu().numpy()  # NxHxW

    for j in range(len(xyxy)):
        tid = int(ids[j])
        x1, y1, x2, y2 = xyxy[j]
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)

        color = color_for_id(tid if tid >= 0 else (j + 1))

        # Mask overlay (if segmentation mask exists)
        if masks is not None and j < masks.shape[0]:
            mask = masks[j] > 0.5
            draw_mask_overlay(vis, mask, color, MASK_ALPHA)

        # Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        # Label: track id + class name
        class_name = model.names[int(cls[j])] if hasattr(model, "names") else str(int(cls[j]))
        label = f"ID {tid} | {class_name}" if tid >= 0 else f"{class_name}"

        # Text background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(vis, (x1, y_text - th - 6), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(
            vis, label, (x1 + 3, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA
        )

    out_path = os.path.join(OUT_DIR, f"{i:06d}.jpg")
    cv2.imwrite(out_path, vis)
    if writer:
        writer.write(vis)

    if (i + 1) % 50 == 0:
        print(f"Processed {i+1}/{len(paths)} frames...")

if writer:
    writer.release()

print("Done.")
print(f"Annotated frames saved to: {OUT_DIR}")
if OUT_VIDEO_PATH:
    print(f"Video saved to: {OUT_VIDEO_PATH}")