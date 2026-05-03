"""
Streaming Pi3X depth prediction with live Open3D visualization.

Accumulates RGB point clouds, camera trajectory, frustums, and a world frame
in an Open3D window. Old point clouds are evicted when ``--max_pcds`` is reached.

Usage:
    # From a webcam:
    python run_stream.py --source camera --camera_id 0

    # From a directory of RGB images (simulates a stream):
    python run_stream.py --source dir --rgb_dir /path/to/rgb

    # From a video file:
    python run_stream.py --source video --video_path /path/to/video.mp4

    # Tune visualization:
    python run_stream.py --source dir --rgb_dir /path --max_pcds 40 --subsample 4
"""

import argparse
import time
import math
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import open3d as o3d
import torch
from torchvision import transforms

from pi3.models.pi3x import Pi3X
from pi3.pipe.pi3x_vo_stream import Pi3XVOStream


REPO_ROOT = Path(__file__).resolve().parent
LOCAL_PI3X_CKPT = REPO_ROOT / "chkp" / "Pi3X" / "model.safetensors"

# ──────────────────────────────────────────────────────────────
#  Image preprocessing (matches pi3/utils/basic.py logic)
# ──────────────────────────────────────────────────────────────

def compute_target_size(w_orig, h_orig, pixel_limit=255000):
    scale = math.sqrt(pixel_limit / (w_orig * h_orig)) if w_orig * h_orig > 0 else 1
    w_t, h_t = w_orig * scale, h_orig * scale
    k, m = round(w_t / 14), round(h_t / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > w_t / h_t:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def preprocess_frame(bgr_frame, target_w, target_h, to_tensor):
    """BGR numpy (H,W,3) uint8 -> (3, target_h, target_w) float32 tensor in [0,1]."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return to_tensor(rgb_resized)


def build_intrinsics(fx, fy, cx, cy, orig_w, orig_h, target_w, target_h, device):
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    K = torch.tensor([
        [fx * scale_x, 0.0, cx * scale_x],
        [0.0, fy * scale_y, cy * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32, device=device)
    return K.view(1, 1, 3, 3)


# ──────────────────────────────────────────────────────────────
#  Frame sources
# ──────────────────────────────────────────────────────────────

def camera_source(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    print(f"[source] Camera {camera_id} opened")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def directory_source(rgb_dir):
    exts = {'.png', '.jpg', '.jpeg'}
    files = sorted(p for p in Path(rgb_dir).iterdir() if p.suffix.lower() in exts)
    if not files:
        raise ValueError(f"No images found in {rgb_dir}")
    print(f"[source] Directory: {rgb_dir} ({len(files)} images)")
    for f in files:
        frame = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if frame is not None:
            yield frame


def video_source(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    print(f"[source] Video: {video_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────
#  Open3D geometry builders
# ──────────────────────────────────────────────────────────────

def make_frustum(pose, fx, fy, cx, cy, w, h, scale=0.15, color=(0.0, 1.0, 0.0)):
    """Camera frustum wireframe as an Open3D LineSet."""
    corners_uv = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    corners_cam = np.zeros((4, 3))
    for i, (u, v) in enumerate(corners_uv):
        corners_cam[i] = [(u - cx) / fx * scale, (v - cy) / fy * scale, scale]

    pts_cam = np.vstack([[[0, 0, 0]], corners_cam])
    pts_world = (pose[:3, :3] @ pts_cam.T).T + pose[:3, 3]

    lines = [[0, 1], [0, 2], [0, 3], [0, 4],
             [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts_world),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(lines))
    return ls


def make_trajectory_lineset(positions, color=(1.0, 0.5, 0.0)):
    """Line connecting consecutive camera origins."""
    n = len(positions)
    if n < 2:
        return None
    pts = np.array(positions)
    lines = [[i, i + 1] for i in range(n - 1)]
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines))
    ls.colors = o3d.utility.Vector3dVector([list(color)] * len(lines))
    return ls


def build_frame_pcd(points_np, rgb_np, fx, fy, cx, cy, subsample=4, conf_np=None, conf_thre=0.0):
    """
    Back-project one frame's global 3D points into an Open3D PointCloud with RGB colors.

    Args:
        points_np: (H, W, 3) global 3D points
        rgb_np:    (H, W, 3) uint8 RGB image (at inference resolution)
        subsample: pixel stride for subsampling
        conf_np:   (H, W) optional confidence map
        conf_thre: discard points below this confidence
    """
    H, W, _ = points_np.shape
    vs = np.arange(0, H, subsample)
    us = np.arange(0, W, subsample)
    u, v = np.meshgrid(us, vs)
    u, v = u.ravel(), v.ravel()

    pts = points_np[v, u]  # (N, 3)
    z = pts[:, 2] if pts.shape[1] == 3 else np.zeros(len(pts))

    valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-6)
    if conf_np is not None:
        c = conf_np[v, u]
        valid &= c > conf_thre

    pts = pts[valid]
    colors = rgb_np[v[valid], u[valid]].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    if len(pts) > 0:
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ──────────────────────────────────────────────────────────────
#  Live Open3D Visualizer
# ──────────────────────────────────────────────────────────────

class StreamVisualizer:
    """
    Non-blocking Open3D visualizer that incrementally adds point clouds,
    frustums, and trajectory, evicting the oldest clouds when max_pcds
    is exceeded.
    """

    def __init__(self, max_pcds=50, frustum_scale=0.15, subsample=4):
        self.max_pcds = max_pcds
        self.frustum_scale = frustum_scale
        self.subsample = subsample

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Pi3X Stream", width=1400, height=900)

        # Render options
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0

        # World frame (always present)
        self.world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        self.vis.add_geometry(self.world_frame)

        # Ring buffer of (pcd, frustum) — evict oldest when full
        self._pcd_queue = deque()       # deque of o3d.geometry.PointCloud
        self._frustum_queue = deque()   # deque of o3d.geometry.LineSet

        # Trajectory line — rebuilt each update
        self._cam_positions = []
        self._traj_lineset = None

        self._first_update = True

    def add_chunk_results(self, results, rgb_frames, fx, fy, cx, cy, infer_w, infer_h):
        """
        Add new frames from a streaming chunk result.

        Args:
            results: dict from Pi3XVOStream with 'points', 'poses', 'conf', 'depth'
            rgb_frames: list of BGR numpy arrays (original resolution) for these frames
            fx, fy, cx, cy: intrinsics in *inference* resolution
            infer_w, infer_h: inference image size (for frustum corners)
        """
        points = results['points'].cpu().numpy()  # (n, H, W, 3)
        poses = results['poses'].cpu().numpy()     # (n, 4, 4)
        conf = results['conf'].cpu().numpy()       # (n, H, W)
        n = points.shape[0]
        pH, pW = points.shape[1], points.shape[2]

        for i in range(n):
            # Build RGB at inference resolution for coloring
            rgb_resized = cv2.resize(
                cv2.cvtColor(rgb_frames[i], cv2.COLOR_BGR2RGB),
                (pW, pH), interpolation=cv2.INTER_LINEAR
            )

            pcd = build_frame_pcd(
                points[i], rgb_resized,
                fx, fy, cx, cy,
                subsample=self.subsample,
                conf_np=conf[i], conf_thre=0.01,
            )

            frustum = make_frustum(
                poses[i], fx, fy, cx, cy, infer_w, infer_h,
                scale=self.frustum_scale, color=(0.0, 1.0, 0.0),
            )

            # Add to viewer
            self.vis.add_geometry(pcd)
            self.vis.add_geometry(frustum)
            self._pcd_queue.append(pcd)
            self._frustum_queue.append(frustum)

            # Camera position for trajectory
            self._cam_positions.append(poses[i][:3, 3].copy())

            # Evict oldest if over budget
            while len(self._pcd_queue) > self.max_pcds:
                old_pcd = self._pcd_queue.popleft()
                old_frustum = self._frustum_queue.popleft()
                self.vis.remove_geometry(old_pcd, reset_bounding_box=False)
                self.vis.remove_geometry(old_frustum, reset_bounding_box=False)

        # Update trajectory line
        if self._traj_lineset is not None:
            self.vis.remove_geometry(self._traj_lineset, reset_bounding_box=False)
            self._traj_lineset = None

        traj = make_trajectory_lineset(self._cam_positions, color=(1.0, 0.5, 0.0))
        if traj is not None:
            self._traj_lineset = traj
            self.vis.add_geometry(traj)

        # Reset viewpoint on first update so the scene is visible
        if self._first_update:
            self.vis.reset_view_point(True)
            self._first_update = False

    def poll(self):
        """Pump the Open3D event loop. Returns False if the window was closed."""
        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def run_event_loop(self):
        """Block until the user closes the window (call after stream ends)."""
        print("[vis] Stream finished. Close the Open3D window to exit.")
        self.vis.run()

    def destroy(self):
        self.vis.destroy_window()


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Streaming Pi3X depth + live 3D visualization")
    parser.add_argument("--source", choices=["camera", "dir", "video"], default="dir")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--rgb_dir", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Optional: save depth PNGs and trajectory here")
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--fx", type=float, default=800.0)
    parser.add_argument("--fy", type=float, default=800.0)
    parser.add_argument("--cx", type=float, default=640.0)
    parser.add_argument("--cy", type=float, default=360.0)
    # Visualization
    parser.add_argument("--max_pcds", type=int, default=50,
                        help="Max point clouds kept in the viewer (oldest evicted)")
    parser.add_argument("--subsample", type=int, default=4,
                        help="Pixel stride for point cloud subsampling")
    parser.add_argument("--frustum_scale", type=float, default=0.15,
                        help="Camera frustum size in meters")
    parser.add_argument("--no_vis", action="store_true",
                        help="Disable Open3D visualization (save only)")
    args = parser.parse_args()

    # Optional output directory
    out = None
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'trajectory.txt').write_text('')

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = (torch.bfloat16
             if device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    # Load model
    print("Loading Pi3X model...")
    if LOCAL_PI3X_CKPT.is_file():
        model = Pi3X().to(device).eval()
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(LOCAL_PI3X_CKPT)), strict=False)
    else:
        model = Pi3X.from_pretrained('yyfz233/Pi3X').to(device).eval()
    print("Model loaded.")

    # Frame source
    if args.source == "camera":
        source_iter = camera_source(args.camera_id)
    elif args.source == "dir":
        if args.rgb_dir is None:
            raise ValueError("--rgb_dir required for dir source")
        source_iter = directory_source(args.rgb_dir)
    elif args.source == "video":
        if args.video_path is None:
            raise ValueError("--video_path required for video source")
        source_iter = video_source(args.video_path)

    # Peek first frame for resolution
    source_iter = iter(source_iter)
    first_frame = next(source_iter)
    orig_h, orig_w = first_frame.shape[:2]
    target_w, target_h = compute_target_size(orig_w, orig_h)
    print(f"Original: {orig_w}x{orig_h} -> Inference: {target_w}x{target_h}")

    # Intrinsics (scaled to inference resolution)
    intrinsics_tensor = build_intrinsics(
        args.fx, args.fy, args.cx, args.cy,
        orig_w, orig_h, target_w, target_h, device
    )
    K = intrinsics_tensor[0, 0].cpu().numpy()
    fx_s, fy_s, cx_s, cy_s = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"Scaled intrinsics:\n{K}")

    # Streaming pipeline
    stream = Pi3XVOStream(
        model=model,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        conf_thre=0.05,
        intrinsics=intrinsics_tensor,
        dtype=dtype,
    )

    # Visualizer
    vis = None
    if not args.no_vis:
        vis = StreamVisualizer(
            max_pcds=args.max_pcds,
            frustum_scale=args.frustum_scale,
            subsample=args.subsample,
        )

    to_tensor = transforms.ToTensor()
    frame_offset = 0
    total_frames = 0
    t_start = time.perf_counter()

    # We need to keep the raw BGR frames for the current chunk so we can
    # color the point clouds. The stream pipeline buffers `overlap` frames
    # internally; we mirror that here for the RGB side.
    rgb_buffer = []

    def process_results(results, n_new_from_start):
        """Handle a chunk result: visualize and optionally save."""
        nonlocal frame_offset
        n_new = results['points'].shape[0]

        # The newest `n_new` RGB frames in rgb_buffer correspond to the result.
        # But the result may skip `overlap` frames that were already emitted.
        # n_new_from_start tells us where in rgb_buffer the new frames start.
        chunk_rgb = rgb_buffer[n_new_from_start: n_new_from_start + n_new]

        if vis is not None:
            vis.add_chunk_results(
                results, chunk_rgb,
                fx_s, fy_s, cx_s, cy_s,
                target_w, target_h,
            )

        # Optional save
        if out is not None:
            depth = results['depth'].cpu().numpy()
            poses = results['poses'].cpu().numpy()
            for i in range(n_new):
                idx = frame_offset + i
                d = depth[i]
                valid = np.logical_and(d > 0, np.isfinite(d))
                d_u16 = np.zeros_like(d, dtype=np.uint16)
                if valid.sum() > 0:
                    q = np.clip(np.round(d[valid] / 0.001), 1, 65535).astype(np.uint16)
                    d_u16[valid] = q
                cv2.imwrite(str(out / f'depth{idx:06d}.png'), d_u16)

            with open(out / 'trajectory.txt', 'a') as f:
                for i in range(n_new):
                    f.write(' '.join(map(str, poses[i].flatten())) + '\n')

        print(f"  Emitted frames {frame_offset}..{frame_offset + n_new - 1}")
        frame_offset += n_new

    # ── Feed frames ──────────────────────────────────────────

    def feed(bgr_frame):
        nonlocal total_frames
        rgb_buffer.append(bgr_frame)
        frame_tensor = preprocess_frame(bgr_frame, target_w, target_h, to_tensor).to(device)
        results = stream.push_frame(frame_tensor)
        total_frames += 1

        if results is not None:
            # For the first chunk all frames are new (n_new_from_start=0).
            # For subsequent chunks the first `overlap` frames in the buffer
            # were already emitted, so new output starts at index `overlap`.
            is_first = (frame_offset == 0)
            n_new_start = 0 if is_first else stream.overlap
            process_results(results, n_new_start)

            # Trim rgb_buffer to keep only the overlap tail
            del rgb_buffer[:-stream.overlap]

        if vis is not None:
            vis.poll()

    # First frame
    feed(first_frame)

    # Remaining frames
    for bgr_frame in source_iter:
        feed(bgr_frame)

    # Flush
    results = stream.flush()
    if results is not None:
        is_first = (frame_offset == 0)
        n_new_start = 0 if is_first else stream.overlap
        process_results(results, n_new_start)
        del rgb_buffer[:-stream.overlap]

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s  ({total_frames/elapsed:.1f} fps input)")

    # Keep the window open until the user closes it
    if vis is not None:
        vis.run_event_loop()
        vis.destroy()


if __name__ == "__main__":
    main()
