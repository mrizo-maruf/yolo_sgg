import os
import glob
from typing import Any

import cv2
import numpy as np
import torch


def process_depth_model(cfg):
    """
    Run DepthAnything3 offline and save depth PNGs.

    Expects cfg fields:
      - rgb_dir: path to RGB images
      - depth_dir: path to original depth dir (used for output location)
      - depth_model: model name or None
      - dav3_offline_png_depth_scale (optional): meters per PNG unit
      - min_depth / max_depth (optional)
    """
    if cfg.depth_model is None:
        return cfg

    model = _load_model(cfg.depth_model, cfg)
    rgb_files = _list_rgb_files(cfg.rgb_dir)
    if not rgb_files:
        raise ValueError(f"No images found in {cfg.rgb_dir}")

    depth_parent_dir = os.path.dirname(cfg.depth_dir)
    out_dir = os.path.join(depth_parent_dir, "dav3_depth")
    os.makedirs(out_dir, exist_ok=True)

    scale = _resolve_png_scale(cfg)
    min_depth = float(getattr(cfg, "min_depth", 0.01))
    max_depth = float(getattr(cfg, "max_depth", 0.0))

    prediction = _predict_sequence(model, rgb_files)
    _save_prediction(
        prediction,
        rgb_files,
        out_dir,
        scale,
        min_depth,
        max_depth,
    )
    _save_poses(prediction, depth_parent_dir)

    _write_meta(out_dir, scale)
    cfg.depth_dir = out_dir
    return cfg


def _load_model(model_name: str, cfg):
    try:
        from depth_anything_3.api import DepthAnything3  # type: ignore
    except Exception as exc:
        raise RuntimeError("DepthAnything3 is not installed.") from exc

    if hasattr(DepthAnything3, "from_pretrained"):
        model = DepthAnything3.from_pretrained(model_name)
    else:
        model = DepthAnything3(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def _predict_depth(model, rgb: np.ndarray) -> np.ndarray | None:
    prediction: Any = model.inference([rgb], export_format="mini_npz")
    if prediction is None:
        return None

    depth = getattr(prediction, "depth", None)
    if depth is None:
        return None

    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[0]
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
    return depth.astype(np.float32)


def _predict_sequence(model, rgb_files: list[str]) -> Any:
    return model.inference(
        rgb_files,
        export_format="mini_npz",
        use_ray_pose=True,
    )


def _save_prediction(
    prediction: Any,
    rgb_files: list[str],
    out_dir: str,
    scale: float,
    min_depth: float,
    max_depth: float,
) -> None:
    depth = getattr(prediction, "depth", None)
    if depth is None:
        raise ValueError("Prediction has no depth output")

    depth = np.asarray(depth)
    if depth.ndim == 2:
        depth = depth[None, ...]
    for i in range(depth.shape[0]):
        rgb = cv2.imread(rgb_files[i])
        if rgb is None:
            continue
        h, w = rgb.shape[:2]
        dm = depth[i]
        if dm.shape[:2] != (h, w):
            dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)

        dm = dm.astype(np.float32)
        dm[~np.isfinite(dm)] = 0.0
        dm[dm < min_depth] = 0.0
        if max_depth > 0.0:
            dm[dm > max_depth] = 0.0

        depth_u16 = np.clip(dm / scale, 0.0, 65535.0).astype(np.uint16)
        out_path = os.path.join(out_dir, f"depth{i:06d}.png")
        cv2.imwrite(out_path, depth_u16)


def _save_poses(prediction: Any, depth_parent_dir: str) -> None:
    exts = getattr(prediction, "extrinsics", None)
    if exts is None:
        raise RuntimeError(
            "DepthAnything3 did not return camera poses. "
            "Use a DA3 model with camera decoder or enable ray pose output."
        )
    exts = np.asarray(exts)
    if exts.ndim != 3:
        return
    if exts.shape[1:] == (3, 4):
        exts = _to_homogeneous(exts)
    elif exts.shape[1:] != (4, 4):
        return
    traj_path = os.path.join(depth_parent_dir, "dav3_camera_poses.txt")
    with open(traj_path, "w", encoding="utf-8") as f:
        for i in range(exts.shape[0]):
            pose = exts[i].reshape(-1)
            f.write(" ".join(map(str, pose.tolist())) + "\n")


def _to_homogeneous(exts_3x4: np.ndarray) -> np.ndarray:
    n = exts_3x4.shape[0]
    exts = np.zeros((n, 4, 4), dtype=exts_3x4.dtype)
    exts[:, :3, :4] = exts_3x4
    exts[:, 3, 3] = 1.0
    return exts


def _list_rgb_files(rgb_dir: str) -> list[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(rgb_dir, ext)))
    files.sort()
    return files


def _resolve_png_scale(cfg) -> float:
    scale = getattr(cfg, "dav3_offline_png_depth_scale", None)
    if scale is None:
        scale = getattr(cfg, "png_depth_scale", None)
    if scale is None:
        return 0.001
    return float(scale)


def _write_meta(out_dir: str, scale: float) -> None:
    path = os.path.join(out_dir, "dav3_depth_meta.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"png_depth_scale: {scale}\n")
