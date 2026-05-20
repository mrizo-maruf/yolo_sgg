#!/usr/bin/env python3
"""Export chunked dav3_online predictions into dav3_offline layout.

This is intentionally a thin wrapper around DAv3OnlineDepthProvider: the saved
depth/pose files match the online provider behavior for a given
``dav3_window_size`` / ``dav3_overlap`` config.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.benchmark_utils import build_loader_kwargs
from data_loaders import get_loader
from depth_providers.factory import build_depth_provider


def _scene_list(dataset: str, scene_path: Path, multi: bool) -> list[Path]:
    if not multi:
        return [scene_path]
    loader_cls = get_loader(dataset)
    return [Path(p) for p in loader_cls.discover_scenes(str(scene_path))]


def _save_depth_png(depth_m: np.ndarray, out_path: Path, png_scale: float) -> None:
    dm = np.asarray(depth_m, dtype=np.float32)
    dm[~np.isfinite(dm)] = 0.0
    depth_u16 = np.clip(dm / float(png_scale), 0.0, 65535.0).astype(np.uint16)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), depth_u16):
        raise RuntimeError(f"failed to write depth PNG: {out_path}")


def _write_meta(out_dir: Path, png_scale: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dav3_depth_meta.txt").write_text(
        f"png_depth_scale: {float(png_scale)}\n",
        encoding="utf-8",
    )


def _write_pose_file(path: Path, poses: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for pose in poses:
            p = np.asarray(pose, dtype=np.float32)
            if p.shape != (4, 4):
                p = np.eye(4, dtype=np.float32)
            f.write(" ".join(map(str, p.reshape(-1).tolist())) + "\n")


def export_scene(scene_dir: Path, dataset: str, cfg, *, skip_existing: bool) -> None:
    chunk = int(cfg.get("dav3_window_size", cfg.get("pi3_window_size", 5)))
    overlap = int(cfg.get("dav3_overlap", cfg.get("pi3_overlap", 3)))
    suffix = f"{chunk}_{overlap}"
    depth_dir = scene_dir / f"dav3_depth_{suffix}"
    pose_path = scene_dir / f"dav3_traj_{suffix}.txt"
    png_scale = float(cfg.get("dav3_offline_png_depth_scale", 0.001) or 0.001)

    loader_cls = get_loader(dataset)
    provider = build_depth_provider("dav3_online", dataset, str(scene_dir), cfg)
    loader = loader_cls(
        str(scene_dir),
        depth_provider=provider,
        **build_loader_kwargs(dataset, cfg),
    )
    n_frames = loader.get_num_frames()

    existing_depths = list(depth_dir.glob("depth*.png"))
    if skip_existing and pose_path.exists() and len(existing_depths) >= n_frames:
        print(f"[skip] {scene_dir.name}: {depth_dir.name} already has {len(existing_depths)} frames")
        provider.close()
        return

    print(f"[export] {scene_dir.name}: dav3_online ({chunk}, {overlap}) -> {depth_dir.name}")
    depth_dir.mkdir(parents=True, exist_ok=True)

    frame_keys: list[int] = []
    for frame_idx in range(n_frames):
        loader.get_rgb(frame_idx)
        frame_keys.append(int(loader.provider_frame_key(frame_idx)))

    if hasattr(provider, "drain"):
        provider.drain()

    poses: list[np.ndarray] = []
    for frame_idx, frame_key in enumerate(frame_keys):
        depth = provider.get_depth(frame_key)
        if depth is None:
            raise RuntimeError(f"{scene_dir.name}: missing DAv3 depth for frame key {frame_key}")
        pose = provider.get_pose(frame_key)
        if pose is None:
            raise RuntimeError(f"{scene_dir.name}: missing DAv3 pose for frame key {frame_key}")

        out_png = depth_dir / f"depth{frame_key:06d}.png"
        _save_depth_png(depth, out_png, png_scale)
        poses.append(np.asarray(pose, dtype=np.float32))

    _write_pose_file(pose_path, poses)
    _write_meta(depth_dir, png_scale)
    provider.close()
    print(f"[done] {scene_dir.name}: wrote {len(frame_keys)} depths and {pose_path.name}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["isaacsim", "scanetpp", "thud_synthetic", "coda"])
    p.add_argument("--scene_path", required=True, help="Single scene dir, or dataset root with --multi.")
    p.add_argument("--multi", action="store_true", help="Export all scenes under scene_path.")
    p.add_argument("--skip_existing", action="store_true", help="Skip scenes whose output already exists.")
    args = p.parse_args()

    cfg_dir = PROJECT_ROOT / "configs"
    cfg = OmegaConf.load(cfg_dir / "core_tracking.yaml")
    ds_yaml = cfg_dir / f"{args.dataset}.yaml"
    if ds_yaml.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(ds_yaml))

    scene_path = Path(args.scene_path).expanduser().resolve()
    scenes = _scene_list(args.dataset, scene_path, args.multi)
    if not scenes:
        print(f"No scenes found under {scene_path}", file=sys.stderr)
        return 1

    for scene in scenes:
        export_scene(scene, args.dataset, cfg, skip_existing=bool(args.skip_existing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
