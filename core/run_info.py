"""Pretty-printed run-configuration banner — shared by ``new_run.py`` and
``benchmark/benchmark.py`` so the two entry points show identical info.

Call once per scene, right after the loader / depth provider are built.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def print_run_banner(
    *,
    title: str,
    dataset_name: str,
    scene_label: str,
    scene_path: str,
    n_frames: int,
    intrinsics,
    dp_type: str,
    cfg,
    classes: Optional[Iterable[str]] = None,
    extras: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Print a labeled config block summarising the active run.

    Parameters
    ----------
    title
        e.g. ``"RUN"`` for new_run.py, ``"BENCHMARK"`` for benchmark.py.
    classes
        Class list passed to YOLO when running open-vocabulary, or
        ``None`` for closed-set.
    extras
        Optional ``{section_title: [line, ...]}`` map appended at the
        end. Caller-specific info (scene graph for new_run.py, match
        mode for benchmark.py).
    """
    print()
    print("=" * 60)
    print(f"  {title} — {scene_label}  (dataset: {dataset_name})")
    print("=" * 60)
    print(f"Scene path:   {scene_path}")
    print(f"Frames:       {n_frames}")
    print(
        f"Intrinsics:   fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
        f"cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}  "
        f"image={intrinsics.width}x{intrinsics.height}"
    )

    _print_yolo(cfg, classes)
    _print_depth(dp_type, cfg, scene_path)
    _print_tracking(cfg)

    if extras:
        for section, lines in extras.items():
            print()
            print(f"{section}:")
            for line in lines:
                print(f"  {line}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _resolve_path(scene_path: str, cfg_value: Any) -> Optional[Path]:
    if cfg_value is None:
        return None
    p = Path(str(cfg_value))
    return p if p.is_absolute() else Path(scene_path) / p


def _path_status(p: Optional[Path], required: bool) -> str:
    if p is None:
        return "(none)"
    if p.exists():
        return str(p)
    return f"{p} [MISSING — required!]" if required else f"{p} [MISSING]"


def _print_yolo(cfg, classes: Optional[Iterable[str]]) -> None:
    print()
    print("YOLO:")
    print(f"  model:             {cfg.get('yolo_model', 'yoloe-11l-seg.pt')}")
    is_open = bool(cfg.get("is_open_vocabulary", False))
    if is_open:
        cls_list = list(classes) if classes else []
        cls_str = ", ".join(cls_list) if cls_list else "(none)"
        print(f"  vocabulary:        open  ({len(cls_list)} classes: {cls_str})")
    else:
        print(f"  vocabulary:        closed")
    print(f"  conf:              {cfg.get('conf', 0.25)}    iou: {cfg.get('iou', 0.5)}")
    print(f"  tracker:           {cfg.get('tracker_cfg', 'botsort.yaml')}")


def _print_depth(dp_type: str, cfg, scene_path: str) -> None:
    print()
    print(f"Depth provider:    {dp_type}")
    if dp_type == "gt":
        print("  source:            ground-truth depth from scene directory")
    elif dp_type == "pi3_online":
        _print_pi3_online_section(cfg, scene_path)
    elif dp_type == "pi3_offline":
        _print_pi3_offline_section(cfg, scene_path)
    elif dp_type == "dav3_online":
        _print_dav3_online_section(cfg)
    elif dp_type == "dav3_offline":
        _print_dav3_offline_section(cfg, scene_path)


def _print_pi3_online_section(cfg, scene_path: str) -> None:
    print(f"  model:             {cfg.get('pi3_model', 'yyfz233/Pi3X')}")
    print(
        f"  chunk_size:        {cfg.get('pi3_window_size', 5)}    "
        f"overlap: {cfg.get('pi3_overlap', 3)}"
    )
    print(f"  use_intrinsics:    {bool(cfg.get('pi3_use_intrinsics', True))}")
    print(f"  conf_threshold:    {cfg.get('pi3_conf_threshold', 0.05)}")
    inject = list(cfg.get("pi3_inject_condition", []) or [])
    print(f"  inject_condition:  {inject if inject else '[]'}")
    print(
        f"  pixel_limit:       {cfg.get('pi3_pixel_limit', 255000)}    "
        f"use_original_size: {bool(cfg.get('pi3_use_original_size', False))}"
    )
    transform_raw = cfg.get(
        "pi3_online_transform_path", cfg.get("pi3_offline_transform_path"),
    )
    tp = _resolve_path(scene_path, transform_raw)
    require = bool(cfg.get("pi3_online_require_transform", False))
    print(f"  Sim(3) JSON:       {_path_status(tp, require)}")


def _print_pi3_offline_section(cfg, scene_path: str) -> None:
    depth_dir = _resolve_path(
        scene_path, cfg.get("pi3_offline_depth_dir", "pi3_depth"),
    )
    pose_path = _resolve_path(
        scene_path, cfg.get("pi3_offline_pose_path", "pi3_camera_poses.txt"),
    )
    transform = _resolve_path(scene_path, cfg.get("pi3_offline_transform_path"))
    require_t = bool(cfg.get("pi3_offline_require_transform", True))

    print(f"  depth_dir:         {depth_dir}")
    print(f"  pose_path:         {pose_path}")
    print(f"  Sim(3) JSON:       {_path_status(transform, require_t)}")
    scale = cfg.get("pi3_offline_png_depth_scale")
    if scale is None:
        print("  png_depth_scale:   (read from pi3_depth_meta.txt; default 0.001)")
    else:
        print(f"  png_depth_scale:   {scale} (cfg override)")


def _print_dav3_online_section(cfg) -> None:
    print(f"  model:             {cfg.get('dav3_model', 'depth-anything/DA3-LARGE')}")
    print(f"  device:            {cfg.get('device', '0')}")
    print(
        f"  chunk_size:        {cfg.get('dav3_window_size', cfg.get('pi3_window_size', 5))}    "
        f"overlap: {cfg.get('dav3_overlap', cfg.get('pi3_overlap', 3))}"
    )
    print(f"  process_res:       {cfg.get('dav3_process_res', 504)}")
    print(f"  use_ray_pose:      {bool(cfg.get('dav3_use_ray_pose', True))}")
    print(f"  use_intrinsics:    {bool(cfg.get('dav3_use_intrinsics', False))}")
    print(f"  scale_mode:        {cfg.get('dav3_scale_mode', 'none')}")
    print(f"  conf_percentile:   {cfg.get('dav3_conf_percentile', None)}")
    print(f"  mask_sky:          {bool(cfg.get('dav3_mask_sky', False))}")


def _print_dav3_offline_section(cfg, scene_path: str) -> None:
    depth_dir = _resolve_path(
        scene_path, cfg.get("dav3_offline_depth_dir", "dav3_depth"),
    )
    pose_path = _resolve_path(
        scene_path, cfg.get("dav3_offline_pose_path", "dav3_camera_poses.txt"),
    )
    print(f"  depth_dir:         {depth_dir}")
    print(f"  pose_path:         {pose_path}")
    scale = cfg.get("dav3_offline_png_depth_scale")
    if scale is not None:
        print(f"  png_depth_scale:   {scale}")


def _print_tracking(cfg) -> None:
    print()
    print("Tracking:")
    print(
        f"  voxel_size:        {cfg.get('registry_voxel_size', 0.0)} m  "
        f"(0 = random sub-sample)"
    )
    print(
        f"  max_pts/obj:       {cfg.get('max_points_per_obj', 'n/a')}    "
        f"accumulated cap: {cfg.get('max_accumulated_points', 'n/a')}"
    )
    print(
        f"  cascade:           overlap_th={cfg.get('tracking_overlap_threshold', 'n/a')}, "
        f"dist_th={cfg.get('tracking_distance_threshold', 'n/a')}"
    )
    me = int(cfg.get("merge_every_n_frames", 0))
    me_str = f"every {me} frames" if me > 0 else "disabled"
    print(
        f"  dedup merge:       {me_str}  "
        f"(iou={cfg.get('merge_iou_threshold', 'n/a')}, "
        f"containment={cfg.get('merge_containment_threshold', 'n/a')})"
    )
