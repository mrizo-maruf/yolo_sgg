"""Depth provider factory — build a DepthProvider from config.

Usage::

    from depth_providers.factory import build_depth_provider

    dp = build_depth_provider("gt", "isaacsim", "/path/to/scene", cfg)
    loader = IsaacSimLoader(scene_dir, depth_provider=dp, ...)

Supported provider types:

* ``gt``            – Ground-truth depth from the scene directory.
* ``pi3_online``    – Pi3 streaming depth+pose (native provider).
* ``pi3_offline``   – Pre-computed Pi3 depth + pose (IsaacSim uses Sim(3) alignment).
* ``dav3_online``   – DepthAnything V3 streaming (v2 online, adapted).
* ``dav3_offline``  – Pre-computed DAv3 metric-depth PNGs.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from .base import DepthProvider


# Default depth sub-directories per dataset.
_DEPTH_DIRS: dict[str, str] = {
    "isaacsim": "depth",
    "thud_synthetic": "Depth",
    "coda": "depth",
    "scanetpp": "gt_depth",
}

# Provider type choices (for CLI help / validation).
PROVIDER_CHOICES = ("gt", "pi3_online", "pi3_offline", "dav3_online", "dav3_offline")


def build_depth_provider(
    provider_type: str,
    dataset_name: str,
    scene_dir: str,
    cfg,
) -> DepthProvider:
    """Construct a :class:`DepthProvider` from configuration.

    Parameters
    ----------
    provider_type : str
        One of ``PROVIDER_CHOICES``.
    dataset_name : str
        Dataset key (``isaacsim``, ``thud_synthetic``, ``coda``, ``scanetpp``).
    scene_dir : str
        Absolute path to the scene directory.
    cfg : OmegaConf
        Merged configuration object.

    Returns
    -------
    DepthProvider
    """
    scene_p = Path(scene_dir)

    if provider_type == "gt":
        return _build_gt(dataset_name, scene_p, cfg)
    if provider_type == "pi3_online":
        return _build_pi3_online(dataset_name, cfg)
    if provider_type == "pi3_offline":
        return _build_pi3_offline(dataset_name, scene_p, cfg)
    if provider_type == "dav3_online":
        return _build_dav3_online(cfg)
    if provider_type == "dav3_offline":
        return _build_dav3_offline(dataset_name, scene_p, cfg)

    raise ValueError(
        f"Unknown depth provider type: {provider_type!r}. "
        f"Choose from: {', '.join(PROVIDER_CHOICES)}"
    )


# ───────────────────────────────────────────────────────────────────────
# GT providers
# ───────────────────────────────────────────────────────────────────────

def _build_gt(dataset_name: str, scene_p: Path, cfg) -> DepthProvider:
    if dataset_name == "isaacsim":
        from .gt_depth import IsaacSimDepthProvider

        depth_dir = scene_p / "depth"
        pose_path = scene_p / "traj.txt"
        pose_lookup_cfg = str(cfg.get("pose_lookup_mode", "auto")).lower()
        pose_lookup = (
            _infer_pose_lookup_mode(depth_dir, "depth*.png", default="frame_number")
            if pose_lookup_cfg == "auto"
            else pose_lookup_cfg
        )
        return IsaacSimDepthProvider(
            depth_dir=str(depth_dir),
            png_max_value=int(cfg.get("png_max_value", 65535)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            min_depth=float(cfg.get("min_depth", 0.01)),
            pose_path=str(pose_path) if pose_path.exists() else None,
            pose_lookup=pose_lookup,
        )

    if dataset_name == "thud_synthetic":
        from .gt_depth import THUDSyntheticDepthProvider

        depth_dir = scene_p / "Depth"
        return THUDSyntheticDepthProvider(
            depth_dir=str(depth_dir),
            scale=float(cfg.get("depth_scale", 1000.0)),
        )

    if dataset_name == "coda":
        from .gt_depth import CODaDepthProvider

        depth_dir = scene_p / "depth"
        pose_path = scene_p / "traj.txt"
        return CODaDepthProvider(
            depth_dir=str(depth_dir),
            max_depth=float(cfg.get("max_depth", 80.0)),
            pose_path=str(pose_path) if pose_path.exists() else None,
        )

    if dataset_name == "scanetpp":
        from .gt_depth import ScanNetPPDepthProvider

        depth_dir = scene_p / "gt_depth"
        pose_path = scene_p / "traj.txt"
        return ScanNetPPDepthProvider(
            depth_dir=str(depth_dir),
            filename_pattern="frame_{frame_idx:06d}.png",
            depth_scale=float(cfg.get("depth_scale", 1000.0)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            min_depth=float(cfg.get("min_depth", 0.01)),
            pose_path=str(pose_path) if pose_path.exists() else None,
        )

    raise ValueError(f"No GT depth provider for dataset: {dataset_name!r}")


# ───────────────────────────────────────────────────────────────────────
# Offline predicted (Pi3 / DAv3) — standard metric-depth PNGs
# ───────────────────────────────────────────────────────────────────────

_FILENAME_PATTERNS: dict[str, str] = {
    "isaacsim": "depth{frame_number:06d}.png",
    "thud_synthetic": "depth_{frame_idx}.png",
    "coda": "depth{frame_number:06d}.png",
    "scanetpp": "frame_{frame_idx:06d}.png",
}

_DEFAULT_MAX_DEPTH: dict[str, float] = {
    "isaacsim": 10.0,
    "thud_synthetic": 100.0,
    "coda": 80.0,
    "scanetpp": 10.0,
}

_TRAILING_INT_RE = re.compile(r"(\d+)$")


def _resolve_scene_path(scene_p: Path, value, default_rel: str) -> Path:
    raw = str(value) if value is not None else default_rel
    p = Path(raw)
    return p if p.is_absolute() else scene_p / p


def _maybe_existing_path(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _infer_pose_lookup_mode(
    frame_dir: Path,
    pattern: str,
    default: str = "frame_number",
) -> str:
    """Infer 0-based ('index') vs 1-based ('frame_number') frame IDs."""
    if not frame_dir.exists():
        return default

    ids: list[int] = []
    for p in sorted(frame_dir.glob(pattern)):
        m = _TRAILING_INT_RE.search(p.stem)
        if not m:
            continue
        ids.append(int(m.group(1)))
        if len(ids) >= 64:
            break

    if not ids:
        return default
    return "index" if min(ids) == 0 else "frame_number"


def _build_metric_png_offline(
    dataset_name: str,
    scene_p: Path,
    cfg,
    *,
    depth_subdir_key: str,
    depth_default: str,
    pose_path_key: Optional[str] = None,
    pose_default: str = "traj.txt",
) -> DepthProvider:
    """MetricPngDepthProvider with dataset-appropriate defaults."""

    if dataset_name == "scanetpp":
        from .gt_depth import ScanNetPPDepthProvider

        depth_dir = scene_p / "pi3_depth"
        pose_path = scene_p / "pi3_traj.txt"
        return ScanNetPPDepthProvider(
            depth_dir=str(depth_dir),
            filename_pattern="frame_{frame_idx:04d}.png",
            depth_scale=float(cfg.get("depth_scale", 1000.0)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            min_depth=float(cfg.get("min_depth", 0.01)),
            pose_path=str(pose_path) if pose_path.exists() else None,
        )

    from .gt_depth import MetricPngDepthProvider

    depth_dir = _resolve_scene_path(
        scene_p,
        cfg.get(depth_subdir_key, cfg.get("predicted_depth_dir")),
        depth_default,
    )

    pose_path = None
    if pose_path_key is not None:
        pose_candidate = _resolve_scene_path(
            scene_p,
            cfg.get(pose_path_key, cfg.get("predicted_pose_path")),
            pose_default,
        )
        pose_path = _maybe_existing_path(pose_candidate)

    return MetricPngDepthProvider(
        depth_dir=str(depth_dir),
        filename_pattern=_FILENAME_PATTERNS.get(
            dataset_name, "depth{frame_idx:06d}.png"
        ),
        png_max_value=int(cfg.get("png_max_value", 65535)),
        max_depth=float(
            cfg.get("max_depth", _DEFAULT_MAX_DEPTH.get(dataset_name, 10.0))
        ),
        min_depth=float(cfg.get("min_depth", 0.01)),
        pose_path=pose_path,
    )


def _build_pi3_offline(dataset_name: str, scene_p: Path, cfg) -> DepthProvider:
    if dataset_name == "isaacsim":
        from .pi3_offline import IsaacSimOfflinePi3DepthProvider

        depth_dir = _resolve_scene_path(
            scene_p,
            cfg.get("pi3_offline_depth_dir", cfg.get("predicted_depth_dir")),
            "pi3_depth",
        )
        pose_path = _resolve_scene_path(
            scene_p,
            cfg.get("pi3_offline_pose_path", cfg.get("predicted_pose_path")),
            "pi3_camera_poses.txt",
        )
        transform_path = _resolve_scene_path(
            scene_p,
            cfg.get("pi3_offline_transform_path"),
            "pi3_to_world_transform.json",
        )

        png_scale = cfg.get("pi3_offline_png_depth_scale", cfg.get("pi3_png_depth_scale"))
        if png_scale is not None:
            png_scale = float(png_scale)

        pose_lookup_cfg = str(
            cfg.get("pi3_offline_pose_lookup_mode", cfg.get("pose_lookup_mode", "auto"))
        ).lower()
        pose_lookup = (
            _infer_pose_lookup_mode(depth_dir, "depth*.png", default="frame_number")
            if pose_lookup_cfg == "auto"
            else pose_lookup_cfg
        )

        return IsaacSimOfflinePi3DepthProvider(
            depth_dir=str(depth_dir),
            pose_path=_maybe_existing_path(pose_path),
            transform_path=str(transform_path),
            png_depth_scale=png_scale,
            min_depth=float(cfg.get("min_depth", 0.01)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            pose_lookup=pose_lookup,
            require_transform=bool(cfg.get("pi3_offline_require_transform", True)),
        )

    # Other datasets: keep generic metric-PNG behavior.
    return _build_metric_png_offline(
        dataset_name,
        scene_p,
        cfg,
        depth_subdir_key="pi3_offline_depth_dir",
        depth_default="pi3_depth",
        pose_path_key="pi3_offline_pose_path",
        pose_default="pi3_traj.txt",
    )


def _build_dav3_offline(dataset_name: str, scene_p: Path, cfg) -> DepthProvider:
    return _build_metric_png_offline(
        dataset_name,
        scene_p,
        cfg,
        depth_subdir_key="dav3_offline_depth_dir",
        depth_default="dav3_depth",
        pose_path_key="dav3_offline_pose_path",
        pose_default="dav3_traj.txt",
    )


# ───────────────────────────────────────────────────────────────────────
# Online providers (adapted from v2)
# ───────────────────────────────────────────────────────────────────────

class _OnlineProviderAdapter(DepthProvider):
    """Wrap a v2 ``OnlineDepthProvider`` so it satisfies the main
    ``depth_providers.base.DepthProvider`` interface (and gains the
    default ``get_masked_pcds`` implementation).
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    # --- DepthProvider interface ---
    def get_depth(self, frame_idx):
        return self._inner.get_depth(frame_idx)

    def get_pose(self, frame_idx):
        return self._inner.get_pose(frame_idx)

    # --- OnlineDepthProvider interface ---
    def feed_frame(self, frame_idx, rgb):
        self._inner.feed_frame(frame_idx, rgb)

    def warmup(self):
        self._inner.warmup()

    def close(self):
        self._inner.close()


def _parse_inject_condition(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x) for x in raw]


def _build_pi3_intrinsics_from_cfg(dataset_name: str, cfg) -> tuple[Optional[np.ndarray], Optional[tuple[int, int]]]:
    """Best-effort camera K extraction for Pi3 intrinsics injection.

    Returns
    -------
    (K, image_size)
        K is a 3x3 float32 matrix (or None).
        image_size is (H, W) where K is defined (or None).
    """
    iw = cfg.get("image_width")
    ih = cfg.get("image_height")
    if iw is None or ih is None:
        return None, None

    w = int(iw)
    h = int(ih)
    image_size = (h, w)

    fx = cfg.get("fx")
    fy = cfg.get("fy")
    cx = cfg.get("cx")
    cy = cfg.get("cy")

    if fx is not None and fy is not None and cx is not None and cy is not None:
        K = np.array(
            [
                [float(fx), 0.0, float(cx)],
                [0.0, float(fy), float(cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return K, image_size

    if dataset_name == "isaacsim":
        focal = cfg.get("focal_length")
        h_ap = cfg.get("horizontal_aperture")
        v_ap = cfg.get("vertical_aperture")
        if focal is not None and h_ap is not None and v_ap is not None:
            fx = float(focal) / float(h_ap) * w
            fy = float(focal) / float(v_ap) * h
            cx = float(w) / 2.0
            cy = float(h) / 2.0
            K = np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K, image_size

    return None, image_size


def _build_pi3_online(dataset_name: str, cfg) -> DepthProvider:
    from .pi3_online import Pi3OnlineDepthProvider

    target_h = cfg.get("pi3_target_height", cfg.get("image_height"))
    target_w = cfg.get("pi3_target_width", cfg.get("image_width"))
    target_size = (
        (int(target_h), int(target_w))
        if target_h is not None and target_w is not None
        else None
    )

    K, K_size = _build_pi3_intrinsics_from_cfg(dataset_name, cfg)
    if not bool(cfg.get("pi3_use_intrinsics", True)):
        K = None
        K_size = None

    return Pi3OnlineDepthProvider(
        model_name=str(cfg.get("pi3_model", cfg.get("pi3_model_name", "yyfz233/Pi3X"))),
        chunk_size=int(cfg.get("pi3_chunk_size", cfg.get("pi3_window_size", 30))),
        overlap=int(cfg.get("pi3_overlap", 10)),
        target_size=target_size,
        device=(
            str(cfg.get("pi3_device"))
            if cfg.get("pi3_device") is not None
            else (str(cfg.get("device")) if cfg.get("device") is not None else None)
        ),
        max_cache=int(cfg.get("pi3_online_max_cache", cfg.get("online_max_cache", 128))),
        min_depth=float(cfg.get("min_depth", 0.01)),
        max_depth=float(cfg.get("max_depth", 0.0)),
        conf_threshold=float(cfg.get("pi3_conf_threshold", 0.05)),
        inject_condition=_parse_inject_condition(cfg.get("pi3_inject_condition", [])),
        intrinsics=K,
        intrinsics_image_size=K_size,
        use_original_size=bool(cfg.get("pi3_use_original_size", False)),
        pixel_limit=int(cfg.get("pi3_pixel_limit", 255000)),
        patch_size=int(cfg.get("pi3_patch_size", 14)),
    )


def _build_dav3_online(cfg) -> DepthProvider:
    from v2.depth_providers.dav3_online import DAv3StreamingDepthProvider

    inner = DAv3StreamingDepthProvider(
        model_name=str(cfg.get("dav3_model", "depth-anything-v3")),
        device=str(cfg.get("device", "cuda")),
    )
    return _OnlineProviderAdapter(inner)
