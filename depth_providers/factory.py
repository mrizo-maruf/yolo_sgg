"""Depth provider factory — build a DepthProvider from config.

Usage::

    from depth_providers.factory import build_depth_provider

    dp = build_depth_provider("gt", "isaacsim", "/path/to/scene", cfg)
    loader = IsaacSimLoader(scene_dir, depth_provider=dp, ...)

Supported provider types:

* ``gt``            – Ground-truth depth from the scene directory.
* ``pi3_online``    – Pi3 streaming (v2 online provider, adapted).
* ``pi3_offline``   – Pre-computed Pi3 metric-depth PNGs.
* ``dav3_online``   – DepthAnything V3 streaming (v2 online, adapted).
* ``dav3_offline``  – Pre-computed DAv3 metric-depth PNGs.
"""
from __future__ import annotations

from pathlib import Path

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
        return _build_pi3_online(cfg)
    if provider_type == "pi3_offline":
        return _build_offline_predicted(dataset_name, scene_p, cfg)
    if provider_type == "dav3_online":
        return _build_dav3_online(cfg)
    if provider_type == "dav3_offline":
        return _build_offline_predicted(dataset_name, scene_p, cfg)

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
        return IsaacSimDepthProvider(
            depth_dir=str(depth_dir),
            png_max_value=int(cfg.get("png_max_value", 65535)),
            max_depth=float(cfg.get("max_depth", 10.0)),
            min_depth=float(cfg.get("min_depth", 0.01)),
            pose_path=str(pose_path) if pose_path.exists() else None,
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


def _build_offline_predicted(
    dataset_name: str, scene_p: Path, cfg,
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

    depth_subdir = str(
        cfg.get("predicted_depth_dir", _DEPTH_DIRS.get(dataset_name, "depth"))
    )
    depth_dir = scene_p / depth_subdir
    pose_path = scene_p / "traj.txt"

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
        pose_path=str(pose_path) if pose_path.exists() else None,
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


def _build_pi3_online(cfg) -> DepthProvider:
    from v2.depth_providers.pi3_online import Pi3OnlineDepthProvider

    inner = Pi3OnlineDepthProvider(
        model_name=str(cfg.get("pi3_model", "yyfz233/Pi3X")),
        window_size=int(cfg.get("pi3_window_size", 13)),
        overlap=int(cfg.get("pi3_overlap", 5)),
        device=str(cfg.get("device", "0")) if cfg.get("device") else None,
    )
    return _OnlineProviderAdapter(inner)


def _build_dav3_online(cfg) -> DepthProvider:
    from v2.depth_providers.dav3_online import DAv3StreamingDepthProvider

    inner = DAv3StreamingDepthProvider(
        model_name=str(cfg.get("dav3_model", "depth-anything-v3")),
        device=str(cfg.get("device", "cuda")),
    )
    return _OnlineProviderAdapter(inner)
