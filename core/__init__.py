"""Core tracking package.

Old symbols (TrackedFrame, run_tracking, helpers, yolo_runner) are
available via lazy import so that ``import core.types`` or other new
sub-modules no longer trigger the heavy ultralytics dependency chain.
"""

__all__ = [
    "TrackedFrame",
    "run_tracking",
    "extract_yolo_ids",
    "apply_class_filter",
    "build_pred_instances",
    "should_skip_class",
    "run_yolo_tracking_stream",
]


def __getattr__(name: str):
    if name in ("TrackedFrame", "run_tracking"):
        from .tracker import TrackedFrame, run_tracking  # noqa: F811
        return TrackedFrame if name == "TrackedFrame" else run_tracking
    if name in (
        "extract_yolo_ids",
        "apply_class_filter",
        "build_pred_instances",
        "should_skip_class",
    ):
        from . import helpers
        return getattr(helpers, name)
    if name == "run_yolo_tracking_stream":
        from .yolo_runner import run_yolo_tracking_stream
        return run_yolo_tracking_stream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
