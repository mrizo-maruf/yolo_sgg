"""Dataset loaders with pluggable depth providers."""

from .base import DatasetLoader
from .isaacsim import IsaacSimLoader
from .thud_synthetic import THUDSyntheticLoader
from .camera import CameraLoader

__all__ = [
    "DatasetLoader",
    "IsaacSimLoader",
    "THUDSyntheticLoader",
    "CameraLoader",
]

# Registry for string-based lookup
_REGISTRY = {
    "isaacsim": "v2.loaders.isaacsim.IsaacSimLoader",
    "thud_synthetic": "v2.loaders.thud_synthetic.THUDSyntheticLoader",
    "camera": "v2.loaders.camera.CameraLoader",
}


def get_loader(name: str) -> type:
    """Return loader class by dataset name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY.keys())}")
    module_path, cls_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(module_path), cls_name)
