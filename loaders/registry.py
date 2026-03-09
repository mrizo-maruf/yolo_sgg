"""
Loader registry — maps dataset name → loader class.

Usage:
    from loaders import get_loader
    LoaderCls = get_loader("isaacsim")
    loader = LoaderCls(scene_dir, ...)
"""
from __future__ import annotations

from typing import Dict, Type

from .base import DatasetLoader

# Lazy imports to avoid pulling heavy deps at import time
LOADER_REGISTRY: Dict[str, str] = {
    "isaacsim":        "loaders.isaacsim.IsaacSimLoader",
    "thud_synthetic":  "loaders.thud_synthetic.THUDSyntheticDatasetLoader",
    "thud_real":       "loaders.thud_real.THUDRealDatasetLoader",
}


def get_loader(name: str) -> Type[DatasetLoader]:
    """Return the loader class for *name*.

    Raises ``KeyError`` if *name* is not registered.
    """
    if name not in LOADER_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(LOADER_REGISTRY.keys())}"
        )
    module_path, cls_name = LOADER_REGISTRY[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)
