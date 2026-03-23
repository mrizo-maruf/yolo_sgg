"""
Loader registry — maps dataset name -> loader class.

Usage:
    from data_loaders import get_loader
    LoaderCls = get_loader("isaacsim")
    loader = LoaderCls(scene_dir, ...)
"""
from __future__ import annotations

from typing import Dict, Type

from .base import DatasetLoader

LOADER_REGISTRY: Dict[str, str] = {
    "isaacsim":       "data_loaders.isaacsim.IsaacSimLoader",
    "thud_synthetic": "data_loaders.thud_synthetic.THUDSyntheticLoader",
    "coda":           "data_loaders.coda.CODaLoader",
    "scanepp":        "data_loaders.scanepp.ScanNetPPLoader",
}


def get_loader(name: str) -> Type[DatasetLoader]:
    """Return the loader class for *name*."""
    if name not in LOADER_REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(LOADER_REGISTRY.keys())}"
        )
    module_path, cls_name = LOADER_REGISTRY[name].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)
