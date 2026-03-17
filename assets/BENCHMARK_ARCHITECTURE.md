# Generic Benchmarking & Depth Provider Architecture Proposal

## Problem

We have multiple team members, each benchmarking on different datasets (IsaacSim, THUD Synthetic, THUD Real, and future ones). Currently there are **two separate benchmark scripts** (`benchmark_tracking_isaac.py`, `benchmark_tracking_thud.py`) that are ~70% identical code. This creates:

- Duplicated helper functions (`_build_pred_instances`, `_visualize_frame`, `_print_aggregate`, `_plot_cross_scene`, `_build_overall_plot_results`)
- Inconsistent CLI interfaces per dataset
- Each new dataset requires copying/pasting an entire benchmark file
- Bug fixes must be applied in multiple places

## Current Flow (What Works Well)

The project already has strong abstractions that should be preserved:

```
DatasetLoader (ABC)          ← each dataset implements this
    ↓
run_tracking() generator     ← dataset-agnostic, yields TrackedFrame
    ↓
MetricsAccumulator           ← dataset-agnostic metric computation
    ↓
save_metrics / plot_results  ← dataset-agnostic output
```

The `loaders/registry.py` already maps `"isaacsim" → IsaacSimLoader`, etc., but the benchmarks **don't use it** — they hardcode loader imports.

## Proposed Architecture

### One script: `benchmark/benchmark.py`

```
python benchmark/benchmark.py --dataset isaacsim /path/to/scene_1
python benchmark/benchmark.py --dataset thud_synthetic /path/to/Gym/static/Capture_1
python benchmark/benchmark.py --dataset thud_real /path/to/Real_Scenes/10L --multi
python benchmark/benchmark.py --dataset my_new_dataset /path/to/data
```

### Module structure

```
benchmark/
├── benchmark.py             ← single CLI entry point
├── runner.py                ← benchmark_scene() + benchmark_dataset()
├── common.py                ← shared helpers (build_pred_instances, etc.)
├── visualization.py         ← existing (unchanged)
├── __init__.py
```

---

## Design Details

### 1. Unified Loader Construction

Currently Isaac passes 12+ individual params; THUD uses a factory. Standardize to:

```python
# In runner.py
loader_cls = get_loader(cfg.dataset)          # from loaders/registry.py
loader = loader_cls.from_config(scene_path, cfg)
```

Each loader adds a `@classmethod from_config(cls, scene_dir, cfg)` that pulls the params it needs from the OmegaConf config. This is **one method per loader**, not a change to the ABC — just a convention.

```python
# loaders/isaacsim.py
class IsaacSimLoader(DatasetLoader):
    @classmethod
    def from_config(cls, scene_dir: str, cfg: OmegaConf) -> "IsaacSimLoader":
        return cls(
            scene_dir=scene_dir,
            skip_labels=set(cfg.get("loader_skip_labels", [])),
            image_width=int(cfg.get("image_width", 1280)),
            image_height=int(cfg.get("image_height", 720)),
            focal_length=float(cfg.get("focal_length", 50)),
            horizontal_aperture=float(cfg.get("horizontal_aperture", 80)),
            vertical_aperture=float(cfg.get("vertical_aperture", 45)),
            pi3=cfg.get("pi3", False),
            dav3=cfg.get("dav3", False),
            ...
        )
```

### 2. Standardize Camera Intrinsics Override

Both benchmarks do this differently. Unify into `runner.py`:

```python
intrinsics = loader.get_camera_intrinsics()
if intrinsics is not None:
    K, img_h, img_w = intrinsics
    yutils.fx = float(K[0, 0])
    yutils.fy = float(K[1, 1])
    yutils.cx = float(K[0, 2])
    yutils.cy = float(K[1, 2])
    yutils.IMAGE_WIDTH = img_w
    yutils.IMAGE_HEIGHT = img_h
```

This is already what `run.py` does via `_override_intrinsics()`. The benchmark should do the same.

### 3. Standardize Class Name Resolution

Currently scattered: Isaac uses `cfg.{scene_name}_class_names`, THUD walks path parts. Move to the loader:

```python
# Add to DatasetLoader ABC (optional method with default)
def get_class_names_to_track(self, cfg: OmegaConf) -> Optional[List[str]]:
    """Return open-vocabulary class names for this scene, or None."""
    return list(cfg.get("track_classes", [])) or None
```

Override per dataset:

```python
# IsaacSimLoader
def get_class_names_to_track(self, cfg):
    key = f"{self.scene_label}_class_names"
    names = cfg.get(key)
    return list(names) if names else None

# THUDSyntheticDatasetLoader  
def get_class_names_to_track(self, cfg):
    for part in reversed(Path(self._scene_dir).parts):
        key = f"{part}_class_names"
        if cfg.get(key):
            return list(cfg.get(key))
    return None
```

### 4. Unified Scene Discovery

The `DatasetLoader` ABC already declares `discover_scenes()`. Isaac's benchmark uses raw `iterdir()` instead. Fix Isaac's loader:

```python
# IsaacSimLoader (already exists, just needs to be used)
@classmethod
def discover_scenes(cls, root: str, **kwargs) -> List[str]:
    return sorted(str(d) for d in Path(root).iterdir()
                  if d.is_dir() and (d/"rgb").exists() and (d/"depth").exists())
```

Then `runner.py` just calls:

```python
scenes = loader_cls.discover_scenes(root_path)
```

### 5. Extract Duplicated Helpers → `benchmark/common.py`

```python
# benchmark/common.py

def build_pred_instances(frame_objs, track_ids, masks_clean) -> List[PredInstance]:
    """Convert tracker output into PredInstance list for metric matching."""
    ...

def visualize_frame(rgb_path, gt_instances, pred_instances,
                    mapping, ious, frame_idx, vis_cfg, vis_save):
    """Optional per-frame visualization with GT-vs-pred matching."""
    ...

def print_aggregate(overall: Dict, keys: List[str], title: str = "AGGREGATE"):
    ...

def plot_cross_scene(all_results, agg_keys, out_dir, title_prefix=""):
    ...

def build_overall_plot_results(all_results: Dict[str, Dict]) -> Dict:
    ...
```

### 6. Unified `benchmark_scene()` in `runner.py`

```python
def benchmark_scene(scene_path: str, cfg: OmegaConf) -> Dict:
    dataset = cfg.dataset
    loader_cls = get_loader(dataset)
    loader = loader_cls.from_config(scene_path, cfg)

    # Camera intrinsics → global override
    intrinsics = loader.get_camera_intrinsics()
    if intrinsics:
        _override_intrinsics(*intrinsics)

    # Data preparation (identical for all datasets)
    rgb_paths = loader.get_rgb_paths()
    depth_paths = loader.get_depth_paths()
    depth_cache = loader.build_depth_cache(max_cached=cfg.get("max_depth_cached", 64))
    poses = loader.get_all_poses()
    point_extractor = loader.get_point_extractor(poses=poses)
    class_names = loader.get_class_names_to_track(cfg)

    acc = MetricsAccumulator()

    for tf in run_tracking(
        rgb_paths=rgb_paths,
        depth_paths=depth_paths,
        depth_cache=depth_cache,
        poses=poses,
        cfg=cfg,
        class_names_to_track=class_names,
        point_extractor=point_extractor,
    ):
        gt = loader.get_gt_instances(tf.frame_idx)
        preds = build_pred_instances(tf.frame_objs, tf.track_ids, tf.masks_clean)
        mapping, ious = match_greedy(gt, preds)
        acc.add_frame(FrameRecord(tf.frame_idx, gt, preds, mapping, ious))

    return acc.compute()
```

Notice: **zero dataset-specific logic**. Isaac, THUD synthetic, THUD real, and future datasets all go through this exact same path. The differences live entirely in the loader.

### 7. Unified CLI

```python
def _build_parser():
    p = argparse.ArgumentParser(description="Generic tracking benchmark")
    p.add_argument("path", help="Scene or dataset root directory")
    p.add_argument("--dataset", required=True,
                   choices=["isaacsim", "thud_synthetic", "thud_real"],
                   help="Dataset type")
    p.add_argument("--multi", action="store_true",
                   help="Run on all scenes under path")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory for metrics/plots")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--vis-interval", type=int, default=10)
    p.add_argument("--rerun", action="store_true")
    return p
```

Config loading (same pattern as `run.py`):

```python
cfg = OmegaConf.merge(
    OmegaConf.load("configs/core_tracking.yaml"),
    OmegaConf.load(f"configs/{args.dataset}.yaml"),
)
```

---

## Adding a New Dataset (Checklist for Team Members)

A team member adding dataset `"foo"` only needs to:

| Step | File | What to do |
|------|------|-----------|
| 1 | `loaders/foo.py` | Create `FooLoader(DatasetLoader)` implementing the ABC + `from_config()` |
| 2 | `loaders/registry.py` | Add `"foo": "loaders.foo.FooLoader"` |
| 3 | `configs/foo.yaml` | Create with `dataset: foo` and any dataset-specific params |
| 4 | *(optional)* | Override `get_point_extractor()` if depth uses non-standard encoding |
| 5 | *(optional)* | Override `get_class_names_to_track()` if class name discovery is custom |

**No changes to benchmark code, tracker, metrics, or visualization.** Run with:

```bash
python benchmark/benchmark.py --dataset foo /path/to/data
python benchmark/benchmark.py --dataset foo /path/to/root --multi
```

---

## Migration Path

This can be done incrementally without breaking existing scripts:

1. **Phase 1**: Create `benchmark/common.py` — extract the 5 duplicated helpers from both benchmark files. Update both files to import from `common.py`. Both scripts still work independently.

2. **Phase 2**: Add `from_config()` classmethod to each loader. Add `get_class_names_to_track()` to each loader.

3. **Phase 3**: Create `benchmark/runner.py` with unified `benchmark_scene()` and `benchmark_dataset()`.

4. **Phase 4**: Create `benchmark/benchmark.py` as the single CLI. Keep old scripts as thin wrappers calling `runner.benchmark_scene()` for backwards compatibility.

5. **Phase 5**: Once team is migrated, deprecate old scripts.

---

## Output Structure

Standardize output for all datasets:

```
results/
└── {dataset}_{timestamp}/
    ├── config.yaml                    # frozen config snapshot
    ├── {scene_name}_metrics.json      # per-scene metrics
    ├── {scene_name}_plots.png         # per-scene plots
    ├── aggregate_metrics.json         # multi-scene aggregate
    └── cross_scene_comparison.png     # multi-scene bar chart
```

---

## Summary of Changes

| Current | Proposed |
|---------|----------|
| 2 benchmark scripts (~600 lines each, ~70% identical) | 1 unified script + shared runner (~400 lines total) |
| Loader hardcoded in benchmark | Loader resolved via registry |
| CLI differs per dataset | Single CLI with `--dataset` flag |
| 5 duplicated helper functions | Extracted to `benchmark/common.py` |
| Adding a dataset = copy script + modify | Adding a dataset = write loader + config + register |
| Camera intrinsics setup differs per benchmark | Uniform intrinsics flow via `loader.get_camera_intrinsics()` |
| Class name resolution is ad-hoc | `loader.get_class_names_to_track(cfg)` |
| Scene discovery is ad-hoc (Isaac) | `loader_cls.discover_scenes(root)` |

---
---

# Part 2: Depth & Pose Provider Architecture

## Problem

Currently depth and camera poses come from three completely different code paths:

| Source | How it works today | Where the code lives |
|--------|-------------------|---------------------|
| **GT depth** (default) | 16-bit PNG → `raw / 65535 * MAX_DEPTH` | `YOLOE/utils.load_depth_as_meters()` |
| **Pi3X** | Offline pre-processing → 8-bit PNGs + `depth_scale.json` + `pi3_traj.txt` | `pi3_utils.py` (preprocessing), `loaders/isaacsim.py` (loading) |
| **DAv3** | Flag exists but stubbed | `isaacsim.yaml` flag, passed to inner loader |
| **THUD custom** | Raw uint16 → custom per-axis divisor formula | `loaders/thud_synthetic.py` + inline closure |
| **VGGT** | Direct 3D point cloud (no depth maps) | `vggt_utils/yolo_ssg_vggt.py` (standalone script, not integrated) |

Problems:
- Pi3 is hardwired into `IsaacSimLoader` via `if self._pi3:` branches — adding DAv3 means adding more branches
- Each depth model has a different preprocessing step, output format, and loading path
- No way to swap depth sources without editing loader code
- VGGT (direct point cloud, no depth) doesn't fit the depth-map abstraction at all
- Camera poses from predicted models vs GT are tangled with depth loading

## Key Insight: Two Orthogonal Axes

The pipeline has **two independent choices**:

```
1. Dataset     — IsaacSim, THUD Synthetic, THUD Real, Custom, ...
2. Depth/Pose  — GT, Pi3X, DAv3, DepthPro, VGGT, ...
```

These should be **independent**. You should be able to run:

```bash
python run.py --dataset isaacsim --depth-source pi3    /path/to/scene
python run.py --dataset isaacsim --depth-source dav3   /path/to/scene
python run.py --dataset isaacsim --depth-source gt     /path/to/scene
python run.py --dataset thud_real --depth-source pi3   /path/to/scene
```

Currently Pi3 is baked into `IsaacSimLoader` — it can't be used with THUD without duplicating code.

## Proposed Architecture

### New module: `depth_providers/`

```
depth_providers/
├── __init__.py          ← registry: get_depth_provider("pi3") → Pi3Provider
├── base.py              ← DepthProvider ABC
├── gt.py                ← GTDepthProvider (passthrough, uses loader's own depth)
├── pi3.py               ← Pi3Provider
├── depth_anything.py    ← DepthAnythingProvider (DAv3)
├── vggt.py              ← VGGTProvider (direct point cloud, no depth maps)
└── depth_pro.py         ← (future)
```

### `DepthProvider` ABC

```python
# depth_providers/base.py

class DepthProvider(ABC):
    """Provides depth maps and camera poses for a scene.

    A depth provider either wraps the dataset's own GT depth (passthrough)
    or runs / loads predictions from a depth model.

    Two modes of operation:
      1. Depth-map mode (default): provides per-frame depth images + poses.
         The pipeline uses standard pinhole unprojection.
      2. Point-cloud mode (VGGT-style): provides per-frame 3D points directly.
         The pipeline skips depth unprojection entirely.
    """

    @abstractmethod
    def setup(self, scene_dir: str, rgb_paths: List[str], cfg: OmegaConf) -> None:
        """One-time setup: run model inference or locate pre-computed outputs.

        Called once before processing begins. For offline models (Pi3, DAv3),
        this is where inference happens (or cached results are found).
        For GT, this is a no-op.
        """
        ...

    @abstractmethod
    def get_depth_paths(self) -> List[str]:
        """Return ordered depth image paths (same length as rgb_paths).

        For point-cloud-mode providers (VGGT), return empty list.
        """
        ...

    @abstractmethod
    def load_depth(self, path: str) -> np.ndarray:
        """Load a single depth frame → float32 metres (H, W).

        For point-cloud-mode providers, this raises NotImplementedError.
        """
        ...

    @abstractmethod
    def get_poses(self) -> Optional[List[np.ndarray]]:
        """Return camera-to-world 4×4 poses, one per frame (or None)."""
        ...

    @abstractmethod
    def get_intrinsics(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """Return (K_3x3, H, W) or None.

        Predicted models may produce their own intrinsics.
        If None, the dataset loader's intrinsics are used.
        """
        ...

    # --- Optional: point-cloud mode ---

    @property
    def provides_pointclouds(self) -> bool:
        """True if this provider gives 3D points directly (no depth maps)."""
        return False

    def get_point_extractor(self) -> Optional[Callable]:
        """Return a custom point extractor for point-cloud-mode providers.

        Signature: (depth_or_data, mask, frame_idx, max_points,
                    o3_nb_neighbors, o3std_ratio, track_id) → (N, 3)
        """
        return None
```

### `GTDepthProvider` — Passthrough

```python
# depth_providers/gt.py

class GTDepthProvider(DepthProvider):
    """Passthrough: uses the dataset loader's own depth and poses."""

    def setup(self, scene_dir, rgb_paths, cfg, *, loader=None):
        self._loader = loader   # the DatasetLoader instance

    def get_depth_paths(self):
        return self._loader.get_depth_paths()

    def load_depth(self, path):
        return self._loader.load_depth(path)

    def get_poses(self):
        return self._loader.get_all_poses()

    def get_intrinsics(self):
        return None   # use loader's intrinsics
```

### `Pi3Provider`

```python
# depth_providers/pi3.py

class Pi3Provider(DepthProvider):
    """Depth + poses from Pi3X visual odometry model."""

    def setup(self, scene_dir, rgb_paths, cfg, **kwargs):
        self._scene_dir = Path(scene_dir)
        self._output_dir = self._scene_dir / "pi3_depth"
        self._traj_path = self._scene_dir / "pi3_traj.txt"

        if self._output_dir.exists() and self._traj_path.exists():
            print(f"[Pi3] Using cached predictions from {self._output_dir}")
        else:
            print(f"[Pi3] Running Pi3X inference...")
            self._run_inference(rgb_paths, cfg)

        # Load depth scale for metric recovery
        with open(self._output_dir / "depth_scale.json") as f:
            scale = json.load(f)
        self._depth_min = float(scale["global_depth_min"])
        self._depth_max = float(scale["global_depth_max"])

        self._poses = yutils.load_camera_poses(str(self._traj_path))
        self._depth_paths = sorted(
            str(p) for p in self._output_dir.glob("*.png")
            if p.name != "depth_video.mp4"
        )

    def get_depth_paths(self):
        return self._depth_paths

    def load_depth(self, path):
        d = np.array(Image.open(path))  # uint8
        depth_m = d.astype(np.float32) / 255.0 * (
            self._depth_max - self._depth_min
        ) + self._depth_min
        depth_m[d == 0] = 0.0
        return depth_m

    def get_poses(self):
        return self._poses

    def get_intrinsics(self):
        return None   # Pi3 doesn't predict intrinsics, use loader's

    def _run_inference(self, rgb_paths, cfg):
        # Move current pi3_utils.process_depth_model() logic here
        ...
```

### `VGGTProvider` — Point-Cloud Mode

```python
# depth_providers/vggt.py

class VGGTProvider(DepthProvider):
    """Direct 3D point clouds from VGGT — no depth maps."""

    def setup(self, scene_dir, rgb_paths, cfg, **kwargs):
        self._frame_data_dir = Path(scene_dir) / "frame_data_vggt"
        # ... load camera JSONs, verify data exists

    @property
    def provides_pointclouds(self):
        return True

    def get_depth_paths(self):
        return []   # no depth maps

    def load_depth(self, path):
        raise NotImplementedError("VGGT provides point clouds, not depth maps")

    def get_poses(self):
        return self._poses   # from camera JSONs

    def get_intrinsics(self):
        return self._K, self._H, self._W   # from camera JSONs

    def get_point_extractor(self):
        """Return extractor that filters VGGT world-points by mask projection."""
        # Move logic from vggt_utils/yolo_ssg_vggt.py here
        def _extract(depth_or_data, mask, frame_idx, max_points, ...):
            vggt_points, vggt_colors = self._load_frame_pointcloud(frame_idx)
            # project to 2D, filter by mask, return world-frame points
            ...
        return _extract
```

### `DepthAnythingProvider` — Future

```python
# depth_providers/depth_anything.py

class DepthAnythingProvider(DepthProvider):
    """Depth from Depth Anything v3 (monocular metric depth)."""

    def setup(self, scene_dir, rgb_paths, cfg, **kwargs):
        # Check for cached outputs or run inference
        # DAv3 predicts per-frame metric depth but NOT camera poses
        ...

    def get_poses(self):
        return None   # DAv3 is depth-only, no pose estimation
        # Pipeline will fall back to loader's GT poses or fail gracefully
```

### Provider Registry

```python
# depth_providers/__init__.py

PROVIDER_REGISTRY = {
    "gt":              "depth_providers.gt.GTDepthProvider",
    "pi3":             "depth_providers.pi3.Pi3Provider",
    "depth_anything":  "depth_providers.depth_anything.DepthAnythingProvider",
    "vggt":            "depth_providers.vggt.VGGTProvider",
}

def get_depth_provider(name: str) -> type:
    """Resolve provider name → class (lazy import)."""
    ...
```

## Integration into the Pipeline

### Updated `run.py` / `runner.py` flow

```python
# 1. Build dataset loader (unchanged)
loader = loader_cls.from_config(scene_path, cfg)

# 2. Build depth provider
depth_source = cfg.get("depth_source", "gt")
provider_cls = get_depth_provider(depth_source)
provider = provider_cls()
provider.setup(scene_path, loader.get_rgb_paths(), cfg, loader=loader)

# 3. Get data from PROVIDER instead of LOADER
depth_paths = provider.get_depth_paths() or loader.get_depth_paths()
poses = provider.get_poses() or loader.get_all_poses()

# 4. Provider may override intrinsics (some models predict them)
provider_intrinsics = provider.get_intrinsics()
if provider_intrinsics is not None:
    K, img_h, img_w = provider_intrinsics
    _override_intrinsics(K, img_h, img_w)

# 5. Provider may supply a custom point extractor (VGGT)
point_extractor = provider.get_point_extractor() or loader.get_point_extractor(poses=poses)

# 6. Build depth cache using PROVIDER's load function
depth_cache = LazyDepthCache(
    load_fn=provider.load_depth,
    paths=depth_paths,
    max_cached=cfg.get("max_depth_cached", 64),
)

# 7. Everything downstream is unchanged
for tf in run_tracking(
    rgb_paths=rgb_paths,
    depth_paths=depth_paths,
    depth_cache=depth_cache,
    poses=poses,
    cfg=cfg,
    point_extractor=point_extractor,
):
    ...
```

### CLI

```bash
# GT depth (default, same as today)
python run.py --dataset isaacsim /path/to/scene

# Pi3 predicted depth
python run.py --dataset isaacsim --depth-source pi3 /path/to/scene

# Depth Anything v3 (depth only, GT poses)
python run.py --dataset isaacsim --depth-source depth_anything /path/to/scene

# VGGT (direct point clouds, no depth maps)
python run.py --dataset isaacsim --depth-source vggt /path/to/scene

# Works with any dataset × any provider
python run.py --dataset thud_real --depth-source pi3 /path/to/scene
```

### Config

```yaml
# configs/core_tracking.yaml
depth_source: gt       # gt | pi3 | depth_anything | vggt

# Or override per-dataset in configs/isaacsim.yaml:
# depth_source: pi3
```

## How This Cleans Up Existing Code

### What gets removed from loaders

The `pi3`/`dav3` flags and branches are **removed** from `IsaacSimLoader`:

```python
# BEFORE (isaacsim.py) — tangled
class IsaacSimLoader(DatasetLoader):
    def __init__(self, ..., pi3=False, dav3=False):
        if pi3:
            self._depth_dir = scene_dir / "pi3_depth"
            self._poses = load_poses("pi3_traj.txt")
            # load depth_scale.json...
        else:
            self._depth_dir = scene_dir / "depth"
            self._poses = load_poses("traj.txt")

    def load_depth(self, path):
        if self._pi3:
            return self._load_pi3_depth(path)   # 8-bit recovery
        return load_depth_as_meters(path)        # 16-bit standard

# AFTER — clean
class IsaacSimLoader(DatasetLoader):
    def __init__(self, scene_dir, ...):
        self._depth_dir = scene_dir / "depth"
        self._poses = load_poses("traj.txt")

    def load_depth(self, path):
        return load_depth_as_meters(path)    # always GT
```

The Pi3 logic moves from `IsaacSimLoader` to `Pi3Provider` where it belongs. The loader only knows about GT data. The provider only knows about the depth model.

## Depth Provider Comparison Matrix

| Provider | Produces depth maps? | Produces poses? | Produces intrinsics? | Needs GPU? | Offline? |
|----------|---------------------|----------------|---------------------|-----------|---------|
| GT | Yes (from loader) | Yes (from loader) | No (from loader) | No | — |
| Pi3X | Yes (8-bit PNG) | Yes (VO pipeline) | No | Yes | Yes (cached) |
| DAv3 | Yes (metric depth) | **No** (depth only) | No | Yes | Yes (cached) |
| VGGT | **No** (point clouds) | Yes | Yes | Yes | Yes (cached) |
| DepthPro | Yes (metric depth) | **No** | No | Yes | Yes (cached) |

Key implications:
- **Pose fallback**: When a provider doesn't produce poses (DAv3, DepthPro), the pipeline falls back to the loader's GT poses. This is correct — these are depth-only models.
- **Point-cloud mode**: VGGT bypasses depth entirely. The `provides_pointclouds` flag lets the pipeline know to skip depth cache construction and use the custom point extractor instead.

## Adding a New Depth Model (Checklist)

| Step | File | What to do |
|------|------|-----------|
| 1 | `depth_providers/my_model.py` | Create `MyModelProvider(DepthProvider)` |
| 2 | `depth_providers/__init__.py` | Add `"my_model": "depth_providers.my_model.MyModelProvider"` |
| 3 | *(optional)* `configs/` | Add default params if needed |

That's it. No changes to loaders, tracker, metrics, or benchmark code.

The `setup()` method handles the model inference or reads cached outputs. The rest of the interface (`load_depth`, `get_poses`, `get_intrinsics`) is fixed.

## Migration Path

1. **Phase 1**: Create `depth_providers/base.py` and `depth_providers/gt.py`. Wire into `run.py` as default — zero behavior change.

2. **Phase 2**: Move Pi3 logic from `IsaacSimLoader` + `pi3_utils.py` → `depth_providers/pi3.py`. Remove `pi3`/`dav3` flags from `IsaacSimLoader`.

3. **Phase 3**: Move VGGT logic from `vggt_utils/yolo_ssg_vggt.py` → `depth_providers/vggt.py`. Wire up point-cloud mode.

4. **Phase 4**: Implement `depth_providers/depth_anything.py` when DAv3 is ready.

## Full System Diagram (Datasets × Depth Providers)

```
                 ┌─────────────────────────────────────────┐
                 │              CLI / Config                │
                 │  --dataset X  --depth-source Y           │
                 └────────┬──────────────┬─────────────────┘
                          │              │
             ┌────────────▼──┐    ┌──────▼───────────┐
             │ DatasetLoader  │    │ DepthProvider     │
             │ (from registry)│    │ (from registry)   │
             ├───────────────┤    ├──────────────────┤
             │ • RGB paths    │    │ • depth paths     │
             │ • GT depth     │    │ • load_depth()    │
             │ • GT poses     │    │ • poses           │
             │ • GT instances │    │ • intrinsics      │
             │ • skip labels  │    │ • point_extractor │
             │ • scene_label  │    │                    │
             └───────┬───────┘    └──────┬────────────┘
                     │                   │
                     │    ┌──────────────┘
                     │    │  Provider overrides depth/poses
                     │    │  Loader provides GT + metadata
                     ▼    ▼
              ┌──────────────────┐
              │  run_tracking()  │  ← unchanged generator
              │  (core/tracker)  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  TrackedFrame    │
              └────────┬─────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
        SSG edges   Benchmark   Rerun
        (run.py)    (runner)    (vis)
```

The two registries are fully independent: any dataset × any depth source.

---

## Part 3 — Offline vs Online/Streaming Depth & Pose Modes

### 3.1 Motivation

A depth/pose provider can operate in two fundamentally different ways:

| | **Offline** | **Online / Streaming** |
|---|---|---|
| **When** | Before tracking — a separate prep step | During tracking — per-frame, inside the loop |
| **I/O** | Reads RGB → writes depth PNGs + poses file to disk | No disk artefacts; results live in memory only |
| **Latency** | Total wall-time paid up-front; tracking is fast | Each frame pays inference cost during tracking |
| **Multi-frame** | Can use any window (future + past) | Causal only — can buffer a small sliding window |
| **Reproducibility** | Easy — re-run tracking on saved files | Must re-run model to reproduce |
| **Example** | Pi3X today: `run_pi3x()` → `output_dir/` → loader | Depth-Anything v3 single-frame model called per frame |

Both modes must be supported **through the same `DepthProvider` ABC**, so
the tracker never knows (or cares) which mode produced the depth map.

---

### 3.2 Two Sub-interfaces

```
DepthProvider (ABC)                          ← unchanged contract
├── OfflineDepthProvider                     ← new
│   └── prepare(rgb_paths, output_dir)       ← batch pre-compute
│       returns (depth_paths, poses, depth_cache)
│
└── StreamingDepthProvider                   ← new
    └── predict_frame(rgb: np.ndarray,       ← single-frame or sliding-window
                      frame_idx: int,
                      rgb_path: str,
                      context: StreamCtx)
        returns FramePrediction(depth_m, T_w_c)
```

```python
# depth_providers/base.py  (extended)

from dataclasses import dataclass

@dataclass
class FramePrediction:
    depth_m: np.ndarray          # H×W float32 metres
    T_w_c: Optional[np.ndarray]  # 4×4 camera-to-world, or None


class OfflineDepthProvider(DepthProvider):
    """Run the model on ALL frames, save artefacts to disk, then return
    paths/poses so the tracker can load them normally."""

    @abstractmethod
    def prepare(
        self,
        rgb_paths: List[str],
        output_dir: str | Path,
    ) -> Tuple[List[str], Optional[List[np.ndarray]]]:
        """Batch pre-compute depth and poses.

        Returns
        -------
        depth_paths : list[str]
            One depth image per RGB frame, written to *output_dir*.
        poses : list[np.ndarray] | None
            Camera-to-world poses (same length as *depth_paths*).
            None if the model does not produce poses.
        """
        ...

    # DepthProvider contract — fulfilled from disk artefacts
    def get_depth_paths(self) -> List[str]:
        """Available after prepare()."""
        return self._depth_paths

    def build_depth_cache(self, max_cached: int = 64) -> LazyDepthCache:
        return LazyDepthCache(self.load_depth, self._depth_paths, max_cached)


class StreamingDepthProvider(DepthProvider):
    """Compute depth (and optionally pose) frame-by-frame during tracking."""

    @abstractmethod
    def predict_frame(
        self,
        rgb: np.ndarray,
        frame_idx: int,
        rgb_path: str,
    ) -> FramePrediction:
        """Return depth map (and optionally pose) for a single frame.

        Implementations may buffer a small sliding window internally
        (e.g. Pi3 VO needs ~13 frames with 5-frame overlap).
        """
        ...
```

---

### 3.3 How Streaming Fits the Existing Tracker — Zero Changes

The key insight: **`LazyDepthCache` already calls a `load_fn` on demand**.
A streaming provider only needs to supply a `load_fn` that runs inference
instead of reading a file from disk.

```python
class StreamingDepthCache:
    """Drop-in replacement for LazyDepthCache backed by a streaming model."""

    def __init__(self, provider: StreamingDepthProvider, rgb_paths: List[str]):
        self._provider = provider
        self._rgb_paths = {p: i for i, p in enumerate(rgb_paths)}
        self._depth_by_rgb: Dict[str, np.ndarray] = {}
        self._poses: Dict[int, np.ndarray] = {}

    def get(self, depth_path: str, default=None) -> np.ndarray | None:
        # depth_path == rgb_path for streaming (no separate depth files)
        rgb_path = depth_path
        if rgb_path in self._depth_by_rgb:
            return self._depth_by_rgb[rgb_path]

        frame_idx = self._rgb_paths.get(rgb_path)
        if frame_idx is None:
            return default

        rgb = np.array(Image.open(rgb_path))
        pred = self._provider.predict_frame(rgb, frame_idx, rgb_path)

        self._depth_by_rgb[rgb_path] = pred.depth_m
        if pred.T_w_c is not None:
            self._poses[frame_idx] = pred.T_w_c
        return pred.depth_m

    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        return self._poses.get(frame_idx)

    def __contains__(self, key): return key in self._rgb_paths
    def __len__(self):           return len(self._rgb_paths)
```

For poses, wrap the list lookup to also check streaming results:

```python
class StreamingPoseList:
    """Wraps a base pose list and overlays streaming predictions."""

    def __init__(self, base_poses, streaming_cache: StreamingDepthCache):
        self._base = base_poses
        self._cache = streaming_cache

    def __getitem__(self, idx):
        # Streaming pose takes priority over base (GT or None)
        p = self._cache.get_pose(idx)
        if p is not None:
            return p
        if self._base is not None and idx < len(self._base):
            return self._base[idx]
        return None

    def __len__(self):
        return len(self._base) if self._base else 0
```

**Result**: `run_tracking()` receives `(StreamingDepthCache, StreamingPoseList)`
in place of `(LazyDepthCache, list[np.ndarray])` — same duck-typed interface,
zero changes to the tracker.

---

### 3.4 Concrete Provider Examples

#### Offline — Pi3XOfflineProvider

```python
class Pi3XOfflineProvider(OfflineDepthProvider):
    """Wraps pi3_utils.run_pi3x() as a batch pre-compute step."""

    def __init__(self, model_path: str = "pi3x_model.pt"):
        self._model_path = model_path

    def prepare(self, rgb_paths, output_dir):
        from pi3_utils import run_pi3x
        run_pi3x(rgb_paths, output_dir, model_path=self._model_path)
        # Now load artefacts
        self._output_dir = Path(output_dir)
        scale = json.load(open(self._output_dir / "depth_scale.json"))
        self._depth_min = scale["global_depth_min"]
        self._depth_max = scale["global_depth_max"]
        self._depth_paths = sorted(str(p) for p in self._output_dir.glob("*.png"))
        poses = load_camera_poses(str(self._output_dir / "pi3_traj.txt"))
        return self._depth_paths, poses

    def load_depth(self, path):
        d = np.array(Image.open(path)).astype(np.float32)
        return d / 255.0 * (self._depth_max - self._depth_min) + self._depth_min

    def get_poses(self):
        return load_camera_poses(str(self._output_dir / "pi3_traj.txt"))
```

#### Streaming — DepthAnythingStreamingProvider

```python
class DAv3StreamingProvider(StreamingDepthProvider):
    """Single-frame depth model, no poses."""

    def __init__(self, model_size: str = "large"):
        from depth_anything_v3 import DepthAnythingV3
        self._model = DepthAnythingV3.from_pretrained(model_size)
        self._model.eval().cuda()

    def predict_frame(self, rgb, frame_idx, rgb_path):
        with torch.no_grad():
            depth_m = self._model.infer(rgb)  # H×W float32 metres
        return FramePrediction(depth_m=depth_m, T_w_c=None)

    # DepthProvider contract — not meaningful for streaming
    def get_depth_paths(self):  return []
    def load_depth(self, path): raise NotImplementedError
    def get_poses(self):        return None
```

#### Streaming — Pi3XStreamingProvider (sliding window)

```python
class Pi3XStreamingProvider(StreamingDepthProvider):
    """Pi3 VO in streaming mode with a causal sliding window."""

    def __init__(self, chunk_size=13, overlap=5):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._buffer: List[np.ndarray] = []
        self._buffer_paths: List[str] = []

    def predict_frame(self, rgb, frame_idx, rgb_path):
        self._buffer.append(rgb)
        self._buffer_paths.append(rgb_path)

        if len(self._buffer) < self._chunk_size:
            # Not enough frames yet — return a fast single-frame fallback
            return FramePrediction(
                depth_m=self._single_frame_fallback(rgb),
                T_w_c=None,
            )

        # Run Pi3 on the chunk
        depths, poses = self._run_pi3_chunk(self._buffer)
        result = FramePrediction(
            depth_m=depths[-1],  # latest frame's depth
            T_w_c=poses[-1],
        )

        # Slide window: keep last `overlap` frames
        self._buffer = self._buffer[-self._overlap:]
        self._buffer_paths = self._buffer_paths[-self._overlap:]
        return result
```

---

### 3.5 Integration — Unified Factory

```python
# depth_providers/factory.py

def create_provider(cfg) -> Tuple[DepthProvider, bool]:
    """Build a depth provider from config.

    Returns
    -------
    provider : DepthProvider
    is_streaming : bool
    """
    name = cfg.depth_source          # "gt", "pi3_offline", "pi3_streaming", "dav3", …
    mode = cfg.get("depth_mode", "auto")  # "offline", "streaming", "auto"

    provider = PROVIDER_REGISTRY[name](cfg)

    if mode == "auto":
        is_streaming = isinstance(provider, StreamingDepthProvider)
    else:
        is_streaming = (mode == "streaming")

    return provider, is_streaming
```

#### Usage in `run.py` / `bench.py`

```python
provider, is_streaming = create_provider(cfg)

if is_streaming:
    # No prep step — build streaming wrappers
    cache = StreamingDepthCache(provider, rgb_paths)
    poses = StreamingPoseList(gt_poses, cache)
    depth_paths = rgb_paths  # depth_path == rgb_path (placeholder keys)
else:
    if isinstance(provider, OfflineDepthProvider):
        # Run batch pre-compute once
        depth_paths, pred_poses = provider.prepare(rgb_paths, output_dir)
        cache = provider.build_depth_cache()
        poses = pred_poses if pred_poses is not None else gt_poses
    else:
        # GT or simple offline loader
        depth_paths = provider.get_depth_paths()
        cache = LazyDepthCache(provider.load_depth, depth_paths)
        poses = provider.get_poses() or gt_poses

# Tracker sees the same interface either way
for frame in run_tracking(rgb_paths, depth_paths, cache, poses, cfg, ...):
    ...
```

---

### 3.6 Config Examples

```yaml
# configs/core_tracking.yaml  (defaults)
depth_source: gt
depth_mode: auto        # "offline" | "streaming" | "auto"

# --- offline Pi3 ---
# python run.py --dataset isaacsim --depth-source pi3_offline
# → prepare() runs pi3 batch, saves to disk, then tracking uses saved files

# --- streaming DAv3 ---
# python run.py --dataset isaacsim --depth-source dav3 --depth-mode streaming
# → no prep step, model called per-frame inside tracking loop

# --- streaming Pi3 VO ---
# python run.py --dataset isaacsim --depth-source pi3_streaming
# → sliding-window Pi3 called per-frame, first few frames use fallback

# --- force offline mode for a streaming-capable model ---
# python run.py --dataset isaacsim --depth-source dav3 --depth-mode offline
# → pre-compute all depth maps, save PNGs, then load during tracking
```

---

### 3.7 Decision Matrix

| Provider | Natural mode | Can offline? | Can stream? | Produces poses? |
|---|---|---|---|---|
| GT | offline (already on disk) | ✓ | — | ✓ (from dataset) |
| Pi3X VO | offline (batch, multi-frame) | ✓ | ✓ (sliding window) | ✓ |
| DAv3 | streaming (single-frame) | ✓ (save PNGs) | ✓ | ✗ |
| VGGT | streaming (point-cloud) | ✓ (save NPZ) | ✓ | ✓ |
| ZoeDepth | streaming (single-frame) | ✓ | ✓ | ✗ |

Models that don't produce poses (DAv3, ZoeDepth) must be paired with a
separate pose source — either GT poses from the dataset loader, or a
dedicated VO/SLAM module (future work).

---

### 3.8 Comparison: Offline vs Streaming Trade-offs

```
                    OFFLINE                          STREAMING
                    ───────                          ─────────
 Prep step?        Yes (run model → save to disk)   No
 Tracking speed?   Fast (load from disk/cache)      Slower (inference per frame)
 Memory?           Disk + LRU cache                 GPU + single frame in memory
 Reproducible?     Yes (saved artefacts)             Must re-run model
 Multi-frame?      Yes (full sequence available)     Causal sliding window only
 Debug?            Easy (inspect saved PNGs)         Harder (in-memory only)
 CI-friendly?      Yes (pre-computed fixtures)       No (needs GPU at test time)
```

**Recommendation**: Use **offline for benchmarking** (reproducible, debuggable).
Use **streaming for live demos and real-time applications**.

---

### 3.9 Migration Path

| Phase | Change | Effort |
|---|---|---|
| 1 | Add `OfflineDepthProvider` / `StreamingDepthProvider` ABCs | Small |
| 2 | Implement `StreamingDepthCache` + `StreamingPoseList` | Small |
| 3 | Port Pi3X to `Pi3XOfflineProvider` (trivial, wraps existing code) | Small |
| 4 | Implement `DAv3StreamingProvider` as first streaming provider | Medium |
| 5 | Add `--depth-mode` CLI flag and `create_provider()` factory | Small |
| 6 | Implement `Pi3XStreamingProvider` with sliding window | Medium |
| 7 | (Optional) Add `OfflineAdapter` — generic wrapper that converts any streaming provider into offline by pre-running all frames | Small |

Phase 7 note — a generic `OfflineAdapter`:

```python
class OfflineAdapter(OfflineDepthProvider):
    """Wraps any StreamingDepthProvider to pre-compute all frames."""

    def __init__(self, streaming: StreamingDepthProvider):
        self._streaming = streaming

    def prepare(self, rgb_paths, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        depth_paths, poses = [], []
        for i, rp in enumerate(rgb_paths):
            rgb = np.array(Image.open(rp))
            pred = self._streaming.predict_frame(rgb, i, rp)
            dp = str(out / f"{i:06d}.npy")
            np.save(dp, pred.depth_m)
            depth_paths.append(dp)
            if pred.T_w_c is not None:
                poses.append(pred.T_w_c)
        self._depth_paths = depth_paths
        return depth_paths, poses or None

    def load_depth(self, path):
        return np.load(path)
```

This means any streaming model can be used offline too — `--depth-mode offline`
with a streaming provider automatically wraps it with `OfflineAdapter`.

---
---

# Part 4 — Live Camera Input, Temporal Consistency, and Memory-Efficient Loading

## 4.1 Three Input Sources

The pipeline needs to handle RGB from three fundamentally different sources:

```
┌─────────────────────────────────────────────────────────────┐
│                     RGB Source                              │
├─────────────────┬──────────────────┬────────────────────────┤
│  (A) Dataset    │  (B) Pre-saved   │  (C) Live Camera       │
│  on disk        │  depth on disk   │  streaming             │
│                 │                  │                        │
│  RGB + depth    │  RGB on disk,    │  RGB from camera feed, │
│  files on disk  │  depth saved by  │  no files on disk      │
│                 │  offline model   │                        │
│                 │                  │                        │
│  Loader gives   │  OfflineProvider │  Camera gives frames,  │
│  both paths     │  gives depth     │  StreamingProvider      │
│                 │  paths           │  gives depth per frame │
└────────┬────────┴────────┬─────────┴───────────┬────────────┘
         │                 │                     │
         ▼                 ▼                     ▼
    run_tracking()    run_tracking()      run_tracking_live()
    (existing)        (existing)          (new generator)
```

### Source (C): Live Camera — New Abstraction

```python
# camera_sources/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraFrame:
    """Single frame from a camera source."""
    rgb: np.ndarray            # H×W×3 uint8
    frame_idx: int
    timestamp_ns: int          # monotonic nanosecond timestamp
    camera_id: str = "cam0"    # for multi-camera setups


class CameraSource(ABC):
    """Protocol for live camera input."""

    @abstractmethod
    def open(self) -> None:
        """Initialize camera / open connection."""
        ...

    @abstractmethod
    def read(self) -> Optional[CameraFrame]:
        """Return next frame, or None if stream ended."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""
        ...

    def get_intrinsics(self):
        """Return (K, H, W) if known, else None."""
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
```

Concrete sources:

```python
# camera_sources/usb.py
class USBCamera(CameraSource):
    def __init__(self, device_id: int = 0, fps: int = 30):
        self._dev = device_id
        self._fps = fps
        self._cap = None
        self._idx = 0

    def open(self):
        self._cap = cv2.VideoCapture(self._dev)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read(self):
        ret, bgr = self._cap.read()
        if not ret:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ts = int(time.monotonic_ns())
        frame = CameraFrame(rgb=rgb, frame_idx=self._idx, timestamp_ns=ts)
        self._idx += 1
        return frame

    def close(self):
        if self._cap:
            self._cap.release()


# camera_sources/ros.py  (future)
class ROSCamera(CameraSource):
    """Subscribe to a ROS Image topic."""
    def __init__(self, topic: str = "/camera/color/image_raw"): ...


# camera_sources/realsense.py  (future)
class RealSenseCamera(CameraSource):
    """Intel RealSense — can also give hardware depth."""
    ...
```

### Live Tracking Generator

```python
# core/tracker.py  — new function alongside run_tracking()

def run_tracking_live(
    camera: CameraSource,
    depth_provider: StreamingDepthProvider,
    cfg: OmegaConf,
    object_registry: Optional[GlobalObjectRegistry] = None,
    class_names_to_track: Optional[List[str]] = None,
    point_extractor=None,
) -> Generator[TrackedFrame, None, None]:
    """Live tracking loop — RGB from camera, depth from streaming model.

    Unlike run_tracking() which takes pre-built path lists, this
    generator pulls frames from a live camera source and runs depth
    prediction on each frame in real time.
    """
    if object_registry is None:
        object_registry = GlobalObjectRegistry(...)

    model = YOLOEModel(cfg.yolo_model)
    if class_names_to_track:
        model.set_classes(class_names_to_track)

    with camera:
        while True:
            cam_frame = camera.read()
            if cam_frame is None:
                break

            # 1. YOLO detection on raw RGB
            out = model.track(
                source=[cam_frame.rgb],
                tracker=cfg.get("tracker_cfg") or yutils.TRACKER_CFG,
                device=cfg.get("device") or yutils.DEVICE,
                conf=float(cfg.conf),
                persist=True, agnostic_nms=True,
            )
            yolo_res = out[0] if isinstance(out, (list, tuple)) else out

            # 2. Depth prediction (streaming)
            pred = depth_provider.predict_frame(
                cam_frame.rgb, cam_frame.frame_idx, rgb_path=""
            )

            # 3. Same mask → 3D → tracking pipeline
            _, masks_clean = yutils.preprocess_mask(...)
            track_ids, class_names = extract_yolo_ids(yolo_res, masks_clean)
            frame_objs, graph = yutils.create_3d_objects_with_tracking(
                track_ids, masks_clean, ...,
                depth_m=pred.depth_m,
                T_w_c=pred.T_w_c,
                frame_idx=cam_frame.frame_idx,
                object_registry=object_registry,
                point_extractor=point_extractor,
            )

            yield TrackedFrame(
                frame_idx=cam_frame.frame_idx,
                rgb_path="",       # no file on disk
                depth_path="",
                frame_objs=frame_objs,
                scene_graph=graph,
                masks_clean=masks_clean,
                track_ids=track_ids,
                class_names=class_names,
                depth_m=pred.depth_m,
                T_w_c=pred.T_w_c,
                yolo_result=yolo_res,
                timings={},
            )
```

---

## 4.2 Temporal Consistency Parameters

Both Pi3X and DA3-Streaming use chunked inference with overlap for temporal
consistency. This is a common pattern across depth/pose models:

| Model | `chunk_size` | `overlap` | Extra | What it means |
|-------|-------------|-----------|-------|---------------|
| Pi3X VO | 13 | 5 | — | 13 frames per batch, 5-frame overlap for alignment between consecutive chunks |
| DA3-Streaming | configurable | configurable | `loop_enable`, `ref_view_strategy` | Same windowing + optional loop closure for long sequences |
| VGGT | ~12-16 | 4-6 | — | Similar multi-view attention window |
| UniDepth | 1 | 0 | — | Single-frame, no temporal context |

### Unified Temporal Config

These should be part of the `DepthProvider` config, not hardcoded:

```yaml
# configs/core_tracking.yaml
depth_source: gt
depth_mode: auto

# Temporal consistency params — override per provider
depth_provider:
  chunk_size: 13       # frames per inference batch
  overlap: 5           # overlap between consecutive chunks
  loop_closure: false  # DA3-style loop detection
  ref_view_strategy: "first_last"  # reference view selection
  conf_threshold: 1.5  # confidence threshold for alignment
```

### Provider Access to Temporal Params

```python
class DepthProvider(ABC):
    """Base class — extended with temporal-awareness."""

    def configure(self, cfg: OmegaConf):
        """Pull temporal params from config with model-specific defaults."""
        dp_cfg = cfg.get("depth_provider", {})
        self._chunk_size = int(dp_cfg.get("chunk_size", self.DEFAULT_CHUNK_SIZE))
        self._overlap = int(dp_cfg.get("overlap", self.DEFAULT_OVERLAP))

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap

    # Each provider declares its own defaults
    DEFAULT_CHUNK_SIZE: int = 1    # single-frame by default
    DEFAULT_OVERLAP: int = 0
```

```python
class Pi3XStreamingProvider(StreamingDepthProvider):
    DEFAULT_CHUNK_SIZE = 13
    DEFAULT_OVERLAP = 5

class DA3StreamingProvider(StreamingDepthProvider):
    DEFAULT_CHUNK_SIZE = 20     # from DA3 base_config.yaml
    DEFAULT_OVERLAP = 5
```

### Temporal Buffer for Streaming Providers

The sliding window buffer from Part 3 should be generalized:

```python
class TemporalBuffer:
    """Sliding window buffer for multi-frame depth models.

    Accumulates frames until chunk_size is reached, then yields
    predictions for the non-overlap portion. Keeps the last
    `overlap` frames for the next chunk.
    """

    def __init__(self, chunk_size: int, overlap: int):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._frames: List[np.ndarray] = []
        self._indices: List[int] = []
        self._ready = False

    def push(self, rgb: np.ndarray, frame_idx: int) -> bool:
        """Add a frame. Returns True when chunk is ready for inference."""
        self._frames.append(rgb)
        self._indices.append(frame_idx)
        self._ready = len(self._frames) >= self._chunk_size
        return self._ready

    def get_chunk(self) -> Tuple[List[np.ndarray], List[int]]:
        """Return the current chunk of frames and their indices."""
        return self._frames.copy(), self._indices.copy()

    def slide(self):
        """Slide the window: keep last `overlap` frames."""
        self._frames = self._frames[-self._overlap:]
        self._indices = self._indices[-self._overlap:]
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def pending_count(self) -> int:
        return len(self._frames)
```

Usage inside a provider:

```python
class Pi3XStreamingProvider(StreamingDepthProvider):

    def __init__(self, cfg):
        super().__init__()
        self.configure(cfg)
        self._buffer = TemporalBuffer(self._chunk_size, self._overlap)
        self._model = Pi3XVO(Pi3X.from_pretrained(...).eval().cuda())
        self._pending_results: Dict[int, FramePrediction] = {}

    def predict_frame(self, rgb, frame_idx, rgb_path):
        # Check if we already have a result (from previous chunk overlap)
        if frame_idx in self._pending_results:
            return self._pending_results.pop(frame_idx)

        if self._buffer.push(rgb, frame_idx):
            # Full chunk — run inference
            frames, indices = self._buffer.get_chunk()
            depths, poses = self._run_pi3_chunk(frames)

            # Cache results for all frames in this chunk
            for i, idx in enumerate(indices):
                self._pending_results[idx] = FramePrediction(
                    depth_m=depths[i], T_w_c=poses[i]
                )

            self._buffer.slide()
            return self._pending_results.pop(frame_idx)

        # Not enough frames yet — use lightweight single-frame fallback
        return self._single_frame_fallback(rgb)
```

---

## 4.3 Memory-Efficient Loading for Large Datasets

### Current Status

| What | Current Approach | Memory Impact |
|------|-----------------|---------------|
| **RGB images** | Lazy — `cv2.imread()` per frame in `run_yolo_tracking_stream` | **Good** — only 1 frame in memory |
| **Depth maps** | `LazyDepthCache` with LRU eviction (`max_cached=64`) | **Good** — at most 64 depth maps |
| **Poses** | All loaded into a `List[np.ndarray]` upfront | **OK** — 2K poses × 128 bytes ≈ 256 KB |
| **RGB path list** | All paths in memory as `List[str]` | **OK** — 2K strings ≈ negligible |
| **YOLO model** | Loaded once, persistent tracker state | **Fixed** — ~200-800 MB GPU |

**Good news**: The pipeline is already mostly lazy for the heavy data (RGB, depth). The main risk areas for 1-2K+ frame datasets are:

### Risk 1: Offline Depth Provider Pre-Compute

When an `OfflineDepthProvider.prepare()` runs on 2K frames, the model may
try to load all images into a single tensor (as Pi3X does: `load_images_as_tensor()`).

**Solution**: Providers should process in chunks internally, flushing to disk between chunks.

```python
class Pi3XOfflineProvider(OfflineDepthProvider):

    def prepare(self, rgb_paths, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_depth_paths = []
        all_poses = []

        # Process in chunks to avoid OOM on large sequences
        step = self._chunk_size - self._overlap
        n = len(rgb_paths)

        for chunk_start in range(0, n, step):
            chunk_end = min(chunk_start + self._chunk_size, n)
            chunk_rgb = rgb_paths[chunk_start:chunk_end]

            # Load only this chunk into GPU
            imgs = load_images_as_tensor_from_paths(chunk_rgb).to(self._device)

            with torch.no_grad():
                results = self._pipe(
                    imgs=imgs[None],
                    chunk_size=len(chunk_rgb),  # single chunk
                    overlap=0,
                )

            # Extract and save immediately — free GPU memory
            depths = self._extract_depths(results)
            poses = results['camera_poses'][0].cpu().numpy()

            for i, global_idx in enumerate(range(chunk_start, chunk_end)):
                dp = str(out / f"{global_idx:06d}.npy")
                np.save(dp, depths[i])
                all_depth_paths.append(dp)

            all_poses.extend(poses)

            del imgs, results
            torch.cuda.empty_cache()

        # Align chunks (overlap-based Sim3 alignment)
        all_poses = self._align_chunks(all_poses, step, self._overlap)

        self._depth_paths = all_depth_paths
        return all_depth_paths, all_poses
```

### Risk 2: DA3-Streaming Long Sequences

DA3-Streaming already handles this — it processes in chunks, saves
intermediate `.npy` files to `_tmp_results_unaligned/`, and then does
alignment as a second pass. This pattern should be the standard for all
offline providers processing long sequences.

### Risk 3: Ground Truth Depth for Large Datasets

Datasets with 2K+ depth maps as 16-bit PNGs can be 5-20 MB each.
`LazyDepthCache` already handles this with LRU eviction, but the
`max_cached` parameter should be tunable:

```yaml
# configs/core_tracking.yaml
max_depth_cached: 64      # default LRU cache size

# For memory-constrained systems:
# max_depth_cached: 16

# For fast SSDs with small depth maps:
# max_depth_cached: 256
```

### Risk 4: RGB for Streaming Depth Providers

When a streaming depth provider needs multi-frame context
(chunk of RGB frames), it holds a `TemporalBuffer` with up to
`chunk_size` frames in RAM. For HD frames:

- 1920×1080×3 = ~6 MB per frame
- 13 frames = ~78 MB (Pi3X)
- 20 frames = ~120 MB (DA3)

This is acceptable on most systems. For 4K or very long chunks,
the buffer could optionally store file paths and reload on demand,
but this is premature optimisation — not needed now.

### Summary: Memory Budget per Component

```
Component              Memory (2K-frame HD dataset)
─────────────────────  ──────────────────────────────
RGB (single frame)     ~6 MB
LazyDepthCache (64)    ~64 × 6 MB = ~384 MB
Poses (all in RAM)     ~256 KB
YOLO model (GPU)       ~200–800 MB
Depth model (GPU)      ~500–2000 MB  (if streaming)
TemporalBuffer         ~13 × 6 = ~78 MB  (Pi3X)
Object registry        Variable, grows with objects
─────────────────────  ──────────────────────────────
Total estimate         ~1–3.5 GB  (manageable)
```

---

## 4.4 Unified Architecture Diagram (All Parts Combined)

```
                    ┌──────────────┐     ┌────────────────┐
                    │ CameraSource │     │ DatasetLoader   │
                    │ (live feed)  │     │ (files on disk) │
                    └──────┬───────┘     └───────┬─────────┘
                           │                     │
                     CameraFrame            rgb_paths,
                     (rgb + ts)             depth_paths,
                           │                gt_poses
                           │                     │
            ┌──────────────┴─────────────────────┤
            │                                    │
            ▼                                    ▼
  ┌───────────────────┐              ┌───────────────────────┐
  │ StreamingProvider  │              │ OfflineProvider        │
  │ (per-frame infer) │              │ (batch pre-compute)   │
  │                   │              │                       │
  │ ┌───────────────┐ │              │  prepare(rgb_paths)   │
  │ │TemporalBuffer │ │              │  → depth PNGs + poses │
  │ │ chunk_size=13 │ │              │  → disk artefacts     │
  │ │ overlap=5     │ │              │                       │
  │ └───────────────┘ │              └───────────┬───────────┘
  │                   │                          │
  │ predict_frame()   │                          │
  │ → FramePrediction │                          │
  └────────┬──────────┘                          │
           │                                     │
           ▼                                     ▼
  ┌──────────────────┐              ┌────────────────────────┐
  │StreamingDepthCache│              │ LazyDepthCache (LRU)   │
  │StreamingPoseList  │              │ List[poses]            │
  └────────┬──────────┘              └───────────┬────────────┘
           │                                     │
           │  same duck-typed interface          │
           └─────────────┬───────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   run_tracking()     │  ← unchanged generator
              │   or                 │
              │   run_tracking_live()│  ← new for camera input
              └──────────┬───────────┘
                         │
                     TrackedFrame
                         │
           ┌─────────────┼──────────────┐
           ▼             ▼              ▼
      SSG edges     Benchmark      Rerun (vis)
      (run.py)      (runner.py)    (rerun_utils)
```

---

## 4.5 Config Examples (All Modes)

```yaml
# --- Mode A: Dataset files, GT depth (default, same as today) ---
# python run.py --dataset isaacsim /path/to/scene
depth_source: gt
depth_mode: auto

# --- Mode B: Dataset files, offline Pi3X depth ---
# python run.py --dataset isaacsim --depth-source pi3_offline /path/to/scene
depth_source: pi3_offline
depth_mode: offline
depth_provider:
  chunk_size: 13
  overlap: 5

# --- Mode C: Dataset files, streaming DA3 depth ---
# python run.py --dataset isaacsim --depth-source dav3 --depth-mode streaming
depth_source: dav3
depth_mode: streaming
depth_provider:
  chunk_size: 20
  overlap: 5
  loop_closure: false

# --- Mode D: Live camera, streaming depth ---
# python run.py --camera usb:0 --depth-source dav3
camera:
  source: usb
  device_id: 0
  fps: 30
depth_source: dav3
depth_mode: streaming

# --- Mode E: Live camera, streaming Pi3X with temporal buffer ---
# python run.py --camera realsense --depth-source pi3_streaming
camera:
  source: realsense
  serial: "12345678"
depth_source: pi3_streaming
depth_provider:
  chunk_size: 13
  overlap: 5

# --- Memory tuning ---
max_depth_cached: 64        # LRU cache size for LazyDepthCache
```

---

## 4.6 CLI Extension for Live Camera

```bash
# Dataset mode (existing)
python run.py --dataset isaacsim /path/to/scene
python run.py --dataset thud_real /path/to/scene --depth-source pi3_offline

# Live camera mode (new)
python run.py --camera usb:0 --depth-source dav3
python run.py --camera realsense --depth-source pi3_streaming
python run.py --camera ros:/camera/image_raw --depth-source dav3

# Benchmark mode (dataset only — live camera doesn't make sense for benchmarks)
python benchmark/benchmark.py --dataset isaacsim /path/to/scene --depth-source pi3_offline
```

---

## 4.7 Migration Path (Part 4 Additions)

| Phase | Change | Effort |
|---|---|---|
| 8 | Add `TemporalBuffer` utility class | Small |
| 9 | Integrate `TemporalBuffer` into Pi3X and DA3 streaming providers | Medium |
| 10 | Add `depth_provider:` config section with temporal params | Small |
| 11 | Add `max_depth_cached` to config, thread through to `LazyDepthCache` | Small |
| 12 | Create `camera_sources/` module with `CameraSource` ABC + `USBCamera` | Medium |
| 13 | Add `run_tracking_live()` generator to `core/tracker.py` | Medium |
| 14 | Add `--camera` CLI flag, wire into `run.py` | Small |
| 15 | (Future) Add `RealSenseCamera`, `ROSCamera` sources | Per-device |
| 16 | (Future) Chunk-based offline processing for Pi3X `prepare()` to avoid OOM on 2K+ frame datasets | Medium |
