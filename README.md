# YOLO-SGG — 3D Scene Graph Generation

Builds persistent **3D scene graphs** from RGB-D or monocular video by tracking
objects across frames and predicting spatial relationships between them.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

![pipeline](/assets/image2.jpeg)

---

## Quick start

```bash
conda create -n yolo_sgg python=3.10 -y && conda activate yolo_sgg
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

```bash
# Run the pipeline on one scene, with visualisation and graph export
python new_run.py --scene_path /path/to/scene_1 --rerun --vis_graph --save_graph

# Evaluate tracking against ground truth
python -m benchmark.benchmark --dataset isaacsim --scene_path /path/to/scene_1
```

Optional extras (VL-SAT edge predictor, `pytorch3d`) are listed and commented in
[requirements.txt](requirements.txt).

---

## Pipeline

Per frame, [`run_tracking()`](core/new_tracker.py) is a generator that yields a
`TrackedFrame`; [`new_run.py`](new_run.py) feeds each one into the scene graph.

| # | Stage | Where |
|---|-------|-------|
| 1 | **2D detection + tracking** — YOLOE instance segmentation with BoT-SORT track IDs | [core/new_yolo_runner.py](core/new_yolo_runner.py) |
| 2 | **Mask preprocessing** — morphological erosion, then class filtering via skip lists | [core/mask_utils.py](core/mask_utils.py) |
| 3 | **Depth + 3D lifting** — fetch depth from the configured provider, project each masked region into a world-frame point cloud | [depth_providers/](depth_providers/) |
| 4 | **Cascaded global matching** — assign a persistent global ID per detection (see below) | [core/tracker_utils.py](core/tracker_utils.py) |
| 5 | **Registry update** — accumulate points, recompute bboxes, voxel-downsample, compute frustum visibility | [core/object_registry.py](core/object_registry.py) |
| 6 | **Reprojection recovery** — re-emit registry objects inside the frustum that YOLO missed this frame | [core/new_tracker.py](core/new_tracker.py) |
| 7 | **Periodic dedup** — merge registry objects whose bboxes converged onto the same physical thing | [core/object_registry.py](core/object_registry.py) |
| 8 | **Scene graph** — build a local per-frame graph, merge it into the global `nx.MultiDiGraph` | [scene_graph/graph.py](scene_graph/graph.py) |

### Cascaded global matching

Each detection's cleaned point cloud and 3D bbox are matched against the registry
by a four-level cascade — **first hit wins**, a miss creates a new object. All
levels above 0 score candidates with
[`compute_overlap_score`](core/tracker_utils.py) (centroid-distance gate →
size-similarity gate → `IoU ≥ overlap_th` or `containment ≥ 0.3`).

| Level | Strategy | Recovers from |
|-------|----------|---------------|
| **0** | Trust the BoT-SORT track ID outright, no 3D check | — (pure-2D baseline; skips depth entirely) |
| **1** | Same ID, but spatially verified in 3D | BoT-SORT ID switches |
| **2** | Best overlap against the previous frame, ignoring the ID | new/changed track IDs on recently seen objects |
| **3** | Best overlap across the whole registry, class-gated and distance-relaxed | occlusion, objects out of view for many frames |

After a match the registry rebinds `yolo_to_global[tid]`, so a correction made at
L2/L3 lets the cheap L1 path succeed on later frames. `--level N` truncates the
cascade for ablations, and each run prints a per-level hit histogram.

### Relationship types

- **Egocentric** (camera-relative: proximity, `middle_furniture`) — recomputed
  every frame.
- **Allocentric** (object-relative: support, hanging, embedded, alignment,
  `oppo_support`) — persist across frames; conflicts are resolved by dropping
  the older edge.

Three predictors are available in [scene_graph/predictors/](scene_graph/predictors/)
and selected with `--edges`: `sv` (geometric/SceneVerse), `bs` (egocentric
baseline), `vlsat` (neural VL-SAT), or `all`.

---

## Datasets & depth

Loaders are registered in [data_loaders/registry.py](data_loaders/registry.py)
and selected with `--dataset`:

`isaacsim` (default) · `thud_synthetic` · `coda` · `scanetpp` · `r3scan` · `habitat`

Depth sources come from [depth_providers/factory.py](depth_providers/factory.py),
selected with `--depth_provider`:

| Provider | Description |
|----------|-------------|
| `gt` | ground-truth depth PNGs from the scene directory (default) |
| `pi3_online` | Pi3X streaming inference with a background feeder thread |
| `pi3_offline` | precomputed Pi3 depth + Sim(3) alignment JSON |
| `dav3_online` / `dav3_offline` | DepthAnything V3 metric depth |

Expected on-disk layout for each dataset is documented in
[datasets_structure.md](datasets_structure.md).

---

## Configuration

Config is OmegaConf, merged in this order — later wins:

```
configs/core_tracking.yaml  →  configs/{dataset}.yaml  →  CLI flags
```

[configs/core_tracking.yaml](configs/core_tracking.yaml) holds every default and
documents each option inline. The knobs that matter most:

| Key | Default | Effect |
|-----|---------|--------|
| `yolo_model`, `conf`, `iou` | `yoloe-11l-seg-pf.pt`, 0.25, 0.5 | detector and thresholds |
| `kernel_size` | 13 | mask erosion strength (too high erases small objects) |
| `registry_voxel_size` | 0.04 | point-cloud compression; lower for small/detailed objects |
| `tracking_overlap_threshold` / `tracking_distance_threshold` | 0.1 / 1.0 | matching cascade gates |
| `merge_every_n_frames` | 1 | duplicate-object merge cadence; 0 disables |
| `matching_level` | 3 | cascade depth (ablation) |
| `max_depth_cached` | 64 | depth LRU size; use >0 for sequences over ~200 frames |
| `skip_classes` | wall, floor, … | structural classes excluded from the graph |

---

## CLI reference

**[`new_run.py`](new_run.py)** — run the pipeline on a scene.

| Flag | Purpose |
|------|---------|
| `--scene_path PATH` | scene directory (required) |
| `--dataset NAME` | loader to use (default `isaacsim`) |
| `--depth_provider NAME` | override the config's depth source |
| `--yolo_model PATH`, `--is_open_vocab` | model overrides |
| `--edges all\|bs\|sv\|vlsat` | which edge predictors to run (comma-separated combos allowed) |
| `--level 0..3` | truncate the matching cascade |
| `--rerun`, `--vis_edge`, `--vis_graph` | Rerun 3D/2D views, edge overlay, per-frame graph plot |
| `--save_graph`, `--save_global_graph` | export graphs as JSON |
| `--print_resources` | per-frame latency and GPU memory |
| `--debug_sync`, `--debug_gt` | print RGB/depth/pose alignment; add GT panels to Rerun |

**[`benchmark/benchmark.py`](benchmark/benchmark.py)** — evaluate tracking against
ground truth. Shares `build_default_registry()` and `run_tracking()` with
`new_run.py`, so results reflect the real pipeline.

```bash
# Single scene
python -m benchmark.benchmark --dataset isaacsim --scene_path /data/scene_1

# Every scene under a dataset root
python -m benchmark.benchmark --dataset isaacsim --scene_path /data --multi
```

Adds `--match_mode mask2d|bbox2d|bbox3d`, `--rerun` (four debug 2D views per
frame), `--vis`/`--vis_interval`/`--vis_save`/`--no_show`, and `--output_dir`.
Requires a loader with `get_gt_instances()` support.

---

## Outputs

- **Scene graphs** — local and global `nx.MultiDiGraph` as JSON in
  `results/scene_graphs/`, alongside `tracked_objects.json`
- **Metrics & plots** — timestamped run directories under
  `results/benchmark_metrics/`
- **Visualisation** — live Rerun 3D point clouds, bboxes and edges; 2D graph plots

---

## Documentation

| Doc | Covers |
|-----|--------|
| [datasets_structure.md](datasets_structure.md) | expected on-disk layout per dataset |
| [rerun.md](rerun.md) · [run.md](run.md) | visualisation setup and example invocations |
| [scene_graph/edge.md](scene_graph/edge.md) | how edges are stored and directed |
| [per_frame_graphs.md](per_frame_graphs.md) | generating per-frame graphs |
| [depth_providers/readme.md](depth_providers/readme.md) | async Pi3 architecture |
| [metrics/METRICS_README.md](metrics/METRICS_README.md) | tracking metric definitions |
| [benchmark/reame.md](benchmark/reame.md) · [assets/BENCHMARK_ARCHITECTURE.md](assets/BENCHMARK_ARCHITECTURE.md) | benchmark usage and design |
| [3RScan_3DSSG_loader.md](3RScan_3DSSG_loader.md) | 3RScan + 3DSSG loader |

---

## Project structure

```
yolo_sgg/
├── new_run.py              # Main entry point
├── configs/                # core_tracking.yaml + one YAML per dataset
├── core/
│   ├── new_tracker.py      # run_tracking() generator — the central loop
│   ├── new_yolo_runner.py  # YOLOE + BoT-SORT streaming
│   ├── object_registry.py  # GlobalObjectRegistry (persistent object store)
│   ├── tracker_utils.py    # matching cascade
│   ├── mask_utils.py       # mask preprocessing / class filtering
│   └── geometry.py         # point-cloud cleaning, bbox fitting
├── data_loaders/           # isaacsim, thud_synthetic, coda, scanetpp, r3scan, habitat
├── depth_providers/        # gt, pi3_online/offline, dav3_online/offline
├── scene_graph/
│   ├── graph.py            # SceneGraph — local→global merge, edge reconciliation
│   └── predictors/         # Basic (SceneVerse), Baseline, VL-SAT
├── benchmark/              # evaluation harness + debug visualisation
├── metrics/                # tracking metric implementations
└── tools/                  # data collection, Sim(3) fitting, offline visualisers
```

---

## Roadmap

- [ ] video visualisation
- [ ] faster SceneVerse edge predictor
- [ ] prompt-conditioned detection model
- [ ] richer graph-update logic
