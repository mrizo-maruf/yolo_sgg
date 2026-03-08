# YOLO-SGG Architecture Proposal (Tracking + Scene Graph + Benchmarking)

## Why refactor now

Current pipeline logic is concentrated in single scripts (`yolo_ssg.py`, `benchmark_thud.py`, `benchmark_tracking.py`) and duplicates dataflow across runtime and benchmarking code. Tracking setup is also hard-coded (`TRACKER_CFG = "botsort.yaml"`) and scene-graph relation prediction is called as one fixed function (`edges(...)`).

This proposal introduces a modular architecture with plugin-like components for:
- YOLO model variants (prompted / non-prompted)
- tracker backends (BoT-SORT, ByteTrack, others)
- edge predictors (SceneVerse rules, learned models)
- depth/reconstruction providers (RGB-D, Depth Anything v3, Pi3 family)
- unified benchmarking/metrics across THUD, CODa, IsaacSim

---

## Observed current state (summary)

- Main runtime script (`yolo_ssg.py`) performs end-to-end execution in one function: model inference, mask preprocessing, 3D construction, edge prediction, graph merge, and visualization.
- Tracking stream is created in `YOLOE.utils.track_objects_in_video_stream(...)` and currently binds to `TRACKER_CFG` global.
- Scene-graph edge generation is invoked through `ssg.ssg_main.edges(...)` directly.
- Benchmarking exists in separate scripts (`benchmark_thud.py`, `benchmark_tracking.py`) with overlapping evaluation/visualization logic.
- Metrics utilities (`metrics/metrics_utils.py`) exist but are not the unified entrypoint for all datasets.

---

## Target architecture

## 1) High-level layered design

```text
+--------------------------------------------------------------+
|                         CLI / Apps                           |
|   run.py            benchmark.py            visualize.py      |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                     Orchestration Layer                      |
| TrackingSceneGraphPipeline  |  BenchmarkRunner              |
| ConfigResolver              |  ExperimentRegistry           |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                       Domain Services                        |
| DetectorService  TrackerService  ReconstructionService       |
| SceneGraphService  GraphMergeService  VisualizationService   |
| MetricsService                                               |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                     Plugin / Adapter Layer                   |
| Detectors: YOLOEPrompt, YOLOENoPrompt, ...                  |
| Trackers: BotSortAdapter, ByteTrackAdapter, ...             |
| EdgePredictors: SceneVerseRules, GNNPredictor, ...          |
| Depth: RGBDDepth, DepthAnythingV3, Pi3Depth                 |
| DatasetAdapters: THUD, CODa, IsaacSim                       |
| VizAdapters: Rerun, Open3D, Matplotlib                      |
+-----------------------------+--------------------------------+
                              |
+-----------------------------v--------------------------------+
|                         Data Layer                           |
| FramePacket / TrackState / Object3D / SceneGraphSnapshot    |
| Event logs / metrics JSON / artifacts                        |
+--------------------------------------------------------------+
```

## 2) Core interfaces (suggested)

```python
class Detector(Protocol):
    def infer(self, frame: FramePacket) -> DetectionBatch: ...

class Tracker(Protocol):
    def update(self, detections: DetectionBatch, frame_idx: int) -> TrackBatch: ...

class Reconstruction(Protocol):
    def build_objects(self, frame: FramePacket, tracks: TrackBatch) -> list[Object3D]: ...

class EdgePredictor(Protocol):
    def predict(self, graph: nx.MultiDiGraph, objects: list[Object3D], context: FrameContext) -> None: ...

class DatasetAdapter(Protocol):
    def iter_frames(self) -> Iterator[FramePacket]: ...
    def load_gt(self) -> GroundTruthBundle: ...

class MetricsEvaluator(Protocol):
    def evaluate(self, pred: PredictionBundle, gt: GroundTruthBundle) -> MetricReport: ...
```

## 3) Proposed package structure

```text
yolo_sgg/
  apps/
    run.py
    benchmark.py
    visualize.py
  core/
    config/
      schema.py
      resolver.py
    pipeline/
      tracking_sg_pipeline.py
      benchmark_pipeline.py
    domain/
      models.py            # FramePacket, Object3D, SceneGraphSnapshot...
      events.py
  plugins/
    detectors/
      yoloe_prompt.py
      yoloe_no_prompt.py
    trackers/
      botsort.py
      bytetrack.py
    reconstruction/
      rgbd_reconstruction.py
      depth_anything_v3.py
      pi3_reconstruction.py
    edge_predictors/
      sceneverse_rules.py
      learned_gnn.py
    datasets/
      thud_adapter.py
      coda_adapter.py
      isaacsim_adapter.py
    visualization/
      rerun_adapter.py
      open3d_adapter.py
  services/
    graph_merge_service.py
    metrics_service.py
    artifact_logger.py
```

---

## Key class suggestions

- `TrackingSceneGraphPipeline`  
  Orchestrates per-frame flow: detect -> track -> reconstruct -> edge predict -> merge -> emit snapshot.

- `PipelineFactory`  
  Reads config and wires chosen plugins (detector/tracker/edge predictor/reconstruction).

- `GraphMergeService`  
  Encapsulates persistent graph update policy (ID matching, edge conflict rules).

- `BenchmarkRunner`  
  Runs experiments over datasets/scenes, captures outputs to a common result schema.

- `UnifiedMetricsService`  
  Computes common metrics (MOTA/MOTP/IDF1/HOTA + SG metrics) regardless of source dataset.

- `RerunVisualizationService`  
  Consumes pipeline events and publishes streams (tracks, bboxes, relations, metrics over time).

- `ExperimentConfig` (dataclass + OmegaConf schema)
  Single config tree for runtime and benchmarking.

---

## Flow diagrams for draw.io

## A) Runtime sequence

```text
DatasetAdapter -> Pipeline: next FramePacket
Pipeline -> Detector: infer(frame)
Detector -> Pipeline: DetectionBatch
Pipeline -> Tracker: update(detections)
Tracker -> Pipeline: TrackBatch
Pipeline -> Reconstruction: build_objects(frame, tracks)
Reconstruction -> Pipeline: [Object3D]
Pipeline -> SceneGraphService: build current graph
Pipeline -> EdgePredictor: predict(graph, objects, context)
Pipeline -> GraphMergeService: merge(persistent, current)
Pipeline -> VisualizationService: publish(snapshot)
Pipeline -> ArtifactLogger: save frame outputs
```

## B) Benchmark sequence

```text
BenchmarkRunner -> DatasetAdapter: iter scenes
BenchmarkRunner -> TrackingSceneGraphPipeline: run(scene)
Pipeline -> ArtifactLogger: prediction bundle
BenchmarkRunner -> DatasetAdapter: load_gt(scene)
BenchmarkRunner -> UnifiedMetricsService: evaluate(pred, gt)
UnifiedMetricsService -> BenchmarkRunner: MetricReport
BenchmarkRunner -> ResultsStore: aggregate + leaderboard
```

---

## Unified configuration example

```yaml
app:
  mode: run            # run | benchmark | visualize
  seed: 42

pipeline:
  detector: yoloe_prompt
  tracker: botsort
  reconstruction: rgbd
  edge_predictor: sceneverse_rules

plugins:
  detector:
    model_path: weights/yoloe-11l-seg-pf.pt
    conf: 0.25
    iou: 0.5
    prompts: []
  tracker:
    config_file: cfg/trackers/botsort.yaml
  reconstruction:
    max_points_per_obj: 2000
    source: rgbd         # rgbd | depth_anything_v3 | pi3
  edge_predictor:
    proximity_threshold: 0.3

dataset:
  name: isaacsim         # thud | coda | isaacsim
  scene: scene_7

benchmark:
  metrics:
    tracking: [MOTA, MOTP, IDF1, HOTA]
    scene_graph: [edge_precision, edge_recall, edge_f1]
```

---

## Migration roadmap (incremental)

1. **Create `apps/run.py` as thin wrapper** over existing `yolo_ssg.main` behavior (no logic change).
2. **Extract services** from `yolo_ssg.py`: tracking, reconstruction, edge prediction, merge.
3. **Introduce plugin interfaces** and keep current implementations as defaults.
4. **Unify benchmarks** by moving THUD/Isaac/CODa adapters under one `BenchmarkRunner`.
5. **Unify metrics schema + reports** with one `UnifiedMetricsService` output format.
6. **Add rerun event bus** for live visualization and offline replay.
7. **Add monocular depth adapters** (`DepthAnythingV3`, `Pi3`) behind `Reconstruction` interface.
8. **Deprecate direct GT/depth camera dependence** in runtime path, keep GT only in benchmark path.

---

## Design decisions to align with your goals

- Keep **core tracker + SSG logic** in reusable services.
- Support **switchable edge predictors** by interface rather than branching in one file.
- Support **YOLO model variants** and tracker configs via config-driven factory.
- Make benchmarking a **first-class app** with the same runtime pipeline.
- Build metrics once, use everywhere (THUD/CODa/IsaacSim).
- Prepare for future **reconstruction-method migration** without touching tracking/SSG core.

