# Depth Providers — Async Pi3 Architecture

## Overview

The 3D tracking pipeline needs **depth + pose** for every RGB frame.
When ground-truth depth exists on disk, reads are instant.
When depth comes from a neural model like **Pi3**, inference costs ~500 ms
per window — a bottleneck that the async architecture removes by overlapping
Pi3 with YOLO + 3D tracking.

---

## Pipeline Components

```
┌────────────┐     ┌────────────┐     ┌──────────────┐     ┌────────────────┐
│  RGB source│────▶│ YOLO (100ms│────▶│ 3D tracking  │────▶│ scene graph /  │
│  (loader)  │     │  per frame)│     │ + registry   │     │ visualization  │
└─────┬──────┘     └────────────┘     └──────┬───────┘     └────────────────┘
      │                                      │
      │  feed_frame(idx, rgb)                │  get_depth(idx)
      ▼                                      │  get_pose(idx)
┌─────────────────────────────────────────────┘
│         Pi3OnlineDepthProvider
│  ┌──────────┐    ┌───────────────┐    ┌─────────────┐
│  │  buffer   │──▶│ Pi3 inference │──▶│ depth/pose  │
│  │  (queue)  │   │  (~500ms/win) │   │   cache     │
│  └──────────┘    └───────────────┘    └─────────────┘
└──────────────────────────────────────────────────────
```

### Call chain in the main loop

```python
# new_yolo_runner.py — per frame:
rgb, path = loader.get_rgb(idx)          # ← calls feed_frame(fnum, rgb) internally
result = yolo_model.track(rgb)           # ~100 ms

# new_tracker.py — per frame:
T_w_c  = loader.get_pose(idx)           # ← reads from Pi3 cache (or GT)
depth  = loader.get_depth(idx)          # ← reads from Pi3 cache (or GT)
pcds   = depth_provider.get_masked_pcds(...)
registry.update(...)
```

`loader.get_rgb()` is the injection point: the dataset loader reads the
image **and** calls `depth_provider.feed_frame(fnum, rgb)`, which hands the
frame to Pi3.

---

## Synchronous Mode (`async_inference=False`)

Everything runs on the **main thread**, sequentially:

```
Frame 0: get_rgb ──▶ feed_frame ──────────────────────────────────────
Frame 1: get_rgb ──▶ feed_frame ──────────────────────────────────────
  ...
Frame 12: get_rgb ─▶ feed_frame ─▶ [_process_window: 500ms BLOCKS] ──
Frame 12: YOLO (100ms) ──▶ get_depth(12) ──▶ tracking ───────────────
Frame 13: get_rgb ──▶ feed_frame ──────────────────────────────────────
  ...
```

**Problem:** Every `window_size` frames (default 13), the main thread
freezes for ~500 ms while Pi3 runs.  With `window_size=13` and
`overlap=5`, Pi3 produces 8 new depth maps per window:

| Component | Time |
|-----------|------|
| Pi3 (amortized per frame) | 500 / 8 = **62 ms** |
| YOLO per frame | **100 ms** |
| **Total per frame** | **~162 ms** (but bursty — 500 ms stall every 8 frames) |

---

## Asynchronous Mode (`async_inference=True`, the default)

Pi3 inference runs in a **background thread**.  The main thread never
blocks on Pi3 unless it asks for a depth map that hasn't been computed
yet.

### Architecture

```
MAIN THREAD                              BACKGROUND THREAD (pi3-worker)
─────────────────                        ──────────────────────────────
                                         _async_worker() loop:
get_rgb(0):                                │
  feed_frame(0, rgb)                       │
    → queue.put((0, rgb))  ──────────▶     queue.get() → buffer[0] = rgb
YOLO(frame 0)  ~100ms                     │
                                           │
get_rgb(1):                                │
  feed_frame(1, rgb)                       │
    → queue.put((1, rgb))  ──────────▶     queue.get() → buffer[1] = rgb
YOLO(frame 1)  ~100ms                     │
  ...                                      ...
                                           │
get_rgb(12):                               │
  feed_frame(12, rgb)                      │
    → queue.put((12, rgb)) ──────────▶     queue.get() → buffer[12] = rgb
                                           len(buffer) == 13 == window_size
                                           → _process_window()  [500ms,
                                              runs on GPU in this thread]
YOLO(frame 12) ~100ms                     │  ... Pi3 running ...
get_depth(12):                             │  ... Pi3 running ...
  event[12] not set yet                    │  ... Pi3 running ...
  → ev.wait()  ◀─  BLOCKS ~300ms  ────────│  ... Pi3 running ...
                                           │  done! cache depth[0..12]
                                           │  event[0..12].set() ─────▶  unblocked!
  depth = cache[12]  ✓                     │
tracking(frame 12)                         │
                                           │
get_rgb(13):                               │
  feed_frame(13, rgb)                      │
    → queue.put()  ───────────────────▶    queue.get() → buffer = overlap + frame 13
YOLO(frame 13) ~100ms                     │ (accumulating next window)
get_depth(13):                             │
  not cached, ev.wait()                    │
  ... (Pi3 not triggered yet,              │
       only 6 frames in buffer)            │
                                           │
  ...frames 14-20 accumulate...            buffer full → _process_window() [500ms]
                                           │
YOLO(frame 20) ~100ms                     │  ... Pi3 running in parallel ...
get_depth(20):                             │  done! event[13..20].set()
  ev.wait() → ready ✓                     │
```

### Steady state (after warmup)

Once the pipeline warms up, YOLO processes frames faster (100 ms each)
than Pi3 needs them (62 ms amortized), so the worker stays ahead:

```
MAIN THREAD                    PI3 WORKER
────────────                   ──────────
YOLO frame N    (100ms)        processing window [N-12..N] (500ms)
YOLO frame N+1  (100ms)          ↓ still running
YOLO frame N+2  (100ms)          ↓ still running
YOLO frame N+3  (100ms)          ↓ still running
YOLO frame N+4  (100ms)        done → events set
get_depth(N)    → cached ✓     starts next window
YOLO frame N+5  (100ms)          ↓
...                            ...
```

**Result:** The main thread almost never blocks on `get_depth()` because
Pi3 finishes its window while YOLO is still processing subsequent frames.

| Component | Time |
|-----------|------|
| YOLO per frame | **100 ms** |
| Pi3 (overlapped) | **~0 ms wait** (runs in parallel) |
| **Effective per frame** | **~100 ms** |

---

## Pre-recorded RGB Images (Offline)

This is the typical research/benchmarking scenario:
all RGB frames exist as files on disk.

```
data/
  scene/
    RGB/rgb_2.png, rgb_3.png, ..., rgb_500.png
```

### Flow

1. `new_run.py` creates the loader + `Pi3OnlineDepthProvider(async_inference=True)`
2. The loader iterates frame indices 0, 1, 2, ...
3. `get_rgb(idx)` reads the PNG from disk (~2 ms) and calls `feed_frame(fnum, rgb)`
4. `feed_frame` pushes `(fnum, rgb)` onto the input queue — **returns instantly**
5. The main thread proceeds to run YOLO (~100 ms)
6. Meanwhile, the Pi3 worker accumulates frames and runs inference when the buffer fills
7. After YOLO + mask preprocessing, the tracker calls `get_depth(idx)`:
   - If Pi3 already finished this window → **cache hit, no wait**
   - If Pi3 is still running → `event.wait()` blocks until it finishes
8. `get_pose(idx)` works the same way

### Timeline (100 frames, window=13, overlap=5)

```
Wall clock ──────────────────────────────────────────────────────────▶

Main:  │ YOLO×13 (1.3s) │ wait~0 │ YOLO×8 │ wait~0 │ YOLO×8 │ ...
Pi3:   │ buffering...    │ win1(0.5s)│ buff.. │win2(0.5s)│buff.│...
                          ▲                    ▲
                   first window fires    second window fires
```

- **First window:** 13 frames × 100ms = 1.3s of YOLO.  Pi3 starts at
  frame 12 and takes 500ms → finishes after 1.8s.  `get_depth(0)` is
  called at ~1.4s → waits ~400ms.  After this initial stall, Pi3 stays
  ahead.
- **Steady state:** 8 new frames × 100ms = 800ms between windows.
  Pi3 takes 500ms → finishes 300ms before the main thread needs results.
  **Zero wait.**

### Throughput comparison

| Mode | 100 frames | Per-frame |
|------|-----------|-----------|
| **Sync** | ~16s (stalls every 8 frames) | ~160 ms |
| **Async** | ~10.5s (1 initial stall, rest free) | ~105 ms |

---

## Real-time Camera (Online)

Frames arrive from a live camera at some FPS (e.g. 15-30 Hz).

### Key difference from offline

- You cannot read ahead: frames arrive at the camera's pace
- YOLO + tracking must finish within the inter-frame interval to avoid dropping frames
- Pi3 results may lag behind by one window; the pipeline must tolerate this

### Flow

```python
# (pseudocode for a real-time loop)
camera = open_camera(...)
provider = Pi3OnlineDepthProvider(async_inference=True, window_size=13, overlap=5)
provider.warmup()

for frame_idx in itertools.count():
    rgb = camera.read()                       # blocks until next frame arrives
    provider.feed_frame(frame_idx, rgb)       # instant (enqueues)
    detections = yolo.track(rgb)              # ~100 ms
    depth = provider.get_depth(frame_idx)     # waits if needed
    pose  = provider.get_pose(frame_idx)
    if depth is not None:
        # full 3D tracking
        pcds  = provider.get_masked_pcds(...)
        registry.update(...)
    else:
        # fallback: 2D-only tracking for this frame
        ...
```

### Timeline at 15 FPS (~67 ms between frames)

```
Camera:  │f0│f1│f2│f3│f4│f5│f6│f7│f8│f9│f10│f11│f12│f13│...
          67ms apart

Main:    read+feed ──▶ YOLO ──▶ get_depth ──▶ tracking ──▶ ...
         ~2ms          100ms     wait?         ~30ms

Pi3:     buffering frames 0-12 ──┐
                                 ├─ inference (500ms) ──▶ cache[0..12]
                                 │
                                 ▼
         At frame 12: 12×67 = 804ms have passed since frame 0.
         Pi3 starts at t=804ms, finishes at t=1304ms.
         get_depth(12) is called at ~t=870ms → waits ~434ms.
```

### Handling the latency gap

At 15 FPS, the inter-frame interval (67 ms) is shorter than YOLO (100 ms),
so you'll drop some camera frames anyway.  Two strategies:

**Strategy A: Accept initial lag, run 3D when depth is available**

```python
depth = provider.get_depth(frame_idx)
if depth is None:
    # First ~13 frames: no depth yet, run 2D-only tracking
    track_2d(detections)
else:
    track_3d(detections, depth, pose)
```

**Strategy B: Reduce window size for lower latency**

```python
# Smaller window = faster first result, but slightly less accurate depth
provider = Pi3OnlineDepthProvider(window_size=7, overlap=3, async_inference=True)
```

| Window size | First depth available after | Pi3 per window |
|-------------|---------------------------|----------------|
| 13 | ~13 frames | ~500 ms |
| 7 | ~7 frames | ~300 ms |
| 4 | ~4 frames | ~200 ms |

### Real-time throughput budget (15 FPS target = 67 ms/frame)

| Component | Time | Bottleneck? |
|-----------|------|-------------|
| Camera read | ~2 ms | No |
| feed_frame (queue push) | <0.1 ms | No |
| YOLO | ~100 ms | **Yes — limits to ~10 FPS** |
| get_depth (cache hit) | <0.1 ms | No |
| get_depth (cache miss, wait) | 0-500 ms | Yes, first window only |
| 3D tracking | ~30 ms | No |
| Pi3 (background, amortized) | ~62 ms/frame | No (hidden) |

**YOLO is the real bottleneck for real-time**, not Pi3.  The async
architecture ensures Pi3 is never on the critical path after warmup.

---

## Configuration Reference

```python
Pi3OnlineDepthProvider(
    model_name="yyfz233/Pi3X",
    window_size=13,          # frames per Pi3 inference batch
    overlap=5,               # overlap between consecutive windows
    async_inference=True,    # True = background thread (recommended)
    device="cuda",           # GPU device for Pi3
    max_cache=128,           # how many depth/pose maps to keep in memory
    min_depth=0.01,          # clip depths below this (metres)
    max_depth=0.0,           # clip depths above this (0 = no limit)
    intrinsics=K,            # 3×3 camera matrix (optional)
)
```

### Tuning tips

- **`window_size`**: Larger → better depth quality (more temporal context),
  but higher latency for the first result and more GPU memory.
- **`overlap`**: Larger → smoother depth transitions between windows,
  but more redundant computation.  Keep it < `window_size`.
- **`async_inference=False`**: Use for debugging or profiling only.
  Forces everything onto the main thread so timings are deterministic.
- **`max_cache`**: If processing very long sequences, increase this to
  avoid evicting depth maps that the tracker hasn't consumed yet.

---

## File Map

| File | Role |
|------|------|
| `base.py` | `DepthProvider` ABC — `get_depth`, `get_pose`, `unproject`, `get_masked_pcds` |
| `gt_depth.py` | GT depth readers for each dataset (IsaacSim, THUD, ScanNet++, CODa) |
| `pi3_offline.py` | Pi3 with pre-computed depth maps on disk |
| `pi3_online.py` | Pi3 streaming inference with sync/async modes (this doc) |
| `factory.py` | `build_depth_provider()` — config-driven construction |
