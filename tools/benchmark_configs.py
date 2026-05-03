#!/usr/bin/env python3
"""Benchmark Pi3 depth configs vs GT — encoding-aware.

Drop-in replacement for the original ``benchmark_configs.py`` that decodes
GT and Pi3 PNGs with the *correct* per-dataset formulas, then refits the
Sim(3) (and rigid R,t) alignment between the two reconstructions.

Why this exists
---------------
The original script applied a single ``png / 1000`` decoder to both GT and
Pi3 depth.  That happens to be right for ScanNet++ (both are linear mm),
but it is wrong for IsaacSim, whose GT script encodes range-normalized
depth as::

    png = (depth_m - MIN) / (MAX - MIN) * 65535     # MIN=0.01, MAX=10

Reading those PNGs as ``png / 1000`` inflates the GT cloud ~6.5×, so the
fitted Sim(3) inherits that bogus scale and the rerun visualization shows
oversized objects relative to the camera.

This version
------------
- ``--dataset_type {isaacsim, scanetpp}`` selects the decoders.
- For IsaacSim GT we apply ``MIN + (png/65535)*(MAX-MIN)`` and treat
  ``png == 0`` as invalid.
- For Pi3 offline we read ``png_depth_scale`` from ``pi3_depth_meta.txt``
  inside each ``pi3_depth_<chunk>_<ovlp>/`` folder (fallback 0.001).
- Alignment JSON also stores ``sim3_matrix_4x4`` so it can be consumed
  directly by ``IsaacSimOfflinePi3DepthProvider`` /
  ``Pi3OnlineDepthProvider.set_sim3_transform``.

Usage
-----
::

    python tools/benchmark_configs.py \
        --dataset /path/to/IsaacSim_bench_pi3 \
        --dataset_type isaacsim \
        --configs 5_3 10_5 16_8 20_10

    python tools/benchmark_configs.py \
        --dataset /path/to/scannetpp_scenes \
        --dataset_type scanetpp \
        --configs 5_3 10_5


    # Refit alignments for all IsaacSim scenes and write the runtime alias for the
    # config you actually want (here 5_3 — pick whichever matches what cfg points at):
    python3 tools/benchmark_configs.py \
    --dataset /home/maribjonov_mr/Downloads/IsaacSim_bench_pi3/IsaacSim_bench_pi3 \
    --dataset_type isaacsim \
    --configs 5_3 \
    --save_transform_alias \
    --save tools/bench_out

    # Or to compare all configs head-to-head:
    python3 tools/benchmark_configs.py \
    --dataset /home/maribjonov_mr/Downloads/IsaacSim_bench_pi3/IsaacSim_bench_pi3 \
    --dataset_type isaacsim \
    --configs 5_3 10_5 16_8 20_10 \
    --save tools/bench_out

"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Depth decoders
# ──────────────────────────────────────────────────────────────────────────

_DEPTH_SCALE_RE = re.compile(r"png_depth_scale:\s*([0-9eE+.\-]+)")


def _read_pi3_meta_scale(depth_dir: Path, fallback: float = 0.001) -> float:
    """Return ``png_depth_scale`` from ``pi3_depth_meta.txt`` (or fallback)."""
    for name in ("pi3_depth_meta.txt", "depth_scale.txt", "meta.txt"):
        p = depth_dir / name
        if not p.exists():
            continue
        m = _DEPTH_SCALE_RE.search(p.read_text(encoding="utf-8"))
        if m:
            try:
                v = float(m.group(1))
                if v > 0:
                    return v
            except ValueError:
                continue
    return fallback


def decode_gt_depth(
    png: np.ndarray,
    dataset_type: str,
    isaacsim_min: float = 0.01,
    isaacsim_max: float = 10.0,
    isaacsim_png_max: int = 65535,
    scanetpp_scale: float = 1000.0,
) -> np.ndarray:
    """Decode a uint16 GT depth PNG into metres."""
    if dataset_type == "isaacsim":
        # range-normalized: depth_m = MIN + (png / PNG_MAX) * (MAX - MIN)
        # png == 0 represents invalid / clipped-near; treat as 0.
        rng = isaacsim_max - isaacsim_min
        dm = isaacsim_min + (png.astype(np.float32) / float(isaacsim_png_max)) * rng
        dm[png == 0] = 0.0
        return dm
    if dataset_type == "scanetpp":
        return png.astype(np.float32) / float(scanetpp_scale)
    raise ValueError(f"unknown dataset_type {dataset_type!r}")


def decode_pi3_depth(png: np.ndarray, scale: float) -> np.ndarray:
    """Decode a uint16 Pi3 depth PNG into metres (linear mm by default)."""
    return png.astype(np.float32) * float(scale)


# ──────────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────────

def load_poses(path: Path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                poses.append(np.array(vals).reshape(4, 4))
    return poses


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)


def transform_points_rigid(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return (pose @ np.concatenate([points, ones], 1).T).T[:, :3]


def transform_points_similarity(points, scale, R, t):
    return (scale * (R @ points.T).T) + t


def umeyama_similarity(src, dst):
    n = src.shape[0]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mu, dst - dst_mu
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    src_var = np.mean(np.sum(src_c ** 2, axis=1))
    scale = np.trace(np.diag(S) @ D) / src_var
    t = dst_mu - scale * (R @ src_mu)
    return float(scale), R, t


def rigid_alignment(src, dst):
    n = src.shape[0]
    src_mu, dst_mu = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mu, dst - dst_mu
    cov = (dst_c.T @ src_c) / n
    U, _S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    t = dst_mu - R @ src_mu
    return 1.0, R, t


def sim3_to_4x4(scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3, 3] = t
    return M


# ──────────────────────────────────────────────────────────────────────────
# Scene helpers
# ──────────────────────────────────────────────────────────────────────────

def _gt_depth_dir(scene_dir: Path, dataset_type: str) -> Path:
    """Return the GT depth folder for the given dataset layout."""
    if dataset_type == "isaacsim":
        return scene_dir / "depth"
    if dataset_type == "scanetpp":
        return scene_dir / "gt_depth"
    raise ValueError(dataset_type)


def _gt_traj_path(scene_dir: Path, dataset_type: str) -> Path:
    return scene_dir / "traj.txt"


def _backproject_world(
    depth_file: Path,
    pose: np.ndarray,
    K: np.ndarray,
    stride: int,
    decoder,
    valid_min: float,
    valid_max: float,
) -> np.ndarray:
    png = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
    if png is None:
        return np.empty((0, 3), dtype=np.float32)
    if png.ndim == 3:
        png = png[..., 0]
    dm = decoder(png)
    pts_cam = backproject(dm, K)[::stride, ::stride].reshape(-1, 3)
    z = pts_cam[:, 2]
    valid = (z > valid_min) & (z < valid_max) & np.isfinite(z)
    pts_cam = pts_cam[valid]
    if pts_cam.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    return transform_points_rigid(pts_cam, pose)


def reconstruct_scene(
    depth_files,
    poses,
    K,
    stride,
    n_frames,
    voxel_size,
    decoder,
    valid_min: float,
    valid_max: float,
):
    """Merge per-frame backprojections into one (optionally voxelized) cloud."""
    import open3d as o3d

    merged = o3d.geometry.PointCloud()
    for i in range(n_frames):
        pts = _backproject_world(
            depth_files[i], poses[i], K, stride, decoder, valid_min, valid_max
        )
        if pts.shape[0] == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        merged += pcd

    if voxel_size > 0 and len(merged.points) > 0:
        merged = merged.voxel_down_sample(voxel_size)

    return np.asarray(merged.points)


def compute_metrics(gt_pts, pred_pts, threshold=0.05):
    if gt_pts.shape[0] == 0 or pred_pts.shape[0] == 0:
        return None
    from scipy.spatial import cKDTree

    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    dist_pg, _ = gt_tree.query(pred_pts)
    dist_gp, _ = pred_tree.query(gt_pts)

    accuracy = float(np.mean(dist_pg))
    completion = float(np.mean(dist_gp))
    chamfer = (accuracy + completion) / 2.0
    rmse = float(np.sqrt(np.mean(dist_pg ** 2)))

    precision = float(np.mean(dist_pg < threshold))
    recall = float(np.mean(dist_gp < threshold))
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "completion": completion,
        "chamfer": chamfer,
        "rmse": rmse,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ──────────────────────────────────────────────────────────────────────────
# Per-config evaluation
# ──────────────────────────────────────────────────────────────────────────

def discover_configs(scene_dir: Path, explicit=None):
    if explicit:
        valid = []
        for c in explicit:
            if (scene_dir / f"pi3_depth_{c}").is_dir() and (
                scene_dir / f"pi3_traj_{c}.txt"
            ).is_file():
                valid.append(c)
        return sorted(valid)
    out = []
    for d in sorted(scene_dir.iterdir()):
        if d.is_dir() and d.name.startswith("pi3_depth_"):
            cfg = d.name[len("pi3_depth_"):]
            if (scene_dir / f"pi3_traj_{cfg}.txt").is_file():
                out.append(cfg)
    return sorted(out)


def evaluate_config(
    scene_dir: Path,
    config: str,
    dataset_type: str,
    K: np.ndarray,
    align_frame: int,
    stride: int,
    voxel_size: float,
    max_eval_points: int,
    valid_min: float,
    valid_max: float,
    save_transform_alias: bool,
):
    gt_depth_dir = _gt_depth_dir(scene_dir, dataset_type)
    pi3_depth_dir = scene_dir / f"pi3_depth_{config}"
    gt_traj = _gt_traj_path(scene_dir, dataset_type)
    pi3_traj = scene_dir / f"pi3_traj_{config}.txt"

    if not all(p.exists() for p in [gt_depth_dir, pi3_depth_dir, gt_traj, pi3_traj]):
        return None

    gt_poses = load_poses(gt_traj)
    pi3_poses = load_poses(pi3_traj)

    gt_files = sorted(gt_depth_dir.glob("*.png"))
    pi3_files = sorted(pi3_depth_dir.glob("*.png"))

    n = min(len(gt_files), len(pi3_files), len(gt_poses), len(pi3_poses))
    if n == 0 or align_frame >= n:
        return None

    pi3_scale = _read_pi3_meta_scale(pi3_depth_dir)
    gt_decoder = lambda png: decode_gt_depth(png, dataset_type)  # noqa: E731
    pi3_decoder = lambda png: decode_pi3_depth(png, pi3_scale)   # noqa: E731

    # ── 1. Single-frame correspondences for alignment ────────────────────
    gt_png = cv2.imread(str(gt_files[align_frame]), cv2.IMREAD_UNCHANGED)
    pi3_png = cv2.imread(str(pi3_files[align_frame]), cv2.IMREAD_UNCHANGED)
    if gt_png is None or pi3_png is None:
        return None
    if gt_png.ndim == 3:
        gt_png = gt_png[..., 0]
    if pi3_png.ndim == 3:
        pi3_png = pi3_png[..., 0]

    gt_dm = gt_decoder(gt_png)
    pi3_dm = pi3_decoder(pi3_png)

    if gt_dm.shape != pi3_dm.shape:
        # Resize Pi3 depth to GT shape (Pi3 export should already match RGB,
        # but be defensive).
        pi3_dm = cv2.resize(
            pi3_dm, (gt_dm.shape[1], gt_dm.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    gt_cam = backproject(gt_dm, K)[::stride, ::stride].reshape(-1, 3)
    pi3_cam = backproject(pi3_dm, K)[::stride, ::stride].reshape(-1, 3)

    valid = (
        (gt_cam[:, 2] > valid_min)
        & (gt_cam[:, 2] < valid_max)
        & (pi3_cam[:, 2] > valid_min)
        & (pi3_cam[:, 2] < valid_max)
    )
    if int(valid.sum()) < 100:
        return None

    gt_corr = transform_points_rigid(gt_cam[valid], gt_poses[align_frame])
    pi3_corr = transform_points_rigid(pi3_cam[valid], pi3_poses[align_frame])

    align_cap = 200_000
    if gt_corr.shape[0] > align_cap:
        rng = np.random.default_rng(42)
        idx = rng.choice(gt_corr.shape[0], size=align_cap, replace=False)
        gt_corr, pi3_corr = gt_corr[idx], pi3_corr[idx]

    s_rigid, R_rigid, t_rigid = rigid_alignment(pi3_corr, gt_corr)
    s_sim, R_sim, t_sim = umeyama_similarity(pi3_corr, gt_corr)

    print(
        f"    [{config}] pi3_scale={pi3_scale:.6g} "
        f"sim3.scale={s_sim:.4f} valid_corr={int(valid.sum())}",
        end=" ",
        flush=True,
    )

    # ── 2. Full-scene reconstruction ────────────────────────────────────
    print("recon GT…", end=" ", flush=True)
    gt_pts = reconstruct_scene(
        gt_files, gt_poses, K, stride, n, voxel_size, gt_decoder, valid_min, valid_max
    )
    print(f"{gt_pts.shape[0]}", end=" ", flush=True)

    print("Pi3…", end=" ", flush=True)
    pi3_pts = reconstruct_scene(
        pi3_files, pi3_poses, K, stride, n, voxel_size, pi3_decoder, valid_min, valid_max
    )
    print(f"{pi3_pts.shape[0]}", end=" ", flush=True)

    if gt_pts.shape[0] == 0 or pi3_pts.shape[0] == 0:
        return None

    # ── 3. Subsample for KD-tree ────────────────────────────────────────
    rng = np.random.default_rng(0)
    gt_eval = gt_pts
    if max_eval_points > 0 and gt_eval.shape[0] > max_eval_points:
        idx = rng.choice(gt_eval.shape[0], size=max_eval_points, replace=False)
        gt_eval = gt_eval[idx]

    out = {}
    for mode, (s, R, t) in [
        ("rigid", (s_rigid, R_rigid, t_rigid)),
        ("similarity", (s_sim, R_sim, t_sim)),
    ]:
        pi3_aligned = transform_points_similarity(pi3_pts, s, R, t)
        pi3_eval = pi3_aligned
        if max_eval_points > 0 and pi3_eval.shape[0] > max_eval_points:
            idx = rng.choice(pi3_eval.shape[0], size=max_eval_points, replace=False)
            pi3_eval = pi3_eval[idx]
        m = compute_metrics(gt_eval, pi3_eval)
        if m is None:
            return None
        out[mode] = m

    # ── 4. Save the alignment ──────────────────────────────────────────
    sim3_4x4 = sim3_to_4x4(s_sim, R_sim, t_sim)
    payload = {
        "config": config,
        "dataset_type": dataset_type,
        "pi3_png_depth_scale": pi3_scale,
        "scale": s_sim,
        "rotation": R_sim.tolist(),
        "translation": t_sim.tolist(),
        "sim3_matrix_4x4": sim3_4x4.tolist(),
        "rigid_R": R_rigid.tolist(),
        "rigid_t": t_rigid.tolist(),
    }
    out_path = scene_dir / f"align_pi3_{config}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {out_path.name}", end=" ", flush=True)

    # Convenience alias consumed by IsaacSimOfflinePi3DepthProvider /
    # Pi3OnlineDepthProvider.set_sim3_transform.
    if save_transform_alias and dataset_type == "isaacsim":
        alias = scene_dir / "pi3_to_world_transform.json"
        with open(alias, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"+ {alias.name}", end=" ", flush=True)

    return out


# ──────────────────────────────────────────────────────────────────────────
# Plotting + main
# ──────────────────────────────────────────────────────────────────────────

METRIC_LABELS = {
    "accuracy":   "Accuracy (m)",
    "completion": "Completion (m)",
    "chamfer":    "Chamfer (m)",
    "rmse":       "RMSE (m)",
    "precision":  "Precision @5cm",
    "recall":     "Recall @5cm",
    "f1":         "F1 @5cm",
}


def make_label(c: str) -> str:
    parts = c.split("_")
    if len(parts) == 2:
        return f"chunk={parts[0]} / ovlp={parts[1]}"
    return c


def plot_results(all_metrics, mode_label, output_path=None):
    configs = list(all_metrics.keys())
    metrics = list(METRIC_LABELS.keys())
    n_configs = len(configs)
    n_metrics = len(metrics)
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_configs, 1)))
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = [all_metrics[c].get(metric, 0) for c in configs]
        labels = [make_label(c) for c in configs]
        bars = ax.bar(range(n_configs), vals, color=colors[:n_configs])
        ax.set_xticks(range(n_configs))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=7,
            )
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(
        f"3D Reconstruction Metrics — {mode_label} alignment\n"
        f"(averaged over scenes)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot: {output_path}")
    plt.show()


def _default_intrinsics(dataset_type: str):
    if dataset_type == "isaacsim":
        # focal_length=50, h_aperture=80, v_aperture=45, image=1280x720
        return dict(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
    if dataset_type == "scanetpp":
        return dict(fx=692.52, fy=693.83, cx=459.76, cy=344.76)
    return dict(fx=692.52, fy=693.83, cx=459.76, cy=344.76)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="Dataset root containing scene folders.")
    p.add_argument("--dataset_type", choices=["isaacsim", "scanetpp"],
                   required=True,
                   help="Selects depth decoders + GT directory layout.")
    p.add_argument("--configs", nargs="*", default=None,
                   help="Explicit config suffixes (e.g. 5_3 10_5 16_8). "
                        "Auto-discover if omitted.")
    p.add_argument("--align_frame", type=int, default=0)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--voxel_size", type=float, default=0.02)
    p.add_argument("--max_eval_points", type=int, default=500_000)
    p.add_argument("--valid_min", type=float, default=0.05,
                   help="Discard depth pixels below this many metres.")
    p.add_argument("--valid_max", type=float, default=10.0,
                   help="Discard depth pixels above this many metres.")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--save", type=str, default=None,
                   help="Directory to save plot PNGs.")
    p.add_argument("--save_transform_alias", action="store_true",
                   help="For IsaacSim, also write `pi3_to_world_transform.json`"
                        " (consumed by IsaacSimOfflinePi3DepthProvider).")
    args = p.parse_args()

    K_def = _default_intrinsics(args.dataset_type)
    fx = args.fx if args.fx is not None else K_def["fx"]
    fy = args.fy if args.fy is not None else K_def["fy"]
    cx = args.cx if args.cx is not None else K_def["cx"]
    cy = args.cy if args.cy is not None else K_def["cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"[K] fx={fx} fy={fy} cx={cx} cy={cy}")

    scenes = sorted(
        d for d in args.dataset.iterdir()
        if d.is_dir()
        and _gt_traj_path(d, args.dataset_type).is_file()
        and _gt_depth_dir(d, args.dataset_type).is_dir()
    )
    if not scenes:
        raise RuntimeError(
            f"No valid scenes under {args.dataset} for dataset_type="
            f"{args.dataset_type}"
        )
    print(f"Found {len(scenes)} scene(s)")

    configs = args.configs or discover_configs(scenes[0])
    if not configs:
        raise RuntimeError(
            f"No pi3_depth_*/pi3_traj_*.txt configs in {scenes[0]}. "
            "Pass --configs explicitly."
        )
    print(f"Configs: {configs}")

    rigid_acc = {c: defaultdict(list) for c in configs}
    sim_acc = {c: defaultdict(list) for c in configs}
    counts = {c: 0 for c in configs}

    for scene in scenes:
        scene_cfgs = discover_configs(scene, explicit=configs)
        if not scene_cfgs:
            print(f"  [{scene.name}] no matching configs, skipping")
            continue
        for cfg in scene_cfgs:
            print(f"  [{scene.name}] {cfg} … ", end="", flush=True)
            res = evaluate_config(
                scene, cfg, args.dataset_type, K,
                args.align_frame, args.stride, args.voxel_size,
                args.max_eval_points, args.valid_min, args.valid_max,
                args.save_transform_alias,
            )
            if res is None:
                print("SKIP")
                continue
            counts[cfg] += 1
            for k, v in res["rigid"].items():
                rigid_acc[cfg][k].append(v)
            for k, v in res["similarity"].items():
                sim_acc[cfg][k].append(v)
            print("OK")

    rigid_avg, sim_avg = {}, {}
    for c in configs:
        if counts[c] == 0:
            print(f"  WARN: no valid scenes for config {c}")
            continue
        rigid_avg[c] = {k: float(np.mean(v)) for k, v in rigid_acc[c].items()}
        sim_avg[c] = {k: float(np.mean(v)) for k, v in sim_acc[c].items()}
    if not rigid_avg:
        raise RuntimeError("No valid results.")

    print("\n" + "=" * 80)
    print("RESULTS (averaged over scenes)")
    print("=" * 80)
    for mode, avg in [("Rigid (R,t)", rigid_avg), ("Similarity (s,R,t)", sim_avg)]:
        print(f"\n  {mode}:")
        header = f"  {'Config':<20}"
        for m in METRIC_LABELS:
            header += f"  {m:>12}"
        print(header)
        print("  " + "-" * (20 + 14 * len(METRIC_LABELS)))
        for c in avg:
            row = f"  {make_label(c):<20}"
            for m in METRIC_LABELS:
                row += f"  {avg[c].get(m, 0):>12.5f}"
            print(row)

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    plot_results(
        rigid_avg, "Rigid (R, t)",
        output_path=save_dir / "benchmark_rigid.png" if save_dir else None,
    )
    plot_results(
        sim_avg, "Similarity (s, R, t)",
        output_path=save_dir / "benchmark_similarity.png" if save_dir else None,
    )


if __name__ == "__main__":
    main()
