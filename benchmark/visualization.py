"""
Visualisation helpers for tracking benchmarks.

All functions take raw numpy arrays / dicts — no framework-specific types.
Optional Open3D support is guarded behind ``HAS_OPEN3D``.
"""

from __future__ import annotations

from datetime import datetime
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def color_from_id(obj_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR colour from an integer ID."""
    if obj_id < 0:
        return (128, 128, 128)
    rng = random.Random(int(obj_id) * 7 + 13)
    return (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))


# ---------------------------------------------------------------------------
# 2-D mask overlay
# ---------------------------------------------------------------------------

def draw_masks_with_labels(
    rgb: np.ndarray,
    masks: List[np.ndarray],
    ids: List[int],
    labels: List[str],
    alpha: float = 0.5,
    title: str = "",
) -> np.ndarray:
    """Overlay coloured masks on *rgb* (H,W,3 uint8) and return annotated copy."""
    vis = rgb.copy()
    for mask, oid, label in zip(masks, ids, labels):
        if mask is None:
            continue
        m = mask.astype(bool)
        c = color_from_id(oid)
        vis[m] = (vis[m] * (1 - alpha) + np.array(c[::-1]) * alpha).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        cv2.rectangle(vis, (x1, y1), (int(xs.max()), int(ys.max())), c[::-1], 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), c[::-1], -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(vis, title, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# GT-vs-Pred matching visualisation (3 panels)
# ---------------------------------------------------------------------------

def visualize_matching(
    rgb: np.ndarray,
    gt_masks: List[np.ndarray],
    gt_ids: List[int],
    gt_labels: List[str],
    pred_masks: List[np.ndarray],
    pred_ids: List[int],
    pred_labels: List[str],
    mapping: Dict[int, int],
    ious: Dict[int, float],
    frame_idx: int = 0,
    alpha: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = True,
) -> np.ndarray:
    """3-panel view: GT | Combined matching | Predictions.

    Parameters
    ----------
    mapping : dict  gt_id → pred_id  (only matched pairs)
    ious    : dict  gt_id → IoU
    """
    if rgb is None:
        return None
    H, W = rgb.shape[:2]

    matched_gt = set(mapping.keys())
    matched_pred = set(mapping.values())

    # --- GT panel ---
    gt_panel = rgb.copy()
    gt_centroids: Dict[int, Tuple[int, int]] = {}
    for mask, gid, label in zip(gt_masks, gt_ids, gt_labels):
        if mask is None:
            continue
        m = mask.astype(bool)
        c = color_from_id(gid)
        gt_panel[m] = (gt_panel[m] * (1 - alpha) + np.array(c[::-1]) * alpha).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        gt_centroids[gid] = (cx, cy)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        border = (0, 255, 0) if gid in matched_gt else (0, 0, 255)
        cv2.rectangle(gt_panel, (x1, y1), (x2, y2), border, 2)
        iou_v = ious.get(gid, 0.0)
        txt = f"GT:{gid} IoU:{iou_v:.2f}" if gid in matched_gt else f"GT:{gid} [FN]"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(gt_panel, (x1, y1 - th - 6), (x1 + tw + 4, y1), border, -1)
        cv2.putText(gt_panel, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Pred panel ---
    pred_panel = rgb.copy()
    pred_centroids: Dict[int, Tuple[int, int]] = {}
    for mask, pid, label in zip(pred_masks, pred_ids, pred_labels):
        if mask is None:
            continue
        m = mask.astype(bool)
        c = color_from_id(pid)
        pred_panel[m] = (pred_panel[m] * (1 - alpha) + np.array(c[::-1]) * alpha).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        pred_centroids[pid] = (cx, cy)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        border = (255, 165, 0) if pid in matched_pred else (0, 128, 255)
        cv2.rectangle(pred_panel, (x1, y1), (x2, y2), border, 2)
        txt = f"P:{pid} {label}" if pid in matched_pred else f"P:{pid} [FP]"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(pred_panel, (x1, y1 - th - 6), (x1 + tw + 4, y1), border, -1)
        cv2.putText(pred_panel, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Combined panel with matching lines ---
    combined = rgb.copy()
    for gid, pid in mapping.items():
        if gid in gt_centroids and pid in pred_centroids:
            g_pt = gt_centroids[gid]
            p_pt = pred_centroids[pid]
            iou_v = ious.get(gid, 0.0)
            lc = (0, 255, 0) if iou_v >= 0.7 else (0, 255, 255) if iou_v >= 0.5 else (0, 165, 255)
            cv2.line(combined, g_pt, p_pt, lc, 2, cv2.LINE_AA)
            mid = ((g_pt[0] + p_pt[0]) // 2, (g_pt[1] + p_pt[1]) // 2)
            cv2.putText(combined, f"{iou_v:.2f}", mid,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, lc, 1, cv2.LINE_AA)

    n_m = len(mapping)
    avg_iou = float(np.mean(list(ious.values()))) if ious else 0.0
    cv2.putText(gt_panel, f"GT: {len(gt_ids)} (matched {n_m})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, f"Frame {frame_idx} | {n_m} matches | IoU {avg_iou:.3f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(pred_panel, f"Pred: {len(pred_ids)} (matched {n_m})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    full = np.hstack([gt_panel, combined, pred_panel])

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        axes[0].imshow(cv2.cvtColor(gt_panel, cv2.COLOR_BGR2RGB)); axes[0].set_title("Ground Truth"); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)); axes[1].set_title(f"Matching – Frame {frame_idx}"); axes[1].axis("off")
        axes[2].imshow(cv2.cvtColor(pred_panel, cv2.COLOR_BGR2RGB)); axes[2].set_title("Predictions"); axes[2].axis("off")
        plt.suptitle(f"GT ↔ Pred Matching – Frame {frame_idx}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
    elif save_path:
        cv2.imwrite(save_path, full)

    return full


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def plot_results(results: Dict, output_dir: Optional[str] = None) -> None:
    """Generate standard plots from a metrics dict."""
    out = Path(output_dir)
    if out:
        out.mkdir(parents=True, exist_ok=True)

    # 1) Per-object T-mIoU histogram
    per_obj = results.get("T_mIoU_per_object", {})
    if per_obj:
        fig, ax = plt.subplots(figsize=(10, 6))
        vals = list(per_obj.values())
        ax.hist(vals, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(results["T_mIoU"], color="r", linestyle="--",
                   label=f"Mean: {results['T_mIoU']:.3f}")
        ax.set_xlabel("T-mIoU"); ax.set_ylabel("# Objects")
        ax.set_title("Per-Object Temporal Mean IoU"); ax.legend()
        if out:
            fig.savefig(out / "per_object_tmiou.png", dpi=150, bbox_inches="tight")
        # plt.show()

    # 2) Per-class bar chart
    per_cls = results.get("per_class_metrics", {})
    if per_cls:
        classes = list(per_cls.keys())
        vals = [per_cls[c]["T_mIoU"] for c in classes]
        stds = [per_cls[c]["T_mIoU_std"] for c in classes]
        idx = np.argsort(vals)[::-1]
        classes = [classes[i] for i in idx]
        vals = [vals[i] for i in idx]
        stds = [stds[i] for i in idx]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(len(classes)), vals, yerr=stds, capsize=3, alpha=0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylabel("T-mIoU"); ax.set_title("Per-Class T-mIoU")
        plt.tight_layout()
        if out:
            fig.savefig(out / "per_class_tmiou.png", dpi=150, bbox_inches="tight")
        # plt.show()

    # 3) Radar chart
    names = ["T-mIoU", "T-SR", "ID Consistency", "MOTA", "MOTP"]
    keys = ["T_mIoU", "T_SR", "ID_consistency", "MOTA", "MOTP"]
    vals = [max(0, min(1, results.get(k, 0))) for k in keys]
    vals.append(vals[0])
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, "o-", linewidth=2, markersize=8)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), names)
    ax.set_ylim(0, 1); ax.set_title("Tracking Metrics", y=1.08)
    for a, v in zip(angles[:-1], vals[:-1]):
        ax.annotate(f"{v:.2f}", xy=(a, v), xytext=(a, v + 0.1), ha="center", fontsize=9)
    if out:
        print(f"DEBUG: benchmark/visualization.py: Saving radar chart to {out / 'metrics_radar.png'}")
        fig.savefig(out / "metrics_radar.png", dpi=150, bbox_inches="tight")
    # plt.show()

    # 4) MOTA breakdown – stacked bar showing FN, FP, IDSW contributions
    fn_ratio = results.get("MOTA_FN_ratio", 0.0)
    fp_ratio = results.get("MOTA_FP_ratio", 0.0)
    idsw_ratio = results.get("MOTA_IDSW_ratio", 0.0)
    mota_val = results.get("MOTA", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 4a) Stacked bar: penalty breakdown
    ax = axes[0]
    penalties = [fn_ratio, fp_ratio, idsw_ratio]
    labels_p = ["FN / GT", "FP / GT", "IDSW / GT"]
    colors_p = ["#e74c3c", "#e67e22", "#f1c40f"]
    bars = ax.bar(labels_p, penalties, color=colors_p, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, penalties):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ratio (penalty)")
    ax.set_title("MOTA Penalty Breakdown")
    ax.set_ylim(0, max(max(penalties) * 1.2, 0.1))

    # 4b) Waterfall: 1.0 → subtract penalties → MOTA
    ax = axes[1]
    waterfall_labels = ["Perfect", "− FN", "− FP", "− IDSW", "= MOTA"]
    waterfall_vals = [1.0, -fn_ratio, -fp_ratio, -idsw_ratio, mota_val]
    bottoms = [0, 0, 0, 0, 0]
    running = 1.0
    bar_colors = ["#2ecc71", "#e74c3c", "#e67e22", "#f1c40f", "#3498db"]
    for i in range(1, 4):
        bottoms[i] = running + waterfall_vals[i]
        running += waterfall_vals[i]
    # "Perfect" starts at 0, height=1.0
    # penalty bars: start at new level, height = penalty (drawn downward)
    # MOTA bar: starts at 0, height = mota_val
    bar_heights = [1.0, fn_ratio, fp_ratio, idsw_ratio, max(0, mota_val)]
    bar_bottoms = [0.0, 1.0 - fn_ratio, 1.0 - fn_ratio - fp_ratio,
                   1.0 - fn_ratio - fp_ratio - idsw_ratio, 0.0]

    wbars = ax.bar(waterfall_labels, bar_heights, bottom=bar_bottoms,
                   color=bar_colors, edgecolor="black", alpha=0.85)
    for bar, bh, bb in zip(wbars, bar_heights, bar_bottoms):
        ax.text(bar.get_x() + bar.get_width() / 2, bb + bh / 2,
                f"{bh:.3f}", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("MOTA")
    ax.set_title("MOTA Waterfall")
    ax.set_ylim(min(0, mota_val - 0.1), 1.15)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    if out:
        fig.savefig(out / "mota_breakdown.png", dpi=150, bbox_inches="tight")
    # plt.show()


# ---------------------------------------------------------------------------
# 3-D bbox visualisation  (Open3D, optional)
# ---------------------------------------------------------------------------

def visualize_3d_bboxes(
    gt_bboxes: List[Dict],
    pred_bboxes: List[Dict],
    frame_idx: int = 0,
    window_title: str = "",
) -> None:
    """Show GT (red) vs Pred (green) AABBs in Open3D.

    Each bbox dict must have ``aabb`` = [xmin,ymin,zmin,xmax,ymax,zmax]
    and ``track_id``.
    """
    if not HAS_OPEN3D:
        print("[vis] Open3D not available – skipping 3-D view.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title or f"Frame {frame_idx}", width=1280, height=720)

    def _lineset(aabb, color):
        mn, mx = np.asarray(aabb[:3]), np.asarray(aabb[3:])
        pts = np.array([
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
        ])
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(edges)
        ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
        return ls

    for b in gt_bboxes:
        vis.add_geometry(_lineset(b["aabb"], [1, 0, 0]))
    for b in pred_bboxes:
        vis.add_geometry(_lineset(b["aabb"], [0, 1, 0]))

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord)
    vis.run()
    vis.destroy_window()
