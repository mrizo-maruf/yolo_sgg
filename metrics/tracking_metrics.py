"""
Dataset-agnostic 2D tracking metrics.

Computes:
    - T-mIoU   : (1/T) * Σ IoU(M̂_t^o, M_t^o)  per object, averaged across objects.
    - T-SR      : (frames tracked) / (GT frames)  per object, averaged.
    - ID Switches : #transitions in the predicted ID assigned to a GT object.
    - MOTA      : 1 − (FN + FP + IDSW) / GT
    - MOTP      : mean IoU over matched (GT, pred) pairs.
    - Per-class  aggregations of the above.

Design
------
* All functions operate on plain numpy arrays and Python dicts.
* No YOLO / Isaac Sim / Open3D / framework-specific imports.
* A lightweight ``FrameRecord`` is the only data structure callers need to fill.
"""

from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from polars import datetime


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU between two binary masks of the same spatial shape."""
    if mask_a is None or mask_b is None:
        return 0.0
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if a.shape != b.shape:
        # Attempt nearest-neighbour resize via simple indexing.
        import cv2
        b = cv2.resize(mask_b.astype(np.uint8),
                       (a.shape[1], a.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def bbox_iou_2d(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    """IoU between two 2-D boxes ``(x1, y1, x2, y2)``."""
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Data containers  (caller fills these, metrics funcs consume them)
# ---------------------------------------------------------------------------

@dataclass
class GTInstance:
    """One ground-truth object in a single frame."""
    track_id: int
    class_name: str
    mask: Optional[np.ndarray] = None          # H×W bool / uint8
    bbox_xyxy: Optional[Tuple[float, ...]] = None  # (x1,y1,x2,y2)


@dataclass
class PredInstance:
    """One predicted object in a single frame."""
    pred_id: int          # Pipeline-assigned ID (global tracking ID)
    class_name: Optional[str] = None
    mask: Optional[np.ndarray] = None
    bbox_xyxy: Optional[Tuple[float, ...]] = None


@dataclass
class FrameRecord:
    """Matched GT ↔ Pred data for one frame.

    ``mapping``  : gt_track_id → pred_id   (only for successfully matched pairs)
    ``ious``     : gt_track_id → IoU value  (for each matched pair)
    """
    frame_idx: int
    gt_objects: List[GTInstance]
    pred_objects: List[PredInstance]
    mapping: Dict[int, int]       # gt_track_id → pred_id
    ious: Dict[int, float]        # gt_track_id → IoU


# ---------------------------------------------------------------------------
# Greedy matcher  (can be swapped for Hungarian)
# ---------------------------------------------------------------------------

def match_greedy(
    gt_objects: List[GTInstance],
    pred_objects: List[PredInstance],
    iou_threshold: float = 0.3,
    use_masks: bool = True,
) -> Tuple[Dict[int, int], Dict[int, float]]:
    """Greedy bipartite matching by mask (or bbox) IoU.

    Returns
    -------
    mapping : dict   gt_track_id → pred_id
    ious    : dict   gt_track_id → IoU
    """
    if not gt_objects or not pred_objects:
        return {}, {}

    n_gt, n_pred = len(gt_objects), len(pred_objects)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, gt in enumerate(gt_objects):
        for j, pr in enumerate(pred_objects):
            if use_masks:
                iou_matrix[i, j] = mask_iou(gt.mask, pr.mask)
            else:
                iou_matrix[i, j] = bbox_iou_2d(gt.bbox_xyxy, pr.bbox_xyxy)

    # collect valid pairs, sort descending
    pairs = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i, j] >= iou_threshold:
                pairs.append((iou_matrix[i, j], i, j))
    pairs.sort(key=lambda t: -t[0])

    mapping: Dict[int, int] = {}
    ious: Dict[int, float] = {}
    used_gt, used_pred = set(), set()
    for val, gi, pj in pairs:
        if gi in used_gt or pj in used_pred:
            continue
        gt_id = gt_objects[gi].track_id
        pr_id = pred_objects[pj].pred_id
        mapping[gt_id] = pr_id
        ious[gt_id] = val
        used_gt.add(gi)
        used_pred.add(pj)

    return mapping, ious


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """Accumulates per-frame records and computes final metrics.

    Usage::

        acc = MetricsAccumulator()
        for frame in frames:
            acc.add_frame(frame_record)
        results = acc.compute()
    """

    def __init__(self) -> None:
        # per GT-object histories  (gt_track_id → list over frames)
        self._ious: Dict[int, List[float]] = defaultdict(list)
        self._tracked: Dict[int, List[bool]] = defaultdict(list)
        self._pred_ids: Dict[int, List[int]] = defaultdict(list)
        self._gt_class: Dict[int, str] = {}

        # per-class IoUs
        self._class_ious: Dict[str, List[float]] = defaultdict(list)
        self._class_counts: Dict[str, int] = defaultdict(int)

        # global counters
        self.total_gt: int = 0
        self.total_pred: int = 0
        self.total_matches: int = 0
        self.total_fp: int = 0
        self.total_fn: int = 0
        self.frames_processed: int = 0

    # ---- feed data --------------------------------------------------------

    def add_frame(self, rec: FrameRecord) -> None:
        self.frames_processed += 1
        n_gt = len(rec.gt_objects)
        n_pred = len(rec.pred_objects)
        n_matched = len(rec.mapping)

        self.total_gt += n_gt
        self.total_pred += n_pred
        self.total_matches += n_matched
        self.total_fp += (n_pred - n_matched)
        self.total_fn += (n_gt - n_matched)

        matched_gt_ids = set(rec.mapping.keys())

        for gt in rec.gt_objects:
            gid = gt.track_id
            if gid not in self._gt_class:
                self._gt_class[gid] = gt.class_name

            if gid in matched_gt_ids:
                iou = rec.ious.get(gid, 0.0)
                self._ious[gid].append(iou)
                self._tracked[gid].append(True)
                self._pred_ids[gid].append(rec.mapping[gid])
                self._class_ious[gt.class_name].append(iou)
                self._class_counts[gt.class_name] += 1
            else:
                self._ious[gid].append(0.0)
                self._tracked[gid].append(False)
                self._pred_ids[gid].append(-1)

    # ---- compute metrics --------------------------------------------------

    def compute(self) -> Dict:
        """Return a flat dict with all tracking metrics."""
        res: Dict = {}
        res["frames_processed"] = self.frames_processed
        res["unique_gt_objects"] = len(self._ious)
        res["total_gt_instances"] = self.total_gt
        res["total_pred_instances"] = self.total_pred
        res["total_matches"] = self.total_matches
        res["total_false_positives"] = self.total_fp
        res["total_false_negatives"] = self.total_fn

        # ---- T-mIoU (matched frames only — pure mask quality) ---------------
        t_miou_per_obj = {}
        for gid, v in self._ious.items():
            matched_ious = [x for x in v if x > 0]
            t_miou_per_obj[gid] = float(np.mean(matched_ious)) if matched_ious else 0.0
        res["T_mIoU"] = float(np.mean(list(t_miou_per_obj.values()))) if t_miou_per_obj else 0.0
        res["T_mIoU_std"] = float(np.std(list(t_miou_per_obj.values()))) if t_miou_per_obj else 0.0
        res["T_mIoU_per_object"] = t_miou_per_obj

        # ---- T-SR (ratio: frames tracked / GT frames per object) ----------
        t_sr_per_obj = {
            gid: sum(v) / len(v)
            for gid, v in self._tracked.items() if v
        }
        res["T_SR"] = float(np.mean(list(t_sr_per_obj.values()))) if t_sr_per_obj else 0.0
        res["T_SR_per_object"] = t_sr_per_obj

        # ---- ID Switches ---------------------------------------------------
        id_sw_per_obj: Dict[int, int] = {}
        total_idsw = 0
        for gid, pids in self._pred_ids.items():
            valid = [p for p in pids if p >= 0]
            sw = sum(1 for i in range(1, len(valid)) if valid[i] != valid[i - 1]) if len(valid) >= 2 else 0
            id_sw_per_obj[gid] = sw
            total_idsw += sw
        res["ID_switches_total"] = total_idsw
        res["ID_switches_per_object"] = id_sw_per_obj

        # ---- ID Consistency -------------------------------------------------
        id_cons_per_obj: Dict[int, float] = {}
        for gid, pids in self._pred_ids.items():
            valid = [p for p in pids if p >= 0]
            if not valid:
                id_cons_per_obj[gid] = 0.0
                continue
            most_common = Counter(valid).most_common(1)[0][0]
            id_cons_per_obj[gid] = sum(1 for p in valid if p == most_common) / len(valid)
        res["ID_consistency"] = float(np.mean(list(id_cons_per_obj.values()))) if id_cons_per_obj else 0.0
        res["ID_consistency_per_object"] = id_cons_per_obj



        # ---- MOTA -----------------------------------------------------------
        if self.total_gt > 0:
            mota = 1.0 - (self.total_fn + self.total_fp + total_idsw) / self.total_gt
            res["MOTA"] = max(-1.0, mota)
            # Diagnostic breakdown: which term hurts most?
            res["MOTA_FN_ratio"] = self.total_fn / self.total_gt
            res["MOTA_FP_ratio"] = self.total_fp / self.total_gt
            res["MOTA_IDSW_ratio"] = total_idsw / self.total_gt
        else:
            res["MOTA"] = 0.0
            res["MOTA_FN_ratio"] = 0.0
            res["MOTA_FP_ratio"] = 0.0
            res["MOTA_IDSW_ratio"] = 0.0

        # ---- MOTP -----------------------------------------------------------
        all_matched_ious = [
            iou for ious in self._ious.values() for iou in ious if iou > 0
        ]
        res["MOTP"] = float(np.mean(all_matched_ious)) if all_matched_ious else 0.0

        # ---- Per-class T-mIoU -----------------------------------------------
        per_class: Dict[str, Dict] = {}
        for cls, ious in self._class_ious.items():
            per_class[cls] = {
                "T_mIoU": float(np.mean(ious)),
                "T_mIoU_std": float(np.std(ious)),
                "count": self._class_counts[cls],
            }
        res["per_class_metrics"] = per_class

        return res


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _make_serializable(obj):
    """Recursively convert numpy types for JSON."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def save_metrics(results: Dict, output_dir: str | Path, scene_name: str = "scene") -> None:
    """Persist metrics as JSON + CSV."""
    # make outdir with date time to avoid overwriting previous results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _make_serializable(results)

    # JSON
    json_path = output_dir / f"{scene_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # CSV (flat key→value)
    # csv_path = output_dir / f"{scene_name}_metrics.csv"
    # with open(csv_path, "w", newline="") as f:
    #     w = csv.writer(f)
    #     w.writerow(["Metric", "Value"])
    #     for k, v in data.items():
    #         if not isinstance(v, dict):
    #             w.writerow([k, v])

    print(f"[metrics] Saved {json_path}")


def print_summary(results: Dict, title: str = "TRACKING METRICS") -> None:
    """Pretty-print the most important metrics to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    fmt = "  {:<28s} {:>10s}"
    def _f(k, prec=4):
        v = results.get(k, 0.0)
        return f"{v:.{prec}f}"

    print(fmt.format("T-mIoU", _f("T_mIoU")))
    print(fmt.format("T-mIoU (std)", _f("T_mIoU_std")))
    print(fmt.format("T-SR", _f("T_SR")))
    print(fmt.format("ID Consistency", _f("ID_consistency")))
    print(fmt.format("MOTA", _f("MOTA")))
    print(fmt.format("  -> FN / GT", _f("MOTA_FN_ratio")))
    print(fmt.format("  -> FP / GT", _f("MOTA_FP_ratio")))
    print(fmt.format("  -> IDSW / GT", _f("MOTA_IDSW_ratio")))
    print(fmt.format("MOTP", _f("MOTP")))
    print(fmt.format("ID Switches (total)", str(results.get("ID_switches_total", 0))))
    print()
    print(fmt.format("Frames processed", str(results.get("frames_processed", 0))))
    print(fmt.format("Unique GT objects", str(results.get("unique_gt_objects", 0))))
    print(fmt.format("Total GT instances", str(results.get("total_gt_instances", 0))))
    print(fmt.format("Total predictions", str(results.get("total_pred_instances", 0))))
    print(fmt.format("Total matches", str(results.get("total_matches", 0))))
    print(fmt.format("False positives", str(results.get("total_false_positives", 0))))
    print(fmt.format("False negatives", str(results.get("total_false_negatives", 0))))

    per_cls = results.get("per_class_metrics", {})
    if per_cls:
        print(f"\n  Per-class T-mIoU:")
        for cls, m in sorted(per_cls.items(), key=lambda x: -x[1]["T_mIoU"]):
            print(f"    {cls:20s}  {m['T_mIoU']:.4f} ± {m['T_mIoU_std']:.4f}  (n={m['count']})")
    print(sep)
