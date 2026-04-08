#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_poses(path: Path) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 16:
                continue
            poses.append(
                np.array(list(map(float, vals)), dtype=np.float64).reshape(4, 4)
            )
    return poses


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src/dst must be Nx3 and the same shape")
    n = src.shape[0]
    if n < 3:
        raise ValueError("need at least 3 points for Sim(3) alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / float(n)
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1.0
    r = u @ s_mat @ vt
    var_src = (src_c ** 2).sum() / float(n)
    scale = float(np.trace(np.diag(d) @ s_mat) / var_src)
    t = mu_dst - scale * (r @ mu_src)
    return scale, r, t


def _build_sim3(scale: float, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    sim3 = np.eye(4, dtype=np.float64)
    sim3[:3, :3] = scale * r
    sim3[:3, 3] = t
    return sim3


def main() -> int:
    p = argparse.ArgumentParser(description="Compute Pi3-to-world Sim(3) transform.")
    p.add_argument("--scene_path", required=True, help="IsaacSim scene directory.")
    p.add_argument("--gt_pose", default="traj.txt", help="GT pose file (16 floats/line).")
    p.add_argument(
        "--pi3_pose",
        default="pi3_camera_poses.txt",
        help="Pi3 pose file (16 floats/line).",
    )
    p.add_argument(
        "--out",
        default="pi3_to_world_transform.json",
        help="Output JSON filename.",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Use at most N frames (0 = all).",
    )
    args = p.parse_args()

    scene_p = Path(args.scene_path).expanduser().resolve()
    gt_path = Path(args.gt_pose)
    pi3_path = Path(args.pi3_pose)
    out_path = Path(args.out)

    if not gt_path.is_absolute():
        gt_path = scene_p / gt_path
    if not pi3_path.is_absolute():
        pi3_path = scene_p / pi3_path
    if not out_path.is_absolute():
        out_path = scene_p / out_path

    if not gt_path.exists():
        raise FileNotFoundError(f"GT pose file not found: {gt_path}")
    if not pi3_path.exists():
        raise FileNotFoundError(f"Pi3 pose file not found: {pi3_path}")

    gt_poses = _load_poses(gt_path)
    pi3_poses = _load_poses(pi3_path)
    n = min(len(gt_poses), len(pi3_poses))
    if args.max_frames > 0:
        n = min(n, int(args.max_frames))
    if n < 3:
        raise ValueError("not enough poses to compute alignment")

    gt_centers = np.stack([p[:3, 3] for p in gt_poses[:n]], axis=0)
    pi3_centers = np.stack([p[:3, 3] for p in pi3_poses[:n]], axis=0)

    scale, r, t = _umeyama(pi3_centers, gt_centers)
    sim3 = _build_sim3(scale, r, t)

    payload = {"sim3_matrix_4x4": sim3.tolist()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
