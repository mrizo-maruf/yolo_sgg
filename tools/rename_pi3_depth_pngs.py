#!/usr/bin/env python3
"""Rename `frame*.png` -> `depth*.png` inside every pi3_depth* folder of a scene.

Usage:
    python3 tools/rename_pi3_depth_pngs.py <scene_path> [--dry-run]

Renames files like `frame000020.png` to `depth000020.png` in:
    <scene_path>/pi3_depth/
    <scene_path>/pi3_depth_5_3/
    <scene_path>/pi3_depth_10_5/
    ... (any folder whose name starts with "pi3_depth")
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_FRAME_RE = re.compile(r"^frame(\d+)\.png$")


def rename_in_dir(folder: Path, dry_run: bool) -> tuple[int, int]:
    renamed = 0
    skipped = 0
    for src in sorted(folder.glob("frame*.png")):
        m = _FRAME_RE.match(src.name)
        if not m:
            skipped += 1
            continue
        dst = folder / f"depth{m.group(1)}.png"
        if dst.exists():
            print(f"  SKIP (target exists): {src.name} -> {dst.name}")
            skipped += 1
            continue
        if dry_run:
            print(f"  DRY: {src.name} -> {dst.name}")
        else:
            src.rename(dst)
            print(f"  {src.name} -> {dst.name}")
        renamed += 1
    return renamed, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scene_path", type=Path, help="Path to scene directory")
    p.add_argument("--dry-run", action="store_true", help="Print actions without renaming")
    args = p.parse_args()

    scene = args.scene_path
    if not scene.is_dir():
        print(f"ERROR: not a directory: {scene}", file=sys.stderr)
        return 1

    folders = sorted(d for d in scene.iterdir() if d.is_dir() and d.name.startswith("pi3_depth"))
    if not folders:
        print(f"No pi3_depth* folders found in {scene}")
        return 0

    total_renamed = 0
    total_skipped = 0
    for folder in folders:
        print(f"\n[{folder.name}]")
        r, s = rename_in_dir(folder, args.dry_run)
        total_renamed += r
        total_skipped += s
        print(f"  -> renamed={r} skipped={s}")

    print(f"\nDone. Total renamed: {total_renamed}, skipped: {total_skipped}"
          f"{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
