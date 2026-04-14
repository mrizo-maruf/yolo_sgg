"""Simple ordered synchronization helpers for frame-keyed streams."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


_TRAILING_INT_RE = re.compile(r"(\d+)$")


def extract_trailing_int(stem_or_name: str) -> Optional[int]:
    """Extract trailing integer from a filename stem/name."""
    m = _TRAILING_INT_RE.search(str(stem_or_name))
    if not m:
        return None
    return int(m.group(1))


def sorted_files_with_ids(directory: Path, pattern: str) -> tuple[List[Path], List[int]]:
    """Return sorted files and parallel numeric IDs (or order index fallback)."""
    files = sorted(directory.glob(pattern))
    ids: List[int] = []
    for i, p in enumerate(files):
        fid = extract_trailing_int(p.stem)
        ids.append(i if fid is None else fid)
    return files, ids


@dataclass(frozen=True)
class OrderedIndexMap:
    """Map frame keys to a stable ordered index.

    Resolution strategy:
    1) Exact key match in ids
    2) rank fallback: if key is one before first id (common 1-based request
       with 0-based files), shift by +1 rank
    3) rank fallback by nearest insertion position
    """

    ids: Sequence[int]

    def resolve_index(self, key: int) -> Optional[int]:
        if not self.ids:
            return None

        try:
            return self.ids.index(int(key))
        except ValueError:
            pass

        k = int(key)
        first = int(self.ids[0])
        if k == first + 1:
            return 0

        # insertion-position rank fallback
        pos = 0
        n = len(self.ids)
        while pos < n and int(self.ids[pos]) < k:
            pos += 1
        if pos <= 0:
            return 0
        if pos >= n:
            return n - 1
        return pos

    def resolve_frame_number_index(self, frame_number: int) -> Optional[int]:
        """Resolve 1-based frame numbers to ordered index robustly."""
        if not self.ids:
            return None

        idx = int(frame_number) - 1
        if 0 <= idx < len(self.ids):
            return idx
        return self.resolve_index(int(frame_number))
