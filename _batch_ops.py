"""
Draw._batch_ops
================
Shared batch-math primitives for anything in Draw that needs to operate
on many objects at once: vectorized transforms and spatial partitioning
for proximity/overlap-style queries.

Why this exists as its own module
----------------------------------
``_connectors.py``'s proximity/overlap senses currently do pairwise
distance checks — fine at tens of shapes, O(n²) and increasingly slow
past a few hundred. Rather than hand-rolling a grid inside
``_connectors.py`` and then hand-rolling a *second* one inside
``_room.py`` later for anchor/border math, both should call into this
one shared layer. Put the numeric heavy-lifting here once.

numpy is optional: every function here has a plain-Python fallback so
Draw doesn't gain a hard numpy dependency just for small scenes (numpy's
call overhead isn't worth it below ~30 items anyway — see the threshold
in batch_translate).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # numpy is optional — everything degrades gracefully
    np = None


# ── vectorized transforms ────────────────────────────────────────────────

def batch_translate(points: Sequence[Tuple[float, float]], dx: float, dy: float):
    """Translate many (x, y) points by (dx, dy). Uses numpy above a small
    size threshold (per-call numpy overhead isn't worth it for a handful
    of points); falls back to a plain loop otherwise. Same signature and
    return type (list of tuples) either way, so callers never branch on
    whether numpy is installed."""
    if np is not None and len(points) > 32:
        arr = np.asarray(points, dtype=float)
        arr[:, 0] += dx
        arr[:, 1] += dy
        return [tuple(p) for p in arr.tolist()]
    return [(x + dx, y + dy) for x, y in points]


def batch_distances(origin: Tuple[float, float], points: Sequence[Tuple[float, float]]):
    """Euclidean distance from `origin` to every point in `points`."""
    ox, oy = origin
    if np is not None and len(points) > 32:
        arr = np.asarray(points, dtype=float)
        d = np.sqrt((arr[:, 0] - ox) ** 2 + (arr[:, 1] - oy) ** 2)
        return d.tolist()
    return [math.hypot(x - ox, y - oy) for x, y in points]


# ── spatial partitioning ─────────────────────────────────────────────────

class SpatialGrid:
    """Uniform grid for cheap approximate-neighbor queries, replacing
    O(n²) pairwise distance checks in connector proximity/overlap senses
    (and, later, room anchor/border math).

    Usage pattern — rebuild once per tick, query many times per tick:

        grid = SpatialGrid(cell_size=100)
        grid.rebuild((ip, x, y) for ip, x, y in all_shape_centers)
        for ip, x, y in moving_shapes:
            candidates = grid.nearby(x, y, radius=50)   # small candidate set
            for other_ip, ox, oy in candidates:
                if math.hypot(x - ox, y - oy) <= 50:
                    ...  # exact check only against the small candidate set
    """

    def __init__(self, cell_size: float = 100.0):
        if cell_size <= 0:
            raise ValueError("SpatialGrid cell_size must be > 0")
        self.cell_size = cell_size
        self._cells: dict = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def rebuild(self, items: Iterable[Tuple[object, float, float]]) -> None:
        """items: iterable of (id, x, y). Call once per tick before any
        nearby() queries that tick — cheap (single pass, no allocation
        beyond the cell dict) compared to the O(n²) it replaces."""
        self._cells.clear()
        for item_id, x, y in items:
            self._cells.setdefault(self._key(x, y), []).append((item_id, x, y))

    def nearby(self, x: float, y: float, radius: float):
        """Return every tracked item in this cell and its 8 neighbors —
        a cheap superset guaranteed to contain everything within `radius`.
        Caller still does the exact distance check, but only against this
        small candidate set instead of every object in the scene."""
        cx, cy = self._key(x, y)
        span = max(1, math.ceil(radius / self.cell_size))
        out = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                out.extend(self._cells.get((cx + dx, cy + dy), []))
        return out

    def clear(self) -> None:
        self._cells.clear()
