"""
Draw._optimize
==============
Modular High-Performance Optimization Engine & Pluggable Backend Manager.

Architecture (V3):
- `BackendManager`: Pluggable backend registry (`register_backend`) with priority selection.
- `GeometryEngine`: Vectorized point transformation & AABB math.
- `CollisionEngine`: Fast AABB overlap tests & point-in-box hit testing.
- `SpatialEngine`: Coordinates spatial grid indexing and acceleration.
- `SpatialGridIndex`: Thread-safe 2D spatial grid with fast `update()` for moving objects.
- `RenderProfileManager`: Self-applying QPainter render hints.
- `FastObjectPool`: Object pool with automatic reset callbacks.
- `GCTuner`: Low-stutter GC threshold manager.
- Thread Safety: All shared state synchronized via `threading.RLock()`.
"""

from __future__ import annotations

import gc
import importlib
import math
import sys
import threading
import time
import weakref
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Protocol, Set, Tuple, runtime_checkable

# Global Reentrant Lock for Thread Safety
_LOCK = threading.RLock()

# Optional NumPy vector acceleration
_HAS_NUMPY = False
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None


# ── 1. Abstract Backend Interface & Implementations ────────────────────────
class BackendInterface:
    """Base interface for high-performance math backends (Cython, NumPy, C++, Rust, GPU)."""

    name: str = "Base"
    priority: int = 0

    def is_available(self) -> bool:
        return False

    def transform_vertices(
        self, vertices: list, dx: float, dy: float, angle_deg: float, scale_x: float = 1.0, scale_y: float = 1.0
    ) -> list:
        raise NotImplementedError

    def compute_aabb(self, vertices: list) -> Tuple[float, float, float, float]:
        raise NotImplementedError

    def aabb_intersections(self, boxes_a: list, boxes_b: list) -> list:
        raise NotImplementedError

    def point_in_boxes(self, px: float, py: float, boxes: list) -> list:
        raise NotImplementedError


class PythonBackend(BackendInterface):
    """Pure Python fallback implementation."""

    name = "Python"
    priority = 0

    def is_available(self) -> bool:
        return True

    def transform_vertices(
        self, vertices: list, dx: float, dy: float, angle_deg: float, scale_x: float = 1.0, scale_y: float = 1.0
    ) -> list:
        if not vertices:
            return []
        if angle_deg == 0.0 and scale_x == 1.0 and scale_y == 1.0:
            return [(x + dx, y + dy) for x, y in vertices]

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = []
        for vx, vy in vertices:
            sx, sy = vx * scale_x, vy * scale_y
            rx = sx * cos_a - sy * sin_a + dx
            ry = sx * sin_a + sy * cos_a + dy
            out.append((rx, ry))
        return out

    def compute_aabb(self, vertices: list) -> Tuple[float, float, float, float]:
        if not vertices:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def aabb_intersections(self, boxes_a: list, boxes_b: list) -> list:
        results = []
        for ax, ay, aw, ah in boxes_a:
            row = []
            for bx, by, bw, bh in boxes_b:
                is_overlap = (ax < bx + bw) and (ax + aw > bx) and (ay < by + bh) and (ay + ah > by)
                row.append(is_overlap)
            results.append(row)
        return results

    def point_in_boxes(self, px: float, py: float, boxes: list) -> list:
        matched = []
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            if bx <= px <= bx + bw and by <= py <= by + bh:
                matched.append(idx)
        return matched


class NumPyBackend(BackendInterface):
    """Vectorized C-array backend using NumPy."""

    name = "NumPy"
    priority = 50

    def is_available(self) -> bool:
        return _HAS_NUMPY

    def transform_vertices(
        self, vertices: list, dx: float, dy: float, angle_deg: float, scale_x: float = 1.0, scale_y: float = 1.0
    ) -> list:
        if not vertices:
            return []
        pts = np.asarray(vertices, dtype=np.float64)
        if scale_x != 1.0 or scale_y != 1.0:
            pts = pts * np.array([scale_x, scale_y], dtype=np.float64)
        if angle_deg != 0.0:
            rad = np.radians(angle_deg)
            c, s = np.cos(rad), np.sin(rad)
            rot_matrix = np.array([[c, -s], [s, c]], dtype=np.float64)
            pts = pts @ rot_matrix.T
        pts = pts + np.array([dx, dy], dtype=np.float64)
        return pts.tolist()

    def compute_aabb(self, vertices: list) -> Tuple[float, float, float, float]:
        if not vertices:
            return (0.0, 0.0, 0.0, 0.0)
        pts = np.asarray(vertices, dtype=np.float64)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        return (float(min_xy[0]), float(min_xy[1]), float(max_xy[0] - min_xy[0]), float(max_xy[1] - min_xy[1]))

    def aabb_intersections(self, boxes_a: list, boxes_b: list) -> list:
        if not boxes_a or not boxes_b:
            return []
        a = np.asarray(boxes_a, dtype=np.float64)
        b = np.asarray(boxes_b, dtype=np.float64)
        ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]

        overlap = (ax1[:, None] < bx2[None, :]) & (ax2[:, None] > bx1[None, :]) & \
                  (ay1[:, None] < by2[None, :]) & (ay2[:, None] > by1[None, :])
        return overlap.tolist()

    def point_in_boxes(self, px: float, py: float, boxes: list) -> list:
        if not boxes:
            return []
        b = np.asarray(boxes, dtype=np.float64)
        bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        inside = (px >= bx1) & (px <= bx2) & (py >= by1) & (py <= by2)
        return np.where(inside)[0].tolist()


class CythonImportLoader(BackendInterface):
    """Dynamic Cython / Native C extension import loader."""

    name = "Cython"
    priority = 100

    _candidates = [
        "_draw_fast_cython",
        "_draw_fast",
        "draw_accel",
        "_draw_c",
        "draw_fast",
    ]

    def __init__(self):
        self._module = None
        self._module_name = None
        self._discover()

    def _discover(self):
        for candidate in self._candidates:
            try:
                self._module = importlib.import_module(candidate)
                self._module_name = candidate
                break
            except ImportError:
                pass

    def is_available(self) -> bool:
        return self._module is not None

    def transform_vertices(
        self, vertices: list, dx: float, dy: float, angle_deg: float, scale_x: float = 1.0, scale_y: float = 1.0
    ) -> list:
        if hasattr(self._module, "transform_points"):
            return self._module.transform_points(vertices, dx, dy, angle_deg, scale_x, scale_y)
        raise NotImplementedError

    def compute_aabb(self, vertices: list) -> Tuple[float, float, float, float]:
        if hasattr(self._module, "compute_aabb"):
            return self._module.compute_aabb(vertices)
        raise NotImplementedError

    def aabb_intersections(self, boxes_a: list, boxes_b: list) -> list:
        if hasattr(self._module, "batch_aabb_intersections"):
            return self._module.batch_aabb_intersections(boxes_a, boxes_b)
        raise NotImplementedError

    def point_in_boxes(self, px: float, py: float, boxes: list) -> list:
        if hasattr(self._module, "point_in_boxes"):
            return self._module.point_in_boxes(px, py, boxes)
        raise NotImplementedError


# ── 2. Pluggable Backend Manager ──────────────────────────────────────────
class BackendManager:
    """Registry and selector for performance backends."""

    _backends: List[BackendInterface] = []
    _active_backend: Optional[BackendInterface] = None

    @classmethod
    def _init_default_backends(cls):
        with _LOCK:
            if not cls._backends:
                cls._backends = [
                    CythonImportLoader(),
                    NumPyBackend(),
                    PythonBackend(),
                ]
                cls._resolve_active_backend()

    @classmethod
    def register_backend(cls, backend: BackendInterface) -> None:
        """Register a custom backend (e.g. GPU, Rust, custom C extension)."""
        with _LOCK:
            cls._backends.append(backend)
            cls._resolve_active_backend()

    @classmethod
    def _resolve_active_backend(cls) -> None:
        sorted_backends = sorted(cls._backends, key=lambda b: b.priority, reverse=True)
        for b in sorted_backends:
            if b.is_available():
                cls._active_backend = b
                break

    @classmethod
    def get_active_backend(cls) -> BackendInterface:
        cls._init_default_backends()
        return cls._active_backend or PythonBackend()

    @classmethod
    def get_active_backend_name(cls) -> str:
        return cls.get_active_backend().name


# ── 3. Decoupled Specialized Engines ──────────────────────────────────────
class GeometryEngine:
    """Specialized engine for 2D vertex transformations and bounding box geometry."""

    @classmethod
    def batch_transform_vertices(
        cls, vertices: list, dx: float, dy: float, angle_deg: float, scale_x: float = 1.0, scale_y: float = 1.0
    ) -> list:
        backend = BackendManager.get_active_backend()
        try:
            return backend.transform_vertices(vertices, dx, dy, angle_deg, scale_x, scale_y)
        except NotImplementedError:
            return PythonBackend().transform_vertices(vertices, dx, dy, angle_deg, scale_x, scale_y)

    @classmethod
    def compute_aabb(cls, vertices: list) -> Tuple[float, float, float, float]:
        backend = BackendManager.get_active_backend()
        try:
            return backend.compute_aabb(vertices)
        except NotImplementedError:
            return PythonBackend().compute_aabb(vertices)


class CollisionEngine:
    """Specialized engine for AABB overlap testing and point-in-box hit detection."""

    @classmethod
    def batch_aabb_intersections(cls, boxes_a: list, boxes_b: list) -> list:
        backend = BackendManager.get_active_backend()
        try:
            return backend.aabb_intersections(boxes_a, boxes_b)
        except NotImplementedError:
            return PythonBackend().aabb_intersections(boxes_a, boxes_b)

    @classmethod
    def point_in_boxes(cls, px: float, py: float, boxes: list) -> list:
        backend = BackendManager.get_active_backend()
        try:
            return backend.point_in_boxes(px, py, boxes)
        except NotImplementedError:
            return PythonBackend().point_in_boxes(px, py, boxes)


# ── 4. Spatial Grid Indexing Engine ───────────────────────────────────────
class SpatialGridIndex:
    """
    Thread-safe Uniform 2D Spatial Grid Indexing structure for $O(1)$ spatial queries,
    hit-testing, and fast `update()` calls for moving objects.
    """

    def __init__(self, cell_size: float = 100.0):
        self.cell_size = max(10.0, float(cell_size))
        self.grid: Dict[Tuple[int, int], Set[Any]] = {}
        self.item_boxes: Dict[Any, Tuple[float, float, float, float]] = {}

    def clear(self) -> None:
        with _LOCK:
            self.grid.clear()
            self.item_boxes.clear()

    def _cells_for_box(self, x: float, y: float, w: float, h: float) -> Set[Tuple[int, int]]:
        cs = self.cell_size
        min_cx = int(math.floor(x / cs))
        max_cx = int(math.floor((x + w) / cs))
        min_cy = int(math.floor(y / cs))
        max_cy = int(math.floor((y + h) / cs))

        cells = set()
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cells.add((cx, cy))
        return cells

    def insert(self, item_id: Any, bbox: Tuple[float, float, float, float]) -> None:
        """Inserts an item with bounding box (x, y, w, h)."""
        with _LOCK:
            if item_id in self.item_boxes:
                self._remove_internal(item_id)
            self.item_boxes[item_id] = bbox
            for cell in self._cells_for_box(*bbox):
                self.grid.setdefault(cell, set()).add(item_id)

    def _remove_internal(self, item_id: Any) -> None:
        bbox = self.item_boxes.pop(item_id, None)
        if bbox:
            for cell in self._cells_for_box(*bbox):
                if cell in self.grid:
                    self.grid[cell].discard(item_id)
                    if not self.grid[cell]:
                        del self.grid[cell]

    def remove(self, item_id: Any) -> None:
        """Removes an item from spatial grid."""
        with _LOCK:
            self._remove_internal(item_id)

    def update(self, item_id: Any, new_bbox: Tuple[float, float, float, float]) -> None:
        """
        Fast update for moving objects. Avoids cell re-allocation if new bounding box
        occupies the exact same spatial grid cells.
        """
        with _LOCK:
            old_bbox = self.item_boxes.get(item_id)
            if old_bbox == new_bbox:
                return

            if old_bbox is not None:
                old_cells = self._cells_for_box(*old_bbox)
                new_cells = self._cells_for_box(*new_bbox)
                if old_cells == new_cells:
                    self.item_boxes[item_id] = new_bbox
                    return

            # Re-index if cells changed
            self._remove_internal(item_id)
            self.insert(item_id, new_bbox)

    def query_rect(self, bbox: Tuple[float, float, float, float]) -> Set[Any]:
        """Returns set of candidate item IDs intersecting bounding box."""
        with _LOCK:
            candidates: Set[Any] = set()
            for cell in self._cells_for_box(*bbox):
                if cell in self.grid:
                    candidates.update(self.grid[cell])
            return candidates

    def query_point(self, px: float, py: float) -> Set[Any]:
        """Returns candidate item IDs containing point (px, py)."""
        with _LOCK:
            cs = self.cell_size
            cell = (int(math.floor(px / cs)), int(math.floor(py / cs)))
            return set(self.grid.get(cell, ()))


class SpatialEngine:
    """Specialized engine managing spatial grid indices and spatial acceleration."""

    _global_grid = SpatialGridIndex(cell_size=100.0)

    @classmethod
    def get_global_grid(cls) -> SpatialGridIndex:
        return cls._global_grid


# ── 5. Global State & Diagnostics ─────────────────────────────────────────
_state: Dict[str, Any] = {
    "mode": "dev",
    "gc_tuned": False,
    "render_profile": "fast",
}


def performance_mode() -> str:
    with _LOCK:
        return _state["mode"]


def set_performance_mode(mode: str) -> None:
    if mode not in ("dev", "max"):
        raise ValueError("Draw.performance_mode must be 'dev' or 'max', got %r" % (mode,))
    with _LOCK:
        _state["mode"] = mode
        if mode == "max":
            GCTuner.tune_for_animation()


def performance_info() -> dict:
    with _LOCK:
        active_backend = BackendManager.get_active_backend()
        cython_loader = next((b for b in BackendManager._backends if isinstance(b, CythonImportLoader)), None)
        cython_mod = cython_loader._module_name if cython_loader else None

        return {
            "backend": active_backend.name,
            "numpy": _HAS_NUMPY,
            "cython": cython_loader.is_available() if cython_loader else False,
            "cython_module": cython_mod,
            "gc": _state["gc_tuned"],
            "render_profile": _state["render_profile"],
            "mode": _state["mode"],
            "registered_classes": [c.__name__ for c in _compilable_classes],
            "live_instances": len(_live_instances),
        }


# ── 6. Compilable Protocol & Registry ─────────────────────────────────────
@runtime_checkable
class Compilable(Protocol):
    def _sig(self) -> Hashable: ...
    def _is_dirty(self) -> bool: ...
    def _compile(self) -> Any: ...


_compilable_classes: List[type] = []
_live_instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def compilable(cls: type) -> type:
    with _LOCK:
        if cls not in _compilable_classes:
            _compilable_classes.append(cls)
    return cls


def register_instance(obj: Any) -> None:
    with _LOCK:
        _live_instances[id(obj)] = obj


class FastInit:
    __slots__ = ()

    @classmethod
    def _from_compiled(cls, **resolved):
        obj = object.__new__(cls)
        for k, v in resolved.items():
            setattr(obj, k, v)
        return obj


# ── 7. Garbage Collection Tuner ──────────────────────────────────────────
class GCTuner:
    _original_thresholds: Tuple[int, int, int] = gc.get_threshold()

    @classmethod
    def tune_for_animation(cls) -> None:
        with _LOCK:
            gc.set_threshold(50000, 500, 500)
            _state["gc_tuned"] = True

    @classmethod
    def restore(cls) -> None:
        with _LOCK:
            gc.set_threshold(*cls._original_thresholds)
            _state["gc_tuned"] = False


# ── 8. Render Profile Manager ─────────────────────────────────────────────
class RenderProfileManager:
    PROFILES = {
        "quality": {"antialiasing": True, "smooth_pixmap": True, "text_antialiasing": True},
        "fast": {"antialiasing": True, "smooth_pixmap": False, "text_antialiasing": False},
        "ultra": {"antialiasing": False, "smooth_pixmap": False, "text_antialiasing": False},
    }

    @classmethod
    def set_profile(cls, profile_name: str) -> None:
        with _LOCK:
            if profile_name in cls.PROFILES:
                _state["render_profile"] = profile_name

    @classmethod
    def get_profile_settings(cls, profile_name: Optional[str] = None) -> dict:
        with _LOCK:
            name = profile_name or _state.get("render_profile", "fast")
            return cls.PROFILES.get(name, cls.PROFILES["fast"])

    @classmethod
    def apply(cls, painter: Any, profile_name: Optional[str] = None) -> bool:
        settings = cls.get_profile_settings(profile_name)
        if painter is None or not hasattr(painter, "setRenderHint"):
            return False

        try:
            from PySide6.QtGui import QPainter  # type: ignore
            if hasattr(QPainter, "RenderHint"):
                rh = QPainter.RenderHint
                painter.setRenderHint(rh.Antialiasing, settings["antialiasing"])
                painter.setRenderHint(rh.SmoothPixmapTransform, settings["smooth_pixmap"])
                if hasattr(rh, "TextAntialiasing"):
                    painter.setRenderHint(rh.TextAntialiasing, settings["text_antialiasing"])
                return True
        except ImportError:
            pass
        return False


# ── 9. Fast Object Pool with Thread Safety & Reset Hooks ────────────────
class FastObjectPool:
    def __init__(self, factory_fn: Callable[[], Any], reset_fn: Optional[Callable[[Any], None]] = None, max_size: int = 1000):
        self.factory_fn = factory_fn
        self.reset_fn = reset_fn
        self.max_size = max_size
        self._pool: List[Any] = []
        self._pool_lock = threading.RLock()

    def acquire(self) -> Any:
        with self._pool_lock:
            return self._pool.pop() if self._pool else self.factory_fn()

    def release(self, obj: Any) -> None:
        if obj is None:
            return

        if self.reset_fn is not None:
            self.reset_fn(obj)
        elif hasattr(obj, "reset") and callable(getattr(obj, "reset")):
            obj.reset()

        with self._pool_lock:
            if len(self._pool) < self.max_size:
                self._pool.append(obj)


# ── 10. Main Generic Optimizer Walk ──────────────────────────────────────
def optimize(
    scene: Optional[Iterable[Any]] = None,
    mode: Optional[str] = None,
    gc_tune: bool = True,
    render_profile: Optional[str] = None,
) -> dict:
    t0 = time.perf_counter()

    if mode is not None:
        set_performance_mode(mode)

    if render_profile is not None:
        RenderProfileManager.set_profile(render_profile)

    if gc_tune and performance_mode() == "max":
        GCTuner.tune_for_animation()

    with _LOCK:
        targets = list(scene) if scene is not None else list(_live_instances.values())
        classes = tuple(_compilable_classes) if _compilable_classes else ()

    compiled = 0
    skipped = 0
    for obj in targets:
        if classes and not isinstance(obj, classes):
            continue
        if not (hasattr(obj, "_sig") and hasattr(obj, "_is_dirty") and hasattr(obj, "_compile")):
            continue
        if obj._is_dirty():
            obj._compile()
            compiled += 1
        else:
            skipped += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    info = performance_info()
    info.update({
        "compiled": compiled,
        "skipped": skipped,
        "considered": len(targets),
        "elapsed_ms": round(elapsed_ms, 3),
    })
    return info


__all__ = [
    "compilable", "register_instance", "FastInit", "Compilable",
    "performance_mode", "set_performance_mode", "performance_info", "optimize",
    "BackendInterface", "BackendManager", "PythonBackend", "NumPyBackend",
    "GeometryEngine", "CollisionEngine", "SpatialEngine", "SpatialGridIndex",
    "RenderProfileManager", "FastObjectPool", "GCTuner",
]
