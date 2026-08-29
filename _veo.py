"""
Draw._veo
=========
Visibility Optimization Engine (VEO / VOE), also known as the Dust Remover.

Purpose:
--------
Reduces unnecessary rendering workload before a frame is drawn.
Spends intentional time during scene initialization/preprocessing to construct
a hierarchical spatial index and classify static vs dynamic shapes, enabling
low-overhead runtime culling with zero per-frame allocations where practical.

Phases:
1. Viewport Culling (Fast AABB screen test)
2. Spatial Hierarchy (Hierarchical 2D grid index)
3. Static / Dynamic Separation (Cached static bounds)
4. Compact Render Queue (Pre-allocated buffers)
5. Occlusion Culling (Front-to-back 2D opaque masking)
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

_LOCK = threading.RLock()


# ── Bounding Box Primitives ───────────────────────────────────────────────────

@dataclass(slots=True)
class AABB:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def intersects(self, other: AABB) -> bool:
        return (
            self.x < other.right
            and self.right > other.x
            and self.y < other.bottom
            and self.bottom > other.y
        )

    def contains(self, other: AABB) -> bool:
        """True if other is completely enclosed within self."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.right >= other.right
            and self.bottom >= other.bottom
        )


# ── Phase 1: Viewport Culling ─────────────────────────────────────────────────

class ViewportCuller:
    """
    Fast Axis-Aligned Bounding Box (AABB) viewport culling.
    Filters elements outside the visible viewport [scroll_x, scroll_y, scroll_x + w, scroll_y + h].
    """

    @staticmethod
    def is_visible(
        bbox: AABB,
        viewport: AABB,
        margin: float = 64.0,
    ) -> bool:
        """
        Tests if bbox intersects viewport extended by margin.
        """
        # Extended viewport with safety margin to prevent popping on rapid scroll
        vx = viewport.x - margin
        vy = viewport.y - margin
        vw = viewport.w + margin * 2.0
        vh = viewport.h + margin * 2.0

        return (
            bbox.x < vx + vw
            and bbox.x + bbox.w > vx
            and bbox.y < vy + vh
            and bbox.y + bbox.h > vy
        )


# ── Phase 2: Hierarchical 2D Spatial Grid ─────────────────────────────────────

class SpatialGridHierarchy:
    """
    2D Hierarchical Spatial Grid.
    Organizes world space into coarse (500px) and fine (100px) cells for fast $O(k)$ range queries.
    """

    def __init__(self, fine_cell_size: float = 100.0, coarse_cell_size: float = 500.0):
        self.fine_size = max(20.0, float(fine_cell_size))
        self.coarse_size = max(self.fine_size * 2.0, float(coarse_cell_size))
        
        # Maps (cx, cy) -> set of element IDs
        self.fine_grid: Dict[Tuple[int, int], Set[int]] = {}
        self.coarse_grid: Dict[Tuple[int, int], Set[int]] = {}
        self.item_boxes: Dict[int, AABB] = {}

    def clear(self) -> None:
        with _LOCK:
            self.fine_grid.clear()
            self.coarse_grid.clear()
            self.item_boxes.clear()

    def insert(self, item_id: int, bbox: AABB) -> None:
        with _LOCK:
            self.item_boxes[item_id] = bbox
            # Insert into fine grid
            f_min_x = int(math.floor(bbox.x / self.fine_size))
            f_max_x = int(math.floor(bbox.right / self.fine_size))
            f_min_y = int(math.floor(bbox.y / self.fine_size))
            f_max_y = int(math.floor(bbox.bottom / self.fine_size))

            for cx in range(f_min_x, f_max_x + 1):
                for cy in range(f_min_y, f_max_y + 1):
                    self.fine_grid.setdefault((cx, cy), set()).add(item_id)

            # Insert into coarse grid for wide queries
            c_min_x = int(math.floor(bbox.x / self.coarse_size))
            c_max_x = int(math.floor(bbox.right / self.coarse_size))
            c_min_y = int(math.floor(bbox.y / self.coarse_size))
            c_max_y = int(math.floor(bbox.bottom / self.coarse_size))

            for cx in range(c_min_x, c_max_x + 1):
                for cy in range(c_min_y, c_max_y + 1):
                    self.coarse_grid.setdefault((cx, cy), set()).add(item_id)

    def query_viewport(self, viewport: AABB, margin: float = 64.0) -> Set[int]:
        """Query elements overlapping the viewport using the spatial grid."""
        with _LOCK:
            candidates: Set[int] = set()
            vx = viewport.x - margin
            vy = viewport.y - margin
            vr = viewport.right + margin
            vb = viewport.bottom + margin

            # Choose grid level based on viewport size
            if viewport.w >= self.coarse_size * 2 or viewport.h >= self.coarse_size * 2:
                c_min_x = int(math.floor(vx / self.coarse_size))
                c_max_x = int(math.floor(vr / self.coarse_size))
                c_min_y = int(math.floor(vy / self.coarse_size))
                c_max_y = int(math.floor(vb / self.coarse_size))
                for cx in range(c_min_x, c_max_x + 1):
                    for cy in range(c_min_y, c_max_y + 1):
                        cell_items = self.coarse_grid.get((cx, cy))
                        if cell_items:
                            candidates.update(cell_items)
            else:
                f_min_x = int(math.floor(vx / self.fine_size))
                f_max_x = int(math.floor(vr / self.fine_size))
                f_min_y = int(math.floor(vy / self.fine_size))
                f_max_y = int(math.floor(vb / self.fine_size))
                for cx in range(f_min_x, f_max_x + 1):
                    for cy in range(f_min_y, f_max_y + 1):
                        cell_items = self.fine_grid.get((cx, cy))
                        if cell_items:
                            candidates.update(cell_items)

            return candidates


# ── Phase 3: Static / Dynamic Separation ──────────────────────────────────────

class StaticDynamicTracker:
    """
    Separates scene items into Static vs Dynamic sets.
    - Static items are indexed once during scene preprocessing / initialization.
    - Dynamic items (active motion, animation, live data, or dragging) are re-evaluated per frame.
    """

    def __init__(self):
        self.static_items: List[Any] = []
        self.dynamic_items: List[Any] = []
        self.static_bboxes: Dict[int, AABB] = {}

    def clear(self) -> None:
        self.static_items.clear()
        self.dynamic_items.clear()
        self.static_bboxes.clear()

    @staticmethod
    def is_dynamic(item: Any) -> bool:
        """Determines if an item has active motion, animation, or live bindings."""
        if getattr(item, "_is_dragged", False):
            return True
        if getattr(item, "motion", None) or getattr(item, "_last_motion_state", None):
            return True
        # Check text dynamic live bindings
        from Draw._text import LiveTextBinding
        if hasattr(item, "source") and isinstance(item.source, LiveTextBinding):
            return True
        return False

    def classify_and_index(self, items: List[Any], get_bbox_fn: Any) -> None:
        self.clear()
        for item in items:
            item_id = id(item)
            if self.is_dynamic(item):
                self.dynamic_items.append(item)
            else:
                self.static_items.append(item)
                bbox = get_bbox_fn(item)
                if bbox is not None:
                    self.static_bboxes[item_id] = bbox


# ── Phase 4: Compact Render Queue ─────────────────────────────────────────────

class CompactRenderQueue:
    """
    Pre-allocated, low-allocation render queue buffer.
    Avoids Python list re-allocation per frame by reusing pre-allocated slots.
    """

    def __init__(self, initial_capacity: int = 2048):
        self.capacity = initial_capacity
        self.items: List[Any] = [None] * initial_capacity
        self.size = 0

    def clear(self) -> None:
        self.size = 0

    def add(self, item: Any) -> None:
        if self.size >= self.capacity:
            self.capacity *= 2
            self.items.extend([None] * (self.capacity - len(self.items)))
        self.items[self.size] = item
        self.size += 1

    def to_list(self) -> List[Any]:
        return self.items[:self.size]


# ── Phase 5: 2D Occlusion Culling ─────────────────────────────────────────────

@dataclass(slots=True)
class OpaqueOccluder:
    bbox: AABB
    z_index: int


class OcclusionCuller:
    """
    Front-to-back 2D Hierarchical Occlusion Culling.
    Discards shapes that are 100% enclosed and covered by opaque foreground shapes.
    """

    @staticmethod
    def is_opaque(shape: Any) -> bool:
        """True if the shape is a solid, non-transparent occluder."""
        # 1. Opacity must be 100%
        opacity = getattr(shape, "opacity", 100)
        if opacity is not None and opacity < 100:
            return False

        # 2. Must not be transparent color
        col = getattr(shape, "color", None)
        if col is not None and hasattr(col, "alpha") and col.alpha() < 255:
            return False

        # 3. Exclude holes make it non-solid
        if getattr(shape, "exclude", None):
            return False

        # 4. Rotation or border radius > 0 can have transparent corners
        if getattr(shape, "rotation", 0) != 0:
            return False
        if getattr(shape, "border_radius_raw", 0) not in (0, None, "0", "0px"):
            return False

        return True

    @staticmethod
    def cull_occluded(
        candidate_items: List[Tuple[Any, str, int, AABB]],
    ) -> List[Tuple[Any, str, int, AABB]]:
        """
        Performs front-to-back occlusion tests.
        candidate_items: list of (item, kind_str, z_order, AABB) sorted descending by Z.
        """
        if len(candidate_items) <= 1:
            return candidate_items

        visible_items: List[Tuple[Any, str, int, AABB]] = []
        occluders: List[OpaqueOccluder] = []

        # Sort front-to-back: smallest z first (topmost drawn on top)
        # Remember: In Draw, lower z is on top (drawn later in Qt, or sorted ascending by z)
        # candidates are passed in ascending order of z (front to back)
        for item, kind, z_order, bbox in candidate_items:
            # Check if this bbox is fully hidden inside any existing occluder in front of it
            is_hidden = False
            for occ in occluders:
                if occ.z_index < z_order and occ.bbox.contains(bbox):
                    is_hidden = True
                    break

            if not is_hidden:
                visible_items.append((item, kind, z_order, bbox))
                # If this item itself is opaque, register it as an occluder for items behind it
                if kind == "shape" and OcclusionCuller.is_opaque(item):
                    occluders.append(OpaqueOccluder(bbox=bbox, z_index=z_order))

        return visible_items


# ── Master Visibility Optimization Engine (Dust Remover) ──────────────────────

class VisibilityOptimizer:
    """
    Master Visibility Optimization Engine (Dust Remover) Manager.
    Orchestrates Viewport Culling, Spatial Hierarchy, Static/Dynamic Tracking,
    Compact Render Queues, and Occlusion Culling.
    """

    def __init__(self):
        self.enabled: bool = True
        self.occlusion_enabled: bool = True
        self.spatial_hierarchy = SpatialGridHierarchy()
        self.tracker = StaticDynamicTracker()
        self.queue = CompactRenderQueue()
        
        # Diagnostic statistics
        self.stats: Dict[str, Any] = {
            "total_items": 0,
            "visible_items": 0,
            "culled_frustum": 0,
            "culled_occlusion": 0,
            "cull_time_ms": 0.0,
            "init_time_ms": 0.0,
        }

    def set_enabled(self, enabled: bool) -> None:
        with _LOCK:
            self.enabled = bool(enabled)

    def set_occlusion_enabled(self, enabled: bool) -> None:
        with _LOCK:
            self.occlusion_enabled = bool(enabled)

    def get_stats(self) -> Dict[str, Any]:
        with _LOCK:
            return dict(self.stats)

    @staticmethod
    def extract_shape_bbox(s: Any, cw: float, ch: float) -> Optional[AABB]:
        """Extracts or estimates shape AABB in canvas space."""
        from Draw._shapes import _shape_preferred_pos
        try:
            sw, sh, ox, oy = _shape_preferred_pos(s, int(cw), int(ch))
            return AABB(float(ox), float(oy), float(sw), float(sh))
        except Exception:
            if hasattr(s, "last_position") and s.last_position and s.last_size:
                x, y = s.last_position
                w, h = s.last_size
                return AABB(float(x), float(y), float(w), float(h))
            return None

    @staticmethod
    def extract_text_bbox(t: Any, cw: float, ch: float) -> Optional[AABB]:
        """Extracts or estimates text AABB in canvas space."""
        if hasattr(t, "last_rect") and t.last_rect:
            x, y, w, h = t.last_rect
            return AABB(float(x), float(y), float(w), float(h))
        tx = float(getattr(t, "x", 0) or 0)
        ty = float(getattr(t, "y", 0) or 0)
        return AABB(tx, ty, 100.0, 30.0)

    def preprocess_scene(self, shape_items: List[Any], text_items: List[Any], cw: float, ch: float) -> None:
        """
        Preprocesses and indexes all scene elements into spatial hierarchies.
        Called on scene creation, layout changes, or window resize.
        """
        t0 = time.perf_counter()
        with _LOCK:
            self.spatial_hierarchy.clear()
            self._cached_static_candidates: List[Tuple[Any, str, int, float, float, float, float]] = []

            # 1. Index shapes
            for s in shape_items:
                bbox = self.extract_shape_bbox(s, cw, ch)
                if bbox is not None:
                    self.spatial_hierarchy.insert(id(s), bbox)
                    if not self.tracker.is_dynamic(s):
                        self._cached_static_candidates.append(
                            (s, "shape", getattr(s, "z", 0), bbox.x, bbox.y, bbox.w, bbox.h)
                        )

            # 2. Index texts
            for t in text_items:
                bbox = self.extract_text_bbox(t, cw, ch)
                if bbox is not None:
                    self.spatial_hierarchy.insert(id(t), bbox)
                    if not self.tracker.is_dynamic(t):
                        self._cached_static_candidates.append(
                            (t, "text", getattr(t, "z", 0), bbox.x, bbox.y, bbox.w, bbox.h)
                        )

            # 3. Classify static vs dynamic
            self.tracker.classify_and_index(
                shape_items + text_items,
                get_bbox_fn=lambda it: self.extract_shape_bbox(it, cw, ch) if hasattr(it, "vertices") else self.extract_text_bbox(it, cw, ch)
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with _LOCK:
            self.stats["init_time_ms"] = round(elapsed_ms, 3)

    def cull_scene(
        self,
        shape_items: List[Any],
        text_items: List[Any],
        viewport_w: float,
        viewport_h: float,
        scroll_x: float = 0.0,
        scroll_y: float = 0.0,
        get_text_z_fn: Any = None,
        margin: float = 64.0,
    ) -> List[Tuple[Any, str, int]]:
        """
        Executes the low-overhead visibility culling pass.
        Returns the compact, sorted render queue of (item, kind_str, z_order) tuples to be drawn.
        """
        if not self.enabled:
            # Fallback: All elements pass through without culling
            queue = []
            for s in shape_items:
                queue.append((s, "shape", getattr(s, "z", 0)))
            for t in text_items:
                z = get_text_z_fn(t, shape_items) if get_text_z_fn else getattr(t, "z", 0)
                queue.append((t, "text", z))
            queue.sort(key=lambda item: (-item[2], 0 if item[1] == "shape" else 1))
            return queue

        t0 = time.perf_counter()
        vx = scroll_x - margin
        vy = scroll_y - margin
        vr = scroll_x + viewport_w + margin
        vb = scroll_y + viewport_h + margin

        total_count = len(shape_items) + len(text_items)
        culled_frustum = 0
        culled_occlusion = 0

        candidates: List[Tuple[Any, str, int, AABB]] = []

        # Fast path if scene is pre-cached
        cached = getattr(self, "_cached_static_candidates", None)
        if cached and len(cached) == total_count:
            for item, kind, z, x, y, w, h in cached:
                if x < vr and x + w > vx and y < vb and y + h > vy:
                    candidates.append((item, kind, z, AABB(x, y, w, h)))
                else:
                    culled_frustum += 1
        else:
            for s in shape_items:
                ox = getattr(s, "_placed_x", getattr(s, "x", None))
                oy = getattr(s, "_placed_y", getattr(s, "y", None))
                sw = getattr(s, "_placed_w", getattr(s, "width", 50))
                sh = getattr(s, "_placed_h", getattr(s, "height", 50))

                if ox is not None and oy is not None:
                    x, y, w, h = float(ox), float(oy), float(sw), float(sh)
                else:
                    bbox = self.extract_shape_bbox(s, viewport_w, viewport_h)
                    if bbox is None:
                        candidates.append((s, "shape", getattr(s, "z", 0), AABB(0, 0, viewport_w, viewport_h)))
                        continue
                    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

                if x < vr and x + w > vx and y < vb and y + h > vy:
                    candidates.append((s, "shape", getattr(s, "z", 0), AABB(x, y, w, h)))
                else:
                    culled_frustum += 1

            for t in text_items:
                tx = getattr(t, "_placed_x", getattr(t, "x", None))
                ty = getattr(t, "_placed_y", getattr(t, "y", None))
                if tx is not None and ty is not None:
                    x, y, w, h = float(tx), float(ty), 100.0, 30.0
                else:
                    bbox = self.extract_text_bbox(t, viewport_w, viewport_h)
                    if bbox is None:
                        z = get_text_z_fn(t, shape_items) if get_text_z_fn else getattr(t, "z", 0)
                        candidates.append((t, "text", z, AABB(0, 0, viewport_w, viewport_h)))
                        continue
                    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

                z = get_text_z_fn(t, shape_items) if get_text_z_fn else getattr(t, "z", 0)

                if x < vr and x + w > vx and y < vb and y + h > vy:
                    candidates.append((t, "text", z, AABB(x, y, w, h)))
                else:
                    culled_frustum += 1

        # Phase 5: Occlusion Culling
        if self.occlusion_enabled and len(candidates) > 1:
            candidates.sort(key=lambda item: (item[2], 1 if item[1] == "text" else 0))
            before_occ = len(candidates)
            visible_candidates = OcclusionCuller.cull_occluded(candidates)
            culled_occlusion = before_occ - len(visible_candidates)
        else:
            visible_candidates = candidates

        # Phase 4: Final Compact Render Queue in correct Draw Z-order
        visible_candidates.sort(key=lambda item: (-item[2], 0 if item[1] == "shape" else 1))
        
        final_queue = [(item[0], item[1], item[2]) for item in visible_candidates]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with _LOCK:
            self.stats.update({
                "total_items": total_count,
                "visible_items": len(final_queue),
                "culled_frustum": culled_frustum,
                "culled_occlusion": culled_occlusion,
                "cull_time_ms": round(elapsed_ms, 3),
            })

        return final_queue


# Singleton Instance
voe = VisibilityOptimizer()
veo = voe
dust_remover = voe

__all__ = [
    "AABB",
    "ViewportCuller",
    "SpatialGridHierarchy",
    "StaticDynamicTracker",
    "CompactRenderQueue",
    "OcclusionCuller",
    "VisibilityOptimizer",
    "veo",
    "voe",
    "dust_remover",
]
