"""
Draw._overlap
Overlap / flow resolution for shape, text, and path placement.

FlowSpec  — per-item configuration produced by parse_flow_spec().
Strategies — HorizontalStackStrategy (default), VerticalStackStrategy,
             GridPackStrategy, RadialSpiralStrategy, PackingStrategy,
             SmartStackStrategy, OverlapAllowedStrategy.

Public helpers used by the canvas paint loop:
    parse_flow_spec(raw, *, flow_provided, overlap, ...) -> FlowSpec
    get_strategy_for_flow(flow_spec) -> strategy instance
    flow_occupied_rect(x, y, w, h, flow_spec) -> Rect
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol


# ── Rectangle ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Rect:
    """Immutable axis-aligned bounding box."""
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:    return self.x
    @property
    def right(self) -> float:   return self.x + self.w
    @property
    def top(self) -> float:     return self.y
    @property
    def bottom(self) -> float:  return self.y + self.h
    @property
    def center_x(self) -> float: return self.x + self.w / 2
    @property
    def center_y(self) -> float: return self.y + self.h / 2

    def intersects(self, other: "Rect") -> bool:
        return (self.left < other.right and self.right > other.left and
                self.top < other.bottom and self.bottom > other.top)

    def contains_point(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def contains_rect(self, other: "Rect") -> bool:
        return (self.left <= other.left and self.right >= other.right and
                self.top <= other.top and self.bottom >= other.bottom)

    def expand(self, padding: float) -> "Rect":
        return Rect(self.x - padding, self.y - padding,
                    self.w + padding * 2, self.h + padding * 2)

    def move_to(self, x: float, y: float) -> "Rect":
        return Rect(x, y, self.w, self.h)

    def rotated_bounds(self, angle_degrees: float) -> "Rect":
        """Compute axis-aligned bounding box enclosing rotated rectangle."""
        if not angle_degrees:
            return self
        rad = math.radians(angle_degrees)
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))
        new_w = self.w * cos_a + self.h * sin_a
        new_h = self.w * sin_a + self.h * cos_a
        cx, cy = self.center_x, self.center_y
        return Rect(cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h)


class QuadTree:
    """Spatial partitioning tree for fast O(N log N) 2D collision queries."""

    def __init__(self, bounds: Rect, max_objects: int = 10, max_levels: int = 5, level: int = 0):
        self.bounds = bounds
        self.max_objects = max_objects
        self.max_levels = max_levels
        self.level = level
        self.objects: List[Rect] = []
        self.nodes: List[QuadTree] = []

    def clear(self) -> None:
        self.objects.clear()
        for node in self.nodes:
            node.clear()
        self.nodes.clear()

    def _split(self) -> None:
        sub_w = self.bounds.w / 2.0
        sub_h = self.bounds.h / 2.0
        x = self.bounds.x
        y = self.bounds.y

        self.nodes.append(QuadTree(Rect(x + sub_w, y, sub_w, sub_h), self.max_objects, self.max_levels, self.level + 1))
        self.nodes.append(QuadTree(Rect(x, y, sub_w, sub_h), self.max_objects, self.max_levels, self.level + 1))
        self.nodes.append(QuadTree(Rect(x, y + sub_h, sub_w, sub_h), self.max_objects, self.max_levels, self.level + 1))
        self.nodes.append(QuadTree(Rect(x + sub_w, y + sub_h, sub_w, sub_h), self.max_objects, self.max_levels, self.level + 1))

    def _get_index(self, rect: Rect) -> int:
        index = -1
        vert_midpoint = self.bounds.x + (self.bounds.w / 2.0)
        horiz_midpoint = self.bounds.y + (self.bounds.h / 2.0)

        top_quad = (rect.top < horiz_midpoint and rect.bottom < horiz_midpoint)
        bot_quad = (rect.top > horiz_midpoint)

        if rect.left < vert_midpoint and rect.right < vert_midpoint:
            if top_quad:
                index = 1
            elif bot_quad:
                index = 2
        elif rect.left > vert_midpoint:
            if top_quad:
                index = 0
            elif bot_quad:
                index = 3

        return index

    def insert(self, rect: Rect) -> None:
        if self.nodes:
            index = self._get_index(rect)
            if index != -1:
                self.nodes[index].insert(rect)
                return

        self.objects.append(rect)

        if len(self.objects) > self.max_objects and self.level < self.max_levels:
            if not self.nodes:
                self._split()

            i = 0
            while i < len(self.objects):
                index = self._get_index(self.objects[i])
                if index != -1:
                    self.nodes[index].insert(self.objects.pop(i))
                else:
                    i += 1

    def retrieve(self, return_objects: List[Rect], rect: Rect) -> List[Rect]:
        index = self._get_index(rect)
        if index != -1 and self.nodes:
            self.nodes[index].retrieve(return_objects, rect)

        return_objects.extend(self.objects)
        return return_objects


# ── FlowSpec ─────────────────────────────────────────────────────────────────

@dataclass
class FlowSpec:
    """
    Resolved flow/anti-overlap specification for one shape, text, or path item.

    Fields
    ------
    enabled     : whether flow placement is active for this item
    mode        : layout algorithm  ("horizontal"|"vertical"|"grid"|"spiral"|"pack"|"smart")
    direction   : movement axis hint  ("right"|"down"|"right_wrap"|"down_wrap"|"left"|"up")
    gap         : minimum pixels between items  (default 4)
    padding_x   : extra horizontal padding added to the occupied rect
    padding_y   : extra vertical padding added to the occupied rect
    role        : "item" (participates) | "blocker" (occupies but not moved)
                  | "ignore" (neither moved nor occupies registry)
    wrap        : wrap to next row/column at canvas edge
    scope       : "window" | "cell"
    area_expand : (ex, ey) extra pixels added to collision rect
    area_move   : preferred move direction ("right"|"left"|"down"|"up")
    """
    enabled:      bool                  = False
    mode:         str                   = "horizontal"
    direction:    str                   = "right"
    gap:          float                 = 4.0
    padding_x:    float                 = 0.0
    padding_y:    float                 = 0.0
    role:         str                   = "item"
    wrap:         bool                  = True
    scope:        str                   = "window"
    area_expand:  Tuple[float, float]   = field(default=(0.0, 0.0))
    area_move:    Optional[str]         = field(default=None)


# ── parse_flow_spec ───────────────────────────────────────────────────────────

_DIRECTION_ALIASES: Dict[str, str] = {
    "right":      "right",
    "left":       "left",
    "down":       "down",
    "up":         "up",
    "right_wrap": "right_wrap",
    "down_wrap":  "down_wrap",
    "horizontal": "right",
    "vertical":   "down",
    "h":          "right",
    "v":          "down",
}

_MODE_ALIASES: Dict[str, str] = {
    "horizontal": "horizontal",
    "vertical":   "vertical",
    "grid":       "grid",
    "spiral":     "spiral",
    "pack":       "pack",
    "smart":      "smart",
    "right":      "horizontal",
    "down":       "vertical",
    "right_wrap": "horizontal",
    "down_wrap":  "vertical",
    "left":       "horizontal",
    "up":         "vertical",
}

_ROLE_VALUES = {"item", "blocker", "ignore"}


def parse_flow_spec(
    raw: Any = None,
    *,
    flow_provided: bool = False,
    overlap: bool = True,
    closest_rect_area: bool = False,
    area_expand: Tuple[float, float] = (0.0, 0.0),
    area_move: Optional[str] = None,
) -> FlowSpec:
    """
    Convert the raw ``flow=`` value from a shape/text/path dict into a FlowSpec.

    Accepted forms
    --------------
    True                           → enabled with defaults
    False                          → disabled
    "horizontal" | "vertical" | … → enabled, named mode
    {"direction": "down_wrap", "gap": 9, "role": "blocker", ...}
    None + overlap=False           → legacy: enabled (backward-compat)
    None + overlap=True            → disabled
    """
    # Legacy: overlap=False with no explicit flow keyword → enable flow
    if not flow_provided and not overlap:
        return FlowSpec(enabled=True, mode="horizontal", direction="right",
                        gap=4.0, role="item", wrap=True,
                        area_expand=area_expand, area_move=area_move)

    if raw is False:
        return FlowSpec(enabled=False, area_expand=area_expand, area_move=area_move)

    if raw is True:
        return FlowSpec(enabled=True, mode="horizontal", direction="right",
                        gap=4.0, role="item", wrap=True,
                        area_expand=area_expand, area_move=area_move)

    if isinstance(raw, str):
        mode_key = raw.strip().lower()
        mode = _MODE_ALIASES.get(mode_key, "horizontal")
        direction = _DIRECTION_ALIASES.get(mode_key, "right")
        wrap = True
        return FlowSpec(enabled=True, mode=mode, direction=direction,
                        gap=4.0, role="item", wrap=wrap,
                        area_expand=area_expand, area_move=area_move)

    if isinstance(raw, dict):
        direction_raw = str(raw.get("direction", "right")).strip().lower()
        direction = _DIRECTION_ALIASES.get(direction_raw, "right")
        mode_raw = str(raw.get("mode", direction_raw)).strip().lower()
        mode = _MODE_ALIASES.get(mode_raw, "horizontal")

        gap = float(raw.get("gap", 4.0))
        role_raw = str(raw.get("role", "item")).strip().lower()
        role = role_raw if role_raw in _ROLE_VALUES else "item"
        wrap_raw = raw.get("wrap", True)
        wrap = bool(wrap_raw) if isinstance(wrap_raw, bool) else str(wrap_raw).lower() != "false"
        scope = str(raw.get("scope", "window")).strip().lower()

        pad = raw.get("padding", None)
        if isinstance(pad, (list, tuple)) and len(pad) >= 2:
            padding_x, padding_y = float(pad[0]), float(pad[1])
        elif isinstance(pad, (int, float)):
            padding_x = padding_y = float(pad)
        else:
            padding_x = float(raw.get("padding_x", 0.0))
            padding_y = float(raw.get("padding_y", 0.0))

        ae_raw = raw.get("area_expand", None)
        if isinstance(ae_raw, (list, tuple)) and len(ae_raw) >= 2:
            area_expand = (float(ae_raw[0]), float(ae_raw[1]))
        elif isinstance(ae_raw, (int, float)):
            area_expand = (float(ae_raw), float(ae_raw))

        am_raw = raw.get("area_move", area_move)
        area_move_final = str(am_raw).strip().lower() if am_raw is not None else None

        return FlowSpec(
            enabled=True, mode=mode, direction=direction, gap=gap,
            padding_x=padding_x, padding_y=padding_y,
            role=role, wrap=wrap, scope=scope,
            area_expand=area_expand, area_move=area_move_final,
        )

    if closest_rect_area:
        return FlowSpec(enabled=True, mode="horizontal", direction="right",
                        gap=4.0, role="item", wrap=True,
                        area_expand=area_expand, area_move=area_move)

    return FlowSpec(enabled=False, area_expand=area_expand, area_move=area_move)


# ── flow_occupied_rect ────────────────────────────────────────────────────────

def flow_occupied_rect(x: float, y: float, w: float, h: float,
                       flow_spec: FlowSpec) -> Rect:
    """
    Return the Rect that a placed item registers in the global occupied list.
    Includes padding and area_expand.
    """
    ex, ey = flow_spec.area_expand
    px = flow_spec.padding_x + ex
    py = flow_spec.padding_y + ey
    return Rect(x - px, y - py, w + px * 2, h + py * 2)


# ── get_strategy_for_flow ─────────────────────────────────────────────────────

def get_strategy_for_flow(flow_spec: FlowSpec):
    """Return the appropriate placement strategy instance for a FlowSpec."""
    gap = max(0.0, flow_spec.gap)
    mode = flow_spec.mode
    direction = flow_spec.direction
    wrap = flow_spec.wrap

    if mode == "grid":
        return GridPackStrategy(spacing=gap)
    if mode == "spiral":
        return RadialSpiralStrategy(spacing=max(gap, 4.0))
    if mode == "pack":
        return PackingStrategy(spacing=gap)
    if mode == "smart":
        prefer_h = direction in ("right", "right_wrap")
        return SmartStackStrategy(spacing=gap, prefer_horizontal=prefer_h)
    if mode == "vertical" or direction in ("down", "down_wrap", "up"):
        return VerticalStackStrategy(spacing=gap, wrap=wrap)
    # default: horizontal
    return HorizontalStackStrategy(spacing=gap, wrap=wrap)


# ── Strategy Protocol ─────────────────────────────────────────────────────────

class PlacementStrategy(Protocol):
    def find_position(
        self,
        shape_rect: Rect,
        occupied: List[Rect],
        canvas_bounds: Rect,
        preferred_x: float,
        preferred_y: float,
    ) -> Optional[Tuple[float, float]]: ...


# ── Strategies ────────────────────────────────────────────────────────────────

class OverlapAllowedStrategy:
    """Always returns preferred position (overlap=True / flow disabled)."""
    def find_position(self, shape_rect, occupied, canvas_bounds,
                      preferred_x, preferred_y):
        return (preferred_x, preferred_y)


class HorizontalStackStrategy:
    """Stack left-to-right, wrapping to next row when hitting the edge."""
    def __init__(self, spacing: float = 4.0, wrap: bool = True):
        self.spacing = spacing
        self.wrap = wrap

    def find_position(self, shape_rect: Rect, occupied: List[Rect],
                      canvas_bounds: Rect, preferred_x: float,
                      preferred_y: float) -> Optional[Tuple[float, float]]:
        test_x, test_y = preferred_x, preferred_y
        for _ in range(1000):
            test_rect = shape_rect.move_to(test_x, test_y)
            if not canvas_bounds.contains_rect(test_rect):
                if not self.wrap:
                    return None
                test_x = canvas_bounds.left
                test_y += shape_rect.h + self.spacing
                if test_y + shape_rect.h > canvas_bounds.bottom:
                    return None
                continue
            collision = False
            for occ in occupied:
                if test_rect.intersects(occ.expand(self.spacing / 2)):
                    collision = True
                    test_x = occ.right + self.spacing
                    break
            if not collision:
                return (test_x, test_y)
        return (preferred_x, preferred_y)


class VerticalStackStrategy:
    """Stack top-to-bottom, wrapping to next column when hitting the edge."""
    def __init__(self, spacing: float = 4.0, wrap: bool = True):
        self.spacing = spacing
        self.wrap = wrap

    def find_position(self, shape_rect: Rect, occupied: List[Rect],
                      canvas_bounds: Rect, preferred_x: float,
                      preferred_y: float) -> Optional[Tuple[float, float]]:
        test_x, test_y = preferred_x, preferred_y
        for _ in range(1000):
            test_rect = shape_rect.move_to(test_x, test_y)
            if not canvas_bounds.contains_rect(test_rect):
                if not self.wrap:
                    return None
                test_x += shape_rect.w + self.spacing
                test_y = canvas_bounds.top
                if test_x + shape_rect.w > canvas_bounds.right:
                    return None
                continue
            collision = False
            for occ in occupied:
                if test_rect.intersects(occ.expand(self.spacing / 2)):
                    collision = True
                    test_y = occ.bottom + self.spacing
                    break
            if not collision:
                return (test_x, test_y)
        return (preferred_x, preferred_y)


class GridPackStrategy:
    """Pack shapes into a grid, filling left-to-right then top-to-bottom."""
    def __init__(self, spacing: float = 4.0, columns: Optional[int] = None,
                 start_x: Optional[float] = None, start_y: Optional[float] = None):
        self.spacing = spacing
        self.columns = columns
        self.start_x = start_x
        self.start_y = start_y
        self._row = 0
        self._col = 0

    def find_position(self, shape_rect: Rect, occupied: List[Rect],
                      canvas_bounds: Rect, preferred_x: float,
                      preferred_y: float) -> Optional[Tuple[float, float]]:
        ox = self.start_x if self.start_x is not None else preferred_x
        oy = self.start_y if self.start_y is not None else preferred_y
        if self.columns is None:
            max_cols = max(1, int(
                (canvas_bounds.w - (ox - canvas_bounds.left))
                / (shape_rect.w + self.spacing)
            ))
        else:
            max_cols = self.columns
        for _ in range(1000):
            x = ox + self._col * (shape_rect.w + self.spacing)
            y = oy + self._row * (shape_rect.h + self.spacing)
            test_rect = shape_rect.move_to(x, y)
            if canvas_bounds.contains_rect(test_rect):
                if not any(test_rect.intersects(o.expand(self.spacing / 2)) for o in occupied):
                    self._col += 1
                    if self._col >= max_cols:
                        self._col = 0
                        self._row += 1
                    return (x, y)
            self._col += 1
            if self._col >= max_cols:
                self._col = 0
                self._row += 1
            if y > canvas_bounds.bottom:
                return None
        return (preferred_x, preferred_y)


class RadialSpiralStrategy:
    """Expanding spiral around the preferred position."""
    def __init__(self, spacing: float = 8.0, max_radius: float = 500.0):
        self.spacing = spacing
        self.max_radius = max_radius

    def find_position(self, shape_rect: Rect, occupied: List[Rect],
                      canvas_bounds: Rect, preferred_x: float,
                      preferred_y: float) -> Optional[Tuple[float, float]]:
        test_rect = shape_rect.move_to(preferred_x, preferred_y)
        if self._valid(test_rect, occupied, canvas_bounds):
            return (preferred_x, preferred_y)
        angle, radius = 0.0, self.spacing
        while radius <= self.max_radius:
            x = preferred_x + radius * math.cos(angle)
            y = preferred_y + radius * math.sin(angle)
            test_rect = shape_rect.move_to(x, y)
            if self._valid(test_rect, occupied, canvas_bounds):
                return (x, y)
            angle += math.pi / 8
            if angle >= 2 * math.pi:
                angle = 0.0
                radius += self.spacing
        return None

    def _valid(self, r: Rect, occupied: List[Rect], bounds: Rect) -> bool:
        if not bounds.contains_rect(r):
            return False
        return not any(r.intersects(o.expand(self.spacing / 2)) for o in occupied)


class PackingStrategy:
    """Shelf-packing: fill each shelf before starting the next row."""
    def __init__(self, spacing: float = 4.0):
        self.spacing = spacing
        self._shelves: List[Tuple[float, float, float, float]] = []

    def find_position(self, shape_rect: Rect, occupied: List[Rect],
                      canvas_bounds: Rect, preferred_x: float,
                      preferred_y: float) -> Optional[Tuple[float, float]]:
        if not self._shelves:
            self._shelves.append((canvas_bounds.left, canvas_bounds.top, 0.0, shape_rect.h))
        for i, (sx, sy, sw_used, sh) in enumerate(self._shelves):
            if shape_rect.h <= sh:
                tx, ty = sx + sw_used, sy
                test_rect = shape_rect.move_to(tx, ty)
                if canvas_bounds.contains_rect(test_rect):
                    if not any(test_rect.intersects(o.expand(self.spacing / 2)) for o in occupied):
                        self._shelves[i] = (sx, sy, sw_used + shape_rect.w + self.spacing, sh)
                        return (tx, ty)
        last = self._shelves[-1]
        new_y = last[1] + last[3] + self.spacing
        test_rect = shape_rect.move_to(canvas_bounds.left, new_y)
        if canvas_bounds.contains_rect(test_rect):
            self._shelves.append((canvas_bounds.left, new_y, shape_rect.w + self.spacing, shape_rect.h))
            return (canvas_bounds.left, new_y)
        return None


class SmartStackStrategy:
    """Try horizontal first, fall back to vertical (or vice-versa)."""
    def __init__(self, spacing: float = 4.0, prefer_horizontal: bool = True):
        self.spacing = spacing
        self.prefer_horizontal = prefer_horizontal
        self._h = HorizontalStackStrategy(spacing, wrap=False)
        self._v = VerticalStackStrategy(spacing, wrap=False)

    def find_position(self, shape_rect, occupied, canvas_bounds,
                      preferred_x, preferred_y):
        first, second = (self._h, self._v) if self.prefer_horizontal else (self._v, self._h)
        pos = first.find_position(shape_rect, occupied, canvas_bounds, preferred_x, preferred_y)
        if pos is not None:
            return pos
        pos = second.find_position(shape_rect, occupied, canvas_bounds, preferred_x, preferred_y)
        return pos if pos is not None else (preferred_x, preferred_y)


# ── legacy helper ─────────────────────────────────────────────────────────────

def get_default_strategy(overlap: bool) -> PlacementStrategy:
    """Legacy helper kept for external callers."""
    if overlap:
        return OverlapAllowedStrategy()
    return HorizontalStackStrategy(spacing=4.0, wrap=True)