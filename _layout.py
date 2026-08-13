"""Draw layout containers, geometry, parsing, and rendering helpers.

This module deliberately owns *layout* concerns only.  Widgets and Draw's
shape/text systems remain independent: a layout can host any object and asks
that object for no particular base class.  The painter integration is kept at
the edge of the module so geometry can be reused by ``_room``, ``_list`` and
future widget implementations without creating import cycles.

Public compatibility
--------------------
``Draw.table(...)``, ``Draw.set(...)``, :class:`TableLayout`,
:class:`CombinedCellLayout`, ``cell_rect`` and all accepted table dimension
keys retain their existing behaviour.  ``set`` is still the registry
singleton; its less descriptive name is retained because it is public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

# Qt is only required by the rendering boundary.  QRectF also remains the
# returned geometry type for compatibility with the rest of Draw.
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QPainter, QPen


# ============================================================================
# Geometry and size resolution
# ============================================================================

def _parse_size(value: object, parent_px: int, default_pct: float = 1.0) -> int:
    """Resolve Draw's pixel/percentage size syntax in one canonical place."""
    if value is None:
        return max(1, int(parent_px * default_pct))
    if isinstance(value, (int, float)):
        return max(1, int(value))
    if isinstance(value, str):
        raw = value.strip()
        try:
            if raw.endswith("%"):
                return max(1, int(float(raw[:-1]) / 100.0 * parent_px))
            if raw.endswith("px"):
                return max(1, int(float(raw[:-2])))
            return max(1, int(float(raw)))
        except ValueError:
            return max(1, int(parent_px * default_pct))
    return max(1, int(parent_px * default_pct))


def _parse_cell_ref(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError("Draw.set / Draw.shapes: 'columns' must be a 2-item tuple or list.")
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise TypeError("Draw.set / Draw.shapes: 'columns' values must be integers.") from exc


def _parse_int_value(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.strip().rstrip("%")
    return int(value)


def _parse_margin_value(value: object, default: object = 0) -> object:
    """Keep percentage margins unresolved until a canvas size is available."""
    if value is None or value == "":
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            return raw
        try:
            return int(float(raw.rstrip("px").strip()))
        except ValueError:
            return default
    return default


def _resolve_margin(value: object, canvas_px: int) -> int:
    """Resolve one raw margin value to a non-negative pixel count."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        try:
            amount = float(raw[:-1]) / 100.0 * canvas_px if raw.endswith("%") else float(raw)
            return max(0, int(amount))
        except ValueError:
            pass
    return 0


@dataclass(frozen=True)
class ResolvedMargins:
    """Pixel margins used by a single geometry calculation."""

    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0


@dataclass(frozen=True)
class TableGeometry:
    """A calculated table frame and tracks, independent of rendering."""

    frame: QRectF
    column_widths: tuple[float, ...]
    row_heights: tuple[float, ...]
    horizontal_gap: float
    vertical_gap: float
    origin: str

    def cell_rect(self, column: int, row: int) -> QRectF:
        x = self.frame.left() + sum(self.column_widths[:column]) + column * self.horizontal_gap
        if self.origin == "top-left":
            y = self.frame.top() + sum(self.row_heights[:row]) + row * self.vertical_gap
        else:
            y = self.frame.bottom() - sum(self.row_heights[:row + 1]) - row * self.vertical_gap
        return QRectF(x, y, self.column_widths[column], self.row_heights[row])


# ============================================================================
# Base layout and widget integration
# ============================================================================

@dataclass
class LayoutElement:
    """An optional placement record for a widget, shape, or nested layout.

    ``widget`` is intentionally typed as ``object``.  This keeps `_layout`
    decoupled from `_widget`, `_native`, `_graph`, and future widget modules.
    A renderer may use ``slot`` or ``metadata`` to decide how to host it.
    """

    widget: object
    slot: object = None
    metadata: dict[str, object] = field(default_factory=dict)


class Layout:
    """Common contract for layout containers.

    Subclasses supply geometry; this base supplies child hosting and a single
    structured metadata extension point.  The metadata groups deliberately
    mirror Draw's ecosystem without importing any of its optional systems.
    """

    METADATA_GROUPS = (
        "general", "layout", "style", "color", "motion", "events",
        "physics", "data", "accessibility", "advanced",
    )

    def __init__(self, ip: object = None, *, metadata: Optional[dict[str, object]] = None,
                 elements: Optional[Iterable[LayoutElement | object]] = None) -> None:
        """Create an empty extension layout with the standard metadata groups."""
        self.ip = ip
        self._init_layout_base(metadata, elements)

    def _init_layout_base(self, metadata: Optional[dict[str, object]] = None,
                          elements: Optional[Iterable[LayoutElement | object]] = None) -> None:
        self.metadata: dict[str, dict[str, object]] = {
            group: {} for group in self.METADATA_GROUPS
        }
        if metadata:
            for group, values in metadata.items():
                if isinstance(values, dict):
                    self.metadata.setdefault(str(group), {}).update(values)
        self._elements: list[LayoutElement] = []
        for element in elements or ():
            self.add(element)

    def add(self, widget: LayoutElement | object, *, slot: object = None,
            metadata: Optional[dict[str, object]] = None) -> LayoutElement:
        """Host an object without requiring a widget import or inheritance."""
        element = widget if isinstance(widget, LayoutElement) else LayoutElement(widget, slot, metadata or {})
        self._elements.append(element)
        return element

    def remove(self, widget: object) -> bool:
        for element in self._elements:
            if element.widget is widget:
                self._elements.remove(element)
                return True
        return False

    def elements(self) -> Iterator[LayoutElement]:
        return iter(tuple(self._elements))

    def set_metadata(self, group: str, **values: object) -> None:
        self.metadata.setdefault(group, {}).update(values)

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        raise NotImplementedError

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        raise NotImplementedError


# ============================================================================
# Table / grid layouts
# ============================================================================

@dataclass
class TableMargins:
    """Raw table margins; percentages are resolved only during geometry."""

    top: object = 0
    bottom: object = 0
    left: object = 0
    right: object = 0
    show: bool = False
    cell_margin: int = 0
    cell_margin_horizontal: int = 0
    cell_margin_vertical: int = 0

    def resolve(self, canvas_width: int, canvas_height: int) -> ResolvedMargins:
        return ResolvedMargins(
            top=_resolve_margin(self.top, canvas_height),
            right=_resolve_margin(self.right, canvas_width),
            bottom=_resolve_margin(self.bottom, canvas_height),
            left=_resolve_margin(self.left, canvas_width),
        )


@dataclass
class TableCellStyle:
    """A style override for every cell (``cell=None``) or one cell."""

    cell: Optional[tuple[int, int]]
    values: dict[str, object] = field(default_factory=dict)


@dataclass
class TableLayout(Layout):
    """A rectangular layout container with independently testable geometry."""

    ip: object
    rows: int
    columns: int
    width_raw: object
    height_raw: object
    margins: TableMargins
    cell_styles: list[TableCellStyle] = field(default_factory=list)
    origin: str = "bottom-left"
    # `_list.py` uses these for measured, non-uniform tracks.
    col_widths: Optional[list] = field(default=None)
    row_height: Optional[int] = field(default=None)
    metadata_input: dict[str, object] = field(default_factory=dict, repr=False, compare=False)
    layout_elements: list[LayoutElement | object] = field(default_factory=list, repr=False, compare=False)
    _geometry_cache: dict[tuple[object, ...], TableGeometry] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._init_layout_base(self.metadata_input, self.layout_elements)

    def _cache_key(self, canvas_width: int, canvas_height: int) -> tuple[object, ...]:
        # This signature detects normal direct mutations to the legacy public
        # fields, preventing stale cached geometry without imposing setters.
        margin = self.margins
        return (canvas_width, canvas_height, self.rows, self.columns, self.width_raw,
                self.height_raw, self.origin, margin.top, margin.right, margin.bottom,
                margin.left, margin.cell_margin, margin.cell_margin_horizontal,
                margin.cell_margin_vertical, tuple(self.col_widths or ()), self.row_height)

    def geometry(self, canvas_width: int, canvas_height: int) -> TableGeometry:
        """Return cached, render-free geometry for this canvas size."""
        key = self._cache_key(canvas_width, canvas_height)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached

        margins = self.margins.resolve(canvas_width, canvas_height)
        available_width = max(1, canvas_width - margins.left - margins.right)
        available_height = max(1, canvas_height - margins.top - margins.bottom)
        width = min(available_width, _parse_size(self.width_raw, available_width))
        height = min(available_height, _parse_size(self.height_raw, available_height))
        y = margins.top if self.origin == "top-left" else canvas_height - margins.bottom - height
        frame = QRectF(float(margins.left), float(y), float(width), float(height))
        horizontal_gap = float(self.margins.cell_margin_horizontal or self.margins.cell_margin)
        vertical_gap = float(self.margins.cell_margin_vertical or self.margins.cell_margin)

        if self.col_widths is not None and len(self.col_widths) == self.columns:
            widths = tuple(float(value) for value in self.col_widths)
        else:
            width_per_column = max(0.0, (frame.width() - (self.columns - 1) * horizontal_gap) / self.columns)
            widths = (width_per_column,) * self.columns
        if self.row_height is not None:
            heights = (float(self.row_height),) * self.rows
        else:
            height_per_row = max(0.0, (frame.height() - (self.rows - 1) * vertical_gap) / self.rows)
            heights = (height_per_row,) * self.rows

        geometry = TableGeometry(frame, widths, heights, horizontal_gap, vertical_gap, self.origin)
        # A resized canvas creates a new entry; keep the cache bounded for
        # long-running resizable windows.
        if len(self._geometry_cache) >= 8:
            self._geometry_cache.pop(next(iter(self._geometry_cache)))
        self._geometry_cache[key] = geometry
        return geometry

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        return self.geometry(canvas_width, canvas_height).frame

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        column, row = _parse_cell_ref(cell)
        if not 0 <= column < self.columns:
            raise ValueError(f"Draw.table: column index {column} is out of range for {self.columns} columns.")
        if not 0 <= row < self.rows:
            raise ValueError(f"Draw.table: row index {row} is out of range for {self.rows} rows.")
        return self.geometry(canvas_width, canvas_height).cell_rect(column, row)


class GridLayout(TableLayout):
    """Enhanced grid layout container supporting fractional flex column weights."""

    def geometry(self, canvas_width: int, canvas_height: int) -> TableGeometry:
        key = self._cache_key(canvas_width, canvas_height)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached

        margins = self.margins.resolve(canvas_width, canvas_height)
        available_width = max(1, canvas_width - margins.left - margins.right)
        available_height = max(1, canvas_height - margins.top - margins.bottom)
        width = min(available_width, _parse_size(self.width_raw, available_width))
        height = min(available_height, _parse_size(self.height_raw, available_height))
        y = margins.top if self.origin == "top-left" else canvas_height - margins.bottom - height
        frame = QRectF(float(margins.left), float(y), float(width), float(height))
        horizontal_gap = float(self.margins.cell_margin_horizontal or self.margins.cell_margin)
        vertical_gap = float(self.margins.cell_margin_vertical or self.margins.cell_margin)

        # Support fractional fr weights in col_widths (e.g. ["1fr", "2fr", 100])
        if self.col_widths is not None and len(self.col_widths) == self.columns:
            total_gaps = (self.columns - 1) * horizontal_gap
            fixed_sum = 0.0
            fr_sum = 0.0
            for w_val in self.col_widths:
                if isinstance(w_val, str) and w_val.strip().endswith("fr"):
                    try:
                        fr_sum += float(w_val.strip()[:-2])
                    except ValueError:
                        fixed_sum += 50.0
                elif isinstance(w_val, (int, float)):
                    fixed_sum += float(w_val)
                else:
                    fixed_sum += 50.0
            
            rem_width = max(0.0, frame.width() - total_gaps - fixed_sum)
            calc_widths = []
            for w_val in self.col_widths:
                if isinstance(w_val, str) and w_val.strip().endswith("fr"):
                    try:
                        ratio = float(w_val.strip()[:-2])
                        calc_widths.append((ratio / fr_sum * rem_width) if fr_sum > 0 else 50.0)
                    except ValueError:
                        calc_widths.append(50.0)
                elif isinstance(w_val, (int, float)):
                    calc_widths.append(float(w_val))
                else:
                    calc_widths.append(50.0)
            widths = tuple(calc_widths)
        else:
            width_per_column = max(0.0, (frame.width() - (self.columns - 1) * horizontal_gap) / self.columns)
            widths = (width_per_column,) * self.columns

        if self.row_height is not None:
            heights = (float(self.row_height),) * self.rows
        else:
            height_per_row = max(0.0, (frame.height() - (self.rows - 1) * vertical_gap) / self.rows)
            heights = (height_per_row,) * self.rows

        geometry = TableGeometry(frame, widths, heights, horizontal_gap, vertical_gap, self.origin)
        if len(self._geometry_cache) >= 8:
            self._geometry_cache.pop(next(iter(self._geometry_cache)))
        self._geometry_cache[key] = geometry
        return geometry


@dataclass
class ListLayout(Layout):
    """One-dimensional linear stack layout container (horizontal or vertical)."""

    ip: object = None
    orientation: str = "vertical"
    spacing: int = 4
    width_raw: object = None
    height_raw: object = None
    origin: str = "top-left"
    margins: TableMargins = field(default_factory=TableMargins)
    metadata_input: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._init_layout_base(self.metadata_input)

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        margins = self.margins.resolve(canvas_width, canvas_height)
        avail_w = max(1, canvas_width - margins.left - margins.right)
        avail_h = max(1, canvas_height - margins.top - margins.bottom)
        w = min(avail_w, _parse_size(self.width_raw, avail_w))
        h = min(avail_h, _parse_size(self.height_raw, avail_h))
        y = margins.top if self.origin == "top-left" else canvas_height - margins.bottom - h
        return QRectF(float(margins.left), float(y), float(w), float(h))

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        idx = cell[0] if isinstance(cell, (tuple, list)) else int(cell)
        f = self.frame(canvas_width, canvas_height)
        elems = self._elements
        count = max(1, len(elems))
        if self.orientation == "horizontal":
            item_w = max(0.0, (f.width() - (count - 1) * self.spacing) / count)
            x = f.left() + idx * (item_w + self.spacing)
            return QRectF(x, f.top(), item_w, f.height())
        else:
            item_h = max(0.0, (f.height() - (count - 1) * self.spacing) / count)
            y = f.top() + idx * (item_h + self.spacing)
            return QRectF(f.left(), y, f.width(), item_h)


@dataclass
class StackLayout(Layout):
    """Layered Z-index placement container where children overlay each other."""

    ip: object = None
    width_raw: object = None
    height_raw: object = None
    origin: str = "top-left"
    margins: TableMargins = field(default_factory=TableMargins)
    metadata_input: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._init_layout_base(self.metadata_input)

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        margins = self.margins.resolve(canvas_width, canvas_height)
        avail_w = max(1, canvas_width - margins.left - margins.right)
        avail_h = max(1, canvas_height - margins.top - margins.bottom)
        w = min(avail_w, _parse_size(self.width_raw, avail_w))
        h = min(avail_h, _parse_size(self.height_raw, avail_h))
        y = margins.top if self.origin == "top-left" else canvas_height - margins.bottom - h
        return QRectF(float(margins.left), float(y), float(w), float(h))

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        # All stack layers occupy the full stack frame
        return self.frame(canvas_width, canvas_height)


@dataclass
class FlowLayout(Layout):
    """Wrapping horizontal flow placement container."""

    ip: object = None
    item_width: int = 100
    item_height: int = 40
    horizontal_gap: int = 4
    vertical_gap: int = 4
    width_raw: object = None
    height_raw: object = None
    origin: str = "top-left"
    margins: TableMargins = field(default_factory=TableMargins)
    metadata_input: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._init_layout_base(self.metadata_input)

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        margins = self.margins.resolve(canvas_width, canvas_height)
        avail_w = max(1, canvas_width - margins.left - margins.right)
        avail_h = max(1, canvas_height - margins.top - margins.bottom)
        w = min(avail_w, _parse_size(self.width_raw, avail_w))
        h = min(avail_h, _parse_size(self.height_raw, avail_h))
        y = margins.top if self.origin == "top-left" else canvas_height - margins.bottom - h
        return QRectF(float(margins.left), float(y), float(w), float(h))

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        idx = cell[0] if isinstance(cell, (tuple, list)) else int(cell)
        f = self.frame(canvas_width, canvas_height)
        cols_per_row = max(1, int((f.width() + self.horizontal_gap) / (self.item_width + self.horizontal_gap)))
        row = idx // cols_per_row
        col = idx % cols_per_row
        x = f.left() + col * (self.item_width + self.horizontal_gap)
        y = f.top() + row * (self.item_height + self.vertical_gap)
        return QRectF(x, y, float(self.item_width), float(self.item_height))


@dataclass
class CanvasLayout(Layout):
    """Absolute-positioned child placement container."""

    ip: object = None
    width_raw: object = None
    height_raw: object = None
    origin: str = "top-left"
    margins: TableMargins = field(default_factory=TableMargins)
    metadata_input: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._init_layout_base(self.metadata_input)

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        margins = self.margins.resolve(canvas_width, canvas_height)
        avail_w = max(1, canvas_width - margins.left - margins.right)
        avail_h = max(1, canvas_height - margins.top - margins.bottom)
        w = min(avail_w, _parse_size(self.width_raw, avail_w))
        h = min(avail_h, _parse_size(self.height_raw, avail_h))
        y = margins.top if self.origin == "top-left" else canvas_height - margins.bottom - h
        return QRectF(float(margins.left), float(y), float(w), float(h))

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        f = self.frame(canvas_width, canvas_height)
        if isinstance(cell, (tuple, list)) and len(cell) >= 4:
            return QRectF(f.left() + cell[0], f.top() + cell[1], cell[2], cell[3])
        idx = cell[0] if isinstance(cell, (tuple, list)) else int(cell)
        if idx < len(self._elements):
            meta = self._elements[idx].metadata
            x = float(meta.get("x", 0))
            y = float(meta.get("y", 0))
            w = float(meta.get("width", 100))
            h = float(meta.get("height", 40))
            return QRectF(f.left() + x, f.top() + y, w, h)
        return QRectF(f.left(), f.top(), 100.0, 40.0)


@dataclass
class CombinedCellLayout(Layout):
    """A virtual container spanning selected cells of a base table layout."""

    ip: object
    base: TableLayout
    cells: list
    metadata_input: dict[str, object] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._init_layout_base(self.metadata_input)

    def frame(self, canvas_width: int, canvas_height: int) -> QRectF:
        return self.cell_rect(canvas_width, canvas_height, (0, 0))

    def cell_rect(self, canvas_width: int, canvas_height: int, cell: tuple[int, int]) -> QRectF:
        # `cell` is intentionally ignored: legacy code treats a merged layout
        # as one logical cell.
        rects = [self.base.cell_rect(canvas_width, canvas_height, item) for item in self.cells]
        min_x, min_y = min(rect.left() for rect in rects), min(rect.top() for rect in rects)
        max_x, max_y = max(rect.right() for rect in rects), max(rect.bottom() for rect in rects)
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


# ============================================================================
# Parser: raw specifications -> runtime layouts
# ============================================================================

def _extract_style_values(raw: dict) -> dict[str, object]:
    return {key: value for key, value in raw.items() if key not in {"all", "columns", "cells", "style"}}


def _parse_table_margins(raw: object) -> TableMargins:
    if raw is None:
        return TableMargins()
    if not isinstance(raw, dict):
        raise TypeError("Draw.table: 'margin' must be a dict.")
    all_margin = _parse_margin_value(raw.get("all", 0))
    return TableMargins(
        top=_parse_margin_value(raw.get("top", all_margin), all_margin),
        bottom=_parse_margin_value(raw.get("bottom", all_margin), all_margin),
        left=_parse_margin_value(raw.get("left", all_margin), all_margin),
        right=_parse_margin_value(raw.get("right", all_margin), all_margin),
        show=bool(raw.get("show", False)),
        cell_margin=_parse_int_value(raw.get("cell_margin", 0)),
        cell_margin_horizontal=_parse_int_value(raw.get("cell_margin_horizontal", 0)),
        cell_margin_vertical=_parse_int_value(raw.get("cell_margin_vertical", 0)),
    )


def _parse_table_cell_styles(raw: object) -> list[TableCellStyle]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [style for item in raw for style in _parse_table_cell_styles(item)]
    if not isinstance(raw, dict):
        raise TypeError("Draw.table: 'customise' must be a dict or list of dicts.")
    styles = _parse_table_cell_styles(raw["cells"]) if raw.get("cells") is not None else []
    if "all" in raw:
        if isinstance(raw["all"], dict):
            styles.append(TableCellStyle(None, _extract_style_values(raw["all"])))
        elif bool(raw["all"]):
            styles.append(TableCellStyle(None, _extract_style_values(raw)))
    if "columns" in raw:
        payload = raw.get("style") if isinstance(raw.get("style"), dict) else raw
        styles.append(TableCellStyle(_parse_cell_ref(raw["columns"]), _extract_style_values(payload)))
    if not styles:
        values = _extract_style_values(raw)
        if values:
            styles.append(TableCellStyle(None, values))
    return styles


def _layout_metadata(raw: dict) -> dict[str, object]:
    """Read optional extension metadata without making it part of table math."""
    return {name: raw[name] for name in Layout.METADATA_GROUPS if isinstance(raw.get(name), dict)}


def _parse_table_layout(ip: object, raw: dict) -> TableLayout:
    rows, columns = int(raw.get("rows", raw.get("y", 0))), int(raw.get("columns", raw.get("x", 0)))
    if rows <= 0:
        raise ValueError("Draw.table: 'rows' (or 'y') must be greater than 0.")
    if columns <= 0:
        raise ValueError("Draw.table: 'columns' (or 'x') must be greater than 0.")
    widths = raw.get("col_widths")
    return TableLayout(
        ip, rows, columns, raw.get("width"), raw.get("height"),
        _parse_table_margins(raw.get("margin")), _parse_table_cell_styles(raw.get("customise")),
        str(raw.get("origin", "bottom-left")).strip().lower(),
        [int(width) for width in widths] if isinstance(widths, (list, tuple)) else None,
        int(raw["row_height"]) if raw.get("row_height") is not None else None,
        _layout_metadata(raw),
    )


# ============================================================================
# Rendering helpers (Qt boundary)
# ============================================================================

def _table_style_for_cell(layout: TableLayout, cell: tuple[int, int]) -> dict[str, object]:
    values: dict[str, object] = {}
    if layout.margins.show:
        values.update({"color": None, "border_width": 1, "border_color": "#AAB4C3", "border_style": "solid", "opacity": 100})
    for style in layout.cell_styles:
        if style.cell is None or style.cell == cell:
            values.update(style.values)
    return values


def _draw_one_layout(painter: QPainter, layout: Layout, canvas_width: int, canvas_height: int) -> None:
    """Paint table decoration only; layouts never paint or import widgets."""
    if not isinstance(layout, TableLayout) or not (layout.margins.show or layout.cell_styles):
        return
    # Lazy import keeps the layout core independent of `_window` at import time.
    from Draw._window import _parse_color

    for row in range(layout.rows):
        for column in range(layout.columns):
            style, rect = _table_style_for_cell(layout, (column, row)), layout.cell_rect(canvas_width, canvas_height, (column, row))
            border_width = max(0, int(style.get("border_width", 0)))
            border_style = str(style.get("border_style", "solid")).lower()
            painter.save()
            painter.setOpacity(max(0, min(100, int(style.get("opacity", 100)))) / 100.0)
            painter.setBrush(Qt.BrushStyle.NoBrush if style.get("color") is None else QBrush(_parse_color(style["color"])))
            if border_width > 0 and border_style != "none":
                pen = QPen(_parse_color(style.get("border_color", "#AAB4C3")), border_width)
                pen.setStyle({"solid": Qt.PenStyle.SolidLine, "dashed": Qt.PenStyle.DashLine, "dotted": Qt.PenStyle.DotLine}.get(border_style, Qt.PenStyle.SolidLine))
                painter.setPen(pen)
            else:
                painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            painter.restore()


# ============================================================================
# Registry and compatibility factory
# ============================================================================

LayoutParser = Callable[[object, dict], Layout]


def _parse_list_layout(ip: object, raw: dict) -> ListLayout:
    return ListLayout(
        ip=ip,
        orientation=str(raw.get("orientation", "vertical")).strip().lower(),
        spacing=int(raw.get("spacing", 4)),
        width_raw=raw.get("width"),
        height_raw=raw.get("height"),
        origin=str(raw.get("origin", "top-left")).strip().lower(),
        margins=_parse_table_margins(raw.get("margin")),
        metadata_input=_layout_metadata(raw),
    )


def _parse_stack_layout(ip: object, raw: dict) -> StackLayout:
    return StackLayout(
        ip=ip,
        width_raw=raw.get("width"),
        height_raw=raw.get("height"),
        origin=str(raw.get("origin", "top-left")).strip().lower(),
        margins=_parse_table_margins(raw.get("margin")),
        metadata_input=_layout_metadata(raw),
    )


def _parse_flow_layout(ip: object, raw: dict) -> FlowLayout:
    return FlowLayout(
        ip=ip,
        item_width=int(raw.get("item_width", 100)),
        item_height=int(raw.get("item_height", 40)),
        horizontal_gap=int(raw.get("horizontal_gap", 4)),
        vertical_gap=int(raw.get("vertical_gap", 4)),
        width_raw=raw.get("width"),
        height_raw=raw.get("height"),
        origin=str(raw.get("origin", "top-left")).strip().lower(),
        margins=_parse_table_margins(raw.get("margin")),
        metadata_input=_layout_metadata(raw),
    )


def _parse_canvas_layout(ip: object, raw: dict) -> CanvasLayout:
    return CanvasLayout(
        ip=ip,
        width_raw=raw.get("width"),
        height_raw=raw.get("height"),
        origin=str(raw.get("origin", "top-left")).strip().lower(),
        margins=_parse_table_margins(raw.get("margin")),
        metadata_input=_layout_metadata(raw),
    )


class _SetRegistry:
    """Layout registry and parser dispatch exposed through ``Draw.table``."""

    def __init__(self) -> None:
        self._layouts: dict[object, Layout] = {}
        self._parsers: dict[str, LayoutParser] = {
            "table": _parse_table_layout,
            "grid": _parse_table_layout,
            "list": _parse_list_layout,
            "stack": _parse_stack_layout,
            "flow": _parse_flow_layout,
            "canvas": _parse_canvas_layout,
        }

    def register_parser(self, layout_type: str, parser: LayoutParser) -> None:
        if not callable(parser):
            raise TypeError("Draw.table: layout parser must be callable.")
        self._parsers[str(layout_type).strip().lower()] = parser

    def register(self, layout: Layout, ip: object = None) -> Layout:
        key = layout.ip if ip is None else ip
        if key is not None:
            try:
                self._layouts[key] = layout
            except TypeError:
                # Existing behaviour: anonymous/unhashable ids still return a layout.
                pass
        return layout

    def __call__(self, *, ip: object = None, tag: object = None, dimension: Optional[dict] = None,
                 dimention: Optional[dict] = None, cell_combine: object = None,
                 join: Optional[dict] = None) -> Layout:
        effective_ip = ip if ip is not None else tag
        if cell_combine is not None:
            if not isinstance(cell_combine, (list, tuple)) or not cell_combine:
                raise ValueError("Draw.table: 'cell_combine' must be a non-empty list of (col, row) tuples.")
            base = self.resolve(effective_ip)
            if not isinstance(base, TableLayout):
                raise TypeError("Draw.table: 'cell_combine' requires a TableLayout.")
            return self.register(CombinedCellLayout(effective_ip, base, [_parse_cell_ref(cell) for cell in cell_combine]))

        raw = dimension if dimension is not None else dimention
        if raw is None:
            raise ValueError("Draw.table: 'dimension' (or 'dimention') is required.")
        if not isinstance(raw, dict):
            raise TypeError("Draw.table: 'dimension' must be a dict.")
        layout_type = raw.get("type", "table")
        layout_type = str(layout_type[0] if isinstance(layout_type, (list, tuple)) and layout_type else layout_type).strip().lower()
        parser = self._parsers.get(layout_type)
        if parser is None:
            raise ValueError(f"Draw.table: unsupported dimension type={layout_type!r}.")
        layout = self.register(parser(effective_ip, raw))
        if join is not None:
            self._register_joined_cells(layout, join)
        return layout

    def _register_joined_cells(self, layout: Layout, join: object) -> None:
        if not isinstance(join, dict):
            raise TypeError("Draw.table: 'join' must be a dict.")
        if not isinstance(layout, TableLayout):
            raise TypeError("Draw.table: 'join' requires a TableLayout.")
        for key, value in join.items():
            if not isinstance(value, (list, tuple)):
                continue
            cells, name = [], None
            for item in value:
                if isinstance(item, str):
                    name = item
                else:
                    try:
                        cells.append(_parse_cell_ref(item))
                    except (TypeError, ValueError):
                        pass
            if not name and isinstance(key, str) and key != "column":
                name = key
            if cells:
                merged = CombinedCellLayout(name or key, layout, cells)
                if name:
                    self.register(merged, name)
                self.register(merged, key)

    def resolve(self, ref: object) -> Layout:
        if isinstance(ref, Layout):
            return ref
        try:
            return self._layouts[ref]
        except TypeError as exc:
            raise TypeError("Draw.shapes: 'get_ip' must be a Draw.table(...) layout or a hashable ip.") from exc
        except KeyError as exc:
            raise KeyError(f"Draw.shapes: no layout registered for get_ip={ref!r}.") from exc


set = _SetRegistry()
table = set


__all__ = [
    "Layout", "LayoutElement", "TableLayout", "GridLayout", "ListLayout", "StackLayout",
    "FlowLayout", "CanvasLayout", "CombinedCellLayout", "TableMargins", "TableCellStyle",
    "TableGeometry", "set", "table",
]
