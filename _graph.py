"""
Draw._graph
===========
Comprehensive chart rendering engine built on top of Draw.shapes, Draw.text, Draw.point,
Draw._motion, Draw.senses, Draw._align, Draw.room, Draw._live, and Draw._colour.

Supports:
- Chart Types: bar, stacked_bar, line, area, stacked_area, dot/scatter, pie, donut, radar/spider
- Reactive Pipeline: name of graph -> Draw.color (_colour.py) -> Draw.live (_live.py)
- Universal Color System integration (Draw.color string IP bindings, HSV math expressions, live color arrays)
- Pie & Donut Circle Fitting with dynamic inner radius & title clearance
- Live Data Updates via Draw._live (LiveRef, dynamic callbacks, Draw.live.set, Draw.graph.update_live)
- Interactive Hold & Move (chart dragging across canvas)
- 2D Motion Engine integration via Draw._motion (animated chart entry, springs, path trimming)
- Automatic hover highlights and floating tooltip cards via Draw.senses & Draw.connectors
- Stateful _GraphRegistry for Draw.room layout engine integration & _room_size interop
- Alignment anchoring via Draw._align
- Multi-series auto-legend generation & multi-series scaling
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from Draw._window import window as _window_registry
from Draw._shapes import shapes as _shapes_registry
from Draw._colour import _parse_color
from Draw._tools import safe_cast, is_color, distance, angle, is_number
from Draw._align import calculate_alignment_pos, calculate_chart_region, calculate_radial_center, zlayer, ZLayer
from Draw._layout import set as _layout_registry
from Draw._validation import validate_keys
from Draw._live import live as _live_registry, LiveRef, LiveTextBinding, resolve_live_text, is_live_text_binding


_GRAPH_TYPES = {
    "bar", "stacked_bar", "line", "area", "stacked_area",
    "dot", "scatter", "pie", "donut", "radar", "spider"
}

_SHAPE_PASSTHROUGH_KEYS = {
    "curve_mode", "bend", "bend_amount", "warp", "exclude", "symmetry",
    "border_style", "opacity", "type", "src", "hitbox_mode", "hit_box",
}

KNOWN_GRAPH_CUSTOMISE_KEYS: frozenset = frozenset({
    "margin", "show_line", "title", "x_title", "y_title",
    "title_font_size", "title_color", "axis_title_size", "axis_title_color",
    "grid_lines", "grid_color", "grid_width", "axis_color",
    "y_tick_color", "max_x", "min_y", "max_y",
    "color", "border_width", "border_color", "border_style", "border_radius",
    "bar_width", "line_width", "line_color", "area_opacity",
    "point_size", "point_vertices",
    "label_color", "label_font_size",
    "show_values", "value_color", "value_font_size", "value_decimals",
    "group_width_ratio", "flow",
    # Live & Interactivity
    "live", "live_key", "draggable", "hold_and_move",
    # Motion & Animation
    "animate", "animation_type", "duration", "stiffness", "damping",
    # Hover & Tooltip
    "hover", "tooltip", "tooltip_color", "tooltip_bg", "tooltip_font_size",
    # Pie & Donut & Radar
    "inner_radius", "hole_radius", "hole_color", "center_text", "center_subtext",
    "spider_grid", "spider_levels",
    # Line chart vector path options
    "smooth", "curve",
    # Legend
    "show_legend", "legend_position", "legend_color", "legend_font_size",
    # Shape passthrough keys
    "curve_mode", "bend", "bend_amount", "warp", "exclude", "symmetry",
    "opacity", "type", "src", "hitbox_mode", "hit_box",
    # Color extensions
    "gradient", "stops", "color_ip",
    # Dimension / Coordinate / Alignment overrides
    "width", "height", "x", "y", "size", "align",
})


# ── Live Data Resolver ────────────────────────────────────────────────────────

def resolve_live_data(raw: Any) -> Any:
    """Unwrap LiveRef, LiveTextBinding, callables, or lists of live references."""
    if isinstance(raw, LiveRef):
        return _live_registry.get(raw.key, [])
    if is_live_text_binding(raw):
        return resolve_live_text(raw)
    if callable(raw):
        return raw()
    if isinstance(raw, list):
        resolved_list = []
        for item in raw:
            if isinstance(item, LiveRef):
                resolved_list.append(_live_registry.get(item.key, item))
            elif is_live_text_binding(item):
                resolved_list.append(resolve_live_text(item))
            elif callable(item):
                resolved_list.append(item())
            else:
                resolved_list.append(item)
        return resolved_list
    return raw


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ── GraphDef ─────────────────────────────────────────────────────────────────

@dataclass
class GraphDef:
    """Internal record for one rendered graph instance, stored in _GraphRegistry."""
    ip: str
    window_tag: str
    bounds: Tuple[float, float, float, float]
    series_data: List[dict]
    customise: dict
    offset_x: float = 0.0
    offset_y: float = 0.0
    graph_type: str = "bar"
    get_ip: object = None
    columns: object = None
    align: object = None
    animate: bool = False
    hover: bool = True
    tooltip: bool = True
    draggable: bool = False
    child_ips: Set[str] = field(default_factory=set)
    child_text_ips: Set[str] = field(default_factory=set)
    child_point_ips: Set[str] = field(default_factory=set)
    element_info: dict = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.graph_type

    @property
    def series(self) -> List[dict]:
        return self.series_data


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _normalize_graph_type(raw: object, *, field_name: str) -> str:
    token = str(raw or "").strip().lower()
    if token == "scatter":
        token = "dot"
    elif token == "spider":
        token = "radar"
    if token not in _GRAPH_TYPES:
        allowed = ", ".join(sorted(_GRAPH_TYPES))
        raise ValueError(f"Draw.graph: {field_name} must be one of [{allowed}].")
    return token


def _as_float(value: object, *, field: str) -> float:
    result = safe_cast(value, float, None)
    if result is None:
        raise TypeError(f"Draw.graph: {field} must contain only numeric values.")
    return result


def _as_int(value: object, *, field: str, default: int) -> int:
    if value is None:
        return default
    result = safe_cast(value, int, None)
    if result is None:
        raise TypeError(f"Draw.graph: '{field}' must be an integer.")
    return result


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _make_value_label(value: float, decimals: int) -> str:
    if decimals <= 0 and float(value).is_integer():
        return str(int(value))
    return f"{value:.{max(0, decimals)}f}"


def _extract_color(c: dict, key: str, default: str) -> object:
    val = resolve_live_data(c.get(key, default))
    if isinstance(val, list):
        return [resolve_live_data(v) for v in val]
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        from Draw._colour import color as _color_registry
        if _color_registry.has_binding(val):
            resolved = _color_registry.resolve_for_shape(val)
            if resolved and "body_color" in resolved:
                r, g, b, a = resolved["body_color"]
                return f"rgba({r},{g},{b},{a/255.0:.2f})"
    if not is_color(val):
        warnings.warn(
            f"Draw.graph: Invalid color '{val}' for {key}, falling back to '{default}'.",
            UserWarning,
            stacklevel=3,
        )
        return default
    return val


# ── Legend helpers ────────────────────────────────────────────────────────────

def _build_legend(
    *,
    namespace: str = "legend",
    series_list: List[dict],
    get_color_fn,
    left: float,
    top: float,
    chart_w: float,
    chart_h: float,
    legend_position: str,
    legend_color: object,
    legend_font_size: int,
    value_decimals: int,
) -> Tuple[List[dict], List[dict]]:
    """Return (shapes, texts) for a multi-series chart legend."""
    swatch_size = 12
    row_h = legend_font_size + 6
    padding = 10
    max_title_len = max(
        (len(str(s.get("title", s.get("name", f"Series {i + 1}"))).strip())
         for i, s in enumerate(series_list)),
        default=10,
    )
    col_w = max(120, int(max_title_len * legend_font_size * 0.6 + 36))

    n = len(series_list)
    legend_h = n * row_h + padding * 2

    pos = str(legend_position or "top-right").strip().lower()
    if pos == "outside-right":
        # Special case: legend sits outside the chart area to the right
        lx = left + chart_w + padding * 2
        ly = top + padding
    else:
        # ── Universal layout: delegate to _align.py ──
        lx, ly = calculate_alignment_pos(
            pos, sw=float(col_w), sh=float(legend_h),
            cw=chart_w, ch=chart_h, pad=float(padding),
            offset_x=left, offset_y=top,
        )

    shapes: List[dict] = []
    texts: List[dict] = []

    shapes.append({
        "vertices": 4,
        "border_radius": "4px",
        "width": f"{col_w}px",
        "height": f"{int(legend_h)}px",
        "ip": f"{namespace}:legend_bg",
        "customise": {
            "x": int(lx),
            "y": int(ly),
            "color": "#1A2333",
            "opacity": 75,
            "overlap": True,
            "z": zlayer(ZLayer.LEGEND_BG),
        },
    })

    for i, series in enumerate(series_list):
        title = str(series.get("title", series.get("name", f"Series {i + 1}"))).strip()
        c = get_color_fn(i)
        sy = ly + padding + i * row_h
        shapes.append({
            "vertices": 4,
            "border_radius": "2px",
            "width": f"{swatch_size}px",
            "height": f"{swatch_size}px",
            "ip": f"{namespace}:legend_swatch:{i}",
            "customise": {
                "x": int(lx + 8),
                "y": int(sy + (legend_font_size - swatch_size) / 2 + 2),
                "color": c,
                "overlap": True,
                "z": zlayer(ZLayer.LEGEND_ITEM, i),
            },
        })
        texts.append({
            "text": title,
            "ip": f"{namespace}:legend_text:{i}",
            "customise": {
                "x": int(lx + 8 + swatch_size + 6),
                "y": int(sy + 2),
                "font_size": legend_font_size,
                "color": legend_color,
                "z": zlayer(ZLayer.LEGEND_ITEM, i),
            },
        })

    return shapes, texts


# ── Main registry ─────────────────────────────────────────────────────────────

class _GraphRegistry:
    """
    Singleton exposed as Draw.graph.

    Supports live data updating via Draw._live (LiveRef, dynamic callbacks, Draw.live.set),
    interactive Hold & Move canvas dragging, 2D Motion animations, automatic hover tooltips,
    alignment positioning, and stateful Draw.room layout engine integration.
    """

    def __init__(self) -> None:
        # (window_tag, ip) -> GraphDef
        self._registry: Dict[Tuple[str, str], GraphDef] = {}
        # Active hover senses listeners
        self._hover_listeners: Dict[str, Any] = {}
        # Active drag listeners for Hold & Move
        self._drag_listeners: Dict[str, Any] = {}

    # ── public call ──────────────────────────────────────────────────────

    def __call__(
        self,
        *,
        ip: object = "",
        graph_ip: object = None,
        type: str = "bar",
        graph: Optional[List[dict]] = None,
        tag: Optional[str] = None,
        display: Optional[str] = None,
        align: object = None,
        get_ip: object = None,
        columns: object = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        size: Optional[Union[List[float], Tuple[float, float]]] = None,
        animate: bool = False,
        hover: bool = True,
        tooltip: bool = True,
        draggable: bool = False,
        hold_and_move: bool = False,
        customise: Optional[dict] = None,
        **kwargs
    ) -> None:
        """
        Render a graph to the screen.
        """
        default_type = _normalize_graph_type(type, field_name="'type'")
        if not isinstance(graph, list):
            raise TypeError("Draw.graph: 'graph' must be a list of dicts.")

        if isinstance(customise, dict):
            if "width" in customise and width is None:
                width = customise["width"]
            if "height" in customise and height is None:
                height = customise["height"]
            if "x" in customise and x is None:
                x = customise["x"]
            if "y" in customise and y is None:
                y = customise["y"]
            if "size" in customise and size is None:
                size = customise["size"]
            if "align" in customise and align is None:
                align = customise["align"]
        if kwargs:
            if "width" in kwargs and width is None:
                width = kwargs["width"]
            if "height" in kwargs and height is None:
                height = kwargs["height"]
            if "x" in kwargs and x is None:
                x = kwargs["x"]
            if "y" in kwargs and y is None:
                y = kwargs["y"]

        if graph_ip is not None:
            warnings.warn(
                "Draw.graph: 'graph_ip' is deprecated, use 'ip' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        effective_ip = graph_ip if graph_ip is not None else ip

        window_tag = display if display is not None else tag
        if window_tag is None:
            window_tag = str(effective_ip or "").strip()
        if not window_tag:
            raise ValueError("Draw.graph: provide 'tag' or 'display' (or a non-empty 'ip').")

        win = _window_registry.get(window_tag)
        plot_w = max(1.0, float(win.width()))
        plot_h = max(1.0, float(win.height()))

        cell_x, cell_y = 0.0, 0.0
        cell_w, cell_h = plot_w, plot_h

        if get_ip is not None:
            layout = _layout_registry.resolve(get_ip)
            if columns is not None:
                rect = layout.cell_rect(plot_w, plot_h, columns)
                cell_x, cell_y = float(rect.x()), float(rect.y())
                cell_w, cell_h = float(rect.width()), float(rect.height())
        else:
            # Direct region size/position overrides
            if size is not None and isinstance(size, (list, tuple)) and len(size) >= 2:
                cell_w = max(1.0, float(size[0]))
                cell_h = max(1.0, float(size[1]))
            if width is not None:
                cell_w = max(1.0, float(width))
            if height is not None:
                cell_h = max(1.0, float(height))
            if align is not None:
                from Draw._align import calculate_alignment_pos
                cell_x, cell_y = calculate_alignment_pos(
                    str(align), sw=float(cell_w), sh=float(cell_h),
                    cw=plot_w, ch=plot_h, window_tag=window_tag
                )
            if x is not None:
                cell_x = float(x)
            if y is not None:
                cell_y = float(y)

        # Retrieve drag offsets if graph was previously moved via Hold & Move
        ip_str = str(effective_ip or "").strip()
        existing = self._registry.get((window_tag, ip_str))
        off_x = existing.offset_x if existing else 0.0
        off_y = existing.offset_y if existing else 0.0
        cell_x += off_x
        cell_y += off_y

        entries: List[dict] = []
        for series_index, series in enumerate(graph):
            if not isinstance(series, dict):
                raise TypeError("Draw.graph: each item in 'graph' must be a dict.")

            # Resolve live data bindings for series 'x' and 'y' via Draw._live
            resolved_series = dict(series)
            if "x" in series:
                resolved_series["x"] = resolve_live_data(series["x"])
            if "y" in series:
                resolved_series["y"] = resolve_live_data(series["y"])

            series_type = _normalize_graph_type(
                resolved_series.get("type", default_type),
                field_name=f"graph[{series_index}].type",
            )
            entries.append({
                "series_index": series_index,
                "series_type": series_type,
                "series": resolved_series,
            })

        if len(entries) == 0:
            return

        bar_entries = [item for item in entries if item["series_type"] == "bar"]
        bar_rank_by_index = {
            item["series_index"]: rank for rank, item in enumerate(bar_entries)
        }
        total_bar_series = max(1, len(bar_entries))

        top_c_dict = dict(customise) if isinstance(customise, dict) else {}
        global_slots = 1
        first_customise: dict = dict(top_c_dict)
        for item in entries:
            series = item["series"]
            x_raw = series.get("x")
            if not isinstance(x_raw, list):
                raise TypeError("Draw.graph: series 'x' must be a list.")
            c = series.get("customise", {}) or {}
            if not isinstance(c, dict):
                raise TypeError("Draw.graph: series 'customise' must be a dict.")
            merged_c = dict(top_c_dict)
            merged_c.update(c)
            series["customise"] = merged_c
            item["series"]["customise"] = merged_c
            global_slots = max(global_slots, len(x_raw))
            if "max_x" in merged_c:
                global_slots = max(
                    global_slots,
                    _as_int(merged_c.get("max_x"), field="max_x", default=len(x_raw)),
                )
            if not first_customise:
                first_customise = merged_c

        namespace = str(effective_ip or window_tag)
        chart_bounds: Optional[Tuple[float, float, float, float]] = None

        is_draggable = draggable or hold_and_move or _as_bool(first_customise.get("draggable", False)) or _as_bool(first_customise.get("hold_and_move", False))

        first_type = entries[0]["series_type"]
        if first_type in {"pie", "donut"}:
            chart_bounds = self._render_pie_donut(
                window_tag=window_tag,
                namespace=namespace,
                graph_type=first_type,
                series_list=[e["series"] for e in entries],
                cell_w=cell_w,
                cell_h=cell_h,
                cell_x=cell_x,
                cell_y=cell_y,
                align=align,
                animate=animate or _as_bool(first_customise.get("animate", False)),
                enable_hover=hover and _as_bool(first_customise.get("hover", True), default=True),
                enable_tooltip=tooltip and _as_bool(first_customise.get("tooltip", True), default=True),
                draggable=is_draggable,
                ip_str=ip_str,
            )
        elif first_type in {"radar", "spider"}:
            chart_bounds = self._render_radar(
                window_tag=window_tag,
                namespace=namespace,
                series_list=[e["series"] for e in entries],
                cell_w=cell_w,
                cell_h=cell_h,
                cell_x=cell_x,
                cell_y=cell_y,
                align=align,
                animate=animate or _as_bool(first_customise.get("animate", False)),
                enable_hover=hover and _as_bool(first_customise.get("hover", True), default=True),
                enable_tooltip=tooltip and _as_bool(first_customise.get("tooltip", True), default=True),
                draggable=is_draggable,
                ip_str=ip_str,
            )
        else:
            # Cartesian charts (bar, stacked_bar, line, area, stacked_area, dot, scatter)
            for idx, item in enumerate(entries):
                series_index = int(item["series_index"])
                bounds = self._render_cartesian_series(
                    window_tag=window_tag,
                    namespace=namespace,
                    graph_type=item["series_type"],
                    series=item["series"],
                    series_index=series_index,
                    all_series=[e["series"] for e in entries],
                    cell_w=cell_w,
                    cell_h=cell_h,
                    cell_x=cell_x,
                    cell_y=cell_y,
                    align=align,
                    slot_count=global_slots,
                    bar_rank=bar_rank_by_index.get(series_index, 0),
                    bar_series_total=total_bar_series,
                    emit_frame=(idx == 0),
                    emit_x_labels=(idx == 0),
                    emit_legend=(idx == 0),
                    animate=animate or _as_bool(item["series"].get("customise", {}).get("animate", False)),
                    enable_hover=hover and _as_bool(item["series"].get("customise", {}).get("hover", True), default=True),
                    enable_tooltip=tooltip and _as_bool(item["series"].get("customise", {}).get("tooltip", True), default=True),
                    draggable=is_draggable,
                    ip_str=ip_str,
                )
                if idx == 0 and bounds is not None:
                    chart_bounds = bounds

        # Store in registry for room-engine interop
        if ip_str and chart_bounds is not None:
            gdef = GraphDef(
                ip=ip_str,
                window_tag=window_tag,
                bounds=chart_bounds,
                series_data=graph,
                customise=first_customise,
                offset_x=off_x,
                offset_y=off_y,
                graph_type=default_type,
                get_ip=get_ip,
                columns=columns,
                align=align,
                animate=animate,
                hover=hover,
                tooltip=tooltip,
                draggable=is_draggable,
            )
            gdef.child_ips.add(f"{namespace}:graph_container")
            gdef.child_ips.add(f"{namespace}:legend_bg")
            gdef.child_ips.add(f"{namespace}:hole")
            for i in range(len(graph)):
                gdef.child_ips.add(f"{namespace}:legend_swatch:{i}")
                gdef.child_text_ips.add(f"{namespace}:legend_text:{i}")
            gdef.child_text_ips.add(f"{namespace}:center_text")
            self._registry[(window_tag, ip_str)] = gdef

    # ── Cartesian series renderer ───────────────────────────────────────────

    def _render_cartesian_series(
        self,
        *,
        window_tag: str,
        namespace: str,
        graph_type: str,
        series: dict,
        series_index: int,
        all_series: List[dict],
        cell_w: float,
        cell_h: float,
        cell_x: float,
        cell_y: float,
        align: object,
        slot_count: int,
        bar_rank: int,
        bar_series_total: int,
        emit_frame: bool,
        emit_x_labels: bool,
        emit_legend: bool,
        animate: bool,
        enable_hover: bool,
        enable_tooltip: bool,
        draggable: bool,
        ip_str: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Render cartesian chart series (bar, line, area, dot)."""
        x_raw = series.get("x")
        y_raw = series.get("y")
        if not isinstance(x_raw, list):
            raise TypeError("Draw.graph: series 'x' must be a list.")
        if not isinstance(y_raw, list):
            raise TypeError("Draw.graph: series 'y' must be a list.")
        if len(x_raw) != len(y_raw):
            raise ValueError("Draw.graph: series 'x' and 'y' must have the same length.")
        if len(x_raw) == 0:
            return None

        y_values = [_as_float(v, field="y") for v in y_raw]
        x_labels = [str(v) for v in x_raw]

        c = series.get("customise", {}) or {}
        if not isinstance(c, dict):
            raise TypeError("Draw.graph: series 'customise' must be a dict.")

        validate_keys(c, KNOWN_GRAPH_CUSTOMISE_KEYS, kind="Draw.graph")

        margin = max(0, _as_int(c.get("margin"), field="margin", default=48))

        # Check layout properties across all series so all series share identical padding & bounding box
        has_show_line = any(_as_bool((s.get("customise", {}) or {}).get("show_line", False)) for s in all_series)
        has_show_legend = any(_as_bool((s.get("customise", {}) or {}).get("show_legend", False)) for s in all_series)
        title_str = next((str((s.get("customise", {}) or {}).get("title", "")).strip() for s in all_series if str((s.get("customise", {}) or {}).get("title", "")).strip()), "")
        x_title_str = next((str((s.get("customise", {}) or {}).get("x_title", "")).strip() for s in all_series if str((s.get("customise", {}) or {}).get("x_title", "")).strip()), "")
        y_title_str = next((str((s.get("customise", {}) or {}).get("y_title", "")).strip() for s in all_series if str((s.get("customise", {}) or {}).get("y_title", "")).strip()), "")

        show_line = _as_bool(c.get("show_line", has_show_line), default=False)
        title = str(c.get("title", title_str)).strip()
        x_title = str(c.get("x_title", x_title_str)).strip()
        y_title = str(c.get("y_title", y_title_str)).strip()

        # ── Universal layout: delegate to _align.py ──
        left, top, right, bottom, chart_w, chart_h = calculate_chart_region(
            cell_x, cell_y, cell_w, cell_h,
            margin=float(margin),
            has_y_axis=(has_show_line or bool(y_title_str)),
            has_title=bool(title_str),
            has_x_title=bool(x_title_str),
            align=align,
            window_tag=window_tag,
        )
        step_x = chart_w / max(1, slot_count)

        # Collect data range across all series for unified Y scaling
        all_y_vals: List[float] = []
        for s in all_series:
            raw_y = s.get("y", [])
            if isinstance(raw_y, list):
                all_y_vals.extend(_as_float(v, field="y") for v in raw_y)
        if not all_y_vals:
            all_y_vals = y_values
        data_min = min(all_y_vals)
        data_max = max(all_y_vals)

        has_bar_or_area = any(
            _normalize_graph_type(s.get("type", "bar"), field_name="type") in {"bar", "stacked_bar", "area", "stacked_area"}
            for s in all_series
        )

        min_y_raw = c.get("min_y", None)
        max_y_raw = c.get("max_y", None)
        if min_y_raw is not None:
            min_y = _as_float(min_y_raw, field="min_y")
        elif has_bar_or_area:
            min_y = min(0.0, data_min)
        else:
            data_span = abs(data_max - data_min) if data_max != data_min else abs(data_min) or 1.0
            min_y = data_min - data_span * 0.05
        if max_y_raw is not None:
            max_y = _as_float(max_y_raw, field="max_y")
        elif has_bar_or_area:
            data_top = max(0.0, data_max)
            max_y = data_top + max(1.0, abs(data_top - min_y) * 0.08)
        else:
            data_span = abs(data_max - data_min) if data_max != data_min else abs(data_max) or 1.0
            max_y = data_max + data_span * 0.05
        if max_y <= min_y:
            max_y = min_y + 1.0
        y_range = max_y - min_y

        def y_to_px(y: float) -> float:
            ratio = (y - min_y) / y_range
            return bottom - (ratio * chart_h)

        baseline_y = y_to_px(0.0)

        color_val = _extract_color(c, "color", "blue")
        border_width = _as_int(c.get("border_width"), field="border_width", default=0)
        border_color = _extract_color(c, "border_color", "black")
        bar_width_input = c.get("bar_width", None)
        line_width = max(1, _as_int(c.get("line_width"), field="line_width", default=3))
        point_size_raw = c.get("point_size", None)
        point_vertices = max(3, _as_int(c.get("point_vertices"), field="point_vertices", default=24))
        label_color = _extract_color(c, "label_color", "#E0E0E0")
        label_font_size = max(8, _as_int(c.get("label_font_size"), field="label_font_size", default=11))
        show_values = _as_bool(c.get("show_values", False), default=False)
        value_color = _extract_color(c, "value_color", "#BBD4FF")
        value_font_size = max(8, _as_int(c.get("value_font_size"), field="value_font_size", default=10))
        value_decimals = _as_int(c.get("value_decimals"), field="value_decimals", default=0)
        use_curve = _as_bool(c.get("smooth", c.get("curve", False)), default=False)

        def get_color(idx: int) -> object:
            if isinstance(color_val, list):
                return color_val[idx % len(color_val)] if len(color_val) > 0 else "blue"
            return color_val

        shapes: List[dict] = []
        texts: List[dict] = []
        points: List[Tuple[float, float]] = []

        container_ip = f"{namespace}:graph_container"
        if emit_frame:
            shapes.append({
                "vertices": 4,
                "width": f"{int(cell_w)}px",
                "height": f"{int(cell_h)}px",
                "ip": container_ip,
                "customise": {
                    "x": int(cell_x),
                    "y": int(cell_y),
                    "color": "transparent",
                    "opacity": 0,
                    "overlap": True,
                    "z": zlayer(ZLayer.CHART_GRID),
                },
            })

            # ── Attach Hold & Move drag listener ──
            if draggable and ip_str:
                self._attach_drag_listener(
                    window_tag=window_tag,
                    container_ip=container_ip,
                    graph_ip=ip_str,
                )

        if emit_frame and show_line:
            grid_lines = max(2, _as_int(c.get("grid_lines"), field="grid_lines", default=5))
            grid_color = _extract_color(c, "grid_color", "#D6DDE6")
            grid_width = max(1, _as_int(c.get("grid_width"), field="grid_width", default=1))
            axis_color = _extract_color(c, "axis_color", "#8DA0B8")
            y_tick_color = _extract_color(c, "y_tick_color", "#9CAEC2")

            for gi in range(grid_lines + 1):
                gy = top + (chart_h * gi / grid_lines)
                tick_val = max_y - (y_range * gi / grid_lines)
                shapes.append({
                    "vertices": 4,
                    "border_radius": "1px",
                    "width": f"{int(chart_w)}px",
                    "height": f"{grid_width}px",
                    "ip": f"{namespace}:grid_line:{gi}",
                    "customise": {
                        "x": int(left),
                        "y": int(round(gy)),
                        "color": grid_color,
                        "opacity": 60,
                        "overlap": True,
                        "z": zlayer(ZLayer.CHART_GRID, gi),
                    },
                })
                texts.append({
                    "text": _make_value_label(tick_val, value_decimals),
                    "ip": f"{namespace}:grid_tick:{gi}",
                    "customise": {
                        "x": int(left - 48),
                        "y": int(round(gy - 8)),
                        "font_size": 10,
                        "color": y_tick_color,
                        "z": zlayer(ZLayer.CHART_AXIS, gi),
                    },
                })

            shapes.append({
                "vertices": 4,
                "border_radius": "1px",
                "width": f"{int(chart_w)}px",
                "height": "2px",
                "ip": f"{namespace}:axis_baseline",
                "customise": {
                    "x": int(left),
                    "y": int(round(baseline_y)),
                    "color": axis_color,
                    "overlap": True,
                    "z": zlayer(ZLayer.CHART_AXIS),
                },
            })
            shapes.append({
                "vertices": 4,
                "border_radius": "1px",
                "width": "2px",
                "height": f"{int(chart_h)}px",
                "ip": f"{namespace}:y_axis_line",
                "customise": {
                    "x": int(left),
                    "y": int(top),
                    "color": axis_color,
                    "overlap": True,
                    "z": zlayer(ZLayer.CHART_AXIS),
                },
            })

        if emit_frame:
            if title:
                texts.append({
                    "text": title,
                    "ip": f"{namespace}:title",
                    "customise": {
                        "x": int(left),
                        "y": int(top - 28),
                        "font_size": max(10, _as_int(c.get("title_font_size"), field="title_font_size", default=16)),
                        "color": _extract_color(c, "title_color", "#DCEBFF"),
                        "bold": True,
                        "z": zlayer(ZLayer.CHART_LABEL),
                    },
                })
            if x_title:
                texts.append({
                    "text": x_title,
                    "ip": f"{namespace}:x_title",
                    "customise": {
                        "x": int(left + chart_w / 2 - 40),
                        "y": int(bottom + 28),
                        "font_size": max(9, _as_int(c.get("axis_title_size"), field="axis_title_size", default=12)),
                        "color": _extract_color(c, "axis_title_color", "#AEC9E7"),
                        "z": zlayer(ZLayer.CHART_AXIS),
                    },
                })
            if y_title:
                texts.append({
                    "text": y_title,
                    "ip": f"{namespace}:y_title",
                    "customise": {
                        "x": int(left - 30),
                        "y": int(top + chart_h / 2),
                        "rotation": -90,
                        "font_size": max(9, _as_int(c.get("axis_title_size"), field="axis_title_size", default=12)),
                        "color": _extract_color(c, "axis_title_color", "#AEC9E7"),
                        "z": zlayer(ZLayer.CHART_AXIS),
                    },
                })

        group_ratio = max(0.1, min(1.0, float(c.get("group_width_ratio", 0.82))))
        group_width = step_x * group_ratio
        slot_bar_band = group_width / max(1, bar_series_total)
        if bar_width_input is None:
            bar_width = slot_bar_band
        else:
            bar_width = min(slot_bar_band, max(2.0, float(bar_width_input)))

        for idx, (label, y_val) in enumerate(zip(x_labels, y_values)):
            center_x = left + (idx + 0.5) * step_x
            point_y = y_to_px(y_val)
            item_color = get_color(idx)

            if graph_type in {"bar", "stacked_bar"}:
                bar_h = abs(baseline_y - point_y)
                bar_top = min(baseline_y, point_y)
                group_left = center_x - (group_width / 2.0)
                bar_x = group_left + (bar_rank * slot_bar_band) + (slot_bar_band - bar_width) / 2.0
                hit_id = f"{namespace}:bar:{series_index}:{idx}"

                bar_item = {
                    "vertices": 4,
                    "border_radius": "2px",
                    "width": f"{bar_width}px",
                    "height": f"{max(1, int(round(bar_h)))}px",
                    "ip": hit_id,
                }

                custom = {
                    "x": int(round(bar_x)),
                    "y": int(round(bar_top)),
                    "color": item_color,
                    "border_width": border_width,
                    "border_color": border_color,
                    "overlap": True,
                    "z": zlayer(ZLayer.CHART_SERIES, idx),
                }

                for k in _SHAPE_PASSTHROUGH_KEYS:
                    if k in c:
                        custom[k] = c[k]

                bar_item["customise"] = custom
                shapes.append(bar_item)

                try:
                    from Draw._shapes import hitbox as _hitbox_registry
                    _hitbox_registry(
                        ip=hit_id,
                        type=["Fullgeometry"],
                        box={
                            "x": f"{int(round(bar_x))}px",
                            "y": f"{int(round(bar_top))}px",
                            "width": f"{bar_width}px",
                            "height": f"{max(1, int(round(bar_h)))}px",
                        },
                    )
                except Exception:
                    pass

                if animate:
                    try:
                        from Draw._motion import motion as _motion_registry
                        if hit_id not in _motion_registry._connected_motions:
                            _motion_registry(
                                ip=hit_id,
                                motion=[
                                    {
                                        "type": "fade",
                                        "from": 0,
                                        "to": 100,
                                        "duration": 0.4,
                                    }
                                ]
                            )
                    except Exception:
                        pass

                if show_values:
                    val_str = _make_value_label(y_val, value_decimals)
                    try:
                        from Draw._text import measure_text
                        val_w, val_h = measure_text(val_str, font_size=value_font_size)
                    except Exception:
                        val_w, val_h = len(val_str) * value_font_size * 0.6, value_font_size * 1.2

                    bar_center_x = bar_x + (bar_width / 2.0)
                    val_x = bar_center_x - (val_w / 2.0)
                    value_y = (bar_top - val_h - 6) if y_val >= 0 else (bar_top + bar_h + 6)

                    texts.append({
                        "text": val_str,
                        "ip": f"{hit_id}:value",
                        "customise": {
                            "x": int(round(val_x)),
                            "y": int(round(value_y)),
                            "font_size": value_font_size,
                            "color": value_color,
                            "z": zlayer(ZLayer.CHART_LABEL, idx),
                        },
                    })

                if enable_hover or enable_tooltip:
                    self._attach_hover_tooltip(
                        window_tag=window_tag,
                        hit_id=hit_id,
                        series_title=str(series.get("title", f"Series {series_index + 1}")),
                        label=label,
                        val_str=_make_value_label(y_val, value_decimals),
                        px=int(round(bar_x + bar_width / 2.0)),
                        py=int(round(bar_top)),
                    )

            else:
                # Line, Area, Dot, Scatter
                if point_size_raw is None:
                    point_size = max(6, min(16, int(bar_width * 0.6)))
                else:
                    point_size = max(3, _as_int(point_size_raw, field="point_size", default=10))
                hit_id = f"{namespace}:point:{series_index}:{idx}"

                point_item = {
                    "vertices": point_vertices,
                    "width": f"{point_size}px",
                    "height": f"{point_size}px",
                    "ip": hit_id,
                }

                px = int(round(center_x - point_size / 2.0))
                py = int(round(point_y - point_size / 2.0))
                point_custom = {
                    "x": px,
                    "y": py,
                    "color": item_color,
                    "border_width": border_width,
                    "border_color": border_color,
                    "overlap": True,
                    "z": zlayer(ZLayer.CHART_SERIES_FG, idx),
                }

                for k in _SHAPE_PASSTHROUGH_KEYS:
                    if k in c:
                        point_custom[k] = c[k]

                point_item["customise"] = point_custom
                shapes.append(point_item)
                points.append((center_x, point_y))

                try:
                    from Draw._shapes import hitbox as _hitbox_registry
                    _hitbox_registry(
                        ip=hit_id,
                        type=["Fullgeometry"],
                        box={
                            "x": f"{px}px",
                            "y": f"{py}px",
                            "width": f"{point_size}px",
                            "height": f"{point_size}px",
                        },
                    )
                except Exception:
                    pass

                if animate:
                    try:
                        from Draw._motion import motion as _motion_registry
                        _motion_registry(
                            ip=hit_id,
                            motion=[
                                {
                                    "type": "scale",
                                    "from": [0, 0],
                                    "to": [1, 1],
                                    "graph": "ease_out_back",
                                    "duration": 0.6,
                                    "delay": idx * 0.05,
                                }
                            ]
                        )
                    except Exception:
                        pass

                if show_values:
                    val_str = _make_value_label(y_val, value_decimals)
                    try:
                        from Draw._text import measure_text
                        val_w, val_h = measure_text(val_str, font_size=value_font_size)
                    except Exception:
                        val_w, val_h = len(val_str) * value_font_size * 0.6, value_font_size * 1.2

                    val_x = center_x - (val_w / 2.0)
                    val_y = point_y - val_h - 6
                    texts.append({
                        "text": val_str,
                        "ip": f"{hit_id}:value",
                        "customise": {
                            "x": int(round(val_x)),
                            "y": int(round(val_y)),
                            "font_size": value_font_size,
                            "color": value_color,
                            "z": zlayer(ZLayer.CHART_LABEL, idx),
                        },
                    })

                if enable_hover or enable_tooltip:
                    self._attach_hover_tooltip(
                        window_tag=window_tag,
                        hit_id=hit_id,
                        series_title=str(series.get("title", f"Series {series_index + 1}")),
                        label=label,
                        val_str=_make_value_label(y_val, value_decimals),
                        px=int(round(center_x)),
                        py=int(round(point_y)),
                    )

            if emit_x_labels:
                texts.append({
                    "text": label,
                    "ip": f"{namespace}:x_label:{idx}",
                    "customise": {
                        "x": int(round(center_x - len(label) * label_font_size * 0.3)),
                        "y": int(round(bottom + 8)),
                        "font_size": label_font_size,
                        "color": label_color,
                        "z": zlayer(ZLayer.CHART_AXIS, idx),
                    },
                })

        # ── Line & Area chart vector paths ──────────────────────────────
        if graph_type in {"line", "area", "stacked_area"} and len(points) >= 2:
            line_color = _extract_color(c, "line_color", "blue")
            if not is_color(line_color):
                line_color = color_val if not isinstance(color_val, list) else "blue"

            coord_str = " ; ".join(f"{px},{py}" for px, py in points)
            path_ip = f"{namespace}:line:{series_index}"

            try:
                from Draw._point import point as _point_registry
                _point_registry(
                    tag=window_tag,
                    graph=[float(_window_registry.get(window_tag).width()),
                           float(_window_registry.get(window_tag).height())],
                    points=[{
                        "path": coord_str,
                        "colour": line_color,
                        "width": line_width,
                        "edge": "curve" if use_curve else "straight",
                        "smooth": "40%" if use_curve else "0%",
                        "fill": graph_type in {"area", "stacked_area"},
                        "ip": path_ip,
                    }],
                )

                if animate:
                    try:
                        from Draw._motion import motion as _motion_registry
                        if path_ip not in _motion_registry._connected_motions:
                            _motion_registry(
                                ip=path_ip,
                                motion=[
                                    {
                                        "type": "trim_path",
                                        "from": [0, 0],
                                        "to": [0, 100],
                                        "duration": 1.2,
                                        "graph": "ease_out_cubic",
                                    }
                                ]
                            )
                    except Exception:
                        pass
            except Exception:
                pass

        # ── Multi-series Legend ──────────────────────────────────────────
        if emit_legend and (has_show_legend or _as_bool(c.get("show_legend", False))):
            legend_position = str(c.get("legend_position", "top-right")).strip().lower()
            legend_color = _extract_color(c, "legend_color", "#D0DCF0")
            legend_font_size = max(8, _as_int(c.get("legend_font_size"), field="legend_font_size", default=11))

            def _legend_color(i: int) -> object:
                sc = all_series[i].get("customise", {}) or {}
                return _extract_color(sc, "color", "blue") if sc else "blue"

            leg_shapes, leg_texts = _build_legend(
                namespace=namespace,
                series_list=all_series,
                get_color_fn=_legend_color,
                left=left,
                top=top,
                chart_w=chart_w,
                chart_h=chart_h,
                legend_position=legend_position,
                legend_color=legend_color,
                legend_font_size=legend_font_size,
                value_decimals=value_decimals,
            )
            shapes.extend(leg_shapes)
            texts.extend(leg_texts)

        _shapes_registry(
            tag=window_tag,
            shapes=shapes,
            text=texts,
        )

        return (cell_x, cell_y, cell_w, cell_h)

    # ── Pie & Donut Renderer ───────────────────────────────────────────────

    def _render_pie_donut(
        self,
        *,
        window_tag: str,
        namespace: str,
        graph_type: str,
        series_list: List[dict],
        cell_w: float,
        cell_h: float,
        cell_x: float,
        cell_y: float,
        align: object,
        animate: bool,
        enable_hover: bool,
        enable_tooltip: bool,
        draggable: bool,
        ip_str: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Render Circular Pie or Donut Chart."""
        series = series_list[0]
        x_raw = series.get("x", [])
        y_raw = series.get("y", [])
        if not isinstance(x_raw, list) or not isinstance(y_raw, list) or len(x_raw) == 0:
            return None

        labels = [str(v) for v in x_raw]
        values = [_as_float(v, field="y") for v in y_raw]
        total_val = sum(max(0.0, v) for v in values)
        if total_val <= 0:
            total_val = 1.0

        c = series.get("customise", {}) or {}
        raw_colors = resolve_live_data(c.get("color", series.get("color", ["#6366F1", "#10B981", "#EC4899", "#F59E0B", "#8B5CF6", "#3B82F6"])))
        if not isinstance(raw_colors, list):
            raw_colors = [raw_colors]

        colors = [_extract_color({"c": col}, "c", "#3B82F6") for col in raw_colors]

        margin = max(0, _as_int(c.get("margin"), field="margin", default=30))
        title_str = str(c.get("title", series.get("title", ""))).strip()

        # ── Universal layout: delegate to _align.py ──
        pcx, pcy, radius = calculate_radial_center(
            cell_x, cell_y, cell_w, cell_h,
            margin=float(margin),
            has_labels=len(labels) > 0,
            has_title=bool(title_str),
            align=align,
            window_tag=window_tag,
        )
        chart_w = radius * 2.0
        chart_h = chart_w

        shapes: List[dict] = []
        texts: List[dict] = []

        if title_str:
            tx, ty = calculate_alignment_pos(
                "top-left", sw=180, sh=24, cw=cell_w, ch=cell_h,
                pad=margin, offset_x=cell_x, offset_y=cell_y, window_tag=window_tag,
            )
            texts.append({
                "text": title_str,
                "ip": f"{namespace}:title",
                "customise": {
                    "x": int(tx + 10),
                    "y": int(ty + 14),
                    "font_size": 15,
                    "color": _extract_color(c, "title_color", "#DCEBFF"),
                    "bold": True,
                }
            })

        container_ip = f"{namespace}:graph_container"
        shapes.append({
            "vertices": 4,
            "width": f"{int(chart_w)}px",
            "height": f"{int(chart_h)}px",
            "ip": container_ip,
            "customise": {
                "x": int(pcx - radius),
                "y": int(pcy - radius),
                "color": "transparent",
                "opacity": 0,
                "overlap": True,
                "z": 100,
            },
        })

        if draggable and ip_str:
            self._attach_drag_listener(
                window_tag=window_tag,
                container_ip=container_ip,
                graph_ip=ip_str,
            )

        inner_r = radius * 0.55 if graph_type == "donut" else 0.0

        from Draw._colour import color as _color_registry

        start_angle = -90.0
        for idx, (label, val) in enumerate(zip(labels, values)):
            pct = max(0.0, val) / total_val
            sweep = pct * 360.0
            mid_angle = start_angle + sweep / 2.0
            mid_rad = math.radians(mid_angle)

            item_color = colors[idx % len(colors)]
            wedge_ip = f"{namespace}:wedge:0:{idx}"

            # Query Draw.color (color.py) for registered color bindings by IP
            for target_ip in [wedge_ip, f"{namespace}:pie:{idx}", f"{namespace}:pie_wedge:{idx}", namespace, ip_str]:
                if target_ip and _color_registry.has_binding(target_ip):
                    resolved = _color_registry.resolve_for_shape(target_ip)
                    if resolved and "body_color" in resolved:
                        r, g, b, a = resolved["body_color"]
                        item_color = f"rgba({r},{g},{b},{a/255.0:.2f})"
                        break

            num_steps = max(8, int(round(sweep / 6.0)))
            poly_pts = []
            for i in range(num_steps + 1):
                ang = math.radians(start_angle + (sweep * i / num_steps))
                poly_pts.append((pcx + radius * math.cos(ang), pcy + radius * math.sin(ang)))

            if graph_type == "donut":
                for i in range(num_steps, -1, -1):
                    ang = math.radians(start_angle + (sweep * i / num_steps))
                    poly_pts.append((pcx + inner_r * math.cos(ang), pcy + inner_r * math.sin(ang)))
            else:
                poly_pts.append((pcx, pcy))

            shapes.append({
                "vertices": len(poly_pts),
                "width": f"{int(radius * 2)}px",
                "height": f"{int(radius * 2)}px",
                "x": int(pcx - radius),
                "y": int(pcy - radius),
                "ip": wedge_ip,
                "custom_vertices": poly_pts,
                "customise": {
                    "color": item_color,
                    "opacity": 90,
                    "overlap": True,
                    "z": 5,
                }
            })

            label_r = (radius + inner_r) / 2.0 if graph_type == "donut" else radius * 0.65
            lx = pcx + label_r * math.cos(mid_rad) - 14
            ly = pcy + label_r * math.sin(mid_rad) - 7
            if sweep > 12:
                texts.append({
                    "text": f"{pct*100:.0f}%",
                    "ip": f"{namespace}:wedge_label:0:{idx}",
                    "customise": {
                        "x": int(lx),
                        "y": int(ly),
                        "font_size": 11,
                        "color": "#FFFFFF",
                        "bold": True,
                    }
                })

            if enable_hover or enable_tooltip:
                self._attach_hover_tooltip(
                    window_tag=window_tag,
                    hit_id=wedge_ip,
                    series_title=title_str or "Distribution",
                    label=label,
                    val_str=f"{val:.1f} ({pct*100:.1f}%)",
                    px=int(pcx + radius * 0.7 * math.cos(mid_rad)),
                    py=int(pcy + radius * 0.7 * math.sin(mid_rad)),
                )

            start_angle += sweep

        if graph_type == "donut":
            hole_r = inner_r
            hole_color = _extract_color(c, "hole_color", "#1E293B")
            shapes.append({
                "vertices": 64,
                "width": f"{int(hole_r * 2)}px",
                "height": f"{int(hole_r * 2)}px",
                "ip": f"{namespace}:hole",
                "customise": {
                    "x": int(pcx - hole_r),
                    "y": int(pcy - hole_r),
                    "color": hole_color,
                    "overlap": True,
                    "z": 8,
                }
            })
            center_txt = c.get("center_text", f"{total_val:.0f}")
            texts.append({
                "text": str(center_txt),
                "ip": f"{namespace}:center_text",
                "customise": {
                    "x": int(pcx - len(str(center_txt)) * 5),
                    "y": int(pcy - 10),
                    "font_size": 18,
                    "color": "#F8FAFC",
                    "bold": True,
                    "z": 9,
                }
            })

        # Category legend
        leg_x = int(pcx + radius + 15)
        leg_y = int(pcy - (len(labels) * 22) / 2.0)
        for idx, (label, val) in enumerate(zip(labels, values)):
            pct = max(0.0, val) / total_val
            item_color = colors[idx % len(colors)]
            shapes.append({
                "vertices": 4,
                "width": "12px",
                "height": "12px",
                "border_radius": "3px",
                "ip": f"{namespace}:leg_box:{idx}",
                "customise": {
                    "x": leg_x,
                    "y": leg_y + idx * 22,
                    "color": item_color,
                    "overlap": True,
                    "z": 10,
                }
            })
            texts.append({
                "text": f"{label} ({pct*100:.0f}%)",
                "ip": f"{namespace}:leg_txt:{idx}",
                "customise": {
                    "x": leg_x + 18,
                    "y": leg_y + idx * 22 - 2,
                    "font_size": 11,
                    "color": "#E2E8F0",
                }
            })

        _shapes_registry(tag=window_tag, shapes=shapes, text=texts)
        return (cell_x, cell_y, cell_w, cell_h)

    # ── Radar / Spider Renderer ─────────────────────────────────────────────

    def _render_radar(
        self,
        *,
        window_tag: str,
        namespace: str,
        series_list: List[dict],
        cell_w: float,
        cell_h: float,
        cell_x: float,
        cell_y: float,
        align: object,
        animate: bool,
        enable_hover: bool,
        enable_tooltip: bool,
        draggable: bool,
        ip_str: str,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Render Radar / Spider Chart."""
        first = series_list[0]
        categories = [str(v) for v in first.get("x", [])]
        num_vars = len(categories)
        if num_vars < 3:
            return None

        c = first.get("customise", {}) or {}
        margin = max(0, _as_int(c.get("margin"), field="margin", default=35))

        # ── Universal layout: delegate to _align.py ──
        rcx, rcy, radius = calculate_radial_center(
            cell_x, cell_y, cell_w, cell_h,
            margin=float(margin),
            has_labels=True,
            has_title=False,
            align=align,
            window_tag=window_tag,
        )
        chart_w = radius * 2.0
        chart_h = chart_w

        shapes: List[dict] = []
        texts: List[dict] = []

        container_ip = f"{namespace}:graph_container"
        shapes.append({
            "vertices": 4,
            "width": f"{int(chart_w)}px",
            "height": f"{int(chart_h)}px",
            "ip": container_ip,
            "customise": {
                "x": int(rcx - radius),
                "y": int(rcy - radius),
                "color": "transparent",
                "opacity": 0,
                "overlap": True,
                "z": 100,
            },
        })

        if draggable and ip_str:
            self._attach_drag_listener(
                window_tag=window_tag,
                container_ip=container_ip,
                graph_ip=ip_str,
            )

        levels = max(1, _as_int(c.get("spider_levels"), field="spider_levels", default=4))
        for lvl in range(1, levels + 1):
            r_level = radius * (lvl / levels)
            shapes.append({
                "vertices": num_vars,
                "width": f"{int(r_level * 2)}px",
                "height": f"{int(r_level * 2)}px",
                "ip": f"{namespace}:radar_ring:{lvl}",
                "customise": {
                    "x": int(rcx - r_level),
                    "y": int(rcy - r_level),
                    "color": "transparent",
                    "border_width": 1,
                    "border_color": "#475569",
                    "overlap": True,
                    "z": 2,
                }
            })

        angle_step = (2 * math.pi) / num_vars
        for i, cat in enumerate(categories):
            ang = i * angle_step - math.pi / 2
            cos_a = math.cos(ang)
            sin_a = math.sin(ang)
            label_dist = radius + 12
            tx = rcx + label_dist * cos_a
            ty = rcy + label_dist * sin_a
            if abs(cos_a) < 0.2:
                tx -= len(cat) * 3.5
            elif cos_a < 0:
                tx -= len(cat) * 7.0
            ty -= 6.0
            texts.append({
                "text": cat,
                "ip": f"{namespace}:radar_label:{i}",
                "customise": {
                    "x": int(tx),
                    "y": int(ty),
                    "font_size": 11,
                    "color": "#94A3B8",
                }
            })

        # ── Data webs: plot each series' actual y-values as a filled polygon ──
        default_web_colors = ["#38BDF8", "#F472B6", "#34D399", "#FBBF24", "#A78BFA"]
        all_values = []
        for s in series_list:
            for v in s.get("y", []):
                try:
                    all_values.append(_as_float(v, field="y"))
                except TypeError:
                    pass
        max_y_raw = c.get("max_y", None)
        max_val = _as_float(max_y_raw, field="max_y") if max_y_raw is not None else max([1.0] + all_values)
        if max_val <= 0:
            max_val = 1.0

        for s_idx, s in enumerate(series_list):
            svals = s.get("y", [])
            if len(svals) != num_vars:
                continue
            svals = [_as_float(v, field="y") for v in svals]
            sc = s.get("customise", {}) or {}
            web_color = _extract_color(sc, "color", default_web_colors[s_idx % len(default_web_colors)])
            if isinstance(web_color, list):
                web_color = web_color[0] if web_color else default_web_colors[s_idx % len(default_web_colors)]

            vertices = []
            for i, val in enumerate(svals):
                ratio = max(0.0, min(1.0, val / max_val))
                r_val = radius * ratio
                ang = i * angle_step - math.pi / 2
                vertices.append((rcx + r_val * math.cos(ang), rcy + r_val * math.sin(ang)))

            web_ip = f"{namespace}:radar_web:{s_idx}"
            closed_points = vertices + [vertices[0]]
            coord_str = " ; ".join(f"{vx},{vy}" for vx, vy in closed_points)
            try:
                from Draw._point import point as _point_registry
                _point_registry(
                    tag=window_tag,
                    graph=[float(_window_registry.get(window_tag).width()),
                           float(_window_registry.get(window_tag).height())],
                    points=[{
                        "path": coord_str,
                        "colour": web_color,
                        "width": 2,
                        "edge": "straight",
                        "fill": True,
                        "ip": web_ip,
                    }],
                )
            except Exception:
                pass

            for i, (vx, vy) in enumerate(vertices):
                vertex_ip = f"{namespace}:radar_point:{s_idx}:{i}"
                shapes.append({
                    "vertices": 24,
                    "width": "8px",
                    "height": "8px",
                    "ip": vertex_ip,
                    "customise": {
                        "x": int(round(vx - 4)),
                        "y": int(round(vy - 4)),
                        "color": web_color,
                        "overlap": True,
                        "z": 5,
                    },
                })
                if enable_hover or enable_tooltip:
                    self._attach_hover_tooltip(
                        window_tag=window_tag,
                        hit_id=vertex_ip,
                        series_title=str(s.get("title", f"Series {s_idx + 1}")),
                        label=categories[i],
                        val_str=_make_value_label(svals[i], 0),
                        px=int(round(vx)),
                        py=int(round(vy)),
                    )

        _shapes_registry(tag=window_tag, shapes=shapes, text=texts)
        return (cell_x, cell_y, cell_w, cell_h)

    # ── Hover & Tooltip Attachment ──────────────────────────────────────────

    def _attach_hover_tooltip(
        self,
        *,
        window_tag: str,
        hit_id: str,
        series_title: str,
        label: str,
        val_str: str,
        px: int,
        py: int,
    ) -> None:
        """Register interactive hover senses listener and dynamic tooltip display."""
        try:
            from Draw._senses_redesign import senses as _senses_registry
            sense_hover = _senses_registry("mouse_hover", id=f"hover:{hit_id}", ip=hit_id)

            def _on_tick():
                if sense_hover.consume():
                    tooltip_ip = f"{window_tag}:tooltip"
                    _shapes_registry(
                        tag=window_tag,
                        shapes=[{
                            "vertices": 4,
                            "border_radius": "4px",
                            "width": "140px",
                            "height": "40px",
                            "ip": tooltip_ip,
                            "customise": {
                                "x": px - 70,
                                "y": py - 48,
                                "color": "#0F172A",
                                "border_width": 1,
                                "border_color": "#38BDF8",
                                "opacity": 95,
                                "overlap": True,
                                "z": 999,
                            }
                        }],
                        text=[{
                            "text": f"{series_title}\n{label}: {val_str}",
                            "ip": f"{tooltip_ip}:text",
                            "customise": {
                                "x": px - 62,
                                "y": py - 44,
                                "font_size": 10,
                                "color": "#F8FAFC",
                            }
                        }]
                    )

            self._hover_listeners[hit_id] = _on_tick
        except Exception:
            pass

    # ── Hold & Move Drag Listener ─────────────────────────────────────────────

    def _attach_drag_listener(
        self,
        *,
        window_tag: str,
        container_ip: str,
        graph_ip: str,
    ) -> None:
        """Attach mouse drag listener for Hold & Move canvas chart dragging."""
        try:
            from Draw._shapes import hitbox as _hitbox_registry
            _hitbox_registry(
                ip=container_ip,
                type=["Fullgeometry"],
                box={
                    "x": "0px",
                    "y": "0px",
                    "width": "100%",
                    "height": "100%",
                }
            )

            from Draw._senses_redesign import senses as _senses_registry
            sense_drag = _senses_registry("mouse_drag", id=f"drag:{container_ip}", ip=container_ip)

            def _on_drag_tick():
                if sense_drag.consume():
                    entry = self._registry.get((window_tag, graph_ip))
                    if entry:
                        # Re-render with updated offsets
                        dx = getattr(sense_drag, "dx", 0.0)
                        dy = getattr(sense_drag, "dy", 0.0)
                        entry.offset_x += dx
                        entry.offset_y += dy
                        self.__call__(
                            ip=graph_ip,
                            tag=window_tag,
                            graph=entry.series_data,
                            type=entry.graph_type,
                            get_ip=entry.get_ip,
                            columns=entry.columns,
                            align=entry.align,
                            animate=entry.animate,
                            hover=entry.hover,
                            tooltip=entry.tooltip,
                            draggable=True,
                        )

            self._drag_listeners[container_ip] = _on_drag_tick
        except Exception:
            pass

    # ── Live Updating API ───────────────────────────────────────────────────

    def update_live(self, window_tag: Optional[str] = None) -> None:
        """
        Re-evaluates live data references (Draw._live) across registered graphs
        and updates their rendering in real time.
        """
        for (wt, ip_str), entry in list(self._registry.items()):
            if window_tag is not None and wt != window_tag:
                continue
            self.__call__(
                ip=ip_str,
                tag=wt,
                graph=entry.series_data,
                type=entry.graph_type,
                get_ip=entry.get_ip,
                columns=entry.columns,
                align=entry.align,
                animate=entry.animate,
                hover=entry.hover,
                tooltip=entry.tooltip,
                draggable=entry.draggable,
            )

    # ── Room-engine interop API ───────────────────────────────────────────

    def get_by_ip(self, window_tag: str, ip: str) -> Optional[GraphDef]:
        """Return live GraphDef for (window_tag, ip), or None."""
        return self._registry.get((window_tag, ip))

    def get_pixel_bounds(
        self, window_tag: str, ip: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return (x, y, w, h) of the chart's pixel bounding box, or None."""
        entry = self._registry.get((window_tag, ip))
        if entry is None:
            return None
        return entry.bounds

    def get_intrinsic_size(
        self, window_tag: str, ip: str
    ) -> Optional[Tuple[float, float]]:
        """Return (width, height) for _room_size.resolve_size_spec interop."""
        bounds = self.get_pixel_bounds(window_tag, ip)
        if bounds is None:
            return None
        return (bounds[2], bounds[3])

    def move_by_ip(self, window_tag: str, ip: str, x: float, y: float) -> bool:
        """Shift a registered graph's bounds so its top-left lands at pixel (x, y)."""
        entry = self._registry.get((window_tag, ip))
        if entry is None:
            return False
        bx, by, bw, bh = entry.bounds
        entry.offset_x = x - bx
        entry.offset_y = y - by
        entry.bounds = (float(x), float(y), bw, bh)
        return True

    def resize_by_ip(self, window_tag: str, ip: str, width: float, height: float) -> bool:
        """Resize a registered graph's recorded bounds (for room layout accounting)."""
        entry = self._registry.get((window_tag, ip))
        if entry is None or width < 0 or height < 0:
            return False
        bx, by, _, _ = entry.bounds
        entry.bounds = (bx, by, float(width), float(height))
        return True

    def clear(self, window_tag: str) -> int:
        """Remove all graph registrations for a window. Returns count removed."""
        keys = [(wt, ip) for wt, ip in self._registry if wt == window_tag]
        for k in keys:
            del self._registry[k]
        return len(keys)


graph = _GraphRegistry()
