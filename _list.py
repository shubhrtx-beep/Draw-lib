"""
Draw._list
==========
Helpers for rendering key-value lists as visual table layouts.

The ``list`` parameter accepted by ``Draw.shape()`` has three sections::

    list = [
        {                                    # ── SECTION 1: data dict (required) ──
            "header_1" : ["a", "b", "c"],
            "header_2" : ["d", "e", "f"],
            "header_3" : ["g", "h"],         # uneven lengths OK
        },

        "layout", {                          # ── SECTION 2: layout config (optional) ──

            # ── orientation ───────────────────────────────────
            "rows_vertical"   : "keys",      # keys become row headers  (DEFAULT)
            "rows_horizontal" : "values",    # values fill columns      (DEFAULT)

            # ── origin ────────────────────────────────────────
            "origin"          : "top-left",

            # ── color ─────────────────────────────────────────
            "color"           : "#2c3e50",   # default cell fill color
            "key_color"       : "#e74c3c",   # header cells fill color
            "color_ip"        : "",          # ip prefix for per-cell color tokens
            "alt_color"       : "#34495e",   # alternating row color (zebra stripe)
            "hover_color"     : "#3498db",   # cell color on mouse hover
            "select_color"    : "#1abc9c",   # cell color when selected

            # ── text defaults ──────────────────────────────────
            "font_family"     : "Inter",
            "font_size"       : 14,
            "text_color"      : "#ecf0f1",
            "text_align"      : "center",    # left | center | right
            "bold_keys"       : True,        # bold header cells
            "line_height"     : 1.4,

            # ── spacing ───────────────────────────────────────
            "cell_margin"     : 3,           # gap between cells (px)
            "padding"         : [8, 12],     # [vertical, horizontal] inside cell
            "table_margin"    : [0,0,0,0],   # [top, right, bottom, left] around whole table

            # ── auto sizing ───────────────────────────────────
            "auto_width"      : True,        # auto-fit column width to content
            "auto_height"     : True,        # auto-fit row height to content
            "min_width"       : 60,          # minimum column width (px)
            "max_width"       : 400,         # maximum column width (px)
            "min_height"      : 30,          # minimum row height (px)

            # ── column widths (manual override) ───────────────
            "col_widths"      : [120, "auto", 80],

            # ── sorting ───────────────────────────────────────
            "sortable"        : False,       # enable sorting (static at creation)
            "sort_by"         : None,        # key to sort by (must match a header)
            "sort_order"      : "asc",       # asc | desc

            # ── pagination ────────────────────────────────────
            "paginate"        : False,       # (future) enable pagination
            "page_size"       : 20,          # (future) rows per page

            # ── search / filter ───────────────────────────────
            "searchable"      : False,       # (future)
            "filterable"      : False,       # (future)

            # ── selection ─────────────────────────────────────
            "selectable"      : False,       # (future)
            "select_mode"     : "row",       # (future) row | cell | multi-row

            # ── scroll ────────────────────────────────────────
            "scroll"          : "auto",      # (future) auto | always | never
            "sticky_header"   : True,        # (future)

            # ── border ────────────────────────────────────────
            "table_border"        : False,
            "table_border_color"  : "#7f8c8d",
            "table_border_width"  : 1,
            "cell_border"         : False,
            "cell_border_color"   : "#95a5a6",
            "cell_border_width"   : 1,

            # ── animation ─────────────────────────────────────
            "animate_in"      : "none",      # none | fade | slide | scale
            "animate_speed"   : 300,         # duration in ms
        },

        "shape_edit", {                      # ── SECTION 3: cell shape config (optional) ──
            "shape": [{
                "vertices"     : None,
                "size"         : [100, 40],
                "border_radius": 4,
                "border_width" : 1,
                "border_color" : "#bdc3c7",
                "border_style" : "solid",
                "curve_mode"   : "line",
                "bend"         : [],
                "exclude"      : [],
                "overlap"      : True,
                "opacity"      : 100,
            }],
            "column_shapes" : {              # per-column shape overrides (0-based)
                0 : {"border_radius": 0, "color": "#e74c3c"},
            },
            "row_shapes" : {                 # per-row shape overrides (0-based)
                0 : {"border_width": 2},
            },
        }
    ]

CELL OVERRIDE — any data value can be a dict for per-cell styling::

    "header" : [
        "normal",
        {"text": "special", "color": "#e74c3c", "bold": True},
    ]
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple


# ── parser ────────────────────────────────────────────────────────────────────

def _parse_list_arg(
    raw_list: list,
) -> Tuple[Dict[str, List[Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Split the ``list=[...]`` parameter into three parts:

    Returns
    -------
    data        : ``{"key": [values], ...}``
    layout_cfg  : layout config dict (from the item after ``"layout"``)
    shape_cfg   : shape-edit config dict (from the item after ``"shape_edit"``)
    """
    data: Dict[str, List[Any]] = {}
    layout_cfg: Dict[str, Any] = {}
    shape_cfg: Dict[str, Any] = {}

    if not isinstance(raw_list, list) or not raw_list:
        return data, layout_cfg, shape_cfg

    # First element: the data dict
    if isinstance(raw_list[0], dict):
        data = raw_list[0]

    # Walk the rest looking for section markers
    current_section: Optional[str] = None
    for item in raw_list[1:]:
        if isinstance(item, str):
            current_section = item.strip().lower()
        elif isinstance(item, dict):
            if current_section == "layout":
                layout_cfg = item
            elif current_section == "shape_edit":
                shape_cfg = item
            current_section = None          # reset after consuming the dict

    return data, layout_cfg, shape_cfg


# ── cell value resolver ───────────────────────────────────────────────────────

def _resolve_cell_value(val: Any) -> Tuple[Any, Dict[str, Any]]:
    """
    Resolve a data cell value into (display_text, cell_overrides).

    If ``val`` is a dict, extract "text" and treat the rest as per-cell
    overrides (color, bold, font_size, etc.).
    Otherwise, convert to string with no overrides.
    """
    from Draw._live import is_live_text_binding
    if isinstance(val, dict):
        text = val.get("text", "")
        if not (is_live_text_binding(text) or callable(text)):
            text = str(text)
        overrides = {k: v for k, v in val.items() if k != "text"}
        return text, overrides
    if is_live_text_binding(val) or callable(val):
        return val, {}
    return str(val), {}


# ── auto-width measurement ────────────────────────────────────────────────────

def _estimate_text_width(text: str, font_size: int) -> int:
    """
    Estimate pixel width of a text string given a font size.
    Uses a simple character-width heuristic (average ~0.6 × font_size per char).
    """
    if not text:
        return 0
    # Average char width is roughly 60% of font_size for most fonts
    return int(len(text) * font_size * 0.62)


def _compute_column_widths(
    data: Dict[str, List[Any]],
    keys: List[str],
    layout_cfg: Dict[str, Any],
    col_widths_override: Optional[List[Any]],
    num_cols: int,
    vertical: str,
    font_size: int,
    pad_h: int,
    column_shapes: Dict[int, Dict[str, Any]],
    default_cw: int,
) -> List[int]:
    """
    Compute per-column widths, respecting auto_width, min_width, max_width,
    and any manual col_widths overrides.
    """
    auto_width  = bool(layout_cfg.get("auto_width",  True))
    min_width   = int(layout_cfg.get("min_width",   60))
    max_width   = int(layout_cfg.get("max_width",   400))

    effective: List[int] = []

    for c in range(num_cols):
        # 1. Hard override from col_widths list
        if col_widths_override and c < len(col_widths_override):
            w = col_widths_override[c]
            if isinstance(w, (int, float)):
                effective.append(max(min_width, min(max_width, int(w))))
                continue
            # "auto" → fall through to auto-sizing

        # 2. column_shapes size override
        if c in column_shapes and "size" in column_shapes[c]:
            cs = column_shapes[c]["size"]
            if isinstance(cs, (list, tuple)) and len(cs) >= 1:
                w = int(cs[0])
                effective.append(max(min_width, min(max_width, w)))
                continue

        # 3. Auto-width: measure content
        if auto_width:
            max_chars_w = 0
            if vertical == "keys":
                # col 0 = key labels, col 1+ = values
                if c == 0:
                    for key in keys:
                        max_chars_w = max(max_chars_w, _estimate_text_width(key, font_size))
                else:
                    val_col_idx = c - 1
                    for key in keys:
                        values = data[key]
                        if not isinstance(values, (list, tuple)):
                            values = [values]
                        if val_col_idx < len(values):
                            cell_text, _ = _resolve_cell_value(values[val_col_idx])
                            from Draw._live import is_live_text_binding, resolve_live_text, LiveTextBinding
                            if is_live_text_binding(cell_text):
                                cell_str = resolve_live_text(cell_text)
                            elif callable(cell_text):
                                cell_str = resolve_live_text(LiveTextBinding(cell_text))
                            else:
                                cell_str = str(cell_text)
                            max_chars_w = max(max_chars_w, _estimate_text_width(cell_str, font_size))
            else:
                # col = key header or value column
                if c < len(keys):
                    key = keys[c]
                    max_chars_w = max(max_chars_w, _estimate_text_width(key, font_size))
                    values = data[key]
                    if not isinstance(values, (list, tuple)):
                        values = [values]
                    for val in values:
                        cell_text, _ = _resolve_cell_value(val)
                        from Draw._live import is_live_text_binding, resolve_live_text, LiveTextBinding
                        if is_live_text_binding(cell_text):
                            cell_str = resolve_live_text(cell_text)
                        elif callable(cell_text):
                            cell_str = resolve_live_text(LiveTextBinding(cell_text))
                        else:
                            cell_str = str(cell_text)
                        max_chars_w = max(max_chars_w, _estimate_text_width(cell_str, font_size))

            w = max_chars_w + pad_h * 2 + 8   # 8px breathing room
            effective.append(max(min_width, min(max_width, w)))
            continue

        # 4. Static default
        effective.append(max(min_width, min(max_width, default_cw)))

    return effective


def _compute_row_height(
    layout_cfg: Dict[str, Any],
    font_size: int,
    pad_v: int,
    default_ch: int,
) -> int:
    """
    Compute uniform row height respecting auto_height and min_height.
    """
    auto_height = bool(layout_cfg.get("auto_height", True))
    min_height  = int(layout_cfg.get("min_height",  30))

    if auto_height:
        # Typical single-line row: font_size + vertical padding × 2 + 4px breathing room
        h = font_size + pad_v * 2 + 4
        return max(min_height, h)

    return max(min_height, default_ch)


# ── generator ─────────────────────────────────────────────────────────────────

# Keys from shape_edit.shape[0] that should be forwarded to every cell shape
_CELL_SHAPE_KEYS = {
    "vertices", "size", "border_radius",
    "border_width", "border_color", "border_style",
    "curve_mode", "bend", "exclude", "symmetry", "custom",
    "opacity", "rotation", "hitbox_mode", "hit_box",
    "overlap", "z",
}


def _generate_list_items(
    ip_prefix: Optional[str],
    raw_list: list,
) -> Tuple[List[dict], List[dict], Optional[dict]]:
    """
    Parse the ``list`` parameter and produce shape / text definition dicts
    ready to be appended to ``shape_items`` / ``text_items`` in the shape
    registry.

    Returns
    -------
    shapes     : list of shape definition dicts (one per cell)
    texts      : list of text definition dicts  (one per cell)
    table_def  : a dimension dict suitable for ``Draw.table(dimension=...)``
                 or ``None`` when the data dict is empty
    """
    data, layout_cfg, shape_cfg = _parse_list_arg(raw_list)

    if not data:
        return [], [], None

    # ── orientation ───────────────────────────────────────────────────
    vertical = layout_cfg.get(
        "rows_vertical",
        layout_cfg.get("rows_vertical", "keys"),
    )
    vertical = str(vertical).strip().lower()

    keys = list(data.keys())
    max_values = max(
        (len(v) if isinstance(v, (list, tuple)) else 1 for v in data.values()),
        default=0,
    )

    if vertical == "keys":
        # keys are row headers (col 0), values fill columns 1..N
        num_rows = len(keys)
        num_cols = max_values + 1
    else:
        # keys are column headers (row 0), values fill rows 1..N
        num_cols = len(keys)
        num_rows = max_values + 1

    # ── base cell shape properties ────────────────────────────────────
    cell_template: Dict[str, Any] = {}
    if (
        "shape" in shape_cfg
        and isinstance(shape_cfg["shape"], list)
        and shape_cfg["shape"]
    ):
        src = shape_cfg["shape"][0]
        if isinstance(src, dict):
            for k in _CELL_SHAPE_KEYS:
                if k in src:
                    cell_template[k] = src[k]

    # ── per-column / per-row shape overrides ──────────────────────────
    column_shapes: Dict[int, Dict[str, Any]] = {}
    row_shapes: Dict[int, Dict[str, Any]] = {}

    raw_col_shapes = shape_cfg.get("column_shapes", None)
    if isinstance(raw_col_shapes, dict):
        for col_key, col_val in raw_col_shapes.items():
            if isinstance(col_val, dict):
                try:
                    column_shapes[int(col_key)] = col_val
                except (TypeError, ValueError):
                    pass

    raw_row_shapes = shape_cfg.get("row_shapes", None)
    if isinstance(raw_row_shapes, dict):
        for row_key, row_val in raw_row_shapes.items():
            if isinstance(row_val, dict):
                try:
                    row_shapes[int(row_key)] = row_val
                except (TypeError, ValueError):
                    pass

    # ── table ip ──────────────────────────────────────────────────────
    table_ip = f"{ip_prefix}_list_table" if ip_prefix else "list_table"

    # ── colors ────────────────────────────────────────────────────────
    default_color  = layout_cfg.get("color", None)
    key_color      = layout_cfg.get("key_color", None)
    alt_color      = layout_cfg.get("alt_color", None)
    color_ip_prefix = layout_cfg.get("color_ip", "")
    hover_color    = layout_cfg.get("hover_color", None)
    select_color   = layout_cfg.get("select_color", None)

    # ── text styling defaults ─────────────────────────────────────────
    text_color   = layout_cfg.get("text_color", None)
    font_family  = layout_cfg.get("font_family", None)
    font_size    = int(layout_cfg.get("font_size", 14))
    text_align   = layout_cfg.get("text_align", None)
    bold_keys    = layout_cfg.get("bold_keys", True)
    line_height  = layout_cfg.get("line_height", None)

    # ── padding ───────────────────────────────────────────────────────
    padding_raw = layout_cfg.get("padding", None)
    pad_v, pad_h = 0, 0
    if isinstance(padding_raw, (list, tuple)) and len(padding_raw) >= 2:
        try:
            pad_v = int(padding_raw[0])
            pad_h = int(padding_raw[1])
        except (TypeError, ValueError):
            pass
    elif isinstance(padding_raw, (int, float)):
        pad_v = pad_h = int(padding_raw)

    # ── table margin ──────────────────────────────────────────────────
    table_margin_raw = layout_cfg.get("table_margin", None)
    table_margin_top    = 0
    table_margin_right  = 0
    table_margin_bottom = 0
    table_margin_left   = 0
    if isinstance(table_margin_raw, (list, tuple)):
        parts = list(table_margin_raw)
        if len(parts) >= 1: table_margin_top    = int(parts[0])
        if len(parts) >= 2: table_margin_right  = int(parts[1])
        if len(parts) >= 3: table_margin_bottom = int(parts[2])
        if len(parts) >= 4: table_margin_left   = int(parts[3])
    elif isinstance(table_margin_raw, (int, float)):
        v = int(table_margin_raw)
        table_margin_top = table_margin_right = table_margin_bottom = table_margin_left = v

    # ── table border ──────────────────────────────────────────────────
    table_border        = bool(layout_cfg.get("table_border", False))
    table_border_color  = layout_cfg.get("table_border_color", "#7f8c8d")
    table_border_width  = int(layout_cfg.get("table_border_width", 1))

    # ── cell border ───────────────────────────────────────────────────
    cell_border        = layout_cfg.get("cell_border", False)
    cell_border_color  = layout_cfg.get("cell_border_color", "#95a5a6")
    cell_border_width  = layout_cfg.get("cell_border_width", 1)

    # ── search & filter ───────────────────────────────────────────────
    search_query = layout_cfg.get("search_query", layout_cfg.get("search", None))
    if search_query:
        data = search_list_data(data, str(search_query))

    filter_func = layout_cfg.get("filter_func", layout_cfg.get("filter", None))
    if callable(filter_func):
        data = filter_list_data(data, filter_func)

    # ── sorting ───────────────────────────────────────────────────────
    sortable   = bool(layout_cfg.get("sortable", False))
    sort_by    = layout_cfg.get("sort_by", None)
    sort_order = str(layout_cfg.get("sort_order", "asc")).strip().lower()
    if sort_order not in ("asc", "desc"):
        sort_order = "asc"

    if (sortable or sort_by is not None) and sort_by is not None and sort_by in data:
        data = sort_list_data(data, sort_by, sort_order)

    # ── animation ─────────────────────────────────────────────────────
    animate_in    = str(layout_cfg.get("animate_in", "none")).strip().lower()
    animate_speed = int(layout_cfg.get("animate_speed", 300))
    if animate_in not in ("fade", "slide", "scale"):
        animate_in = "none"

    # ── col_widths (manual column widths) ─────────────────────────────
    col_widths_raw = layout_cfg.get("col_widths", None)
    col_widths: Optional[List[Any]] = None
    if isinstance(col_widths_raw, (list, tuple)):
        col_widths = list(col_widths_raw)

    # ── fallback cell size from template ──────────────────────────────
    cell_size = cell_template.get("size")
    if isinstance(cell_size, (list, tuple)) and len(cell_size) >= 2:
        default_cw = int(cell_size[0])
        default_ch = int(cell_size[1])
    else:
        default_cw = 100
        default_ch = 40

    # ── compute per-column widths (with auto-width logic) ─────────────
    effective_col_widths = _compute_column_widths(
        data=data,
        keys=keys,
        layout_cfg=layout_cfg,
        col_widths_override=col_widths,
        num_cols=num_cols,
        vertical=vertical,
        font_size=font_size,
        pad_h=pad_h,
        column_shapes=column_shapes,
        default_cw=default_cw,
    )

    # ── compute row height (with auto-height logic) ───────────────────
    effective_row_height = _compute_row_height(
        layout_cfg=layout_cfg,
        font_size=font_size,
        pad_v=pad_v,
        default_ch=default_ch,
    )

    # ── build cell dicts ──────────────────────────────────────────────
    shapes: List[dict] = []
    texts: List[dict] = []
    cell_counter = 0   # for staggered animation delay

    def _add_cell(
        label: str,
        col: int,
        row: int,
        *,
        is_key: bool,
        cell_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create one shape dict + one text dict for a single table cell."""
        nonlocal cell_counter

        # Shape dict — deep-copy the template so bend/exclude lists are independent
        sd = copy.deepcopy(cell_template)

        # Override size to use computed column width & row height
        col_w = effective_col_widths[col] if col < len(effective_col_widths) else default_cw
        sd["size"] = [col_w, effective_row_height]

        # Apply per-column shape overrides
        if col in column_shapes:
            col_override = copy.deepcopy(column_shapes[col])
            # Don't let column_shapes.size override the auto-computed width
            # unless it was explicitly set AND col_widths wasn't overriding already
            if "size" in col_override:
                # honour explicit column size overrides
                sd["size"] = col_override.pop("size")
            sd.update(col_override)

        # Apply per-row shape overrides
        if row in row_shapes:
            sd.update(copy.deepcopy(row_shapes[row]))

        # The shape parser reads "column" (not "cell") for the cell reference
        sd["column"] = [col, row]
        sd["get_ip"] = table_ip

        # Assign a per-cell ip
        if color_ip_prefix:
            sd["ip"] = f"{color_ip_prefix}_{col}_{row}"
        else:
            sd["ip"] = f"{table_ip}_{col}_{row}"

        # ── color precedence ──────────────────────────────────────────
        cell_color_override = None
        if cell_overrides and "color" in cell_overrides:
            cell_color_override = cell_overrides["color"]

        if cell_color_override is not None:
            sd["color"] = cell_color_override
        elif is_key and key_color is not None:
            sd["color"] = key_color
        elif not is_key and alt_color is not None and row % 2 == 1:
            sd["color"] = alt_color
        elif default_color is not None:
            sd.setdefault("color", default_color)

        # ── hover / select colors ─────────────────────────────────────
        if hover_color is not None:
            sd["hover_color"] = hover_color
        if select_color is not None:
            sd["select_color"] = select_color

        # ── table border → apply as border on edge cells ───────────────
        if table_border:
            is_top    = (row == 0)
            is_bottom = (row == num_rows - 1)
            is_left   = (col == 0)
            is_right  = (col == num_cols - 1)
            if is_top or is_bottom or is_left or is_right:
                sd.setdefault("border_width", int(table_border_width))
                sd.setdefault("border_color", table_border_color)
                sd.setdefault("border_style", "solid")

        # ── cell border from layout config ────────────────────────────
        if cell_border:
            sd.setdefault("border_width", int(cell_border_width))
            sd.setdefault("border_color", cell_border_color)
            sd.setdefault("border_style", "solid")

        # ── Text dict ─────────────────────────────────────────────────
        td: Dict[str, Any] = {
            "text": label,
            "column": [col, row],
            "get_ip": table_ip,
        }

        customise: Dict[str, Any] = {}

        # Text color
        effective_text_color = text_color
        if cell_overrides and "text_color" in cell_overrides:
            effective_text_color = cell_overrides["text_color"]
        if effective_text_color is not None:
            customise["color"] = effective_text_color

        # Font family
        effective_font_family = font_family
        if cell_overrides and "font_family" in cell_overrides:
            effective_font_family = cell_overrides["font_family"]
        if effective_font_family is not None:
            customise["font_family"] = effective_font_family

        # Font size
        effective_font_size = font_size
        if cell_overrides and "font_size" in cell_overrides:
            effective_font_size = cell_overrides["font_size"]
        customise["font_size"] = int(effective_font_size)

        # Text alignment
        effective_text_align = text_align
        if cell_overrides and "text_align" in cell_overrides:
            effective_text_align = cell_overrides["text_align"]
        if effective_text_align is not None:
            customise["align_text"] = effective_text_align

        # Bold (keys bold by default, or cell override)
        if cell_overrides and "bold" in cell_overrides:
            customise["bold"] = bool(cell_overrides["bold"])
        elif is_key and bold_keys:
            customise["bold"] = True

        # Line height
        effective_line_height = line_height
        if cell_overrides and "line_height" in cell_overrides:
            effective_line_height = cell_overrides["line_height"]
        if effective_line_height is not None:
            customise["line_height"] = float(effective_line_height)

        # Italic
        if cell_overrides and "italic" in cell_overrides:
            customise["italic"] = bool(cell_overrides["italic"])

        # Underline
        if cell_overrides and "underline" in cell_overrides:
            customise["underline"] = bool(cell_overrides["underline"])

        # Background padding (from layout padding)
        if pad_v > 0 or pad_h > 0:
            customise["background_padding"] = max(pad_v, pad_h)

        # ── animation ─────────────────────────────────────────────────
        if animate_in != "none":
            anim_map = {
                "fade":  "fade_in",
                "slide": "slide_in",
                "scale": "scale_in",
            }
            customise["animation"] = anim_map.get(animate_in, "fade_in")
            customise["duration"] = animate_speed / 1000.0   # ms → seconds
            customise["loop"] = False
            # Stagger delay: each cell appears slightly after the previous
            customise["delay"] = cell_counter * 0.04  # 40ms stagger per cell

        if customise:
            td["customise"] = customise

        shapes.append(sd)
        texts.append(td)
        cell_counter += 1

    # ── populate cells ────────────────────────────────────────────────
    if vertical == "keys":
        for row_idx, key in enumerate(keys):
            _add_cell(key, col=0, row=row_idx, is_key=True)
            values = data[key]
            if not isinstance(values, (list, tuple)):
                values = [values]
            for col_idx, val in enumerate(values):
                label, overrides = _resolve_cell_value(val)
                _add_cell(
                    label,
                    col=col_idx + 1,
                    row=row_idx,
                    is_key=False,
                    cell_overrides=overrides if overrides else None,
                )
    else:
        for col_idx, key in enumerate(keys):
            _add_cell(key, col=col_idx, row=0, is_key=True)
            values = data[key]
            if not isinstance(values, (list, tuple)):
                values = [values]
            for row_idx, val in enumerate(values):
                label, overrides = _resolve_cell_value(val)
                _add_cell(
                    label,
                    col=col_idx,
                    row=row_idx + 1,
                    is_key=False,
                    cell_overrides=overrides if overrides else None,
                )

    # ── gap between cells ─────────────────────────────────────────────
    gap = int(layout_cfg.get("cell_margin", 2))

    # ── table dimension dict ──────────────────────────────────────────
    table_def: Dict[str, Any] = {
        "type": "table",
        "rows": num_rows,
        "columns": num_cols,
        "origin": layout_cfg.get("origin", "top-left"),
        "margin": {
            "show": False,
            "cell_margin": gap,
            "top":    table_margin_top,
            "right":  table_margin_right,
            "bottom": table_margin_bottom,
            "left":   table_margin_left,
        },
        # Pass per-column widths and row height so cell_rect positions correctly
        "col_widths": effective_col_widths,
        "row_height": effective_row_height,
    }

    try:
        total_w = sum(effective_col_widths) + gap * max(0, num_cols - 1)
        total_h = effective_row_height * num_rows + gap * max(0, num_rows - 1)
        # Add table margins
        total_w += table_margin_left + table_margin_right
        total_h += table_margin_top  + table_margin_bottom
        table_def["width"]  = total_w
        table_def["height"] = total_h
    except (TypeError, ValueError):
        pass

    return shapes, texts, table_def


def sort_list_data(data: Dict[str, List[Any]], sort_by: str, sort_order: str = "asc") -> Dict[str, List[Any]]:
    """Sort a list data dictionary by column key."""
    if sort_by not in data:
        return data
    keys = list(data.keys())
    sort_values = data[sort_by]
    if not isinstance(sort_values, (list, tuple)):
        sort_values = [sort_values]
    
    indexed = []
    for idx, v in enumerate(sort_values):
        label, _ = _resolve_cell_value(v)
        indexed.append((idx, label))

    rev = (str(sort_order).lower() == "desc")
    try:
        indexed.sort(key=lambda iv: iv[1], reverse=rev)
    except TypeError:
        indexed.sort(key=lambda iv: str(iv[1]), reverse=rev)

    sort_indices = [i for i, _ in indexed]
    sorted_data: Dict[str, List[Any]] = {}
    for k in keys:
        vals = data[k]
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        sorted_data[k] = [vals[i] if i < len(vals) else "" for i in sort_indices]
    return sorted_data


def search_list_data(data: Dict[str, List[Any]], query: str, fields: Optional[List[str]] = None) -> Dict[str, List[Any]]:
    """Filter list data dictionary rows matching a search query string."""
    if not query or not data:
        return data
    q = query.strip().lower()
    keys = list(data.keys())
    search_keys = fields if fields else keys

    max_len = max((len(v) if isinstance(v, (list, tuple)) else 1 for v in data.values()), default=0)
    matched_indices = []

    for row_idx in range(max_len):
        row_match = False
        for k in search_keys:
            if k in data:
                vals = data[k]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                if row_idx < len(vals):
                    label, _ = _resolve_cell_value(vals[row_idx])
                    if q in str(label).lower():
                        row_match = True
                        break
        if row_match:
            matched_indices.append(row_idx)

    filtered_data: Dict[str, List[Any]] = {}
    for k in keys:
        vals = data[k]
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        filtered_data[k] = [vals[i] for i in matched_indices if i < len(vals)]
    return filtered_data


def filter_list_data(data: Dict[str, List[Any]], predicate: Callable[[Dict[str, Any]], bool]) -> Dict[str, List[Any]]:
    """Filter list data dictionary using a row predicate function."""
    if not data or not callable(predicate):
        return data
    keys = list(data.keys())
    max_len = max((len(v) if isinstance(v, (list, tuple)) else 1 for v in data.values()), default=0)
    matched_indices = []

    for row_idx in range(max_len):
        row_dict = {}
        for k in keys:
            vals = data[k]
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            label, _ = _resolve_cell_value(vals[row_idx] if row_idx < len(vals) else "")
            row_dict[k] = label

        try:
            if predicate(row_dict):
                matched_indices.append(row_idx)
        except Exception:
            pass

    filtered_data: Dict[str, List[Any]] = {}
    for k in keys:
        vals = data[k]
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        filtered_data[k] = [vals[i] for i in matched_indices if i < len(vals)]
    return filtered_data