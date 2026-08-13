"""
Draw._widget
=============
High-level widget-grid helper for Draw.

Solves the "Widget templates / composition" gap (Tier 2):
building a grid of repeated components (e.g. calculator buttons) used to
require manual coordinate math for every cell. ``Draw.grid`` lets you
define ONE reusable template (shape + optional text style) and stamp it
across an auto-generated table layout, with per-cell overrides for the
things that actually differ (label, ip, color...).

Usage 
-----
    Draw.grid(
        display    = "calc",
        ip         = "keypad",                 # table layout ip (auto get_ip)
        dimension  = {"rows": 4, "columns": 4}, # or pass an existing get_ip=...
        template   = {                          # shared shape style
            "size": [70, 70],
            "border_radius": 12,
            "color": "#2c3e50",
            "border_width": 1,
            "border_color": "#34495e",
        },
        text_template = {                       # shared text style (optional)
            "font_size": 22,
            "color": "white",
            "align_text": "center",
        },
        items = [
            {"label": "7"}, {"label": "8"}, {"label": "9"}, {"label": "/"},
            {"label": "4"}, {"label": "5"}, {"label": "6"}, {"label": "*"},
            {"label": "1"}, {"label": "2"}, {"label": "3"}, {"label": "-"},
            {"label": "0"}, {"label": "."}, {"label": "="}, {"label": "+"},
        ],
    )

Each entry in ``items`` is placed left-to-right, top-to-bottom into the
grid (row-major order) unless it supplies an explicit ``"cell": [col, row]``.
Per-item keys ("label", "color", "ip", ...) override the template for that
one cell only. ``ip`` defaults to ``f"{grid_ip}_{col}_{row}"`` so every
button gets a stable, predictable identity automatically — no manual
positioning or naming required.

Returns
-------
(shape_items, text_items, table_def) — same convention as
``Draw._list._generate_list_items`` — ready to be merged into
``canvas.shape_items`` / ``canvas.text_items`` by ``Draw.shapes``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple


# Keys that belong on the shape side of a template/item (vs. the text side).
_SHAPE_KEYS = {
    "vertices", "size", "width", "height", "border_radius", "border_width",
    "border_color", "border_style", "curve_mode", "bend", "exclude",
    "symmetry", "overlap", "opacity", "rotation", "z", "color", "custom",
    "area", "flow",
}

_TEXT_KEYS = {
    "font_family", "font_size", "bold", "italic", "underline",
    "strikethrough", "letter_spacing", "line_height", "align_text",
    "background_color", "background_padding", "border_width", "border_color",
    "border_radius", "glow", "glow_color", "glow_radius", "shadow",
    "shadow_color", "shadow_offset", "rotation",
}

# Keys that are control/meta fields on a grid item, never copied into the
# generated shape/text dicts as-is.
_META_KEYS = {"label", "text", "ip", "cell", "column", "row", "col", "onclick"}


def _split_template(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(raw) if isinstance(raw, dict) else {}


def _resolve_cell(item: Dict[str, Any], index: int, columns: int) -> Tuple[int, int]:
    """Return (col, row) for a grid item — explicit 'cell'/'column' wins,
    otherwise row-major auto-placement from its position in the items list."""
    explicit = item.get("cell", item.get("column", None))
    if explicit is not None:
        if isinstance(explicit, (list, tuple)) and len(explicit) == 2:
            return int(explicit[0]), int(explicit[1])
        if isinstance(explicit, dict):
            return int(explicit.get("col", 0)), int(explicit.get("row", 0))

    col = item.get("col", None)
    row = item.get("row", None)
    if col is not None and row is not None:
        return int(col), int(row)

    if columns <= 0:
        columns = 1
    return index % columns, index // columns


def generate_grid_items(
    grid_ip: Optional[str],
    *,
    dimension: Optional[Dict[str, Any]] = None,
    get_ip: object = None,
    template: Optional[Dict[str, Any]] = None,
    text_template: Optional[Dict[str, Any]] = None,
    items: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[dict], List[dict], Optional[dict]]:
    """
    Build (shape_items, text_items, table_def) for a widget grid.

    Parameters
    ----------
    grid_ip       : ip prefix used to name the auto-created table layout
                     and to derive each cell's default ip
                     (f"{grid_ip}_{col}_{row}").
    dimension     : {"rows": int, "columns": int, ...} passed straight
                     through to Draw.table(dimension=...) when no existing
                     layout is supplied via get_ip.
    get_ip        : reuse an existing Draw.table(...) layout instead of
                     creating a new one from `dimension`.
    template      : shared shape style applied to every cell before
                     per-item overrides.
    text_template : shared text style applied to every cell's label
                     before per-item overrides. Omit entirely to skip
                     text generation (shape-only grid).
    items         : list of per-cell dicts. Recognised meta keys:
                       "label"/"text" : the text shown on the cell
                       "ip"           : explicit identity for this cell
                                        (defaults to f"{grid_ip}_{col}_{row}")
                       "cell"/"column": explicit [col, row] placement
                       "onclick"      : ip alias stored for sense lookups
                                        (you still wire it with
                                        Draw.connectors / Draw.senses)
                     All other keys override the shape or text template
                     for that one cell (color, font_size, border_radius...).

    Returns
    -------
    shapes, texts, table_def — same contract as _list._generate_list_items.
    table_def is None when neither `dimension` nor `get_ip` is usable.
    """
    items = list(items or [])
    template = _split_template(template)
    text_template = text_template if isinstance(text_template, dict) else None

    table_ip = f"{grid_ip}_grid_table" if grid_ip else "grid_table"

    table_def: Optional[dict] = None
    layout_ref: object = None

    if get_ip is not None:
        # Caller is reusing an existing Draw.table(...) layout.
        layout_ref = get_ip
    elif dimension is not None:
        if not isinstance(dimension, dict):
            raise TypeError("Draw.grid: 'dimension' must be a dict.")
        table_def = dict(dimension)
        layout_ref = table_ip
    else:
        raise ValueError("Draw.grid: provide either 'dimension' or 'get_ip'.")

    columns = 1
    if isinstance(dimension, dict):
        columns = int(dimension.get("columns", dimension.get("column", 1)) or 1)
    elif get_ip is not None:
        # Resolve column count from the referenced table layout
        try:
            from Draw._layout import set as _layout_set
            ref_layout = _layout_set.resolve(get_ip)
            if hasattr(ref_layout, 'columns'):
                columns = max(1, int(ref_layout.columns))
        except Exception:
            columns = 1

    shapes: List[dict] = []
    texts: List[dict] = []

    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, dict):
            raise TypeError("Draw.grid: every entry in 'items' must be a dict.")

        col, row = _resolve_cell(raw_item, index, columns)
        cell_ip = raw_item.get("ip", None)
        if cell_ip is None:
            cell_ip = f"{grid_ip or table_ip}_{col}_{row}"
        cell_ip = str(cell_ip)

        # ── shape side ───────────────────────────────────────────────
        sd: Dict[str, Any] = copy.deepcopy(template)
        for k, v in raw_item.items():
            if k in _SHAPE_KEYS:
                sd[k] = v
        sd["column"] = [col, row]
        sd["get_ip"] = layout_ref if get_ip is not None else table_ip
        sd["ip"] = cell_ip
        shapes.append(sd)

        # ── text side (only if a text template/label exists) ───────────
        label = raw_item.get("label", raw_item.get("text", None))
        if text_template is not None or label is not None:
            td: Dict[str, Any] = copy.deepcopy(text_template) if text_template else {}
            for k, v in raw_item.items():
                if k in _TEXT_KEYS:
                    td[k] = v
            td["text"] = "" if label is None else str(label)
            td["column"] = [col, row]
            td["get_ip"] = layout_ref if get_ip is not None else table_ip
            td["ip"] = f"{cell_ip}_label"
            texts.append(td)

    return shapes, texts, table_def
