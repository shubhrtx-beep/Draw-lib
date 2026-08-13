"""
Draw._point
===========
Point-based path drawing system.

Draw.point renders sequences of x,y coordinates as stroked (and optionally
filled) paths directly onto a Draw window canvas.  It supports two authoring
styles, per-path animation, curves, and optional hitbox / connector wiring.

──────────────────────────────────────────────────────────────────────────────
QUICK REFERENCE
──────────────────────────────────────────────────────────────────────────────

    import Draw

    Draw.window(tag="main", width=800, height=600, background_color="black")

    Draw.point(
        tag     = "main",          # window to draw on  (or display=)
        graph   = [800, 600],      # canvas coordinate space  [w, h]
        points  = [
            # ── Style 1 : Normal path ──────────────────────────────────────
            {
                "path": "10,10 ; 50,30 ; 90,70",
                "colour": "white",
                "width": 2,
                "edge": "straight",   # "straight" | "curve"
                "fill": False,
            },
            {
                "path": "10,10 ; 50,30 ; 90,70",
                "colour": "cyan",
                "width": 1,
                "edge": "curve",
                "curve_at": {"10%": "20%"},   # bend 20 % when 10 % along line
                "smooth": "40%",
                "fill": True,
                "animation": {
                    "line_animation": {"0%-100%": "ease"},
                    "opacity": {"0%-100%": "linear"},
                },
            },

            # ── Style 2 : Custom path (column-schema) ──────────────────────
            {
                "item":   ["time",   "edge",      "animation",     "colour" ],
                "10,20":  ["0s",     "straight",  "line_animation","white"  ],
                "40,60":  ["1s",     " ",         "line_animation","cyan"   ],
                "80,30":  ["2s",     "curve",     " ",             "blue"   ],

                # optional wiring
                "ip":     "my_path",
                "hit_box": [{"width": "20px", "height": "20px"}],
                "return": ["text", "my_key"],
            },
        ],
    )

    Draw.window.run("main")

──────────────────────────────────────────────────────────────────────────────
STYLE 1 — NORMAL PATH
──────────────────────────────────────────────────────────────────────────────

Key in the points dict  : "path"  (required)
Value                   : "x,y ; x,y ; x,y"  coordinate string

Additional keys (all optional)
───────────────────────────────
    colour / color      str | hex | RGB tuple         default "white"
    width               int | float  (stroke width)   default 1
    edge                "straight" | "curve"           default "straight"
    smooth              "N%"  — Catmull-Rom tension    default "40%"
    curve_at            dict {"N%": "M%"}
                        When the line has advanced N % of its length,
                        deflect the stroke by M % of the segment length.
    fill                bool  — close and fill the path  default False
    fill_colour         colour for fill (default same as colour)
    opacity             0-100                           default 100
    animation           dict  (see ANIMATION below)

──────────────────────────────────────────────────────────────────────────────
STYLE 2 — CUSTOM PATH  (column-schema)
──────────────────────────────────────────────────────────────────────────────

The dict must contain an "item" key whose value is a list of column names.
Every other coordinate key ("x,y") is a list of values in the same order.
Use an empty string or " " to skip a column for that row.
Style columns define the path's defaults from the first non-empty value;
``time`` and ``animation`` remain per-row controls.

Recognised column names:
    time            "Ns"  — draw-on animation start time (seconds)
    edge            "straight" | "curve"
    curve_at        "N%:M%" shorthand
    smooth          "N%"
    animation       animation name string
    graph           easing name  ("linear","ease","ease_in","ease_out", …)
    colour / color  colour value
    width           stroke width
    opacity         0-100
    fill            bool

──────────────────────────────────────────────────────────────────────────────
ANIMATION
──────────────────────────────────────────────────────────────────────────────

    "animation": {
        "line_animation": {"0%-100%": "ease"},   # draw-on reveal
        "opacity":        {"0%-100%": "linear"},
    }

Supported animation names: line_animation, opacity.
Progress range format:  "START%-END%"  e.g. "0%-100%", "25%-75%".

──────────────────────────────────────────────────────────────────────────────
OPTIONAL WIRING KEYS  (per points-block)
──────────────────────────────────────────────────────────────────────────────

    ip       str   — registers the path so Draw.senses / Draw.connectors
                     can reference it
    hit_box  list  — list of {"width": "Npx", "height": "Npx"} dicts that
                     define the clickable bounding box; wired via Draw.hitbox
    return   any   — passed to Draw.input's return-value store on interaction
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt, QTimer
# pyrefly: ignore [missing-import]
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPainterPath, QPainterPathStroker, QPen,
)
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QWidget, QMainWindow

from Draw._app import get_app
from Draw._window import window as _window_registry
from Draw._text import _get_or_create_canvas


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _parse_pct(raw: object, default: float = 0.0) -> float:
    """Parse "40%" → 0.40, 0.4 → 0.4, 40 → 40.0."""
    if raw is None:
        return default
    s = str(raw).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    # Bare values are fractions.  Treat the common ``40`` spelling as 40%,
    # rather than producing an unusable spline tension of 40.
    value = float(s)
    return value / 100.0 if abs(value) > 1.0 else value


def _parse_seconds(raw: object, default: float = 0.0) -> float:
    """Parse "2s" → 2.0, "500ms" → 0.5, 1.5 → 1.5."""
    if raw is None:
        return default
    s = str(raw).strip()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def _parse_color_safe(raw: object) -> QColor:
    if raw is None or str(raw).strip() in {"", " "}:
        return QColor("white")
    from Draw import _bridge
    return _bridge.get_color_parser()(raw)


def _parse_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    return s in {"true", "1", "yes", "on"}


def _parse_coord_string(raw: str) -> List[Tuple[float, float]]:
    """
    Parse "x,y ; x,y ; x,y" into [(x,y), ...].
    Accepts semicolons or newlines as separators.
    """
    pts: List[Tuple[float, float]] = []
    for token in re.split(r"[;\n]+", raw):
        token = token.strip()
        if not token:
            continue
        parts = re.split(r"[,\s]+", token, maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return pts


def _parse_single_coord(raw: str) -> Optional[Tuple[float, float]]:
    """Parse "x,y" → (x, y) or None."""
    raw = raw.strip()
    parts = re.split(r"[,\s]+", raw, maxsplit=1)
    if len(parts) < 2:
        return None
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return None


def _parse_range_key(raw: str) -> Tuple[float, float]:
    """
    Parse "0%-100%" or "25%-75%" → (0.0, 1.0) / (0.25, 0.75).
    Also accepts "0%:100%" (colon separator).
    """
    raw = raw.strip()
    for sep in ("-", ":"):
        if sep in raw:
            parts = raw.split(sep, 1)
            return (_parse_pct(parts[0]), _parse_pct(parts[1]))
    v = _parse_pct(raw)
    return (v, v)


def _parse_curve_at(raw: object) -> List[Tuple[float, float]]:
    """
    Parse curve_at value into list of (progress_frac, deflect_frac).
    Accepts:
        {"10%": "20%"}            → [(0.10, 0.20)]
        "10%:20%"                 → [(0.10, 0.20)]
        [("10%", "20%"), ...]     → [...]
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        result = []
        for k, v in raw.items():
            try:
                result.append((_parse_pct(k), _parse_pct(v)))
            except (TypeError, ValueError):
                pass
        return result
    if isinstance(raw, str):
        for sep in (":", "-"):
            if sep in raw:
                parts = raw.split(sep, 1)
                try:
                    return [(_parse_pct(parts[0]), _parse_pct(parts[1]))]
                except (TypeError, ValueError):
                    pass
    if isinstance(raw, (list, tuple)):
        result = []
        for item in raw:
            result.extend(_parse_curve_at(item))
        return result
    return []


_EASING_MAP: Dict[str, Any] = {
    "linear":       lambda t: t,
    "ease":         lambda t: t * t * (3 - 2 * t),  # smoothstep
    "ease_in":      lambda t: t * t,
    "easein":       lambda t: t * t,
    "ease_out":     lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "easeout":      lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "ease_in_out":  lambda t: (t * t * (3 - 2 * t)),
    "easeinout":    lambda t: (t * t * (3 - 2 * t)),
}


def _resolve_easing(name: object):
    key = str(name or "linear").strip().lower()
    return _EASING_MAP.get(key, _EASING_MAP["linear"])


# ── Animation spec ────────────────────────────────────────────────────────────

@dataclass
class _AnimChannel:
    """One animation channel (e.g. line_animation, opacity)."""
    name: str               # "line_animation" | "opacity" | "colour_transition"
    start: float            # 0.0-1.0  normalised progress
    end: float              # 0.0-1.0
    easing: Any             # callable(t) -> t


def _parse_animation_spec(raw: object) -> List[_AnimChannel]:
    """
    Parse the animation dict into a list of _AnimChannel objects.

        {
            "line_animation": {"0%-100%": "ease"},
            "opacity":        {"0%-50%":  "linear"},
        }
    """
    if not isinstance(raw, dict):
        return []
    channels: List[_AnimChannel] = []
    for name, value in raw.items():
        name_key = str(name).strip().lower()
        if name_key not in {"line_animation", "opacity"}:
            raise ValueError(
                f"Draw.point: unsupported animation channel {name!r}; "
                "use 'line_animation' or 'opacity'."
            )
        if isinstance(value, dict):
            for range_key, easing_raw in value.items():
                start, end = _parse_range_key(str(range_key))
                channels.append(_AnimChannel(
                    name=name_key,
                    start=_clamp01(start),
                    end=_clamp01(end),
                    easing=_resolve_easing(easing_raw),
                ))
        elif isinstance(value, str):
            # shorthand: "opacity": "ease"  → full range
            channels.append(_AnimChannel(
                name=name_key, start=0.0, end=1.0,
                easing=_resolve_easing(value),
            ))
    return channels


# ── PathDef ───────────────────────────────────────────────────────────────────

@dataclass
class PathDef:
    """Internal representation of one parsed path."""
    coords: List[Tuple[float, float]]   # raw coordinate pairs

    colour: QColor
    fill: bool
    fill_colour: QColor
    width: float
    edge: str                           # "straight" | "curve"
    smooth: float                       # Catmull-Rom tension 0-1
    curve_at: List[Tuple[float, float]] # [(progress, deflect), ...]
    opacity: int                        # 0-100

    # per-point time offsets (from custom-path style; length == len(coords) or 0)
    time_offsets: List[float]           # seconds for each point

    # animation
    anim_channels: List[_AnimChannel]

    # wiring
    ip: Optional[str]
    hit_box: List[dict]
    return_spec: Any
    flow: object
    align: Optional[str] = None

    # runtime
    started_at: Optional[float] = field(default=None, init=False)
    _flow_dx: float = field(default=0.0, init=False)
    _flow_dy: float = field(default=0.0, init=False)

    # geometry cache — avoids rebuilding QPainterPath on every paint tick
    # when the path is static (no animation or animation already finished).
    _cached_geom: Optional[Any] = field(default=None, init=False, repr=False)
    _cache_key: Optional[Any] = field(default=None, init=False, repr=False)


# ── Parsing helpers ───────────────────────────────────────────────────────────

_COORD_KEY_RE = re.compile(r"^\s*-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?\s*$")


def _is_coord_key(key: str) -> bool:
    return bool(_COORD_KEY_RE.match(key))


_COLUMN_ALIASES = {
    "colour": "colour",
    "color":  "colour",
    "col":    "colour",
    "width":  "width",
    "w":      "width",
    "edge":   "edge",
    "smooth": "smooth",
    "curve_at": "curve_at",
    "animation": "animation",
    "anim":   "animation",
    "graph":  "graph",
    "time":   "time",
    "opacity": "opacity",
    "fill":   "fill",
    "fill_colour": "fill_colour",
    "fill_color":  "fill_colour",
    "align":  "align",
}


def _normalise_column(name: str) -> str:
    return _COLUMN_ALIASES.get(name.strip().lower(), name.strip().lower())


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", " "):
        return True
    return False


def _parse_style1(raw: dict) -> PathDef:
    """
    Parse a Style-1 normal-path dict.
    Required key: "path"  (or the path string is the first key that looks like coords).
    """
    path_str = raw.get("path", None)
    if path_str is None:
        # Fall back: find first coord-string key
        for k in raw:
            if _is_coord_key(k) or ";" in str(k):
                path_str = k
                break
    if path_str is None:
        raise ValueError("Draw.point: Style-1 path dict must contain a 'path' key.")

    coords = _parse_coord_string(str(path_str))
    if len(coords) < 2:
        raise ValueError(
            f"Draw.point: path needs at least 2 coordinate pairs, got {len(coords)}: {path_str!r}"
        )

    colour_raw = raw.get("colour", raw.get("color", "white"))
    colour = _parse_color_safe(colour_raw)
    width = float(raw.get("width", 1))
    if width < 0:
        raise ValueError("Draw.point: 'width' cannot be negative.")
    edge = str(raw.get("edge", "straight")).strip().lower()
    if edge not in ("straight", "curve"):
        edge = "straight"
    smooth = _clamp01(_parse_pct(raw.get("smooth", "40%"), default=0.40))
    curve_at = [(_clamp01(progress), deflect) for progress, deflect in _parse_curve_at(raw.get("curve_at", None))]
    fill = _parse_bool(raw.get("fill", False))
    fill_colour_raw = raw.get("fill_colour", raw.get("fill_color", colour_raw))
    fill_colour = _parse_color_safe(fill_colour_raw)
    opacity = max(0, min(100, int(raw.get("opacity", 100))))
    anim_channels = _parse_animation_spec(raw.get("animation", None))
    ip = str(raw["ip"]) if "ip" in raw else None
    hit_box = list(raw.get("hit_box", []) or [])
    return_spec = raw.get("return", None)
    from Draw._overlap import parse_flow_spec
    flow = parse_flow_spec(
        raw.get("flow", None),
        flow_provided=("flow" in raw),
        overlap=True,
    )

    return PathDef(
        coords=coords,
        colour=colour,
        fill=fill,
        fill_colour=fill_colour,
        width=width,
        edge=edge,
        smooth=smooth,
        curve_at=curve_at,
        opacity=opacity,
        time_offsets=[],
        anim_channels=anim_channels,
        ip=ip,
        hit_box=hit_box,
        return_spec=return_spec,
        flow=flow,
        align=raw.get("align", None),
    )


def _parse_style2(raw: dict) -> PathDef:
    """
    Parse a Style-2 custom-path (column-schema) dict.

    The "item" key declares column names.
    Every other "x,y" key is a value row.
    Wiring keys (ip, hit_box, return) are extracted separately.
    """
    schema_raw = raw.get("item", [])
    if not isinstance(schema_raw, (list, tuple)):
        schema_raw = [schema_raw]
    schema: List[str] = [_normalise_column(str(c)) for c in schema_raw]

    # Collect ordered coord rows
    coord_rows: List[Tuple[Tuple[float, float], List[Any]]] = []
    for key, values in raw.items():
        if key in ("item", "ip", "hit_box", "return", "flow", "align",
                    "opacity", "display", "tag", "graph", "width",
                    "colour", "color", "edge", "smooth", "fill",
                    "fill_colour", "fill_color"):
            continue
        coord = _parse_single_coord(str(key))
        if coord is None:
            continue
        if not isinstance(values, (list, tuple)):
            values = [values]
        coord_rows.append((coord, list(values)))

    if len(coord_rows) < 2:
        raise ValueError(
            "Draw.point: Style-2 custom-path needs at least 2 coordinate rows."
        )

    # Build defaults
    coords: List[Tuple[float, float]] = []
    time_offsets: List[float] = []
    edge_values: List[str] = []
    colour_values: List[Optional[QColor]] = []
    width_values: List[Optional[float]] = []
    opacity_values: List[Optional[int]] = []
    fill_values: List[Optional[bool]] = []
    fill_colour_values: List[Optional[QColor]] = []
    smooth_values: List[Optional[float]] = []
    curve_at_values: List[object] = []
    anim_names: List[Optional[str]] = []
    graph_names: List[Optional[str]] = []

    for (coord, values) in coord_rows:
        coords.append(coord)
        row: Dict[str, Any] = {}
        for col_idx, col_name in enumerate(schema):
            val = values[col_idx] if col_idx < len(values) else None
            if not _is_blank(val):
                row[col_name] = val

        time_offsets.append(_parse_seconds(row.get("time", None), default=0.0))
        edge_values.append(
            str(row.get("edge", "straight")).strip().lower()
            if "edge" in row else "straight"
        )
        colour_values.append(
            _parse_color_safe(row["colour"]) if "colour" in row else None
        )
        if "width" in row:
            width_value = float(row["width"])
            if width_value < 0:
                raise ValueError("Draw.point: Style-2 'width' cannot be negative.")
            width_values.append(width_value)
        else:
            width_values.append(None)
        opacity_values.append(int(row["opacity"]) if "opacity" in row else None)
        fill_values.append(_parse_bool(row["fill"]) if "fill" in row else None)
        fill_colour_values.append(_parse_color_safe(row["fill_colour"]) if "fill_colour" in row else None)
        smooth_values.append(_clamp01(_parse_pct(row["smooth"])) if "smooth" in row else None)
        curve_at_values.append(row.get("curve_at"))
        anim_names.append(str(row["animation"]) if "animation" in row else None)
        graph_names.append(str(row["graph"]) if "graph" in row else None)

    # Resolve global fallbacks from first non-None value
    def _first(lst: list, default: Any) -> Any:
        for item in lst:
            if item is not None:
                return item
        return default

    colour = _first(colour_values, QColor("white"))
    width = _first(width_values, 1.0)
    opacity = _first(opacity_values, 100)
    fill = _first(fill_values, False)
    fill_colour = _first(fill_colour_values, colour)
    edge = _first(edge_values, "straight")
    if edge not in ("straight", "curve"):
        edge = "straight"
    smooth = _first(smooth_values, 0.40)
    curve_at = [(_clamp01(progress), deflect) for progress, deflect in _parse_curve_at(_first(curve_at_values, None))]

    # Build animation channels from per-row anim_names and graph_names
    anim_channels: List[_AnimChannel] = []
    for i, anim_name in enumerate(anim_names):
        if anim_name is None:
            continue
        if len(coords) < 2:
            continue
        start = float(i) / max(1, len(coords) - 1)
        end = float(i + 1) / max(1, len(coords) - 1)
        easing_name = graph_names[i] if i < len(graph_names) and graph_names[i] else "linear"
        anim_channels.append(_AnimChannel(
            name=anim_name.lower(),
            start=_clamp01(start),
            end=_clamp01(end),
            easing=_resolve_easing(easing_name),
        ))

    ip = str(raw["ip"]) if "ip" in raw else None
    hit_box = list(raw.get("hit_box", []) or [])
    return_spec = raw.get("return", None)
    from Draw._overlap import parse_flow_spec
    flow = parse_flow_spec(
        raw.get("flow", None),
        flow_provided=("flow" in raw),
        overlap=True,
    )

    return PathDef(
        coords=coords,
        colour=colour,
        fill=fill,
        fill_colour=fill_colour,
        width=width,
        edge=edge,
        smooth=smooth,
        curve_at=curve_at,
        opacity=opacity,
        time_offsets=time_offsets,
        anim_channels=anim_channels,
        ip=ip,
        hit_box=hit_box,
        return_spec=return_spec,
        flow=flow,
        align=raw.get("align", None),
    )


def _parse_path_entry(raw: dict) -> PathDef:
    """Detect style and delegate to the right parser."""
    if "item" in raw:
        return _parse_style2(raw)
    return _parse_style1(raw)


# ── Geometry ──────────────────────────────────────────────────────────────────

def _scale_coords(
    coords: List[Tuple[float, float]],
    graph_w: float,
    graph_h: float,
    canvas_w: int,
    canvas_h: int,
) -> List[QPointF]:
    """Map graph-space coordinates to canvas pixels."""
    if graph_w <= 0:
        graph_w = 1.0
    if graph_h <= 0:
        graph_h = 1.0
    sx = canvas_w / graph_w
    sy = canvas_h / graph_h
    return [QPointF(x * sx, y * sy) for (x, y) in coords]


def _catmull_rom_point(
    p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF, t: float, tension: float = 0.5
) -> QPointF:
    """Evaluate a Catmull-Rom spline at parameter t."""
    t2 = t * t
    t3 = t2 * t
    alpha = tension
    x = (
        alpha * ((-p0.x() + 3*p1.x() - 3*p2.x() + p3.x()) * t3
                 + (2*p0.x() - 5*p1.x() + 4*p2.x() - p3.x()) * t2
                 + (-p0.x() + p2.x()) * t)
        + p1.x()
    )
    y = (
        alpha * ((-p0.y() + 3*p1.y() - 3*p2.y() + p3.y()) * t3
                 + (2*p0.y() - 5*p1.y() + 4*p2.y() - p3.y()) * t2
                 + (-p0.y() + p2.y()) * t)
        + p1.y()
    )
    return QPointF(x, y)


def _build_curve_path(
    pts: List[QPointF],
    smooth: float,
    curve_at: List[Tuple[float, float]],
    reveal_frac: float = 1.0,
) -> QPainterPath:
    """
    Build a smooth Catmull-Rom QPainterPath through pts.
    reveal_frac (0-1) controls how far along the path is drawn (line_animation).
    """
    pts = _insert_curve_bends(pts, curve_at)
    if len(pts) < 2:
        return QPainterPath()

    steps_per_segment = 24
    tension = max(0.01, smooth * 0.5)

    # Compute total arc-length for reveal
    arc_pts: List[QPointF] = []
    n = len(pts)
    for seg in range(n - 1):
        p0 = pts[max(0, seg - 1)]
        p1 = pts[seg]
        p2 = pts[min(n - 1, seg + 1)]
        p3 = pts[min(n - 1, seg + 2)]
        for step in range(steps_per_segment):
            t = step / steps_per_segment
            arc_pts.append(_catmull_rom_point(p0, p1, p2, p3, t, tension))
    arc_pts.append(pts[-1])

    # Compute cumulative lengths
    lengths = [0.0]
    for i in range(1, len(arc_pts)):
        dx = arc_pts[i].x() - arc_pts[i - 1].x()
        dy = arc_pts[i].y() - arc_pts[i - 1].y()
        lengths.append(lengths[-1] + math.hypot(dx, dy))
    total = lengths[-1] if lengths[-1] > 0 else 1.0
    cut = total * _clamp01(reveal_frac)

    path = QPainterPath()
    started = False
    for i, pt in enumerate(arc_pts):
        if lengths[i] > cut + 1e-6:
            break
        if not started:
            path.moveTo(pt)
            started = True
        else:
            path.lineTo(pt)

    return path


def _build_straight_path(
    pts: List[QPointF],
    curve_at: List[Tuple[float, float]],
    reveal_frac: float = 1.0,
) -> QPainterPath:
    """
    Build a straight-segment QPainterPath through pts.
    curve_at applies perpendicular deflection at specified progress points.
    reveal_frac trims the path for line_animation.
    """
    if len(pts) < 2:
        return QPainterPath()

    full_pts = _insert_curve_bends(pts, curve_at)

    # Compute cumulative lengths for reveal
    lengths = [0.0]
    for i in range(1, len(full_pts)):
        dx = full_pts[i].x() - full_pts[i - 1].x()
        dy = full_pts[i].y() - full_pts[i - 1].y()
        lengths.append(lengths[-1] + math.hypot(dx, dy))
    total = lengths[-1] if lengths[-1] > 0 else 1.0
    cut = total * _clamp01(reveal_frac)

    path = QPainterPath()
    started = False
    for i, pt in enumerate(full_pts):
        if lengths[i] > cut + 1e-6:
            break
        if not started:
            path.moveTo(pt)
            started = True
        else:
            path.lineTo(pt)

    return path


def _insert_curve_bends(
    pts: List[QPointF], curve_at: List[Tuple[float, float]],
) -> List[QPointF]:
    """Insert ordered perpendicular bend points at whole-path progress values."""
    if len(pts) < 2 or not curve_at:
        return list(pts)
    bends = sorted(curve_at, key=lambda item: item[0])
    result: List[QPointF] = []
    segment_count = len(pts) - 1
    for seg in range(segment_count):
        p1, p2 = pts[seg], pts[seg + 1]
        result.append(p1)
        start, end = seg / segment_count, (seg + 1) / segment_count
        for progress, deflect in bends:
            if start < progress <= end:
                local_t = (progress - start) / max(1e-9, end - start)
                dx, dy = p2.x() - p1.x(), p2.y() - p1.y()
                length = math.hypot(dx, dy)
                if length:
                    x = p1.x() + dx * local_t - dy / length * deflect * length
                    y = p1.y() + dy * local_t + dx / length * deflect * length
                    result.append(QPointF(x, y))
    result.append(pts[-1])
    return result


# ── Animation evaluation ──────────────────────────────────────────────────────

def _eval_channel(ch: _AnimChannel, global_t: float) -> Optional[float]:
    """
    Given global animation progress global_t (0-1), evaluate this channel.
    Returns normalised local progress (0-1) or None if not yet active.
    """
    if global_t < ch.start:
        return None
    if ch.end <= ch.start:
        return 1.0
    raw_t = (global_t - ch.start) / (ch.end - ch.start)
    return _clamp01(ch.easing(_clamp01(raw_t)))


# ── Canvas widget ─────────────────────────────────────────────────────────────

class _PointCanvas(QWidget):
    """
    A lightweight QWidget that renders a list of PathDef objects.
    One instance per Draw.point() call — placed over the window's existing
    Draw canvas so shapes/text and points all layer correctly.
    """
    _TICK_MS = 16   # ~60 fps

    def __init__(self, parent: QMainWindow, paths: List[PathDef],
                 graph_w: float, graph_h: float) -> None:
        super().__init__(parent)
        # Input is intercepted on the shared Draw canvas by eventFilter below.
        # Keeping this paint-only overlay transparent prevents it from hiding
        # shapes/text interaction in areas without a point-path hitbox.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setGeometry(parent.rect())
        self._paths = paths
        self._graph_w = graph_w
        self._graph_h = graph_h
        self._timer = QTimer(self)
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._has_animation = any(p.anim_channels for p in paths)

    def _tick(self) -> None:
        if not self._has_animation:
            return
        now = time.perf_counter()
        all_done = True
        for p in self._paths:
            if not p.anim_channels:
                continue
            if p.started_at is None:
                all_done = False
                break
            elapsed = now - p.started_at
            max_time = max(p.time_offsets) if p.time_offsets else 0.0
            anim_duration = max(max_time, 1.0)
            if elapsed < anim_duration:
                all_done = False
                break
        if all_done:
            self._has_animation = False
            self._timer.stop()
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self.parent():
            self.setGeometry(self.parent().rect())  # type: ignore[union-attr]
        super().resizeEvent(event)

    def _path_bounds(self, p: PathDef, cw: int, ch: int, *, include_flow: bool = True):
        if not p.coords or len(p.coords) < 2:
            return None
        scaled = _scale_coords(p.coords, self._graph_w, self._graph_h, cw, ch)
        if p.edge == "curve":
            geom = _build_curve_path(scaled, p.smooth, p.curve_at, 1.0)
        else:
            geom = _build_straight_path(scaled, p.curve_at, 1.0)
        if geom.isEmpty():
            return None
        bounds = geom.controlPointRect().adjusted(
            -p.width / 2.0,
            -p.width / 2.0,
            p.width / 2.0,
            p.width / 2.0,
        )
        from Draw._overlap import Rect
        x = float(bounds.x()) + (getattr(p, "_flow_dx", 0.0) if include_flow else 0.0)
        y = float(bounds.y()) + (getattr(p, "_flow_dy", 0.0) if include_flow else 0.0)
        return Rect(
            x,
            y,
            float(bounds.width()),
            float(bounds.height()),
        )

    def _compute_flow_offsets(self, cw: int, ch: int) -> None:
        from Draw._overlap import Rect, flow_occupied_rect, get_strategy_for_flow

        canvas_bounds = Rect(0.0, 0.0, float(cw), float(ch))
        occupied = []
        parent = self.parent()
        shared_canvas = getattr(parent, "_draw_canvas", None) if parent is not None else None
        if shared_canvas is not None:
            occupied.extend(list(getattr(shared_canvas, "_global_occupied", []) or []))

        for p in self._paths:
            p._flow_dx = 0.0
            p._flow_dy = 0.0
            flow_spec = getattr(p, "flow", None)
            if (
                flow_spec is None
                or not getattr(flow_spec, "enabled", False)
                or getattr(flow_spec, "role", "item") == "ignore"
            ):
                continue

            # Flow is recalculated from intrinsic geometry; using an old flow
            # offset here would feed the prior frame back into placement.
            bounds = self._path_bounds(p, cw, ch, include_flow=False)
            if bounds is None:
                continue

            if getattr(flow_spec, "role", "item") == "item":
                strategy = get_strategy_for_flow(flow_spec)
                position = strategy.find_position(
                    Rect(0.0, 0.0, bounds.w, bounds.h),
                    occupied,
                    canvas_bounds,
                    bounds.x,
                    bounds.y,
                )
                if position is None:
                    position = (bounds.x, bounds.y)
            else:
                position = (bounds.x, bounds.y)

            p._flow_dx = float(position[0] - bounds.x)
            p._flow_dy = float(position[1] - bounds.y)
            occupied.append(
                flow_occupied_rect(position[0], position[1], bounds.w, bounds.h, flow_spec)
            )

    def _hit_path_at(self, pos: QPointF) -> Optional[PathDef]:
        """Return the frontmost interactive path containing *pos*."""
        cw, ch = self.width(), self.height()
        self._compute_flow_offsets(cw, ch)
        for p in reversed(self._paths):
            if not p.ip or not p.hit_box:
                continue
            scaled = _scale_coords(p.coords, self._graph_w, self._graph_h, cw, ch)
            if p.edge == "curve":
                geom = _build_curve_path(scaled, p.smooth, p.curve_at, 1.0)
            else:
                geom = _build_straight_path(scaled, p.curve_at, 1.0)
            if geom.isEmpty():
                continue
            dx = getattr(p, "_flow_dx", 0.0)
            dy = getattr(p, "_flow_dy", 0.0)
            adjusted_pos = QPointF(pos.x() - dx, pos.y() - dy)
            stroker = QPainterPathStroker()
            stroker.setWidth(max(10.0, p.width + 4.0))
            stroke_area = stroker.createStroke(geom)
            if p.fill:
                closed = QPainterPath(geom)
                closed.closeSubpath()
                if closed.contains(adjusted_pos) or stroke_area.contains(adjusted_pos):
                    return p
            elif stroke_area.contains(adjusted_pos):
                return p
        return None

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        """Route point hitboxes through the shared Draw senses dispatcher."""
        if watched is not getattr(self.parent(), "_draw_canvas", None):
            return False
        event_types = {
            QEvent.Type.MouseButtonPress: "mouse_press",
            QEvent.Type.MouseButtonRelease: "mouse_release",
            QEvent.Type.MouseButtonDblClick: "mouse_doubleclick",
        }
        event_type = event_types.get(event.type())
        if event_type is None:
            return False
        path_def = self._hit_path_at(event.position())
        if path_def is None:
            return False
        button = event.button()
        button_name = {
            Qt.MouseButton.LeftButton: "left",
            Qt.MouseButton.RightButton: "right",
            Qt.MouseButton.MiddleButton: "middle",
        }.get(button, "left")
        from Draw._connectors import senses
        senses.dispatch_mouse_event(event_type, path_def.ip, button_name)
        if event_type == "mouse_press":
            specific = "mouse_leftclick" if button_name == "left" else "mouse_rightclick" if button_name == "right" else "mouse_click"
            senses.dispatch_mouse_event("mouse_click", path_def.ip, button_name)
            senses.dispatch_mouse_event(specific, path_def.ip, button_name)
            if path_def.return_spec is not None:
                from Draw._text import _store_input_return_value
                _store_input_return_value(path_def.return_spec, path_def.ip)
        return True

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            cw, ch = self.width(), self.height()
            now = time.perf_counter()
            self._compute_flow_offsets(cw, ch)

            for path_def in self._paths:
                self._draw_path(painter, path_def, cw, ch, now)
        finally:
            if painter.isActive():
                painter.end()

    def _draw_path(
        self,
        painter: QPainter,
        p: PathDef,
        cw: int,
        ch: int,
        now: float,
    ) -> None:
        if not p.coords or len(p.coords) < 2:
            return

        # Ensure animation has a start time
        if p.started_at is None:
            p.started_at = now

        elapsed = now - p.started_at

        # Compute maximum time offset for normalisation
        max_time = max(p.time_offsets) if p.time_offsets else 0.0
        anim_duration = max(max_time, 1.0)

        # Global animation progress (0→1 over anim_duration seconds, then stays 1)
        global_t = _clamp01(elapsed / anim_duration) if anim_duration > 0 else 1.0

        # Evaluate channels
        reveal_frac = 1.0
        opacity_frac = 1.0
        for anim_ch in p.anim_channels:
            ch_t = _eval_channel(anim_ch, global_t)
            if anim_ch.name == "line_animation":
                reveal_frac = ch_t if ch_t is not None else (0.0 if global_t < anim_ch.start else 1.0)
            elif anim_ch.name == "opacity":
                opacity_frac = ch_t if ch_t is not None else (0.0 if global_t < anim_ch.start else 1.0)

        # Scale coordinates
        scaled = _scale_coords(p.coords, self._graph_w, self._graph_h, cw, ch)

        # Build path geometry — use cache when reveal_frac is stable
        cache_key = (tuple(p.coords), reveal_frac, cw, ch, p.edge, p.smooth)
        if p._cache_key == cache_key and p._cached_geom is not None:
            geom = p._cached_geom
        else:
            if p.edge == "curve":
                geom = _build_curve_path(scaled, p.smooth, p.curve_at, reveal_frac)
            else:
                geom = _build_straight_path(scaled, p.curve_at, reveal_frac)
            p._cached_geom = geom
            p._cache_key = cache_key

        if geom.isEmpty():
            return

        final_opacity = (p.opacity / 100.0) * _clamp01(opacity_frac)

        painter.save()
        painter.setOpacity(final_opacity)
        dx = getattr(p, "_flow_dx", 0.0)
        dy = getattr(p, "_flow_dy", 0.0)
        if dx or dy:
            painter.translate(dx, dy)

        stroke_colour = QColor(p.colour)
        if p.ip:
            from Draw import _bridge
            dynamic = _bridge.resolve_dynamic_color(
                p.ip, x=geom.controlPointRect().x() + dx,
                y=geom.controlPointRect().y() + dy,
                w=geom.controlPointRect().width(), h=geom.controlPointRect().height(),
            )
            if dynamic and dynamic.get("body_color"):
                rgba = dynamic["body_color"]
                stroke_colour = QColor(rgba[0], rgba[1], rgba[2], rgba[3])

        # Fill (closed shape)
        if p.fill:
            closed = QPainterPath(geom)
            closed.closeSubpath()
            fill_c = stroke_colour if p.ip and dynamic and dynamic.get("body_color") else QColor(p.fill_colour)
            painter.setBrush(QBrush(fill_c))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(closed)

        # Stroke
        pen = QPen(stroke_colour, p.width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(geom)

        painter.restore()


# ── Registry / public API ─────────────────────────────────────────────────────

class _PointRegistry:
    """
    Singleton exposed as Draw.point.

    Parameters
    ----------
    tag / display   : window tag to render on
    graph           : [width, height] of the coordinate space (not canvas pixels)
    points          : list of path-definition dicts (Style 1 or Style 2)

    Each dict in *points* is one path (or path group).
    See module docstring for full syntax.
    """

    def __init__(self) -> None:
        # (window_tag, ip) -> {"canvas": _PointCanvas, "path": PathDef,
        #                       "graph_w": float, "graph_h": float}
        # Draw.point() previously had no way to look an item back up after
        # creation at all -- no get_by_ip, no move, nothing -- unlike
        # shapes/text/panels, which is why Draw.room() could never place or
        # move a point path the way it already could shapes and text. This
        # registry is what closes that gap (see get_by_ip / move_by_ip
        # below, and the "point" kind wired into _room.py).
        self._registry: Dict[Tuple[str, str], dict] = {}
        self._canvases: Dict[str, List[_PointCanvas]] = {}

    def __call__(
        self,
        *,
        tag: Optional[str] = None,
        display: Optional[str] = None,
        graph: Optional[List[float]] = None,
        points: Optional[List[dict]] = None,
    ) -> None:
        get_app()

        window_tag = display if display is not None else tag
        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise ValueError(
                    "Draw.point: multiple windows exist; 'tag' or 'display' is required."
                )
            else:
                raise ValueError("Draw.point: no windows exist to draw on.")

        win: QMainWindow = _window_registry.get(window_tag)

        # Parse graph (coordinate space)
        if graph is None:
            graph = [float(win.width()), float(win.height())]
        if not isinstance(graph, (list, tuple)) or len(graph) < 2:
            raise TypeError("Draw.point: 'graph' must be a [width, height] list.")
        graph_w, graph_h = float(graph[0]), float(graph[1])
        if graph_w <= 0 or graph_h <= 0:
            raise ValueError("Draw.point: 'graph' dimensions must be positive.")

        # Ensure the shared canvas exists (so shapes/text are underneath).
        # Point overlays install an event filter here to participate in the
        # same senses pipeline as shapes and text.
        shared_canvas = _get_or_create_canvas(window_tag, win)

        # Parse paths
        if not isinstance(points, list):
            raise TypeError("Draw.point: 'points' must be a list of dicts.")

        parsed: List[PathDef] = []
        for index, entry in enumerate(points):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Draw.point: every item in 'points' must be a dict (got {type(entry).__name__} at index {index})."
                )
            try:
                path_def = _parse_path_entry(entry)
            except Exception as exc:
                raise ValueError(
                    f"Draw.point: error parsing points[{index}]: {exc}"
                ) from exc
            if path_def.align and path_def.coords:
                xs = [c[0] for c in path_def.coords]
                ys = [c[1] for c in path_def.coords]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                sw, sh = max_x - min_x, max_y - min_y
                from Draw._align import calculate_alignment_pos
                target_x, target_y = calculate_alignment_pos(
                    path_def.align, sw, sh, graph_w, graph_h, window_tag=window_tag
                )
                dx = target_x - min_x
                dy = target_y - min_y
                path_def.coords = [(cx + dx, cy + dy) for (cx, cy) in path_def.coords]
            parsed.append(path_def)

            # Register hitboxes if requested
            if path_def.hit_box and path_def.ip:
                try:
                    from Draw._shapes import hitbox as _hitbox_registry
                    for hb in path_def.hit_box:
                        if isinstance(hb, dict):
                            _hitbox_registry(
                                ip=path_def.ip,
                                type=["Fullgeometry"],
                                box=hb,
                            )
                except Exception as exc:
                    print(f"Draw.point: hitbox registration error for ip={path_def.ip!r}: {exc}")

        if not parsed:
            return

        named = [p.ip for p in parsed if p.ip]
        duplicates = {ip for ip in named if named.count(ip) > 1}
        if duplicates:
            raise ValueError(
                "Draw.point: each 'ip' must be unique within one call; "
                f"duplicates: {', '.join(sorted(duplicates))}."
            )

        # ── Reuse an existing canvas in place when possible ─────────────
        # The old behaviour built a brand-new _PointCanvas QWidget on every
        # single call -- including re-renders of the exact same call on a
        # live update / drag tick / animate tick -- then scheduled the old
        # one for deleteLater(). deleteLater() is deferred to the next
        # event-loop pass, so for one frame both the old and new overlay
        # widget could be alive and painting at once, which is what caused
        # visible flicker/desync on anything that re-submits the same ip(s)
        # every frame. If every named path in this call was already
        # registered to the SAME existing canvas, we now just replace that
        # canvas's path list and geometry in place and repaint it --
        # no widget is destroyed or created, so there is nothing to race.
        reuse_canvas: Optional["_PointCanvas"] = None
        if named and len(named) == len(parsed):
            owning_canvases = set()
            all_known = True
            for ip in named:
                entry = self._registry.get((window_tag, ip))
                if entry is None:
                    all_known = False
                    break
                owning_canvases.add(entry["canvas"])
            if all_known and len(owning_canvases) == 1:
                candidate = owning_canvases.pop()
                # Only reuse if that canvas exclusively belongs to this
                # exact ip set (no foreign paths would be silently dropped).
                candidate_ips = {p.ip for p in candidate._paths if p.ip}
                if candidate_ips == set(named):
                    reuse_canvas = candidate

        if reuse_canvas is not None:
            for path_def in parsed:
                path_def._cache_key = None  # force geometry rebuild
            reuse_canvas._paths = parsed
            reuse_canvas._graph_w = graph_w
            reuse_canvas._graph_h = graph_h
            reuse_canvas._has_animation = any(p.anim_channels for p in parsed)
            reuse_canvas.update()
            for path_def in parsed:
                if path_def.ip:
                    self._registry[(window_tag, path_def.ip)] = {
                        "canvas": reuse_canvas,
                        "path": path_def,
                        "graph_w": graph_w,
                        "graph_h": graph_h,
                    }
            return

        # Re-submitting an ip replaces its previous path instead of leaving a
        # stale overlay/timer visible behind the new one.
        for path_def in parsed:
            if path_def.ip:
                self.remove_by_ip(window_tag, path_def.ip)

        # Dispose stale canvases that have no remaining named paths to
        # prevent overlay and QTimer accumulation on repeated calls.
        if window_tag in self._canvases:
            stale = []
            for c in self._canvases[window_tag]:
                has_named = any(p.ip for p in c._paths)
                if not has_named:
                    c._timer.stop()
                    parent = c.parent()
                    if parent is not None and hasattr(parent, "_draw_canvas"):
                        parent._draw_canvas.removeEventFilter(c)
                    c.hide()
                    c.deleteLater()
                    stale.append(c)
            for c in stale:
                self._canvases[window_tag].remove(c)

        # Create the _PointCanvas overlay
        canvas = _PointCanvas(win, parsed, graph_w, graph_h)
        shared_canvas.installEventFilter(canvas)
        canvas.raise_()
        canvas.show()
        self._canvases.setdefault(window_tag, []).append(canvas)
        # Register every named path so it can be looked up/moved later
        # (Draw.room() needs this; get_by_ip()/move_by_ip() below use it).
        for path_def in parsed:
            if path_def.ip:
                self._registry[(window_tag, path_def.ip)] = {
                    "canvas": canvas,
                    "path": path_def,
                    "graph_w": graph_w,
                    "graph_h": graph_h,
                }

    # ── lookup / room-placement API ─────────────────────────────────────
    # Mirrors the shape ("get_by_ip") and text registries so Draw.point()
    # items are reachable the same way. get_pixel_bounds()/move_by_ip() are
    # what let Draw._room.py treat a point path as a placeable object.

    def get_by_ip(self, window_tag: str, ip: str) -> Optional["PathDef"]:
        """Return the live PathDef for (window_tag, ip), or None."""
        entry = self._registry.get((window_tag, ip))
        return entry["path"] if entry else None

    def get_pixel_bounds(self, window_tag: str, ip: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Return (x, y, w, h) of this path's bounding box in canvas pixel
        space, or None if the ip isn't registered or has no geometry yet.
        Reuses _PointCanvas._path_bounds so this is exactly the same
        geometry the renderer itself uses -- no separate copy of the
        graph-to-pixel scaling math.
        """
        entry = self._registry.get((window_tag, ip))
        if entry is None:
            return None
        canvas: "_PointCanvas" = entry["canvas"]
        rect = canvas._path_bounds(entry["path"], canvas.width(), canvas.height())
        if rect is None:
            return None
        return (rect.x, rect.y, rect.w, rect.h)

    def remove_by_ip(self, window_tag: str, ip: str) -> bool:
        """Remove one registered path and dispose an empty overlay safely."""
        entry = self._registry.pop((window_tag, ip), None)
        if entry is None:
            return False
        canvas: "_PointCanvas" = entry["canvas"]
        canvas._paths = [p for p in canvas._paths if p.ip != ip]
        if not canvas._paths:
            canvas._timer.stop()
            parent = canvas.parent()
            if parent is not None and hasattr(parent, "_draw_canvas"):
                parent._draw_canvas.removeEventFilter(canvas)
            canvas.hide()
            canvas.deleteLater()
            canvases = self._canvases.get(window_tag, [])
            if canvas in canvases:
                canvases.remove(canvas)
        return True

    def clear(self, window_tag: str) -> int:
        """Remove every point path registered for a window."""
        ips = [ip for tag, ip in self._registry if tag == window_tag]
        for ip in ips:
            self.remove_by_ip(window_tag, ip)
        return len(ips)

    def move_by_ip(self, window_tag: str, ip: str, x: float, y: float) -> bool:
        """
        Move a registered path so its bounding-box top-left lands at pixel
        (x, y), by translating every coordinate in the path (coords are
        stored in graph space, so the pixel delta is converted through the
        same graph_w/graph_h -> canvas_w/canvas_h ratio the renderer uses).
        Returns False if the ip isn't registered. Resizing isn't supported
        yet -- Draw.room()'s size_spec is ignored for "point" kind for now.
        """
        entry = self._registry.get((window_tag, ip))
        if entry is None:
            return False
        canvas: "_PointCanvas" = entry["canvas"]
        path_def: "PathDef" = entry["path"]
        cw, ch = canvas.width(), canvas.height()
        current = canvas._path_bounds(path_def, cw, ch)
        if current is None or cw <= 0 or ch <= 0:
            return False
        scale_x = entry["graph_w"] / cw
        scale_y = entry["graph_h"] / ch
        dx = (x - current.x) * scale_x
        dy = (y - current.y) * scale_y
        path_def.coords = [(cx + dx, cy + dy) for (cx, cy) in path_def.coords]
        path_def._cache_key = None  # invalidate geometry cache
        canvas.update()
        return True

    def resize_by_ip(self, window_tag: str, ip: str, width: float, height: float) -> bool:
        """Scale a path's graph-space coordinates to a pixel bounding size."""
        entry = self._registry.get((window_tag, ip))
        if entry is None or width < 0 or height < 0:
            return False
        canvas: "_PointCanvas" = entry["canvas"]
        path_def: "PathDef" = entry["path"]
        cw, ch = canvas.width(), canvas.height()
        current = canvas._path_bounds(path_def, cw, ch, include_flow=False)
        if current is None or current.w <= 0 or current.h <= 0 or cw <= 0 or ch <= 0:
            return False
        sx, sy = float(width) / current.w, float(height) / current.h
        graph_x = entry["graph_w"] / cw
        graph_y = entry["graph_h"] / ch
        origin_x, origin_y = current.x * graph_x, current.y * graph_y
        path_def.coords = [
            (origin_x + (px - origin_x) * sx, origin_y + (py - origin_y) * sy)
            for px, py in path_def.coords
        ]
        path_def._cache_key = None  # invalidate geometry cache
        canvas.update()
        return True

    def get_intrinsic_size(
        self, window_tag: str, ip: str,
    ) -> Optional[Tuple[float, float]]:
        """Return the intrinsic (width, height) of a point path in pixels.

        Used by ``_room_size.resolve_size_spec`` so that size keywords
        like ``fit``, ``aspect``, ``stretch_x``, and ``stretch_y`` can
        scale point paths the same way they scale shapes and text.
        """
        bounds = self.get_pixel_bounds(window_tag, ip)
        if bounds is None:
            return None
        return (bounds[2], bounds[3])


point = _PointRegistry()
