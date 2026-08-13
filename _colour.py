"""
Draw._colour  — Universal Colour System
=========================================

Draw.color(
    connection = {"color_ip": "shape_ip", "get_ip": ""},
    color = [
        {"for": "body",   "h": "time * 60", "s": 100, "v": 100},
        {"for": "border", "color": "#e94560", "width": 2},
        {"for": "shadow", "color": "black", "blur": 12, "offset": [0, 4]},
        {"for": "glow",   "color": "cyan",  "blur": 20, "opacity": 60},
    ]
)

Every numeric value can be a math expression string.
If it's a string with math → it gets evaluated every frame.
Built-in variables: time, x, y, w, h, mouse_x, mouse_y
Built-in functions: sin, cos, abs, min, max, lerp, step
"""

from __future__ import annotations

import json
import logging
import math
import time as _time_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from PySide6.QtGui import QColor


# ── safe expression evaluator ────────────────────────────────────────────────

from Draw._calculator import eval_expression, is_expression
from Draw._live import live as _live_registry, LiveRef, LiveTextBinding, resolve_live_text, is_live_text_binding

_logger = logging.getLogger(__name__)


# ── canonical static colour parser ───────────────────────────────────────────
# This is the single source of truth for turning a user-supplied colour value
# (named string, hex string, RGB tuple/list, or QColor) into a QColor. It used
# to live duplicated-by-import-only in Draw._window (every other module —
# _shapes, _text, _point, _layout, _room — imported it from there), which put
# colour-parsing ownership in the window module instead of the colour module.
# It now lives here; Draw._window keeps a two-line backward-compat re-export
# so existing `from Draw._window import _parse_color` call sites keep working
# unchanged. It shares the same named-colour table as _parse_color_string
# further down this file (the "named color table" section) instead of keeping
# a second, separately-maintained copy.


def _parse_color(color) -> QColor:
    """
    Accept named strings, hex strings, RGB tuples/lists, or QColor values.
    Bad input falls back to white.
    """
    if isinstance(color, QColor):
        return color

    if isinstance(color, str):
        lower = color.strip().lower()
        if lower in _NAMED_COLORS:
            return QColor(*_NAMED_COLORS[lower])
        parsed = QColor(color)
        return parsed if parsed.isValid() else QColor("#FFFFFF")

    if isinstance(color, (tuple, list)) and len(color) in (3, 4):
        vals = [max(0, min(255, int(v))) for v in color]
        return QColor(*vals)

    return QColor("#FFFFFF")


# ── color entry definition ───────────────────────────────────────────────────

@dataclass
class ColorEntry:
    """One parsed color block from the color=[] list."""
    target: str              # "body" | "border" | "shadow" | "glow" | "text"

    # Static color (named/hex)
    color_str: Optional[str] = None

    # HSV mode expressions (strings or numbers)
    h: Any = None
    s: Any = None
    v: Any = None

    # RGB mode expressions (strings or numbers)
    r: Any = None
    g: Any = None
    b: Any = None

    # Alpha/opacity (works with all modes)
    opacity: Any = 100

    # Gradient
    gradient: Optional[str] = None    # "linear" | "radial" | "conical"
    angle: Any = 0                    # degrees (or expression)
    center: Optional[list] = None     # [cx%, cy%] for radial/conical
    radius: Any = 50                  # % spread for radial (or expression)
    stops: Optional[list] = None      # [[pos, color], ...]

    # Shadow / Glow extras
    blur: Any = 0
    spread: Any = 0
    offset: Optional[list] = None     # [x, y]

    # Border extras
    width: Any = 0
    style: str = "solid"

    # Tracking: does this entry have any dynamic expressions?
    _is_dynamic: bool = False


@dataclass
class ColorBinding:
    """A registered Draw.color() call, bound to a shape IP."""
    color_ip: str
    get_ip: Optional[str]
    entries: List[ColorEntry]
    created_at: float = field(default_factory=_time_mod.perf_counter)

    @property
    def is_dynamic(self) -> bool:
        return any(e._is_dynamic for e in self.entries)


# ── expression resolver ──────────────────────────────────────────────────────

# Dedup cache so a broken expression warns once, not once per frame (this
# runs every paint cycle — without dedup a bad expression would spam the
# console at ~60fps).
_expr_error_cache: Dict[str, str] = {}


def _resolve_value(raw: Any, variables: Dict[str, float]) -> float:
    """Resolve a value: if it's a math expression string, evaluate it.
    If it's a LiveRef, LiveTextBinding, or callable, unwrap it first.
    If it's a number, return it directly."""
    if raw is None:
        return 0.0
    if isinstance(raw, LiveRef):
        raw = _live_registry.get(raw.key, 0.0)
    elif is_live_text_binding(raw):
        raw = resolve_live_text(raw)
    elif callable(raw):
        raw = raw()
    if raw is None:
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s == "":
            return 0.0
        # Try plain number first (fast path)
        try:
            return float(s)
        except ValueError:
            pass
        # Evaluate as math expression
        try:
            return eval_expression(s, variables)
        except Exception as exc:
            # Broad catch is intentional: eval_expression's own contract
            # normalizes parse/evaluation problems (bad syntax, unknown
            # variable, division by zero) into ValueError, but this is a
            # per-frame paint-cycle call site — any exception type here
            # must degrade to a fallback value rather than crash painting.
            msg = str(exc)
            if _expr_error_cache.get(s) != msg:
                _logger.warning("Draw.color: expression error in %r: %s; using 0.0.", s, msg)
                _expr_error_cache[s] = msg
            return 0.0
    return 0.0


def _resolve_color_from_entry(
    entry: ColorEntry,
    variables: Dict[str, float],
) -> Tuple[int, int, int, int]:
    """
    Resolve a ColorEntry to an (r, g, b, a) tuple.
    Priority: HSV mode > RGB mode > color string > fallback white.
    """
    opacity = max(0, min(100, int(_resolve_value(entry.opacity, variables))))
    alpha = int(opacity / 100.0 * 255)

    # HSV mode
    if entry.h is not None:
        h = _resolve_value(entry.h, variables) % 360
        s = max(0, min(100, _resolve_value(entry.s if entry.s is not None else 100, variables)))
        v = max(0, min(100, _resolve_value(entry.v if entry.v is not None else 100, variables)))
        # Convert HSV to RGB
        r, g, b = _hsv_to_rgb(h, s / 100.0, v / 100.0)
        return (r, g, b, alpha)

    # RGB mode
    if entry.r is not None or entry.g is not None or entry.b is not None:
        r = max(0, min(255, int(_resolve_value(entry.r if entry.r is not None else 0, variables))))
        g = max(0, min(255, int(_resolve_value(entry.g if entry.g is not None else 0, variables))))
        b = max(0, min(255, int(_resolve_value(entry.b if entry.b is not None else 0, variables))))
        return (r, g, b, alpha)

    # Named/hex color string (static or live)
    if entry.color_str is not None:
        cs = entry.color_str
        if isinstance(cs, LiveRef):
            cs = _live_registry.get(cs.key, "#FFFFFF")
        elif is_live_text_binding(cs):
            cs = resolve_live_text(cs)
        elif callable(cs):
            cs = cs()
        rgb = _parse_color_string(str(cs))
        return (rgb[0], rgb[1], rgb[2], alpha)

    # Fallback
    return (255, 255, 255, alpha)


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """Convert HSV (h: 0-360, s: 0-1, v: 0-1) to RGB (0-255 each)."""
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r1, g1, b1 = c, x, 0
    elif h < 120:
        r1, g1, b1 = x, c, 0
    elif h < 180:
        r1, g1, b1 = 0, c, x
    elif h < 240:
        r1, g1, b1 = 0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    return (
        max(0, min(255, int((r1 + m) * 255))),
        max(0, min(255, int((g1 + m) * 255))),
        max(0, min(255, int((b1 + m) * 255))),
    )


# ── named color table ─────────────────────────────────────────────────────────
# Shared by _parse_color_string (dynamic-registry path) and _parse_color
# (static path, above) — one table instead of two independently-maintained
# copies.

_NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "white":  (255, 255, 255),
    "black":  (0, 0, 0),
    "red":    (255, 0, 0),
    "green":  (0, 200, 83),
    "blue":   (33, 150, 243),
    "yellow": (255, 235, 59),
    "orange": (255, 152, 0),
    "purple": (156, 39, 176),
    "pink":   (233, 30, 99),
    "cyan":   (0, 188, 212),
    "teal":   (0, 150, 136),
    "gray":   (158, 158, 158),
    "grey":   (158, 158, 158),
    "brown":  (121, 85, 72),
    "indigo": (63, 81, 181),
    "lime":   (205, 220, 57),
    "amber":  (255, 193, 7),
    "silver": (192, 192, 192),
    "gold":   (255, 215, 0),
}


def _parse_color_string(color: str) -> Tuple[int, int, int]:
    """Parse a color name or hex string to (r, g, b)."""
    c = color.strip().lower()
    if c in _NAMED_COLORS:
        return _NAMED_COLORS[c]
    # Hex
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = h[0]*2 + h[1]*2 + h[2]*2
        if len(h) == 6 or len(h) == 8:
            return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    qc = QColor(color)
    if qc.isValid():
        return (qc.red(), qc.green(), qc.blue())
    return (255, 255, 255)


# ── gradient builder ─────────────────────────────────────────────────────────

def _resolve_gradient_stops(
    stops: List[list],
    variables: Dict[str, float],
) -> List[Tuple[float, Tuple[int, int, int]]]:
    """Resolve gradient stops: [[pos_or_expr, color_str], ...] → [(pos_float, rgb), ...]"""
    resolved = []
    for stop in stops:
        if not isinstance(stop, (list, tuple)) or len(stop) < 2:
            continue
        pos_raw, color_raw = stop[0], stop[1]
        pos = _resolve_value(pos_raw, variables) / 100.0  # normalize 0-100 → 0-1
        pos = max(0.0, min(1.0, pos))
        rgb = _parse_color_string(str(color_raw))
        resolved.append((pos, rgb))
    resolved.sort(key=lambda x: x[0])
    return resolved


# ── parse helpers ─────────────────────────────────────────────────────────────

def _check_dynamic(entry: ColorEntry) -> bool:
    """Check if any field in the entry is a dynamic expression or references live state."""
    for val in [entry.color_str, entry.h, entry.s, entry.v, entry.r, entry.g, entry.b,
                entry.opacity, entry.angle, entry.radius, entry.blur,
                entry.spread, entry.width]:
        if isinstance(val, LiveRef) or is_live_text_binding(val) or callable(val):
            return True
        if isinstance(val, str) and is_expression(val):
            return True
    # Check gradient stops for dynamic positions or live references
    if entry.stops:
        for stop in entry.stops:
            if isinstance(stop, (list, tuple)) and len(stop) >= 2:
                p_val, c_val = stop[0], stop[1]
                if isinstance(p_val, (LiveRef, LiveTextBinding)) or callable(p_val) or (isinstance(p_val, str) and is_expression(p_val)):
                    return True
                if isinstance(c_val, (LiveRef, LiveTextBinding)) or callable(c_val):
                    return True
    return False


def _parse_color_entry(raw: Dict[str, Any]) -> ColorEntry:
    """Parse one dict from the color=[] list into a ColorEntry."""
    target_raw = raw.get("for", "body")
    target = str(target_raw).strip().lower()
    if target not in {"body", "border", "shadow", "glow", "text"}:
        target = "body"

    entry = ColorEntry(
        target=target,
        color_str=raw.get("color"),
        h=raw.get("h"),
        s=raw.get("s"),
        v=raw.get("v"),
        r=raw.get("r"),
        g=raw.get("g"),
        b=raw.get("b"),
        opacity=raw.get("opacity", 100),
        gradient=raw.get("gradient"),
        angle=raw.get("angle", 0),
        center=raw.get("center"),
        radius=raw.get("radius", 50),
        stops=raw.get("stops"),
        blur=raw.get("blur", 0),
        spread=raw.get("spread", 0),
        offset=raw.get("offset"),
        width=raw.get("width", 0),
        style=str(raw.get("style", "solid")).strip().lower(),
    )
    entry._is_dynamic = _check_dynamic(entry)
    return entry


# ── Color Registry ───────────────────────────────────────────────────────────

class _ColorRegistry:
    """
    Singleton registry for Draw.color() bindings.
    Call it as Draw.color(...) to register a color binding.
    The rendering pipeline queries it by shape IP to get resolved colors.
    """

    def __init__(self) -> None:
        self._bindings: Dict[str, ColorBinding] = {}
        # Cache mouse position for reactive colors
        self._mouse_x: float = 0.0
        self._mouse_y: float = 0.0

    def __call__(
        self,
        *,
        connection: Any = None,
        ip: Any = None,                      # canonical ip kwarg (target shape ip)
        color_ip: Any = None,                # legacy alias (overrides ip if both given)
        color_get_ip: Any = None,            # legacy alias for layout get_ip
        color: Optional[List[Dict[str, Any]]] = None,
    ) -> "ColorBinding":
        """
        Draw.color(
            connection = {"color_ip": "my_shape", "get_ip": ""},
            color = [
                {"for": "body", "color": "cyan"},
                {"for": "border", "h": "time * 45", "s": 80, "v": 100, "width": 2},
            ]
        )
        Preferred form (no connection dict needed):
            Draw.color(
                ip           = "my_shape",    # canonical kwarg
                color_get_ip = "my_layout",   # optional
                color = [...],
            )
        Legacy alias ``color_ip`` still works and takes priority if both given.
        """
        # ── resolve: color_ip takes priority over canonical ip ───────────────
        color_ip = color_ip if color_ip is not None else ip
        # ── resolve color_ip and get_ip ──────────────────────────────────────
        # Prefer top-level color_ip/color_get_ip; fall back to connection dict.
        if color_ip is not None:
            resolved_color_ip = str(color_ip)
            resolved_get_ip = str(color_get_ip) if color_get_ip is not None else None
        else:
            if connection is None:
                raise ValueError("Draw.color: provide 'ip' (or 'color_ip') or a 'connection' dict.")
            if isinstance(connection, dict):
                resolved_color_ip = connection.get("color_ip", connection.get("ip", ""))
                resolved_get_ip = connection.get("get_ip")
            elif isinstance(connection, (list, tuple)):
                resolved_color_ip = str(connection[0]) if len(connection) > 0 else ""
                resolved_get_ip = str(connection[1]) if len(connection) > 1 else None
            elif isinstance(connection, str):
                resolved_color_ip = connection
                resolved_get_ip = None
            else:
                raise TypeError("Draw.color: 'connection' must be a dict, list, or string.")

        if not resolved_color_ip:
            raise ValueError("Draw.color: 'ip' (or 'color_ip') is required.")

        # Parse color entries
        if isinstance(color, dict):
            color_items = [color]
        elif isinstance(color, list):
            color_items = color
        elif color is None:
            color_items = []
        else:
            raise TypeError("Draw.color: 'color' must be a dict or a list of dicts.")

        entries: List[ColorEntry] = []
        for raw in color_items:
            if not isinstance(raw, dict):
                raise TypeError("Draw.color: every item in 'color' must be a dict.")
            entries.append(_parse_color_entry(raw))

        # Register
        binding = ColorBinding(
            color_ip=resolved_color_ip,
            get_ip=resolved_get_ip,
            entries=entries,
        )
        self._bindings[binding.color_ip] = binding
        self._request_repaint(binding.color_ip)
        return binding

    def _request_repaint(self, color_ip: str) -> None:
        """
        Draw.color() only updates internal binding state -- nothing else
        was telling the canvas to actually redraw with the new color, so a
        color/gradient change could sit applied-but-invisible until some
        unrelated repaint happened to occur (e.g. a hover change elsewhere).
        Find any window canvas currently showing this shape and repaint it.
        """
        try:
            from Draw._window import window as _window_registry
        except ImportError:
            return
        for tag in _window_registry.list_tags():
            try:
                canvas = _window_registry.get_canvas(tag)
            except Exception:
                continue
            for s in getattr(canvas, "shape_items", []):
                if getattr(s, "ip", None) == color_ip:
                    canvas.update()
                    break

    # ── query API (used by renderer) ─────────────────────────────────────

    def get_binding(self, color_ip: str) -> Optional[ColorBinding]:
        """Get the color binding for a shape IP."""
        return self._bindings.get(color_ip)

    def has_binding(self, color_ip: str) -> bool:
        return color_ip in self._bindings

    def has_any_dynamic(self) -> bool:
        """Check if any registered binding has dynamic expressions."""
        return any(b.is_dynamic for b in self._bindings.values())

    def resolve_for_shape(
        self,
        color_ip: str,
        shape_x: float = 0.0,
        shape_y: float = 0.0,
        shape_w: float = 0.0,
        shape_h: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve all color entries for a shape IP at the current moment.
        Returns a dict with keys like:
            body_color: (r, g, b, a)
            body_gradient: {...}
            border_color: (r, g, b, a)
            border_width: int
            border_style: str
            shadow_color: (r, g, b, a)
            shadow_blur: float
            shadow_offset: (x, y)
            glow_color: (r, g, b, a)
            glow_blur: float
            etc.
        """
        binding = self._bindings.get(color_ip)
        if binding is None:
            return None

        now = _time_mod.perf_counter()
        elapsed = now - binding.created_at

        variables: Dict[str, float] = {
            "time": elapsed,
            "x": shape_x,
            "y": shape_y,
            "w": shape_w,
            "h": shape_h,
            "mouse_x": self._mouse_x,
            "mouse_y": self._mouse_y,
        }

        # Expose numeric values stored in Draw.live to math expressions
        for lk, lv in getattr(_live_registry, "_values", {}).items():
            if isinstance(lv, (int, float)):
                variables[lk] = float(lv)
            elif isinstance(lv, str):
                try:
                    variables[lk] = float(lv)
                except ValueError:
                    pass

        result: Dict[str, Any] = {}

        for entry in binding.entries:
            target = entry.target  # body, border, shadow, glow, text

            # Resolve color
            rgba = _resolve_color_from_entry(entry, variables)
            result[f"{target}_color"] = rgba

            # Gradient
            if entry.gradient and entry.stops:
                grad_type = entry.gradient
                angle = _resolve_value(entry.angle, variables)
                center = entry.center or [50, 50]
                radius = _resolve_value(entry.radius, variables)
                stops = _resolve_gradient_stops(entry.stops, variables)
                result[f"{target}_gradient"] = {
                    "type": grad_type,
                    "angle": angle,
                    "center": center,
                    "radius": radius,
                    "stops": stops,
                }

            # Shadow/Glow extras
            if target in ("shadow", "glow"):
                result[f"{target}_blur"] = max(0, _resolve_value(entry.blur, variables))
                result[f"{target}_spread"] = max(0, _resolve_value(entry.spread, variables))
                if entry.offset and isinstance(entry.offset, (list, tuple)) and len(entry.offset) >= 2:
                    result[f"{target}_offset"] = (
                        _resolve_value(entry.offset[0], variables),
                        _resolve_value(entry.offset[1], variables),
                    )
                else:
                    result[f"{target}_offset"] = (0, 0)

            # Border extras
            if target == "border":
                result["border_width"] = max(0, int(_resolve_value(entry.width, variables)))
                result["border_style"] = entry.style

        return result

    def update_mouse(self, mx: float, my: float) -> None:
        """Update cached mouse position for reactive expressions."""
        self._mouse_x = mx
        self._mouse_y = my

    def remove(self, color_ip: str) -> bool:
        """Remove a color binding."""
        return self._bindings.pop(color_ip, None) is not None

    def clear(self) -> None:
        """Remove all color bindings."""
        self._bindings.clear()

    def list_bindings(self) -> List[str]:
        """List all registered color IPs."""
        return list(self._bindings.keys())


# ── singleton ────────────────────────────────────────────────────────────────

color = _ColorRegistry()


# ── legacy colour() helper (kept for backward compat) ────────────────────────

def _normalize_gradient_list(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise TypeError("Draw.colour: 'gradient_colors' must be a list.")

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            color_val = item.get("color")
            stop = item.get("stop", item.get("position"))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            color_val, stop = item
        else:
            raise TypeError(
                "Draw.colour: gradient color items must be {'color', 'stop'} or (color, stop)."
            )
        if color_val is None or stop is None:
            raise ValueError("Draw.colour: gradient color items require both color and stop.")
        normalized.append({"color": color_val, "stop": stop})
    return normalized


def colour(**kwargs: Any) -> dict[str, Any]:
    """
    Build a customise dict for Draw.shapes / Draw.text with friendly aliases.
    (Legacy helper — kept for backward compatibility.)
    """
    style: dict[str, Any] = {}

    alias_map = {
        "fill": "color",
        "color": "color",
        "border": "border_color",
        "border_width": "border_width",
        "border_style": "border_style",
        "opacity": "opacity",
        "glow": "glow",
        "glow_color": "glow_color",
        "glow_radius": "glow_radius",
        "shadow": "shadow",
        "shadow_color": "shadow_color",
        "shadow_offset": "shadow_offset",
        "shadow_blur": "shadow_blur",
        "gradient": "gradient",
        "gradient_start": "gradient_start",
        "gradient_end": "gradient_end",
        "gradient_angle": "gradient_angle",
        "animation": "animation",
        "from_color": "from_color",
        "to_color": "to_color",
        "duration": "duration",
        "loop": "loop",
        "ease": "ease",
        "pulse_intensity_min": "pulse_intensity_min",
        "pulse_intensity_max": "pulse_intensity_max",
        "pulse_speed": "pulse_speed",
        "rainbow_speed": "rainbow_speed",
        "rainbow_saturation": "rainbow_saturation",
        "strobe_on_color": "strobe_on_color",
        "strobe_off_color": "strobe_off_color",
        "strobe_frequency": "strobe_frequency",
        "hue_rotation_speed": "hue_rotation_speed",
        "wave_direction": "wave_direction",
        "wave_speed": "wave_speed",
        "filter": "filter",
        "filter_intensity": "filter_intensity",
    }

    for source, target in alias_map.items():
        if source in kwargs:
            style[target] = kwargs[source]

    gradient_colors = _normalize_gradient_list(kwargs.get("gradient_colors"))
    if gradient_colors is not None:
        style["gradient_colors"] = gradient_colors

    wave_colors = _normalize_gradient_list(kwargs.get("wave_colors"))
    if wave_colors is not None:
        style["wave_colors"] = wave_colors

    if kwargs.get("neon"):
        style["glow"] = True
        style.setdefault("glow_color", kwargs.get("neon_color", style.get("color", "cyan")))
        style.setdefault("glow_radius", int(kwargs.get("neon_radius", 15)))
        style.setdefault("shadow", True)
        style.setdefault("shadow_blur", int(kwargs.get("neon_blur", 3)))

    if kwargs.get("water_dip"):
        style.setdefault("animation", "wave_color")
        style.setdefault("wave_direction", "horizontal")
        style.setdefault("wave_speed", kwargs.get("water_dip_speed", 1.5))
        dip_color = kwargs.get("water_dip_color", style.get("color", "cyan"))
        base_color = style.get("color", "cyan")
        style.setdefault(
            "wave_colors",
            [
                {"color": base_color, "stop": 0},
                {"color": dip_color, "stop": 50},
                {"color": base_color, "stop": 100},
            ],
        )

    if kwargs.get("wave"):
        style.setdefault("animation", "wave_color")
        style.setdefault("wave_direction", kwargs.get("wave_direction", "horizontal"))
        style.setdefault("wave_speed", kwargs.get("wave_speed", 2.0))
        if "wave_colors" not in style:
            base_color = style.get("color", "blue")
            style["wave_colors"] = [
                {"color": base_color, "stop": 0},
                {"color": kwargs.get("wave_color", "cyan"), "stop": 50},
                {"color": base_color, "stop": 100},
            ]

    if kwargs.get("metallic"):
        style.setdefault("gradient", "linear")
        style.setdefault("gradient_angle", kwargs.get("metallic_direction", 135))
        base = kwargs.get("metallic_color", "silver")
        shine = kwargs.get("metallic_shine", 0.8)
        style.setdefault(
            "gradient_colors",
            [
                {"color": base, "stop": 0},
                {"color": "white", "stop": max(1, min(99, int(50 * float(shine))))},
                {"color": base, "stop": 100},
            ],
        )

    if kwargs.get("burn"):
        style.setdefault("animation", "strobe")
        style.setdefault("strobe_on_color", kwargs.get("burn_color", "red"))
        style.setdefault("strobe_off_color", kwargs.get("burn_outer_color", "orange"))
        style.setdefault("strobe_frequency", max(0.5, float(kwargs.get("burn_intensity", 1.0)) * 4.0))

    if kwargs.get("frost"):
        style.setdefault("gradient", "linear")
        style.setdefault("gradient_angle", 45)
        base = style.get("color", "cyan")
        frost = kwargs.get("frost_color", "lightblue")
        style.setdefault(
            "gradient_colors",
            [
                {"color": base, "stop": 0},
                {"color": frost, "stop": 50},
                {"color": "white", "stop": 100},
            ],
        )

    return style


# ── token save/load (unchanged) ──────────────────────────────────────────────

def save_tokens(path: str, tokens: dict[str, Any]) -> str:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return str(file_path)


def load_tokens(path: str) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


__all__ = ["color", "colour", "save_tokens", "load_tokens"]
