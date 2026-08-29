"""
Draw._text
Part 3 — Text rendering inside windows.

How it works
------------
Text is drawn on the same shared canvas that shapes use (_DrawCanvas).
Each Draw.text() call appends a TextDef to that canvas.
The canvas redraws everything (shapes + text) on every paintEvent.

Usage
-----
    import Draw

    Draw.window(tag="main", title="Text Demo", width=800, height=600)

    Draw.text(
        tag="main",
        text="Hello, World!",
        customise={
            "align": "center",
            "color": "white",
            "font_size": 48,
            "font_family": "Arial",
            "bold": True,
            "italic": False,
            "underline": False,
            "strikethrough": False,
            "opacity": 100,
            "rotation": 0,
            "glow": True,
            "glow_color": "cyan",
            "glow_radius": 12,
            "shadow": True,
            "shadow_color": "black",
            "shadow_offset": (3, 3),
            "letter_spacing": 2,
            "line_height": 1.4,
            "align_text": "center",   # text alignment inside bounding box
            "background_color": None,
            "background_padding": 8,
            "border_width": 0,
            "border_color": "black",
            "border_radius": 6,
        }
    )

    Draw.window.run("main")
"""

from __future__ import annotations
import math

import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
# pyrefly: ignore [missing-import]      
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
# pyrefly: ignore [missing-import]
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontDatabase, QFontMetricsF, QPainter,
    QPainterPath, QPen, QMouseEvent, QKeyEvent, QWheelEvent,
)
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QMainWindow, QWidget

from Draw._app import get_app
from Draw._live import (
    InputTextMarker,
    LiveTextBinding,
    resolve_live_text,
    live as _live_registry,
    input as _input_registry,
)
from Draw._shapes import _DrawCanvas
from Draw._window import window as _window_registry, _parse_color, _ALIGN_VALUES

_INPUT_TYPE_ALIASES = {
    "typr_input": "type_input",
}

_INPUT_TYPE_VALUES = {
    "all",
    "str",
    "string",
    "int",
    "float",
    "color",
    "border_color",
}

_INPUT_SUBMIT_KEY_ALIASES = {
    "enter": "return",
    "return": "return",
    "spacebar": "space",
}

_COLOR_CANDIDATE_RE = re.compile(r"^[#a-zA-Z0-9(),._\-\s]*$")
_INT_CANDIDATE_RE = re.compile(r"^-?\d*$")
_FLOAT_CANDIDATE_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d*)?$")
_LIVE_TEXT_TICK_INTERVAL_MS = 16

_INPUT_TRANSFORM_ALIASES = {
    "none": "none",
    "raw": "none",
    "upper": "upper",
    "uppercase": "upper",
    "lower": "lower",
    "lowercase": "lower",
    "title": "title",
    "capitalize": "title",
}

# ── Font cache (Phase 10) ──────────────────────────────────────────────
_FONT_CACHE_MAX = 64
_font_cache: dict[tuple, tuple] = {}  # key -> (QFont, QFontMetricsF)
_font_cache_order: list[tuple] = []   # LRU eviction order

def _get_cached_font(family, size, bold, italic, underline, strikethrough, letter_spacing, word_spacing):
    key = (family, size, bold, italic, underline, strikethrough, letter_spacing, word_spacing)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    font = QFont(family)
    font.setPixelSize(max(1, size))
    font.setBold(bold)
    font.setItalic(italic)
    font.setUnderline(underline)
    font.setStrikeOut(strikethrough)
    if letter_spacing != 0:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
    if word_spacing != 0:
        font.setWordSpacing(word_spacing)
    fm = QFontMetricsF(font)
    if len(_font_cache) >= _FONT_CACHE_MAX:
        oldest = _font_cache_order.pop(0)
        _font_cache.pop(oldest, None)
    _font_cache[key] = (font, fm)
    _font_cache_order.append(key)
    return (font, fm)


# ── shared canvas import (lazy, avoids circular import) ──────────────────────
# _DrawCanvas lives in _shapes and is imported here as a compatibility alias.
# The canvas is stored on the QMainWindow as an attribute "_draw_canvas".


def _get_or_create_canvas(tag: str, win: QMainWindow) -> "_DrawCanvas":
    """Return the shared _DrawCanvas for this window, creating it if needed."""
    if not hasattr(win, "_draw_canvas"):
        canvas = _DrawCanvas(win)
        if win.centralWidget() is None:
            win.setCentralWidget(canvas)
        else:
            # Preserve existing centralWidget (e.g. QTabWidget) and position canvas below tab bar
            canvas.setParent(win)
            tab_bar_h = 42
            canvas.setGeometry(0, tab_bar_h, win.width(), max(1, win.height() - tab_bar_h))
            canvas.show()
            win.centralWidget().raise_()

            # Install resize handler to update canvas size on window resize
            orig_resize = win.resizeEvent
            def _on_win_resize(event):
                if orig_resize:
                    orig_resize(event)
                if hasattr(win, "_draw_canvas"):
                    c = win._draw_canvas
                    c.setGeometry(0, tab_bar_h, win.width(), max(1, win.height() - tab_bar_h))
            win.resizeEvent = _on_win_resize

        canvas._window_tag = tag           # type: ignore[attr-defined]
        win._draw_canvas = canvas          # type: ignore[attr-defined]
    return win._draw_canvas                # type: ignore[attr-defined]


# ── text definition ───────────────────────────────────────────────────────────

@dataclass
class TextDef:
    text: str
    ip: Optional[str]              # identifier for this text item

    # position
    x: Optional[int]
    y: Optional[int]
    align: Optional[str]           # canvas-level position (9 values)

    # font
    font_family: str
    font_size: int
    bold: bool
    italic: bool
    underline: bool
    strikethrough: bool
    letter_spacing: float          # extra px between characters
    word_spacing: float            # extra px between words
    line_height: float             # multiplier (1.0 = normal)

    # colours
    color: QColor
    opacity: int                   # 0-100

    # inner text alignment
    align_text: str                # "left" | "center" | "right"

    # background box
    background_color: Optional[QColor]
    background_padding: int
    border_width: int
    border_color: QColor
    border_radius: float

    # effects
    glow: bool
    glow_color: QColor
    glow_radius: int
    shadow: bool
    shadow_color: QColor
    shadow_offset: Tuple[int, int]

    # transform
    rotation: float
    elide: Optional[str] = None    # "right" | "middle" | "left" | None

    # ── animation ─────────────────────────────────────────────────────
    # animation        : name string or None
    # anim_duration    : seconds for one cycle
    # anim_loop        : repeat forever
    # anim_delay       : seconds before starting
    # anim_ease        : easing name ("linear","ease","ease_in","ease_out","ease_in_out")
    # from_color       : start color for color_transition
    # to_color         : end color for color_transition
    # pulse_min/max    : opacity range for pulse
    # slide_direction  : "left"|"right"|"top"|"bottom" for slide_in
    # slide_distance   : px to travel (default = 60)
    # shake_intensity  : px jitter radius for shake
    # wave_amplitude   : px vertical bob for wave
    # wave_speed       : cycles per second for wave
    # wave_char_offset : phase offset per character (radians)
    animation: Optional[str] = None
    anim_duration: float = 1.0
    anim_loop: bool = True
    anim_delay: float = 0.0
    anim_ease: str = "linear"
    from_color: Optional["QColor"] = None
    to_color: Optional["QColor"] = None
    pulse_min: float = 0.3
    pulse_max: float = 1.0
    pulse_speed: float = 1.0
    slide_direction: str = "left"
    slide_distance: float = 60.0
    shake_intensity: float = 4.0
    wave_amplitude: float = 6.0
    wave_speed: float = 2.0
    wave_char_offset: float = 0.4
    # runtime: set on first paint
    anim_started_at: Optional[float] = None

    # dynamic sources (Draw.live.text / Draw.input.text)
    source: Optional[object] = None
    input_enabled: bool = False
    input_take_input: str = "return"
    input_submit_keys: Tuple[str, ...] = ("return",)
    input_type: str = "all"
    input_return_spec: object = None
    input_buffer: str = ""
    input_placeholder: str = ""
    input_min_length: int = 0
    input_max_length: Optional[int] = None
    input_allow_empty: bool = True
    input_transform: str = "none"
    input_pattern: Optional[str] = None
    input_allowed_chars: Optional[frozenset[str]] = None
    input_live_update: bool = False
    input_clear_on_submit: bool = False
    input_selected: bool = False
    input_caret: bool = True
    input_caret_blink: bool = True
    input_caret_color: Optional[QColor] = None
    input_caret_width: float = 2.0
    input_caret_height_ratio: float = 0.88
    input_caret_blink_interval: float = 0.55
    layout: Optional[object] = None
    cell: Optional[object] = None
    max_width: Optional[float] = None  # auto-wrap text when set
    auto_align_in_ip: bool = False     # auto-wrap and fit inside parent shape hitbox
    hitbox_ip: Optional[str] = None    # target shape IP to fit within
    min_font_size: int = 10            # minimum font size during auto-scaling
    last_rect: Optional[Tuple[float, float, float, float]] = None

    # ── overlap avoidance ──────────────────────────────────────────────
    # When True, the text box is placed using the same closest-rectangle
    # collision strategy as shapes with overlap=False.
    # Set via customise={"closest_rect_area": True}
    closest_rect_area: bool = False
    flow: object = None

    # Runtime: placed position set by paintEvent overlap pass.
    # None = use normal positioning from x/y/align/cell.
    _placed_x: Optional[float] = None
    _placed_y: Optional[float] = None
    input_cursor_position: int = 0

    # ── text-on-arc ───────────────────────────────────────────────────
    # arc_radius    : radius in px of the arc circle (None = disabled)
    # arc_angle     : total angle span of the text in degrees (default 180)
    # arc_direction : "up" (text curves upward/above centre) or "down"
    # arc_start     : starting angle offset in degrees (default = auto-centre)
    arc_radius: Optional[float] = None
    arc_angle: float = 180.0
    arc_direction: str = "up"
    arc_start: Optional[float] = None

    # ── text stroke / outline ─────────────────────────────────────────
    # outline_width  : stroke thickness in px around each glyph (0 = off)
    # outline_color  : QColor for the stroke (default: black)
    outline_width: float = 0.0
    outline_color: Optional["QColor"] = None

    # ── gradient fill on text ─────────────────────────────────────────
    # text_gradient  : same dict format as shape gradients
    #   {"type": "linear", "angle": 90, "stops": [[0, [r,g,b]], [1, [r,g,b]]]}
    #   {"type": "radial", "center": [50,50], "radius": 50, "stops": [...]}
    text_gradient: Optional[Dict[str, Any]] = None

    # ── per-character colors ──────────────────────────────────────────
    # char_colors : list of color values cycling across characters.
    # Each entry can be a QColor, an (r,g,b) tuple, or a hex string.
    # Shorter lists cycle (modulo). Ignored if text_gradient is set.
    char_colors: Optional[List[Any]] = None

    # Z-order, overlap, and runtime metrics
    z: object = 0
    overlap: bool = True
    html: bool = False
    last_position: Optional[Tuple[float, float]] = None
    last_size: Optional[Tuple[float, float]] = None


def _normalize_input_type(raw: object) -> str:
    token = str(raw if raw is not None else "all").strip().lower()
    token = _INPUT_TYPE_ALIASES.get(token, token)
    if token not in _INPUT_TYPE_VALUES:
        return "all"
    if token == "string":
        return "str"
    return token


def _normalize_submit_key(raw: object) -> str:
    token = str(raw if raw is not None else "return").strip().lower()
    return _INPUT_SUBMIT_KEY_ALIASES.get(token, token)


def _normalize_submit_keys(raw: object) -> Tuple[str, ...]:
    if isinstance(raw, (list, tuple, set)):
        parts = [str(part).strip() for part in raw]
    else:
        text = str(raw if raw is not None else "return")
        text = text.replace("|", ",")
        parts = [part.strip() for part in text.split(",")]
    keys: list[str] = []
    for part in parts:
        if part == "":
            continue
        key = _normalize_submit_key(part)
        if key not in keys:
            keys.append(key)
    if not keys:
        keys.append("return")
    return tuple(keys)


def _input_bool(raw: object, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"true", "1", "yes", "on"}:
            return True
        if value in {"false", "0", "no", "off", ""}:
            return False
    return bool(raw)


def _normalize_input_transform(raw: object) -> str:
    token = str(raw if raw is not None else "none").strip().lower()
    return _INPUT_TRANSFORM_ALIASES.get(token, "none")


def _normalize_input_allowed_chars(raw: object) -> Optional[frozenset[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == "":
            return None
        return frozenset(raw)
    if isinstance(raw, (list, tuple, set)):
        chars: list[str] = []
        for item in raw:
            text = str(item)
            if len(text) == 0:
                continue
            chars.extend(text)
        if len(chars) == 0:
            return None
        return frozenset(chars)
    text = str(raw)
    if text == "":
        return None
    return frozenset(text)


def _apply_input_transform(value: str, transform: str) -> str:
    mode = _normalize_input_transform(transform)
    if mode == "upper":
        return value.upper()
    if mode == "lower":
        return value.lower()
    if mode == "title":
        return value.title()
    return value


def _is_input_candidate_allowed(
    value: str,
    input_type: str,
    *,
    max_length: Optional[int] = None,
    allowed_chars: Optional[frozenset[str]] = None,
) -> bool:
    if max_length is not None and max_length >= 0 and len(value) > max_length:
        return False
    if allowed_chars is not None and any(ch not in allowed_chars for ch in value):
        return False
    mode = _normalize_input_type(input_type)
    if mode in {"all", "str"}:
        return True
    if mode == "int":
        return _INT_CANDIDATE_RE.fullmatch(value) is not None
    if mode == "float":
        return _FLOAT_CANDIDATE_RE.fullmatch(value) is not None
    if mode in {"color", "border_color"}:
        return _COLOR_CANDIDATE_RE.fullmatch(value) is not None
    return True


def _is_input_final_allowed(
    value: str,
    input_type: str,
    *,
    min_length: int = 0,
    max_length: Optional[int] = None,
    allow_empty: bool = True,
    pattern: Optional[str] = None,
    allowed_chars: Optional[frozenset[str]] = None,
) -> bool:
    if not allow_empty and value == "":
        return False
    if min_length > 0 and len(value) < min_length:
        return False
    if max_length is not None and max_length >= 0 and len(value) > max_length:
        return False
    if allowed_chars is not None and any(ch not in allowed_chars for ch in value):
        return False
    if pattern is not None and pattern != "":
        try:
            if re.fullmatch(pattern, value) is None:
                return False
        except re.error:
            return False

    mode = _normalize_input_type(input_type)
    if mode == "int":
        return re.fullmatch(r"-?\d+", value) is not None
    if mode == "float":
        if value in {"", "-", ".", "-."}:
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False
    if mode in {"color", "border_color"}:
        if value.strip() == "":
            return False
        try:
            _parse_color(value)
            return True
        except Exception:
            return False
    return True


def _store_input_return_value(return_spec: object, value: str) -> None:
    target = return_spec
    if isinstance(return_spec, (list, tuple)) and len(return_spec) > 0:
        head = str(return_spec[0]).strip().lower()
        if head in {"text", "value"}:
            target = return_spec[1] if len(return_spec) > 1 else None

    _input_registry.set(None, value)

    if target is None:
        return

    try:
        if isinstance(target, str):
            _input_registry.set(target, value)
            _live_registry.set(target, value)
            return
        if isinstance(target, list):
            target.clear()
            target.append(value)
            return
        if isinstance(target, dict):
            target["text"] = value
            return
        if callable(target):
            target(value)
            return
        if hasattr(target, "set") and callable(getattr(target, "set")):
            target.set(value)
            return
        if hasattr(target, "value"):
            setattr(target, "value", value)
    except Exception as exc:
        print(f"Draw.input.text: failed to assign return target: {exc}")


# ── text animation helpers ────────────────────────────────────────────────────

_SUPPORTED_TEXT_ANIMATIONS = {
    "fade_in", "fade_out", "pulse", "blink",
    "color_transition", "rainbow", "rotate_hue",
    "typewriter",
    "slide_in", "scale_in", "rotate_in",
    "wave", "shake",
}

_TEXT_ANIM_ALIASES = {
    "fadein":        "fade_in",
    "fadeout":       "fade_out",
    "colour_transition": "color_transition",
    "colourtransition":  "color_transition",
    "colortransition":   "color_transition",
    "type":          "typewriter",
    "typing":        "typewriter",
    "write_on":      "typewriter",
    "writeon":       "typewriter",
    "write":         "typewriter",
    "slidein":       "slide_in",
    "scalein":       "scale_in",
    "rotatein":      "rotate_in",
}

_TEXT_EASE_FNS = {
    "linear":      lambda t: t,
    "ease":        lambda t: t * t * (3.0 - 2.0 * t),
    "ease_in":     lambda t: t * t,
    "easein":      lambda t: t * t,
    "ease_out":    lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "easeout":     lambda t: 1.0 - (1.0 - t) * (1.0 - t),
    "ease_in_out": lambda t: t * t * (3.0 - 2.0 * t),
    "easeinout":   lambda t: t * t * (3.0 - 2.0 * t),
}


def _text_ease(name: str, t: float) -> float:
    fn = _TEXT_EASE_FNS.get(name, _TEXT_EASE_FNS["linear"])
    return max(0.0, min(1.0, fn(max(0.0, min(1.0, t)))))


def _text_anim_t(t: "TextDef", now: float) -> float:
    """
    Return normalised animation progress [0, 1].
    Handles delay, duration, loop, and easing.
    """
    if t.anim_started_at is None:
        t.anim_started_at = now
    elapsed = now - t.anim_started_at - t.anim_delay
    if elapsed < 0.0:
        return 0.0
    duration = max(1e-6, t.anim_duration)
    if t.anim_loop:
        raw = (elapsed % duration) / duration
    else:
        raw = min(1.0, elapsed / duration)
    return _text_ease(t.anim_ease, raw)


def _lerp_color_qc(c1: "QColor", c2: "QColor", t: float) -> "QColor":
    t = max(0.0, min(1.0, t))
    return QColor(
        int(c1.red()   + (c2.red()   - c1.red())   * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue()  + (c2.blue()  - c1.blue())  * t),
        int(c1.alpha() + (c2.alpha() - c1.alpha()) * t),
    )


def _hue_shift_text(color: "QColor", delta: float) -> "QColor":
    h, s, v, a = color.getHsvF()
    h = (h + delta) % 1.0 if h >= 0.0 else delta % 1.0
    return QColor.fromHsvF(h, s, v, a)


def _text_is_animated(t: "TextDef") -> bool:
    return t.animation is not None


def _resolve_text_value(t: "TextDef") -> str:
    if t.input_enabled:
        if t.input_buffer == "" and t.input_placeholder != "":
            return t.input_placeholder
        return t.input_buffer
    return t.text


# ── shared canvas ─────────────────────────────────────────────────────────────

# ── text drawing ──────────────────────────────────────────────────────────────

def _draw_text_on_arc(painter: QPainter, t: "TextDef", ox: float, oy: float,
                      box_w: float, box_h: float, font: "QFont",
                      fm: "QFontMetricsF", text_color: "QColor") -> None:
    """
    Render text glyphs placed along a circular arc.
    Each glyph is individually rotated to follow the tangent of the arc.

    The arc circle centre is at (ox + box_w/2, oy + box_h/2 ± arc_radius)
    depending on arc_direction:
      "up"   — text curves upward (centre below the text box)
      "down" — text curves downward (centre above the text box)
    """
    import math as _math
    from PySide6.QtGui import QPainterPath as _Path, QBrush as _Brush, QTransform as _T

    render_text = t.input_buffer if t.input_enabled else t.text
    if not render_text:
        return

    R = t.arc_radius
    direction = t.arc_direction.lower()

    # Centre of the arc circle
    cx = ox + box_w / 2.0
    if direction == "down":
        cy = oy - R                     # centre above: text droops downward
    else:
        cy = oy + box_h + R             # centre below: text arcs upward

    # Measure total text width
    total_w = fm.horizontalAdvance(render_text)
    half_span = _math.asin(min(1.0, (total_w / 2.0) / R)) if R > 0 else 0.0

    # Start angle: position text centred on the top (or bottom) of the arc
    if t.arc_start is not None:
        start_angle = _math.radians(t.arc_start)
    else:
        if direction == "down":
            # centred at bottom of circle (270° = -π/2)
            start_angle = _math.pi / 2.0 - half_span
        else:
            # centred at top of circle (90° = π/2), going left→right
            start_angle = _math.pi + half_span   # start at left of arc

    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)

    cursor_angle = start_angle
    for ch in render_text:
        ch_w = fm.horizontalAdvance(ch)
        # Angle to advance for this half-character (place glyph at its centre)
        half_ch = _math.asin(min(1.0, (ch_w / 2.0) / R)) if R > 0 else 0.0
        cursor_angle += (half_ch if direction != "down" else -half_ch)

        gx = cx + R * _math.cos(cursor_angle)
        gy = cy + R * _math.sin(cursor_angle)

        # Tangent rotation: perpendicular to the radius vector
        if direction == "down":
            rot_deg = _math.degrees(cursor_angle) - 90.0
        else:
            rot_deg = _math.degrees(cursor_angle) + 90.0

        glyph_path = _Path()
        glyph_path.addText(QPointF(0, 0), font, ch)

        transform = _T()
        transform.translate(gx, gy)
        transform.rotate(rot_deg)
        transform.translate(-ch_w / 2.0, fm.ascent() / 2.0)
        glyph_path = transform.map(glyph_path)

        painter.fillPath(glyph_path, _Brush(text_color))

        cursor_angle += (half_ch if direction != "down" else -half_ch)

    painter.restore()


def _draw_one_text(painter: QPainter, t: TextDef, cw: int, ch: int, canvas=None):
    """Paint a single TextDef onto the canvas, applying any animation."""

    now = time.perf_counter()
    anim = t.animation

    # ── build font ────────────────────────────────────────────────────
    font, fm = _get_cached_font(
        t.font_family, t.font_size, t.bold, t.italic,
        t.underline, t.strikethrough, t.letter_spacing, t.word_spacing
    )
    painter.setFont(font)

    # ── resolve animation state ───────────────────────────────────────
    anim_opacity   = t.opacity / 100.0   # final opacity multiplier 0-1
    anim_color     = QColor(t.color)
    anim_x_offset  = 0.0
    anim_y_offset  = 0.0
    anim_scale     = 1.0
    anim_rotation  = t.rotation
    anim_char_reveal = -1                # -1 = show all; >=0 = show N chars (typewriter)

    if anim is not None:
        prog = _text_anim_t(t, now)

        if anim == "fade_in":
            anim_opacity = (t.opacity / 100.0) * prog

        elif anim == "fade_out":
            anim_opacity = (t.opacity / 100.0) * (1.0 - prog)

        elif anim == "pulse":
            raw = (math.sin(now * max(0.01, t.pulse_speed) * math.tau) + 1.0) * 0.5
            level = t.pulse_min + (t.pulse_max - t.pulse_min) * raw
            anim_opacity = (t.opacity / 100.0) * max(0.0, min(1.0, level))

        elif anim == "blink":
            cycle = int(now * max(0.1, t.pulse_speed) * 2)
            anim_opacity = (t.opacity / 100.0) if cycle % 2 == 0 else 0.0

        elif anim == "color_transition":
            fc = t.from_color if t.from_color is not None else t.color
            tc = t.to_color   if t.to_color   is not None else t.color
            anim_color = _lerp_color_qc(fc, tc, prog)

        elif anim == "rainbow":
            delta = now * max(0.01, t.wave_speed) * 0.15
            anim_color = _hue_shift_text(t.color, delta)

        elif anim == "rotate_hue":
            delta = (now * t.wave_speed) / 360.0
            anim_color = _hue_shift_text(t.color, delta)

        elif anim == "typewriter":
            full_text = _resolve_text_value(t)
            total_chars = max(1, len(full_text))
            anim_char_reveal = int(prog * total_chars)

        elif anim == "slide_in":
            dist = t.slide_distance * (1.0 - prog)
            d = t.slide_direction.lower()
            if d == "left":
                anim_x_offset = -dist
            elif d == "right":
                anim_x_offset = dist
            elif d == "top":
                anim_y_offset = -dist
            elif d == "bottom":
                anim_y_offset = dist

        elif anim == "scale_in":
            anim_scale = prog
            anim_opacity = (t.opacity / 100.0) * prog

        elif anim == "rotate_in":
            anim_rotation = t.rotation + (1.0 - prog) * 360.0
            anim_opacity = (t.opacity / 100.0) * prog

        # wave and shake are per-character/per-frame — handled in draw loop below

    # ── resolve displayed text ────────────────────────────────────────
    placeholder_active = (
        t.input_enabled and t.input_buffer == "" and t.input_placeholder != ""
    )
    render_text = _resolve_text_value(t)

    # Apply typewriter reveal
    if anim_char_reveal >= 0:
        render_text = render_text[:anim_char_reveal]

    lines = render_text.split("\n") if render_text != "" else [""]
    line_h = fm.height() * t.line_height

    # ── auto_align_in_ip: auto-fit and auto-wrap inside parent shape hitbox ──
    _effective_max_w = t.max_width
    if t.auto_align_in_ip or getattr(t, "hitbox_ip", None):
        target_shape = None
        if canvas is not None and hasattr(canvas, "shape_items"):
            # 1. Match by explicit hitbox_ip or prefix of t.ip
            if getattr(t, "hitbox_ip", None):
                target_shape = next((s for s in canvas.shape_items if s.ip == t.hitbox_ip), None)
            if target_shape is None and t.ip:
                prefix_cand = t.ip.rsplit("_", 1)[0] if "_" in t.ip else t.ip
                target_shape = next((s for s in canvas.shape_items if s.ip and (s.ip == prefix_cand or s.ip.startswith(prefix_cand))), None)
            # 2. Or match enclosing shape bounding box
            if target_shape is None and t.x is not None and t.y is not None:
                for s in canvas.shape_items:
                    if s.ip and not s.ip.startswith("scroller_"):
                        sx = getattr(s, "_placed_x", getattr(s, "x", 0) or 0)
                        sy = getattr(s, "_placed_y", getattr(s, "y", 0) or 0)
                        sw = getattr(s, "_placed_w", getattr(s, "width", 0) or 0)
                        sh = getattr(s, "_placed_h", getattr(s, "height", 0) or 0)
                        if s.last_size:
                            sw, sh = s.last_size
                        if sx <= float(t.x) <= sx + sw and sy <= float(t.y) <= sy + sh:
                            target_shape = s
                            break

        if target_shape is not None:
            sw = getattr(target_shape, "_placed_w", getattr(target_shape, "width", 0) or 0)
            sh = getattr(target_shape, "_placed_h", getattr(target_shape, "height", 0) or 0)
            if target_shape.last_size:
                sw, sh = target_shape.last_size
            pad_inset = float(t.background_padding) + 16.0
            avail_w = max(40.0, float(sw) - pad_inset * 2.0)
            avail_h = max(20.0, float(sh) - pad_inset * 2.0)
            _effective_max_w = avail_w

    # ── auto-wrap with QTextDocument when max_width or auto_align_in_ip is set ──
    _use_qtextdoc = False
    if _effective_max_w is not None and _effective_max_w > 0:
        from PySide6.QtGui import QTextDocument
        _text_doc = QTextDocument()
        _text_doc.setDefaultFont(font)
        _text_doc.setTextWidth(_effective_max_w)
        # Render as HTML only if caller explicitly set html=True
        if getattr(t, "html", False):
            _text_doc.setHtml(render_text)
        else:
            _text_doc.setPlainText(render_text)

        # If auto_align_in_ip, scale font down if height exceeds hitbox
        if t.auto_align_in_ip and 'avail_h' in locals():
            curr_size = font.pixelSize()
            min_size = getattr(t, "min_font_size", 10)
            while _text_doc.size().height() > avail_h and curr_size > min_size:
                curr_size -= 1
                font, fm = _get_cached_font(
                    t.font_family, curr_size, t.bold, t.italic,
                    t.underline, t.strikethrough, t.letter_spacing, t.word_spacing
                )
                _text_doc.setDefaultFont(font)

        max_w = _effective_max_w
        total_h = _text_doc.size().height()
        _use_qtextdoc = True
    else:
        max_w  = max((fm.horizontalAdvance(ln) for ln in lines), default=0.0)
        total_h = line_h * len(lines)

    # ── position bounding box ─────────────────────────────────────────
    pad   = t.background_padding
    box_w = max_w + pad * 2
    box_h = total_h + pad * 2

    cell_rect = None
    if t.layout is not None and t.cell is not None:
        from Draw._layout import set as _layout_registry
        try:
            if isinstance(t.cell, str):
                layout_obj = _layout_registry.resolve(t.cell)
                cell_rect = layout_obj.cell_rect(cw, ch, (0, 0))
            else:
                layout_obj = _layout_registry.resolve(t.layout)
                cell_rect = layout_obj.cell_rect(cw, ch, t.cell)
        except Exception as exc:
            print(f"Draw.text: failed to position text in layout cell: {exc}")

    if cell_rect is not None:
        if t.x is not None or t.y is not None:
            ox = cell_rect.left() + (float(t.x) if t.x is not None else (cell_rect.width() - box_w) / 2.0)
            oy = cell_rect.top() + (float(t.y) if t.y is not None else (cell_rect.height() - box_h) / 2.0)
        elif t.align is not None:
            w_tag = getattr(canvas, "_window_tag", None) if canvas is not None else None
            ox_rel, oy_rel = _text_align_pos(t.align, box_w, box_h, int(cell_rect.width()), int(cell_rect.height()), window_tag=w_tag)
            ox = cell_rect.left() + ox_rel
            oy = cell_rect.top() + oy_rel
        else:
            ox = cell_rect.left() + (cell_rect.width() - box_w) / 2.0
            oy = cell_rect.top() + (cell_rect.height() - box_h) / 2.0
    else:
        if t.x is not None or t.y is not None:
            ox = float(t.x) if t.x is not None else (cw - box_w) / 2.0
            oy = float(t.y) if t.y is not None else (ch - box_h) / 2.0
        elif t.align is not None:
            w_tag = getattr(canvas, "_window_tag", None) if canvas is not None else None
            ox, oy = _text_align_pos(t.align, box_w, box_h, cw, ch, window_tag=w_tag)
        else:
            ox, oy = (cw - box_w) / 2.0, (ch - box_h) / 2.0

    # ── closest_rect_area override: use position computed by overlap pass ──
    flow_spec = getattr(t, "flow", None)
    if (
        (t.closest_rect_area or getattr(flow_spec, "enabled", False))
        and t._placed_x is not None
        and t._placed_y is not None
    ):
        ox = t._placed_x
        oy = t._placed_y

    # Apply slide / shake offsets
    if anim == "shake" and anim is not None:
        intensity = t.shake_intensity
        ox += (math.sin(now * 47.3 + 1.1) * intensity)
        oy += (math.sin(now * 53.7 + 2.3) * intensity)

    ox += anim_x_offset
    oy += anim_y_offset

    # ── Draw.motion override (geometry/color set via Draw.motion(...)) ──
    # Mirrors the shape pipeline: any MotionRecord attached to this TextDef
    # (via Draw.motion(motion_ip=t.ip, ...)) is resolved into a state dict
    # by the canvas paint loop and stored on t._last_motion_state. We apply
    # it here as the final geometry/color override before painting.
    motion_state = getattr(t, "_last_motion_state", None)
    if motion_state:
        if "x" in motion_state:
            ox = float(motion_state["x"])
        if "y" in motion_state:
            oy = float(motion_state["y"])
        if "color" in motion_state:
            anim_color = motion_state["color"]
        if "opacity" in motion_state:
            anim_opacity = max(0.0, min(1.0, float(motion_state["opacity"]) / 100.0))

    if canvas is not None and not (t.ip and t.ip.startswith("scroller_")):
        ox -= canvas._scroll_x
        oy -= canvas._scroll_y

    t.last_position = (ox, oy)
    t.last_size = (box_w, box_h)
    t.last_rect = (ox, oy, box_w, box_h)

    painter.save()
    painter.setOpacity(max(0.0, min(1.0, anim_opacity)))

    # ── scale_in: scale around box centre ────────────────────────────
    if anim in ("scale_in", "rotate_in") and anim_scale != 1.0 or anim_rotation != t.rotation:
        cx_box = ox + box_w / 2.0
        cy_box = oy + box_h / 2.0
        painter.translate(cx_box, cy_box)
        if anim_scale != 1.0:
            painter.scale(anim_scale, anim_scale)
        if anim_rotation != 0.0:
            painter.rotate(anim_rotation)
        painter.translate(-cx_box, -cy_box)
    elif t.rotation != 0.0 and anim not in ("rotate_in",):
        cx_box = ox + box_w / 2.0
        cy_box = oy + box_h / 2.0
        painter.translate(cx_box, cy_box)
        painter.rotate(t.rotation)
        painter.translate(-cx_box, -cy_box)

    # ── shadow ────────────────────────────────────────────────────────
    if t.shadow:
        dx, dy = t.shadow_offset
        _draw_text_shadow(painter, t, ox, oy, box_w, box_h, dx, dy, lines,
                          line_h, pad, fm)

    # ── background box ────────────────────────────────────────────────
    if t.background_color is not None:
        rect = QRectF(ox, oy, box_w, box_h)
        painter.setBrush(QBrush(t.background_color))
        if t.border_width > 0:
            painter.setPen(QPen(t.border_color, t.border_width))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        if t.border_radius > 0:
            painter.drawRoundedRect(rect, t.border_radius, t.border_radius)
        else:
            painter.drawRect(rect)
    elif t.border_width > 0:
        rect = QRectF(ox, oy, box_w, box_h)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(t.border_color, t.border_width))
        if t.border_radius > 0:
            painter.drawRoundedRect(rect, t.border_radius, t.border_radius)
        else:
            painter.drawRect(rect)

    # ── glow ──────────────────────────────────────────────────────────
    if t.glow:
        _draw_text_glow(painter, t, lines, ox + pad, oy + pad, line_h, fm,
                        box_w - pad * 2)

    # ── dynamic color binding (Draw.color(ip=t.ip, ...)) ────────────────
    # Mirrors the shape pipeline in _shapes.py: shapes were already wired to
    # Draw.color()'s dynamic/gradient registry, but text never queried it at
    # all, so Draw.color(ip=<text ip>, ...) silently did nothing for
    # Draw.text() items even though _colour.py's own entry targets already
    # include "text". This closes that gap via the shared bridge helper.
    from Draw import _bridge
    dyn_gradient = None
    color_data = _bridge.resolve_dynamic_color(t.ip, x=ox, y=oy, w=box_w, h=box_h)
    if color_data:
        dyn_rgba = color_data.get("text_color")
        if dyn_rgba:
            anim_color = QColor(dyn_rgba[0], dyn_rgba[1], dyn_rgba[2], dyn_rgba[3])
        dyn_gradient = color_data.get("text_gradient")

    # ── draw each line ────────────────────────────────────────────────
    text_color = QColor(anim_color)
    if placeholder_active:
        text_color.setAlpha(max(30, int(text_color.alpha() * 0.55)))
    painter.setFont(font)

    text_area_w = box_w - pad * 2
    qt_align = {
        "left":   Qt.AlignmentFlag.AlignLeft,
        "center": Qt.AlignmentFlag.AlignHCenter,
        "right":  Qt.AlignmentFlag.AlignRight,
    }.get(t.align_text, Qt.AlignmentFlag.AlignLeft)

    # ── gradient / per-character fill ──────────────────────────────────
    # text_gradient and char_colors were parsed onto TextDef at customise()
    # time but nothing downstream ever read them back -- dead fields.
    # dyn_gradient (from a Draw.color() dynamic binding, resolved above)
    # takes priority over a static customise={"text_gradient": ...} value
    # if both are set.
    _effective_gradient = dyn_gradient or t.text_gradient
    _grad_brush = None
    if _effective_gradient:
        _build_gradient_brush = _bridge.get_gradient_brush_builder()
        _grad_brush = _build_gradient_brush(_effective_gradient, ox, oy, int(box_w), int(box_h))

    def _glyph_brush(char_idx: int) -> QBrush:
        if _grad_brush is not None:
            return _grad_brush
        if t.char_colors:
            return QBrush(t.char_colors[char_idx % len(t.char_colors)])
        return QBrush(text_color)

    # ── arc text: render glyphs on circular path ──────────────────────
    arc_r = getattr(t, "arc_radius", None)
    if arc_r is not None and arc_r > 0:
        _draw_text_on_arc(painter, t, ox, oy, box_w, box_h, font, fm, text_color)
    elif anim == "wave":
        # Per-character wave: draw each char individually with a y offset
        amp   = t.wave_amplitude
        spd   = t.wave_speed
        phase = t.wave_char_offset
        char_idx = 0
        for i, line in enumerate(lines):
            line_y_base = oy + pad + i * line_h + fm.ascent()
            cursor_x    = ox + pad
            if t.align_text == "center":
                cursor_x = ox + pad + (text_area_w - fm.horizontalAdvance(line)) / 2.0
            elif t.align_text == "right":
                cursor_x = ox + pad + text_area_w - fm.horizontalAdvance(line)
            for ch_char in line:
                bob = math.sin(now * spd * math.tau + char_idx * phase) * amp
                path = QPainterPath()
                path.addText(QPointF(cursor_x, line_y_base + bob), font, ch_char)
                painter.fillPath(path, _glyph_brush(char_idx))
                cursor_x += fm.horizontalAdvance(ch_char)
                char_idx += 1
    elif (_grad_brush is not None or t.char_colors) and not _use_qtextdoc:
        # Gradient / per-character fill on plain (non-wrapped) text: draw
        # glyph-by-glyph instead of a single drawText() call -- the same
        # technique the wave animation above already used for its per-char
        # pen, just without the bob offset.
        char_idx = 0
        for i, line in enumerate(lines):
            line_y_base = oy + pad + i * line_h + fm.ascent()
            cursor_x    = ox + pad
            if t.align_text == "center":
                cursor_x = ox + pad + (text_area_w - fm.horizontalAdvance(line)) / 2.0
            elif t.align_text == "right":
                cursor_x = ox + pad + text_area_w - fm.horizontalAdvance(line)
            for ch_char in line:
                path = QPainterPath()
                path.addText(QPointF(cursor_x, line_y_base), font, ch_char)
                painter.fillPath(path, _glyph_brush(char_idx))
                cursor_x += fm.horizontalAdvance(ch_char)
                char_idx += 1
    else:
        # Normal line-by-line draw
        painter.setPen(QPen(text_color))
        if _use_qtextdoc:
            # Use QTextDocument for wrapped/rich text rendering
            painter.save()
            painter.translate(ox + pad, oy + pad)
            _text_doc.setDefaultStyleSheet(f"body {{ color: {text_color.name()}; }}")
            from PySide6.QtGui import QAbstractTextDocumentLayout
            ctx = QAbstractTextDocumentLayout.PaintContext()
            ctx.palette.setColor(ctx.palette.ColorRole.Text, text_color)
            _text_doc.documentLayout().draw(painter, ctx)
            painter.restore()
        else:
            for i, line in enumerate(lines):
                line_y = oy + pad + i * line_h
                line_rect = QRectF(ox + pad, line_y, text_area_w, line_h)
                painter.drawText(line_rect,
                                 qt_align | Qt.AlignmentFlag.AlignVCenter,
                                 line)

    if t.input_enabled and t.input_selected and t.input_caret:
        _draw_input_caret(painter, t, ox, oy, pad, line_h, fm, text_area_w)

    painter.restore()


def _draw_input_caret(
    painter: QPainter,
    t: TextDef,
    ox: float,
    oy: float,
    pad: int,
    line_h: float,
    fm: QFontMetricsF,
    text_area_w: float,
) -> None:
    if t.input_caret_blink:
        interval = max(0.1, float(t.input_caret_blink_interval))
        if (time.monotonic() % (interval * 2.0)) >= interval:
            return

    cursor_pos = getattr(t, "input_cursor_position", len(t.input_buffer))
    cursor_pos = max(0, min(len(t.input_buffer), cursor_pos))

    # Split the buffer up to the cursor position to find the active line and column offset
    text_before_cursor = t.input_buffer[:cursor_pos]
    lines_before = text_before_cursor.split("\n")
    line_index = len(lines_before) - 1
    caret_line_prefix = lines_before[-1]
    caret_line_prefix_w = fm.horizontalAdvance(caret_line_prefix)

    # Get the full active line to determine alignment offsets correctly
    all_lines = t.input_buffer.split("\n") if t.input_buffer != "" else [""]
    full_line = all_lines[line_index] if line_index < len(all_lines) else ""
    full_line_w = fm.horizontalAdvance(full_line)

    if t.align_text == "center":
        line_start_x = ox + pad + (text_area_w - full_line_w) / 2.0
    elif t.align_text == "right":
        line_start_x = ox + pad + text_area_w - full_line_w
    else:
        line_start_x = ox + pad

    caret_x = line_start_x + caret_line_prefix_w
    caret_h = max(1.0, line_h * max(0.1, min(1.0, t.input_caret_height_ratio)))
    caret_y = oy + pad + line_index * line_h + (line_h - caret_h) / 2.0
    color = QColor(t.input_caret_color or t.color)
    painter.setPen(QPen(color, max(1.0, float(t.input_caret_width))))
    painter.drawLine(QPointF(caret_x, caret_y), QPointF(caret_x, caret_y + caret_h))


def _text_align_pos(align: str, bw: float, bh: float,
                    cw: int, ch: int, window_tag: Optional[str] = None) -> Tuple[float, float]:
    from Draw._align import calculate_alignment_pos
    return calculate_alignment_pos(align, bw, bh, float(cw), float(ch), window_tag=window_tag)


def _draw_text_shadow(painter, t, ox, oy, bw, bh,
                      dx, dy, lines, line_h, pad, fm):
    sc = QColor(t.shadow_color)
    sc.setAlpha(140)
    painter.setPen(QPen(sc))
    painter.setFont(painter.font())
    text_area_w = bw - pad * 2
    qt_align = {
        "left":   Qt.AlignmentFlag.AlignLeft,
        "center": Qt.AlignmentFlag.AlignHCenter,
        "right":  Qt.AlignmentFlag.AlignRight,
    }.get(t.align_text, Qt.AlignmentFlag.AlignLeft)
    for i, line in enumerate(lines):
        line_y = oy + pad + i * line_h
        line_rect = QRectF(ox + pad + dx, line_y + dy, text_area_w, line_h)
        painter.drawText(line_rect,
                         qt_align | Qt.AlignmentFlag.AlignVCenter,
                         line)


def _draw_text_glow(painter, t, lines, tx, ty, line_h, fm, area_w):
    """Build a QPainterPath from glyph outlines and stroke it for glow."""
    qt_align = {
        "left":   Qt.AlignmentFlag.AlignLeft,
        "center": Qt.AlignmentFlag.AlignHCenter,
        "right":  Qt.AlignmentFlag.AlignRight,
    }.get(t.align_text, Qt.AlignmentFlag.AlignLeft)

    steps = 6
    for step in range(steps, 0, -1):
        alpha = int(70 * step / steps)
        width = t.glow_radius * step / steps
        gc = QColor(t.glow_color)
        gc.setAlpha(alpha)
        pen = QPen(gc, width * 2)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for i, line in enumerate(lines):
            line_y = ty + i * line_h
            line_rect = QRectF(tx, line_y, area_w, line_h)
            # Draw outlined text path for glow
            path = QPainterPath()
            path.addText(
                QPointF(
                    tx if t.align_text == "left"
                    else tx + (area_w - fm.horizontalAdvance(line)) / 2
                    if t.align_text == "center"
                    else tx + area_w - fm.horizontalAdvance(line),
                    line_y + fm.ascent(),
                ),
                painter.font(),
                line,
            )
            painter.drawPath(path)


def measure_text(
    text: str,
    *,
    font_family: str = "Arial",
    font_size: int = 24,
    bold: bool = False,
    italic: bool = False,
    letter_spacing: float = 0.0,
    line_height: float = 1.2,
    max_width: Optional[float] = None,
    background_padding: int = 0,
    html: bool = False,
) -> Tuple[float, float]:
    """
    Headlessly measure the rendered (width, height) of a text string using
    the exact same font/wrap logic _draw_one_text uses, WITHOUT needing a
    live QPainter or an open window. This mirrors what CSS layout engines
    (intrinsic size) and Qt's own QFontMetrics/QTextDocument give you for
    free, but exposes it as a single public call so callers can do real
    box-model layout (auto-wrap + stack by measured height) instead of
    estimating with len(text) * px-per-char heuristics.

    Returns (box_w, box_h) including background_padding on all sides,
    matching how `box_w`/`box_h` are computed inside _draw_one_text.
    """
    get_app()  # QFontMetricsF / QTextDocument need a live QApplication

    font, fm = _get_cached_font(
        font_family, font_size, bold, italic,
        False, False, letter_spacing, 0.0
    )

    lines = text.split("\n") if text != "" else [""]
    line_h = fm.height() * line_height

    if max_width is not None and max_width > 0:
        from PySide6.QtGui import QTextDocument
        doc = QTextDocument()
        doc.setDefaultFont(font)
        doc.setTextWidth(max_width)
        if html:
            doc.setHtml(text)
        else:
            doc.setPlainText(text)
        max_w = max_width
        total_h = doc.size().height()
    else:
        max_w = max((fm.horizontalAdvance(ln) for ln in lines), default=0.0)
        total_h = line_h * len(lines)

    pad = background_padding
    return (max_w + pad * 2, total_h + pad * 2)


# ── registry ─────────────────────────────────────────────────────────────────

class _TextRegistry:
    """Singleton exposed as Draw.text."""

    @staticmethod
    def register_font(font_path: str) -> bool:
        """Register a custom TTF / OTF font file for use in Draw.text."""
        get_app()
        font_id = QFontDatabase.addApplicationFont(font_path)
        return font_id != -1

    @staticmethod
    def measure(
        text: str,
        customise: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Public measurement API: Draw.text.measure("some string", customise={...}).

        Accepts the same style keys as Draw.text()'s `customise` dict
        (font_family, font_size, bold, italic, letter_spacing, line_height,
        max_width / width, background_padding) and returns the (width,
        height) box that string would occupy if drawn with those settings —
        without creating any TextDef or touching the canvas.

        Use this to lay out message bubbles, chat logs, cards, etc. by
        real measured size instead of guessing with character counts,
        the same way you'd call element.getBoundingClientRect() in CSS
        or QFontMetrics.boundingRect() directly in Qt.
        """
        c = dict(customise or {})
        c.update(kwargs)
        max_w = c.get("width", c.get("max_width", None))
        return measure_text(
            text,
            font_family=c.get("font_family", "Arial"),
            font_size=int(c.get("font_size", 24)),
            bold=bool(c.get("bold", False)),
            italic=bool(c.get("italic", False)),
            letter_spacing=float(c.get("letter_spacing", 0)),
            line_height=float(c.get("line_height", 1.2)),
            max_width=float(max_w) if max_w is not None else None,
            background_padding=int(c.get("background_padding", 0)),
        )

    def __call__(
        self,
        *,
        tag: Optional[str] = None,
        display: Optional[str] = None,
        text: object,
        ip: Optional[str] = None,
        customise: dict = None,
        properties: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Add text to the window identified by *tag* or *display*.

        Parameters
        ----------
        tag / display: Must match an existing Draw.window tag.
        text      : String, callable, Draw.live.text(...), or Draw.input.text.
                    Use \\n for line breaks in plain strings.
        customise : dict with any of the following keys (all optional):
        properties: optional input behavior config for Draw.input.text.
                    Supports: take_input, type_input, placeholder, min_length,
                    max_length, allow_empty, transform, pattern, allowed_chars,
                    live_update, clear_on_submit, return, caret/cursor,
                    caret_color/cursor_color, caret_width/cursor_width,
                    caret_blink/cursor_blink.

            Position
            --------
            x, y              : absolute pixel position of the text box
            align             : canvas position —
                                "center" | "top" | "bottom" | "left" | "right" |
                                "top-left" | "top-right" |
                                "bottom-left" | "bottom-right"

            Font
            ----
            font_family       : font name  (default "Arial")
            font_size         : pixel size (default 24)
            bold              : True/False (default False)
            italic            : True/False (default False)
            underline         : True/False (default False)
            strikethrough     : True/False (default False)
            letter_spacing    : extra px between letters (default 0)
            line_height       : line-height multiplier  (default 1.2)

            Colour & opacity
            ----------------
            color             : text colour — name / hex / RGB tuple (default "black")
            opacity           : 0-100 (default 100)

            Text alignment (inside the bounding box)
            -----------------------------------------
            align_text        : "left" | "center" | "right"  (default "left")

            Background box
            --------------
            background_color  : fill colour or None (default None = transparent)
            background_padding: px padding around text  (default 6)
            border_width      : px  (default 0)
            border_color      : color  (default "black")
            border_radius     : corner radius px  (default 0)

            Effects
            -------
            glow              : True/False (default False)
            glow_color        : color  (default same as text color)
            glow_radius       : px  (default 12)
            shadow            : True/False (default False)
            shadow_color      : color  (default "black")
            shadow_offset     : (dx, dy) tuple  (default (3, 3))

            Transform
            ---------
            rotation          : degrees clockwise  (default 0)
        """
        get_app()
        c = dict(customise or {})
        c.update(kwargs)
        ip_val = ip or c.get("ip", None)

        window_tag = display or tag
        if isinstance(text, (list, tuple)):
            for item in text:
                if isinstance(item, dict):
                    item_dict = dict(item)
                    item_text = item_dict.pop("text", "")
                    item_ip = item_dict.pop("ip", ip)
                    item_props = item_dict.pop("properties", properties)
                    self.__call__(
                        tag=window_tag,
                        display=window_tag,
                        text=item_text,
                        ip=item_ip,
                        customise=item_dict,
                        properties=item_props
                    )
            return

        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise ValueError(
                    "Draw.text: multiple windows exist; 'tag' or 'display' is required."
                )
            else:
                raise ValueError("Draw.text: no windows exist to draw text on.")

        win: QMainWindow = _window_registry.get(window_tag)
        canvas = _get_or_create_canvas(window_tag, win)

        text_source = None
        text_value = text
        if isinstance(text, InputTextMarker):
            text_source = text
            text_value = text.initial
        elif isinstance(text, LiveTextBinding):
            text_source = text
            text_value = resolve_live_text(text)
        elif callable(text):
            text_source = LiveTextBinding(text)
            text_value = resolve_live_text(text_source)

        tdef = self._parse(text_value, c, ip=ip_val, source=text_source, properties=properties)

        # Enforce ip uniqueness: replace any existing item sharing this ip
        # instead of stacking a duplicate on top of it (prevents ghosting/
        # meshed text when callers redraw with the same ip without first
        # calling remove_by_ip).
        if ip_val is not None:
            canvas.text_items = [t for t in canvas.text_items if t.ip != ip_val]

        canvas.text_items.append(tdef)
        canvas.update()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(
        text: str,
        c: dict,
        ip: Optional[str] = None,
        *,
        source: object = None,
        properties: Optional[dict] = None,
        layout: Optional[object] = None,
        cell: Optional[object] = None,
    ) -> TextDef:
        text = "" if text is None else str(text)
        ip_val = ip if ip is not None else c.get("ip", None)

        # position
        x     = c.get("x", None)
        y     = c.get("y", None)
        align = c.get("align", None)
        if align is not None and align not in _ALIGN_VALUES:
            raise ValueError(f"Draw.text: invalid align='{align}'.")

        # cell and layout
        cell_raw = cell if cell is not None else c.get("column", c.get("columns", None))
        layout_val = layout if layout is not None else c.get("layout", c.get("get_ip", None))

        parsed_cell = cell_raw
        if cell_raw is not None and not isinstance(cell_raw, str):
            from Draw._layout import _parse_cell_ref
            try:
                parsed_cell = _parse_cell_ref(cell_raw)
            except Exception:
                pass

        # font
        font_family    = str(c.get("font_family", "Arial"))
        font_size      = int(c.get("font_size", 24))
        bold           = bool(c.get("bold", False))
        italic         = bool(c.get("italic", False))
        underline      = bool(c.get("underline", False))
        strikethrough  = bool(c.get("strikethrough", False))
        letter_spacing = float(c.get("letter_spacing", 0))
        word_spacing   = float(c.get("word_spacing", 0))
        line_height    = float(c.get("line_height", 1.2))
        elide          = c.get("elide", None)
        max_width      = c.get("width", c.get("max_width", None))
        auto_align_in_ip = bool(c.get("auto_align_in_ip", c.get("auto_align", c.get("fit_in_ip", c.get("fit_hitbox", False)))))
        hitbox_ip      = c.get("hitbox", c.get("parent_shape", c.get("fit_ip", None)))
        min_font_size  = int(c.get("min_font_size", c.get("min_size", 10)))

        # colour
        color   = _parse_color(c.get("color", "black"))
        opacity = max(0, min(100, int(c.get("opacity", 100))))

        # inner alignment
        align_text = str(c.get("align_text", "left")).lower()
        if align_text not in ("left", "center", "right"):
            align_text = "left"

        # background box
        bg_raw = c.get("background_color", None)
        background_color = _parse_color(bg_raw) if bg_raw is not None else None
        background_padding = int(c.get("background_padding", 6))
        border_width  = int(c.get("border_width", 0))
        border_color  = _parse_color(c.get("border_color", "black"))
        border_radius = float(c.get("border_radius", 0))

        # glow
        glow        = bool(c.get("glow", False))
        glow_color  = _parse_color(c.get("glow_color", c.get("color", "black")))
        glow_radius = int(c.get("glow_radius", 12))

        # shadow
        shadow        = bool(c.get("shadow", False))
        shadow_color  = _parse_color(c.get("shadow_color", "black"))
        so            = c.get("shadow_offset", (3, 3))
        shadow_offset = (int(so[0]), int(so[1]))

        # transform
        rotation = float(c.get("rotation", 0.0))
        html     = bool(c.get("html", c.get("rich", c.get("rich_text", False))))

        # ── text-on-arc ───────────────────────────────────────────────
        arc_radius_raw = c.get("arc_radius", None)
        arc_radius: Optional[float] = float(arc_radius_raw) if arc_radius_raw is not None else None
        arc_angle = float(c.get("arc_angle", 180.0))
        arc_direction = str(c.get("arc_direction", "up")).strip().lower()
        if arc_direction not in ("up", "down"):
            arc_direction = "up"
        arc_start_raw = c.get("arc_start", None)
        arc_start: Optional[float] = float(arc_start_raw) if arc_start_raw is not None else None

        # ── text stroke / outline ─────────────────────────────────────
        outline_width = float(c.get("outline_width", 0.0))
        outline_color_raw = c.get("outline_color", None)
        if outline_color_raw is not None:
            from Draw._shapes import _parse_color as _pc
            outline_color = _pc(outline_color_raw)
        else:
            outline_color = None

        # ── gradient fill on text ─────────────────────────────────────
        text_gradient_raw = c.get("text_gradient", None)
        text_gradient = dict(text_gradient_raw) if isinstance(text_gradient_raw, dict) else None

        # ── per-character colors ──────────────────────────────────────
        char_colors_raw = c.get("char_colors", None)
        char_colors: Optional[List[Any]] = None
        if isinstance(char_colors_raw, (list, tuple)) and len(char_colors_raw) > 0:
            from Draw._shapes import _parse_color as _pcc
            char_colors = []
            for cv in char_colors_raw:
                if isinstance(cv, QColor):
                    char_colors.append(cv)
                elif isinstance(cv, (list, tuple)) and len(cv) >= 3:
                    char_colors.append(QColor(int(cv[0]), int(cv[1]), int(cv[2])))
                elif isinstance(cv, str):
                    try:
                        char_colors.append(_pcc(cv))
                    except Exception:
                        char_colors.append(QColor("white"))

        # overlap avoidance
        overlap = _input_bool(c.get("overlap", True), True)
        closest_rect_area = bool(c.get("closest_rect_area", False))
        from Draw._overlap import parse_flow_spec
        flow = parse_flow_spec(
            c.get("flow", None),
            flow_provided=("flow" in c),
            overlap=overlap,
            closest_rect_area=closest_rect_area,
        )

        if "z" in c and c["z"] is not None:
            z_raw = c["z"]
            if z_raw == "as_shape":
                z = "as_shape"
            else:
                try:
                    z = int(z_raw)
                except (ValueError, TypeError):
                    from Draw._tools import next_z
                    z = int(next_z())
        else:
            from Draw._tools import next_z
            z = int(next_z())

        # ── animation ─────────────────────────────────────────────────
        anim_raw = str(c.get("animation", "") or "").strip().lower()
        anim_raw = _TEXT_ANIM_ALIASES.get(anim_raw, anim_raw)
        animation: Optional[str] = anim_raw if anim_raw in _SUPPORTED_TEXT_ANIMATIONS else None

        anim_duration  = max(1e-6, float(c.get("duration",  c.get("anim_duration",  1.0))))
        anim_loop      = bool(c.get("loop",      c.get("anim_loop",      True)))
        anim_delay     = max(0.0,  float(c.get("delay",     c.get("anim_delay",     0.0))))
        anim_ease_raw  = str(c.get("ease",       c.get("anim_ease",      "linear"))).strip().lower()
        anim_ease      = anim_ease_raw if anim_ease_raw in _TEXT_EASE_FNS else "linear"

        from_color_raw = c.get("from_color", None)
        to_color_raw   = c.get("to_color",   None)
        from_color: Optional[QColor] = _parse_color(from_color_raw) if from_color_raw else None
        to_color:   Optional[QColor] = _parse_color(to_color_raw)   if to_color_raw   else None

        pulse_min       = max(0.0, min(1.0, float(c.get("pulse_min",       c.get("pulse_intensity_min", 0.3)))))
        pulse_max       = max(0.0, min(1.0, float(c.get("pulse_max",       c.get("pulse_intensity_max", 1.0)))))
        pulse_speed     = max(0.01, float(c.get("pulse_speed",    1.0)))
        slide_direction = str(c.get("slide_direction", "left")).strip().lower()
        if slide_direction not in ("left", "right", "top", "bottom"):
            slide_direction = "left"
        slide_distance  = max(0.0, float(c.get("slide_distance", 60.0)))
        shake_intensity = max(0.0, float(c.get("shake_intensity", 4.0)))
        wave_amplitude  = max(0.0, float(c.get("wave_amplitude",  6.0)))
        wave_speed      = max(0.01, float(c.get("wave_speed",     2.0)))
        wave_char_offset= float(c.get("wave_char_offset", 0.4))

        prop = properties or {}
        if not isinstance(prop, dict):
            raise TypeError("Draw.text: text 'properties' must be a dict.")

        input_enabled = bool(prop.get("input", False))
        if isinstance(source, InputTextMarker):
            input_enabled = True

        input_submit_keys = _normalize_submit_keys(
            prop.get("take_input", prop.get("submit_key", "return"))
        )
        input_take_input = input_submit_keys[0]
        input_type = _normalize_input_type(
            prop.get("type_input", prop.get("typr_input", prop.get("input_type", "all")))
        )
        input_return_spec = prop.get("return", None)
        input_placeholder = str(prop.get("placeholder", prop.get("placeholder_text", "")) or "")

        min_length_raw = prop.get("min_length", prop.get("min_chars", 0))
        input_min_length = max(0, int(min_length_raw))

        max_length_raw = prop.get("max_length", prop.get("max_chars", None))
        input_max_length: Optional[int]
        if max_length_raw is None or max_length_raw == "":
            input_max_length = None
        else:
            input_max_length = max(0, int(max_length_raw))
            if input_max_length < input_min_length:
                input_max_length = input_min_length

        input_allow_empty = bool(prop.get("allow_empty", prop.get("allow_blank", True)))
        input_transform = _normalize_input_transform(
            prop.get("transform", prop.get("input_case", "none"))
        )
        input_pattern_raw = prop.get("pattern", prop.get("regex", None))
        input_pattern = None if input_pattern_raw in (None, "") else str(input_pattern_raw)
        input_allowed_chars = _normalize_input_allowed_chars(
            prop.get("allowed_chars", prop.get("allow_chars", None))
        )
        input_live_update = _input_bool(prop.get("live_update", prop.get("live", False)))
        input_clear_on_submit = _input_bool(prop.get("clear_on_submit", False))
        input_caret = _input_bool(prop.get("caret", prop.get("cursor", True)), True)
        input_caret_blink = _input_bool(
            prop.get("caret_blink", prop.get("cursor_blink", True)),
            True,
        )
        input_caret_color_raw = prop.get("caret_color", prop.get("cursor_color", None))
        input_caret_color = None if input_caret_color_raw in (None, "") else _parse_color(input_caret_color_raw)
        input_caret_width = max(
            1.0,
            float(prop.get("caret_width", prop.get("cursor_width", 2))),
        )
        input_caret_height_ratio = max(
            0.1,
            min(1.0, float(prop.get("caret_height", prop.get("cursor_height", 0.88)))),
        )
        input_caret_blink_interval = max(
            0.1,
            float(prop.get("caret_blink_interval", prop.get("cursor_blink_interval", 0.55))),
        )

        input_buffer = str(text)
        if isinstance(source, InputTextMarker):
            marker_initial = getattr(source, "initial", "")
            if marker_initial != "":
                input_buffer = marker_initial
        input_buffer = _apply_input_transform(input_buffer, input_transform)
        if input_max_length is not None and len(input_buffer) > input_max_length:
            input_buffer = input_buffer[:input_max_length]

        return TextDef(
            text               = text,
            ip                 = ip_val,
            x                  = x,
            y                  = y,
            align              = align,
            font_family        = font_family,
            font_size          = font_size,
            bold               = bold,
            italic             = italic,
            underline          = underline,
            strikethrough      = strikethrough,
            letter_spacing     = letter_spacing,
            word_spacing       = word_spacing,
            line_height        = line_height,
            elide              = str(elide) if elide else None,
            color              = color,
            opacity            = opacity,
            align_text         = align_text,
            background_color   = background_color,
            background_padding = background_padding,
            border_width       = border_width,
            border_color       = border_color,
            border_radius      = border_radius,
            glow               = glow,
            glow_color         = glow_color,
            glow_radius        = glow_radius,
            shadow             = shadow,
            shadow_color       = shadow_color,
            shadow_offset      = shadow_offset,
            rotation           = rotation,
            animation          = animation,
            anim_duration      = anim_duration,
            anim_loop          = anim_loop,
            anim_delay         = anim_delay,
            anim_ease          = anim_ease,
            from_color         = from_color,
            to_color           = to_color,
            pulse_min          = pulse_min,
            pulse_max          = pulse_max,
            pulse_speed        = pulse_speed,
            slide_direction    = slide_direction,
            slide_distance     = slide_distance,
            shake_intensity    = shake_intensity,
            wave_amplitude     = wave_amplitude,
            wave_speed         = wave_speed,
            wave_char_offset   = wave_char_offset,
            source             = source,
            input_enabled      = input_enabled,
            input_take_input   = input_take_input,
            input_submit_keys  = input_submit_keys,
            input_type         = input_type,
            input_return_spec  = input_return_spec,
            input_buffer       = input_buffer,
            input_placeholder  = input_placeholder,
            input_min_length   = input_min_length,
            input_max_length   = input_max_length,
            input_allow_empty  = input_allow_empty,
            input_transform    = input_transform,
            input_pattern      = input_pattern,
            input_allowed_chars = input_allowed_chars,
            input_live_update  = input_live_update,
            input_clear_on_submit = input_clear_on_submit,
            input_caret        = input_caret,
            input_caret_blink  = input_caret_blink,
            input_caret_color  = input_caret_color,
            input_caret_width  = input_caret_width,
            input_caret_height_ratio = input_caret_height_ratio,
            input_caret_blink_interval = input_caret_blink_interval,
            layout             = layout_val,
            cell               = parsed_cell,
            max_width          = float(max_width) if max_width is not None else None,
            auto_align_in_ip   = auto_align_in_ip,
            hitbox_ip          = str(hitbox_ip) if hitbox_ip is not None else None,
            min_font_size      = min_font_size,
            closest_rect_area  = closest_rect_area,
            flow               = flow,
            arc_radius         = arc_radius,
            arc_angle          = arc_angle,
            arc_direction      = arc_direction,
            arc_start          = arc_start,
            outline_width      = outline_width,
            outline_color      = outline_color,
            text_gradient      = text_gradient,
            char_colors        = char_colors,
            z                  = z,
            overlap            = overlap,
            html               = html,
        )

    # ------------------------------------------------------------------ #

    def clear(self, tag: str) -> None:
        """Remove all text items from a window's canvas."""
        win = _window_registry.get(tag)
        if hasattr(win, "_draw_canvas"):
            win._draw_canvas.text_items.clear()
            win._draw_canvas.update()

    def remove(self, tag: str, index: int) -> None:
        """Remove a single text item by insertion index."""
        win = _window_registry.get(tag)
        if hasattr(win, "_draw_canvas"):
            items = win._draw_canvas.text_items
            if 0 <= index < len(items):
                items.pop(index)
                win._draw_canvas.update()

    def list_texts(self, tag: str) -> list[TextDef]:
        """Return all TextDef objects for a window."""
        win = _window_registry.get(tag)
        if hasattr(win, "_draw_canvas"):
            return list(win._draw_canvas.text_items)
        return []

    def update_by_ip(
        self,
        tag_or_ip: str,
        ip_or_text: Optional[str] = None,
        new_text: Optional[str] = None,
        *,
        text: Optional[str] = None,
        **style_overrides
    ) -> bool:
        """
        Update the text content and style of a text item identified by ip.
        Supports both signatures:
        1. update_by_ip(ip, text="new_text", **style)
        2. update_by_ip(tag, ip, new_text, **style)
        """
        if new_text is not None:
            tag = tag_or_ip
            ip = ip_or_text
            txt_val = new_text
        else:
            tag = None
            ip = tag_or_ip
            txt_val = text if text is not None else (ip_or_text if ip_or_text is not None else "")

        tags = [tag] if tag is not None else _window_registry.list_all_tags()

        for t_tag in tags:
            win = _window_registry.get(t_tag)
            if not hasattr(win, "_draw_canvas"):
                continue
            canvas = win._draw_canvas
            for i, tdef in enumerate(canvas.text_items):
                if tdef.ip == ip:
                    # Replace with new text def preserving the ip
                    c = {
                        "x": tdef.x, "y": tdef.y, "align": tdef.align,
                        "font_family": tdef.font_family, "font_size": tdef.font_size,
                        "bold": tdef.bold, "italic": tdef.italic,
                        "underline": tdef.underline, "strikethrough": tdef.strikethrough,
                        "letter_spacing": tdef.letter_spacing, "line_height": tdef.line_height,
                        "color": tdef.color, "opacity": tdef.opacity,
                        "align_text": tdef.align_text,
                        "background_color": tdef.background_color,
                        "background_padding": tdef.background_padding,
                        "border_width": tdef.border_width, "border_color": tdef.border_color,
                        "border_radius": tdef.border_radius,
                        "glow": tdef.glow, "glow_color": tdef.glow_color,
                        "glow_radius": tdef.glow_radius,
                        "shadow": tdef.shadow, "shadow_color": tdef.shadow_color,
                        "shadow_offset": tdef.shadow_offset,
                        "rotation": tdef.rotation,
                        "animation": tdef.animation,
                        "duration": tdef.anim_duration,
                        "loop": tdef.anim_loop,
                        "delay": tdef.anim_delay,
                        "ease": tdef.anim_ease,
                        "from_color": tdef.from_color,
                        "to_color": tdef.to_color,
                        "pulse_min": tdef.pulse_min,
                        "pulse_max": tdef.pulse_max,
                        "pulse_speed": tdef.pulse_speed,
                        "slide_direction": tdef.slide_direction,
                        "slide_distance": tdef.slide_distance,
                        "shake_intensity": tdef.shake_intensity,
                        "wave_amplitude": tdef.wave_amplitude,
                        "wave_speed": tdef.wave_speed,
                        "wave_char_offset": tdef.wave_char_offset,
                        "closest_rect_area": tdef.closest_rect_area,
                        "flow": tdef.flow,
                    }
                    c.update(style_overrides)
                    properties = {}
                    if tdef.input_enabled:
                        properties = {
                            "input": True,
                            "take_input": list(tdef.input_submit_keys),
                            "type_input": tdef.input_type,
                            "return": tdef.input_return_spec,
                            "placeholder": tdef.input_placeholder,
                            "min_length": tdef.input_min_length,
                            "max_length": tdef.input_max_length,
                            "allow_empty": tdef.input_allow_empty,
                            "transform": tdef.input_transform,
                            "pattern": tdef.input_pattern,
                            "allowed_chars": (
                                None if tdef.input_allowed_chars is None else "".join(sorted(tdef.input_allowed_chars))
                            ),
                            "live_update": tdef.input_live_update,
                            "clear_on_submit": tdef.input_clear_on_submit,
                            "caret": tdef.input_caret,
                            "caret_blink": tdef.input_caret_blink,
                            "caret_color": tdef.input_caret_color,
                            "caret_width": tdef.input_caret_width,
                            "caret_height": tdef.input_caret_height_ratio,
                            "caret_blink_interval": tdef.input_caret_blink_interval,
                        }
                    canvas.text_items[i] = self._parse(
                        txt_val,
                        c,
                        ip=ip,
                        source=tdef.source,
                        properties=properties,
                        layout=tdef.layout,
                        cell=tdef.cell,
                    )
                    canvas.update()
                    return True
        return False

    def remove_by_ip(self, tag: str, ip: str) -> int:
        """Remove all text items with the given ip. Returns count removed."""
        win = _window_registry.get(tag)
        if not hasattr(win, "_draw_canvas"):
            return 0
        canvas = win._draw_canvas
        before = len(canvas.text_items)
        canvas.text_items = [t for t in canvas.text_items if t.ip != ip]
        removed = before - len(canvas.text_items)
        if removed > 0:
            canvas.update()
        return removed


# ── singleton ─────────────────────────────────────────────────────────────────
text = _TextRegistry()


# ══════════════════════════════════════════════════════════════════════════
# Draw.lineedit / Draw.textedit  — native Qt text-entry widgets
# ══════════════════════════════════════════════════════════════════════════
#
# Draw.text(..., properties={"take_input": True}) simulates typing by hand
# -painting a caret onto the canvas (see _handle_input_key_press above) —
# it never touches a real QLineEdit/QTextEdit, so it has no native IME,
# undo/redo, OS copy-paste shortcuts, or accessibility support.
#
# Draw.lineedit / Draw.textedit instead embed Qt's actual QLineEdit /
# QTextEdit as native child widgets stacked on top of the canvas (same
# parenting trick Draw.panel uses for its content surface). You get all of
# Qt's native text-editing behaviour for free, at the cost of the widget no
# longer being paintable/motion-animatable the way a TextDef is.
#
#     Draw.lineedit(ip="name", display="main", x=20, y=20, width=200,
#                   placeholder="Enter name", on_change=fn, on_submit=fn)
#     Draw.textedit(ip="notes", display="main", x=20, y=60, width=300,
#                   height=150, placeholder="Notes...", on_change=fn)
#
#     Draw.lineedit.get_text(ip) -> str
#     Draw.lineedit.set_text(ip, "hello")
#     Draw.textedit.append_text(ip, "new line")
#     Draw.lineedit.list() / .move(ip, x=, y=) / .resize(ip, width=, height=)
#     Draw.lineedit.close(ip)
# ══════════════════════════════════════════════════════════════════════════

# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QLineEdit, QTextEdit


def _qss_from_edit_style(style: Optional[dict]) -> str:
    """Build a minimal Qt stylesheet string from a Draw-style dict."""
    if not style:
        return ""
    bg     = style.get("background_color")
    fg     = style.get("text_color") or style.get("color")
    border = style.get("border_color")
    bw     = style.get("border_width", 1 if border else 0)
    br     = style.get("border_radius", 0)
    fsize  = style.get("font_size")

    rules = []
    if bg:
        rules.append(f"background-color: {bg};")
    if fg:
        rules.append(f"color: {fg};")
    if border:
        rules.append(f"border: {bw}px solid {border};")
    if br:
        rules.append(f"border-radius: {br}px;")
    if fsize:
        rules.append(f"font-size: {fsize}px;")
    return " ".join(rules)


def _resolve_edit_window(display: Optional[str], label: str) -> str:
    if display is not None:
        return display
    tags = _window_registry.list_tags()
    if len(tags) == 1:
        return tags[0]
    if len(tags) > 1:
        raise ValueError(f"{label}: multiple windows — 'display' is required.")
    raise ValueError(f"{label}: no windows exist. Call Draw.window() first.")


# ── Draw.lineedit ─────────────────────────────────────────────────────────────

@dataclass
class LineEditDef:
    ip: str
    display: str
    x: int
    y: int
    width: int
    height: int
    placeholder: str
    read_only: bool
    password: bool
    style: dict
    on_change: Optional[Any] = None
    on_submit: Optional[Any] = None
    _widget: Optional["QLineEdit"] = None


class _LineEditRegistry:
    """Public API: Draw.lineedit(ip="...", display="main", ...) → native QLineEdit."""

    def __init__(self):
        self._items: dict = {}

    def __call__(
        self,
        *,
        ip: str,
        display: Optional[str] = None,
        x: int = 20,
        y: int = 20,
        width: int = 200,
        height: int = 32,
        placeholder: str = "",
        text: str = "",
        read_only: bool = False,
        password: bool = False,
        max_length: Optional[int] = None,
        style: Optional[dict] = None,
        on_change: Optional[Any] = None,
        on_submit: Optional[Any] = None,
    ) -> LineEditDef:
        if not ip or not isinstance(ip, str):
            raise ValueError("Draw.lineedit: 'ip' is required.")
        if ip in self._items:
            return self._items[ip]

        window_tag = _resolve_edit_window(display, "Draw.lineedit")
        win = _window_registry.get(window_tag)
        canvas = _get_or_create_canvas(window_tag, win)

        widget = QLineEdit(canvas)
        widget.setGeometry(x, y, width, height)
        widget.setPlaceholderText(placeholder)
        if text:
            widget.setText(text)
        widget.setReadOnly(read_only)
        if password:
            widget.setEchoMode(QLineEdit.EchoMode.Password)
        if max_length:
            widget.setMaxLength(int(max_length))
        qss = _qss_from_edit_style(style)
        if qss:
            widget.setStyleSheet("QLineEdit { " + qss + " }")

        ldef = LineEditDef(
            ip=ip, display=window_tag, x=x, y=y, width=width, height=height,
            placeholder=placeholder, read_only=read_only, password=password,
            style=dict(style or {}), on_change=on_change, on_submit=on_submit,
        )
        ldef._widget = widget

        if on_change is not None:
            widget.textChanged.connect(lambda s: on_change(s))
        if on_submit is not None:
            widget.returnPressed.connect(lambda: on_submit(widget.text()))

        widget.show()
        widget.raise_()
        self._items[ip] = ldef
        return ldef

    def get_text(self, ip: str) -> str:
        item = self._items.get(ip)
        return item._widget.text() if item and item._widget else ""

    def set_text(self, ip: str, value: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.setText(str(value))

    def clear(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.clear()

    def set_focus(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.setFocus()

    def select_all(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.selectAll()

    def get(self, ip: str) -> Optional[LineEditDef]:
        return self._items.get(ip)

    def list(self) -> List[str]:
        return list(self._items.keys())

    def move(self, ip: str, *, x: int, y: int) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item.x, item.y = x, y
            item._widget.move(x, y)

    def resize(self, ip: str, *, width: int, height: int) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item.width, item.height = width, height
            item._widget.resize(width, height)

    def show(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.show()

    def hide(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.hide()

    def close(self, ip: str) -> None:
        item = self._items.pop(ip, None)
        if item and item._widget:
            item._widget.deleteLater()


lineedit = _LineEditRegistry()


# ── Draw.textedit ─────────────────────────────────────────────────────────────

@dataclass
class TextEditDef:
    ip: str
    display: str
    x: int
    y: int
    width: int
    height: int
    placeholder: str
    read_only: bool
    wrap: bool
    style: dict
    on_change: Optional[Any] = None
    _widget: Optional["QTextEdit"] = None


class _TextEditRegistry:
    """Public API: Draw.textedit(ip="...", display="main", ...) → native QTextEdit."""

    def __init__(self):
        self._items: dict = {}

    def __call__(
        self,
        *,
        ip: str,
        display: Optional[str] = None,
        x: int = 20,
        y: int = 20,
        width: int = 300,
        height: int = 150,
        placeholder: str = "",
        text: str = "",
        read_only: bool = False,
        wrap: bool = True,
        style: Optional[dict] = None,
        on_change: Optional[Any] = None,
    ) -> TextEditDef:
        if not ip or not isinstance(ip, str):
            raise ValueError("Draw.textedit: 'ip' is required.")
        if ip in self._items:
            return self._items[ip]

        window_tag = _resolve_edit_window(display, "Draw.textedit")
        win = _window_registry.get(window_tag)
        canvas = _get_or_create_canvas(window_tag, win)

        widget = QTextEdit(canvas)
        widget.setGeometry(x, y, width, height)
        widget.setPlaceholderText(placeholder)
        if text:
            widget.setPlainText(text)
        widget.setReadOnly(read_only)
        widget.setLineWrapMode(
            QTextEdit.LineWrapMode.WidgetWidth if wrap else QTextEdit.LineWrapMode.NoWrap
        )
        qss = _qss_from_edit_style(style)
        if qss:
            widget.setStyleSheet("QTextEdit { " + qss + " }")

        tdef = TextEditDef(
            ip=ip, display=window_tag, x=x, y=y, width=width, height=height,
            placeholder=placeholder, read_only=read_only, wrap=wrap,
            style=dict(style or {}), on_change=on_change,
        )
        tdef._widget = widget

        if on_change is not None:
            widget.textChanged.connect(lambda: on_change(widget.toPlainText()))

        widget.show()
        widget.raise_()
        self._items[ip] = tdef
        return tdef

    def get_text(self, ip: str) -> str:
        item = self._items.get(ip)
        return item._widget.toPlainText() if item and item._widget else ""

    def set_text(self, ip: str, value: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.setPlainText(str(value))

    def append_text(self, ip: str, value: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.append(str(value))

    def clear(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.clear()

    def set_focus(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.setFocus()

    def select_all(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.selectAll()

    def get(self, ip: str) -> Optional[TextEditDef]:
        return self._items.get(ip)

    def list(self) -> List[str]:
        return list(self._items.keys())

    def move(self, ip: str, *, x: int, y: int) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item.x, item.y = x, y
            item._widget.move(x, y)

    def resize(self, ip: str, *, width: int, height: int) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item.width, item.height = width, height
            item._widget.resize(width, height)

    def show(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.show()

    def hide(self, ip: str) -> None:
        item = self._items.get(ip)
        if item and item._widget:
            item._widget.hide()

    def close(self, ip: str) -> None:
        item = self._items.pop(ip, None)
        if item and item._widget:
            item._widget.deleteLater()


textedit = _TextEditRegistry()
