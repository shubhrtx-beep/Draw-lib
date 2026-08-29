"""
Draw._shapes  v3
================
Redesigned shape system — geometry-first.

PUBLIC API
----------
    Draw.shape(
        display  = "window_tag",

        connections = [{            # future: connection specs (not processed yet)
            "ip"     : "",
            "get_ip" : "",
            "colour" : "",
            "set"    : "",
        }],

        shape    = [
            {
                # ── Base geometry ─────────────────────────────────────────
                "vertices"     : 6,            # sides (3+); None or 4 = rect
                "size"         : [200, 150],   # [width, height]  or "200px"/"50%"
                "align"        : "center",     # canvas-level position
                "x"            : None,         # absolute pixel x
                "y"            : None,         # absolute pixel y
                "border_radius": 10,           # px or "50%" for circle
                "rotation"     : 0,            # degrees clockwise

                # ── Appearance ────────────────────────────────────────────
                "color"        : "cyan",
                "border_color" : "white",
                "border_width" : 2,
                "border_style" : "solid",      # solid | dashed | dotted
                "opacity"      : 100,          # 0-100
                "z"            : 0,            # layer (higher = further back)
                "overlap"      : True,

                # ── Hitbox ────────────────────────────────────────────────
                "hitbox_mode"  : None,         # None | "shape" | "closed_rec"
                "hit_box"      : "shape",      # "shape" or "closed_rec"


                # ── Edge style (applies to all sides uniformly) ───────────
                # "line"   straight edges (default)
                # "smooth" Catmull-Rom spline through vertices
                # "slope"  slight outward quadratic bulge on every edge
                # "wave"   sinusoidal displacement along every edge
                # "arc"    circular arc on every edge (like border-radius)
                "curve_mode"   : "line",

                # ── Custom dict (alternative nesting) ─────────────────────
                # bend, exclude, symmetry, properties can live here instead
                # of at top-level. Top-level keys take priority.
                "custom"       : {
                    "bend" : [
                        {
                            "side"      : 1,
                            "affect"    : "-100% -> 100%",
                            "offset"    : 30,
                            "smooth"    : 60,
                            "direction" : "out",
                        }
                    ],
                    "exclude" : [
                        {
                            "scale"  : [100, 100],
                            "let_it" : "center",
                            "1"      : {
                                "line"          : "-30,-30  30,-30  30,30  -30,30",
                                "edges"         : "8,8,8,8",
                                "width_between" : "0px",
                                "curve_mode"    : "line",
                            },
                        }
                    ],
                    "symmetry" : {
                        "type"  : "radial",
                        "count" : 4,
                    },
                    "properties" : {
                        "mirror" : "20",
                    },
                },

                # ── Per-side deformation (also available inside custom) ───
                "bend"         : [],
                "exclude"      : [],
                "symmetry"     : None,

                # ── Identity ──────────────────────────────────────────────
                "ip"           : "my-shape",
            }
        ],

        text     = [
            {
                # Text items now support shape-like positioning properties
                "text"         : "Hello",
                "vertices"     : None,
                "size"         : [100, 20],
                "x"            : None,
                "y"            : None,
                "align"        : None,
                "rotation"     : 0,
                "hitbox_mode"  : None,
                "hit_box"      : "shape",
                # Per-letter styling
                "custom"       : {
                    "letter" : "rotate='', scale='' etc.",
                },
            }
        ],
    )
"""

from __future__ import annotations

import math
import logging
import re
import time
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
# pyrefly: ignore [missing-import]
from PySide6.QtGui import (
    QBrush, QColor, QConicalGradient, QLinearGradient, QPainter, QPainterPath,
    QFont, QFontMetricsF, QKeyEvent, QMouseEvent, QPainterPathStroker, QPen, QRadialGradient, QTransform, QWheelEvent,
)
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QMainWindow, QWidget

from Draw._app import get_app
from Draw._window import window as _window_registry, _parse_color, _ALIGN_VALUES

_logger = logging.getLogger(__name__)


# ── constants ─────────────────────────────────────────────────────────────────

_CURVE_MODES = {"line", "smooth", "slope", "wave", "arc", "bend_all"}
_ALIGN_VALUES_SET = set(_ALIGN_VALUES)

# ── In-memory Image Cache (prevent per-frame disk thrashing) ──────────────────
_PIXMAP_CACHE: Dict[str, Tuple[float, int, Any]] = {}
_MAX_PIXMAP_CACHE_ENTRIES: int = 256


def _get_cached_pixmap(src: object) -> Optional[Any]:
    """
    Retrieve or load a QPixmap with memory caching and mtime invalidation.
    Prevents reading from disk on every animation/render frame (60 FPS disk thrashing).
    """
    if not src:
        return None
    import os
    from PySide6.QtGui import QPixmap

    path_str = os.fspath(str(src))
    try:
        if os.path.exists(path_str):
            stat = os.stat(path_str)
            mtime = stat.st_mtime
            size = stat.st_size
            if path_str in _PIXMAP_CACHE:
                cached_mtime, cached_size, pm = _PIXMAP_CACHE[path_str]
                if cached_mtime == mtime and cached_size == size:
                    return pm
            pm = QPixmap(path_str)
            if not pm.isNull():
                if len(_PIXMAP_CACHE) >= _MAX_PIXMAP_CACHE_ENTRIES:
                    _PIXMAP_CACHE.pop(next(iter(_PIXMAP_CACHE)), None)
                _PIXMAP_CACHE[path_str] = (mtime, size, pm)
                return pm
            return None
    except Exception:
        pass

    # Fallback for Qt resource aliases / non-filesystem paths
    if path_str in _PIXMAP_CACHE:
        return _PIXMAP_CACHE[path_str][2]
    pm = QPixmap(path_str)
    if not pm.isNull():
        if len(_PIXMAP_CACHE) >= _MAX_PIXMAP_CACHE_ENTRIES:
            _PIXMAP_CACHE.pop(next(iter(_PIXMAP_CACHE)), None)
        _PIXMAP_CACHE[path_str] = (0.0, 0, pm)
        return pm
    return None


# ── small parse helpers ───────────────────────────────────────────────────────

def _as_int(v: object, default: int = 0) -> int:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(v: object, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"false", "0", "no", "off", ""}:
        return False
    return bool(v)


def _merge_customise(raw: dict) -> dict:
    """Merge legacy 'customise' values into a shape dict; top-level wins."""
    customise = raw.get("customise", None)
    if not isinstance(customise, dict):
        return raw
    merged = dict(customise)
    for key, value in raw.items():
        if key != "customise":
            merged[key] = value
    return merged


def _parse_size(value: object, parent_px: int, default_pct: float = 0.5) -> int:
    if value is None:
        return max(1, int(parent_px * default_pct))
    if isinstance(value, (int, float)):
        return max(1, int(value))
    if isinstance(value, str):
        v = value.strip()
        if v.endswith("%"):
            return max(1, int(float(v[:-1]) / 100.0 * parent_px))
        if v.endswith("px"):
            return max(1, int(float(v[:-2])))
        try:
            return max(1, int(float(v)))
        except ValueError:
            return max(1, int(parent_px * default_pct))
    return max(1, int(parent_px * default_pct))


def _parse_px_value(raw: object, default: float = 0.0) -> float:
    """Parse "20px", "20", 20, etc. to float."""
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    s = s.replace("px", "").strip()
    try:
        return float(s)
    except ValueError:
        return default


def _parse_border_radius(value: object, w: int, h: int) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if v.endswith("%"):
            return float(v[:-1]) / 100.0 * min(w, h) / 2
        return _parse_px_value(v)
    return 0.0


def _clamp01(t: float) -> float:
    return max(0.0, min(1.0, t))


def _lerp_pt(a: QPointF, b: QPointF, t: float) -> QPointF:
    return QPointF(a.x() + (b.x() - a.x()) * t, a.y() + (b.y() - a.y()) * t)


# ── affect range parser ───────────────────────────────────────────────────────

def _parse_affect_range(raw: object) -> Tuple[float, float]:
    """
    Parse the affect string to (start_t, end_t) in 0..1 space.

    Input coordinate system: -100% = start of side, 0% = midpoint, +100% = end.
    Examples:
        "-100% -> 100%"  →  (0.0, 1.0)   full side
        "-20% -> 40%"    →  (0.4, 0.7)
        "0% -> 100%"     →  (0.5, 1.0)   second half
    """
    raw_str = str(raw).strip() if raw is not None else "-100% -> 100%"
    nums = re.findall(r'-?\d+(?:\.\d+)?', raw_str)
    if len(nums) >= 2:
        a = float(nums[0])
        b = float(nums[1])
        # -100..100 → 0..1
        t_start = (a + 100.0) / 200.0
        t_end   = (b + 100.0) / 200.0
    else:
        t_start, t_end = 0.0, 1.0

    if t_start > t_end:
        t_start, t_end = t_end, t_start
    return _clamp01(t_start), _clamp01(t_end)


# ── coordinate string parser ──────────────────────────────────────────────────

def _parse_coord_string(raw: str) -> List[Tuple[float, float]]:
    """Parse "x,y  x,y  x,y" or "x,y;x,y" into [(x,y),...]."""
    pts: List[Tuple[float, float]] = []
    for token in re.split(r'[;\n]+', raw):
        token = token.strip()
        if not token:
            continue
        parts = re.split(r'[\s,]+', token)
        # group pairs
        i = 0
        while i + 1 < len(parts):
            try:
                pts.append((float(parts[i]), float(parts[i + 1])))
                i += 2
            except ValueError:
                i += 1
    return pts


# ── ShapeDef ──────────────────────────────────────────────────────────────────

class _DrawCanvas(QWidget):
    """
    One transparent QWidget per window.
    Holds ShapeDef, TextDef objects and repaints all of them.
    Also dispatches mouse / keyboard events into Draw.senses so that
    Draw.connectors can react to user input.
    """

    def __init__(self, parent: QMainWindow):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # receive keyboard events
        self.setMouseTracking(True)                       # receive hover events
        self.layout_items: list = []
        self.shape_items: list = []   # ShapeDef objects (drawn first)
        self.text_items:  List[TextDef] = []
        self._z_order_dirty: bool = True
        self._sorted_shapes: list = []
        self.setGeometry(parent.rect())
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._tick_animation)
        self._animation_timer.start()
        # Track which ips the cursor is currently hovering over
        self._hovered_ips: set = set()
        self._active_input_index: int = -1
        self._live_text_error_cache: dict[int, str] = {}
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._scroll_x = 0.0
        self._scroll_y = 0.0
        self._dragged_shape = None
        self._drag_offset = (0.0, 0.0)
        self._last_tick_time = time.perf_counter()
        # Persistent global occupied registry — maps shape id() → placed Rect
        # Rebuilt whenever shapes are added/removed (_occupied_dirty = True)
        self._global_occupied: list = []   # list of Rect (area rects of all placed shapes)
        self._occupied_dirty: bool = True  # force rebuild on first paint
        # ── Incremental rendering registry ────────────────────────────────────
        # Maps ip → ShapeDef for O(1) lookup when re-submitting shapes.
        # Cleared by _ShapeRegistry.clear() and remove_by_ip().
        self._shape_by_ip: dict = {}
        # Maps ip → int content hash so that visually identical re-submissions
        # cost a single hash() call and zero allocation.
        self._shape_hash_by_ip: dict = {}
        self._scroller_configs: list = []  # set by _ShapeRegistry when scrollers are built
        # -- Tier 3/4 interactivity state (drag / long-press / touch / focus) --
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self._drag_origin = None          # (x, y) where the current drag began
        self._drag_started = False        # True once movement passed the threshold
        self._drag_threshold_px = 4.0
        self.longpress_delay_ms = 500
        self.longpress_repeat_ms = 80
        self._longpress_timer = QTimer(self)
        self._longpress_timer.timeout.connect(self._on_longpress_timeout)
        self._longpress_ip = None
        self._longpress_fired_once = False

        # ── raw click tracking (independent of hitting an existing shape) ──
        # Used by Draw.senses.first_click / .last_release / .region /
        # .capture_region to answer "where did the user start/end a
        # left-click drag on this canvas", regardless of whether anything
        # was under the cursor at the time.
        self._last_lclick_press_pos = None    # (x, y) or None
        self._last_lclick_release_pos = None  # (x, y) or None

        # ── Draw.shape(..., properties=["builder"]) drag-to-size queue ──
        # Shapes submitted without an explicit size are queued here instead
        # of being placed immediately. The next left-click drag on this
        # canvas consumes one queued entry: press = first corner, live drag
        # = growing preview, release = finalize at an ABSOLUTE pixel size.
        self._builder_queue: list = []
        self._builder_active: Optional[dict] = None
        self._focused_ip = None           # currently keyboard-focused shape/text ip

    def _has_active_shape_animation(self) -> bool:
        from Draw._shapes import _shape_is_dynamic
        return any(_shape_is_dynamic(shape) or getattr(shape, "motion", None) for shape in self.shape_items)

    def _has_live_text_source(self) -> bool:
        from Draw._text import LiveTextBinding
        return any(isinstance(t.source, LiveTextBinding) for t in self.text_items)

    def _compute_content_bounds(self) -> Tuple[float, float]:
        """Compute the maximum (content_width, content_height) on canvas."""
        cw = float(self.width()) if self.width() > 0 else 1000.0
        ch = float(self.height()) if self.height() > 0 else 800.0
        max_w = cw
        max_h = ch
        for s in self.shape_items:
            if s.ip and (s.ip.startswith("scroller_") or any(s.ip in (c.get("thumb_ip"), c.get("track_ip")) for c in getattr(self, "_scroller_configs", []))):
                continue
            sx = getattr(s, "_placed_x", getattr(s, "x", 0) or 0)
            sy = getattr(s, "_placed_y", getattr(s, "y", 0) or 0)
            sw = getattr(s, "_placed_w", getattr(s, "width", 0) or 0)
            sh = getattr(s, "_placed_h", getattr(s, "height", 0) or 0)
            if s.last_size:
                sw, sh = s.last_size
            max_w = max(max_w, float(sx) + float(sw) + 40.0)
            max_h = max(max_h, float(sy) + float(sh) + 40.0)
        for t in self.text_items:
            if t.ip and (t.ip.startswith("scroller_") or any(t.ip in (c.get("thumb_ip"), c.get("track_ip")) for c in getattr(self, "_scroller_configs", []))):
                continue
            tx = float(t.x or 0)
            ty = float(t.y or 0)
            tw = t.last_rect[2] if t.last_rect else 100.0
            th = t.last_rect[3] if t.last_rect else 30.0
            max_w = max(max_w, tx + tw + 40.0)
            max_h = max(max_h, ty + th + 40.0)
        return max_w, max_h

    def _get_max_scroll_range(self) -> Tuple[float, float]:
        """Return (max_scroll_x, max_scroll_y) bounds."""
        cw, ch = float(self.width()) if self.width() > 0 else 1000.0, float(self.height()) if self.height() > 0 else 800.0
        cont_w, cont_h = self._compute_content_bounds()
        default_max_x = max(0.0, cont_w - cw)
        default_max_y = max(0.0, cont_h - ch)
        max_x = default_max_x
        max_y = default_max_y
        for cfg in getattr(self, "_scroller_configs", []):
            if cfg.get("max_x") is not None:
                max_x = float(cfg["max_x"])
            if cfg.get("max_y") is not None:
                max_y = float(cfg["max_y"])
        return max(0.0, max_x), max(0.0, max_y)

    def _update_scroller_thumbs(self) -> None:
        """Reposition scroller thumbs to match current scroll offsets accurately."""
        cw, ch = float(self.width()) if self.width() > 0 else 1000.0, float(self.height()) if self.height() > 0 else 800.0
        max_scroll_x, max_scroll_y = self._get_max_scroll_range()

        # Clamp canvas scroll offsets within valid bounds
        if max_scroll_y > 0.0:
            self._scroll_y = max(0.0, min(max_scroll_y, self._scroll_y))
        else:
            self._scroll_y = 0.0

        if max_scroll_x > 0.0:
            self._scroll_x = max(0.0, min(max_scroll_x, self._scroll_x))
        else:
            self._scroll_x = 0.0

        if not self._scroller_configs:
            return
        ip_to_shape = {s.ip: s for s in self.shape_items if s.ip}
        changed = False

        for cfg in self._scroller_configs:
            thumb = ip_to_shape.get(cfg["thumb_ip"])
            if thumb is None:
                continue
            if cfg["direction"] == "vertical":
                track_h = float(cfg["track_h"])
                total_h = max_scroll_y + ch
                thumb_ratio = max(0.05, min(1.0, ch / max(1.0, total_h))) if total_h > 0 else 0.2
                thumb_h = max(24.0, min(track_h, track_h * thumb_ratio))
                cfg["thumb_h"] = thumb_h
                thumb.height = int(thumb_h)
                thumb.size_raw = [float(cfg["track_w"]), float(thumb_h)]
                thumb.last_size = (float(cfg["track_w"]), float(thumb_h))

                travel = max(1.0, track_h - thumb_h)
                scroll_range = float(cfg.get("max_y")) if cfg.get("max_y") is not None else max(1.0, max_scroll_y)
                t = max(0.0, min(1.0, self._scroll_y / max(1.0, scroll_range)))
                new_y = cfg["track_y"] + t * travel
                thumb.y = int(new_y)
                thumb._placed_y = float(new_y)
                thumb.last_position = (float(cfg["track_x"]), float(new_y))
            else:
                track_w = float(cfg["track_w"])
                total_w = max_scroll_x + cw
                thumb_ratio = max(0.05, min(1.0, cw / max(1.0, total_w))) if total_w > 0 else 0.2
                thumb_w = max(24.0, min(track_w, track_w * thumb_ratio))
                cfg["thumb_w"] = thumb_w
                thumb.width = int(thumb_w)
                thumb.size_raw = [float(thumb_w), float(cfg["track_h"])]
                thumb.last_size = (float(thumb_w), float(cfg["track_h"]))

                travel = max(1.0, track_w - thumb_w)
                scroll_range = float(cfg.get("max_x")) if cfg.get("max_x") is not None else max(1.0, max_scroll_x)
                t = max(0.0, min(1.0, self._scroll_x / max(1.0, scroll_range)))
                new_x = cfg["track_x"] + t * travel
                thumb.x = int(new_x)
                thumb._placed_x = float(new_x)
                thumb.last_position = (float(new_x), float(cfg["track_y"]))
            changed = True
        if changed:
            self.update()

    def _refresh_live_text_bindings(self) -> bool:
        from Draw._text import LiveTextBinding, resolve_live_text
        changed = False
        for t in self.text_items:
            if not isinstance(t.source, LiveTextBinding):
                continue
            key = id(t)
            try:
                value = resolve_live_text(t.source)
                self._live_text_error_cache.pop(key, None)
            except Exception as exc:
                value = ""
                msg = str(exc)
                if self._live_text_error_cache.get(key) != msg:
                    _logger.warning("Draw.live.text: source error: %s", exc)
                    self._live_text_error_cache[key] = msg
            if value != t.text:
                t.text = value
                changed = True
        return changed

    def _tick_animation(self) -> None:
        from Draw._motion import motion as _motion_registry
        from Draw._text import _text_is_animated
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if dt < 0.0:
            dt = 0.016

        for s in self.shape_items:
            _motion_registry.tick_shape_triggers(s, dt)

        timeline_changed = _motion_registry.tick_timelines(now)

        shape_animating = self._has_active_shape_animation()
        live_changed = self._refresh_live_text_bindings() if self._has_live_text_source() else False
        custom_changed = _motion_registry.tick_custom(now)
        caret_animating = any(
            t.input_enabled and t.input_selected and t.input_caret and t.input_caret_blink
            for t in self.text_items
        )
        text_animating = any(_text_is_animated(t) for t in self.text_items)
        self._update_scroller_thumbs()
        if shape_animating or live_changed or custom_changed or caret_animating or text_animating or timeline_changed:
            self.update()

    def _input_targets(self) -> list[TextDef]:
        return [t for t in self.text_items if t.input_enabled]

    def _active_input_target(self) -> Optional[TextDef]:
        targets = self._input_targets()
        if not targets:
            self._active_input_index = -1
            self._sync_input_selection()
            return None
        if self._active_input_index < 0:
            self._sync_input_selection()
            return None
        if self._active_input_index >= len(targets):
            self._active_input_index = len(targets) - 1
        self._sync_input_selection()
        return targets[self._active_input_index]

    def _cycle_active_input_target(self) -> None:
        targets = self._input_targets()
        if not targets:
            self._active_input_index = -1
            self._sync_input_selection()
            return
        if self._active_input_index < 0:
            self._active_input_index = 0
            self._sync_input_selection()
            return
        self._active_input_index = (self._active_input_index + 1) % len(targets)
        self._sync_input_selection()

    def _sync_input_selection(self) -> None:
        targets = self._input_targets()

        previously_selected_ip = None
        for t in self.text_items:
            if t.input_enabled and t.input_selected:
                previously_selected_ip = t.ip
                break

        already_selected = False
        if 0 <= self._active_input_index < len(targets):
            already_selected = targets[self._active_input_index].input_selected

        for target in targets:
            target.input_selected = False
        if 0 <= self._active_input_index < len(targets):
            t = targets[self._active_input_index]
            t.input_selected = True
            if not already_selected:
                t.input_cursor_position = len(t.input_buffer)
        elif self._active_input_index != -1:
            self._active_input_index = -1

        # Focus state sense dispatches
        currently_selected_ip = None
        for t in self.text_items:
            if t.input_enabled and t.input_selected:
                currently_selected_ip = t.ip
                break

        if previously_selected_ip != currently_selected_ip:
            from Draw._connectors import senses as _senses
            if previously_selected_ip is not None:
                for record in _senses._ip_index.get(previously_selected_ip, []):
                    if record.sense_type == "focus_out":
                        record.trigger()
            if currently_selected_ip is not None:
                for record in _senses._ip_index.get(currently_selected_ip, []):
                    if record.sense_type == "focus_in":
                        record.trigger()

    def _clear_active_input_target(self) -> None:
        target = self._active_input_target()
        if target is not None:
            self._commit_input_target(target)

        had_active = self._active_input_index != -1 or any(
            t.input_enabled and t.input_selected for t in self.text_items
        )
        self._active_input_index = -1
        self._sync_input_selection()
        if had_active:
            self.update()

    @staticmethod
    def _text_contains_point(target: TextDef, pos: QPointF) -> bool:
        if target.last_rect is None:
            return False
        x, y, w, h = target.last_rect
        return x <= pos.x() <= x + w and y <= pos.y() <= y + h

    def _select_input_at_point(self, pos: QPointF, hit_ips: set[str]) -> bool:
        targets = self._input_targets()
        for index in range(len(targets) - 1, -1, -1):
            target = targets[index]
            ip_hit = target.ip is not None and target.ip in hit_ips
            if ip_hit or self._text_contains_point(target, pos):
                self._active_input_index = index
                self._sync_input_selection()
                self.setFocus()
                self.update()
                return True
        self._clear_active_input_target()
        return False

    @staticmethod
    def _event_submit_key(event: QKeyEvent, key_name: str) -> Optional[str]:
        from Draw._text import _normalize_submit_key
        qt_key = event.key()
        if qt_key in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            return "return"
        if qt_key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            return "tab"
        if qt_key == Qt.Key.Key_Escape:
            return "escape"
        normalized = _normalize_submit_key(key_name)
        if normalized in {"return", "tab", "escape"}:
            return normalized
        return None

    @staticmethod
    def _emit_input_live_value(target: TextDef, value: str) -> None:
        from Draw._text import _input_registry, _store_input_return_value
        if target.ip is not None:
            _input_registry.set(target.ip, value)
        _store_input_return_value(target.input_return_spec, value)

    def _commit_input_target(self, target: TextDef) -> bool:
        from Draw._text import (
            _input_registry,
            _is_input_final_allowed,
            _store_input_return_value,
        )
        value = target.input_buffer
        if not _is_input_final_allowed(
            value,
            target.input_type,
            min_length=target.input_min_length,
            max_length=target.input_max_length,
            allow_empty=target.input_allow_empty,
            pattern=target.input_pattern,
            allowed_chars=target.input_allowed_chars,
        ):
            return False

        target.text = value
        if target.ip is not None:
            _input_registry.set(target.ip, value)
        _store_input_return_value(target.input_return_spec, value)
        if target.input_clear_on_submit:
            target.input_buffer = ""
            target.text = ""
            target.input_cursor_position = 0
        self.update()
        return True

    def _handle_input_key_press(self, event: QKeyEvent, key_name: str) -> bool:
        from Draw._text import _apply_input_transform, _is_input_candidate_allowed
        target = self._active_input_target()
        if target is None:
            return False

        qt_key = event.key()
        submit_key = self._event_submit_key(event, key_name)
        submit_keys = set(target.input_submit_keys)

        typed = ""
        if submit_key is not None:
            if submit_key in submit_keys:
                return self._commit_input_target(target)
            if submit_key == "tab":
                self._cycle_active_input_target()
                self.update()
                return True
            if submit_key == "return":
                typed = "\n"
            else:
                return False

        cursor_pos = getattr(target, "input_cursor_position", len(target.input_buffer))
        cursor_pos = max(0, min(len(target.input_buffer), cursor_pos))

        if qt_key == Qt.Key.Key_Backspace:
            if cursor_pos > 0:
                target.input_buffer = target.input_buffer[:cursor_pos - 1] + target.input_buffer[cursor_pos:]
                target.input_cursor_position = cursor_pos - 1
                target.text = target.input_buffer
                if target.input_live_update:
                    self._emit_input_live_value(target, target.input_buffer)
                self.update()
            return True

        if qt_key == Qt.Key.Key_Delete:
            if cursor_pos < len(target.input_buffer):
                target.input_buffer = target.input_buffer[:cursor_pos] + target.input_buffer[cursor_pos + 1:]
                target.text = target.input_buffer
                if target.input_live_update:
                    self._emit_input_live_value(target, target.input_buffer)
                self.update()
            return True

        if qt_key == Qt.Key.Key_Left:
            if cursor_pos > 0:
                target.input_cursor_position = cursor_pos - 1
                self.update()
            return True

        if qt_key == Qt.Key.Key_Right:
            if cursor_pos < len(target.input_buffer):
                target.input_cursor_position = cursor_pos + 1
                self.update()
            return True

        if qt_key == Qt.Key.Key_Home:
            target.input_cursor_position = 0
            self.update()
            return True

        if qt_key == Qt.Key.Key_End:
            target.input_cursor_position = len(target.input_buffer)
            self.update()
            return True

        if typed == "":
            typed = event.text()
        if typed == "":
            return False
        if any(not (ch.isprintable() or ch == "\n") for ch in typed):
            return False
        if typed == " " and "space" in submit_keys:
            return self._commit_input_target(target)

        candidate = target.input_buffer[:cursor_pos] + typed + target.input_buffer[cursor_pos:]
        candidate = _apply_input_transform(candidate, target.input_transform)
        if not _is_input_candidate_allowed(
            candidate,
            target.input_type,
            max_length=target.input_max_length,
            allowed_chars=target.input_allowed_chars,
        ):
            return False

        target.input_buffer = candidate
        target.input_cursor_position = cursor_pos + len(typed)
        target.text = target.input_buffer
        if target.input_live_update:
            self._emit_input_live_value(target, target.input_buffer)
        self.update()
        return True

    # ── helper: build {ip -> ShapeDef} mapping ──────────────────────────────
    def _ip_shape_map(self) -> dict:
        """Return a dict mapping registered ip strings to their ShapeDef."""
        result = {}
        for shape in self.shape_items:
            if shape.ip is not None:
                result[shape.ip] = shape
        return result

    # ── helper: find which shape ips are under a given canvas point ──────────
    def _shapes_at_point(self, pos: QPointF) -> list:
        """
        Return list of (ip_str, element_def) tuples where pos is inside the shape/text.
        Uses the ip stored on each ShapeDef/TextDef for reliable matching.
        Returned in top-to-bottom draw order (frontmost visible element first).
        """
        from Draw._shapes import _shape_contains_point
        hits = []

        def _get_text_z(t, shapes):
            z_val = getattr(t, "z", 0)
            if z_val == "as_shape":
                if getattr(t, "ip", None):
                    for s in shapes:
                        if s.ip == t.ip:
                            return s.z
                return 0
            try:
                return int(z_val) if z_val is not None else 0
            except (ValueError, TypeError):
                return 0

        # Ascending order by z: smaller z values are drawn later (on top in paintEvent)
        candidates = []
        for shape in self.shape_items:
            candidates.append((shape, "shape", shape.z))
        for t in self.text_items:
            candidates.append((t, "text", _get_text_z(t, self.shape_items)))

        candidates.sort(key=lambda item: (item[2], 1 if item[1] == "text" else 0))

        for item, kind, _ in candidates:
            if kind == "shape":
                shape = item
                if shape.last_position is None or shape.ip is None:
                    continue
                if _shape_contains_point(shape, pos):
                    hits.append((shape.ip, shape))
            else:
                t = item
                if t.last_rect is None or t.ip is None:
                    continue
                tx, ty, tw, th = t.last_rect
                if tx <= pos.x() <= tx + tw and ty <= pos.y() <= ty + th:
                    hits.append((t.ip, t))

        return hits

    # ── mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_press
        handle_canvas_mouse_press(self, event)
        super().mousePressEvent(event)
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_release
        handle_canvas_mouse_release(self, event)
        super().mouseReleaseEvent(event)
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_double_click
        handle_canvas_mouse_double_click(self, event)
        super().mouseDoubleClickEvent(event)
    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_move
        handle_canvas_mouse_move(self, event)
        super().mouseMoveEvent(event)
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_wheel
        handle_canvas_wheel(self, event)
        super().wheelEvent(event)
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        from Draw._connectors import senses as _senses, _qt_key_to_name
        
        mods = event.modifiers()
        modifiers = []
        if mods & Qt.KeyboardModifier.ShiftModifier:
            modifiers.append("shift")
        if mods & Qt.KeyboardModifier.ControlModifier:
            modifiers.append("ctrl")
        if mods & Qt.KeyboardModifier.AltModifier:
            modifiers.append("alt")
        if mods & Qt.KeyboardModifier.MetaModifier:
            modifiers.append("meta")

        txt = event.text()
        if txt and txt.isprintable() and len(txt) == 1:
            key_name = txt
        else:
            key_name = _qt_key_to_name(event.key())

        if self._active_input_target() is None:
            qt_key = event.key()
            if qt_key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                self._cycle_focus(forward=(qt_key == Qt.Key.Key_Tab and "shift" not in modifiers))
            elif qt_key == Qt.Key.Key_Left:
                self._move_focus_spatial("left")
            elif qt_key == Qt.Key.Key_Right:
                self._move_focus_spatial("right")
            elif qt_key == Qt.Key.Key_Up:
                self._move_focus_spatial("up")
            elif qt_key == Qt.Key.Key_Down:
                self._move_focus_spatial("down")
            elif qt_key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and self._focused_ip:
                self._activate_focused_shape()

        self._handle_input_key_press(event, key_name)
        _senses.dispatch_key_event("key_press", key_name, modifiers)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        from Draw._connectors import senses as _senses, _qt_key_to_name
        
        mods = event.modifiers()
        modifiers = []
        if mods & Qt.KeyboardModifier.ShiftModifier:
            modifiers.append("shift")
        if mods & Qt.KeyboardModifier.ControlModifier:
            modifiers.append("ctrl")
        if mods & Qt.KeyboardModifier.AltModifier:
            modifiers.append("alt")
        if mods & Qt.KeyboardModifier.MetaModifier:
            modifiers.append("meta")

        txt = event.text()
        if txt and txt.isprintable() and len(txt) == 1:
            key_name = txt
        else:
            key_name = _qt_key_to_name(event.key())

        _senses.dispatch_key_event("key_release", key_name, modifiers)
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._clear_active_input_target()
        super().focusOutEvent(event)

    # -- long-press / hold detection (Tier 3) --------------------------------

    def _on_longpress_timeout(self) -> None:
        from Draw._connectors import handle_canvas_longpress_timeout
        handle_canvas_longpress_timeout(self)
    def contextMenuEvent(self, event) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_context_menu
        if not handle_canvas_context_menu(self, event):
            super().contextMenuEvent(event)
    def event(self, ev) -> bool:  # noqa: N802
        from Draw._connectors import handle_canvas_event
        if handle_canvas_event(self, ev):
            return True
        return super().event(ev)
    def _handle_touch_event(self, ev) -> None:
        from Draw._connectors import handle_canvas_touch_event
        handle_canvas_touch_event(self, ev)
    def _focus_rect_for_ip(self, ip: str):
        for s in self.shape_items:
            if s.ip == ip and s.last_position and s.last_size:
                x, y = s.last_position
                w, h = s.last_size
                return (x, y, w, h)
        for t in self.text_items:
            if t.ip == ip and t.last_rect:
                return t.last_rect
        return None

    def _focusable_ips(self) -> list:
        from Draw._connectors import senses as _focus_senses
        seen = set()
        result = []
        for s in self.shape_items:
            if s.ip and s.ip not in seen and _focus_senses.get_by_ip(s.ip):
                seen.add(s.ip)
                result.append(s.ip)
        for t in self.text_items:
            if t.ip and t.ip not in seen and _focus_senses.get_by_ip(t.ip):
                seen.add(t.ip)
                result.append(t.ip)

        def _sort_key(ip):
            rect = self._focus_rect_for_ip(ip)
            return (round(rect[1], 1), round(rect[0], 1)) if rect else (0.0, 0.0)

        result.sort(key=_sort_key)
        return result

    def _cycle_focus(self, forward: bool = True) -> None:
        from Draw._connectors import senses as _focus_senses
        ips = self._focusable_ips()
        if not ips:
            return
        if self._focused_ip in ips:
            idx = ips.index(self._focused_ip)
            new_idx = (idx + (1 if forward else -1)) % len(ips)
        else:
            new_idx = 0 if forward else len(ips) - 1
        old_ip = self._focused_ip
        new_ip = ips[new_idx]
        if old_ip:
            _focus_senses.dispatch_mouse_event("focus_out", old_ip, None)
        self._focused_ip = new_ip
        _focus_senses.dispatch_mouse_event("focus_in", new_ip, None)
        self.update()

    def _move_focus_spatial(self, direction: str) -> None:
        from Draw._connectors import senses as _focus_senses
        ips = self._focusable_ips()
        if not ips:
            return
        if self._focused_ip not in ips:
            self._cycle_focus(forward=True)
            return
        cur_rect = self._focus_rect_for_ip(self._focused_ip)
        if cur_rect is None:
            return
        cx = cur_rect[0] + cur_rect[2] / 2.0
        cy = cur_rect[1] + cur_rect[3] / 2.0
        best_ip, best_dist = None, None
        for ip in ips:
            if ip == self._focused_ip:
                continue
            rect = self._focus_rect_for_ip(ip)
            if rect is None:
                continue
            ox = rect[0] + rect[2] / 2.0
            oy = rect[1] + rect[3] / 2.0
            dx, dy = ox - cx, oy - cy
            if direction == "right" and dx <= 0:
                continue
            if direction == "left" and dx >= 0:
                continue
            if direction == "down" and dy <= 0:
                continue
            if direction == "up" and dy >= 0:
                continue
            dist = (dx * dx + dy * dy) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_ip = ip
        if best_ip is None:
            return
        _focus_senses.dispatch_mouse_event("focus_out", self._focused_ip, None)
        self._focused_ip = best_ip
        _focus_senses.dispatch_mouse_event("focus_in", best_ip, None)
        self.update()

    def _activate_focused_shape(self) -> None:
        from Draw._connectors import senses as _focus_senses
        ip = self._focused_ip
        if not ip:
            return
        _focus_senses.dispatch_mouse_event("mouse_click", ip, "left")
        _focus_senses.dispatch_mouse_event("mouse_leftclick", ip, "left")
        _focus_senses.dispatch_mouse_event("mouse_press", ip, "left")
        _focus_senses.dispatch_mouse_event("mouse_release", ip, "left")

    # ── resize ────────────────────────────────────────────────────────────────

    def resizeEvent(self, event):       # noqa: N802
        if self.parent():
            self.setGeometry(self.parent().rect())  # type: ignore[union-attr]
        super().resizeEvent(event)

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _event):       # noqa: N802
        from Draw.debug import debug as _debug_manager
        if _debug_manager.should_stop():
            return

        from Draw._layout import _draw_one_layout
        from Draw._motion import motion as _motion_registry
        from Draw._overlap import Rect, flow_occupied_rect, get_strategy_for_flow
        from Draw._text import _draw_one_text, _text_align_pos
        from Draw._shapes import (
            _apply_motion_geometry,
            _draw_one_shape,
            _shape_preferred_geometry,
            _shape_hit_geometry,
            _parse_size,
        )

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            cw, ch = self.width(), self.height()
            _last_canvas_size["size"] = (cw, ch)
            now = time.perf_counter()

            for layout in self.layout_items:
                _draw_one_layout(painter, layout, cw, ch)

            # Sort by z descending: higher z value = further back (drawn first)
            if getattr(self, "_z_order_dirty", True):
                self._sorted_shapes = sorted(self.shape_items, key=lambda s: -s.z)
                self._z_order_dirty = False
            sorted_shapes = self._sorted_shapes

            # ── Persistent global occupied registry ───────────────────────
            # When shapes are added/removed we set _occupied_dirty=True.
            # On the first dirty paint we walk every shape in z-order, compute
            # its preferred position, run collision avoidance against ALL
            # already-placed shapes (not just same-area), store the final
            # position on the shape, and build _global_occupied.
            # On subsequent clean paints we just re-use stored positions
            # (motion/animation updates last_position directly).

            canvas_bounds = Rect(0.0, 0.0, float(cw), float(ch))

            if self._occupied_dirty:
                self._global_occupied = []
                for s in sorted_shapes:
                    # skip overlap=True shapes — they don't affect the registry
                    flow_spec = getattr(s, "flow", None)
                    if (
                        flow_spec is None
                        or not getattr(flow_spec, "enabled", False)
                        or getattr(flow_spec, "role", "item") == "ignore"
                    ):
                        # still draw at preferred position
                        try:
                            _, _, _, _, sw, sh, px, py = _shape_preferred_geometry(s, cw, ch, window_tag=getattr(self, "_window_tag", None))
                        except Exception:
                            continue
                        s._placed_x = float(px)
                        s._placed_y = float(py)
                        s._placed_w = int(sw)
                        s._placed_h = int(sh)
                        continue

                    try:
                        origin_x, origin_y, area_w, area_h, sw, sh, preferred_x, preferred_y = _shape_preferred_geometry(s, cw, ch, window_tag=getattr(self, "_window_tag", None))
                    except Exception as exc:
                        _logger.warning("Draw.shapes: overlap registry skip: %s", exc)
                        continue

                    if getattr(flow_spec, "role", "item") == "item":
                        shape_rect = Rect(0.0, 0.0, float(sw), float(sh))
                        strategy = get_strategy_for_flow(flow_spec)
                        placement_bounds = canvas_bounds
                        if getattr(flow_spec, "scope", "window") == "cell":
                            placement_bounds = Rect(
                                float(origin_x),
                                float(origin_y),
                                float(area_w),
                                float(area_h),
                            )
                        position = strategy.find_position(
                            shape_rect,
                            self._global_occupied,
                            placement_bounds,
                            preferred_x,
                            preferred_y,
                        )
                        if position is None:
                            position = (preferred_x, preferred_y)
                    else:
                        position = (preferred_x, preferred_y)

                    final_x, final_y = position
                    s._placed_x = float(final_x)
                    s._placed_y = float(final_y)
                    s._placed_w = int(sw)
                    s._placed_h = int(sh)

                    self._global_occupied.append(
                        flow_occupied_rect(final_x, final_y, float(sw), float(sh), flow_spec)
                    )

                self._occupied_dirty = False

            # ── Text overlap placement pass ────────────────────────────────
            # For every text item that has closest_rect_area=True, compute its
            # natural bounding box, then use the same HorizontalStackStrategy
            # as shapes to find a non-overlapping position.  Items without
            # closest_rect_area skip the registry and draw at their natural pos.
            # We share _global_occupied so text also avoids placed shapes.

            text_occupied: list = list(self._global_occupied)  # start from shapes

            for t in self.text_items:
                flow_spec = getattr(t, "flow", None)
                if (
                    flow_spec is None
                    or not getattr(flow_spec, "enabled", False)
                    or getattr(flow_spec, "role", "item") == "ignore"
                ):
                    # reset any stale placed position from a previous dirty cycle
                    t._placed_x = None
                    t._placed_y = None
                    continue

                # ── measure natural position & size ──────────────────────
                try:
                    font = QFont(t.font_family)
                    font.setPixelSize(max(1, t.font_size))
                    font.setBold(t.bold)
                    font.setItalic(t.italic)
                    if t.letter_spacing != 0:
                        font.setLetterSpacing(
                            QFont.SpacingType.AbsoluteSpacing, t.letter_spacing
                        )
                    fm = QFontMetricsF(font)
                    render_text = t.input_buffer if t.input_enabled else t.text
                    lines = render_text.split("\n") if render_text else [""]
                    line_h = fm.height() * t.line_height
                    max_w = max(
                        (fm.horizontalAdvance(ln) for ln in lines), default=0.0
                    )
                    total_h = line_h * len(lines)
                    pad = t.background_padding
                    box_w = max_w + pad * 2
                    box_h = total_h + pad * 2
                except Exception:
                    t._placed_x = None
                    t._placed_y = None
                    continue

                # ── natural (preferred) origin ────────────────────────────
                cell_rect_t = None
                if t.layout is not None and t.cell is not None:
                    from Draw._layout import set as _layout_registry
                    try:
                        if isinstance(t.cell, str):
                            lo = _layout_registry.resolve(t.cell)
                            cell_rect_t = lo.cell_rect(cw, ch, (0, 0))
                        else:
                            lo = _layout_registry.resolve(t.layout)
                            cell_rect_t = lo.cell_rect(cw, ch, t.cell)
                    except Exception:
                        cell_rect_t = None

                if cell_rect_t is not None:
                    if t.x is not None and t.y is not None:
                        pref_x = cell_rect_t.left() + float(t.x)
                        pref_y = cell_rect_t.top() + float(t.y)
                    elif t.align is not None:
                        rx, ry = _text_align_pos(
                            t.align, box_w, box_h,
                            int(cell_rect_t.width()), int(cell_rect_t.height()),
                        )
                        pref_x = cell_rect_t.left() + rx
                        pref_y = cell_rect_t.top() + ry
                    else:
                        pref_x = cell_rect_t.left() + (cell_rect_t.width() - box_w) / 2.0
                        pref_y = cell_rect_t.top() + (cell_rect_t.height() - box_h) / 2.0
                elif t.x is not None and t.y is not None:
                    pref_x, pref_y = float(t.x), float(t.y)
                elif t.align is not None:
                    pref_x, pref_y = _text_align_pos(t.align, box_w, box_h, cw, ch)
                else:
                    pref_x, pref_y = (cw - box_w) / 2.0, (ch - box_h) / 2.0

                # ── find non-overlapping position ─────────────────────────
                if getattr(flow_spec, "role", "item") == "item":
                    text_rect = Rect(0.0, 0.0, float(box_w), float(box_h))
                    strategy = get_strategy_for_flow(flow_spec)
                    placement_bounds = canvas_bounds
                    if getattr(flow_spec, "scope", "window") == "cell" and cell_rect_t is not None:
                        placement_bounds = Rect(
                            float(cell_rect_t.left()),
                            float(cell_rect_t.top()),
                            float(cell_rect_t.width()),
                            float(cell_rect_t.height()),
                        )
                    position = strategy.find_position(
                        text_rect, text_occupied, placement_bounds, pref_x, pref_y
                    )
                    if position is None:
                        position = (pref_x, pref_y)
                else:
                    position = (pref_x, pref_y)

                t._placed_x, t._placed_y = position

                # register so subsequent text items avoid this one too
                text_occupied.append(
                    flow_occupied_rect(
                        float(t._placed_x),
                        float(t._placed_y),
                        float(box_w),
                        float(box_h),
                        flow_spec,
                    )
                )

            # ── Interleaved Draw Pass by Z-Order ──────────────────────────
            # We build the sorted queue of all shapes and texts.
            # Higher z value = further back (drawn first).
            # For equal z, shapes are drawn before texts (so text is on top of shapes).
            def _get_text_z(t, shapes):
                z_val = getattr(t, "z", 0)
                if z_val == "as_shape":
                    if getattr(t, "ip", None):
                        for s in shapes:
                            if s.ip == t.ip:
                                return s.z
                    return 0
                try:
                    return int(z_val) if z_val is not None else 0
                except (ValueError, TypeError):
                    return 0

            draw_queue = []
            for s in self.shape_items:
                draw_queue.append((s, "shape", s.z))
            for t in self.text_items:
                draw_queue.append((t, "text", _get_text_z(t, self.shape_items)))

            # Sort: descending by z, then shape (0) before text (1)
            draw_queue.sort(key=lambda item: (-item[2], 0 if item[1] == "shape" else 1))

            for item_tuple in draw_queue:
                item, kind_str, _ = item_tuple
                if kind_str == "shape":
                    s = item
                    try:
                        index = self.shape_items.index(s)
                    except ValueError:
                        index = 0
                    try:
                        (
                            origin_x,
                            origin_y,
                            area_w,
                            area_h,
                            sw,
                            sh,
                            preferred_x,
                            preferred_y,
                        ) = _shape_preferred_geometry(s, cw, ch, window_tag=getattr(self, "_window_tag", None))
                    except Exception as exc:
                        s.last_position = None
                        s.last_size = None
                        s.last_hit_position = None
                        s.last_hit_size = None
                        _logger.warning("Draw.shapes: skipping shape %s due to error: %s", index, exc)
                        continue

                    # Use the stored placed position (computed by overlap registry).
                    # Exception: a dynamic "ip:other_ip" align depends on another
                    # shape's live position, which may have moved since the
                    # earlier overlap-registry pre-pass computed _placed_x/_y —
                    # always use the freshly-resolved preferred_x/y instead of
                    # that stale cache, same rationale as the bbox/path caches.
                    is_dynamic_align_now = isinstance(s.align, str) and s.align.startswith("ip:")
                    if is_dynamic_align_now:
                        final_x, final_y = preferred_x, preferred_y
                    else:
                        final_x = getattr(s, "_placed_x", preferred_x)
                        final_y = getattr(s, "_placed_y", preferred_y)
                    sw = getattr(s, "_placed_w", sw)
                    sh = getattr(s, "_placed_h", sh)

                    def _is_scroller_component_shape(shape_obj, canvas_obj) -> bool:
                        if not shape_obj or not getattr(shape_obj, "ip", None):
                            return False
                        ip_s = shape_obj.ip
                        if ip_s.startswith("scroller_") or "_track" in ip_s or "_thumb" in ip_s:
                            return True
                        for sc_cfg in getattr(canvas_obj, "_scroller_configs", []):
                            if ip_s in (sc_cfg.get("thumb_ip"), sc_cfg.get("track_ip")):
                                return True
                        return False

                    if not _is_scroller_component_shape(s, self):
                        final_x -= self._scroll_x
                        final_y -= self._scroll_y

                    # While a shape is being dragged, its live cursor-tracked position
                    # (set every mouseMoveEvent as s._drag_x/_drag_y) overrides whatever
                    # the layout/overlap system computed as its "placed" position.
                    if getattr(s, "_is_dragged", False) and not _is_scroller_component_shape(s, self):
                        final_x = getattr(s, "_drag_x", final_x)
                        final_y = getattr(s, "_drag_y", final_y)

                    hit_sw, hit_sh = sw, sh
                    # Note: hitbox_mode="resetgeometry" custom sizing removed (ShapeDef
                    # has no hitbox_width_raw/hitbox_height_raw; use Draw.hitbox registry directly)

                    shape_motion_state = _motion_registry.compute_shape_state(
                        s, now, _parse_color, float(final_x), float(final_y), sw, sh, self
                    )
                    s._last_motion_state = shape_motion_state
                    anim_x, anim_y, anim_sw, anim_sh = _apply_motion_geometry(
                        shape_motion_state,
                        float(final_x),
                        float(final_y),
                        sw,
                        sh,
                    )
                    s.last_position = (float(anim_x), float(anim_y))
                    s.last_size = (anim_sw, anim_sh)
                    hx, hy, hsw, hsh = _shape_hit_geometry(s)
                    hitbox_motion_state = _motion_registry.compute_hitbox_state(
                        s, now, _parse_color, hx, hy, hsw, hsh, self
                    )
                    hx, hy, hsw, hsh = _apply_motion_geometry(
                        hitbox_motion_state,
                        hx,
                        hy,
                        hsw,
                        hsh,
                    )
                    s.last_hit_position = (float(hx), float(hy))
                    s.last_hit_size = (int(hsw), int(hsh))
                    _draw_one_shape(
                        painter,
                        s,
                        cw,
                        ch,
                        position_override=(final_x, final_y),
                        motion_state=shape_motion_state,
                        canvas=self,
                    )
                else:
                    t = item
                    # Resolve Draw.motion(...) records attached to this text item
                    # (t.motion is set dynamically by MotionRegistry.__call__, same
                    # mechanism shapes use — see Draw._motion.find_elements_by_ip).
                    # Reference geometry comes from last frame's placed rect since
                    # the natural rect for this frame isn't computed until
                    # _draw_one_text runs below; this matches how a moving shape's
                    # motion expression also reads its own last-known position.
                    if getattr(t, "motion", None):
                        ref_x, ref_y, ref_w, ref_h = t.last_rect if t.last_rect else (
                            float(t.x or 0), float(t.y or 0), 0.0, 0.0
                        )
                        t._last_motion_state = _motion_registry.compute_shape_state(
                            t, now, _parse_color, ref_x, ref_y, ref_w, ref_h, self
                        )
                    else:
                        t._last_motion_state = None
                    _draw_one_text(painter, t, cw, ch, canvas=self)

            # -- keyboard-accessibility focus outline (Tier 4) ---------------
            if self._focused_ip:
                _focus_rect = self._focus_rect_for_ip(self._focused_ip)
                if _focus_rect is not None:
                    _fx, _fy, _fw, _fh = _focus_rect
                    painter.save()
                    _focus_pen = QPen(QColor(66, 133, 244))
                    _focus_pen.setWidth(2)
                    _focus_pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(_focus_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(
                        QRectF(_fx - 3, _fy - 3, _fw + 6, _fh + 6), 4, 4
                    )
                    painter.restore()
        finally:
            if painter.isActive():
                painter.end()
            t_frame_ms = (time.perf_counter() - now) * 1000.0
            draw_calls = len(self.shape_items) + len(self.text_items)
            vert_count = sum(s.vertices or 4 for s in self.shape_items)
            tri_count = sum(max(1, (s.vertices or 4) - 2) for s in self.shape_items)
            _debug_manager.record_frame(t_frame_ms, draw_calls=draw_calls, vertices=vert_count, triangles=tri_count)
            _debug_manager.record_timing("Render", t_frame_ms)





from Draw._optimize import compilable, register_instance


@compilable
@dataclass
class ShapeDef:
    # Geometry
    vertices: Optional[int]
    size_raw: object                       # [w, h], "200px", or None
    border_radius_raw: object

    # Position
    x: Optional[int]
    y: Optional[int]
    align: Optional[str]
    rotation: float

    # Appearance
    color: QColor
    border_color: QColor
    border_width: int
    border_style: str
    opacity: int


    # Geometry modifiers
    curve_mode: str                         # line|smooth|slope|wave|arc|bend_all
    bend: List[Dict[str, Any]]
    bend_amount: float                      # global bend strength for bend_all mode (px, default 40)
    warp: Optional[List[List[Tuple[float,float]]]]  # mesh warp: grid of (dx,dy) displacement vectors
    exclude: List[Dict[str, Any]]
    symmetry: Optional[Dict[str, Any]]

    # Hitbox
    hitbox_mode: Optional[str]              # None | "shape" | "closed_rec"
    hit_box: str                            # "shape" or "closed_rec"

    # Custom dict (alternative nesting for bend/exclude/symmetry/properties)
    custom: Optional[Dict[str, Any]]

    # Layer / collision
    z: int
    overlap: bool
    flow: object

    # Identity
    ip: Optional[str]

    # Layout positioning fields
    layout: Optional[object] = None
    cell: Optional[object] = None

    # Area / overlap control (from area=[{...}] parameter)
    area_expand: Tuple[float, float] = field(default=(0.0, 0.0))
    area_move: Optional[str] = field(default=None)   # "right"|"left"|"down"|"up" etc.

    # Image / video support
    shape_type: str = field(default="vector")
    src: Optional[str] = field(default=None)
    _video_loop: bool = field(default=True)
    _video_autoplay: bool = field(default=True)
    _video_muted: bool = field(default=False)

    # Interaction control
    inside: Optional[str] = field(default=None)
    move_path: Optional[str] = field(default=None)
    custom_vertices: Optional[List[Tuple[float, float]]] = field(default=None)

    # Runtime (set during paint)
    last_position: Optional[Tuple[float, float]] = field(default=None, init=False)
    last_size: Optional[Tuple[int, int]] = field(default=None, init=False)
    last_rotation: Optional[float] = field(default=None, init=False)
    last_rotation_pivot: Optional[Tuple[float, float]] = field(default=None, init=False)
    _shadow_cache: Optional[object] = field(default=None, init=False)
    _shadow_cache_key: Optional[object] = field(default=None, init=False)

    # ── Dirty-rendering / path cache (Phase 2.2) ────────────────────────
    # `dirty=True` forces the next paint to rebuild this shape's QPainterPath
    # from scratch (polygon generation, curve/bend deformation, symmetry,
    # exclude holes — the expensive part of drawing a shape). It starts True
    # so every shape gets a real build on its first paint. It is set back to
    # True by anything that edits geometry-affecting fields (see
    # _PATH_AFFECTING_KEYS, _update_shape_inplace, and the sub_edit path in
    # _ShapeRegistry.__call__). Position-only changes (x/y, drag, scroll,
    # position-only motion) do NOT set dirty — _get_shape_path() reuses the
    # cached path and just translates it, which is far cheaper than a full
    # rebuild.
    dirty: bool = field(default=True, init=False)
    _path_cache: Optional[object] = field(default=None, init=False)       # last built QPainterPath
    _path_cache_key: Optional[object] = field(default=None, init=False)   # signature of shape-affecting props
    _path_cache_origin: Tuple[float, float] = field(default=(0.0, 0.0), init=False)

    # ── Bounding-box cache (Phase 2.3) ───────────────────────────────────
    # `_shape_preferred_pos()` computes (sw, sh, ox, oy) — this shape's
    # actual on-canvas size + origin — by parsing size_raw (which may be a
    # percentage/px string) and resolving align. This was recomputed from
    # scratch on every single paint for every shape. The result depends
    # only on (cw, ch, size_raw, x, y, align), so we cache it keyed on that
    # signature: unrelated per-frame repaints reuse the cached tuple, and
    # any change to those fields naturally produces a different key — no
    # separate dirty flag needed.
    _bbox_cache: Optional[Tuple[int, int, float, float]] = field(default=None, init=False)
    _bbox_cache_key: Optional[object] = field(default=None, init=False)

    # ── Generic optimize() hook (Draw._optimize.Compilable) ──────────────
    # ShapeDef is the reference implementation of the Compilable protocol.
    # It deliberately reuses the existing dirty flag / path cache / bbox
    # cache above instead of introducing parallel state — _sig/_is_dirty/
    # _compile are a thin adapter over what already exists, not a new
    # cache. See _optimize.py for what future modules (ConnectorRecord,
    # a future RoomDef, etc.) should copy.
    _compiled_sig: Optional[object] = field(default=None, init=False)

    def __post_init__(self) -> None:
        register_instance(self)

    def dispose(self) -> None:
        """Release video player resources and clean up cached media objects."""
        old_player = getattr(self, "_video_player", None)
        if old_player and hasattr(old_player, "stop"):
            try:
                old_player.stop()
            except Exception:
                pass
        for attr in ["_video_player", "_video_audio", "_video_sink", "_video_frame", "_video_canvas_ref"]:
            if hasattr(self, attr):
                delattr(self, attr)

    def _sig(self):
        """Cheap hashable signature of every field that can change this
        shape's built QPainterPath or its bbox — the two caches above.
        Mirrors the key tuples already used in _get_shape_path() /
        _shape_preferred_pos(), so this never drifts out of sync with
        what those functions actually treat as cache-relevant."""
        return (
            self.vertices, _hashable(self.size_raw), self.x, self.y,
            self.align, self.rotation, self.curve_mode,
            _hashable(self.bend), self.bend_amount, _hashable(self.warp),
            _hashable(self.exclude), _hashable(self.symmetry),
            str(self.border_radius_raw),
        )

    def _is_dirty(self) -> bool:
        """True if this shape hasn't been built yet, or something changed
        since the last build. Reuses the existing `dirty` flag (already
        set correctly by _update_shape_inplace / sub_edit) rather than
        re-deriving it — `_sig()` is only used as a defensive fallback."""
        return bool(self.dirty) or self._path_cache is None or self._compiled_sig != self._sig()

    def _compile(self):
        """Eagerly warm this shape's path + bbox caches right now, using
        the most recently known canvas size, instead of waiting for it to
        get discovered lazily on the next paint. Best-effort: if this
        shape hasn't been painted in any window yet there's no real
        canvas size to warm against, so this silently no-ops and the
        first paint still builds it correctly either way — Draw.optimize()
        is a head start, not the only build path."""
        cw, ch = _last_canvas_size.get("size", (None, None))
        if cw is None:
            self._compiled_sig = self._sig()
            return self._path_cache
        try:
            sw, sh, ox, oy = _shape_preferred_pos(self, cw, ch)
            _get_shape_path(self, ox, oy, sw, sh, self.rotation)
        except Exception:
            pass
        self._compiled_sig = self._sig()
        return self._path_cache


# Last canvas size any _DrawCanvas painted at — updated in paintEvent().
# Used only as a best-effort hint by ShapeDef._compile() to pre-warm
# caches; never required for correctness.
_last_canvas_size: Dict[str, Tuple[int, int]] = {}


_UNIT_POLY_CACHE: Dict[Tuple[int, float], List[Tuple[float, float]]] = {}


def _get_unit_polygon(n: int, start_angle_deg: float = -90.0) -> List[Tuple[float, float]]:
    key = (n, start_angle_deg)
    unit = _UNIT_POLY_CACHE.get(key)
    if unit is not None:
        return unit
    unit = []
    for i in range(n):
        a = math.radians(start_angle_deg + 360.0 * i / n)
        unit.append((math.cos(a), math.sin(a)))
    if len(_UNIT_POLY_CACHE) < 128:
        _UNIT_POLY_CACHE[key] = unit
    return unit


def _regular_polygon_points(
    cx: float, cy: float,
    rx: float, ry: float,
    n: int,
    start_angle_deg: float = -90.0,
) -> List[QPointF]:
    unit = _get_unit_polygon(n, start_angle_deg)
    return [QPointF(cx + rx * ux, cy + ry * uy) for ux, uy in unit]


# ── catmull-rom smooth path ────────────────────────────────────────────────────

def _catmull_rom_path(pts: List[QPointF], tension: float = 0.5) -> QPainterPath:
    """Close smooth Catmull-Rom spline through all pts."""
    n = len(pts)
    path = QPainterPath()
    if n < 2:
        return path

    steps = 20

    def cr_pt(p0: QPointF, p1: QPointF, p2: QPointF, p3: QPointF, t: float) -> QPointF:
        t2, t3 = t * t, t * t * t
        x = tension * (
            (-p0.x() + 3*p1.x() - 3*p2.x() + p3.x()) * t3
            + (2*p0.x() - 5*p1.x() + 4*p2.x() - p3.x()) * t2
            + (-p0.x() + p2.x()) * t
        ) + p1.x()
        y = tension * (
            (-p0.y() + 3*p1.y() - 3*p2.y() + p3.y()) * t3
            + (2*p0.y() - 5*p1.y() + 4*p2.y() - p3.y()) * t2
            + (-p0.y() + p2.y()) * t
        ) + p1.y()
        return QPointF(x, y)

    started = False
    for seg in range(n):
        p0 = pts[(seg - 1) % n]
        p1 = pts[seg]
        p2 = pts[(seg + 1) % n]
        p3 = pts[(seg + 2) % n]
        for step in range(steps):
            pt = cr_pt(p0, p1, p2, p3, step / steps)
            if not started:
                path.moveTo(pt)
                started = True
            else:
                path.lineTo(pt)
    path.closeSubpath()
    return path


# ── bent-side drawing ─────────────────────────────────────────────────────────

def _perpendicular_unit(p1: QPointF, p2: QPointF) -> Tuple[float, float]:
    """Unit vector perpendicular (right-hand-side = outward for CW polygon)."""
    dx = p2.x() - p1.x()
    dy = p2.y() - p1.y()
    length = math.hypot(dx, dy) or 1e-9
    # Rotate 90° CW: (dy, -dx) so that positive offset pushes outward
    return dy / length, -dx / length


def _draw_bent_side(
    path: QPainterPath,
    p_start: QPointF,
    p_end: QPointF,
    bends: List[Dict[str, Any]],
) -> None:
    """
    Draw one polygon side (from p_start to p_end) with bend deformations.
    Multiple bends on the same side are stacked (last wins for same region).
    We walk the side in t=0..1, drawing straight segments except in affected
    zones where we emit a quadratic bezier.
    """
    # Collect all bend zones
    zones: List[Tuple[float, float, float]] = []  # (t_start, t_end, offset)
    for b in bends:
        t_s, t_e = _parse_affect_range(b.get("affect", "-100% -> 100%"))
        offset = _as_float(b.get("offset", 0))
        if b.get("direction", "out") == "in":
            offset = -offset
        zones.append((t_s, t_e, offset))

    if not zones:
        path.lineTo(p_end)
        return

    perp_x, perp_y = _perpendicular_unit(p_start, p_end)

    # If entire side is one zone
    if len(zones) == 1:
        t_s, t_e, offset = zones[0]
        bp_s = _lerp_pt(p_start, p_end, t_s)
        bp_e = _lerp_pt(p_start, p_end, t_e)
        t_mid = (t_s + t_e) / 2.0
        bp_mid = _lerp_pt(p_start, p_end, t_mid)
        ctrl = QPointF(bp_mid.x() + perp_x * offset,
                       bp_mid.y() + perp_y * offset)
        path.lineTo(bp_s)
        path.quadTo(ctrl, bp_e)
        path.lineTo(p_end)
    else:
        # Multiple zones: sort by start and walk
        zones.sort(key=lambda z: z[0])
        cursor = 0.0
        for (t_s, t_e, offset) in zones:
            t_s = max(cursor, t_s)
            if t_s >= t_e:
                continue
            # straight from cursor to t_s
            path.lineTo(_lerp_pt(p_start, p_end, t_s))
            # bent portion
            t_mid = (t_s + t_e) / 2.0
            bp_mid = _lerp_pt(p_start, p_end, t_mid)
            ctrl = QPointF(bp_mid.x() + perp_x * offset,
                           bp_mid.y() + perp_y * offset)
            path.quadTo(ctrl, _lerp_pt(p_start, p_end, t_e))
            cursor = t_e
        path.lineTo(p_end)


def _draw_slope_side(path: QPainterPath, p1: QPointF, p2: QPointF,
                     bulge: float = 0.15) -> None:
    """Slight outward quadratic bulge."""
    perp_x, perp_y = _perpendicular_unit(p1, p2)
    seg_len = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
    push = seg_len * bulge
    mid = _lerp_pt(p1, p2, 0.5)
    ctrl = QPointF(mid.x() + perp_x * push, mid.y() + perp_y * push)
    path.quadTo(ctrl, p2)


def _draw_arc_side(path: QPainterPath, p1: QPointF, p2: QPointF,
                   radius_frac: float = 0.25) -> None:
    """Inward arc (like corner rounding on all edges)."""
    perp_x, perp_y = _perpendicular_unit(p1, p2)
    seg_len = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
    push = seg_len * radius_frac
    mid = _lerp_pt(p1, p2, 0.5)
    ctrl = QPointF(mid.x() - perp_x * push, mid.y() - perp_y * push)
    path.quadTo(ctrl, p2)


def _draw_wave_side(path: QPainterPath, p1: QPointF, p2: QPointF,
                    amplitude_frac: float = 0.08, cycles: int = 3) -> None:
    """Sinusoidal wave along the edge."""
    perp_x, perp_y = _perpendicular_unit(p1, p2)
    seg_len = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
    amp = seg_len * amplitude_frac
    steps = max(24, cycles * 12)
    for s in range(1, steps + 1):
        t = s / steps
        base = _lerp_pt(p1, p2, t)
        wave = math.sin(t * cycles * math.tau) * amp
        path.lineTo(QPointF(base.x() + perp_x * wave,
                            base.y() + perp_y * wave))


def _draw_bend_all_path(
    pts: List[QPointF],
    bend_amount: float,
) -> QPainterPath:
    """
    Apply a uniform outward arc-bend to every side of the polygon.
    bend_amount > 0 = bulge outward; < 0 = pinch inward.
    Each side is drawn as a single quadratic bezier with the control point
    pushed perpendicularly by bend_amount pixels.
    """
    n = len(pts)
    path = QPainterPath()
    if n < 2:
        return path
    path.moveTo(pts[0])
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        perp_x, perp_y = _perpendicular_unit(p1, p2)
        mid = _lerp_pt(p1, p2, 0.5)
        ctrl = QPointF(mid.x() + perp_x * bend_amount,
                       mid.y() + perp_y * bend_amount)
        path.quadTo(ctrl, p2)
    path.closeSubpath()
    return path


def _apply_warp_to_path(
    path: QPainterPath,
    warp: List[List[Tuple[float, float]]],
    ox: float, oy: float,
    sw: int, sh: int,
) -> QPainterPath:
    """
    Apply a mesh-warp deformation to a QPainterPath.
    
    `warp` is a 2-D list of (dx, dy) displacement tuples forming a grid.
    Grid size can be 2×2, 3×3, etc. Each cell represents a region of the
    shape bounding box; control points inside that region are displaced by
    bilinear interpolation of the surrounding corner displacements.
    
    Example — pinch the top-right corner inward by 30px:
        warp = [[(0,0), (-30,30)],
                [(0,0), (0,0)]]
    """
    rows = len(warp)
    cols = len(warp[0]) if rows > 0 else 0
    if rows < 2 or cols < 2:
        return path

    # Sample the original path as a polygon with many points
    poly = path.toFillPolygon()
    if poly.isEmpty():
        return path

    new_path = QPainterPath()
    first = True
    for pt in poly:
        # Normalised position within bounding box (0..1)
        u = max(0.0, min(1.0, (pt.x() - ox) / sw)) if sw > 0 else 0.0
        v = max(0.0, min(1.0, (pt.y() - oy) / sh)) if sh > 0 else 0.0

        # Grid cell indices
        col_f = u * (cols - 1)
        row_f = v * (rows - 1)
        c0 = max(0, min(cols - 2, int(col_f)))
        r0 = max(0, min(rows - 2, int(row_f)))
        tc = col_f - c0  # fractional within cell (0..1)
        tr = row_f - r0

        # Bilinear interpolation of the four corner displacements
        dx00, dy00 = warp[r0][c0]
        dx10, dy10 = warp[r0][c0 + 1]
        dx01, dy01 = warp[r0 + 1][c0]
        dx11, dy11 = warp[r0 + 1][c0 + 1]

        dx = (dx00 * (1 - tc) * (1 - tr) + dx10 * tc * (1 - tr) +
              dx01 * (1 - tc) * tr       + dx11 * tc * tr)
        dy = (dy00 * (1 - tc) * (1 - tr) + dy10 * tc * (1 - tr) +
              dy01 * (1 - tc) * tr       + dy11 * tc * tr)

        new_pt = QPointF(pt.x() + dx, pt.y() + dy)
        if first:
            new_path.moveTo(new_pt)
            first = False
        else:
            new_path.lineTo(new_pt)

    new_path.closeSubpath()
    return new_path


# ── main edge-path builder ────────────────────────────────────────────────────

def _build_edge_path(
    pts: List[QPointF],
    curve_mode: str,
    bends: List[Dict[str, Any]],
    bend_amount: float = 40.0,
) -> QPainterPath:
    """
    Build a closed path through pts, applying curve_mode and per-side bends.
    Side numbering is 1-indexed (side 1 = pts[0]→pts[1]).
    bend_amount is used only for curve_mode="bend_all".
    """
    n = len(pts)
    if n < 2:
        return QPainterPath()

    # Smooth uses catmull-rom (bends ignored for now in smooth mode)
    if curve_mode == "smooth":
        return _catmull_rom_path(pts)

    # bend_all: uniform outward/inward bezier arc on every side
    if curve_mode == "bend_all":
        return _draw_bend_all_path(pts, bend_amount)

    # Build side→bend-specs index
    side_bends: Dict[int, List[Dict[str, Any]]] = {}
    for b in bends:
        s = _as_int(b.get("side", 1), default=1)
        side_bends.setdefault(s, []).append(b)

    path = QPainterPath()
    path.moveTo(pts[0])

    for i in range(n):
        p_start = pts[i]
        p_end   = pts[(i + 1) % n]
        side_num = i + 1

        if side_num in side_bends:
            _draw_bent_side(path, p_start, p_end, side_bends[side_num])
        elif curve_mode == "slope":
            _draw_slope_side(path, p_start, p_end)
        elif curve_mode == "arc":
            _draw_arc_side(path, p_start, p_end)
        elif curve_mode == "wave":
            _draw_wave_side(path, p_start, p_end)
        else:
            path.lineTo(p_end)

    path.closeSubpath()
    return path


# ── exclude (cutout) builder ──────────────────────────────────────────────────

def _build_exclude_paths(
    exclude_specs: List[Dict[str, Any]],
    ox: float, oy: float,
    sw: int, sh: int,
) -> List[QPainterPath]:
    """
    Build hole sub-paths from the 'exclude' specs.
    Uses the scale/let_it coordinate mapping described in the docstring.
    """
    cx = ox + sw / 2.0
    cy = oy + sh / 2.0
    holes: List[QPainterPath] = []

    for spec in exclude_specs:
        scale_raw = spec.get("scale", [100, 100])
        if isinstance(scale_raw, (list, tuple)) and len(scale_raw) >= 2:
            scale_w = max(1e-9, float(scale_raw[0]))
            scale_h = max(1e-9, float(scale_raw[1]))
        else:
            v = max(1e-9, float(scale_raw))
            scale_w = scale_h = v

        let_it = str(spec.get("let_it", "center")).lower().strip()
        # Determine origin in canvas coordinates
        origin_map = {
            "center":       (cx, cy),
            "top-left":     (ox, oy),
            "top":          (cx, oy),
            "top-right":    (ox + sw, oy),
            "left":         (ox, cy),
            "right":        (ox + sw, cy),
            "bottom-left":  (ox, oy + sh),
            "bottom":       (cx, oy + sh),
            "bottom-right": (ox + sw, oy + sh),
        }
        orig_x, orig_y = origin_map.get(let_it, (cx, cy))

        # Scale factors: grid → canvas
        px_per_gx = sw / scale_w
        px_per_gy = sh / scale_h

        def to_canvas(gx: float, gy: float) -> QPointF:
            return QPointF(orig_x + gx * px_per_gx,
                           orig_y + gy * px_per_gy)

        # Gather line specs (integer or string-digit keys)
        for key, value in spec.items():
            if key in ("scale", "let_it"):
                continue
            if not isinstance(value, dict):
                continue

            line_str  = str(value.get("line", ""))
            edges_str = str(value.get("edges", ""))
            width_raw = value.get("width_between", "0px")
            hole_mode = str(value.get("curve_mode", "line")).strip().lower()

            coords = _parse_coord_string(line_str)
            if not coords:
                continue

            canvas_pts = [to_canvas(gx, gy) for gx, gy in coords]

            # Parse per-point edge roundness
            edge_vals: List[float] = []
            for part in edges_str.split(","):
                part = part.strip()
                if part:
                    try:
                        edge_vals.append(float(part))
                    except ValueError:
                        edge_vals.append(0.0)

            width_px = _parse_px_value(width_raw)

            if width_px > 0.5:
                # Ribbon-shaped hole: stroke the line
                stroke_path = QPainterPath()
                stroke_path.moveTo(canvas_pts[0])
                for pt in canvas_pts[1:]:
                    stroke_path.lineTo(pt)
                stroker = QPainterPathStroker()
                stroker.setWidth(width_px)
                stroker.setCapStyle(Qt.PenCapStyle.RoundCap)
                stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                holes.append(stroker.createStroke(stroke_path))
            else:
                # Polygon hole
                hole = QPainterPath()
                if hole_mode == "smooth" and len(canvas_pts) >= 3:
                    hole = _catmull_rom_path(canvas_pts)
                else:
                    hole.moveTo(canvas_pts[0])
                    n_pts = len(canvas_pts)
                    for i, pt in enumerate(canvas_pts[1:], start=1):
                        r = edge_vals[i] if i < len(edge_vals) else 0.0
                        if r > 0.5 and i < n_pts - 1:
                            # Rounded corner via quadratic bezier
                            prev = canvas_pts[i - 1]
                            nxt  = canvas_pts[(i + 1) % n_pts]
                            d_in  = math.hypot(pt.x() - prev.x(), pt.y() - prev.y()) or 1
                            d_out = math.hypot(nxt.x() - pt.x(),  nxt.y() - pt.y())  or 1
                            t_in  = min(r / d_in,  0.5)
                            t_out = min(r / d_out, 0.5)
                            corner_s = _lerp_pt(prev, pt,  1.0 - t_in)
                            corner_e = _lerp_pt(pt,   nxt, t_out)
                            hole.lineTo(corner_s)
                            hole.quadTo(pt, corner_e)
                        else:
                            hole.lineTo(pt)
                    hole.closeSubpath()
                holes.append(hole)

    return holes


# ── symmetry ──────────────────────────────────────────────────────────────────

def _apply_symmetry(
    path: QPainterPath,
    sym: Dict[str, Any],
    cx: float, cy: float,
) -> QPainterPath:
    sym_type = str(sym.get("type", "radial")).strip().lower()

    if sym_type == "radial":
        count = max(2, _as_int(sym.get("count", 4)))
        result = QPainterPath()
        result.setFillRule(Qt.FillRule.WindingFill)
        for i in range(count):
            angle = 360.0 * i / count
            t = QTransform()
            t.translate(cx, cy)
            t.rotate(angle)
            t.translate(-cx, -cy)
            result.addPath(t.map(path))
        return result

    if sym_type == "mirror":
        axis = str(sym.get("axis", "vertical")).strip().lower()
        result = QPainterPath(path)
        t = QTransform()
        if axis == "vertical":
            t.translate(cx * 2.0, 0.0)
            t.scale(-1.0, 1.0)
        elif axis == "horizontal":
            t.translate(0.0, cy * 2.0)
            t.scale(1.0, -1.0)
        elif axis == "diagonal":
            t.translate(cx, cy)
            t.rotate(90.0)
            t.scale(-1.0, 1.0)
            t.rotate(-90.0)
            t.translate(-cx, -cy)
        result.addPath(t.map(path))
        return result

    return path


# ── main path builder ─────────────────────────────────────────────────────────

def _build_shape_path(
    s: ShapeDef,
    ox: float, oy: float,
    sw: int, sh: int,
    rotation: Optional[float] = None,
    pivot: Optional[Tuple[float, float]] = None,
    motion_state: Optional[dict] = None,
) -> QPainterPath:
    """
    Assemble the full QPainterPath for a ShapeDef, applying (in order):
      1. Base polygon / rectangle
      2. Edge curve mode + per-side bends
      3. Rotation (around `pivot` if given, else this shape's own center —
         see Draw._bridge.resolve_point_ref / _motion.py's "center" key)
      4. Symmetry
      5. Exclude holes
    """
    cx = ox + sw / 2.0
    cy = oy + sh / 2.0
    rx = sw / 2.0
    ry = sh / 2.0
    n = s.vertices
    if motion_state and "vertices_count" in motion_state:
        n = motion_state["vertices_count"]

    # ── 1. Base path ──────────────────────────────────────────────────
    if s.custom_vertices and len(s.custom_vertices) >= 3:
        path = QPainterPath()
        path.moveTo(QPointF(float(s.custom_vertices[0][0]), float(s.custom_vertices[0][1])))
        for vx, vy in s.custom_vertices[1:]:
            path.lineTo(QPointF(float(vx), float(vy)))
        path.closeSubpath()
        base_path = path

    elif motion_state and "vertices" in motion_state:
        custom_vertices = motion_state["vertices"]
        ref_x = float(motion_state.get("ref_x", ox))
        ref_y = float(motion_state.get("ref_y", oy))
        ref_w = float(motion_state.get("ref_w", sw))
        ref_h = float(motion_state.get("ref_h", sh))
        
        cx_ref = ref_x + ref_w / 2.0
        cy_ref = ref_y + ref_h / 2.0
        cx_tgt = ox + sw / 2.0
        cy_tgt = oy + sh / 2.0
        
        scale_x = sw / ref_w if ref_w > 0.0 else 1.0
        scale_y = sh / ref_h if ref_h > 0.0 else 1.0
        
        mapped_pts = []
        for vx, vy in custom_vertices:
            dx = vx - cx_ref
            dy = vy - cy_ref
            vx_tgt = cx_tgt + dx * scale_x
            vy_tgt = cy_tgt + dy * scale_y
            mapped_pts.append(QPointF(vx_tgt, vy_tgt))
            
        path = QPainterPath()
        if mapped_pts:
            path.moveTo(mapped_pts[0])
            for pt in mapped_pts[1:]:
                path.lineTo(pt)
            path.closeSubpath()
        base_path = path

    elif n is None or n < 3:
        if n == 2:
            path = QPainterPath()
            path.moveTo(cx, oy)
            path.lineTo(cx, oy + sh)
            return path
        if n == 1:
            path = QPainterPath()
            r = min(rx, ry) * 0.1
            path.addEllipse(QPointF(cx, cy), r, r)
            return path
        # default: rectangle
        br = _parse_border_radius(s.border_radius_raw, sw, sh)
        path = QPainterPath()
        if br > 0:
            path.addRoundedRect(QRectF(ox, oy, sw, sh), br, br)
        else:
            path.addRect(QRectF(ox, oy, sw, sh))
        base_path = path

    elif n == 4 and (s.curve_mode == "line") and (not s.bend):
        br = _parse_border_radius(s.border_radius_raw, sw, sh)
        path = QPainterPath()
        if br > 0:
            path.addRoundedRect(QRectF(ox, oy, sw, sh), br, br)
        else:
            path.addRect(QRectF(ox, oy, sw, sh))
        base_path = path

    else:
        pts = _regular_polygon_points(cx, cy, rx, ry, n)
        base_path = _build_edge_path(pts, s.curve_mode, s.bend, getattr(s, "bend_amount", 40.0))

    # ── 2. Rotation ───────────────────────────────────────────────────
    rot = s.rotation if rotation is None else rotation
    if rot != 0.0:
        eff_pivot = pivot if pivot is not None else getattr(s, "rotation_center", getattr(s, "pivot", None))
        pcx, pcy = eff_pivot if eff_pivot is not None else (cx, cy)
        t = QTransform()
        t.translate(pcx, pcy)
        t.rotate(rot)
        t.translate(-pcx, -pcy)
        base_path = t.map(base_path)

    # ── 3. Symmetry ───────────────────────────────────────────────────
    if s.symmetry:
        base_path = _apply_symmetry(base_path, s.symmetry, cx, cy)

    # ── 4. Exclude holes ──────────────────────────────────────────────
    if s.exclude:
        base_path.setFillRule(Qt.FillRule.OddEvenFill)
        holes = _build_exclude_paths(s.exclude, ox, oy, sw, sh)
        for hole in holes:
            base_path.addPath(hole)

    return base_path


# ── dirty rendering / path cache (Phase 2.2) ───────────────────────────────────
# Which raw-dict keys affect the *shape* of the QPainterPath (as opposed to
# purely visual properties like colour/opacity, or purely positional ones
# like x/y). Used by _update_shape_inplace and the sub_edit path to decide
# whether a re-submitted shape needs its cached path invalidated.
_PATH_AFFECTING_KEYS = frozenset({
    "vertices", "size", "width", "height", "border_radius",
    "curve_mode", "bend", "bend_amount", "warp", "exclude", "symmetry",
    "rotation", "custom_vertices",
})


def _hashable(value: Any) -> Any:
    """Turn a possibly nested list/dict structure into something hashable and
    comparable, so it can be used inside a path-cache signature tuple."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value


def _get_shape_path(s: ShapeDef, ox: float, oy: float, sw: int, sh: int, rot: float, pivot: Optional[Tuple[float, float]] = None) -> "QPainterPath":
    """
    Dirty-rendering path cache (Phase 2.2 — "if not shape.dirty: skip").

    Rebuilding a shape's QPainterPath (polygon generation, bend/curve
    deformation, symmetry, exclude holes) is by far the expensive part of
    drawing a shape — much more expensive than simply moving it. In a scene
    with hundreds/thousands of shapes, most of them are either fully static
    or only being translated (drag, scroll, position-only motion) on any
    given frame. For those we reuse the last built path and cheaply call
    QPainterPath.translated() instead of rebuilding it from scratch.

    Cache invalidation:
      - s.dirty is True → explicit signal that a geometry-affecting field
        changed (set by _update_shape_inplace / sub_edit / on creation).
      - the signature key changed → defensive fallback in case something
        mutated the shape without going through the tracked paths above.
      - the shape has per-frame motion-driven custom vertices → always
        rebuilt, never cached (those are meant to change every frame).
      - `pivot` is given (an external ip: rotation center) → always
        rebuilt, never cached or translated. The translate() shortcut only
        holds when the rotation pivot moves in lockstep with (ox, oy) —
        i.e. the shape's own center — which isn't true for an external,
        independently-moving anchor point.
    """
    motion_state = getattr(s, "_last_motion_state", None)
    if pivot is not None or (motion_state and ("vertices" in motion_state or "vertices_count" in motion_state)):
        return _build_shape_path(s, ox, oy, sw, sh, rotation=rot, pivot=pivot, motion_state=motion_state)

    key = (
        sw, sh, rot,
        s.vertices, s.curve_mode,
        _hashable(s.bend), s.bend_amount,
        _hashable(s.exclude), _hashable(s.symmetry),
        str(s.border_radius_raw),
    )

    cache = s._path_cache
    if (not s.dirty) and cache is not None and s._path_cache_key == key:
        origin_x, origin_y = s._path_cache_origin
        if origin_x == ox and origin_y == oy:
            return cache
        return cache.translated(ox - origin_x, oy - origin_y)

    # Cache miss (dirty, or the signature changed) — full rebuild.
    path = _build_shape_path(s, ox, oy, sw, sh, rotation=rot, motion_state=motion_state)
    s._path_cache = path
    s._path_cache_key = key
    s._path_cache_origin = (ox, oy)
    s.dirty = False
    return path


# ── position helpers ──────────────────────────────────────────────────────────

def _align_pos(
    align: str, sw: int, sh: int, cw: int, ch: int,
    window_tag: Optional[str] = None,
) -> Tuple[float, float]:
    from Draw._align import calculate_alignment_pos
    return calculate_alignment_pos(align, float(sw), float(sh), float(cw), float(ch), window_tag=window_tag)


def _shape_preferred_pos(
    s: ShapeDef, cw: int, ch: int, window_tag: Optional[str] = None
) -> Tuple[int, int, float, float]:
    """Return (sw, sh, ox, oy).

    Phase 2.3: cached — see ShapeDef._bbox_cache. size_raw is hashed via
    _hashable() since it may be a list (e.g. [200, 300]).

    Dynamic ip: anchors — when `align` is an "ip:other_ip" reference (see
    Draw._bridge.resolve_point_ref), this shape's position depends on
    another object's live geometry, which the cache key below doesn't
    track — so that case always recomputes and never touches
    s._bbox_cache, same rationale as _get_shape_path's pivot bypass.
    """
    is_dynamic_align = isinstance(s.align, str) and s.align.startswith("ip:")

    key = (cw, ch, _hashable(s.size_raw), s.x, s.y, s.align)
    if (not is_dynamic_align) and s._bbox_cache is not None and s._bbox_cache_key == key:
        return s._bbox_cache

    size_raw = s.size_raw
    if isinstance(size_raw, (list, tuple)) and len(size_raw) >= 2:
        sw = _parse_size(size_raw[0], cw)
        sh = _parse_size(size_raw[1], ch)
    elif size_raw is not None:
        sw = sh = _parse_size(size_raw, min(cw, ch))
    else:
        sw = _parse_size(None, cw, 0.5)
        sh = _parse_size(None, ch, 0.5)

    if s.x is not None and s.y is not None:
        ox, oy = float(s.x), float(s.y)
    elif s.align is not None:
        ox, oy = _align_pos(s.align, sw, sh, cw, ch, window_tag=window_tag)
    else:
        ox, oy = (cw - sw) / 2.0, (ch - sh) / 2.0

    result = (sw, sh, ox, oy)
    if not is_dynamic_align:
        s._bbox_cache = result
        s._bbox_cache_key = key
    return result


# ── gradient brush builder ────────────────────────────────────────────────────

def _build_gradient_brush(
    grad_info: dict,
    ox: float, oy: float,
    sw: int, sh: int,
) -> Optional[QBrush]:
    """Build a QBrush from a resolved gradient dict."""
    grad_type = grad_info.get("type", "linear")
    stops = grad_info.get("stops", [])
    if not stops:
        return None

    cx = ox + sw / 2.0
    cy = oy + sh / 2.0

    if grad_type == "linear":
        angle = grad_info.get("angle", 0)
        rad = math.radians(angle)
        half_w = sw / 2.0
        half_h = sh / 2.0
        dx = math.cos(rad) * half_w
        dy = math.sin(rad) * half_h
        gradient = QLinearGradient(
            QPointF(cx - dx, cy - dy),
            QPointF(cx + dx, cy + dy),
        )
    elif grad_type == "radial":
        center = grad_info.get("center", [50, 50])
        radius_pct = grad_info.get("radius", 50)
        gcx = ox + sw * (center[0] / 100.0)
        gcy = oy + sh * (center[1] / 100.0)
        grad_radius = min(sw, sh) * (radius_pct / 100.0)
        gradient = QRadialGradient(QPointF(gcx, gcy), max(1.0, grad_radius))
    elif grad_type == "conical":
        center = grad_info.get("center", [50, 50])
        angle = grad_info.get("angle", 0)
        gcx = ox + sw * (center[0] / 100.0)
        gcy = oy + sh * (center[1] / 100.0)
        gradient = QConicalGradient(QPointF(gcx, gcy), angle)
    else:
        return None

    for pos, rgb in stops:
        gradient.setColorAt(pos, QColor(rgb[0], rgb[1], rgb[2]))

    return QBrush(gradient)


# ── rendering ─────────────────────────────────────────────────────────────────

def _draw_one_shape(
    painter: QPainter,
    s: ShapeDef,
    cw: int,
    ch: int,
    position_override: Optional[Tuple[float, float]] = None,
    motion_state: Optional[dict] = None,   # kept for canvas compatibility
    canvas: Optional[QWidget] = None,
) -> None:
    """Paint one ShapeDef. Called from _DrawCanvas.paintEvent."""
    from Draw._colour import color as _color_registry

    cell_rect = None
    if s.layout is not None and s.cell is not None:
        from Draw._layout import set as _layout_registry
        try:
            if isinstance(s.cell, str):
                layout_obj = _layout_registry.resolve(s.cell)
                cell_rect = layout_obj.cell_rect(cw, ch, (0, 0))
            else:
                layout_obj = _layout_registry.resolve(s.layout)
                cell_rect = layout_obj.cell_rect(cw, ch, s.cell)
        except Exception:
            pass

    if cell_rect is not None:
        sw, sh, ox, oy = _shape_preferred_pos(s, int(cell_rect.width()), int(cell_rect.height()))
        if position_override is not None:
            ox, oy = position_override
        else:
            # BUGFIX: ox/oy from _shape_preferred_pos are local to the cell
            # (0,0 == cell origin). They must be translated by the cell's
            # own position on the canvas, or shapes always paint at the
            # canvas origin instead of inside their assigned grid cell.
            ox += cell_rect.left()
            oy += cell_rect.top()
    else:
        sw, sh, ox, oy = _shape_preferred_pos(s, cw, ch)
        if position_override is not None:
            ox, oy = position_override

    # Apply any motion geometry overrides (from motion system if still used)
    if motion_state:
        if "x" in motion_state:
            ox = float(motion_state["x"])
        if "y" in motion_state:
            oy = float(motion_state["y"])
        if "width" in motion_state:
            sw = max(1, int(motion_state["width"]))
        if "height" in motion_state:
            sh = max(1, int(motion_state["height"]))
        if "scale_x" in motion_state:
            new_sw = max(1, int(round(sw * float(motion_state["scale_x"]))))
            ox -= (new_sw - sw) / 2.0
            sw = new_sw
        if "scale_y" in motion_state:
            new_sh = max(1, int(round(sh * float(motion_state["scale_y"]))))
            oy -= (new_sh - sh) / 2.0
            sh = new_sh

    rot = float(motion_state.get("rotation", s.rotation)) if motion_state else s.rotation
    pivot = (motion_state.get("rotation_center") if motion_state else None) or getattr(s, "rotation_center", getattr(s, "pivot", None))
    s.last_position = (ox, oy)
    s.last_size = (sw, sh)
    s.last_rotation = rot
    s.last_rotation_pivot = pivot

    # ── image rendering ───────────────────────────────────────────────
    if s.shape_type == "image" and s.src:
        from PySide6.QtCore import QRectF
        pixmap = _get_cached_pixmap(s.src)
        if pixmap is not None and not pixmap.isNull():
            painter.save()
            eff_opacity = motion_state.get("opacity", s.opacity) if motion_state else s.opacity
            painter.setOpacity(eff_opacity / 100.0)
            if rot != 0:
                pcx, pcy = pivot if pivot is not None else (ox + sw / 2, oy + sh / 2)
                painter.translate(pcx, pcy)
                painter.rotate(rot)
                painter.translate(-pcx, -pcy)
            # Apply border radius clipping if specified
            br = _parse_border_radius(s.border_radius_raw, sw, sh) if s.border_radius_raw else 0
            if br > 0:
                clip_path = QPainterPath()
                clip_path.addRoundedRect(QRectF(ox, oy, sw, sh), br, br)
                painter.setClipPath(clip_path)
            from Draw.debug import debug as _debug_manager
            _debug_manager.gpu_uploads += 1
            painter.drawPixmap(int(ox), int(oy), sw, sh, pixmap)
            painter.restore()
        return

    # ── video rendering ───────────────────────────────────────────────
    if s.shape_type == "video" and s.src:
        from PySide6.QtCore import QUrl, QRectF
        painter.save()
        eff_opacity = motion_state.get("opacity", s.opacity) if motion_state else s.opacity
        painter.setOpacity(eff_opacity / 100.0)
        # Lazy-init: attach a media player and frame grabber to the ShapeDef
        if not hasattr(s, '_video_player') or s._video_player is None:
            try:
                from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
                player = QMediaPlayer()
                audio = QAudioOutput()
                sink = QVideoSink()
                player.setAudioOutput(audio)
                player.setVideoSink(sink)
                player.setSource(QUrl.fromLocalFile(str(s.src)))
                s._video_player = player
                s._video_audio = audio
                s._video_sink = sink
                s._video_frame = None
                s._video_canvas_ref = None
                # Connect frame capture — also trigger canvas repaint
                def _on_frame(frame, _s=s):
                    if frame.isValid():
                        _s._video_frame = frame.toImage()
                        canvas_ref = getattr(_s, '_video_canvas_ref', None)
                        if canvas_ref is not None:
                            try:
                                canvas_ref.update()
                            except Exception:
                                pass
                sink.videoFrameChanged.connect(_on_frame)
                if getattr(s, '_video_muted', False):
                    audio.setMuted(True)
                if getattr(s, '_video_loop', True):
                    player.setLoops(QMediaPlayer.Loops.Infinite)
                if getattr(s, '_video_autoplay', True):
                    player.play()
            except Exception as e:
                _logger.warning("Draw.video: failed to init player: %s", e)
                s._video_player = False  # mark as failed
        # Store canvas ref so the frame callback can trigger repaints
        if getattr(s, '_video_canvas_ref', None) is None:
            s._video_canvas_ref = canvas
        # Paint the latest captured frame
        if hasattr(s, '_video_frame') and s._video_frame is not None:
            from PySide6.QtGui import QPixmap
            if rot != 0:
                pcx, pcy = pivot if pivot is not None else (ox + sw / 2, oy + sh / 2)
                painter.translate(pcx, pcy)
                painter.rotate(rot)
                painter.translate(-pcx, -pcy)
            br = _parse_border_radius(s.border_radius_raw, sw, sh) if s.border_radius_raw else 0
            if br > 0:
                clip_path = QPainterPath()
                clip_path.addRoundedRect(QRectF(ox, oy, sw, sh), br, br)
                painter.setClipPath(clip_path)
            pixmap = QPixmap.fromImage(s._video_frame)
            painter.drawPixmap(int(ox), int(oy), sw, sh, pixmap)
        elif s._video_player is not False:
            # Not yet decoded — draw dark placeholder
            from PySide6.QtCore import QRectF
            painter.setBrush(QBrush(QColor(20, 20, 20)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(ox, oy, sw, sh))
        painter.restore()
        return

    path = _get_shape_path(s, ox, oy, sw, sh, rot, pivot=pivot)

    # ── apply mesh warp deformation ───────────────────────────────────
    # (always re-applied fresh — warp is position-dependent and cheap
    # relative to a full path rebuild, so it isn't part of the path cache)
    if getattr(s, "warp", None):
        path = _apply_warp_to_path(path, s.warp, ox, oy, sw, sh)

    # ── resolve dynamic colors from Draw.color() registry ────────────
    color_data = None
    if s.ip and _color_registry.has_binding(s.ip):
        color_data = _color_registry.resolve_for_shape(
            s.ip,
            shape_x=ox, shape_y=oy,
            shape_w=float(sw), shape_h=float(sh),
        )

    # ── determine effective colors ────────────────────────────────────
    if color_data:
        # Body color
        body_rgba = color_data.get("body_color")
        if body_rgba:
            body_color = QColor(body_rgba[0], body_rgba[1], body_rgba[2], body_rgba[3])
        else:
            body_color = s.color

        # Body gradient (overrides solid body color)
        body_gradient = color_data.get("body_gradient")

        # Border
        border_rgba = color_data.get("border_color")
        if border_rgba:
            eff_border_color = QColor(border_rgba[0], border_rgba[1], border_rgba[2], border_rgba[3])
        else:
            eff_border_color = s.border_color
        eff_border_width = color_data.get("border_width", s.border_width)
        eff_border_style = color_data.get("border_style", s.border_style)

        # Shadow
        shadow_rgba = color_data.get("shadow_color")
        shadow_blur = color_data.get("shadow_blur", 0)
        shadow_offset = color_data.get("shadow_offset", (0, 0))

        # Glow
        glow_rgba = color_data.get("glow_color")
        glow_blur = color_data.get("glow_blur", 0)
    else:
        body_color = motion_state.get("color", s.color) if (motion_state and "color" in motion_state) else s.color
        body_gradient = None
        eff_border_color = s.border_color
        eff_border_width = s.border_width
        eff_border_style = s.border_style
        shadow_rgba = None
        shadow_blur = 0
        shadow_offset = (0, 0)
        glow_rgba = None
        glow_blur = 0

    # ── draw shadow/glow (cached to pixmap) ───────────────────────────
    has_shadow = shadow_rgba and shadow_blur > 0
    has_glow = glow_rgba and glow_blur > 0
    if has_shadow or has_glow:
        cache_key = (shadow_rgba, shadow_blur, shadow_offset, glow_rgba, glow_blur, sw, sh, rot)
        if s._shadow_cache_key != cache_key or s._shadow_cache is None:
            # Compute expand size for shadow/glow buffer
            expand = max(shadow_blur * 2 if has_shadow else 0, glow_blur * 4 if has_glow else 0)
            sdx = shadow_offset[0] if has_shadow else 0
            sdy = shadow_offset[1] if has_shadow else 0
            buf_w = int(sw + expand + abs(sdx) + 2)
            buf_h = int(sh + expand + abs(sdy) + 2)
            from PySide6.QtGui import QPixmap as _QPixmap
            from PySide6.QtCore import QRectF as _QRectF
            cache_pm = _QPixmap(buf_w, buf_h)
            cache_pm.fill(Qt.GlobalColor.transparent)
            cp = QPainter(cache_pm)
            cp.setRenderHint(QPainter.RenderHint.Antialiasing)
            off_x = expand / 2 + (abs(sdx) if sdx < 0 else 0)
            off_y = expand / 2 + (abs(sdy) if sdy < 0 else 0)
            # Draw shadow into cache
            if has_shadow:
                shadow_color = QColor(shadow_rgba[0], shadow_rgba[1], shadow_rgba[2], shadow_rgba[3])
                blur_steps = max(1, int(shadow_blur / 2))
                for i in range(blur_steps, 0, -1):
                    t_val = i / blur_steps
                    alpha = int(shadow_color.alpha() * (1.0 - t_val) * 0.5)
                    if alpha <= 0:
                        continue
                    sc = QColor(shadow_color.red(), shadow_color.green(), shadow_color.blue(), alpha)
                    exp = t_val * shadow_blur
                    sp = _build_shape_path(s, off_x + sdx - exp/2, off_y + sdy - exp/2,
                                           int(sw + exp), int(sh + exp), rotation=rot)
                    cp.setPen(Qt.PenStyle.NoPen)
                    cp.setBrush(QBrush(sc))
                    cp.drawPath(sp)
            # Draw glow into cache
            if has_glow:
                glow_color = QColor(glow_rgba[0], glow_rgba[1], glow_rgba[2], glow_rgba[3])
                glow_steps = max(1, int(glow_blur / 2))
                for i in range(glow_steps, 0, -1):
                    t_val = i / glow_steps
                    alpha = int(glow_color.alpha() * (1.0 - t_val) * 0.4)
                    if alpha <= 0:
                        continue
                    gc = QColor(glow_color.red(), glow_color.green(), glow_color.blue(), alpha)
                    exp = t_val * glow_blur * 2
                    gp = _build_shape_path(s, off_x - exp/2, off_y - exp/2,
                                           int(sw + exp), int(sh + exp), rotation=rot)
                    cp.setPen(Qt.PenStyle.NoPen)
                    cp.setBrush(QBrush(gc))
                    cp.drawPath(gp)
            cp.end()
            s._shadow_cache = cache_pm
            s._shadow_cache_key = cache_key
        # Paint cached shadow/glow
        expand = max(shadow_blur * 2 if has_shadow else 0, glow_blur * 4 if has_glow else 0)
        sdx = shadow_offset[0] if has_shadow else 0
        sdy = shadow_offset[1] if has_shadow else 0
        draw_x = ox - expand / 2 - (abs(sdx) if sdx < 0 else 0)
        draw_y = oy - expand / 2 - (abs(sdy) if sdy < 0 else 0)
        painter.drawPixmap(int(draw_x), int(draw_y), s._shadow_cache)

    # ── draw main shape ───────────────────────────────────────────────
    painter.save()
    eff_opacity = motion_state.get("opacity", s.opacity) if motion_state else s.opacity
    painter.setOpacity(eff_opacity / 100.0)

    if s.vertices != 2:
        # Try gradient first, then solid color
        body_c = _parse_color(body_color) if not isinstance(body_color, QColor) else body_color
        if body_gradient:
            grad_brush = _build_gradient_brush(body_gradient, ox, oy, sw, sh)
            if grad_brush:
                painter.setBrush(grad_brush)
            else:
                painter.setBrush(QBrush(body_c))
        else:
            painter.setBrush(QBrush(body_c))
    else:
        painter.setBrush(Qt.BrushStyle.NoBrush)

    style_map = {
        "solid":  Qt.PenStyle.SolidLine,
        "dashed": Qt.PenStyle.DashLine,
        "dotted": Qt.PenStyle.DotLine,
        "none":   Qt.PenStyle.NoPen,
    }
    if eff_border_width > 0 and eff_border_style != "none":
        pen_c = _parse_color(eff_border_color) if not isinstance(eff_border_color, QColor) else eff_border_color
        pen = QPen(pen_c, eff_border_width)
        pen.setStyle(style_map.get(eff_border_style, Qt.PenStyle.SolidLine))
        painter.setPen(pen)
    else:
        painter.setPen(Qt.PenStyle.NoPen)

    painter.drawPath(path)
    painter.restore()


# ── hit testing (used by mouse events) ───────────────────────────────────────

def _shape_contains_point(s: ShapeDef, point: QPointF) -> bool:
    if s.last_position is None or s.last_size is None:
        return False
    ox, oy = s.last_position
    sw, sh = s.last_size
    rot = getattr(s, "last_rotation", s.rotation)
    if rot is None:
        rot = s.rotation
    pivot = getattr(s, "last_rotation_pivot", None)
    # Hit-testing runs on every mouse-move; reuse the paint-time path cache
    # (Phase 2.2) instead of rebuilding — ox/oy/sw/sh/rot match what was
    # just painted, so this is normally a cache hit.
    path = _get_shape_path(s, ox, oy, sw, sh, rot, pivot=pivot)
    if s.vertices == 2:
        rect = path.controlPointRect().adjusted(-4.0, -4.0, 4.0, 4.0)
        return rect.contains(point)
    return path.contains(point)


def _shape_is_dynamic(s: ShapeDef) -> bool:
    """Return True if the shape has a dynamic color binding."""
    from Draw._colour import color as _color_registry
    if s.ip and _color_registry.has_binding(s.ip):
        binding = _color_registry.get_binding(s.ip)
        if binding and binding.is_dynamic:
            return True
    return False


# ── canvas compatibility stubs (used by _DrawCanvas) ────────────────────

def _shape_preferred_geometry(s: ShapeDef, cw: int, ch: int, window_tag: Optional[str] = None):
    """Compatibility shim returning (origin_x, origin_y, area_w, area_h, sw, sh, ox, oy)."""
    if s.layout is not None and s.cell is not None:
        from Draw._layout import set as _layout_registry
        try:
            if isinstance(s.cell, str):
                layout_obj = _layout_registry.resolve(s.cell)
                cell_rect = layout_obj.cell_rect(cw, ch, (0, 0))
            else:
                layout_obj = _layout_registry.resolve(s.layout)
                cell_rect = layout_obj.cell_rect(cw, ch, s.cell)

            sw, sh, ox, oy = _shape_preferred_pos(s, int(cell_rect.width()), int(cell_rect.height()), window_tag=window_tag)
            return (
                float(cell_rect.left()),
                float(cell_rect.top()),
                float(cell_rect.width()),
                float(cell_rect.height()),
                sw,
                sh,
                float(cell_rect.left() + ox),
                float(cell_rect.top() + oy),
            )
        except Exception as exc:
            _logger.warning("Draw.shapes: failed to position shape in layout cell: %s", exc)

    sw, sh, ox, oy = _shape_preferred_pos(s, cw, ch, window_tag=window_tag)
    return 0.0, 0.0, float(cw), float(ch), sw, sh, ox, oy


def _shape_hit_geometry(s: ShapeDef):
    """Compatibility shim returning (hx, hy, hsw, hsh)."""
    if s.last_position is None or s.last_size is None:
        return 0.0, 0.0, 0, 0
    ox, oy = s.last_position
    sw, sh = s.last_size
    return ox, oy, sw, sh


def _apply_motion_geometry(state: dict, ox, oy, sw, sh):
    """Compatibility shim for motion system."""
    if not state:
        return ox, oy, sw, sh
    if "x" in state:
        ox = float(state["x"])
    if "y" in state:
        oy = float(state["y"])
    if "width" in state:
        sw = max(1, int(round(float(state["width"]))))
    if "height" in state:
        sh = max(1, int(round(float(state["height"]))))
    if "scale_x" in state:
        new_sw = max(1, int(round(sw * float(state["scale_x"]))))
        ox -= (new_sw - sw) / 2.0
        sw = new_sw
    if "scale_y" in state:
        new_sh = max(1, int(round(sh * float(state["scale_y"]))))
        oy -= (new_sh - sh) / 2.0
        sh = new_sh
    return float(ox), float(oy), int(sw), int(sh)


# ── shape parser ──────────────────────────────────────────────────────────────

def _parse_area_spec(raw_area: object) -> Tuple[Tuple[float, float], Optional[str]]:
    """
    Parse area=[{"expand": (ex, ey), "move": "right"}] or area={"expand":...}.
    Returns (expand_xy, move_direction).
    expand_xy: extra pixels to add to the collision rect on each axis.
    move_direction: preferred direction to shift when overlapping, or None.
    """
    if raw_area is None:
        return (0.0, 0.0), None

    # Accept list or single dict
    spec: dict = {}
    if isinstance(raw_area, (list, tuple)):
        for item in raw_area:
            if isinstance(item, dict):
                spec.update(item)
    elif isinstance(raw_area, dict):
        spec = raw_area

    # expand
    expand_raw = spec.get("expand", (0.0, 0.0))
    if isinstance(expand_raw, (list, tuple)) and len(expand_raw) >= 2:
        ex = max(0.0, float(expand_raw[0]))
        ey = max(0.0, float(expand_raw[1]))
    elif isinstance(expand_raw, (int, float)):
        ex = ey = max(0.0, float(expand_raw))
    else:
        ex = ey = 0.0

    # move direction
    move_raw = spec.get("move", None)
    move: Optional[str] = str(move_raw).strip().lower() if move_raw is not None else None
    _VALID_MOVES = {"right", "left", "down", "up", "right_wrap", "down_wrap"}
    if move not in _VALID_MOVES:
        move = None

    return (ex, ey), move


def _init_video_player(s: ShapeDef) -> None:
    """Initialize video player components immediately on creation or update."""
    if s.shape_type == "video" and s.src:
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
            player = QMediaPlayer()
            audio = QAudioOutput()
            sink = QVideoSink()
            player.setAudioOutput(audio)
            player.setVideoSink(sink)
            player.setSource(QUrl.fromLocalFile(str(s.src)))
            s._video_player = player
            s._video_audio = audio
            s._video_sink = sink
            s._video_frame = None
            s._video_canvas_ref = None

            # Connect frame capture — also trigger canvas repaint
            def _on_frame(frame, _s=s):
                if frame.isValid():
                    _s._video_frame = frame.toImage()
                    canvas_ref = getattr(_s, '_video_canvas_ref', None)
                    if canvas_ref is not None:
                        try:
                            canvas_ref.update()
                        except Exception:
                            pass
            sink.videoFrameChanged.connect(_on_frame)

            if s._video_muted:
                audio.setMuted(True)
            if s._video_loop:
                player.setLoops(QMediaPlayer.Loops.Infinite)
            if s._video_autoplay:
                player.play()
        except Exception as e:
            _logger.warning("Draw.video: failed to init player: %s", e)
            s._video_player = False


def _parse_shape(raw: dict, ip_str: Optional[str] = None, layout_override: Optional[object] = None,
                 area_raw: object = None) -> ShapeDef:
    """Parse a shape dict into a ShapeDef."""
    if not isinstance(raw, dict):
        raise TypeError("Draw.shape: every item in 'shape' must be a dict.")
    raw = _merge_customise(raw)

    from Draw._validation import validate_keys, KNOWN_SHAPE_KEYS
    validate_keys(raw, KNOWN_SHAPE_KEYS, kind="Draw.shape", obj_id=ip_str)

    vertices_raw = raw.get("vertices", None)
    vertices: Optional[int] = None if vertices_raw is None else max(1, int(vertices_raw))

    # Size: accept "size": [w, h] or separate width/height keys for compat
    size_raw = raw.get("size", None)
    if size_raw is None:
        # fall back to separate width/height
        w = raw.get("width", None)
        h = raw.get("height", None)
        if w is not None or h is not None:
            size_raw = [w or h, h or w]

    border_radius_raw = raw.get("border_radius", None)

    x_raw = raw.get("x", None)
    y_raw = raw.get("y", None)
    x = int(x_raw) if x_raw is not None else None
    y = int(y_raw) if y_raw is not None else None

    align_raw = raw.get("align", None)
    if (
        align_raw is not None
        and align_raw not in _ALIGN_VALUES_SET
        and not (isinstance(align_raw, str) and align_raw.startswith("ip:"))
    ):
        raise ValueError(f"Draw.shape: invalid align='{align_raw}'.")

    rotation = _as_float(raw.get("rotation", 0.0))

    color       = _parse_color(raw.get("color", "blue"))
    border_color = _parse_color(raw.get("border_color", "black"))
    border_width = _as_int(raw.get("border_width", 0))
    border_style = str(raw.get("border_style", "solid")).strip().lower()
    if border_style not in {"solid", "dashed", "dotted", "none"}:
        border_style = "solid"
    opacity = max(0, min(100, _as_int(raw.get("opacity", 100), default=100)))

    curve_mode_raw = str(raw.get("curve_mode", "line")).strip().lower()
    curve_mode = curve_mode_raw if curve_mode_raw in _CURVE_MODES else "line"

    # ── custom dict: alternative nesting for bend/exclude/symmetry/properties ──
    custom_raw = raw.get("custom", None)
    custom: Optional[Dict[str, Any]] = dict(custom_raw) if isinstance(custom_raw, dict) else None

    # bend/exclude/symmetry: check top-level first, then fall back to custom dict
    bend_raw = raw.get("bend", None)
    if bend_raw is None and custom:
        bend_raw = custom.get("bend", [])
    bend: List[Dict[str, Any]] = list(bend_raw) if isinstance(bend_raw, list) else []

    bend_amount = _as_float(raw.get("bend_amount", custom.get("bend_amount", 40.0) if custom else 40.0))

    warp_raw = raw.get("warp", custom.get("warp", None) if custom else None)
    warp: Optional[List[List[Tuple[float, float]]]] = warp_raw if isinstance(warp_raw, list) else None

    exclude_raw = raw.get("exclude", None)
    if exclude_raw is None and custom:
        exclude_raw = custom.get("exclude", [])
    exclude: List[Dict[str, Any]] = list(exclude_raw) if isinstance(exclude_raw, list) else []

    symmetry_raw = raw.get("symmetry", None)
    if symmetry_raw is None and custom:
        symmetry_raw = custom.get("symmetry", None)
    symmetry: Optional[Dict[str, Any]] = dict(symmetry_raw) if isinstance(symmetry_raw, dict) else None

    # ── hitbox fields ──
    hitbox_mode_raw = raw.get("hitbox_mode", None)
    hitbox_mode: Optional[str] = str(hitbox_mode_raw).strip().lower() if hitbox_mode_raw is not None else None

    hit_box_raw = raw.get("hit_box", "shape")
    hit_box = str(hit_box_raw).strip().lower()
    if hit_box not in {"shape", "closed_rec"}:
        hit_box = "shape"

    if "z" in raw and raw["z"] is not None:
        z = _as_int(raw["z"])
    else:
        from Draw._tools import next_z
        z = int(next_z())
    overlap = _as_bool(raw.get("overlap", True))

    ip_field = raw.get("ip", None)
    if isinstance(ip_field, (list, tuple)):
        effective_ip = str(ip_field[0]) if ip_field else ip_str
    elif ip_field is not None:
        effective_ip = str(ip_field)
    else:
        effective_ip = ip_str

    layout_raw = raw.get("layout", raw.get("get_ip", layout_override))
    cell_raw = raw.get("column", raw.get("columns", None))

    parsed_cell = cell_raw
    if cell_raw is not None and not isinstance(cell_raw, str):
        from Draw._layout import _parse_cell_ref
        try:
            parsed_cell = _parse_cell_ref(cell_raw)
        except Exception:
            pass

    # ── area spec: per-shape collision zone and preferred move direction ──
    # Accept from the shape dict itself ("area" key) OR the caller's area_raw
    raw_area_spec = raw.get("area", area_raw)
    area_expand, area_move = _parse_area_spec(raw_area_spec)
    from Draw._overlap import parse_flow_spec
    flow = parse_flow_spec(
        raw.get("flow", None),
        flow_provided=("flow" in raw),
        overlap=overlap,
        area_expand=area_expand,
        area_move=area_move,
    )

    s = ShapeDef(
        vertices       = vertices,
        size_raw       = size_raw,
        border_radius_raw = border_radius_raw,
        x              = x,
        y              = y,
        align          = align_raw,
        rotation       = rotation,
        color          = color,
        border_color   = border_color,
        border_width   = border_width,
        border_style   = border_style,
        opacity        = opacity,
        curve_mode     = curve_mode,
        bend           = bend,
        bend_amount    = bend_amount,
        warp           = warp,
        exclude        = exclude,
        symmetry       = symmetry,
        hitbox_mode    = hitbox_mode,
        hit_box        = hit_box,
        custom         = custom,
        z              = z,
        overlap        = overlap,
        flow           = flow,
        ip             = effective_ip,
        layout         = layout_raw,
        cell           = parsed_cell,
        area_expand    = area_expand,
        area_move      = area_move,
        shape_type     = str(raw.get("type", "vector")).strip().lower(),
        src            = raw.get("src", None),
        _video_loop    = bool(raw.get("loop", True)),
        _video_autoplay= bool(raw.get("autoplay", True)),
        _video_muted   = bool(raw.get("muted", False)),
        inside         = raw.get("inside", None),
        move_path      = raw.get("move_path", None),
        custom_vertices = raw.get("custom_vertices", None),
    )
    if "custom_vertices" in raw and isinstance(raw["custom_vertices"], (list, tuple)):
        sw_v = 1.0
        sh_v = 1.0
        if isinstance(size_raw, (list, tuple)) and len(size_raw) >= 2:
            try:
                sw_v = float(str(size_raw[0]).rstrip("px"))
                sh_v = float(str(size_raw[1]).rstrip("px"))
            except Exception:
                pass
        ox_v = float(x if x is not None else 0)
        oy_v = float(y if y is not None else 0)
        s._last_motion_state = {
            "vertices": raw["custom_vertices"],
            "ref_x": ox_v,
            "ref_y": oy_v,
            "ref_w": sw_v,
            "ref_h": sh_v,
        }
    _init_video_player(s)
    return s



# ── Draw.shape(..., properties=["builder"]) — drag-to-size creation ──────────
#
# Normal Draw.shape() calls place a shape immediately at whatever size/x/y
# was given. When the "builder" property is active and a shape dict has NO
# width/height/size, it isn't placed at all yet — it's queued here, and the
# next left-click drag the user makes on that canvas defines its corners.
# The shape previews live while dragging and is finalized with an ABSOLUTE
# pixel size (size_raw = [w, h]) on release — never a relative/percentage
# size, since a drag always measures literal on-screen pixels.

def _start_builder_preview(canvas: "_DrawCanvas", entry: dict, pos: "QPointF") -> None:
    """Begin a drag-to-size shape: create a 1x1 preview at the press point."""
    raw = dict(entry["raw"])
    raw["ip"] = entry["ip"]
    raw["x"] = int(pos.x())
    raw["y"] = int(pos.y())
    raw["size"] = [1, 1]
    shape_def = _parse_shape(raw, entry["ip"])
    shape_def.dirty = True
    canvas.shape_items.append(shape_def)
    canvas._z_order_dirty = True
    canvas._shape_by_ip[entry["ip"]] = shape_def
    canvas._occupied_dirty = True
    entry["shape_def"] = shape_def
    entry["start"] = (pos.x(), pos.y())
    canvas.update()


def _update_builder_preview(canvas: "_DrawCanvas", entry: dict, pos: "QPointF") -> None:
    """Live-resize the preview shape while the drag is in progress."""
    s = entry.get("shape_def")
    if s is None:
        return
    x0, y0 = entry["start"]
    x1, y1 = pos.x(), pos.y()
    nx, ny = min(x0, x1), min(y0, y1)
    nw, nh = max(1, abs(x1 - x0)), max(1, abs(y1 - y0))
    s.x, s.y = int(nx), int(ny)
    s.size_raw = [nw, nh]
    s.dirty = True
    canvas.update()


def _finalize_builder_shape(canvas: "_DrawCanvas", entry: dict, pos: "QPointF") -> None:
    """Lock in the shape's final ABSOLUTE size/position on release."""
    s = entry.get("shape_def")
    if s is None:
        return
    x0, y0 = entry["start"]
    x1, y1 = pos.x(), pos.y()
    nx, ny = min(x0, x1), min(y0, y1)
    nw, nh = max(1, abs(x1 - x0)), max(1, abs(y1 - y0))
    s.x, s.y = int(nx), int(ny)
    s.size_raw = [nw, nh]     # absolute pixel size — drag is finished
    s.dirty = True
    canvas._occupied_dirty = True
    canvas._shape_hash_by_ip[entry["ip"]] = _compute_content_hash(dict(entry["raw"]))
    canvas.update()
    on_build = entry.get("on_build")
    if callable(on_build):
        on_build(entry["ip"], int(nx), int(ny), int(nw), int(nh))


# ── place_with_ip geometry resolver ──────────────────────────────────────────

def _resolve_place_with_ip(
    place_ip: str,
    hit_box_ip: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Return a placement constraint dict from a registered hitbox or shape ip.

    Returns a dict with keys: x, y, width, height, mode
    or None if the ip cannot be resolved.

    Priority:
      1. Draw.hitbox registry (explicit hitbox definitions)
      2. ShapeDef.last_position / last_size (shapes already drawn)
    """
    # Check hitbox registry first
    hb = hitbox.get(place_ip)
    if hb is not None:
        return {
            "x":      hb.get("x"),
            "y":      hb.get("y"),
            "width":  hb.get("width"),
            "height": hb.get("height"),
            "mode":   hb.get("mode", "resetgeometry"),
        }

    # Try to find a drawn shape with this ip across all windows
    try:
        from Draw._window import window as _wr
        for tag in _wr.list_all_tags():
            win = _wr.get(tag)
            if hasattr(win, "_draw_canvas"):
                for s in win._draw_canvas.shape_items:
                    if s.ip == place_ip and s.last_position and s.last_size:
                        ox, oy = s.last_position
                        sw, sh = s.last_size
                        return {
                            "x": ox, "y": oy,
                            "width": sw, "height": sh,
                            "mode": "resetgeometry",
                        }
    except Exception:
        pass
    return None


def _apply_place_with_ip(
    shapes_raw: List[dict],
    place_ip: str,
    hit_box_ip: Optional[str] = None,
) -> List[dict]:
    """
    Offset every shape's x/y by the region origin so that shapes written
    with region-relative coordinates land at the correct screen position.

    Coordinates in shape dicts are treated as *region-relative* (i.e. 0,0
    means the top-left corner of the region).  This function adds the region
    origin (rx, ry) to whatever x/y the shape specifies.  If no x/y is given
    the shape lands at the region origin.

    Both resetgeometry and fullgeometry hitboxes are supported — the only
    difference is that fullgeometry also constrains width/height to the region
    dimensions when none are supplied.
    """
    region = _resolve_place_with_ip(place_ip, hit_box_ip)
    if region is None:
        _logger.warning("Draw.shapes: place_with_ip='%s' not found — shapes drawn without placement.", place_ip)
        return shapes_raw

    rx = _parse_px_value(region.get("x"), 0.0)
    ry = _parse_px_value(region.get("y"), 0.0)
    rw = _parse_px_value(region.get("width"), 0.0)
    rh = _parse_px_value(region.get("height"), 0.0)
    mode = str(region.get("mode", "resetgeometry")).strip().lower()

    result = []
    for raw in shapes_raw:
        raw = dict(raw)

        # x/y are region-relative — offset by region origin
        local_x = float(raw["x"]) if raw.get("x") is not None else 0.0
        local_y = float(raw["y"]) if raw.get("y") is not None else 0.0
        raw["x"] = int(rx + local_x)
        raw["y"] = int(ry + local_y)

        # In fullgeometry mode also default width/height to region size
        if mode == "fullgeometry":
            if raw.get("width") is None and raw.get("size") is None:
                raw["width"] = int(rw)
            if raw.get("height") is None and raw.get("size") is None:
                raw["height"] = int(rh)

        raw["_place_region"] = {"x": rx, "y": ry, "w": rw, "h": rh, "mode": mode}
        result.append(raw)
    return result


# ── scroller widget builder ───────────────────────────────────────────────────

_SCROLLER_PLACEMENTS = {"top", "bottom", "left", "right"}


def _build_scroller_shapes(
    scroller_spec: List[dict],
    canvas_w: int,
    canvas_h: int,
    place_region: Optional[Dict[str, Any]] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Build shape + text dicts for a scroller widget.
    Returns (shape_items, text_items, scroller_configs).
    scroller_configs is a list of dicts used by _DrawCanvas to update thumb positions.
    """
    shape_items: List[dict] = []
    text_items: List[dict] = []
    scroll_configs: List[dict] = []

    ref_x = float(place_region["x"]) if place_region else 0.0
    ref_y = float(place_region["y"]) if place_region else 0.0
    ref_w = float(place_region["w"]) if place_region else float(canvas_w)
    ref_h = float(place_region["h"]) if place_region else float(canvas_h)

    for spec in scroller_spec:
        place = str(spec.get("place", "right")).strip().lower()
        if place not in _SCROLLER_PLACEMENTS:
            place = "right"

        track_color   = spec.get("color", "#2a2f3a")
        thumb_color   = spec.get("thumb_color", "#5b6880")
        border_width  = _as_int(spec.get("border_width", 0))
        border_color  = spec.get("border_color", "#000000")
        track_opacity = _as_int(spec.get("opacity", 100), default=100)
        z_val         = _as_int(spec.get("z", 20))
        track_ip      = spec.get("ip", f"scroller_{place}_track")
        thumb_ip      = spec.get("thumb_ip", f"scroller_{place}_thumb")

        if place in ("top", "bottom"):
            # Horizontal scroller
            track_h = _parse_px_value(spec.get("height", spec.get("width", None)), 12.0)
            track_w = ref_w
            track_x = ref_x
            track_y = ref_y if place == "top" else ref_y + ref_h - track_h

            thumb_w = max(30.0, track_w * 0.2)
            thumb_h = track_h
            thumb_x = track_x  # starts at left
            thumb_y = track_y
        else:
            # Vertical scroller (left / right)
            track_w = _parse_px_value(spec.get("width", spec.get("height", None)), 12.0)
            track_h = ref_h
            track_x = ref_x if place == "left" else ref_x + ref_w - track_w
            track_y = ref_y

            thumb_w = track_w
            thumb_h = max(30.0, track_h * 0.2)
            thumb_x = track_x
            thumb_y = track_y  # starts at top

        # Track shape (foreground overlay layer)
        track = {
            "vertices": 4,
            "border_radius": float(track_w) / 2.0 if place in ("left", "right") else float(track_h) / 2.0,
            "x": int(track_x),
            "y": int(track_y),
            "width": int(track_w),
            "height": int(track_h),
            "color": track_color,
            "border_width": border_width,
            "border_color": border_color,
            "opacity": track_opacity,
            "z": -999,
            "overlap": True,
            "ip": track_ip,
        }
        # Thumb shape (top-most interactive layer)
        thumb = {
            "vertices": 4,
            "border_radius": float(thumb_w) / 2.0 if place in ("left", "right") else float(thumb_h) / 2.0,
            "x": int(thumb_x),
            "y": int(thumb_y),
            "width": int(thumb_w),
            "height": int(thumb_h),
            "color": thumb_color,
            "border_width": 0,
            "opacity": track_opacity,
            "z": -1000,
            "overlap": True,
            "ip": thumb_ip,
        }
        shape_items.extend([track, thumb])

        # Config for dynamic thumb positioning
        scroll_configs.append({
            "track_ip":  track_ip,
            "thumb_ip":  thumb_ip,
            "direction": "vertical" if place in ("left", "right") else "horizontal",
            "track_x":   float(track_x),
            "track_y":   float(track_y),
            "track_w":   float(track_w),
            "track_h":   float(track_h),
            "thumb_w":   float(thumb_w),
            "thumb_h":   float(thumb_h),
        })

    return shape_items, text_items, scroll_configs


# ── overlap / placement (lightweight version) ─────────────────────────────────

def _find_non_overlapping_pos(
    sw: int, sh: int,
    ox: float, oy: float,
    occupied: List[Tuple[float, float, int, int]],
    cw: int, ch: int,
    spacing: float = 4.0,
) -> Tuple[float, float]:
    """Simple horizontal-stack placement when overlap=False."""
    tx, ty = ox, oy
    for _ in range(1000):
        collision = False
        for ex, ey, ew, eh in occupied:
            if (tx < ex + ew + spacing and tx + sw + spacing > ex and
                    ty < ey + eh + spacing and ty + sh + spacing > ey):
                tx = ex + ew + spacing
                collision = True
                break
        if not collision:
            if tx + sw <= cw and ty + sh <= ch:
                return tx, ty
            # wrap
            tx = 0
            ty += sh + spacing
            if ty + sh > ch:
                return ox, oy
    return ox, oy


# ── incremental rendering helpers ─────────────────────────────────────────────

def _compute_content_hash(raw: dict) -> int:
    """
    Compute a cheap, stable hash of a shape dict's key visual and layout
    properties.  Called on every re-submitted shape; if the hash matches the
    cached value the shape update is skipped entirely (zero allocation).

    Falls back to ``id(raw)`` (always "changed") when any value is unhashable,
    which keeps correctness at the cost of always updating that shape.
    """
    try:
        size = raw.get("size")
        size_str = (
            str(size) if size is not None
            else str(raw.get("width", "")) + "x" + str(raw.get("height", ""))
        )
        return hash((
            raw.get("vertices"),
            raw.get("x"),
            raw.get("y"),
            raw.get("align"),
            size_str,
            str(raw.get("border_radius", 0)),
            str(raw.get("color", "")),
            str(raw.get("border_color", "")),
            raw.get("border_width", 0),
            raw.get("border_style", "solid"),
            raw.get("opacity", 100),
            raw.get("rotation", 0.0),
            raw.get("z", 0),
            raw.get("overlap", True),
            raw.get("curve_mode", "line"),
            str(raw.get("bend", "")),
            raw.get("bend_amount", 40.0),
            str(raw.get("warp", "")),
            str(raw.get("exclude", "")),
            str(raw.get("symmetry", "")),
        ))
    except TypeError:
        return id(raw)


def _is_layout_affecting(old: ShapeDef, raw_new: dict) -> bool:
    """
    Return True when the diff between *old* (an existing ShapeDef) and
    *raw_new* (the incoming raw dict) will change the shape's on-screen
    position or bounding-box size.  Only layout-affecting changes require an
    overlap-registry rebuild (_occupied_dirty = True).

    Pure visual changes — colour, opacity, border style, rotation, z-order —
    are layout-neutral and do not require a rebuild.
    """
    new_x = int(raw_new["x"]) if raw_new.get("x") is not None else None
    new_y = int(raw_new["y"]) if raw_new.get("y") is not None else None
    new_align = raw_new.get("align")

    vertices_raw = raw_new.get("vertices")
    new_vertices = None if vertices_raw is None else max(1, int(vertices_raw))

    new_size = raw_new.get("size")
    if new_size is None:
        w = raw_new.get("width")
        h = raw_new.get("height")
        if w is not None or h is not None:
            new_size = [w, h]

    return (
        old.x        != new_x        or
        old.y        != new_y        or
        old.align    != new_align    or
        old.vertices != new_vertices or
        str(old.size_raw) != str(new_size) or
        "flow" in raw_new             or
        "area" in raw_new             or
        "border_radius" in raw_new
    )


def _update_shape_inplace(existing: ShapeDef, raw: dict) -> None:
    """
    Merge a raw shape dict into an already-parsed ``ShapeDef``, updating only
    the fields that are explicitly present in *raw*.

    This avoids the cost of re-parsing and re-allocating the object on every
    ``Draw.shapes()`` call.  When combined with ``_is_layout_affecting`` it
    also prevents unnecessary overlap-registry rebuilds for visual-only changes.

    Also drives Phase 2.2 dirty rendering: if any of the incoming keys can
    change the shape's QPainterPath (_PATH_AFFECTING_KEYS), ``existing.dirty``
    is set True so the next paint rebuilds the path instead of reusing the
    cached one. Purely visual (colour/opacity/...) or position-only (x/y)
    changes leave dirty untouched — the cached path is still valid, only its
    on-screen offset changes.
    """
    raw = _merge_customise(raw)
    if _PATH_AFFECTING_KEYS.intersection(raw.keys()):
        existing.dirty = True
    existing._gpu_cache = None
    if "color" in raw or "opacity" in raw:
        existing._cached_rgba = None
        existing._cached_rgba_key = None

    # ── visual properties ────────────────────────────────────────────────
    if "color" in raw:
        existing.color = _parse_color(raw["color"])
    if "border_color" in raw:
        existing.border_color = _parse_color(raw["border_color"])
    if "border_width" in raw:
        existing.border_width = _as_int(raw["border_width"])
    if "border_style" in raw:
        bs = str(raw["border_style"]).strip().lower()
        existing.border_style = bs if bs in {"solid", "dashed", "dotted", "none"} else "solid"
    if "opacity" in raw:
        existing.opacity = max(0, min(100, _as_int(raw["opacity"], default=100)))
    if "rotation" in raw:
        existing.rotation = _as_float(raw["rotation"])
    if "z" in raw:
        existing.z = _as_int(raw["z"])
    if "overlap" in raw:
        existing.overlap = _as_bool(raw["overlap"])
    if "border_radius" in raw:
        existing.border_radius_raw = raw["border_radius"]
    if "curve_mode" in raw:
        cm = str(raw["curve_mode"]).strip().lower()
        existing.curve_mode = cm if cm in _CURVE_MODES else "line"

    # ── geometry properties ──────────────────────────────────────────────
    if "x" in raw:
        existing.x = int(raw["x"]) if raw["x"] is not None else None
    if "y" in raw:
        existing.y = int(raw["y"]) if raw["y"] is not None else None
    if "align" in raw:
        a = raw["align"]
        existing.align = str(a) if (a is None or a in _ALIGN_VALUES_SET) else None
    if "vertices" in raw:
        v = raw["vertices"]
        existing.vertices = None if v is None else max(1, int(v))

    w_raw    = raw.get("width")
    h_raw    = raw.get("height")
    size_raw = raw.get("size")
    if size_raw is not None:
        existing.size_raw = size_raw
    elif w_raw is not None or h_raw is not None:
        old_size = existing.size_raw
        if isinstance(old_size, (list, tuple)) and len(old_size) >= 2:
            old_w, old_h = old_size[0], old_size[1]
        else:
            old_w = old_h = old_size
        existing.size_raw = [
            w_raw if w_raw is not None else old_w,
            h_raw if h_raw is not None else old_h,
        ]

    # ── custom_vertices (pie/donut wedges, arbitrary polygons) ─────────────
    # Re-submitted shapes (same ip, e.g. an animation tick, a live-data
    # refresh, or a drag reposition) go through this in-place path instead
    # of _parse_shape. Without this block, a new custom_vertices list on a
    # re-submission was silently dropped: existing.custom_vertices stayed
    # frozen at its first value while existing.vertices (the plain point
    # *count*, handled above) kept updating — and _build_shape_path falls
    # back to drawing a REGULAR n-gon from that stale count once the
    # motion-state remap below also goes stale, which is what turned pie/
    # donut wedges into one solid n-gon "circle" after the first repaint.
    if "custom_vertices" in raw:
        cv = raw["custom_vertices"]
        existing.custom_vertices = cv if isinstance(cv, (list, tuple)) else None
        if isinstance(cv, (list, tuple)):
            sw_v = 1.0
            sh_v = 1.0
            cur_size = existing.size_raw
            if isinstance(cur_size, (list, tuple)) and len(cur_size) >= 2:
                try:
                    sw_v = float(str(cur_size[0]).rstrip("px"))
                    sh_v = float(str(cur_size[1]).rstrip("px"))
                except Exception:
                    pass
            ox_v = float(existing.x if existing.x is not None else 0)
            oy_v = float(existing.y if existing.y is not None else 0)
            existing._last_motion_state = {
                "vertices": cv,
                "ref_x": ox_v,
                "ref_y": oy_v,
                "ref_w": sw_v,
                "ref_h": sh_v,
            }
        else:
            existing._last_motion_state = None

    # ── structural deformers ──────────────────────────────────────────────
    if "bend" in raw:
        existing.bend = list(raw["bend"]) if isinstance(raw["bend"], list) else []
    if "bend_amount" in raw:
        existing.bend_amount = _as_float(raw["bend_amount"])
    if "warp" in raw:
        existing.warp = raw["warp"] if isinstance(raw["warp"], list) else None
    if "exclude" in raw:
        existing.exclude = list(raw["exclude"]) if isinstance(raw["exclude"], list) else []
    if "symmetry" in raw:
        sym = raw["symmetry"]
        existing.symmetry = dict(sym) if isinstance(sym, dict) else None

    # ── flow / overlap spec ───────────────────────────────────────────────
    if "flow" in raw or "area" in raw or "overlap" in raw:
        raw_area_spec = raw.get("area")
        area_expand, area_move = _parse_area_spec(raw_area_spec)
        from Draw._overlap import parse_flow_spec
        existing.flow = parse_flow_spec(
            raw.get("flow"),
            flow_provided=("flow" in raw),
            overlap=existing.overlap,
            area_expand=area_expand,
            area_move=area_move,
        )
        existing.area_expand = area_expand
        existing.area_move   = area_move

    # ── drag constraints ──────────────────────────────────────────────────
    if "inside" in raw:
        existing.inside = raw["inside"]
    if "move_path" in raw:
        existing.move_path = raw["move_path"]

    # ── video attributes ──────────────────────────────────────────────────
    if "type" in raw:
        existing.shape_type = str(raw["type"]).strip().lower()

    if "loop" in raw:
        existing._video_loop = bool(raw["loop"])
        player = getattr(existing, "_video_player", None)
        if player and hasattr(player, "setLoops"):
            from PySide6.QtMultimedia import QMediaPlayer
            player.setLoops(QMediaPlayer.Loops.Infinite if existing._video_loop else QMediaPlayer.Loops.Once)

    if "autoplay" in raw:
        existing._video_autoplay = bool(raw["autoplay"])

    if "muted" in raw:
        existing._video_muted = bool(raw["muted"])
        audio = getattr(existing, "_video_audio", None)
        if audio and hasattr(audio, "setMuted"):
            audio.setMuted(existing._video_muted)

    if "src" in raw:
        new_src = str(raw["src"]) if raw["src"] is not None else None
        if new_src != existing.src:
            # Stop existing player
            old_player = getattr(existing, "_video_player", None)
            if old_player and hasattr(old_player, "stop"):
                try:
                    old_player.stop()
                except Exception:
                    pass
            # Clear old video attributes
            for attr in ["_video_player", "_video_audio", "_video_sink", "_video_frame", "_video_canvas_ref"]:
                if hasattr(existing, attr):
                    delattr(existing, attr)
            existing.src = new_src
            _init_video_player(existing)


# ── registry ──────────────────────────────────────────────────────────────────

class _ShapeRegistry:
    """
    Internal registry.  Exposes the new ``Draw.shape`` public API and keeps
    legacy ``Draw.shapes`` working for backward compatibility.
    """

    def __call__(
        self,
        *,
        display: Optional[str] = None,
        tag: Optional[str] = None,           # legacy alias
        ip: object = None,                   # canonical ip kwarg
        shape_ip: object = None,             # legacy alias for ip
        shape: Optional[List[dict]] = None,
        shapes: Optional[List[dict]] = None, # legacy alias
        text: Optional[List[dict]] = None,
        list_spec: Optional[list] = None,
        connections: Optional[List[dict]] = None,
        get_ip: object = None,
        shape_get_ip: object = None,         # legacy alias for get_ip
        show: object = None,
        sub_edit: object = None,             # edit existing shapes by ip
        properties: object = None,           # widget specs e.g. [scroller]
        area: object = None,                 # area spec: [{"expand":(ex,ey),"move":"right"}]
        place_with_ip: object = None,        # confine shapes to a region defined by this ip
        hit_box: object = None,              # call-level hitbox ip (resetgeometry / fullgeometry)
        **shape_props,                       # single-shape shorthand: Draw.shape(tag=W, ip=.., type=.., ...)
    ) -> None:
        """
        Draw.shapes(display="main", shapes=[{...}])

        New parameters
        --------------
        place_with_ip : str
            Place all shapes in this call inside the region registered under
            this ip (via Draw.hitbox or a previously drawn shape with that ip).
            Supports both resetgeometry (world-coords, clipped) and fullgeometry
            (relative coords) depending on how the hitbox was defined.

        hit_box : str
            Shorthand: register or reference a hitbox ip for this call.
            Alias for place_with_ip when only a hitbox name is given.

        sub_edit : str | list[str]
            One or more shape IPs to update.  Shapes in the `shapes` list
            are matched by ip and their properties are merged into the
            existing ShapeDef (non-destructive update).

        properties : list
            Widget descriptors.  Currently supported:
              "scroller" — adds a scrollbar overlay.
            Each widget entry is a dict with a "type" key (or use the string
            alias directly) plus widget-specific keys.

            Scroller example:
                Draw.shapes(
                    display=WIN,
                    properties=["scroller"],
                    shapes=[
                        {"place": "right", "for": "page",
                         "color": "#111827", "thumb_color": "#5b6880"}
                    ]
                )
        """
        # ── single-shape shorthand ───────────────────────────────────────────
        # Allows: Draw.shape(tag=WIN, ip="my_ip", type="circle", size=40, ...)
        # Any unrecognised kwargs arrive in shape_props; wrap into a one-item list.
        if shape_props:
            single: dict = dict(shape_props)
            if ip is not None:
                single.setdefault("ip", str(ip))
                ip = None   # consumed into the shape dict; don't re-use as fallback
            shape_items = [single] + list(shape or shapes or [])
        else:
            shape_items = list(shape or shapes or [])
        text_items  = list(text or [])

        if not isinstance(shape_items, list):
            raise TypeError("Draw.shapes: 'shapes' must be a list of dicts.")

        # Resolve window tag
        window_tag = display or tag
        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise ValueError(
                    "Draw.shapes: multiple windows exist; 'display' is required."
                )
            else:
                raise ValueError("Draw.shapes: no windows exist to draw on.")

        get_app()
        from Draw._text import _TextRegistry, _get_or_create_canvas
        from Draw._live import (
            LiveTextBinding, is_input_text_marker, is_live_text_binding,
            resolve_live_text,
        )

        win: QMainWindow = _window_registry.get(window_tag)
        canvas = _get_or_create_canvas(window_tag, win)

        # ip is the canonical name; shape_ip is the legacy alias (takes priority if both given)
        effective_ip = shape_ip if shape_ip is not None else ip
        ip_str = str(effective_ip) if effective_ip is not None else None

        # get_ip is the canonical name; shape_get_ip is the legacy alias (takes priority if both given)
        get_ip = shape_get_ip if shape_get_ip is not None else get_ip

        # ── sub_edit: update existing shapes by ip ──────────────────────────
        # ── sub_edit: update existing shapes and texts by ip ────────────────
        created_result_items: List[Any] = []
        if sub_edit is not None:
            edit_ips: List[str] = []
            if isinstance(sub_edit, str):
                edit_ips = [sub_edit]
            elif isinstance(sub_edit, (list, tuple)):
                edit_ips = [str(x) for x in sub_edit]

            if edit_ips:
                for raw in shape_items:
                    if not isinstance(raw, dict):
                        continue
                    raw = _merge_customise(raw)
                    target_ip = str(raw.get("ip", ip_str or "")).strip()
                    # If ip in raw matches an edit target, or edit list is a
                    # wildcard match, update all existing shapes with matching ip
                    for eip in edit_ips:
                        for s in canvas.shape_items:
                            if s.ip != eip:
                                continue
                            if s not in created_result_items:
                                created_result_items.append(s)
                            # Merge raw dict overrides into the existing ShapeDef
                            if "color" in raw:
                                s.color = _parse_color(raw["color"])
                            if "border_color" in raw:
                                s.border_color = _parse_color(raw["border_color"])
                            if "border_width" in raw:
                                s.border_width = _as_int(raw["border_width"])
                            if "border_style" in raw:
                                s.border_style = str(raw["border_style"]).strip().lower()
                            if "opacity" in raw:
                                s.opacity = max(0, min(100, _as_int(raw["opacity"], default=100)))
                            if "rotation" in raw:
                                s.rotation = _as_float(raw["rotation"])
                            if "x" in raw:
                                s.x = int(raw["x"])
                            if "y" in raw:
                                s.y = int(raw["y"])
                            if "z" in raw:
                                s.z = _as_int(raw["z"])
                            if "overlap" in raw:
                                s.overlap = _as_bool(raw["overlap"])
                            if "gradient" in raw or "stops" in raw:
                                from Draw._colour import color as _color_registry
                                _color_registry(
                                    ip=eip,
                                    color=[{
                                        "for":      "body",
                                        "gradient": raw.get("gradient", "linear"),
                                        "angle":    raw.get("angle", 0),
                                        "center":   raw.get("center"),
                                        "radius":   raw.get("radius", 50),
                                        "stops":    raw.get("stops"),
                                    }],
                                )
                            if "area" in raw:
                                s.area_expand, s.area_move = _parse_area_spec(raw.get("area"))
                            if "flow" in raw or "overlap" in raw or "area" in raw:
                                from Draw._overlap import parse_flow_spec
                                s.flow = parse_flow_spec(
                                    raw.get("flow", None),
                                    flow_provided=("flow" in raw),
                                    overlap=getattr(s, "overlap", True),
                                    area_expand=getattr(s, "area_expand", (0.0, 0.0)),
                                    area_move=getattr(s, "area_move", None),
                                )
                            if "vertices" in raw:
                                v = raw["vertices"]
                                s.vertices = None if v is None else max(1, int(v))
                            w_raw = raw.get("width", None)
                            h_raw = raw.get("height", None)
                            size_raw = raw.get("size", None)
                            if size_raw is not None:
                                s.size_raw = size_raw
                            elif w_raw is not None or h_raw is not None:
                                old_w = s.size_raw[0] if isinstance(s.size_raw, (list, tuple)) and len(s.size_raw) >= 1 else s.size_raw
                                old_h = s.size_raw[1] if isinstance(s.size_raw, (list, tuple)) and len(s.size_raw) >= 2 else s.size_raw
                                s.size_raw = [w_raw if w_raw is not None else old_w,
                                              h_raw if h_raw is not None else old_h]
                            if "border_radius" in raw:
                                s.border_radius_raw = raw["border_radius"]
                            if "curve_mode" in raw:
                                cm = str(raw["curve_mode"]).strip().lower()
                                s.curve_mode = cm if cm in _CURVE_MODES else "line"
                            if "bend" in raw:
                                s.bend = list(raw["bend"]) if isinstance(raw["bend"], list) else []
                            if "exclude" in raw:
                                s.exclude = list(raw["exclude"]) if isinstance(raw["exclude"], list) else []
                            if "symmetry" in raw:
                                s.symmetry = dict(raw["symmetry"]) if isinstance(raw["symmetry"], dict) else None
                            if "align" in raw:
                                a = raw["align"]
                                s.align = str(a) if a in _ALIGN_VALUES_SET else None

                            # Phase 2.2 dirty rendering: any geometry-affecting
                            # key means the cached QPainterPath is stale.
                            if _PATH_AFFECTING_KEYS.intersection(raw.keys()):
                                s.dirty = True

                # Also update existing text items matching the target IPs
                for raw in (shape_items + text_items):
                    if not isinstance(raw, dict):
                        continue
                    # For texts: check if the ip matches an edit target
                    raw_ip = raw.get("ip", ip_str)
                    if isinstance(raw_ip, (list, tuple)):
                        target_ip = str(raw_ip[0]) if raw_ip else ip_str
                    else:
                        target_ip = str(raw_ip) if raw_ip is not None else ip_str

                    for eip in edit_ips:
                        for t in canvas.text_items:
                            if t.ip != eip:
                                continue
                            # Merge raw overrides into the existing TextDef
                            if "text" in raw:
                                t.text = str(raw["text"])
                            if "color" in raw:
                                t.color = _parse_color(raw["color"])
                            if "font_size" in raw:
                                t.font_size = int(raw["font_size"])
                            if "bold" in raw:
                                t.bold = bool(raw["bold"])
                            if "italic" in raw:
                                t.italic = bool(raw["italic"])
                            if "underline" in raw:
                                t.underline = bool(raw["underline"])
                            if "strikethrough" in raw:
                                t.strikethrough = bool(raw["strikethrough"])
                            if "x" in raw:
                                t.x = int(raw["x"])
                            if "y" in raw:
                                t.y = int(raw["y"])
                            if "z" in raw:
                                z_val = raw["z"]
                                if z_val == "as_shape":
                                    t.z = "as_shape"
                                else:
                                    try:
                                        t.z = int(z_val) if z_val is not None else 0
                                    except (ValueError, TypeError):
                                        t.z = 0
                            if "overlap" in raw:
                                from Draw._text import _input_bool
                                t.overlap = _input_bool(raw["overlap"], True)
                            if "align" in raw:
                                a = raw["align"]
                                t.align = str(a) if a in _ALIGN_VALUES_SET else None

                canvas._occupied_dirty = True
                canvas.update()
                if created_result_items:
                    return created_result_items[0] if len(created_result_items) == 1 else created_result_items
                return None  # sub_edit only — no new shapes appended

        # ── place_with_ip / hit_box: resolve region FIRST ──────────────────
        effective_place_ip = None
        if place_with_ip is not None:
            effective_place_ip = str(place_with_ip).strip()
        elif hit_box is not None and isinstance(hit_box, str):
            effective_place_ip = str(hit_box).strip()

        place_region: Optional[Dict[str, Any]] = None
        if effective_place_ip:
            region_info = _resolve_place_with_ip(effective_place_ip, None)
            if region_info:
                rx = _parse_px_value(region_info.get("x"), 0.0)
                ry = _parse_px_value(region_info.get("y"), 0.0)
                rw = _parse_px_value(region_info.get("width"), 0.0)
                rh = _parse_px_value(region_info.get("height"), 0.0)
                place_region = {"x": rx, "y": ry, "w": rw, "h": rh,
                                "mode": region_info.get("mode", "resetgeometry")}
            else:
                _logger.warning("Draw.shapes: place_with_ip/hit_box='%s' not found.", effective_place_ip)

        # ── properties: widget descriptors ─────────────────────────────────
        prop_list: List[Any] = []
        if properties is not None:
            if isinstance(properties, str):
                prop_list = [properties]
            elif isinstance(properties, (list, tuple)):
                prop_list = list(properties)
            else:
                prop_list = [properties]

        scroller_specs: List[dict] = []
        for prop in prop_list:
            if isinstance(prop, str) and prop.strip().lower() == "scroller":
                # The shapes list IS the scroller spec — consume it entirely
                scroller_specs.extend(shape_items)
                shape_items = []
            elif isinstance(prop, dict):
                ptype = str(prop.get("type", "")).strip().lower()
                if ptype == "scroller":
                    scroller_specs.append(prop)

        # ── scroller widget shapes (built with already-resolved region) ─────
        if scroller_specs:
            cw = int(win.width())
            ch = int(win.height())
            scroll_shapes, scroll_texts, scroll_cfgs = _build_scroller_shapes(
                scroller_specs, cw, ch, place_region
            )
            # Scroller shapes already have absolute coords — do NOT re-offset
            shape_items.extend(scroll_shapes)
            text_items.extend(scroll_texts)
            # Store configs so canvas can update thumb positions dynamically
            if not hasattr(canvas, '_scroller_configs'):
                canvas._scroller_configs = []
            canvas._scroller_configs.extend(scroll_cfgs)

        # ── properties: "builder" — drag-to-size shape creation ────────────
        # Shapes with an explicit size are placed immediately as before.
        # Shapes with NO width/height/size are queued instead: the next
        # left-click drag on this canvas defines their corners live, and
        # they're finalized with an absolute pixel size on mouse release.
        builder_settings: Optional[dict] = None
        for prop in prop_list:
            if isinstance(prop, str) and prop.strip().lower() == "builder":
                builder_settings = {}
            elif isinstance(prop, dict) and str(prop.get("type", "")).strip().lower() == "builder":
                builder_settings = {k: v for k, v in prop.items() if k != "type"}

        if builder_settings is not None and shape_items:
            _kept_items: List[dict] = []
            for raw in shape_items:
                if not isinstance(raw, dict):
                    _kept_items.append(raw)
                    continue
                merged = _merge_customise(dict(raw))
                has_size = (
                    merged.get("width") is not None
                    or merged.get("height") is not None
                    or merged.get("size") is not None
                )
                if has_size:
                    _kept_items.append(raw)
                    continue
                b_ip = str(merged.get("ip", ip_str) or f"builder_{len(canvas._builder_queue)}")
                merged["ip"] = b_ip
                canvas._builder_queue.append({
                    "ip": b_ip,
                    "raw": merged,
                    "on_build": builder_settings.get("on_build"),
                })
            shape_items = _kept_items

        # ── offset regular shapes into the region (region-relative coords) ──
        if effective_place_ip and shape_items:
            shape_items = _apply_place_with_ip(shape_items, effective_place_ip)

        # ── list_spec ───────────────────────────────────────────────────────
        if list_spec is not None:
            from Draw._list import _generate_list_items
            from Draw._layout import set as _layout_registry
            
            parent_x = 0
            parent_y = 0
            z_val = None
            opacity_val = None
            should_remove_parent = False
            
            if shape_items and isinstance(shape_items[0], dict):
                parent_x = shape_items[0].get("x", 0) or 0
                parent_y = shape_items[0].get("y", 0) or 0
                z_val = shape_items[0].get("z", None)
                opacity_val = shape_items[0].get("opacity", None)
                
                styling_keys = {
                    "vertices", "size", "width", "height", "color", "border_color",
                    "border_width", "border_radius", "border_style", "curve_mode",
                    "bend", "exclude", "symmetry", "custom", "rotation", "hitbox_mode", "hit_box"
                }
                if not any(k in shape_items[0] for k in styling_keys):
                    should_remove_parent = True
            
            list_shapes, list_texts, table_def = _generate_list_items(ip_str, list_spec)
            if table_def:
                table_def.setdefault("margin", {})
                table_def["margin"]["left"] = table_def["margin"].get("left", 0) + parent_x
                table_def["margin"]["top"] = table_def["margin"].get("top", 0) + parent_y
                
                if z_val is not None:
                    for ls in list_shapes:
                        ls["z"] = z_val
                if opacity_val is not None:
                    for ls in list_shapes:
                        ls["opacity"] = opacity_val
                
                table_ip = f"{ip_str}_list_table" if ip_str else "list_table"
                layout_obj = _layout_registry(ip=table_ip, dimension=table_def)
                if layout_obj not in canvas.layout_items:
                    canvas.layout_items.append(layout_obj)
                
                if should_remove_parent:
                    shape_items.pop(0)
                    
                shape_items.extend(list_shapes)
                text_items.extend(list_texts)

        if get_ip is not None:
            from Draw._layout import set as _layout_registry
            try:
                layout_obj = _layout_registry.resolve(get_ip)
                if layout_obj not in canvas.layout_items:
                    canvas.layout_items.append(layout_obj)
            except Exception as exc:
                _logger.warning("Draw.shapes: failed to resolve get_ip layout: %s", exc)

        for raw in shape_items:
            # ── Incremental update ────────────────────────────────────────────
            # If a shape carrying this ip is already on the canvas, update it
            # in-place instead of allocating a new ShapeDef.  If the hash of
            # the incoming raw dict matches the cached hash the shape hasn't
            # changed at all, so we skip it entirely (zero allocation, zero
            # repaint work added).
            raw_merged = _merge_customise(dict(raw))
            _raw_ip_val = raw_merged.get("ip", ip_str)
            if isinstance(_raw_ip_val, (list, tuple)):
                raw_ip = str(_raw_ip_val[0]) if _raw_ip_val else ip_str
            else:
                raw_ip = str(_raw_ip_val) if _raw_ip_val is not None else ip_str

            if raw_ip is not None and hasattr(canvas, "_shape_by_ip"):
                ip_key   = str(raw_ip)
                existing = canvas._shape_by_ip.get(ip_key)
                if existing is not None:
                    new_hash    = _compute_content_hash(raw_merged)
                    cached_hash = canvas._shape_hash_by_ip.get(ip_key, -1)

                    if cached_hash == new_hash:
                        # Identical to what is already drawn — nothing to do.
                        created_result_items.append(existing)
                        continue

                    # Something changed.  Detect whether the layout (position /
                    # size) changed so we know whether to rebuild the overlap
                    # registry.  Then update the existing object in-place.
                    layout_change = _is_layout_affecting(existing, raw_merged)
                    _update_shape_inplace(existing, raw_merged)
                    canvas._shape_hash_by_ip[ip_key] = new_hash

                    if layout_change:
                        canvas._occupied_dirty = True
                    # Object is already in canvas.shape_items — do not append.
                    created_result_items.append(existing)
                    continue

            # Brand-new shape: parse fully and append.
            parsed = _parse_shape(raw, ip_str, layout_override=get_ip, area_raw=area)
            canvas.shape_items.append(parsed)
            canvas._z_order_dirty = True
            created_result_items.append(parsed)
            canvas._occupied_dirty = True
            if parsed.ip is not None and hasattr(canvas, "_shape_by_ip"):
                ip_key = parsed.ip
                canvas._shape_by_ip[ip_key]      = parsed
                canvas._shape_hash_by_ip[ip_key] = _compute_content_hash(raw_merged)

        for raw in text_items:
            if not isinstance(raw, dict):
                raise TypeError("Draw.shapes: every item in 'text' must be a dict.")
            raw_text = raw.get("text", "")
            if raw_text is None:
                raw_text = ""
            text_source = None
            if is_input_text_marker(raw_text):
                text_source = raw_text
                text_value = str(getattr(raw_text, "initial", ""))
            elif is_live_text_binding(raw_text):
                text_source = raw_text
                text_value = resolve_live_text(raw_text)
            elif callable(raw_text):
                text_source = LiveTextBinding(raw_text)
                text_value = resolve_live_text(text_source)
            else:
                text_value = str(raw_text)

            customise = raw.get("customise", {}) or {}
            raw_properties = raw.get("properties", {}) or {}

            # Backward-compat / forgiveness: allow flat style keys directly
            # on the text dict (x, y, font_size, color, align, etc.) instead
            # of requiring everything to be nested under "customise". Without
            # this, flat keys were silently dropped and every such text item
            # fell back to the same default-centered position, causing
            # unrelated text items to visually stack on top of one another.
            _RESERVED_TEXT_KEYS = {"ip", "text", "customise", "properties", "column", "columns", "layout", "get_ip"}
            flat_style = {k: v for k, v in raw.items() if k not in _RESERVED_TEXT_KEYS}
            if flat_style:
                merged_customise = dict(flat_style)
                merged_customise.update(customise)  # explicit customise wins on conflict
                customise = merged_customise

            cell_val = raw.get("column", raw.get("columns", customise.get("column", customise.get("columns", None))))
            text_layout = raw.get("layout", raw.get("get_ip", customise.get("layout", customise.get("get_ip", get_ip))))

            new_tdef = _TextRegistry._parse(
                text_value,
                customise,
                ip=raw.get("ip", ip_str),
                source=text_source,
                properties=raw_properties,
                layout=text_layout,
                cell=cell_val,
            )
            # Enforce ip uniqueness here too (this append site bypasses
            # _TextRegistry.__call__'s own dedupe), so re-running build_chrome()
            # or any batch text= call with a repeated ip replaces the old
            # item instead of stacking a duplicate on top of it.
            if new_tdef.ip is not None:
                canvas.text_items = [t for t in canvas.text_items if t.ip != new_tdef.ip]
            canvas.text_items.append(new_tdef)

        canvas.update()
        if created_result_items:
            return created_result_items[0] if len(created_result_items) == 1 else created_result_items
        return None

    # ── utility methods ───────────────────────────────────────────────────────

    def clear(self, display: str) -> None:
        """Remove all shapes from a window canvas."""
        win = _window_registry.get(display)
        if hasattr(win, "_draw_canvas"):
            canvas = win._draw_canvas
            for s in canvas.shape_items:
                s.dispose()
            canvas.shape_items.clear()
            canvas._z_order_dirty = True
            # Flush incremental-rendering caches so stale hashes and ip →
            # ShapeDef references cannot match shapes added in the next call.
            if hasattr(canvas, "_shape_by_ip"):
                canvas._shape_by_ip.clear()
            if hasattr(canvas, "_shape_hash_by_ip"):
                canvas._shape_hash_by_ip.clear()
            canvas._occupied_dirty = True
            canvas.update()

    def remove(self, display: str, index: int) -> None:
        """Remove a shape by index."""
        win = _window_registry.get(display)
        if hasattr(win, "_draw_canvas"):
            items = win._draw_canvas.shape_items
            if 0 <= index < len(items):
                removed = items.pop(index)
                removed.dispose()
                win._draw_canvas._z_order_dirty = True
                win._draw_canvas._occupied_dirty = True
                win._draw_canvas.update()

    def list_shapes(self, display: str) -> List[ShapeDef]:
        win = _window_registry.get(display)
        if hasattr(win, "_draw_canvas"):
            return list(win._draw_canvas.shape_items)
        return []

    def get_by_ip(self, display: str, ip: str) -> Optional[ShapeDef]:
        win = _window_registry.get(display)
        if hasattr(win, "_draw_canvas"):
            for s in win._draw_canvas.shape_items:
                if s.ip == ip:
                    return s
        return None

    def remove_by_ip(self, display: str, ip: str) -> int:
        win = _window_registry.get(display)
        if not hasattr(win, "_draw_canvas"):
            return 0
        canvas = win._draw_canvas
        before = len(canvas.shape_items)
        for s in canvas.shape_items:
            if s.ip == ip:
                s.dispose()
        canvas.shape_items = [s for s in canvas.shape_items if s.ip != ip]
        canvas._z_order_dirty = True
        removed = before - len(canvas.shape_items)
        if removed:
            # Keep incremental-rendering caches consistent.
            if hasattr(canvas, "_shape_by_ip"):
                canvas._shape_by_ip.pop(ip, None)
            if hasattr(canvas, "_shape_hash_by_ip"):
                canvas._shape_hash_by_ip.pop(ip, None)
            canvas._occupied_dirty = True
            canvas.update()
        return removed


# ── singleton / public export ─────────────────────────────────────────────────

shapes = _ShapeRegistry()
shape  = shapes           # new preferred name


# ── hitbox (kept for graph / connector compatibility) ─────────────────────────

class _HitboxRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, dict] = {}

    def __call__(self, *, tag: object = None, ip: object = None,
                 mode: object = None, type: object = None,
                 box: object = None) -> None:
        hb_id = str(tag if tag is not None else ip)
        if not hb_id:
            raise ValueError("Draw.hitbox: 'ip' is required.")
        mode_raw = mode if mode is not None else type
        resolved_mode = "resetgeometry"
        if isinstance(mode_raw, (list, tuple)) and mode_raw:
            resolved_mode = str(mode_raw[0]).strip().lower()
        elif mode_raw is not None:
            resolved_mode = str(mode_raw).strip().lower()
        box_item = {}
        if isinstance(box, list) and box:
            box_item = box[0] if isinstance(box[0], dict) else {}
        elif isinstance(box, dict):
            box_item = box
        self._items[hb_id] = {
            "mode":   resolved_mode,
            "width":  box_item.get("width"),
            "height": box_item.get("height"),
            "x":      box_item.get("x"),
            "y":      box_item.get("y"),
        }

    def get(self, ip: object) -> Optional[dict]:
        return self._items.get(str(ip))

    def clear(self, ip: Optional[object] = None) -> None:
        if ip is None:
            self._items.clear()
        else:
            self._items.pop(str(ip), None)

    def list(self) -> Dict[str, dict]:
        return dict(self._items)


hitbox = _HitboxRegistry()


# ── container (kept for backward compat) ─────────────────────────────────────

class _ContainerRegistry:
    def __call__(self, *, display: str = None, tag: str = None,
                 mode: str = "full", width: object = None,
                 height: object = None, children: Optional[List[dict]] = None,
                 customise: Optional[dict] = None,
                 ip: Optional[str] = None) -> None:
        win_tag = display or tag
        c = customise or {}
        container_raw: dict = {
            "vertices": 4,
            "size": [width, height] if width or height else None,
            "ip": ip,
            **c,
        }
        shape(display=win_tag, shape=[container_raw])


container = _ContainerRegistry()


# â”€â”€ image â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _ImageRegistry:
    """
    Draw.image â€” render an image file (PNG, JPG, GIF, BMP, SVG) inside a
    shapes call.

    Usage:
        Draw.image(
            display       = "my_window",
            src           = "path/to/photo.png",   # required
            size          = [200, 150],             # [w, h] in px, or single int
            x             = None,                   # absolute x (None = auto)
            y             = None,                   # absolute y (None = auto)
            align         = "center",               # same values as shape align
            opacity       = 100,                    # 0-100
            border_radius = 0,                      # clip corners (px or "50%")
            ip            = None,                   # identity tag
        )

    Internally this creates a shape with type="image" and attaches the
    source path so _draw_one_shape can paint a QPixmap into the slot.
    """

    def __call__(
        self,
        *,
        display:       object = None,
        tag:           object = None,
        src:           object = None,
        source:        object = None,
        size:          object = None,
        width:         object = None,
        height:        object = None,
        x:             object = None,
        y:             object = None,
        align:         object = None,
        opacity:       object = 100,
        border_radius: object = 0,
        ip:            object = None,
        customise:     Optional[dict] = None,
    ) -> None:
        win_tag  = display or tag
        img_src  = src or source
        if not img_src:
            raise ValueError("Draw.image: 'src' (image file path) is required.")

        c = customise or {}
        resolved_size: object = size
        if resolved_size is None and (width is not None or height is not None):
            resolved_size = [width, height]

        image_raw: dict = {
            "type":          "image",
            "src":           str(img_src),
            "size":          resolved_size,
            "x":             x,
            "y":             y,
            "align":         align,
            "opacity":       opacity,
            "border_radius": border_radius,
            "ip":            ip,
            **c,
        }
        shape(display=win_tag, shape=[image_raw])


image = _ImageRegistry()


# â”€â”€ video â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _VideoRegistry:
    """
    Draw.video â€” embed a video file (MP4, AVI, MKV, MOV, WebM) inside a
    shapes call using QMediaPlayer + QVideoWidget rendered into the canvas.

    Usage:
        Draw.video(
            display  = "my_window",
            src      = "path/to/clip.mp4",    # required
            size     = [320, 240],
            x        = None,
            y        = None,
            align    = "center",
            opacity  = 100,
            loop     = True,                   # loop playback
            autoplay = True,                   # start on first draw
            muted    = False,
            ip       = None,
        )
    """

    def __call__(
        self,
        *,
        display:   object = None,
        tag:       object = None,
        src:       object = None,
        source:    object = None,
        size:      object = None,
        width:     object = None,
        height:    object = None,
        x:         object = None,
        y:         object = None,
        align:     object = None,
        opacity:   object = 100,
        loop:      object = True,
        autoplay:  object = True,
        muted:     object = False,
        ip:        object = None,
        customise: Optional[dict] = None,
    ) -> None:
        win_tag   = display or tag
        vid_src   = src or source
        if not vid_src:
            raise ValueError("Draw.video: 'src' (video file path) is required.")

        c = customise or {}
        resolved_size: object = size
        if resolved_size is None and (width is not None or height is not None):
            resolved_size = [width, height]

        video_raw: dict = {
            "type":     "video",
            "src":      str(vid_src),
            "size":     resolved_size,
            "x":        x,
            "y":        y,
            "align":    align,
            "opacity":  opacity,
            "loop":     loop,
            "autoplay": autoplay,
            "muted":    muted,
            "ip":       ip,
            **c,
        }
        shape(display=win_tag, shape=[video_raw])


video = _VideoRegistry()
