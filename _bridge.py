"""
Draw._bridge — central cross-module integration hub.

WHY THIS FILE EXISTS
---------------------
Python can't have every Draw module eagerly import every other Draw module —
_room importing _text which imports _room back is an immediate circular-import
crash at package load time. That's why the codebase has always wired features
together with *lazy, function-local* imports instead: e.g. _shapes.py reaches
into _colour.py with `from Draw._colour import color as _color_registry`
sitting inside a function body, not at the top of the file. That pattern is
correct and stays exactly as-is — this module doesn't replace it.

What this module does is give that existing pattern ONE shared front door.
Before this file, the same "from Draw._colour import color as _color_registry"
line was independently copy-pasted inside _shapes.py in three separate
functions, and _text.py never had its own copy at all — which is exactly how
Draw.text() ended up silently disconnected from Draw.color(). New cross-module
wiring goes here once; every module that needs it calls the accessor below
instead of writing its own local import.

RULES FOR THIS FILE
--------------------
1. Nothing here is imported at Draw package load time (see __init__.py —
   _bridge is never imported there). Only individual functions import it,
   lazily, at call time. This introduces zero new circular-import risk.
2. Every accessor below does its import *inside* the function body, for the
   same reason. Do not move any of these imports to module level.
3. Keep accessors thin (just "go fetch the singleton"). Put actual
   cross-cutting logic (like resolve_dynamic_color below) in a plain function
   here so it has one implementation instead of N copies.
"""

from __future__ import annotations
from typing import Any, Dict, Optional


# ── registry accessors ──────────────────────────────────────────────────────
# One accessor per singleton registry. Add new ones here as new modules need
# to be reachable from elsewhere in the library.

def get_color_registry():
    from Draw._colour import color as _color_registry
    return _color_registry


def get_color_parser():
    """
    Returns Draw._colour._parse_color — the canonical static colour parser
    (named string / hex string / RGB tuple / QColor -> QColor). This is the
    one-time/static counterpart to resolve_dynamic_color below, which handles
    live Draw.color(ip=...) bindings. Any module accepting a colour value
    (background_color, border_color, etc.) should resolve it through this
    accessor instead of writing or importing its own parser.
    """
    from Draw._colour import _parse_color
    return _parse_color


def get_shape_registry():
    from Draw._shapes import shapes as _shape_registry
    return _shape_registry


def get_text_registry():
    from Draw._text import text as _text_registry
    return _text_registry


def get_point_registry():
    from Draw._point import point as _point_registry
    return _point_registry


def get_panel_registry():
    from Draw._panel import panel as _panel_registry
    return _panel_registry


def get_window_registry():
    from Draw._window import window as _window_registry
    return _window_registry


def get_layout_registry():
    from Draw._layout import set as _layout_registry
    return _layout_registry


def get_room_registry():
    """
    Returns Draw._room.room (the _RoomRegistry singleton exposed as
    Draw.room). _room.py itself already reaches shapes/text/window/panel/
    layout/point through their own bridge accessors — this is the reverse
    direction, for any module that needs to ask room() things (e.g. current
    resolved geometry for an id) instead of the other way around.
    """
    from Draw._room import room as _room_registry
    return _room_registry


def get_motion_registry():
    """
    Returns Draw._motion.motion (the MotionRegistry singleton exposed as
    Draw.motion). _shapes.py's own render-tick loop already reaches into
    this twice with its own local import (once for tick_shape_triggers,
    once for the gradient/color-brush path) — this accessor gives any
    future canvas (e.g. a ticking _graph.py) the same one-line access
    instead of a third copy of the same import.
    """
    from Draw._motion import motion as _motion_registry
    return _motion_registry


def get_graph_registry():
    from Draw.graph import graph as _graph_registry
    return _graph_registry


def get_flow_spec_parser():
    """
    Returns Draw._overlap.parse_flow_spec — turns a raw flow=... value from
    a shape/text/path dict into a resolved FlowSpec. This one function was
    independently re-imported 3x inside _shapes.py, 2x inside _point.py,
    and once in _text.py before this accessor existed; all six call sites
    can go through here instead.
    """
    from Draw._overlap import parse_flow_spec
    return parse_flow_spec


def get_overlap_helpers():
    """
    Returns the small bundle of Draw._overlap primitives needed to place an
    item against the anti-overlap/flow system: (Rect, flow_occupied_rect,
    get_strategy_for_flow). _overlap.py has no singleton registry object
    (it's a stateless collection of dataclasses/strategies/functions), so
    unlike the other accessors above this returns a tuple rather than one
    object — callers unpack it the same way they already unpack the
    equivalent raw import.
    """
    from Draw._overlap import Rect, flow_occupied_rect, get_strategy_for_flow
    return Rect, flow_occupied_rect, get_strategy_for_flow


def get_gradient_brush_builder():
    """
    Returns _shapes._build_gradient_brush — the one function in the codebase
    that turns a resolved gradient dict ({"type", "angle"/"center", "stops"})
    into a QBrush. Shapes already had this; text needed it too, so instead of
    a second copy of the gradient math living in _text.py, text goes through
    this accessor and reuses the exact same implementation.
    """
    from Draw._shapes import _build_gradient_brush
    return _build_gradient_brush


# ── shared cross-cutting helpers ────────────────────────────────────────────

def get_motion_state(ip: str, window_tag: Optional[str] = None) -> Optional[dict]:
    """
    Returns the current motion state dict for the element registered under `ip`
    (or None if no active motion state exists).
    """
    if not ip:
        return None
    motion_reg = get_motion_registry()
    elements = motion_reg.find_elements_by_ip(str(ip))
    if elements:
        el = elements[0]
        state = getattr(el, "_last_motion_state", None)
        if state is not None:
            return state
    return None


def get_shape_rect(window_tag: str, ip: str) -> Optional[Any]:
    """
    Return (x, y, w, h) — the current on-canvas bounding box of the shape
    registered under `ip` on window `window_tag` — or None if it isn't
    registered yet. Inspects active motion state overrides if present.
    """
    registry = get_shape_registry()
    s = registry.get_by_ip(window_tag, ip)
    if s is None:
        return None
    motion_state = getattr(s, "_last_motion_state", None)

    try:
        from Draw._shapes import _shape_preferred_pos
        win_reg = get_window_registry()
        try:
            win = win_reg.get(window_tag)
            cw = float(win.width())
            ch = float(win.height())
        except Exception:
            cw, ch = 1000.0, 800.0
        sw, sh, ox, oy = _shape_preferred_pos(s, int(cw), int(ch))
        s.last_position = (ox, oy)
        s.last_size = (sw, sh)
        if motion_state:
            ox = motion_state.get("x", ox)
            oy = motion_state.get("y", oy)
            sw = motion_state.get("width", sw)
            sh = motion_state.get("height", sh)
        return (float(ox), float(oy), float(sw), float(sh))
    except Exception:
        pass

    if s.last_position and s.last_size:
        x, y = s.last_position
        w, h = s.last_size
        if motion_state:
            x = motion_state.get("x", x)
            y = motion_state.get("y", y)
            w = motion_state.get("width", w)
            h = motion_state.get("height", h)
        return (float(x), float(y), float(w), float(h))
    return None


def get_shape_center(window_tag: str, ip: str) -> Optional[Any]:
    """
    Return (cx, cy) — the current center point of the object registered
    under `ip` on window `window_tag` — or None if it can't be resolved
    yet. Checks the shape registry first, then falls back to Draw.point's
    registered paths (Draw.point items are addressable by ip too, but
    keep separate bounding-box math from ShapeDef — see _point.py's
    get_pixel_bounds).
    """
    rect = get_shape_rect(window_tag, ip)
    if rect is not None:
        x, y, w, h = rect
        return (x + w / 2.0, y + h / 2.0)
    try:
        point_registry = get_point_registry()
        bounds = point_registry.get_pixel_bounds(window_tag, ip)
    except Exception:
        bounds = None
    if bounds is not None:
        x, y, w, h = bounds
        return (x + w / 2.0, y + h / 2.0)
    return None


def get_anchor_point(rect, anchor: str):
    """
    Resolve a named anchor ("center"/"top"/"bottom"/"left"/"right"/
    "top-left"/"top-right"/"bottom-left"/"bottom-right", underscores also
    accepted) on `rect` (x, y, w, h) into an absolute (x, y) point.
    Unknown anchor names fall back to "center". Shared by _connectors.py,
    _motion.py, and _shapes.py so this alignment math has one
    implementation instead of N near-identical copies.
    """
    x, y, w, h = rect
    cx, cy = x + w / 2.0, y + h / 2.0
    a = str(anchor).strip().lower().replace("_", "-")
    points = {
        "center":       (cx,      cy),
        "top":          (cx,      y),
        "bottom":       (cx,      y + h),
        "left":         (x,       cy),
        "right":        (x + w,   cy),
        "top-left":     (x,       y),
        "top-right":    (x + w,   y),
        "bottom-left":  (x,       y + h),
        "bottom-right": (x + w,   y + h),
    }
    return points.get(a, (cx, cy))


def resolve_point_ref(ref: Any, window_tag: Optional[str], self_rect=None):
    """
    Resolve a "point reference" value into an absolute (x, y) point, or
    None if it can't be resolved right now. This is the single shared rule
    for center=/align=/anchor-style parameters across _motion.py (rotation
    pivots), _shapes.py (alignment), and _connectors.py (joint anchors):

        None / "center"        -> center of self_rect
        "ip:other_ip"          -> center of the object registered as
                                   other_ip on this window
        "ip:other_ip:anchor"   -> a named anchor (see get_anchor_point) on
                                   that other object, e.g.
                                   "ip:a:top-right"
        (x, y) tuple/list      -> used as-is, no lookup
        any other anchor word  -> that anchor on self_rect (e.g. "top-left")

    Returns None if an "ip:" reference can't currently resolve (object not
    registered/painted yet) and there's no self_rect to fall back on.
    """
    if ref is None:
        ref = "center"

    if isinstance(ref, (tuple, list)) and len(ref) >= 2:
        return (float(ref[0]), float(ref[1]))

    if isinstance(ref, str) and ref.startswith("ip:"):
        body = ref[3:]
        other_ip, _, anchor = body.partition(":")
        anchor = anchor or "center"
        if not window_tag:
            return None
        other_rect = get_shape_rect(window_tag, other_ip)
        if other_rect is not None:
            return get_anchor_point(other_rect, anchor)
        # Check if it is registered in the point registry (e.g. a Draw.point path)
        try:
            point_registry = get_point_registry()
            bounds = point_registry.get_pixel_bounds(window_tag, other_ip)
        except Exception:
            bounds = None
        if bounds is not None:
            return get_anchor_point(bounds, anchor)
        return None

    if self_rect is not None:
        return get_anchor_point(self_rect, str(ref))

    return None


def resolve_dynamic_color(
    ip: Optional[str],
    x: float = 0.0, y: float = 0.0, w: float = 0.0, h: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """
    Look up a Draw.color(ip=...) binding for `ip` and resolve it at the
    current moment (handles "time"/"x"/"y"/"w"/"h"/"mouse_x"/"mouse_y"
    expressions, gradients, shadow/glow extras — see _colour.py
    _ColorRegistry.resolve_for_shape). Returns None if no binding is
    registered for this ip.

    This is the single shared entry point _shapes.py and _text.py both use
    to ask "does this ip have a dynamic color binding, and if so what does
    it resolve to right now" — previously _shapes.py had its own inline
    has_binding()/resolve_for_shape() calls (duplicated 3x within the file)
    and _text.py had none at all.
    """
    if not ip:
        return None
    registry = get_color_registry()
    if not registry.has_binding(ip):
        return None
    return registry.resolve_for_shape(ip, shape_x=x, shape_y=y, shape_w=w, shape_h=h)
