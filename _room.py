"""
Draw._room

Relative Scene Layout System (see Relative_Layout_System_Plan.txt).

Lets objects in a scene be positioned relative to the scene itself or to
other already-placed objects, instead of relying only on absolute x/y.

Usage
-----
    Draw.room(
        display="main",
        general={"padding": 20},
        scene={
            "panel":  "center",
            "title":  ["panel", "top"],
            "graph":  "center",
            "legend": ["graph", "right"],
        },
    )

Supported per-id value shapes (see plan doc for full spec)
------------------------------------------------------------
    "id": "center"                                  scene-relative anchor
    "id": ["parent_id", "placement"]                 object-relative
    "id": ["parent_id", "placement", "inside"]        object-relative, inside parent
    "id": ["parent_id", "placement", offset_px]       object-relative + offset
    "id": ["parent_id", "placement", "padding", N]    explicit padding
    "id": {"x": 200, "y": 100}                        absolute (dict)
    "id": (200, 100)                                  absolute (tuple)

Resolution order
----------------
1. Parse the layout dict, build a dependency graph (object-relative entries
   depend on their parent id).
2. Reject circular references.
3. Resolve every scene-relative / absolute entry first (no dependencies).
4. Resolve object-relative entries in dependency order, walking each one's
   parent geometry (which is by then final).
5. Reject any id for which BOTH the underlying object already carries its
   own `align` (set when it was created via Draw.shapes/Draw.text/etc.)
   AND room() is being asked to place/align it too — "two alignments given".
6. Apply the computed x/y back onto the live object and trigger a repaint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from Draw._room_size import resolve_size_spec, resolve_match, RoomSizeError, KEYWORDS as _SIZE_KEYWORDS


# ── reserved keywords (per plan doc) ───────────────────────────────────────────

_RESERVED_KEYWORDS = {"scene", "parent", "center", "top", "bottom", "left", "right"}

# Anchor / placement names. We accept both the plan doc's underscore style
# ("top_left") and the engine's native hyphen style ("top-left") and
# normalize to hyphen form, since that's what _align_pos / _panel_align_pos
# / _ALIGN_VALUES already use everywhere else in Draw.
_SCENE_ANCHORS = {
    "top_left", "top", "top_right",
    "left", "center", "right",
    "bottom_left", "bottom", "bottom_right",
    # native hyphen spellings also accepted directly
    "top-left", "top-right", "bottom-left", "bottom-right",
}

# Object-relative placements: where the child goes relative to its parent.
_OBJECT_PLACEMENTS = {
    "top", "bottom", "left", "right",
    "top_left", "top_right", "bottom_left", "bottom_right",
    "top-left", "top-right", "bottom-left", "bottom-right",
    "center",
}


def _normalize_anchor(name: str) -> str:
    """'top_left' -> 'top-left'; already-hyphenated names pass through."""
    return str(name).strip().lower().replace("_", "-")


# Dict keys that mark a dict as a *size* spec (see Draw._room_size) rather
# than a text-align modifier ({"align_text"/"valign"}) or something else.
# "anchor_size"/"scale_center"/"padding"/"margin" alone don't count — they're
# only meaningful alongside one of these primary keys.
_SIZE_DICT_PRIMARY_KEYS = frozenset({
    "fit", "fit_width", "fit_height", "fill",
    "width", "height", "size",
    "ratio", "aspect",
    "min_width", "max_width", "min_height", "max_height",
    "scale", "stretch_x", "stretch_y",
    "equal_width", "equal_height", "match",
    "fit_cell",
})


def _looks_like_size_dict(tok: dict) -> bool:
    return any(k in tok for k in _SIZE_DICT_PRIMARY_KEYS)


_VALID_ALIGN_TEXT = {"left", "center", "right"}
_VALID_VALIGN = {"top", "middle", "bottom"}


def _apply_text_align_modifiers(entry: "_Entry", tok: dict, obj_id: str) -> None:
    """Parse a trailing {"align_text": ..., "valign": ...} modifier dict
    used to align text WITHIN the box room() places it into, separate from
    *where* that box itself goes (which align/anchor/placement controls).
    """
    if "align_text" in tok:
        v = str(tok["align_text"]).strip().lower()
        if v not in _VALID_ALIGN_TEXT:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' has invalid align_text={v!r}; "
                f"expected one of {sorted(_VALID_ALIGN_TEXT)}."
            )
        entry.align_text = v
    if "valign" in tok:
        v = str(tok["valign"]).strip().lower()
        if v not in _VALID_VALIGN:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' has invalid valign={v!r}; "
                f"expected one of {sorted(_VALID_VALIGN)}."
            )
        entry.valign = v


# ── small geometry record ──────────────────────────────────────────────────────

class _Geom:
    """Resolved/working geometry for one room() entry."""

    __slots__ = ("ip", "x", "y", "w", "h", "kind", "resolved")

    def __init__(self, ip: str, x: float = 0.0, y: float = 0.0,
                 w: float = 0.0, h: float = 0.0, kind: str = "shape"):
        self.ip = ip
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.kind = kind          # "shape" | "text" | "panel" | "point"
        self.resolved = False

    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


# ── errors ──────────────────────────────────────────────────────────────────────

class RoomLayoutError(ValueError):
    """Raised for any Draw.room() layout problem (cycles, conflicts, bad ids)."""


# ── live-object lookup / geometry resolution ───────────────────────────────────

def _find_live_object(window_tag: str, obj_id: str):
    """
    Locate the live object for obj_id across shape, text, panel, and point
    registries.

    Returns (kind, obj) where kind is "shape" | "text" | "panel" | "point"
    and obj is the underlying ShapeDef / TextDef / PanelDef / PathDef, or
    (None, None) if not found.
    """
    from Draw import _bridge
    _shape_registry = _bridge.get_shape_registry()
    _panel_registry = _bridge.get_panel_registry()
    _window_registry = _bridge.get_window_registry()
    _point_registry = _bridge.get_point_registry()

    # Panels are independent of any particular canvas/window tag.
    p = _panel_registry.get(obj_id)
    if p is not None:
        return "panel", p

    s = _shape_registry.get_by_ip(window_tag, obj_id)
    if s is not None:
        return "shape", s

    pt = _point_registry.get_by_ip(window_tag, obj_id)
    if pt is not None:
        return "point", pt

    try:
        win = _window_registry.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas is not None:
            for t in canvas.text_items:
                if t.ip == obj_id:
                    return "text", t
    except Exception as exc:
        print(
            f"Draw.room: warning: text lookup for '{obj_id}' on display "
            f"'{window_tag}' failed: {exc}"
        )

    return None, None


def _existing_align(kind: str, obj) -> Optional[str]:
    """The align the object was already given at creation time, if any."""
    return getattr(obj, "align", None)


def _current_geom(kind: str, obj, canvas_w: int, canvas_h: int, window_tag: str = "") -> _Geom:
    """Best-effort current geometry of an already-existing object."""
    if kind == "panel":
        return _Geom(obj.ip, float(obj.x), float(obj.y),
                     float(obj.width), float(obj.height), kind="panel")

    if kind == "shape":
        if obj.last_position and obj.last_size:
            ox, oy = obj.last_position
            sw, sh = obj.last_size
            return _Geom(obj.ip, float(ox), float(oy), float(sw), float(sh), kind="shape")
        # Not painted yet — fall back to the same pre-paint estimate the
        # engine itself uses (_shape_preferred_pos), so room() works even
        # before the first repaint.
        from Draw._shapes import _shape_preferred_pos
        sw, sh, ox, oy = _shape_preferred_pos(obj, canvas_w, canvas_h)
        return _Geom(obj.ip, float(ox), float(oy), float(sw), float(sh), kind="shape")

    if kind == "point":
        # obj is a PathDef (registered via Draw.point(..., points=[{"ip": ...}]))
        # Bounds come from the point registry, which reuses the exact same
        # _PointCanvas._path_bounds the renderer itself uses.
        from Draw import _bridge
        _point_registry = _bridge.get_point_registry()
        bounds = _point_registry.get_pixel_bounds(window_tag, obj.ip)
        if bounds is None:
            return _Geom(obj.ip, 0.0, 0.0, 0.0, 0.0, kind="point")
        x, y, w, h = bounds
        return _Geom(obj.ip, float(x), float(y), float(w), float(h), kind="point")

    if kind == "text":
        if getattr(obj, "last_position", None) is not None and getattr(obj, "last_size", None) is not None:
            ox, oy = obj.last_position
            sw, sh = obj.last_size
            return _Geom(obj.ip, float(ox), float(oy), float(sw), float(sh), kind="text")

        # Always compute a real, font-metric-based size for text — never
        # rely solely on last_rect (a stale post-paint cache) or a blind
        # (x, y, 0, 0) guess pre-paint. Without this, Draw.room() was
        # positioning text by treating it as a zero-size point, so anchors
        # like "center" or "right" landed wherever the *origin* should be,
        # not where the actual rendered text box's center/edge is — i.e.
        # Draw.room solved object positioning but never solved text
        # alignment within its own bounding box, since it never knew the
        # box's true dimensions in the first place.
        from Draw._text import measure_text, _resolve_text_value
        try:
            w, h = measure_text(
                _resolve_text_value(obj),
                font_family=getattr(obj, "font_family", "Arial"),
                font_size=getattr(obj, "font_size", 24),
                bold=getattr(obj, "bold", False),
                italic=getattr(obj, "italic", False),
                letter_spacing=getattr(obj, "letter_spacing", 0.0),
                line_height=getattr(obj, "line_height", 1.2),
                max_width=getattr(obj, "max_width", None),
                background_padding=getattr(obj, "background_padding", 0),
            )
        except Exception as exc:
            # Fall back to whatever we have rather than hard-failing room(),
            # but warn — silently landing on a stale/zero size otherwise
            # looks identical to a correctly-measured box, which makes a
            # bad text measurement very hard to diagnose.
            print(
                f"Draw.room: warning: text measurement for '{obj.ip}' "
                f"failed, falling back to last known size: {exc}"
            )
            if obj.last_rect:
                _, _, w, h = obj.last_rect
            else:
                w, h = 0.0, 0.0

        x = float(obj.x) if obj.x is not None else (
            obj.last_rect[0] if obj.last_rect else 0.0
        )
        y = float(obj.y) if obj.y is not None else (
            obj.last_rect[1] if obj.last_rect else 0.0
        )
        return _Geom(obj.ip, x, y, float(w), float(h), kind="text")

    raise RoomLayoutError(
        f"Draw.room: unknown object kind {kind!r} for geometry lookup on '{obj.ip}'."
    )


def _apply_geom(window_tag: str, kind: str, obj, x: float, y: float,
                 w: Optional[float] = None, h: Optional[float] = None,
                 resize: bool = False, z_index: Optional[int] = None,
                 animate: bool = False, duration: float = 0.3, ease: str = "ease_out") -> None:
    """Push the computed x/y (and, if resize=True, w/h) back onto the live
    object and repaint. Also applies z_index when provided."""
    from Draw import _bridge
    _window_registry = _bridge.get_window_registry()
def _apply_text_align_modifiers(entry: "_Entry", tok: dict, obj_id: str) -> None:
    """Parse a trailing {"align_text": ..., "valign": ...} modifier dict
    used to align text WITHIN the box room() places it into, separate from
    *where* that box itself goes (which align/anchor/placement controls).
    """
    if "align_text" in tok:
        v = str(tok["align_text"]).strip().lower()
        if v not in _VALID_ALIGN_TEXT:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' has invalid align_text={v!r}; "
                f"expected one of {sorted(_VALID_ALIGN_TEXT)}."
            )
        entry.align_text = v
    if "valign" in tok:
        v = str(tok["valign"]).strip().lower()
        if v not in _VALID_VALIGN:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' has invalid valign={v!r}; "
                f"expected one of {sorted(_VALID_VALIGN)}."
            )
        entry.valign = v


# ── small geometry record ──────────────────────────────────────────────────────

class _Geom:
    """Resolved/working geometry for one room() entry."""

    __slots__ = ("ip", "x", "y", "w", "h", "kind", "resolved")

    def __init__(self, ip: str, x: float = 0.0, y: float = 0.0,
                 w: float = 0.0, h: float = 0.0, kind: str = "shape"):
        self.ip = ip
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.kind = kind          # "shape" | "text" | "panel" | "point"
        self.resolved = False

    def rect(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)


# ── errors ──────────────────────────────────────────────────────────────────────

class RoomLayoutError(ValueError):
    """Raised for any Draw.room() layout problem (cycles, conflicts, bad ids)."""


# ── live-object lookup / geometry resolution ───────────────────────────────────

def _find_live_object(window_tag: str, obj_id: str):
    """
    Locate the live object for obj_id across shape, text, panel, and point
    registries.

    Returns (kind, obj) where kind is "shape" | "text" | "panel" | "point"
    and obj is the underlying ShapeDef / TextDef / PanelDef / PathDef, or
    (None, None) if not found.
    """
    from Draw import _bridge
    _shape_registry = _bridge.get_shape_registry()
    _panel_registry = _bridge.get_panel_registry()
    _window_registry = _bridge.get_window_registry()
    _point_registry = _bridge.get_point_registry()

    # Panels are independent of any particular canvas/window tag.
    p = _panel_registry.get(obj_id)
    if p is not None:
        return "panel", p

    s = _shape_registry.get_by_ip(window_tag, obj_id)
    if s is not None:
        return "shape", s

    pt = _point_registry.get_by_ip(window_tag, obj_id)
    if pt is not None:
        return "point", pt

    try:
        win = _window_registry.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas is not None:
            for t in canvas.text_items:
                if t.ip == obj_id:
                    return "text", t
    except Exception as exc:
        print(
            f"Draw.room: warning: text lookup for '{obj_id}' on display "
            f"'{window_tag}' failed: {exc}"
        )

    return None, None


def _existing_align(kind: str, obj) -> Optional[str]:
    """The align the object was already given at creation time, if any."""
    return getattr(obj, "align", None)


def _current_geom(kind: str, obj, canvas_w: int, canvas_h: int, window_tag: str = "") -> _Geom:
    """Best-effort current geometry of an already-existing object."""
    if kind == "panel":
        return _Geom(obj.ip, float(obj.x), float(obj.y),
                     float(obj.width), float(obj.height), kind="panel")

    if kind == "shape":
        if obj.last_position and obj.last_size:
            ox, oy = obj.last_position
            sw, sh = obj.last_size
            return _Geom(obj.ip, float(ox), float(oy), float(sw), float(sh), kind="shape")
        # Not painted yet — fall back to the same pre-paint estimate the
        # engine itself uses (_shape_preferred_pos), so room() works even
        # before the first repaint.
        from Draw._shapes import _shape_preferred_pos
        sw, sh, ox, oy = _shape_preferred_pos(obj, canvas_w, canvas_h)
        return _Geom(obj.ip, float(ox), float(oy), float(sw), float(sh), kind="shape")

    if kind == "point":
        # obj is a PathDef (registered via Draw.point(..., points=[{"ip": ...}]))
        # Bounds come from the point registry, which reuses the exact same
        # _PointCanvas._path_bounds the renderer itself uses.
        from Draw import _bridge
        _point_registry = _bridge.get_point_registry()
        bounds = _point_registry.get_pixel_bounds(window_tag, obj.ip)
        if bounds is None:
            return _Geom(obj.ip, 0.0, 0.0, 0.0, 0.0, kind="point")
        x, y, w, h = bounds
        return _Geom(obj.ip, float(x), float(y), float(w), float(h), kind="point")

    if kind == "text":
        if getattr(obj, "last_position", None) is not None and getattr(obj, "last_size", None) is not None:
            ox, oy = obj.last_position
            sw, sh = obj.last_size
            return _Geom(obj.ip, float(ox), float(oy), float(sw), float(sh), kind="text")

        # Always compute a real, font-metric-based size for text — never
        # rely solely on last_rect (a stale post-paint cache) or a blind
        # (x, y, 0, 0) guess pre-paint. Without this, Draw.room() was
        # positioning text by treating it as a zero-size point, so anchors
        # like "center" or "right" landed wherever the *origin* should be,
        # not where the actual rendered text box's center/edge is — i.e.
        # Draw.room solved object positioning but never solved text
        # alignment within its own bounding box, since it never knew the
        # box's true dimensions in the first place.
        from Draw._text import measure_text, _resolve_text_value
        try:
            w, h = measure_text(
                _resolve_text_value(obj),
                font_family=getattr(obj, "font_family", "Arial"),
                font_size=getattr(obj, "font_size", 24),
                bold=getattr(obj, "bold", False),
                italic=getattr(obj, "italic", False),
                letter_spacing=getattr(obj, "letter_spacing", 0.0),
                line_height=getattr(obj, "line_height", 1.2),
                max_width=getattr(obj, "max_width", None),
                background_padding=getattr(obj, "background_padding", 0),
            )
        except Exception as exc:
            # Fall back to whatever we have rather than hard-failing room(),
            # but warn — silently landing on a stale/zero size otherwise
            # looks identical to a correctly-measured box, which makes a
            # bad text measurement very hard to diagnose.
            print(
                f"Draw.room: warning: text measurement for '{obj.ip}' "
                f"failed, falling back to last known size: {exc}"
            )
            if obj.last_rect:
                _, _, w, h = obj.last_rect
            else:
                w, h = 0.0, 0.0

        x = float(obj.x) if obj.x is not None else (
            obj.last_rect[0] if obj.last_rect else 0.0
        )
        y = float(obj.y) if obj.y is not None else (
            obj.last_rect[1] if obj.last_rect else 0.0
        )
        return _Geom(obj.ip, x, y, float(w), float(h), kind="text")

    raise RoomLayoutError(
        f"Draw.room: unknown object kind {kind!r} for geometry lookup on '{obj.ip}'."
    )


def _apply_geom(window_tag: str, kind: str, obj, x: float, y: float,
                 w: Optional[float] = None, h: Optional[float] = None,
                 resize: bool = False, z_index: Optional[int] = None,
                 animate: bool = False, duration: float = 0.3, ease: str = "ease_out") -> None:
    """Push the computed x/y (and, if resize=True, w/h) back onto the live
    object and repaint. Also applies z_index when provided."""
    from Draw import _bridge
    _window_registry = _bridge.get_window_registry()
    _panel_registry = _bridge.get_panel_registry()

    if z_index is not None and hasattr(obj, "z"):
        obj.z = z_index

    if animate:
        try:
            from Draw import motion
            motion(
                display=window_tag,
                target_ip=obj.ip,
                duration=duration,
                ease=ease,
                x=x,
                y=y,
            )
            return
        except Exception:
            pass

    if kind == "panel":
        if resize and w is not None and h is not None:
            _panel_registry.resize(obj.ip, width=int(round(w)), height=int(round(h)))
        _panel_registry.move(obj.ip, x=int(round(x)), y=int(round(y)))
        return

    if kind == "shape":
        if resize and w is not None and h is not None:
            obj.size_raw = [float(w), float(h)]
            obj.last_size = (float(w), float(h))
            obj._bbox_cache = None
            obj._bbox_cache_key = None
            obj.dirty = True
            if hasattr(obj, "_placed_w"):
                try:
                    delattr(obj, "_placed_w")
                except AttributeError:
                    pass
            if hasattr(obj, "_placed_h"):
                try:
                    delattr(obj, "_placed_h")
                except AttributeError:
                    pass
        obj.x = int(round(x))
        obj.y = int(round(y))
        obj.last_position = (float(x), float(y))
        if hasattr(obj, "_placed_x"):
            try:
                delattr(obj, "_placed_x")
            except AttributeError:
                pass
        if hasattr(obj, "_placed_y"):
            try:
                delattr(obj, "_placed_y")
            except AttributeError:
                pass

        # Graph container re-render hook for Draw.room integration
        if getattr(obj, "ip", None) and str(obj.ip).endswith(":graph_container"):
            graph_ip = str(obj.ip)[:-16]
            try:
                from Draw.graph import graph as graph_reg, _render_graph
                gd = graph_reg.get_by_ip(window_tag, graph_ip)
                if gd is not None:
                    _render_graph(gd)
            except Exception:
                pass

        win = _window_registry.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas is not None:
            canvas._occupied_dirty = True
            canvas.update()
        return

    if kind == "point":
        from Draw import _bridge
        _point_registry = _bridge.get_point_registry()
        if resize and w is not None and h is not None:
            _point_registry.resize_by_ip(window_tag, obj.ip, w, h)
        _point_registry.move_by_ip(window_tag, obj.ip, x, y)
        return

    if kind == "text":
        if resize and w is not None:
            # Text has no independent w/h — its box comes from font-metric
            # measurement of its content. The only lever room() has is the
            # wrap width; height still follows from content at that width
            # (measured fresh next time _current_geom runs for this id).
            obj.max_width = float(w)
        obj.x = int(round(x))
        obj.y = int(round(y))
        # Seed last_rect from the size room() already measured via
        # measure_text(), even pre-paint. Without this, calling room()
        # before the first paintEvent left last_rect=None, so any code
        # reading geometry immediately after room() (chained room() calls,
        # tests, or motion picking up a base position) saw stale/missing
        # size info despite room() having computed it correctly internally.
        if obj.last_rect:
            _, _, old_w, old_h = obj.last_rect
            use_w = float(w) if w is not None else old_w
            use_h = float(h) if h is not None else old_h
        else:
            use_w = float(w) if w is not None else 0.0
            use_h = float(h) if h is not None else 0.0
        obj.last_rect = (float(x), float(y), use_w, use_h)
        win = _window_registry.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas is not None:
            canvas.update()
        return

    raise RoomLayoutError(
        f"Draw.room: unknown object kind {kind!r} when applying geometry to '{obj.ip}'."
    )


# ── entry-spec parsing ──────────────────────────────────────────────────────────

class _Entry:
    """One parsed line of the `scene={...}` dict."""

    __slots__ = (
        "id", "kind",                # "scene_relative" | "object_relative" | "absolute"
        "anchor",                    # scene anchor or object placement string
        "parent_id",                 # for object_relative
        "inside",                    # bool
        "padding", "margin", "offset",
        "abs_x", "abs_y",
        "align_text", "valign",      # text-only: alignment WITHIN the placed box
        "size_spec",                 # #1-25 resize spec: keyword str or dict, or None
        "z",                         # explicit Z-index override
        "pin",                       # sticky pinning toggle
        "raw",
    )

    def __init__(self, id_: str):
        self.id = id_
        self.kind = "scene_relative"
        self.anchor: Optional[str] = None
        self.parent_id: Optional[str] = None
        self.inside = False
        self.padding: Optional[float] = None
        self.margin: Optional[Tuple[float, float, float]] = None
        self.offset: Optional[float] = None
        self.abs_x: Optional[float] = None
        self.abs_y: Optional[float] = None
        self.align_text: Optional[str] = None   # "left" | "center" | "right"
        self.valign: Optional[str] = None        # "top" | "middle" | "bottom"
        self.size_spec: Optional[Any] = None
        self.z: Optional[int] = None
        self.pin: bool = False
        self.raw = None


def _parse_margin_str(token: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in str(token).split(",")]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            nums.append(0.0)
    while len(nums) < 3:
        nums.append(0.0)
    return (nums[0], nums[1], nums[2])


def _parse_entry(obj_id: str, value: Any) -> _Entry:
    entry = _Entry(obj_id)

    # ── absolute: dict {"x":..., "y":...} ──────────────────────────────────
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            entry.kind = "absolute"
            entry.abs_x = float(value["x"])
            entry.abs_y = float(value["y"])
            return entry
        # ── size-only: bare dict with no position, e.g.
        # {"width":"50%","height":"30%"} or {"size":"80%"} — resize the
        # object in place, leaving its current position untouched. ──────
        if _looks_like_size_dict(value):
            entry.kind = "size_only"
            entry.size_spec = value
            return entry
        raise RoomLayoutError(
            f"Draw.room: '{obj_id}' dict value needs either both 'x'/'y' "
            f"(absolute position) or a recognized size key (e.g. 'width', "
            f"'height', 'size', 'fit', 'ratio', ...); got {sorted(value.keys())!r}."
        )

    # ── absolute: tuple (x, y) ───────────────────────────────────────────────
    if isinstance(value, tuple) and len(value) == 2 and all(
        isinstance(v, (int, float)) for v in value
    ):
        entry.kind = "absolute"
        entry.abs_x = float(value[0])
        entry.abs_y = float(value[1])
        return entry

    # ── scene-relative: bare string anchor ──────────────────────────────────
    if isinstance(value, str):
        anchor = _normalize_anchor(value)
        if anchor in {_normalize_anchor(a) for a in _SCENE_ANCHORS}:
            entry.kind = "scene_relative"
            entry.anchor = anchor
            return entry
        # ── size-only: bare keyword, e.g. "box": "auto" — no position,
        # just a (possibly no-op) resize. Anchor names and size keywords
        # are disjoint sets, so this can't be ambiguous. ─────────────────
        kw = value.strip().lower()
        if kw in _SIZE_KEYWORDS:
            entry.kind = "size_only"
            entry.size_spec = kw
            return entry
        raise RoomLayoutError(
            f"Draw.room: '{obj_id}' has unknown scene placement '{value}'."
        )

    # ── object-relative: ["parent_id", "placement", ...modifiers] ──────────
    if isinstance(value, list):
        if len(value) < 2:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' object-relative spec needs at least "
                f"[parent_id, placement]."
            )
        parent_id, placement = value[0], value[1]
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' parent id must be a non-empty string."
            )
        placement_norm = _normalize_anchor(placement)
        if placement_norm not in {_normalize_anchor(a) for a in _OBJECT_PLACEMENTS}:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' has unknown placement '{placement}' "
                f"relative to '{parent_id}'."
            )

        entry.kind = "object_relative"
        entry.parent_id = parent_id.strip()
        entry.anchor = placement_norm

        # remaining modifiers: "inside", numeric offset, "padding", N,
        # "margin_x,y,z", literal margin string, a trailing {"align_text":
        # ..., "valign": ...} dict for text-only in-box alignment, etc.
        rest = list(value[2:])
        i = 0
        while i < len(rest):
            tok = rest[i]
            if isinstance(tok, str) and tok.strip().lower() == "inside":
                entry.inside = True
                i += 1
                continue
            if isinstance(tok, str) and tok.strip().lower() == "padding":
                if i + 1 < len(rest) and isinstance(rest[i + 1], (int, float)):
                    entry.padding = float(rest[i + 1])
                    i += 2
                    continue
                i += 1
                continue
            if isinstance(tok, str) and "," in tok:
                # "margin_x,y,z" or an offset string with commas
                entry.margin = _parse_margin_str(tok)
                i += 1
                continue
            if isinstance(tok, (int, float)):
                # bare numeric offset, e.g. ["panel", "bottom", 15]
                entry.offset = float(tok)
                i += 1
                continue
            if isinstance(tok, str) and tok.strip().lower() == "offset":
                if i + 1 < len(rest) and isinstance(rest[i + 1], (int, float)):
                    entry.offset = float(rest[i + 1])
                    i += 2
                    continue
                i += 1
                continue
            if isinstance(tok, str) and tok.strip().lower() in _SIZE_KEYWORDS:
                # Inline resize keyword, e.g. ["panel", "center", "fit"].
                entry.size_spec = tok.strip().lower()
                i += 1
                continue
            if isinstance(tok, dict) and _looks_like_size_dict(tok):
                # Inline resize dict, e.g.
                # ["panel", "center", {"fit": True, "padding": 20}] or
                # ["panel", "center", {"width": ["panel", "/2"]}].
                entry.size_spec = tok
                i += 1
                continue
            if isinstance(tok, dict):
                # Text-only modifiers for aligning WITHIN the placed box,
                # e.g. ["panel", "top", "inside", {"align_text": "left"}].
                _apply_text_align_modifiers(entry, tok, obj_id)
                i += 1
                continue
            # unrecognized token — ignore for forward-compat, per design's
            # "allow future additions without breaking existing syntax", but
            # surface it so a typo'd modifier (e.g. "padidng") doesn't just
            # vanish silently — that would otherwise look like the modifier
            # was simply never applied, with no clue why.
            print(
                f"Draw.room: warning: '{obj_id}' has an unrecognized "
                f"placement modifier {tok!r} — ignoring it. Recognized "
                f"modifiers: 'inside', 'padding', 'offset', a bare number, "
                f"a 'x,y,z' margin string, a size keyword/dict, or an "
                f"{{'align_text'/'valign'}} dict."
            )
            i += 1

        # ── conflicting gap modifiers ────────────────────────────────────
        # padding / offset / margin all ultimately resolve to the same
        # "gap" value in room()'s apply step, with silent last-one-wins
        # precedence. That means an explicit padding=10 can be quietly
        # discarded by a later bare-number offset in the same spec, with
        # no indication anything was overridden. Fail fast instead so the
        # ambiguity is caught at parse time rather than producing a
        # mysteriously-wrong layout.
        gap_modifiers_set = sum(
            m is not None for m in (entry.padding, entry.offset, entry.margin)
        )
        if gap_modifiers_set > 1:
            raise RoomLayoutError(
                f"Draw.room: '{obj_id}' specifies more than one of "
                f"padding/offset/margin in the same placement spec — only "
                f"one gap modifier is allowed per entry, since the last one "
                f"parsed would otherwise silently override the others."
            )

        return entry

    raise RoomLayoutError(
        f"Draw.room: '{obj_id}' has an unsupported placement value: {value!r}."
    )


# ── dependency graph / cycle detection ─────────────────────────────────────────

def _build_dependency_order(entries: Dict[str, _Entry]) -> List[str]:
    """
    Topologically sort entries so every object_relative entry is resolved
    after its parent. Raises RoomLayoutError on circular references.
    """
    visiting: set = set()
    visited: set = set()
    order: List[str] = []

    def visit(node_id: str, chain: List[str]):
        if node_id in visited:
            return
        if node_id in visiting:
            cycle = " -> ".join(chain + [node_id])
            raise RoomLayoutError(f"Draw.room: circular reference detected: {cycle}")
        entry = entries.get(node_id)
        if entry is None or entry.kind != "object_relative":
            visited.add(node_id)
            order.append(node_id)
            return
        visiting.add(node_id)
        parent_id = entry.parent_id
        if parent_id in entries:
            visit(parent_id, chain + [node_id])
        visiting.discard(node_id)
        visited.add(node_id)
        order.append(node_id)

    for obj_id in entries:
        visit(obj_id, [])
    return order


# ── placement math ──────────────────────────────────────────────────────────────

def _scene_anchor_pos(anchor: str, sw: float, sh: float, cw: float, ch: float,
                       pad: float) -> Tuple[float, float]:
    from Draw._align import calculate_alignment_pos
    return calculate_alignment_pos(anchor, sw, sh, cw, ch, pad=pad)


def _object_relative_pos(
    placement: str,
    parent: _Geom,
    sw: float, sh: float,
    inside: bool,
    gap: float,
) -> Tuple[float, float]:
    """
    Position a child of size (sw, sh) relative to `parent`'s rect.
    `gap` is the offset/padding pushed out from the parent edge (outside
    placement) or in from the parent edge (inside placement).
    """
    px, py, pw, ph = parent.rect()

    if not inside:
        # Child sits OUTSIDE the parent, adjacent to the given edge.
        outside = {
            "top":          (px + (pw - sw) / 2.0,  py - sh - gap),
            "bottom":       (px + (pw - sw) / 2.0,  py + ph + gap),
            "left":         (px - sw - gap,          py + (ph - sh) / 2.0),
            "right":        (px + pw + gap,          py + (ph - sh) / 2.0),
            "top-left":     (px - sw - gap,          py - sh - gap),
            "top-right":    (px + pw + gap,          py - sh - gap),
            "bottom-left":  (px - sw - gap,          py + ph + gap),
            "bottom-right": (px + pw + gap,          py + ph + gap),
            "center":       (px + (pw - sw) / 2.0,   py + (ph - sh) / 2.0),
        }
        return outside.get(placement, (px, py))

    # Child sits INSIDE the parent, against the given edge.
    inside_map = {
        "top":          (px + (pw - sw) / 2.0,  py + gap),
        "bottom":       (px + (pw - sw) / 2.0,  py + ph - sh - gap),
        "left":         (px + gap,               py + (ph - sh) / 2.0),
        "right":        (px + pw - sw - gap,     py + (ph - sh) / 2.0),
        "top-left":     (px + gap,               py + gap),
        "top-right":    (px + pw - sw - gap,     py + gap),
        "bottom-left":  (px + gap,               py + ph - sh - gap),
        "bottom-right": (px + pw - sw - gap,     py + ph - sh - gap),
        "center":       (px + (pw - sw) / 2.0,   py + (ph - sh) / 2.0),
    }
    return inside_map.get(placement, (px, py))


def _anchor_size_reposition(
    mode: str, old_x: float, old_y: float, old_w: float, old_h: float,
    new_w: float, new_h: float,
) -> Tuple[float, float]:
    """#20 Anchor Size / #21 Scale From Center — after a resize, keep one
    edge/corner/center point of the OLD rect fixed instead of re-deriving
    position from the placement anchor. `mode` is a scene-anchor-style
    name: 'left'/'right'/'top'/'bottom'/'center'/'top-left'/etc.
    """
    mode = _normalize_anchor(mode)
    if mode in ("left", "top-left", "bottom-left"):
        x = old_x
    elif mode in ("right", "top-right", "bottom-right"):
        x = old_x + old_w - new_w
    else:  # "center", "top", "bottom"
        x = old_x + (old_w - new_w) / 2.0
    if mode in ("top", "top-left", "top-right"):
        y = old_y
    elif mode in ("bottom", "bottom-left", "bottom-right"):
        y = old_y + old_h - new_h
    else:  # "center", "left", "right"
        y = old_y + (old_h - new_h) / 2.0
    return x, y


def _resolve_fit_cell(
    spec: dict, obj_id: str, kind: str, obj, canvas_w: int, canvas_h: int,
) -> Tuple[float, float, float, float]:
    """#19 Grid Cell Fit — {"fit_cell": True} (reuse the object's own
    Draw.shapes(columns=..., get_ip=...) binding), or an explicit
    {"fit_cell": True, "layout": ip, "cell": (col, row)} / {"fit_cell":
    [ip, (col, row)]} / {"fit_cell": ip} (cell defaults to (0, 0), matching
    Draw._layout's own string-cell shorthand for combined cells).
    Returns the cell's (x, y, w, h) directly — fitting a cell means
    occupying it exactly, so this bypasses room()'s normal anchor-based
    repositioning rather than feeding into it.
    """
    from Draw import _bridge
    _layout_registry = _bridge.get_layout_registry()

    fc = spec.get("fit_cell")
    layout_ref = spec.get("layout")
    cell_ref = spec.get("cell")

    if isinstance(fc, (list, tuple)) and len(fc) == 2:
        layout_ref, cell_ref = fc[0], fc[1]
    elif isinstance(fc, str):
        layout_ref = fc

    if layout_ref is None:
        # No explicit binding given — fall back to the object's own native
        # table-cell binding (set at creation via Draw.shapes(columns=...,
        # get_ip=...) or Draw.text(...) equivalent). Only shapes/text carry
        # these fields; panels don't, so this always needs an explicit
        # layout/cell for panels.
        layout_ref = getattr(obj, "layout", None)
        cell_ref = getattr(obj, "cell", None)
        if layout_ref is None:
            raise RoomSizeError(
                f"'{obj_id}': fit_cell needs either an explicit "
                f"{{'layout': <table ip>, 'cell': (col, row)}} or the "
                f"object must already be bound to a table cell (created "
                f"with columns=... and get_ip=...). Plain {kind}s created "
                f"without either of those have no cell to fit."
            )

    if cell_ref is None:
        cell_ref = (0, 0)

    try:
        if isinstance(cell_ref, str):
            # A string cell ref names a combined-cell layout directly
            # (Draw._layout's own convention — see ShapeDef.cell usage).
            layout_obj = _layout_registry.resolve(cell_ref)
            rect = layout_obj.cell_rect(canvas_w, canvas_h, (0, 0))
        else:
            layout_obj = _layout_registry.resolve(layout_ref)
            rect = layout_obj.cell_rect(canvas_w, canvas_h, tuple(cell_ref))
    except RoomSizeError:
        raise
    except Exception as exc:
        raise RoomSizeError(f"'{obj_id}': fit_cell failed: {exc}") from exc

    return (
        float(rect.left()), float(rect.top()),
        float(rect.width()), float(rect.height()),
    )


# ── public registry ─────────────────────────────────────────────────────────────

class _RoomRegistry:
    """
    Public API: Draw.room(display="main", scene={...}, general={...})

    Resolves a relative-layout scene description and writes the computed
    x/y straight onto the live shape / text / panel objects, then repaints.
    """

    def __call__(
        self,
        *,
        scene: Dict[str, Any],
        type: str = "scene",
        display: Optional[str] = None,
        tag: Optional[str] = None,
        general: Optional[Dict[str, Any]] = None,
        sizes: Optional[Dict[str, Any]] = None,
        container: Optional[str] = None,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        if not isinstance(scene, dict):
            raise RoomLayoutError("Draw.room: 'scene' must be a dict.")
        if not scene and not sizes:
            raise RoomLayoutError(
                "Draw.room: at least one of 'scene' or 'sizes' must be non-empty."
            )

        from Draw import _bridge
        _window_registry = _bridge.get_window_registry()

        window_tag = display or tag
        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise RoomLayoutError(
                    "Draw.room: multiple windows exist; 'display' is required."
                )
            else:
                raise RoomLayoutError("Draw.room: no windows exist to lay out.")

        win = _window_registry.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        canvas_w = canvas.width() if canvas is not None else win.width()
        canvas_h = canvas.height() if canvas is not None else win.height()

        # ── container= : scope this entire scene to a rectangle ────────────
        # Without container=, every anchor/percentage in `scene` resolves
        # against the whole window canvas (canvas_w/canvas_h above), and
        # every applied x/y is a window-absolute coordinate. With
        # container="some_ip", the container's own live rect (found via the
        # same _find_live_object/_current_geom machinery used for
        # object_relative parents — so a container can be a shape, panel,
        # or anything else room() already knows how to measure) becomes the
        # local "canvas" for this call: percentages/anchors are computed
        # against the container's own (w, h), and origin_x/origin_y below
        # get added back in once, at the very end, when writing final
        # positions to the live objects — so everything in between (parent
        # references between entries *within this same scene*, dependency
        # ordering, etc.) works in the container's local coordinate space
        # exactly like a normal window-scoped room() call would.
        #
        # Known limitation (v1): an object_relative entry's `parent` must
        # be another id in this same scene when container= is used — a
        # parent looked up live from outside this call would come back in
        # window-absolute coordinates, which don't mix with the
        # container-local coordinates used everywhere else in this call.
        origin_x, origin_y = 0.0, 0.0
        if container is not None:
            if container in scene or (sizes and container in sizes):
                raise RoomLayoutError(
                    f"Draw.room: container='{container}' cannot also appear "
                    f"as an entry in 'scene' or 'sizes' of the same call."
                )
            c_kind, c_obj = _find_live_object(window_tag, container)
            if c_obj is None:
                raise RoomLayoutError(
                    f"Draw.room: container='{container}' not found (no shape/"
                    f"text/panel/point with that ip exists on display "
                    f"'{window_tag}')."
                )
            c_geom = _current_geom(c_kind, c_obj, canvas_w, canvas_h, window_tag)
            canvas_w, canvas_h = c_geom.w, c_geom.h
            origin_x, origin_y = c_geom.x, c_geom.y

        general = general or {}
        default_padding = float(general.get("padding", 0))

        # ── general Z-axis settings (prevents Z-overlap flickering) ──────
        gen_z_base = general.get("z", general.get("z_index", general.get("z_base", None)))
        gen_z_step = int(general.get("z_step", 1))
        gen_z_auto = bool(general.get("z_auto", gen_z_base is not None))

        # ── general animation settings ────────────────────────────────────
        gen_animate = bool(general.get("animate", False))
        gen_duration = float(general.get("duration", 0.3))
        gen_ease = str(general.get("ease", "ease_out"))

        sizes = sizes or {}

        # ── reserved-keyword check on user ids ──────────────────────────────
        for obj_id in list(scene.keys()) + list(sizes.keys()):
            if obj_id in _RESERVED_KEYWORDS:
                raise RoomLayoutError(
                    f"Draw.room: '{obj_id}' is a reserved keyword and cannot "
                    f"be used as an object id."
                )

        # ── parse every entry ────────────────────────────────────────────────
        entries: Dict[str, _Entry] = {}
        for obj_id, value in scene.items():
            entries[obj_id] = _parse_entry(obj_id, value)

        # ── merge in the sizes= param ────────────────────────────────────────
        # An id already in `scene` gets its size_spec set/overridden (sizes=
        # wins over an inline "fit"-style token, since it's the more
        # explicit of the two). An id NOT in `scene` gets a brand-new
        # size_only pseudo-entry — a pure resize with no repositioning,
        # e.g. Draw.room(scene={...}, sizes={"logo": "half"}) where "logo"
        # isn't otherwise part of this room() call at all.
        for obj_id, size_spec in sizes.items():
            if obj_id in entries:
                entries[obj_id].size_spec = size_spec
            else:
                e = _Entry(obj_id)
                e.kind = "size_only"
                e.size_spec = size_spec
                entries[obj_id] = e

        # object_relative entries may reference parents outside `scene`
        # (already-existing objects on the canvas) — that's fine, they just
        # won't appear in the dependency-ordering dict and are resolved by
        # live lookup instead.
        order = _build_dependency_order(entries)

        # ── locate + validate live objects, catch double-alignment ──────────
        live: Dict[str, Tuple[str, Any]] = {}
        for obj_id, entry in entries.items():
            kind, obj = _find_live_object(window_tag, obj_id)
            if obj is None:
                raise RoomLayoutError(
                    f"Draw.room: no shape/text/panel/point with ip='{obj_id}' was "
                    f"found on display '{window_tag}'."
                )
            existing_align = _existing_align(kind, obj)
            if existing_align is not None and entry.kind in ("scene_relative", "object_relative"):
                # The object was already given an alignment when it was
                # created (Draw.shapes/Draw.text/Draw.panel align=...),
                # and Draw.room is also being asked to place/align it.
                # (size_only / absolute entries don't align anything, so
                # they can't conflict with an existing align= — a pure
                # resize is fine even on an already-aligned object.)
                raise RoomLayoutError(
                    f"Draw.room: '{obj_id}' already has align='{existing_align}' "
                    f"set when it was created, and Draw.room is also trying to "
                    f"align it to '{entry.anchor}'. Two alignments are given — "
                    f"remove one of them (either drop align= on the object, or "
                    f"remove '{obj_id}' from the room scene)."
                )
            live[obj_id] = (kind, obj)

        # ── resolve geometry in dependency order ────────────────────────────
        resolved: Dict[str, _Geom] = {}

        def resolve_parent_geom(parent_id: str) -> _Geom:
            if parent_id in resolved:
                return resolved[parent_id]
            # parent isn't part of this room() call — read it live as-is
            kind, obj = _find_live_object(window_tag, parent_id)
            if obj is None:
                raise RoomLayoutError(
                    f"Draw.room: parent '{parent_id}' not found (no shape/"
                    f"text/panel/point with that ip exists)."
                )
            geom = _current_geom(kind, obj, canvas_w, canvas_h, window_tag)
            resolved[parent_id] = geom
            return geom

        for obj_id in order:
            entry = entries[obj_id]
            kind, obj = live[obj_id]
            base = _current_geom(kind, obj, canvas_w, canvas_h, window_tag)
            sw, sh = base.w, base.h

            if entry.kind == "absolute":
                x, y = entry.abs_x, entry.abs_y

            elif entry.kind == "scene_relative":
                x, y = _scene_anchor_pos(entry.anchor, sw, sh, canvas_w, canvas_h,
                                          default_padding)

            elif entry.kind == "size_only":
                # No position spec at all — leave it exactly where it
                # already is. Any resize happens in the size pass below.
                x, y = base.x, base.y

            elif entry.kind == "object_relative":
                parent_geom = resolve_parent_geom(entry.parent_id)
                gap = default_padding
                if entry.padding is not None:
                    gap = entry.padding
                if entry.offset is not None:
                    gap = entry.offset
                if entry.margin is not None:
                    gap = entry.margin[0]
                x, y = _object_relative_pos(
                    entry.anchor, parent_geom, sw, sh, entry.inside, gap
                )

                # ── text-only: align_text / valign WITHIN the parent box ──
                # This is the piece Draw.room never solved before: placement
                # only ever centered/edge-anchored the text's own (sw, sh)
                # box as a whole. If the caller wants left-aligned text
                # inside a wide panel (e.g. a sidebar row label that should
                # sit at the panel's left edge with padding, not centered
                # in it), that's expressed here via align_text/valign acting
                # on the *parent's* full rect rather than re-deriving the
                # anchor math.
                if kind == "text" and entry.inside and (entry.align_text or entry.valign):
                    px, py, pw, ph = parent_geom.rect()
                    if entry.align_text == "left":
                        x = px + gap
                    elif entry.align_text == "right":
                        x = px + pw - sw - gap
                    elif entry.align_text == "center":
                        x = px + (pw - sw) / 2.0
                    if entry.valign == "top":
                        y = py + gap
                    elif entry.valign == "bottom":
                        y = py + ph - sh - gap
                    elif entry.valign == "middle":
                        y = py + (ph - sh) / 2.0

            geom = _Geom(obj_id, x, y, sw, sh, kind=kind)
            geom.resolved = True
            resolved[obj_id] = geom

        # ── size pass (separate, runs after all positions are resolved) ─────
        # Only touches ids that actually have a size_spec (inline "fit"
        # token, an inline size dict, or an entry from sizes=). Everything
        # else keeps the exact (w, h) it already had from the position
        # pass above, so plain old room() calls are unaffected byte-for-
        # byte. ref_lookup prefers an id's *already size-resolved* geometry
        # (so e.g. "legend" sized off of "graph" sees graph's NEW size if
        # graph was resized earlier in `order`), falling back to a fresh
        # live lookup for ids untouched by this room() call.
        def _size_ref_lookup(ref_id: str) -> Optional[Tuple[float, float]]:
            g = resolved.get(ref_id)
            if g is not None:
                return (g.w, g.h)
            kind2, obj2 = _find_live_object(window_tag, ref_id)
            if obj2 is None:
                return None
            g2 = _current_geom(kind2, obj2, canvas_w, canvas_h, window_tag)
            return (g2.w, g2.h)

        for obj_id in order:
            entry = entries[obj_id]
            if entry.size_spec is None:
                continue

            kind, obj = live[obj_id]
            base_geom = resolved[obj_id]
            old_x, old_y, old_w, old_h = base_geom.rect()

            if entry.kind == "object_relative":
                parent_geom = resolve_parent_geom(entry.parent_id)
                parent_w, parent_h = parent_geom.w, parent_geom.h
            else:
                # scene_relative / absolute / size_only entries have no
                # explicit sizing parent — the canvas is the natural root.
                parent_w, parent_h = float(canvas_w), float(canvas_h)

            spec = entry.size_spec

            # ── #19 fit_cell: pins x/y/w/h directly to a table cell's
            # rect, bypassing the normal anchor-based repositioning below
            # entirely (occupying a cell IS the position, not just a size).
            if isinstance(spec, dict) and "fit_cell" in spec:
                try:
                    fx, fy, new_w, new_h = _resolve_fit_cell(
                        spec, obj_id, kind, obj, canvas_w, canvas_h
                    )
                except RoomSizeError as exc:
                    raise RoomLayoutError(f"Draw.room: '{obj_id}': {exc}") from exc
                new_geom = _Geom(obj_id, fx, fy, new_w, new_h, kind=kind)
                new_geom.resolved = True
                resolved[obj_id] = new_geom
                continue

            try:
                if isinstance(spec, dict) and "match" in spec:
                    new_w, new_h = resolve_match(spec, _size_ref_lookup)
                    scale_center = bool(spec.get("scale_center", False))
                    anchor_size = spec.get("anchor_size")
                else:
                    new_w, new_h, scale_center = resolve_size_spec(
                        spec,
                        base_w=old_w, base_h=old_h,
                        parent_w=parent_w, parent_h=parent_h,
                        ref_lookup=_size_ref_lookup,
                    )
                    anchor_size = spec.get("anchor_size") if isinstance(spec, dict) else None
            except RoomSizeError as exc:
                raise RoomLayoutError(f"Draw.room: '{obj_id}': {exc}") from exc

            new_w = max(0.0, float(new_w))
            new_h = max(0.0, float(new_h))

            # ── #20/#21: keep a specific edge/corner/center fixed instead
            # of re-deriving position from the placement anchor ──────────
            if anchor_size or scale_center:
                new_x, new_y = _anchor_size_reposition(
                    anchor_size or "center", old_x, old_y, old_w, old_h, new_w, new_h,
                )
            elif entry.kind == "scene_relative":
                new_x, new_y = _scene_anchor_pos(
                    entry.anchor, new_w, new_h, canvas_w, canvas_h, default_padding
                )
            elif entry.kind == "object_relative":
                gap = default_padding
                if entry.padding is not None:
                    gap = entry.padding
                if entry.offset is not None:
                    gap = entry.offset
                if entry.margin is not None:
                    gap = entry.margin[0]
                parent_geom = resolve_parent_geom(entry.parent_id)
                new_x, new_y = _object_relative_pos(
                    entry.anchor, parent_geom, new_w, new_h, entry.inside, gap
                )
            else:
                # size_only / absolute: no placement anchor to re-derive
                # from — keep the old top-left corner fixed by default.
                new_x, new_y = old_x, old_y

            new_geom = _Geom(obj_id, new_x, new_y, new_w, new_h, kind=kind)
            new_geom.resolved = True
            resolved[obj_id] = new_geom

        # ── apply ────────────────────────────────────────────────────────────
        for idx, obj_id in enumerate(order):
            geom = resolved[obj_id]
            kind, obj = live[obj_id]
            entry = entries[obj_id]

            # Determine Z-index to prevent Z-fighting / overlap bugs
            z_val = entry.z
            if z_val is None and gen_z_base is not None:
                z_val = int(gen_z_base) + idx * gen_z_step
            elif z_val is None and gen_z_auto:
                z_val = idx * gen_z_step

            _apply_geom(
                window_tag, kind, obj,
                geom.x + origin_x, geom.y + origin_y, geom.w, geom.h,
                resize=(entry.size_spec is not None),
                z_index=z_val,
                animate=gen_animate,
                duration=gen_duration,
                ease=gen_ease,
            )

        return {
            obj_id: (
                resolved[obj_id].x + origin_x,
                resolved[obj_id].y + origin_y,
                resolved[obj_id].w,
                resolved[obj_id].h,
            )
            for obj_id in order
        }

    def responsive(
        self,
        *,
        display: Optional[str] = None,
        tag: Optional[str] = None,
        breakpoints: Dict[str, Dict[str, Any]],
        general: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Responsive layout manager.
        Automatically re-evaluates layout scenes based on window width breakpoints.
        """
        from Draw import _bridge
        _window_registry = _bridge.get_window_registry()
        window_tag = display or tag or _window_registry.list_tags()[0]
        win = _window_registry.get(window_tag)

        def _evaluate_responsive():
            cur_w = win.width()
            for condition, layout_kwargs in breakpoints.items():
                cond = condition.strip()
                match = False
                if cond.startswith("<="):
                    val = float(cond[2:].strip())
                    match = cur_w <= val
                elif cond.startswith("<"):
                    val = float(cond[1:].strip())
                    match = cur_w < val
                elif cond.startswith(">="):
                    val = float(cond[2:].strip())
                    match = cur_w >= val
                elif cond.startswith(">"):
                    val = float(cond[1:].strip())
                    match = cur_w > val
                elif cond.startswith("=="):
                    val = float(cond[2:].strip())
                    match = cur_w == val

                if match:
                    scene_spec = layout_kwargs.get("scene", {})
                    sizes_spec = layout_kwargs.get("sizes", None)
                    container_spec = layout_kwargs.get("container", None)
                    gen_spec = dict(general or {})
                    gen_spec.update(layout_kwargs.get("general", {}))
                    self(
                        display=window_tag,
                        scene=scene_spec,
                        sizes=sizes_spec,
                        container=container_spec,
                        general=gen_spec,
                    )
                    break

        _evaluate_responsive()
        win._draw_on_resize = _evaluate_responsive


room = _RoomRegistry()
