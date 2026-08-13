"""
Draw._senses_redesign
======================
Unified input / event / sense system for Draw.

This module replaces the hard-coded event handlers that lived as free
functions in ``_connectors.py`` with a single, composable architecture
built on two primitives:

    InputEvent   — a normalised event object carrying all platform data
    EventProcessor — receives raw Qt events, derives gestures, dispatches

Design principles
-----------------
1. **One event type, many sources.**  Mouse, keyboard, touch, and spatial
   events all share the same ``InputEvent`` envelope.  This makes filters
   and callbacks uniform.

2. **Gesture derivation, not enumeration.**  Drag, long-press, and
   double-click are *derived* from press/move/release streams inside the
   ``EventProcessor``, not hard-coded in dozens of ``handle_canvas_*``
   functions.

3. **Registry indexes.**  The ``_SenseRegistryV2`` keeps ip→senses and
   type→senses indexes for O(1) dispatch, identical to the pattern the
   original ``_SenseRegistry`` established.

4. **Cursor management.**  Shapes with registered interaction senses
   automatically show a ``PointingHandCursor`` on hover — the long-
   requested ``mouse_type`` capability — without any extra API.

5. **Backward-compatible.**  The new registry exposes exactly the same
   ``__call__``, ``get``, ``get_by_ip``, ``dispatch_mouse_event``,
   ``dispatch_key_event``, ``evaluate_callable_senses``, and
   ``evaluate_proximity_senses`` signatures that ``_connectors.py``'s
   ``_SenseRegistry`` provides, so the existing ``_ConnectorRegistry``
   and canvas delegation code continue working without modification.

Thread-safety
-------------
All methods that mutate shared state (``_items``, indexes) acquire
``_lock``.  The ``EventProcessor`` is only ever called from the Qt main
thread (from canvas event overrides), so it does **not** need locking.
The ``evaluate_*`` helpers are designed to be safe from either thread.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  InputEvent — normalised event envelope
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class InputEvent:
    """A single, normalised input event.

    Every interaction — mouse press, key stroke, touch point, spatial
    proximity check — is expressed as an ``InputEvent``.  This lets every
    downstream consumer (senses, connectors, gesture derivers) operate on
    a single type instead of switching on Qt event subclasses.

    Fields
    ------
    kind : str
        Canonical event name (``mouse_press``, ``key_press``, …).
    ip : str | None
        Shape identity this event was dispatched against, or ``None`` for
        global / unscoped events (keyboard).
    button : str | None
        ``"left"`` / ``"right"`` / ``"middle"`` for mouse events.
    x : float
        Canvas-local x coordinate (mouse/touch).
    y : float
        Canvas-local y coordinate (mouse/touch).
    key : str | None
        Human-readable key name for keyboard events.
    modifiers : list[str]
        Active modifier keys (``"shift"``, ``"ctrl"``, ``"alt"``, ``"meta"``).
    delta : float
        Scroll wheel delta (positive = up).
    timestamp : float
        ``time.perf_counter()`` when the event was created.
    meta : dict
        Arbitrary extra data (touch_id, drag offsets, …).
    """
    kind:       str
    ip:         Optional[str]       = None
    button:     Optional[str]       = None
    x:          float               = 0.0
    y:          float               = 0.0
    key:        Optional[str]       = None
    modifiers:  List[str]           = field(default_factory=list)
    delta:      float               = 0.0
    timestamp:  float               = field(default_factory=time.perf_counter)
    meta:       Dict[str, Any]      = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
#  Sense type constants and aliases
# ══════════════════════════════════════════════════════════════════════════════

_MOUSE_SENSE_TYPES: Set[str] = {
    "mouse_click", "mouse_leftclick", "mouse_rightclick",
    "mouse_doubleclick", "mouse_leftdoubleclick", "mouse_rightdoubleclick",
    "mouse_middledoubleclick", "mouse_hover", "mouse_leave",
    "mouse_press", "mouse_release",
    "mouse_scroll_up", "mouse_scroll_down",
    "drag_start", "drag_move", "drag_end",
    "mouse_longpress", "context_menu",
}
_KEYBOARD_SENSE_TYPES: Set[str] = {"key_press", "key_release"}
_CAMERA_SENSE_TYPES:   Set[str] = {"camera_motion", "camera_face", "camera_gesture"}
_FOCUS_SENSE_TYPES:    Set[str] = {"focus_in", "focus_out"}
_SPATIAL_SENSE_TYPES:  Set[str] = {"proximity", "overlap", "distance"}
_TOUCH_SENSE_TYPES:    Set[str] = {"touch_start", "touch_move", "touch_end"}

_ALL_SENSE_TYPES: Set[str] = (
    _MOUSE_SENSE_TYPES | _KEYBOARD_SENSE_TYPES
    | _CAMERA_SENSE_TYPES | _FOCUS_SENSE_TYPES
    | _SPATIAL_SENSE_TYPES | _TOUCH_SENSE_TYPES
)

_SENSE_ALIASES: Dict[str, str] = {
    "click":        "mouse_click",
    "leftclick":    "mouse_leftclick",
    "rightclick":   "mouse_rightclick",
    "doubleclick":  "mouse_doubleclick",
    "dblclick":     "mouse_doubleclick",
    "leftdoubleclick":   "mouse_leftdoubleclick",
    "leftdblclick":      "mouse_leftdoubleclick",
    "rightdoubleclick":  "mouse_rightdoubleclick",
    "rightdblclick":     "mouse_rightdoubleclick",
    "middledoubleclick": "mouse_middledoubleclick",
    "middledblclick":    "mouse_middledoubleclick",
    "middleclick":       "mouse_click",
    "hover":        "mouse_hover",
    "leave":        "mouse_leave",
    "press":        "mouse_press",
    "release":      "mouse_release",
    "scroll_up":    "mouse_scroll_up",
    "scroll_down":  "mouse_scroll_down",
    "keypress":     "key_press",
    "keyrelease":   "key_release",
    "key":          "key_press",
    "focus":        "focus_in",
    "focusin":      "focus_in",
    "blur":         "focus_out",
    "focusout":     "focus_out",
    "drag":         "drag_start",
    "dragstart":    "drag_start",
    "drag_stop":    "drag_end",
    "dragend":      "drag_end",
    "dragmove":     "drag_move",
    "longpress":    "mouse_longpress",
    "long_press":   "mouse_longpress",
    "hold":         "mouse_longpress",
    "contextmenu":  "context_menu",
    "right_click":  "mouse_rightclick",
    # spatial aliases
    "near":         "proximity",
    "close":        "proximity",
    "collide":      "overlap",
    # touch aliases
    "touchstart":   "touch_start",
    "touchmove":    "touch_move",
    "touchend":     "touch_end",
}


def _normalize_sense_type(raw: object) -> str:
    """Convert an alias or raw string to the canonical sense type name."""
    token = str(raw or "").strip().lower()
    return _SENSE_ALIASES.get(token, token)


# ══════════════════════════════════════════════════════════════════════════════
#  Cursor shape constants
# ══════════════════════════════════════════════════════════════════════════════

# Maps user-facing mouse_type strings → Qt CursorShape values.
# Actual Qt imports happen lazily to avoid import-order issues.
_CURSOR_ALIASES: Dict[str, str] = {
    "pointer":   "PointingHandCursor",
    "hand":      "PointingHandCursor",
    "grab":      "OpenHandCursor",
    "grabbing":  "ClosedHandCursor",
    "crosshair": "CrossCursor",
    "text":      "IBeamCursor",
    "move":      "SizeAllCursor",
    "wait":      "WaitCursor",
    "not-allowed": "ForbiddenCursor",
    "default":   "ArrowCursor",
    "none":      "BlankCursor",
}


# ══════════════════════════════════════════════════════════════════════════════
#  SenseRecord — one registered sense (backward-compatible with original)
# ══════════════════════════════════════════════════════════════════════════════

def _as_bool(value: object) -> bool:
    """Coerce truthy/falsy values including string forms."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


@dataclass
class SenseRecord:
    """Represents one registered sense condition.

    This is API-identical to the original ``_connectors.SenseRecord`` so
    that every downstream consumer (``_ConnectorRegistry``, user code)
    works without changes.
    """
    id: str
    sense_type: str
    ip: Optional[str]
    key: Optional[List[str]]
    active: bool
    value: Any
    meta: Dict[str, Any]
    debounce: Optional[float] = None

    # ── cursor shape (new capability) ────────────────────────────────────────
    mouse_type: Optional[str] = None

    # ── internal bookkeeping ─────────────────────────────────────────────────
    _triggered: bool     = field(default=False, init=False, repr=False)
    _trigger_time: float = field(default=0.0,   init=False, repr=False)
    _last_trigger_time: float = field(default=0.0, init=False, repr=False)
    _trigger_count: int  = field(default=0,     init=False, repr=False)

    def trigger(self) -> None:
        """Mark this sense as having fired right now."""
        now = time.perf_counter()
        if self.debounce is not None and self.debounce > 0:
            debounce_sec = self.debounce / 1000.0 if self.debounce > 2.0 else self.debounce
            if now - self._last_trigger_time < debounce_sec:
                return
        self._triggered       = True
        self._trigger_time    = now
        self._last_trigger_time = now
        self._trigger_count  += 1
        self.active           = True

    def consume(self) -> bool:
        """Return True and clear the trigger flag, or False if not triggered."""
        if self._triggered:
            self._triggered = False
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "sense_type": self.sense_type,
            "ip": self.ip, "key": self.key, "active": self.active,
            "value": self.value, "meta": dict(self.meta),
            "trigger_count": self._trigger_count, "debounce": self.debounce,
            "mouse_type": self.mouse_type,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  _SenseRegistryV2 — the public Draw.senses singleton
# ══════════════════════════════════════════════════════════════════════════════

class _SenseRegistryV2:
    """Unified sense registry.

    Drop-in replacement for ``_connectors._SenseRegistry`` with the same
    ``__call__``, ``get``, ``clear``, ``dispatch_*``, and ``evaluate_*``
    signatures.  New capabilities:

    * ``mouse_type`` parameter on registration — shapes with any active
      interaction sense automatically change the cursor on hover.
    * Touch event dispatching via the existing ``dispatch_mouse_event``
      codepath (touch events map to the same canonical names).
    """

    def __init__(self) -> None:
        self._items: Dict[str, SenseRecord]            = {}
        self._counter: int                             = 0
        self._ip_index: Dict[str, List[SenseRecord]]   = {}
        self._type_index: Dict[str, List[SenseRecord]] = {}
        self._lock = threading.Lock()
        self._callable_sense_error_cache: Dict[str, str] = {}

    # ── registration (Draw.senses(...)) ──────────────────────────────────────

    def __call__(
        self,
        expression: object = None,
        *,
        id: Optional[object]    = None,
        ip: Optional[object]    = None,
        type: Optional[object]  = None,
        sense_type: Optional[object] = None,
        key: Optional[object]   = None,
        active: Optional[object] = None,
        debounce: Optional[float] = None,
        mouse_type: Optional[str] = None,
        # proximity-sense extras
        target: Optional[str]   = None,
        threshold: Optional[float] = None,
        **meta: Any,
    ) -> SenseRecord:
        """Register a sense and return its ``SenseRecord``.

        Parameters
        ----------
        expression
            Positional first argument.  Can be a known sense-type string
            (``"click"``), a callable predicate (``lambda: x > 5``), or a
            static boolean value.
        id
            Explicit sense id.  Auto-generated if omitted.
        ip
            Shape identity this sense is scoped to.
        type / sense_type
            Explicit sense-type string (alias for passing the type via
            keyword).
        key
            Key filter for keyboard senses.
        active
            Initial active state.
        debounce
            Minimum interval between successive triggers (ms if >2, else
            seconds — same heuristic as the original).
        mouse_type
            Cursor shape shown when hovering an ip with this sense.
            ``"pointer"`` / ``"hand"`` / ``"grab"`` / ``"crosshair"`` / …
            Omit to inherit the default arrow cursor.
        target
            For proximity/overlap senses: the other shape ip.
        threshold
            For proximity senses: distance threshold in px.
        **meta
            Arbitrary metadata stored on the record.
        """
        sense_id  = str(id) if id is not None else f"sense_{self._counter}"
        self._counter += 1

        raw_type      = sense_type or type or expression
        resolved_type = _normalize_sense_type(raw_type) if raw_type is not None else ""

        if resolved_type in _ALL_SENSE_TYPES:
            record_type  = resolved_type
            record_value = None
            is_active    = _as_bool(active) if active is not None else False
        elif callable(expression):
            record_type  = "callable"
            record_value = expression
            is_active    = _as_bool(active) if active is not None else False
        else:
            record_type  = "value"
            record_value = expression
            is_active    = _as_bool(active if active is not None else expression)

        key_filter: Optional[List[str]] = None
        if key is not None:
            if isinstance(key, (list, tuple)):
                key_filter = [str(k).strip() for k in key]
            else:
                key_filter = [str(key).strip()]

        ip_str = str(ip) if ip is not None else None

        debounce_val = debounce
        if debounce_val is None:
            debounce_val = meta.pop("debounce", None)
        if debounce_val is not None:
            debounce_val = float(debounce_val)

        # Resolve mouse_type
        resolved_mouse = None
        if mouse_type is not None:
            resolved_mouse = str(mouse_type).strip().lower()
        elif ip_str is not None and resolved_type in _MOUSE_SENSE_TYPES:
            # Auto-assign hand cursor for clickable shapes
            if resolved_type in {
                "mouse_click", "mouse_leftclick", "mouse_rightclick",
                "mouse_press", "context_menu",
            }:
                resolved_mouse = "pointer"

        # proximity extras go into meta
        if target is not None:
            meta["target"]    = target
        if threshold is not None:
            meta["threshold"] = float(threshold)

        record = SenseRecord(
            id=sense_id, sense_type=record_type, ip=ip_str,
            key=key_filter, active=is_active, value=record_value,
            meta=dict(meta), debounce=debounce_val,
            mouse_type=resolved_mouse,
        )
        with self._lock:
            self._items[sense_id] = record
            if ip_str is not None:
                self._ip_index.setdefault(ip_str, []).append(record)
            self._type_index.setdefault(record_type, []).append(record)
        return record

    # ── query helpers ─────────────────────────────────────────────────────────

    def get(self, id: object) -> Optional[SenseRecord]:
        return self._items.get(str(id))

    def get_by_ip(self, ip: str) -> List[SenseRecord]:
        return list(self._ip_index.get(ip, []))

    def get_by_type(self, sense_type: str) -> List[SenseRecord]:
        return list(self._type_index.get(sense_type, []))

    def list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._items.values()]

    def clear(self, id: Optional[object] = None) -> None:
        with self._lock:
            if id is None:
                self._items.clear(); self._ip_index.clear()
                self._type_index.clear()
                self._counter = 0
                return
            key    = str(id)
            record = self._items.pop(key, None)
            if record is not None:
                if record.ip and record.ip in self._ip_index:
                    try: self._ip_index[record.ip].remove(record)
                    except ValueError: pass
                if record.sense_type in self._type_index:
                    try: self._type_index[record.sense_type].remove(record)
                    except ValueError: pass

    def get_menu(self, ip: str) -> List[Dict[str, Any]]:
        """Retrieve context-menu items registered for an ip."""
        for record in self._ip_index.get(ip, []):
            if record.sense_type == "context_menu":
                menu = record.meta.get("menu") or record.meta.get("items")
                if isinstance(menu, list):
                    return menu
        return []

    # ── cursor management (new) ───────────────────────────────────────────────

    def get_cursor_for_ip(self, ip: str) -> Optional[str]:
        """Return the mouse_type string for an ip, or None if default.

        Called by the ``EventProcessor`` on hover to decide if the canvas
        cursor should change.  Returns the *first* non-None mouse_type
        among the senses registered for this ip.
        """
        for record in self._ip_index.get(ip, []):
            if record.mouse_type is not None:
                return record.mouse_type
        return None

    # ── canvas click/region sensing (backward compat) ─────────────────────────

    def _resolve_canvas(self, display: Optional[str] = None):
        from Draw._window import window as _wr
        tag = display
        if tag is None:
            tags = _wr.list_tags()
            if len(tags) == 1:
                tag = tags[0]
            elif len(tags) > 1:
                raise ValueError("Draw.senses: multiple windows — 'display' is required.")
            else:
                raise ValueError("Draw.senses: no windows exist. Call Draw.window() first.")
        win = _wr.get(tag)
        from Draw._text import _get_or_create_canvas
        return _get_or_create_canvas(tag, win)

    def first_click(self, display: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """(x, y) where the most recent LEFT mouse-button press landed."""
        canvas = self._resolve_canvas(display)
        return getattr(canvas, "_last_lclick_press_pos", None)

    def last_release(self, display: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """(x, y) where the most recent LEFT mouse-button release landed."""
        canvas = self._resolve_canvas(display)
        return getattr(canvas, "_last_lclick_release_pos", None)

    def region(self, display: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Rectangle spanned by the last completed left-click drag."""
        canvas = self._resolve_canvas(display)
        start = getattr(canvas, "_last_lclick_press_pos", None)
        end   = getattr(canvas, "_last_lclick_release_pos", None)
        if start is None or end is None:
            return None
        x0, y0 = start
        x1, y1 = end
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        return {"start": (x0, y0), "end": (x1, y1), "rect": (x, y, w, h)}

    def capture_region(self, display: Optional[str] = None, as_array: bool = True):
        """Grab pixels inside the last press→release rectangle."""
        from PySide6.QtCore import QRect  # pyrefly: ignore [missing-import]
        canvas = self._resolve_canvas(display)
        reg = self.region(display)
        if reg is None:
            return None
        x, y, w, h = reg["rect"]
        if w < 1 or h < 1:
            return None
        pixmap = canvas.grab(QRect(int(x), int(y), int(w), int(h)))
        if not as_array:
            return pixmap
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "Draw.senses.capture_region(as_array=True) requires numpy. "
                "Install it with: pip install numpy — or pass as_array=False "
                "to get a QPixmap instead."
            ) from exc
        from PySide6.QtGui import QImage  # pyrefly: ignore [missing-import]
        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        iw, ih = image.width(), image.height()
        if iw == 0 or ih == 0:
            return np.zeros((0, 0, 4), dtype=np.uint8)
        bpl = image.bytesPerLine()
        buf = image.constBits()
        if hasattr(buf, "setsize"):
            buf.setsize(ih * bpl)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((ih, bpl // 4, 4))
        return arr[:, :iw, :].copy()

    def clear_region(self, display: Optional[str] = None) -> None:
        """Reset the tracked press/release positions."""
        canvas = self._resolve_canvas(display)
        canvas._last_lclick_press_pos = None
        canvas._last_lclick_release_pos = None

    # ── event dispatchers (called by EventProcessor / canvas) ─────────────────

    def dispatch_mouse_event(
        self,
        event_type: str,
        ip: str,
        button: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dispatch a mouse/touch event to matching senses.

        This is the *same signature* as the original ``_SenseRegistry`` so
        that every existing ``handle_canvas_*`` function keeps working.
        """
        for record in self._ip_index.get(ip, []):
            matched = False
            if record.sense_type == event_type:
                matched = True
            elif event_type == "mouse_click" and record.sense_type in (
                "mouse_leftclick"  if button == "left"  else "",
                "mouse_rightclick" if button == "right" else "",
            ):
                matched = True
            elif event_type in ("mouse_leftclick", "mouse_rightclick") \
                    and record.sense_type == "mouse_click":
                matched = True
            elif event_type == "mouse_doubleclick" and record.sense_type in (
                "mouse_leftdoubleclick"   if button == "left"   else "",
                "mouse_rightdoubleclick"  if button == "right"  else "",
                "mouse_middledoubleclick" if button == "middle" else "",
            ):
                matched = True
            elif event_type in (
                "mouse_leftdoubleclick", "mouse_rightdoubleclick", "mouse_middledoubleclick",
            ) and record.sense_type == "mouse_doubleclick":
                matched = True
            if matched:
                if meta:
                    record.meta.update(meta)
                record.trigger()

    def dispatch_key_event(
        self,
        event_type: str,
        key_name: str,
        modifiers: Optional[List[str]] = None,
    ) -> None:
        """Dispatch a keyboard event to matching senses."""
        if modifiers is None:
            modifiers = []

        def normalize_key(k: str) -> str:
            k = k.lower().strip()
            return {"enter":"return","ret":"return","esc":"escape",
                    "back":"backspace","del":"delete","spacebar":"space",
                    "space":" "}.get(k, k)

        norm = normalize_key(key_name)
        with self._lock:
            records = list(self._type_index.get(event_type, []))
        for record in records:
            key_matched = record.key is None or any(
                normalize_key(k) == norm for k in record.key
            )
            modifier_matched = True
            req = record.meta.get("modifier") or record.meta.get("modifiers")
            if req is not None:
                req_list = [str(req).lower().strip()] if isinstance(req, str) \
                           else [str(m).lower().strip() for m in req]
                for rm in req_list:
                    if rm == "control": rm = "ctrl"
                    if rm not in modifiers:
                        modifier_matched = False; break
            if key_matched and modifier_matched:
                record.meta["pressed_key"]       = key_name
                record.meta["active_modifiers"]  = modifiers
                record.trigger()

    def dispatch_input_event(self, event: InputEvent) -> None:
        """Dispatch a unified ``InputEvent`` to matching senses.

        This is the *new* entry point used by ``EventProcessor``.  It
        delegates to ``dispatch_mouse_event`` or ``dispatch_key_event``
        depending on the event kind.
        """
        kind = event.kind
        if kind in _KEYBOARD_SENSE_TYPES:
            self.dispatch_key_event(kind, event.key or "", event.modifiers)
        elif event.ip is not None:
            self.dispatch_mouse_event(kind, event.ip, event.button, event.meta)

    # ── tick-based evaluators (unchanged contract) ────────────────────────────

    def evaluate_callable_senses(self) -> None:
        """Poll all callable-predicate senses (called from the bg ticker)."""
        with self._lock:
            callable_senses = list(self._type_index.get("callable", []))
        for record in callable_senses:
            try:
                result = bool(record.value())
                if result and not record.active:
                    record.trigger()
                record.active = result
            except Exception as exc:
                msg = str(exc)
                if self._callable_sense_error_cache.get(record.id) != msg:
                    _logger.warning(
                        "Draw.connectors: callable sense '%s' "
                        "predicate raised: %s — treating as inactive.",
                        record.id, msg
                    )
                    self._callable_sense_error_cache[record.id] = msg
                record.active = False

    def evaluate_proximity_senses(self, window_tag: str) -> None:
        """Check distance between shapes for proximity/overlap/distance senses."""
        with self._lock:
            prox_senses = list(self._type_index.get("proximity", []))
            prox_senses += list(self._type_index.get("overlap", []))
            prox_senses += list(self._type_index.get("distance", []))

        for record in prox_senses:
            ip_a = record.ip
            ip_b = record.meta.get("target")
            if not ip_a or not ip_b:
                continue
            ca = _get_shape_center(window_tag, ip_a)
            cb = _get_shape_center(window_tag, ip_b)
            if ca is None or cb is None:
                continue
            dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            record.meta["distance"] = dist

            threshold = float(record.meta.get("threshold", 50.0))
            if record.sense_type == "proximity":
                if dist <= threshold:
                    record.meta["overlap"] = dist < threshold * 0.5
                    record.trigger()
                else:
                    record.active = False
            elif record.sense_type == "overlap":
                ra = _get_shape_rect(window_tag, ip_a)
                rb = _get_shape_rect(window_tag, ip_b)
                if ra and rb:
                    ax, ay, aw, ah = ra
                    bx, by, bw, bh = rb
                    overlapping = (ax < bx + bw and ax + aw > bx and
                                   ay < by + bh and ay + ah > by)
                    if overlapping:
                        record.meta["overlap"] = True
                        record.trigger()
                    else:
                        record.active = False
            elif record.sense_type == "distance":
                record.meta["distance"] = dist
                record.trigger()   # always fires, distance in meta


# ══════════════════════════════════════════════════════════════════════════════
#  EventProcessor — gesture derivation and unified dispatch
# ══════════════════════════════════════════════════════════════════════════════

class EventProcessor:
    """Receives raw Qt events from the canvas, derives gestures, dispatches.

    This replaces the free ``handle_canvas_*`` functions in
    ``_connectors.py``.  It is instantiated *per-canvas* and holds
    gesture-derivation state (drag tracking, hover set, long-press timer).

    The processor does **not** own the sense registry — it receives it as
    a constructor argument and dispatches through it.  This makes it
    trivial to test in isolation.
    """

    def __init__(self, registry: _SenseRegistryV2, canvas: Any) -> None:
        self._reg    = registry
        self._canvas = canvas

        # ── gesture derivation state ──────────────────────────────────────
        self._hovered_ips: Set[str] = set()
        self._drag_origin: Optional[Tuple[float, float]] = None
        self._drag_started: bool = False
        self._dragged_shape: Any = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)

    # ── public entry points (called from canvas event overrides) ──────────

    def on_mouse_press(self, event: InputEvent) -> None:
        """Handle mouse-press: hit-test, begin drag tracking, dispatch."""
        canvas = self._canvas
        from PySide6.QtCore import QPointF  # pyrefly: ignore [missing-import]
        pos = QPointF(event.x, event.y)

        canvas._mouse_x = event.x
        canvas._mouse_y = event.y

        # raw click tracking
        if event.button == "left":
            canvas._last_lclick_press_pos = (event.x, event.y)
            if getattr(canvas, "_builder_active", None) is None and getattr(canvas, "_builder_queue", None):
                from Draw._shapes import _start_builder_preview
                canvas._builder_active = canvas._builder_queue.pop(0)
                _start_builder_preview(canvas, canvas._builder_active, pos)

        hits = canvas._shapes_at_point(pos)
        canvas._select_input_at_point(pos, {ip for ip, _shape in hits})

        hit_shapes = [s for _, s in hits if hasattr(s, "z")]
        for s in canvas.shape_items:
            s._is_pressed = (s in hit_shapes)

        # drag capture
        from Draw._motion import VelocityTracker
        draggable_hits = [s for s in hit_shapes if not getattr(s, "locked", False)]
        if event.button == "left" and draggable_hits:
            hits_sorted = sorted(draggable_hits, key=lambda s: s.z)
            canvas._dragged_shape = hits_sorted[0]
            canvas._dragged_shape._is_dragged = True
            canvas._dragged_shape._vel_tracker_x = VelocityTracker()
            canvas._dragged_shape._vel_tracker_y = VelocityTracker()

            sx, sy = canvas._dragged_shape.last_position if getattr(canvas._dragged_shape, "last_position", None) else (event.x, event.y)
            canvas._drag_offset = (event.x - sx, event.y - sy)
            canvas._dragged_shape._drag_x = event.x - canvas._drag_offset[0]
            canvas._dragged_shape._drag_y = event.y - canvas._drag_offset[1]

            t = time.perf_counter()
            canvas._dragged_shape._vel_tracker_x.add_sample(t, canvas._dragged_shape._drag_x)
            canvas._dragged_shape._vel_tracker_y.add_sample(t, canvas._dragged_shape._drag_y)

            canvas._drag_origin = (event.x, event.y)
            canvas._drag_started = False
            if canvas._dragged_shape.ip:
                canvas._longpress_ip = canvas._dragged_shape.ip
                canvas._longpress_fired_once = False
                canvas._longpress_timer.start(canvas.longpress_delay_ms)

        # dispatch to senses
        button_name = event.button or "left"
        specific = (
            "mouse_leftclick" if button_name == "left" else
            "mouse_rightclick" if button_name == "right" else
            "mouse_click"
        )
        for ip_str, _shape in hits:
            self._reg.dispatch_mouse_event("mouse_click", ip_str, button_name)
            self._reg.dispatch_mouse_event(specific, ip_str, button_name)
            self._reg.dispatch_mouse_event("mouse_press", ip_str, button_name)
        canvas.setFocus()

    def on_mouse_release(self, event: InputEvent) -> None:
        """Handle mouse-release: drop tracking, dispatch."""
        from PySide6.QtCore import QPointF  # pyrefly: ignore [missing-import]
        canvas = self._canvas
        pos = QPointF(event.x, event.y)
        button_name = event.button or "left"

        canvas._mouse_x = event.x
        canvas._mouse_y = event.y
        canvas._longpress_timer.stop()

        if button_name == "left":
            canvas._last_lclick_release_pos = (event.x, event.y)
            if getattr(canvas, "_builder_active", None) is not None:
                from Draw._shapes import _finalize_builder_shape
                _finalize_builder_shape(canvas, canvas._builder_active, pos)
                canvas._builder_active = None

        if getattr(canvas, "_dragged_shape", None) is not None:
            s = canvas._dragged_shape
            s._is_dragged = False
            vx = s._vel_tracker_x.get_velocity()
            vy = s._vel_tracker_y.get_velocity()
            s._release_velocities = {"x": vx, "y": vy, "position": [vx, vy], "scroll_y": vy}
            if canvas._drag_started:
                if s.ip and s.ip.startswith("scroller_"):
                    s._placed_x = getattr(s, "_drag_x", getattr(s, "_placed_x", s.x))
                    s._placed_y = getattr(s, "_drag_y", getattr(s, "_placed_y", s.y))
                else:
                    s._placed_x = getattr(s, "_drag_x", getattr(s, "_placed_x", s.x)) + canvas._scroll_x
                    s._placed_y = getattr(s, "_drag_y", getattr(s, "_placed_y", s.y)) + canvas._scroll_y
            if canvas._drag_started and s.ip:
                _ox, _oy = canvas._drag_origin if canvas._drag_origin else (event.x, event.y)
                _ddx, _ddy = event.x - _ox, event.y - _oy
                self._reg.dispatch_mouse_event("drag_end", s.ip, "left", meta={"drag_x": event.x, "drag_y": event.y, "drag_origin_x": _ox, "drag_origin_y": _oy, "drag_dx": _ddx, "drag_dy": _ddy})
            canvas._drag_started = False
            canvas._drag_origin = None
            s.motion_started_at = time.perf_counter()
            canvas._dragged_shape = None
        for s in canvas.shape_items:
            s._is_pressed = False
            s._is_dragged = False
        for ip_str, _shape in canvas._shapes_at_point(pos):
            self._reg.dispatch_mouse_event("mouse_release", ip_str, button_name)

    def on_mouse_double_click(self, event: InputEvent) -> None:
        """Handle double-click: dispatch to matching senses."""
        from PySide6.QtCore import QPointF  # pyrefly: ignore [missing-import]
        canvas = self._canvas
        pos = QPointF(event.x, event.y)
        button_name = event.button or "left"
        for ip_str, _shape in canvas._shapes_at_point(pos):
            self._reg.dispatch_mouse_event("mouse_doubleclick", ip_str, button_name)

    def on_mouse_move(self, event: InputEvent) -> None:
        """Handle mouse-move: hover, drag tracking, cursor shape."""
        from PySide6.QtCore import QPointF  # pyrefly: ignore [missing-import]
        from Draw._colour import color as _color_registry
        canvas = self._canvas
        pos = QPointF(event.x, event.y)
        canvas._mouse_x = event.x
        canvas._mouse_y = event.y
        _color_registry.update_mouse(event.x, event.y)

        if getattr(canvas, "_builder_active", None) is not None:
            from Draw._shapes import _update_builder_preview
            _update_builder_preview(canvas, canvas._builder_active, pos)

        now_hovered = {ip for ip, _ in canvas._shapes_at_point(pos)}
        for s in canvas.shape_items:
            s._is_hovered = (s.ip in now_hovered) if s.ip else False

        # drag tracking
        if getattr(canvas, "_dragged_shape", None) is not None:
            s = canvas._dragged_shape
            s._drag_x = event.x - canvas._drag_offset[0]
            s._drag_y = event.y - canvas._drag_offset[1]
            t_now = time.perf_counter()
            s._vel_tracker_x.add_sample(t_now, s._drag_x)
            s._vel_tracker_y.add_sample(t_now, s._drag_y)
            if canvas._drag_origin is not None and s.ip:
                _mox, _moy = canvas._drag_origin
                _mdx, _mdy = event.x - _mox, event.y - _moy
                if not canvas._drag_started and (_mdx * _mdx + _mdy * _mdy) ** 0.5 >= canvas._drag_threshold_px:
                    canvas._drag_started = True
                    if canvas._longpress_timer.isActive():
                        canvas._longpress_timer.stop()
                    self._reg.dispatch_mouse_event("drag_start", s.ip, "left", meta={"drag_x": event.x, "drag_y": event.y, "drag_origin_x": _mox, "drag_origin_y": _moy, "drag_dx": _mdx, "drag_dy": _mdy})
                if canvas._drag_started:
                    self._reg.dispatch_mouse_event("drag_move", s.ip, "left", meta={"drag_x": event.x, "drag_y": event.y, "drag_origin_x": _mox, "drag_origin_y": _moy, "drag_dx": _mdx, "drag_dy": _mdy})
            if s.ip:
                for cfg in canvas._scroller_configs:
                    if s.ip == cfg["thumb_ip"]:
                        if cfg["direction"] == "vertical":
                            travel = max(1.0, float(cfg["track_h"]) - float(cfg["thumb_h"]))
                            raw_t = (event.y - cfg["track_y"] - float(cfg["thumb_h"]) / 2.0) / travel
                            canvas._scroll_y = max(0.0, min(1.0, raw_t)) * max(1.0, float(canvas.height()))
                        else:
                            travel = max(1.0, float(cfg["track_w"]) - float(cfg["thumb_w"]))
                            raw_t = (event.x - cfg["track_x"] - float(cfg["thumb_w"]) / 2.0) / travel
                            canvas._scroll_x = max(0.0, min(1.0, raw_t)) * max(1.0, float(canvas.width()))
                        canvas._update_scroller_thumbs()
                        break

        # hover enter/leave
        if now_hovered != self._hovered_ips:
            for ip_str in now_hovered - self._hovered_ips:
                self._reg.dispatch_mouse_event("mouse_hover", ip_str, None)
            for ip_str in self._hovered_ips - now_hovered:
                self._reg.dispatch_mouse_event("mouse_leave", ip_str, None)
            self._hovered_ips = now_hovered
            canvas.update()
        else:
            self._hovered_ips = now_hovered

        # ── cursor shape management ──────────────────────────────────────
        self._update_cursor(now_hovered)

        # dragged shape must repaint on every move
        if getattr(canvas, "_dragged_shape", None) is not None:
            canvas.update()

    def on_wheel(self, event: InputEvent) -> None:
        """Handle scroll wheel."""
        from PySide6.QtCore import QPointF  # pyrefly: ignore [missing-import]
        canvas = self._canvas
        pos = QPointF(event.x, event.y)
        canvas._mouse_x = event.x
        canvas._mouse_y = event.y

        delta_y = event.meta.get("delta_y", event.delta)
        delta_x = event.meta.get("delta_x", 0.0)
        canvas._scroll_y = max(0.0, canvas._scroll_y - delta_y)
        canvas._scroll_x = max(0.0, canvas._scroll_x - delta_x)
        direction = "mouse_scroll_up" if delta_y > 0 else "mouse_scroll_down"
        for ip_str, _shape in canvas._shapes_at_point(pos):
            self._reg.dispatch_mouse_event(direction, ip_str, None)
        canvas._update_scroller_thumbs()

    # ── cursor shape helper ───────────────────────────────────────────────────

    def _update_cursor(self, hovered_ips: Set[str]) -> None:
        """Set canvas cursor to pointer/hand if hovering an interactive shape."""
        from PySide6.QtCore import Qt  # pyrefly: ignore [missing-import]
        from PySide6.QtGui import QCursor  # pyrefly: ignore [missing-import]

        # Find the first mouse_type among hovered shapes
        for ip in hovered_ips:
            cursor_name = self._reg.get_cursor_for_ip(ip)
            if cursor_name is not None:
                qt_attr = _CURSOR_ALIASES.get(cursor_name, "PointingHandCursor")
                cursor_shape = getattr(Qt.CursorShape, qt_attr, Qt.CursorShape.ArrowCursor)
                self._canvas.setCursor(QCursor(cursor_shape))
                return
        # No interactive shape hovered — restore default
        self._canvas.unsetCursor()


# ══════════════════════════════════════════════════════════════════════════════
#  Shape geometry helpers (identical to _connectors.py — needed for
#  spatial senses to avoid circular imports)
# ══════════════════════════════════════════════════════════════════════════════

def _get_shape_center(window_tag: str, ip: str) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) of shape ``ip`` on ``window_tag``, or None."""
    from Draw import _bridge
    return _bridge.get_shape_center(window_tag, ip)


def _get_shape_rect(window_tag: str, ip: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (x, y, w, h) of shape ``ip``, or None."""
    from Draw import _bridge
    return _bridge.get_shape_rect(window_tag, ip)


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level singleton
# ══════════════════════════════════════════════════════════════════════════════

senses_v2 = _SenseRegistryV2()
