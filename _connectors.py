"""
Draw._connectors  (full revision)
================
Draw.senses     — register input / logic / proximity conditions on shape IPs.
Draw.connectors — wire shapes together through a sense, with optional payload,
                  callback, and built-in shape-link behaviours.

Thread-safety
-------------
The connector tick engine runs in a daemon background thread.  All GUI
operations — canvas redraws, position writes, style changes — must happen on
the main thread.  Every ``work`` callback and every built-in joint/sync step
is dispatched through ``_GUIDispatcher``, a QObject whose ``_fire`` signal is
connected with ``Qt.ConnectionType.QueuedConnection``.  When the background
thread emits the signal, Qt posts a message to the main thread's event queue
so the slot executes safely there.

────────────────────────────────────────────────────────────────────────────────
NEW CAPABILITY: Shape links & joints
────────────────────────────────────────────────────────────────────────────────

Draw.connectors now supports a ``link`` parameter that describes a
*physical or logical relationship* between two canvas shapes (from_ip → to_ip).
This makes it easy to build pendulums, rope chains, springs, magnets, and
synchronized shape pairs without any manual position math.

link types
----------
"pin"
    Rigid attachment — the "join two shapes" joint.  ``to_ip`` is pinned to
    a fixed anchor point on ``from_ip`` (e.g. the bottom-center).  Every
    frame the child's position is recomputed so the anchor stays fixed.
    Parameters:
        anchor       str    anchor point on from_ip.  Default "bottom".
        rotate_with  bool   if True (default), to_ip rotates together with
                            from_ip — e.g. join a circle and a stick, and
                            the stick swings around and spins with the
                            circle as it rotates, keeping its attach point
                            and its orientation relative to the circle
                            fixed.  Pass False for the old rotation-blind
                            behaviour.

"pendulum"
    ``to_ip`` swings as a pendulum whose pivot is the anchor point on
    ``from_ip``.  Physics: gravity + angular damping.  Parameters:
        length     (px)    rod length.  Default = distance between centres.
        gravity    (px/s²) downward acceleration. Default 980.
        damping    (0–1)   energy loss per frame.  Default 0.02.
        angle      (deg)   initial angle from vertical.  Default 45.

"spring"
    ``to_ip`` is connected to ``from_ip`` by a spring joint.  It oscillates
    around the rest length.  Parameters:
        rest_length (px)   natural spring length.  Default = current distance.
        stiffness   (N/px) spring constant.  Default 200.
        damping     (0–1)  energy loss.  Default 0.05.
        mass        (kg)   bob mass.  Default 1.0.

"rope"
    Like "pin" but only pulls (not pushes): the child hangs below the pivot
    and the rope goes taut when the distance exceeds ``length``.  Gravity
    pulls the child down.  Parameters:
        length     (px)   maximum rope length.
        gravity    (px/s²) Default 980.
        damping    (0–1)  Default 0.03.

"sync"
    Copies properties from ``from_ip`` to ``to_ip`` every frame.  Parameters:
        properties  list  which properties to mirror.
                          Options: "x", "y", "position", "color",
                          "opacity", "size", "rotation".
                          Default: ["position"].
        offset_x    px    constant x offset applied to the copied position.
        offset_y    px    constant y offset.

"orbit"
    ``to_ip`` orbits around ``from_ip`` at a fixed radius and angular speed.
    Parameters:
        radius      (px)   orbit radius.  Default = current distance.
        speed       (deg/s) degrees per second. Default 90.
        angle       (deg)  starting angle. Default 0 (rightward).

"magnet"
    ``to_ip`` is attracted to ``from_ip`` with a force that falls off with
    distance.  Parameters:
        strength    (0–∞)  attraction multiplier.  Default 200.
        max_speed   (px/s) velocity cap.  Default 300.
        damping     (0–1)  friction.  Default 0.08.

"distance_lock"
    Hard constraint: keeps ``to_ip`` exactly ``length`` px away from
    ``from_ip``.  Good for rigid rods in articulated chains.
        length      (px)   fixed distance.  Default = current distance.
        anchor      str    anchor point on from_ip: "center"|"top"|"bottom"|
                           "left"|"right"|"top-left"…  Default "center".

Anchor points on a shape
------------------------
Used in "pin", "pendulum", "rope", "distance_lock" to specify which point
on from_ip the joint attaches to:
    "center", "top", "bottom", "left", "right",
    "top-left", "top-right", "bottom-left", "bottom-right"

Proximity senses (new)
----------------------
Use Draw.senses with sense_type "proximity" to detect when two shapes are
within a threshold distance of each other:

    Draw.senses("proximity", id="near", ip="ball", target="wall",
                threshold=30)

``record.meta`` will contain:
    "distance"   — current distance between shape centres
    "overlap"    — True if shapes overlap (distance < sum of radii approx.)

Connector groups
----------------
Tag connectors with ``group="physics"`` to pause / resume the whole group:

    Draw.connectors.pause_group("physics")
    Draw.connectors.resume_group("physics")

Grab / drag-and-drop (new)
---------------------------
Pass ``draggable=True`` (alias ``grab=True``) on a linked connector to let
the person pick up ``to_ip`` with the mouse — it already supports dragging
on the canvas — without the joint fighting them every frame:

    Draw.connectors(from_ip="circle", to_ip="stick", link="pin",
                     anchor="right", draggable=True)

While the shape is held, the joint yields control entirely. The instant it's
released, the joint re-baselines itself around the dropped position — a
"pin" re-attaches at the new spot/orientation, a pendulum/orbit recomputes
its rod length/radius and swing angle from there, and spring/rope/magnet
simply pull back toward their configured rest length/length/strength from
wherever it landed.

────────────────────────────────────────────────────────────────────────────────
Mouse sense types
────────────────────────────────────────────────────────────────────────────────
"mouse_click", "mouse_leftclick", "mouse_rightclick",
"mouse_doubleclick", "mouse_leftdoubleclick", "mouse_rightdoubleclick",
"mouse_middledoubleclick", "mouse_hover", "mouse_leave",
"mouse_press", "mouse_release",
"mouse_scroll_up", "mouse_scroll_down",
"drag_start", "drag_move", "drag_end",
"mouse_longpress" (alias "hold"),
"context_menu"

Double-click is button-aware: "mouse_doubleclick" fires for any button,
while "mouse_leftdoubleclick" / "mouse_rightdoubleclick" /
"mouse_middledoubleclick" fire only for that specific button.

Keyboard: "key_press", "key_release"  (requires key=[...] parameter)

Custom callable: Draw.senses(lambda: condition, id="my_sense")
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

_logger = logging.getLogger(__name__)

from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal, Slot

from Draw._motion import TargetRef, motion as _motion_registry
from Draw._optimize import compilable, register_instance


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)




def _canvas_button_name(button: Qt.MouseButton) -> str:
    if button == Qt.MouseButton.LeftButton:
        return "left"
    if button == Qt.MouseButton.RightButton:
        return "right"
    return "middle"


def handle_canvas_mouse_press(canvas: Any, event: Any) -> None:
    """Handle canvas mouse press, including click senses and drag capture."""
    from Draw._motion import VelocityTracker

    pos = QPointF(event.position())
    canvas._mouse_x = pos.x()
    canvas._mouse_y = pos.y()
    btn = event.button()
    button_name = _canvas_button_name(btn)
    specific = (
        "mouse_leftclick" if button_name == "left" else
        "mouse_rightclick" if button_name == "right" else
        "mouse_click"
    )

    # ── raw click tracking + Draw.shape builder (drag-to-size) queue ───────
    # Recorded unconditionally, independent of whether anything was hit —
    # Draw.senses.first_click/region/capture_region rely on this even when
    # the click lands on empty canvas.
    if btn == Qt.MouseButton.LeftButton:
        canvas._last_lclick_press_pos = (pos.x(), pos.y())
        if getattr(canvas, "_builder_active", None) is None and getattr(canvas, "_builder_queue", None):
            from Draw._shapes import _start_builder_preview
            canvas._builder_active = canvas._builder_queue.pop(0)
            _start_builder_preview(canvas, canvas._builder_active, pos)

    hits = canvas._shapes_at_point(pos)
    canvas._select_input_at_point(pos, {ip for ip, _shape in hits})

    hit_shapes = [s for _, s in hits if hasattr(s, "z")]
    for s in canvas.shape_items:
        s._is_pressed = (s in hit_shapes)

    draggable_hits = [s for s in hit_shapes if not connectors.is_locked(getattr(s, "ip", None))]

    # If clicked on track, immediately grab the corresponding thumb
    if btn == Qt.MouseButton.LeftButton:
        track_hit = None
        for ip_str, s in hits:
            for sc_cfg in getattr(canvas, "_scroller_configs", []):
                if ip_str in (sc_cfg.get("thumb_ip"), sc_cfg.get("track_ip")):
                    track_hit = sc_cfg
                    break
            if track_hit:
                break
        if track_hit:
            thumb_shape = next((s for s in canvas.shape_items if s.ip == track_hit.get("thumb_ip")), None)
            if thumb_shape:
                draggable_hits = [thumb_shape]

    if btn == Qt.MouseButton.LeftButton and draggable_hits:
        hits_sorted = sorted(draggable_hits, key=lambda s: s.z)
        canvas._dragged_shape = hits_sorted[0]
        canvas._dragged_shape._is_dragged = True

        canvas._dragged_shape._vel_tracker_x = VelocityTracker()
        canvas._dragged_shape._vel_tracker_y = VelocityTracker()

        sx, sy = canvas._dragged_shape.last_position if getattr(canvas._dragged_shape, "last_position", None) else (pos.x(), pos.y())
        canvas._drag_offset = (pos.x() - sx, pos.y() - sy)
        canvas._dragged_shape._drag_x = pos.x() - canvas._drag_offset[0]
        canvas._dragged_shape._drag_y = pos.y() - canvas._drag_offset[1]
        canvas._dragged_shape._drag_start_x = canvas._dragged_shape._drag_x
        canvas._dragged_shape._drag_start_y = canvas._dragged_shape._drag_y

        t = time.perf_counter()
        canvas._dragged_shape._vel_tracker_x.add_sample(t, canvas._dragged_shape._drag_x)
        canvas._dragged_shape._vel_tracker_y.add_sample(t, canvas._dragged_shape._drag_y)

        canvas._drag_origin = (pos.x(), pos.y())
        canvas._drag_started = False
        if canvas._dragged_shape.ip:
            canvas._longpress_ip = canvas._dragged_shape.ip
            canvas._longpress_fired_once = False
            canvas._longpress_timer.start(canvas.longpress_delay_ms)

    for ip_str, _shape in hits:
        senses.dispatch_mouse_event("mouse_click", ip_str, button_name)
        senses.dispatch_mouse_event(specific, ip_str, button_name)
        senses.dispatch_mouse_event("mouse_press", ip_str, button_name)
    canvas.setFocus()


def handle_canvas_mouse_release(canvas: Any, event: Any) -> None:
    """Handle canvas mouse release, drop persistence, and release senses."""
    pos = QPointF(event.position())
    canvas._mouse_x = pos.x()
    canvas._mouse_y = pos.y()
    button_name = _canvas_button_name(event.button())
    canvas._longpress_timer.stop()

    if button_name == "left":
        canvas._last_lclick_release_pos = (pos.x(), pos.y())
        if getattr(canvas, "_builder_active", None) is not None:
            from Draw._shapes import _finalize_builder_shape
            _finalize_builder_shape(canvas, canvas._builder_active, pos)
            canvas._builder_active = None

    if getattr(canvas, "_dragged_shape", None) is not None:
        s = canvas._dragged_shape
        s._is_dragged = False
        vx = s._vel_tracker_x.get_velocity()
        vy = s._vel_tracker_y.get_velocity()
        is_scroller_component = False
        if s.ip:
            if s.ip.startswith("scroller_") or "_track" in s.ip or "_thumb" in s.ip:
                is_scroller_component = True
            for sc_cfg in getattr(canvas, "_scroller_configs", []):
                if s.ip in (sc_cfg.get("thumb_ip"), sc_cfg.get("track_ip")):
                    is_scroller_component = True
                    break

        if canvas._drag_started and not is_scroller_component:
            s._placed_x = getattr(s, "_drag_x", getattr(s, "_placed_x", s.x)) + canvas._scroll_x
            s._placed_y = getattr(s, "_drag_y", getattr(s, "_placed_y", s.y)) + canvas._scroll_y

        if is_scroller_component and abs(vy) > 100.0:
            from Draw._scroller import scroller as _scroller_reg
            max_scroll_x, max_scroll_y = canvas._get_max_scroll_range() if hasattr(canvas, "_get_max_scroll_range") else (0.0, 0.0)
            target_y = canvas._scroll_y + vy * 0.25
            _scroller_reg.scroll_to(y=target_y, display=getattr(canvas, "_window_tag", None), animate=True, duration=0.35)

        if canvas._drag_started and s.ip:
            _ox, _oy = canvas._drag_origin if canvas._drag_origin else (pos.x(), pos.y())
            _ddx, _ddy = pos.x() - _ox, pos.y() - _oy
            senses.dispatch_mouse_event("drag_end", s.ip, "left", meta={"drag_x": pos.x(), "drag_y": pos.y(), "drag_origin_x": _ox, "drag_origin_y": _oy, "drag_dx": _ddx, "drag_dy": _ddy})
        canvas._drag_started = False
        canvas._drag_origin = None
        s.motion_started_at = time.perf_counter()
        canvas._dragged_shape = None
    for s in canvas.shape_items:
        s._is_pressed = False
        s._is_dragged = False
    for ip_str, _shape in canvas._shapes_at_point(pos):
        senses.dispatch_mouse_event("mouse_release", ip_str, button_name)


def handle_canvas_mouse_double_click(canvas: Any, event: Any) -> None:
    pos = QPointF(event.position())
    button_name = _canvas_button_name(event.button())
    for ip_str, _shape in canvas._shapes_at_point(pos):
        senses.dispatch_mouse_event("mouse_doubleclick", ip_str, button_name)


def handle_canvas_mouse_move(canvas: Any, event: Any) -> None:
    from Draw._colour import color as _color_registry
    pos = QPointF(event.position())
    canvas._mouse_x = pos.x()
    canvas._mouse_y = pos.y()
    _color_registry.update_mouse(pos.x(), pos.y())

    if getattr(canvas, "_builder_active", None) is not None:
        from Draw._shapes import _update_builder_preview
        _update_builder_preview(canvas, canvas._builder_active, pos)

    now_hovered = {ip for ip, _ in canvas._shapes_at_point(pos)}
    for s in canvas.shape_items:
        s._is_hovered = (s.ip in now_hovered) if s.ip else False
    if getattr(canvas, "_dragged_shape", None) is not None:
        s = canvas._dragged_shape
        s._drag_x = pos.x() - canvas._drag_offset[0]
        s._drag_y = pos.y() - canvas._drag_offset[1]

        move_path = getattr(s, "move_path", None)
        if move_path == "horizontal" or move_path == "x":
            s._drag_y = getattr(s, "_drag_start_y", s._drag_y)
        elif move_path == "vertical" or move_path == "y":
            s._drag_x = getattr(s, "_drag_start_x", s._drag_x)

        inside_ip = getattr(s, "inside", None)
        if inside_ip:
            boundary_shape = None
            if hasattr(canvas, "_shape_by_ip"):
                boundary_shape = canvas._shape_by_ip.get(inside_ip)
            if not boundary_shape:
                for item in canvas.shape_items:
                    if item.ip == inside_ip:
                        boundary_shape = item
                        break
            if boundary_shape and getattr(boundary_shape, "last_position", None) and getattr(boundary_shape, "last_size", None):
                bx, by = boundary_shape.last_position
                bw, bh = boundary_shape.last_size
                sw, sh = getattr(s, "last_size", (0, 0))
                if sw and sh:
                    s._drag_x = max(bx, min(bx + bw - sw, s._drag_x))
                    s._drag_y = max(by, min(by + bh - sh, s._drag_y))
                else:
                    s._drag_x = max(bx, min(bx + bw, s._drag_x))
                    s._drag_y = max(by, min(by + bh, s._drag_y))

        t_now = time.perf_counter()
        s._vel_tracker_x.add_sample(t_now, s._drag_x)
        s._vel_tracker_y.add_sample(t_now, s._drag_y)
        if canvas._drag_origin is not None and s.ip:
            _mox, _moy = canvas._drag_origin
            _mdx, _mdy = pos.x() - _mox, pos.y() - _moy
            if not canvas._drag_started and (_mdx * _mdx + _mdy * _mdy) ** 0.5 >= canvas._drag_threshold_px:
                canvas._drag_started = True
                if canvas._longpress_timer.isActive():
                    canvas._longpress_timer.stop()
                senses.dispatch_mouse_event("drag_start", s.ip, "left", meta={"drag_x": pos.x(), "drag_y": pos.y(), "drag_origin_x": _mox, "drag_origin_y": _moy, "drag_dx": _mdx, "drag_dy": _mdy})
            if canvas._drag_started:
                senses.dispatch_mouse_event("drag_move", s.ip, "left", meta={"drag_x": pos.x(), "drag_y": pos.y(), "drag_origin_x": _mox, "drag_origin_y": _moy, "drag_dx": _mdx, "drag_dy": _mdy})
        if s.ip:
            for cfg in getattr(canvas, "_scroller_configs", []):
                if s.ip in (cfg.get("thumb_ip"), cfg.get("track_ip")):
                    max_scroll_x, max_scroll_y = canvas._get_max_scroll_range() if hasattr(canvas, "_get_max_scroll_range") else (0.0, 0.0)
                    if cfg.get("direction") == "vertical":
                        track_h = float(cfg["track_h"])
                        thumb_h = float(cfg.get("thumb_h", 30.0))
                        travel = max(1.0, track_h - thumb_h)
                        drag_off_y = canvas._drag_offset[1] if getattr(canvas, "_drag_offset", None) else thumb_h / 2.0
                        thumb_target_y = pos.y() - drag_off_y
                        raw_t = (thumb_target_y - cfg["track_y"]) / travel
                        scroll_range = float(cfg.get("max_y")) if cfg.get("max_y") is not None else max(1.0, max_scroll_y)
                        canvas._scroll_y = max(0.0, min(1.0, raw_t)) * scroll_range
                    else:
                        track_w = float(cfg["track_w"])
                        thumb_w = float(cfg.get("thumb_w", 30.0))
                        travel = max(1.0, track_w - thumb_w)
                        drag_off_x = canvas._drag_offset[0] if getattr(canvas, "_drag_offset", None) else thumb_w / 2.0
                        thumb_target_x = pos.x() - drag_off_x
                        raw_t = (thumb_target_x - cfg["track_x"]) / travel
                        scroll_range = float(cfg.get("max_x")) if cfg.get("max_x") is not None else max(1.0, max_scroll_x)
                        canvas._scroll_x = max(0.0, min(1.0, raw_t)) * scroll_range
                    canvas._update_scroller_thumbs()
                    break
    if now_hovered != canvas._hovered_ips:
        for ip_str in now_hovered - canvas._hovered_ips:
            senses.dispatch_mouse_event("mouse_hover", ip_str, None)
        for ip_str in canvas._hovered_ips - now_hovered:
            senses.dispatch_mouse_event("mouse_leave", ip_str, None)
        canvas._hovered_ips = now_hovered
        canvas.update()
    else:
        canvas._hovered_ips = now_hovered

    # A shape being actively dragged must repaint on every move, not only
    # when the hover set changes -- otherwise the canvas only redraws when
    # hover-detection happens to flicker, producing a stepped/jerky drag
    # instead of smooth tracking of the cursor.
    if getattr(canvas, "_dragged_shape", None) is not None:
        canvas.update()


def handle_canvas_wheel(canvas: Any, event: Any) -> None:
    pos = QPointF(event.position())
    canvas._mouse_x = pos.x()
    canvas._mouse_y = pos.y()
    delta = event.angleDelta().y()
    delta_x = event.angleDelta().x()

    configs = getattr(canvas, "_scroller_configs", [])
    has_vertical = any(cfg.get("direction") in ("vertical", "both") for cfg in configs) or (not configs)
    has_horizontal = any(cfg.get("direction") in ("horizontal", "both") for cfg in configs)

    max_scroll_x, max_scroll_y = canvas._get_max_scroll_range() if hasattr(canvas, "_get_max_scroll_range") else (0.0, 0.0)

    if has_vertical:
        if max_scroll_y > 0.0:
            new_y = max(0.0, min(max_scroll_y, canvas._scroll_y - delta))
            canvas._scroll_y = new_y
        else:
            canvas._scroll_y = 0.0

    if has_horizontal:
        if max_scroll_x > 0.0:
            dx = delta_x if delta_x != 0 else (delta if not has_vertical else 0)
            new_x = max(0.0, min(max_scroll_x, canvas._scroll_x - dx))
            canvas._scroll_x = new_x
        else:
            canvas._scroll_x = 0.0

    direction = "mouse_scroll_up" if delta > 0 else "mouse_scroll_down"
    for ip_str, _shape in canvas._shapes_at_point(pos):
        senses.dispatch_mouse_event(direction, ip_str, None)
    if hasattr(canvas, "_update_scroller_thumbs"):
        canvas._update_scroller_thumbs()
    canvas.update()


def handle_canvas_context_menu(canvas: Any, event: Any) -> bool:
    pos = QPointF(event.pos())
    hits = canvas._shapes_at_point(pos)
    if not hits:
        return False
    ip_str, _shape = hits[0]
    senses.dispatch_mouse_event("context_menu", ip_str, "right", meta={"x": pos.x(), "y": pos.y()})
    items = senses.get_menu(ip_str)
    if items:
        from PySide6.QtWidgets import QMenu
        menu = QMenu(canvas)
        for entry in items:
            action = menu.addAction(str(entry.get("label", "")))
            cb = entry.get("callback")
            if callable(cb):
                action.triggered.connect(lambda checked=False, _cb=cb, _ip=ip_str: _cb(_ip))
        menu.exec(event.globalPos())
    event.accept()
    return True


def handle_canvas_touch_event(canvas: Any, event: Any) -> None:
    sense_type = {QEvent.Type.TouchBegin: "touch_start", QEvent.Type.TouchUpdate: "touch_move", QEvent.Type.TouchEnd: "touch_end"}.get(event.type())
    if sense_type is None:
        return
    points = event.points() if hasattr(event, "points") else event.touchPoints()
    for tp in points:
        pos = tp.position() if hasattr(tp, "position") else tp.pos()
        try:
            touch_id = tp.id()
        except Exception:
            touch_id = -1
        for ip_str, _shape in canvas._shapes_at_point(pos):
            senses.dispatch_mouse_event(sense_type, ip_str, None, meta={"touch_id": touch_id, "touch_x": pos.x(), "touch_y": pos.y()})
    event.accept()


def handle_canvas_event(canvas: Any, event: Any) -> bool:
    if event.type() in (QEvent.Type.TouchBegin, QEvent.Type.TouchUpdate, QEvent.Type.TouchEnd):
        handle_canvas_touch_event(canvas, event)
        return True
    return False


def handle_canvas_longpress_timeout(canvas: Any) -> None:
    if not canvas._longpress_ip:
        canvas._longpress_timer.stop()
        return
    senses.dispatch_mouse_event("mouse_longpress", canvas._longpress_ip, "left", meta={"repeat": canvas._longpress_fired_once})
    canvas._longpress_fired_once = True
    canvas._longpress_timer.start(canvas.longpress_repeat_ms)

# ══════════════════════════════════════════════════════════════════════════════
#  GUI dispatcher — routes bg-thread callbacks to main thread
# ══════════════════════════════════════════════════════════════════════════════

class _GUIDispatcher(QObject):
    """
    Routes connector work-callbacks from the background ticker thread to the
    Qt main thread via a queued signal/slot connection.
    Must be constructed on the main thread.
    """
    _fire: Signal = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self._fire.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(object, object)
    def _run(self, fn: Callable, record: Any) -> None:
        """Execute the user callback on the main thread."""
        try:
            fn(record)
        except Exception as exc:
            _logger.warning("Draw.connectors: work callback error: %s", exc)

    def post(self, fn: Callable, record: Any) -> None:
        """Queue fn(record) for execution on the main thread. Thread-safe."""
        self._fire.emit(fn, record)


# ══════════════════════════════════════════════════════════════════════════════
#  Sense type registry
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

_ALL_SENSE_TYPES: Set[str] = (
    _MOUSE_SENSE_TYPES | _KEYBOARD_SENSE_TYPES
    | _CAMERA_SENSE_TYPES | _FOCUS_SENSE_TYPES
    | _SPATIAL_SENSE_TYPES
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
}


def _normalize_sense_type(raw: object) -> str:
    token = str(raw or "").strip().lower()
    return _SENSE_ALIASES.get(token, token)


# ══════════════════════════════════════════════════════════════════════════════
#  Live shape geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_shape_center(window_tag: str, ip: str) -> Optional[Tuple[float, float]]:
    """Return (cx, cy) of shape ``ip`` on ``window_tag``, or None.
    Thin wrapper — see Draw._bridge.get_shape_center for the shared impl."""
    from Draw import _bridge
    return _bridge.get_shape_center(window_tag, ip)


def _get_shape_rect(window_tag: str, ip: str) -> Optional[Tuple[float, float, float, float]]:
    """Return (x, y, w, h) of shape ``ip``, or None.
    Thin wrapper — see Draw._bridge.get_shape_rect for the shared impl."""
    from Draw import _bridge
    return _bridge.get_shape_rect(window_tag, ip)


def _set_shape_pos(window_tag: str, ip: str, x: float, y: float) -> None:
    """Write (x, y) back to shape ``ip`` and schedule a repaint."""
    from Draw._shapes import shapes as _sr
    from Draw._window import window as _wr
    s = _sr.get_by_ip(window_tag, ip)
    if s is None:
        return
    s.x = int(round(x))
    s.y = int(round(y))
    if s.last_position:
        s.last_position = (float(x), float(y))
    try:
        win    = _wr.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas is not None:
            canvas.update()
    except Exception:
        pass


def _current_rotation(s: object) -> float:
    """Return the shape's *currently rendered* rotation in degrees.

    ``s.rotation`` is only the static/base value set at creation time — a
    shape animated with ``Draw.motion``/timeline updates ``s.last_rotation``
    each frame instead (see ``_shapes.py``'s paint-time state merge) without
    touching the base field. Joins that need to track a spinning parent
    (e.g. "pin" with rotate_with) must read the live value, not the base one.
    """
    live = getattr(s, "last_rotation", None)
    if live is not None:
        return float(live)
    return float(getattr(s, "rotation", 0.0) or 0.0)


def _set_shape_rotation(window_tag: str, ip: str, rotation: float) -> None:
    """Write ``rotation`` (degrees) back to shape ``ip`` and schedule a repaint."""
    from Draw._shapes import shapes as _sr
    from Draw._window import window as _wr
    s = _sr.get_by_ip(window_tag, ip)
    if s is None:
        return
    s.rotation = float(rotation)
    s.last_rotation = float(rotation)
    try:
        win    = _wr.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas is not None:
            canvas.update()
    except Exception:
        pass


def _is_shape_dragged(window_tag: str, ip: str) -> bool:
    """True if shape ``ip`` is currently being grabbed/dragged by the mouse.

    Used by the ``draggable``/``grab`` connector option to yield control
    back to the user while they're holding a joint-controlled shape, instead
    of the joint solver fighting the mouse every frame.
    """
    try:
        from Draw._shapes import shapes as _sr
        s = _sr.get_by_ip(window_tag, ip)
        return bool(s is not None and getattr(s, "_is_dragged", False))
    except Exception:
        return False


def _anchor_point(rect: Tuple[float, float, float, float], anchor: str) -> Tuple[float, float]:
    """Return the (x, y) of a named anchor on a shape rect.
    Thin wrapper — see Draw._bridge.get_anchor_point for the shared impl."""
    from Draw import _bridge
    return _bridge.get_anchor_point(rect, anchor)


# ══════════════════════════════════════════════════════════════════════════════
#  SenseRecord
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SenseRecord:
    """Represents one registered sense condition."""
    id: str
    sense_type: str
    ip: Optional[str]
    key: Optional[List[str]]
    active: bool
    value: Any
    meta: Dict[str, Any]
    debounce: Optional[float] = None

    _triggered: bool     = field(default=False, init=False, repr=False)
    _trigger_time: float = field(default=0.0,   init=False, repr=False)
    _last_trigger_time: float = field(default=0.0, init=False, repr=False)
    _trigger_count: int  = field(default=0,     init=False, repr=False)

    def trigger(self) -> None:
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
        }


# ══════════════════════════════════════════════════════════════════════════════
#  _SenseRegistry
# ══════════════════════════════════════════════════════════════════════════════

class _SenseRegistry:
    """Singleton exposed as Draw.senses."""

    def __init__(self) -> None:
        self._items: Dict[str, SenseRecord]        = {}
        self._counter: int                         = 0
        self._ip_index: Dict[str, List[SenseRecord]]   = {}
        self._type_index: Dict[str, List[SenseRecord]] = {}
        self._lock = threading.Lock()
        self._callable_sense_error_cache: Dict[str, str] = {}

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
        # proximity-sense extras
        target: Optional[str]   = None,
        threshold: Optional[float] = None,
        **meta: Any,
    ) -> SenseRecord:
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

        # proximity extras go into meta
        if target is not None:
            meta["target"]    = target
        if threshold is not None:
            meta["threshold"] = float(threshold)

        if "draggable" in meta:
            drag_val = meta.pop("draggable")
            if ip_str:
                if _as_bool(drag_val):
                    connectors.unlock(ip_str)
                else:
                    connectors.lock(ip_str)

        if "locked" in meta:
            locked_val = meta.pop("locked")
            if ip_str:
                if _as_bool(locked_val):
                    connectors.lock(ip_str)
                else:
                    connectors.unlock(ip_str)

        record = SenseRecord(
            id=sense_id, sense_type=record_type, ip=ip_str,
            key=key_filter, active=is_active, value=record_value,
            meta=dict(meta), debounce=debounce_val,
        )
        self._items[sense_id] = record
        if ip_str is not None:
            self._ip_index.setdefault(ip_str, []).append(record)
        with self._lock:
            self._type_index.setdefault(record_type, []).append(record)
        return record

    def get(self, id: object) -> Optional[SenseRecord]:
        return self._items.get(str(id))

    def get_by_ip(self, ip: str) -> List[SenseRecord]:
        return list(self._ip_index.get(ip, []))

    def get_by_type(self, sense_type: str) -> List[SenseRecord]:
        return list(self._type_index.get(sense_type, []))

    def list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._items.values()]

    def clear(self, id: Optional[object] = None) -> None:
        if id is None:
            self._items.clear(); self._ip_index.clear()
            with self._lock:
                self._type_index.clear()
            self._counter = 0
            return
        key    = str(id)
        record = self._items.pop(key, None)
        if record is not None:
            if record.ip and record.ip in self._ip_index:
                try: self._ip_index[record.ip].remove(record)
                except ValueError: pass
            with self._lock:
                if record.sense_type in self._type_index:
                    try: self._type_index[record.sense_type].remove(record)
                    except ValueError: pass

    def get_menu(self, ip: str) -> List[Dict[str, Any]]:
        """Retrieve the menu items registered for context_menu sense on this ip."""
        for record in self._ip_index.get(ip, []):
            if record.sense_type == "context_menu":
                menu = record.meta.get("menu") or record.meta.get("items")
                if isinstance(menu, list):
                    return menu
        return []

    # ── canvas click/region sensing (not tied to any single shape ip) ─────────
    #
    #   Draw.senses.first_click(display="main")   -> (x, y) | None
    #   Draw.senses.last_release(display="main")  -> (x, y) | None
    #   Draw.senses.region(display="main")        -> {"start", "end", "rect"}
    #   Draw.senses.capture_region(display="main") -> numpy uint8 (H, W, 4)
    #
    # "region" is defined by the most recent completed left-click drag on
    # that canvas: press = one corner, release = the other. Works even if
    # the click didn't land on any registered shape.

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
        """(x, y) where the most recent LEFT mouse-button press landed, or None."""
        canvas = self._resolve_canvas(display)
        return getattr(canvas, "_last_lclick_press_pos", None)

    def last_release(self, display: Optional[str] = None) -> Optional[Tuple[float, float]]:
        """(x, y) where the most recent LEFT mouse-button release landed, or None."""
        canvas = self._resolve_canvas(display)
        return getattr(canvas, "_last_lclick_release_pos", None)

    def region(self, display: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        The rectangle spanned by the last completed left-click drag on this
        canvas (press → release). Returns
            {"start": (x1, y1), "end": (x2, y2), "rect": (x, y, w, h)}
        or None if no press/release pair has happened yet.
        """
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
        """
        Grab the pixels inside the last press→release rectangle on this
        canvas. Returns a numpy uint8 array shaped (H, W, 4) in RGBA order
        by default, or a QPixmap when as_array=False. Returns None if no
        press/release pair exists yet, or the rectangle has zero area.
        """
        # pyrefly: ignore [missing-import]
        from PySide6.QtCore import QRect
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
        # pyrefly: ignore [missing-import]
        from PySide6.QtGui import QImage
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
        """Reset the tracked press/release positions for this canvas."""
        canvas = self._resolve_canvas(display)
        canvas._last_lclick_press_pos = None
        canvas._last_lclick_release_pos = None

    # ── event dispatchers (called by canvas) ──────────────────────────────────

    def dispatch_mouse_event(
        self,
        event_type: str,
        ip: str,
        button: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
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
        if modifiers is None:
            modifiers = []

        def normalize_key(k: str) -> str:
            k_low = str(k).lower()
            if k_low == " " or k_low.strip() in {"space", "spacebar"}:
                return "space"
            k_clean = k_low.strip()
            return {"enter":"return","ret":"return","esc":"escape",
                    "back":"backspace","del":"delete"}.get(k_clean, k_clean)

        norm = normalize_key(key_name)
        for record in self._type_index.get(event_type, []):
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

    def evaluate_callable_senses(self) -> None:
        with self._lock:
            callable_senses = list(self._type_index.get("callable", []))
        for record in callable_senses:
            try:
                result = bool(record.value())
                if result and not record.active:
                    record.trigger()
                record.active = result
            except Exception as exc:
                # Previously silent — a broken predicate would just quietly
                # stop firing with no indication why. Dedup by record id so
                # a persistently-broken sense warns once, not every tick.
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
        """Check distance between two shapes for proximity/overlap senses."""
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
                # overlap fires only when shapes' bounding boxes intersect
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
#  Physics state for joints
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _JointState:
    """Per-connector runtime physics state."""
    # pendulum / rope
    angle:      float = 0.0         # radians from vertical (pendulum) or unused
    angle_vel:  float = 0.0         # angular velocity (rad/s)
    # spring / magnet
    vx:         float = 0.0
    vy:         float = 0.0
    # orbit
    orbit_angle: float = 0.0        # current orbit angle in radians
    # pin / rigid-join rotation-follow (lazily captured on first tick, and
    # re-captured after a drag+drop so a grabbed joint re-attaches where
    # it was dropped instead of snapping back)
    pin_local_dx:   Optional[float] = None
    pin_local_dy:   Optional[float] = None
    pin_rot_offset: Optional[float] = None
    # universal
    last_time:  float = field(default_factory=time.perf_counter)


# ══════════════════════════════════════════════════════════════════════════════
#  ConnectorRecord
# ══════════════════════════════════════════════════════════════════════════════

@compilable
@dataclass
class ConnectorRecord:
    """Represents one registered connector."""
    id:       str
    from_ip:  Optional[str]
    sense:    Optional[SenseRecord]
    to_ip:    Optional[str]
    active:   bool
    work:     Optional[Callable]
    payload:  Dict[str, Any]
    from_ref: Optional[TargetRef] = None
    to_ref:   Optional[TargetRef] = None

    # ── link / joint ──────────────────────────────────────────────────────────
    link:         Optional[str]        = None   # "pin"|"pendulum"|"spring"|…
    link_params:  Dict[str, Any]       = field(default_factory=dict)
    _joint:       Optional[_JointState] = field(default=None, init=False, repr=False)
    group:        Optional[str]        = None   # connector group tag
    paused:       bool                 = False  # set by pause_group / resume_group

    # ── drag / grab ────────────────────────────────────────────────────────
    draggable:    bool                 = False  # if True, user can grab to_ip
    _was_dragging: bool                = field(default=False, init=False, repr=False)

    # ── Generic optimize() hook (Draw._optimize.Compilable) ──────────────
    # Every joint solver below (_solve_pin, _solve_pendulum, ...) used to
    # re-read and re-coerce raw link_params (p.get("anchor", "bottom"),
    # _as_bool(p.get("rotate_with", True)), etc.) on every single tick —
    # the same "parse the dict every frame" tax ShapeDef used to pay
    # before its path/bbox caches. _compile() resolves link_params once
    # into typed, defaulted `_resolved_params`; solvers read that instead.
    # Follows the exact pattern documented in _optimize.py — copy this
    # block for any future connector-like record.
    _resolved_params: Optional[Dict[str, Any]] = field(default=None, init=False, repr=False)
    _compiled_sig:    Optional[object]         = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        register_instance(self)

    def _sig(self):
        """Hashable signature of everything that can change what
        link_params should resolve to. link_params values are generally
        flat (str/float/bool) — nested dict/list values fall back to a
        cheap tuple-ified form so an unusual value still hashes instead
        of raising."""
        def _h(v):
            if isinstance(v, dict):
                return tuple(sorted((k, _h(x)) for k, x in v.items()))
            if isinstance(v, (list, tuple)):
                return tuple(_h(x) for x in v)
            return v
        return (self.link, _h(self.link_params), self.active, self.paused)

    def _is_dirty(self) -> bool:
        return self._resolved_params is None or self._compiled_sig != self._sig()

    def _compile(self):
        """Resolve+coerce this connector's link_params once. Only keys
        the active solver for `self.link` actually reads are given typed
        defaults here; unknown/extra keys pass through untouched exactly
        as they do today when solvers read link_params directly.

        A few keys are deliberately NOT resolved here even though their
        solver reads them: spring's `rest_length`, rope's `length`, and
        distance_lock's `length` all default to "whatever the current
        live distance between the shapes is" on ticks where the user
        hasn't set them explicitly (see each solver). That default is
        recomputed from live shape geometry every tick by design — caching
        it here would freeze the joint to a stale distance from whenever
        _compile() first ran instead of tracking a moving shape. Those
        three stay read from raw self.link_params inside their solvers.
        """
        p = self.link_params
        resolved: Dict[str, Any] = dict(p)
        link = self.link
        if link == "pin":
            resolved["anchor"] = p.get("anchor", "bottom")
            resolved["rotate_with"] = _as_bool(p.get("rotate_with", True))
        elif link == "pendulum":
            resolved["anchor"]  = p.get("anchor", "bottom")
            resolved["length"]  = float(p.get("length", 150.0))
            resolved["gravity"] = float(p.get("gravity", 980.0))
            resolved["damping"] = float(p.get("damping", 0.02))
        elif link == "spring":
            resolved["anchor"]    = p.get("anchor", "center")
            resolved["stiffness"] = float(p.get("stiffness", 200.0))
            resolved["damping"]   = float(p.get("damping", 0.05))
            resolved["mass"]      = float(p.get("mass", 1.0))
            # rest_length: intentionally not resolved — live-tracking default.
        elif link == "rope":
            resolved["anchor"]  = p.get("anchor", "bottom")
            resolved["gravity"] = float(p.get("gravity", 980.0))
            resolved["damping"] = float(p.get("damping", 0.03))
            # length: intentionally not resolved — live-tracking default.
        elif link in {"orbit", "benzene"}:
            resolved["anchor"] = p.get("anchor", "center")
            resolved["radius"] = float(p.get("radius", 100.0))
            resolved["speed"]  = float(p.get("speed", 90.0))
        elif link == "magnet":
            resolved["anchor"]    = p.get("anchor", "center")
            resolved["strength"]  = float(p.get("strength", 200.0))
            resolved["max_speed"] = float(p.get("max_speed", 300.0))
            resolved["damping"]   = float(p.get("damping", 0.08))
        elif link == "distance_lock":
            resolved["anchor"] = p.get("anchor", "center")
            # length: intentionally not resolved — live-tracking default.
        elif link == "sync":
            resolved["properties"] = p.get("properties", ["position"])
            resolved["offset_x"]   = float(p.get("offset_x", 0.0))
            resolved["offset_y"]   = float(p.get("offset_y", 0.0))
        self._resolved_params = resolved
        self._compiled_sig = self._sig()
        return resolved

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "from_ip":     self.from_ip,
            "sense_id":    self.sense.id if self.sense else None,
            "sense_active":self.sense.active if self.sense else None,
            "to_ip":       self.to_ip,
            "active":      self.active,
            "link":        self.link,
            "group":       self.group,
            "work":        self.work,
            "payload":     dict(self.payload),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Joint solvers (called on main thread via dispatcher)
# ══════════════════════════════════════════════════════════════════════════════

def _solve_pin(record: ConnectorRecord, window_tag: str) -> None:
    """Rigid attachment: pin to_ip to an anchor point on from_ip.

    This is the "join two shapes" joint — e.g. join a circle and a stick,
    and the stick stays attached to the circle's edge. By default
    (``rotate_with=True``) the join is rigid in rotation too: when from_ip
    spins, its anchor point sweeps around with it, and to_ip's own
    orientation turns to match — the way a stick glued to a spinning wheel
    would. Pass ``rotate_with=False`` to keep the old behaviour where the
    anchor is recomputed on the unrotated bounding box every frame.
    """
    from Draw._shapes import shapes as _sr
    if record._is_dirty():
        record._compile()
    p           = record._resolved_params
    anchor      = p["anchor"]
    rotate_with = p["rotate_with"]

    sa = _sr.get_by_ip(window_tag, record.from_ip)
    sb = _sr.get_by_ip(window_tag, record.to_ip)
    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if sa is None or sb is None or rect_a is None or rect_b is None:
        return

    bw, bh = rect_b[2], rect_b[3]

    if not rotate_with:
        ax, ay = _anchor_point(rect_a, anchor)
        _set_shape_pos(window_tag, record.to_ip, ax - bw / 2.0, ay - bh / 2.0)
        return

    rot_a = _current_rotation(sa)
    a_cx  = rect_a[0] + rect_a[2] / 2.0
    a_cy  = rect_a[1] + rect_a[3] / 2.0

    js = record._joint
    if js is not None and js.pin_local_dx is None:
        # Capture the anchor's offset from from_ip's centre in its
        # *unrotated* local frame, plus the relative rotation between the
        # two shapes — once, the first time this pin ticks (or right after
        # a drag+drop re-attach) — so the join stays fixed to the shape as
        # it spins instead of being recomputed fresh (and drifting) each
        # frame.
        raw_ax, raw_ay = _anchor_point(rect_a, anchor)
        local_dx, local_dy = raw_ax - a_cx, raw_ay - a_cy
        rad0 = math.radians(-rot_a)
        cos0, sin0 = math.cos(rad0), math.sin(rad0)
        js.pin_local_dx = local_dx * cos0 - local_dy * sin0
        js.pin_local_dy = local_dx * sin0 + local_dy * cos0
        js.pin_rot_offset = _current_rotation(sb) - rot_a

    if js is not None and js.pin_local_dx is not None:
        rad = math.radians(rot_a)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        off_x = js.pin_local_dx * cos_r - js.pin_local_dy * sin_r
        off_y = js.pin_local_dx * sin_r + js.pin_local_dy * cos_r
        ax, ay = a_cx + off_x, a_cy + off_y
    else:
        ax, ay = _anchor_point(rect_a, anchor)

    _set_shape_pos(window_tag, record.to_ip, ax - bw / 2.0, ay - bh / 2.0)

    if js is not None and js.pin_rot_offset is not None:
        _set_shape_rotation(window_tag, record.to_ip, rot_a + js.pin_rot_offset)


def _solve_pendulum(record: ConnectorRecord, window_tag: str, now: float) -> None:
    """Gravity pendulum physics."""
    if record._is_dirty():
        record._compile()
    p  = record._resolved_params
    js = record._joint
    dt = min(now - js.last_time, 0.05)
    js.last_time = now

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    anchor  = p["anchor"]
    ax, ay  = _anchor_point(rect_a, anchor)
    length  = p["length"]
    gravity = p["gravity"]
    damping = p["damping"]

    # θ̈ = -(g/L) sin(θ) - d·θ̇
    alpha       = -(gravity / length) * math.sin(js.angle) - damping * js.angle_vel
    js.angle_vel += alpha * dt
    js.angle    += js.angle_vel * dt

    bw, bh = rect_b[2], rect_b[3]
    bx = ax + length * math.sin(js.angle) - bw / 2.0
    by = ay + length * math.cos(js.angle) - bh / 2.0
    _set_shape_pos(window_tag, record.to_ip, bx, by)


def _solve_spring(record: ConnectorRecord, window_tag: str, now: float) -> None:
    """Spring joint between two shapes."""
    if record._is_dirty():
        record._compile()
    p  = record._resolved_params
    js = record._joint
    dt = min(now - js.last_time, 0.05)
    js.last_time = now

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    anchor     = p["anchor"]
    ax, ay     = _anchor_point(rect_a, anchor)
    bx_c = rect_b[0] + rect_b[2] / 2.0
    by_c = rect_b[1] + rect_b[3] / 2.0
    dx, dy     = bx_c - ax, by_c - ay
    dist       = math.hypot(dx, dy) or 1.0
    # rest_length's default tracks the live current distance — read raw,
    # never cached (see the note in ConnectorRecord._compile()).
    rest       = float(record.link_params.get("rest_length", dist))
    stiffness  = p["stiffness"]
    damping    = p["damping"]
    mass       = p["mass"]

    stretch    = dist - rest
    fx         = -(stiffness * stretch + damping * js.vx) * (dx / dist)
    fy         = -(stiffness * stretch + damping * js.vy) * (dy / dist)
    js.vx     += (fx / mass) * dt
    js.vy     += (fy / mass) * dt

    new_bx = bx_c + js.vx * dt - rect_b[2] / 2.0
    new_by = by_c + js.vy * dt - rect_b[3] / 2.0
    _set_shape_pos(window_tag, record.to_ip, new_bx, new_by)


def _solve_rope(record: ConnectorRecord, window_tag: str, now: float) -> None:
    """Rope: gravity + max-length constraint."""
    if record._is_dirty():
        record._compile()
    p  = record._resolved_params
    js = record._joint
    dt = min(now - js.last_time, 0.05)
    js.last_time = now

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    anchor  = p["anchor"]
    ax, ay  = _anchor_point(rect_a, anchor)
    gravity = p["gravity"]
    damping = p["damping"]
    bw, bh  = rect_b[2], rect_b[3]
    bx_c    = rect_b[0] + bw / 2.0
    by_c    = rect_b[1] + bh / 2.0
    # length's default tracks the live current distance — read raw,
    # never cached (see the note in ConnectorRecord._compile()).
    length  = float(record.link_params.get("length", math.hypot(bx_c - ax, by_c - ay) or 100.0))

    # gravity
    js.vy += gravity * dt
    js.vx *= (1.0 - damping)
    js.vy *= (1.0 - damping)

    new_cx = bx_c + js.vx * dt
    new_cy = by_c + js.vy * dt

    # taut constraint
    dist = math.hypot(new_cx - ax, new_cy - ay)
    if dist > length:
        ratio  = length / dist
        new_cx = ax + (new_cx - ax) * ratio
        new_cy = ay + (new_cy - ay) * ratio
        # project velocity
        nx, ny = (new_cx - ax) / length, (new_cy - ay) / length
        dot    = js.vx * nx + js.vy * ny
        js.vx -= dot * nx
        js.vy -= dot * ny

    _set_shape_pos(window_tag, record.to_ip, new_cx - bw / 2.0, new_cy - bh / 2.0)


def _solve_orbit(record: ConnectorRecord, window_tag: str, now: float) -> None:
    """Orbital motion around from_ip."""
    if record._is_dirty():
        record._compile()
    p  = record._resolved_params
    js = record._joint
    dt = min(now - js.last_time, 0.05)
    js.last_time = now

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    anchor  = p["anchor"]
    cx, cy  = _anchor_point(rect_a, anchor)
    radius  = p["radius"]
    speed   = p["speed"]   # deg/s
    bw, bh  = rect_b[2], rect_b[3]

    js.orbit_angle += math.radians(speed) * dt
    bx = cx + radius * math.cos(js.orbit_angle) - bw / 2.0
    by = cy + radius * math.sin(js.orbit_angle) - bh / 2.0
    _set_shape_pos(window_tag, record.to_ip, bx, by)


def _solve_benzene(record: ConnectorRecord, window_tag: str, now: float) -> None:
    """Hexagonal orbital motion around from_ip."""
    if record._is_dirty():
        record._compile()
    p  = record._resolved_params
    js = record._joint
    dt = min(now - js.last_time, 0.05)
    js.last_time = now

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    anchor  = p["anchor"]
    cx, cy  = _anchor_point(rect_a, anchor)
    base_radius = p["radius"]
    speed   = p["speed"]   # deg/s
    bw, bh  = rect_b[2], rect_b[3]

    js.orbit_angle += math.radians(speed) * dt
    
    # Regular hexagon in polar coordinates
    pi_over_3 = math.pi / 3.0
    pi_over_6 = math.pi / 6.0
    theta = (js.orbit_angle % pi_over_3) - pi_over_6
    
    # Guard against cos(theta) close to 0 (though theta is bound in [-pi/6, pi/6])
    cos_t = math.cos(theta)
    radius = base_radius * math.cos(pi_over_6) / (cos_t if abs(cos_t) > 1e-9 else 1e-9)
    
    bx = cx + radius * math.cos(js.orbit_angle) - bw / 2.0
    by = cy + radius * math.sin(js.orbit_angle) - bh / 2.0
    _set_shape_pos(window_tag, record.to_ip, bx, by)


def _solve_magnet(record: ConnectorRecord, window_tag: str, now: float) -> None:
    """Magnetic attraction: to_ip is pulled toward from_ip."""
    if record._is_dirty():
        record._compile()
    p  = record._resolved_params
    js = record._joint
    dt = min(now - js.last_time, 0.05)
    js.last_time = now

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    anchor    = p["anchor"]
    tx, ty    = _anchor_point(rect_a, anchor)
    bx_c      = rect_b[0] + rect_b[2] / 2.0
    by_c      = rect_b[1] + rect_b[3] / 2.0
    dx, dy    = tx - bx_c, ty - by_c
    dist      = math.hypot(dx, dy) or 1.0
    strength  = p["strength"]
    max_speed = p["max_speed"]
    damping   = p["damping"]

    force  = strength / max(dist, 1.0)
    js.vx += (dx / dist) * force * dt
    js.vy += (dy / dist) * force * dt
    js.vx *= (1.0 - damping)
    js.vy *= (1.0 - damping)
    speed  = math.hypot(js.vx, js.vy)
    if speed > max_speed:
        js.vx = js.vx / speed * max_speed
        js.vy = js.vy / speed * max_speed

    new_bx = bx_c + js.vx * dt - rect_b[2] / 2.0
    new_by = by_c + js.vy * dt - rect_b[3] / 2.0
    _set_shape_pos(window_tag, record.to_ip, new_bx, new_by)


def _solve_distance_lock(record: ConnectorRecord, window_tag: str) -> None:
    """Hard distance constraint: keeps shapes exactly ``length`` px apart."""
    if record._is_dirty():
        record._compile()
    p      = record._resolved_params
    anchor = p["anchor"]
    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    ax, ay  = _anchor_point(rect_a, anchor)
    bx_c    = rect_b[0] + rect_b[2] / 2.0
    by_c    = rect_b[1] + rect_b[3] / 2.0
    dx, dy  = bx_c - ax, by_c - ay
    dist    = math.hypot(dx, dy) or 1.0
    # length's default tracks the live current distance — read raw,
    # never cached (see the note in ConnectorRecord._compile()).
    length  = float(record.link_params.get("length", dist))

    ratio   = length / dist
    new_cx  = ax + dx * ratio
    new_cy  = ay + dy * ratio
    _set_shape_pos(window_tag, record.to_ip,
                   new_cx - rect_b[2] / 2.0, new_cy - rect_b[3] / 2.0)


def _solve_sync(record: ConnectorRecord, window_tag: str) -> None:
    """Copy properties from from_ip to to_ip."""
    from Draw._shapes import shapes as _sr
    if record._is_dirty():
        record._compile()
    p          = record._resolved_params
    properties = p["properties"]
    offset_x   = p["offset_x"]
    offset_y   = p["offset_y"]

    sa = _sr.get_by_ip(window_tag, record.from_ip)
    sb = _sr.get_by_ip(window_tag, record.to_ip)
    if sa is None or sb is None:
        return

    if "position" in properties or "x" in properties or "y" in properties:
        if sa.last_position:
            sx, sy = sa.last_position
            if "position" in properties or "x" in properties:
                sb.x = int(round(sx + offset_x))
            if "position" in properties or "y" in properties:
                sb.y = int(round(sy + offset_y))
            if sb.last_position:
                sb.last_position = (sb.x, sb.y)
    if "color" in properties:
        sb.color = sa.color
    if "opacity" in properties:
        sb.opacity = sa.opacity
    if "size" in properties and sa.last_size:
        sb.size = list(sa.last_size)
    if "rotation" in properties:
        sb.rotation = sa.rotation

    try:
        from Draw._window import window as _wr
        win    = _wr.get(window_tag)
        canvas = getattr(win, "_draw_canvas", None)
        if canvas:
            canvas.update()
    except Exception:
        pass


def _reinit_joint_after_drag(record: ConnectorRecord, window_tag: str) -> None:
    """Re-baseline a joint's physics/anchor state right after the user
    manually grabs to_ip and lets go, so the joint continues naturally from
    the dropped position instead of snapping back or exploding on the next
    tick (a large stale ``dt`` combined with old velocity/anchor state).
    """
    link = record.link
    p    = record.link_params
    js   = record._joint

    rect_a = _get_shape_rect(window_tag, record.from_ip)
    rect_b = _get_shape_rect(window_tag, record.to_ip)
    if rect_a is None or rect_b is None:
        return

    b_cx = rect_b[0] + rect_b[2] / 2.0
    b_cy = rect_b[1] + rect_b[3] / 2.0

    if js is not None:
        js.vx = 0.0
        js.vy = 0.0
        js.last_time = time.perf_counter()

    if link == "pin":
        # Forget the captured local offset/rotation — _solve_pin will
        # recapture it next tick from wherever the shape was just dropped,
        # effectively re-attaching the join at the new spot.
        if js is not None:
            js.pin_local_dx   = None
            js.pin_local_dy   = None
            js.pin_rot_offset = None
        return

    anchor = p.get("anchor", "center")
    ax, ay = _anchor_point(rect_a, anchor)
    dx, dy = b_cx - ax, b_cy - ay
    dist   = math.hypot(dx, dy) or 1.0

    if link == "pendulum" and js is not None:
        js.angle     = math.atan2(dx, dy)
        js.angle_vel = 0.0
        p["length"]  = dist
    elif link in {"orbit", "benzene"} and js is not None:
        js.orbit_angle = math.atan2(dy, dx)
        p["radius"]    = dist
    elif link == "distance_lock":
        p["length"] = dist
    # spring / rope / magnet: rest_length / length / strength stay as
    # configured — the shape will spring/hang/pull back toward them from
    # wherever it was dropped, which is the whole point of grabbing them.


# ── joint dispatch table ──────────────────────────────────────────────────────

_JOINT_SOLVERS: Dict[str, Callable] = {
    "pin":           _solve_pin,
    "pendulum":      _solve_pendulum,
    "spring":        _solve_spring,
    "rope":          _solve_rope,
    "orbit":         _solve_orbit,
    "benzene":       _solve_benzene,
    "magnet":        _solve_magnet,
    "distance_lock": _solve_distance_lock,
    "sync":          _solve_sync,
}

_STATEFUL_JOINTS: Set[str] = {"pendulum", "spring", "rope", "orbit", "benzene", "magnet"}


# ══════════════════════════════════════════════════════════════════════════════
#  _ConnectorRegistry
# ══════════════════════════════════════════════════════════════════════════════

class _ConnectorRegistry:
    """Singleton exposed as Draw.connectors."""

    def __init__(self, senses_registry: _SenseRegistry) -> None:
        self._items: Dict[str, ConnectorRecord] = {}
        self._counter: int                      = 0
        self._senses                            = senses_registry
        self._lock                              = threading.Lock()
        self._ticker_running                    = False
        self._ticker_thread: Optional[threading.Thread] = None
        self._gui_dispatcher: Optional[_GUIDispatcher]  = None
        self._window_tag: Optional[str]         = None
        self._locked_ips: Set[str]              = set()
        self._unlocked_ips: Set[str]            = set()

    def lock(self, ip: Optional[str]) -> None:
        """Lock shape IP so it cannot be dragged on canvas."""
        if ip is not None:
            self._unlocked_ips.discard(str(ip))
            self._locked_ips.add(str(ip))

    def unlock(self, ip: Optional[str]) -> None:
        """Unlock shape IP so it can be dragged on canvas."""
        if ip is not None:
            self._locked_ips.discard(str(ip))
            self._unlocked_ips.add(str(ip))

    def is_locked(self, ip: Optional[str]) -> bool:
        """Return True if shape IP is locked. Defaults to True (cannot move unless unlocked)."""
        if ip is None:
            return True
        if str(ip) in self._unlocked_ips:
            return False
        return True

    # ── public call ───────────────────────────────────────────────────────────

    def __call__(
        self,
        spec: object = None,
        payload: Optional[Dict[str, Any]] = None,
        *,
        id: Optional[object]           = None,
        ip: Optional[object]           = None,   # canonical ip kwarg (maps to from_ip)
        from_ip: Optional[object]      = None,
        connector_ip: Optional[object] = None,   # legacy alias for from_ip
        sense: Optional[object]        = None,
        to_ip: Optional[object]        = None,
        connector_get_ip: Optional[object] = None,  # legacy alias for to_ip
        work: Optional[Callable]       = None,
        active: Optional[object]       = None,
        return_value: object           = None,
        # ── shape link / joint ──────────────────────────────────────────────
        link: Optional[str]            = None,
        anchor: Optional[str]          = None,
        length: Optional[float]        = None,
        gravity: Optional[float]       = None,
        damping: Optional[float]       = None,
        stiffness: Optional[float]     = None,
        mass: Optional[float]          = None,
        rest_length: Optional[float]   = None,
        radius: Optional[float]        = None,
        speed: Optional[float]         = None,
        angle: Optional[float]         = None,
        strength: Optional[float]      = None,
        max_speed: Optional[float]     = None,
        properties: Optional[List[str]] = None,
        offset_x: Optional[float]      = None,
        offset_y: Optional[float]      = None,
        rotate_with: Optional[object]  = None,
        display: Optional[str]         = None,
        group: Optional[str]           = None,
        draggable: Optional[object]    = None,
        grab: Optional[object]         = None,
        locked: Optional[object]       = None,
    ) -> ConnectorRecord:

        # ── resolve: connector_ip / ip → from_ip ────────────────────────────
        connector_ip = connector_ip if connector_ip is not None else ip
        if connector_ip is not None and from_ip is None:
            from_ip = connector_ip
        if connector_get_ip is not None and to_ip is None:
            to_ip = connector_get_ip

        if locked is not None:
            target_lock = str(from_ip or to_ip or ip) if (from_ip or to_ip or ip) else None
            if target_lock:
                if _as_bool(locked):
                    self.lock(target_lock)
                else:
                    self.unlock(target_lock)

        if draggable is not None or grab is not None:
            drag_val = draggable if draggable is not None else grab
            target_drag = str(from_ip or to_ip or ip) if (from_ip or to_ip or ip) else None
            if target_drag:
                if _as_bool(drag_val):
                    self.unlock(target_drag)
                else:
                    self.lock(target_drag)

        if spec is not None:
            pf, ps, pt, pw, sp = self._parse_spec(spec)
            from_ip = from_ip or pf
            sense   = sense   or ps
            to_ip   = to_ip   or pt
            work    = work    or pw
            merged_payload = dict(sp)
            if payload: merged_payload.update(payload)
        else:
            merged_payload = dict(payload or {})

        if from_ip is None and to_ip is None and sense is None and link is None:
            raise ValueError(
                "Draw.connectors: provide 'spec' list/tuple/dict OR "
                "keyword arguments (from_ip, sense, to_ip, link…)."
            )

        connector_id = str(id) if id is not None else f"connector_{self._counter}"
        self._counter += 1

        resolved_sense = self._resolve_sense(sense)

        if "active" in merged_payload:
            is_active = _as_bool(merged_payload.pop("active"))
        elif active is not None:
            is_active = _as_bool(active)
        else:
            is_active = True

        if return_value is not None:
            merged_payload["return"] = return_value

        from_ref = (_motion_registry.parse_target_ref(from_ip)
                    if from_ip else TargetRef("shape", None))
        to_ref   = (_motion_registry.parse_target_ref(to_ip)
                    if to_ip else TargetRef("shape", None))

        # ── build link_params from convenience kwargs ─────────────────────
        link_params: Dict[str, Any] = {}
        if anchor      is not None: link_params["anchor"]      = anchor
        if length      is not None: link_params["length"]      = length
        if gravity     is not None: link_params["gravity"]     = gravity
        if damping     is not None: link_params["damping"]     = damping
        if stiffness   is not None: link_params["stiffness"]   = stiffness
        if mass        is not None: link_params["mass"]        = mass
        if rest_length is not None: link_params["rest_length"] = rest_length
        if radius      is not None: link_params["radius"]      = radius
        if speed       is not None: link_params["speed"]       = speed
        if angle       is not None: link_params["angle"]       = angle
        if strength    is not None: link_params["strength"]    = strength
        if max_speed   is not None: link_params["max_speed"]   = max_speed
        if properties  is not None: link_params["properties"]  = properties
        if offset_x    is not None: link_params["offset_x"]    = offset_x
        if offset_y    is not None: link_params["offset_y"]    = offset_y
        if rotate_with is not None: link_params["rotate_with"] = _as_bool(rotate_with)

        # ── drag / grab option ─────────────────────────────────────────────
        # Lets the person grab to_ip with the mouse (it already has full
        # drag support on the canvas) without the joint solver fighting
        # them every frame; the joint re-baselines itself once they let go.
        draggable_raw = draggable if draggable is not None else grab
        is_draggable  = _as_bool(draggable_raw) if draggable_raw is not None else False

        # resolve window tag for joint solvers
        if display is not None:
            self._window_tag = display

        # normalise link name
        link_norm = str(link).lower().strip() if link else None

        record = ConnectorRecord(
            id=connector_id,
            from_ip=from_ref.ip, sense=resolved_sense,
            to_ip=to_ref.ip,     active=is_active,
            work=work,           payload=merged_payload,
            from_ref=from_ref,   to_ref=to_ref,
            link=link_norm,      link_params=link_params,
            group=group,
            draggable=is_draggable,
        )

        # initialise physics state
        if link_norm in _STATEFUL_JOINTS or link_norm == "pin":
            js = _JointState()
            if link_norm == "pendulum":
                init_angle = float(link_params.get("angle", 45.0))
                js.angle = math.radians(init_angle)
            elif link_norm in {"orbit", "benzene"}:
                init_angle = float(link_params.get("angle", 0.0))
                js.orbit_angle = math.radians(init_angle)
            record._joint = js

        with self._lock:
            self._items[connector_id] = record

        self._ensure_ticker()
        return record

    # ── group control ─────────────────────────────────────────────────────────

    def pause_group(self, group: str) -> None:
        """Pause all connectors in a group (physics + callbacks stop)."""
        with self._lock:
            for r in self._items.values():
                if r.group == group:
                    r.paused = True

    def resume_group(self, group: str) -> None:
        """Resume a paused connector group."""
        with self._lock:
            for r in self._items.values():
                if r.group == group:
                    r.paused = False
                    # reset physics clocks so dt doesn't explode
                    if r._joint:
                        r._joint.last_time = time.perf_counter()

    def set_active(self, id: object, active: bool) -> None:
        """Enable or disable one connector by id."""
        rec = self._items.get(str(id))
        if rec:
            rec.active = active

    # ── query helpers ─────────────────────────────────────────────────────────

    def get(self, id: object) -> Optional[ConnectorRecord]:
        return self._items.get(str(id))

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._items.values()]

    def clear(self, id: Optional[object] = None) -> None:
        with self._lock:
            if id is None:
                self._items.clear(); self._counter = 0
            else:
                self._items.pop(str(id), None)

    def cleanup_window(self, tag: str, ips: Optional[set] = None) -> int:
        """Remove connector records that belonged to a now-closed window.

        ConnectorRecord doesn't track which window it was created for (the
        registry only ever resolves a single shared window_tag for the
        ticker — see _resolve_window_tag), so there's nothing on the record
        itself to filter by. Instead, records are matched by ip membership
        against the closed window's shape/text ips, which is a reliable
        proxy: a from_ip/to_ip whose shape no longer exists anywhere is
        exactly the case where a joint should stop being solved.

        Also clears self._window_tag if it pointed at the closed window, so
        the ticker re-resolves (to the one remaining open window, or to
        None) instead of staying pinned to a tag with no canvas behind it —
        without this, connectors registered for a *new* window after the
        old one closes could silently never tick.

        Called automatically from Draw.window's close handler. Returns the
        number of connector records removed.
        """
        removed = 0
        with self._lock:
            if self._window_tag == tag:
                self._window_tag = None
            if ips:
                for cid, record in list(self._items.items()):
                    if (record.from_ip and record.from_ip in ips) or (
                        record.to_ip and record.to_ip in ips
                    ):
                        del self._items[cid]
                        removed += 1
        return removed

    # ── tick engine ───────────────────────────────────────────────────────────

    def _ensure_ticker(self) -> None:
        if self._gui_dispatcher is None:
            self._gui_dispatcher = _GUIDispatcher()
        if not self._ticker_running:
            self._ticker_running = True
            t = threading.Thread(target=self._tick_loop, daemon=True)
            t.start()
            self._ticker_thread = t

    def _resolve_window_tag(self) -> Optional[str]:
        if self._window_tag:
            return self._window_tag
        try:
            from Draw._window import window as _wr
            tags = _wr.list_tags()
            if len(tags) == 1:
                self._window_tag = tags[0]
                return self._window_tag
        except Exception:
            pass
        return None

    def _tick_loop(self) -> None:
        """Background thread: evaluate senses, dispatch joints and callbacks."""
        while self._ticker_running:
            try:
                now = time.perf_counter()
                # NOTE: tick_custom() is intentionally NOT called here. It is
                # already driven every ~16ms by the GUI-thread QTimer in
                # _text.py's _DrawCanvas._tick_animation. Calling it from both
                # the daemon ticker and the GUI timer caused an unsynchronized
                # cross-thread race on CustomMotionRecord.started_at /
                # current_value. See Make_it_Stable.md Component 1.
                self._senses.evaluate_callable_senses()

                window_tag = self._resolve_window_tag()
                if window_tag:
                    self._senses.evaluate_proximity_senses(window_tag)

                with self._lock:
                    connectors = list(self._items.values())

                for record in connectors:
                    if not record.active or record.paused:
                        continue

                    # ── joint / link tick (runs every frame) ─────────────
                    if record.link and record.from_ip and record.to_ip and window_tag:
                        dispatcher = self._gui_dispatcher
                        if dispatcher:
                            link = record.link
                            if link == "pin":
                                dispatcher.post(_solve_pin,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "pendulum":
                                dispatcher.post(_solve_pendulum,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "spring":
                                dispatcher.post(_solve_spring,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "rope":
                                dispatcher.post(_solve_rope,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "orbit":
                                dispatcher.post(_solve_orbit,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "benzene":
                                dispatcher.post(_solve_benzene,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "magnet":
                                dispatcher.post(_solve_magnet,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "distance_lock":
                                dispatcher.post(_solve_distance_lock,
                                                _LinkCallArgs(record, window_tag, now))
                            elif link == "sync":
                                dispatcher.post(_solve_sync,
                                                _LinkCallArgs(record, window_tag, now))

                    # ── sense + work callback ────────────────────────────
                    sense = record.sense
                    if sense is None:
                        continue
                    if sense.consume() and record.work is not None:
                        dispatcher = self._gui_dispatcher
                        if dispatcher:
                            dispatcher.post(record.work, record)
                        else:
                            try:
                                record.work(record)
                            except Exception as exc:
                                _logger.warning("Draw.connectors: callback error: %s", exc)

            except Exception as exc:
                _logger.warning("Draw.connectors: tick error: %s", exc)

            time.sleep(0.016)   # ~60 fps

    def stop_ticker(self) -> None:
        self._ticker_running = False
        self._gui_dispatcher = None

    # ── spec parser ───────────────────────────────────────────────────────────

    def _parse_spec(self, spec: object):
        if isinstance(spec, dict):
            return (
                spec.get("from_ip", spec.get("get_ip")),
                spec.get("sense"),
                spec.get("to_ip", spec.get("target_ip")),
                spec.get("work"),
                {k: v for k, v in spec.items()
                 if k not in {"from_ip","get_ip","sense","to_ip","target_ip","work"}},
            )
        if isinstance(spec, (list, tuple)):
            if len(spec) < 3:
                raise ValueError(
                    "Draw.connectors: list spec needs [from_ip, sense, to_ip, ...]."
                )
            extra: Dict[str, Any] = {}
            if len(spec) > 3 and isinstance(spec[3], dict):
                extra = dict(spec[3])
            work_val = spec[3] if len(spec) > 3 and callable(spec[3]) else None
            if work_val is None and len(spec) > 4 and callable(spec[4]):
                work_val = spec[4]
            return spec[0], spec[1], spec[2], work_val, extra
        raise TypeError("Draw.connectors: 'spec' must be a dict, list, or tuple.")

    def _resolve_sense(self, raw: object) -> Optional[SenseRecord]:
        if raw is None:                    return None
        if isinstance(raw, SenseRecord):   return raw
        if isinstance(raw, str):           return self._senses.get(raw)
        if isinstance(raw, dict) and "id" in raw:
            return self._senses.get(raw["id"])
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  _LinkCallArgs — thin wrapper so _GUIDispatcher.post can unpack correctly
# ══════════════════════════════════════════════════════════════════════════════

class _LinkCallArgs:
    """
    Used as the second argument to dispatcher.post() for joint solvers.
    The dispatcher calls  fn(record)  — but joint solvers need
    (record, window_tag, now).  We subclass this into each solver's signature
    by making the object callable so that fn(this) unpacks itself.
    """
    __slots__ = ("record", "window_tag", "now")

    def __init__(self, record: ConnectorRecord, window_tag: str, now: float):
        self.record     = record
        self.window_tag = window_tag
        self.now        = now


# Patch joint solvers so they accept a _LinkCallArgs as the second argument
# (the dispatcher calls fn(record) → but record here is a _LinkCallArgs)

def _wrap_joint(solver: Callable, needs_time: bool) -> Callable:
    if needs_time:
        def _fn(args: _LinkCallArgs) -> None:
            solver(args.record, args.window_tag, args.now)
    else:
        def _fn(args: _LinkCallArgs) -> None:
            solver(args.record, args.window_tag)
    return _fn


_pin_dispatch           = _wrap_joint(_solve_pin,           needs_time=False)
_pendulum_dispatch      = _wrap_joint(_solve_pendulum,      needs_time=True)
_spring_dispatch        = _wrap_joint(_solve_spring,        needs_time=True)
_rope_dispatch          = _wrap_joint(_solve_rope,          needs_time=True)
_orbit_dispatch         = _wrap_joint(_solve_orbit,         needs_time=True)
_benzene_dispatch       = _wrap_joint(_solve_benzene,       needs_time=True)
_magnet_dispatch        = _wrap_joint(_solve_magnet,        needs_time=True)
_distance_lock_dispatch = _wrap_joint(_solve_distance_lock, needs_time=False)
_sync_dispatch          = _wrap_joint(_solve_sync,          needs_time=False)
_reinit_drag_dispatch   = _wrap_joint(_reinit_joint_after_drag, needs_time=False)


# Fix the tick loop to use wrapped dispatchers
def _patched_tick_loop(self: _ConnectorRegistry) -> None:
    _DISPATCH_MAP = {
        "pin":           _pin_dispatch,
        "pendulum":      _pendulum_dispatch,
        "spring":        _spring_dispatch,
        "rope":          _rope_dispatch,
        "orbit":         _orbit_dispatch,
        "benzene":       _benzene_dispatch,
        "magnet":        _magnet_dispatch,
        "distance_lock": _distance_lock_dispatch,
        "sync":          _sync_dispatch,
    }
    while self._ticker_running:
        try:
            now = time.perf_counter()
            self._senses.evaluate_callable_senses()
            window_tag = self._resolve_window_tag()
            if window_tag:
                self._senses.evaluate_proximity_senses(window_tag)

            with self._lock:
                connectors = list(self._items.values())

            for record in connectors:
                if not record.active or record.paused:
                    continue
                if record.link and record.from_ip and record.to_ip and window_tag:
                    # ── grab / drag-and-drop ──────────────────────────────
                    # While the person is manually holding to_ip (it already
                    # supports mouse drag on the canvas), let go of it —
                    # don't fight the mouse by writing a joint-computed
                    # position over it every frame.
                    being_dragged = (
                        record.draggable
                        and _is_shape_dragged(window_tag, record.to_ip)
                    )
                    if being_dragged:
                        record._was_dragging = True
                    else:
                        if record._was_dragging and self._gui_dispatcher:
                            # Just let go — re-baseline the joint around
                            # wherever the shape was dropped before resuming
                            # normal control this same tick.
                            self._gui_dispatcher.post(
                                _reinit_drag_dispatch,
                                _LinkCallArgs(record, window_tag, now),
                            )
                            record._was_dragging = False

                        dispatch_fn = _DISPATCH_MAP.get(record.link)
                        if dispatch_fn and self._gui_dispatcher:
                            self._gui_dispatcher.post(
                                dispatch_fn, _LinkCallArgs(record, window_tag, now)
                            )
                sense = record.sense
                if sense is None:
                    continue
                if sense.consume() and record.work is not None:
                    if self._gui_dispatcher:
                        self._gui_dispatcher.post(record.work, record)
                    else:
                        try:
                            record.work(record)
                        except Exception as exc:
                            _logger.warning("Draw.connectors: callback error: %s", exc)
        except Exception as exc:
            _logger.warning("Draw.connectors: tick error: %s", exc)
        time.sleep(0.016)


# Replace the method with the clean dispatch-map version
_ConnectorRegistry._tick_loop = _patched_tick_loop


# ── Key name helper ────────────────────────────────────────────────────────────

def _qt_key_to_name(qt_key: int) -> str:
    """Convert a Qt.Key integer to a human-readable name like 'Return', 'A'."""
    from PySide6.QtCore import Qt
    try:
        name = Qt.Key(qt_key).name
        return name.replace("Key_", "") if name.startswith("Key_") else name
    except ValueError:
        return str(qt_key)


# ══════════════════════════════════════════════════════════════════════════════
#  Singletons
# ══════════════════════════════════════════════════════════════════════════════

senses     = _SenseRegistry()
connectors = _ConnectorRegistry(senses)
