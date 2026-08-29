"""
Draw._simulate — programmatic input simulation, for tests and automation.

Without this, driving the canvas from code means reaching past the public
API into Draw._connectors.handle_canvas_mouse_* and hand-building a raw
QMouseEvent (the pattern already used in test_drag.py). This module wraps
that behind a small public surface: Draw.simulate.click(...) / .drag(...).

Not meant for real end-user interaction — it's for automated tests, demos,
and scripted walkthroughs.
"""

from __future__ import annotations

from typing import Optional

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QEvent, QPointF, Qt
# pyrefly: ignore [missing-import]
from PySide6.QtGui import QMouseEvent

from Draw._window import window as _window_registry

_BUTTON_MAP = {
    "left":   Qt.MouseButton.LeftButton,
    "right":  Qt.MouseButton.RightButton,
    "middle": Qt.MouseButton.MiddleButton,
}


def _resolve_canvas(display: Optional[str]):
    tags = _window_registry.list_tags()
    if not tags:
        raise ValueError("Draw.simulate: no active window found.")
    tag = display or (tags[0] if len(tags) == 1 else None)
    if tag is None:
        raise ValueError(
            "Draw.simulate: 'display' is required when multiple windows exist."
        )
    return _window_registry.get_canvas(tag)


def _shape_center(canvas, ip: str) -> QPointF:
    for s in canvas.shape_items:
        if s.ip == ip:
            if s.last_position and s.last_size:
                x, y = s.last_position
                w, h = s.last_size
                return QPointF(x + w / 2.0, y + h / 2.0)
            break
    raise ValueError(f"Draw.simulate: shape ip={ip!r} not found or not yet painted.")


def _mouse_event(event_type, pos: QPointF, button) -> QMouseEvent:
    return QMouseEvent(event_type, pos, pos, button, button, Qt.KeyboardModifier.NoModifier)


def click(
    ip: str,
    *,
    display: Optional[str] = None,
    button: str = "left",
    double: bool = False,
    x: Optional[float] = None,
    y: Optional[float] = None,
) -> None:
    """
    Simulate a mouse click (or double-click) on shape ``ip``.

    button : "left" | "right" | "middle"
    double : fire a double-click instead of a single click.
    x / y  : override the click position (canvas coords); defaults to the
             shape's current center.
    """
    from Draw._connectors import (
        handle_canvas_mouse_press,
        handle_canvas_mouse_release,
        handle_canvas_mouse_double_click,
    )

    canvas = _resolve_canvas(display)
    pos = QPointF(x, y) if x is not None and y is not None else _shape_center(canvas, ip)
    qt_button = _BUTTON_MAP.get(button, Qt.MouseButton.LeftButton)

    if double:
        ev = _mouse_event(QEvent.Type.MouseButtonDblClick, pos, qt_button)
        handle_canvas_mouse_double_click(canvas, ev)
    else:
        press_ev = _mouse_event(QEvent.Type.MouseButtonPress, pos, qt_button)
        release_ev = _mouse_event(QEvent.Type.MouseButtonRelease, pos, qt_button)
        handle_canvas_mouse_press(canvas, press_ev)
        handle_canvas_mouse_release(canvas, release_ev)


def drag(ip: str, to_x: float, to_y: float, *, display: Optional[str] = None) -> None:
    """Simulate picking up shape ``ip`` and dragging it to (to_x, to_y)."""
    from Draw._connectors import (
        handle_canvas_mouse_press,
        handle_canvas_mouse_move,
        handle_canvas_mouse_release,
    )

    canvas = _resolve_canvas(display)
    start_pos = _shape_center(canvas, ip)
    end_pos = QPointF(to_x, to_y)

    press_ev = _mouse_event(QEvent.Type.MouseButtonPress, start_pos, Qt.MouseButton.LeftButton)
    handle_canvas_mouse_press(canvas, press_ev)

    move_ev = _mouse_event(QEvent.Type.MouseMove, end_pos, Qt.MouseButton.LeftButton)
    handle_canvas_mouse_move(canvas, move_ev)
    # The canvas only recomputes last_position during an actual paint pass
    # (final_x/final_y are overridden from _drag_x/_drag_y there). Force one
    # synchronously so the caller sees the moved position right away instead
    # of waiting for Qt's next scheduled repaint.
    canvas.grab()

    release_ev = _mouse_event(QEvent.Type.MouseButtonRelease, end_pos, Qt.MouseButton.LeftButton)
    handle_canvas_mouse_release(canvas, release_ev)
    canvas.grab()


class _Simulate:
    """Singleton exposed as Draw.simulate."""
    click = staticmethod(click)
    drag = staticmethod(drag)


simulate = _Simulate()
