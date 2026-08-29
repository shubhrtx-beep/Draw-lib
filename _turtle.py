"""
Draw._turtle
============
Bridge Python's stdlib ``turtle`` module into a PySide6 Draw window.

Draw.turtle() / Draw.pen() creates a genuine (hidden, never mapped to
screen) tkinter.Canvas plus a real turtle.TurtleScreen / turtle.RawTurtle
bound to it, and returns that RawTurtle directly — every method on it
(forward, left, circle, dot, stamp, begin_fill/end_fill, write, shape,
speed, ...) is 100% authentic stdlib turtle.

Only two event-loop integration points are patched:

  Canvas.update()   Mirrors the canvas's current item list onto a
                     transparent Qt overlay and pumps Qt's event loop.

  Canvas.after(...)  Delay-only form does a non-blocking paced wait;
                     delay+callback form is rescheduled via QTimer.

Known limitations (v1): mouse/keyboard event forwarding (onclick/onkey)
is not wired up. Custom raster shapes via screen.register_shape() /
screen.bgpic() are not mirrored.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QCoreApplication, QPointF, Qt, QTimer
# pyrefly: ignore [missing-import]
from PySide6.QtGui import (
    QBrush, QColor, QPainter, QPainterPath, QPen,
)
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QWidget, QMainWindow

from Draw._app import get_app
from Draw._window import window as _window_registry
from Draw._text import _get_or_create_canvas


def _parse_color_safe(raw: object) -> QColor:
    if raw is None or str(raw).strip() in {"", " "}:
        return QColor("white")
    from Draw import _bridge
    return _bridge.get_color_parser()(raw)


class _TurtleCanvasWidget(QWidget):
    """
    Mirrors a hidden tkinter.Canvas's current item list onto Qt via
    QPainter. One instance per Draw.turtle() call — layered over the
    window's shared Draw canvas, same as _PointCanvas.
    """

    def __init__(self, parent: QMainWindow, tk_canvas, width: int, height: int) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setFixedSize(int(width), int(height))
        self._tk = tk_canvas
        self._half_w = width / 2.0
        self._half_h = height / 2.0
        self._items: List[Tuple[str, List[Tuple[float, float]], Dict[str, Any]]] = []

    def _sync_from_tk(self) -> None:
        """Read the tk canvas's current item list and translate turtle's
        centre-origin canvas-space coordinates into this widget's
        top-left-origin pixel space."""
        cv = self._tk
        items: List[Tuple[str, List[Tuple[float, float]], Dict[str, Any]]] = []
        try:
            for item_id in cv.find_all():
                kind = cv.type(item_id)
                raw = cv.coords(item_id)
                pts = [
                    (raw[i] + self._half_w, raw[i + 1] + self._half_h)
                    for i in range(0, len(raw) - 1, 2)
                ]
                style: Dict[str, Any] = {}
                if kind in ("line", "polygon"):
                    style["fill"] = cv.itemcget(item_id, "fill") or None
                    if kind == "polygon":
                        style["outline"] = cv.itemcget(item_id, "outline") or None
                    try:
                        style["width"] = float(cv.itemcget(item_id, "width") or 1)
                    except (TypeError, ValueError):
                        style["width"] = 1.0
                elif kind == "text":
                    style["fill"] = cv.itemcget(item_id, "fill") or "black"
                    style["text"] = cv.itemcget(item_id, "text")
                items.append((kind, pts, style))
        except Exception:
            pass
        self._items = items

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for kind, pts, style in self._items:
                if kind == "line" and len(pts) >= 2:
                    fill = style.get("fill")
                    if not fill:
                        continue
                    pen = QPen(QColor(fill), max(1.0, float(style.get("width", 1.0))))
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                    painter.setPen(pen)
                    path = QPainterPath()
                    path.moveTo(QPointF(*pts[0]))
                    for p in pts[1:]:
                        path.lineTo(QPointF(*p))
                    painter.drawPath(path)

                elif kind == "polygon" and len(pts) >= 3:
                    fill = style.get("fill")
                    outline = style.get("outline")
                    path = QPainterPath()
                    path.moveTo(QPointF(*pts[0]))
                    for p in pts[1:]:
                        path.lineTo(QPointF(*p))
                    path.closeSubpath()
                    painter.setBrush(QBrush(QColor(fill)) if fill else Qt.BrushStyle.NoBrush)
                    if outline:
                        painter.setPen(QPen(QColor(outline), max(1.0, float(style.get("width", 1.0)))))
                    else:
                        painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPath(path)

                elif kind == "text" and pts:
                    text = style.get("text") or ""
                    if not text:
                        continue
                    painter.setPen(QPen(QColor(style.get("fill", "black"))))
                    painter.drawText(QPointF(*pts[0]), text)
                # "image" items intentionally unsupported.
        finally:
            if painter.isActive():
                painter.end()


class _PenRegistry:
    """
    Singleton exposed as Draw.pen / Draw.turtle.

    Returns a genuine turtle.RawTurtle bound to a hidden tkinter.Canvas
    that's mirrored live onto a Qt overlay in the Draw window.

        t = Draw.pen(display="main", width=400, height=400, bg="white")
        t.speed(6)
        t.forward(100)
        t.left(90)
        t.circle(50)

    Parameters
    ----------
    tag / display : window to draw on (required if >1 window exists)
    width / height : pixel size of the turtle canvas (default: window size)
    x / y         : top-left placement within the window (default: 0, 0)
    bg            : background colour (default "white")
    mode          : "standard" | "logo" | "world"
    speed         : 0-10, forwarded to RawTurtle.speed()
    shape         : initial cursor shape name
    ip            : optional id to look this turtle screen up again later
    """

    def __init__(self):
        self._tk_root = None
        self._screens: Dict[str, Any] = {}

    def _get_root(self):
        if self._tk_root is None:
            import tkinter as _tk
            root = _tk.Tk()
            root.withdraw()
            self._tk_root = root
        return self._tk_root

    def __call__(
        self,
        *,
        tag: Optional[str] = None,
        display: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        x: int = 0,
        y: int = 0,
        bg: object = "white",
        mode: str = "standard",
        speed: object = None,
        shape: Optional[str] = None,
        ip: Optional[str] = None,
    ):
        get_app()

        window_tag = display if display is not None else tag
        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise ValueError(
                    "Draw.pen: multiple windows exist; 'tag' or 'display' is required."
                )
            else:
                raise ValueError("Draw.pen: no windows exist to draw on.")

        win: QMainWindow = _window_registry.get(window_tag)
        _get_or_create_canvas(window_tag, win)

        w = int(width) if width is not None else max(1, win.width())
        h = int(height) if height is not None else max(1, win.height())
        bg_str = bg if isinstance(bg, str) else _parse_color_safe(bg).name()

        import tkinter as _tk
        import turtle as _turtle_mod

        root = self._get_root()
        qt_widget_holder: Dict[str, Any] = {}

        class _BridgeCanvas(_tk.Canvas):
            """A real tkinter.Canvas with update()/after()/after_idle()
            patched for Qt event-loop integration."""

            def update(self_cv):  # noqa: N805
                try:
                    _tk.Canvas.update(self_cv)
                except Exception:
                    pass
                qt = qt_widget_holder.get("w")
                if qt is not None:
                    qt._sync_from_tk()
                    qt.repaint()
                QCoreApplication.processEvents()

            def after(self_cv, ms, func=None, *args):  # noqa: N805
                if func is None:
                    import time as _time
                    deadline = _time.monotonic() + ms / 1000.0
                    while _time.monotonic() < deadline:
                        QCoreApplication.processEvents()
                        _time.sleep(0.001)
                    return None
                QTimer.singleShot(int(ms), lambda: func(*args))
                return "bridged-after"

            def after_idle(self_cv, func, *args):  # noqa: N805
                QTimer.singleShot(0, lambda: func(*args))
                return "bridged-after"

        cv = _BridgeCanvas(root, width=w, height=h, bg=bg_str, highlightthickness=0)

        qt_widget = _TurtleCanvasWidget(win, cv, w, h)
        qt_widget.move(int(x), int(y))
        qt_widget.raise_()
        qt_widget.show()
        qt_widget_holder["w"] = qt_widget

        screen = _turtle_mod.TurtleScreen(cv, mode=mode)
        rt = _turtle_mod.RawTurtle(screen)
        if speed is not None:
            rt.speed(speed)
        if shape is not None:
            rt.shape(shape)

        rt._draw_qt_widget = qt_widget
        rt._draw_tk_canvas = cv
        rt._draw_screen = screen

        key = ip if ip is not None else f"{window_tag}:{len(self._screens)}"
        self._screens[key] = (cv, qt_widget, screen, rt)

        return rt


turtle = _PenRegistry()
