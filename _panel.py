"""
Draw._panel  v1
===============
Floating mini-windows embedded inside an existing Draw window canvas.
Each panel gets its own draw surface so Draw.shape(display="panel-ip")
works identically to drawing on a full window.

PUBLIC API
----------
    Draw.panel(
        ip               = "settings",      # unique id  (REQUIRED)
        display          = "main",          # parent window tag
        title            = "Settings",
        width            = 300,
        height           = 400,
        x                = 50,
        y                = 50,
        align            = None,            # center | top-left | ... (same as Draw.window)
        background_color = "#1e1e2e",
        title_color      = "#cdd6f4",
        title_background = "#313244",
        border_color     = "#45475a",
        border_width     = 1,
        border_radius    = 10,
        transparency     = 100,             # 0-100
        draggable        = True,
        resizable        = False,
        closable         = True,
        minimizable      = False,
        frameless        = False,           # hides title bar entirely
        always_on_top    = False,           # paint above other panels
        visible          = True,
        min_width        = 80,
        min_height       = 60,
        max_width        = None,
        max_height       = None,
        shadow           = True,            # drop-shadow around panel
        shadow_color     = "#000000",
        shadow_blur      = 12,
        properties       = {},              # arbitrary metadata dict
    )

    # Draw inside the panel just like a window:
    Draw.shape(display="settings", shape=[{...}])
    Draw.text(display="settings", ...)

    # Control
    Draw.panel.show("settings")
    Draw.panel.hide("settings")
    Draw.panel.close("settings")
    Draw.panel.move("settings", x=100, y=200)
    Draw.panel.resize("settings", width=400, height=500)
    Draw.panel.get("settings")        # → PanelDef
    Draw.panel.list()                 # → [ip, ...]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor, QCursor, QPainter, QPainterPath, QBrush, QPen,
)
from PySide6.QtWidgets import QFrame, QWidget

from Draw._colour import _parse_color
from Draw._window import window as _window_registry, _ALIGN_VALUES


# ── PanelDef ──────────────────────────────────────────────────────────────────

@dataclass
class PanelDef:
    ip: str
    display: str                    # parent window tag
    title: str
    width: int
    height: int
    x: int
    y: int
    align: Optional[str]
    background_color: QColor
    title_color: QColor
    title_background: QColor
    border_color: QColor
    border_width: int
    border_radius: float
    transparency: int               # 0-100
    draggable: bool
    resizable: bool
    closable: bool
    minimizable: bool
    frameless: bool
    always_on_top: bool
    visible: bool
    min_width: int
    min_height: int
    max_width: Optional[int]
    max_height: Optional[int]
    shadow: bool
    shadow_color: QColor
    shadow_blur: int
    properties: Dict[str, Any]      # arbitrary user metadata

    # runtime
    minimized: bool = field(default=False, init=False)
    _frame: Optional["_PanelFrame"] = field(default=None, init=False)


# ── Title bar height ──────────────────────────────────────────────────────────

_TITLE_BAR_H = 32
_BTN_SIZE    = 16
_BTN_MARGIN  = 8
_RESIZE_GRIP = 8


# ── _PanelFrame ───────────────────────────────────────────────────────────────

class _PanelFrame(QFrame):
    """
    Floating panel widget embedded in a parent canvas.
    Provides title bar, drag, optional resize, close/minimise buttons,
    and an inner draw surface compatible with _get_or_create_canvas.
    """

    def __init__(self, pdef: PanelDef, parent: QWidget):
        super().__init__(parent)
        self._pdef       = pdef
        self._dragging   = False
        self._drag_start = QPoint()
        self._resizing   = False
        self._resize_start_pos  = QPoint()
        self._resize_start_size = QSize()
        self._minimized  = False

        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        shadow_pad = pdef.shadow_blur if pdef.shadow else 0
        self.setGeometry(
            pdef.x - shadow_pad,
            pdef.y - shadow_pad,
            pdef.width  + shadow_pad * 2,
            pdef.height + shadow_pad * 2,
        )
        self._shadow_pad = shadow_pad

        if pdef.always_on_top:
            self.raise_()

        # ── inner content canvas ─────────────────────────────────────────────
        # Import here to avoid circular; _DrawCanvas needs QWidget parent.
        from Draw._shapes import _DrawCanvas

        title_h = 0 if pdef.frameless else _TITLE_BAR_H
        content_x = shadow_pad + pdef.border_width
        content_y = shadow_pad + pdef.border_width + title_h
        content_w = max(1, pdef.width  - pdef.border_width * 2)
        content_h = max(1, pdef.height - pdef.border_width * 2 - title_h)

        self._content_widget = QWidget(self)
        self._content_widget.setGeometry(content_x, content_y, content_w, content_h)
        self._content_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        # Create the canvas and attach it — _get_or_create_canvas checks
        # hasattr(win, "_draw_canvas") first so it will find this directly.
        canvas = _DrawCanvas(self._content_widget)
        canvas.setGeometry(self._content_widget.rect())
        self._draw_canvas = canvas   # <- what _get_or_create_canvas looks for

        self.setVisible(pdef.visible)

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _title_rect(self) -> QRect:
        sp = self._shadow_pad
        bw = self._pdef.border_width
        return QRect(sp + bw, sp + bw,
                     self._pdef.width - bw * 2, _TITLE_BAR_H)

    def _body_rect(self) -> QRect:
        sp = self._shadow_pad
        bw = self._pdef.border_width
        title_h = 0 if self._pdef.frameless else _TITLE_BAR_H
        return QRect(sp + bw, sp + bw,
                     self._pdef.width - bw * 2,
                     self._pdef.height - bw * 2)

    def _close_btn_rect(self) -> Optional[QRect]:
        if self._pdef.frameless or not self._pdef.closable:
            return None
        tr = self._title_rect()
        cx = tr.right() - _BTN_MARGIN - _BTN_SIZE
        cy = tr.top() + (tr.height() - _BTN_SIZE) // 2
        return QRect(cx, cy, _BTN_SIZE, _BTN_SIZE)

    def _min_btn_rect(self) -> Optional[QRect]:
        if self._pdef.frameless or not self._pdef.minimizable:
            return None
        close_r = self._close_btn_rect()
        offset = (close_r.left() - _BTN_MARGIN - _BTN_SIZE
                  if close_r else
                  self._title_rect().right() - _BTN_MARGIN - _BTN_SIZE)
        tr = self._title_rect()
        cy = tr.top() + (tr.height() - _BTN_SIZE) // 2
        return QRect(offset, cy, _BTN_SIZE, _BTN_SIZE)

    # ── paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        p    = self._pdef
        sp   = self._shadow_pad
        bw   = p.border_width
        br   = p.border_radius
        full = QRect(sp, sp, p.width, p.height)

        # ── drop shadow ──────────────────────────────────────────────────────
        if p.shadow and p.shadow_blur > 0:
            sc = QColor(p.shadow_color)
            steps = max(1, p.shadow_blur // 2)
            for i in range(steps, 0, -1):
                t = i / steps
                alpha = int(sc.alpha() * (1.0 - t) * 0.5)
                if alpha <= 0:
                    continue
                shadow_c = QColor(sc.red(), sc.green(), sc.blue(), alpha)
                expand = t * p.shadow_blur
                sr = QRect(
                    int(sp - expand / 2),
                    int(sp - expand / 2),
                    int(p.width  + expand),
                    int(p.height + expand),
                )
                path = QPainterPath()
                path.addRoundedRect(sr.x(), sr.y(), sr.width(), sr.height(), br, br)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(shadow_c))
                painter.drawPath(path)

        painter.setOpacity(p.transparency / 100.0)

        # ── panel background ─────────────────────────────────────────────────
        bg_path = QPainterPath()
        bg_path.addRoundedRect(full.x(), full.y(), full.width(), full.height(), br, br)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(p.background_color))
        painter.drawPath(bg_path)

        # ── title bar ────────────────────────────────────────────────────────
        if not p.frameless:
            tr = self._title_rect()
            title_path = QPainterPath()
            # top-rounded only
            title_path.addRoundedRect(tr.x(), tr.y(), tr.width(), tr.height(), br, br)
            # square bottom corners by overdrawing
            title_path.addRect(tr.x(), tr.y() + tr.height() // 2,
                                tr.width(), tr.height() // 2)
            painter.setBrush(QBrush(p.title_background))
            painter.drawPath(title_path)

            # title text
            painter.setPen(QPen(p.title_color))
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            text_r = QRect(tr.x() + 10, tr.y(), tr.width() - 60, tr.height())
            painter.drawText(text_r, Qt.AlignmentFlag.AlignVCenter, p.title)

            # ── close button ─────────────────────────────────────────────────
            cb = self._close_btn_rect()
            if cb:
                painter.setBrush(QBrush(QColor("#f38ba8")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(cb)

            # ── minimise button ──────────────────────────────────────────────
            mb = self._min_btn_rect()
            if mb:
                painter.setBrush(QBrush(QColor("#f9e2af")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(mb)

        # ── border ───────────────────────────────────────────────────────────
        if bw > 0:
            border_path = QPainterPath()
            border_path.addRoundedRect(full.x() + bw / 2, full.y() + bw / 2,
                                        full.width() - bw, full.height() - bw, br, br)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(p.border_color, bw))
            painter.drawPath(border_path)

        painter.end()

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        pos = event.pos()

        # close button
        cb = self._close_btn_rect()
        if cb and cb.contains(pos):
            self._registry_close()
            return

        # minimise button
        mb = self._min_btn_rect()
        if mb and mb.contains(pos):
            self._toggle_minimize()
            return

        # drag via title bar
        if not self._pdef.frameless and self._pdef.draggable:
            tr = self._title_rect()
            if tr.contains(pos):
                self._dragging   = True
                self._drag_start = event.globalPosition().toPoint() - self.pos()
                self.setCursor(QCursor(Qt.CursorShape.SizeAllCursor))
                return

        # resize grip (bottom-right corner)
        if self._pdef.resizable:
            sp = self._shadow_pad
            grip = QRect(sp + self._pdef.width  - _RESIZE_GRIP,
                         sp + self._pdef.height - _RESIZE_GRIP,
                         _RESIZE_GRIP, _RESIZE_GRIP)
            if grip.contains(pos):
                self._resizing = True
                self._resize_start_pos  = event.globalPosition().toPoint()
                self._resize_start_size = QSize(self._pdef.width, self._pdef.height)
                return

        event.ignore()

    def mouseMoveEvent(self, event):
        if self._dragging:
            new_pos = event.globalPosition().toPoint() - self._drag_start
            self.move(new_pos)
            self._pdef.x = new_pos.x() + self._shadow_pad
            self._pdef.y = new_pos.y() + self._shadow_pad
            return

        if self._resizing:
            delta   = event.globalPosition().toPoint() - self._resize_start_pos
            new_w   = max(self._pdef.min_width,  self._resize_start_size.width()  + delta.x())
            new_h   = max(self._pdef.min_height, self._resize_start_size.height() + delta.y())
            if self._pdef.max_width  is not None: new_w = min(new_w, self._pdef.max_width)
            if self._pdef.max_height is not None: new_h = min(new_h, self._pdef.max_height)
            self._pdef.width  = new_w
            self._pdef.height = new_h
            sp = self._shadow_pad
            self.resize(new_w + sp * 2, new_h + sp * 2)
            self._sync_content_geometry()
            self.update()
            return

        event.ignore()

    def mouseReleaseEvent(self, event):
        if self._dragging or self._resizing:
            self._dragging  = False
            self._resizing  = False
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            return
        event.ignore()

    def resizeEvent(self, event):
        self._sync_content_geometry()
        super().resizeEvent(event)

    def _sync_content_geometry(self):
        p = self._pdef
        sp   = self._shadow_pad
        bw   = p.border_width
        title_h = 0 if p.frameless else _TITLE_BAR_H
        cx = sp + bw
        cy = sp + bw + title_h
        cw = max(1, p.width  - bw * 2)
        ch = max(1, p.height - bw * 2 - title_h)
        self._content_widget.setGeometry(cx, cy, cw, ch)
        if hasattr(self, "_draw_canvas"):
            self._draw_canvas.setGeometry(0, 0, cw, ch)

    def _toggle_minimize(self):
        p = self._pdef
        sp = self._shadow_pad
        if not self._minimized:
            self._minimized   = True
            p.minimized       = True
            title_h = _TITLE_BAR_H
            self.resize(p.width + sp * 2, title_h + sp * 2 + p.border_width * 2)
            self._content_widget.hide()
        else:
            self._minimized   = False
            p.minimized       = False
            self.resize(p.width + sp * 2, p.height + sp * 2)
            self._content_widget.show()
            self._sync_content_geometry()
        self.update()

    def _registry_close(self):
        ip = self._pdef.ip
        panel._panels.pop(ip, None)
        # remove from window registry so Draw.shape(display=ip) stops working
        _window_registry._windows.pop(ip, None)
        self.hide()
        self.deleteLater()


# ── alignment helper (uses Draw._align) ──────────────────────────────────────

def _panel_align_pos(align: str, pw: int, ph: int, cw: int, ch: int) -> Tuple[int, int]:
    from Draw._align import calculate_alignment_pos
    x, y = calculate_alignment_pos(align, float(pw), float(ph), float(cw), float(ch))
    return (int(x), int(y))



# ── registry ──────────────────────────────────────────────────────────────────

class _PanelRegistry:
    """
    Public API:  Draw.panel(ip="...", display="main", ...)
    """

    def __init__(self):
        self._panels: Dict[str, PanelDef] = {}

    def __call__(
        self,
        *,
        ip: str,
        display: Optional[str] = None,
        title: str = "",
        width: int = 300,
        height: int = 400,
        x: Optional[int] = None,
        y: Optional[int] = None,
        align: Optional[str] = None,
        background_color: object = "#1e1e2e",
        title_color: object = "#cdd6f4",
        title_background: object = "#313244",
        border_color: object = "#45475a",
        border_width: int = 1,
        border_radius: float = 10.0,
        transparency: int = 100,
        draggable: bool = True,
        resizable: bool = False,
        closable: bool = True,
        minimizable: bool = False,
        frameless: bool = False,
        always_on_top: bool = False,
        visible: bool = True,
        min_width: int = 80,
        min_height: int = 60,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        shadow: bool = True,
        shadow_color: object = "#000000",
        shadow_blur: int = 12,
        properties: Optional[Dict[str, Any]] = None,
    ) -> PanelDef:
        if not ip or not isinstance(ip, str):
            raise ValueError("Draw.panel: 'ip' is required.")

        # Return existing panel if already created
        if ip in self._panels:
            return self._panels[ip]

        # Resolve parent window
        window_tag = display
        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise ValueError("Draw.panel: multiple windows — 'display' is required.")
            else:
                raise ValueError("Draw.panel: no windows exist. Call Draw.window() first.")

        parent_win = _window_registry.get(window_tag)
        if parent_win is None:
            raise ValueError(f"Draw.panel: window '{window_tag}' not found.")
        # Get or create the parent canvas
        from Draw._text import _get_or_create_canvas
        parent_canvas = _get_or_create_canvas(window_tag, parent_win)

        # Resolve position
        if align is not None:
            if align not in _ALIGN_VALUES:
                raise ValueError(f"Draw.panel: invalid align='{align}'.")
            px, py = _panel_align_pos(
                align, width, height,
                parent_canvas.width(), parent_canvas.height(),
            )
        else:
            px = x if x is not None else 20
            py = y if y is not None else 20

        pdef = PanelDef(
            ip               = ip,
            display          = window_tag,
            title            = title,
            width            = width,
            height           = height,
            x                = px,
            y                = py,
            align            = align,
            background_color = _parse_color(background_color),
            title_color      = _parse_color(title_color),
            title_background = _parse_color(title_background),
            border_color     = _parse_color(border_color),
            border_width     = border_width,
            border_radius    = border_radius,
            transparency     = max(0, min(100, transparency)),
            draggable        = draggable,
            resizable        = resizable,
            closable         = closable,
            minimizable      = minimizable,
            frameless        = frameless,
            always_on_top    = always_on_top,
            visible          = visible,
            min_width        = min_width,
            min_height       = min_height,
            max_width        = max_width,
            max_height       = max_height,
            shadow           = shadow,
            shadow_color     = _parse_color(shadow_color),
            shadow_blur      = shadow_blur,
            properties       = dict(properties or {}),
        )

        # Build the frame widget (child of parent canvas)
        frame = _PanelFrame(pdef, parent_canvas)
        frame.show()
        pdef._frame = frame

        # Register under ip in window registry so Draw.shape(display=ip) works
        _window_registry._windows[ip] = frame

        self._panels[ip] = pdef
        return pdef

    # ── control methods ──────────────────────────────────────────────────────

    def show(self, ip: str) -> None:
        p = self._panels.get(ip)
        if p and p._frame:
            p.visible = True
            p._frame.show()

    def hide(self, ip: str) -> None:
        p = self._panels.get(ip)
        if p and p._frame:
            p.visible = False
            p._frame.hide()

    def close(self, ip: str) -> None:
        p = self._panels.get(ip)
        if p and p._frame:
            p._frame._registry_close()

    def move(self, ip: str, *, x: int, y: int) -> None:
        p = self._panels.get(ip)
        if p and p._frame:
            sp = p._frame._shadow_pad
            p.x, p.y = x, y
            p._frame.move(x - sp, y - sp)

    def resize(self, ip: str, *, width: int, height: int) -> None:
        p = self._panels.get(ip)
        if not p or not p._frame:
            return
        p.width  = max(p.min_width, width)
        p.height = max(p.min_height, height)
        sp = p._frame._shadow_pad
        p._frame.resize(p.width + sp * 2, p.height + sp * 2)
        p._frame._sync_content_geometry()
        p._frame.update()

    def get(self, ip: str) -> Optional[PanelDef]:
        return self._panels.get(ip)

    def list(self) -> List[str]:
        return list(self._panels.keys())

    def clear(self, ip: str) -> None:
        """Remove all shapes/text drawn inside the panel."""
        p = self._panels.get(ip)
        if p and p._frame and hasattr(p._frame, "_draw_canvas"):
            canvas = p._frame._draw_canvas
            canvas.shape_items.clear()
            canvas.text_items.clear()
            if hasattr(canvas, "_shape_by_ip"):
                canvas._shape_by_ip.clear()
            if hasattr(canvas, "_shape_hash_by_ip"):
                canvas._shape_hash_by_ip.clear()
            canvas._occupied_dirty = True
            canvas.update()

    def update_style(self, ip: str, **kwargs) -> None:
        """
        Live-update panel appearance without recreating it.
        Accepts the same style keys as Draw.panel().
        """
        p = self._panels.get(ip)
        if not p or not p._frame:
            return
        color_keys = {"background_color", "title_color", "title_background",
                      "border_color", "shadow_color"}
        for k, v in kwargs.items():
            if not hasattr(p, k):
                continue
            if k in color_keys:
                setattr(p, k, _parse_color(v))
            elif k == "transparency":
                p.transparency = max(0, min(100, int(v)))
            else:
                setattr(p, k, v)
        p._frame.update()


# ── singleton ─────────────────────────────────────────────────────────────────

panel = _PanelRegistry()
