"""
Draw._native  v1
================
Two things live here:

  Draw.widget(ip=..., type=..., ...)   — a factory for *real* native Qt
      controls: checkbox, radio, button, listbox, combobox, slider,
      spinbox, progressbar, tabs, scrollarea, and canvas (an embedded
      paintable Draw surface you can Draw.shape()/Draw.text() into).

  Draw.box(ip=..., direction=..., children=[...])  — a *real*
      QVBoxLayout/QHBoxLayout container that reparents already-created
      native widgets (from Draw.widget, Draw.filetree, Draw.lineedit,
      Draw.textedit, or a nested Draw.box) into itself and lets Qt manage
      their geometry automatically — true vbox/hbox stacking, unlike
      Draw.room()'s one-shot relative-position math for painted shapes.

Both stack on top of the shared canvas the same way Draw.panel embeds
its content widget: a real QWidget, child-parented, raised above paint.

PUBLIC API
----------
    Draw.widget(ip="agree", type="checkbox", display="main",
                 x=20, y=20, text="I agree", checked=False, on_toggle=fn)

    Draw.widget(ip="fruit", type="combobox", display="main",
                 x=20, y=60, width=160, items=["Apple", "Banana"],
                 on_change=fn)

    Draw.widget(ip="log", type="listbox", display="main",
                 x=20, y=100, width=200, height=150, items=["a", "b"],
                 on_select=fn)

    Draw.widget(ip="pages", type="tabs", display="main",
                 x=20, y=20, width=400, height=300)
    Draw.widget.add_tab("pages", "Home", child_ip="fruit")

    Draw.widget(ip="scene", type="canvas", display="main",
                 x=20, y=20, width=400, height=300)
    Draw.shape(display="scene", shape=[...])   # paints straight onto it

    Draw.widget.get_value(ip) / .set_value(ip, value)
    Draw.widget.get(ip) / .list() / .move(ip, x=, y=) / .resize(ip, width=, height=)
    Draw.widget.show(ip) / .hide(ip) / .close(ip)

    Draw.box(ip="sidebar", display="main", direction="vertical",
             x=20, y=20, width=280, height=500, spacing=8,
             children=["fruit", "log"])
    Draw.box.add_child("sidebar", "agree")
    Draw.box.add_stretch("sidebar")
    Draw.box.get(ip) / .list() / .move / .resize / .show / .hide / .close
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from PySide6.QtCore import Qt
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QBoxLayout,
    QCheckBox, QRadioButton, QPushButton, QListWidget, QListWidgetItem,
    QComboBox, QSlider, QSpinBox, QProgressBar, QTabWidget, QScrollArea,
    QButtonGroup,
)

from Draw._window import window as _window_registry
from Draw._text import (
    _get_or_create_canvas,
    _qss_from_edit_style,
    _resolve_edit_window,
)
from Draw._file_tree import filetree as _filetree_registry
from Draw._text import lineedit as _lineedit_registry, textedit as _textedit_registry

_WIDGET_TYPES = {
    "checkbox", "radio", "button", "listbox", "combobox",
    "slider", "spinbox", "progressbar", "tabs", "scrollarea", "canvas",
}


# ── shared widget lookup across all Draw native registries ───────────────────

def _resolve_native_widget(ip: str) -> Optional["QWidget"]:
    """Find the underlying QWidget for *ip* across every native registry."""
    item = widget._items.get(ip)
    if item and item._widget:
        return item._widget
    b = box._boxes.get(ip)
    if b and b._widget:
        return b._widget
    t = _filetree_registry.get(ip)
    if t and t._widget:
        return t._widget
    l = _lineedit_registry.get(ip)
    if l and l._widget:
        return l._widget
    e = _textedit_registry.get(ip)
    if e and e._widget:
        return e._widget
    return None


# ── Draw.widget ────────────────────────────────────────────────────────────────

@dataclass
class NativeWidgetDef:
    ip: str
    type: str
    display: str
    x: int
    y: int
    width: int
    height: int
    style: dict
    properties: dict
    _widget: Optional["QWidget"] = field(default=None, init=False)


class _NativeWidgetRegistry:
    """Public API: Draw.widget(ip="...", type="checkbox"/"radio"/"button"/
    "listbox"/"combobox"/"slider"/"spinbox"/"progressbar"/"tabs"/
    "scrollarea"/"canvas", display="main", ...)"""

    def __init__(self):
        self._items: Dict[str, NativeWidgetDef] = {}
        self._button_groups: Dict[str, "QButtonGroup"] = {}

    def __call__(
        self,
        *,
        ip: str,
        type: str,
        display: Optional[str] = None,
        x: int = 20,
        y: int = 20,
        width: int = 120,
        height: int = 32,
        text: str = "",
        items: Optional[List[str]] = None,
        checked: bool = False,
        group: Optional[str] = None,
        min: int = 0,
        max: int = 100,
        value: int = 0,
        editable: bool = False,
        orientation: str = "horizontal",
        style: Optional[dict] = None,
        on_change: Optional[Callable[[Any], None]] = None,
        on_click: Optional[Callable[[], None]] = None,
        on_toggle: Optional[Callable[[bool], None]] = None,
        on_select: Optional[Callable[[str], None]] = None,
        properties: Optional[dict] = None,
    ) -> NativeWidgetDef:
        if not ip or not isinstance(ip, str):
            raise ValueError("Draw.widget: 'ip' is required.")
        if ip in self._items:
            return self._items[ip]
        if type not in _WIDGET_TYPES:
            raise ValueError(f"Draw.widget: invalid type={type!r}. Choose from: {sorted(_WIDGET_TYPES)}")

        window_tag = _resolve_edit_window(display, "Draw.widget")
        win = _window_registry.get(window_tag)
        canvas = _get_or_create_canvas(window_tag, win)

        w: QWidget
        if type == "checkbox":
            w = QCheckBox(text, canvas)
            w.setChecked(checked)
            if on_toggle is not None:
                w.toggled.connect(lambda v: on_toggle(v))

        elif type == "radio":
            w = QRadioButton(text, canvas)
            w.setChecked(checked)
            if group:
                grp = self._button_groups.setdefault(group, QButtonGroup())
                grp.addButton(w)
            if on_toggle is not None:
                w.toggled.connect(lambda v: on_toggle(v))

        elif type == "button":
            w = QPushButton(text, canvas)
            if on_click is not None:
                w.clicked.connect(lambda: on_click())

        elif type == "listbox":
            w = QListWidget(canvas)
            for it in (items or []):
                QListWidgetItem(str(it), w)
            if on_select is not None:
                w.itemSelectionChanged.connect(
                    lambda: on_select(w.currentItem().text() if w.currentItem() else None)
                )
            if on_change is not None:
                w.currentTextChanged.connect(lambda s: on_change(s))

        elif type == "combobox":
            w = QComboBox(canvas)
            w.addItems([str(it) for it in (items or [])])
            w.setEditable(editable)
            if on_change is not None:
                w.currentTextChanged.connect(lambda s: on_change(s))

        elif type == "slider":
            orient = Qt.Orientation.Vertical if orientation == "vertical" else Qt.Orientation.Horizontal
            w = QSlider(orient, canvas)
            w.setRange(min, max)
            w.setValue(value)
            if on_change is not None:
                w.valueChanged.connect(lambda v: on_change(v))

        elif type == "spinbox":
            w = QSpinBox(canvas)
            w.setRange(min, max)
            w.setValue(value)
            if on_change is not None:
                w.valueChanged.connect(lambda v: on_change(v))

        elif type == "progressbar":
            w = QProgressBar(canvas)
            w.setRange(min, max)
            w.setValue(value)

        elif type == "tabs":
            w = QTabWidget(canvas)

        elif type == "scrollarea":
            w = QScrollArea(canvas)
            w.setWidgetResizable(True)

        elif type == "canvas":
            # A nested, independently-paintable Draw surface: register it
            # under its own ip so Draw.shape(display=ip)/Draw.text(display=ip)
            # work directly, exactly like Draw.panel's content canvas does.
            from Draw._shapes import _DrawCanvas
            w = _DrawCanvas(canvas)
            w._draw_canvas = w   # self-reference so _get_or_create_canvas short-circuits
            _window_registry._windows[ip] = w

        else:  # pragma: no cover - guarded above
            raise ValueError(f"Draw.widget: unhandled type={type!r}")

        w.setGeometry(x, y, width, height)
        if style:
            if isinstance(style, str):
                w.setStyleSheet(style)
            else:
                qss = _qss_from_edit_style(style)
                if qss:
                    w.setStyleSheet(f"{w.metaObject().className()} {{ {qss} }}")
        else:
            # Sleek modern dark theme defaults
            if type == "slider":
                w.setStyleSheet(
                    "QSlider::groove:horizontal { height: 4px; background: #334155; border-radius: 2px; }"
                    "QSlider::sub-page:horizontal { background: #eab308; border-radius: 2px; }"
                    "QSlider::handle:horizontal { background: #facc15; border: 1px solid #fef08a; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; }"
                    "QSlider::groove:vertical { width: 4px; background: #334155; border-radius: 2px; }"
                    "QSlider::sub-page:vertical { background: #eab308; border-radius: 2px; }"
                    "QSlider::handle:vertical { background: #facc15; border: 1px solid #fef08a; height: 12px; margin-left: -4px; margin-right: -4px; border-radius: 6px; }"
                )
            elif type == "combobox":
                w.setStyleSheet(
                    "QComboBox { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 4px 8px; font-size: 11px; }"
                    "QComboBox::drop-down { border: none; width: 20px; }"
                    "QComboBox QAbstractItemView { background-color: #0f172a; color: #f8fafc; selection-background-color: #eab308; selection-color: #000; }"
                )
            elif type == "button":
                w.setStyleSheet(
                    "QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 6px 12px; font-weight: bold; }"
                    "QPushButton:hover { background-color: #334155; color: #fff; }"
                    "QPushButton:pressed { background-color: #facc15; color: #000; }"
                )
            elif type == "checkbox":
                w.setStyleSheet(
                    "QCheckBox { color: #cbd5e1; font-size: 11px; spacing: 6px; }"
                    "QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #475569; background: #1e293b; }"
                    "QCheckBox::indicator:checked { background: #facc15; border-color: #fef08a; }"
                )

        ndef = NativeWidgetDef(
            ip=ip, type=type, display=window_tag, x=x, y=y, width=width, height=height,
            style=dict(style) if isinstance(style, dict) else ({"raw": style} if style else {}),
            properties=dict(properties or {}),
        )
        ndef._widget = w
        self._items[ip] = ndef

        w.show()
        w.raise_()
        return ndef

    # -- type-specific helpers --------------------------------------------------

    def add_tab(self, ip: str, label: str, *, child_ip: Optional[str] = None,
                widget_obj: Optional["QWidget"] = None) -> None:
        item = self._items.get(ip)
        if not item or item.type != "tabs" or not item._widget:
            return
        child = widget_obj
        if child is None and child_ip is not None:
            child = _resolve_native_widget(child_ip)
        if child is None:
            child = QWidget()
        item._widget.addTab(child, label)

    def add_item(self, ip: str, text: str) -> None:
        item = self._items.get(ip)
        if not item or not item._widget:
            return
        if item.type == "listbox":
            QListWidgetItem(str(text), item._widget)
        elif item.type == "combobox":
            item._widget.addItem(str(text))

    def set_content(self, ip: str, child_ip: str) -> None:
        """For type='scrollarea': set the widget it scrolls."""
        item = self._items.get(ip)
        if not item or item.type != "scrollarea" or not item._widget:
            return
        child = _resolve_native_widget(child_ip)
        if child is not None:
            item._widget.setWidget(child)

    # -- generic value access --------------------------------------------------

    def get_value(self, ip: str) -> Any:
        item = self._items.get(ip)
        if not item or not item._widget:
            return None
        w, t = item._widget, item.type
        if t in ("checkbox", "radio"):
            return w.isChecked()
        if t == "listbox":
            cur = w.currentItem()
            return cur.text() if cur else None
        if t == "combobox":
            return w.currentText()
        if t in ("slider", "spinbox", "progressbar"):
            return w.value()
        return None

    def set_value(self, ip: str, value: Any) -> None:
        item = self._items.get(ip)
        if not item or not item._widget:
            return
        w, t = item._widget, item.type
        if t in ("checkbox", "radio"):
            w.setChecked(bool(value))
        elif t == "combobox":
            idx = w.findText(str(value))
            if idx >= 0:
                w.setCurrentIndex(idx)
        elif t in ("slider", "spinbox", "progressbar"):
            w.setValue(int(value))

    # -- control API --------------------------------------------------------------

    def get(self, ip: str) -> Optional[NativeWidgetDef]:
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
            if item.type == "canvas":
                _window_registry._windows.pop(ip, None)
            item._widget.deleteLater()


widget = _NativeWidgetRegistry()


# ── Draw.box ─────────────────────────────────────────────────────────────────

@dataclass
class BoxDef:
    ip: str
    display: str
    direction: str
    x: int
    y: int
    width: int
    height: int
    spacing: int
    margins: Tuple[int, int, int, int]
    style: dict
    _widget: Optional["QWidget"] = field(default=None, init=False)
    _layout: Optional["QBoxLayout"] = field(default=None, init=False)


class _BoxRegistry:
    """Public API: Draw.box(ip="...", display="main", direction="vertical"|
    "horizontal", children=[ip, ...]) → a real QVBoxLayout/QHBoxLayout
    container that reparents existing native widgets into itself."""

    def __init__(self):
        self._boxes: Dict[str, BoxDef] = {}

    def __call__(
        self,
        *,
        ip: str,
        display: Optional[str] = None,
        direction: str = "vertical",
        x: int = 20,
        y: int = 20,
        width: int = 300,
        height: int = 400,
        spacing: int = 8,
        margins: Tuple[int, int, int, int] = (8, 8, 8, 8),
        children: Optional[List[str]] = None,
        style: Optional[dict] = None,
    ) -> BoxDef:
        if not ip or not isinstance(ip, str):
            raise ValueError("Draw.box: 'ip' is required.")
        if ip in self._boxes:
            return self._boxes[ip]
        if direction not in ("vertical", "horizontal"):
            raise ValueError("Draw.box: direction must be 'vertical' or 'horizontal'.")

        window_tag = _resolve_edit_window(display, "Draw.box")
        win = _window_registry.get(window_tag)
        canvas = _get_or_create_canvas(window_tag, win)

        container = QWidget(canvas)
        container.setObjectName(ip)
        container.setGeometry(x, y, width, height)
        layout_cls = QVBoxLayout if direction == "vertical" else QHBoxLayout
        layout = layout_cls(container)
        layout.setSpacing(spacing)
        layout.setContentsMargins(*margins)
        qss = _qss_from_edit_style(style)
        if qss:
            container.setStyleSheet(f"QWidget#{ip} {{ {qss} }}")

        bdef = BoxDef(
            ip=ip, display=window_tag, direction=direction,
            x=x, y=y, width=width, height=height,
            spacing=spacing, margins=margins, style=dict(style or {}),
        )
        bdef._widget = container
        bdef._layout = layout
        self._boxes[ip] = bdef

        container.show()
        container.raise_()

        for child_ip in (children or []):
            self.add_child(ip, child_ip)

        return bdef

    def add_child(self, ip: str, child_ip: str, *, stretch: int = 0) -> None:
        bdef = self._boxes.get(ip)
        if not bdef or not bdef._layout:
            return
        child = _resolve_native_widget(child_ip)
        if child is None:
            return
        child.setParent(bdef._widget)
        bdef._layout.addWidget(child, stretch)
        child.show()

    def add_stretch(self, ip: str, stretch: int = 1) -> None:
        bdef = self._boxes.get(ip)
        if bdef and bdef._layout:
            bdef._layout.addStretch(stretch)

    def add_spacing(self, ip: str, pixels: int) -> None:
        bdef = self._boxes.get(ip)
        if bdef and bdef._layout:
            bdef._layout.addSpacing(pixels)

    def get(self, ip: str) -> Optional[BoxDef]:
        return self._boxes.get(ip)

    def list(self) -> List[str]:
        return list(self._boxes.keys())

    def move(self, ip: str, *, x: int, y: int) -> None:
        bdef = self._boxes.get(ip)
        if bdef and bdef._widget:
            bdef.x, bdef.y = x, y
            bdef._widget.move(x, y)

    def resize(self, ip: str, *, width: int, height: int) -> None:
        bdef = self._boxes.get(ip)
        if bdef and bdef._widget:
            bdef.width, bdef.height = width, height
            bdef._widget.resize(width, height)

    def show(self, ip: str) -> None:
        bdef = self._boxes.get(ip)
        if bdef and bdef._widget:
            bdef._widget.show()

    def hide(self, ip: str) -> None:
        bdef = self._boxes.get(ip)
        if bdef and bdef._widget:
            bdef._widget.hide()

    def close(self, ip: str) -> None:
        bdef = self._boxes.pop(ip, None)
        if bdef and bdef._widget:
            bdef._widget.deleteLater()


box = _BoxRegistry()


# ── High-Level Convenience Wrappers ──────────────────────────────────────────

def slider(
    ip: str,
    *,
    display: Optional[str] = None,
    x: int = 20,
    y: int = 20,
    width: int = 180,
    height: int = 28,
    min: int = 0,
    max: int = 100,
    value: int = 0,
    orientation: str = "horizontal",
    style: Optional[Any] = None,
    on_change: Optional[Callable[[int], None]] = None,
    **kwargs,
) -> NativeWidgetDef:
    """Convenience helper to create a styled native slider."""
    return widget(
        ip=ip,
        type="slider",
        display=display,
        x=x,
        y=y,
        width=width,
        height=height,
        min=min,
        max=max,
        value=value,
        orientation=orientation,
        style=style,
        on_change=on_change,
        **kwargs,
    )


def button(
    ip: str,
    *,
    display: Optional[str] = None,
    x: int = 20,
    y: int = 20,
    width: int = 120,
    height: int = 34,
    text: str = "",
    style: Optional[Any] = None,
    on_click: Optional[Callable[[], None]] = None,
    **kwargs,
) -> NativeWidgetDef:
    """Convenience helper to create a styled native push button."""
    return widget(
        ip=ip,
        type="button",
        display=display,
        x=x,
        y=y,
        width=width,
        height=height,
        text=text,
        style=style,
        on_click=on_click,
        **kwargs,
    )


def combobox(
    ip: str,
    *,
    display: Optional[str] = None,
    x: int = 20,
    y: int = 20,
    width: int = 200,
    height: int = 32,
    items: Optional[List[str]] = None,
    style: Optional[Any] = None,
    on_change: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> NativeWidgetDef:
    """Convenience helper to create a styled native combobox dropdown."""
    return widget(
        ip=ip,
        type="combobox",
        display=display,
        x=x,
        y=y,
        width=width,
        height=height,
        items=items,
        style=style,
        on_change=on_change,
        **kwargs,
    )


def checkbox(
    ip: str,
    *,
    display: Optional[str] = None,
    x: int = 20,
    y: int = 20,
    width: int = 140,
    height: int = 28,
    text: str = "",
    checked: bool = False,
    style: Optional[Any] = None,
    on_toggle: Optional[Callable[[bool], None]] = None,
    **kwargs,
) -> NativeWidgetDef:
    """Convenience helper to create a styled native checkbox."""
    return widget(
        ip=ip,
        type="checkbox",
        display=display,
        x=x,
        y=y,
        width=width,
        height=height,
        text=text,
        checked=checked,
        style=style,
        on_toggle=on_toggle,
        **kwargs,
    )

