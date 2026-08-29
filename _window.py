"""
Draw._window

Window creation and lifecycle management for Draw.

Usage:
    import Draw
    Draw.window(tag="main", title="Hello", width=800, height=600, style="background-color: #1e1e2e; color: white;")
    Draw.window.show("main")
    Draw.window.run()
    Draw.window.quit()
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

# pyrefly: ignore [missing-import]
from PySide6.QtCore import Qt, QTimer, QPoint, QObject, Signal, Slot
# pyrefly: ignore [missing-import]
from PySide6.QtGui import QColor, QCloseEvent, QIcon, QMouseEvent, QResizeEvent, QMoveEvent, QAction
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QApplication, QMainWindow, QMenuBar, QStatusBar, QMenu

from Draw._app import get_app


# Canonical static colour parser now lives in Draw._colour — this is a
# backward-compat re-export so existing `from Draw._window import _parse_color`
# call sites (in _shapes.py, _text.py, _point.py, _layout.py) keep working
# unchanged. New code should import these from Draw._colour directly.
from Draw._colour import _parse_color, _NAMED_COLORS
from Draw._align import _ALIGN_PRESETS as _ALIGN_VALUES



class _ManagedWindow(QMainWindow):
    """
    QMainWindow that tells the registry when Qt actually closes it, supports
    frameless dragging, and dispatches window event callbacks.
    """

    def __init__(
        self,
        *,
        tag: str,
        on_closed: Callable[[str], None],
        flags: Qt.WindowType,
        draggable: bool = True,
        on_resize: Optional[Callable[[int, int], None]] = None,
        on_move: Optional[Callable[[int, int], None]] = None,
        on_focus: Optional[Callable[[bool], None]] = None,
        on_close: Optional[Callable[[], bool]] = None,
    ) -> None:
        super().__init__(flags=flags)
        self._draw_tag = tag
        self._draw_on_closed = on_closed
        self._draw_draggable = draggable
        self._draw_on_resize = on_resize
        self._draw_on_move = on_move
        self._draw_on_focus = on_focus
        self._draw_on_close = on_close
        self._drag_pos: Optional[QPoint] = None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API name
        if self._draw_on_close is not None:
            try:
                result = self._draw_on_close()
                if result is False:
                    event.ignore()
                    return
            except Exception:
                pass
        super().closeEvent(event)
        if event.isAccepted():
            self._draw_on_closed(self._draw_tag)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        if self._draw_on_resize is not None:
            try:
                sz = event.size()
                self._draw_on_resize(sz.width(), sz.height())
            except Exception:
                pass

    def moveEvent(self, event: QMoveEvent) -> None:  # noqa: N802 - Qt API name
        super().moveEvent(event)
        if self._draw_on_move is not None:
            try:
                pos = event.pos()
                self._draw_on_move(pos.x(), pos.y())
            except Exception:
                pass

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().changeEvent(event)
        if event.type() == event.Type.ActivationChange and self._draw_on_focus is not None:
            try:
                self._draw_on_focus(self.isActiveWindow())
            except Exception:
                pass

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        if self._draw_draggable and event.button() == Qt.MouseButton.LeftButton:
            if self.windowFlags() & Qt.WindowType.FramelessWindowHint:
                self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API name
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.key() == Qt.Key.Key_Escape or (
            event.key() == Qt.Key.Key_Q
            and (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)


def _apply_align(win: QMainWindow, align: str) -> None:
    """Move the window to the requested screen position."""
    screen = win.screen()
    if screen is None:
        return

    sg = screen.availableGeometry()
    w, h = win.width(), win.height()
    cx, cy = sg.x(), sg.y()
    sw, sh = sg.width(), sg.height()

    from Draw._align import calculate_alignment_pos
    pos_x, pos_y = calculate_alignment_pos(align, float(w), float(h), float(sw), float(sh), offset_x=float(cx), offset_y=float(cy))
    win.move(int(pos_x), int(pos_y))


def _tick_dynamic_bg(win: QMainWindow) -> None:
    """
    Timer callback for one window's dynamic background colour. Resolves the
    window's Draw.color(ip=...) binding (if any) via _bridge, and only
    touches the palette / repaints when the resolved colour actually
    changed since the last tick.
    """
    ip = getattr(win, "_draw_color_ip", None)
    if not ip:
        return
    from Draw import _bridge
    resolved = _bridge.resolve_dynamic_color(
        ip, x=0.0, y=0.0, w=float(win.width()), h=float(win.height())
    )
    if not resolved:
        return
    rgba = resolved.get("body_color")
    if rgba is None or rgba == win._draw_bg_last_rgba:
        return
    win._draw_bg_last_rgba = rgba
    r, g, b, a = rgba
    palette = win.palette()
    palette.setColor(win.backgroundRole(), QColor(int(r), int(g), int(b), int(a)))
    win.setPalette(palette)
    win.update()


def _start_dynamic_bg_timer(win: QMainWindow) -> None:
    """Start a 60fps poll for win's dynamic background colour."""
    timer = QTimer(win)
    timer.setInterval(16)
    timer.timeout.connect(lambda: _tick_dynamic_bg(win))
    timer.start()
    win._draw_bg_timer = timer


def _stop_connectors() -> None:
    """Best-effort shutdown for Draw's connector ticker."""
    try:
        from Draw._connectors import connectors

        connectors.stop_ticker()
    except Exception:
        pass


class _ShutdownDispatcher(QObject):
    """
    Routes an emergency shutdown request (e.g. from Draw.debug's watchdog
    daemon thread) onto the Qt GUI thread.

    Qt widgets may only be touched on the thread that owns them. Emitting a
    Qt signal with the default (Auto) connection type is automatically
    delivered as a *queued* call when the emitting thread differs from the
    receiving QObject's thread, and as a normal direct call when it's the
    same thread — so this is safe to invoke from any thread without ever
    running Qt GUI code off-thread. Mirrors the ``_GUIDispatcher`` pattern
    already used by Draw._connectors for its background ticker thread.

    Must be constructed on the Qt GUI thread.
    """

    _fire: Signal = Signal(str)

    def __init__(self, registry: "_WindowRegistry") -> None:
        super().__init__()
        self._registry = registry
        self._fire.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(str)
    def _run(self, reason: str) -> None:
        self._registry.close_all()

    def post(self, reason: str) -> None:
        """Thread-safe: request shutdown from any thread."""
        self._fire.emit(reason)


class _WindowRegistry:
    """Holds all windows created by Draw.window()."""

    def __init__(self):
        self._windows: dict[str, QMainWindow] = {}
        self._shutdown_connected = False
        self._shutdown_dispatcher: Optional[_ShutdownDispatcher] = None

    def __call__(
        self,
        *,
        tag: str = "main",
        title: str = "",
        width: int = 800,
        height: int = 600,
        x: Optional[int] = None,
        y: Optional[int] = None,
        align: Optional[str] = None,
        background_color: Union[str, Tuple[int, int, int], QColor] = "white",
        transparency: int = 100,
        frameless: bool = False,
        always_on_top: bool = False,
        resizable: bool = True,
        min_width: Optional[int] = None,
        min_height: Optional[int] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        ip: Optional[str] = None,
        # Extended professional parameters:
        style: Optional[str] = None,
        css: Optional[str] = None,
        icon: Optional[str] = None,
        modal: Union[bool, str] = False,
        screen: Optional[Union[int, str]] = None,
        draggable: bool = True,
        on_resize: Optional[Callable[[int, int], None]] = None,
        on_move: Optional[Callable[[int, int], None]] = None,
        on_focus: Optional[Callable[[bool], None]] = None,
        on_close: Optional[Callable[[], bool]] = None,
    ) -> QMainWindow:
        """
        Create or return a window identified by *tag*.

        Parameters
        ----------
        tag            : Unique identifier. Required.
        title          : Window title-bar text.
        width / height : Size in pixels.
        x / y          : Screen position. Ignored when *align* is set.
        align          : center, top, bottom, left, right, top-left,
                         top-right, bottom-left, or bottom-right.
        background_color : Named colour, hex string, RGB tuple, or QColor.
        transparency   : 0 is invisible, 100 is fully opaque.
        frameless      : Remove the OS window frame / title-bar.
        always_on_top  : Keep the window above all others.
        resizable      : Allow the user to resize the window.
        min/max_width/height : Size constraints.
        ip             : Optional dynamic color IP binding.
        style / css    : Direct text-based CSS / QSS styling string.
        icon           : File path to window icon.
        modal          : Set modal window state (True / "application" / "window").
        screen         : Target monitor index or screen name.
        draggable      : Allow dragging frameless windows by mouse.
        on_resize      : Callback(width, height) triggered on window resize.
        on_move        : Callback(x, y) triggered on window move.
        on_focus       : Callback(is_active) triggered on focus change.
        on_close       : Callback() -> bool. Return False to prevent close.
        """
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("Draw.window: 'tag' must be a non-empty string.")

        tag = tag.strip()
        if tag in self._windows:
            return self._windows[tag]

        if align is not None and align not in _ALIGN_VALUES:
            raise ValueError(
                f"Draw.window: invalid align='{align}'. "
                f"Choose from: {sorted(_ALIGN_VALUES)}"
            )

        if not 0 <= transparency <= 100:
            raise ValueError(
                "Draw.window: 'transparency' must be between 0 and 100."
            )

        app = get_app()
        self._connect_shutdown_hook(app)

        flags = Qt.WindowType.Window
        if frameless:
            flags |= Qt.WindowType.FramelessWindowHint
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint

        win = _ManagedWindow(
            tag=tag,
            on_closed=self._forget,
            flags=flags,
            draggable=draggable,
            on_resize=on_resize,
            on_move=on_move,
            on_focus=on_focus,
            on_close=on_close,
        )
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.setWindowTitle(title)
        win.resize(width, height)

        if screen is not None:
            screens = app.screens()
            target_screen = None
            if isinstance(screen, int) and 0 <= screen < len(screens):
                target_screen = screens[screen]
            elif isinstance(screen, str):
                for s in screens:
                    if s.name() == screen:
                        target_screen = s
                        break
            if target_screen is not None:
                win.setScreen(target_screen)

        if not resizable:
            win.setFixedSize(width, height)
        else:
            if min_width is not None or min_height is not None:
                win.setMinimumSize(min_width or 0, min_height or 0)
            if max_width is not None or max_height is not None:
                win.setMaximumSize(
                    max_width or 16_777_215,
                    max_height or 16_777_215,
                )

        color = _parse_color(background_color)
        palette = win.palette()
        palette.setColor(win.backgroundRole(), color)
        win.setPalette(palette)
        win.setAutoFillBackground(True)
        win.setWindowOpacity(transparency / 100.0)

        style_str = css if css is not None else style
        if style_str:
            win.setStyleSheet(style_str)

        if icon:
            win.setWindowIcon(QIcon(icon))

        if modal:
            if modal is True or modal == "application":
                win.setWindowModality(Qt.WindowModality.ApplicationModal)
            elif modal == "window":
                win.setWindowModality(Qt.WindowModality.WindowModal)

        if x is not None and y is not None:
            win.move(x, y)
        elif align is not None:
            _apply_align(win, align)

        win._draw_color_ip = str(ip).strip() if ip else None
        win._draw_bg_last_rgba = None
        if win._draw_color_ip:
            _start_dynamic_bg_timer(win)

        self._windows[tag] = win
        return win

    def set_style(self, tag: str, css: str) -> None:
        """Apply a direct text-based CSS / QSS stylesheet to the window."""
        win = self.get(tag)
        win.setStyleSheet(css)

    def set_css(self, tag: str, css: str) -> None:
        """Alias for set_style. Apply direct text-based CSS / QSS styling."""
        self.set_style(tag, css)

    def set_icon(self, tag: str, icon_path: str) -> None:
        """Set the window icon from a file path."""
        win = self.get(tag)
        win.setWindowIcon(QIcon(icon_path))

    def set_transparency(self, tag: str, transparency: int) -> None:
        """Set window transparency (0 = invisible, 100 = fully opaque)."""
        if not 0 <= transparency <= 100:
            raise ValueError("Draw.window: 'transparency' must be between 0 and 100.")
        win = self.get(tag)
        win.setWindowOpacity(transparency / 100.0)

    def maximize(self, tag: str = "main") -> None:
        """Maximize the window."""
        self.get(tag).showMaximized()

    def minimize(self, tag: str = "main") -> None:
        """Minimize the window."""
        self.get(tag).showMinimized()

    def fullscreen(self, tag: str = "main") -> None:
        """Set window to full screen."""
        self.get(tag).showFullScreen()

    def restore(self, tag: str = "main") -> None:
        """Restore window to normal state (un-maximize / un-fullscreen)."""
        self.get(tag).showNormal()

    def is_maximized(self, tag: str = "main") -> bool:
        """Return True if window is maximized."""
        return self.get(tag).isMaximized()

    def is_minimized(self, tag: str = "main") -> bool:
        """Return True if window is minimized."""
        return self.get(tag).isMinimized()

    def is_fullscreen(self, tag: str = "main") -> bool:
        """Return True if window is full screen."""
        return self.get(tag).isFullScreen()

    def set_menu(self, tag: str, schema: list[dict[str, list[Union[str, Tuple[str, Callable]]]]]) -> QMenuBar:
        """
        Build a native window menu bar from a declarative schema.

        Schema format example:
            [
                {
                    "File": [
                        ("New", on_new_func),
                        ("Open", on_open_func),
                        "---",  # Separator
                        ("Exit", Draw.quit)
                    ]
                },
                {
                    "Help": [
                        ("About", on_about_func)
                    ]
                }
            ]
        """
        win = self.get(tag)
        menubar = win.menuBar()
        menubar.clear()

        for menu_dict in schema:
            for title, items in menu_dict.items():
                qmenu = menubar.addMenu(title)
                for item in items:
                    if item == "---":
                        qmenu.addSeparator()
                    elif isinstance(item, tuple) and len(item) == 2:
                        label, callback = item
                        action = QAction(label, win)
                        action.triggered.connect(callback)
                        qmenu.addAction(action)
                    elif isinstance(item, str):
                        qmenu.addAction(item)
        return menubar

    def set_status(self, tag: str, message: str, timeout: int = 0) -> None:
        """Display a message in the window's status bar."""
        win = self.get(tag)
        win.statusBar().showMessage(message, timeout)

    def show(self, tag: str) -> None:
        """Show the window with the given tag."""
        self.get(tag).show()

    def get_canvas(self, tag: str):
        """
        Return the shared _DrawCanvas widget for this window, creating it
        if it doesn't exist yet.
        """
        from Draw._text import _get_or_create_canvas
        win = self.get(tag)
        return _get_or_create_canvas(tag, win)

    def hide(self, tag: str) -> None:
        """Hide the window with the given tag."""
        self.get(tag).hide()

    def close(self, tag: str) -> bool:
        """
        Ask Qt to close a window.

        Returns True when the window accepted the close event.
        """
        win = self.get(tag)
        return bool(win.close())

    def close_all(self) -> None:
        """Close every Draw window and stop the app if no windows remain."""
        for tag in list(self._windows):
            self.close(tag)

        if not self._windows:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def request_shutdown(self, reason: str = "") -> None:
        """
        Thread-safe emergency shutdown: close every Draw window and quit.

        Unlike close_all(), this is safe to call from *any* thread,
        including Draw.debug's watchdog daemon thread. Qt widgets may only
        be touched on the GUI thread that owns them, so this never closes
        windows directly — it hands the request to a queued Qt
        signal/slot connection (_ShutdownDispatcher), which Qt marshals
        onto the GUI thread's event loop automatically.
        """
        if self._shutdown_dispatcher is None:
            # No window/app has been set up on the GUI thread yet, so
            # there is nothing to shut down.
            return
        self._shutdown_dispatcher.post(reason)

    def quit(self) -> None:
        """
        Quit the Draw app cleanly.
        """
        _stop_connectors()
        for win in list(self._windows.values()):
            win.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def get(self, tag: str) -> QMainWindow:
        """Return the QMainWindow for *tag*, or raise KeyError."""
        tag = str(tag).strip()
        if tag not in self._windows:
            raise KeyError(f"Draw.window: no window with tag='{tag}'.")
        return self._windows[tag]

    def run(self, tag: Optional[str] = None) -> int:
        """
        Show window(s), start the event loop, and return Qt's exit code.
        """
        app = get_app()
        self._connect_shutdown_hook(app)

        if tag is not None:
            self.show(tag)
        else:
            for win in list(self._windows.values()):
                if isinstance(win, QMainWindow):   # skip embedded panels
                    win.show()

        return int(app.exec())

    def list_tags(self) -> list[str]:
        """Return tags of real QMainWindow instances only (excludes panels)."""
        return [t for t, w in self._windows.items() if isinstance(w, QMainWindow)]

    def list_all_tags(self) -> list[str]:
        """Return ALL registered tags including panels."""
        return list(self._windows.keys())

    def _forget(self, tag: str) -> None:
        win = self._windows.pop(tag, None)
        if win is not None:
            try:
                from Draw._connectors import connectors as _connector_registry
                ips: set = set()
                canvas = getattr(win, "_draw_canvas", None)
                if canvas is not None:
                    ips |= {s.ip for s in getattr(canvas, "shape_items", []) if s.ip}
                    ips |= {t.ip for t in getattr(canvas, "text_items", []) if t.ip}
                _connector_registry.cleanup_window(tag, ips)
            except Exception:
                pass

        # If no real QMainWindow remains open, quit Qt event loop cleanly
        remaining = [w for w in self._windows.values() if isinstance(w, QMainWindow)]
        if not remaining:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def _connect_shutdown_hook(self, app: QApplication) -> None:
        # Always called on the Qt GUI thread (from __call__/run(), Draw's
        # documented entry points), so this is a safe place to construct
        # the shutdown dispatcher with correct GUI-thread affinity.
        if self._shutdown_dispatcher is None:
            self._shutdown_dispatcher = _ShutdownDispatcher(self)

        if self._shutdown_connected:
            return
        app.aboutToQuit.connect(_stop_connectors)
        app.aboutToQuit.connect(self._windows.clear)
        self._shutdown_connected = True


window = _WindowRegistry()
