"""
Draw._screen
============
Live display surface for Draw (host-provided pixel buffer & interaction stream).

Draw.screen is a lightweight display surface.
It does NOT render shapes or perform graphics calculations.

Its only responsibilities are:
- Display pixels provided by the host (images, video frames, camera frames,
  3D viewport output, AI-generated frames, OpenGL/Vulkan framebuffers, pixel buffers).
- Receive user interaction (mouse, touch, stylus, keyboard, wheel, drag).
- Return interaction data to the host (relative coordinates, drawing mode streams).

Philosophy:
Draw.screen is only a window into data.
It never owns the data.
It never renders objects.
It never performs drawing.
It simply displays frames and reports user interaction.
"""

from __future__ import annotations

import os
import sys
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# pyrefly: ignore [missing-import]
from PySide6.QtCore import (
    QCoreApplication, QEvent, QMetaObject, QObject, QPoint, QPointF,
    QRect, QRectF, QSize, Qt, QTimer, Signal, Slot,
)
# pyrefly: ignore [missing-import]
from PySide6.QtGui import (
    QBrush, QColor, QCursor, QImage, QKeyEvent, QMouseEvent,
    QPainter, QPixmap, QTabletEvent, QTouchEvent, QWheelEvent,
)
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QMainWindow, QWidget

from Draw._app import get_app
from Draw._window import window as _window_registry, _ALIGN_VALUES
from Draw._align import calculate_alignment_pos

_logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Frame Conversion Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _to_qimage(frame: Any, target_size: Optional[Tuple[int, int]] = None) -> Optional[QImage]:
    """
    Convert any host frame/pixel format into a PySide6 QImage.
    Supports:
      - QImage (direct / copy)
      - QPixmap (.toImage())
      - numpy.ndarray (2D grayscale, 3D RGB/BGR/RGBA/BGRA, uint8/float32)
      - PIL.Image.Image
      - bytes / bytearray / memoryview (compressed PNG/JPEG/WebP or raw buffer)
      - file path (str / os.PathLike)
      - nested lists / tuples of RGB values
      - None (clears the display)
    """
    if frame is None:
        return None

    if isinstance(frame, QImage):
        return frame

    if isinstance(frame, QPixmap):
        return frame.toImage()

    # File path / string path
    if isinstance(frame, (str, os.PathLike)):
        path_str = str(frame)
        if os.path.exists(path_str):
            img = QImage(path_str)
            if not img.isNull():
                return img

    # PIL Image
    if hasattr(frame, "__array_interface__") or hasattr(frame, "tobytes"):
        # Check for PIL Image
        if hasattr(frame, "mode") and hasattr(frame, "size"):
            try:
                mode = frame.mode
                w, h = frame.size
                if mode == "RGBA":
                    data = frame.tobytes("raw", "RGBA")
                    img = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888)
                    return img.copy()
                elif mode == "RGB":
                    data = frame.tobytes("raw", "RGB")
                    img = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
                    return img.copy()
                elif mode == "L":
                    data = frame.tobytes("raw", "L")
                    img = QImage(data, w, h, w, QImage.Format.Format_Grayscale8)
                    return img.copy()
                else:
                    rgba = frame.convert("RGBA")
                    data = rgba.tobytes("raw", "RGBA")
                    img = QImage(data, w, h, w * 4, QImage.Format.Format_RGBA8888)
                    return img.copy()
            except Exception as exc:
                _logger.debug("PIL conversion fallback: %s", exc)

    # Numpy array / array-like
    if hasattr(frame, "shape") and hasattr(frame, "dtype") and hasattr(frame, "data"):
        try:
            arr = frame
            # Handle float arrays (0.0 to 1.0 or 0.0 to 255.0)
            if "float" in str(arr.dtype):
                import numpy as np  # type: ignore[import-not-found]
                max_val = float(arr.max()) if arr.size > 0 else 1.0
                scale = 255.0 if max_val <= 1.01 else 1.0
                arr = (arr * scale).clip(0, 255).astype(np.uint8)
            elif str(arr.dtype) != "uint8":
                import numpy as np  # type: ignore[import-not-found]
                arr = arr.astype(np.uint8)

            shape = arr.shape
            if len(shape) == 2:
                # 2D Grayscale: (H, W)
                h, w = shape
                bytes_per_line = int(arr.strides[0]) if hasattr(arr, "strides") else w
                img = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                return img.copy()
            elif len(shape) == 3:
                h, w, channels = shape
                bytes_per_line = int(arr.strides[0]) if hasattr(arr, "strides") else w * channels
                if channels == 4:
                    # RGBA or BGRA
                    img = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
                    return img.copy()
                elif channels == 3:
                    # RGB888
                    img = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    return img.copy()
                elif channels == 1:
                    img = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                    return img.copy()
        except Exception as exc:
            _logger.debug("Numpy conversion fallback: %s", exc)

    # Raw bytes / bytearray / memoryview
    if isinstance(frame, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(frame)
        # Try loading as encoded image (PNG, JPEG, WebP, BMP, GIF)
        img = QImage.fromData(raw_bytes)
        if not img.isNull():
            return img

        # If raw pixel buffer and target size given, construct image directly
        if target_size and len(target_size) == 2:
            tw, th = int(target_size[0]), int(target_size[1])
            total_bytes = len(raw_bytes)
            if tw > 0 and th > 0:
                if total_bytes == tw * th * 4:
                    return QImage(raw_bytes, tw, th, tw * 4, QImage.Format.Format_RGBA8888).copy()
                elif total_bytes == tw * th * 3:
                    return QImage(raw_bytes, tw, th, tw * 3, QImage.Format.Format_RGB888).copy()
                elif total_bytes == tw * th:
                    return QImage(raw_bytes, tw, th, tw, QImage.Format.Format_Grayscale8).copy()

    # Nested list of pixels
    if isinstance(frame, (list, tuple)) and frame:
        try:
            import numpy as np  # type: ignore[import-not-found]
            arr = np.array(frame, dtype=np.uint8)
            return _to_qimage(arr, target_size=target_size)
        except Exception:
            pass

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Thread-Safe Dispatcher for Background Host Feeds (OpenCV, AI, Camera, etc.)
# ══════════════════════════════════════════════════════════════════════════════

class _ScreenDispatcher(QObject):
    """Dispatches frame updates safely from background worker threads to Qt."""
    _update_signal: Signal = Signal(object, object)  # (ScreenSurface, frame)
    _clear_signal: Signal = Signal(object)           # (ScreenSurface)
    _resize_signal: Signal = Signal(object, int, int)# (ScreenSurface, w, h)

    def __init__(self) -> None:
        super().__init__()
        self._update_signal.connect(self._do_update, Qt.ConnectionType.QueuedConnection)
        self._clear_signal.connect(self._do_clear, Qt.ConnectionType.QueuedConnection)
        self._resize_signal.connect(self._do_resize, Qt.ConnectionType.QueuedConnection)

    @Slot(object, object)
    def _do_update(self, screen_surface: Any, frame: Any) -> None:
        try:
            screen_surface._apply_frame_main_thread(frame)
        except Exception as exc:
            _logger.warning("Draw.screen update error: %s", exc)

    @Slot(object)
    def _do_clear(self, screen_surface: Any) -> None:
        try:
            screen_surface._apply_clear_main_thread()
        except Exception as exc:
            _logger.warning("Draw.screen clear error: %s", exc)

    @Slot(object, int, int)
    def _do_resize(self, screen_surface: Any, w: int, h: int) -> None:
        try:
            screen_surface._apply_resize_main_thread(w, h)
        except Exception as exc:
            _logger.warning("Draw.screen resize error: %s", exc)


_dispatcher: Optional[_ScreenDispatcher] = None


def _get_dispatcher() -> _ScreenDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = _ScreenDispatcher()
    return _dispatcher


# ══════════════════════════════════════════════════════════════════════════════
#  ScreenSurface: Display Surface Widget & Host Interaction Interface
# ══════════════════════════════════════════════════════════════════════════════

class ScreenSurface(QWidget):
    """
    Lightweight display surface widget embedded in a Draw window or canvas.
    Displays host-provided pixel frames and streams user interaction back to host.
    """

    def __init__(
        self,
        *,
        ip: str,
        display: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        x: Optional[int] = None,
        y: Optional[int] = None,
        align: str = "center",
        source: Any = None,
        senses: Any = None,
        drawing_mode: bool = False,
        background_color: Optional[Union[str, QColor]] = "#000000",
        scaling: str = "fit",
        parent: Optional[QWidget] = None,
        **callbacks: Any,
    ) -> None:
        super().__init__(parent)
        self.ip: str = str(ip or "")
        self.display: Optional[str] = display
        self._width: int = max(1, int(width))
        self._height: int = max(1, int(height))
        self._x: Optional[int] = x
        self._y: Optional[int] = y
        self.align: str = align or "center"
        self._senses: Any = senses
        self.drawing_mode: bool = bool(drawing_mode)
        self.scaling: str = str(scaling or "fit").lower()

        # Appearance & frame state
        self._bg_color: Optional[QColor] = QColor(background_color) if background_color else None
        self._current_frame: Optional[QImage] = None
        self._frame_lock = threading.Lock()

        # Coordinate reporting tracking
        self.mouse_x: float = 0.0
        self.mouse_y: float = 0.0
        self._is_mouse_down: bool = False
        self._last_button: str = ""
        self._is_hovered: bool = False

        # Host Event Callbacks
        self.on_mouse_move: Optional[Callable] = callbacks.get("on_mouse_move")
        self.on_mouse_down: Optional[Callable] = callbacks.get("on_mouse_down")
        self.on_mouse_up: Optional[Callable] = callbacks.get("on_mouse_up")
        self.on_wheel: Optional[Callable] = callbacks.get("on_wheel")
        self.on_touch: Optional[Callable] = callbacks.get("on_touch")
        self.on_stylus: Optional[Callable] = callbacks.get("on_stylus")
        self.on_enter: Optional[Callable] = callbacks.get("on_enter")
        self.on_leave: Optional[Callable] = callbacks.get("on_leave")
        self.on_resize: Optional[Callable] = callbacks.get("on_resize")
        self.on_click: Optional[Callable] = callbacks.get("on_click")
        self.on_drag: Optional[Callable] = callbacks.get("on_drag")
        self.on_key: Optional[Callable] = callbacks.get("on_key", callbacks.get("on_key_press"))

        # Drawing mode callbacks (optional host functions)
        self.mouse_down_callback: Optional[Callable] = callbacks.get("mouse_down")
        self.mouse_move_callback: Optional[Callable] = callbacks.get("mouse_move")
        self.mouse_up_callback: Optional[Callable] = callbacks.get("mouse_up")

        # Custom event listeners registry
        self._event_listeners: Dict[str, List[Callable]] = {}

        # Qt Widget attributes & interaction flags
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedSize(self._width, self._height)

        if source is not None:
            self.update(source)

    # ── Properties & Coordinate Access ───────────────────────────────────────

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def size(self) -> List[int]:
        return [self._width, self._height]

    @property
    def x(self) -> float:
        return float(self.pos().x())

    @property
    def y(self) -> float:
        return float(self.pos().y())

    @property
    def last_x(self) -> float:
        """Last reported cursor X coordinate relative to screen."""
        return self.mouse_x

    @property
    def last_y(self) -> float:
        """Last reported cursor Y coordinate relative to screen."""
        return self.mouse_y

    @property
    def is_drawing_mode(self) -> bool:
        return self.drawing_mode

    def enable_drawing_mode(self, enabled: bool = True) -> None:
        """Toggle drawing mode reporting (mouse_down, mouse_move, mouse_up)."""
        self.drawing_mode = bool(enabled)

    # ── Host Functions ───────────────────────────────────────────────────────

    def update(self, frame: Any = None) -> None:
        """
        Replace current frame on the display.
        Thread-safe: Can be called from any background thread (camera, video, AI).
        """
        app = QCoreApplication.instance()
        current_thread = threading.current_thread()
        main_thread = threading.main_thread()

        if app is not None and current_thread == main_thread:
            self._apply_frame_main_thread(frame)
        else:
            dispatcher = _get_dispatcher()
            dispatcher._update_signal.emit(self, frame)

    def _apply_frame_main_thread(self, frame: Any) -> None:
        """Internal main-thread frame ingestion and repaint trigger."""
        qimg = _to_qimage(frame, target_size=(self._width, self._height))
        with self._frame_lock:
            self._current_frame = qimg
        super().update()

    def clear(self) -> None:
        """Clear display content."""
        app = QCoreApplication.instance()
        if app is not None and threading.current_thread() == threading.main_thread():
            self._apply_clear_main_thread()
        else:
            dispatcher = _get_dispatcher()
            dispatcher._clear_signal.emit(self)

    def _apply_clear_main_thread(self) -> None:
        with self._frame_lock:
            self._current_frame = None
        super().update()

    def capture(self) -> Optional[QImage]:
        """Capture current display frame as a QImage."""
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()

        # If no frame yet, capture current widget surface
        pixmap = self.grab()
        return pixmap.toImage()

    def resize(self, *args: Any, **kwargs: Any) -> None:
        """
        Resize display.
        Usage:
            screen.resize(640, 480)
            screen.resize([640, 480])
            screen.resize(size=[640, 480])
            screen.resize(width=640, height=480)
        """
        w, h = self._width, self._height
        if len(args) == 1:
            val = args[0]
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                w, h = int(val[0]), int(val[1])
            elif isinstance(val, (int, float)):
                w = h = int(val)
        elif len(args) >= 2:
            w, h = int(args[0]), int(args[1])

        if "size" in kwargs:
            val = kwargs["size"]
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                w, h = int(val[0]), int(val[1])
        if "width" in kwargs:
            w = int(kwargs["width"])
        if "height" in kwargs:
            h = int(kwargs["height"])

        w = max(1, w)
        h = max(1, h)

        app = QCoreApplication.instance()
        if app is not None and threading.current_thread() == threading.main_thread():
            self._apply_resize_main_thread(w, h)
        else:
            dispatcher = _get_dispatcher()
            dispatcher._resize_signal.emit(self, w, h)

    def _apply_resize_main_thread(self, w: int, h: int) -> None:
        self._width = w
        self._height = h
        self.setFixedSize(w, h)
        self._reposition_in_parent()
        self._emit_event("on_resize", w, h)
        if self.on_resize:
            try:
                self.on_resize(w, h)
            except TypeError:
                self.on_resize((w, h))

    def _reposition_in_parent(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        pw = parent.width()
        ph = parent.height()
        if self._x is not None and self._y is not None:
            self.move(int(self._x), int(self._y))
        elif self.align:
            ax, ay = calculate_alignment_pos(
                self.align, float(self._width), float(self._height),
                float(pw), float(ph)
            )
            self.move(int(ax), int(ay))

    # ── Event Subscription API ───────────────────────────────────────────────

    def on(self, event_name: str, callback: Optional[Callable] = None) -> Any:
        """
        Register an event listener or use as a decorator.
        Examples:
            screen.on("mouse_move", my_callback)

            @screen.on("mouse_down")
            def on_down(x, y, button, event):
                ...
        """
        norm = event_name.strip().lower()
        if not norm.startswith("on_") and norm not in ("mouse_down", "mouse_move", "mouse_up"):
            norm_key = f"on_{norm}"
        else:
            norm_key = norm

        def decorator(fn: Callable) -> Callable:
            self._event_listeners.setdefault(norm_key, []).append(fn)
            return fn

        if callback is not None:
            return decorator(callback)
        return decorator

    def _emit_event(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        listeners = list(self._event_listeners.get(event_name, []))
        prop_fn = getattr(self, event_name, None)
        if prop_fn is not None and callable(prop_fn) and prop_fn not in listeners:
            listeners.append(prop_fn)

        for fn in listeners:
            try:
                fn(*args, **kwargs)
            except TypeError:
                try:
                    fn(*args)
                except Exception as exc:
                    _logger.debug("Event %s listener error: %s", event_name, exc)
            except Exception as exc:
                _logger.debug("Event %s listener error: %s", event_name, exc)

    def _dispatch_senses(self, sense_type: str, button: Optional[str] = None, meta: Optional[dict] = None) -> None:
        if not self.ip:
            return
        try:
            from Draw._connectors import senses as _senses_registry
            _senses_registry.dispatch_mouse_event(sense_type, self.ip, button, meta=meta or {})
        except Exception:
            pass

    # ── Paint Event ──────────────────────────────────────────────────────────

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            # Paint background if configured
            if self._bg_color is not None:
                painter.fillRect(self.rect(), QBrush(self._bg_color))

            with self._frame_lock:
                frame = self._current_frame

            if frame is not None and not frame.isNull():
                target_rect = self.rect()
                fw, fh = frame.width(), frame.height()

                if self.scaling == "stretch":
                    painter.drawImage(target_rect, frame)
                elif self.scaling == "center":
                    cx = (self._width - fw) // 2
                    cy = (self._height - fh) // 2
                    painter.drawImage(QPoint(cx, cy), frame)
                elif self.scaling == "none":
                    painter.drawImage(QPoint(0, 0), frame)
                else:
                    # "fit" (aspect-ratio preserving)
                    scaled = frame.scaled(
                        self.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    cx = (self._width - scaled.width()) // 2
                    cy = (self._height - scaled.height()) // 2
                    painter.drawImage(QPoint(cx, cy), scaled)
        finally:
            if painter.isActive():
                painter.end()

    # ── Interaction & Input Events ───────────────────────────────────────────

    @staticmethod
    def _button_name(btn: Qt.MouseButton) -> str:
        if btn == Qt.MouseButton.LeftButton:
            return "left"
        elif btn == Qt.MouseButton.RightButton:
            return "right"
        elif btn == Qt.MouseButton.MiddleButton:
            return "middle"
        return "left"

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        rx, ry = float(pos.x()), float(pos.y())
        self.mouse_x = rx
        self.mouse_y = ry
        self._is_mouse_down = True
        btn_name = self._button_name(event.button())
        self._last_button = btn_name

        # Coordinate Output: reports x, y relative to screen
        # 1. Forward to drawing mode host functions if enabled
        if self.drawing_mode:
            if self.mouse_down_callback:
                try:
                    self.mouse_down_callback(rx, ry, btn_name)
                except TypeError:
                    self.mouse_down_callback(rx, ry)
            self._emit_event("mouse_down", rx, ry, btn_name)

        # 2. Host Event Forwarding: on_mouse_down
        self._emit_event("on_mouse_down", rx, ry, btn_name, event)
        if self.on_mouse_down:
            try:
                self.on_mouse_down(rx, ry, btn_name, event)
            except TypeError:
                try:
                    self.on_mouse_down(rx, ry, btn_name)
                except TypeError:
                    self.on_mouse_down(rx, ry)

        # 3. Sense & connector dispatches
        self._dispatch_senses("mouse_press", btn_name, {"x": rx, "y": ry, "button": btn_name})
        self._dispatch_senses("mouse_click", btn_name, {"x": rx, "y": ry, "button": btn_name})
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        rx, ry = float(pos.x()), float(pos.y())
        prev_x, prev_y = self.mouse_x, self.mouse_y
        self.mouse_x = rx
        self.mouse_y = ry

        # Coordinate Output: reports x, y relative to screen
        # 1. Forward to drawing mode host functions if enabled
        if self.drawing_mode:
            if self.mouse_move_callback:
                try:
                    self.mouse_move_callback(rx, ry, self._last_button)
                except TypeError:
                    self.mouse_move_callback(rx, ry)
            self._emit_event("mouse_move", rx, ry, self._last_button)

        # 2. Host Event Forwarding: on_mouse_move
        self._emit_event("on_mouse_move", rx, ry, event)
        if self.on_mouse_move:
            try:
                self.on_mouse_move(rx, ry, event)
            except TypeError:
                self.on_mouse_move(rx, ry)

        # 3. Mouse Drag reporting
        if self._is_mouse_down:
            dx, dy = rx - prev_x, ry - prev_y
            self._emit_event("on_drag", rx, ry, dx, dy, self._last_button)
            if self.on_drag:
                try:
                    self.on_drag(rx, ry, dx, dy, self._last_button)
                except TypeError:
                    self.on_drag(rx, ry, dx, dy)
            self._dispatch_senses("drag_move", self._last_button, {"x": rx, "y": ry, "dx": dx, "dy": dy})

        # 4. Senses hover
        self._dispatch_senses("mouse_hover", None, {"x": rx, "y": ry})
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        rx, ry = float(pos.x()), float(pos.y())
        self.mouse_x = rx
        self.mouse_y = ry
        self._is_mouse_down = False
        btn_name = self._button_name(event.button())

        # Coordinate Output: reports x, y relative to screen
        # 1. Forward to drawing mode host functions if enabled
        if self.drawing_mode:
            if self.mouse_up_callback:
                try:
                    self.mouse_up_callback(rx, ry, btn_name)
                except TypeError:
                    self.mouse_up_callback(rx, ry)
            self._emit_event("mouse_up", rx, ry, btn_name)

        # 2. Host Event Forwarding: on_mouse_up, on_click
        self._emit_event("on_mouse_up", rx, ry, btn_name, event)
        if self.on_mouse_up:
            try:
                self.on_mouse_up(rx, ry, btn_name, event)
            except TypeError:
                try:
                    self.on_mouse_up(rx, ry, btn_name)
                except TypeError:
                    self.on_mouse_up(rx, ry)

        self._emit_event("on_click", rx, ry, btn_name)
        if self.on_click:
            try:
                self.on_click(rx, ry, btn_name)
            except TypeError:
                self.on_click(rx, ry)

        # 3. Sense dispatches
        self._dispatch_senses("mouse_release", btn_name, {"x": rx, "y": ry, "button": btn_name})
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        rx, ry = float(pos.x()), float(pos.y())
        delta_x = event.angleDelta().x()
        delta_y = event.angleDelta().y()

        self._emit_event("on_wheel", delta_x, delta_y, rx, ry, event)
        if self.on_wheel:
            try:
                self.on_wheel(delta_x, delta_y, rx, ry, event)
            except TypeError:
                try:
                    self.on_wheel(delta_y, rx, ry)
                except TypeError:
                    self.on_wheel(delta_y)

        sense_type = "mouse_scroll_up" if delta_y > 0 else "mouse_scroll_down"
        self._dispatch_senses(sense_type, None, {"delta_x": delta_x, "delta_y": delta_y, "x": rx, "y": ry})
        event.accept()

    def enterEvent(self, event: Any) -> None:
        self._is_hovered = True
        self._emit_event("on_enter", event)
        if self.on_enter:
            try:
                self.on_enter(event)
            except TypeError:
                self.on_enter()
        self._dispatch_senses("mouse_hover", None, {"hover": True})
        try:
            super().enterEvent(event)
        except Exception:
            pass

    def leaveEvent(self, event: Any) -> None:
        self._is_hovered = False
        self._emit_event("on_leave", event)
        if self.on_leave:
            try:
                self.on_leave(event)
            except TypeError:
                self.on_leave()
        self._dispatch_senses("mouse_leave", None, {"hover": False})
        try:
            super().leaveEvent(event)
        except Exception:
            pass

    def touchEvent(self, event: QTouchEvent) -> None:
        points = event.points() if hasattr(event, "points") else event.touchPoints()
        point_data = []
        for tp in points:
            tpos = tp.position() if hasattr(tp, "position") else tp.pos()
            tid = tp.id() if hasattr(tp, "id") else 0
            state = str(tp.state()) if hasattr(tp, "state") else ""
            point_data.append({
                "id": tid,
                "x": float(tpos.x()),
                "y": float(tpos.y()),
                "state": state,
            })

        self._emit_event("on_touch", point_data, event)
        if self.on_touch:
            try:
                self.on_touch(point_data, event)
            except TypeError:
                self.on_touch(point_data)

        if point_data:
            first = point_data[0]
            self.mouse_x = first["x"]
            self.mouse_y = first["y"]
            self._dispatch_senses("touch_start", None, {"points": point_data})
        event.accept()

    def tabletEvent(self, event: QTabletEvent) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        rx, ry = float(pos.x()), float(pos.y())
        pressure = event.pressure()
        tilt_x = event.xTilt()
        tilt_y = event.yTilt()

        self._emit_event("on_stylus", pressure, rx, ry, tilt_x, tilt_y, event)
        if self.on_stylus:
            try:
                self.on_stylus(pressure, rx, ry, tilt_x, tilt_y, event)
            except TypeError:
                try:
                    self.on_stylus(pressure, rx, ry)
                except TypeError:
                    self.on_stylus(pressure)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key_text = event.text()
        key_code = event.key()
        self._emit_event("on_key", key_text, key_code, event)
        if self.on_key:
            try:
                self.on_key(key_text, key_code, event)
            except TypeError:
                try:
                    self.on_key(key_text)
                except TypeError:
                    self.on_key(key_code)
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  _ScreenRegistry: Exposed as Draw.screen
# ══════════════════════════════════════════════════════════════════════════════

class _ScreenRegistry:
    """
    Singleton manager and factory for Draw.screen.

    API:
        Draw.screen(
            ip="",
            size=[640, 480],
            align="center",
            source=None,
            senses=None,
        )
    """

    def __init__(self) -> None:
        self._screens: Dict[str, ScreenSurface] = {}

    def __call__(
        self,
        ip: str = "",
        *,
        size: Union[List[int], Tuple[int, int], int] = (640, 480),
        align: str = "center",
        source: Any = None,
        senses: Any = None,
        display: Optional[str] = None,
        tag: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        drawing_mode: bool = False,
        background_color: Optional[Union[str, QColor]] = "#000000",
        scaling: str = "fit",
        **callbacks: Any,
    ) -> ScreenSurface:
        get_app()

        screen_ip = str(ip or "screen_main")
        win_tag = display or tag

        # Resolve target window
        if win_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                win_tag = tags[0]
            elif len(tags) > 1:
                win_tag = tags[0]
            else:
                # Auto-create default window if none exist
                win_tag = "main"
                _window_registry(tag=win_tag, width=max(800, size[0] if isinstance(size, (list, tuple)) else 800), height=max(600, size[1] if isinstance(size, (list, tuple)) else 600))

        win: QMainWindow = _window_registry.get(win_tag)

        # Parse width and height
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            w, h = int(size[0]), int(size[1])
        elif isinstance(size, (int, float)):
            w = h = int(size)
        else:
            w, h = 640, 480

        # Check if screen with this IP already exists; reuse and reconfigure if so
        if screen_ip in self._screens:
            existing = self._screens[screen_ip]
            if existing.parentWidget() == win:
                if (w, h) != (existing.width, existing.height):
                    existing.resize(w, h)
                if source is not None:
                    existing.update(source)
                return existing

        # Create new ScreenSurface widget
        surface = ScreenSurface(
            ip=screen_ip,
            display=win_tag,
            width=w,
            height=h,
            x=x,
            y=y,
            align=align,
            source=source,
            senses=senses,
            drawing_mode=drawing_mode,
            background_color=background_color,
            scaling=scaling,
            parent=win,
            **callbacks,
        )

        # Position surface on window
        if x is not None and y is not None:
            surface.move(int(x), int(y))
        else:
            ww = win.width()
            wh = win.height()
            ax, ay = calculate_alignment_pos(align or "center", float(w), float(h), float(ww), float(wh))
            surface.move(int(ax), int(ay))

        surface.show()
        surface.raise_()

        self._screens[screen_ip] = surface
        return surface

    def get(self, ip: str) -> Optional[ScreenSurface]:
        """Return the ScreenSurface registered under *ip*, or None."""
        return self._screens.get(str(ip))

    def __getitem__(self, ip: str) -> ScreenSurface:
        screen_surface = self.get(ip)
        if screen_surface is None:
            raise KeyError(f"Draw.screen: no screen with ip '{ip}' exists.")
        return screen_surface

    def list(self) -> List[str]:
        """Return list of all registered screen IPs."""
        return list(self._screens.keys())

    def update(self, ip: str, frame: Any) -> None:
        """Update the frame for the screen with the given *ip*."""
        screen_surface = self.get(ip)
        if screen_surface is not None:
            screen_surface.update(frame)

    def clear(self, ip: Optional[str] = None) -> None:
        """Clear screen with given *ip* or all screens if ip is None."""
        if ip is not None:
            screen_surface = self.get(ip)
            if screen_surface is not None:
                screen_surface.clear()
        else:
            for s in self._screens.values():
                s.clear()

    def capture(self, ip: str) -> Optional[QImage]:
        """Capture current frame of screen with given *ip*."""
        screen_surface = self.get(ip)
        if screen_surface is not None:
            return screen_surface.capture()
        return None

    def resize(self, ip: str, size: Union[List[int], Tuple[int, int]]) -> None:
        """Resize screen with given *ip*."""
        screen_surface = self.get(ip)
        if screen_surface is not None:
            screen_surface.resize(size)

    def remove(self, ip: str) -> bool:
        """Remove and close the screen surface for *ip*."""
        screen_surface = self._screens.pop(str(ip), None)
        if screen_surface is not None:
            screen_surface.hide()
            screen_surface.deleteLater()
            return True
        return False


screen = _ScreenRegistry()
