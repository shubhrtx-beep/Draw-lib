"""Shared, dependency-light Qt utilities for Draw.

This module is Draw's lowest-level convenience layer.  It exposes Qt value
types and small, side-effect-free helpers that are useful to several Draw
subsystems (especially connection geometry).  It deliberately does not know
about Draw widgets, scenes, layouts, colours, cursors, or rendering.

Qt Multimedia and Qt Widgets are imported only by the helper that needs them,
so importing :mod:`Draw._tools` remains cheap and safe for headless geometry
and resource work.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, TypeVar, Union, overload

from PySide6.QtCore import (
    QDateTime,
    QElapsedTimer,
    QLine,
    QLineF,
    QMargins,
    QMarginsF,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    QSizeF,
    QStandardPaths,
    QSysInfo,
    QTimer,
    QUrl,
)
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap, QPolygon, QPolygonF


PointLike = Union[QPoint, QPointF, tuple[float, float], list[float]]
RectLike = Union[QRect, QRectF]
PolygonLike = Union[QPolygon, QPolygonF, Iterable[PointLike]]
_T = TypeVar("_T")


# ============================================================================
# Qt Geometry -- common connection primitives, not layout policy
# ============================================================================

def _pointf(value: PointLike) -> QPointF:
    """Coerce a point-like value to ``QPointF`` for internal calculations."""
    if isinstance(value, QPointF):
        return QPointF(value)
    if isinstance(value, QPoint):
        return QPointF(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return QPointF(float(value[0]), float(value[1]))
    raise TypeError("expected QPoint, QPointF, or a two-item numeric sequence")


def _rectf(value: RectLike) -> QRectF:
    """Return a normalized floating-point rectangle."""
    return QRectF(value).normalized()


def center(value: Union[RectLike, QLine, QLineF, QPolygon, QPolygonF]) -> QPointF:
    """Return the geometric centre of a rectangle, line, or polygon bounds."""
    if isinstance(value, (QRect, QRectF)):
        return _rectf(value).center()
    if isinstance(value, (QLine, QLineF)):
        return midpoint(value.p1(), value.p2())
    if isinstance(value, (QPolygon, QPolygonF)):
        return polygon_center(value)
    raise TypeError("center() accepts QRect, QLine, or QPolygon values")


def edge_center(rect: RectLike, edge: Literal["top", "bottom", "left", "right"]) -> QPointF:
    """Return the centre point of the named normalized rectangle edge."""
    box = _rectf(rect)
    key = edge.lower().replace("_", "-")
    if key == "top":
        return QPointF(box.center().x(), box.top())
    if key == "bottom":
        return QPointF(box.center().x(), box.bottom())
    if key == "left":
        return QPointF(box.left(), box.center().y())
    if key == "right":
        return QPointF(box.right(), box.center().y())
    raise ValueError("edge must be 'top', 'bottom', 'left', or 'right'")


def corner(rect: RectLike, name: str) -> QPointF:
    """Return a named rectangle corner (``top-left``, ``top-right``, etc.)."""
    box = _rectf(rect)
    key = name.lower().replace("_", "-")
    corners = {"top-left": box.topLeft, "top-right": box.topRight,
               "bottom-left": box.bottomLeft, "bottom-right": box.bottomRight}
    try:
        return corners[key]()
    except KeyError as exc:
        raise ValueError("corner must be top-left, top-right, bottom-left, or bottom-right") from exc


def midpoint(first: PointLike, second: PointLike) -> QPointF:
    """Return the point halfway between two points."""
    a, b = _pointf(first), _pointf(second)
    return QPointF((a.x() + b.x()) / 2.0, (a.y() + b.y()) / 2.0)


def nearest_edge(point: PointLike, rect: RectLike) -> str:
    """Return the rectangle edge nearest to ``point`` as ``top/bottom/left/right``."""
    p, box = _pointf(point), _rectf(rect)
    distances = {"top": abs(p.y() - box.top()), "bottom": abs(p.y() - box.bottom()),
                 "left": abs(p.x() - box.left()), "right": abs(p.x() - box.right())}
    return min(distances, key=distances.__getitem__)


def nearest_point(point: PointLike, candidates: Iterable[PointLike]) -> Optional[QPointF]:
    """Return a copy of the candidate closest to ``point``, or ``None`` if empty."""
    source = _pointf(point)
    closest: Optional[QPointF] = None
    closest_distance = math.inf
    for candidate in candidates:
        candidate_point = _pointf(candidate)
        value = distance(source, candidate_point)
        if value < closest_distance:
            closest, closest_distance = candidate_point, value
    return closest


def nearest_corner(point: PointLike, rect: RectLike) -> QPointF:
    """Return the corner of ``rect`` closest to ``point``."""
    box = _rectf(rect)
    result = nearest_point(point, (box.topLeft(), box.topRight(), box.bottomLeft(), box.bottomRight()))
    assert result is not None
    return result


def distance(first: PointLike, second: PointLike) -> float:
    """Return Euclidean distance between two points in Qt coordinate space."""
    a, b = _pointf(first), _pointf(second)
    return math.hypot(b.x() - a.x(), b.y() - a.y())


def angle(first: PointLike, second: PointLike) -> float:
    """Return the clockwise screen-space angle from ``first`` to ``second`` in degrees."""
    a, b = _pointf(first), _pointf(second)
    return math.degrees(math.atan2(b.y() - a.y(), b.x() - a.x()))


def direction_vector(first: PointLike, second: PointLike) -> QPointF:
    """Return the unnormalised vector pointing from ``first`` to ``second``."""
    a, b = _pointf(first), _pointf(second)
    return QPointF(b.x() - a.x(), b.y() - a.y())


def normalize_vector(vector: PointLike) -> QPointF:
    """Return a unit vector, or ``QPointF(0, 0)`` for a zero-length vector."""
    value = _pointf(vector)
    length = math.hypot(value.x(), value.y())
    return QPointF() if length == 0.0 else QPointF(value.x() / length, value.y() / length)


def project_point(point: PointLike, line_start: PointLike, line_end: PointLike) -> QPointF:
    """Project ``point`` onto the infinite line defined by ``line_start`` and ``line_end``."""
    p, a, b = _pointf(point), _pointf(line_start), _pointf(line_end)
    vector = direction_vector(a, b)
    denominator = vector.x() ** 2 + vector.y() ** 2
    if denominator == 0.0:
        return a
    factor = ((p.x() - a.x()) * vector.x() + (p.y() - a.y()) * vector.y()) / denominator
    return QPointF(a.x() + factor * vector.x(), a.y() + factor * vector.y())


def snap_point(point: PointLike, grid: Union[float, QSize, QSizeF, tuple[float, float]]) -> QPointF:
    """Snap a point to the nearest positive grid increment in each dimension."""
    p = _pointf(point)
    if isinstance(grid, (int, float)):
        width = height = float(grid)
    elif isinstance(grid, (QSize, QSizeF)):
        width, height = float(grid.width()), float(grid.height())
    else:
        width, height = float(grid[0]), float(grid[1])
    if width <= 0 or height <= 0:
        raise ValueError("grid dimensions must be positive")
    return QPointF(round(p.x() / width) * width, round(p.y() / height) * height)


def bounding_rect(points: PolygonLike) -> QRectF:
    """Return normalized floating bounds for points or a Qt polygon; empty input is null."""
    if isinstance(points, (QPolygon, QPolygonF)):
        return QRectF(points.boundingRect()).normalized()
    values = [_pointf(point) for point in points]
    if not values:
        return QRectF()
    return QRectF(min(p.x() for p in values), min(p.y() for p in values),
                  max(p.x() for p in values) - min(p.x() for p in values),
                  max(p.y() for p in values) - min(p.y() for p in values))


def line_intersection(first: Union[QLine, QLineF], second: Union[QLine, QLineF]) -> Optional[QPointF]:
    """Return the bounded-segment intersection of two lines, or ``None`` when absent."""
    a, b = QLineF(first), QLineF(second)
    result = a.intersects(b)
    kind, point = result[0], result[1]
    return point if kind == QLineF.IntersectionType.BoundedIntersection else None


def rect_intersection(first: RectLike, second: RectLike) -> Optional[QRectF]:
    """Return the positive-area overlap of two rectangles, or ``None`` if they do not overlap."""
    overlap = _rectf(first).intersected(_rectf(second))
    return overlap if not overlap.isEmpty() else None


def polygon_bounds(polygon: PolygonLike) -> QRectF:
    """Return the bounding rectangle of a polygon or point collection."""
    return bounding_rect(polygon)


def polygon_center(polygon: PolygonLike) -> QPointF:
    """Return the centre of polygon bounds (a stable connector anchor)."""
    return bounding_rect(polygon).center()


def contains(rect: RectLike, point: PointLike) -> bool:
    """Return whether a normalized rectangle contains a point."""
    return _rectf(rect).contains(_pointf(point))


def intersects(first: RectLike, second: RectLike) -> bool:
    """Return whether two normalized rectangles overlap or touch."""
    return _rectf(first).intersects(_rectf(second))


def aspect_ratio(size: Union[QSize, QSizeF, RectLike]) -> float:
    """Return width divided by height, or ``0.0`` when height is zero."""
    return float(size.width()) / float(size.height()) if size.height() else 0.0


# ============================================================================
# Qt URL and resource paths -- portable boundary between files and Qt APIs
# ============================================================================

def url(value: Union[str, Path, QUrl]) -> QUrl:
    """Return a ``QUrl``; filesystem paths become local-file URLs."""
    if isinstance(value, QUrl):
        return QUrl(value)
    text = os.fspath(value)
    parsed = QUrl(text)
    return parsed if parsed.isValid() and parsed.scheme() else QUrl.fromLocalFile(str(Path(text).expanduser()))


def file_to_url(path: Union[str, Path]) -> QUrl:
    """Convert a filesystem path to a local-file ``QUrl``."""
    return QUrl.fromLocalFile(str(Path(path).expanduser()))


def url_to_file(value: Union[str, QUrl]) -> str:
    """Return a local filesystem path for a local URL, otherwise an empty string."""
    parsed = url(value) if isinstance(value, str) else value
    return parsed.toLocalFile() if parsed.isLocalFile() else ""


def is_url(value: object) -> bool:
    """Return whether ``value`` is a valid URL with a scheme or a local-file URL."""
    if not isinstance(value, (str, QUrl)):
        return False
    parsed = value if isinstance(value, QUrl) else QUrl.fromUserInput(value)
    return parsed.isValid() and (bool(parsed.scheme()) or parsed.isLocalFile())


def resource_path(*parts: Union[str, Path]) -> str:
    """Return an absolute path rooted at the Draw package's resource directory.

    The function does not require the resource to exist, making it suitable for
    both packaged assets and paths that will be written by the caller.
    """
    return str(Path(__file__).resolve().parent.joinpath(*(os.fspath(part) for part in parts)))


# ============================================================================
# Multimedia -- lazy QtMultimedia construction only, never playback policy
# ============================================================================

def _multimedia() -> Any:
    """Import QtMultimedia lazily to preserve geometry-only import cost."""
    from PySide6 import QtMultimedia
    return QtMultimedia


def media_player(parent: Optional[Any] = None, *, audio_output: Union[bool, Any] = True) -> Any:
    """Create a ``QMediaPlayer`` with an optional ``QAudioOutput`` attached."""
    multimedia = _multimedia()
    player = multimedia.QMediaPlayer(parent)
    if audio_output is not False:
        player.setAudioOutput(multimedia.QAudioOutput(parent) if audio_output is True else audio_output)
    return player


def audio_output(parent: Optional[Any] = None) -> Any:
    """Create and return a Qt ``QAudioOutput``."""
    return _multimedia().QAudioOutput(parent)


def video_sink(parent: Optional[Any] = None) -> Any:
    """Create and return a Qt ``QVideoSink`` for frame consumers."""
    return _multimedia().QVideoSink(parent)


def sound_effect(parent: Optional[Any] = None) -> Any:
    """Create and return a Qt ``QSoundEffect``."""
    return _multimedia().QSoundEffect(parent)


def camera(device: Optional[Any] = None, parent: Optional[Any] = None) -> Any:
    """Create a ``QCamera``, optionally for the supplied Qt camera device."""
    klass = _multimedia().QCamera
    return klass(parent) if device is None else klass(device, parent)


def media_capture_session(parent: Optional[Any] = None, **targets: Any) -> Any:
    """Create a ``QMediaCaptureSession`` and attach supported named targets.

    Accepted target names are ``camera``, ``audio_input``, ``audio_output``,
    ``recorder`` and ``video_sink``.  Unknown names raise ``ValueError``.
    """
    session = _multimedia().QMediaCaptureSession(parent)
    setters = {"camera": "setCamera", "audio_input": "setAudioInput", "audio_output": "setAudioOutput",
               "recorder": "setRecorder", "video_sink": "setVideoSink"}
    for name, target in targets.items():
        try:
            getattr(session, setters[name])(target)
        except KeyError as exc:
            raise ValueError(f"unsupported capture-session target: {name}") from exc
    return session


# ============================================================================
# Clipboard and dialogs -- small GUI boundaries, imported lazily
# ============================================================================

def _clipboard() -> Any:
    """Return Qt's clipboard, creating Draw's application only when required."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        from Draw._app import get_app
        app = get_app()
    clipboard = app.clipboard()
    if clipboard is None:
        raise RuntimeError("could not access the system clipboard")
    return clipboard


def copy_text(value: object) -> None:
    """Copy plain text to the system clipboard."""
    # Reuse the established public clipboard implementation instead of
    # maintaining a second text-clipboard policy in this foundation module.
    from Draw._clipboard import copy
    copy(value)


def paste_text() -> str:
    """Return plain text from the system clipboard, or an empty string."""
    from Draw._clipboard import read
    return read()


def copy_image(image: Union[QImage, QPixmap]) -> None:
    """Copy a ``QImage`` or ``QPixmap`` to the system clipboard."""
    _clipboard().setImage(image.toImage() if isinstance(image, QPixmap) else image)


def paste_image() -> QImage:
    """Return a clipboard image; a null ``QImage`` indicates no image content."""
    return _clipboard().image()


def open_file_dialog(title: str = "Open File", directory: str = "", filter: str = "All Files (*)") -> str:
    """Open a native file picker and return the selected path, or an empty string."""
    from Draw._filedialog import open as open_file
    return open_file(title, directory, filter)


def save_file_dialog(title: str = "Save File", directory: str = "", filter: str = "All Files (*)") -> str:
    """Open a native save picker and return the selected path, or an empty string."""
    from Draw._filedialog import save as save_file
    return save_file(title, directory, filter)


def select_directory(title: str = "Select Directory", directory: str = "") -> str:
    """Open a native directory picker and return the selected path, or an empty string."""
    from PySide6.QtWidgets import QFileDialog
    return QFileDialog.getExistingDirectory(None, title, directory) or ""


def message_box(text: str, title: str = "Draw", *, level: str = "information", parent: Optional[Any] = None) -> Any:
    """Show a simple message box and return its standard-button result."""
    from PySide6.QtWidgets import QMessageBox
    levels = {"information": QMessageBox.Information, "warning": QMessageBox.Warning,
              "critical": QMessageBox.Critical, "question": QMessageBox.Question}
    try:
        return QMessageBox(levels[level.lower()], title, text, QMessageBox.Ok, parent).exec()
    except KeyError as exc:
        raise ValueError("level must be information, warning, critical, or question") from exc


def color_dialog(initial: Optional[QColor] = None, parent: Optional[Any] = None) -> QColor:
    """Open a colour picker and return its colour (invalid when cancelled)."""
    from PySide6.QtWidgets import QColorDialog
    return QColorDialog.getColor(initial or QColor(), parent)


def font_dialog(initial: Optional[Any] = None, parent: Optional[Any] = None) -> tuple[Any, bool]:
    """Open a font picker and return ``(font, accepted)``."""
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QFontDialog
    return QFontDialog.getFont(initial if initial is not None else QFont(), parent)


# ============================================================================
# Images -- value-type conversions with no image-widget policy
# ============================================================================

def load_image(path: Union[str, Path], *, as_pixmap: bool = False) -> Union[QImage, QPixmap]:
    """Load an image path as ``QImage`` (or ``QPixmap`` when requested)."""
    return QPixmap(os.fspath(path)) if as_pixmap else QImage(os.fspath(path))


def save_image(image: Union[QImage, QPixmap], path: Union[str, Path], format: Optional[str] = None, quality: int = -1) -> bool:
    """Save an image and return Qt's success flag; format is inferred when omitted."""
    source = image.toImage() if isinstance(image, QPixmap) else image
    return source.save(os.fspath(path), format, quality)


def create_icon(source: Union[str, Path, QImage, QPixmap, QIcon]) -> QIcon:
    """Create an icon from a path or Qt image value, preserving an existing icon."""
    if isinstance(source, QIcon):
        return QIcon(source)
    if isinstance(source, QImage):
        return QIcon(QPixmap.fromImage(source))
    return QIcon(source if isinstance(source, QPixmap) else os.fspath(source))


def thumbnail(source: Union[str, Path, QImage, QPixmap], size: Union[int, QSize, QSizeF]) -> QImage:
    """Return an aspect-ratio-preserving ``QImage`` thumbnail using smooth scaling."""
    from PySide6.QtCore import Qt
    image = QImage(os.fspath(source)) if isinstance(source, (str, Path)) else (source.toImage() if isinstance(source, QPixmap) else source)
    target = QSize(size, size) if isinstance(size, int) else QSize(size)
    return image.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ============================================================================
# Screen, platform, validation, and time -- cross-module environment facts
# ============================================================================

def _screen(screen: Optional[Any] = None) -> Optional[Any]:
    """Return the requested or primary screen without constructing an application."""
    if screen is not None:
        return screen
    from PySide6.QtGui import QGuiApplication
    return QGuiApplication.primaryScreen()


def screen_size(screen: Optional[Any] = None) -> Optional[QSize]:
    """Return the screen geometry size, or ``None`` when no GUI screen exists."""
    active = _screen(screen)
    return active.geometry().size() if active is not None else None


def available_geometry(screen: Optional[Any] = None) -> Optional[QRect]:
    """Return usable desktop geometry, excluding taskbars and docks when available."""
    active = _screen(screen)
    return active.availableGeometry() if active is not None else None


def device_pixel_ratio(screen: Optional[Any] = None) -> Optional[float]:
    """Return screen device-pixel ratio, or ``None`` when no GUI screen exists."""
    active = _screen(screen)
    return float(active.devicePixelRatio()) if active is not None else None


def dpi(screen: Optional[Any] = None) -> Optional[float]:
    """Return physical horizontal screen DPI, or ``None`` without a screen."""
    active = _screen(screen)
    return float(active.physicalDotsPerInchX()) if active is not None else None


def logical_dpi(screen: Optional[Any] = None) -> Optional[float]:
    """Return logical horizontal screen DPI, or ``None`` without a screen."""
    active = _screen(screen)
    return float(active.logicalDotsPerInchX()) if active is not None else None


def is_number(value: object) -> bool:
    """Return whether a value can be converted to a finite floating-point number."""
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def is_path(value: object, *, exists: bool = False) -> bool:
    """Return whether a value is path-like; optionally require that it exists."""
    if not isinstance(value, (str, Path)):
        return False
    try:
        path = Path(value).expanduser()
        return path.exists() if exists else bool(str(path))
    except (OSError, ValueError):
        return False


def is_color(value: object) -> bool:
    """Return whether Qt or Draw can interpret ``value`` as a valid ``QColor``."""
    if isinstance(value, QColor):
        return value.isValid()
    if isinstance(value, str):
        return QColor(value).isValid()
    if isinstance(value, (tuple, list)) and len(value) in (3, 4):
        return all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in value)
    return False


@overload
def safe_cast(value: object, target: Callable[[object], _T], default: _T) -> _T: ...
@overload
def safe_cast(value: object, target: Callable[[object], _T], default: None = None) -> Optional[_T]: ...
def safe_cast(value: object, target: Callable[[object], _T], default: Optional[_T] = None) -> Optional[_T]:
    """Cast a value and return ``default`` if the target raises ``TypeError`` or ``ValueError``."""
    try:
        return target(value)
    except (TypeError, ValueError):
        return default


def platform_name() -> str:
    """Return Qt's platform product name (for example ``windows`` or ``osx``)."""
    return QSysInfo.productType()


def is_windows() -> bool:
    """Return whether Draw is running on Windows."""
    return platform_name() == "windows"


def is_linux() -> bool:
    """Return whether Draw is running on a Linux platform."""
    return platform_name() == "linux"


def is_macos() -> bool:
    """Return whether Draw is running on macOS."""
    return platform_name() in {"osx", "macos"}


def temporary_directory() -> str:
    """Return Qt's writable temporary directory, or the system temporary path."""
    return QStandardPaths.writableLocation(QStandardPaths.TempLocation) or os.getenv("TMP", "")


def home_directory() -> str:
    """Return Qt's writable home directory."""
    return QStandardPaths.writableLocation(QStandardPaths.HomeLocation)


def now(*, utc: bool = False) -> QDateTime:
    """Return the current local or UTC Qt date-time."""
    return QDateTime.currentDateTimeUtc() if utc else QDateTime.currentDateTime()


def elapsed_ms(timer: QElapsedTimer) -> int:
    """Return milliseconds elapsed on a started ``QElapsedTimer``."""
    return timer.elapsed()


def single_shot(milliseconds: int, callback: Callable[[], Any]) -> None:
    """Schedule a callback once through Qt's event loop."""
    QTimer.singleShot(milliseconds, callback)


# ============================================================================
# Auto Z-Indexing Manager
# ============================================================================

_current_z_counter: float = 0.0


def next_z(step: float = 1.0) -> float:
    """
    Return the current auto Z value and increment by step.
    Elements created first receive lower Z values (e.g. 0.0, 1.0, 2.0...).
    """
    global _current_z_counter
    val = _current_z_counter
    _current_z_counter += step
    return val


def auto_z(reset: bool = False, start: float = 0.0, step: float = 1.0) -> float:
    """
    Auto Z-index manager tool call.
    If reset=True, resets counter to start. Returns next sequential Z value.
    """
    global _current_z_counter
    if reset:
        _current_z_counter = float(start)
    val = _current_z_counter
    _current_z_counter += step
    return val


def reset_z(start: float = 0.0) -> None:
    """Reset global auto Z counter to start."""
    global _current_z_counter
    _current_z_counter = float(start)


_MULTIMEDIA_TYPES = {"QMediaPlayer", "QAudioOutput", "QVideoSink", "QSoundEffect", "QCamera", "QMediaCaptureSession"}


def __getattr__(name: str) -> Any:
    """Lazily expose requested Qt Multimedia classes without import-time cost."""
    if name in _MULTIMEDIA_TYPES:
        return getattr(_multimedia(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Qt value types
    "QRect", "QRectF", "QPoint", "QPointF", "QSize", "QSizeF", "QLine", "QLineF",
    "QMargins", "QMarginsF", "QPolygon", "QPolygonF", "QUrl", "QImage", "QPixmap", "QIcon",
    "QTimer", "QElapsedTimer", "QDateTime",
    # Geometry
    "center", "edge_center", "corner", "midpoint", "nearest_edge", "nearest_point", "nearest_corner",
    "distance", "angle", "direction_vector", "normalize_vector", "project_point", "snap_point",
    "bounding_rect", "line_intersection", "rect_intersection", "polygon_bounds", "polygon_center",
    "contains", "intersects", "aspect_ratio",
    # Auto Z Manager
    "next_z", "auto_z", "reset_z",
    # URLs, multimedia, clipboard, dialogs, images
    "url", "file_to_url", "url_to_file", "is_url", "resource_path",
    "media_player", "audio_output", "video_sink", "sound_effect", "camera", "media_capture_session",
    "copy_text", "paste_text", "copy_image", "paste_image",
    "open_file_dialog", "save_file_dialog", "select_directory", "message_box", "color_dialog", "font_dialog",
    "load_image", "save_image", "create_icon", "thumbnail",
    # System utilities
    "screen_size", "available_geometry", "device_pixel_ratio", "dpi", "logical_dpi",
    "is_number", "is_path", "is_color", "safe_cast",
    "platform_name", "is_windows", "is_linux", "is_macos", "temporary_directory", "home_directory",
    "now", "elapsed_ms", "single_shot",
] + sorted(_MULTIMEDIA_TYPES)
