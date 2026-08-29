"""
Draw._clipboard
===============
System clipboard integration for the Draw UI library.

Provides clipboard operations for plain text, rich text/HTML, custom MIME data types,
clipboard clearing, and real-time clipboard change listeners.

Usage
-----
    import Draw

    Draw.clipboard.copy("Hello, world!")
    value = Draw.clipboard.read()

    Draw.clipboard.copy_html("<h1>Hello</h1>", text="Hello")
    html_val = Draw.clipboard.read_html()

    Draw.clipboard.clear()

    Draw.clipboard.on_change(lambda: print("Clipboard changed!"))
"""

from __future__ import annotations

from typing import Callable, Optional, Union

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QByteArray, QMimeData
# pyrefly: ignore [missing-import]
from PySide6.QtGui import QClipboard
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QApplication

_LISTENERS: list[Callable[[], None]] = []
_LISTENER_CONNECTED = False


def _get_clipboard() -> QClipboard:
    """Return Qt's global clipboard, creating a QApplication if needed."""
    app = QApplication.instance()
    if app is None:
        from Draw._app import get_app
        app = get_app()
    clipboard = app.clipboard()
    if clipboard is None:
        raise RuntimeError(
            "Draw.clipboard: could not access system clipboard. "
            "Ensure a QApplication is running."
        )
    return clipboard


def _dispatch_change() -> None:
    """Dispatch dataChanged signal to registered Python callbacks."""
    for callback in list(_LISTENERS):
        try:
            callback()
        except Exception:
            pass


def copy(text: object) -> None:
    """
    Write plain text to the OS clipboard.
    """
    _get_clipboard().setText(str(text) if not isinstance(text, str) else text)


def read() -> str:
    """
    Read plain text from the OS clipboard.
    """
    text: Optional[str] = _get_clipboard().text()
    return text if text is not None else ""


def copy_html(html: str, text: Optional[str] = None) -> None:
    """
    Write Rich Text / HTML content to the OS clipboard.

    Parameters
    ----------
    html : str
        The HTML string to place on the clipboard.
    text : str, optional
        Plain text fallback for applications that do not support HTML.
        If omitted, *html* is stripped or used as plain text fallback.
    """
    mime = QMimeData()
    mime.setHtml(html)
    if text is not None:
        mime.setText(text)
    else:
        mime.setText(html)
    _get_clipboard().setMimeData(mime)


def read_html() -> str:
    """
    Read HTML / Rich Text content from the OS clipboard.

    Returns
    -------
    str
        The HTML string on the clipboard, or an empty string if no HTML is present.
    """
    cb = _get_clipboard()
    mime = cb.mimeData()
    if mime is not None and mime.hasHtml():
        return mime.html()
    return ""


def clear() -> None:
    """
    Clear all content from the operating system clipboard.
    """
    _get_clipboard().clear()


def copy_data(mime_type: str, data: Union[bytes, str]) -> None:
    """
    Write custom MIME data (binary bytes or str) to the OS clipboard.

    Parameters
    ----------
    mime_type : str
        MIME identifier string (e.g. "application/json", "text/csv").
    data : bytes or str
        Payload bytes or text to store.
    """
    mime = QMimeData()
    if isinstance(data, str):
        payload = QByteArray(data.encode("utf-8"))
    else:
        payload = QByteArray(data)
    mime.setData(mime_type, payload)
    _get_clipboard().setMimeData(mime)


def read_data(mime_type: str) -> bytes:
    """
    Read custom MIME data from the OS clipboard by MIME type.

    Parameters
    ----------
    mime_type : str
        MIME type identifier to query.

    Returns
    -------
    bytes
        Raw bytes stored for that MIME type, or empty bytes if unavailable.
    """
    cb = _get_clipboard()
    mime = cb.mimeData()
    if mime is not None and mime.hasFormat(mime_type):
        qba: QByteArray = mime.data(mime_type)
        return bytes(qba.data())
    return b""


def on_change(callback: Callable[[], None]) -> None:
    """
    Register a callback function to be invoked whenever the OS clipboard content changes.
    """
    global _LISTENER_CONNECTED
    if callback not in _LISTENERS:
        _LISTENERS.append(callback)

    if not _LISTENER_CONNECTED:
        cb = _get_clipboard()
        cb.dataChanged.connect(_dispatch_change)
        _LISTENER_CONNECTED = True
