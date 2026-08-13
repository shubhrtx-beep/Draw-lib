"""
Draw._filedialog
================
Native file dialog integration for the Draw UI library.

Provides helper functions for opening native OS file selection dialogs:

    Draw.filedialog.open_file(caption="Select File", filter="All Files (*.*)")
    Draw.filedialog.save_file(caption="Save File", filter="All Files (*.*)")
    Draw.filedialog.select_folder(caption="Select Folder")

Both use PySide6's QFileDialog.
"""

from __future__ import annotations

from typing import Optional, Tuple
from PySide6.QtWidgets import QApplication, QFileDialog


def _get_parent_widget():
    app = QApplication.instance()
    if app is None:
        from Draw._app import get_app
        get_app()
    try:
        from Draw._window import window as _window_registry
        tags = _window_registry.list_tags()
        if tags:
            return _window_registry.get(tags[0])
    except Exception:
        pass
    return None


def open_file(caption: str = "Select File", dir: str = "", filter: str = "All Files (*.*)") -> Optional[str]:
    parent = _get_parent_widget()
    file_path, _ = QFileDialog.getOpenFileName(parent, caption, dir, filter)
    return file_path if file_path else None


def save_file(caption: str = "Save File", dir: str = "", filter: str = "All Files (*.*)") -> Optional[str]:
    parent = _get_parent_widget()
    file_path, _ = QFileDialog.getSaveFileName(parent, caption, dir, filter)
    return file_path if file_path else None


def select_folder(caption: str = "Select Folder", dir: str = "") -> Optional[str]:
    parent = _get_parent_widget()
    folder_path = QFileDialog.getExistingDirectory(parent, caption, dir)
    return folder_path if folder_path else None
