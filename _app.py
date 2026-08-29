"""
Draw._app
Manages the single QApplication instance required by PySide6.
"""

import sys
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QApplication

_app = None


def get_app() -> QApplication:
    """Return the existing QApplication or create one if it doesn't exist."""
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(sys.argv)
    return _app
