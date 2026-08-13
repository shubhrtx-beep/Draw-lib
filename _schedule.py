"""
Draw._schedule — generic delayed / repeating callback helper.

Draw.motion / Draw.timeline animate shape *properties* over time. This
module fills a different gap: running arbitrary, non-shape logic after a
delay, or on a repeating interval, without reaching for QTimer directly.

Usage
-----
    Draw.after(1.5, lambda: print("1.5s later"))

    handle = Draw.every(0.5, tick_fn)   # repeats every 0.5s
    handle.stop()                        # cancel the repeat later
"""

from __future__ import annotations

from typing import Callable, Optional

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QTimer

from Draw._app import get_app


# Keep a hard reference to every live timer. Without this, a QTimer created
# with no parent and no other Python reference can be garbage-collected
# before it ever fires, silently swallowing the callback.
_active_timers: set = set()


class _ScheduledHandle:
    """Returned by Draw.after()/Draw.every(); call .stop() to cancel."""

    def __init__(self, timer: QTimer) -> None:
        self._timer = timer

    def stop(self) -> None:
        self._timer.stop()
        _active_timers.discard(self._timer)

    @property
    def active(self) -> bool:
        return self._timer.isActive()


def after(seconds: float, fn: Callable[[], None]) -> _ScheduledHandle:
    """
    Run ``fn()`` once, ``seconds`` from now. Requires the Qt event loop to
    be running (i.e. called after Draw.window(...) and before/around
    Draw.get_app().exec()).
    """
    get_app()  # ensure a QApplication exists
    timer = QTimer()
    timer.setSingleShot(True)
    _active_timers.add(timer)

    def _fire():
        try:
            fn()
        finally:
            _active_timers.discard(timer)

    timer.timeout.connect(_fire)
    timer.start(max(0, int(seconds * 1000)))
    return _ScheduledHandle(timer)


def every(seconds: float, fn: Callable[[], None]) -> _ScheduledHandle:
    """
    Run ``fn()`` repeatedly every ``seconds``, until ``.stop()`` is called
    on the returned handle.
    """
    get_app()
    timer = QTimer()
    timer.timeout.connect(fn)
    _active_timers.add(timer)
    timer.start(max(1, int(seconds * 1000)))
    return _ScheduledHandle(timer)
