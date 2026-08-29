"""
Draw._scroller
==============
High-level scrolling controller built on top of Draw modules.

It does not implement rendering or animation itself. Instead, it coordinates:
- Draw.shape      (scrollbar track & thumb shapes)
- Draw.motion     (smooth, spring, inertia, and custom motion physics)
- Draw.senses     (gesture detection: wheel, drag, touch, keyboard)
- Draw.connectors (synchronization between room, thumb, motion, and senses)
- Draw.room       (room alignment, relative layouts, and room scrolling)

Public API
----------
::

    # Basic vertical scrolling
    Draw.scroller(
        ip="main_scroll",
        room="page",
        scroll="vertical",
        max_y=2500,
        speed=1.0,
        align="right",
        thumb="thumb_shape",
        motion="scroll_motion",
        senses="scroll_sense",
        connectors="scroll_connector",
    )

    # Programmatic control
    Draw.scroller.scroll_to(300)
    Draw.scroller.scroll_by(-50)
    Draw.scroller.scroll_to_ip("card_5")
    Draw.scroller.center_ip("card_5")
    pos = Draw.scroller.get_scroll()
    Draw.scroller.set_scroll(100)
    Draw.scroller.lock()
    Draw.scroller.unlock()
    Draw.scroller.enable()
    Draw.scroller.disable()
    Draw.scroller.remove("main_scroll")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  ScrollerConfig — one registered scroller instance
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScrollerConfig:
    """Represents one active scroller created by ``Draw.scroller(...)``."""
    id: str
    display: str                                              # window tag
    direction: str = "vertical"                               # "vertical" | "horizontal" | "both"
    scroll: str = "vertical"                                  # alias for direction
    ip: str = ""                                              # unique identifier
    room: Optional[Any] = None                                # room identifier / name
    max_x: Optional[float] = None                             # max horizontal scroll distance
    max_y: Optional[float] = None                             # max vertical scroll distance
    speed: float = 1.0                                        # scroll speed multiplier
    align: str = "right"                                      # scrollbar placement ("left", "right", "top", "bottom")
    thumb: Optional[Any] = None                               # thumb shape IP or bool
    motion: Optional[Any] = None                              # motion IP or preset ("spring", "inertia", etc.)
    senses: Optional[Any] = None                              # sense IP
    connectors: Optional[Any] = None                          # connector IP
    target_ip: Optional[str] = None                           # scroll shapes inside this ip, or whole canvas
    sense_ids: List[str] = field(default_factory=list)        # registered sense IDs
    connector_ids: List[str] = field(default_factory=list)    # registered connector IDs
    thumb_ip: Optional[str] = None                            # thumb shape ip (if visible)
    track_ip: Optional[str] = None                            # track shape ip (if visible)
    inertia: bool = False                                     # enable fling momentum
    snap: Optional[float] = None                              # snap to grid interval (px)
    keyboard: bool = True                                     # register arrow / page key senses
    show_thumb: bool = False                                  # whether track+thumb shapes exist
    enabled: bool = True                                      # whether scrolling is enabled
    locked: bool = False                                      # whether user interaction is locked
    thumb_color: str = "#5b6880"                              # thumb shape color
    track_color: str = "#2a2f3a"                              # track background color
    thumb_width: float = 12.0                                 # track / thumb thickness in px
    track_opacity: int = 100                                  # track background opacity (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ip": self.ip or self.id,
            "display": self.display,
            "direction": self.direction,
            "scroll": self.scroll,
            "room": self.room,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "speed": self.speed,
            "align": self.align,
            "thumb": self.thumb,
            "motion": self.motion,
            "senses": self.senses,
            "connectors": self.connectors,
            "target_ip": self.target_ip,
            "sense_ids": list(self.sense_ids),
            "connector_ids": list(self.connector_ids),
            "thumb_ip": self.thumb_ip,
            "track_ip": self.track_ip,
            "speed_val": self.speed,
            "inertia": self.inertia,
            "snap": self.snap,
            "keyboard": self.keyboard,
            "show_thumb": self.show_thumb,
            "enabled": self.enabled,
            "locked": self.locked,
            "thumb_color": self.thumb_color,
            "track_color": self.track_color,
            "thumb_width": self.thumb_width,
            "track_opacity": self.track_opacity,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  _ScrollerRegistry — the public Draw.scroller singleton
# ══════════════════════════════════════════════════════════════════════════════

class _ScrollerRegistry:
    """High-level scroller API — coordinates Draw.shape, Draw.motion,
    Draw.senses, Draw.connectors, and Draw.room.

    Callable as ``Draw.scroller(...)`` to create a scroller. Also exposes
    methods for programmatic scrolling, querying, locking, and cleanup.
    """

    def __init__(self) -> None:
        self._items: Dict[str, ScrollerConfig] = {}
        self._counter: int = 0
        self._lock = threading.Lock()

    # ── create a scroller ─────────────────────────────────────────────────

    def __call__(
        self,
        ip: Optional[str] = None,
        *,
        room: Optional[Any] = None,
        scroll: str = "vertical",
        direction: Optional[str] = None,
        max_x: Optional[float] = None,
        max_y: Optional[float] = None,
        speed: float = 1.0,
        align: str = "right",
        thumb: Optional[Any] = None,
        motion: Optional[Any] = None,
        senses: Optional[Any] = None,
        connectors: Optional[Any] = None,
        display: Optional[str] = None,
        id: Optional[str] = None,
        target_ip: Optional[str] = None,
        inertia: bool = False,
        snap: Optional[float] = None,
        keyboard: bool = True,
        show_thumb: Optional[bool] = None,
        thumb_color: str = "#5b6880",
        track_color: str = "#2a2f3a",
        thumb_width: float = 12.0,
        track_opacity: int = 100,
        **kwargs: Any,
    ) -> ScrollerConfig:
        """Create a scroller controller for a canvas, room, or region.

        Parameters
        ----------
        ip           : Unique identifier (e.g. "main_scroll").
        room         : Room that will be moved while scrolling (e.g. "page").
        scroll       : Scrolling direction: "vertical" | "horizontal" | "both".
        direction    : Alias for scroll.
        max_x        : Maximum horizontal scrolling distance (px).
        max_y        : Maximum vertical scrolling distance (px).
        speed        : Scrolling multiplier (1.0 = default, 2.5, 0.5, etc.).
        align        : Scrollbar position ("right", "left", "top", "bottom").
        thumb        : Shape IP used as scrollbar thumb, or bool.
        motion       : Motion IP or preset ("spring", "inertia", "smooth", "instant").
        senses       : Sense IP used to receive user interaction.
        connectors   : Connector IP used to synchronize room, thumb, and scroll value.
        display      : Window tag. Auto-resolved if only one window exists.
        id           : Alias for ip.
        target_ip    : Scope scrolling to shapes inside this region (None = canvas).
        inertia      : Enable fling momentum after touch/drag release.
        snap         : Snap scroll position to this grid interval (px).
        keyboard     : Register arrow, PageUp/Down, Home/End key senses.
        show_thumb   : Explicitly show/hide scrollbar track + thumb shapes.
        thumb_color  : Thumb shape colour.
        track_color  : Track background colour.
        thumb_width  : Track / thumb thickness in px.
        track_opacity: Track background opacity (0-100).
        """
        # ── resolve display tag ───────────────────────────────────────────
        tag = self._resolve_tag(display)

        # ── allocate identifier ───────────────────────────────────────────
        raw_id = ip if ip is not None else (id if id is not None else kwargs.get("name"))
        scroller_id = str(raw_id) if raw_id is not None else f"scroller_{self._counter}"
        self._counter += 1

        # ── resolve direction / scroll mode ───────────────────────────────
        scroll_mode = (direction or scroll or "vertical").strip().lower()
        if scroll_mode not in ("vertical", "horizontal", "both"):
            scroll_mode = "vertical"

        # ── resolve thumb visibility ──────────────────────────────────────
        has_thumb_spec = False
        thumb_shape_ip: Optional[str] = None
        if thumb is not None:
            if isinstance(thumb, str):
                thumb_shape_ip = thumb.strip()
                has_thumb_spec = True
            elif isinstance(thumb, bool):
                has_thumb_spec = thumb
        if show_thumb is not None:
            has_thumb_spec = bool(show_thumb)

        # ── motion / inertia resolution ───────────────────────────────────
        enable_inertia = bool(inertia)
        if motion == "inertia" or kwargs.get("physics") == "inertia":
            enable_inertia = True

        cfg = ScrollerConfig(
            id=scroller_id,
            display=tag,
            direction=scroll_mode,
            scroll=scroll_mode,
            ip=scroller_id,
            room=room,
            max_x=float(max_x) if max_x is not None else None,
            max_y=float(max_y) if max_y is not None else None,
            speed=float(speed),
            align=str(align).strip().lower() if align else "right",
            thumb=thumb,
            motion=motion,
            senses=senses,
            connectors=connectors,
            target_ip=target_ip,
            inertia=enable_inertia,
            snap=float(snap) if snap is not None else None,
            keyboard=bool(keyboard),
            show_thumb=has_thumb_spec,
            thumb_ip=thumb_shape_ip,
            thumb_color=thumb_color,
            track_color=track_color,
            thumb_width=float(thumb_width),
            track_opacity=int(track_opacity),
        )

        # ── 1. Register senses (gesture detection: wheel, touch, keys) ────
        self._register_senses(cfg)

        # ── 2. Wire connectors (sense → scroll offset synchronization) ────
        self._register_connectors(cfg)

        # ── 3. Optionally build or link thumb/track shapes ────────────────
        if cfg.show_thumb:
            self._build_thumb_shapes(
                cfg,
                thumb_color=thumb_color,
                track_color=track_color,
                thumb_width=thumb_width,
                track_opacity=track_opacity,
            )

        # ── 4. Register keyboard senses if requested ──────────────────────
        if keyboard:
            self._register_keyboard_senses(cfg)

        with self._lock:
            self._items[scroller_id] = cfg

        _logger.debug(
            "Draw.scroller: created '%s' (%s) on display='%s'",
            scroller_id, scroll_mode, tag,
        )
        return cfg

    # ── programmatic scroll control ───────────────────────────────────────

    def scroll_to(
        self,
        value: Optional[float] = None,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
        display: Optional[str] = None,
        direction: Optional[str] = None,
        ip: Optional[str] = None,
        animate: bool = False,
        duration: float = 0.25,
    ) -> None:
        """Set the scroll offset to an absolute pixel coordinate.

        Parameters
        ----------
        value    : Single offset (applied to y for vertical, x for horizontal).
        x        : Target x offset (px).
        y        : Target y offset (px).
        display  : Window tag.
        direction: "vertical" | "horizontal" | "both".
        ip       : Specific scroller id to apply max constraints from.
        animate  : Smoothly animate the transition.
        duration : Animation duration in seconds (when animate=True).
        """
        tag = self._resolve_tag(display)
        canvas = self._resolve_canvas(tag)
        if canvas is None:
            return

        cfg = self.get(ip) if ip else self._find_first_cfg(tag)
        dir_mode = direction or (cfg.direction if cfg else "vertical")

        target_x = canvas._scroll_x if hasattr(canvas, "_scroll_x") else 0.0
        target_y = canvas._scroll_y if hasattr(canvas, "_scroll_y") else 0.0

        if value is not None:
            val = float(value)
            if dir_mode == "horizontal":
                target_x = val
            elif dir_mode == "both":
                target_x = val
                target_y = val
            else:
                target_y = val

        if x is not None:
            target_x = float(x)
        if y is not None:
            target_y = float(y)

        # Apply clamping limits
        max_scroll_x, max_scroll_y = canvas._get_max_scroll_range() if hasattr(canvas, "_get_max_scroll_range") else (0.0, 0.0)
        bound_max_x = float(cfg.max_x) if cfg and cfg.max_x is not None else max(0.0, max_scroll_x)
        bound_max_y = float(cfg.max_y) if cfg and cfg.max_y is not None else max(0.0, max_scroll_y)

        target_x = max(0.0, min(bound_max_x, target_x))
        target_y = max(0.0, min(bound_max_y, target_y))

        if animate and (getattr(cfg, "motion", None) or animate):
            self._animate_scroll_to(canvas, target_x, target_y, duration=duration)
        else:
            canvas._scroll_x = target_x
            canvas._scroll_y = target_y
            if hasattr(canvas, "_update_scroller_thumbs"):
                canvas._update_scroller_thumbs()
            self._sync_room_offset(cfg, target_x, target_y)
            canvas.update()

    def scroll_by(
        self,
        delta: Optional[float] = None,
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        display: Optional[str] = None,
        direction: Optional[str] = None,
        ip: Optional[str] = None,
        animate: bool = False,
        duration: float = 0.25,
    ) -> None:
        """Adjust the scroll offset by a relative delta (px)."""
        tag = self._resolve_tag(display)
        canvas = self._resolve_canvas(tag)
        if canvas is None:
            return

        cfg = self.get(ip) if ip else self._find_first_cfg(tag)
        dir_mode = direction or (cfg.direction if cfg else "vertical")

        cur_x = getattr(canvas, "_scroll_x", 0.0)
        cur_y = getattr(canvas, "_scroll_y", 0.0)

        add_x = dx
        add_y = dy

        if delta is not None:
            d = float(delta)
            if dir_mode == "horizontal":
                add_x += d
            elif dir_mode == "both":
                add_x += d
                add_y += d
            else:
                add_y += d

        self.scroll_to(
            x=cur_x + add_x,
            y=cur_y + add_y,
            display=tag,
            ip=ip,
            animate=animate,
            duration=duration,
        )

    def scroll_to_ip(
        self,
        target_ip: str,
        *,
        display: Optional[str] = None,
        ip: Optional[str] = None,
        animate: bool = False,
        align: str = "center",
        offset: float = 0.0,
    ) -> None:
        """Scroll until a shape or item with `target_ip` becomes visible.

        Parameters
        ----------
        target_ip: Shape or text IP to scroll into view.
        display  : Window tag.
        ip       : Scroller ID.
        animate  : Smoothly animate the transition.
        align    : Viewport alignment ("top", "bottom", "left", "right", "center", "visible").
        offset   : Extra margin / padding offset in pixels.
        """
        tag = self._resolve_tag(display)
        canvas = self._resolve_canvas(tag)
        if canvas is None:
            return

        from Draw import _bridge
        rect = _bridge.get_shape_rect(tag, str(target_ip))
        if rect is None:
            try:
                from Draw._room import _find_live_object, _current_geom
                kind, obj = _find_live_object(tag, str(target_ip))
                if obj is not None:
                    cw = canvas.width() if hasattr(canvas, "width") else 1000
                    ch = canvas.height() if hasattr(canvas, "height") else 800
                    geom = _current_geom(kind, obj, cw, ch, tag)
                    rect = (geom.x, geom.y, geom.w, geom.h)
            except Exception:
                rect = None

        if rect is None:
            for t in getattr(canvas, "text_items", []):
                if getattr(t, "ip", None) == str(target_ip) and getattr(t, "last_rect", None):
                    rect = t.last_rect
                    break
            if rect is None:
                for s in getattr(canvas, "shape_items", []):
                    if getattr(s, "ip", None) == str(target_ip):
                        px = getattr(s, "last_position", (s.x, s.y))
                        sz = getattr(s, "last_size", (getattr(s, "width", 100), getattr(s, "height", 100)))
                        if px and sz:
                            rect = (px[0], px[1], sz[0], sz[1])
                        break

        if rect is None:
            return

        unscrolled_x = rect[0] + getattr(canvas, "_scroll_x", 0.0)
        unscrolled_y = rect[1] + getattr(canvas, "_scroll_y", 0.0)
        sw, sh = rect[2], rect[3]

        cw = float(canvas.width()) if hasattr(canvas, "width") and canvas.width() > 0 else 1000.0
        ch = float(canvas.height()) if hasattr(canvas, "height") and canvas.height() > 0 else 800.0

        align_norm = str(align).strip().lower()
        target_x = getattr(canvas, "_scroll_x", 0.0)
        target_y = getattr(canvas, "_scroll_y", 0.0)

        if align_norm in ("top", "top-left", "top-right"):
            target_y = unscrolled_y - offset
        elif align_norm in ("bottom", "bottom-left", "bottom-right"):
            target_y = (unscrolled_y + sh) - ch + offset
        elif align_norm in ("center", "middle"):
            target_y = unscrolled_y + sh / 2.0 - ch / 2.0 + offset
            target_x = unscrolled_x + sw / 2.0 - cw / 2.0 + offset
        elif align_norm in ("left",):
            target_x = unscrolled_x - offset
        elif align_norm in ("right",):
            target_x = (unscrolled_x + sw) - cw + offset
        elif align_norm in ("visible",):
            cur_x = getattr(canvas, "_scroll_x", 0.0)
            cur_y = getattr(canvas, "_scroll_y", 0.0)
            if unscrolled_y < cur_y:
                target_y = unscrolled_y - offset
            elif unscrolled_y + sh > cur_y + ch:
                target_y = unscrolled_y + sh - ch + offset
            if unscrolled_x < cur_x:
                target_x = unscrolled_x - offset
            elif unscrolled_x + sw > cur_x + cw:
                target_x = unscrolled_x + sw - cw + offset

        self.scroll_to(x=target_x, y=target_y, display=tag, ip=ip, animate=animate)

    def center_ip(
        self,
        target_ip: str,
        *,
        display: Optional[str] = None,
        ip: Optional[str] = None,
        animate: bool = False,
    ) -> None:
        """Center a shape inside the room / viewport."""
        self.scroll_to_ip(
            target_ip,
            display=display,
            ip=ip,
            animate=animate,
            align="center",
        )

    def set_scroll(
        self,
        value: Optional[float] = None,
        *,
        x: Optional[float] = None,
        y: Optional[float] = None,
        display: Optional[str] = None,
        direction: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> None:
        """Set the scroll offset immediately (alias for scroll_to(..., animate=False))."""
        self.scroll_to(
            value,
            x=x,
            y=y,
            display=display,
            direction=direction,
            ip=ip,
            animate=False,
        )

    def get_scroll(
        self,
        display: Optional[str] = None,
        ip: Optional[str] = None,
    ) -> Dict[str, float]:
        """Return current scroll offsets ``{"x": ..., "y": ...}``."""
        tag = self._resolve_tag(display)
        canvas = self._resolve_canvas(tag)
        if canvas is None:
            return {"x": 0.0, "y": 0.0}
        return {
            "x": getattr(canvas, "_scroll_x", 0.0),
            "y": getattr(canvas, "_scroll_y", 0.0),
        }

    # ── interactivity controls ────────────────────────────────────────────

    def enable(self, ip: Optional[str] = None, *, display: Optional[str] = None) -> None:
        """Enable scrolling for the specified scroller (or all if omitted)."""
        with self._lock:
            targets = [self._items[ip]] if (ip and ip in self._items) else list(self._items.values())
            for cfg in targets:
                cfg.enabled = True

    def disable(self, ip: Optional[str] = None, *, display: Optional[str] = None) -> None:
        """Disable scrolling for the specified scroller (or all if omitted)."""
        with self._lock:
            targets = [self._items[ip]] if (ip and ip in self._items) else list(self._items.values())
            for cfg in targets:
                cfg.enabled = False

    def lock(self, ip: Optional[str] = None, *, display: Optional[str] = None) -> None:
        """Prevent user scrolling (locks wheel, drag, touch, keyboard interaction)."""
        with self._lock:
            targets = [self._items[ip]] if (ip and ip in self._items) else list(self._items.values())
            for cfg in targets:
                cfg.locked = True

    def unlock(self, ip: Optional[str] = None, *, display: Optional[str] = None) -> None:
        """Unlock scrolling to allow user interaction."""
        with self._lock:
            targets = [self._items[ip]] if (ip and ip in self._items) else list(self._items.values())
            for cfg in targets:
                cfg.locked = False

    # ── query / cleanup ───────────────────────────────────────────────────

    def get(self, scroller_id: str) -> Optional[ScrollerConfig]:
        """Return a scroller config by id, or None."""
        return self._items.get(str(scroller_id))

    def list(self) -> List[Dict[str, Any]]:
        """List all active scrollers as dicts."""
        return [cfg.to_dict() for cfg in self._items.values()]

    def remove(self, scroller_id: Optional[str] = None) -> None:
        """Remove a scroller (and clean up its senses + connectors).

        If *scroller_id* is None, removes all scrollers.
        """
        from Draw._connectors import senses, connectors

        with self._lock:
            if scroller_id is None:
                configs = list(self._items.values())
                self._items.clear()
            else:
                cfg = self._items.pop(str(scroller_id), None)
                configs = [cfg] if cfg else []

        for cfg in configs:
            # clean up senses
            for sid in cfg.sense_ids:
                try:
                    senses.clear(sid)
                except Exception:
                    pass
            # clean up connectors
            for cid in cfg.connector_ids:
                try:
                    connectors.clear(cid)
                except Exception:
                    pass
            _logger.debug("Draw.scroller: removed '%s'", cfg.id)

    # ══════════════════════════════════════════════════════════════════════
    #  Internal wiring helpers
    # ══════════════════════════════════════════════════════════════════════

    def _resolve_tag(self, display: Optional[str]) -> str:
        """Resolve the window tag, auto-picking the sole window if any."""
        from Draw._window import window as _wr
        if display is not None:
            return str(display)
        tags = _wr.list_tags()
        if len(tags) == 1:
            return tags[0]
        if len(tags) > 1:
            raise ValueError(
                "Draw.scroller: multiple windows — 'display' is required."
            )
        raise ValueError(
            "Draw.scroller: no windows exist. Call Draw.window() first."
        )

    def _resolve_canvas(self, tag: str):
        """Return the _DrawCanvas for *tag*, or None."""
        try:
            from Draw._window import window as _wr
            win = _wr.get(tag)
            from Draw._text import _get_or_create_canvas
            return _get_or_create_canvas(tag, win)
        except Exception:
            return None

    def _find_first_cfg(self, tag: str) -> Optional[ScrollerConfig]:
        """Find the first ScrollerConfig associated with display *tag*."""
        for cfg in self._items.values():
            if cfg.display == tag:
                return cfg
        return None

    # ── sense registration (step 1) ───────────────────────────────────────

    def _register_senses(self, cfg: ScrollerConfig) -> None:
        """Register scroll-wheel and touch senses via Draw.senses."""
        from Draw._connectors import senses

        prefix = cfg.senses or cfg.id

        # Wheel scroll senses (detected on any shape or canvas)
        up_id = f"{prefix}_wheel_up"
        senses("mouse_scroll_up", id=up_id)
        cfg.sense_ids.append(up_id)

        down_id = f"{prefix}_wheel_down"
        senses("mouse_scroll_down", id=down_id)
        cfg.sense_ids.append(down_id)

        # Touch senses
        touch_id = f"{prefix}_touch_move"
        senses("touch_move", id=touch_id)
        cfg.sense_ids.append(touch_id)

        # Custom thumb drag sense if thumb IP is given
        if cfg.thumb_ip:
            thumb_drag_id = f"{prefix}_thumb_drag"
            senses("drag_move", ip=cfg.thumb_ip, id=thumb_drag_id)
            cfg.sense_ids.append(thumb_drag_id)

    def _register_keyboard_senses(self, cfg: ScrollerConfig) -> None:
        """Register arrow, Page Up / Page Down, Home / End key senses."""
        from Draw._connectors import senses

        prefix = cfg.senses or cfg.id
        is_vert = cfg.direction in ("vertical", "both")
        is_horiz = cfg.direction in ("horizontal", "both")

        # Vertical Arrow keys
        if is_vert:
            arrow_down_id = f"{prefix}_key_down"
            senses("key_press", id=arrow_down_id, key=["Down"])
            cfg.sense_ids.append(arrow_down_id)

            arrow_up_id = f"{prefix}_key_up"
            senses("key_press", id=arrow_up_id, key=["Up"])
            cfg.sense_ids.append(arrow_up_id)

            # Page Up / Page Down
            pgdown_id = f"{prefix}_key_pgdn"
            senses("key_press", id=pgdown_id, key=["Page_Down", "PageDown"])
            cfg.sense_ids.append(pgdown_id)

            pgup_id = f"{prefix}_key_pgup"
            senses("key_press", id=pgup_id, key=["Page_Up", "PageUp"])
            cfg.sense_ids.append(pgup_id)

        # Horizontal Arrow keys
        if is_horiz:
            arrow_right_id = f"{prefix}_key_right"
            senses("key_press", id=arrow_right_id, key=["Right"])
            cfg.sense_ids.append(arrow_right_id)

            arrow_left_id = f"{prefix}_key_left"
            senses("key_press", id=arrow_left_id, key=["Left"])
            cfg.sense_ids.append(arrow_left_id)

        # Home / End
        home_id = f"{prefix}_key_home"
        senses("key_press", id=home_id, key=["Home"])
        cfg.sense_ids.append(home_id)

        end_id = f"{prefix}_key_end"
        senses("key_press", id=end_id, key=["End"])
        cfg.sense_ids.append(end_id)

    # ── connector wiring (step 2) ─────────────────────────────────────────

    def _register_connectors(self, cfg: ScrollerConfig) -> None:
        """Wire senses to scroll-offset updates via connectors."""
        from Draw._connectors import senses, connectors
        from Draw._align import calc

        prefix = cfg.connectors or cfg.id
        sense_prefix = cfg.senses or cfg.id
        tag = cfg.display
        speed = cfg.speed
        is_vert = cfg.direction in ("vertical", "both")
        is_horiz = cfg.direction in ("horizontal", "both")
        snap = cfg.snap

        # ── wheel scroll connectors ───────────────────────────────────────
        step_px = 40.0 * speed  # one notch ≈ 40 px

        def _make_wheel_work(delta_y: float, delta_x: float = 0.0):
            """Create a work callback for wheel and touch gestures."""
            def _work(record):
                if not cfg.enabled or cfg.locked:
                    return
                canvas = self._resolve_canvas(tag)
                if canvas is None:
                    return

                max_scroll_x, max_scroll_y = canvas._get_max_scroll_range() if hasattr(canvas, "_get_max_scroll_range") else (0.0, 0.0)
                bound_max_y = float(cfg.max_y) if cfg.max_y is not None else max(0.0, max_scroll_y)
                bound_max_x = float(cfg.max_x) if cfg.max_x is not None else max(0.0, max_scroll_x)

                if is_vert and delta_y != 0.0:
                    new_y = max(0.0, min(bound_max_y, canvas._scroll_y + delta_y))
                    if snap:
                        new_y = calc.snap_to_grid(new_y, snap)
                    canvas._scroll_y = new_y

                if is_horiz and (delta_x != 0.0 or not is_vert):
                    dx = delta_x if delta_x != 0.0 else delta_y
                    new_x = max(0.0, min(bound_max_x, canvas._scroll_x + dx))
                    if snap:
                        new_x = calc.snap_to_grid(new_x, snap)
                    canvas._scroll_x = new_x

                if hasattr(canvas, "_update_scroller_thumbs"):
                    canvas._update_scroller_thumbs()
                self._sync_room_offset(cfg, canvas._scroll_x, canvas._scroll_y)
                canvas.update()
            return _work

        # Scroll down
        down_conn_id = f"{prefix}_conn_down"
        down_sense = senses.get(f"{sense_prefix}_wheel_down")
        if down_sense:
            connectors(
                id=down_conn_id,
                sense=down_sense,
                work=_make_wheel_work(step_px),
            )
            cfg.connector_ids.append(down_conn_id)

        # Scroll up
        up_conn_id = f"{prefix}_conn_up"
        up_sense = senses.get(f"{sense_prefix}_wheel_up")
        if up_sense:
            connectors(
                id=up_conn_id,
                sense=up_sense,
                work=_make_wheel_work(-step_px),
            )
            cfg.connector_ids.append(up_conn_id)

        # ── keyboard connectors ───────────────────────────────────────────
        if cfg.keyboard:
            arrow_step = 30.0 * speed
            page_step = 300.0 * speed

            if is_vert:
                for suffix, sense_suf, dy, dx in [
                    ("conn_key_down", "key_down", arrow_step, 0.0),
                    ("conn_key_up", "key_up", -arrow_step, 0.0),
                    ("conn_key_pgdn", "key_pgdn", page_step, 0.0),
                    ("conn_key_pgup", "key_pgup", -page_step, 0.0),
                ]:
                    conn_id = f"{prefix}_{suffix}"
                    s = senses.get(f"{sense_prefix}_{sense_suf}")
                    if s:
                        connectors(
                            id=conn_id,
                            sense=s,
                            work=_make_wheel_work(dy, dx),
                        )
                        cfg.connector_ids.append(conn_id)

            if is_horiz:
                for suffix, sense_suf, dx in [
                    ("conn_key_right", "key_right", arrow_step),
                    ("conn_key_left", "key_left", -arrow_step),
                ]:
                    conn_id = f"{prefix}_{suffix}"
                    s = senses.get(f"{sense_prefix}_{sense_suf}")
                    if s:
                        connectors(
                            id=conn_id,
                            sense=s,
                            work=_make_wheel_work(0.0, dx),
                        )
                        cfg.connector_ids.append(conn_id)

            # Home → scroll to origin
            home_conn_id = f"{prefix}_conn_key_home"
            home_sense = senses.get(f"{sense_prefix}_key_home")
            if home_sense:
                def _home_work(record):
                    if not cfg.enabled or cfg.locked:
                        return
                    self.scroll_to(0, display=tag, direction=cfg.direction, ip=cfg.id)
                connectors(id=home_conn_id, sense=home_sense, work=_home_work)
                cfg.connector_ids.append(home_conn_id)

            # End → scroll to max bounds
            end_conn_id = f"{prefix}_conn_key_end"
            end_sense = senses.get(f"{sense_prefix}_key_end")
            if end_sense:
                def _end_work(record):
                    if not cfg.enabled or cfg.locked:
                        return
                    target_val = cfg.max_y if is_vert and cfg.max_y is not None else (
                        cfg.max_x if is_horiz and cfg.max_x is not None else 99999.0
                    )
                    self.scroll_to(target_val, display=tag, direction=cfg.direction, ip=cfg.id)
                connectors(id=end_conn_id, sense=end_sense, work=_end_work)
                cfg.connector_ids.append(end_conn_id)

    # ── thumb / track shape building (step 3) ─────────────────────────────

    def _build_thumb_shapes(
        self,
        cfg: ScrollerConfig,
        *,
        thumb_color: str,
        track_color: str,
        thumb_width: float,
        track_opacity: int,
    ) -> None:
        """Build visible scrollbar shapes using Draw.shapes scroller overlay."""
        from Draw._shapes import shapes as _sr

        prefix = cfg.id
        is_vert = cfg.direction in ("vertical", "both")
        is_horiz = cfg.direction in ("horizontal", "both")

        shapes_list = []

        if is_vert:
            place_v = "left" if cfg.align == "left" else "right"
            track_v = f"{prefix}_track_v"
            thumb_v = cfg.thumb_ip or f"{prefix}_thumb_v"
            cfg.track_ip = track_v
            cfg.thumb_ip = thumb_v

            shapes_list.append({
                "place": place_v,
                "ip": track_v,
                "thumb_ip": thumb_v,
                "color": track_color,
                "thumb_color": thumb_color,
                "width": thumb_width,
                "opacity": track_opacity,
                "max_y": cfg.max_y,
                "direction": "vertical",
            })

        if is_horiz and cfg.direction != "vertical":
            place_h = "top" if cfg.align == "top" else "bottom"
            track_h = f"{prefix}_track_h"
            thumb_h = f"{prefix}_thumb_h" if is_vert else (cfg.thumb_ip or f"{prefix}_thumb_h")

            shapes_list.append({
                "place": place_h,
                "ip": track_h,
                "thumb_ip": thumb_h,
                "color": track_color,
                "thumb_color": thumb_color,
                "height": thumb_width,
                "opacity": track_opacity,
                "max_x": cfg.max_x,
                "direction": "horizontal",
            })

        if shapes_list:
            try:
                _sr(
                    display=cfg.display,
                    properties=["scroller"],
                    shapes=shapes_list,
                )
            except Exception as exc:
                _logger.debug("Draw.scroller: thumb shape build skipped: %s", exc)

    # ── room & animation synchronization ─────────────────────────────────

    def _sync_room_offset(self, cfg: Optional[ScrollerConfig], x: float, y: float) -> None:
        """Synchronize room offset if a room IP / panel is linked."""
        if not cfg or not cfg.room:
            return
        room_ref = cfg.room
        try:
            from Draw._bridge import get_panel_registry
            panels = get_panel_registry()
            if isinstance(room_ref, str) and panels.get(room_ref):
                panels.move(room_ref, x=-int(round(x)), y=-int(round(y)))
        except Exception:
            pass

    def _animate_scroll_to(
        self,
        canvas: Any,
        target_x: float,
        target_y: float,
        duration: float = 0.25,
    ) -> None:
        """Smoothly interpolate canvas scroll offset to target coordinates."""
        start_x = getattr(canvas, "_scroll_x", 0.0)
        start_y = getattr(canvas, "_scroll_y", 0.0)
        start_time = time.perf_counter()

        def _step():
            elapsed = time.perf_counter() - start_time
            t = min(1.0, elapsed / max(0.01, duration))
            # Smooth ease-out cubic curve
            ease_t = 1.0 - (1.0 - t) ** 3
            canvas._scroll_x = start_x + (target_x - start_x) * ease_t
            canvas._scroll_y = start_y + (target_y - start_y) * ease_t
            canvas._update_scroller_thumbs()
            canvas.update()
            if t < 1.0:
                try:
                    from PySide6.QtCore import QTimer
                    QTimer.singleShot(16, _step)
                except Exception:
                    canvas._scroll_x = target_x
                    canvas._scroll_y = target_y
                    canvas._update_scroller_thumbs()
                    canvas.update()

        _step()


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level singleton
# ══════════════════════════════════════════════════════════════════════════════

scroller = _ScrollerRegistry()
