"""
Draw._checkpoint  v1
====================
Scene-level state management for large Draw projects.
Snapshot, offload, reload, and swap canvas content by name.

CONCEPTS
--------
  checkpoint ip   A named slot that stores a canvas snapshot.
  snapshot        A frozen copy of all shapes + text on a canvas.
  offload         Clear the canvas (frees paint load) without losing the snapshot.
  reload          Restore a snapshot back onto the canvas.
  new             Offload current state then run a builder function.
  load            Restore a different (previously saved) checkpoint.

OPERATIONS
----------
    Draw.checkpoint(
        ip        = "scene-1",      # checkpoint name  (REQUIRED)
        display   = "main",         # target window / panel tag
        save      = True,           # snapshot current canvas NOW
        reload    = True,           # restore this checkpoint's snapshot
        offload   = True,           # clear canvas  (snapshot kept in memory)
        new       = build_fn,       # offload + call build_fn()
        load      = "scene-2",      # restore a DIFFERENT checkpoint
        path      = "saves/s1.pkl", # persist snapshot to disk (pickle)
        on_save   = fn,             # hook(ip, state) called after save
        on_load   = fn,             # hook(ip, state) called after load
        properties = {              # arbitrary metadata attached to snapshot
            "description": "Main menu",
            "version": 2,
        },
    )

TYPICAL PATTERNS
----------------
    # 1. Save current scene, swap to new scene
    Draw.checkpoint(ip="menu",  display="main", save=True)
    Draw.checkpoint(ip="game",  display="main", new=build_game_scene)

    # 2. Return to menu
    Draw.checkpoint(ip="menu",  display="main", reload=True)

    # 3. Offload heavy scene temporarily
    Draw.checkpoint(ip="map",   display="main", offload=True)
    # ... do other work ...
    Draw.checkpoint(ip="map",   display="main", reload=True)

    # 4. Save to disk
    Draw.checkpoint(ip="save1", display="main", save=True, path="saves/slot1.pkl")

    # 5. Load from disk
    Draw.checkpoint(ip="save1", display="main", load="save1", path="saves/slot1.pkl")

STATE ACCESS
------------
    state = Draw.checkpoint.get("scene-1")
    Draw.checkpoint.list()
    Draw.checkpoint.delete("scene-1")
    Draw.checkpoint.clear_all()
"""

from __future__ import annotations

import copy
import io
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_logger = logging.getLogger(__name__)


# ── restricted unpickler (security hardening) ─────────────────────────────────

# Only allow deserializing known-safe module/class pairs.  This prevents
# arbitrary code execution when loading checkpoint files from untrusted sources.
_SAFE_PICKLE_CLASSES: dict[str, set[str]] = {
    "Draw._checkpoint":   {"CheckpointState"},
    "Draw._shapes":       {"ShapeDef"},
    "Draw._text":         {"TextDef"},
    "Draw._layout":       {"TableLayout", "CombinedCellLayout"},
    "Draw._overlap":      {"Rect", "FlowSpec"},
    "PySide6.QtGui":      {"QColor"},
    "PySide6.QtCore":     {"QPointF", "QRectF"},
    "builtins":           {"dict", "list", "tuple", "set", "frozenset",
                           "int", "float", "str", "bool", "bytes",
                           "complex", "type", "NoneType"},
    "collections":        {"OrderedDict", "defaultdict"},
    "copy":               {"_reconstructor"},
    "copyreg":            {"_reconstructor"},
}


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that refuses to instantiate classes not on the allow-list."""

    def find_class(self, module: str, name: str) -> type:
        allowed = _SAFE_PICKLE_CLASSES.get(module)
        if allowed is not None and name in allowed:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Draw.checkpoint: blocked attempt to unpickle '{module}.{name}'. "
            f"Only known-safe Draw/Qt types are permitted."
        )


# ── CheckpointState ───────────────────────────────────────────────────────────

@dataclass
class CheckpointState:
    ip: str
    display: str                        # window/panel tag this belongs to
    properties: Dict[str, Any]

    # Stored canvas data — deep copies of ShapeDef / TextDef lists
    shape_items: list = field(default_factory=list)
    text_items:  list = field(default_factory=list)
    layout_items: list = field(default_factory=list)

    # ip → ShapeDef mapping (rebuilt on reload)
    shape_by_ip: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    saved_at: Optional[str] = None      # ISO timestamp of last save


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_canvas(display: Optional[str]):
    """Return (window_tag, win, canvas) for a display tag."""
    from Draw._window import window as _wr
    from Draw._text  import _get_or_create_canvas

    if display is None:
        tags = _wr.list_tags()
        if len(tags) == 1:
            display = tags[0]
        elif len(tags) > 1:
            raise ValueError("Draw.checkpoint: multiple windows — 'display' is required.")
        else:
            raise ValueError("Draw.checkpoint: no windows exist.")

    win    = _wr.get(display)
    canvas = _get_or_create_canvas(display, win)
    return display, win, canvas


def _snapshot_canvas(canvas) -> tuple[list, list, list, dict]:
    """Deep-copy current canvas state. Returns (shapes, texts, layouts, shape_by_ip)."""
    shapes  = copy.deepcopy(canvas.shape_items)
    texts   = copy.deepcopy(canvas.text_items)
    layouts = copy.deepcopy(getattr(canvas, "layout_items", []))
    by_ip: dict = {}
    for s in shapes:
        if s.ip is not None:
            by_ip[s.ip] = s
    return shapes, texts, layouts, by_ip


def _restore_canvas(canvas, state: CheckpointState) -> None:
    """Replace canvas content with the stored snapshot."""
    canvas.shape_items  = copy.deepcopy(state.shape_items)
    canvas.text_items   = copy.deepcopy(state.text_items)
    canvas.layout_items = copy.deepcopy(state.layout_items)

    # Rebuild fast-lookup dict
    if hasattr(canvas, "_shape_by_ip"):
        canvas._shape_by_ip.clear()
        for s in canvas.shape_items:
            if s.ip is not None:
                canvas._shape_by_ip[s.ip] = s
    if hasattr(canvas, "_shape_hash_by_ip"):
        canvas._shape_hash_by_ip.clear()

    canvas._occupied_dirty = True
    canvas.update()


def _clear_canvas(canvas) -> None:
    """Clear canvas without saving state."""
    canvas.shape_items  = []
    canvas.text_items   = []
    canvas.layout_items = []
    if hasattr(canvas, "_shape_by_ip"):
        canvas._shape_by_ip.clear()
    if hasattr(canvas, "_shape_hash_by_ip"):
        canvas._shape_hash_by_ip.clear()
    canvas._occupied_dirty = True
    canvas.update()


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── rest_ip helper ────────────────────────────────────────────────────────────

def _reset_all_ip_namespaces(display: Optional[str] = None) -> None:
    """
    Reset all IP namespaces so that IPs created after this point are
    independent of any IPs created before it.

    Clears:
      • Shape ip lookup caches on every canvas (or the named display's canvas)
      • Colour bindings registry
      • Motion connected-motions registry
      • Connector records registry
      • Sense records registry
    """
    from Draw._window import window as _wr

    # ── shapes: clear ip caches on canvases ──────────────────────────────────
    tags = [display] if display else _wr.list_tags()
    for tag in tags:
        try:
            win = _wr.get(tag)
            if hasattr(win, "_draw_canvas"):
                canvas = win._draw_canvas
                if hasattr(canvas, "_shape_by_ip"):
                    canvas._shape_by_ip.clear()
                if hasattr(canvas, "_shape_hash_by_ip"):
                    canvas._shape_hash_by_ip.clear()
        except Exception as _e:
            import warnings
            warnings.warn(f"Draw.checkpoint rest_ip: partial reset failed — {_e}", stacklevel=3)

    # ── colour bindings ───────────────────────────────────────────────────────
    try:
        from Draw._colour import colour as _colour_registry
        _colour_registry._bindings.clear()
    except Exception as _e:
        import warnings
        warnings.warn(f"Draw.checkpoint rest_ip: partial reset failed — {_e}", stacklevel=3)

    # ── motion bindings ───────────────────────────────────────────────────────
    try:
        from Draw._motion import motion as _motion_reg
        if hasattr(_motion_reg, "_connected_motions"):
            _motion_reg._connected_motions.clear()
    except Exception as _e:
        import warnings
        warnings.warn(f"Draw.checkpoint rest_ip: partial reset failed — {_e}", stacklevel=3)

    # ── connectors & senses ───────────────────────────────────────────────────
    try:
        from Draw._connectors import connectors as _conn_reg, senses as _sense_reg
        _conn_reg.clear()
        _sense_reg.clear()
    except Exception as _e:
        import warnings
        warnings.warn(f"Draw.checkpoint rest_ip: partial reset failed — {_e}", stacklevel=3)


# ── Registry ──────────────────────────────────────────────────────────────────

class _CheckpointRegistry:
    """
    Public API:  Draw.checkpoint(ip="...", ...)
    """

    def __init__(self):
        self._states: Dict[str, CheckpointState] = {}

    def __call__(
        self,
        *,
        ip: str = None,
        display:    Optional[str]      = None,
        save:       bool               = False,
        reload:     bool               = False,
        offload:    bool               = False,
        new:        Optional[Callable] = None,
        load:       Optional[str]      = None,
        path:       Optional[str]      = None,
        on_save:    Optional[Callable] = None,
        on_load:    Optional[Callable] = None,
        properties: Optional[Dict[str, Any]] = None,
        rest_ip:    bool               = False,
    ) -> Optional[CheckpointState]:
        """
        Perform one or more checkpoint operations.

        Operations execute in this order:
          rest_ip → save → offload → new → reload → load

        Parameters
        ----------
        ip          Name of this checkpoint slot (required unless rest_ip=True alone).
        display     Window or panel tag (auto-detected if only one window).
        save        Snapshot current canvas into this slot.
        offload     Clear the canvas (snapshot kept in memory / on disk).
        new         Callable: offload first, then call new() to rebuild canvas.
        reload      Restore THIS slot's snapshot onto the canvas.
        load        ip of a DIFFERENT slot to restore (cross-slot load).
        path        File path for persistent save/load (.pkl).
        on_save     Callback(ip, state) fired after a save.
        on_load     Callback(ip, state) fired after a reload/load.
        properties  Metadata dict attached to the snapshot.
        rest_ip     Reset IP namespace: all IPs defined before this call are
                    invalidated.  IPs created after this checkpoint are
                    independent — even if they share the same name string,
                    they belong to a new scope and cannot connect to the
                    old ones.  Internally this clears the shape ip lookup
                    caches, colour bindings, motion bindings, connector
                    records, and sense records so stale connections are
                    severed cleanly.
        """
        # ── 0. REST_IP — reset all IP namespaces ─────────────────────────────
        if rest_ip:
            _reset_all_ip_namespaces(display)

        if ip is None:
            # rest_ip-only call — no checkpoint slot needed
            if rest_ip:
                return None
            raise ValueError("Draw.checkpoint: 'ip' is required.")

        if not isinstance(ip, str) or not ip:
            raise ValueError("Draw.checkpoint: 'ip' must be a non-empty string.")

        tag, win, canvas = _get_canvas(display)

        # Ensure a state slot exists
        state = self._states.get(ip)
        if state is None:
            state = CheckpointState(
                ip         = ip,
                display    = tag,
                properties = dict(properties or {}),
            )
            self._states[ip] = state
        elif properties:
            state.properties.update(properties)

        # ── 1. SAVE ───────────────────────────────────────────────────────────
        if save:
            shapes, texts, layouts, by_ip = _snapshot_canvas(canvas)
            state.shape_items  = shapes
            state.text_items   = texts
            state.layout_items = layouts
            state.shape_by_ip  = by_ip
            state.display      = tag
            state.saved_at     = _now_iso()
            if path:
                self._save_to_file(state, path)
            if on_save:
                try:
                    on_save(ip, state)
                except Exception as e:
                    _logger.exception("Draw.checkpoint: on_save callback failed for %r", ip)

        # ── 2. OFFLOAD ────────────────────────────────────────────────────────
        if offload:
            _clear_canvas(canvas)

        # ── 3. NEW ────────────────────────────────────────────────────────────
        if new is not None:
            _clear_canvas(canvas)
            try:
                new()
            except Exception as e:
                _logger.exception("Draw.checkpoint: new() callback failed for %r", ip)

        # ── 4. RELOAD (this slot) ─────────────────────────────────────────────
        if reload:
            if not state.shape_items and not state.text_items and path:
                loaded = self._load_from_file(ip, path)
                if loaded:
                    state = loaded
                    self._states[ip] = state
            if state.shape_items or state.text_items:
                _restore_canvas(canvas, state)
            else:
                _logger.warning("Draw.checkpoint: %r has no saved state to reload.", ip)
            if on_load:
                try:
                    on_load(ip, state)
                except Exception as e:
                    _logger.exception("Draw.checkpoint: on_load callback failed for %r", ip)

        # ── 5. LOAD (different slot) ──────────────────────────────────────────
        if load is not None and load != ip:
            target = self._states.get(load)
            if target is None and path:
                target = self._load_from_file(load, path)
                if target:
                    self._states[load] = target
            if target is None:
                _logger.warning("Draw.checkpoint: checkpoint %r not found.", load)
            else:
                _restore_canvas(canvas, target)
                if on_load:
                    try:
                        on_load(load, target)
                    except Exception as e:
                        _logger.exception("Draw.checkpoint: on_load callback failed for %r", load)

        return state

    # ── disk persistence ─────────────────────────────────────────────────────

    def _save_to_file(self, state: CheckpointState, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            _logger.exception("Draw.checkpoint: failed to save to %r", path)

    def _load_from_file(self, ip: str, path: str) -> Optional[CheckpointState]:
        if not os.path.exists(path):
            _logger.warning("Draw.checkpoint: file not found: %r", path)
            return None
        try:
            with open(path, "rb") as f:
                state = _RestrictedUnpickler(f).load()
                if not isinstance(state, CheckpointState):
                    _logger.warning("Draw.checkpoint: %r does not contain a valid checkpoint.", path)
                    return None
                return state
        except Exception as e:
            _logger.exception("Draw.checkpoint: failed to load %r", path)
            return None

    # ── utility methods ──────────────────────────────────────────────────────

    def get(self, ip: str) -> Optional[CheckpointState]:
        """Return the CheckpointState for ip, or None."""
        return self._states.get(ip)

    def list(self) -> List[str]:
        """List all saved checkpoint ips."""
        return list(self._states.keys())

    def info(self, ip: str) -> Optional[Dict[str, Any]]:
        """Return a summary dict for a checkpoint (no shape data)."""
        s = self._states.get(ip)
        if s is None:
            return None
        return {
            "ip":           s.ip,
            "display":      s.display,
            "saved_at":     s.saved_at,
            "shape_count":  len(s.shape_items),
            "text_count":   len(s.text_items),
            "properties":   dict(s.properties),
        }

    def delete(self, ip: str) -> bool:
        """Delete a checkpoint slot (does NOT affect the canvas)."""
        if ip in self._states:
            del self._states[ip]
            return True
        return False

    def clear_all(self) -> None:
        """Delete all checkpoint slots."""
        self._states.clear()

    def rename(self, old_ip: str, new_ip: str) -> bool:
        """Rename a checkpoint slot."""
        if old_ip not in self._states:
            return False
        state = self._states.pop(old_ip)
        state.ip = new_ip
        self._states[new_ip] = state
        return True

    def copy(self, src_ip: str, dst_ip: str) -> Optional[CheckpointState]:
        """Duplicate a checkpoint under a new ip."""
        src = self._states.get(src_ip)
        if src is None:
            return None
        dup = copy.deepcopy(src)
        dup.ip = dst_ip
        self._states[dst_ip] = dup
        return dup

    def diff(self, ip_a: str, ip_b: str) -> Dict[str, Any]:
        """
        Lightweight diff between two snapshots.
        Returns counts of shapes/texts added, removed, common.
        """
        a = self._states.get(ip_a)
        b = self._states.get(ip_b)
        if a is None or b is None:
            return {"error": "one or both checkpoints not found"}

        a_ips = {s.ip for s in a.shape_items if s.ip}
        b_ips = {s.ip for s in b.shape_items if s.ip}

        return {
            "shapes_only_in_a":   sorted(a_ips - b_ips),
            "shapes_only_in_b":   sorted(b_ips - a_ips),
            "shapes_in_both":     sorted(a_ips & b_ips),
            "shape_count_a":      len(a.shape_items),
            "shape_count_b":      len(b.shape_items),
            "text_count_a":       len(a.text_items),
            "text_count_b":       len(b.text_items),
        }


# ── singleton ─────────────────────────────────────────────────────────────────

checkpoint = _CheckpointRegistry()
