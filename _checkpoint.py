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
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtGui import QColor
from Draw._colour import _parse_color

_logger = logging.getLogger(__name__)


# ── JSON Serialization & Deserialization Helpers (Safe Deserialization) ───────

def _color_to_str(c: Any) -> Optional[str]:
    if c is None:
        return None
    if hasattr(c, "name"):
        try:
            return c.name(QColor.NameFormat.HexArgb)
        except Exception:
            return str(c)
    return str(c)


def _str_to_color(s: Any, default: str = "#000000") -> QColor:
    if s is None:
        return _parse_color(default)
    return _parse_color(s)


def _serialize_shape(s: Any) -> dict:
    if not hasattr(s, "shape_type"):
        if isinstance(s, dict):
            return dict(s)
        return {}

    flow_val = s.flow
    if hasattr(flow_val, "enabled"):
        flow_val = {
            "__type__": "FlowSpec",
            "enabled": flow_val.enabled,
            "mode": flow_val.mode,
            "direction": flow_val.direction,
            "gap": flow_val.gap,
            "padding_x": flow_val.padding_x,
            "padding_y": flow_val.padding_y,
            "role": flow_val.role,
            "wrap": flow_val.wrap,
            "scope": flow_val.scope,
            "area_expand": list(flow_val.area_expand) if flow_val.area_expand else [0.0, 0.0],
            "area_move": flow_val.area_move,
        }

    return {
        "__type__": "ShapeDef",
        "vertices": s.vertices,
        "size_raw": s.size_raw,
        "border_radius_raw": s.border_radius_raw,
        "x": s.x,
        "y": s.y,
        "align": s.align,
        "rotation": float(s.rotation) if s.rotation is not None else 0.0,
        "color": _color_to_str(s.color),
        "border_color": _color_to_str(s.border_color),
        "border_width": s.border_width,
        "border_style": s.border_style,
        "opacity": s.opacity,
        "curve_mode": s.curve_mode,
        "bend": copy.deepcopy(s.bend),
        "bend_amount": s.bend_amount,
        "warp": copy.deepcopy(s.warp),
        "exclude": copy.deepcopy(s.exclude),
        "symmetry": copy.deepcopy(s.symmetry),
        "hitbox_mode": s.hitbox_mode,
        "hit_box": s.hit_box,
        "custom": copy.deepcopy(s.custom),
        "z": s.z,
        "overlap": s.overlap,
        "flow": flow_val,
        "ip": s.ip,
        "layout": s.layout,
        "cell": s.cell,
        "area_expand": list(s.area_expand) if s.area_expand else [0.0, 0.0],
        "area_move": s.area_move,
        "shape_type": s.shape_type,
        "src": s.src,
        "_video_loop": getattr(s, "_video_loop", True),
        "_video_autoplay": getattr(s, "_video_autoplay", True),
        "_video_muted": getattr(s, "_video_muted", False),
        "inside": s.inside,
        "move_path": s.move_path,
        "custom_vertices": copy.deepcopy(s.custom_vertices),
    }


def _deserialize_shape(d: dict) -> Any:
    from Draw._shapes import ShapeDef
    from Draw._overlap import FlowSpec

    flow_raw = d.get("flow")
    if isinstance(flow_raw, dict) and flow_raw.get("__type__") == "FlowSpec":
        flow_obj = FlowSpec(
            enabled=bool(flow_raw.get("enabled", False)),
            mode=str(flow_raw.get("mode", "horizontal")),
            direction=str(flow_raw.get("direction", "right")),
            gap=int(flow_raw.get("gap", 4)),
            padding_x=int(flow_raw.get("padding_x", 0)),
            padding_y=int(flow_raw.get("padding_y", 0)),
            role=str(flow_raw.get("role", "item")),
            wrap=bool(flow_raw.get("wrap", False)),
            scope=str(flow_raw.get("scope", "window")),
            area_expand=tuple(flow_raw.get("area_expand", (0.0, 0.0))),
            area_move=flow_raw.get("area_move"),
        )
    else:
        flow_obj = flow_raw

    area_exp = d.get("area_expand", (0.0, 0.0))
    if isinstance(area_exp, (list, tuple)) and len(area_exp) == 2:
        area_exp_tuple = (float(area_exp[0]), float(area_exp[1]))
    else:
        area_exp_tuple = (0.0, 0.0)

    return ShapeDef(
        vertices=d.get("vertices"),
        size_raw=d.get("size_raw"),
        border_radius_raw=d.get("border_radius_raw"),
        x=d.get("x"),
        y=d.get("y"),
        align=d.get("align"),
        rotation=float(d.get("rotation", 0.0)),
        color=_str_to_color(d.get("color"), "#FFFFFF"),
        border_color=_str_to_color(d.get("border_color"), "#000000"),
        border_width=int(d.get("border_width", 0)),
        border_style=str(d.get("border_style", "solid")),
        opacity=int(d.get("opacity", 100)),
        curve_mode=str(d.get("curve_mode", "line")),
        bend=d.get("bend", []),
        bend_amount=float(d.get("bend_amount", 40.0)),
        warp=d.get("warp"),
        exclude=d.get("exclude", []),
        symmetry=d.get("symmetry"),
        hitbox_mode=d.get("hitbox_mode"),
        hit_box=str(d.get("hit_box", "shape")),
        custom=d.get("custom"),
        z=d.get("z", 0),
        overlap=bool(d.get("overlap", True)),
        flow=flow_obj,
        ip=d.get("ip"),
        layout=d.get("layout"),
        cell=d.get("cell"),
        area_expand=area_exp_tuple,
        area_move=d.get("area_move"),
        shape_type=str(d.get("shape_type", "vector")),
        src=d.get("src"),
        _video_loop=bool(d.get("_video_loop", True)),
        _video_autoplay=bool(d.get("_video_autoplay", True)),
        _video_muted=bool(d.get("_video_muted", False)),
        inside=d.get("inside"),
        move_path=d.get("move_path"),
        custom_vertices=d.get("custom_vertices"),
    )


def _serialize_text(t: Any) -> dict:
    if not hasattr(t, "font_family"):
        if isinstance(t, dict):
            return dict(t)
        return {}

    flow_val = t.flow
    if hasattr(flow_val, "enabled"):
        flow_val = {
            "__type__": "FlowSpec",
            "enabled": flow_val.enabled,
            "mode": flow_val.mode,
            "direction": flow_val.direction,
            "gap": flow_val.gap,
            "padding_x": flow_val.padding_x,
            "padding_y": flow_val.padding_y,
            "role": flow_val.role,
            "wrap": flow_val.wrap,
            "scope": flow_val.scope,
            "area_expand": list(flow_val.area_expand) if flow_val.area_expand else [0.0, 0.0],
            "area_move": flow_val.area_move,
        }

    char_c = None
    if getattr(t, "char_colors", None):
        char_c = [_color_to_str(c) for c in t.char_colors]

    return {
        "__type__": "TextDef",
        "text": t.text,
        "ip": t.ip,
        "x": t.x,
        "y": t.y,
        "align": t.align,
        "font_family": t.font_family,
        "font_size": t.font_size,
        "bold": t.bold,
        "italic": t.italic,
        "underline": t.underline,
        "strikethrough": t.strikethrough,
        "letter_spacing": t.letter_spacing,
        "word_spacing": t.word_spacing,
        "line_height": t.line_height,
        "elide": t.elide,
        "color": _color_to_str(t.color),
        "opacity": t.opacity,
        "align_text": t.align_text,
        "background_color": _color_to_str(t.background_color),
        "background_padding": t.background_padding,
        "border_width": t.border_width,
        "border_color": _color_to_str(t.border_color),
        "border_radius": t.border_radius,
        "glow": t.glow,
        "glow_color": _color_to_str(t.glow_color),
        "glow_radius": t.glow_radius,
        "shadow": t.shadow,
        "shadow_color": _color_to_str(t.shadow_color),
        "shadow_offset": list(t.shadow_offset) if t.shadow_offset else [3, 3],
        "rotation": t.rotation,
        "animation": t.animation,
        "anim_duration": t.anim_duration,
        "anim_loop": t.anim_loop,
        "anim_delay": t.anim_delay,
        "anim_ease": t.anim_ease,
        "from_color": _color_to_str(t.from_color),
        "to_color": _color_to_str(t.to_color),
        "pulse_min": t.pulse_min,
        "pulse_max": t.pulse_max,
        "pulse_speed": t.pulse_speed,
        "slide_direction": t.slide_direction,
        "slide_distance": t.slide_distance,
        "shake_intensity": t.shake_intensity,
        "wave_amplitude": t.wave_amplitude,
        "wave_speed": t.wave_speed,
        "wave_char_offset": t.wave_char_offset,
        "source": str(t.source) if t.source is not None else None,
        "input_enabled": t.input_enabled,
        "input_take_input": t.input_take_input,
        "input_submit_keys": list(t.input_submit_keys) if t.input_submit_keys else ["return"],
        "input_type": t.input_type,
        "input_buffer": t.input_buffer,
        "input_placeholder": t.input_placeholder,
        "input_min_length": t.input_min_length,
        "input_max_length": t.input_max_length,
        "input_allow_empty": t.input_allow_empty,
        "input_transform": t.input_transform,
        "input_pattern": t.input_pattern,
        "input_live_update": t.input_live_update,
        "input_clear_on_submit": t.input_clear_on_submit,
        "input_caret": t.input_caret,
        "input_caret_blink": t.input_caret_blink,
        "input_caret_color": _color_to_str(t.input_caret_color),
        "input_caret_width": t.input_caret_width,
        "input_caret_height_ratio": t.input_caret_height_ratio,
        "input_caret_blink_interval": t.input_caret_blink_interval,
        "layout": t.layout,
        "cell": t.cell,
        "max_width": t.max_width,
        "auto_align_in_ip": t.auto_align_in_ip,
        "hitbox_ip": t.hitbox_ip,
        "min_font_size": t.min_font_size,
        "closest_rect_area": t.closest_rect_area,
        "flow": flow_val,
        "arc_radius": t.arc_radius,
        "arc_angle": t.arc_angle,
        "arc_direction": t.arc_direction,
        "arc_start": t.arc_start,
        "outline_width": t.outline_width,
        "outline_color": _color_to_str(t.outline_color),
        "text_gradient": copy.deepcopy(t.text_gradient),
        "char_colors": char_c,
        "z": t.z,
        "overlap": t.overlap,
        "html": getattr(t, "html", False),
    }


def _deserialize_text(d: dict) -> Any:
    from Draw._text import TextDef
    from Draw._overlap import FlowSpec

    flow_raw = d.get("flow")
    if isinstance(flow_raw, dict) and flow_raw.get("__type__") == "FlowSpec":
        flow_obj = FlowSpec(
            enabled=bool(flow_raw.get("enabled", False)),
            mode=str(flow_raw.get("mode", "horizontal")),
            direction=str(flow_raw.get("direction", "right")),
            gap=int(flow_raw.get("gap", 4)),
            padding_x=int(flow_raw.get("padding_x", 0)),
            padding_y=int(flow_raw.get("padding_y", 0)),
            role=str(flow_raw.get("role", "item")),
            wrap=bool(flow_raw.get("wrap", False)),
            scope=str(flow_raw.get("scope", "window")),
            area_expand=tuple(flow_raw.get("area_expand", (0.0, 0.0))),
            area_move=flow_raw.get("area_move"),
        )
    else:
        flow_obj = flow_raw

    so_raw = d.get("shadow_offset", (3, 3))
    if isinstance(so_raw, (list, tuple)) and len(so_raw) == 2:
        so_tuple = (int(so_raw[0]), int(so_raw[1]))
    else:
        so_tuple = (3, 3)

    bg_c = d.get("background_color")
    from_c = d.get("from_color")
    to_c = d.get("to_color")
    caret_c = d.get("input_caret_color")
    outline_c = d.get("outline_color")

    char_c_raw = d.get("char_colors")
    char_colors = [_str_to_color(c) for c in char_c_raw] if char_c_raw else None

    return TextDef(
        text=str(d.get("text", "")),
        ip=d.get("ip"),
        x=d.get("x"),
        y=d.get("y"),
        align=d.get("align"),
        font_family=str(d.get("font_family", "Arial")),
        font_size=int(d.get("font_size", 24)),
        bold=bool(d.get("bold", False)),
        italic=bool(d.get("italic", False)),
        underline=bool(d.get("underline", False)),
        strikethrough=bool(d.get("strikethrough", False)),
        letter_spacing=float(d.get("letter_spacing", 0.0)),
        word_spacing=float(d.get("word_spacing", 0.0)),
        line_height=float(d.get("line_height", 1.2)),
        elide=d.get("elide"),
        color=_str_to_color(d.get("color"), "#000000"),
        opacity=int(d.get("opacity", 100)),
        align_text=str(d.get("align_text", "left")),
        background_color=_str_to_color(bg_c) if bg_c else None,
        background_padding=int(d.get("background_padding", 6)),
        border_width=int(d.get("border_width", 0)),
        border_color=_str_to_color(d.get("border_color"), "#000000"),
        border_radius=float(d.get("border_radius", 0.0)),
        glow=bool(d.get("glow", False)),
        glow_color=_str_to_color(d.get("glow_color"), "#000000"),
        glow_radius=int(d.get("glow_radius", 12)),
        shadow=bool(d.get("shadow", False)),
        shadow_color=_str_to_color(d.get("shadow_color"), "#000000"),
        shadow_offset=so_tuple,
        rotation=float(d.get("rotation", 0.0)),
        animation=d.get("animation"),
        anim_duration=float(d.get("anim_duration", 1.0)),
        anim_loop=bool(d.get("anim_loop", True)),
        anim_delay=float(d.get("anim_delay", 0.0)),
        anim_ease=str(d.get("anim_ease", "linear")),
        from_color=_str_to_color(from_c) if from_c else None,
        to_color=_str_to_color(to_c) if to_c else None,
        pulse_min=float(d.get("pulse_min", 0.3)),
        pulse_max=float(d.get("pulse_max", 1.0)),
        pulse_speed=float(d.get("pulse_speed", 1.0)),
        slide_direction=str(d.get("slide_direction", "left")),
        slide_distance=float(d.get("slide_distance", 60.0)),
        shake_intensity=float(d.get("shake_intensity", 4.0)),
        wave_amplitude=float(d.get("wave_amplitude", 6.0)),
        wave_speed=float(d.get("wave_speed", 2.0)),
        wave_char_offset=float(d.get("wave_char_offset", 0.4)),
        source=d.get("source"),
        input_enabled=bool(d.get("input_enabled", False)),
        input_take_input=str(d.get("input_take_input", "return")),
        input_submit_keys=tuple(d.get("input_submit_keys", ("return",))),
        input_type=str(d.get("input_type", "all")),
        input_buffer=str(d.get("input_buffer", "")),
        input_placeholder=str(d.get("input_placeholder", "")),
        input_min_length=int(d.get("input_min_length", 0)),
        input_max_length=d.get("input_max_length"),
        input_allow_empty=bool(d.get("input_allow_empty", True)),
        input_transform=str(d.get("input_transform", "none")),
        input_pattern=d.get("input_pattern"),
        input_live_update=bool(d.get("input_live_update", False)),
        input_clear_on_submit=bool(d.get("input_clear_on_submit", False)),
        input_caret=bool(d.get("input_caret", True)),
        input_caret_blink=bool(d.get("input_caret_blink", True)),
        input_caret_color=_str_to_color(caret_c) if caret_c else None,
        input_caret_width=float(d.get("input_caret_width", 2.0)),
        input_caret_height_ratio=float(d.get("input_caret_height_ratio", 0.88)),
        input_caret_blink_interval=float(d.get("input_caret_blink_interval", 0.55)),
        layout=d.get("layout"),
        cell=d.get("cell"),
        max_width=float(d["max_width"]) if d.get("max_width") is not None else None,
        auto_align_in_ip=bool(d.get("auto_align_in_ip", False)),
        hitbox_ip=d.get("hitbox_ip"),
        min_font_size=int(d.get("min_font_size", 10)),
        closest_rect_area=bool(d.get("closest_rect_area", False)),
        flow=flow_obj,
        arc_radius=float(d["arc_radius"]) if d.get("arc_radius") is not None else None,
        arc_angle=float(d.get("arc_angle", 180.0)),
        arc_direction=str(d.get("arc_direction", "up")),
        arc_start=float(d["arc_start"]) if d.get("arc_start") is not None else None,
        outline_width=float(d.get("outline_width", 0.0)),
        outline_color=_str_to_color(outline_c) if outline_c else None,
        text_gradient=d.get("text_gradient"),
        char_colors=char_colors,
        z=d.get("z", 0),
        overlap=bool(d.get("overlap", True)),
        html=bool(d.get("html", False)),
    )


def _state_to_dict(state: CheckpointState) -> dict:
    return {
        "version": 1,
        "format": "Draw.checkpoint.json",
        "ip": state.ip,
        "display": state.display,
        "saved_at": state.saved_at,
        "properties": copy.deepcopy(state.properties),
        "shape_items": [_serialize_shape(s) for s in state.shape_items],
        "text_items": [_serialize_text(t) for t in state.text_items],
        "layout_items": copy.deepcopy(state.layout_items) if isinstance(state.layout_items, list) else [],
    }


def _state_from_dict(data: dict) -> CheckpointState:
    if not isinstance(data, dict):
        raise ValueError("Invalid checkpoint JSON data: expected object.")

    shapes = [_deserialize_shape(s) for s in data.get("shape_items", []) if isinstance(s, dict)]
    texts = [_deserialize_text(t) for t in data.get("text_items", []) if isinstance(t, dict)]
    layouts = data.get("layout_items", [])
    if not isinstance(layouts, list):
        layouts = []

    by_ip = {}
    for s in shapes:
        if s.ip is not None:
            by_ip[s.ip] = s

    return CheckpointState(
        ip=str(data.get("ip", "unnamed")),
        display=str(data.get("display", "main")),
        properties=dict(data.get("properties", {})),
        shape_items=shapes,
        text_items=texts,
        layout_items=layouts,
        shape_by_ip=by_ip,
        saved_at=data.get("saved_at"),
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
            payload = _state_to_dict(state)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            _logger.exception("Draw.checkpoint: failed to save to %r", path)

    def _load_from_file(self, ip: str, path: str) -> Optional[CheckpointState]:
        if not os.path.exists(path):
            _logger.warning("Draw.checkpoint: file not found: %r", path)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                _logger.warning("Draw.checkpoint: %r does not contain valid checkpoint JSON.", path)
                return None
            state = _state_from_dict(data)
            return state
        except Exception as e:
            _logger.exception("Draw.checkpoint: failed to load %r: %s", path, e)
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
