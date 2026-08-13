from __future__ import annotations
import copy
from typing import Any, List, Optional, Tuple, Dict
from Draw._shapes import shapes as draw_shape
from Draw._motion import motion as draw_motion
from Draw._checkpoint import _get_canvas

def loader(
    *,
    display: Optional[str] = None,
    tag: Optional[str] = None,
    ip: Optional[str] = None,
    shape: object = "circle",
    color: object = "cyan",
    motion: object = "spin",
    **kwargs
) -> None:
    """
    Exposes a loader (loading spinner / progress bar) on the display.
    Combines shape, color, and motion specifications under the hood.
    
    Parameters
    ----------
    display / tag: Display window tag.
    ip: Unique ID for the loader.
    shape: String ("circle", "ring", "bar"), custom shape dict/list, or IP of an existing shape.
    color: Color string/value, or IP of an existing color binding.
    motion: String ("spin", "pulse", "progress"), custom motion dict/list, or IP of an existing motion.
    kwargs: Extra shape properties (x, y, size, align, z, overlap, etc.)
    """
    target_display = display or tag
    
    # 1. Resolve active canvas
    try:
        _, _, canvas = _get_canvas(target_display)
    except Exception:
        canvas = None

    if ip is None:
        ip = f"loader_{id(kwargs)}"

    # 2. Resolve shape
    shape_list = []
    resolved_shape_by_ip = False
    
    if isinstance(shape, str) and canvas is not None and shape not in ("circle", "ring", "bar"):
        # Check if shape is an IP of an existing shape on the canvas
        existing_shape = None
        for s in canvas.shape_items:
            if s.ip == shape:
                existing_shape = s
                break
        
        if existing_shape is not None:
            # Duplicate the shape properties
            shape_list = [{
                "vertices": existing_shape.vertices,
                "size": list(existing_shape.size_raw) if isinstance(existing_shape.size_raw, (list, tuple)) else existing_shape.size_raw,
                "border_radius": existing_shape.border_radius_raw,
                "color": existing_shape.color,
                "border_color": existing_shape.border_color,
                "border_width": existing_shape.border_width,
                "border_style": existing_shape.border_style,
                "opacity": existing_shape.opacity,
                "rotation": existing_shape.rotation,
                "x": existing_shape.x,
                "y": existing_shape.y,
                "align": existing_shape.align,
                "z": existing_shape.z,
                "overlap": existing_shape.overlap,
                "curve_mode": existing_shape.curve_mode,
                "bend": existing_shape.bend,
                "warp": existing_shape.warp,
                "exclude": existing_shape.exclude,
                "symmetry": existing_shape.symmetry,
            }]
            resolved_shape_by_ip = True

    if not resolved_shape_by_ip:
        loader_color = color if color is not None else "cyan"
        if isinstance(shape, str):
            if shape in ("circle", "ring"):
                shape_list = [{
                    "vertices": 36,
                    "color": "transparent",
                    "border_width": 4,
                    "border_color": loader_color,
                    "border_style": "dashed" if shape == "circle" else "solid",
                    "size": [40, 40],
                }]
            elif shape == "bar":
                shape_list = [{
                    "vertices": 4,
                    "border_radius": 4,
                    "color": loader_color,
                    "size": [150, 8],
                }]
            else:
                shape_list = [{
                    "vertices": 36,
                    "color": loader_color,
                    "size": [40, 40],
                }]
        elif isinstance(shape, dict):
            s_dict = dict(shape)
            s_dict.setdefault("color", loader_color)
            shape_list = [s_dict]
        elif isinstance(shape, list):
            shape_list = [dict(s) for s in shape]

    for s in shape_list:
        s.setdefault("ip", ip)
        for k, v in kwargs.items():
            s[k] = v

    # 3. Resolve color (check if color parameter is an IP of an existing color binding)
    resolved_color_by_ip = False
    if isinstance(color, str):
        from Draw._colour import color as _color_registry
        existing_binding = _color_registry.get_binding(color)
        if existing_binding is not None:
            new_binding = copy.deepcopy(existing_binding)
            new_binding.color_ip = ip
            _color_registry._bindings[ip] = new_binding
            resolved_color_by_ip = True

    # If shape did not resolve from an existing shape IP, but color was resolved by IP,
    # update the generated shape color setting to the loader's IP so dynamic color applies
    if resolved_color_by_ip and not resolved_shape_by_ip:
        for s in shape_list:
            s["color"] = ip

    # 4. Resolve motion
    motion_spec = []
    existing_motions = None
    
    if isinstance(motion, str):
        from Draw._motion import motion as _motion_reg
        if motion in _motion_reg._connected_motions:
            existing_motions = _motion_reg._connected_motions[motion]
        elif motion in ("spin", "rotate"):
            motion_spec = [{
                "attribute": "rotate",
                "from": 0,
                "to": 360,
                "duration": 1.2,
                "repeat": True
            }]
        elif motion == "pulse":
            motion_spec = [{
                "attribute": "opacity",
                "from": 20,
                "to": 100,
                "duration": 1.0,
                "repeat": True,
                "reverse": True
            }]
        elif motion == "progress":
            motion_spec = [{
                "attribute": "scale",
                "from": [0.0, 1.0],
                "to": [1.0, 1.0],
                "duration": 2.0,
                "repeat": True
            }]
    elif isinstance(motion, dict):
        motion_spec = [motion]
    elif isinstance(motion, list):
        motion_spec = motion

    # 5. Execute shape rendering
    draw_shape(display=target_display, shape=shape_list)

    # 6. Apply motion
    if existing_motions is not None:
        from Draw._motion import motion as _motion_reg
        copied_motions = copy.deepcopy(existing_motions)
        _motion_reg._connected_motions[ip] = copied_motions
        if canvas is not None:
            for s in canvas.shape_items:
                if s.ip == ip:
                    if not hasattr(s, "_trigger_progresses"):
                        s._trigger_progresses = {}
                    s.motion = copied_motions
    elif motion_spec:
        draw_motion(ip=ip, motion=motion_spec)
