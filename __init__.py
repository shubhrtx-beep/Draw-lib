"""
Draw - A simple UI library built on PySide6
Part 1: Window Management
Part 2: Shapes
Part 3: Panel  (mini-windows inside a window)
Part 4: Checkpoint  (scene state management)
"""

import importlib

from Draw._app import get_app
from Draw._window import window
from Draw._shapes import shapes, shapes as shape, hitbox, container, image, video
from Draw._text import text, lineedit, textedit
from Draw._layout import table, set as set_layout

__version__ = "0.1.0"

quit = window.quit
close_all = window.close_all

onwindow = "onwindow"
onscreen = "onscreen"
onip = "onip"

_LAZY_IMPORTS = {
    "graph": ("Draw._graph", "graph"),
    "turtle": ("Draw._turtle", "turtle"),
    "point": ("Draw._point", "point"),
    "connectors": ("Draw._connectors", "connectors"),
    "senses": ("Draw._connectors", "senses"),
    "calculator": ("Draw._calculator", "calculator"),
    "calculater": ("Draw._calculator", "calculater"),
    "motion": ("Draw._motion", "motion"),
    "custom": ("Draw._motion", "custom"),
    "timeline": ("Draw._motion", "timeline"),
    "scroller": ("Draw._scroller", "scroller"),
    "calc": ("Draw._align", "calc"),
    "panel": ("Draw._panel", "panel"),
    "screen": ("Draw._screen", "screen"),
    "checkpoint": ("Draw._checkpoint", "checkpoint"),
    "after": ("Draw._schedule", "after"),
    "every": ("Draw._schedule", "every"),
    "simulate": ("Draw._simulate", "simulate"),
    "optimize": ("Draw._optimize", "optimize"),
    "performance_mode": ("Draw._optimize", "performance_mode"),
    "set_performance_mode": ("Draw._optimize", "set_performance_mode"),
    "performance_info": ("Draw._optimize", "performance_info"),
    "colour": ("Draw._colour", "colour"),
    "color": ("Draw._colour", "color"),
    "save_tokens": ("Draw._colour", "save_tokens"),
    "load_tokens": ("Draw._colour", "load_tokens"),
    "super": ("Draw._super", "super_engine"),
    "super_mode": ("Draw._super", "super_mode"),
    "debug": ("Draw.debug", "debug"),
    "live": ("Draw._live", "live"),
    "input_field": ("Draw._live", "input"),
    "loader": ("Draw._loader", "loader"),
    "filetree": ("Draw._file_tree", "filetree"),
    "widget": ("Draw._native", "widget"),
    "box": ("Draw._native", "box"),
    "slider": ("Draw._native", "slider"),
    "button": ("Draw._native", "button"),
    "combobox": ("Draw._native", "combobox"),
    "checkbox": ("Draw._native", "checkbox"),
    "clipboard": ("Draw._clipboard", None),
    "filedialog": ("Draw._filedialog", None),
    "tools": ("Draw._tools", None),
    "grid": ("Draw._widget", "generate_grid_items"),
    "room": ("Draw._room", "room"),
    "profiler": ("Draw._profiler", "profiler"),
    "veo": ("Draw._veo", "veo"),
    "voe": ("Draw._voe", "voe"),
    "dust_remover": ("Draw._veo", "dust_remover"),
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        if attr_name is None:
            attr = module
        else:
            attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    try:
        submod = importlib.import_module(f".{name}", __name__)
        globals()[name] = submod
        return submod
    except ModuleNotFoundError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # window
    "__version__",
    "window",
    "get_app",
    "quit",
    "close_all",
    # layout
    "table",
    "set_layout",
    "grid",
    # shapes
    "shape",
    "shapes",
    "hitbox",
    "container",
    "image",
    "video",
    "loader",
    "text",
    "lineedit",
    "textedit",
    "filetree",
    "widget",
    "box",
    "slider",
    "button",
    "combobox",
    "checkbox",
    # graph / point
    "graph",
    "point",
    "turtle",
    # connectivity
    "connectors",
    "senses",
    # math
    "calculater",
    "calculator",
    "clipboard",
    "filedialog",
    "tools",
    # live / input
    "live",
    "input_field",
    # motion
    "motion",
    "custom",
    "timeline",
    # scroller
    "scroller",
    # calculation engine
    "calc",
    # colour
    "colour",
    "color",
    "save_tokens",
    "load_tokens",
    # room (relative layout)
    "room",
    # panel
    "panel",
    # screen (live display)
    "screen",
    # checkpoint
    "checkpoint",
    # templates (native Qt overlays)
    "templates",
    # scheduling
    "after",
    "every",
    # input simulation (testing/automation)
    "simulate",
    # performance / optimization / opengl engine
    "optimize",
    "performance_mode",
    "set_performance_mode",
    "performance_info",
    "super",
    "super_mode",
    # constants
    "onwindow",
    "onscreen",
    "onip",
    # debug system
    "debug",
    "profiler",
    # visibility optimization engine
    "veo",
    "voe",
    "dust_remover",
]
