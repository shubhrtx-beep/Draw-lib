"""
Draw - A simple UI library built on PySide6
Part 1: Window Management
Part 2: Shapes
Part 3: Panel  (mini-windows inside a window)
Part 4: Checkpoint  (scene state management)
"""

from Draw._window      import window
from Draw._room        import room
from Draw._app         import get_app
from Draw._layout      import table, set as set_layout   # layout / grid
from Draw._widget      import generate_grid_items as grid
from Draw._shapes      import shapes
from Draw._shapes      import shapes as shape
from Draw._shapes      import hitbox
from Draw._shapes      import container
from Draw._shapes      import image, video
from Draw._loader      import loader
from Draw._text        import text
from Draw._text        import lineedit, textedit
from Draw._file_tree   import filetree
from Draw._native      import widget
from Draw._native      import box

from Draw._connectors  import connectors, senses
from Draw._calculator  import calculator, calculater  # calculater kept as alias
from Draw._clipboard  import copy as _cb_copy, read as _cb_read
from .                 import _clipboard as clipboard
from .                 import _filedialog as filedialog
from .                 import _tools as tools
from Draw._live        import live
from Draw._live        import input as input_field   # avoids shadowing built-in input()
from Draw._motion      import motion, custom, timeline
from Draw._scroller    import scroller
from Draw._align       import calc
from Draw._point       import point
from Draw._graph       import graph
from Draw._turtle      import turtle
from Draw._panel       import panel
from Draw._screen      import screen
from Draw._checkpoint  import checkpoint

from Draw._schedule     import after, every
from Draw._simulate     import simulate
from Draw._optimize     import optimize, performance_mode, set_performance_mode, performance_info

from Draw._colour import colour, color, save_tokens, load_tokens
from Draw._super import super_engine as super, super_mode
from Draw.debug import debug

__version__ = "0.1.0"

quit      = window.quit
close_all = window.close_all

onwindow = "onwindow"
onscreen = "onscreen"
onip     = "onip"


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
]
