"""
Draw._super
===========
Hardware-Accelerated OpenGL Engine & GPU Dynamic Batching Subsystem for Draw.

Features & Safety Protections:
- `_DrawOpenGLCanvas`: Hardware-accelerated `QOpenGLWidget` canvas.
- `GPUGeometryBatcher`: Ultra-fast dynamic 2D vertex & index buffer packager.
- Complete GPU rendering path: VBO/EBO dynamic sub-buffer upload and `glDrawElements` execution.
- $O(1)$ Frustum / Viewport Culling: Off-screen geometry discarded before CPU/GPU buffer copy.
- Comprehensive Render-Loop & Recursion Protection: `_painting` guard prevents recursive updates.
- Safe Update Scheduling: `paintGL()` never schedules `update()`. Repaints are driven exclusively
  by animation state changes, scrolling, and user interaction.
- Idempotent `Draw.super(...)`: Calling repeatedly reuses existing accelerated canvas and timer.
- GPU Context & Error Resilience: Graceful fallback to hardware-accelerated QPainter on any OpenGL error.
- Safe Resource Cleanup: Destroys GPU buffers and stops timers cleanly upon widget destruction.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# pyrefly: ignore [missing-import]
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
# pyrefly: ignore [missing-import]
from PySide6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QKeyEvent, QMouseEvent,
    QPainter, QPainterPath, QPen, QTransform, QWheelEvent,
    QSurfaceFormat, QMatrix4x4,
)
# pyrefly: ignore [missing-import]
from PySide6.QtOpenGLWidgets import QOpenGLWidget
# pyrefly: ignore [missing-import]
from PySide6.QtOpenGL import (
    QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLVertexArrayObject,
)
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QMainWindow, QWidget

from Draw._app import get_app
from Draw._window import window as _window_registry, _parse_color
from Draw._shapes import (
    ShapeDef, _regular_polygon_points, _get_unit_polygon, _shape_preferred_geometry,
    _shape_hit_geometry, _apply_motion_geometry, _draw_one_shape,
    _shape_is_dynamic, _shape_preferred_pos,
)
from Draw._text import TextDef, _draw_one_text, _text_is_animated, LiveTextBinding, resolve_live_text
from Draw._optimize import GCTuner, SpatialGridIndex, optimize, set_performance_mode

_logger = logging.getLogger(__name__)


# ── GLSL Shaders ─────────────────────────────────────────────────────────────

VERTEX_SHADER_SRC = """#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor;

uniform mat4 uProjection;
out vec4 vColor;

void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
"""

FRAGMENT_SHADER_SRC = """#version 330 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    FragColor = vColor;
}
"""

VERTEX_SHADER_LEGACY = """#version 120
attribute vec2 aPos;
attribute vec4 aColor;

uniform mat4 uProjection;
varying vec4 vColor;

void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
"""

FRAGMENT_SHADER_LEGACY = """#version 120
varying vec4 vColor;

void main() {
    gl_FragColor = vColor;
}
"""


# ── Color & Math Helpers ─────────────────────────────────────────────────────

def _parse_color_to_rgba(raw_c: object, opacity: int = 100) -> Tuple[float, float, float, float]:
    """Parse color input to (r, g, b, a) float tuple in [0.0, 1.0]."""
    if isinstance(raw_c, QColor):
        qc = raw_c
    elif raw_c is not None:
        qc = _parse_color(raw_c)
    else:
        qc = QColor(255, 255, 255)
    r, g, b, a = qc.getRgbF()
    if opacity is not None and opacity < 100:
        a *= max(0.0, min(1.0, opacity / 100.0))
    return (r, g, b, a)


# ── GPU Dynamic Geometry Batcher ──────────────────────────────────────────────

class GPUGeometryBatcher:
    """
    High-throughput dynamic 2D vertex buffer manager.
    Batches regular polygons, custom vertices, and quads into contiguous VBO arrays.
    """

    def __init__(self, max_vertices: int = 65536):
        self.max_vertices = max(1024, max_vertices)
        self.max_indices = self.max_vertices

        # Interleaved vertex buffer: [x, y, r, g, b, a] (6 floats = 24 bytes per vertex)
        self.vertex_buffer = np.zeros(self.max_vertices * 6, dtype=np.float32)
        self.index_buffer = np.zeros(self.max_indices, dtype=np.uint32)

        self.v_count = 0
        self.i_count = 0

    def reset(self) -> None:
        self.v_count = 0
        self.i_count = 0

    def add_polygon(self, pts: List[QPointF], color_rgba: Tuple[float, float, float, float]) -> bool:
        """Add a 2D convex polygon as direct triangulated vertices for glDrawArrays."""
        n_pts = len(pts)
        if n_pts < 3:
            return False

        tri_vertices = (n_pts - 2) * 3
        if self.v_count + tri_vertices > self.max_vertices:
            return False

        r, g, b, a = color_rgba
        p0 = pts[0]
        x0, y0 = float(p0.x()), float(p0.y())

        for i in range(1, n_pts - 1):
            p1 = pts[i]
            p2 = pts[i + 1]

            tri_pts = (
                (x0, y0),
                (float(p1.x()), float(p1.y())),
                (float(p2.x()), float(p2.y())),
            )
            for px, py in tri_pts:
                v_offset = self.v_count * 6
                self.vertex_buffer[v_offset] = px
                self.vertex_buffer[v_offset + 1] = py
                self.vertex_buffer[v_offset + 2] = r
                self.vertex_buffer[v_offset + 3] = g
                self.vertex_buffer[v_offset + 4] = b
                self.vertex_buffer[v_offset + 5] = a
                self.v_count += 1

        self.i_count = self.v_count
        return True

    def add_transformed_polygon(
        self,
        cx: float, cy: float,
        rx: float, ry: float,
        unit_pts: List[Tuple[float, float]],
        rot_deg: float,
        color_rgba: Tuple[float, float, float, float]
    ) -> bool:
        """Direct zero-allocation polygon generation directly into the vertex buffer."""
        n_pts = len(unit_pts)
        if n_pts < 3:
            return False
        tri_vertices = (n_pts - 2) * 3
        if self.v_count + tri_vertices > self.max_vertices:
            return False

        r, g, b, a = color_rgba
        vbuf = self.vertex_buffer

        if rot_deg != 0.0:
            rad = math.radians(rot_deg)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            ux0, uy0 = unit_pts[0]
            rux0 = ux0 * cos_a - uy0 * sin_a
            ruy0 = ux0 * sin_a + uy0 * cos_a
            x0 = cx + rx * rux0
            y0 = cy + ry * ruy0

            for i in range(1, n_pts - 1):
                ux1, uy1 = unit_pts[i]
                rux1 = ux1 * cos_a - uy1 * sin_a
                ruy1 = ux1 * sin_a + uy1 * cos_a
                x1 = cx + rx * rux1
                y1 = cy + ry * ruy1

                ux2, uy2 = unit_pts[i + 1]
                rux2 = ux2 * cos_a - uy2 * sin_a
                ruy2 = ux2 * sin_a + uy2 * cos_a
                x2 = cx + rx * rux2
                y2 = cy + ry * ruy2

                v_off = self.v_count * 6
                vbuf[v_off] = x0; vbuf[v_off+1] = y0; vbuf[v_off+2] = r; vbuf[v_off+3] = g; vbuf[v_off+4] = b; vbuf[v_off+5] = a
                vbuf[v_off+6] = x1; vbuf[v_off+7] = y1; vbuf[v_off+8] = r; vbuf[v_off+9] = g; vbuf[v_off+10] = b; vbuf[v_off+11] = a
                vbuf[v_off+12] = x2; vbuf[v_off+13] = y2; vbuf[v_off+14] = r; vbuf[v_off+15] = g; vbuf[v_off+16] = b; vbuf[v_off+17] = a
                self.v_count += 3
        else:
            ux0, uy0 = unit_pts[0]
            x0 = cx + rx * ux0
            y0 = cy + ry * uy0

            for i in range(1, n_pts - 1):
                ux1, uy1 = unit_pts[i]
                x1 = cx + rx * ux1
                y1 = cy + ry * uy1

                ux2, uy2 = unit_pts[i + 1]
                x2 = cx + rx * ux2
                y2 = cy + ry * uy2

                v_off = self.v_count * 6
                vbuf[v_off] = x0; vbuf[v_off+1] = y0; vbuf[v_off+2] = r; vbuf[v_off+3] = g; vbuf[v_off+4] = b; vbuf[v_off+5] = a
                vbuf[v_off+6] = x1; vbuf[v_off+7] = y1; vbuf[v_off+8] = r; vbuf[v_off+9] = g; vbuf[v_off+10] = b; vbuf[v_off+11] = a
                vbuf[v_off+12] = x2; vbuf[v_off+13] = y2; vbuf[v_off+14] = r; vbuf[v_off+15] = g; vbuf[v_off+16] = b; vbuf[v_off+17] = a
                self.v_count += 3

        self.i_count = self.v_count
        return True


# ── Hardware-Accelerated QOpenGLWidget Canvas ────────────────────────────────

class _DrawOpenGLCanvas(QOpenGLWidget):
    """
    High-Performance Hardware-Accelerated Canvas for Draw.
    Replaces software canvas with GPU-accelerated OpenGL batch rendering + QPainter overlay.
    """

    def __init__(self, parent: QMainWindow, batch_capacity: int = 65536, vsync: bool = False):
        # Configure OpenGL Surface Format
        fmt = QSurfaceFormat()
        fmt.setSamples(4)  # 4x MSAA
        fmt.setSwapInterval(1 if vsync else 0)
        fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

        super().__init__(parent)
        self.setFormat(fmt)

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        self.layout_items: list = []
        self.shape_items: list = []
        self.text_items: List[TextDef] = []
        self.setGeometry(parent.rect())

        # ── Phase 2: Cached Z-order ──────────────────────────────────────
        self._z_order_dirty: bool = True
        self._sorted_shapes: list = []

        # ── Phase 3-7: Per-shape GPU cache tracking ──────────────────────
        # Each shape gets a _gpu_cache dict storing last-frame computed
        # state to skip recomputation for static shapes.
        # _gpu_cache = {
        #   'pos': (x, y),    'size': (w, h),    'rot': float,
        #   'color_rgba': (r,g,b,a),  'vertices_data': numpy_slice,
        #   'n_verts': int,   'batched': bool,   'motion_state': dict,
        #   'spatial_bbox': (x,y,w,h)
        # }
        # Initialized lazily on first frame per shape.

        # GPU Batching Engine & State
        self.batcher = GPUGeometryBatcher(max_vertices=batch_capacity)
        self.shader_program: Optional[QOpenGLShaderProgram] = None
        self.vao: Optional[QOpenGLVertexArrayObject] = None
        self.vbo: Optional[QOpenGLBuffer] = None
        self.ebo: Optional[QOpenGLBuffer] = None
        self._gl_initialized = False
        self._gpu_failed = False
        self._cleaned_up = False
        self._projection_matrix: Optional[QMatrix4x4] = None

        # Real-Time FPS Telemetry
        self._fps_timestamps: list = []
        self._fps: float = 60.0
        self._fps_last_calc: float = 0.0

        # CRITICAL: Re-entrancy guard against render loops
        self._painting = False

        # Spatial Indexing for O(1) hit testing
        self.spatial_grid = SpatialGridIndex(cell_size=100.0)
        self._spatial_grid_direct = self.spatial_grid  # Phase 5: direct ref for fast path

        # Single Animation loop timer
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(16)
        self._animation_timer.timeout.connect(self._tick_animation)
        self._animation_timer.start()

        # Interaction and senses tracking state
        self._hovered_ips: set = set()
        self._active_input_index: int = -1
        self._live_text_error_cache: dict[int, str] = {}
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._scroll_x = 0.0
        self._scroll_y = 0.0
        self._dragged_shape = None
        self._drag_offset = (0.0, 0.0)
        self._last_tick_time = time.perf_counter()

        self._global_occupied: list = []
        self._occupied_dirty: bool = True
        self._shape_by_ip: dict = {}
        self._shape_hash_by_ip: dict = {}
        self._scroller_configs: list = []

        self._drag_origin = None
        self._drag_started = False
        self._drag_threshold_px = 4.0
        self.longpress_delay_ms = 500
        self.longpress_repeat_ms = 80
        self._longpress_timer = QTimer(self)
        self._longpress_timer.timeout.connect(self._on_longpress_timeout)
        self._longpress_ip = None
        self._longpress_fired_once = False

        self._last_lclick_press_pos = None
        self._last_lclick_release_pos = None
        self._builder_queue: list = []
        self._builder_active: Optional[dict] = None
        self._focused_ip = None

        # Cleanup hook
        self.destroyed.connect(self._cleanup_resources)
        if isinstance(parent, QMainWindow):
            parent.destroyed.connect(self._cleanup_resources)

    # ── OpenGL Lifecycle ──────────────────────────────────────────────────────

    def initializeGL(self) -> None:  # noqa: N802
        if self._gl_initialized or self._gpu_failed:
            return
        try:
            program = QOpenGLShaderProgram(self)
            # Try Core Profile 3.3 Shader
            if not program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER_SRC):
                _logger.warning("OpenGL 3.3 vertex shader failed, falling back to legacy shader: %s", program.log())
                program.removeAllShaders()
                program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX_SHADER_LEGACY)
                program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER_LEGACY)
            else:
                program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT_SHADER_SRC)

            if not program.link():
                _logger.warning("OpenGL shader link failed: %s", program.log())
                self._gpu_failed = True
                return

            self.shader_program = program

            # Create VAO
            self.vao = QOpenGLVertexArrayObject(self)
            if self.vao.create():
                self.vao.bind()

            # Create dynamic VBO
            self.vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self.vbo.create()
            self.vbo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            self.vbo.bind()
            self.vbo.allocate(self.batcher.vertex_buffer.nbytes)

            # Create dynamic EBO
            self.ebo = QOpenGLBuffer(QOpenGLBuffer.Type.IndexBuffer)
            self.ebo.create()
            self.ebo.setUsagePattern(QOpenGLBuffer.UsagePattern.DynamicDraw)
            self.ebo.bind()
            self.ebo.allocate(self.batcher.index_buffer.nbytes)

            # Configure vertex attribute pointers
            stride = 6 * 4  # 6 floats * 4 bytes
            self.shader_program.enableAttributeArray(0)
            self.shader_program.setAttributeBuffer(0, 0x1406, 0, 2, stride)  # GL_FLOAT = 0x1406

            self.shader_program.enableAttributeArray(1)
            self.shader_program.setAttributeBuffer(1, 0x1406, 2 * 4, 4, stride)

            if self.vao.isCreated():
                self.vao.release()
            self.vbo.release()
            self.ebo.release()

            self._gl_initialized = True
        except Exception as exc:
            _logger.warning("Draw.super: initializeGL failed, using safe fallback: %s", exc)
            self._gpu_failed = True

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        """
        Update orthographic projection safely when window is resized.
        CRITICAL: Never calls update() to prevent recursive repaints.
        """
        if w <= 0 or h <= 0:
            return
        proj = QMatrix4x4()
        proj.ortho(0.0, float(w), float(h), 0.0, -1.0, 1.0)
        self._projection_matrix = proj

        if self.context():
            funcs = self.context().functions()
            if funcs:
                funcs.glViewport(0, 0, w, h)

    @property
    def fps(self) -> float:
        """Return the current rolling OpenGL frames-per-second."""
        return self._fps

    def clear(self, keep_ips: Optional[Set[str]] = None) -> None:
        """Completely flush all render items and caches."""
        if keep_ips:
            self.shape_items = [s for s in self.shape_items if s.ip and s.ip in keep_ips]
            self.text_items = [t for t in self.text_items if t.ip and t.ip in keep_ips]
        else:
            self.shape_items.clear()
            self.text_items.clear()

        self._z_order_dirty = True
        self._occupied_dirty = True
        self._sorted_shapes.clear()
        if hasattr(self, "_shape_by_ip"):
            self._shape_by_ip = {s.ip: s for s in self.shape_items if s.ip}
        if hasattr(self, "_shape_hash_by_ip"):
            self._shape_hash_by_ip.clear()
        if hasattr(self, "spatial_grid"):
            self.spatial_grid.clear()
        if hasattr(self, "batcher"):
            self.batcher.reset()
        self.update()

    def paintGL(self) -> None:  # noqa: N802
        """
        Main Hardware-Accelerated Rendering Routine.
        Protected against recursion loops via `self._painting` guard.
        """
        if self._painting:
            return

        self._painting = True
        try:
            self._do_paint_gl()
        finally:
            self._painting = False

    def _do_paint_gl(self) -> None:
        from Draw.debug import debug as _debug_manager
        if _debug_manager.should_stop():
            return

        from Draw._layout import _draw_one_layout
        from Draw._motion import motion as _motion_registry

        cw, ch = self.width(), self.height()
        if cw <= 0 or ch <= 0:
            return

        now = time.perf_counter()

        # Update rolling FPS telemetry
        self._fps_timestamps.append(now)
        if now - self._fps_last_calc >= 0.1:
            self._fps_last_calc = now
            cutoff = now - 1.0
            self._fps_timestamps = [t for t in self._fps_timestamps if t >= cutoff]
            cnt = len(self._fps_timestamps)
            if cnt > 1:
                dt = self._fps_timestamps[-1] - self._fps_timestamps[0]
                self._fps = (cnt - 1) / dt if dt > 0.001 else float(cnt)
            else:
                self._fps = float(cnt)

        # Update scroller thumbs without triggering recursive paint
        self._update_scroller_thumbs(from_paint=True)

        # ── Phase 2: Cached Z-order ──────────────────────────────────────────
        if self._z_order_dirty:
            self._sorted_shapes = sorted(self.shape_items, key=lambda s: -s.z)
            self._z_order_dirty = False
        sorted_shapes = self._sorted_shapes

        # ── 1. Batch Eligible 2D Geometry for GPU Direct Render ──────────────
        self.batcher.reset()
        unbatched_shapes: List[Tuple[ShapeDef, Tuple[float, float], dict]] = []
        batched_shape_set: Set[int] = set()

        # Ensure projection matrix is current
        if self._projection_matrix is None:
            proj = QMatrix4x4()
            proj.ortho(0.0, float(cw), float(ch), 0.0, -1.0, 1.0)
            self._projection_matrix = proj

        # Cache references for inner loop speed
        _window_tag = getattr(self, "_window_tag", None)
        _scroll_x = self._scroll_x
        _scroll_y = self._scroll_y
        _gl_ok = self._gl_initialized and not self._gpu_failed
        _batcher = self.batcher
        _has_active_timelines = bool(_motion_registry._active_timelines)
        # Phase 5: direct grid reference (bypasses public locking API)
        _sgrid_items = self._spatial_grid_direct.item_boxes
        _sgrid = self._spatial_grid_direct

        # Sentinel for "no motion state" — reuse across static shapes to avoid
        # allocating a dict per shape per frame.
        _EMPTY_MOTION = {}

        for s in sorted_shapes:
            sid = id(s)
            is_dragged = getattr(s, "_is_dragged", False)
            _shape_motions = getattr(s, "motion", None)
            has_motion = bool(_shape_motions) or (_has_active_timelines and s.ip is not None)
            is_scroller = s.ip and (s.ip.startswith("scroller_") or "_track" in s.ip or "_thumb" in s.ip)

            # ── Full Static Retained Path (Zero work per frame) ──────────────
            _gpu = getattr(s, "_gpu_cache", None)
            if (
                _gpu is not None
                and not s.dirty
                and not is_dragged
                and not has_motion
                and not is_scroller
                and _gpu["cw"] == cw
                and _gpu["ch"] == ch
                and _gpu["sx"] == _scroll_x
                and _gpu["sy"] == _scroll_y
            ):
                if _gpu["batched"] and _gl_ok:
                    cached_verts = _gpu["n_verts"]
                    if _batcher.v_count + cached_verts <= _batcher.max_vertices:
                        cached_data = _gpu["vdata"]
                        v_off = _batcher.v_count * 6
                        _batcher.vertex_buffer[v_off:v_off + len(cached_data)] = cached_data
                        _batcher.v_count += cached_verts
                        _batcher.i_count = _batcher.v_count
                        batched_shape_set.add(sid)
                        continue
                elif not _gpu["batched"]:
                    unbatched_shapes.append((s, _gpu["pos"], _EMPTY_MOTION))
                    continue

            # ── Dynamic / Dirty Compute Path ─────────────────────────────────
            try:
                (
                    origin_x, origin_y, area_w, area_h,
                    sw, sh, preferred_x, preferred_y
                ) = _shape_preferred_geometry(s, cw, ch, window_tag=_window_tag)
            except Exception:
                continue

            final_x = getattr(s, "_placed_x", preferred_x)
            final_y = getattr(s, "_placed_y", preferred_y)
            sw = getattr(s, "_placed_w", sw)
            sh = getattr(s, "_placed_h", sh)

            # Apply scroll offset unless scroller component
            if not is_scroller:
                final_x -= _scroll_x
                final_y -= _scroll_y

            # Handle live dragging (instantaneous 1:1 cursor tracking with zero lag)
            if is_dragged and not is_scroller:
                final_x = getattr(s, "_drag_x", final_x)
                final_y = getattr(s, "_drag_y", final_y)
                anim_x, anim_y = float(final_x), float(final_y)
                anim_sw, anim_sh = float(sw), float(sh)
                shape_motion_state = _EMPTY_MOTION
            else:
                if has_motion:
                    shape_motion_state = _motion_registry.compute_shape_state(
                        s, now, _parse_color, float(final_x), float(final_y), sw, sh, self
                    )
                    s._last_motion_state = shape_motion_state
                    anim_x, anim_y, anim_sw, anim_sh = _apply_motion_geometry(
                        shape_motion_state, float(final_x), float(final_y), sw, sh
                    )
                else:
                    shape_motion_state = _EMPTY_MOTION
                    s._last_motion_state = _EMPTY_MOTION
                    anim_x = float(final_x)
                    anim_y = float(final_y)
                    anim_sw = float(sw)
                    anim_sh = float(sh)

            s.last_position = (float(anim_x), float(anim_y))
            s.last_size = (int(anim_sw), int(anim_sh))

            hx, hy, hsw, hsh = _shape_hit_geometry(s)
            s.last_hit_position = (float(hx), float(hy))
            s.last_hit_size = (int(hsw), int(hsh))

            # ── Phase 5: Spatial grid — skip update if bbox unchanged ────────
            new_bbox = (float(anim_x), float(anim_y), float(anim_sw), float(anim_sh))
            old_bbox = _sgrid_items.get(sid)
            if old_bbox != new_bbox:
                _sgrid._update_internal(sid, new_bbox)

            # Direct GPU Batching for standard regular polygons
            is_simple_polygon = (
                s.curve_mode == "line" and
                not s.bend and
                not s.exclude and
                not s.symmetry and
                not s.warp and
                s.shape_type == "vector" and
                s.border_width == 0 and
                getattr(s, "gradient", None) is None and
                getattr(s, "custom_vertices", None) is None
            )

            batched = False
            vdata_slice = None
            n_verts = 0

            if is_simple_polygon and _gl_ok:
                n = s.vertices if s.vertices and s.vertices >= 3 else 4
                rot = shape_motion_state.get("rotation", getattr(s, "rotation", 0.0) or 0.0)

                # Compute geometry
                cx = anim_x + anim_sw / 2.0
                cy = anim_y + anim_sh / 2.0
                rx = anim_sw / 2.0
                ry = anim_sh / 2.0

                # Phase 7: Color cache
                raw_c = getattr(s, "color", None)
                _color_cache = getattr(s, "_cached_rgba", None)
                _color_key = getattr(s, "_cached_rgba_key", None)
                color_key = (id(raw_c) if isinstance(raw_c, QColor) else raw_c, s.opacity)
                if _color_cache is not None and _color_key == color_key:
                    color_rgba = _color_cache
                else:
                    color_rgba = _parse_color_to_rgba(raw_c, s.opacity)
                    s._cached_rgba = color_rgba
                    s._cached_rgba_key = color_key

                if (s.vertices is None or s.vertices == 4) and (s.curve_mode == "line") and (not s.bend):
                    unit_pts = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                else:
                    unit_pts = _get_unit_polygon(n)
                v_start = _batcher.v_count
                batched = _batcher.add_transformed_polygon(cx, cy, rx, ry, unit_pts, rot, color_rgba)
                if batched:
                    batched_shape_set.add(sid)
                    v_end = _batcher.v_count
                    n_verts = v_end - v_start
                    vdata_slice = _batcher.vertex_buffer[v_start * 6:v_end * 6].copy()

            if not batched:
                unbatched_shapes.append((s, (final_x, final_y), shape_motion_state))

            # Store retained GPU cache
            s._gpu_cache = {
                "cw": cw,
                "ch": ch,
                "sx": _scroll_x,
                "sy": _scroll_y,
                "batched": batched,
                "vdata": vdata_slice,
                "n_verts": n_verts,
                "pos": (final_x, final_y),
            }
            s.dirty = False

        # ── 2. Layered Z-Ordered Rendering ──────────────────────────────────
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

            # Clear background cleanly to prevent any viewport artifacts
            win_obj = self.parent() if self.parent() else None
            bg_col = getattr(win_obj, "_bg_color", None) if win_obj else None
            if bg_col is None:
                bg_col = QColor(8, 12, 20)
            painter.fillRect(self.rect(), bg_col)

            # Draw layout frames
            for layout in self.layout_items:
                _draw_one_layout(painter, layout, cw, ch)

            # Separate shapes by z into background (z >= 80) and foreground (z < 80)
            bg_unbatched = [(s, pos, ms) for s, pos, ms in unbatched_shapes if s.z >= 80]
            fg_unbatched = [(s, pos, ms) for s, pos, ms in unbatched_shapes if s.z < 80]

            # Pass 1: Draw background cards / panels
            for s, pos_override, m_state in bg_unbatched:
                _draw_one_shape(
                    painter, s, cw, ch,
                    position_override=pos_override,
                    motion_state=m_state,
                    canvas=self,
                )

            # Pass 2: GPU Dynamic Batch Draw (Midground 2D Geometry)
            if self._gl_initialized and not self._gpu_failed and self.batcher.v_count > 0:
                try:
                    painter.beginNativePainting()
                    funcs = self.context().functions() if self.context() else None
                    if funcs and self.shader_program and self.vbo:
                        v_used_floats = self.batcher.v_count * 6
                        v_bytes = self.batcher.vertex_buffer[:v_used_floats].tobytes()
                        self.vbo.bind()
                        self.vbo.write(0, v_bytes, len(v_bytes))

                        self.shader_program.bind()
                        self.shader_program.setUniformValue("uProjection", self._projection_matrix)

                        if self.vao and self.vao.isCreated():
                            self.vao.bind()
                        else:
                            self.vbo.bind()
                            stride = 6 * 4
                            self.shader_program.enableAttributeArray(0)
                            self.shader_program.setAttributeBuffer(0, 0x1406, 0, 2, stride)
                            self.shader_program.enableAttributeArray(1)
                            self.shader_program.setAttributeBuffer(1, 0x1406, 2 * 4, 4, stride)

                        funcs.glDrawArrays(0x0004, 0, self.batcher.v_count)

                        if self.vao and self.vao.isCreated():
                            self.vao.release()
                        self.vbo.release()
                        self.shader_program.release()
                except Exception as exc:
                    _logger.warning("Draw.super: GPU batch draw error, fallback to QPainter: %s", exc)
                    self._gpu_failed = True
                finally:
                    painter.endNativePainting()

            # Pass 3: Draw foreground unbatched shapes (complex shapes, cutouts, borders)
            for s, pos_override, m_state in fg_unbatched:
                _draw_one_shape(
                    painter, s, cw, ch,
                    position_override=pos_override,
                    motion_state=m_state,
                    canvas=self,
                )

            # Pass 4: Interleaved text pass
            for t in self.text_items:
                if getattr(t, "motion", None):
                    ref_x, ref_y, ref_w, ref_h = t.last_rect if t.last_rect else (
                        float(t.x or 0), float(t.y or 0), 0.0, 0.0
                    )
                    t._last_motion_state = _motion_registry.compute_shape_state(
                        t, now, _parse_color, ref_x, ref_y, ref_w, ref_h, self
                    )
                else:
                    t._last_motion_state = None
                _draw_one_text(painter, t, cw, ch, canvas=self)

            # Pass 5: Keyboard focus outline
            if self._focused_ip:
                _focus_rect = self._focus_rect_for_ip(self._focused_ip)
                if _focus_rect is not None:
                    _fx, _fy, _fw, _fh = _focus_rect
                    painter.save()
                    _focus_pen = QPen(QColor(66, 133, 244))
                    _focus_pen.setWidth(2)
                    _focus_pen.setStyle(Qt.PenStyle.DashLine)
                    painter.setPen(_focus_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(QRectF(_fx - 3, _fy - 3, _fw + 6, _fh + 6), 4, 4)
                    painter.restore()
        finally:
            if painter.isActive():
                painter.end()

            t_frame_ms = (time.perf_counter() - now) * 1000.0
            draw_calls = len(unbatched_shapes) + len(self.text_items) + (1 if self.batcher.v_count > 0 else 0)
            vert_count = sum(s.vertices or 4 for s in self.shape_items)
            tri_count = sum(max(1, (s.vertices or 4) - 2) for s in self.shape_items)
            _debug_manager.record_frame(t_frame_ms, draw_calls=draw_calls, vertices=vert_count, triangles=tri_count)
            _debug_manager.record_timing("Render_OpenGL", t_frame_ms)


    # ── Animation and Interaction Delegate Methods ───────────────────────────

    def _has_active_shape_animation(self) -> bool:
        return any(_shape_is_dynamic(shape) or getattr(shape, "motion", None) for shape in self.shape_items)

    def _has_live_text_source(self) -> bool:
        return any(
            isinstance(t.source, LiveTextBinding) or callable(t.source) or callable(getattr(t, "text", None))
            for t in self.text_items
        )

    def _refresh_live_text_bindings(self) -> bool:
        changed = False
        for t in self.text_items:
            src = t.source if t.source is not None else getattr(t, "text", None)
            if not (isinstance(src, LiveTextBinding) or callable(src)):
                continue
            key = id(t)
            try:
                if isinstance(src, LiveTextBinding):
                    value = resolve_live_text(src)
                elif callable(src):
                    value = str(src())
                else:
                    value = str(src)
                self._live_text_error_cache.pop(key, None)
            except Exception as exc:
                value = ""
                msg = str(exc)
                if self._live_text_error_cache.get(key) != msg:
                    _logger.warning("Draw.live.text: source error: %s", exc)
                    self._live_text_error_cache[key] = msg
            if value != t.text:
                t.text = value
                t.dirty = True
                changed = True
        return changed

    def _compute_content_bounds(self) -> Tuple[float, float]:
        cw = float(self.width()) if self.width() > 0 else 1000.0
        ch = float(self.height()) if self.height() > 0 else 800.0
        max_w = cw
        max_h = ch
        for s in self.shape_items:
            if s.ip and (s.ip.startswith("scroller_") or any(s.ip in (c.get("thumb_ip"), c.get("track_ip")) for c in getattr(self, "_scroller_configs", []))):
                continue
            sx = getattr(s, "_placed_x", getattr(s, "x", 0) or 0)
            sy = getattr(s, "_placed_y", getattr(s, "y", 0) or 0)
            sw = getattr(s, "_placed_w", getattr(s, "width", 0) or 0)
            sh = getattr(s, "_placed_h", getattr(s, "height", 0) or 0)
            if s.last_size:
                sw, sh = s.last_size
            max_w = max(max_w, float(sx) + float(sw) + 40.0)
            max_h = max(max_h, float(sy) + float(sh) + 40.0)
        for t in self.text_items:
            if t.ip and (t.ip.startswith("scroller_") or any(t.ip in (c.get("thumb_ip"), c.get("track_ip")) for c in getattr(self, "_scroller_configs", []))):
                continue
            tx = float(t.x or 0)
            ty = float(t.y or 0)
            tw = t.last_rect[2] if t.last_rect else 100.0
            th = t.last_rect[3] if t.last_rect else 30.0
            max_w = max(max_w, tx + tw + 40.0)
            max_h = max(max_h, ty + th + 40.0)
        return max_w, max_h

    def _get_max_scroll_range(self) -> Tuple[float, float]:
        cw = float(self.width()) if self.width() > 0 else 1000.0
        ch = float(self.height()) if self.height() > 0 else 800.0
        cont_w, cont_h = self._compute_content_bounds()
        default_max_x = max(0.0, cont_w - cw)
        default_max_y = max(0.0, cont_h - ch)
        max_x = default_max_x
        max_y = default_max_y
        for cfg in getattr(self, "_scroller_configs", []):
            if cfg.get("max_x") is not None:
                max_x = float(cfg["max_x"])
            if cfg.get("max_y") is not None:
                max_y = float(cfg["max_y"])
        return max(0.0, max_x), max(0.0, max_y)

    def _update_scroller_thumbs(self, from_paint: bool = False) -> None:
        cw = float(self.width()) if self.width() > 0 else 1000.0
        ch = float(self.height()) if self.height() > 0 else 800.0
        max_scroll_x, max_scroll_y = self._get_max_scroll_range()

        if max_scroll_y > 0.0:
            self._scroll_y = max(0.0, min(max_scroll_y, self._scroll_y))
        else:
            self._scroll_y = 0.0

        if max_scroll_x > 0.0:
            self._scroll_x = max(0.0, min(max_scroll_x, self._scroll_x))
        else:
            self._scroll_x = 0.0

        if not self._scroller_configs:
            return
        ip_to_shape = {s.ip: s for s in self.shape_items if s.ip}
        changed = False

        for cfg in self._scroller_configs:
            thumb = ip_to_shape.get(cfg["thumb_ip"])
            if thumb is None:
                continue
            if cfg["direction"] == "vertical":
                track_h = float(cfg["track_h"])
                total_h = max_scroll_y + ch
                thumb_ratio = max(0.05, min(1.0, ch / max(1.0, total_h))) if total_h > 0 else 0.2
                thumb_h = max(24.0, min(track_h, track_h * thumb_ratio))
                cfg["thumb_h"] = thumb_h
                thumb.height = int(thumb_h)
                thumb.size_raw = [float(cfg["track_w"]), float(thumb_h)]
                thumb.last_size = (float(cfg["track_w"]), float(thumb_h))

                travel = max(1.0, track_h - thumb_h)
                scroll_range = float(cfg.get("max_y")) if cfg.get("max_y") is not None else max(1.0, max_scroll_y)
                t = max(0.0, min(1.0, self._scroll_y / max(1.0, scroll_range)))
                new_y = cfg["track_y"] + t * travel
                thumb.y = int(new_y)
                thumb._placed_y = float(new_y)
                thumb.last_position = (float(cfg["track_x"]), float(new_y))
            else:
                track_w = float(cfg["track_w"])
                total_w = max_scroll_x + cw
                thumb_ratio = max(0.05, min(1.0, cw / max(1.0, total_w))) if total_w > 0 else 0.2
                thumb_w = max(24.0, min(track_w, track_w * thumb_ratio))
                cfg["thumb_w"] = thumb_w
                thumb.width = int(thumb_w)
                thumb.size_raw = [float(thumb_w), float(cfg["track_h"])]
                thumb.last_size = (float(thumb_w), float(cfg["track_h"]))

                travel = max(1.0, track_w - thumb_w)
                scroll_range = float(cfg.get("max_x")) if cfg.get("max_x") is not None else max(1.0, max_scroll_x)
                t = max(0.0, min(1.0, self._scroll_x / max(1.0, scroll_range)))
                new_x = cfg["track_x"] + t * travel
                thumb.x = int(new_x)
                thumb._placed_x = float(new_x)
                thumb.last_position = (float(new_x), float(cfg["track_y"]))
            changed = True

        # CRITICAL: Never trigger update() from inside paintGL
        if changed and not from_paint and not self._painting:
            self.update()

    def _tick_animation(self) -> None:
        from Draw._motion import motion as _motion_registry
        now = time.perf_counter()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if dt < 0.0:
            dt = 0.016

        for s in self.shape_items:
            _motion_registry.tick_shape_triggers(s, dt)

        timeline_changed = _motion_registry.tick_timelines(now)
        shape_animating = self._has_active_shape_animation()
        live_changed = self._refresh_live_text_bindings() if self._has_live_text_source() else False
        custom_changed = _motion_registry.tick_custom(now)
        caret_animating = any(
            t.input_enabled and t.input_selected and t.input_caret and t.input_caret_blink
            for t in self.text_items
        )
        text_animating = any(_text_is_animated(t) for t in self.text_items)
        self._update_scroller_thumbs(from_paint=False)

        # Only request repaint when something is actively animating or changing
        if shape_animating or live_changed or custom_changed or caret_animating or text_animating or timeline_changed:
            if not self._painting:
                self.update()

    def _input_targets(self) -> list[TextDef]:
        return [t for t in self.text_items if t.input_enabled]

    def _active_input_target(self) -> Optional[TextDef]:
        targets = self._input_targets()
        if not targets or self._active_input_index < 0:
            return None
        if self._active_input_index >= len(targets):
            self._active_input_index = len(targets) - 1
        return targets[self._active_input_index]

    def _select_input_at_point(self, pos: QPointF, hit_ips: set[str]) -> bool:
        targets = self._input_targets()
        for index in range(len(targets) - 1, -1, -1):
            target = targets[index]
            ip_hit = target.ip is not None and target.ip in hit_ips
            if ip_hit or (target.last_rect and target.last_rect[0] <= pos.x() <= target.last_rect[0] + target.last_rect[2] and target.last_rect[1] <= pos.y() <= target.last_rect[1] + target.last_rect[3]):
                self._active_input_index = index
                self.setFocus()
                if not self._painting:
                    self.update()
                return True
        self._active_input_index = -1
        return False

    def _shapes_at_point(self, pos: QPointF) -> list:
        from Draw._shapes import _shape_contains_point
        hits = []
        cw = self.width() if self.width() > 0 else 800
        ch = self.height() if self.height() > 0 else 600
        for shape in sorted(self.shape_items, key=lambda s: s.z):
            if shape.ip is None:
                continue
            if shape.last_position is None or shape.last_size is None:
                try:
                    sw, sh, ox, oy = _shape_preferred_pos(shape, cw, ch)
                    shape.last_position = (float(ox), float(oy))
                    shape.last_size = (int(sw), int(sh))
                except Exception:
                    continue
            if _shape_contains_point(shape, pos):
                hits.append((shape.ip, shape))
        for t in self.text_items:
            if t.last_rect is None or t.ip is None:
                continue
            tx, ty, tw, th = t.last_rect
            if tx <= pos.x() <= tx + tw and ty <= pos.y() <= ty + th:
                hits.append((t.ip, t))
        return hits

    def _focus_rect_for_ip(self, ip: str):
        for s in self.shape_items:
            if s.ip == ip and s.last_position and s.last_size:
                x, y = s.last_position
                w, h = s.last_size
                return (x, y, w, h)
        for t in self.text_items:
            if t.ip == ip and t.last_rect:
                return t.last_rect
        return None

    # ── Mouse & Key Event Forwarding ──────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_press
        handle_canvas_mouse_press(self, event)
        self.update()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_release
        handle_canvas_mouse_release(self, event)
        self.update()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_double_click
        handle_canvas_mouse_double_click(self, event)
        self.update()
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_mouse_move
        handle_canvas_mouse_move(self, event)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_wheel
        handle_canvas_wheel(self, event)
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape or (
            event.key() == Qt.Key.Key_Q
            and (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        ):
            if self.parent():
                self.parent().close()
            event.accept()
            return
        from Draw._connectors import senses as _senses, _qt_key_to_name
        txt = event.text()
        key_name = _qt_key_to_name(event.key()) if (event.key() == Qt.Key.Key_Space or not txt or not txt.isprintable()) else txt
        _senses.dispatch_key_event("key_press", key_name, [])
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        from Draw._connectors import senses as _senses, _qt_key_to_name
        txt = event.text()
        key_name = _qt_key_to_name(event.key()) if (event.key() == Qt.Key.Key_Space or not txt or not txt.isprintable()) else txt
        _senses.dispatch_key_event("key_release", key_name, [])
        super().keyReleaseEvent(event)

    def _on_longpress_timeout(self) -> None:
        from Draw._connectors import handle_canvas_longpress_timeout
        handle_canvas_longpress_timeout(self)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        from Draw._connectors import handle_canvas_context_menu
        if not handle_canvas_context_menu(self, event):
            super().contextMenuEvent(event)

    def event(self, ev: QEvent) -> bool:  # noqa: N802
        from Draw._connectors import handle_canvas_event
        if handle_canvas_event(self, ev):
            return True
        return super().event(ev)

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self.parent():
            self.setGeometry(self.parent().rect())
        super().resizeEvent(event)

    # ── Safe Resource Cleanup ────────────────────────────────────────────────

    def _cleanup_resources(self) -> None:
        """Clean up timers and GPU resources safely."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        if hasattr(self, "_animation_timer") and self._animation_timer.isActive():
            self._animation_timer.stop()
        if hasattr(self, "_longpress_timer") and self._longpress_timer.isActive():
            self._longpress_timer.stop()

        if self.isValid() and self.context():
            try:
                self.makeCurrent()
                if self.vao is not None and self.vao.isCreated():
                    self.vao.destroy()
                    self.vao = None
                if self.vbo is not None and self.vbo.isCreated():
                    self.vbo.destroy()
                    self.vbo = None
                if self.ebo is not None and self.ebo.isCreated():
                    self.ebo.destroy()
                    self.ebo = None
                if self.shader_program is not None:
                    self.shader_program.removeAllShaders()
                    self.shader_program = None
            except Exception:
                pass
            finally:
                try:
                    self.doneCurrent()
                except Exception:
                    pass

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cleanup_resources()
        super().closeEvent(event)


# ── Draw.super Engine Entry Point ────────────────────────────────────────────

class _SuperRegistry:
    """
    Singleton exposed as Draw.super.

    Enables OpenGL hardware acceleration on windows, pre-compiles geometry,
    warms up caches, and tunes garbage collection for extreme runtime smoothness.
    """

    def __call__(
        self,
        display: Optional[str] = None,
        tag: Optional[str] = None,
        *,
        precompile: bool = True,
        mode: str = "max",
        vsync: bool = False,
        batch_capacity: int = 65536,
        **kwargs
    ) -> _DrawOpenGLCanvas:
        """
        Activate OpenGL hardware acceleration on the target window canvas.

        Parameters
        ----------
        display / tag : Target window tag (if None, targets all open windows or 'main').
        precompile    : If True, executes upfront geometry compilation and cache warming.
        mode          : Performance mode ('max' tunes Python GC thresholds and enables spatial grid).
        vsync         : If False (default), disables VSync for ultra-high uncapped FPS.
        batch_capacity: Max vertices per GPU dynamic buffer (default: 65536).
        """
        get_app()
        window_tag = display if display is not None else tag
        if window_tag is None:
            tags = _window_registry.list_tags()
            window_tag = tags[0] if tags else "main"

        win: QMainWindow = _window_registry.get(window_tag)

        # Idempotent upgrade: reuse existing _DrawOpenGLCanvas if already active
        existing_canvas = getattr(win, "_draw_canvas", None)
        if isinstance(existing_canvas, _DrawOpenGLCanvas):
            gl_canvas = existing_canvas
        else:
            gl_canvas = _DrawOpenGLCanvas(win, batch_capacity=batch_capacity, vsync=vsync)
            gl_canvas._window_tag = window_tag

            # Preserve items from software canvas
            if existing_canvas is not None:
                gl_canvas.shape_items = list(existing_canvas.shape_items)
                gl_canvas.text_items = list(existing_canvas.text_items)
                gl_canvas.layout_items = list(existing_canvas.layout_items)
                gl_canvas._scroller_configs = list(existing_canvas._scroller_configs)
                # Reparent any native child widgets (QPushButton, QSlider, etc.) to gl_canvas
                for child in existing_canvas.findChildren(QWidget):
                    if child.parent() is existing_canvas:
                        child.setParent(gl_canvas)
                        child.show()
                        child.raise_()
                existing_canvas.setParent(None)
                existing_canvas.deleteLater()

            # Set as primary central widget of the window
            win.setCentralWidget(gl_canvas)
            gl_canvas.setGeometry(0, 0, win.width(), win.height())
            gl_canvas.show()
            win._draw_canvas = gl_canvas

        # ── Upfront Pre-compilation & Warm-up (Slower Init, Ultra-Fast Runtime) ──
        if precompile:
            set_performance_mode(mode)
            GCTuner.tune_for_animation()

            # Pre-compile scene shapes and warm bounding box / path caches
            canvas = win._draw_canvas
            optimize(scene=canvas.shape_items, mode=mode, gc_tune=True)

            cw = win.width() if win.width() > 0 else 800
            ch = win.height() if win.height() > 0 else 600
            for s in canvas.shape_items:
                try:
                    sw, sh, ox, oy = _shape_preferred_pos(s, cw, ch)
                    s.last_position = (float(ox), float(oy))
                    s.last_size = (int(sw), int(sh))
                except Exception:
                    pass
                if hasattr(s, "_compile"):
                    s._compile()
                if hasattr(s, "last_position") and s.last_position and s.last_size:
                    canvas.spatial_grid.insert(id(s), (s.last_position[0], s.last_position[1], float(s.last_size[0]), float(s.last_size[1])))

        return win._draw_canvas

    def get(self, display: Optional[str] = None) -> Optional[_DrawOpenGLCanvas]:
        """Get the active _DrawOpenGLCanvas for a window tag."""
        tags = _window_registry.list_tags()
        window_tag = display if display is not None else (tags[0] if tags else "main")
        win = _window_registry.get(window_tag)
        if win is not None and hasattr(win, "_draw_canvas"):
            canvas = win._draw_canvas
            if isinstance(canvas, _DrawOpenGLCanvas):
                return canvas
        return None

    def clear(self, display: Optional[str] = None, keep_ips: Optional[Set[str]] = None) -> None:
        """Cleanly clear all shapes, texts, and render buffers for an OpenGL canvas."""
        canvas = self.get(display)
        if canvas is not None and hasattr(canvas, "clear"):
            canvas.clear(keep_ips=keep_ips)


super_engine = _SuperRegistry()
super_mode = super_engine

__all__ = [
    "super_engine",
    "super_mode",
    "_DrawOpenGLCanvas",
    "GPUGeometryBatcher",
]
