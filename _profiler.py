"""
Draw._profiler
==============
Specialized High-Resolution Render Loop & Performance Profiler for Draw.

Measures the 8 core rendering pipeline stages:
1. Motion calculation
2. Geometry generation
3. Spatial grid
4. Shape state/update
5. GPU buffer upload
6. OpenGL draw
7. Qt paint/event processing
8. Text/FPS display

Features:
- High-precision CPU timing (time.perf_counter_ns)
- Rolling-window statistics (Mean, Median, P95, P99, Max, and % Frame Share)
- Both Context Manager and explicit measurement APIs
- Built-in live telemetry HUD string generator
- Complete ASCII telemetry dashboard & JSON export
- Seamless auto-hook into Draw.super OpenGL and software canvas
"""

from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class ProfilerCategory:
    MOTION = "Motion calculation"
    GEOMETRY = "Geometry generation"
    SPATIAL_GRID = "Spatial grid"
    SHAPE_STATE = "Shape state/update"
    GPU_UPLOAD = "GPU buffer upload"
    OPENGL_DRAW = "OpenGL draw"
    QT_PAINT = "Qt paint/event processing"
    TEXT_FPS = "Text/FPS display"

    ALL_CATEGORIES = [
        MOTION,
        GEOMETRY,
        SPATIAL_GRID,
        SHAPE_STATE,
        GPU_UPLOAD,
        OPENGL_DRAW,
        QT_PAINT,
        TEXT_FPS,
    ]


class CategoryTimer:
    """Context manager for zero-overhead block timing."""
    __slots__ = ("_profiler", "_category", "_start_ns")

    def __init__(self, profiler: "RenderLoopProfiler", category: str):
        self._profiler = profiler
        self._category = category
        self._start_ns = 0

    def __enter__(self):
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter_ns() - self._start_ns) / 1_000_000.0
        self._profiler.record(self._category, duration_ms)


class RenderLoopProfiler:
    """
    Subsystem profiler measuring high-resolution per-frame timings across the
    8 fundamental Draw pipeline categories.
    """

    def __init__(self, window_size: int = 120):
        self.window_size = window_size
        self._lock = threading.RLock()
        self.categories = list(ProfilerCategory.ALL_CATEGORIES)

        # Per-frame accumulator (current active frame)
        self._current_frame: Dict[str, float] = {cat: 0.0 for cat in self.categories}

        # Rolling history of completed frame breakdowns (ms)
        self._history: Dict[str, collections.deque] = {
            cat: collections.deque(maxlen=window_size) for cat in self.categories
        }
        self._frame_times = collections.deque(maxlen=window_size)
        self._paint_intervals = collections.deque(maxlen=window_size)

        self._last_paint_timestamp: Optional[float] = None
        self._frame_count = 0
        self._is_measuring = False

    def reset(self) -> None:
        """Reset all recorded metrics and history buffers."""
        with self._lock:
            self._current_frame = {cat: 0.0 for cat in self.categories}
            for cat in self.categories:
                self._history[cat].clear()
            self._frame_times.clear()
            self._paint_intervals.clear()
            self._last_paint_timestamp = None
            self._frame_count = 0

    def start_frame(self) -> None:
        """Mark the beginning of a new rendering frame."""
        now = time.perf_counter()
        with self._lock:
            if self._last_paint_timestamp is not None:
                interval_ms = (now - self._last_paint_timestamp) * 1000.0
                if 0.1 < interval_ms < 1000.0:
                    self._paint_intervals.append(interval_ms)
            self._last_paint_timestamp = now
            self._current_frame = {cat: 0.0 for cat in self.categories}
            self._is_measuring = True

    def record(self, category: str, duration_ms: float) -> None:
        """Add time (in ms) to the current frame's category accumulator."""
        with self._lock:
            if category in self._current_frame:
                self._current_frame[category] += duration_ms

    def measure(self, category: str) -> CategoryTimer:
        """Context manager to measure a block: `with profiler.measure('Motion calculation'): ...`"""
        return CategoryTimer(self, category)

    def end_frame(self) -> float:
        """
        Mark the completion of a rendering frame.
        Computes total frame render time and pushes to rolling window.
        """
        with self._lock:
            total_render_ms = sum(self._current_frame.values())
            self._frame_times.append(total_render_ms)
            for cat in self.categories:
                self._history[cat].append(self._current_frame[cat])

            self._frame_count += 1
            self._is_measuring = False
            return total_render_ms

    # ── Statistics & Analysis ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean, median, P95, P99, max, and frame-time percentage for every category.
        """
        with self._lock:
            stats: Dict[str, Dict[str, float]] = {}
            total_avg = np.mean(self._frame_times) if self._frame_times else 0.001

            for cat in self.categories:
                data = list(self._history[cat])
                if not data:
                    stats[cat] = {
                        "mean": 0.0,
                        "median": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                        "max": 0.0,
                        "pct": 0.0,
                    }
                    continue

                arr = np.array(data)
                mean_val = float(np.mean(arr))
                stats[cat] = {
                    "mean": round(mean_val, 4),
                    "median": round(float(np.median(arr)), 4),
                    "p95": round(float(np.percentile(arr, 95)), 4),
                    "p99": round(float(np.percentile(arr, 99)), 4),
                    "max": round(float(np.max(arr)), 4),
                    "pct": round((mean_val / max(total_avg, 0.0001)) * 100.0, 1),
                }

            # Add total render stats
            if self._frame_times:
                arr = np.array(list(self._frame_times))
                stats["Total Render Time"] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "median": round(float(np.median(arr)), 4),
                    "p95": round(float(np.percentile(arr, 95)), 4),
                    "p99": round(float(np.percentile(arr, 99)), 4),
                    "max": round(float(np.max(arr)), 4),
                    "pct": 100.0,
                }
            else:
                stats["Total Render Time"] = {
                    "mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "pct": 100.0
                }

            # Add display interval stats
            if self._paint_intervals:
                arr = np.array(list(self._paint_intervals))
                stats["Time Between Paints"] = {
                    "mean": round(float(np.mean(arr)), 4),
                    "median": round(float(np.median(arr)), 4),
                    "p95": round(float(np.percentile(arr, 95)), 4),
                    "p99": round(float(np.percentile(arr, 99)), 4),
                    "max": round(float(np.max(arr)), 4),
                    "pct": 0.0,
                }

            return stats

    def get_fps_summary(self) -> Dict[str, float]:
        """Returns dual display rate and pure render throughput metrics."""
        with self._lock:
            avg_interval = np.mean(self._paint_intervals) if self._paint_intervals else 16.666
            avg_render = np.mean(self._frame_times) if self._frame_times else 2.5
            return {
                "display_fps": round(1000.0 / max(avg_interval, 0.001), 1),
                "display_interval_ms": round(float(avg_interval), 2),
                "engine_throughput_fps": round(1000.0 / max(avg_render, 0.001), 1),
                "cpu_render_time_ms": round(float(avg_render), 2),
                "total_frames_profiled": self._frame_count,
            }

    # ── Formatting & Reporting ───────────────────────────────────────────────

    def format_table(self) -> str:
        """Format the profiler breakdown into a clean ASCII report table."""
        stats = self.get_stats()
        fps_info = self.get_fps_summary()

        lines = [
            "=" * 86,
            "                   DRAW RENDERING PIPELINE PROFILER REPORT                    ",
            "=" * 86,
            f"Display Rate: {fps_info['display_fps']} FPS ({fps_info['display_interval_ms']} ms) | "
            f"Engine Speed: {fps_info['engine_throughput_fps']} FPS ({fps_info['cpu_render_time_ms']} ms) | "
            f"Frames: {fps_info['total_frames_profiled']:,}",
            "-" * 86,
            f"{'Pipeline Stage':<30} | {'Mean (ms)':>9} | {'Median':>8} | {'P95':>8} | {'Max':>8} | {'Share %':>7}",
            "-" * 86,
        ]

        for cat in self.categories:
            st = stats.get(cat, {})
            lines.append(
                f"{cat:<30} | {st.get('mean', 0.0):9.4f} | {st.get('median', 0.0):8.4f} | "
                f"{st.get('p95', 0.0):8.4f} | {st.get('max', 0.0):8.4f} | {st.get('pct', 0.0):6.1f}%"
            )

        lines.append("-" * 86)
        tot = stats.get("Total Render Time", {})
        lines.append(
            f"{'Total Render Time':<30} | {tot.get('mean', 0.0):9.4f} | {tot.get('median', 0.0):8.4f} | "
            f"{tot.get('p95', 0.0):8.4f} | {tot.get('max', 0.0):8.4f} | 100.0%"
        )
        tbp = stats.get("Time Between Paints", {})
        if tbp:
            lines.append(
                f"{'Time Between Paints':<30} | {tbp.get('mean', 0.0):9.4f} | {tbp.get('median', 0.0):8.4f} | "
                f"{tbp.get('p95', 0.0):8.4f} | {tbp.get('max', 0.0):8.4f} |    ---"
            )
        lines.append("=" * 86)
        return "\n".join(lines)

    def live_hud_string(self) -> str:
        """Compact single-line live HUD string for Draw.live bindings."""
        stats = self.get_stats()
        fps_info = self.get_fps_summary()

        m_ms = stats.get(ProfilerCategory.MOTION, {}).get("mean", 0.0)
        g_ms = stats.get(ProfilerCategory.GEOMETRY, {}).get("mean", 0.0)
        s_ms = stats.get(ProfilerCategory.SPATIAL_GRID, {}).get("mean", 0.0)
        v_ms = stats.get(ProfilerCategory.GPU_UPLOAD, {}).get("mean", 0.0)
        q_ms = stats.get(ProfilerCategory.QT_PAINT, {}).get("mean", 0.0)

        return (
            f"FPS: {fps_info['display_fps']:4.1f} | Engine: {fps_info['engine_throughput_fps']:5.1f} FPS "
            f"[Motion: {m_ms:3.2f}ms | Geom: {g_ms:3.2f}ms | Grid: {s_ms:3.2f}ms | "
            f"GPU: {v_ms:3.2f}ms | Paint: {q_ms:3.2f}ms]"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profiler metrics to a dictionary."""
        return {
            "fps_summary": self.get_fps_summary(),
            "pipeline_stages": self.get_stats(),
        }

    def export_json(self, file_path: str) -> None:
        """Export profiler metrics to a JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global singleton instance
profiler = RenderLoopProfiler(window_size=120)


# ── Automatic OpenGL Canvas Profiling Hook ────────────────────────────────────

def attach_profiler_to_canvas(canvas: Any) -> None:
    """
    Hooks the RenderLoopProfiler directly into a _DrawOpenGLCanvas or software canvas
    to measure all 8 stages with zero manual boilerplate.
    """
    from Draw._motion import motion as _motion_registry
    from Draw._shapes import (
        _shape_preferred_geometry, _shape_hit_geometry,
        _apply_motion_geometry, _draw_one_shape, _get_unit_polygon, _parse_color,
    )
    from Draw._super import _parse_color_to_rgba
    from Draw._text import _draw_one_text
    from Draw._layout import _draw_one_layout
    from PySide6.QtGui import QPainter, QColor, QPen, QMatrix4x4
    from PySide6.QtCore import QRectF, Qt

    orig_do_paint_gl = getattr(canvas, "_do_paint_gl", None)
    if orig_do_paint_gl is None:
        return

    def profiled_paint_gl():
        profiler.start_frame()
        cw, ch = canvas.width(), canvas.height()
        if cw <= 0 or ch <= 0:
            profiler.end_frame()
            return

        now = time.perf_counter()

        # 1. Scroller & Z-Order State Update
        with profiler.measure(ProfilerCategory.SHAPE_STATE):
            canvas._update_scroller_thumbs(from_paint=True)
            if canvas._z_order_dirty:
                canvas._sorted_shapes = sorted(canvas.shape_items, key=lambda s: -s.z)
                canvas._z_order_dirty = False
            sorted_shapes = canvas._sorted_shapes
            canvas.batcher.reset()

        unbatched_shapes = []
        batched_shape_set = set()

        if canvas._projection_matrix is None:
            proj = QMatrix4x4()
            proj.ortho(0.0, float(cw), float(ch), 0.0, -1.0, 1.0)
            canvas._projection_matrix = proj

        _window_tag = getattr(canvas, "_window_tag", None)
        _scroll_x = canvas._scroll_x
        _scroll_y = canvas._scroll_y
        _gl_ok = canvas._gl_initialized and not canvas._gpu_failed
        _batcher = canvas.batcher
        _has_active_timelines = bool(_motion_registry._active_timelines)
        _sgrid_items = canvas._spatial_grid_direct.item_boxes
        _sgrid = canvas._spatial_grid_direct
        _EMPTY_MOTION = {}

        for s in sorted_shapes:
            sid = id(s)
            is_dragged = getattr(s, "_is_dragged", False)
            _shape_motions = getattr(s, "motion", None)
            has_motion = bool(_shape_motions) or (_has_active_timelines and s.ip is not None)
            is_scroller = s.ip and (s.ip.startswith("scroller_") or "_track" in s.ip or "_thumb" in s.ip)

            # Static Retained Path
            with profiler.measure(ProfilerCategory.SHAPE_STATE):
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

            # Dynamic Geometry Generation
            with profiler.measure(ProfilerCategory.GEOMETRY):
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

                if not is_scroller:
                    final_x -= _scroll_x
                    final_y -= _scroll_y

            # Motion Calculation
            with profiler.measure(ProfilerCategory.MOTION):
                if is_dragged and not is_scroller:
                    final_x = getattr(s, "_drag_x", final_x)
                    final_y = getattr(s, "_drag_y", final_y)
                    anim_x, anim_y = float(final_x), float(final_y)
                    anim_sw, anim_sh = float(sw), float(sh)
                    shape_motion_state = _EMPTY_MOTION
                else:
                    if has_motion:
                        shape_motion_state = _motion_registry.compute_shape_state(
                            s, now, _parse_color, float(final_x), float(final_y), sw, sh, canvas
                        )
                        s._last_motion_state = shape_motion_state
                        anim_x, anim_y, anim_sw, anim_sh = _apply_motion_geometry(
                            shape_motion_state, float(final_x), float(final_y), sw, sh
                        )
                    else:
                        shape_motion_state = _EMPTY_MOTION
                        s._last_motion_state = _EMPTY_MOTION
                        anim_x, anim_y = float(final_x), float(final_y)
                        anim_sw, anim_sh = float(sw), float(sh)

                s.last_position = (float(anim_x), float(anim_y))
                s.last_size = (int(anim_sw), int(anim_sh))

            # Hitbox Geometry
            with profiler.measure(ProfilerCategory.GEOMETRY):
                hx, hy, hsw, hsh = _shape_hit_geometry(s)
                s.last_hit_position = (float(hx), float(hy))
                s.last_hit_size = (int(hsw), int(hsh))

            # Spatial Grid
            with profiler.measure(ProfilerCategory.SPATIAL_GRID):
                new_bbox = (float(anim_x), float(anim_y), float(anim_sw), float(anim_sh))
                old_bbox = _sgrid_items.get(sid)
                if old_bbox != new_bbox:
                    _sgrid._update_internal(sid, new_bbox)

            # Shape State & Color
            with profiler.measure(ProfilerCategory.SHAPE_STATE):
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
                    cx = anim_x + anim_sw / 2.0
                    cy = anim_y + anim_sh / 2.0
                    rx = anim_sw / 2.0
                    ry = anim_sh / 2.0
                    unit_pts = _get_unit_polygon(n)

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

                    v_start = _batcher.v_count
                    batched = _batcher.add_transformed_polygon(cx, cy, rx, ry, unit_pts, rot, color_rgba)
                    if batched:
                        batched_shape_set.add(sid)
                        v_end = _batcher.v_count
                        n_verts = v_end - v_start
                        vdata_slice = _batcher.vertex_buffer[v_start * 6:v_end * 6].copy()

                if not batched:
                    unbatched_shapes.append((s, (final_x, final_y), shape_motion_state))

                s._gpu_cache = {
                    "cw": cw, "ch": ch, "sx": _scroll_x, "sy": _scroll_y,
                    "batched": batched, "vdata": vdata_slice, "n_verts": n_verts, "pos": (final_x, final_y),
                }
                s.dirty = False

        # 2. QPainter Pass & Background Rendering
        with profiler.measure(ProfilerCategory.QT_PAINT):
            painter = QPainter(canvas)
            try:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

                win_obj = canvas.parent() if canvas.parent() else None
                bg_col = getattr(win_obj, "_bg_color", None) if win_obj else None
                if bg_col is None:
                    bg_col = QColor(8, 12, 20)
                painter.fillRect(canvas.rect(), bg_col)

                for layout in canvas.layout_items:
                    _draw_one_layout(painter, layout, cw, ch)

                bg_unbatched = [(s, pos, ms) for s, pos, ms in unbatched_shapes if s.z >= 80]
                fg_unbatched = [(s, pos, ms) for s, pos, ms in unbatched_shapes if s.z < 80]

                for s, pos_override, m_state in bg_unbatched:
                    _draw_one_shape(painter, s, cw, ch, position_override=pos_override, motion_state=m_state, canvas=canvas)
            finally:
                pass

        # 3. GPU Buffer Upload & OpenGL Draw Pass
        if canvas._gl_initialized and not canvas._gpu_failed and canvas.batcher.v_count > 0:
            try:
                painter.beginNativePainting()
                funcs = canvas.context().functions() if canvas.context() else None
                if funcs and canvas.shader_program and canvas.vbo:
                    # Stage 5: GPU buffer upload
                    with profiler.measure(ProfilerCategory.GPU_UPLOAD):
                        v_used_floats = canvas.batcher.v_count * 6
                        v_bytes = canvas.batcher.vertex_buffer[:v_used_floats].tobytes()
                        canvas.vbo.bind()
                        canvas.vbo.write(0, v_bytes, len(v_bytes))

                    # Stage 6: OpenGL draw
                    with profiler.measure(ProfilerCategory.OPENGL_DRAW):
                        canvas.shader_program.bind()
                        canvas.shader_program.setUniformValue("uProjection", canvas._projection_matrix)

                        if canvas.vao and canvas.vao.isCreated():
                            canvas.vao.bind()
                        else:
                            canvas.vbo.bind()
                            stride = 6 * 4
                            canvas.shader_program.enableAttributeArray(0)
                            canvas.shader_program.setAttributeBuffer(0, 0x1406, 0, 2, stride)
                            canvas.shader_program.enableAttributeArray(1)
                            canvas.shader_program.setAttributeBuffer(1, 0x1406, 2 * 4, 4, stride)

                        funcs.glDrawArrays(0x0004, 0, canvas.batcher.v_count)

                        if canvas.vao and canvas.vao.isCreated():
                            canvas.vao.release()
                        canvas.vbo.release()
                        canvas.shader_program.release()
            except Exception:
                canvas._gpu_failed = True
            finally:
                painter.endNativePainting()

        # 4. Foreground Shapes Pass
        with profiler.measure(ProfilerCategory.QT_PAINT):
            for s, pos_override, m_state in fg_unbatched:
                _draw_one_shape(painter, s, cw, ch, position_override=pos_override, motion_state=m_state, canvas=canvas)

        # 5. Text & FPS Display Pass
        with profiler.measure(ProfilerCategory.TEXT_FPS):
            for t in canvas.text_items:
                if getattr(t, "motion", None):
                    ref_x, ref_y, ref_w, ref_h = t.last_rect if t.last_rect else (float(t.x or 0), float(t.y or 0), 0.0, 0.0)
                    t._last_motion_state = _motion_registry.compute_shape_state(t, now, _parse_color, ref_x, ref_y, ref_w, ref_h, canvas)
                else:
                    t._last_motion_state = None
                _draw_one_text(painter, t, cw, ch, canvas=canvas)

            if canvas._focused_ip:
                _focus_rect = canvas._focus_rect_for_ip(canvas._focused_ip)
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

        with profiler.measure(ProfilerCategory.QT_PAINT):
            if painter.isActive():
                painter.end()

        profiler.end_frame()

    canvas._do_paint_gl = profiled_paint_gl


__all__ = [
    "profiler",
    "RenderLoopProfiler",
    "ProfilerCategory",
    "attach_profiler_to_canvas",
]
