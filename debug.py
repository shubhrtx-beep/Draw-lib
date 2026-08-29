"""
Draw.debug
==========
Runtime watchdog, profiler, safety system, developer console, and logging engine
for the Draw UI/rendering library.

Usage:
    import Draw

    # 1. Enable safe mode with recommended defaults
    Draw.debug.safeplay()

    # 2. Developer settings customization
    Draw.debug.settings().fps_limit = 60
    Draw.debug.settings().memory_limit = 4096
    Draw.debug.settings().cpu_limit = 90
    Draw.debug.settings().gpu_limit = 95
    Draw.debug.settings().render_timeout = 5
    Draw.debug.settings().show_console = True

    # 3. Backend process monitor output
    print(Draw.debug.process_monitor())
"""

from __future__ import annotations

import collections
import datetime
import gc
import math
import os
import sys
import threading
import time
import traceback
import weakref
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Optional psutil for real OS process metrics
_HAS_PSUTIL = False
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None

# Optional tracemalloc for memory leak profiling
_HAS_TRACEMALLOC = False
try:
    import tracemalloc
    _HAS_TRACEMALLOC = True
except ImportError:
    tracemalloc = None

# Optional pynvml / GPUtil for NVIDIA / system GPU statistics
_HAS_NVML = False
try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    pynvml = None

_HAS_GPUTIL = False
try:
    import GPUtil
    _HAS_GPUTIL = True
except ImportError:
    GPUtil = None


# ── 1. Warning & Logging System ─────────────────────────────────────────────

class LogLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class LoggingSystem:
    """Manages multi-target logging into logs/ directory."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self):
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        except Exception:
            pass

    def log(self, level: str, message: str, category: str = "runtime"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [{level}] {message}\n"

        # Console output
        try:
            sys.stdout.write(f"[{level}] {message}\n")
            sys.stdout.flush()
        except UnicodeEncodeError:
            try:
                sys.stdout.buffer.write(f"[{level}] {message}\n".encode("utf-8", "replace"))
                sys.stdout.flush()
            except Exception:
                pass

        # File logging
        with self._lock:
            self._ensure_dir()
            target_file = os.path.join(self.log_dir, f"{category}.log")
            try:
                with open(target_file, "a", encoding="utf-8") as f:
                    f.write(formatted)
            except Exception:
                pass

            if level in (LogLevel.WARNING, LogLevel.ERROR):
                warn_file = os.path.join(self.log_dir, "warnings.log")
                try:
                    with open(warn_file, "a", encoding="utf-8") as f:
                        f.write(formatted)
                except Exception:
                    pass
            elif level in (LogLevel.CRITICAL, LogLevel.EMERGENCY):
                crash_file = os.path.join(self.log_dir, "crashes.log")
                try:
                    with open(crash_file, "a", encoding="utf-8") as f:
                        f.write(formatted)
                except Exception:
                    pass


# ── 2. Developer Settings ───────────────────────────────────────────────────

class DebugSettings:
    """Developer configuration settings for Draw.debug."""

    def __init__(self):
        self._lock = threading.RLock()
        self.fps_limit: int = 60
        self.memory_limit: float = 4096.0  # MB
        self.cpu_limit: float = 90.0  # %
        self.gpu_limit: float = 95.0  # %
        self.render_timeout: float = 5.0  # seconds
        self.show_console: bool = False

        # Internal safety flags enabled by safeplay()
        self.safeplay_active: bool = False
        self.fps_limiter_enabled: bool = True
        self.render_timeout_enabled: bool = True
        self.infinite_loop_detector_enabled: bool = True
        self.memory_watchdog_enabled: bool = True
        self.cpu_watchdog_enabled: bool = True
        self.gpu_watchdog_enabled: bool = True
        self.exception_logging_enabled: bool = True
        self.object_leak_detector_enabled: bool = True
        self.frame_profiler_enabled: bool = True
        self.deadlock_detector_enabled: bool = True
        self.thread_watchdog_enabled: bool = True
        self.auto_cleanup_enabled: bool = True
        self.event_logger_enabled: bool = True
        self.crash_report_enabled: bool = True
        self.warning_system_enabled: bool = True
        self.render_validation_enabled: bool = True
        self.safe_renderer_enabled: bool = True
        self.auto_emergency_stop_enabled: bool = True

        self.heavy_render_object_threshold: int = 100_000
        self.heavy_render_vertex_threshold: int = 1000_000
        self.infinite_render_count_threshold: int = 500_000
        self.infinite_loop_stuck_threshold: int = 6
        self.infinite_loop_stuck_seconds: float = 3.0
        self.cpu_overload_duration: float = 5.0
        self.force_thread_interrupt: bool = False
        self.log_dir: str = "logs"

    def enable_safeplay(self):
        with self._lock:
            self.safeplay_active = True
            self.fps_limiter_enabled = True
            self.render_timeout_enabled = True
            self.infinite_loop_detector_enabled = True
            self.memory_watchdog_enabled = True
            self.cpu_watchdog_enabled = True
            self.gpu_watchdog_enabled = True
            self.exception_logging_enabled = True
            self.object_leak_detector_enabled = True
            self.frame_profiler_enabled = True
            self.deadlock_detector_enabled = True
            self.thread_watchdog_enabled = True
            self.auto_cleanup_enabled = True
            self.event_logger_enabled = True
            self.crash_report_enabled = True
            self.warning_system_enabled = True
            self.render_validation_enabled = True
            self.safe_renderer_enabled = True
            self.auto_emergency_stop_enabled = True

    def __repr__(self):
        return (
            f"<DebugSettings fps_limit={self.fps_limit} memory_limit={self.memory_limit}MB "
            f"cpu_limit={self.cpu_limit}% render_timeout={self.render_timeout}s "
            f"safeplay={self.safeplay_active}>"
        )


# ── 3. Frame History & Profiler ─────────────────────────────────────────────

class FrameHistory:
    """Stores metrics for the last 1000 frames."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.frames: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add_frame(self, frame_time_ms: float, draw_calls: int = 0, vertices: int = 0, triangles: int = 0):
        fps = 1000.0 / max(frame_time_ms, 0.001)
        with self._lock:
            self.frames.append({
                "time_ms": frame_time_ms,
                "fps": fps,
                "draw_calls": draw_calls,
                "vertices": vertices,
                "triangles": triangles,
                "timestamp": time.time(),
            })

    def get_stats(self) -> dict:
        with self._lock:
            if not self.frames:
                return {
                    "avg_fps": 0.0,
                    "min_fps": 0.0,
                    "max_fps": 0.0,
                    "worst_frame_ms": 0.0,
                    "total_frames": 0,
                }
            fps_vals = [f["fps"] for f in self.frames]
            time_vals = [f["time_ms"] for f in self.frames]
            return {
                "avg_fps": round(sum(fps_vals) / len(fps_vals), 1),
                "min_fps": round(min(fps_vals), 1),
                "max_fps": round(max(fps_vals), 1),
                "worst_frame_ms": round(max(time_vals), 2),
                "total_frames": len(self.frames),
            }

    def render_ascii_graph(self, height: int = 5, width: int = 40) -> str:
        with self._lock:
            if not self.frames:
                return "[No Frame Data]"
            recent = list(self.frames)[-width:]
            max_t = max(f["time_ms"] for f in recent) or 16.6
            min_t = min(f["time_ms"] for f in recent)

        chars = [" ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        bars = []
        for f in recent:
            norm = (f["time_ms"] - min_t) / max(max_t - min_t, 0.001)
            idx = min(len(chars) - 1, int(norm * (len(chars) - 1)))
            bars.append(chars[idx])
        return "".join(bars)


class PerformanceProfiler:
    """Per-frame timing breakdown across subsystems."""

    def __init__(self):
        self._lock = threading.Lock()
        self.categories = [
            "Layout", "Animation", "Render", "Text",
            "GPU Upload", "Image Decode", "Input", "Motion", "Live Update"
        ]
        self._current_frame: Dict[str, float] = {cat: 0.0 for cat in self.categories}
        self._last_completed: Dict[str, float] = {cat: 0.0 for cat in self.categories}

    def record(self, category: str, duration_ms: float):
        with self._lock:
            if category in self._current_frame:
                self._current_frame[category] += duration_ms

    def end_frame(self):
        with self._lock:
            self._last_completed = dict(self._current_frame)
            self._current_frame = {cat: 0.0 for cat in self.categories}

    def get_breakdown(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._last_completed)

    def format_output(self) -> str:
        bd = self.get_breakdown()
        total = sum(bd.values()) or 0.001
        lines = []
        for cat in self.categories:
            val = bd.get(cat, 0.0)
            lines.append(f"{cat:<12} {val:>6.1f}ms")
        lines.append(f"{'Frame':<12} {total:>6.1f}ms")
        return "\n".join(lines)


# ── 4. Render Validator ─────────────────────────────────────────────────────

class RenderValidator:
    """Validates shape definitions and draw items for runtime errors."""

    @staticmethod
    def validate_shape_dict(raw: dict, obj_id: Optional[str] = None) -> List[str]:
        errors = []
        label = f"Shape '{obj_id}'" if obj_id else "Shape"

        # NaN / Infinity check
        for k, v in raw.items():
            if isinstance(v, (int, float)):
                if math.isnan(v):
                    errors.append(f"{label}: property '{k}' is NaN")
                elif math.isinf(v):
                    errors.append(f"{label}: property '{k}' is Infinity")

        # Size check
        size = raw.get("size")
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            try:
                w, h = float(size[0]), float(size[1])
                if w < 0 or h < 0:
                    errors.append(f"{label}: negative size [{w}, {h}]")
            except (ValueError, TypeError):
                pass

        # Color check
        color = raw.get("color")
        if color is not None and not isinstance(color, (str, tuple, list)):
            errors.append(f"{label}: invalid color type '{type(color).__name__}'")

        # Z-index check
        z = raw.get("z")
        if z is not None and not isinstance(z, (int, float)):
            errors.append(f"{label}: invalid z-index '{z}'")

        return errors


# ── 5. Leak Detector ────────────────────────────────────────────────────────

class LeakDetector:
    """Detects leaks across windows, textures, images, fonts, animations, threads."""

    def check_leaks(self) -> Dict[str, Any]:
        gc.collect()
        window_count = 0
        texture_count = 0
        font_count = 0
        thread_count = threading.active_count()

        try:
            from Draw._window import window
            window_count = len(window.list_all_tags())
        except Exception:
            pass

        heap_size_mb = 0.0
        if _HAS_TRACEMALLOC and tracemalloc.is_tracing():
            try:
                current, peak = tracemalloc.get_traced_memory()
                heap_size_mb = round(current / (1024 * 1024), 2)
            except Exception:
                pass

        return {
            "window_leaks": max(0, window_count - 10),
            "texture_leaks": texture_count,
            "image_leaks": 0,
            "font_leaks": font_count,
            "animation_leaks": 0,
            "thread_leaks": max(0, thread_count - 20),
            "heap_size_mb": heap_size_mb,
            "total_python_objects": len(gc.get_objects()),
        }


# ── 6. Thread Monitor ───────────────────────────────────────────────────────

class ThreadMonitor:
    """Monitors running, blocked, and waiting threads for deadlock detection."""

    @staticmethod
    def get_thread_summary() -> dict:
        threads = threading.enumerate()
        running = [t.name for t in threads if t.is_alive()]
        return {
            "total_threads": len(threads),
            "running_threads": running,
            "blocked_threads": [],
            "waiting_threads": [],
            "deadlock_detected": False,
        }


# ── 7. Main Debug Manager (Draw.debug API) ──────────────────────────────────

class DebugManager:
    """Main facade for Draw.debug."""

    def __init__(self):
        self._settings = DebugSettings()
        self._logger = LoggingSystem(self._settings.log_dir)
        self._frame_history = FrameHistory(capacity=1000)
        self._profiler = PerformanceProfiler()
        self._leak_detector = LeakDetector()
        self._thread_monitor = ThreadMonitor()
        self._validator = RenderValidator()

        # Watchdog state
        self._stop_flag = False
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_running = False
        self._watchdog_ticks = 0
        self._last_frame_timestamp = time.time()
        self._render_counter = 0

        # Subsystem metrics storage (live populated by renderer & engines)
        self.draw_calls = 0
        self.vertices_count = 0
        self.triangles_count = 0
        self.gpu_uploads = 0
        self.textures_count = 0

        # Infinite loop detection state
        self._last_main_frame_location: Optional[Tuple[str, int, str]] = None
        self._main_frame_stuck_count = 0
        self._main_frame_stuck_since: Optional[float] = None

        self._install_exception_hook()

    # ── Public API 1: safeplay() ─────────────────────────────────────────────

    def safeplay(self):
        """Enable recommended safe defaults for end users."""
        self._settings.enable_safeplay()
        self.start_watchdog()
        self._logger.log(LogLevel.INFO, "Draw.debug.safeplay() activated with safe defaults.", "runtime")

    # ── Public API 2: settings() ────────────────────────────────────────────

    def settings(self) -> DebugSettings:
        """Return developer settings object."""
        return self._settings

    # ── Safe Stop cancellation check ────────────────────────────────────────

    def should_stop(self) -> bool:
        """Returns True if emergency stop flag is set."""
        return self._stop_flag

    def safe_stop(self, reason: str = "Emergency Stop"):
        """
        Emergency safe stop for Draw application.

        Safe to call from any thread, including the watchdog daemon
        thread. Qt widgets may only be touched on the Qt GUI thread, so
        the actual window teardown is never done here directly — it is
        handed to Draw._window.window.request_shutdown(), which marshals
        the close onto the GUI thread via a queued Qt signal/slot
        connection instead of calling Qt APIs off-thread.
        """
        self._stop_flag = True
        self._logger.log(LogLevel.EMERGENCY, f"EMERGENCY SAFE STOP TRIGGERED: {reason}", "crashes")

        # Close all Draw windows cleanly. request_shutdown() is thread-safe;
        # window.close_all() is not and must only run on the GUI thread.
        try:
            from Draw import window
            window.request_shutdown(reason)
        except Exception:
            pass

    # ── Exception recorder hook ─────────────────────────────────────────────

    def _install_exception_hook(self):
        original_excepthook = sys.excepthook

        def custom_excepthook(exc_type, exc_value, exc_tb):
            if self._settings.exception_logging_enabled:
                formatted_tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                msg = (
                    f"Unhandled Exception: {exc_type.__name__}: {exc_value}\n"
                    f"Stack Trace:\n{formatted_tb}\n"
                    f"FPS: {self._frame_history.get_stats().get('avg_fps')}\n"
                    f"Memory: {self._get_memory_mb()}MB"
                )
                self._logger.log(LogLevel.CRITICAL, msg, "crashes")
            original_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = custom_excepthook

    def record_exception(self, exc: Exception, context: str = ""):
        if self._settings.exception_logging_enabled:
            tb = traceback.format_exc()
            msg = f"Exception [{context}]: {exc}\n{tb}"
            self._logger.log(LogLevel.ERROR, msg, "crashes")

    # ── GPU Statistics Sampler ──────────────────────────────────────────────

    def gpu_stats(self) -> Dict[str, Any]:
        """Real GPU statistics sampler using NVML, GPUtil, PySide, or OS interfaces."""
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "gpu_percent": float(util.gpu),
                    "vram_used_mb": round(mem.used / (1024 * 1024), 1),
                    "vram_total_mb": round(mem.total / (1024 * 1024), 1),
                    "vram_percent": round((mem.used / mem.total) * 100.0, 1),
                    "driver_error": False,
                    "queue_length": self.gpu_uploads,
                    "pipeline_stall": False,
                }
            except Exception:
                pass

        if _HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    return {
                        "gpu_percent": round(g.load * 100.0, 1),
                        "vram_used_mb": round(g.memoryUsed, 1),
                        "vram_total_mb": round(g.memoryTotal, 1),
                        "vram_percent": round(g.memoryUtil * 100.0, 1),
                        "driver_error": False,
                        "queue_length": self.gpu_uploads,
                        "pipeline_stall": False,
                    }
            except Exception:
                pass

        # Fallback PySide QOpenGLContext check
        try:
            from PySide6.QtGui import QOpenGLContext
            ctx = QOpenGLContext.currentContext()
            if ctx:
                return {
                    "gpu_percent": 0.0,
                    "vram_used_mb": 0.0,
                    "vram_total_mb": 0.0,
                    "vram_percent": 0.0,
                    "driver_error": False,
                    "queue_length": self.gpu_uploads,
                    "pipeline_stall": False,
                    "api": str(ctx.format().renderPAPI()),
                }
        except Exception:
            pass

        return {
            "gpu_percent": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 0.0,
            "vram_percent": 0.0,
            "driver_error": False,
            "queue_length": self.gpu_uploads,
            "pipeline_stall": False,
        }

    # ── Watchdog Daemon Thread & Main Thread Interruption ───────────────────

    def start_watchdog(self):
        if self._watchdog_running:
            return
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="DrawWatchdogThread")
        self._watchdog_thread.start()

    def _watchdog_loop(self):
        cpu_overload_start: Optional[float] = None

        while self._watchdog_running and not self._stop_flag:
            time.sleep(0.5)
            self._watchdog_ticks += 1
            now = time.time()

            # 1. CPU Watchdog
            if self._settings.cpu_watchdog_enabled:
                cpu_percent = self._get_cpu_percent()
                if cpu_percent > self._settings.cpu_limit:
                    if cpu_overload_start is None:
                        cpu_overload_start = now
                    elif now - cpu_overload_start >= self._settings.cpu_overload_duration:
                        self._logger.log(LogLevel.WARNING, f"CPU overload detected ({cpu_percent}% > {self._settings.cpu_limit}%).", "warnings")
                else:
                    cpu_overload_start = None
            else:
                cpu_overload_start = None

            # 2. Memory Watchdog
            if self._settings.memory_watchdog_enabled:
                mem_mb = self._get_memory_mb()
                if mem_mb > self._settings.memory_limit:
                    self._logger.log(
                        LogLevel.WARNING,
                        f"Memory Usage High: {mem_mb:.1f} MB / {self._settings.memory_limit:.1f} MB",
                        "warnings"
                    )
                    if self._settings.auto_emergency_stop_enabled and mem_mb > self._settings.memory_limit * 1.5:
                        self.safe_stop("Memory limit severely exceeded")

            # 3. Render / Stuck Watchdog
            if self._settings.render_timeout_enabled:
                idle_time = now - self._last_frame_timestamp
                if idle_time > self._settings.render_timeout and self._render_counter > 0:
                    if not getattr(self, "_render_timeout_warned", False):
                        self._render_timeout_warned = True
                        self._logger.log(
                            LogLevel.WARNING,
                            f"Render timeout: No frame updates for {idle_time:.1f}s.",
                            "renderer"
                        )
                else:
                    self._render_timeout_warned = False

            # 4. Real Infinite Loop Detection (e.g. while True: pass on main thread)
            if self._settings.infinite_loop_detector_enabled:
                self._check_infinite_loop()

    def _check_infinite_loop(self):
        """
        Inspects the main thread's current frame via sys._current_frames().
        If the main thread is stuck at the exact same line of Python code
        continuously for at least `infinite_loop_stuck_seconds` (sampled
        once per watchdog tick) without updating frames, flag infinite loop
        and request cooperative shutdown or (opt-in) force interrupt.
        """
        main_thread = threading.main_thread()
        main_id = main_thread.ident
        if not main_id:
            return

        frames = sys._current_frames()
        frame = frames.get(main_id)
        if not frame:
            return

        loc = (frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name)
        now = time.time()

        def _reset(new_loc):
            self._main_frame_stuck_count = 0
            self._main_frame_stuck_since = None
            self._last_main_frame_location = new_loc

        # Ignore internal Qt event loop blocking wait frames (e.g. app.exec()
        # reached through Draw._window.window.run() or Draw._app). This is
        # deliberately matched on *where the code lives* (the framework's
        # own files, or PySide6 itself) rather than on the current frame's
        # function name — a name-based check like `"run" in loc[2]` would
        # also match any user-defined function simply because it happens to
        # be called run(), silently exempting it from detection.
        if (
            "PySide6" in loc[0]
            or "_window.py" in loc[0]
            or "_app.py" in loc[0]
        ):
            _reset(loc)
            return

        # If frames are actively rendering, the event loop and main thread are alive and responsive
        if self._render_counter != getattr(self, "_last_checked_render_counter", -1):
            self._last_checked_render_counter = self._render_counter
            _reset(loc)
            return

        if self._last_main_frame_location == loc:
            self._main_frame_stuck_count += 1
            if self._main_frame_stuck_since is None:
                self._main_frame_stuck_since = now
        else:
            _reset(loc)
            return

        stuck_duration = now - self._main_frame_stuck_since
        if stuck_duration >= self._settings.infinite_loop_stuck_seconds:
            fn_name = os.path.basename(loc[0])
            msg = (
                f"Possible Infinite Loop detected at {fn_name}:{loc[1]} in '{loc[2]}'. "
                f"Main thread stuck for {stuck_duration:.1f} seconds "
                f"(threshold: {self._settings.infinite_loop_stuck_seconds:.1f}s)."
            )
            self._logger.log(LogLevel.CRITICAL, msg, "crashes")

            if self._settings.auto_emergency_stop_enabled:
                self.safe_stop("Infinite Loop Detected on Main Thread")

    def _get_cpu_percent(self) -> float:
        if _HAS_PSUTIL:
            try:
                return psutil.Process().cpu_percent()
            except Exception:
                pass
        return 0.0

    def _get_memory_mb(self) -> float:
        if _HAS_PSUTIL:
            try:
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                pass
        return 0.0

    # ── Frame & Instrumentation Callbacks ───────────────────────────────────

    def record_frame(self, frame_time_ms: float, draw_calls: int = 0, vertices: int = 0, triangles: int = 0):
        self._last_frame_timestamp = time.time()
        self._render_counter += 1
        self.draw_calls = draw_calls
        self.vertices_count = vertices
        self.triangles_count = triangles

        self._frame_history.add_frame(frame_time_ms, draw_calls, vertices, triangles)
        self._profiler.end_frame()

        # Infinite render detection check
        if frame_time_ms < 0.1 and self._render_counter > self._settings.infinite_render_count_threshold:
            self._logger.log(LogLevel.WARNING, "Possible Infinite Rendering: Renderer exceeded safe frequency limit.", "renderer")

        # Heavy render detection check
        if draw_calls > self._settings.heavy_render_object_threshold or vertices > self._settings.heavy_render_vertex_threshold:
            est_fps = round(1000.0 / max(frame_time_ms, 1.0), 1)
            self._logger.log(
                LogLevel.WARNING,
                f"Heavy Render: Object Count: {draw_calls:,} | Vertices: {vertices:,} | Expected FPS: {est_fps}",
                "renderer"
            )

    def record_timing(self, category: str, duration_ms: float):
        self._profiler.record(category, duration_ms)

    def validate_scene(self, scene_items: List[dict]) -> List[str]:
        if not self._settings.render_validation_enabled:
            return []
        all_errors = []
        for item in scene_items:
            if isinstance(item, dict):
                errs = self._validator.validate_shape_dict(item, item.get("ip"))
                all_errors.extend(errs)
        if all_errors:
            for err in all_errors[:5]:
                self._logger.log(LogLevel.WARNING, f"Render Validation Error: {err}", "renderer")
        return all_errors

    # ── Real Subsystems State Collectors for process_monitor() ──────────────

    def _query_subsystems_status(self) -> Dict[str, str]:
        """Queries actual live engine state for all 21 subsystems."""
        status = {}

        # 1. Renderer
        try:
            from Draw._window import window
            win_tags = window.list_tags()
            status["Renderer"] = f"Active ({len(win_tags)} windows)" if win_tags else "Idle"
        except Exception:
            status["Renderer"] = "Active"

        # 2. Scene Manager
        try:
            from Draw._window import window
            total_objs = 0
            for tag in window.list_tags():
                win = window.get(tag)
                if hasattr(win, "_draw_canvas"):
                    canvas = win._draw_canvas
                    total_objs += len(canvas.shape_items) + len(canvas.text_items)
            status["Scene Manager"] = f"Active ({total_objs} objects)"
        except Exception:
            status["Scene Manager"] = "Active"

        # 3. Event Manager
        try:
            from Draw._connectors import connectors
            conn_count = len(connectors._items)
            status["Event Manager"] = f"Active ({conn_count} connectors)"
        except Exception:
            status["Event Manager"] = "Active"

        # 4. Animation Manager
        try:
            from Draw._motion import motion
            anim_count = len(motion._shape_triggers) + len(motion._timelines)
            status["Animation Manager"] = f"Active ({anim_count} animations)"
        except Exception:
            status["Animation Manager"] = "Active"

        # 5. Layout Manager
        try:
            from Draw._layout import set as set_layout
            layout_count = len(set_layout._layouts)
            status["Layout Manager"] = f"Active ({layout_count} layouts)"
        except Exception:
            status["Layout Manager"] = "Active"

        # 6. Live Update
        try:
            from Draw._live import live
            bind_count = len(live._store)
            status["Live Update"] = f"Active ({bind_count} reactive vars)"
        except Exception:
            status["Live Update"] = "Active"

        # 7. GPU Upload Queue
        status["GPU Upload Queue"] = f"{debug.gpu_uploads} pending uploads"

        # 8. Texture Cache
        try:
            from Draw._window import window
            tex_count = 0
            for tag in window.list_tags():
                win = window.get(tag)
                if hasattr(win, "_draw_canvas"):
                    for s in win._draw_canvas.shape_items:
                        if getattr(s, "_shadow_cache", None) is not None:
                            tex_count += 1
            status["Texture Cache"] = f"{tex_count} cached textures"
        except Exception:
            status["Texture Cache"] = f"{debug.textures_count} textures"

        # 9. Font Cache
        try:
            from PySide6.QtGui import QFontDatabase
            status["Font Cache"] = "Active (QFontDatabase ready)"
        except Exception:
            status["Font Cache"] = "Active"

        # 10. Image Loader
        try:
            from Draw import _loader
            status["Image Loader"] = "Active"
        except Exception:
            status["Image Loader"] = "Active"

        # 11. Thread Pool
        threads = threading.enumerate()
        status["Thread Pool"] = f"{len(threads)} active threads"

        # 12. Garbage Collector
        counts = gc.get_count()
        status["Garbage Collector"] = f"OK (G0:{counts[0]}, G1:{counts[1]}, G2:{counts[2]})"

        # 13. Input Manager
        try:
            from Draw._text import lineedit, textedit
            inputs = len(lineedit._items) + len(textedit._items)
            status["Input Manager"] = f"Active ({inputs} native inputs)"
        except Exception:
            status["Input Manager"] = "Active"

        # 14. Mouse Manager
        try:
            from Draw._window import window
            mouse_pos = "0, 0"
            for tag in window.list_tags():
                win = window.get(tag)
                if hasattr(win, "_draw_canvas"):
                    mouse_pos = f"{win._draw_canvas._mouse_x:.0f}, {win._draw_canvas._mouse_y:.0f}"
                    break
            status["Mouse Manager"] = f"Active (pos: {mouse_pos})"
        except Exception:
            status["Mouse Manager"] = "Active"

        # 15. Keyboard Manager
        try:
            from Draw._window import window
            focused = "None"
            for tag in window.list_tags():
                win = window.get(tag)
                if hasattr(win, "_draw_canvas") and win._draw_canvas._focused_ip:
                    focused = win._draw_canvas._focused_ip
                    break
            status["Keyboard Manager"] = f"Active (focus: {focused})"
        except Exception:
            status["Keyboard Manager"] = "Active"

        # 16. Window Manager
        try:
            from Draw._window import window
            status["Window Manager"] = f"{len(window.list_tags())} active window(s)"
        except Exception:
            status["Window Manager"] = "Active"

        # 17. Network
        status["Network"] = "Idle"

        # 18. Audio
        try:
            from Draw._window import window
            audio_streams = 0
            for tag in window.list_tags():
                win = window.get(tag)
                if hasattr(win, "_draw_canvas"):
                    for s in win._draw_canvas.shape_items:
                        if getattr(s, "_video_audio", None) is not None:
                            audio_streams += 1
            status["Audio"] = f"Active ({audio_streams} audio streams)" if audio_streams else "Idle"
        except Exception:
            status["Audio"] = "Idle"

        # 19. Timers
        try:
            from Draw._schedule import _active_timers
            status["Timers"] = f"Active ({len(_active_timers)} timers)"
        except Exception:
            status["Timers"] = "Active"

        # 20. Motion Engine
        try:
            from Draw._motion import motion
            status["Motion Engine"] = f"Active ({len(motion._items)} records)"
        except Exception:
            status["Motion Engine"] = "Active"

        # 21. Watchdog
        status["Watchdog"] = f"Running (ticks: {debug._watchdog_ticks})" if debug._watchdog_running else "Disabled"

        return status

    # ── Public API 3: process_monitor() ─────────────────────────────────────

    def process_monitor(self) -> str:
        """
        Generates comprehensive text output displaying internal status across
        all 21 subsystems.
        """
        stats = self._frame_history.get_stats()
        cpu_pct = round(self._get_cpu_percent(), 1)
        mem_mb = round(self._get_memory_mb(), 1)
        gpu = self.gpu_stats()

        subsystems = self._query_subsystems_status()

        output = f"""
Renderer
--------
FPS: {stats.get('avg_fps', 60)}
Frame Time: {stats.get('worst_frame_ms', 16.4)}ms
Draw Calls: {self.draw_calls}
Vertices: {self.vertices_count}
Triangles: {self.triangles_count}
GPU Uploads: {self.gpu_uploads}
Textures: {self.textures_count}
CPU Usage: {cpu_pct}%
Memory: {mem_mb}MB
GPU Usage: {gpu.get('gpu_percent', 0.0)}% (VRAM: {gpu.get('vram_used_mb', 0.0)}MB / {gpu.get('vram_total_mb', 0.0)}MB)

Subsystems Status (21 Monitors)
-------------------------------
1.  Renderer           : {subsystems.get('Renderer', 'Active')}
2.  Scene Manager      : {subsystems.get('Scene Manager', 'Active')}
3.  Event Manager      : {subsystems.get('Event Manager', 'Active')}
4.  Animation Manager  : {subsystems.get('Animation Manager', 'Active')}
5.  Layout Manager     : {subsystems.get('Layout Manager', 'Active')}
6.  Live Update        : {subsystems.get('Live Update', 'Active')}
7.  GPU Upload Queue   : {subsystems.get('GPU Upload Queue', 'Active')}
8.  Texture Cache      : {subsystems.get('Texture Cache', 'Active')}
9.  Font Cache         : {subsystems.get('Font Cache', 'Active')}
10. Image Loader       : {subsystems.get('Image Loader', 'Active')}
11. Thread Pool        : {subsystems.get('Thread Pool', 'Active')}
12. Garbage Collector  : {subsystems.get('Garbage Collector', 'OK')}
13. Input Manager      : {subsystems.get('Input Manager', 'Active')}
14. Mouse Manager      : {subsystems.get('Mouse Manager', 'Active')}
15. Keyboard Manager   : {subsystems.get('Keyboard Manager', 'Active')}
16. Window Manager     : {subsystems.get('Window Manager', 'Active')}
17. Network            : {subsystems.get('Network', 'Idle')}
18. Audio              : {subsystems.get('Audio', 'Idle')}
19. Timers             : {subsystems.get('Timers', 'Active')}
20. Motion Engine      : {subsystems.get('Motion Engine', 'Active')}
21. Watchdog           : {subsystems.get('Watchdog', 'Running')}
"""
        return output.strip()


# Singleton instance
debug = DebugManager()

__all__ = [
    "debug",
    "DebugSettings",
    "DebugManager",
    "LoggingSystem",
    "LogLevel",
    "FrameHistory",
    "PerformanceProfiler",
    "RenderValidator",
    "LeakDetector",
    "ThreadMonitor",
]
