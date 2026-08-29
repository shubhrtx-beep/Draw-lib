"""
Draw._motion
============
Structured 2D motion parsing and runtime physics/animation engine for Draw.

Public time values are seconds and may be floats.
"""

from __future__ import annotations

import ast
import copy
import math
import threading
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtGui import QPainterPath, QPolygonF
from PySide6.QtCore import QPointF, QRectF

MotionHandler = Callable[["MotionRecord", Any, dict[str, Any]], None]
GraphFunc = Callable[[float], float]

_BUILTIN_MOTION_TYPES = {
    "move",
    "expand",
    "size",
    "scale",
    "rotate",
    "rotation",
    "opacity",
    "alpha",
    "blur",
    "color",
    "colour",
    "glow",
    "path",
    "transform",
    "custom",
    "x",
    "y",
    "pos",
    "position",
    "vertices",
    "morph",
    "benzene",
    "skew",
    "shear",
    "rotate_x",
    "rotate_y",
    "rotate_3d",
    "perspective",
    "trim_path",
    "stroke_dash",
    "shake",
    "wiggle",
    "wave",
    "pulse",
    "gravity",
    "bounce_physics",
    "spring",
    "inertia",
    "pendulum",
    "orbit",
    "polygon_orbit",
    "lissajous",
    "spiral",
    "attractor",
    "stretch_squash",
    "noise",
    "projectile",
}

_PROCEDURAL_TYPES = {
    "shake",
    "wiggle",
    "wave",
    "pulse",
    "gravity",
    "bounce_physics",
    "spring",
    "inertia",
    "pendulum",
    "orbit",
    "polygon_orbit",
    "benzene",
    "lissajous",
    "spiral",
    "attractor",
    "stretch_squash",
    "noise",
    "projectile",
}

_TARGET_ALIASES = {
    "shape": "shape",
    "shapes": "shape",
    "hitbox": "hitbox",
    "hitboxes": "hitbox",
    "connector": "connector",
    "connectors": "connector",
    "custom": "custom.motion",
    "custom_motion": "custom.motion",
    "custom.motion": "custom.motion",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_bool(value: object, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off", ""}:
            return False
    return bool(value)


def _resolve_dynamic(value: Any) -> Any:
    if callable(value):
        return value()
    return value


def _normalize_target(raw: object, *, default: str = "shape") -> str:
    token = str(raw or default).strip().lower()
    normalized = _TARGET_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(sorted(set(_TARGET_ALIASES.values())))
        raise ValueError(f"Draw.motion: target must be one of [{allowed}].")
    return normalized


@dataclass
class TargetRef:
    target: str
    ip: Optional[str]

    def to_dict(self) -> dict[str, Optional[str]]:
        return {"target": self.target, "ip": self.ip}


class DrawExprSolver:
    """Evaluates motion expressions safely using AST parsing instead of exec()."""
    def __init__(self, expression: str, parameters: Optional[dict[str, Any]] = None):
        self.expression = expression
        self.parameters = parameters or {}
        try:
            self._ast = ast.parse(expression, mode='exec')
        except SyntaxError as e:
            raise ValueError(f"DrawExprSolver: invalid expression: {e}") from e

    def evaluate(self, locals_dict: dict[str, Any]) -> dict[str, Any]:
        ctx = dict(_EXPR_GLOBALS)
        eval_locals = {}
        eval_locals.update(self.parameters)
        eval_locals.update(locals_dict)
        for node in self._ast.body:
            if isinstance(node, ast.Assign):
                value = _safe_eval_node(node.value, ctx, eval_locals)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        eval_locals[target.id] = value
            elif isinstance(node, ast.Expr):
                _safe_eval_node(node.value, ctx, eval_locals)
        return eval_locals


def _safe_eval_node(node: ast.AST, ctx: dict, local: dict) -> Any:
    """Recursively evaluate a single AST expression node safely."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in local:
            return local[node.id]
        if node.id in ctx:
            return ctx[node.id]
        raise ValueError(f"DrawExprSolver: unknown variable '{node.id}'")
    if isinstance(node, ast.BinOp):
        left = _safe_eval_node(node.left, ctx, local)
        right = _safe_eval_node(node.right, ctx, local)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
        }
        op_func = ops.get(type(node.op))
        if op_func is None:
            raise ValueError(f"DrawExprSolver: unsupported operator {type(node.op).__name__}")
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_node(node.operand, ctx, local)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"DrawExprSolver: unsupported unary op {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        func = _safe_eval_node(node.func, ctx, local)
        if not callable(func):
            raise ValueError(f"DrawExprSolver: '{func}' is not callable")
        args = [_safe_eval_node(a, ctx, local) for a in node.args]
        return func(*args)
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left, ctx, local)
        for op, comparator in zip(node.ops, node.comparators):
            right = _safe_eval_node(comparator, ctx, local)
            cmp_ops = {
                ast.Gt: lambda a, b: a > b,
                ast.GtE: lambda a, b: a >= b,
                ast.Lt: lambda a, b: a < b,
                ast.LtE: lambda a, b: a <= b,
                ast.Eq: lambda a, b: a == b,
                ast.NotEq: lambda a, b: a != b,
            }
            cmp_func = cmp_ops.get(type(op))
            if cmp_func is None:
                raise ValueError(f"DrawExprSolver: unsupported comparison {type(op).__name__}")
            if not cmp_func(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.IfExp):
        test = _safe_eval_node(node.test, ctx, local)
        return _safe_eval_node(node.body, ctx, local) if test else _safe_eval_node(node.orelse, ctx, local)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _safe_eval_node(v, ctx, local)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = _safe_eval_node(v, ctx, local)
                if result:
                    return result
            return result
    if isinstance(node, ast.Subscript):
        value = _safe_eval_node(node.value, ctx, local)
        sl = _safe_eval_node(node.slice, ctx, local)
        return value[sl]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval_node(e, ctx, local) for e in node.elts)
    if isinstance(node, ast.List):
        return [_safe_eval_node(e, ctx, local) for e in node.elts]
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("__"):
            raise ValueError(f"DrawExprSolver: dunder attribute '{node.attr}' access disallowed")
        value = _safe_eval_node(node.value, ctx, local)
        return getattr(value, node.attr)
    raise ValueError(f"DrawExprSolver: unsupported expression node {type(node).__name__}")


# ── Smooth 2D Simplex Noise Solver ──────────────────────────────────────────

_GRAD2 = [
    (1, 1), (-1, 1), (1, -1), (-1, -1),
    (1, 0), (-1, 0), (0, 1), (0, -1),
]

_PERM = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
] * 2


def _simplex_noise_2d(xin: float, yin: float) -> float:
    """Fast, smooth 2D Simplex continuous gradient noise in range [-1, 1]."""
    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0

    s = (xin + yin) * F2
    i = math.floor(xin + s)
    j = math.floor(yin + s)

    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = xin - X0
    y0 = yin - Y0

    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ii = int(i) & 255
    jj = int(j) & 255

    gi0 = _PERM[ii + _PERM[jj]] % 8
    gi1 = _PERM[ii + i1 + _PERM[jj + j1]] % 8
    gi2 = _PERM[ii + 1 + _PERM[jj + 1]] % 8

    n0 = n1 = n2 = 0.0

    t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 >= 0:
        t0 *= t0
        g0 = _GRAD2[gi0]
        n0 = t0 * t0 * (g0[0] * x0 + g0[1] * y0)

    t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 >= 0:
        t1 *= t1
        g1 = _GRAD2[gi1]
        n1 = t1 * t1 * (g1[0] * x1 + g1[1] * y1)

    t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 >= 0:
        t2 *= t2
        g2 = _GRAD2[gi2]
        n2 = t2 * t2 * (g2[0] * x2 + g2[1] * y2)

    return 70.0 * (n0 + n1 + n2)


def _pseudo_noise(x: float, y: float = 0.0, z: float = 0.0) -> float:
    """Continuous smooth 2D noise in range [-1, 1]."""
    return _simplex_noise_2d(x, y + z)


def solve_wiggle(t: float, frequency: float = 5.0, amplitude: float = 10.0, octaves: int = 2) -> tuple[float, float]:
    """Computes smooth procedural multi-octave 2D displacement wiggle."""
    dx = 0.0
    dy = 0.0
    for i in range(1, max(1, octaves) + 1):
        freq = frequency * i * 0.8
        amp = amplitude / (i * 0.75)
        dx += _simplex_noise_2d(t * freq, i * 13.5) * amp
        dy += _simplex_noise_2d(t * freq + 100.0, i * 27.3) * amp
    return dx, dy


def solve_wave(t: float, frequency: float = 2.0, amplitude: float = 20.0, phase: float = 0.0) -> float:
    """Computes continuous sinusoidal wave value."""
    return math.sin(t * frequency * 2.0 * math.pi + phase) * amplitude


# ── Analytical 2D Physics Solvers ──────────────────────────────────────────

def solve_gravity_bounce(
    t: float,
    v0: float = 0.0,
    g: float = 980.0,
    restitution: float = 0.7,
    floor_y: float = 500.0,
    start_y: float = 0.0,
) -> float:
    """
    Physical falling body with ground bounce dynamics.
    Evaluates in O(1) time analytically to eliminate frame drop bottlenecks.
    """
    if t <= 0.0:
        return start_y
    if start_y >= floor_y:
        return floor_y
    if g <= 0.0:
        return start_y + v0 * t

    dy = floor_y - start_y
    disc = v0 * v0 + 2.0 * g * dy
    if disc <= 0.0:
        return floor_y

    t1 = (-v0 + math.sqrt(disc)) / g
    if t <= t1:
        return start_y + v0 * t + 0.5 * g * t * t

    v_impact1 = v0 + g * t1
    v_rebound = v_impact1 * restitution
    t_curr = t - t1

    while v_rebound > 1.0:
        t_bounce = (2.0 * v_rebound) / g
        if t_curr <= t_bounce:
            return floor_y - (v_rebound * t_curr - 0.5 * g * t_curr * t_curr)
        t_curr -= t_bounce
        v_rebound *= restitution

    return floor_y


def solve_projectile_2d(
    t: float,
    vx0: float = 200.0,
    vy0: float = -300.0,
    g: float = 980.0,
    drag: float = 0.05,
    restitution: float = 0.75,
    floor_y: float = 500.0,
    wall_x: Optional[float] = None,
    start_x: float = 0.0,
    start_y: float = 0.0,
) -> tuple[float, float]:
    """Analytical 2D parabolic projectile motion with air drag and boundary reflection."""
    if t <= 0.0:
        return start_x, start_y

    d_factor = math.exp(-drag * t) if drag > 0.0 else 1.0
    dx = vx0 * (1.0 - d_factor) / drag if drag > 0.0 else vx0 * t
    cur_x = start_x + dx

    if wall_x is not None and cur_x >= wall_x:
        cur_x = wall_x - abs(cur_x - wall_x) * restitution

    cur_y = solve_gravity_bounce(t, vy0, g, restitution, floor_y, start_y)
    return cur_x, cur_y


def solve_pendulum(
    t: float,
    amplitude: float = 45.0,
    frequency: float = 1.5,
    damping: float = 0.25,
    phase: float = 0.0,
) -> float:
    """Computes angular damped pendulum oscillation in degrees."""
    if t <= 0.0:
        return amplitude
    envelope = math.exp(-damping * t)
    return amplitude * envelope * math.cos(2.0 * math.pi * frequency * t + phase)


def solve_lissajous(
    t: float,
    amplitude_x: float = 100.0,
    amplitude_y: float = 80.0,
    freq_x: float = 3.0,
    freq_y: float = 2.0,
    phase_x: float = 0.0,
    phase_y: float = math.pi / 2.0,
) -> tuple[float, float]:
    """Computes 2D Lissajous curve orbital coordinates."""
    x = amplitude_x * math.sin(2.0 * math.pi * freq_x * t + phase_x)
    y = amplitude_y * math.sin(2.0 * math.pi * freq_y * t + phase_y)
    return x, y


def solve_spiral(
    t: float,
    a: float = 10.0,
    b: float = 15.0,
    frequency: float = 1.0,
    logarithmic: bool = False,
) -> tuple[float, float]:
    """Computes 2D Archimedean or Logarithmic spiral orbital coordinates."""
    theta = 2.0 * math.pi * frequency * t
    if logarithmic:
        r = a * math.exp(0.1 * b * theta)
    else:
        r = a + b * theta
    return r * math.cos(theta), r * math.sin(theta)


def solve_polygon_orbit(
    t: float,
    radius: float = 100.0,
    sides: int = 6,
    frequency: float = 0.5,
) -> tuple[float, float]:
    """General N-sided regular polygon orbital path (generalizes hexagon/benzene)."""
    sides = max(3, sides)
    angle_rad = 2.0 * math.pi * frequency * t
    sector = (2.0 * math.pi) / sides
    half_sector = sector / 2.0
    theta = (angle_rad % sector) - half_sector
    cos_t = math.cos(theta)
    r = radius * math.cos(half_sector) / (cos_t if abs(cos_t) > 1e-9 else 1e-9)
    return r * math.cos(angle_rad), r * math.sin(angle_rad)


def solve_stretch_squash(
    t: float,
    amplitude: float = 0.3,
    frequency: float = 2.0,
    damping: float = 1.0,
) -> tuple[float, float]:
    """Area-preserving 2D Stretch & Squash (scale_y = 1 / scale_x)."""
    env = math.exp(-damping * t) if damping > 0.0 else 1.0
    delta = amplitude * env * math.sin(2.0 * math.pi * frequency * t)
    sx = max(0.1, 1.0 + delta)
    sy = 1.0 / sx
    return sx, sy


def solve_attractor_2d(
    t: float,
    cur_x: float,
    cur_y: float,
    target_x: float,
    target_y: float,
    strength: float = 5.0,
) -> tuple[float, float]:
    """Computes 2D point attraction toward target coordinates."""
    factor = 1.0 - math.exp(-strength * max(0.0, t))
    new_x = cur_x + (target_x - cur_x) * factor
    new_y = cur_y + (target_y - cur_y) * factor
    return new_x, new_y


_EXPR_GLOBALS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "atan2": math.atan2,
    "abs": abs,
    "min": min,
    "max": max,
    "exp": math.exp,
    "pow": pow,
    "sqrt": math.sqrt,
    "round": round,
    "floor": math.floor,
    "degrees": math.degrees,
    "radians": math.radians,
    "sign": lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
    "fract": lambda x: x - math.floor(x),
    "smoothstep": lambda e0, e1, x: 0.0 if x <= e0 else (1.0 if x >= e1 else pow((x - e0)/(e1 - e0), 2) * (3 - 2 * (x - e0)/(e1 - e0))),
    "mod": lambda a, b: math.fmod(a, b),
    "lerp": lambda a, b, t: a + (b - a) * t,
    "step": lambda edge, n: 1.0 if n >= edge else 0.0,
    "clamp": lambda n, lo, hi: max(lo, min(hi, n)),
    "noise": _pseudo_noise,
    "simplex": _simplex_noise_2d,
}


@dataclass
class MotionRecord:
    motion_type: str
    target: str
    from_value: Any
    to_value: Any
    start: float
    end: float
    graph: str = "linear"
    easing: Optional[str] = None
    repeat: bool = False
    reverse: bool = False
    delay: float = 0.0
    index: Optional[int] = None
    custom_data: Any = None
    raw: Dict[str, Any] = field(default_factory=dict)

    trigger: Optional[str] = None
    compiled_solver: Optional[DrawExprSolver] = None
    resolved_from_vertices: Optional[list[tuple[float, float]]] = None
    resolved_to_vertices: Optional[list[tuple[float, float]]] = None
    resolved_path_vertices: Optional[list[tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.motion_type,
            "target": self.target,
            "from": self.from_value,
            "to": self.to_value,
            "time": {"start": self.start, "end": self.end},
            "graph": self.graph,
            "easing": self.easing,
            "repeat": self.repeat,
            "reverse": self.reverse,
            "delay": self.delay,
            "index": self.index,
            "custom_data": self.custom_data,
        }


@dataclass
class CustomMotionRecord:
    ip: Optional[str]
    get_ip: Optional[str]
    return_value: Any
    tools: list
    get_custom: Any
    motion: List[MotionRecord]
    current_value: Any = None
    started_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ip": self.ip,
            "get_ip": self.get_ip,
            "return_value": self.return_value,
            "tools": list(self.tools),
            "get_custom": self.get_custom,
            "motion": [m.to_dict() for m in self.motion],
            "current_value": self.current_value,
        }


class Timeline:
    def __init__(
        self,
        tracks: Optional[list] = None,
        targets: Optional[list] = None,
        stagger: float = 0.0,
        motion_list: Optional[list] = None,
        duration: float = 0.4,
        graph: str = "ease_out",
        sequence: Optional[list] = None,
        repeat: bool = False,
        reverse: bool = False,
        **kwargs
    ):
        motion_list = motion_list or kwargs.get("motion") or []

        self.tracks = tracks or []
        self.targets = targets or []
        self.stagger = stagger
        self.motion_list = motion_list
        self.duration = duration
        self.graph = graph
        self.sequence = sequence or []
        self.repeat = repeat
        self.reverse = reverse

        self.started_at = None
        self.records: list[tuple[str, MotionRecord]] = []
        self.parse_timeline()

    def parse_timeline(self):
        if self.tracks:
            for track in self.tracks:
                target = track.get("target")
                raw_motions = track.get("motion", [])
                time_spec = track.get("time", {"start": 0.0, "end": 0.3})
                for rm in raw_motions:
                    item = copy.deepcopy(rm)
                    item["time"] = time_spec
                    if "graph" not in item:
                        item["graph"] = track.get("graph", "linear")
                    record = motion._parse_motion_item(item, 0)
                    self.records.append((target, record))
        elif self.targets:
            for idx, target in enumerate(self.targets):
                start = idx * self.stagger
                end = start + self.duration
                for rm in self.motion_list:
                    item = copy.deepcopy(rm)
                    item["time"] = {"start": start, "end": end}
                    item["graph"] = self.graph
                    record = motion._parse_motion_item(item, 0)
                    self.records.append((target, record))
        elif self.sequence:
            curr_time = 0.0
            for item in self.sequence:
                target = item.get("target")
                raw_motions = item.get("motion", [])
                duration = float(item.get("duration", 0.4))
                delay = float(item.get("delay", 0.0))

                start = curr_time + delay
                end = start + duration
                curr_time = end

                for rm in raw_motions:
                    m_item = copy.deepcopy(rm)
                    m_item["time"] = {"start": start, "end": end}
                    if "graph" not in m_item:
                        m_item["graph"] = item.get("graph", "linear")
                    record = motion._parse_motion_item(m_item, 0)
                    self.records.append((target, record))


def solve_bezier_u(x: float, x1: float, x2: float) -> float:
    low = 0.0
    high = 1.0
    for _ in range(20):
        u = (low + high) / 2.0
        cx = 3.0 * (1.0 - u) * (1.0 - u) * u * x1 + 3.0 * (1.0 - u) * u * u * x2 + u * u * u
        if cx < x:
            low = u
        else:
            high = u
    return (low + high) / 2.0


def solve_cubic_bezier(t: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Evaluates a 2D Cubic Bezier curve at parameterized progress t in [0, 1]."""
    x = max(0.0, min(1.0, t))
    u = solve_bezier_u(x, x1, x2)
    return 3.0 * (1.0 - u) * (1.0 - u) * u * y1 + 3.0 * (1.0 - u) * u * u * y2 + u * u * u


def get_distance_and_direction(v0: Any, v1: Any) -> tuple[float, Any]:
    if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
        diff = float(v1) - float(v0)
        return abs(diff), (1.0 if diff >= 0 else -1.0)
    if isinstance(v0, (list, tuple)) and isinstance(v1, (list, tuple)) and len(v0) == len(v1):
        squared_sum = sum((float(b) - float(a)) ** 2 for a, b in zip(v0, v1))
        dist = math.sqrt(squared_sum)
        return dist, 1.0
    return 0.0, 1.0


def evaluate_keyframe_segment(
    t: float,
    t0: float, v0: Any, outgoing: dict,
    t1: float, v1: Any, incoming: dict,
) -> Any:
    dt = t1 - t0
    if dt <= 0.0:
        return v1

    dist, _ = get_distance_and_direction(v0, v1)
    x = max(0.0, min(1.0, (t - t0) / dt))

    infl_out = float(outgoing.get("influence", 33.333)) / 100.0
    infl_in = float(incoming.get("influence", 33.333)) / 100.0

    linear_speed = dist / dt if dt > 0 else 0.0
    speed_out = float(outgoing.get("speed", linear_speed))
    speed_in = float(incoming.get("speed", linear_speed))

    x1 = infl_out
    x2 = 1.0 - infl_in

    if dist > 0.0:
        y1 = infl_out * speed_out * dt / dist
        y2 = 1.0 - infl_in * speed_in * dt / dist
    else:
        y1 = 0.0
        y2 = 1.0

    u = solve_bezier_u(x, x1, x2)
    y = 3.0 * (1.0 - u) * (1.0 - u) * u * y1 + 3.0 * (1.0 - u) * u * u * y2 + u * u * u
    y = max(0.0, min(1.0, y))

    return motion.interpolate(v0, v1, y)


def solve_spring(
    t: float,
    v_from: float, v_to: float, v0: float,
    stiffness: float, damping: float, mass: float,
) -> float:
    if t <= 0.0:
        return v_from
    z0 = v_from - v_to
    k = stiffness
    c = damping
    m = mass

    if m <= 0.0 or k <= 0.0:
        return v_to

    w0 = math.sqrt(k / m)
    zeta = c / (2.0 * math.sqrt(m * k))

    if zeta < 1.0:
        wd = w0 * math.sqrt(1.0 - zeta * zeta)
        a = z0
        b = (v0 + zeta * w0 * z0) / wd
        zt = math.exp(-zeta * w0 * t) * (a * math.cos(wd * t) + b * math.sin(wd * t))
    elif abs(zeta - 1.0) < 1e-6:
        a = z0
        b = v0 + w0 * z0
        zt = math.exp(-w0 * t) * (a + b * t)
    else:
        w_star = w0 * math.sqrt(zeta * zeta - 1.0)
        a = z0
        b = (v0 + zeta * w0 * z0) / w_star
        zt = math.exp(-zeta * w0 * t) * (a * math.cosh(w_star * t) + b * math.sinh(w_star * t))

    return v_to + zt


def solve_inertia(
    t: float,
    v_from: float, v0: float,
    friction: float,
    bounds: Optional[dict[str, float]] = None,
) -> float:
    if t <= 0.0:
        return float(v_from) if v_from is not None else 0.0
    f = friction
    if f >= 1.0 or f <= 0.0:
        val = v_from + v0 * t
    else:
        ln_f = math.log(f)
        val = v_from + (v0 / (60.0 * ln_f)) * (math.pow(f, 60.0 * t) - 1.0)

    if bounds:
        min_val = bounds.get("min")
        max_val = bounds.get("max")
        if min_val is not None:
            val = max(min_val, val)
        if max_val is not None:
            val = min(max_val, val)
    return val


def parse_svg_path(path_str: str) -> QPainterPath:
    path = QPainterPath()
    tokens = re.findall(r'([MmLlHhVvCcSsZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)', path_str)

    commands = []
    for cmd, val in tokens:
        if cmd:
            commands.append((cmd, []))
        elif val:
            if not commands:
                commands.append(('L', [float(val)]))
            else:
                commands[-1][1].append(float(val))

    curr_x, curr_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0

    for cmd, args in commands:
        cmd_lower = cmd.lower()
        if cmd_lower == 'm':
            for i in range(0, len(args), 2):
                if i + 1 < len(args):
                    dx, dy = args[i], args[i+1]
                    if cmd == 'm':
                        curr_x += dx
                        curr_y += dy
                    else:
                        curr_x = dx
                        curr_y = dy
                    if i == 0:
                        path.moveTo(curr_x, curr_y)
                        start_x, start_y = curr_x, curr_y
                    else:
                        path.lineTo(curr_x, curr_y)
        elif cmd_lower == 'l':
            for i in range(0, len(args), 2):
                if i + 1 < len(args):
                    dx, dy = args[i], args[i+1]
                    if cmd == 'l':
                        curr_x += dx
                        curr_y += dy
                    else:
                        curr_x = dx
                        curr_y = dy
                    path.lineTo(curr_x, curr_y)
        elif cmd_lower == 'h':
            for dx in args:
                if cmd == 'h':
                    curr_x += dx
                else:
                    curr_x = dx
                path.lineTo(curr_x, curr_y)
        elif cmd_lower == 'v':
            for dy in args:
                if cmd == 'v':
                    curr_y += dy
                else:
                    curr_y = dy
                path.lineTo(curr_x, curr_y)
        elif cmd_lower == 'c':
            for i in range(0, len(args), 6):
                if i + 5 < len(args):
                    dx1, dy1 = args[i], args[i+1]
                    dx2, dy2 = args[i+2], args[i+3]
                    dx3, dy3 = args[i+4], args[i+5]
                    if cmd == 'c':
                        x1, y1 = curr_x + dx1, curr_y + dy1
                        x2, y2 = curr_x + dx2, curr_y + dy2
                        x3, y3 = curr_x + dx3, curr_y + dy3
                    else:
                        x1, y1 = dx1, dy1
                        x2, y2 = dx2, dy2
                        x3, y3 = dx3, dy3
                    path.cubicTo(x1, y1, x2, y2, x3, y3)
                    curr_x, curr_y = x3, y3
        elif cmd_lower == 's':
            for i in range(0, len(args), 4):
                if i + 3 < len(args):
                    dx2, dy2 = args[i], args[i+1]
                    dx3, dy3 = args[i+2], args[i+3]
                    x1, y1 = curr_x, curr_y
                    if cmd == 's':
                        x2, y2 = curr_x + dx2, curr_y + dy2
                        x3, y3 = curr_x + dx3, curr_y + dy3
                    else:
                        x2, y2 = dx2, dy2
                        x3, y3 = dx3, dy3
                    path.cubicTo(x1, y1, x2, y2, x3, y3)
                    curr_x, curr_y = x3, y3
        elif cmd_lower == 'z':
            path.closeSubpath()
            curr_x, curr_y = start_x, start_y

    return path


def shape_spec_to_path(spec: dict[str, Any]) -> QPainterPath:
    path = QPainterPath()
    shape_type = spec.get("shape", "").strip().lower()

    if shape_type == "svg":
        svg_path_str = spec.get("path", "")
        path = parse_svg_path(svg_path_str)
    elif shape_type == "circle":
        radius = float(spec.get("radius", 50.0))
        cx = float(spec.get("cx", 0.0))
        cy = float(spec.get("cy", 0.0))
        path.addEllipse(QPointF(cx, cy), radius, radius)
    elif shape_type in ("rectangle", "rect"):
        w = float(spec.get("width", 100.0))
        h = float(spec.get("height", 100.0))
        rx = float(spec.get("border_radius", spec.get("rx", 0.0)))
        ry = float(spec.get("border_radius", spec.get("ry", rx)))
        cx = float(spec.get("cx", 0.0))
        cy = float(spec.get("cy", 0.0))
        rect = QRectF(cx - w/2.0, cy - h/2.0, w, h)
        if rx > 0:
            path.addRoundedRect(rect, rx, ry)
        else:
            path.addRect(rect)
    return path


def path_to_vertices(path: QPainterPath) -> list[tuple[float, float]]:
    polys = path.toSubpathPolygons()
    vertices = []
    for poly in polys:
        for pt in poly:
            vertices.append((pt.x(), pt.y()))
    filtered = []
    for pt in vertices:
        if not filtered or pt != filtered[-1]:
            filtered.append(pt)
    return filtered


def resample_subdivide(vertices: list[tuple[float, float]], target_count: int) -> list[tuple[float, float]]:
    pts = list(vertices)
    if not pts:
        return [(0.0, 0.0)] * target_count
    while len(pts) < target_count:
        max_len = -1.0
        insert_idx = -1
        midpoint = (0.0, 0.0)
        for i in range(len(pts)):
            p1 = pts[i]
            p2 = pts[(i + 1) % len(pts)]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist > max_len:
                max_len = dist
                insert_idx = i
                midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        pts.insert(insert_idx + 1, midpoint)
    return pts


def resample_distribute(vertices: list[tuple[float, float]], target_count: int) -> list[tuple[float, float]]:
    if not vertices:
        return [(0.0, 0.0)] * target_count
    pts = list(vertices)
    if len(pts) > 1 and pts[0] != pts[-1]:
        pts.append(pts[0])

    cum_len = [0.0]
    total = 0.0
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i+1]
        total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        cum_len.append(total)

    if total == 0.0:
        return [pts[0]] * target_count

    resampled = []
    for j in range(target_count):
        target_dist = j * (total / target_count)
        idx = 0
        while idx < len(cum_len) - 2 and cum_len[idx+1] < target_dist:
            idx += 1
        d0 = cum_len[idx]
        d1 = cum_len[idx+1]
        segment_len = d1 - d0
        t = (target_dist - d0) / segment_len if segment_len > 0.0 else 0.0
        p1 = pts[idx]
        p2 = pts[idx+1]
        rx = p1[0] + t * (p2[0] - p1[0])
        ry = p1[1] + t * (p2[1] - p1[1])
        resampled.append((rx, ry))
    return resampled


def resample_polar_map(vertices: list[tuple[float, float]], target_count: int) -> list[tuple[float, float]]:
    if not vertices:
        return [(0.0, 0.0)] * target_count
    cx = sum(p[0] for p in vertices) / len(vertices)
    cy = sum(p[1] for p in vertices) / len(vertices)

    resampled = []
    for j in range(target_count):
        angle = j * (2.0 * math.pi / target_count)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        intersection = None
        min_positive_t = float('inf')

        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1

            det = dx * sin_a - dy * cos_a
            if abs(det) > 1e-9:
                t_ray = (dx * (y1 - cy) - dy * (x1 - cx)) / det
                u_seg = (cos_a * (y1 - cy) - sin_a * (x1 - cx)) / det
                if 0.0 <= u_seg <= 1.0 and t_ray >= 0.0:
                    if t_ray < min_positive_t:
                        min_positive_t = t_ray
                        intersection = (cx + t_ray * cos_a, cy + t_ray * sin_a)

        if intersection is not None:
            resampled.append(intersection)
        else:
            closest_v = None
            min_d_angle = float('inf')
            for p in vertices:
                v_angle = math.atan2(p[1] - cy, p[0] - cx)
                d_angle = abs(math.atan2(math.sin(v_angle - angle), math.cos(v_angle - angle)))
                if d_angle < min_d_angle:
                    min_d_angle = d_angle
                    closest_v = p
            resampled.append(closest_v if closest_v is not None else (cx, cy))
    return resampled


def resolve_morph_vertices(from_spec: dict, to_spec: dict, resample_mode: str) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    from_path = shape_spec_to_path(from_spec)
    to_path = shape_spec_to_path(to_spec)

    from_pts = path_to_vertices(from_path)
    to_pts = path_to_vertices(to_path)

    target_count = max(len(from_pts), len(to_pts), 64)

    resample_fn = resample_distribute
    if resample_mode == "subdivide":
        resample_fn = resample_subdivide
    elif resample_mode == "polar_map":
        resample_fn = resample_polar_map

    resolved_from = resample_fn(from_pts, target_count)
    resolved_to = resample_fn(to_pts, target_count)
    return resolved_from, resolved_to


def resolve_motion_path_vertices(path_spec: Any, closed: bool = False) -> list[tuple[float, float]]:
    if isinstance(path_spec, str):
        vertices = path_to_vertices(parse_svg_path(path_spec))
    elif isinstance(path_spec, dict):
        vertices = path_to_vertices(shape_spec_to_path(path_spec))
    elif isinstance(path_spec, (list, tuple)):
        vertices = [(float(p[0]), float(p[1])) for p in path_spec]
    else:
        raise ValueError(
            "Draw.motion: 'path' must be an SVG path string, a shape spec "
            "dict, or a list of (x, y) points."
        )
    if closed and len(vertices) > 1 and vertices[0] != vertices[-1]:
        vertices = vertices + [vertices[0]]
    return vertices


def sample_polyline_at(vertices: list[tuple[float, float]], u: float) -> tuple[float, float, float]:
    if not vertices:
        return 0.0, 0.0, 0.0
    if len(vertices) == 1:
        return vertices[0][0], vertices[0][1], 0.0

    u = max(0.0, min(1.0, u))
    cum_len = [0.0]
    total = 0.0
    for i in range(len(vertices) - 1):
        p1, p2 = vertices[i], vertices[i + 1]
        total += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        cum_len.append(total)

    if total == 0.0:
        return vertices[0][0], vertices[0][1], 0.0

    target = u * total
    idx = 0
    while idx < len(cum_len) - 2 and cum_len[idx + 1] < target:
        idx += 1

    p1, p2 = vertices[idx], vertices[idx + 1]
    seg_len = cum_len[idx + 1] - cum_len[idx]
    t_seg = 0.0 if seg_len <= 0.0 else (target - cum_len[idx]) / seg_len

    x = p1[0] + t_seg * (p2[0] - p1[0])
    y = p1[1] + t_seg * (p2[1] - p1[1])
    heading = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    return x, y, heading


def find_elements_by_ip(ip: str) -> list[object]:
    from Draw._window import window as _window_registry
    elements = []
    for tag in _window_registry.list_all_tags():
        win = _window_registry.get(tag)
        if hasattr(win, '_draw_canvas'):
            canvas = win._draw_canvas
            for s in canvas.shape_items:
                if getattr(s, "ip", None) == ip:
                    elements.append(s)
            for t in canvas.text_items:
                if getattr(t, "ip", None) == ip:
                    elements.append(t)
    try:
        from Draw._file_tree import filetree as _file_tree_registry
        ft = _file_tree_registry.get(ip)
        if ft is not None:
            elements.append(ft)
    except Exception:
        pass
    return elements


class MotionRegistry:
    """Singleton exposed as Draw.motion."""

    def __init__(self) -> None:
        self._type_handlers: dict[str, Optional[MotionHandler]] = {
            name: None for name in _BUILTIN_MOTION_TYPES
        }

        # Comprehensive Easing Functions Suite
        def _ease_in_quad(t: float) -> float: return t * t
        def _ease_out_quad(t: float) -> float: return 1.0 - (1.0 - t) * (1.0 - t)
        def _ease_in_out_quad(t: float) -> float: return 2.0 * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0

        def _ease_in_cubic(t: float) -> float: return t * t * t
        def _ease_out_cubic(t: float) -> float: return 1.0 - pow(1.0 - t, 3.0)
        def _ease_in_out_cubic(t: float) -> float: return 4.0 * t * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0

        def _ease_in_quart(t: float) -> float: return t * t * t * t
        def _ease_out_quart(t: float) -> float: return 1.0 - pow(1.0 - t, 4.0)
        def _ease_in_out_quart(t: float) -> float: return 8.0 * t * t * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 4.0) / 2.0

        def _ease_in_quint(t: float) -> float: return t * t * t * t * t
        def _ease_out_quint(t: float) -> float: return 1.0 - pow(1.0 - t, 5.0)
        def _ease_in_out_quint(t: float) -> float: return 16.0 * t * t * t * t * t if t < 0.5 else 1.0 - pow(-2.0 * t + 2.0, 5.0) / 2.0

        def _ease_in_expo(t: float) -> float: return 0.0 if t == 0.0 else pow(2.0, 10.0 * t - 10.0)
        def _ease_out_expo(t: float) -> float: return 1.0 if t == 1.0 else 1.0 - pow(2.0, -10.0 * t)
        def _ease_in_out_expo(t: float) -> float:
            if t == 0.0: return 0.0
            if t == 1.0: return 1.0
            return pow(2.0, 20.0 * t - 10.0) / 2.0 if t < 0.5 else (2.0 - pow(2.0, -20.0 * t + 10.0)) / 2.0

        def _ease_in_circ(t: float) -> float: return 1.0 - math.sqrt(max(0.0, 1.0 - pow(t, 2.0)))
        def _ease_out_circ(t: float) -> float: return math.sqrt(max(0.0, 1.0 - pow(t - 1.0, 2.0)))
        def _ease_in_out_circ(t: float) -> float:
            if t < 0.5: return (1.0 - math.sqrt(max(0.0, 1.0 - pow(2.0 * t, 2.0)))) / 2.0
            return (math.sqrt(max(0.0, 1.0 - pow(-2.0 * t + 2.0, 2.0))) + 1.0) / 2.0

        def _ease_in_back(t: float) -> float:
            c1 = 1.70158
            c3 = c1 + 1.0
            return c3 * t * t * t - c1 * t * t

        def _ease_out_back(t: float) -> float:
            c1 = 1.70158
            c3 = c1 + 1.0
            return 1.0 + c3 * pow(t - 1.0, 3.0) + c1 * pow(t - 1.0, 2.0)

        def _ease_in_out_back(t: float) -> float:
            c1 = 1.70158
            c2 = c1 * 1.525
            if t < 0.5:
                return (pow(2.0 * t, 2.0) * ((c2 + 1.0) * 2.0 * t - c2)) / 2.0
            return (pow(2.0 * t - 2.0, 2.0) * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0

        def _ease_in_elastic(t: float) -> float:
            if t <= 0.0: return 0.0
            if t >= 1.0: return 1.0
            c4 = (2.0 * math.pi) / 3.0
            return -pow(2.0, 10.0 * t - 10.0) * math.sin((t * 10.0 - 10.75) * c4)

        def _ease_out_elastic(t: float) -> float:
            if t <= 0.0: return 0.0
            if t >= 1.0: return 1.0
            c4 = (2.0 * math.pi) / 3.0
            return pow(2.0, -10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0

        def _ease_in_out_elastic(t: float) -> float:
            if t <= 0.0: return 0.0
            if t >= 1.0: return 1.0
            c5 = (2.0 * math.pi) / 4.5
            if t < 0.5:
                return -(pow(2.0, 20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0
            return (pow(2.0, -20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0 + 1.0

        def _ease_out_bounce(t: float) -> float:
            n1 = 7.5625
            d1 = 2.75
            if t < 1.0 / d1:
                return n1 * t * t
            elif t < 2.0 / d1:
                t -= 1.5 / d1
                return n1 * t * t + 0.75
            elif t < 2.5 / d1:
                t -= 2.25 / d1
                return n1 * t * t + 0.9375
            else:
                t -= 2.625 / d1
                return n1 * t * t + 0.984375

        def _ease_in_bounce(t: float) -> float:
            return 1.0 - _ease_out_bounce(1.0 - t)

        def _ease_in_out_bounce(t: float) -> float:
            if t < 0.5:
                return (1.0 - _ease_out_bounce(1.0 - 2.0 * t)) / 2.0
            return (1.0 + _ease_out_bounce(2.0 * t - 1.0)) / 2.0

        self._graphs: dict[str, GraphFunc] = {
            "linear": lambda t: t,
            "ease_in": _ease_in_quad,
            "easein": _ease_in_quad,
            "ease_in_quad": _ease_in_quad,
            "ease_out": _ease_out_quad,
            "easeout": _ease_out_quad,
            "ease_out_quad": _ease_out_quad,
            "ease_in_out": self._ease_in_out,
            "easeinout": self._ease_in_out,
            "easeinoutquad": self._ease_in_out,
            "ease_in_out_quad": _ease_in_out_quad,

            "ease_in_cubic": _ease_in_cubic,
            "ease_out_cubic": _ease_out_cubic,
            "ease_in_out_cubic": _ease_in_out_cubic,

            "ease_in_quart": _ease_in_quart,
            "ease_out_quart": _ease_out_quart,
            "ease_in_out_quart": _ease_in_out_quart,

            "ease_in_quint": _ease_in_quint,
            "ease_out_quint": _ease_out_quint,
            "ease_in_out_quint": _ease_in_out_quint,

            "ease_in_expo": _ease_in_expo,
            "ease_out_expo": _ease_out_expo,
            "ease_in_out_expo": _ease_in_out_expo,

            "ease_in_circ": _ease_in_circ,
            "ease_out_circ": _ease_out_circ,
            "ease_in_out_circ": _ease_in_out_circ,

            "back": _ease_out_back,
            "ease_in_back": _ease_in_back,
            "ease_out_back": _ease_out_back,
            "ease_in_out_back": _ease_in_out_back,

            "elastic": _ease_out_elastic,
            "ease_in_elastic": _ease_in_elastic,
            "ease_out_elastic": _ease_out_elastic,
            "ease_in_out_elastic": _ease_in_out_elastic,

            "bounce": _ease_out_bounce,
            "ease_in_bounce": _ease_in_bounce,
            "ease_out_bounce": _ease_out_bounce,
            "ease_in_out_bounce": _ease_in_out_bounce,
        }
        self._custom_records: dict[str, CustomMotionRecord] = {}
        self._custom_lock = threading.Lock()
        self._custom_counter = 0

        self._active_timelines: list[Timeline] = []
        self._connected_motions: dict[str, list[MotionRecord]] = {}

    @staticmethod
    def _ease_in_out(t: float) -> float:
        if t < 0.5:
            return 2.0 * t * t
        return 1.0 - pow(-2.0 * t + 2.0, 2.0) / 2.0

    def register_type(self, name: object, handler: MotionHandler) -> str:
        token = str(name).strip().lower()
        if not token:
            raise ValueError("Draw.motion.register_type: name is required.")
        if not callable(handler):
            raise TypeError("Draw.motion.register_type: handler must be callable.")
        self._type_handlers[token] = handler
        return token

    def register_graph(self, name: object, fn: GraphFunc) -> str:
        token = str(name).strip().lower()
        if not token:
            raise ValueError("Draw.motion.register_graph: name is required.")
        if not callable(fn):
            raise TypeError("Draw.motion.register_graph: fn must be callable.")
        self._graphs[token] = fn
        return token

    def parse_target_ref(self, raw: object, *, default_target: str = "shape") -> TargetRef:
        if raw is None:
            return TargetRef(_normalize_target(default_target), None)
        if isinstance(raw, TargetRef):
            return raw
        if isinstance(raw, dict):
            target = raw.get("target", raw.get("type", default_target))
            ip = raw.get("ip", raw.get("get_ip", raw.get("id")))
            return TargetRef(_normalize_target(target, default=default_target), None if ip is None else str(ip))
        return TargetRef(_normalize_target(default_target), str(raw))

    def parse_motion_list(
        self,
        raw_motion: object,
        time_line: object = None,
    ) -> list[MotionRecord]:
        if raw_motion in (None, ""):
            return []
        if not isinstance(raw_motion, list):
            raise TypeError("Draw.motion: 'motion' must be a list of dicts.")

        timeline = self._parse_time_line(time_line)
        records: list[MotionRecord] = []
        for index, raw_item in enumerate(raw_motion):
            if not isinstance(raw_item, dict):
                raise TypeError("Draw.motion: every motion item must be a dict.")
            item = copy.deepcopy(raw_item)
            if "time" not in item and index < len(timeline):
                item["time"] = timeline[index]
            records.append(self._parse_motion_item(item, index))
        return records

    def _parse_time_line(self, raw: object) -> list[dict[str, float]]:
        if raw in (None, ""):
            return []
        if not isinstance(raw, list):
            raise TypeError("Draw.motion: 'time_line' must be a list.")
        result: list[dict[str, float]] = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise TypeError("Draw.motion: every time_line item must be a dict.")
            if "start" in entry or "end" in entry:
                result.append({"start": float(entry.get("start", 0.0)), "end": float(entry["end"])})
                continue
            if len(entry) != 1:
                raise ValueError("Draw.motion: time_line items must contain one start:end pair.")
            start_raw, end_raw = next(iter(entry.items()))
            result.append({"start": float(start_raw), "end": float(end_raw)})
        return result

    def _parse_motion_item(self, item: dict[str, Any], fallback_index: int) -> MotionRecord:
        attr = item.get("attribute")
        mtype = item.get("type")

        motion_type = attr or mtype or "custom"
        motion_type = str(motion_type).strip().lower()

        if mtype == "morph":
            motion_type = "morph"

        if motion_type == "color" and "color" in item and ("from" not in item or "to" not in item):
            self._expand_color_shorthand(item)

        start = 0.0
        end = float('inf')
        time_spec = item.get("time")
        if time_spec is not None:
            if isinstance(time_spec, dict):
                start = float(time_spec.get("start", 0.0))
                end = float(time_spec.get("end", float('inf')))
            elif isinstance(time_spec, (int, float)):
                end = float(time_spec)
        elif "duration" in item:
            start = float(item.get("delay", 0.0) or 0.0)
            end = start + float(item["duration"])
        elif "keyframes" in item:
            kfs = item["keyframes"]
            if isinstance(kfs, list) and kfs:
                start = float(kfs[0].get("time", 0.0))
                end = float(kfs[-1].get("time", 0.0))

        has_solver = "solver" in item
        has_keyframes = "keyframes" in item
        is_inertia = (mtype == "inertia")
        has_path = (motion_type == "path" and "path" in item)

        if not (has_solver or has_keyframes or is_inertia or has_path or motion_type in _PROCEDURAL_TYPES):
            for required in ("from", "to"):
                if required not in item:
                    raise ValueError(f"Draw.motion: motion '{motion_type}' requires '{required}'.")

        target_raw = item.get("target", item.get("motion_of", "shape"))
        graph = str(item.get("graph", item.get("easing", "linear"))).strip().lower()
        if graph not in self._graphs:
            raise ValueError(f"Draw.motion: unknown graph '{graph}'.")

        compiled_solver = None
        solver_spec = item.get("solver")
        if solver_spec and isinstance(solver_spec, dict):
            expr = solver_spec.get("expression")
            params = solver_spec.get("parameters")
            if expr:
                compiled_solver = DrawExprSolver(expr, params)

        return MotionRecord(
            motion_type=motion_type,
            target=_normalize_target(target_raw),
            from_value=item.get("from"),
            to_value=item.get("to"),
            start=start,
            end=end,
            graph=graph,
            easing=str(item["easing"]).strip().lower() if item.get("easing") is not None else None,
            repeat=_as_bool(item.get("repeat"), default=False),
            reverse=_as_bool(item.get("reverse"), default=False),
            delay=float(item.get("delay", 0.0) or 0.0),
            index=int(item["index"]) if item.get("index") is not None else fallback_index,
            custom_data=item.get("custom_data"),
            raw=dict(item),
            compiled_solver=compiled_solver,
        )

    @staticmethod
    def _expand_color_shorthand(item: dict[str, Any]) -> None:
        color_spec = item.get("color")
        if not isinstance(color_spec, (list, tuple)) or len(color_spec) < 2:
            raise ValueError("Draw.motion: color shorthand must be [from, to, index?, graph?].")
        item["from"] = color_spec[0]
        item["to"] = color_spec[1]
        if len(color_spec) > 2 and "index" not in item:
            item["index"] = color_spec[2]
        if len(color_spec) > 3 and "graph" not in item:
            item["graph"] = color_spec[3]

    def progress(self, record: MotionRecord, now: float, started_at: float) -> Optional[float]:
        elapsed = now - started_at - record.delay
        if elapsed < record.start:
            return None

        duration = record.end - record.start
        if duration <= 0.0:
            raw_t = 1.0
        elif elapsed > record.end and record.repeat:
            cycle_pos = (elapsed - record.start) / duration
            cycle_index = int(cycle_pos)
            raw_t = cycle_pos - cycle_index
            if record.reverse and cycle_index % 2 == 1:
                raw_t = 1.0 - raw_t
        else:
            raw_t = _clamp01((elapsed - record.start) / max(duration, 1e-12))

        graph = self._graphs.get(record.graph, self._graphs["linear"])
        return _clamp01(graph(_clamp01(raw_t)))

    def interpolate(self, start: Any, end: Any, t: float) -> Any:
        start = _resolve_dynamic(start)
        end = _resolve_dynamic(end)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return float(start) + (float(end) - float(start)) * t
        if isinstance(start, dict) and isinstance(end, dict):
            keys = set(start) | set(end)
            return {
                key: self.interpolate(start.get(key, end.get(key)), end.get(key, start.get(key)), t)
                for key in keys
            }
        if isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == len(end):
            return [self.interpolate(a, b, t) for a, b in zip(start, end)]
        return end if t >= 1.0 else start

    def compute_shape_state(self, shape: object, now: float, parse_color: Callable[[Any], Any], x=0.0, y=0.0, w=0.0, h=0.0, canvas=None) -> dict[str, Any]:
        shape_motions = getattr(shape, "motion", None)
        shape_ip = getattr(shape, "ip", None)
        if not shape_motions and (not self._active_timelines or not shape_ip):
            return {}

        started_at = self._ensure_started(shape, now)
        state: dict[str, Any] = {
            "ref_x": float(x),
            "ref_y": float(y),
            "ref_w": float(w),
            "ref_h": float(h),
            "_now_t": max(0.0, now - started_at),
        }
        for record in shape_motions or []:
            if record.target != "shape":
                continue
            self._apply_record(shape, record, state, now, started_at, parse_color, x, y, w, h, canvas)

        if shape_ip:
            for timeline in self._active_timelines:
                if timeline.started_at is None:
                    timeline.started_at = now
                for target_ip, record in timeline.records:
                    if target_ip == shape_ip:
                        self._apply_record(shape, record, state, now, timeline.started_at, parse_color, x, y, w, h, canvas)
        return state

    def compute_hitbox_state(self, shape: object, now: float, parse_color: Callable[[Any], Any], x=0.0, y=0.0, w=0.0, h=0.0, canvas=None) -> dict[str, Any]:
        shape_motions = getattr(shape, "motion", None)
        if not shape_motions:
            return {}

        started_at = self._ensure_started(shape, now)
        state: dict[str, Any] = {
            "ref_x": float(x),
            "ref_y": float(y),
            "ref_w": float(w),
            "ref_h": float(h),
            "_now_t": max(0.0, now - started_at),
        }
        for record in shape_motions:
            if record.target != "hitbox":
                continue
            self._apply_record(shape, record, state, now, started_at, parse_color, x, y, w, h, canvas)
        return state

    @staticmethod
    def _ensure_started(owner: object, now: float) -> float:
        started_at = getattr(owner, "motion_started_at", None)
        if started_at is None:
            setattr(owner, "motion_started_at", now)
            return now
        return float(started_at)

    def _apply_record(
        self,
        shape: object,
        record: MotionRecord,
        state: dict[str, Any],
        now: float,
        started_at: float,
        parse_color: Callable[[Any], Any],
        x: float, y: float, w: float, h: float,
        canvas: Optional[Any],
    ) -> None:
        center_ref = record.raw.get("center") if isinstance(getattr(record, "raw", None), dict) else None
        if center_ref is not None and record.motion_type in ("rotate", "rotation", "transform"):
            from Draw import _bridge
            window_tag = getattr(canvas, "_window_tag", None)
            pivot = _bridge.resolve_point_ref(center_ref, window_tag, self_rect=(float(x), float(y), float(w), float(h)))
            if pivot is not None:
                state["rotation_center"] = pivot

        if record.compiled_solver is not None:
            elapsed_time = now - started_at - record.delay
            last_now = getattr(shape, "_last_now", None)
            dt = now - last_now if last_now is not None else 0.016
            setattr(shape, "_last_now", now)

            locals_dict = {
                "time": elapsed_time,
                "dt": dt,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "mouse_x": getattr(canvas, "_mouse_x", 0.0) if canvas else 0.0,
                "mouse_y": getattr(canvas, "_mouse_y", 0.0) if canvas else 0.0,
                "scroll_x": getattr(canvas, "_scroll_x", 0.0) if canvas else 0.0,
                "scroll_y": getattr(canvas, "_scroll_y", 0.0) if canvas else 0.0,
                "drag_x": getattr(shape, "_drag_x", x) if hasattr(shape, "_drag_x") else x,
                "drag_y": getattr(shape, "_drag_y", y) if hasattr(shape, "_drag_y") else y,
            }

            if record.motion_type == "vertices":
                total = getattr(shape, "vertices", None) or 64
                custom_vertices = []
                for idx in range(total):
                    iter_locals = dict(locals_dict)
                    iter_locals["index"] = float(idx)
                    iter_locals["total"] = float(total)
                    res = record.compiled_solver.evaluate(iter_locals)
                    vx = float(res.get("x", x))
                    vy = float(res.get("y", y))
                    custom_vertices.append((vx, vy))
                state["vertices"] = custom_vertices
            else:
                res = record.compiled_solver.evaluate(locals_dict)
                attr_name = record.motion_type
                if attr_name == "position":
                    if "x" in res:
                        state["x"] = float(res["x"])
                    if "y" in res:
                        state["y"] = float(res["y"])
                elif attr_name in res:
                    state[attr_name] = res[attr_name]
            return

        if record.motion_type == "morph":
            if record.trigger:
                tp = shape._trigger_progresses.get(record.trigger, 0.0) if hasattr(shape, "_trigger_progresses") else 0.0
                t = tp
            else:
                elapsed = now - started_at - record.delay
                duration = record.end - record.start
                t = 1.0 if duration <= 0.0 else max(0.0, min(1.0, elapsed / duration))

            graph = self._graphs.get(record.graph, self._graphs["linear"])
            t_eased = graph(t)

            if not hasattr(record, "resolved_from_vertices") or record.resolved_from_vertices is None:
                rf, rt = resolve_morph_vertices(record.from_value, record.to_value, record.raw.get("resample", "distribute"))
                record.resolved_from_vertices = rf
                record.resolved_to_vertices = rt

            interpolated = []
            for p1, p2 in zip(record.resolved_from_vertices, record.resolved_to_vertices):
                ix = p1[0] + t_eased * (p2[0] - p1[0])
                iy = p1[1] + t_eased * (p2[1] - p1[1])
                interpolated.append((ix, iy))
            state["vertices"] = interpolated
            return

        if record.motion_type == "path" and record.raw.get("path") is not None:
            if record.trigger:
                tp = shape._trigger_progresses.get(record.trigger, 0.0) if hasattr(shape, "_trigger_progresses") else 0.0
                graph = self._graphs.get(record.graph, self._graphs["linear"])
                t_eased = graph(_clamp01(tp))
            else:
                t_eased = self.progress(record, now, started_at)
                if t_eased is None:
                    return

            if record.resolved_path_vertices is None:
                record.resolved_path_vertices = resolve_motion_path_vertices(
                    record.raw["path"],
                    closed=_as_bool(record.raw.get("closed"), default=False),
                )

            px, py, heading = sample_polyline_at(record.resolved_path_vertices, t_eased)
            offset = record.raw.get("offset") or record.raw.get("at")
            if offset:
                px += float(offset[0])
                py += float(offset[1])

            state["x"] = px
            state["y"] = py
            if _as_bool(record.raw.get("orient"), default=False):
                state["rotation"] = heading + float(record.raw.get("orient_offset", 0.0))
            return

        if "keyframes" in record.raw:
            keyframes = record.raw["keyframes"]
            if record.trigger:
                tp = shape._trigger_progresses.get(record.trigger, 0.0) if hasattr(shape, "_trigger_progresses") else 0.0
                t = record.start + tp * (record.end - record.start)
            else:
                elapsed = now - started_at - record.delay
                duration = record.end - record.start
                if duration > 0.0:
                    if elapsed > record.end and record.repeat:
                        cycle_pos = (elapsed - record.start) / duration
                        cycle_index = int(cycle_pos)
                        t_in_cycle = cycle_pos - cycle_index
                        if record.reverse and cycle_index % 2 == 1:
                            t_in_cycle = 1.0 - t_in_cycle
                        t = record.start + t_in_cycle * duration
                    else:
                        t = max(record.start, min(record.end, elapsed))
                else:
                    t = record.start
            value = self.evaluate_keyframes(t, keyframes)
            self._apply_value(record, value, state, parse_color)
            return

        if record.raw.get("type") == "spring" or record.motion_type == "spring":
            stiffness = float(record.raw.get("stiffness", 300.0))
            damping = float(record.raw.get("damping", 25.0))
            mass = float(record.raw.get("mass", 1.0))

            v0 = 0.0
            vel_spec = record.raw.get("velocity", 0.0)
            if vel_spec == "inherit":
                v0_dict = getattr(shape, "_release_velocities", {})
                v0 = v0_dict.get(record.motion_type, 0.0)
            elif isinstance(vel_spec, (int, float)):
                v0 = float(vel_spec)

            elapsed = now - started_at - record.delay
            v_from = record.from_value if record.from_value is not None else 0.0
            v_to = record.to_value if record.to_value is not None else 1.0

            if isinstance(v_from, (list, tuple)) and isinstance(v_to, (list, tuple)) and len(v_from) == len(v_to):
                v0_list = v0 if isinstance(v0, (list, tuple)) and len(v0) == len(v_from) else [v0] * len(v_from)
                res_val = [
                    solve_spring(elapsed, vf, vt, float(v0_list[idx]), stiffness, damping, mass)
                    for idx, (vf, vt) in enumerate(zip(v_from, v_to))
                ]
            else:
                res_val = solve_spring(elapsed, float(v_from), float(v_to), float(v0), stiffness, damping, mass)
            self._apply_value(record, res_val, state, parse_color)
            return

        if record.raw.get("type") == "inertia" or record.motion_type == "inertia":
            friction = float(record.raw.get("friction", 0.92))
            bounds = record.raw.get("bounds")

            v0 = 0.0
            vel_spec = record.raw.get("velocity", 0.0)
            if vel_spec == "inherit":
                v0_dict = getattr(shape, "_release_velocities", {})
                v0 = v0_dict.get(record.motion_type, 0.0)
            elif isinstance(vel_spec, (int, float)):
                v0 = float(vel_spec)

            elapsed = now - started_at - record.delay
            v_from = record.from_value
            if v_from is None:
                if record.motion_type == "scroll_y" and canvas:
                    v_from = canvas._scroll_y
                elif record.motion_type == "x" and hasattr(shape, "last_position") and shape.last_position:
                    v_from = shape.last_position[0]
                elif record.motion_type == "y" and hasattr(shape, "last_position") and shape.last_position:
                    v_from = shape.last_position[1]
                else:
                    v_from = 0.0
                record.from_value = v_from

            res_val = solve_inertia(elapsed, float(v_from), float(v0), friction, bounds)
            if record.motion_type == "scroll_y" and canvas:
                canvas._scroll_y = res_val
            self._apply_value(record, res_val, state, parse_color)
            return

        t = self.progress(record, now, started_at)
        now_t = state.get("_now_t", 0.0)

        # ── Procedural & Parametric 2D Curve Motions ────────────────────────
        if record.motion_type in {"orbit", "polygon_orbit", "benzene"}:
            sides = int(record.raw.get("sides", 6))
            radius_val = float(record.raw.get("radius", 100.0))
            freq = float(record.raw.get("frequency", 0.5))
            dx, dy = solve_polygon_orbit(now_t, radius_val, sides, freq)
            state["x"] = state.get("x", state.get("ref_x", x)) + dx
            state["y"] = state.get("y", state.get("ref_y", y)) + dy
            return

        if record.motion_type == "lissajous":
            ax = float(record.raw.get("amplitude_x", 100.0))
            ay = float(record.raw.get("amplitude_y", 80.0))
            fx = float(record.raw.get("freq_x", 3.0))
            fy = float(record.raw.get("freq_y", 2.0))
            px = float(record.raw.get("phase_x", 0.0))
            py = float(record.raw.get("phase_y", math.pi / 2.0))
            dx, dy = solve_lissajous(now_t, ax, ay, fx, fy, px, py)
            state["x"] = state.get("x", state.get("ref_x", x)) + dx
            state["y"] = state.get("y", state.get("ref_y", y)) + dy
            return

        if record.motion_type == "spiral":
            a_val = float(record.raw.get("a", 10.0))
            b_val = float(record.raw.get("b", 15.0))
            freq = float(record.raw.get("frequency", 1.0))
            is_log = _as_bool(record.raw.get("logarithmic"), default=False)
            dx, dy = solve_spiral(now_t, a_val, b_val, freq, is_log)
            state["x"] = state.get("x", state.get("ref_x", x)) + dx
            state["y"] = state.get("y", state.get("ref_y", y)) + dy
            return

        if record.motion_type == "pendulum":
            amp = float(record.raw.get("amplitude", 45.0))
            freq = float(record.raw.get("frequency", 1.5))
            damp = float(record.raw.get("damping", 0.25))
            rot_deg = solve_pendulum(now_t, amp, freq, damp)
            state["rotation"] = state.get("rotation", 0.0) + rot_deg
            return

        if record.motion_type == "stretch_squash":
            amp = float(record.raw.get("amplitude", 0.3))
            freq = float(record.raw.get("frequency", 2.0))
            damp = float(record.raw.get("damping", 1.0))
            sx, sy = solve_stretch_squash(now_t, amp, freq, damp)
            state["scale_x"] = state.get("scale_x", 1.0) * sx
            state["scale_y"] = state.get("scale_y", 1.0) * sy
            return

        if record.motion_type in {"projectile", "gravity_2d"}:
            vx0 = float(record.raw.get("vx0", record.raw.get("velocity_x", 200.0)))
            vy0 = float(record.raw.get("vy0", record.raw.get("velocity_y", -300.0)))
            g = float(record.raw.get("g", 980.0))
            drag = float(record.raw.get("drag", 0.05))
            restitution = float(record.raw.get("restitution", 0.75))
            floor_y = float(record.raw.get("floor_y", state.get("ref_y", y) + 300.0))
            wall_x = float(record.raw["wall_x"]) if "wall_x" in record.raw else None
            start_x_val = float(record.from_value[0] if isinstance(record.from_value, (list, tuple)) else state.get("ref_x", x))
            start_y_val = float(record.from_value[1] if isinstance(record.from_value, (list, tuple)) else state.get("ref_y", y))
            px, py = solve_projectile_2d(now_t, vx0, vy0, g, drag, restitution, floor_y, wall_x, start_x_val, start_y_val)
            state["x"] = px
            state["y"] = py
            return

        if record.motion_type == "attractor":
            tx = float(record.raw.get("target_x", canvas._mouse_x if canvas and hasattr(canvas, "_mouse_x") else x))
            ty = float(record.raw.get("target_y", canvas._mouse_y if canvas and hasattr(canvas, "_mouse_y") else y))
            str_val = float(record.raw.get("strength", 5.0))
            cur_x_val = state.get("x", state.get("ref_x", x))
            cur_y_val = state.get("y", state.get("ref_y", y))
            nx, ny = solve_attractor_2d(now_t, cur_x_val, cur_y_val, tx, ty, str_val)
            state["x"] = nx
            state["y"] = ny
            return

        if t is None:
            return

        if record.motion_type == "color":
            value = self.interpolate_color(record.from_value, record.to_value, t, parse_color)
        else:
            value = self.interpolate(record.from_value, record.to_value, t)
        self._apply_value(record, value, state, parse_color)

    def evaluate_keyframes(self, t: float, keyframes: list[dict[str, Any]]) -> Any:
        if not keyframes:
            return 0.0
        kfs = sorted(keyframes, key=lambda k: float(k.get("time", 0.0)))
        t_first = float(kfs[0].get("time", 0.0))
        t_last = float(kfs[-1].get("time", 0.0))
        if t <= t_first:
            return kfs[0].get("value")
        if t >= t_last:
            return kfs[-1].get("value")

        for i in range(len(kfs) - 1):
            kf0 = kfs[i]
            kf1 = kfs[i+1]
            t0 = float(kf0.get("time", 0.0))
            t1 = float(kf1.get("time", 0.0))
            if t0 <= t < t1:
                outgoing = kf0.get("outgoing", {})
                incoming = kf1.get("incoming", {})
                return evaluate_keyframe_segment(t, t0, kf0.get("value"), outgoing, t1, kf1.get("value"), incoming)
        return kfs[-1].get("value")

    def _apply_value(self, record: MotionRecord, value: Any, state: dict[str, Any], parse_color: Callable[[Any], Any]) -> None:
        handler = self._type_handlers.get(record.motion_type)
        if handler is not None:
            handler(record, value, state)
            return
        self._apply_builtin(record, value, state, parse_color)

    def tick_shape_triggers(self, shape: object, dt: float) -> None:
        if not hasattr(shape, "motion") or not shape.motion:
            return
        if not hasattr(shape, "_trigger_progresses"):
            shape._trigger_progresses = {}

        for record in shape.motion:
            if record.trigger:
                trigger_name = record.trigger
                is_active = False
                if trigger_name == "hover":
                    is_active = getattr(shape, "_is_hovered", False)
                elif trigger_name == "press":
                    is_active = getattr(shape, "_is_pressed", False)
                elif trigger_name == "visible":
                    is_active = getattr(shape, "_is_visible", True)
                elif trigger_name == "drag":
                    is_active = getattr(shape, "_is_dragged", False)
                elif trigger_name == "scroll":
                    is_active = True

                curr_p = shape._trigger_progresses.get(trigger_name, 0.0)
                if math.isinf(record.end):
                    duration = 0.3
                else:
                    duration = max(1e-5, record.end - record.start)
                if is_active:
                    new_p = min(1.0, curr_p + dt / duration)
                else:
                    new_p = max(0.0, curr_p - dt / duration)
                shape._trigger_progresses[trigger_name] = new_p

    def tick_timelines(self, now: float) -> bool:
        to_remove = []
        changed = False
        for timeline in self._active_timelines:
            if timeline.started_at is None:
                timeline.started_at = now
                changed = True
            elapsed = now - timeline.started_at
            max_end = max((rec.end for target, rec in timeline.records), default=0.0)
            if elapsed > max_end:
                if timeline.repeat:
                    timeline.started_at = now
                    changed = True
                else:
                    to_remove.append(timeline)
                    changed = True
        for tl in to_remove:
            self._active_timelines.remove(tl)
        return changed

    def create_timeline(self, *args, **kwargs) -> Timeline:
        tl = Timeline(*args, **kwargs)
        self._active_timelines.append(tl)
        return tl

    def __call__(
        self,
        connection: dict[str, Any] = None,
        motion: list[dict[str, Any]] = None,
        trigger: Optional[str] = None,
        *,
        ip: object = None,
        motion_ip: object = None,
        motion_get_ip: object = None,
    ) -> None:
        motion_ip = motion_ip if motion_ip is not None else ip
        if motion_ip is not None:
            connection = {"motion_ip": str(motion_ip)}
            if motion_get_ip is not None:
                connection["motion_get_ip"] = str(motion_get_ip)
        elif connection is None:
            raise ValueError("Draw.motion: provide 'ip' (or 'motion_ip') or a 'connection' dict.")

        parsed_records = self.parse_motion_list(motion or [])
        for rec in parsed_records:
            rec.trigger = trigger

        if isinstance(connection, str):
            resolved_ip = connection
        else:
            resolved_ip = connection.get("motion_ip", connection.get("motion_get_ip", connection.get("ip", connection.get("get_ip", connection.get("id")))))
        if resolved_ip is None:
            raise ValueError("Draw.motion: could not resolve target IP from connection dict.")
        ip_str = str(resolved_ip)
        self._connected_motions.setdefault(ip_str, []).extend(parsed_records)

        elements = find_elements_by_ip(ip_str)
        for el in elements:
            if not hasattr(el, "_trigger_progresses"):
                el._trigger_progresses = {}
            if not hasattr(el, "motion") or el.motion is None:
                el.motion = []
            el.motion.extend(parsed_records)

    def interpolate_color(
        self,
        start: Any,
        end: Any,
        t: float,
        parse_color: Callable[[Any], Any],
    ) -> Any:
        c1 = parse_color(_resolve_dynamic(start))
        c2 = parse_color(_resolve_dynamic(end))
        color_type = c1.__class__
        return color_type(
            int(round(float(c1.red()) + (float(c2.red()) - float(c1.red())) * t)),
            int(round(float(c1.green()) + (float(c2.green()) - float(c1.green())) * t)),
            int(round(float(c1.blue()) + (float(c2.blue()) - float(c1.blue())) * t)),
            int(round(float(c1.alpha()) + (float(c2.alpha()) - float(c1.alpha())) * t)),
        )

    def _apply_builtin(
        self,
        record: MotionRecord,
        value: Any,
        state: dict[str, Any],
        parse_color: Callable[[Any], Any],
    ) -> None:
        motion_type = record.motion_type
        if motion_type in {"move", "path", "position", "pos"}:
            self._merge_xy(value, state)
        elif motion_type == "x":
            state["x"] = float(value)
        elif motion_type == "y":
            state["y"] = float(value)
        elif motion_type == "vertices":
            if isinstance(value, (int, float)):
                state["vertices_count"] = max(3, int(round(float(value))))
            else:
                state["vertices"] = value
        elif motion_type in {"expand", "size"}:
            self._merge_size(value, state)
        elif motion_type == "scale":
            self._merge_scale(value, state)
        elif motion_type in {"skew", "shear"}:
            self._merge_skew(value, state)
        elif motion_type in {"rotate_x", "rotate_y", "rotate_3d", "perspective"}:
            self._merge_3d_transform(motion_type, value, state)
        elif motion_type == "trim_path":
            self._merge_trim_path(value, state)
        elif motion_type == "stroke_dash":
            self._merge_stroke_dash(value, state)
        elif motion_type in {"shake", "wiggle"}:
            freq = float(record.raw.get("frequency", 5.0))
            amp = float(record.raw.get("amplitude", float(value) if isinstance(value, (int, float)) else 10.0))
            octs = int(record.raw.get("octaves", 2))
            now_t = state.get("_now_t", 0.0)
            dx, dy = solve_wiggle(now_t, freq, amp, octs)
            state["x"] = state.get("x", state.get("ref_x", 0.0)) + dx
            state["y"] = state.get("y", state.get("ref_y", 0.0)) + dy
        elif motion_type in {"wave", "pulse"}:
            freq = float(record.raw.get("frequency", 2.0))
            amp = float(record.raw.get("amplitude", float(value) if isinstance(value, (int, float)) else 20.0))
            phase = float(record.raw.get("phase", 0.0))
            now_t = state.get("_now_t", 0.0)
            wv = solve_wave(now_t, freq, amp, phase)
            if motion_type == "wave":
                state["y"] = state.get("y", state.get("ref_y", 0.0)) + wv
            else:
                scale_base = state.get("scale_x", 1.0)
                state["scale_x"] = scale_base + (wv / 100.0)
                state["scale_y"] = scale_base + (wv / 100.0)
        elif motion_type in {"gravity", "bounce_physics"}:
            v0 = float(record.raw.get("velocity", 0.0))
            g = float(record.raw.get("g", 980.0))
            restitution = float(record.raw.get("restitution", 0.7))
            floor_y = float(record.raw.get("floor_y", state.get("ref_y", 0.0) + 300.0))
            start_y = float(record.from_value if record.from_value is not None else state.get("ref_y", 0.0))
            now_t = state.get("_now_t", 0.0)
            state["y"] = solve_gravity_bounce(now_t, v0, g, restitution, floor_y, start_y)
        elif motion_type in {"rotate", "rotation"}:
            state["rotation"] = float(value.get("rotation", 0.0) if isinstance(value, dict) else value)
        elif motion_type in {"opacity", "alpha"}:
            state["opacity"] = max(0, min(100, int(round(float(value)))))
        elif motion_type == "blur":
            state["blur"] = max(0, int(round(float(value))))
        elif motion_type in {"color", "colour"}:
            state["color"] = parse_color(value)
        elif motion_type == "glow":
            state["glow"] = True
            if isinstance(value, dict):
                if "radius" in value:
                    state["glow_radius"] = max(0, int(round(float(value["radius"]))))
                if "glow_radius" in value:
                    state["glow_radius"] = max(0, int(round(float(value["glow_radius"]))))
                if "color" in value:
                    state["glow_color"] = parse_color(value["color"])
                if "glow_color" in value:
                    state["glow_color"] = parse_color(value["glow_color"])
            else:
                state["glow_radius"] = max(0, int(round(float(value))))
        elif motion_type == "transform" and isinstance(value, dict):
            self._merge_xy(value, state)
            self._merge_size(value, state)
            self._merge_scale(value, state)
            self._merge_skew(value, state)
            if "rotation" in value:
                state["rotation"] = float(value["rotation"])
            if "opacity" in value:
                state["opacity"] = max(0, min(100, int(round(float(value["opacity"])))))
        elif motion_type == "custom":
            state["custom"] = value

    @staticmethod
    def _merge_xy(value: Any, state: dict[str, Any]) -> None:
        if isinstance(value, dict):
            if "x" in value:
                state["x"] = float(value["x"])
            if "y" in value:
                state["y"] = float(value["y"])
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            state["x"] = float(value[0])
            state["y"] = float(value[1])

    @staticmethod
    def _merge_size(value: Any, state: dict[str, Any]) -> None:
        if isinstance(value, dict):
            if "width" in value:
                state["width"] = max(1, int(round(float(value["width"]))))
            if "height" in value:
                state["height"] = max(1, int(round(float(value["height"]))))
            if "w" in value:
                state["width"] = max(1, int(round(float(value["w"]))))
            if "h" in value:
                state["height"] = max(1, int(round(float(value["h"]))))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            state["width"] = max(1, int(round(float(value[0]))))
            state["height"] = max(1, int(round(float(value[1]))))
        elif isinstance(value, (int, float)):
            size = max(1, int(round(float(value))))
            state["width"] = size
            state["height"] = size

    @staticmethod
    def _merge_scale(value: Any, state: dict[str, Any]) -> None:
        if isinstance(value, dict):
            sx = value.get("x", value.get("scale_x", value.get("width", value.get("scale", 1.0))))
            sy = value.get("y", value.get("scale_y", value.get("height", value.get("scale", sx))))
            state["scale_x"] = float(sx)
            state["scale_y"] = float(sy)
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            state["scale_x"] = float(value[0])
            state["scale_y"] = float(value[1])
        else:
            state["scale_x"] = float(value)
            state["scale_y"] = float(value)

    @staticmethod
    def _merge_skew(value: Any, state: dict[str, Any]) -> None:
        if isinstance(value, dict):
            state["skew_x"] = float(value.get("x", value.get("skew_x", 0.0)))
            state["skew_y"] = float(value.get("y", value.get("skew_y", 0.0)))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            state["skew_x"] = float(value[0])
            state["skew_y"] = float(value[1])
        else:
            state["skew_x"] = float(value)
            state["skew_y"] = 0.0

    @staticmethod
    def _merge_3d_transform(mtype: str, value: Any, state: dict[str, Any]) -> None:
        if mtype == "rotate_x":
            state["rotate_x"] = float(value.get("rotate_x", value) if isinstance(value, dict) else value)
        elif mtype == "rotate_y":
            state["rotate_y"] = float(value.get("rotate_y", value) if isinstance(value, dict) else value)
        elif mtype == "perspective":
            state["perspective"] = float(value.get("perspective", value) if isinstance(value, dict) else value)
        elif mtype == "rotate_3d" and isinstance(value, dict):
            state["rotate_x"] = float(value.get("rotate_x", value.get("x", 0.0)))
            state["rotate_y"] = float(value.get("rotate_y", value.get("y", 0.0)))
            state["rotation"] = float(value.get("rotate_z", value.get("z", value.get("rotation", 0.0))))
            if "perspective" in value:
                state["perspective"] = float(value["perspective"])

    @staticmethod
    def _merge_trim_path(value: Any, state: dict[str, Any]) -> None:
        if isinstance(value, dict):
            state["trim_start"] = _clamp01(float(value.get("start", 0.0)))
            state["trim_end"] = _clamp01(float(value.get("end", 1.0)))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            state["trim_start"] = _clamp01(float(value[0]))
            state["trim_end"] = _clamp01(float(value[1]))
        else:
            state["trim_start"] = 0.0
            state["trim_end"] = _clamp01(float(value))

    @staticmethod
    def _merge_stroke_dash(value: Any, state: dict[str, Any]) -> None:
        if isinstance(value, dict):
            state["dash_offset"] = float(value.get("offset", 0.0))
            if "array" in value:
                state["dash_array"] = [float(v) for v in value["array"]]
        elif isinstance(value, (list, tuple)):
            state["dash_array"] = [float(v) for v in value]
        else:
            state["dash_offset"] = float(value)

    def register_custom(
        self,
        *,
        ip: object = None,
        get_ip: object = None,
        return_value: Any = None,
        tools: Optional[list] = None,
        get_custom: Any = None,
        motion: object = None,
    ) -> CustomMotionRecord:
        records = self.parse_motion_list(motion or [])
        custom_ip = str(ip) if ip is not None else f"custom.motion.{self._custom_counter}"
        self._custom_counter += 1
        record = CustomMotionRecord(
            ip=custom_ip,
            get_ip=str(get_ip) if get_ip is not None else None,
            return_value=return_value,
            tools=list(tools or []),
            get_custom=get_custom,
            motion=records,
        )
        with self._custom_lock:
            self._custom_records[custom_ip] = record
        return record

    def get_custom(self, ip: object) -> Optional[CustomMotionRecord]:
        return self._custom_records.get(str(ip))

    def list_custom(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._custom_records.values()]

    def clear_custom(self, ip: object = None) -> None:
        with self._custom_lock:
            if ip is None:
                self._custom_records.clear()
                self._custom_counter = 0
            else:
                self._custom_records.pop(str(ip), None)

    def tick_custom(self, now: float) -> bool:
        changed = False
        with self._custom_lock:
            records_snapshot = list(self._custom_records.values())
            for record in records_snapshot:
                if record.started_at is None:
                    record.started_at = now
            started_ats = {id(r): r.started_at for r in records_snapshot}

        for record in records_snapshot:
            started_at = started_ats[id(record)]
            new_value = record.current_value
            for motion_record in record.motion:
                t = self.progress(motion_record, now, started_at)
                if t is None:
                    continue
                if record.get_custom is not None and callable(record.get_custom):
                    new_value = record.get_custom(record, t)
                elif record.return_value is not None:
                    new_value = self._resolve_return_value(record.return_value)
                else:
                    new_value = self.interpolate(motion_record.from_value, motion_record.to_value, t)
            try:
                is_changed = bool(new_value != record.current_value)
            except Exception:
                is_changed = True
            if is_changed:
                with self._custom_lock:
                    record.current_value = new_value
                changed = True
        return changed

    @staticmethod
    def _resolve_return_value(value: Any) -> Any:
        value = _resolve_dynamic(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return _resolve_dynamic(value[-1])
        return value

    def studio(self, **kwargs) -> Any:
        """Launch the comprehensive interactive Motion Studio test window with the yellow ball."""
        from Draw.examples.test_motion_window import MotionStudioWindow
        from Draw._app import get_app
        _ = get_app()
        win = MotionStudioWindow(**kwargs)
        win.show()
        return win


class _CustomNamespace:
    """Namespace exposed as Draw.custom."""

    def __init__(self, motion_registry: MotionRegistry) -> None:
        self._motion = motion_registry

    def motion(
        self,
        *,
        ip: object = None,
        get_ip: object = None,
        return_value: Any = None,
        tools: Optional[list] = None,
        get_custom: Any = None,
        motion: object = None,
    ) -> CustomMotionRecord:
        return self._motion.register_custom(
            ip=ip,
            get_ip=get_ip,
            return_value=return_value,
            tools=tools,
            get_custom=get_custom,
            motion=motion,
        )


motion = MotionRegistry()
custom = _CustomNamespace(motion)
def timeline(*args, **kwargs) -> Timeline:
    return motion.create_timeline(*args, **kwargs)


class VelocityTracker:
    """Tracks position samples over time to compute smooth release velocities."""
    def __init__(self, max_samples: int = 10, max_age: float = 0.25) -> None:
        self.samples: List[tuple[float, float]] = []
        self.max_samples = max_samples
        self.max_age = max_age

    def add_sample(self, t: float, val: float) -> None:
        self.samples.append((float(t), float(val)))
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)

    def get_velocity(self, now: Optional[float] = None) -> float:
        if len(self.samples) < 2:
            return 0.0
        import time as _time
        now_t = now if now is not None else _time.perf_counter()
        valid = [(t, v) for t, v in self.samples if (now_t - t) <= self.max_age]
        if len(valid) < 2:
            return 0.0
        dt = valid[-1][0] - valid[0][0]
        if dt <= 1e-6:
            return 0.0
        return (valid[-1][1] - valid[0][1]) / dt

