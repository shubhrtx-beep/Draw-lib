"""
Draw._calculator
================
Safe, sandboxed mathematical expression evaluator and trimmed string calculator.
Evaluates user expressions and dynamic formulas (e.g. Draw.color(), Draw.motion())
via Python AST without unsafe eval() or exec().

Execution Pipeline:
    Input Expression
          ↓
    1. Normalize string & validate length (<= MAX_EXPR_LENGTH)
          ↓
    2. ast.parse(mode="eval") (with informative SyntaxError column formatting)
          ↓
    3. Validate AST structure (whitelisted AST node types, node count <= MAX_AST_NODES)
          ↓
    4. Safe evaluation (recursion depth <= MAX_RECURSION_DEPTH, power/exponent limits, overflow guards)
          ↓
    5. Formatting & precision trimming (for Draw.calculator) or direct float (for eval_expression)
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, Optional, Set, Tuple


# ── Complexity and Resource Limits ───────────────────────────────────────────

MAX_EXPR_LENGTH: int = 1024
MAX_AST_NODES: int = 250
MAX_RECURSION_DEPTH: int = 40
MAX_EXPONENT_MAGNITUDE: float = 1000.0
MAX_BASE_FOR_LARGE_EXP: float = 1e6


# ── Allowed AST Operators & Nodes ────────────────────────────────────────────

_ALLOWED_BINARY_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)

_ALLOWED_NODE_TYPES: Tuple[type, ...] = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.UnaryOp,
    ast.BinOp,
    ast.Call,
    ast.Load,
    ast.UAdd,
    ast.USub,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)


# ── Helper Math Functions ────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b by t (clamped to 0.0 - 1.0)."""
    return a + (b - a) * max(0.0, min(1.0, t))


def _step(edge: float, x: float) -> float:
    """Returns 0.0 if x < edge, else 1.0."""
    return 0.0 if x < edge else 1.0


# ── Safe Built-in Functions & Constants Tables ───────────────────────────────

_SAFE_FUNCTIONS: Dict[str, Any] = {
    "sin":      math.sin,
    "cos":      math.cos,
    "tan":      math.tan,
    "asin":     math.asin,
    "acos":     math.acos,
    "atan":     math.atan,
    "sinh":     math.sinh,
    "cosh":     math.cosh,
    "tanh":     math.tanh,
    "abs":      abs,
    "min":      min,
    "max":      max,
    "sqrt":     math.sqrt,
    "log":      math.log10,
    "ln":       math.log,
    "log2":     math.log2,
    "floor":    math.floor,
    "ceil":     math.ceil,
    "round":    round,
    "degrees":  math.degrees,
    "radians":  math.radians,
    "lerp":     _lerp,
    "step":     _step,
}

_SAFE_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": getattr(math, "tau", 2.0 * math.pi),
    "inf": math.inf,
    "infinity": math.inf,
}

# Automatically derived function and variable name set for is_expression() heuristic
_EXPR_FUNCS: Set[str] = set(_SAFE_FUNCTIONS.keys()) | set(_SAFE_CONSTANTS.keys()) | {
    "time", "mouse_x", "mouse_y", "x", "y", "t", "w", "h", "cw", "ch"
}


# ── String Formatting Helpers ────────────────────────────────────────────────

def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _trim_number(value: Any, precision: int) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if value.is_integer():
            return str(int(value))
        text = format(value, f".{max(0, int(precision))}f")
        text = text.rstrip("0").rstrip(".")
        return text or "0"
    return str(value).strip()


def _format_syntax_error(exc: SyntaxError, expr: str) -> str:
    """Build a human-readable syntax error message from a SyntaxError."""
    msg = str(exc.msg) if exc.msg else "syntax error"
    col = exc.offset
    if col is not None:
        pointer = " " * (col - 1) + "^" if col > 0 else ""
        return (
            f"Draw.calculator: syntax error at column {col} in '{expr}'\n"
            f"  {expr}\n"
            f"  {pointer}\n"
            f"  ({msg})"
        )
    return f"Draw.calculator: syntax error in '{expr}' ({msg})"


# ── AST Complexity Validation ────────────────────────────────────────────────

def _validate_ast_complexity(tree: ast.AST, max_nodes: int = MAX_AST_NODES) -> None:
    """
    Validate that the AST does not exceed node limits and contains only whitelisted node types.
    """
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > max_nodes:
            raise ValueError(
                f"Draw.calculator: expression exceeds maximum complexity "
                f"({node_count} nodes > {max_nodes} limit)."
            )
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            node_type = type(node).__name__
            raise ValueError(
                f"Draw.calculator: disallowed expression element '{node_type}'. "
                f"Only arithmetic operators (+, -, *, /, //, %, **), numeric constants, "
                f"whitelisted constants, and safe math functions are permitted."
            )


# ── Safe Exponentiation Math ─────────────────────────────────────────────────

def _safe_pow(left: float, right: float) -> float:
    """
    Safely evaluate left ** right with strict exponent caps and overflow protection.
    Prevents CPU starvation attacks from extreme exponentiation (e.g. 999999 ** 999999).
    """
    if abs(right) > MAX_EXPONENT_MAGNITUDE:
        raise ValueError(
            f"Draw.calculator: exponent magnitude too large ({right} > {MAX_EXPONENT_MAGNITUDE} limit)."
        )
    if abs(left) > MAX_BASE_FOR_LARGE_EXP and right > 100:
        raise ValueError(
            "Draw.calculator: calculation overflow (exponentiation base and power too large)."
        )
    try:
        res = left ** right
        if isinstance(res, complex):
            raise ValueError(
                f"Draw.calculator: negative base ({left}) with fractional exponent ({right}) produced complex result."
            )
        if math.isinf(res) and not math.isinf(left):
            raise OverflowError("Result is infinite")
        return float(res)
    except OverflowError as exc:
        raise ValueError(
            f"Draw.calculator: calculation overflow ({left} ** {right} too large)."
        ) from exc


# ── Safe AST Recursive Evaluator ─────────────────────────────────────────────

def _eval_expr(
    node: ast.AST,
    variables: Optional[Dict[str, float]] = None,
    depth: int = 0,
    max_depth: int = MAX_RECURSION_DEPTH,
) -> float:
    """Safely evaluate an AST node with recursion depth and arithmetic safeguards."""
    if depth > max_depth:
        raise ValueError(
            f"Draw.calculator: expression exceeds maximum recursion depth ({depth} > {max_depth} limit)."
        )

    if isinstance(node, ast.Expression):
        return _eval_expr(node.body, variables, depth + 1, max_depth)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(
            f"Draw.calculator: only numeric constants are allowed, "
            f"got {type(node.value).__name__} ({node.value!r})."
        )

    # Variable or constant lookup (e.g. time, x, y, pi, e, tau, inf)
    if isinstance(node, ast.Name):
        name = node.id
        if variables and name in variables:
            return float(variables[name])
        lower_name = name.lower()
        if lower_name in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[lower_name]
        raise ValueError(
            f"Draw.calculator: unknown variable '{name}'. "
            f"Available variables must be passed via the 'variables' argument."
        )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
        operand = _eval_expr(node.operand, variables, depth + 1, max_depth)
        if isinstance(node.op, ast.UAdd):
            return +operand
        return -operand

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINARY_OPS):
        left = _eval_expr(node.left, variables, depth + 1, max_depth)
        right = _eval_expr(node.right, variables, depth + 1, max_depth)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError(
                        f"Draw.calculator: division by zero ('{left} / {right}')."
                    )
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise ZeroDivisionError(
                        f"Draw.calculator: floor division by zero ('{left} // {right}')."
                    )
                return left // right
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise ZeroDivisionError(
                        f"Draw.calculator: modulo by zero ('{left} % {right}')."
                    )
                return left % right
            if isinstance(node.op, ast.Pow):
                return _safe_pow(left, right)
        except OverflowError as exc:
            raise ValueError(f"Draw.calculator: calculation overflow: {exc}") from exc

    # Function calls: sin(x), cos(x), tan(x), abs(x), min(a,b), max(a,b), lerp(a,b,t), step(e,x)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Draw.calculator: only direct function calls are allowed "
                "(e.g. sin(x), not obj.method(x))."
            )
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            known = ", ".join(sorted(_SAFE_FUNCTIONS.keys()))
            raise ValueError(
                f"Draw.calculator: unknown function '{func_name}'. "
                f"Available functions: {known}."
            )
        args = [_eval_expr(arg, variables, depth + 1, max_depth) for arg in node.args]
        try:
            res = float(_SAFE_FUNCTIONS[func_name](*args))
            if math.isnan(res):
                raise ValueError(f"Draw.calculator: function '{func_name}' returned NaN.")
            return res
        except TypeError as exc:
            raise ValueError(
                f"Draw.calculator: wrong arguments for '{func_name}': {exc}"
            ) from exc
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Draw.calculator: math error in '{func_name}': {exc}"
            ) from exc

    node_type = type(node).__name__
    raise ValueError(
        f"Draw.calculator: unsupported expression element '{node_type}'."
    )


# ── Expression Cache ─────────────────────────────────────────────────────────

_AST_CACHE_MAX = 256
_ast_cache: dict[str, ast.Expression] = {}  # expr_str -> validated AST tree
_ast_cache_order: list[str] = []  # LRU tracking

def clear_expression_cache() -> None:
    """Clear the parsed AST cache."""
    _ast_cache.clear()
    _ast_cache_order.clear()


# ── Public API 1: eval_expression ────────────────────────────────────────────

def eval_expression(
    expr: str,
    variables: Optional[Dict[str, float]] = None,
    max_length: int = MAX_EXPR_LENGTH,
    max_nodes: int = MAX_AST_NODES,
) -> float:
    """
    Evaluate a math expression string with optional variables and strict complexity limits.
    This is the primary entry point for the Draw.color() and Draw.motion() expression systems.
    Uses a sandboxed AST parser with resource limits — no eval() or exec().

    Pipeline:
      1. Normalize & check length (<= max_length)
      2. Parse AST via ast.parse(mode="eval")
      3. Validate AST complexity (<= max_nodes, node whitelist)
      4. Recursively evaluate with depth, exponent, and overflow caps

    Parameters
    ----------
    expr       : Math expression string, e.g. "sin(time * 3) * 50 + 50"
    variables  : Dict of variable names to float values, e.g. {"time": 1.5, "x": 100}
    max_length : Maximum allowed string length (default: 1024)
    max_nodes  : Maximum allowed AST node count (default: 250)

    Returns
    -------
    float result of the expression.

    Raises
    ------
    ValueError with a descriptive message on any syntax, validation, or arithmetic error.
    """
    expr_str = expr.strip() if isinstance(expr, str) else str(expr or "").strip()
    if expr_str == "":
        return 0.0
    if len(expr_str) > max_length:
        raise ValueError(
            f"Draw.calculator: expression exceeds maximum length "
            f"({len(expr_str)} > {max_length} limit)."
        )
    if expr_str in _ast_cache:
        tree = _ast_cache[expr_str]
        _ast_cache_order.remove(expr_str)
        _ast_cache_order.append(expr_str)
    else:
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as exc:
            raise ValueError(_format_syntax_error(exc, expr_str)) from exc

        _validate_ast_complexity(tree, max_nodes=max_nodes)
        _ast_cache[expr_str] = tree
        _ast_cache_order.append(expr_str)
        if len(_ast_cache_order) > _AST_CACHE_MAX:
            oldest = _ast_cache_order.pop(0)
            del _ast_cache[oldest]

    return _eval_expr(tree, variables)


# ── Public API 2: is_expression ──────────────────────────────────────────────

def is_expression(value: Any) -> bool:
    """
    Fast heuristic check to determine whether a string is likely a dynamic math expression.

    Note:
    - This is a fast classification heuristic, not a full syntax validator.
    - Returns False for pure numeric literals (e.g. 10, "42", "-3.14"), percentages ("50%"),
      and hex colors ("#ff0000").
    - Returns True for strings containing mathematical operators (+, -, *, /, %, **) or function
      names (sin, cos, lerp, time, etc.).
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if v == "":
        return False

    # 1. If pure number, it is static
    try:
        float(v)
        return False
    except ValueError:
        pass

    # 2. If percentage literal (e.g. "50%", "-12.5%"), it is static
    if v.endswith("%"):
        try:
            float(v[:-1])
            return False
        except ValueError:
            pass

    # 3. If hex color (e.g. "#FFF", "#112233", "#11223344"), it is static
    if v.startswith("#") and len(v) in (4, 5, 7, 9):
        try:
            int(v[1:], 16)
            return False
        except ValueError:
            pass

    _EXPR_CHARS = set("+-*/%().,")
    lower = v.lower()
    if any(c in _EXPR_CHARS for c in v):
        return True
    if any(fn in lower for fn in _EXPR_FUNCS):
        return True
    return False


# ── Public API 3: calculator & calculater ─────────────────────────────────────

def calculator(
    text: object = None,
    *,
    live_text: object = None,
    empty_answer: object = "0",
    precision: int = 12,
    error_value: Optional[str] = None,
    variables: Optional[Dict[str, float]] = None,
    max_length: int = MAX_EXPR_LENGTH,
    max_nodes: int = MAX_AST_NODES,
) -> str:
    """
    Evaluate a mathematical expression and return a clean, trimmed answer string.

    Pipeline:
      1. Resolve text / live_text input
      2. Normalize & check length
      3. Parse AST safely
      4. Validate AST complexity & node count
      5. Evaluate safely with recursion, power, and overflow limits
      6. Format output to specified precision and trim trailing zeros

    Parameters
    ----------
    text         : Math expression string or number.
    live_text    : Optional live text binding or override.
    empty_answer : Value returned when expression is blank (default: "0").
    precision    : Max decimal places to format float results (default: 12).
    error_value  : If specified (e.g. "Error"), returned on errors instead of raising ValueError.
    variables    : Optional dict of variable names to numbers.
    max_length   : Maximum string character length (default: 1024).
    max_nodes    : Maximum AST nodes allowed (default: 250).
    """
    raw_text = live_text if live_text is not None else text
    expr = _normalize_text(raw_text)
    if expr == "":
        return _normalize_text(empty_answer)

    try:
        val = eval_expression(
            expr,
            variables=variables,
            max_length=max_length,
            max_nodes=max_nodes,
        )
        return _trim_number(val, precision)
    except ZeroDivisionError as exc:
        if error_value is not None:
            return error_value
        raise ValueError(f"Draw.calculator: division by zero in '{expr}'") from exc
    except (ValueError, OverflowError) as exc:
        if error_value is not None:
            return error_value
        raise ValueError(str(exc)) from exc


def calculater(*args: Any, **kwargs: Any) -> str:
    """Backward-compatible alias for Draw.calculator."""
    return calculator(*args, **kwargs)
