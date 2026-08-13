"""
Draw._calculator
Small safe calculator with trimmed string output.
Extended with math functions (sin, cos, abs, min, max, lerp, step)
and variable injection for the Draw.color() expression system.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, Optional


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

# Safe built-in functions available in expressions
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
}


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b by t (0-1)."""
    return a + (b - a) * max(0.0, min(1.0, t))


def _step(edge: float, x: float) -> float:
    """Returns 0.0 if x < edge, else 1.0."""
    return 0.0 if x < edge else 1.0


_SAFE_FUNCTIONS["lerp"] = _lerp
_SAFE_FUNCTIONS["step"] = _step


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
        # Point to the offending character in the expression
        pointer = " " * (col - 1) + "^" if col > 0 else ""
        return (
            f"Draw.calculater: syntax error at column {col} in '{expr}'\n"
            f"  {expr}\n"
            f"  {pointer}\n"
            f"  ({msg})"
        )
    return f"Draw.calculater: syntax error in '{expr}' ({msg})"


def _eval_expr(node: ast.AST, variables: Optional[Dict[str, float]] = None) -> float:
    """Safely evaluate an AST node with optional variable/function support."""
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body, variables)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(
            f"Draw.calculater: only numeric constants are allowed, "
            f"got {type(node.value).__name__} ({node.value!r})."
        )

    # Variable lookup (e.g. time, x, y, mouse_x, etc.)
    if isinstance(node, ast.Name):
        name = node.id
        if variables and name in variables:
            return float(variables[name])
        if name.lower() == "pi":
            return math.pi
        if name.lower() == "e":
            return math.e
        raise ValueError(
            f"Draw.calculater: unknown variable '{name}'. "
            f"Available variables must be passed via the 'variables' argument."
        )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
        operand = _eval_expr(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +operand
        return -operand

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINARY_OPS):
        left = _eval_expr(node.left, variables)
        right = _eval_expr(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError(
                    f"Draw.calculater: division by zero ('{left} / {right}')."
                )
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ZeroDivisionError(
                    f"Draw.calculater: floor division by zero ('{left} // {right}')."
                )
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise ZeroDivisionError(
                    f"Draw.calculater: modulo by zero ('{left} % {right}')."
                )
            return left % right
        if isinstance(node.op, ast.Pow):
            return left ** right

    # Function calls: sin(x), cos(x), abs(x), min(a,b), max(a,b), lerp(a,b,t), step(e,x)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                "Draw.calculater: only simple function calls are allowed "
                "(e.g. sin(x), not obj.method(x))."
            )
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            known = ", ".join(sorted(_SAFE_FUNCTIONS.keys()))
            raise ValueError(
                f"Draw.calculater: unknown function '{func_name}'. "
                f"Available functions: {known}."
            )
        args = [_eval_expr(arg, variables) for arg in node.args]
        try:
            return float(_SAFE_FUNCTIONS[func_name](*args))
        except TypeError as exc:
            raise ValueError(
                f"Draw.calculater: wrong arguments for '{func_name}': {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Draw.calculater: math error in '{func_name}': {exc}"
            ) from exc

    # Unsupported AST node type
    node_type = type(node).__name__
    raise ValueError(
        f"Draw.calculater: unsupported expression element '{node_type}'. "
        f"Only arithmetic operators (+, -, *, /, //, %, **) and safe "
        f"math functions are allowed."
    )


def eval_expression(
    expr: str,
    variables: Optional[Dict[str, float]] = None,
) -> float:
    """
    Evaluate a math expression string with optional variables.
    This is the primary entry point for the Draw.color() expression system.
    Uses a safe AST parser — no eval() or exec().

    Parameters
    ----------
    expr       : Math expression string, e.g. "sin(time * 3) * 50 + 50"
    variables  : Dict of variable names to float values, e.g. {"time": 1.5, "x": 100}

    Returns
    -------
    float result of the expression.

    Raises
    ------
    ValueError with a descriptive message on any parse or evaluation error.
    """
    expr = expr.strip()
    if expr == "":
        return 0.0
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(_format_syntax_error(exc, expr)) from exc
    return _eval_expr(tree, variables)


def is_expression(value: Any) -> bool:
    """Check if a value is a math expression string (not a plain number or color name)."""
    if not isinstance(value, str):
        return False
    v = value.strip()
    if v == "":
        return False
    # If it's a pure number, it's not an expression (it's static)
    try:
        float(v)
        return False
    except ValueError:
        pass
    # If it's a percentage literal like "50%" or "100%", it's not a dynamic math expression
    if v.endswith("%"):
        try:
            float(v[:-1])
            return False
        except ValueError:
            pass
    # If it contains math operators or function names, it's an expression
    _EXPR_CHARS = set("+-*/%().,")
    _EXPR_FUNCS = {"sin", "cos", "abs", "min", "max", "sqrt", "floor", "ceil",
                   "round", "lerp", "step", "time", "mouse_x", "mouse_y"}
    lower = v.lower()
    if any(c in _EXPR_CHARS for c in v):
        return True
    if any(fn in lower for fn in _EXPR_FUNCS):
        return True
    return False


def calculater(
    text: object = None,
    *,
    live_text: object = None,
    empty_answer: object = "0",
    precision: int = 12,
    error_value: Optional[str] = None,
) -> str:
    """
    Evaluate a simple math expression and return a trimmed answer string.

    Rules:
    - If live_text is provided, it is used first.
    - If text is missing/blank after trim, returns empty_answer (trimmed).
    - Supports only numeric expressions with +, -, *, /, //, %, **, and ().
    - On any error: raises ValueError with a specific, human-readable message.
      Pass error_value="Error" (or any string) to return that string instead
      of raising — useful for a calculator app's display.

    Error messages describe the exact problem:
      "syntax error at column 5 in '2 + * 3' (unexpected token)"
      "division by zero"
      "unknown variable 'x'"
      "unknown function 'foo'. Available functions: abs, acos, ..."
      etc.
    """
    raw_text = live_text if live_text is not None else text
    expr = _normalize_text(raw_text)
    if expr == "":
        return _normalize_text(empty_answer)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        msg = _format_syntax_error(exc, expr)
        if error_value is not None:
            return error_value
        raise ValueError(msg) from exc

    try:
        value = _eval_expr(tree)
    except ZeroDivisionError as exc:
        msg = f"Draw.calculater: division by zero in '{expr}'"
        if error_value is not None:
            return error_value
        raise ValueError(msg) from exc
    except ValueError:
        if error_value is not None:
            return error_value
        raise

    return _trim_number(value, precision)


def calculator(
    text: object = None,
    *,
    live_text: object = None,
    empty_answer: object = "0",
    precision: int = 12,
    error_value: Optional[str] = None,
) -> str:
    """Alias for Draw.calculater (correct spelling helper)."""
    return calculater(
        text=text,
        live_text=live_text,
        empty_answer=empty_answer,
        precision=precision,
        error_value=error_value,
    )
