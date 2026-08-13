"""
Draw._room_size
================
Size-resolution engine used by Draw.room()'s ``sizes=`` parameter (and the
inline "fit"/"fill"/... shorthand allowed in scene= placement lists).

Resolves a per-id size spec — keywords, percentages, ip-based references
with arithmetic, aspect ratios, min/max/clamp, padding/margin-adjusted
fit, aspect-preserving scale (fit/cover) — into a concrete (w, h) in
pixels, given the object's current size, its layout parent's size, and a
lookup function for any other referenced object's current size.

This module is deliberately position-agnostic: it only ever returns
(width, height, scale_center). Draw.room()'s existing scene-relative /
object-relative placement math (_scene_anchor_pos / _object_relative_pos)
runs afterward exactly as before, using whatever (w, h) this module
resolves — an object resized via `sizes=` still gets positioned by its
normal `scene=` anchor/placement. The one exception is `scale_center`:
when set, Draw.room() overrides the placement result so the object's OLD
center point stays fixed after the resize, instead of re-anchoring to
the placement edge.

Not implemented (documented rather than faked):
  - `fit_cell`: Draw.room() has no grid-cell concept to size against.
    Use Draw.table's cell sizing for grid layouts instead.
  - bare `{"match": "largest"}` with no explicit sibling list: which ids
    to compare against is inherently ambiguous without one. Use
    `{"match": "largest", "of": ["a", "b", "c"]}` instead.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Tuple

# ip -> (w, h) in pixels, or None if the id doesn't exist / can't be found.
RefLookup = Callable[[str], Optional[Tuple[float, float]]]


class RoomSizeError(ValueError):
    """Raised for any Draw.room() size-spec problem."""


# One-word convenience keywords usable as a bare `sizes={"id": "fit"}`
# value, or inline as a modifier token in scene=["parent","placement","fit"].
KEYWORDS = frozenset({
    "fit", "fit_width", "fit_height", "fill", "auto",
    "stretch", "stretch_x", "stretch_y",
    "square", "half", "third", "quarter", "double",
    "content", "fit_content",
    "parent", "same", "copy", "inherit", "match_parent",
})


def _pct(value: str) -> Optional[float]:
    v = value.strip()
    if v.endswith("%"):
        try:
            return float(v[:-1]) / 100.0
        except ValueError:
            return None
    return None


# "parent", "parent.width", "panel", "panel.width", each optionally
# followed by an operator + number: "-40", "/3", "*1.5", "+20".
_EXPR_RE = re.compile(
    r"^\s*(parent|[A-Za-z_]\w*)\s*(?:\.\s*(width|height))?\s*"
    r"(?:([/*+\-])\s*([0-9]*\.?[0-9]+))?\s*$"
)

# A bare arithmetic-op token used inside a size-spec list, e.g. the "/2"
# in ["panel", "/2"].
_OP_TOKEN_RE = re.compile(r"^([/*+\-])\s*([0-9]*\.?[0-9]+)$")


def _apply_op(base: float, op: Optional[str], num: Optional[float]) -> float:
    if op is None:
        return base
    if op == "/":
        if num == 0:
            raise RoomSizeError("division by zero in size expression.")
        return base / num
    if op == "*":
        return base * num
    if op == "+":
        return base + num
    if op == "-":
        return base - num
    return base


def _eval_ref_expr(
    expr: str, axis: str, parent_w: float, parent_h: float, ref_lookup: RefLookup,
) -> Optional[float]:
    """
    Evaluate a small reference-expression string, e.g.:
        "parent"  "parent-40"  "parent/3"
        "panel"   "panel.width/2"   "panel.width-40"   "graph.height+20"

    `axis` ("width"|"height") is used when the expression doesn't name a
    dimension explicitly (bare "panel" or "parent"). Returns None if
    `expr` doesn't match the expression grammar at all, so the caller can
    fall through to other interpretations instead of hard-failing.
    """
    m = _EXPR_RE.match(expr)
    if not m:
        return None
    ref, dim, op, num = m.groups()
    dim = dim or axis
    if ref == "parent":
        base = parent_w if dim == "width" else parent_h
    else:
        size = ref_lookup(ref)
        if size is None:
            return None  # not a known id either — let caller raise a clearer error
        base = size[0] if dim == "width" else size[1]
    return _apply_op(base, op, float(num) if num is not None else None)


def _resolve_ref_list(
    tokens: list, axis: str, parent_w: float, parent_h: float, ref_lookup: RefLookup,
) -> float:
    """
    ["panel"]           -> panel.<axis>
    ["panel", "/2"]      -> panel.<axis> / 2
    ["panel", "*1.5"]    -> panel.<axis> * 1.5
    ["panel", "+50"]     -> panel.<axis> + 50
    ["panel", "-20"]     -> panel.<axis> - 20
    ["panel", "title"]   -> max(panel.<axis>, title.<axis>)   (#23 multiple refs)
    """
    if not tokens:
        raise RoomSizeError("empty id-reference list in size spec.")

    ref_ids, op, num = [], None, None
    for tok in tokens:
        if isinstance(tok, str):
            m = _OP_TOKEN_RE.match(tok.strip())
            if m:
                op, num = m.group(1), float(m.group(2))
                continue
            ref_ids.append(tok.strip())
        else:
            raise RoomSizeError(
                f"size spec list has a bare number {tok!r} with no "
                f"operator — use a string like '+50' / '-20' / '*1.5' / '/2'."
            )

    if not ref_ids:
        raise RoomSizeError("size spec list has no id references.")

    values = []
    for ref in ref_ids:
        if ref == "parent":
            values.append(parent_w if axis == "width" else parent_h)
            continue
        size = ref_lookup(ref)
        if size is None:
            raise RoomSizeError(f"size spec references unknown id '{ref}'.")
        values.append(size[0] if axis == "width" else size[1])

    base = values[0] if len(values) == 1 else max(values)
    return _apply_op(base, op, num)


def _unwrap_clamp(value: Any) -> Any:
    """#10 Clamp: {"value": X, "min":.., "max":..} -> X (the min/max are
    applied by the caller after resolving X). Anything else passes through
    unchanged."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _apply_inline_clamp(value: float, spec: Any) -> float:
    if isinstance(spec, dict) and "value" in spec:
        if "min" in spec:
            value = max(value, float(spec["min"]))
        if "max" in spec:
            value = min(value, float(spec["max"]))
    return value


def _resolve_axis_value(
    value: Any, axis: str, *, base_w: float, base_h: float,
    parent_w: float, parent_h: float, ref_lookup: RefLookup,
) -> float:
    """Resolve one already-unwrapped width/height value to pixels.
    `value` may be a number (px), a "NN%" string (#2), a reference
    expression string (#16/#22, e.g. "panel.width/2"), or an id-reference
    list (#3/#4/#5/#6/#7/#23, e.g. ["panel", "/2"])."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        pct = _pct(value)
        if pct is not None:
            parent_dim = parent_w if axis == "width" else parent_h
            return parent_dim * pct
        expr_val = _eval_ref_expr(value, axis, parent_w, parent_h, ref_lookup)
        if expr_val is not None:
            return expr_val
        raise RoomSizeError(
            f"could not parse size value {value!r} for '{axis}'. "
            f"Expected a number, a 'NN%' string, or an expression like "
            f"'panel.width/2' or 'parent-40'."
        )
    if isinstance(value, list):
        return _resolve_ref_list(value, axis, parent_w, parent_h, ref_lookup)
    raise RoomSizeError(f"unsupported size value {value!r} for '{axis}'.")


def _resolve_scale(
    spec: dict, base_w: float, base_h: float, parent_w: float, parent_h: float,
) -> Tuple[float, float]:
    """#15 Auto Scale — aspect-preserving fit/contain or cover, like CSS
    object-fit. `base_w`/`base_h` supply the object's natural aspect
    ratio (its current size)."""
    mode = str(spec["scale"]).strip().lower()
    if base_w <= 0 or base_h <= 0 or parent_h <= 0:
        return parent_w, parent_h
    obj_ratio = base_w / base_h
    w, h = parent_w, parent_w / obj_ratio
    if mode in ("fit", "contain"):
        if h > parent_h:
            h = parent_h
            w = h * obj_ratio
    elif mode == "cover":
        if h < parent_h:
            h = parent_h
            w = h * obj_ratio
    else:
        raise RoomSizeError(
            f"unknown scale mode {mode!r}; expected 'fit'/'contain'/'cover'."
        )
    return w, h


def resolve_size_spec(
    spec: Any,
    *,
    base_w: float, base_h: float,        # object's current size (the "auto" fallback)
    parent_w: float, parent_h: float,    # immediate layout parent's size
    ref_lookup: RefLookup,
) -> Tuple[float, float, bool]:
    """
    Resolve one full size spec — a keyword string, or a dict combining
    any of the features documented in the module docstring — into
    (w, h, scale_center) in pixels.

    spec=None returns (base_w, base_h, False) unchanged, so callers can
    call this unconditionally for every room() entry regardless of
    whether that id actually has a size spec.
    """
    if spec is None:
        return base_w, base_h, False

    # ── one-word keywords (#1 fit family, #13 stretch, #14/#25 aliases) ──
    if isinstance(spec, str):
        kw = spec.strip().lower()
        if kw not in KEYWORDS:
            raise RoomSizeError(
                f"unknown size keyword {spec!r}. Recognized: {sorted(KEYWORDS)}."
            )
        if kw in ("auto", "same", "copy", "inherit"):
            return base_w, base_h, False
        if kw == "fit":
            return parent_w, parent_h, False
        if kw == "fit_width":
            return parent_w, base_h, False
        if kw == "fit_height":
            return base_w, parent_h, False
        if kw in ("fill", "parent", "match_parent", "stretch"):
            return parent_w, parent_h, False
        if kw == "stretch_x":
            return parent_w, base_h, False
        if kw == "stretch_y":
            return base_w, parent_h, False
        if kw == "square":
            side = min(base_w, base_h) if (base_w and base_h) else max(base_w, base_h)
            return side, side, False
        if kw == "half":
            return base_w / 2.0, base_h / 2.0, False
        if kw == "third":
            return base_w / 3.0, base_h / 3.0, False
        if kw == "quarter":
            return base_w / 4.0, base_h / 4.0, False
        if kw == "double":
            return base_w * 2.0, base_h * 2.0, False
        if kw in ("content", "fit_content"):
            # #17 fit_content: real content measurement (e.g. measure_text
            # for text objects) happens one layer up in _room.py, which
            # passes the already-measured size in as base_w/base_h — this
            # generic layer has no text-measurement context of its own, so
            # "content" here is just "use whatever base_w/base_h already is".
            return base_w, base_h, False

    if not isinstance(spec, dict):
        raise RoomSizeError(f"unsupported size spec {spec!r}.")

    scale_center = bool(spec.get("scale_center", False))  # #21

    # #15 Auto Scale — short-circuits everything else (own aspect-ratio logic).
    if "scale" in spec:
        w, h = _resolve_scale(spec, base_w, base_h, parent_w, parent_h)
        if "min_width" in spec: w = max(w, float(spec["min_width"]))
        if "max_width" in spec: w = min(w, float(spec["max_width"]))
        if "min_height" in spec: h = max(h, float(spec["min_height"]))
        if "max_height" in spec: h = min(h, float(spec["max_height"]))
        return w, h, scale_center

    # #1 fit / fit_width / fit_height / #6 fill / #13 stretch_x / stretch_y
    fit_w = bool(spec.get("fit", False)) or bool(spec.get("fit_width", False))
    fit_h = bool(spec.get("fit", False)) or bool(spec.get("fit_height", False))
    if spec.get("fill", False):
        fit_w = fit_h = True
    if spec.get("stretch_x", False):
        fit_w = True
    if spec.get("stretch_y", False):
        fit_h = True

    w = parent_w if fit_w else base_w
    h = parent_h if fit_h else base_h

    # #11 padding / #12 margin shrink a fitted size on both edges of that axis.
    if fit_w or fit_h:
        pad = spec.get("padding", spec.get("margin", 0)) or 0
        pad = float(pad)
        if pad:
            if fit_w:
                w = max(0.0, w - 2 * pad)
            if fit_h:
                h = max(0.0, h - 2 * pad)

    # #3/#4/#5/#6/#7/#16/#22/#23/#14 explicit size / width / height (also
    # covers equal_width/equal_height, which are just aliases for width/
    # height with an id-reference value).
    if "size" in spec:
        raw = _unwrap_clamp(spec["size"])
        w = _resolve_axis_value(raw, "width", base_w=base_w, base_h=base_h,
                                 parent_w=parent_w, parent_h=parent_h, ref_lookup=ref_lookup)
        h = _resolve_axis_value(raw, "height", base_w=base_w, base_h=base_h,
                                 parent_w=parent_w, parent_h=parent_h, ref_lookup=ref_lookup)
        w = _apply_inline_clamp(w, spec["size"])
        h = _apply_inline_clamp(h, spec["size"])
    if "width" in spec or "equal_width" in spec:
        wv = spec.get("width", spec.get("equal_width"))
        w = _resolve_axis_value(_unwrap_clamp(wv), "width", base_w=base_w, base_h=base_h,
                                 parent_w=parent_w, parent_h=parent_h, ref_lookup=ref_lookup)
        w = _apply_inline_clamp(w, wv)
    if "height" in spec or "equal_height" in spec:
        hv = spec.get("height", spec.get("equal_height"))
        h = _resolve_axis_value(_unwrap_clamp(hv), "height", base_w=base_w, base_h=base_h,
                                 parent_w=parent_w, parent_h=parent_h, ref_lookup=ref_lookup)
        h = _apply_inline_clamp(h, hv)

    # #8 Aspect ratio — the axis NOT explicitly given is derived from the
    # one that was (width -> height by default, unless only height/
    # equal_height was explicitly given).
    if "ratio" in spec or "aspect" in spec:
        ratio = spec.get("ratio", spec.get("aspect"))
        if isinstance(ratio, str) and ":" in ratio:
            rw, rh = ratio.split(":", 1)
            rh_val = float(rh)
            if rh_val == 0:
                raise RoomSizeError(f"invalid aspect ratio {ratio!r} (denominator is 0).")
            ratio_val = float(rw) / rh_val
        else:
            ratio_val = float(ratio)
        height_given = "height" in spec or "equal_height" in spec
        width_given = "width" in spec or "equal_width" in spec or "size" in spec
        if not height_given:
            h = w / ratio_val if ratio_val else h
        elif not width_given:
            w = h * ratio_val

    # #9 Min / Max — applied last, after everything above.
    if "min_width" in spec: w = max(w, float(spec["min_width"]))
    if "max_width" in spec: w = min(w, float(spec["max_width"]))
    if "min_height" in spec: h = max(h, float(spec["min_height"]))
    if "max_height" in spec: h = min(h, float(spec["max_height"]))

    return w, h, scale_center


def resolve_match(spec: dict, ref_lookup: RefLookup) -> Tuple[float, float]:
    """#18 Match Largest — requires an explicit `of: [ids]` list; there's
    no well-defined default group of "siblings" to compare against
    otherwise. spec = {"match": "largest", "of": ["a", "b", "c"]}."""
    of = spec.get("of")
    if not isinstance(of, list) or not of:
        raise RoomSizeError(
            "{'match': 'largest'} requires an explicit 'of': [ids] "
            "list — there's no implicit sibling group to compare against."
        )
    mode = str(spec.get("match", "largest")).strip().lower()
    sizes = [ref_lookup(i) for i in of]
    sizes = [s for s in sizes if s is not None]
    if not sizes:
        raise RoomSizeError(f"none of the ids in 'of'={of!r} were found.")
    pick = max if mode == "largest" else min
    return pick(s[0] for s in sizes), pick(s[1] for s in sizes)
