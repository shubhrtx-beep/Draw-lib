"""
Draw._validation

Lightweight, dependency-free schema validation used at Draw's public API
boundaries (currently: shape dicts). The goal is fail-fast behavior: a
misspelled key like "colour" or "opactiy" should raise a clear, immediate
ValueError naming the bad key and suggesting the likely correct spelling,
instead of being silently ignored and producing a confusing wrong-looking
shape with no error at all.

This module intentionally does NOT try to validate value *types* or *ranges*
in general — Draw's existing `_as_int` / `_as_float` / `_parse_*` helpers
already coerce and clamp those defensively. The job here is narrower and
cheaper: catch keys that don't exist at all, which coercion can't catch
because `raw.get("colour", default)` just silently returns the default.

Usage
-----
    from Draw._validation import validate_keys

    validate_keys(
        raw, KNOWN_SHAPE_KEYS,
        kind="Draw.shape", obj_id=ip_str,
    )
"""

from __future__ import annotations

import difflib
from typing import Iterable, Optional


# ── shape dict keys ──────────────────────────────────────────────────────────
# Sourced directly from every `raw.get("...")` / `"..." in raw` read inside
# Draw._shapes._parse_shape. Keep this list in sync with that function —
# if you add a new recognized shape key there, add it here too, or every
# shape using it will raise a false-positive "unknown key" error.
KNOWN_SHAPE_KEYS: frozenset[str] = frozenset({
    "vertices", "size", "width", "height", "border_radius",
    "x", "y", "align", "rotation",
    "color", "border_color", "border_width", "border_style", "opacity",
    "curve_mode",
    "custom", "customise",
    "bend", "bend_amount", "warp", "exclude", "symmetry",
    "hitbox_mode", "hit_box",
    "z", "overlap", "flow",
    "ip", "get_ip", "layout",
    "column", "columns",
    "area",
    "type", "src", "loop", "autoplay", "muted",
    "inside", "move_path",
    "locked",
    "custom_vertices",
})


# ── graph customise dict keys ────────────────────────────────────────────────
# Recognized keys inside the 'customise' dict of a Draw.graph() series.
# Keep in sync with Draw._graph._render_series's customise-key reads.
KNOWN_GRAPH_CUSTOMISE_KEYS: frozenset[str] = frozenset({
    "margin", "show_line", "title", "x_title", "y_title",
    "title_font_size", "title_color", "axis_title_size", "axis_title_color",
    "grid_lines", "grid_color", "grid_width", "axis_color",
    "y_tick_color", "max_x", "min_y", "max_y",
    "color", "border_width", "border_color", "border_style",
    "bar_width", "line_width", "line_color",
    "point_size", "point_vertices",
    "label_color", "label_font_size",
    "show_values", "value_color", "value_font_size", "value_decimals",
    "group_width_ratio", "flow",
    # Shape passthrough keys
    "curve_mode", "bend", "bend_amount", "warp", "exclude", "symmetry",
    "opacity", "type", "src", "hitbox_mode", "hit_box",
    # Color extensions
    "gradient", "stops", "color_ip",
})


class DrawValidationError(ValueError):
    """Raised when a Draw API call is given an unrecognized dict key."""


def _suggest(key: str, known: Iterable[str]) -> Optional[str]:
    """Return the closest known key to `key`, or None if nothing is close."""
    matches = difflib.get_close_matches(key, list(known), n=1, cutoff=0.6)
    return matches[0] if matches else None


def validate_keys(
    raw: dict,
    known_keys: Iterable[str],
    *,
    kind: str,
    obj_id: Optional[str] = None,
) -> None:
    """
    Raise DrawValidationError if `raw` contains any key not in `known_keys`.

    `kind` is a short label for the error message (e.g. "Draw.shape").
    `obj_id` is the shape/text/panel ip, if known, for context in the error.
    """
    if not isinstance(raw, dict):
        return  # type errors are handled by the caller; not our job here

    known = set(known_keys)
    unknown = [k for k in raw.keys() if isinstance(k, str) and k not in known]
    if not unknown:
        return

    where = f" ('{obj_id}')" if obj_id else ""
    lines = [f"{kind}{where}: unrecognized key(s): {unknown!r}."]
    for bad_key in unknown:
        suggestion = _suggest(bad_key, known)
        if suggestion:
            lines.append(f"  '{bad_key}' — did you mean '{suggestion}'?")
    raise DrawValidationError("\n".join(lines))
