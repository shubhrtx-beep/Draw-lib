"""
Draw._align
Canonical, high-precision alignment calculation engine for Draw.
Unifies screen, canvas, padded scene, and object-relative alignments.
"""

from __future__ import annotations

from typing import Optional, Tuple

_ALIGN_PRESETS = {
    "center",
    "top",
    "bottom",
    "left",
    "right",
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
}


def normalize_alignment(align: str) -> str:
    """Normalize alignment string (strip, lowercase, underscore to hyphen)."""
    if not isinstance(align, str):
        return "center"
    return align.strip().lower().replace("_", "-")


def is_valid_alignment(align: str) -> bool:
    """Return True if align is a recognized preset or ip reference."""
    if not isinstance(align, str):
        return False
    norm = normalize_alignment(align)
    return norm in _ALIGN_PRESETS or norm.startswith("ip:")


def calculate_alignment_pos(
    align: str,
    sw: float,
    sh: float,
    cw: float,
    ch: float,
    pad: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    window_tag: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Unified canonical alignment calculator for Draw.

    Parameters
    ----------
    align      : Alignment preset ("center", "top", "bottom", "left", "right",
                 "top-left", "top-right", "bottom-left", "bottom-right") or
                 ip point reference ("ip:target_ip").
    sw, sh     : Subject width and height.
    cw, ch     : Container width and height.
    pad        : Optional padding offset inside container.
    offset_x/y : Container origin offset (e.g., screen x/y or parent x/y).
    window_tag : Optional window tag for resolving "ip:" point references.

    Returns
    -------
    Tuple[float, float]
        High-precision (x, y) coordinates.
    """
    if not isinstance(align, str):
        return (offset_x + (cw - sw) / 2.0, offset_y + (ch - sh) / 2.0)

    align_norm = normalize_alignment(align)

    # Support dynamic IP point anchoring ("ip:my_shape")
    if align_norm.startswith("ip:"):
        from Draw import _bridge
        point = _bridge.resolve_point_ref(align, window_tag, self_rect=None)
        if point is not None:
            cx, cy = point
            return (cx - sw / 2.0, cy - sh / 2.0)
        # Fallback to centered if IP is not yet resolvable
        return (offset_x + (cw - sw) / 2.0, offset_y + (ch - sh) / 2.0)

    if align_norm == "center":
        return (offset_x + (cw - sw) / 2.0, offset_y + (ch - sh) / 2.0)
    elif align_norm == "top":
        return (offset_x + (cw - sw) / 2.0, offset_y + pad)
    elif align_norm == "bottom":
        return (offset_x + (cw - sw) / 2.0, offset_y + ch - sh - pad)
    elif align_norm == "left":
        return (offset_x + pad, offset_y + (ch - sh) / 2.0)
    elif align_norm == "right":
        return (offset_x + cw - sw - pad, offset_y + (ch - sh) / 2.0)
    elif align_norm == "top-left":
        return (offset_x + pad, offset_y + pad)
    elif align_norm == "top-right":
        return (offset_x + cw - sw - pad, offset_y + pad)
    elif align_norm == "bottom-left":
        return (offset_x + pad, offset_y + ch - sh - pad)
    elif align_norm == "bottom-right":
        return (offset_x + cw - sw - pad, offset_y + ch - sh - pad)

    return (offset_x + (cw - sw) / 2.0, offset_y + (ch - sh) / 2.0)


def calculate_alignment_rect(
    align: str,
    sw: float,
    sh: float,
    cw: float,
    ch: float,
    pad: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    window_tag: Optional[str] = None,
) -> Tuple[float, float, float, float]:
    """Return aligned bounding tuple (x, y, sw, sh)."""
    x, y = calculate_alignment_pos(align, sw, sh, cw, ch, pad, offset_x, offset_y, window_tag)
    return (x, y, sw, sh)


# ═══════════════════════════════════════════════════════════════════════════
# Universal Layout Primitives
# ═══════════════════════════════════════════════════════════════════════════
# These functions centralise all bounding-box / inset / chart-region
# calculations so that _graph.py, _shapes.py, _panel.py, and any future
# Draw module never need inline pixel math again.


def calculate_inset_rect(
    cw: float,
    ch: float,
    pad_left: float = 0.0,
    pad_top: float = 0.0,
    pad_right: float = 0.0,
    pad_bottom: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    min_w: float = 80.0,
    min_h: float = 80.0,
) -> Tuple[float, float, float, float]:
    """
    Core asymmetric-padding primitive: compute the inner rectangle after
    subtracting independent padding from each side of a container.

    Parameters
    ----------
    cw, ch         : Container width and height.
    pad_left/top/right/bottom : Padding on each side (pixels).
    offset_x/y     : Container origin offset.
    min_w, min_h   : Minimum inner dimensions (clamped, never negative).

    Returns
    -------
    (x, y, width, height) of the inner rect in absolute coordinates.
    """
    inner_x = offset_x + pad_left
    inner_y = offset_y + pad_top
    inner_w = max(min_w, cw - pad_left - pad_right)
    inner_h = max(min_h, ch - pad_top - pad_bottom)
    return (inner_x, inner_y, inner_w, inner_h)


def calculate_chart_region(
    cell_x: float,
    cell_y: float,
    cell_w: float,
    cell_h: float,
    margin: float = 48.0,
    has_y_axis: bool = False,
    has_title: bool = False,
    has_x_title: bool = False,
    align: object = None,
    window_tag: Optional[str] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute a cartesian chart's drawable region within a layout cell.

    Encapsulates the left_pad / top_pad / bottom_pad logic and alignment
    offset that was previously inlined in ``_graph._render_cartesian_series``.

    Parameters
    ----------
    cell_x, cell_y : Cell origin (from Draw.table / get_ip).
    cell_w, cell_h : Cell dimensions.
    margin         : Outer margin from customise dict (default 48).
    has_y_axis     : True if y-axis ticks / y-title are shown (wider left pad).
    has_title      : True if a chart title is shown (taller top pad).
    has_x_title    : True if an x-axis title is shown (taller bottom pad).
    align          : Alignment preset or None.
    window_tag     : Window tag for ip: alignment resolution.

    Returns
    -------
    (left, top, right, bottom, chart_w, chart_h)  — all in absolute px.
    """
    left_pad = 72.0 if has_y_axis else 24.0
    top_pad = 42.0 if has_title else 15.0
    bottom_pad = 36.0 + (22.0 if has_x_title else 0.0)

    # Compute inner rect using the inset primitive
    pad_l = margin + left_pad
    pad_t = margin + top_pad
    pad_r = margin
    pad_b = margin + bottom_pad

    inner_x, inner_y, inner_w, inner_h = calculate_inset_rect(
        cell_w, cell_h,
        pad_left=pad_l, pad_top=pad_t,
        pad_right=pad_r, pad_bottom=pad_b,
        offset_x=0.0, offset_y=0.0,
        min_w=80.0, min_h=80.0,
    )

    # inner_x/inner_y are relative to cell origin (0,0)
    left = inner_x
    top = inner_y
    right = left + inner_w
    bottom = top + inner_h
    chart_w = inner_w
    chart_h = inner_h

    # Apply alignment offset
    if align is not None:
        ax, ay = calculate_alignment_pos(
            align, sw=chart_w, sh=chart_h, cw=cell_w, ch=cell_h,
            offset_x=cell_x, offset_y=cell_y, window_tag=window_tag,
        )
        dx = ax - left
        dy = ay - top
    else:
        dx = cell_x
        dy = cell_y

    left += dx
    right += dx
    top += dy
    bottom += dy

    return (left, top, right, bottom, chart_w, chart_h)


import math as _math


def calculate_radial_center(
    cell_x: float,
    cell_y: float,
    cell_w: float,
    cell_h: float,
    margin: float = 30.0,
    has_labels: bool = False,
    has_title: bool = False,
    align: object = None,
    window_tag: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Compute center and radius for a radial chart (pie, donut, radar)
    within a layout cell.

    Parameters
    ----------
    cell_x, cell_y : Cell origin.
    cell_w, cell_h : Cell dimensions.
    margin         : Outer margin.
    has_labels     : True if category labels sit outside the circle (needs gutter).
    has_title      : True if a chart title is shown.
    align          : Alignment preset or None.
    window_tag     : Window tag for ip: alignment resolution.

    Returns
    -------
    (center_x, center_y, radius)  — all in absolute px.
    """
    label_gutter = 100.0 if has_labels else 20.0
    title_gutter = 40.0 if has_title else 0.0

    chart_w = min(cell_w - label_gutter, cell_h - title_gutter) - margin * 2.0
    chart_w = max(40.0, chart_w)
    chart_h = chart_w
    radius = chart_w / 2.0

    if align is not None:
        ax, ay = calculate_alignment_pos(
            str(align), sw=chart_w, sh=chart_h, cw=cell_w, ch=cell_h,
            offset_x=cell_x, offset_y=cell_y, window_tag=window_tag,
        )
        cx = ax + radius
        cy = ay + radius
    else:
        cx = cell_x + cell_w / 2.0
        cy = cell_y + cell_h / 2.0
        if has_labels:
            # Shift left to leave room for legend on the right
            cx = cell_x + (cell_w - label_gutter) / 2.0
        if has_title:
            cy += title_gutter / 4.0

    return (cx, cy, radius)


# ═══════════════════════════════════════════════════════════════════════════
# Z-Layer System — canonical, centralized depth calculation
# ═══════════════════════════════════════════════════════════════════════════
# Draw._shapes paints in DESCENDING z order (see _DrawCanvas.paintEvent):
#   sorted_shapes = sorted(self.shape_items, key=lambda s: -s.z)
# i.e. the HIGHEST z value is painted FIRST (furthest back / bottom), and the
# LOWEST z value is painted LAST (frontmost / on top). This is the *opposite*
# of typical CSS z-index intuition, and every hand-picked magic z number in
# the codebase so far has assumed the CSS convention ("bigger = more in
# front"), which quietly buries the element instead:
#   - Draw._graph's legend box used z=200 meaning "keep it visible on top",
#     which — under DESCENDING sort — sinks it far behind ordinary content
#     and behind chart series (which mostly fall back to the small,
#     ever-growing Draw._tools.next_z() auto counter).
#   - Draw._graph's legend *text* had no z at all, so it landed wherever
#     next_z() happened to be that frame — sometimes behind the bars it was
#     labelling, producing exactly the "ghost text behind the chart" bug.
#   - A hover tooltip written with z=999 ("bring to front") is instead
#     pinned to the very back and invisible.
#
# `zlayer()` is the fix: one canonical function, colocated with
# `calculate_alignment_pos` because it solves the same class of problem —
# "don't let every module reinvent this calculation" — except for the z axis
# instead of x/y. Callers pass a named ZLayer instead of a magic number, so
# the back/front intent is legible at the call site and can never be
# accidentally inverted again.
#
# Layers are split into two bands so they can NEVER collide with the
# ordinary Draw._tools.next_z() auto-counter (which starts at 0 and only
# grows for the life of the app):
#   - "back" layers (backgrounds, chart structure) sit at large POSITIVE
#     values, comfortably above anything next_z() will realistically reach.
#   - "front" layers (legend, overlays, tooltips, active drag) sit at
#     NEGATIVE values — a guarantee, not a guess, that holds no matter how
#     long the app runs or how many shapes it has created, since next_z()
#     never produces a negative number.

from enum import IntEnum


class ZLayer(IntEnum):
    """Named depth layers, back → front. See module docstring above for the
    DESCENDING-sort rule this hub exists to make foolproof. Treat the exact
    integers as an implementation detail — always refer to layers by name
    via `zlayer()`, never by writing a number directly."""

    # ── back band (large positive; always behind ordinary auto-z content) ──
    FAR_BACK        = 1_000_000   # window/canvas-filling backgrounds
    PANEL_BG        = 900_000     # panel / card / container backgrounds
    CHART_GRID      = 800_000     # chart gridlines, background bands
    CHART_AXIS      = 700_000     # axis lines, axis ticks, axis titles
    CHART_SERIES    = 600_000     # bar / area / pie / donut fills
    CHART_SERIES_FG = 500_000     # line / dot series drawn over fills
    CHART_LABEL     = 400_000     # value labels, data-point labels

    # ── baseline ──
    # Ordinary Draw.shapes()/Draw.text() calls with no explicit layer use
    # Draw._tools.next_z() here: 0, 1, 2, ... growing over the app's life.

    # ── front band (negative; always in front of ordinary auto-z content,
    #    permanently — next_z() can never produce a negative value) ──
    LEGEND_BG       = -100_000    # legend background panel
    LEGEND_ITEM     = -110_000    # legend swatches + labels (over LEGEND_BG)
    OVERLAY         = -200_000    # drag hit-targets, focus rings
    TOOLTIP         = -300_000    # hover tooltips
    DRAG_ACTIVE     = -400_000    # element currently being dragged
    TOPMOST         = -500_000    # always frontmost of everything


def zlayer(layer: "ZLayer | str", offset: int = 0) -> int:
    """
    Canonical z-value calculator — the z-axis counterpart of
    `calculate_alignment_pos`. Every module should call this instead of
    hand-picking a z number.

    Parameters
    ----------
    layer  : A ZLayer member, or its name as a string (e.g. "legend_item",
             case-insensitive).
    offset : Sub-index *within* the layer, for deterministic stacking of
             many elements that share one layer (e.g. legend row `i`, pie
             wedge index `i`). Larger offsets push an element slightly
             further BACK within its own layer (matching the module's
             DESCENDING paint order) — pass e.g. a loop index so item 0
             ends up in front of item 1, item 2, etc. Keep offsets small
             (well under ~1000); each layer has that much headroom before
             the next layer up.

    Returns
    -------
    int  A z value ready to drop straight into a shape/text "z" field.

    Examples
    --------
    >>> zlayer(ZLayer.LEGEND_BG)                # legend panel background
    >>> zlayer(ZLayer.LEGEND_ITEM, i)            # legend row i (swatch/text)
    >>> zlayer("chart_series", idx)              # pie wedge idx
    >>> zlayer(ZLayer.TOOLTIP)                   # always wins, always visible
    """
    if isinstance(layer, str):
        layer = ZLayer[layer.strip().upper()]
    return int(layer) - int(offset)


# ═══════════════════════════════════════════════════════════════════════════
# AlignCalc — reusable spatial / alignment calculation engine
# ═══════════════════════════════════════════════════════════════════════════
# Centralises the maths that every module doing positional computation
# (scrollers, sliders, layout, motion snapping, …) would otherwise inline.
#
# Primary purpose: alignment calculations (hence the name).
# Secondary: general spatial primitives (clamp, lerp, remap, thumb-track
# geometry) that come up whenever you compose _senses + _connectors into
# a scrollable / draggable / slidable widget.
#
# Usage:
#   from Draw._align import calc
#   y = calc.clamp(raw_y, 0, max_h)
#   t = calc.normalize(scroll_y, 0, content_h - viewport_h)
#   thumb_y = calc.thumb_position(t, track_y, track_h, thumb_h)


class AlignCalc:
    """Reusable calculation primitives for alignment, scrolling, and
    general range / position maths.

    Every method is a pure function (no side-effects, no Qt dependencies).
    Instantiate once as a module singleton (``calc = AlignCalc()``) and
    import that everywhere — mirrors the registry-singleton pattern used
    by ``senses``, ``connectors``, ``motion``, etc.
    """

    # ── core primitives ───────────────────────────────────────────────────

    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        """Constrain *value* to [lo, hi]."""
        return max(lo, min(hi, float(value)))

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation from *a* to *b* at fraction *t* ∈ [0, 1]."""
        return float(a) + (float(b) - float(a)) * float(t)

    @staticmethod
    def inverse_lerp(a: float, b: float, value: float) -> float:
        """Compute the *t* fraction such that ``lerp(a, b, t) == value``.

        Returns 0.0 if *a* == *b* (degenerate range).
        """
        denom = float(b) - float(a)
        if abs(denom) < 1e-12:
            return 0.0
        return (float(value) - float(a)) / denom

    @staticmethod
    def remap(
        value: float,
        in_lo: float, in_hi: float,
        out_lo: float, out_hi: float,
        *,
        clamped: bool = True,
    ) -> float:
        """Map *value* from range [in_lo, in_hi] to [out_lo, out_hi].

        If *clamped* (default), the result is constrained to [out_lo, out_hi].
        """
        in_range = float(in_hi) - float(in_lo)
        if abs(in_range) < 1e-12:
            return float(out_lo)
        t = (float(value) - float(in_lo)) / in_range
        if clamped:
            t = max(0.0, min(1.0, t))
        return float(out_lo) + t * (float(out_hi) - float(out_lo))

    @staticmethod
    def smoothstep(edge0: float, edge1: float, x: float) -> float:
        """Hermite interpolation (smooth S-curve) of *x* between edges."""
        rng = float(edge1) - float(edge0)
        if abs(rng) < 1e-12:
            return 0.0 if float(x) < float(edge0) else 1.0
        t = max(0.0, min(1.0, (float(x) - float(edge0)) / rng))
        return t * t * (3.0 - 2.0 * t)

    # ── scroll / slider geometry ──────────────────────────────────────────

    @staticmethod
    def normalize(scroll: float, min_scroll: float, max_scroll: float) -> float:
        """Convert a raw scroll offset to a 0-1 fraction.

        ``normalize(scroll_y, 0, content_h - viewport_h) → 0..1``
        """
        rng = float(max_scroll) - float(min_scroll)
        if rng <= 0.0:
            return 0.0
        return max(0.0, min(1.0, (float(scroll) - float(min_scroll)) / rng))

    @staticmethod
    def denormalize(t: float, min_scroll: float, max_scroll: float) -> float:
        """Convert a 0-1 fraction back to a raw scroll offset."""
        t = max(0.0, min(1.0, float(t)))
        return float(min_scroll) + t * (float(max_scroll) - float(min_scroll))

    @staticmethod
    def thumb_position(
        t: float,
        track_origin: float,
        track_length: float,
        thumb_length: float,
    ) -> float:
        """Compute a scrollbar thumb's coordinate along its track.

        Parameters
        ----------
        t             : Normalised scroll fraction (0 = start, 1 = end).
        track_origin  : Track's x or y screen coordinate.
        track_length  : Total track length in px.
        thumb_length  : Thumb length in px.

        Returns
        -------
        float   Thumb's x or y coordinate.
        """
        travel = max(1.0, float(track_length) - float(thumb_length))
        t = max(0.0, min(1.0, float(t)))
        return float(track_origin) + t * travel

    @staticmethod
    def thumb_fraction(
        thumb_pos: float,
        track_origin: float,
        track_length: float,
        thumb_length: float,
    ) -> float:
        """Inverse of ``thumb_position``: compute the scroll fraction from
        the thumb's current coordinate."""
        travel = max(1.0, float(track_length) - float(thumb_length))
        return max(0.0, min(1.0,
            (float(thumb_pos) - float(track_origin)) / travel
        ))

    # ── alignment helpers (delegating to module-level functions) ───────────

    def align_pos(
        self,
        align: str,
        sw: float, sh: float,
        cw: float, ch: float,
        pad: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        window_tag: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Convenience wrapper around ``calculate_alignment_pos``."""
        return calculate_alignment_pos(
            align, sw, sh, cw, ch, pad, offset_x, offset_y, window_tag,
        )

    def align_rect(
        self,
        align: str,
        sw: float, sh: float,
        cw: float, ch: float,
        pad: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        window_tag: Optional[str] = None,
    ) -> Tuple[float, float, float, float]:
        """Convenience wrapper around ``calculate_alignment_rect``."""
        return calculate_alignment_rect(
            align, sw, sh, cw, ch, pad, offset_x, offset_y, window_tag,
        )

    def inset(
        self,
        cw: float, ch: float,
        pad_left: float = 0.0, pad_top: float = 0.0,
        pad_right: float = 0.0, pad_bottom: float = 0.0,
        offset_x: float = 0.0, offset_y: float = 0.0,
        min_w: float = 80.0, min_h: float = 80.0,
    ) -> Tuple[float, float, float, float]:
        """Convenience wrapper around ``calculate_inset_rect``."""
        return calculate_inset_rect(
            cw, ch, pad_left, pad_top, pad_right, pad_bottom,
            offset_x, offset_y, min_w, min_h,
        )

    # ── distance / geometry ───────────────────────────────────────────────

    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Euclidean distance between two points."""
        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def midpoint(
        x1: float, y1: float, x2: float, y2: float,
    ) -> Tuple[float, float]:
        """Midpoint between two points."""
        return ((float(x1) + float(x2)) / 2.0,
                (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def rect_contains(
        rx: float, ry: float, rw: float, rh: float,
        px: float, py: float,
    ) -> bool:
        """True if point (px, py) is inside rect (rx, ry, rw, rh)."""
        return (float(rx) <= float(px) <= float(rx) + float(rw)
                and float(ry) <= float(py) <= float(ry) + float(rh))

    @staticmethod
    def rects_overlap(
        ax: float, ay: float, aw: float, ah: float,
        bx: float, by: float, bw: float, bh: float,
    ) -> bool:
        """True if two axis-aligned rectangles overlap."""
        return (float(ax) < float(bx) + float(bw)
                and float(ax) + float(aw) > float(bx)
                and float(ay) < float(by) + float(bh)
                and float(ay) + float(ah) > float(by))

    @staticmethod
    def snap_to_grid(value: float, grid: float) -> float:
        """Snap *value* to the nearest multiple of *grid*."""
        if grid <= 0:
            return float(value)
        return round(float(value) / float(grid)) * float(grid)


# Module-level singleton — import as ``from Draw._align import calc``
calc = AlignCalc()
