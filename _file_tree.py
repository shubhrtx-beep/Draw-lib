"""
Draw._file_tree  v1
====================
A genuine native QTreeWidget-backed file/folder tree, embedded inside an
existing Draw window exactly the way Draw.panel embeds a floating frame —
but here the content IS the real Qt widget, not a QPainter-drawn surface.

Why this module exists
-----------------------
Draw.list / Draw.widget grids are hand-painted (see _list.py, _widget.py):
rows and cells are QPainterPath rectangles + drawText calls with no native
scrolling, selection model, or expand/collapse machinery behind them.
That's fine for dashboards, but it can never behave like a real file
explorer. QTreeWidget already solves indentation, lazy children, keyboard
nav (arrows/Home/End/type-ahead), multi-select, and native scrollbars for
free — so this module wraps it directly instead of reimplementing it.

PUBLIC API
----------
    Draw.filetree(
        ip           = "explorer",       # unique id (REQUIRED)
        display      = "main",           # parent window tag
        root_path    = "C:/Users/me",    # real folder to scan  -- OR --
        data         = {...},            # nested dict tree (no filesystem)
        x = 20, y = 20, width = 260, height = 400,
        show_header  = True,
        header_label = "Name",
        show_files   = True,             # include files, not just folders
        file_filter  = None,             # e.g. [".py", ".txt"]
        lazy         = True,             # populate a folder's children only
                                          # when the user expands it
        alternating_rows = True,
        style        = {...},            # background_color / text_color /
                                          # border_color / font_size
        on_select       = None,           # callback(path_or_key)
        on_double_click = None,           # callback(path_or_key)
        on_expand       = None,           # callback(path_or_key)
    )

    Draw.filetree.refresh(ip)                 # re-scan root_path from disk
    Draw.filetree.expand_all(ip)
    Draw.filetree.collapse_all(ip)
    Draw.filetree.selected(ip)   -> str | None
    Draw.filetree.get(ip)        -> FileTreeDef
    Draw.filetree.list()         -> [ip, ...]
    Draw.filetree.move(ip, x=, y=)
    Draw.filetree.resize(ip, width=, height=)
    Draw.filetree.show(ip) / .hide(ip) / .close(ip)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# pyrefly: ignore [missing-import]
from PySide6.QtCore import Qt
# pyrefly: ignore [missing-import]
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QAbstractItemView

from Draw._window import window as _window_registry

# ── internal markers ──────────────────────────────────────────────────────────

_LAZY_ROLE = Qt.ItemDataRole.UserRole + 1      # True on folders not yet scanned
_PATH_ROLE = Qt.ItemDataRole.UserRole          # full path (fs mode) or dict-key path
_ISDIR_ROLE = Qt.ItemDataRole.UserRole + 2


def _qss_from_style(style: Optional[Dict[str, Any]], ip: Optional[str] = None) -> str:
    """Build a minimal Qt stylesheet string from a Draw-style dict."""
    if not style:
        return ""
    bg     = style.get("background_color")
    fg     = style.get("text_color") or style.get("color")
    border = style.get("border_color")
    bw     = style.get("border_width", 1 if border else 0)
    br     = style.get("border_radius", 0)
    fsize  = style.get("font_size")
    sel_bg = style.get("selection_color", "#3d7aed")

    target = f"QTreeWidget#{ip}" if ip else "QTreeWidget"
    rules = []
    base = []
    if bg:
        base.append(f"background-color: {bg};")
    if fg:
        base.append(f"color: {fg};")
    if border:
        base.append(f"border: {bw}px solid {border};")
    if br:
        base.append(f"border-radius: {br}px;")
    if fsize:
        base.append(f"font-size: {fsize}px;")
    if base:
        rules.append(f"{target} {{ " + " ".join(base) + " }}")
    rules.append(
        f"{target}::item:selected {{ background-color: " + str(sel_bg) + "; }}"
    )
    return "\n".join(rules)


def _scan_dir_entries(path: str, file_filter: Optional[List[str]], show_files: bool):
    """Return (dirs, files) entry lists for *path*, dirs first, sorted, best-effort."""
    dirs, files = [], []
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                    elif show_files:
                        if file_filter:
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext not in file_filter:
                                continue
                        files.append(entry.name)
                except OSError:
                    continue
    except OSError:
        pass
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    return dirs, files


# ── FileTreeDef ────────────────────────────────────────────────────────────────

@dataclass
class FileTreeDef:
    ip: str
    display: str
    root_path: Optional[str]
    data: Optional[Dict[str, Any]]
    x: int
    y: int
    width: int
    height: int
    show_header: bool
    header_label: str
    show_files: bool
    file_filter: Optional[List[str]]
    lazy: bool
    alternating_rows: bool
    style: Dict[str, Any]
    on_select: Optional[Callable[[str], None]]
    on_double_click: Optional[Callable[[str], None]]
    on_expand: Optional[Callable[[str], None]]

    # runtime
    _widget: Optional["QTreeWidget"] = field(default=None, init=False)
    motion: List[Any] = field(default_factory=list, init=False)

    def apply_motion_state(self, state: Dict[str, Any]) -> None:
        if not self._widget:
            return
        if "x" in state or "y" in state:
            nx = int(round(float(state.get("x", self.x))))
            ny = int(round(float(state.get("y", self.y))))
            self.x, self.y = nx, ny
            self._widget.move(nx, ny)
        if "width" in state or "height" in state:
            nw = int(round(float(state.get("width", self.width))))
            nh = int(round(float(state.get("height", self.height))))
            self.width, self.height = nw, nh
            self._widget.resize(nw, nh)


# ── registry ────────────────────────────────────────────────────────────────────

class _FileTreeRegistry:
    """Public API: Draw.filetree(ip="...", display="main", root_path=... , ...)"""

    def __init__(self):
        self._trees: Dict[str, FileTreeDef] = {}

    # -- construction ---------------------------------------------------------

    def __call__(
        self,
        *,
        ip: str,
        display: Optional[str] = None,
        root_path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        x: int = 20,
        y: int = 20,
        width: int = 260,
        height: int = 400,
        show_header: bool = True,
        header_label: str = "Name",
        show_files: bool = True,
        file_filter: Optional[List[str]] = None,
        lazy: bool = True,
        alternating_rows: bool = True,
        style: Optional[Dict[str, Any]] = None,
        on_select: Optional[Callable[[str], None]] = None,
        on_double_click: Optional[Callable[[str], None]] = None,
        on_expand: Optional[Callable[[str], None]] = None,
    ) -> FileTreeDef:
        if not ip or not isinstance(ip, str):
            raise ValueError("Draw.filetree: 'ip' is required.")
        if ip in self._trees:
            return self._trees[ip]
        if root_path is None and data is None:
            raise ValueError("Draw.filetree: pass either 'root_path' or 'data'.")

        window_tag = display
        if window_tag is None:
            tags = _window_registry.list_tags()
            if len(tags) == 1:
                window_tag = tags[0]
            elif len(tags) > 1:
                raise ValueError("Draw.filetree: multiple windows — 'display' is required.")
            else:
                raise ValueError("Draw.filetree: no windows exist. Call Draw.window() first.")

        win = _window_registry.get(window_tag)
        from Draw._text import _get_or_create_canvas
        canvas = _get_or_create_canvas(window_tag, win)

        tdef = FileTreeDef(
            ip=ip, display=window_tag, root_path=root_path, data=data,
            x=x, y=y, width=width, height=height,
            show_header=show_header, header_label=header_label,
            show_files=show_files, file_filter=(
                [f.lower() for f in file_filter] if file_filter else None
            ),
            lazy=lazy, alternating_rows=alternating_rows,
            style=dict(style or {}),
            on_select=on_select, on_double_click=on_double_click, on_expand=on_expand,
        )

        widget = QTreeWidget(canvas)
        widget.setObjectName(ip)
        widget.setGeometry(x, y, width, height)
        widget.setColumnCount(1)
        widget.setHeaderLabels([header_label])
        widget.setHeaderHidden(not show_header)
        widget.setAlternatingRowColors(alternating_rows)
        widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        widget.setUniformRowHeights(True)
        widget.setAnimated(True)
        qss = _qss_from_style(style, ip)
        if qss:
            widget.setStyleSheet(qss)

        tdef._widget = widget
        self._trees[ip] = tdef

        if root_path is not None:
            self._populate_fs_root(tdef)
        else:
            self._populate_data_root(tdef)

        widget.itemExpanded.connect(lambda item: self._on_item_expanded(tdef, item))
        widget.itemSelectionChanged.connect(lambda: self._on_selection_changed(tdef))
        widget.itemDoubleClicked.connect(
            lambda item, _col: self._on_double_clicked(tdef, item)
        )

        widget.show()
        widget.raise_()
        return tdef

    # -- filesystem population -------------------------------------------------

    def _make_fs_item(self, parent, name: str, path: str, is_dir: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [name])
        item.setData(0, _PATH_ROLE, path)
        item.setData(0, _ISDIR_ROLE, is_dir)
        return item

    def _populate_fs_root(self, tdef: FileTreeDef) -> None:
        widget = tdef._widget
        widget.clear()
        root = tdef.root_path
        if not root or not os.path.isdir(root):
            placeholder = QTreeWidgetItem(widget, [f"(not found: {root})"])
            placeholder.setDisabled(True)
            return
        root_item = self._make_fs_item(widget, os.path.basename(root.rstrip("/\\")) or root, root, True)
        self._fill_fs_children(tdef, root_item, root)
        root_item.setExpanded(True)

    def _fill_fs_children(self, tdef: FileTreeDef, parent_item: QTreeWidgetItem, path: str) -> None:
        dirs, files = _scan_dir_entries(path, tdef.file_filter, tdef.show_files)
        for name in dirs:
            child_path = os.path.join(path, name)
            child_item = self._make_fs_item(parent_item, name, child_path, True)
            if tdef.lazy:
                dummy = QTreeWidgetItem(child_item, ["…"])
                dummy.setData(0, _LAZY_ROLE, True)
            else:
                self._fill_fs_children(tdef, child_item, child_path)
        for name in files:
            self._make_fs_item(parent_item, name, os.path.join(path, name), False)

    def _on_item_expanded(self, tdef: FileTreeDef, item: QTreeWidgetItem) -> None:
        if tdef.on_expand:
            path = item.data(0, _PATH_ROLE)
            if path is not None:
                tdef.on_expand(path)
        if not tdef.lazy or tdef.root_path is None:
            return
        if item.childCount() == 1 and item.child(0).data(0, _LAZY_ROLE):
            item.takeChildren()
            path = item.data(0, _PATH_ROLE)
            if path:
                self._fill_fs_children(tdef, item, path)

    # -- nested-dict population -------------------------------------------------

    def _populate_data_root(self, tdef: FileTreeDef) -> None:
        widget = tdef._widget
        widget.clear()
        self._fill_data_children(widget, tdef.data or {}, "")

    def _fill_data_children(self, parent, node: Dict[str, Any], key_path: str) -> None:
        for key, value in node.items():
            full_key = f"{key_path}/{key}" if key_path else key
            if isinstance(value, dict):
                item = QTreeWidgetItem(parent, [str(key)])
                item.setData(0, _PATH_ROLE, full_key)
                item.setData(0, _ISDIR_ROLE, True)
                self._fill_data_children(item, value, full_key)
            else:
                label = str(key) if value in (None, "") else f"{key}"
                item = QTreeWidgetItem(parent, [label])
                item.setData(0, _PATH_ROLE, full_key)
                item.setData(0, _ISDIR_ROLE, False)

    # -- selection / callbacks ---------------------------------------------------

    def _on_selection_changed(self, tdef: FileTreeDef) -> None:
        if not tdef.on_select:
            return
        items = tdef._widget.selectedItems()
        if items:
            path = items[0].data(0, _PATH_ROLE)
            if path is not None:
                tdef.on_select(path)

    def _on_double_clicked(self, tdef: FileTreeDef, item: QTreeWidgetItem) -> None:
        if tdef.on_double_click:
            path = item.data(0, _PATH_ROLE)
            if path is not None:
                tdef.on_double_click(path)

    # -- control API --------------------------------------------------------------

    def refresh(self, ip: str) -> None:
        tdef = self._trees.get(ip)
        if not tdef:
            return
        if tdef.root_path is not None:
            self._populate_fs_root(tdef)
        else:
            self._populate_data_root(tdef)

    def expand_all(self, ip: str) -> None:
        tdef = self._trees.get(ip)
        if tdef and tdef._widget:
            tdef._widget.expandAll()

    def collapse_all(self, ip: str) -> None:
        tdef = self._trees.get(ip)
        if tdef and tdef._widget:
            tdef._widget.collapseAll()

    def selected(self, ip: str) -> Optional[str]:
        tdef = self._trees.get(ip)
        if not tdef or not tdef._widget:
            return None
        items = tdef._widget.selectedItems()
        return items[0].data(0, _PATH_ROLE) if items else None

    def get(self, ip: str) -> Optional[FileTreeDef]:
        return self._trees.get(ip)

    def list(self) -> List[str]:
        return list(self._trees.keys())

    def move(self, ip: str, *, x: int, y: int) -> None:
        tdef = self._trees.get(ip)
        if tdef and tdef._widget:
            tdef.x, tdef.y = x, y
            tdef._widget.move(x, y)

    def resize(self, ip: str, *, width: int, height: int) -> None:
        tdef = self._trees.get(ip)
        if tdef and tdef._widget:
            tdef.width, tdef.height = width, height
            tdef._widget.resize(width, height)

    def show(self, ip: str) -> None:
        tdef = self._trees.get(ip)
        if tdef and tdef._widget:
            tdef._widget.show()

    def hide(self, ip: str) -> None:
        tdef = self._trees.get(ip)
        if tdef and tdef._widget:
            tdef._widget.hide()

    def close(self, ip: str) -> None:
        tdef = self._trees.pop(ip, None)
        if tdef and tdef._widget:
            tdef._widget.deleteLater()


# ── singleton ─────────────────────────────────────────────────────────────────
filetree = _FileTreeRegistry()
