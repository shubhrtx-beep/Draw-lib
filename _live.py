"""
Draw._live
Helpers for dynamic text (`Draw.live.text(...)`) and input markers (`Draw.input.text`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LiveRef:
    key: str


@dataclass(frozen=True)
class LiveTextBinding:
    source: Any
    fallback: str = ""


@dataclass(frozen=True)
class InputTextMarker:
    initial: str = ""


class _LiveRegistry:
    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def text(self, source: Any = None, *, fallback: object = "") -> LiveTextBinding:
        fallback_text = "" if fallback is None else str(fallback)
        return LiveTextBinding(source=source, fallback=fallback_text)

    def ref(self, key: object) -> LiveRef:
        return LiveRef(str(key))

    def set(self, key: object, value: Any) -> Any:
        self._values[str(key)] = value
        return value

    def get(self, key: object, default: Any = None) -> Any:
        return self._values.get(str(key), default)

    def clear(self, key: object | None = None) -> None:
        if key is None:
            self._values.clear()
        else:
            self._values.pop(str(key), None)

    def __getitem__(self, key: object) -> Any:
        return self._values[str(key)]

    def __setitem__(self, key: object, value: Any) -> None:
        self._values[str(key)] = value


class _InputRegistry:
    def __init__(self) -> None:
        self._marker = InputTextMarker("")
        self._values: dict[str, str] = {}
        self._last_value: str = ""

    @property
    def text(self) -> InputTextMarker:
        return self._marker

    def field(self, initial: object = "") -> InputTextMarker:
        initial_text = "" if initial is None else str(initial)
        return InputTextMarker(initial=initial_text)

    def set(self, key: object | None, value: object) -> str:
        text = "" if value is None else str(value)
        self._last_value = text
        if key is not None:
            self._values[str(key)] = text
        return text

    def get(self, key: object | None = None, default: object = "") -> str:
        if key is None:
            if self._last_value != "":
                return self._last_value
            return "" if default is None else str(default)
        return self._values.get(str(key), "" if default is None else str(default))

    def clear(self, key: object | None = None) -> None:
        if key is None:
            self._values.clear()
            self._last_value = ""
        else:
            self._values.pop(str(key), None)


def is_live_text_binding(value: object) -> bool:
    return isinstance(value, LiveTextBinding)


def is_input_text_marker(value: object) -> bool:
    return isinstance(value, InputTextMarker)


def resolve_live_text(binding: LiveTextBinding) -> str:
    value: Any
    source = binding.source
    if isinstance(source, LiveRef):
        value = live.get(source.key, binding.fallback)
    elif callable(source):
        value = source()
    else:
        value = source

    if value is None:
        value = binding.fallback
    return "" if value is None else str(value)


live = _LiveRegistry()
input = _InputRegistry()

