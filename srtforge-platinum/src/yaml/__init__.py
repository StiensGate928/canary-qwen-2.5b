"""Minimal YAML parser for development environments without PyYAML."""

from __future__ import annotations

from typing import Any, Dict, Tuple

__all__ = ["safe_load"]


def _convert(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def safe_load(stream: Any) -> Dict[str, Any]:
    if hasattr(stream, "read"):
        content = stream.read()
    else:
        content = str(stream)

    root: Dict[str, Any] = {}
    stack: list[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for raw_line in content.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        while stack and indent <= stack[-1][0] and len(stack) > 1:
            stack.pop()
        current = stack[-1][1]
        key, sep, value = raw_line.strip().partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if value == "":
            new_dict: Dict[str, Any] = {}
            current[key] = new_dict
            stack.append((indent, new_dict))
        else:
            current[key] = _convert(value)
    return root
