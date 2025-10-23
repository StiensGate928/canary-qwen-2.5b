"""Subtitle readability helpers."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(slots=True)
class ReadingConstraints:
    max_chars_per_line: int
    max_lines: int
    min_duration_s: float


def wrap_text(text: str, constraints: ReadingConstraints) -> List[str]:
    """Wrap text into lines that respect readability limits.

    The implementation uses :class:`textwrap.TextWrapper` with a hard line
    count limit. When text cannot fully fit within the specified number of
    lines, the final line is truncated with an ellipsis character (\u2026).
    """

    if not text.strip():
        return []

    wrapper = textwrap.TextWrapper(
        width=constraints.max_chars_per_line,
        max_lines=constraints.max_lines,
        placeholder="…",
        break_long_words=False,
        break_on_hyphens=False,
    )
    lines = wrapper.wrap(text)

    # Handle exceptionally long tokens that exceed the width by slicing them
    # into chunks to avoid empty output.
    if not lines:
        chunks: List[str] = []
        for idx in range(0, len(text), constraints.max_chars_per_line):
            chunks.append(text[idx : idx + constraints.max_chars_per_line])
        return chunks[: constraints.max_lines]

    return lines


def reading_speed_ok(text: str, duration: float, constraints: ReadingConstraints) -> bool:
    """Return True if the subtitle respects a basic reading speed heuristic."""
    if duration <= 0:
        return False
    cps = len(text.replace(" ", "")) / duration
    # Accept up to ~21 CPS which is common for anime subtitles.
    return cps <= 21.0


def format_lines(lines: Iterable[str]) -> str:
    return "\n".join(lines)


__all__ = ["ReadingConstraints", "wrap_text", "reading_speed_ok", "format_lines"]
