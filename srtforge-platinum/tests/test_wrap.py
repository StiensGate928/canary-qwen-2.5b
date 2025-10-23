"""Tests for subtitle line wrapping."""

from __future__ import annotations

from srtforge.srt.readability import ReadingConstraints, wrap_text


def test_wrap_text_respects_line_length() -> None:
    constraints = ReadingConstraints(max_chars_per_line=10, max_lines=2, min_duration_s=1.0)
    text = "This is a small wrapping test"
    lines = wrap_text(text, constraints)
    assert all(len(line) <= 10 for line in lines)
    assert len(lines) <= 2


def test_wrap_text_compacts_lines_when_needed() -> None:
    constraints = ReadingConstraints(max_chars_per_line=6, max_lines=2, min_duration_s=1.0)
    text = "one two three four"
    lines = wrap_text(text, constraints)
    assert len(lines) == 2
