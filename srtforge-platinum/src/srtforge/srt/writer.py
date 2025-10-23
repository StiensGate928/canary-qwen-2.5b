"""SRT writer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from ..config import ReadingConfig
from ..utils.timing import TimeInterval, ensure_min_duration, to_srt_timestamp
from .readability import ReadingConstraints, format_lines, wrap_text


@dataclass(slots=True)
class SubtitleSegment:
    index: int
    start: float
    end: float
    text: str

    @property
    def interval(self) -> TimeInterval:
        return TimeInterval(self.start, self.end)


class SRTWriter:
    """Render subtitle segments to disk."""

    def __init__(self, config: ReadingConfig) -> None:
        self._constraints = ReadingConstraints(
            max_chars_per_line=config.max_chars_per_line,
            max_lines=config.max_lines,
            min_duration_s=config.min_duration_s,
        )

    def _apply_readability(self, segment: SubtitleSegment) -> SubtitleSegment:
        wrapped = wrap_text(segment.text, self._constraints)
        enforced = ensure_min_duration([segment.interval], self._constraints.min_duration_s)[0]
        return SubtitleSegment(
            index=segment.index,
            start=enforced.start,
            end=enforced.end,
            text=format_lines(wrapped),
        )

    def render(self, segments: Iterable[SubtitleSegment]) -> List[str]:
        rendered: List[str] = []
        for segment in segments:
            adjusted = self._apply_readability(segment)
            rendered.extend(
                [
                    str(adjusted.index),
                    f"{to_srt_timestamp(adjusted.start)} --> {to_srt_timestamp(adjusted.end)}",
                    adjusted.text,
                    "",
                ]
            )
        return rendered

    def write(self, segments: Iterable[SubtitleSegment], target: Path) -> Path:
        lines = self.render(segments)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines).strip() + "\n")
        return target


__all__ = ["SubtitleSegment", "SRTWriter"]
