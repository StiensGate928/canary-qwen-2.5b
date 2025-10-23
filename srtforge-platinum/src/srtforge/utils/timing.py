"""Time conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(slots=True)
class TimeInterval:
    """Represents a span of audio in seconds."""

    start: float
    end: float

    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def to_srt_timestamp(value: float) -> str:
    """Convert seconds to an SRT timestamp string."""
    if value < 0:
        value = 0.0
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = int(value % 60)
    milliseconds = int(round((value - int(value)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def ensure_min_duration(intervals: Iterable[TimeInterval], minimum: float) -> List[TimeInterval]:
    """Ensure each interval is at least ``minimum`` seconds long."""
    adjusted: List[TimeInterval] = []
    for interval in intervals:
        if interval.duration() < minimum:
            padding = (minimum - interval.duration()) / 2
            adjusted.append(TimeInterval(interval.start - padding, interval.end + padding))
        else:
            adjusted.append(interval)
    return adjusted


__all__ = ["TimeInterval", "to_srt_timestamp", "ensure_min_duration"]
