"""Voice activity detection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from ..utils.timing import TimeInterval

_SILENCE_START_RE = re.compile(r"silence_start: (?P<start>[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end: (?P<end>[0-9.]+) \| silence_duration: (?P<duration>[0-9.]+)")


@dataclass(slots=True)
class SilenceInterval:
    """Represents a silence span detected by FFmpeg."""

    start: float
    end: float
    duration: float


@dataclass(slots=True)
class SpeechInterval(TimeInterval):
    """Represents a speech span."""

    pass


def parse_silencedetect(log_lines: Iterable[str]) -> List[SilenceInterval]:
    """Parse FFmpeg silencedetect output into silence intervals."""
    silences: List[SilenceInterval] = []
    current_start: float | None = None
    for raw_line in log_lines:
        line = raw_line.strip()
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            current_start = float(start_match.group("start"))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match and current_start is not None:
            end = float(end_match.group("end"))
            duration = float(end_match.group("duration"))
            silences.append(SilenceInterval(current_start, end, duration))
            current_start = None
    return silences


def invert_silences(silences: Sequence[SilenceInterval], total_duration: float) -> List[SpeechInterval]:
    """Convert silence intervals into speech intervals across the file."""
    speech: List[SpeechInterval] = []
    cursor = 0.0
    for silence in silences:
        if silence.start > cursor:
            speech.append(SpeechInterval(cursor, silence.start))
        cursor = max(cursor, silence.end)
    if cursor < total_duration:
        speech.append(SpeechInterval(cursor, total_duration))
    return speech


def merge_speech_spans(
    spans: Iterable[Tuple[float, float]],
    pad: float = 0.0,
    merge_gap: float = 0.0,
) -> List[SpeechInterval]:
    """Merge overlapping spans applying a symmetric pad and gap tolerance."""
    padded = [SpeechInterval(start=max(0.0, s - pad), end=e + pad) for s, e in spans]
    padded.sort(key=lambda interval: interval.start)
    merged: List[SpeechInterval] = []
    for interval in padded:
        if not merged:
            merged.append(interval)
            continue
        last = merged[-1]
        if interval.start <= last.end + merge_gap:
            merged[-1] = SpeechInterval(last.start, max(last.end, interval.end))
        else:
            merged.append(interval)
    return merged


__all__ = [
    "SilenceInterval",
    "SpeechInterval",
    "parse_silencedetect",
    "invert_silences",
    "merge_speech_spans",
]
