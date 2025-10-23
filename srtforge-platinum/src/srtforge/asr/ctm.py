"""CTM helpers used by the ROVER combination stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .canary_salm import CanaryChunkResult
from .parakeet import ParakeetChunkResult, ParakeetWord


@dataclass(slots=True)
class CTMEntry:
    """Represents a single CTM word entry."""

    speaker: str
    start: float
    duration: float
    word: str
    confidence: float | None = None


def _safe_duration(start: float, end: float) -> float:
    duration = max(0.0, end - start)
    return duration


def from_parakeet(chunk: ParakeetChunkResult, speaker: str = "P") -> List[CTMEntry]:
    entries: List[CTMEntry] = []
    for word in chunk.words:
        entries.append(
            CTMEntry(
                speaker=speaker,
                start=word.start,
                duration=_safe_duration(word.start, word.end),
                word=word.word,
                confidence=word.confidence,
            )
        )
    return entries


def from_uniform_text(chunk: CanaryChunkResult, speaker: str = "C") -> List[CTMEntry]:
    words = [w for w in chunk.text.strip().split() if w]
    if not words:
        return []
    duration = _safe_duration(chunk.start, chunk.end)
    per_word = duration / len(words)
    entries: List[CTMEntry] = []
    cursor = chunk.start
    for word in words:
        entries.append(
            CTMEntry(
                speaker=speaker,
                start=cursor,
                duration=per_word,
                word=word,
                confidence=None,
            )
        )
        cursor += per_word
    return entries


def merge_chunk_ctms(ctms: Sequence[Iterable[CTMEntry]]) -> List[CTMEntry]:
    merged: List[CTMEntry] = []
    for entries in ctms:
        merged.extend(entries)
    merged.sort(key=lambda entry: entry.start)
    return merged


def text_from_ctm(entries: Iterable[CTMEntry]) -> str:
    return " ".join(entry.word for entry in entries)


__all__ = [
    "CTMEntry",
    "from_parakeet",
    "from_uniform_text",
    "merge_chunk_ctms",
    "text_from_ctm",
]
