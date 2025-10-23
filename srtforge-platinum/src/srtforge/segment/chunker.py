"""Chunking strategy for long-form transcription."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .vad import SpeechInterval


@dataclass(slots=True)
class Chunk:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class Chunker:
    """Generate overlapping windows suitable for ASR models."""

    def __init__(self, max_len: float, overlap: float) -> None:
        if overlap >= max_len:
            raise ValueError("Chunk overlap must be smaller than maximum length")
        self._max_len = max_len
        self._overlap = overlap

    def chunk(self, spans: Sequence[SpeechInterval]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for span in spans:
            start = span.start
            while start < span.end:
                chunk_end = min(span.end, start + self._max_len)
                chunks.append(Chunk(start, chunk_end))
                if chunk_end >= span.end:
                    break
                start = max(start + 1e-6, chunk_end - self._overlap)
        return chunks

    def chunk_from_pairs(self, spans: Iterable[tuple[float, float]]) -> List[Chunk]:
        intervals = [SpeechInterval(start, end) for start, end in spans]
        return self.chunk(intervals)


__all__ = ["Chunk", "Chunker"]
