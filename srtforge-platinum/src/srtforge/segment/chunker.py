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

    def __init__(self, max_len: float, overlap: float, min_len: float) -> None:
        if max_len <= 0:
            raise ValueError("Maximum chunk length must be positive")
        if min_len <= 0:
            raise ValueError("Minimum chunk length must be positive")
        if min_len > max_len:
            raise ValueError("Minimum chunk length cannot exceed maximum length")
        if overlap >= max_len:
            raise ValueError("Chunk overlap must be smaller than maximum length")
        if overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        self._max_len = max_len
        self._overlap = overlap
        self._min_len = min_len

    def chunk(self, spans: Sequence[SpeechInterval]) -> List[Chunk]:
        chunks: List[Chunk] = []
        if not spans:
            return chunks
        step = max(1e-6, self._max_len - self._overlap)
        for span in spans:
            span_start = span.start
            span_end = span.end
            if span_end - span_start <= 0:
                continue
            span_duration = span_end - span_start
            if span_duration <= self._max_len:
                # Small spans fit in a single chunk; only merge with the
                # previous chunk if it helps maintain minimum duration.
                if (
                    chunks
                    and span_duration < self._min_len
                    and span_end - chunks[-1].start <= self._max_len
                ):
                    previous = chunks.pop()
                    chunks.append(Chunk(previous.start, span_end))
                else:
                    chunks.append(Chunk(span_start, span_end))
                continue

            start = span_start
            while start + self._max_len < span_end:
                chunk_end = start + self._max_len
                chunks.append(Chunk(start, chunk_end))
                start += step

            final_start = max(span_start, span_end - self._max_len)
            if span_end - final_start < self._min_len and span_duration >= self._min_len:
                final_start = max(span_start, span_end - self._min_len)
            chunks.append(Chunk(final_start, span_end))
        return chunks

    def chunk_from_pairs(self, spans: Iterable[tuple[float, float]]) -> List[Chunk]:
        intervals = [SpeechInterval(start, end) for start, end in spans]
        return self.chunk(intervals)


__all__ = ["Chunk", "Chunker"]
