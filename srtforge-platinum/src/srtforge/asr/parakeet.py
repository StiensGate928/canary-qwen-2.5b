"""Parakeet ASR integration stubs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..config import PipelineConfig
from ..logging import get_logger

LOGGER = get_logger("asr.parakeet")


@dataclass(slots=True)
class ParakeetWord:
    word: str
    start: float
    end: float
    confidence: float = 0.9


@dataclass(slots=True)
class ParakeetChunkResult:
    chunk_index: int
    words: List[ParakeetWord]


class ParakeetASR:
    """Simulated Parakeet inference returning deterministic word timings."""

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def transcribe_chunks(self, chunks: Iterable[tuple[int, float, float]]) -> List[ParakeetChunkResult]:
        results: List[ParakeetChunkResult] = []
        for index, start, end in chunks:
            duration = max(0.01, end - start)
            word_duration = duration / 3
            words = [
                ParakeetWord(word="chunk", start=start, end=start + word_duration, confidence=0.8),
                ParakeetWord(
                    word=str(index),
                    start=start + word_duration,
                    end=start + 2 * word_duration,
                    confidence=0.85,
                ),
                ParakeetWord(word="transcript", start=start + 2 * word_duration, end=end, confidence=0.9),
            ]
            results.append(ParakeetChunkResult(index=index, words=words))
        return results


__all__ = ["ParakeetASR", "ParakeetChunkResult", "ParakeetWord"]
