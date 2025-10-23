"""CTM helpers for chunk-level processing."""

from __future__ import annotations

from typing import List, Tuple


def ctm_from_uniform_words(words: List[str], chunk_start: float, chunk_end: float, utt_id: str) -> List[str]:
    """Generate a CTM by uniformly distributing ``words`` across the chunk."""

    duration = max(0.0, chunk_end - chunk_start)
    count = max(1, len(words))
    step = duration / count if count else duration
    entries: List[str] = []
    for index, word in enumerate(words):
        start = chunk_start + index * step
        entries.append(
            f"{utt_id} 1 {start:.3f} {step:.3f} {word} 1.00"
        )
    return entries


def ctm_from_word_times(
    words_ts: List[Tuple[str, float, float]],
    chunk_offset: float,
    utt_id: str,
) -> List[str]:
    """Convert model-provided word timestamps to CTM entries."""

    entries: List[str] = []
    for word, start, end in words_ts:
        duration = max(0.0, end - start)
        entries.append(
            f"{utt_id} 1 {start + chunk_offset:.3f} {duration:.3f} {word} 1.00"
        )
    return entries


__all__ = ["ctm_from_uniform_words", "ctm_from_word_times"]
