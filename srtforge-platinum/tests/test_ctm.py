"""Tests for CTM conversion helpers."""

from __future__ import annotations

from srtforge.asr import ctm
from srtforge.asr.canary_salm import CanaryChunkResult
from srtforge.asr.parakeet import ParakeetChunkResult, ParakeetWord


def test_uniform_text_allocates_evenly() -> None:
    chunk = CanaryChunkResult(chunk_index=0, start=0.0, end=4.0, text="hello world test")
    entries = ctm.from_uniform_text(chunk)
    assert len(entries) == 3
    assert entries[0].duration == entries[1].duration == entries[2].duration


def test_merge_chunk_ctms_sorts_entries() -> None:
    parakeet = ParakeetChunkResult(
        chunk_index=0,
        words=[
            ParakeetWord(word="alpha", start=1.0, end=2.0, confidence=0.9),
            ParakeetWord(word="beta", start=0.0, end=0.5, confidence=0.8),
        ],
    )
    merged = ctm.merge_chunk_ctms([ctm.from_parakeet(parakeet)])
    assert [entry.word for entry in merged] == ["beta", "alpha"]
