"""Tests for VAD log parsing."""

from __future__ import annotations

from srtforge.segment import vad


def test_parse_silencedetect_extracts_intervals() -> None:
    log_lines = [
        "[silencedetect @ 0x] silence_start: 1.000",
        "[silencedetect @ 0x] silence_end: 2.000 | silence_duration: 1.000",
        "[silencedetect @ 0x] silence_start: 5.500",
        "[silencedetect @ 0x] silence_end: 6.000 | silence_duration: 0.500",
    ]
    silences = vad.parse_silencedetect(log_lines)
    assert len(silences) == 2
    assert silences[0].start == 1.0
    assert silences[0].end == 2.0
    assert silences[1].duration == 0.5


def test_merge_speech_spans_merges_overlap() -> None:
    spans = [(0.0, 1.0), (0.8, 1.5), (2.0, 3.0)]
    merged = vad.merge_speech_spans(spans, pad=0.1)
    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[0].end > 1.4
