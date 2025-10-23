"""Tests for CTM conversion helpers."""

from __future__ import annotations

from srtforge.asr.ctm import ctm_from_uniform_words, ctm_from_word_times


def test_uniform_words_distribute_evenly() -> None:
    entries = ctm_from_uniform_words(["hello", "world"], 0.0, 4.0, "utt")
    assert len(entries) == 2
    first = entries[0].split()
    second = entries[1].split()
    assert first[0] == second[0] == "utt"
    assert float(first[2]) == 0.0
    assert float(first[3]) == float(second[3]) == 2.0


def test_word_times_preserve_offsets() -> None:
    words = [("hi", 0.1, 0.4), ("there", 0.4, 0.9)]
    entries = ctm_from_word_times(words, 1.0, "utt")
    assert entries[0].split()[4] == "hi"
    assert entries[1].split()[4] == "there"
    assert entries[0].split()[2] == "1.100"
    assert entries[1].split()[2] == "1.400"
