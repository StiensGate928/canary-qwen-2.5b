"""Tests for per-chunk ROVER combination."""

from __future__ import annotations

from typing import List

from srtforge.combine import rover


def test_rover_runs_per_chunk(monkeypatch) -> None:
    canary_ctms: List[List[str]] = [
        ["utt 1 0.000 0.500 hello 1.00"],
        ["utt 1 0.000 0.400 world 1.00"],
        ["utt 1 0.000 0.300 again 1.00"],
    ]
    parakeet_ctms: List[List[str]] = [
        ["utt 1 0.000 0.500 hello 0.90"],
        ["utt 1 0.000 0.400 there 0.90"],
        ["utt 1 0.000 0.300 friend 0.90"],
    ]
    chunk_ranges = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

    def fake_run_rover(a, b, out, method="maxconf"):
        out.write_text(a.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(rover, "_run_rover", fake_run_rover)

    consensus = rover.combine_per_chunk(chunk_ranges, canary_ctms, parakeet_ctms)
    assert len(consensus) == 3
    for chunk in consensus:
        assert chunk[0].split()[2].startswith("0.")
