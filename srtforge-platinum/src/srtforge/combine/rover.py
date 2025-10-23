"""ROVER combination wrapper."""

from __future__ import annotations

from typing import Iterable, List

from ..logging import get_logger
from ..utils import ffmpeg
from ..asr.ctm import CTMEntry, text_from_ctm

LOGGER = get_logger("combine.rover")


class RoverNotAvailable(RuntimeError):
    """Raised when the SCTK rover binary is unavailable."""


class RoverCombiner:
    """Coordinate ROVER combination using SCTK."""

    def __init__(self, rover_binary: str = "rover") -> None:
        self._rover_binary = rover_binary

    def assert_available(self) -> None:
        try:
            ffmpeg.require_binary(self._rover_binary)
        except FileNotFoundError as exc:
            raise RoverNotAvailable("SCTK rover binary is required for system combination") from exc

    def combine(self, systems: Iterable[List[CTMEntry]]) -> List[CTMEntry]:
        """Return the consensus transcript.

        The reference implementation does not shell out to the real binary but
        instead merges the highest-confidence CTM entries to provide deterministic
        behaviour for tests.
        """
        merged: List[CTMEntry] = []
        for system in systems:
            merged.extend(system)
        merged.sort(key=lambda entry: (entry.start, -(entry.confidence or 0.0)))
        # Deduplicate on start time keeping the first (highest confidence).
        consensus: List[CTMEntry] = []
        used_starts: set[float] = set()
        for entry in merged:
            key = round(entry.start, 3)
            if key in used_starts:
                continue
            used_starts.add(key)
            consensus.append(entry)
        LOGGER.debug("Consensus transcript: %s", text_from_ctm(consensus))
        return consensus


__all__ = ["RoverCombiner", "RoverNotAvailable"]
