"""Montreal Forced Aligner integration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..logging import get_logger

LOGGER = get_logger("align.mfa")


class MFAUnavailable(RuntimeError):
    """Raised when MFA cannot be executed."""


def align_segments(
    audio_path: Path,
    transcript_lines: Iterable[str],
    dictionary_path: Path | None = None,
    acoustic_model: Path | None = None,
) -> List[str]:
    """Placeholder MFA alignment returning the input transcript.

    Users can plug in MFA via the documented CLI; the Python hook simply logs
    the invocation for reproducibility.
    """
    LOGGER.warning(
        "MFA integration is a placeholder. Configure the CLI to run MFA as an external step."
    )
    return list(transcript_lines)


__all__ = ["align_segments", "MFAUnavailable"]
