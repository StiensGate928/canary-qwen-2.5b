"""Audio separation helpers using Demucs."""

from __future__ import annotations

from pathlib import Path

from ..config import FrontendConfig
from ..logging import get_logger

LOGGER = get_logger("audio.separate")


class SeparationUnavailable(RuntimeError):
    """Raised when separation is requested but dependencies are missing."""


class DemucsSeparator:
    """Run Demucs vocal separation as a fallback when center channels are absent."""

    def __init__(self, config: FrontendConfig) -> None:
        self._config = config

    def separate(self, audio_path: Path, temp_dir: Path) -> Path:
        """Pretend to run Demucs and return the expected vocal stem path.

        The project ships with a thin wrapper to keep orchestration manageable
        during development. The actual Demucs command is documented in the
        README and can be executed by advanced users.
        """
        LOGGER.warning(
            "Demucs separation is not executed automatically in the reference implementation." \
            " Please consult the documentation to enable it."
        )
        target = temp_dir / f"{audio_path.stem}_vocals.wav"
        target.touch()
        return target


__all__ = ["DemucsSeparator", "SeparationUnavailable"]
