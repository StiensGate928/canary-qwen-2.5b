"""Audio preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

from ..config import FrontendConfig
from ..logging import get_logger

LOGGER = get_logger("audio.preprocess")


class AudioPreprocessor:
    """Apply denoising, normalization and loudness alignment."""

    def __init__(self, config: FrontendConfig) -> None:
        self._config = config

    def preprocess(self, audio_path: Path, temp_dir: Path) -> Path:
        """Return a placeholder path for the cleaned audio."""
        LOGGER.info("Pretending to preprocess audio: denoise=%s normalize=%s", self._config.denoise, self._config.normalize)
        target = temp_dir / f"{audio_path.stem}_clean.wav"
        target.touch()
        return target


__all__ = ["AudioPreprocessor"]
