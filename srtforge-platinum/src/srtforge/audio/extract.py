"""Audio extraction logic using FFmpeg."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import FrontendConfig
from ..logging import get_logger
from ..utils import ffmpeg

LOGGER = get_logger("audio.extract")


class AudioExtractor:
    """Extract audio streams according to the configured policy."""

    def __init__(self, config: FrontendConfig, ffmpeg_binary: str = "ffmpeg") -> None:
        self._config = config
        self._ffmpeg_binary = ffmpeg_binary

    def extract(self, source: Path, temp_dir: Path, language: Optional[str] = None) -> Path:
        """Prepare the extraction command and create an empty target placeholder."""
        target = temp_dir / f"{source.stem}_extracted.wav"
        LOGGER.info("Preparing audio extraction for %s", source)
        ffmpeg.extract_audio(
            source,
            target,
            prefer_center=self._config.prefer_center,
            language=language,
            ffmpeg_binary=self._ffmpeg_binary,
        )
        return target


__all__ = ["AudioExtractor"]
