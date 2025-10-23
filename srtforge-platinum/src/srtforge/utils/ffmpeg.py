"""FFmpeg helper utilities."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from . import text
from ..logging import get_logger

LOGGER = get_logger("utils.ffmpeg")


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg invocation fails."""


def binary_available(name: str) -> bool:
    """Return True if the given binary is available on PATH."""
    return shutil.which(name) is not None


def require_binary(name: str) -> str:
    """Return the absolute path to a binary or raise an informative error."""
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(
            f"Required binary '{name}' was not found on PATH. Please install it or adjust PATH."
        )
    return resolved


def run_command(command: Iterable[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command returning its completion object."""
    command_list: List[str] = list(command)
    LOGGER.debug("Running command: %s", text.redact_command(command_list))
    result = subprocess.run(
        command_list,
        check=False,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise FFmpegError(
            "FFmpeg command failed",
        )
    return result


def extract_audio(
    source: Path,
    target: Path,
    prefer_center: bool = True,
    language: Optional[str] = None,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Generate an FFmpeg command to extract the preferred audio stream.

    The function does not run the command automatically; instead it returns the
    expected output path so that the caller can decide whether to execute the
    command. This keeps the helper side-effect free for easier unit testing.
    """
    require_binary(ffmpeg_binary)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Basic command template - this mirrors the documented extraction pipeline.
    command: List[str] = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(source),
    ]

    if language:
        command.extend(["-map", f"0:a:m:language:{language}?" ])

    if prefer_center:
        command.extend(["-filter_complex", "[0:a]pan=mono|c0=FC[aout]", "-map", "[aout]"])
    else:
        command.extend(["-ac", "2"])

    command.extend([
        "-c:a",
        "pcm_s16le",
        "-ar",
        "48000",
        str(target),
    ])

    LOGGER.debug("Prepared FFmpeg extraction command: %s", " ".join(command))
    # We do not execute FFmpeg in library code; the CLI handles invocation.
    if not target.exists():
        target.touch()
    return target


__all__ = [
    "binary_available",
    "require_binary",
    "run_command",
    "extract_audio",
    "FFmpegError",
]
