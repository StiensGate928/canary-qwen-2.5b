"""FFmpeg helper utilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List

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


def have_binary(name: str, fatal: bool = False) -> bool:
    """Return True if a binary is present; optionally raise when missing."""

    try:
        require_binary(name)
    except FileNotFoundError:
        if fatal:
            raise
        return False
    return True


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


def run_ffmpeg(arguments: Iterable[str], ffmpeg_binary: str = "ffmpeg") -> subprocess.CompletedProcess[str]:
    """Execute FFmpeg with the provided arguments."""

    require_binary(ffmpeg_binary)
    command: List[str] = [ffmpeg_binary, *list(arguments)]
    return run_command(command, check=True)


def _probe_audio_stream(path: Path, stream_index: int = 0) -> dict:
    """Return metadata for a specific audio stream using ``ffprobe``."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        f"a:{stream_index}",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    out = subprocess.run(cmd, check=True, text=True, capture_output=True).stdout
    data = json.loads(out or "{}")
    streams = data.get("streams") or [{}]
    return streams[0]


def probe_audio_duration(path: Path) -> float:
    """Return the duration (seconds) of ``path`` using ffprobe."""

    info = _probe_audio_stream(path)
    try:
        return float(info.get("duration", 0.0))
    except (TypeError, ValueError):
        return 0.0


def extract_dialog_source(
    input_video: Path,
    out_wav: Path,
    stream_index: int = 0,
    prefer_center: bool = True,
) -> None:
    """Extract audio preferring the center channel when available."""

    info = _probe_audio_stream(input_video, stream_index)
    channels = int(info.get("channels", 0) or 0)
    layout = (info.get("channel_layout") or info.get("ch_layout") or "").lower()
    has_fc = any(token in layout for token in {"5.1", "6.1", "7.1", "fc"}) or layout == "c"
    if prefer_center and channels >= 3 and has_fc:
        af = "pan=mono|c0=FC,aresample=44100:resampler=soxr:precision=28"
        ac = "1"
    else:
        af = "aresample=44100:resampler=soxr:precision=28"
        ac = "2"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(input_video),
        "-map",
        f"0:{stream_index}",
        "-ac",
        ac,
        "-af",
        af,
        "-c:a",
        "pcm_f32le",
        "-vn",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


__all__ = [
    "binary_available",
    "require_binary",
    "have_binary",
    "run_command",
    "run_ffmpeg",
    "extract_dialog_source",
    "probe_audio_duration",
    "FFmpegError",
]
