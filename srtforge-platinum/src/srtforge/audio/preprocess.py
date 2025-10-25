"""Audio preprocessing pipeline."""

from __future__ import annotations

import subprocess as sp
from pathlib import Path


def build_preprocess_filter_chain(
    afftdn_nf: float = -25.0,
    enable_denoise: bool = True,
    enable_normalize: bool = True,
) -> str:
    """Construct the FFmpeg filter chain for preprocessing."""

    filters = ["highpass=f=80"]
    if enable_denoise:
        filters.append(f"afftdn=nf={afftdn_nf}")
    filters.extend(
        [
            "deesser=i=0.6",
            "acompressor=threshold=-22dB:ratio=2:attack=5:release=50:makeup=2",
        ]
    )
    if enable_normalize:
        filters.append("loudnorm=I=-16:LRA=7:TP=-3")
    filters.append("aresample=16000:resampler=soxr:precision=28")
    return ",".join(filters)


def preprocess_and_resample_16k(
    in_wav: Path,
    out_wav: Path,
    afftdn_nf: float = -25.0,
    enable_denoise: bool = True,
    enable_normalize: bool = True,
) -> None:
    """Run the production filter chain before 16 kHz resampling."""

    filters = build_preprocess_filter_chain(
        afftdn_nf=afftdn_nf,
        enable_denoise=enable_denoise,
        enable_normalize=enable_normalize,
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(in_wav),
        "-af",
        filters,
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        str(out_wav),
    ]
    sp.run(cmd, check=True)


__all__ = ["build_preprocess_filter_chain", "preprocess_and_resample_16k"]
