"""Audio preprocessing pipeline."""

from __future__ import annotations

import subprocess as sp
from pathlib import Path


def preprocess_and_resample_16k(
    in_wav: Path,
    out_wav: Path,
    arnndn_model: str = "rnnoise",
    afftdn_nf: float = -25.0,
) -> None:
    """Run the production filter chain before 16 kHz resampling."""

    filters = ",".join(
        [
            "highpass=f=80",
            f"arnndn=m={arnndn_model}" if arnndn_model else "arnndn",
            f"afftdn=nf={afftdn_nf}",
            "deesser=i=6",
            "acompressor=threshold=-22dB:ratio=2:attack=5:release=50:makeup=2",
            "loudnorm=I=-16:LRA=7:TP=-3",
            "aresample=16000:resampler=soxr:precision=28",
        ]
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


__all__ = ["preprocess_and_resample_16k"]
