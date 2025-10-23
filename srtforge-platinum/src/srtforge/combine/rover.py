"""ROVER combination helpers."""

from __future__ import annotations

import subprocess as sp
import tempfile
from pathlib import Path
from typing import List, Tuple

from ..logging import get_logger

log = get_logger("combine.rover")


def _run_rover(hyp_a_ctm: Path, hyp_b_ctm: Path, out_ctm: Path, method: str = "maxconf") -> None:
    cmd = [
        "rover",
        "-h",
        str(hyp_a_ctm),
        "ctm",
        "-h",
        str(hyp_b_ctm),
        "ctm",
        "-o",
        str(out_ctm),
        "-m",
        method,
    ]
    log.debug("ROVER cmd: %s", " ".join(cmd))
    sp.run(cmd, check=True)


def combine_per_chunk(
    chunks: List[Tuple[float, float]],
    canary_ctms: List[List[str]],
    parakeet_ctms: List[List[str]],
    method: str = "maxconf",
) -> List[List[str]]:
    """Run ROVER independently per chunk and return local CTMs."""

    if not (len(chunks) == len(canary_ctms) == len(parakeet_ctms)):
        raise ValueError("Chunk metadata and CTM lists must be aligned")

    consensus_per_chunk: List[List[str]] = []
    with tempfile.TemporaryDirectory(prefix="rover_") as tmpdir:
        tmp_path = Path(tmpdir)
        for index in range(len(chunks)):
            a = tmp_path / f"canary_{index:05d}.ctm"
            b = tmp_path / f"parakeet_{index:05d}.ctm"
            out = tmp_path / f"consensus_{index:05d}.ctm"
            a.write_text("\n".join(canary_ctms[index]) + "\n", encoding="utf-8")
            b.write_text("\n".join(parakeet_ctms[index]) + "\n", encoding="utf-8")
            _run_rover(a, b, out, method=method)
            consensus_lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
            consensus_per_chunk.append(consensus_lines)
    return consensus_per_chunk


__all__ = ["combine_per_chunk"]
