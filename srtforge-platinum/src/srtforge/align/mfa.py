"""Montreal Forced Aligner integration."""

from __future__ import annotations

import json
import shutil
import subprocess as sp
import tempfile
from pathlib import Path
from typing import List, Tuple

from ..logging import get_logger

log = get_logger("align.mfa")


def ensure_mfa() -> Path:
    """Return the path to the MFA binary or raise an informative error."""

    resolved = shutil.which("mfa")
    if not resolved:
        raise RuntimeError(
            "Montreal Forced Aligner not found in PATH. Install 'mfa' or disable MFA."
        )
    return Path(resolved)


def align_chunks_with_mfa(
    chunks: List[Tuple[float, float, Path, str]],
    acoustic_model: str,
    dict_path: Path,
    n_jobs: int = 2,
) -> List[List[Tuple[str, float, float]]]:
    """Align ``chunks`` with MFA and return per-chunk word timings."""

    ensure_mfa()
    results: List[List[Tuple[str, float, float]]] = []
    with tempfile.TemporaryDirectory(prefix="mfa_") as tmpdir:
        tmp_path = Path(tmpdir)
        corpus = tmp_path / "corpus"
        corpus.mkdir(parents=True, exist_ok=True)
        for index, (_, _, wav_path, text) in enumerate(chunks, start=1):
            wav_target = corpus / f"utt_{index:05d}.wav"
            lab_target = corpus / f"utt_{index:05d}.lab"
            wav_target.write_bytes(wav_path.read_bytes())
            lab_target.write_text(text, encoding="utf-8")
        out_dir = tmp_path / "aligned"
        cmd = [
            "mfa",
            "align",
            str(corpus),
            str(dict_path),
            acoustic_model,
            str(out_dir),
            "--clean",
            "-j",
            str(n_jobs),
            "--output_format",
            "json",
        ]
        log.info("Running MFA: %s", " ".join(cmd))
        sp.run(cmd, check=True)
        for index in range(1, len(chunks) + 1):
            json_path = out_dir / f"utt_{index:05d}.json"
            if not json_path.exists():
                results.append([])
                continue
            data = json.loads(json_path.read_text(encoding="utf-8"))
            words: List[Tuple[str, float, float]] = []
            for entry in data.get("words", []):
                word = entry.get("text", "")
                start = float(entry.get("start", 0.0))
                end = float(entry.get("end", start))
                words.append((word, start, end))
            results.append(words)
    return results


__all__ = ["align_chunks_with_mfa", "ensure_mfa"]
