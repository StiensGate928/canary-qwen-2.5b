"""Audio separation helpers using RoFormer-based models."""

from __future__ import annotations

import shutil
import subprocess as sp
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ..logging import get_logger
from ..utils.ffmpeg import FFmpegError, have_binary, run_ffmpeg

LOGGER = get_logger("audio.separate")


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    LOGGER.debug("exec: %s", " ".join(map(str, cmd)))
    try:
        sp.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except sp.CalledProcessError as exc:  # pragma: no cover - delegated tooling
        raise RuntimeError(f"Separation subprocess failed: {exc}") from exc


def _resample_for_sep(in_wav: Path, out_wav: Path, sr_hz: int) -> Path:
    """High-quality SoXr resampling to float32 stereo."""
    try:
        run_ffmpeg(
            [
                "-y",
                "-hide_banner",
                "-i",
                str(in_wav),
                "-ac",
                "2",
                "-af",
                f"aresample={sr_hz}:resampler=soxr:precision=28",
                "-c:a",
                "pcm_f32le",
                str(out_wav),
            ]
        )
    except FFmpegError as exc:  # pragma: no cover - heavy dependency
        raise RuntimeError("Failed to resample audio for separation") from exc
    return out_wav


def _find_vocals_from_folder(folder: Path) -> Path:
    """Pick the most likely vocal stem from a folder of WAV files."""

    candidates = list(folder.glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No WAVs produced in {folder}")

    def score(path: Path) -> tuple[int, int]:
        name = path.name.lower()
        score_has_vocals = int(any(token in name for token in ("voc", "vocal", "voice", "speech", "dialog")))
        score_not_instrumental = 0 if any(token in name for token in ("inst", "instrumental", "music")) else 1
        return score_has_vocals, score_not_instrumental

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def separate_dialogue(
    input_wav: Path,
    output_wav: Path,
    backend: str,
    *,
    sr_frontend_hz: int = 44100,
    prefer_center_first: bool = True,
    # fv4
    fv4_repo_dir: Optional[Path] = None,
    fv4_cfg: Optional[Path] = None,
    fv4_ckpt: Optional[Path] = None,
    fv4_num_overlap: int = 6,
    # bandit
    bandit_repo_dir: Optional[Path] = None,
    bandit_ckpt: Optional[Path] = None,
    bandit_cfg: Optional[Path] = None,
    bandit_model_name: Optional[str] = None,
) -> Path:
    """Extract a dialogue-focused stem using the configured separator."""

    del prefer_center_first  # retained for compatibility with configuration structure

    if backend == "none":
        LOGGER.debug("Separation backend disabled; copying input track")
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_wav, output_wav)
        return output_wav

    have_binary("ffmpeg", fatal=True)

    with tempfile.TemporaryDirectory(prefix="sep_") as tmp:
        tmpdir = Path(tmp)
        prep = tmpdir / "prep_44k.wav"
        _resample_for_sep(input_wav, prep, sr_frontend_hz)

        if backend == "fv4":
            if not (fv4_repo_dir and fv4_cfg and fv4_ckpt):
                raise ValueError("fv4 requires --sep-repo, --sep-cfg and --sep-ckpt")

            in_dir = tmpdir / "in"
            out_dir = tmpdir / "out"
            in_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_in = in_dir / "clip.wav"
            shutil.copy2(prep, tmp_in)
            cmd = [
                sys.executable,
                str(fv4_repo_dir / "inference.py"),
                "--config_path",
                str(fv4_cfg),
                "--model_path",
                str(fv4_ckpt),
                "--input_folder",
                str(in_dir),
                "--store_dir",
                str(out_dir),
            ]
            # FV4 currently rejects the --num_overlap flag; re-enable when supported
            # cmd += ["--num_overlap", str(fv4_num_overlap)]
            _run(cmd, cwd=fv4_repo_dir)
            vocals = _find_vocals_from_folder(out_dir)
            shutil.copy2(vocals, output_wav)
            return output_wav

        if backend == "bandit":
            if not (bandit_repo_dir and bandit_ckpt and bandit_cfg):
                raise ValueError("bandit requires --sep-repo, --sep-ckpt and --sep-cfg")

            bundle = tmpdir / "bundle"
            bundle.mkdir(parents=True, exist_ok=True)
            bundle_ckpt = bundle / Path(bandit_ckpt).name
            bundle_yaml = bundle / "hparams.yaml"
            shutil.copy2(bandit_ckpt, bundle_ckpt)
            shutil.copy2(bandit_cfg, bundle_yaml)

            model_name = bandit_model_name or "BandIt Vocals V7"
            cmd = [
                sys.executable,
                str(bandit_repo_dir / "inference.py"),
                "inference",
                "--ckpt_path",
                str(bundle_ckpt),
                "--file_path",
                str(prep),
                "--model_name",
                model_name,
            ]
            _run(cmd, cwd=bandit_repo_dir)

            produced = sorted(bandit_repo_dir.rglob("*.wav"))
            produced.extend(sorted(bundle.rglob("*.wav")))
            produced.extend(sorted(tmpdir.rglob("*.wav")))
            candidates = [
                path
                for path in produced
                if path != prep
                and "inst" not in path.name.lower()
                and any(token in path.name.lower() for token in ("voc", "voice", "speech", "dialog"))
            ]
            if not candidates:
                candidates = [path for path in produced if path != prep]
            if not candidates:
                raise FileNotFoundError("Bandit inference did not produce a WAV we could locate")
            shutil.copy2(candidates[-1], output_wav)
            return output_wav

        raise ValueError(f"Unknown separation backend: {backend}")


__all__ = ["separate_dialogue"]

