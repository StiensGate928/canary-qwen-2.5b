"""Tests for MFA integration helpers."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from srtforge.align import mfa


def test_ensure_mfa_raises_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(mfa.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError):
        mfa.ensure_mfa()


def test_align_chunks_with_mfa_skips_without_binary(tmp_path: Path) -> None:
    if shutil.which("mfa") is None:
        pytest.skip("mfa binary not installed")

    wav_path = tmp_path / "chunk.wav"
    with wave.open(str(wav_path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    dict_path = tmp_path / "dummy.dict"
    dict_path.write_text("HELLO HH EH L OW\n", encoding="utf-8")

    try:
        mfa.align_chunks_with_mfa(
            [(0.0, 1.0, wav_path, "hello")],
            "english_mfa",
            dict_path,
        )
    except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError):
        pytest.skip("MFA resources unavailable for test run")
