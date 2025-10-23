"""Tests for FFmpeg channel probing logic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from srtforge.utils import ffmpeg


def _run_case(monkeypatch, layout: str, channels: int) -> List[str]:
    recorded: Dict[str, List[str]] = {}

    def fake_probe(path: Path, stream_index: int = 0) -> dict:
        return {"channels": channels, "channel_layout": layout}

    def fake_run(cmd: List[str], check: bool = True) -> None:  # pragma: no cover - trivial
        recorded["cmd"] = cmd

    monkeypatch.setattr(ffmpeg, "_probe_audio_stream", fake_probe)
    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    ffmpeg.extract_dialog_source(Path("input.mkv"), Path("out.wav"), prefer_center=True)
    return recorded["cmd"]


def test_center_channel_used_when_available(monkeypatch) -> None:
    cmd = _run_case(monkeypatch, "5.1", 6)
    assert any("pan=mono|c0=FC" in token for token in cmd)
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "1"
    assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "0:a:0"


def test_center_channel_preserved_when_not_collapsing(monkeypatch) -> None:
    recorded: Dict[str, List[str]] = {}

    def fake_probe(path: Path, stream_index: int = 0) -> dict:
        return {"channels": 6, "channel_layout": "5.1"}

    def fake_run(cmd: List[str], check: bool = True) -> None:  # pragma: no cover - trivial
        recorded["cmd"] = cmd

    monkeypatch.setattr(ffmpeg, "_probe_audio_stream", fake_probe)
    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)
    ffmpeg.extract_dialog_source(
        Path("input.mkv"),
        Path("out.wav"),
        prefer_center=True,
        collapse_to_mono=False,
    )
    cmd = recorded["cmd"]
    assert any("pan=stereo|c0=FC|c1=FC" in token for token in cmd)
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2"


def test_stereo_falls_back_to_downmix(monkeypatch) -> None:
    cmd = _run_case(monkeypatch, "stereo", 2)
    assert not any("pan=mono|c0=FC" in token for token in cmd)
    assert "-ac" in cmd and cmd[cmd.index("-ac") + 1] == "2"


def test_odd_layout_without_fc(monkeypatch) -> None:
    cmd = _run_case(monkeypatch, "quad", 4)
    assert not any("pan=mono|c0=FC" in token for token in cmd)
