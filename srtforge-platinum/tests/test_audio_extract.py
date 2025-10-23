"""Tests for audio extraction helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from srtforge.audio.extract import AudioExtractor
from srtforge.config import FrontendConfig
from srtforge.utils import ffmpeg


def _stub_ffprobe(monkeypatch: pytest.MonkeyPatch, streams: List[Dict]) -> None:
    def fake_run(cmd, check=True, text=True, capture_output=True):  # pragma: no cover - helper
        del check, text, capture_output
        return type("Completed", (), {"stdout": json.dumps({"streams": streams})})()

    monkeypatch.setattr(ffmpeg.subprocess, "run", fake_run)


def test_select_audio_stream_prefers_requested_language(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = [
        {"index": 2, "tags": {"language": "jpn"}},
        {"index": 3, "tags": {"language": "eng"}},
    ]
    _stub_ffprobe(monkeypatch, streams)
    index = ffmpeg.select_audio_stream(Path("movie.mkv"), language="eng")
    assert index == 1


def test_select_audio_stream_supports_two_letter_language(monkeypatch: pytest.MonkeyPatch) -> None:
    streams = [
        {"index": 2, "tags": {"language": "eng"}},
        {"index": 3, "tags": {"language": "deu"}},
    ]
    _stub_ffprobe(monkeypatch, streams)
    index = ffmpeg.select_audio_stream(Path("movie.mkv"), language="en")
    assert index == 0


def test_audio_extractor_passes_selected_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "clip.mkv"
    target = tmp_path / "clip_extracted.wav"
    config = FrontendConfig()
    extractor = AudioExtractor(config)

    def fake_select(path: Path, language: str | None) -> int:
        assert path == source
        assert language == "eng"
        return 2

    recorded: Dict[str, List] = {}

    def fake_extract(
        input_video: Path,
        out_wav: Path,
        *,
        stream_index: int,
        prefer_center: bool,
        collapse_to_mono: bool,
    ) -> None:
        recorded.update(
            {
                "input": input_video,
                "target": out_wav,
                "stream_index": stream_index,
                "prefer_center": prefer_center,
                "collapse": collapse_to_mono,
            }
        )

    monkeypatch.setattr(ffmpeg, "select_audio_stream", fake_select)
    monkeypatch.setattr(ffmpeg, "extract_dialog_source", fake_extract)

    result = extractor.extract(source, tmp_path, language="eng", collapse_to_mono=False)

    assert result == target
    assert recorded["stream_index"] == 2
    assert recorded["prefer_center"] is config.prefer_center
    assert recorded["collapse"] is False
