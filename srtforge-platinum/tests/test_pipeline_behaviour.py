"""Behavioural tests for the CLI pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from srtforge.cli import SubtitlePipeline
from srtforge.config import PipelineConfig
from srtforge.utils import ffmpeg


def _build_pipeline(embed: bool = False) -> SubtitlePipeline:
    config = PipelineConfig()
    config.chunking.max_len = 10.0
    config.chunking.min_len = 1.0
    config.chunking.overlap = 1.0
    config.vad.pad = 0.0
    return SubtitlePipeline(
        config=config,
        keywords=[],
        use_cpu=True,
        enable_parakeet=False,
        enable_rover=False,
        enable_mfa=False,
        embed_subtitles=embed,
        language="eng",
    )


def test_plan_chunks_uses_vad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")
    pipeline = _build_pipeline()

    monkeypatch.setattr(
        ffmpeg,
        "run_silencedetect",
        lambda *args, **kwargs: [
            "[silencedetect @ 0x0] silence_start: 2.0",
            "[silencedetect @ 0x0] silence_end: 4.0 | silence_duration: 2.0",
        ],
    )

    chunks = pipeline._plan_chunks(audio_path, duration=6.0)
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 2.0), (4.0, 6.0)]


def test_plan_chunks_fallback_without_vad(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"")
    pipeline = _build_pipeline()

    def raiser(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ffmpeg, "run_silencedetect", raiser)

    chunks = pipeline._plan_chunks(audio_path, duration=5.0)
    assert [(chunk.start, chunk.end) for chunk in chunks] == [(0.0, 5.0)]


def test_maybe_separate_skips_mono(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video_path = tmp_path / "video.mkv"
    video_path.write_bytes(b"")
    extracted = tmp_path / "audio.wav"
    extracted.write_bytes(b"")
    pipeline = _build_pipeline()

    monkeypatch.setattr(ffmpeg, "probe_audio_channels", lambda _: 1)

    called = False

    def fake_separate(*args, **kwargs):
        nonlocal called
        called = True
        return tmp_path / "separated.wav"

    monkeypatch.setattr("srtforge.cli.separate_dialogue", fake_separate)

    result = pipeline._maybe_separate(extracted, tmp_path, video_path)
    assert result == extracted
    assert not called


def test_maybe_embed_invokes_mux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "clip.mkv"
    video.write_bytes(b"")
    srt = tmp_path / "clip.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest\n")
    pipeline = _build_pipeline(embed=True)

    recorded: dict[str, list[str]] = {}

    def fake_run_ffmpeg(args, ffmpeg_binary="ffmpeg"):
        recorded["args"] = list(args)
        return CompletedProcess(args=list(args), returncode=0)

    monkeypatch.setattr(ffmpeg, "run_ffmpeg", fake_run_ffmpeg)

    pipeline._maybe_embed(video, srt)

    assert recorded["args"][0] == "-y"
    assert recorded["args"][3] == str(video)
    assert recorded["args"][5] == str(srt)
    assert recorded["args"][-1] == str(video.with_suffix(".subs.mkv"))
