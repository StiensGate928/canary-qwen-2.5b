"""Tests for audio preprocessing filter selection."""

from srtforge.audio.preprocess import build_preprocess_filter_chain


def test_filter_chain_defaults_include_processing() -> None:
    chain = build_preprocess_filter_chain()
    assert "afftdn=nf=-25.0" in chain
    assert "loudnorm=I=-16:LRA=7:TP=-3" in chain


def test_filter_chain_respects_toggles() -> None:
    chain = build_preprocess_filter_chain(
        afftdn_nf=-25.0,
        enable_denoise=False,
        enable_normalize=False,
    )
    assert "afftdn" not in chain
    assert "loudnorm" not in chain
    assert chain.startswith("highpass=f=80")
    assert chain.endswith("aresample=16000:resampler=soxr:precision=28")
