"""Tests for SALM prompt budgeting."""

from __future__ import annotations

from typing import List

from srtforge.asr import canary_salm


class _DummyTokenizer:
    def text_to_ids(self, text: str) -> List[int]:
        return list(range(len(text.split())))

    def ids_to_text(self, ids) -> str:  # pragma: no cover - trivial
        return "dummy"


class _DummyModel:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self.audio_locator_tag = "<audio>"

    def generate(self, **_: object):
        return [1, 2, 3]


class _DummyLoader:
    @staticmethod
    def from_pretrained(model_id: str) -> _DummyModel:  # pragma: no cover - trivial
        return _DummyModel()


def test_prompt_respects_token_budget(monkeypatch) -> None:
    monkeypatch.setattr(canary_salm, "SALM", _DummyLoader)
    monkeypatch.setattr(canary_salm, "_SALM_IMPORT_ERROR", None)
    cfg = canary_salm.SALMConfig(
        max_prompt_tokens=16,
        max_total_tokens=64,
        max_new_tokens=32,
    )
    model = canary_salm.CanarySALM(cfg)
    context = [f"Sentence {i}." for i in range(10)]
    keywords = [f"kw{i}" for i in range(200)]
    prompt = model.build_prompt(context, keywords)
    assert model._token_len(prompt) <= cfg.max_prompt_tokens
