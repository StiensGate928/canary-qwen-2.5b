"""Interface to the NVIDIA Canary-Qwen SALM model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

try:  # pragma: no cover - import guard for environments without torch
    import torch
except Exception:  # pragma: no cover - fallback for CPU-only test envs
    torch = None  # type: ignore

try:  # pragma: no cover - exercised via runtime patching in tests
    from nemo.collections.speechlm2.models import SALM  # type: ignore
    _SALM_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - handled in __init__
    SALM = None  # type: ignore
    _SALM_IMPORT_ERROR = exc

from ..logging import get_logger

log = get_logger("asr.canary")


@dataclass(slots=True)
class SALMConfig:
    model_id: str = "nvidia/canary-qwen-2.5b"
    max_total_tokens: int = 1024
    max_new_tokens: int = 256
    carry_sentences: int = 2
    max_prompt_tokens: int = 256
    greedy: bool = True


class CanarySALM:
    """Thin wrapper around NeMo SALM with token-budgeted prompts."""

    def __init__(self, cfg: SALMConfig) -> None:
        if SALM is None:  # pragma: no cover - exercised in integration flows
            hint = "Ensure NeMo is installed."
            if isinstance(_SALM_IMPORT_ERROR, ModuleNotFoundError) and getattr(
                _SALM_IMPORT_ERROR, "name", ""
            ) == "megatron":
                hint += " Install the optional 'megatron-core' package (``pip install megatron-core``) required by newer NeMo releases."
            raise RuntimeError(
                f"NeMo SALM is not available. {hint}"
            ) from _SALM_IMPORT_ERROR
        self.cfg = cfg
        try:
            log.info("Loading SALM model: %s", cfg.model_id)
            self.model = SALM.from_pretrained(cfg.model_id)  # type: ignore[arg-type]
            self.tokenizer = self.model.tokenizer
            self.audio_tag = self.model.audio_locator_tag
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                f"Failed to load SALM '{cfg.model_id}'. Ensure NeMo and checkpoints are available."
            ) from exc

    # --- prompt helpers -------------------------------------------------
    def _token_len(self, text: str) -> int:
        ids = self.tokenizer.text_to_ids(text)
        return len(ids)

    def _truncate_by_tokens(self, text: str, budget: int) -> str:
        if budget <= 0:
            return ""
        if self._token_len(text) <= budget:
            return text
        lo, hi = 0, len(text)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            snippet = text[-mid:]
            if self._token_len(snippet) <= budget:
                best = snippet
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def build_prompt(self, recent_sentences: List[str], keywords: List[str]) -> str:
        context = " ".join(recent_sentences).strip()
        keyword_line = ", ".join(keywords[:50]) if keywords else ""
        base = ""
        if context:
            base += f"Context: {context}\n"
        if keyword_line:
            base += f"Keywords: {keyword_line}\n"
        instruction = f"Transcribe the following: {self.audio_tag}"
        instruction_tokens = self._token_len(instruction)
        budget = max(0, self.cfg.max_prompt_tokens - instruction_tokens)
        base = self._truncate_by_tokens(base, budget)
        return f"{base}{instruction}"

    # --- inference ------------------------------------------------------
    def transcribe_chunk(self, wav_path: Path, prompt: str) -> str:
        prompts = [[{"role": "user", "content": prompt, "audio": [str(wav_path)]}]]
        prompt_tokens = self._token_len(prompt)
        max_new = min(
            self.cfg.max_new_tokens,
            max(0, self.cfg.max_total_tokens - prompt_tokens - 8),
        )
        if max_new < 16:
            log.warning("Prompt exceeds token budget; truncating aggressively.")
            prompt = self._truncate_by_tokens(
                prompt, max(32, self.cfg.max_total_tokens - 64)
            )
            prompt_tokens = self._token_len(prompt)
            max_new = min(
                self.cfg.max_new_tokens,
                max(0, self.cfg.max_total_tokens - prompt_tokens - 8),
            )

        gen_kwargs = {"prompts": prompts}
        if max_new > 0:
            gen_kwargs["max_new_tokens"] = max_new
        try:
            ids = self.model.generate(**gen_kwargs)
        except TypeError:  # pragma: no cover - older NeMo signatures
            ids = self.model.generate(prompts=prompts)

        tokens = ids
        shp = getattr(tokens, "shape", None)
        log.debug("SALM returned tokens type=%s, shape=%s", type(tokens), shp)

        out = tokens
        if isinstance(out, dict) and "tokens" in out:
            out = out["tokens"]

        if torch is not None and isinstance(out, torch.Tensor):
            if out.ndim > 1:
                out = out[0]
            out = out.detach().cpu().tolist()
        elif isinstance(out, (list, tuple)) and out and torch is not None and isinstance(out[0], torch.Tensor):
            out = [int(t.item()) for t in out]
        else:
            out = [int(x) for x in out]

        text = self.tokenizer.ids_to_text(out).strip()
        return text


__all__ = ["CanarySALM", "SALMConfig"]
