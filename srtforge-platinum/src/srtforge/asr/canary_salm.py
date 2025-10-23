"""Interface to the NVIDIA Canary-Qwen SALM model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..config import PipelineConfig
from ..logging import get_logger
from ..utils import text

LOGGER = get_logger("asr.canary")


@dataclass(slots=True)
class CanaryChunkResult:
    """Represents the text emitted by the SALM for a chunk."""

    chunk_index: int
    start: float
    end: float
    text: str


class CanarySALM:
    """Thin wrapper around NeMo's SALM interface.

    The project does not ship the heavy NeMo dependency in tests, but the
    wrapper exposes the intended interface and logs the expected behaviour.
    """

    def __init__(self, config: PipelineConfig, keywords: Optional[List[str]] = None) -> None:
        self._config = config
        self._keywords = keywords or []
        self._context_buffer: List[str] = []

    def _build_prompt(self) -> str:
        if not self._context_buffer:
            return ""
        context = " ".join(self._context_buffer)[-self._config.salm_context.max_context_chars :]
        return context

    def transcribe_chunks(
        self,
        chunks: Iterable[tuple[int, float, float]],
    ) -> List[CanaryChunkResult]:
        """Simulate SALM transcription for each chunk.

        In a production deployment this method would load the NeMo model and
        decode audio tensors. The reference implementation returns placeholder
        text that preserves the structure for downstream components.
        """
        results: List[CanaryChunkResult] = []
        for index, start, end in chunks:
            prompt = self._build_prompt()
            LOGGER.debug("SALM prompt for chunk %s: %s", index, prompt)
            dummy_text = f"Chunk {index} transcript"
            if self._keywords:
                dummy_text += " (" + ", ".join(self._keywords[:3]) + ")"
            results.append(CanaryChunkResult(index, start, end, dummy_text))
            sentences = text.split_sentences(dummy_text)
            if sentences:
                carry = sentences[-self._config.salm_context.carry_sentences :]
                self._context_buffer = carry
        return results


__all__ = ["CanaryChunkResult", "CanarySALM"]
