"""Parakeet ASR integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

try:  # pragma: no cover - exercised when NeMo is installed
    import nemo.collections.asr as nemo_asr
    _PARAKEET_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - handled during init
    nemo_asr = None  # type: ignore
    _PARAKEET_IMPORT_ERROR = exc

from ..logging import get_logger

log = get_logger("asr.parakeet")


@dataclass(slots=True)
class ParakeetConfig:
    model_id: str = "nvidia/parakeet-tdt-0.6b-v2"
    use_gpu: bool = True


class ParakeetASR:
    """Wrapper around NeMo Parakeet models providing word times."""

    def __init__(self, cfg: ParakeetConfig) -> None:
        if nemo_asr is None:  # pragma: no cover - integration runtime
            hint = "Install NeMo to use Parakeet."
            if isinstance(_PARAKEET_IMPORT_ERROR, ModuleNotFoundError) and getattr(
                _PARAKEET_IMPORT_ERROR, "name", ""
            ) == "megatron":
                hint += " Install the optional 'megatron-core' package (``pip install megatron-core``) to satisfy the NeMo dependency."
            raise RuntimeError(
                f"NeMo ASR is not available. {hint}"
            ) from _PARAKEET_IMPORT_ERROR
        self.cfg = cfg
        try:
            log.info("Loading Parakeet model: %s", cfg.model_id)
            try:
                self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                    model_name=cfg.model_id
                )
            except Exception:
                self.model = nemo_asr.models.EncDecTransducerModel.from_pretrained(
                    model_name=cfg.model_id
                )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError(
                f"Failed to load Parakeet '{cfg.model_id}'. Verify the model id and installation."
            ) from exc

    def transcribe_with_word_times(
        self, wav: Path
    ) -> Tuple[str, List[Tuple[str, float, float]]]:
        """Return (text, [(word, start, end), ...]) for ``wav``."""

        try:
            hyps = self.model.transcribe(
                paths2audio_files=[str(wav)], return_hypotheses=True
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            log.warning("Parakeet transcription failed: %s", exc)
            return "", []

        if not hyps or not hyps[0]:
            return "", []
        hyp = hyps[0][0]
        text = getattr(hyp, "text", "")
        timestamps: List[Tuple[str, float, float]] = []
        if hasattr(hyp, "words") and hyp.words:
            for word in hyp.words:
                start = float(getattr(word, "start_offset", 0.0))
                end = float(getattr(word, "end_offset", start))
                timestamps.append((word.word, start, end))
        return text, timestamps


__all__ = ["ParakeetASR", "ParakeetConfig"]
