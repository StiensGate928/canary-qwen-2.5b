"""Parakeet ASR integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

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

    def transcribe_with_word_times(self, wav_path: Path) -> tuple[str, List[Tuple[str, float, float]]]:
        """Return (text, [(word, start, end), ...]) for ``wav_path``."""

        try:
            hyps = self.model.transcribe(
                [str(wav_path)],
                batch_size=1,
                return_hypotheses=True,
                timestamps=True,
            )
        except Exception as exc:  # pragma: no cover - runtime specific
            log.warning("Parakeet transcription failed: %s", exc)
            return "", []

        if not hyps:
            return "", []

        hyp = hyps[0]
        if isinstance(hyp, list) and hyp:
            hyp = hyp[0]

        if hasattr(hyp, "text"):
            text: str = hyp.text
        elif isinstance(hyp, str):
            text = hyp
        else:
            text = ""

        word_times: List[Tuple[str, float, float]] = []

        ts = getattr(hyp, "timestep", None) or getattr(hyp, "timestamp", None)
        if isinstance(ts, dict):
            entries: Union[list, None] = ts.get("word") or ts.get("words")
            if isinstance(entries, list):
                for item in entries:
                    if isinstance(item, (list, tuple)) and len(item) >= 3:
                        w, s, e = item[0], item[1], item[2]
                    elif isinstance(item, dict):
                        w = item.get("word") or item.get("text") or ""
                        s = item.get("start_time") or item.get("start") or item.get("ts")
                        e = item.get("end_time") or item.get("end") or item.get("te")
                    else:
                        continue

                    if s is None or e is None:
                        continue

                    try:
                        s = float(s)
                        e = float(e)
                    except Exception:
                        continue

                    word_times.append((str(w), s, e))

        return text, word_times


__all__ = ["ParakeetASR", "ParakeetConfig"]
