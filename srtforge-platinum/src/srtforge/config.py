"""Configuration models and loader utilities."""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass(slots=True)
class FrontendConfig:
    prefer_center: bool = True
    demucs_model: str = "htdemucs"
    arnndn_model: str = "rnnoise"
    afftdn_nf: float = -25.0
    denoise: bool = True
    normalize: bool = True


@dataclass(slots=True)
class VadConfig:
    noise_db: float = -33.0
    min_silence: float = 0.35
    pad: float = 0.10


@dataclass(slots=True)
class ChunkConfig:
    max_len: float = 36.0
    overlap: float = 1.0


@dataclass(slots=True)
class SalmConfig:
    model_id: str = "nvidia/canary-qwen-2.5b"
    carry_sentences: int = 2
    max_context_chars: int = 220
    keywords_file: Optional[pathlib.Path] = None


@dataclass(slots=True)
class ParakeetConfig:
    model_id: str = "nvidia/parakeet-rnnt-1.1b"
    enabled: bool = True


@dataclass(slots=True)
class CombinationConfig:
    enable_rover: bool = True
    method: str = "maxconf"


@dataclass(slots=True)
class AlignmentConfig:
    use_mfa: bool = False
    mfa_dict: Optional[pathlib.Path] = None
    mfa_acoustic: Optional[pathlib.Path] = None


@dataclass(slots=True)
class ReadingConfig:
    max_chars_per_line: int = 42
    max_lines: int = 2
    min_duration_s: float = 1.0


@dataclass(slots=True)
class PathsConfig:
    temp_dir: pathlib.Path = pathlib.Path("./tmp")
    output_dir: pathlib.Path = pathlib.Path("./output")


@dataclass(slots=True)
class ModelConfig:
    salm: SalmConfig = field(default_factory=SalmConfig)
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)


@dataclass(slots=True)
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    salm_context: SalmConfig = field(default_factory=SalmConfig)
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    combination: CombinationConfig = field(default_factory=CombinationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    reading: ReadingConfig = field(default_factory=ReadingConfig)


def _resolve_path(value: Any) -> Optional[pathlib.Path]:
    if value is None:
        return None
    path = pathlib.Path(value).expanduser()
    return path


def load_config(path: pathlib.Path) -> PipelineConfig:
    """Load configuration from a YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        data: Dict[str, Any] = yaml.safe_load(handle) or {}
    return apply_overrides(PipelineConfig(), data)


def apply_overrides(config: PipelineConfig, overrides: Dict[str, Any]) -> PipelineConfig:
    """Apply dictionary overrides recursively to a configuration object."""

    def merge(target: Any, src: Dict[str, Any]) -> Any:
        if dataclasses.is_dataclass(target):
            for key, value in src.items():
                if not hasattr(target, key):
                    raise KeyError(f"Unknown configuration key: {key}")
                attr = getattr(target, key)
                if dataclasses.is_dataclass(attr) and isinstance(value, dict):
                    merge(attr, value)
                else:
                    if isinstance(attr, pathlib.Path) or key.endswith("_file") or key.endswith("_dir"):
                        setattr(target, key, _resolve_path(value))
                    else:
                        setattr(target, key, value)
            return target
        raise TypeError("Target must be a dataclass instance")

    merge(config, overrides)
    return config


def validate_config(config: PipelineConfig) -> None:
    """Validate logical invariants of the pipeline configuration."""
    if config.chunking.overlap >= config.chunking.max_len:
        raise ValueError("Overlap must be smaller than maximum chunk length")
    if config.reading.max_lines < 1:
        raise ValueError("At least one subtitle line must be allowed")
    if config.reading.max_chars_per_line <= 0:
        raise ValueError("Maximum characters per line must be positive")


def load_keywords(path: Optional[pathlib.Path]) -> List[str]:
    """Load keywords from a file if provided."""
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        raise FileNotFoundError(f"Keyword file not found: {path}") from None
