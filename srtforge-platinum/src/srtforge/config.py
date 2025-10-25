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
    afftdn_nf: float = -25.0
    denoise: bool = True
    normalize: bool = True


@dataclass(slots=True)
class SeparationFv4Config:
    repo_dir: pathlib.Path = pathlib.Path("/opt/mel-band-roformer")
    cfg: pathlib.Path = pathlib.Path("/models/config_vocals_mel_band_roformer.yaml")
    ckpt: pathlib.Path = pathlib.Path("/models/voc_fv4.ckpt")
    num_overlap: int = 6


@dataclass(slots=True)
class SeparationBanditConfig:
    repo_dir: pathlib.Path = pathlib.Path("/opt/bandit")
    ckpt: pathlib.Path = pathlib.Path("/models/bandit_vocals.ckpt")
    cfg: pathlib.Path = pathlib.Path("/models/bandit_vocals.yaml")
    model_name: str = "BandIt Vocals V7"


@dataclass(slots=True)
class SeparationConfig:
    backend: str = "bandit"
    sr_frontend_hz: int = 44100
    prefer_center: bool = True
    fv4: SeparationFv4Config = field(default_factory=SeparationFv4Config)
    bandit: SeparationBanditConfig = field(default_factory=SeparationBanditConfig)


@dataclass(slots=True)
class VadConfig:
    noise_db: float = -30.0
    min_silence: float = 0.40
    pad: float = 0.12
    merge_gap: float = 0.30


@dataclass(slots=True)
class ChunkConfig:
    min_len: float = 30.0
    max_len: float = 36.0
    overlap: float = 1.0


@dataclass(slots=True)
class ModelsConfig:
    salm_id: str = "nvidia/canary-qwen-2.5b"
    parakeet_id: str = "nvidia/parakeet-tdt-0.6b-v2"


@dataclass(slots=True)
class SalmContextConfig:
    carry_sentences: int = 2
    max_prompt_tokens: int = 256
    max_total_tokens: int = 1024
    max_new_tokens: int = 256
    keywords_file: Optional[pathlib.Path] = None


@dataclass(slots=True)
class ParakeetConfig:
    enabled: bool = True
    use_gpu: bool = True


@dataclass(slots=True)
class CombinationConfig:
    enable_rover: bool = True
    method: str = "maxconf"


@dataclass(slots=True)
class AlignmentConfig:
    use_mfa: bool = False
    acoustic_model: str = "english_mfa"
    dict_path: Optional[pathlib.Path] = None


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
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    separation: SeparationConfig = field(default_factory=SeparationConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    salm_context: SalmContextConfig = field(default_factory=SalmContextConfig)
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
    if config.chunking.min_len <= 0:
        raise ValueError("Minimum chunk length must be positive")
    if config.chunking.max_len <= 0:
        raise ValueError("Maximum chunk length must be positive")
    if config.chunking.min_len > config.chunking.max_len:
        raise ValueError("Minimum chunk length cannot exceed maximum chunk length")
    if config.chunking.overlap >= config.chunking.max_len:
        raise ValueError("Overlap must be smaller than maximum chunk length")
    if config.chunking.overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if config.vad.merge_gap < 0:
        raise ValueError("VAD merge gap must be non-negative")
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
