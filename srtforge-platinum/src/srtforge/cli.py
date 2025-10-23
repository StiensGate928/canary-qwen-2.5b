"""Command line interface for srtforge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from .align import mfa
from .asr import ctm
from .asr.canary_salm import CanarySALM
from .asr.parakeet import ParakeetASR
from .combine.rover import RoverCombiner
from .config import (
    PipelineConfig,
    apply_overrides,
    load_config,
    load_keywords,
    validate_config,
)
from .logging import configure_logging, get_logger
from .segment.chunker import Chunk, Chunker
from .srt.writer import SRTWriter, SubtitleSegment

LOGGER = get_logger("cli")


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".wav", ".flac", ".m4a"}


class SubtitlePipeline:
    """High-level orchestration for subtitle generation."""

    def __init__(
        self,
        config: PipelineConfig,
        keywords: Sequence[str],
        use_cpu: bool,
        enable_parakeet: bool,
        enable_rover: bool,
        enable_mfa: bool,
        embed_subtitles: bool,
        language: str,
    ) -> None:
        self._config = config
        self._keywords = list(keywords)
        self._use_cpu = use_cpu
        self._enable_parakeet = enable_parakeet
        self._enable_rover = enable_rover
        self._enable_mfa = enable_mfa
        self._embed = embed_subtitles
        self._language = language

    def _collect_media(self, source: Path, recursive: bool) -> List[Path]:
        if source.is_file():
            return [source]
        pattern = "**/*" if recursive else "*"
        return [p for p in source.glob(pattern) if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    def _plan_chunks(self) -> List[Chunk]:
        chunker = Chunker(
            max_len=self._config.chunking.max_len,
            overlap=self._config.chunking.overlap,
        )
        dummy_span = [(0.0, self._config.chunking.max_len)]
        return chunker.chunk_from_pairs(dummy_span)

    def _simulate_alignment(self, transcript: List[str]) -> List[str]:
        if not self._enable_mfa:
            return transcript
        return mfa.align_segments(
            audio_path=self._config.paths.temp_dir / "dummy.wav",
            transcript_lines=transcript,
            dictionary_path=self._config.alignment.mfa_dict,
            acoustic_model=self._config.alignment.mfa_acoustic,
        )

    def _maybe_embed(self, video_path: Path, srt_path: Path) -> None:
        if not self._embed:
            return
        LOGGER.info("Embedding subtitles into %s (simulated)", video_path)

    def _transcribe(self, video_path: Path) -> Path:
        LOGGER.info("Processing %s", video_path)
        chunks = self._plan_chunks()
        canary = CanarySALM(self._config, keywords=self._keywords)
        canary_results = canary.transcribe_chunks(
            (index, chunk.start, chunk.end) for index, chunk in enumerate(chunks)
        )

        parakeet_results: List[ctm.CTMEntry] | None = None
        if self._enable_parakeet:
            parakeet = ParakeetASR(self._config)
            parakeet_chunks = parakeet.transcribe_chunks(
                (index, chunk.start, chunk.end) for index, chunk in enumerate(chunks)
            )
            parakeet_entries: List[List[ctm.CTMEntry]] = [
                ctm.from_parakeet(chunk) for chunk in parakeet_chunks
            ]
            parakeet_results = ctm.merge_chunk_ctms(parakeet_entries)

        canary_entries = ctm.merge_chunk_ctms([
            ctm.from_uniform_text(result) for result in canary_results
        ])

        consensus_entries = canary_entries
        if self._enable_rover and parakeet_results is not None:
            rover = RoverCombiner()
            try:
                rover.assert_available()
            except Exception:
                LOGGER.warning("ROVER binary not found, falling back to Canary-only transcript")
            else:
                consensus_entries = rover.combine([canary_entries, parakeet_results])

        transcript_lines = self._simulate_alignment([entry.word for entry in consensus_entries])

        srt_segments = [
            SubtitleSegment(
                index=i + 1,
                start=entry.start,
                end=entry.start + entry.duration,
                text=line,
            )
            for i, (entry, line) in enumerate(zip(consensus_entries, transcript_lines))
        ]

        writer = SRTWriter(self._config.reading)
        output_dir = self._config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{video_path.stem}.srt"
        writer.write(srt_segments, target)
        self._maybe_embed(video_path, target)
        return target

    def run(self, source: Path, recursive: bool) -> List[Path]:
        files = self._collect_media(source, recursive)
        if not files:
            raise FileNotFoundError("No media files found to process")
        outputs = [self._transcribe(file) for file in files]
        return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="srtforge", description="Platinum subtitle pipeline")
    parser.add_argument("--version", action="version", version="srtforge 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Transcribe video or directory")
    transcribe.add_argument("path", type=Path, help="Path to media file or directory")
    transcribe.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    transcribe.add_argument("--recursive", action="store_true")
    transcribe.add_argument("--embed", action="store_true")
    transcribe.add_argument("--lang", default="eng")
    transcribe.add_argument("--prefer-center", dest="prefer_center", action="store_true", default=None)
    transcribe.add_argument("--no-prefer-center", dest="prefer_center", action="store_false")
    transcribe.add_argument(
        "--sep",
        choices=["none", "fv4", "bandit"],
        help="Voice isolation backend. Overrides config.separation.backend",
    )
    transcribe.add_argument(
        "--sep-repo",
        type=Path,
        help="Path to separation repo (fv4/bandit)",
    )
    transcribe.add_argument(
        "--sep-ckpt",
        type=Path,
        help="Separation checkpoint (.ckpt)",
    )
    transcribe.add_argument(
        "--sep-cfg",
        type=Path,
        help="Separation config (.yaml)",
    )
    transcribe.add_argument(
        "--bandit-model-name",
        type=str,
        help="Bandit model_name for inference.py (e.g., 'BandIt Vocals V7')",
    )
    transcribe.add_argument("--keywords", type=Path, default=None)
    transcribe.add_argument("--cpu", action="store_true")
    transcribe.add_argument("--with-parakeet", action="store_true")
    transcribe.add_argument("--with-rover", action="store_true")
    transcribe.add_argument("--with-mfa", action="store_true")
    transcribe.add_argument("--verbose", action="store_true")
    transcribe.add_argument("--quiet", action="store_true")
    return parser


def _apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    overrides = {}
    if args.prefer_center is not None:
        overrides.setdefault("frontend", {})["prefer_center"] = args.prefer_center
        overrides.setdefault("separation", {})["prefer_center"] = args.prefer_center
    if args.sep:
        overrides.setdefault("separation", {})["backend"] = args.sep
    selected_backend = args.sep or config.separation.backend
    if args.sep_repo:
        if selected_backend == "fv4":
            overrides.setdefault("separation", {}).setdefault("fv4", {})["repo_dir"] = args.sep_repo
        elif selected_backend == "bandit":
            overrides.setdefault("separation", {}).setdefault("bandit", {})["repo_dir"] = args.sep_repo
    if args.sep_ckpt:
        if selected_backend == "fv4":
            overrides.setdefault("separation", {}).setdefault("fv4", {})["ckpt"] = args.sep_ckpt
        elif selected_backend == "bandit":
            overrides.setdefault("separation", {}).setdefault("bandit", {})["ckpt"] = args.sep_ckpt
    if args.sep_cfg:
        if selected_backend == "fv4":
            overrides.setdefault("separation", {}).setdefault("fv4", {})["cfg"] = args.sep_cfg
        elif selected_backend == "bandit":
            overrides.setdefault("separation", {}).setdefault("bandit", {})["cfg"] = args.sep_cfg
    if args.bandit_model_name:
        overrides.setdefault("separation", {}).setdefault("bandit", {})["model_name"] = args.bandit_model_name
    if args.with_parakeet:
        overrides.setdefault("parakeet", {})["enabled"] = True
    else:
        overrides.setdefault("parakeet", {})["enabled"] = config.parakeet.enabled
    if args.with_rover:
        overrides.setdefault("combination", {})["enable_rover"] = True
    if args.with_mfa:
        overrides.setdefault("alignment", {})["use_mfa"] = True
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=getattr(args, "verbose", False), quiet=getattr(args, "quiet", False))

    if args.command == "transcribe":
        config_path: Path = args.config
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        config = load_config(config_path)
        config = _apply_cli_overrides(config, args)
        validate_config(config)
        keyword_file = args.keywords or config.salm_context.keywords_file
        keywords = load_keywords(keyword_file)
        pipeline = SubtitlePipeline(
            config=config,
            keywords=keywords,
            use_cpu=args.cpu,
            enable_parakeet=args.with_parakeet or config.parakeet.enabled,
            enable_rover=args.with_rover or config.combination.enable_rover,
            enable_mfa=args.with_mfa or config.alignment.use_mfa,
            embed_subtitles=args.embed,
            language=args.lang,
        )
        outputs = pipeline.run(args.path, args.recursive)
        for output in outputs:
            LOGGER.info("Generated %s", output)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
