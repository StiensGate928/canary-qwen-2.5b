"""Command line interface for srtforge."""

from __future__ import annotations

import argparse
import subprocess as sp
from pathlib import Path
from typing import List, Sequence, Tuple

from .align.mfa import align_chunks_with_mfa
from .asr.canary_salm import CanarySALM, SALMConfig
from .asr.ctm import ctm_from_uniform_words, ctm_from_word_times
from .asr.parakeet import ParakeetASR, ParakeetConfig
from .audio.extract import AudioExtractor
from .audio.preprocess import preprocess_and_resample_16k
from .audio.separate import separate_dialogue
from .combine.rover import RoverNotAvailableError, combine_per_chunk
from .config import (
    PipelineConfig,
    apply_overrides,
    load_config,
    load_keywords,
    validate_config,
)
from .logging import configure_logging, get_logger
from .segment import Chunk, Chunker, invert_silences, merge_speech_spans, parse_silencedetect
from .srt.writer import SRTWriter, SubtitleSegment
from .utils import ffmpeg
from .utils.text import update_carry_sentences

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
        self._audio_extractor = AudioExtractor(config.frontend)

    # ------------------------------------------------------------------
    def _collect_media(self, source: Path, recursive: bool) -> List[Path]:
        if source.is_file():
            return [source]
        pattern = "**/*" if recursive else "*"
        return [p for p in source.glob(pattern) if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    def _plan_chunks(self, audio_path: Path, duration: float) -> List[Chunk]:
        chunker = Chunker(
            max_len=self._config.chunking.max_len,
            overlap=self._config.chunking.overlap,
            min_len=self._config.chunking.min_len,
        )
        spans: List[Tuple[float, float]] = [(0.0, duration)]
        vad_cfg = self._config.vad
        try:
            log_lines = ffmpeg.run_silencedetect(
                audio_path,
                noise_db=vad_cfg.noise_db,
                min_silence=vad_cfg.min_silence,
            )
            silences = parse_silencedetect(log_lines)
            speech = invert_silences(silences, duration)
            if speech:
                spans = [(interval.start, interval.end) for interval in speech]
        except Exception as exc:
            LOGGER.warning(
                "VAD failed for %s: %s; falling back to uniform chunking",
                audio_path,
                exc,
            )
        merged = merge_speech_spans(spans, pad=vad_cfg.pad, merge_gap=vad_cfg.merge_gap)
        return chunker.chunk(merged)

    def _cut_chunk(self, source: Path, start: float, end: float, target: Path) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(source),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-ac",
            "1",
            "-c:a",
            "pcm_f32le",
            str(target),
        ]
        sp.run(cmd, check=True)

    def _maybe_separate(self, extracted: Path, temp_dir: Path, video_path: Path) -> Path:
        sep_cfg = self._config.separation
        if not sep_cfg.backend or sep_cfg.backend == "none":
            return extracted
        channels = ffmpeg.probe_audio_channels(extracted)
        if channels <= 1:
            LOGGER.info("Skipping separation for %s: extracted audio already mono", video_path)
            return extracted
        separated = temp_dir / f"{video_path.stem}_separated.wav"
        return separate_dialogue(
            extracted,
            separated,
            sep_cfg.backend,
            sr_frontend_hz=sep_cfg.sr_frontend_hz,
            prefer_center_first=sep_cfg.prefer_center,
            fv4_repo_dir=getattr(sep_cfg.fv4, "repo_dir", None),
            fv4_cfg=getattr(sep_cfg.fv4, "cfg", None),
            fv4_ckpt=getattr(sep_cfg.fv4, "ckpt", None),
            fv4_num_overlap=getattr(sep_cfg.fv4, "num_overlap", 6),
            bandit_repo_dir=getattr(sep_cfg.bandit, "repo_dir", None),
            bandit_ckpt=getattr(sep_cfg.bandit, "ckpt", None),
            bandit_cfg=getattr(sep_cfg.bandit, "cfg", None),
            bandit_model_name=getattr(sep_cfg.bandit, "model_name", None),
        )

    def _maybe_embed(self, video_path: Path, srt_path: Path) -> None:
        if not self._embed:
            return
        output = video_path.with_suffix(".subs.mkv")
        LOGGER.info("Embedding subtitles into %s", output)
        ffmpeg.run_ffmpeg(
            [
                "-y",
                "-hide_banner",
                "-i",
                str(video_path),
                "-i",
                str(srt_path),
                "-map",
                "0",
                "-map",
                "1",
                "-c",
                "copy",
                "-c:s",
                "srt",
                str(output),
            ]
        )

    def _load_salm(self) -> CanarySALM:
        cfg = SALMConfig(
            model_id=self._config.models.salm_id,
            max_total_tokens=self._config.salm_context.max_total_tokens,
            max_new_tokens=self._config.salm_context.max_new_tokens,
            carry_sentences=self._config.salm_context.carry_sentences,
            max_prompt_tokens=self._config.salm_context.max_prompt_tokens,
        )
        model = CanarySALM(cfg)
        if self._use_cpu and hasattr(model.model, "to"):
            model.model.to("cpu")
        return model

    def _load_parakeet(self) -> ParakeetASR | None:
        if not self._enable_parakeet:
            return None
        cfg = ParakeetConfig(
            model_id=self._config.models.parakeet_id,
            use_gpu=self._config.parakeet.use_gpu and not self._use_cpu,
        )
        try:
            model = ParakeetASR(cfg)
        except RuntimeError as exc:
            LOGGER.warning("Parakeet unavailable: %s", exc)
            return None
        if self._use_cpu and hasattr(model.model, "to"):
            model.model.to("cpu")
        return model

    def _transcribe(self, video_path: Path) -> Path:
        LOGGER.info("Processing %s", video_path)
        temp_dir = self._config.paths.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        sep_cfg = self._config.separation
        collapse_to_mono = not (sep_cfg.backend and sep_cfg.backend != "none")
        extracted = self._audio_extractor.extract(
            video_path,
            temp_dir,
            language=self._language,
            collapse_to_mono=collapse_to_mono,
        )

        separated_source = self._maybe_separate(extracted, temp_dir, video_path)

        cleaned = temp_dir / f"{video_path.stem}_clean.wav"
        preprocess_and_resample_16k(
            separated_source,
            cleaned,
            afftdn_nf=self._config.frontend.afftdn_nf,
            enable_denoise=self._config.frontend.denoise,
            enable_normalize=self._config.frontend.normalize,
        )
        duration = ffmpeg.probe_audio_duration(cleaned)
        chunks = self._plan_chunks(cleaned, duration or self._config.chunking.max_len)

        salm = self._load_salm()
        parakeet = self._load_parakeet()

        keywords = self._keywords
        carry: List[str] = []
        chunk_paths: List[Path] = []
        canary_texts: List[str] = []
        parakeet_texts: List[str] = []
        parakeet_wts: List[List[Tuple[str, float, float]]] = []
        chunk_ranges: List[Tuple[float, float]] = []

        for index, chunk in enumerate(chunks):
            chunk_path = temp_dir / f"{video_path.stem}_chunk_{index:05d}.wav"
            self._cut_chunk(cleaned, chunk.start, chunk.end, chunk_path)
            chunk_paths.append(chunk_path)
            chunk_ranges.append((chunk.start, chunk.end))
            prompt = salm.build_prompt(carry, keywords)
            try:
                text = salm.transcribe_chunk(chunk_path, prompt)
            except RuntimeError as exc:
                if "cuda out of memory" in str(exc).lower() and chunk.duration > 24.0:
                    shrink_end = chunk.start + chunk.duration * 0.8
                    LOGGER.warning(
                        "CUDA OOM on chunk %s; retrying with shorter window %.2fs",
                        index,
                        shrink_end - chunk.start,
                    )
                    self._cut_chunk(cleaned, chunk.start, shrink_end, chunk_path)
                    chunks[index] = Chunk(chunk.start, shrink_end)
                    chunk_ranges[-1] = (chunk.start, shrink_end)
                    prompt = salm.build_prompt(carry, keywords)
                    text = salm.transcribe_chunk(chunk_path, prompt)
                else:
                    raise
            canary_texts.append(text)
            carry = update_carry_sentences(
                carry,
                text,
                self._config.salm_context.carry_sentences,
            )

            if parakeet:
                transcript, word_times = parakeet.transcribe_with_word_times(chunk_path)
                parakeet_texts.append(transcript)
                parakeet_wts.append(word_times)

        mfa_word_times: List[List[Tuple[str, float, float]]] = []
        if self._enable_mfa and self._config.alignment.dict_path:
            payload = [
                (chunk.start, chunk.end, path, text)
                for chunk, path, text in zip(chunks, chunk_paths, canary_texts)
            ]
            try:
                mfa_word_times = align_chunks_with_mfa(
                    payload,
                    self._config.alignment.acoustic_model,
                    self._config.alignment.dict_path,
                )
            except Exception as exc:
                LOGGER.warning("MFA failed, falling back to uniform CTM: %s", exc)
                mfa_word_times = []
        elif self._enable_mfa:
            LOGGER.warning("MFA enabled but dictionary path missing; skipping alignment")

        canary_ctms: List[List[str]] = []
        parakeet_ctms: List[List[str]] = []
        for idx, chunk in enumerate(chunks):
            utt_id = f"utt_{idx:05d}"
            chunk_duration = chunk.duration
            words = canary_texts[idx].split()
            if mfa_word_times and idx < len(mfa_word_times) and mfa_word_times[idx]:
                canary_ctm = ctm_from_word_times(mfa_word_times[idx], 0.0, utt_id)
            else:
                canary_ctm = ctm_from_uniform_words(words, 0.0, chunk_duration, utt_id)
            canary_ctms.append(canary_ctm)
            if parakeet:
                wt = parakeet_wts[idx] if idx < len(parakeet_wts) else []
                if wt:
                    parakeet_ctm = ctm_from_word_times(wt, 0.0, utt_id)
                else:
                    parakeet_ctm = ctm_from_uniform_words(
                        parakeet_texts[idx].split(),
                        0.0,
                        chunk_duration,
                        utt_id,
                    )
                parakeet_ctms.append(parakeet_ctm)

        final_texts = list(canary_texts)
        if parakeet and self._enable_rover and parakeet_ctms:
            try:
                consensus_chunks = combine_per_chunk(
                    chunk_ranges,
                    canary_ctms,
                    parakeet_ctms,
                    method=self._config.combination.method,
                )
            except RoverNotAvailableError as exc:
                LOGGER.warning("ROVER disabled: %s", exc)
            else:
                final_texts = [
                    " ".join(line.split()[4] for line in chunk_ctm)
                    for chunk_ctm in consensus_chunks
                ]

        writer = SRTWriter(self._config.reading)
        output_dir = self._config.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{video_path.stem}.srt"
        segments = [
            SubtitleSegment(
                index=i + 1,
                start=chunk.start,
                end=chunk.end,
                text=final_texts[i].strip() if i < len(final_texts) else "",
            )
            for i, chunk in enumerate(chunks)
        ]
        writer.write(segments, target)
        self._maybe_embed(video_path, target)
        return target

    def run(self, source: Path, recursive: bool) -> List[Path]:
        files = self._collect_media(source, recursive)
        if not files:
            raise FileNotFoundError("No media files found to process")
        return [self._transcribe(file) for file in files]


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
    transcribe.add_argument("--sep-repo", type=Path)
    transcribe.add_argument("--sep-ckpt", type=Path)
    transcribe.add_argument("--sep-cfg", type=Path)
    transcribe.add_argument("--bandit-model-name", type=str)
    transcribe.add_argument("--keywords", type=Path, default=None)
    transcribe.add_argument("--cpu", action="store_true")
    transcribe.add_argument("--min-chunk", type=float, dest="min_chunk")
    transcribe.add_argument("--max-chunk", type=float, dest="max_chunk")
    transcribe.add_argument("--overlap", type=float, dest="chunk_overlap")
    transcribe.add_argument("--with-parakeet", action="store_true")
    transcribe.add_argument("--with-rover", action="store_true")
    transcribe.add_argument("--with-mfa", action="store_true")
    transcribe.add_argument("-v", "--verbose", action="store_true")
    transcribe.add_argument("-q", "--quiet", action="store_true")
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
    if args.min_chunk is not None:
        overrides.setdefault("chunking", {})["min_len"] = float(args.min_chunk)
    if args.max_chunk is not None:
        overrides.setdefault("chunking", {})["max_len"] = float(args.max_chunk)
    if args.chunk_overlap is not None:
        overrides.setdefault("chunking", {})["overlap"] = float(args.chunk_overlap)
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
