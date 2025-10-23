# srtforge-platinum

srtforge Platinum is an offline-first subtitle generator that targets demanding
use cases such as 20–30 minute English-dub anime episodes. The project couples
NVIDIA's Canary-Qwen-2.5B SALM model with Parakeet RNNT/TDT, combines their
outputs with SCTK ROVER, and emits production-ready SRT files that can be muxed
back into the original media container.

## Why Platinum?

* **Front-end audio excellence** – Prefer center channels on 5.1 mixes, fall
  back to Demucs vocal separation, then apply denoise/normalize/loudness match
  before ASR.
* **Long-form resilience** – Canary SALM is chunked in 30–36 s windows with
  1 s overlap, contextual carry-over, and optional keyword prompting to keep
  character names straight across episodes.
* **Dual ASR blend** – Parakeet RNNT/TDT provides stable word timings and n-best
  alternatives. ROVER consolidates SALM and Parakeet CTMs into a consensus
  transcript with lower WER.
* **Optional alignment polish** – Montreal Forced Aligner can refine timing for
  the final segments when ultimate accuracy is required.

## Key features

* Python 3.10+ CLI (`srtforge`) with recursive folder processing and MKV embed
  support via FFmpeg.
* Configurable pipeline via YAML (`configs/default.yaml`) and per-run CLI
  overrides.
* Structured logging, informative errors when dependencies are missing, and
  simulated behaviour for development environments without access to large
  models.
* Dockerfiles for CUDA-capable and CPU-only systems.
* GitHub Actions CI covering linting, typing, and tests.

## Installation

### Prerequisites

* Python 3.10 or newer.
* FFmpeg, SoX, SCTK (ROVER). Use `scripts/verify_binaries.sh` to confirm.
* NVIDIA drivers + CUDA toolkit when using GPU acceleration.
* Substantial disk space for model checkpoints (~25 GB total).

### Clone & install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The heavy ML dependencies (PyTorch, NeMo, Demucs) are part of
`requirements.txt`. On CUDA hosts you may prefer to install the wheel matching
your driver manually, e.g.

```bash
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then install NeMo from source to track the current SALM/Parakeet builds:

```bash
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo
pip install -e ".[asr]"
```

Demucs and other Python dependencies are handled by `pip install -r requirements.txt`.

### Model downloads

Use the NeMo CLI to pull the latest models:

```bash
python -m nemo.collections.asr.models.language_modeling.salm.download --model nvidia/canary-qwen-2.5b
python -m nemo.collections.asr.models.asr_model --model nvidia/parakeet-rnnt-1.1b --download
```

You can verify available model IDs with either the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/quick-start#download-a-model)
or the NeMo `nemo_toolkit` helper:

```bash
huggingface-cli list-files nvidia/canary-qwen-2.5b --revision main --tree
python -m nemo.collections.asr.models.asr_model --list_models | grep parakeet
```

Set the `NEMO_CACHE_DIR` environment variable if you want to keep the models in
a dedicated offline directory.

## Usage

Transcribe a single file and embed the resulting subtitles in-place:

```bash
srtforge transcribe "/path/video.mkv" --lang eng --with-parakeet --with-rover --prefer-center --embed
```

Process a folder recursively, keeping subtitles as external `.srt` files:

```bash
srtforge transcribe "/path/series" --recursive --with-parakeet --with-rover --prefer-center
```

Important CLI toggles:

* `--cpu` forces CPU inference (slow but useful for verification).
* `--keywords <file>` injects keywords/character names into the SALM prompt.
* `--with-mfa` enables the Montreal Forced Aligner integration for timing
  refinement.

## Configuration

`configs/default.yaml` controls pipeline behaviour. Highlights include:

* `frontend.prefer_center`: favour 5.1 center extraction; pair with
  `--no-prefer-center` to override.
* `chunking.max_len` / `chunking.overlap`: SALM window size and overlap.
* `salm_context.carry_sentences`: number of sentences persisted between chunks.
* `reading.max_chars_per_line`: subtitle formatting constraints used by the SRT
  renderer.

Provide a keyword list via `configs/keywords.sample.txt` as a template. Lines
starting with `#` are ignored.

## Pipeline overview

1. **Audio extraction** – FFmpeg grabs the preferred stream. The CUDA Dockerfile
   uses `pan=mono|c0=FC` to isolate the center channel when present.
2. **Optional separation** – Demucs vocal extraction engages when the center
   channel is absent.
3. **Preprocessing** – FFmpeg filters (high-pass, RNNoise, afftdn, de-esser,
   compressor, loudnorm) produce a clean 16 kHz mono file for ASR.
4. **Chunking** – VAD + silence detection boundaries are merged and chunked into
   30–36 s windows with 1 s overlap.
5. **SALM inference** – Canary SALM runs greedily with prompt carry-over and
   optional keyword list.
6. **Parakeet inference** – Provides word-level timestamps and n-best outputs.
7. **System combination** – Canary and Parakeet CTMs flow into SCTK ROVER using
   `-m maxconf` for consensus.
8. **Alignment (optional)** – MFA polishes timings with forced alignment.
9. **Rendering** – Subtitle lines are wrapped to 42 chars, limited to two lines,
   and written to SRT. `--embed` invokes FFmpeg muxing to create
   `<video>.subs.mkv` without re-encoding.

The repository ships with lightweight stub implementations for the heavy ML
components so unit tests can run without GPU resources. When deployed with the
full dependencies installed the same interfaces orchestrate the real models.

## Development

* `make setup` – create a virtual environment and install dependencies.
* `make lint` – run Black, Flake8, and MyPy.
* `make test` – execute pytest.
* `make verify` – confirm external binaries (FFmpeg, SoX, ROVER) are present.
* `make docker-gpu` – build the CUDA Docker image.

GitHub Actions CI mirrors the above checks on Python 3.10 and 3.11.

## Offline operation

After downloading the models and building SCTK once, the pipeline runs fully
offline. Configure the NeMo cache via `NEMO_CACHE_DIR` and set
`XDG_CACHE_HOME`/`DEMUCSPATH` as desired to keep checkpoints in a persistent
location. The CLI stores intermediate files in `configs/default.yaml -> paths.temp_dir`.

## Troubleshooting

* Missing `rover`: run `scripts/build_sctk.sh` or install SCTK from your
  package manager.
* NeMo import errors: ensure `nemo_toolkit` and `hydra-core` are installed in the
  active environment. The optional `.[dev]` extra brings in lint/test tooling but
  not NeMo itself.
* GPU memory pressure: reduce `chunking.max_len` to ~24 s windows and consider
  `--cpu` for functional testing.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations. Issues and
pull requests are welcome!

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
