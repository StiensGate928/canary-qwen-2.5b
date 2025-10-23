# Contributing to srtforge-platinum

Thanks for your interest in improving srtforge! This document describes how to
set up a development environment, propose changes, and maintain quality.

## Development workflow

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies via `make setup`.
3. Create a feature branch (`git checkout -b feat/my-change`).
4. Make changes with comprehensive docstrings, type hints, and tests.
5. Run `make lint` and `make test` before submitting a pull request.
6. Open a PR summarising the motivation, design, and testing performed.

## Coding standards

* Python code targets 3.10+ and should use static typing. Prefer dataclasses for
  structured data.
* Follow Black formatting with a line length of 100 and ensure Flake8 and MyPy
  pass cleanly.
* Prefer structured logging via `srtforge.logging.get_logger`.
* Validate user input and fail fast with descriptive error messages.

## Testing

* Unit tests live under `tests/` and are executed with `pytest`.
* Avoid network access or large downloads in tests; use stubs and fixtures.
* Add regression tests whenever fixing a bug or introducing new behaviour.

## Documentation

* Update `README.md` and configuration comments when behaviour changes.
* Keep docstrings concise but informative, noting assumptions and return types.

## Release process

1. Update the version in `src/srtforge/__init__.py`.
2. Update the changelog (create `CHANGELOG.md` if needed).
3. Tag the release (`git tag vX.Y.Z`) and push tags.

## Community standards

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). Please
be respectful and constructive in all project spaces.
