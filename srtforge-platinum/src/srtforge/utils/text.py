"""Utility helpers for text processing."""

from __future__ import annotations

import re
from typing import Iterable, List


def redact_command(command: Iterable[str]) -> str:
    """Return a printable command with obvious secrets redacted."""
    redacted: List[str] = []
    for token in command:
        if any(secret in token.lower() for secret in {"token", "password", "secret"}):
            redacted.append("***")
        else:
            redacted.append(token)
    return " ".join(redacted)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace characters."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    """A light-weight heuristic sentence splitter."""
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


__all__ = ["redact_command", "normalize_whitespace", "split_sentences"]
