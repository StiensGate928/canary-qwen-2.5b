"""Logging helpers with structured defaults."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def configure_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configure application logging."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    level = logging.INFO
    if quiet:
        level = logging.WARNING
    if verbose:
        level = logging.DEBUG

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger("srtforge")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False

    _LOGGER = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger with the configured settings."""
    base = configure_logging()
    return base.getChild(name)
