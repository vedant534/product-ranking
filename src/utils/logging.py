"""Consistent command-line logging configuration."""

import logging
from typing import Union


def configure_logging(level: Union[int, str] = logging.INFO) -> None:
    """Configure concise process-wide logging for scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named project logger."""
    return logging.getLogger(name)
