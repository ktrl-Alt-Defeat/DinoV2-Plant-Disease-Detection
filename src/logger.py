"""Centralised logging configuration.

A single named logger acts as the project root; every module obtains a child of
it through :func:`get_logger`. :func:`configure_logging` is idempotent: calling
it more than once replaces the existing handlers instead of stacking duplicates,
which keeps log lines from being emitted several times.
"""

import logging
import sys
from pathlib import Path
from typing import Final, TextIO

from src.paths import ensure_directory

#: Name of the project-wide root logger.
LOGGER_NAME: Final[str] = "dinov2_leafcare"

#: File name used when file logging is enabled without an explicit name.
DEFAULT_LOG_FILENAME: Final[str] = "run.log"

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_UTF8_ALIASES: Final[frozenset[str]] = frozenset({"utf8", "utf-8"})


def configure_console_encoding(stream: TextIO | None = None) -> None:
    """Switch ``stream`` (``sys.stdout`` by default) to UTF-8 when it is not already.

    Windows consoles default to a legacy code page, which makes non-ASCII log
    records raise ``UnicodeEncodeError``. Reconfiguring the existing stream keeps
    the object identity intact, so redirection and test capture keep working.
    """
    target = sys.stdout if stream is None else stream
    reconfigure = getattr(target, "reconfigure", None)
    if reconfigure is None:
        return
    encoding = (getattr(target, "encoding", "") or "").lower()
    if encoding in _UTF8_ALIASES:
        return
    reconfigure(encoding="utf-8", errors="replace")


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: str | Path | None = None,
    filename: str = DEFAULT_LOG_FILENAME,
) -> logging.Logger:
    """Configure and return the project root logger.

    Args:
        level: Name of a standard logging level, case insensitive.
        log_dir: Directory receiving the log file. When ``None``, only console
            logging is enabled. The directory is created if needed.
        filename: Name of the log file inside ``log_dir``.

    Raises:
        ValueError: If ``level`` is not a known logging level name.
        OSError: If the log directory or file cannot be created.
    """
    configure_console_encoding()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False
    _detach_handlers(logger)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_file = ensure_directory(log_dir) / filename
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the project root logger, or a child of it when ``name`` is given.

    Children inherit the level and handlers installed by
    :func:`configure_logging`, so modules can request a logger at import time
    without worrying about initialisation order.
    """
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def shutdown_logging() -> None:
    """Detach and close every handler of the project root logger.

    Useful for tests and for any process that needs to release the log file
    before the directory is removed.
    """
    _detach_handlers(logging.getLogger(LOGGER_NAME))


def _detach_handlers(logger: logging.Logger) -> None:
    """Remove and close all handlers currently attached to ``logger``."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _resolve_level(level: str) -> int:
    """Translate a level name into its numeric value.

    Raises:
        ValueError: If the name is unknown.
    """
    names_to_levels = logging.getLevelNamesMapping()
    key = level.strip().upper()
    if key not in names_to_levels:
        supported = ", ".join(sorted(names_to_levels))
        raise ValueError(f"Unknown logging level '{level}'. Supported levels: {supported}.")
    return names_to_levels[key]
