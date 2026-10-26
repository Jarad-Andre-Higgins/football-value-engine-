"""Central logging setup: rotating log files, console output, crash capture.

Every entry point calls :func:`setup_logging` once. Library modules then simply
use `logging.getLogger(__name__)`. Records are written to `logs/<name>.log`
with size-based rotation, and any uncaught exception is recorded (with its
traceback) before the process exits, so the system never fails silently.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from types import TracebackType

from src.config import LOG_DIR

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # rotate at 5 MB
DEFAULT_BACKUP_COUNT = 5  # keep up to ~30 MB of history per script

_FILE_HANDLER_NAME = "football_engine_file"
_CONSOLE_HANDLER_NAME = "football_engine_console"


def setup_logging(
    name: str = "football_engine",
    level: int = logging.INFO,
    *,
    console: bool = True,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> logging.Logger:
    """Configure root logging (idempotent) and return a named logger.

    Calling this again replaces the handlers installed by a previous call, so
    duplicate log lines never appear (a classic notebook and test pitfall).

    Args:
        name: Log file stem (``logs/<name>.log``) and the returned logger name.
        level: Minimum severity recorded.
        console: Also stream records to stderr.
        max_bytes: File size at which the log rotates.
        backup_count: Number of rotated files to keep.

    Returns:
        The logger called `name`.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    _remove_engine_handlers(root)
    root.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(
        LOG_DIR / f"{name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.set_name(_FILE_HANDLER_NAME)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.set_name(_CONSOLE_HANDLER_NAME)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    sys.excepthook = _log_uncaught_exception
    return logging.getLogger(name)


def _remove_engine_handlers(root: logging.Logger) -> None:
    """Detach and close handlers installed by a previous `setup_logging`."""
    engine_names = {_FILE_HANDLER_NAME, _CONSOLE_HANDLER_NAME}
    for handler in list(root.handlers):
        if handler.get_name() in engine_names:
            root.removeHandler(handler)
            handler.close()

def _log_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
)-> None:
    """``sys.excepthook`` replacement that writes crashes to the log file."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger("uncaught").critical(
        "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )
 
