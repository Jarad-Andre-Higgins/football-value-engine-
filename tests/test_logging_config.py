"""Unit tests for the rotating-file logging setup."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from src import logging_config


@pytest.fixture(autouse=True)
def _isolate_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Send logs to a temp dir and restore global logging state afterwards."""
    root = logging.getLogger()
    original_level = root.level
    original_hook = sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path)
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def test_records_are_written_to_the_log_file(tmp_path: Path) -> None:
    log = logging_config.setup_logging("unit", console=False)
    log.info("hello engine")
    content = (tmp_path / "unit.log").read_text(encoding="utf-8")
    assert "hello engine" in content
    assert "INFO" in content


def test_setup_is_idempotent() -> None:
    logging_config.setup_logging("unit", console=True)
    logging_config.setup_logging("unit", console=True)
    handlers = logging.getLogger().handlers
    rotating = [h for h in handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating) == 1


def test_log_file_rotates_and_respects_backup_count(tmp_path: Path) -> None:
    log = logging_config.setup_logging(
        "rot", console=False, max_bytes=300, backup_count=2
    )
    for i in range(50):
        log.info("line %03d with enough padding to grow the file quickly", i)
    assert (tmp_path / "rot.log").exists()
    assert (tmp_path / "rot.log.1").exists()
    assert (tmp_path / "rot.log.2").exists()
    assert not (tmp_path / "rot.log.3").exists()


def test_uncaught_exceptions_are_logged(tmp_path: Path) -> None:
    logging_config.setup_logging("crash", console=False)
    error = ValueError("boom")
    sys.excepthook(ValueError, error, error.__traceback__)
    content = (tmp_path / "crash.log").read_text(encoding="utf-8")
    assert "CRITICAL" in content
    assert "Uncaught exception" in content
    assert "boom" in content
