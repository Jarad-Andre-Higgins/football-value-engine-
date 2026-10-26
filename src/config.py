"""Project-wide paths and constants.

Every path is derived from this file's location, so scripts behave the same
whether they are launched from the repo root, an IDE, or cron (which starts in
a different working directory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
SAMPLE_DIR: Final[Path] = DATA_DIR / "sample"

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
