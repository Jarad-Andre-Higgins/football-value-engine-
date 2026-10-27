"""Download historical results and odds from football-data.co.uk.

Completed seasons are cached on disk and never fetched twice; the current
season is refreshed on every run because the site updates it weekly. Each file
is validated before it is saved and written atomically, so a failed download
can never leave a corrupt CSV behind. One bad file never stops the run: it is
logged and reported, and the exit code tells cron whether anything failed.

Usage::

    python -m src.ingest                      # top-5 leagues, 2019/20 to now
    python -m src.ingest --leagues E0 SP1     # only some leagues
    python -m src.ingest --from-season 2023   # shorter history
    python -m src.ingest --force              # re-download everything
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Final, Self

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import RAW_DIR
from src.leagues import (
    LEAGUES,
    TOP_FIVE,
    current_season_start,
    season_code,
    season_label,
)
from src.logging_config import setup_logging

logger = logging.getLogger("src.ingest")  # stable name, even under python -m

DEFAULT_BASE_URL: Final[str] = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_DATA_DIR: Final[Path] = RAW_DIR / "football_data"
DEFAULT_FIRST_SEASON: Final[int] = 2019
USER_AGENT: Final[str] = "football-value-engine/0.1 (personal research project)"
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
)


class FetchStatus(StrEnum):
    """Outcome of fetching one league-season file."""

    DOWNLOADED = "downloaded"
    CACHED = "cached"
    MISSING = "missing"
    FAILED = "failed"


@dataclass(frozen=True)
class FetchResult:
    """What happened to one league-season file.

    Attributes:
        league: football-data.co.uk division code.
        season: Season start year (2025 means 2025/26).
        status: Outcome of the fetch.
        path: Local file, when one exists.
        detail: Extra information, such as an error message or row count.
    """

    league: str
    season: int
    status: FetchStatus
    path: Path | None = None
    detail: str = ""


def looks_like_results_csv(content: bytes) -> bool:
    """Return True if `content` starts with a football-data results header.

    Guards against saving an HTML error page or a truncated response as data.
    """
    header = content.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace")
    columns = {column.strip() for column in header.split(",")}
    return REQUIRED_COLUMNS <= columns


class FootballDataDownloader:
    """Polite, retrying downloader for football-data.co.uk CSV files.

    Args:
        raw_dir: Folder the CSV files are saved in.
        base_url: Root URL; files live at `<base_url>/<season>/<league>.csv`.
        retries: Retries per request on connection errors and 429/5xx responses.
        backoff: Back-off factor between retries, in seconds.
        delay: Pause between consecutive network requests, in seconds.
        timeout: `(connect, read)` timeouts in seconds.
    """

    def __init__(
        self,
        raw_dir: Path = FOOTBALL_DATA_DIR,
        base_url: str = DEFAULT_BASE_URL,
        *,
        retries: int = 3,
        backoff: float = 1.0,
        delay: float = 0.5,
        timeout: tuple[float, float] = (10.0, 30.0),
    ) -> None:
        self.raw_dir = raw_dir
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self._requests_made = 0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    def local_path(self, league: str, season: int) -> Path:
        """Return where the file for `league` and `season` is stored."""
        return self.raw_dir / f"{league}_{season_code(season)}.csv"

    def fetch(
        self,
        league: str,
        season: int,
        *,
        current_season: int,
        force: bool = False,
    ) -> FetchResult:
        """Fetch one league-season file, using the cache when it is safe to.

        Completed seasons never change, so an existing local copy is reused
        unless `force` is set. The season in progress is always refreshed.

        Args:
            league: football-data.co.uk division code, such as `E0`.
            season: Season start year.
            current_season: Start year of the season in progress.
            force: Re-download even if a cached copy exists.

        Returns:
            A :class:`FetchResult`. Network and content problems are reported
            as `FAILED` rather than raised.
        """
        path = self.local_path(league, season)
        if path.exists() and season < current_season and not force:
            return FetchResult(league, season, FetchStatus.CACHED, path)

        url = f"{self.base_url}/{season_code(season)}/{league}.csv"
        if self._requests_made:
            time.sleep(self.delay)
        self._requests_made += 1
        try:
            response = self._session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return FetchResult(league, season, FetchStatus.FAILED, detail=detail)

        if response.status_code == 404:
            return FetchResult(league, season, FetchStatus.MISSING)
        if not response.ok:
            detail = f"HTTP {response.status_code}"
            return FetchResult(league, season, FetchStatus.FAILED, detail=detail)
        if not looks_like_results_csv(response.content):
            detail = "response is not a football-data results CSV"
            return FetchResult(league, season, FetchStatus.FAILED, detail=detail)

        self._write_atomic(path, response.content)
        rows = max(len(response.content.splitlines()) - 1, 0)
        return FetchResult(
            league, season, FetchStatus.DOWNLOADED, path, detail=f"({rows} rows)"
        )

    def download_all(
        self,
        leagues: Iterable[str],
        seasons: Iterable[int],
        *,
        current_season: int,
        force: bool = False,
    ) -> list[FetchResult]:
        """Fetch every league-season combination and log each outcome."""
        season_list = list(seasons)
        results: list[FetchResult] = []
        for league in leagues:
            for season in season_list:
                result = self.fetch(
                    league, season, current_season=current_season, force=force
                )
                _log_result(result)
                results.append(result)
        return results

    @staticmethod
    def _write_atomic(path: Path, content: bytes) -> None:
        """Write `content` so readers never see a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".part")
        temp.write_bytes(content)
        os.replace(temp, path)


def _log_result(result: FetchResult) -> None:
    """Log one fetch outcome at a severity that matches its meaning."""
    label = f"{result.league} {season_label(result.season)}"
    if result.status is FetchStatus.FAILED:
        logger.error("%s: FAILED - %s", label, result.detail)
    elif result.status is FetchStatus.MISSING:
        logger.warning("%s: not published (HTTP 404)", label)
    else:
        message = f"{result.status.value} {result.detail}".strip()
        logger.info("%s: %s", label, message)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download football-data.co.uk results into data/raw/."
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        choices=sorted(LEAGUES),
        default=list(TOP_FIVE),
        help="division codes to download (default: the top five leagues)",
    )
    parser.add_argument(
        "--from-season",
        type=int,
        default=DEFAULT_FIRST_SEASON,
        help=f"first season start year (default: {DEFAULT_FIRST_SEASON})",
    )
    parser.add_argument(
        "--to-season",
        type=int,
        help="last season start year (default: the season in progress)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download files already on disk"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="seconds to pause between requests (default: 0.5)",
    )
    parser.add_argument("--raw-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args(argv)
    if args.to_season is not None and args.to_season < args.from_season:
        parser.error("--to-season must not be earlier than --from-season")
    return args


def main(argv: list[str] | None = None) -> int:
    """Download the requested files and return the process exit code.

    Returns `0` when nothing failed, `1` when at least one file failed.
    """
    args = _parse_args(argv)
    log = setup_logging("ingest")
    current = current_season_start()
    last = current if args.to_season is None else args.to_season
    seasons = range(args.from_season, last + 1)
    log.info(
        "Ingest: leagues=%s seasons=%s to %s -> %s",
        ",".join(args.leagues),
        season_label(args.from_season),
        season_label(last),
        args.raw_dir,
    )

    with FootballDataDownloader(
        args.raw_dir, args.base_url, delay=args.delay
    ) as downloader:
        results = downloader.download_all(
            args.leagues, seasons, current_season=current, force=args.force
        )

    counts = Counter(result.status for result in results)
    log.info(
        "Ingest finished: downloaded=%d cached=%d missing=%d failed=%d",
        counts[FetchStatus.DOWNLOADED],
        counts[FetchStatus.CACHED],
        counts[FetchStatus.MISSING],
        counts[FetchStatus.FAILED],
    )
    return 1 if counts[FetchStatus.FAILED] else 0


if __name__ == "__main__":
    sys.exit(main())
