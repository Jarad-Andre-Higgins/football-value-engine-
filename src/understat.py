"""Download and parse Understat expected-goals data.

Understat publishes per-match xG, plus team-level PPDA and deep completions,
but has no official API. Its pages are empty shells that load JSON from
internal endpoints. This module uses ``getLeagueData`` (one request per
league-season, about 250 KB), identifies itself honestly, and caches completed
seasons so each file is fetched once. The raw JSON is saved untouched, so if
the response shape ever changes it can be diagnosed from disk (``--peek``)
without downloading again.

Be a good guest: this is an unofficial endpoint, used here for personal
research. Do not hammer it and do not redistribute the data.

When Understat is down it answers every request with a web page instead of data
(a "We will come back soon" maintenance page was seen in November 2026). The
downloader says what the page is, and stops after a few failures in a row
instead of trying every file. Finished files stay cached, so run it again later.
Files saved elsewhere, such as an earlier notebook's output or a private Kaggle
dataset (loose files or zip archives), can be restored with ``--import-from``.

Usage::

    python -m src.understat                  # five leagues, 2019/20 to now
    python -m src.understat --peek E0 2025   # show the structure of a saved file
    python -m src.understat --import-from /kaggle/input/understat-cache
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import RAW_DIR
from src.ingest import USER_AGENT, FetchResult, FetchStatus
from src.leagues import current_season_start, season_code, season_label
from src.logging_config import setup_logging

logger = logging.getLogger("src.understat")

BASE_URL: Final[str] = "https://understat.com"
UNDERSTAT_DIR: Final[Path] = RAW_DIR / "understat"
DEFAULT_FIRST_SEASON: Final[int] = 2019
#: football-data division code -> Understat league slug.
SLUGS: Final[dict[str, str]] = {
    "E0": "EPL",
    "SP1": "La_liga",
    "I1": "Serie_A",
    "D1": "Bundesliga",
    "F1": "Ligue_1",
}
_FILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<league>[A-Z0-9]+)_(?P<start>\d{2})(?P<end>\d{2})\.json$"
)

#: Columns returned by :func:`parse_league_payload`.
XG_COLUMNS: Final[tuple[str, ...]] = (
    "league",
    "season",
    "understat_id",
    "date",
    "home_team_raw",
    "away_team_raw",
    "home_goals_u",
    "away_goals_u",
    "home_xg",
    "away_xg",
    "home_npxg",
    "away_npxg",
    "home_ppda",
    "away_ppda",
    "home_deep",
    "away_deep",
)


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
NOT_JSON: Final[str] = "response is not JSON"
#: Phrases (lower case) that mean "the site is down" on a page that should have been data.
_MAINTENANCE_PHRASES: Final[tuple[str, ...]] = (
    "back soon",
    "under maintenance",
    "maintenance mode",
    "scheduled maintenance",
    "temporarily unavailable",
)
#: Phrases that mean a firewall or bot check answered instead of the site.
_FIREWALL_PHRASES: Final[tuple[str, ...]] = (
    "just a moment",
    "attention required",
    "checking your browser",
    "verify you are human",
    "captcha",
    "access denied",
)
_SCAN_LIMIT: Final[int] = 2_000_000  # bytes of a response read to explain it
_SPACES: Final[re.Pattern[str]] = re.compile(r"\s+")
_TAG: Final[re.Pattern[str]] = re.compile(r"<[^>]*>")
#: Parts of a page that are not text a person reads (the title is read separately).
_CODE_BLOCK: Final[re.Pattern[str]] = re.compile(
    r"<(script|style|head|title)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_TITLE: Final[re.Pattern[str]] = re.compile(
    r"<title[^>]*>(.*?)</title\s*>", re.IGNORECASE | re.DOTALL
)
_HTML_START: Final[re.Pattern[bytes]] = re.compile(
    rb"<(!doctype|html|head|body)\b", re.IGNORECASE
)


def payload_problem(content: bytes) -> str | None:
    """Return why ``content`` is not a usable league payload, or ``None`` if it is.

    Understat can answer ``200`` with an empty HTML shell when an endpoint
    changes, which would otherwise fail silently, so the shape is checked.
    """
    try:
        payload = json.loads(content)
    except ValueError:
        return f"{NOT_JSON} (has the endpoint changed?)"
    if not isinstance(payload, dict) or not {"teams", "dates"} <= payload.keys():
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        return f"unexpected payload shape (keys: {keys})"
    if not isinstance(payload["dates"], list) or not payload["dates"]:
        return "payload has no fixtures"
    return None


def _squash(text: str) -> str:
    """Unescape HTML entities and collapse all white space to single spaces."""
    return _SPACES.sub(" ", html.unescape(text)).strip()


def describe_response(status: int, headers: Mapping[str, str], content: bytes) -> str:
    """Say what a response that is not usable data actually is.

    "Not JSON" alone cannot tell a site that is down from one that blocks this
    machine or one that moved its endpoint. The answer is in the page that came
    back, so this reads it.

    Args:
        status: HTTP status code.
        headers: Response headers.
        content: Response body.

    Returns:
        One sentence, such as ``Understat is showing a maintenance page ("We will
        come back soon"): the site is down, so try again later``.
    """
    if not content.strip():
        return "the response is empty"
    raw = content[:_SCAN_LIMIT].decode("utf-8", errors="replace")
    found = _TITLE.search(raw)
    title = _squash(found.group(1)) if found else ""
    words = _squash(_TAG.sub(" ", _CODE_BLOCK.sub(" ", raw)))
    lowered = f"{title} {words}".lower()
    if any(phrase in lowered for phrase in _MAINTENANCE_PHRASES):
        return (
            f'Understat is showing a maintenance page ("{(words or title)[:60]}"): '
            "the site is down, so try again later"
        )
    behind_cloudflare = any(name.lower() == "cf-ray" for name in headers)
    if any(phrase in lowered for phrase in _FIREWALL_PHRASES) or (
        status == 403 and behind_cloudflare
    ):
        return (
            "a firewall or bot-check page answered instead of Understat; "
            "this machine may be blocked"
        )
    if status >= 500:
        return "the server answered with an error, so the site may be down"
    if _HTML_START.search(content[:2048]):
        shown = f'title "{title[:60]}"' if title else f'text "{words[:60]}"'
        return (
            f"a web page came back instead of data ({shown}); has the endpoint changed?"
        )
    return (
        f"the response is not data (it starts with {content[:30]!r}); "
        "has the endpoint changed?"
    )


def unusable_reason(
    content: bytes, status: int = 200, headers: Mapping[str, str] | None = None
) -> str | None:
    """Like :func:`payload_problem`, but says what a non-JSON response actually is."""
    problem = payload_problem(content)
    if problem is not None and problem.startswith(NOT_JSON):
        return f"{NOT_JSON}: {describe_response(status, headers or {}, content)}"
    return problem


class UnderstatDownloader:
    """Polite, retrying, caching downloader for Understat league files.

    Args:
        raw_dir: Folder the JSON files are saved in.
        base_url: Site root; files come from ``<base_url>/getLeagueData/...``.
        retries: Retries per request on connection errors and 429/5xx responses.
        backoff: Back-off factor between retries, in seconds.
        delay: Pause between consecutive network requests, in seconds.
        timeout: ``(connect, read)`` timeouts in seconds.
    """

    def __init__(
        self,
        raw_dir: Path = UNDERSTAT_DIR,
        base_url: str = BASE_URL,
        *,
        retries: int = 3,
        backoff: float = 1.0,
        delay: float = 1.0,
        timeout: tuple[float, float] = (10.0, 30.0),
    ) -> None:
        self.raw_dir = raw_dir
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self._requests_made = 0
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "X-Requested-With": "XMLHttpRequest",  # required, else a 404 is returned
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Referer": f"{self.base_url}/",
            }
        )
        retry = Retry(
            total=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
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
        """Return where the file for ``league`` and ``season`` is stored."""
        return self.raw_dir / f"{league}_{season_code(season)}.json"

    def fetch(
        self, league: str, season: int, *, current_season: int, force: bool = False
    ) -> FetchResult:
        """Fetch one league-season file, reusing the cache for finished seasons.

        Network and content problems are reported as ``FAILED`` rather than raised.
        """
        path = self.local_path(league, season)
        if path.exists() and season < current_season and not force:
            return FetchResult(league, season, FetchStatus.CACHED, path)

        url = f"{self.base_url}/getLeagueData/{SLUGS[league]}/{season}"
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
        status, headers, body = response.status_code, response.headers, response.content
        if not response.ok:
            why = describe_response(status, headers, body)
            return FetchResult(
                league, season, FetchStatus.FAILED, detail=f"HTTP {status}: {why}"
            )
        problem = unusable_reason(body, status, headers)
        if problem is not None:
            kind = headers.get("Content-Type", "no content type").split(";")[0].strip()
            detail = f"{problem} [HTTP {status}, {kind}, {len(body):,} bytes]"
            return FetchResult(league, season, FetchStatus.FAILED, detail=detail)

        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".part")
        temp.write_bytes(response.content)
        os.replace(temp, path)
        return FetchResult(
            league,
            season,
            FetchStatus.DOWNLOADED,
            path,
            f"({len(response.content) // 1024} KB)",
        )

    def download_all(
        self,
        leagues: Iterable[str],
        seasons: Iterable[int],
        *,
        current_season: int,
        force: bool = False,
        max_failures: int = 3,
    ) -> list[FetchResult]:
        """Fetch every league-season combination and log each outcome.

        When ``max_failures`` requests fail in a row the run stops, because a site
        that is down or blocking this machine fails every file the same way and
        trying the rest only wastes time. Files served from the cache do not count
        either way, and any answer from the site (a download or a 404) resets the
        count.

        Args:
            leagues: football-data division codes.
            seasons: Season start years.
            current_season: The season in progress, which is always refreshed.
            force: Re-download files already on disk.
            max_failures: Stop after this many failures in a row (0 never stops).

        Returns:
            One result for each file that was tried, in order. A run that stopped
            early returns fewer results than there are files.
        """
        league_list, season_list = list(leagues), list(seasons)
        results: list[FetchResult] = []
        in_a_row = 0
        for league in league_list:
            for season in season_list:
                result = self.fetch(
                    league, season, current_season=current_season, force=force
                )
                label = f"{league} {season_label(season)}"
                if result.status is FetchStatus.FAILED:
                    logger.error("%s: FAILED - %s", label, result.detail)
                    in_a_row += 1
                elif result.status is FetchStatus.MISSING:
                    logger.warning("%s: not published (HTTP 404)", label)
                else:
                    logger.info(
                        "%s: %s",
                        label,
                        f"{result.status.value} {result.detail}".strip(),
                    )
                if result.status in (FetchStatus.DOWNLOADED, FetchStatus.MISSING):
                    in_a_row = 0
                results.append(result)
                if max_failures and in_a_row >= max_failures:
                    untried = len(league_list) * len(season_list) - len(results)
                    if untried:
                        logger.error(
                            "Stopping after %d failures in a row; %d files were not "
                            "tried. Understat is probably down or not reachable from "
                            "this machine. Finished files are cached, so run the "
                            "command again later.",
                            in_a_row,
                            untried,
                        )
                    return results
        return results


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _number(value: object) -> float:
    """Convert Understat's numbers (sometimes strings, sometimes null) to float."""
    if value is None:
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _is_played(item: Mapping[str, Any]) -> bool:
    """True for a finished match: flagged as a result and with a score."""
    flag = item.get("isResult")
    if isinstance(flag, str):
        flag = flag.strip().lower() == "true"
    goals = item.get("goals") or {}
    return bool(flag) and goals.get("h") is not None and goals.get("a") is not None


def _history_rows(teams: object) -> pd.DataFrame:
    """Flatten ``teams[*].history`` into one row per team per match."""
    entries = teams.values() if isinstance(teams, Mapping) else teams
    records = []
    for team in entries:  # type: ignore[attr-defined]
        for row in team.get("history", []):
            ppda = row.get("ppda") or {}
            defence = _number(ppda.get("def"))
            records.append(
                {
                    "team": team["title"],
                    "h_a": row["h_a"],
                    "day": pd.Timestamp(row["date"]).normalize(),
                    "npxg": _number(row.get("npxG")),
                    "deep": _number(row.get("deep")),
                    "ppda": _number(ppda.get("att")) / defence
                    if defence > 0
                    else float("nan"),
                }
            )
    columns = ["team", "h_a", "day", "npxg", "deep", "ppda"]
    frame = pd.DataFrame(records, columns=columns)
    return frame.drop_duplicates(["team", "h_a", "day"])


def parse_league_payload(
    payload: Mapping[str, Any], league: str, season: int
) -> pd.DataFrame:
    """Turn one ``getLeagueData`` response into one row per played match.

    Per-side npxG, PPDA (passes allowed per defensive action; lower means more
    pressing) and deep completions come from the team histories and are matched
    to fixtures by team and calendar day.

    Args:
        payload: The decoded JSON (keys ``teams``, ``players``, ``dates``).
        league: football-data division code, such as ``E0``.
        season: Season start year.

    Returns:
        DataFrame with the columns in :data:`XG_COLUMNS`.

    Raises:
        ValueError: If the payload does not have the expected structure.
    """
    try:
        fixtures = [item for item in payload["dates"] if _is_played(item)]
        history = _history_rows(payload["teams"])
        rows = [
            {
                "league": league,
                "season": season,
                "understat_id": str(item["id"]),
                "date": pd.Timestamp(item["datetime"]),
                "home_team_raw": str(item["h"]["title"]).strip(),
                "away_team_raw": str(item["a"]["title"]).strip(),
                "home_goals_u": int(item["goals"]["h"]),
                "away_goals_u": int(item["goals"]["a"]),
                "home_xg": _number(item["xG"]["h"]),
                "away_xg": _number(item["xG"]["a"]),
            }
            for item in fixtures
        ]
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"{league} {season}: unexpected Understat payload ({type(exc).__name__}: {exc}); "
            "run python -m src.understat --peek to inspect the saved file"
        ) from exc

    frame = pd.DataFrame(rows, columns=list(XG_COLUMNS[:10]))
    frame["day"] = frame["date"].dt.normalize()
    for side, flag in (("home", "h"), ("away", "a")):
        part = history[history["h_a"] == flag].rename(
            columns={
                "npxg": f"{side}_npxg",
                "deep": f"{side}_deep",
                "ppda": f"{side}_ppda",
            }
        )
        frame = frame.merge(
            part[["team", "day", f"{side}_npxg", f"{side}_deep", f"{side}_ppda"]],
            how="left",
            left_on=[f"{side}_team_raw", "day"],
            right_on=["team", "day"],
        ).drop(columns="team")
    return frame.loc[:, list(XG_COLUMNS)]


def parse_file_name(path: Path) -> tuple[str, int]:
    """Return ``(league code, season start year)`` for ``E0_2526.json``-style names."""
    found = _FILE_PATTERN.match(path.name)
    if found is None or found["league"] not in SLUGS:
        raise ValueError(f"{path.name}: expected a name like E0_2526.json")
    start = int(found["start"])
    if int(found["end"]) != (start + 1) % 100:
        raise ValueError(f"{path.name}: season code is not consecutive")
    return found["league"], (1900 if start >= 90 else 2000) + start


def read_league_file(path: Path) -> pd.DataFrame:
    """Read and parse one saved league file.

    Raises:
        ValueError: If the name, JSON or structure is not as expected.
    """
    league, season = parse_file_name(path)
    try:
        payload = json.loads(path.read_bytes())
    except ValueError as exc:
        raise ValueError(f"{path.name}: not valid JSON ({exc})") from exc
    return parse_league_payload(payload, league, season)


@dataclass(frozen=True)
class ImportReport:
    """What :func:`import_files` did.

    Attributes:
        imported: Names of the files copied into the raw folder.
        present: Names of valid files that were already there and were kept.
        rejected: File name -> why it was refused, for files with an Understat
            name that are not usable data, and zip archives that cannot be read.
        ignored: Number of other ``.json`` files skipped because their names are
            not those of Understat league files.
    """

    imported: tuple[str, ...]
    present: tuple[str, ...]
    rejected: dict[str, str]
    ignored: int


def _zip_members(path: Path) -> list[tuple[str, bytes]]:
    """The ``.json`` files inside a zip archive, as ``(file name, content)``.

    Raises:
        zipfile.BadZipFile: If ``path`` is not a readable zip archive.
    """
    with zipfile.ZipFile(path) as archive:
        return [
            (Path(member.filename).name, archive.read(member))
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".json")
        ]


def import_files(
    source: Path, raw_dir: Path = UNDERSTAT_DIR, *, force: bool = False
) -> ImportReport:
    """Copy saved Understat league files from ``source`` into ``raw_dir``.

    This restores a cache made elsewhere: an earlier notebook's output, a private
    Kaggle dataset, or a computer that could reach Understat when this one could
    not. Each file is checked as a download would be, so a saved error page never
    gets in. Sub-folders of ``source`` are searched, ``.zip`` archives are opened,
    and files already in ``raw_dir`` are kept unless ``force`` is set.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        NotADirectoryError: If ``source`` is not a folder.
    """
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    if not source.is_dir():
        raise NotADirectoryError(f"{source} is not a folder")
    imported: list[str] = []
    present: list[str] = []
    rejected: dict[str, str] = {}
    ignored = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            found = [(path.name, path.read_bytes())]
        elif suffix == ".zip":
            try:
                found = _zip_members(path)
            except zipfile.BadZipFile:
                rejected[path.name] = "not a readable zip archive"
                continue
        else:
            continue
        for name, content in found:
            try:
                parse_file_name(Path(name))
            except ValueError:
                ignored += 1
                continue
            problem = unusable_reason(content)
            if problem is not None:
                rejected[name] = problem
                continue
            target = raw_dir / name
            if target.exists() and not force:
                present.append(name)
                continue
            raw_dir.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".part")
            temp.write_bytes(content)
            os.replace(temp, target)
            imported.append(name)
    return ImportReport(tuple(imported), tuple(present), rejected, ignored)


def describe_file(path: Path) -> str:
    """Summarise a saved file's structure, for diagnosing format changes."""
    payload = json.loads(path.read_bytes())
    lines = [f"{path.name}: top-level keys {sorted(payload)}"]
    played = next((item for item in payload["dates"] if _is_played(item)), None)
    lines.append(f"fixtures: {len(payload['dates'])}; first played match:")
    lines.append(json.dumps(played, indent=2))
    teams = payload["teams"]
    first = next(iter(teams.values() if isinstance(teams, dict) else teams))
    lines.append(
        f"teams: {len(teams)}; first team keys {sorted(first)}; first history row:"
    )
    lines.append(json.dumps(first.get("history", [None])[0], indent=2))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Understat xG data into data/raw/understat/."
    )
    parser.add_argument(
        "--leagues", nargs="+", choices=sorted(SLUGS), default=list(SLUGS)
    )
    parser.add_argument("--from-season", type=int, default=DEFAULT_FIRST_SEASON)
    parser.add_argument("--to-season", type=int, help="default: the season in progress")
    parser.add_argument(
        "--force", action="store_true", help="re-download files already on disk"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="seconds between requests"
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="stop after this many failed downloads in a row (0: never stop)",
    )
    parser.add_argument("--raw-dir", type=Path, default=UNDERSTAT_DIR)
    parser.add_argument("--base-url", default=BASE_URL)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--peek",
        nargs=2,
        metavar=("LEAGUE", "SEASON"),
        help="show the structure of an already downloaded file, e.g. --peek E0 2025",
    )
    mode.add_argument(
        "--import-from",
        type=Path,
        metavar="DIR",
        help="copy saved league files (E0_2526.json ..., or zip archives of them) "
        "from DIR into the raw folder",
    )
    args = parser.parse_args(argv)
    if args.to_season is not None and args.to_season < args.from_season:
        parser.error("--to-season must not be earlier than --from-season")
    if args.max_failures < 0:
        parser.error("--max-failures cannot be negative")
    return args


def _run_import(source: Path, raw_dir: Path, force: bool, log: logging.Logger) -> int:
    """Restore saved files with :func:`import_files` and report; return the exit code."""
    try:
        report = import_files(source, raw_dir, force=force)
    except OSError as exc:
        log.error("Cannot import from %s: %s", source, exc)
        return 1
    for name, reason in report.rejected.items():
        log.error("%s: rejected - %s", name, reason)
    log.info(
        "Import from %s: imported=%d already_present=%d rejected=%d other_json_ignored=%d",
        source,
        len(report.imported),
        len(report.present),
        len(report.rejected),
        report.ignored,
    )
    if not (report.imported or report.present or report.rejected):
        log.error(
            "No Understat league files found in %s (expected names like E0_2526.json)",
            source,
        )
        return 1
    return 1 if report.rejected else 0


def main(argv: list[str] | None = None) -> int:
    """Download, ``--peek`` at or ``--import-from`` Understat files; return the exit code."""
    args = _parse_args(argv)
    log = setup_logging("understat")
    if args.import_from:
        return _run_import(args.import_from, args.raw_dir, args.force, log)
    if args.peek:
        league, season = args.peek[0], int(args.peek[1])
        path = args.raw_dir / f"{league}_{season_code(season)}.json"
        if not path.exists():
            log.error("%s does not exist; download it first", path)
            return 1
        log.info("\n%s", describe_file(path))
        return 0

    current = current_season_start()
    last = current if args.to_season is None else args.to_season
    log.info(
        "Understat: leagues=%s seasons=%s to %s",
        ",".join(args.leagues),
        season_label(args.from_season),
        season_label(last),
    )
    with UnderstatDownloader(
        args.raw_dir, args.base_url, delay=args.delay
    ) as downloader:
        results = downloader.download_all(
            args.leagues,
            range(args.from_season, last + 1),
            current_season=current,
            force=args.force,
            max_failures=args.max_failures,
        )
    counts = Counter(result.status for result in results)
    not_tried = len(args.leagues) * (last - args.from_season + 1) - len(results)
    log.info(
        "Understat finished: downloaded=%d cached=%d missing=%d failed=%d not_tried=%d",
        counts[FetchStatus.DOWNLOADED],
        counts[FetchStatus.CACHED],
        counts[FetchStatus.MISSING],
        counts[FetchStatus.FAILED],
        not_tried,
    )
    return 1 if counts[FetchStatus.FAILED] else 0  # an early stop always has failures


if __name__ == "__main__":
    sys.exit(main())
