"""Build the unified, validated match table from the downloaded results files.

Stages: read each raw file, standardise the columns, resolve team names to
canonical ids (strictly), validate, then write Parquet. Nothing is written when
validation finds an *error*, so later phases can trust
``data/processed/matches.parquet`` without re-checking it.

Usage::

    python -m src.pipeline
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.entities import DEFAULT_REGISTRY_PATH, TeamRegistry
from src.ingest import FOOTBALL_DATA_DIR
from src.leagues import LEAGUES, current_season_start, season_label
from src.logging_config import setup_logging
from src.schema import (
    MATCH_COLUMNS,
    ODDS_COLUMNS,
    ONE_X_TWO_SETS,
    REQUIRED_RAW,
    STAT_COLUMNS,
)

logger = logging.getLogger("src.pipeline")

DEFAULT_OUTPUT: Final[Path] = PROCESSED_DIR / "matches.parquet"
MAX_PLAUSIBLE_GOALS: Final[int] = 10
OVERROUND_RANGE: Final[tuple[float, float]] = (0.90, 1.40)
# Completed seasons that legitimately have fewer matches than n * (n - 1).
KNOWN_SHORT_SEASONS: Final[dict[tuple[str, int], int]] = {("F1", 2019): 279}

_FILE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<league>[A-Z0-9]+)_(?P<start>\d{2})(?P<end>\d{2})\.csv$"
)
_WANTED_RAW: Final[frozenset[str]] = (
    frozenset(REQUIRED_RAW)
    | {"Time", "FTR"}
    | frozenset(STAT_COLUMNS)
    | frozenset(ODDS_COLUMNS)
)
# Odds columns whose coverage per season is reported (standard name -> short label).
_COVERAGE_COLUMNS: Final[dict[str, str]] = {
    "ps_h": "ps",
    "psc_h": "psc",
    "b365_h": "b365",
    "b365c_h": "b365c",
    "avgc_h": "avgc",
    "psc_over25": "psc_ou",
}


class UnresolvedNamesError(ValueError):
    """Raised when team spellings are missing from the registry.

    Attributes:
        problems: One line per unresolved spelling, with the closest suggestions.
    """

    def __init__(self, problems: list[str]) -> None:
        super().__init__(f"{len(problems)} team spelling(s) are not in the registry")
        self.problems = problems


class Severity(StrEnum):
    """How serious a validation finding is."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    Attributes:
        severity: ``ERROR`` blocks the output file; ``WARNING`` is for review.
        code: Short machine-friendly identifier.
        message: Human-readable description, including counts.
    """

    severity: Severity
    code: str
    message: str


ReportFn = Callable[[Severity, str, str], None]


@dataclass(frozen=True)
class BuildResult:
    """Everything produced by :func:`build_match_table`.

    Attributes:
        matches: The unified match table (columns of ``schema.MATCH_COLUMNS``).
        issues: Validation findings.
        columns_found: Raw stat and odds columns seen in at least one file.
    """

    matches: pd.DataFrame
    issues: list[Issue]
    columns_found: set[str]

    @property
    def errors(self) -> list[Issue]:
        """Findings that block writing the table."""
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def parse_file_name(path: Path) -> tuple[str, int]:
    """Return ``(league code, season start year)`` for ``E0_2526.csv``-style names.

    Raises:
        ValueError: If the name does not follow the pattern or the league is unknown.
    """
    found = _FILE_PATTERN.match(path.name)
    if found is None:
        raise ValueError(f"{path.name}: expected a name like E0_2526.csv")
    start, end = int(found["start"]), int(found["end"])
    if end != (start + 1) % 100:
        raise ValueError(
            f"{path.name}: season code {found['start']}{found['end']} is not consecutive"
        )
    if found["league"] not in LEAGUES:
        raise ValueError(f"{path.name}: unknown league code {found['league']}")
    year = (
        1900 + start if start >= 90 else 2000 + start
    )  # football-data begins in 1993/94
    return found["league"], year


def read_raw_file(path: Path) -> tuple[pd.DataFrame, set[str]]:
    """Read one football-data file into standard columns (raw team names kept).

    Args:
        path: A ``<LEAGUE>_<SEASON>.csv`` file from ``python -m src.ingest``.

    Returns:
        The standardised rows, and the raw stat/odds columns the file contained.

    Raises:
        ValueError: If the file name or contents are not as expected.
    """
    league, season = parse_file_name(path)
    try:
        raw = pd.read_csv(
            path,
            usecols=lambda column: str(column).strip() in _WANTED_RAW,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
    except ValueError as exc:
        raise ValueError(f"{path.name}: cannot read results ({exc})") from exc
    raw.columns = [str(column).strip() for column in raw.columns]
    missing = [column for column in REQUIRED_RAW if column not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    raw = raw.dropna(subset=list(REQUIRED_RAW))

    home_goals = pd.to_numeric(raw["FTHG"], errors="coerce")
    away_goals = pd.to_numeric(raw["FTAG"], errors="coerce")
    if home_goals.isna().any() or away_goals.isna().any():
        raise ValueError(f"{path.name}: non-numeric full-time goals")

    columns: dict[str, object] = {
        "league": league,
        "season": season,
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "kickoff_time": _optional_text(raw, "Time"),
        "home_team_raw": raw["HomeTeam"].astype(str).str.strip(),
        "away_team_raw": raw["AwayTeam"].astype(str).str.strip(),
        "home_goals": home_goals.astype("int64"),
        "away_goals": away_goals.astype("int64"),
        "ftr_raw": _optional_text(raw, "FTR"),
    }
    for source, target in STAT_COLUMNS.items():
        columns[target] = _nullable_int(raw, source, path.name)
    for source, target in ODDS_COLUMNS.items():
        if source in raw.columns:
            columns[target] = pd.to_numeric(raw[source], errors="coerce")
        else:
            columns[target] = np.nan
    found = {
        column
        for column in raw.columns
        if column in STAT_COLUMNS or column in ODDS_COLUMNS
    }
    return pd.DataFrame(columns), found


def _optional_text(raw: pd.DataFrame, column: str) -> pd.Series:
    """Return a stripped text column, or an all-missing one if it is absent."""
    if column in raw.columns:
        return raw[column].astype("string").str.strip()
    return pd.Series(pd.NA, index=raw.index, dtype="string")


def _nullable_int(raw: pd.DataFrame, column: str, file_name: str) -> pd.Series:
    """Return an integer column that tolerates gaps, or an all-missing one."""
    if column not in raw.columns:
        return pd.Series(pd.NA, index=raw.index, dtype="Int64")
    numbers = pd.to_numeric(raw[column], errors="coerce")
    try:
        return numbers.astype("Int64")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{file_name}: column {column} has non-integer values"
        ) from exc


# --------------------------------------------------------------------------- #
# Team resolution
# --------------------------------------------------------------------------- #
def resolve_teams(frame: pd.DataFrame, registry: TeamRegistry) -> pd.DataFrame:
    """Add canonical team ids and names; fail loudly on any unknown spelling.

    Raises:
        UnresolvedNamesError: Listing every spelling that did not resolve.
    """
    countries = {code: league.country for code, league in LEAGUES.items()}
    spellings = pd.concat(
        [
            frame[["league", "home_team_raw"]].rename(columns={"home_team_raw": "raw"}),
            frame[["league", "away_team_raw"]].rename(columns={"away_team_raw": "raw"}),
        ]
    ).drop_duplicates()

    resolved: dict[tuple[str, str], str] = {}
    problems: list[str] = []
    for league, raw in spellings.itertuples(index=False):
        country = countries.get(league)
        team_id = registry.resolve(raw, country=country)
        if team_id is not None:
            resolved[(league, raw)] = team_id
            continue
        close = registry.suggest(raw, country=country)
        hint = "; ".join(f"{s.name} ({s.score:.2f})" for s in close) or "no close match"
        problems.append(f"{league} {raw!r}: closest {hint}")
    if problems:
        raise UnresolvedNamesError(problems)

    home_ids = [
        resolved[(lg, raw)] for lg, raw in zip(frame["league"], frame["home_team_raw"])
    ]
    away_ids = [
        resolved[(lg, raw)] for lg, raw in zip(frame["league"], frame["away_team_raw"])
    ]
    return frame.assign(
        home_team_id=home_ids,
        away_team_id=away_ids,
        home_team=[registry.name_of(team_id) for team_id in home_ids],
        away_team=[registry.name_of(team_id) for team_id in away_ids],
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_matches(
    frame: pd.DataFrame, *, today: pd.Timestamp, current_season: int
) -> tuple[pd.DataFrame, list[Issue]]:
    """Check data quality; return the cleaned rows and every finding.

    Errors (which block the output): conflicting duplicates, negative goals,
    a team playing itself, matches dated in the future, and results that
    disagree with the source's own full-time result column. Warnings: exact
    duplicates (dropped), very high scores, odds not above 1.0 (set to missing),
    implausible overrounds, unusual team counts and incomplete finished seasons.

    Args:
        frame: Output of :func:`resolve_teams`.
        today: Matches after this date are errors.
        current_season: Start year of the season in progress (may be partial).
    """
    issues: list[Issue] = []
    data = frame.copy()

    def report(severity: Severity, code: str, message: str) -> None:
        issues.append(Issue(severity, code, message))

    key = ["league", "season", "date", "home_team_id", "away_team_id"]
    exact = data.duplicated([*key, "home_goals", "away_goals"], keep="first")
    if exact.any():
        report(
            Severity.WARNING,
            "duplicate_rows",
            f"{int(exact.sum())} exact duplicate rows dropped",
        )
        data = data.loc[~exact]
    clashing = data.duplicated(key, keep=False)
    if clashing.any():
        report(
            Severity.ERROR,
            "conflicting_duplicates",
            f"{int(clashing.sum())} rows describe one fixture with different scores",
        )

    negative = (data["home_goals"] < 0) | (data["away_goals"] < 0)
    if negative.any():
        report(
            Severity.ERROR,
            "negative_goals",
            f"{int(negative.sum())} rows have negative goals",
        )
    huge = (data["home_goals"] > MAX_PLAUSIBLE_GOALS) | (
        data["away_goals"] > MAX_PLAUSIBLE_GOALS
    )
    if huge.any():
        report(
            Severity.WARNING,
            "very_high_score",
            f"{int(huge.sum())} rows have more than {MAX_PLAUSIBLE_GOALS} goals for one team",
        )
    same_team = data["home_team_id"] == data["away_team_id"]
    if same_team.any():
        report(
            Severity.ERROR,
            "team_plays_itself",
            f"{int(same_team.sum())} rows have the same home and away team",
        )
    future = data["date"] > today
    if future.any():
        report(
            Severity.ERROR,
            "future_dates",
            f"{int(future.sum())} matches are dated after {today.date()}",
        )

    data["result"] = np.where(
        data["home_goals"] > data["away_goals"],
        "H",
        np.where(data["home_goals"] == data["away_goals"], "D", "A"),
    )
    if "ftr_raw" in data.columns:
        stated = data["ftr_raw"].fillna("").astype(str)
        mismatch = (stated != "") & (stated != data["result"])
        if mismatch.any():
            report(
                Severity.ERROR,
                "result_mismatch",
                f"{int(mismatch.sum())} rows disagree with the source's own result column",
            )

    _clean_odds(data, report)
    _check_seasons(data, current_season, report)
    return data, issues


def _clean_odds(data: pd.DataFrame, report: ReportFn) -> None:
    """Null out impossible odds and flag implausible 1X2 overrounds (in place)."""
    columns = [column for column in ODDS_COLUMNS.values() if column in data.columns]
    invalid_counts: dict[str, int] = {}
    for column in columns:
        invalid = data[column] <= 1.0
        if invalid.any():
            invalid_counts[column] = int(invalid.sum())
            data.loc[invalid, column] = np.nan
    if invalid_counts:
        report(
            Severity.WARNING,
            "odds_not_above_one",
            f"{sum(invalid_counts.values())} odds <= 1.0 set to missing in: {', '.join(sorted(invalid_counts))}",
        )

    low, high = OVERROUND_RANGE
    odd_sets: dict[str, int] = {}
    for triple in ONE_X_TWO_SETS:
        if not all(column in data.columns for column in triple):
            continue
        total = sum(1.0 / data[column] for column in triple)
        outside = total.notna() & ((total < low) | (total > high))
        if outside.any():
            odd_sets[triple[0].removesuffix("_h")] = int(outside.sum())
    if odd_sets:
        detail = ", ".join(
            f"{name}: {count}" for name, count in sorted(odd_sets.items())
        )
        report(
            Severity.WARNING,
            "odd_overround",
            f"1X2 odds with an implausible overround ({detail})",
        )


def _check_seasons(data: pd.DataFrame, current_season: int, report: ReportFn) -> None:
    """Flag unusual team counts and incomplete finished seasons."""
    for (league, season), group in data.groupby(["league", "season"]):
        label = f"{league} {season_label(int(season))}"
        n_teams = len(set(group["home_team_id"]) | set(group["away_team_id"]))
        if n_teams not in (18, 20):
            report(Severity.WARNING, "unusual_team_count", f"{label}: {n_teams} teams")
        if int(season) >= current_season:
            continue  # the season in progress is expected to be partial
        expected = KNOWN_SHORT_SEASONS.get(
            (league, int(season)), n_teams * (n_teams - 1)
        )
        if len(group) != expected:
            report(
                Severity.WARNING,
                "incomplete_season",
                f"{label}: {len(group)} of {expected} expected matches",
            )


def odds_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Percentage of matches per season that carry each key odds column.

    Pinnacle's closing odds (``psc``) are the benchmark for closing line value,
    so their coverage decides which seasons Phase 6 can backtest.
    """
    present = table[list(_COVERAGE_COLUMNS)].notna().astype(float) * 100.0
    coverage = present.groupby(table["season"]).mean()
    coverage.columns = list(_COVERAGE_COLUMNS.values())
    coverage.insert(0, "matches", table.groupby("season").size())
    return coverage


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_match_table(
    raw_dir: Path,
    registry: TeamRegistry,
    *,
    today: pd.Timestamp | None = None,
    current_season: int | None = None,
) -> BuildResult:
    """Read, resolve, validate and assemble the unified match table.

    Raises:
        FileNotFoundError: If ``raw_dir`` holds no CSV files.
        ValueError: If a file is malformed.
        UnresolvedNamesError: If any team spelling is not in the registry.
    """
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files in {raw_dir}; run python -m src.ingest first"
        )
    frames, found = [], set()
    for path in paths:
        frame, columns = read_raw_file(path)
        frames.append(frame)
        found |= columns
    resolved = resolve_teams(pd.concat(frames, ignore_index=True), registry)
    cleaned, issues = validate_matches(
        resolved,
        today=today if today is not None else pd.Timestamp.today().normalize(),
        current_season=current_season
        if current_season is not None
        else current_season_start(),
    )
    return BuildResult(_finalise(cleaned), issues, found)


def _finalise(data: pd.DataFrame) -> pd.DataFrame:
    """Sort, add the deterministic ``match_id`` and select the output columns."""
    table = data.sort_values(
        ["date", "league", "home_team_id"], kind="stable"
    ).reset_index(drop=True)
    table["match_id"] = (
        table["league"]
        + "-"
        + table["season"].astype(str)
        + "-"
        + table["date"].dt.strftime("%Y%m%d")
        + "-"
        + table["home_team_id"]
        + "-"
        + table["away_team_id"]
    )
    return table.loc[:, list(MATCH_COLUMNS)]


def load_matches_table(path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load the processed match table written by ``python -m src.pipeline``."""
    return pd.read_parquet(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/processed/matches.parquet from the downloaded results."
    )
    parser.add_argument("--raw-dir", type=Path, default=FOOTBALL_DATA_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the match table and return the process exit code.

    Returns ``0`` on success, ``1`` when the data cannot be built or fails
    validation, and ``2`` when the team registry is missing.
    """
    args = _parse_args(argv)
    log = setup_logging("pipeline")
    if not args.registry.exists():
        log.error("Registry not found: %s", args.registry)
        return 2
    registry = TeamRegistry.from_yaml(args.registry)

    try:
        result = build_match_table(args.raw_dir, registry)
    except UnresolvedNamesError as exc:
        log.error("%s; run python -m src.entities for the review queue", exc)
        for problem in exc.problems:
            log.error("  %s", problem)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        log.error("Cannot build the match table: %s", exc)
        return 1

    table = result.matches
    log.info(
        "Built %d matches: %d leagues, seasons %s to %s",
        len(table),
        table["league"].nunique(),
        season_label(int(table["season"].min())),
        season_label(int(table["season"].max())),
    )
    for issue in result.issues:
        if issue.severity is Severity.ERROR:
            log.error("[%s] %s", issue.code, issue.message)
        else:
            log.warning("[%s] %s", issue.code, issue.message)
    never_seen = sorted((set(STAT_COLUMNS) | set(ODDS_COLUMNS)) - result.columns_found)
    if never_seen:
        log.warning("Raw columns never found in any file: %s", ", ".join(never_seen))
    coverage = odds_coverage(table)
    log.info(
        "Odds coverage by season (%% of matches):\n%s",
        coverage.to_string(float_format="{:.0f}".format),
    )

    if result.errors:
        log.error(
            "Not writing %s: %d validation error(s)", args.output, len(result.errors)
        )
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    log.info(
        "Wrote %s (%d rows, %d columns)", args.output, len(table), len(table.columns)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
