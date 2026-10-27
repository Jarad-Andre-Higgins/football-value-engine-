"""Merge Understat xG into the match table, with cross-source integrity checks.

Stages: parse the saved Understat files, resolve their team spellings to
canonical ids (strictly, with the same registry used for football-data), then
join onto ``matches.parquet``. The join key is league, season and the
home/away pair: each ordered pair meets once per league-season, so no date
matching (and no time-zone trouble) is needed. The two sources must then agree
on the score of nearly every match, which is the proof the join is right.

Usage::

    python -m src.enrich
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import pandas as pd

from src.config import PROCESSED_DIR
from src.entities import DEFAULT_REGISTRY_PATH, TeamRegistry
from src.leagues import current_season_start, season_label
from src.logging_config import setup_logging
from src.pipeline import (
    DEFAULT_OUTPUT as MATCHES_PATH,
)
from src.pipeline import (
    Issue,
    Severity,
    UnresolvedNamesError,
    resolve_teams,
)
from src.understat import UNDERSTAT_DIR, XG_COLUMNS, read_league_file

logger = logging.getLogger("src.enrich")

XG_OUTPUT: Final[Path] = PROCESSED_DIR / "xg_matches.parquet"
ENRICHED_OUTPUT: Final[Path] = PROCESSED_DIR / "matches_enriched.parquet"
KEY: Final[list[str]] = ["league", "season", "home_team_id", "away_team_id"]
EXTRA_COLUMNS: Final[list[str]] = [
    "understat_id",
    "home_xg",
    "away_xg",
    "home_npxg",
    "away_npxg",
    "home_ppda",
    "away_ppda",
    "home_deep",
    "away_deep",
]
MAX_SCORE_DISAGREEMENT: Final[float] = 0.01  # above this share, the join is suspect
MAX_DATE_GAP_DAYS: Final[int] = 3
MAX_PLAUSIBLE_XG: Final[float] = 8.0


def build_xg_table(raw_dir: Path, registry: TeamRegistry) -> pd.DataFrame:
    """Parse every saved Understat file and attach canonical team ids.

    Raises:
        FileNotFoundError: If ``raw_dir`` holds no JSON files.
        ValueError: If a file is malformed.
        UnresolvedNamesError: If any team spelling is not in the registry.
    """
    paths = sorted(raw_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No JSON files in {raw_dir}; run python -m src.understat first"
        )
    frame = pd.concat([read_league_file(path) for path in paths], ignore_index=True)
    return resolve_teams(frame, registry)


def merge_xg(
    matches: pd.DataFrame, xg: pd.DataFrame, *, current_season: int
) -> tuple[pd.DataFrame, list[Issue]]:
    """Join xG onto ``matches`` and cross-check the two sources.

    Errors (which block the output): duplicate Understat fixtures, negative xG,
    and score disagreement above 1% of joined matches. Warnings: a few score
    or date disagreements, implausible xG, Understat matches absent from the
    match table, and finished-season matches without xG.

    Args:
        matches: The unified match table from ``python -m src.pipeline``.
        xg: Output of :func:`build_xg_table`.
        current_season: Start year of the season in progress (may be partial).

    Returns:
        The enriched table (``matches`` plus :data:`EXTRA_COLUMNS`) and findings.
    """
    issues: list[Issue] = []

    def report(severity: Severity, code: str, message: str) -> None:
        issues.append(Issue(severity, code, message))

    repeated = xg.duplicated(KEY, keep=False)
    if repeated.any():
        report(
            Severity.ERROR,
            "duplicate_xg_fixtures",
            f"{int(repeated.sum())} Understat rows share a fixture",
        )
        xg = xg.drop_duplicates(KEY)
    xg = xg.rename(columns={"date": "date_u"})

    merged = matches.merge(
        xg[[*KEY, "date_u", "home_goals_u", "away_goals_u", *EXTRA_COLUMNS]],
        on=KEY,
        how="left",
        indicator=True,
    )
    joined = merged["_merge"] == "both"

    disagree = joined & (
        (merged["home_goals"] != merged["home_goals_u"])
        | (merged["away_goals"] != merged["away_goals_u"])
    )
    if disagree.any():
        share = disagree.sum() / max(int(joined.sum()), 1)
        examples = ", ".join(merged.loc[disagree, "match_id"].head(3))
        severity = (
            Severity.ERROR if share > MAX_SCORE_DISAGREEMENT else Severity.WARNING
        )
        report(
            severity,
            "score_disagreement",
            f"{int(disagree.sum())} joined matches ({share:.1%}) have different scores in the two sources, e.g. {examples}",
        )

    gap = (merged["date"] - merged["date_u"]).abs().dt.days > MAX_DATE_GAP_DAYS
    if (joined & gap).any():
        report(
            Severity.WARNING,
            "date_gap",
            f"{int((joined & gap).sum())} joined matches differ by more than {MAX_DATE_GAP_DAYS} days between sources",
        )

    xg_values = merged[["home_xg", "away_xg"]]
    if (xg_values < 0).any().any():
        report(Severity.ERROR, "negative_xg", "some xG values are negative")
    if (xg_values > MAX_PLAUSIBLE_XG).any().any():
        report(
            Severity.WARNING,
            "very_high_xg",
            f"some xG values exceed {MAX_PLAUSIBLE_XG}",
        )

    orphans = xg.merge(matches[KEY], on=KEY, how="left", indicator=True)
    orphans = orphans[orphans["_merge"] == "left_only"]
    if not orphans.empty:
        by_season = orphans.groupby("season").size().to_dict()
        report(
            Severity.WARNING,
            "xg_without_match",
            f"{len(orphans)} Understat matches are not in the match table (by season: {by_season})",
        )

    missing = (~joined) & (merged["season"] < current_season)
    if missing.any():
        counts = merged[missing].groupby(["league", "season"]).size()
        detail = ", ".join(
            f"{lg} {season_label(int(s))}: {n}" for (lg, s), n in counts.items()
        )
        report(
            Severity.WARNING,
            "match_without_xg",
            f"{int(missing.sum())} finished-season matches have no xG ({detail})",
        )

    enriched = merged.drop(columns=["_merge", "date_u", "home_goals_u", "away_goals_u"])
    return enriched, issues


def xg_coverage(enriched: pd.DataFrame) -> pd.DataFrame:
    """Percentage of matches per season carrying xG, PPDA and deep completions."""
    present = (
        enriched[["home_xg", "home_ppda", "home_deep"]].notna().astype(float) * 100.0
    )
    coverage = present.groupby(enriched["season"]).mean()
    coverage.columns = ["xg", "ppda", "deep"]
    coverage.insert(0, "matches", enriched.groupby("season").size())
    return coverage


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Understat xG into the match table."
    )
    parser.add_argument("--understat-dir", type=Path, default=UNDERSTAT_DIR)
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--xg-output", type=Path, default=XG_OUTPUT)
    parser.add_argument("--output", type=Path, default=ENRICHED_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build ``matches_enriched.parquet`` and return the process exit code.

    Returns ``0`` on success, ``1`` when data is missing, unresolved or fails
    validation, and ``2`` when the team registry is missing.
    """
    args = _parse_args(argv)
    log = setup_logging("enrich")
    if not args.registry.exists():
        log.error("Registry not found: %s", args.registry)
        return 2
    if not args.matches.exists():
        log.error(
            "Match table not found: %s; run python -m src.pipeline first", args.matches
        )
        return 1
    registry = TeamRegistry.from_yaml(args.registry)

    try:
        xg = build_xg_table(args.understat_dir, registry)
    except UnresolvedNamesError as exc:
        log.error("%s; add them as aliases in %s and re-run", exc, args.registry)
        for problem in exc.problems:
            log.error("  %s", problem)
        return 1
    except (FileNotFoundError, ValueError) as exc:
        log.error("Cannot build the xG table: %s", exc)
        return 1

    matches = pd.read_parquet(args.matches)
    enriched, issues = merge_xg(matches, xg, current_season=current_season_start())
    log.info(
        "Understat: %d played matches parsed; match table: %d rows",
        len(xg),
        len(matches),
    )
    for issue in issues:
        if issue.severity is Severity.ERROR:
            log.error("[%s] %s", issue.code, issue.message)
        else:
            log.warning("[%s] %s", issue.code, issue.message)
    log.info(
        "xG coverage by season (%% of matches):\n%s",
        xg_coverage(enriched).to_string(float_format="{:.0f}".format),
    )

    errors = [issue for issue in issues if issue.severity is Severity.ERROR]
    if errors:
        log.error("Not writing outputs: %d validation error(s)", len(errors))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    xg.loc[:, [*XG_COLUMNS, "home_team_id", "away_team_id"]].to_parquet(
        args.xg_output, index=False
    )
    enriched.to_parquet(args.output, index=False)
    log.info("Wrote %s and %s", args.xg_output, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
