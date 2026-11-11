"""Merge Understat xG into the match table, with cross-source integrity checks.

Stages: parse the saved Understat files, resolve their team spellings to
canonical ids (strictly, with the same registry used for football-data), then
join onto ``matches.parquet``. The join key is league, season and the
home/away pair: each ordered pair meets once per league-season, so no date
matching (and no time-zone trouble) is needed. The two sources must then agree
on the score of nearly every match, which is the proof the join is right.

When Understat is unavailable, ``--xg-source shots`` fills the same columns with
a proxy priced from football-data's shot counts (see :mod:`src.xg_proxy`). It
needs neither the Understat files nor the team registry, and it marks every row
``xg_source = "shots_proxy"`` so nobody mistakes it for real xG.

Usage::

    python -m src.enrich                      # real xG from the saved Understat files
    python -m src.enrich --xg-source shots    # fallback: shot-based proxy
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
from src.xg_proxy import ShotModel, fit_shot_model, shot_xg, xg_per_goal

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
SHOT_PROXY_SOURCE: Final[str] = "shots_proxy"


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


def enrich_with_shots(matches: pd.DataFrame) -> tuple[pd.DataFrame, ShotModel]:
    """Add a shot-based xG proxy in place of Understat's, with the same columns.

    The proxy fills ``home_xg`` and ``away_xg``. The Understat-only columns
    (``understat_id``, non-penalty xG, PPDA and deep completions) stay empty, and
    ``xg_source`` records where the xG came from.

    Args:
        matches: The unified match table from ``python -m src.pipeline``.

    Returns:
        The enriched table (``matches``, then :data:`EXTRA_COLUMNS`, then
        ``xg_source``) and the fitted shot prices.

    Raises:
        ValueError: If the shot data cannot support the calibration (see
            :func:`src.xg_proxy.fit_shot_model`).
    """
    model = fit_shot_model(matches)
    proxy = shot_xg(matches, model)
    extra = {
        column: proxy[column] if column in proxy.columns else float("nan")
        for column in EXTRA_COLUMNS
    }
    return matches.assign(**extra, xg_source=SHOT_PROXY_SOURCE), model


def _run_shots(args: argparse.Namespace, log: logging.Logger) -> int:
    """Write ``matches_enriched.parquet`` with the shot-based proxy; return the exit code."""
    if not args.matches.exists():
        log.error(
            "Match table not found: %s; run python -m src.pipeline first", args.matches
        )
        return 1
    matches = pd.read_parquet(args.matches)
    try:
        enriched, model = enrich_with_shots(matches)
    except ValueError as exc:
        log.error("Cannot build the shot-based xG proxy: %s", exc)
        return 1
    log.warning(
        "FALLBACK in use: xG is a shot-based proxy, NOT Understat xG. Prices fitted on "
        "%s (%d team-matches): %.3f goals per shot on target, %.3f per other shot. "
        "Rebuild without --xg-source shots once Understat data is available.",
        season_label(model.season),
        model.n_observations,
        model.on_target,
        model.off_target,
    )
    unpriced = enriched["home_xg"].isna() | enriched["away_xg"].isna()
    if unpriced.any():
        by_season = enriched[unpriced].groupby("season").size().to_dict()
        log.warning(
            "%d of %d matches have no shot data, so no xG proxy (by season: %s)",
            int(unpriced.sum()),
            len(enriched),
            by_season,
        )
    log.info(
        "Proxy xG per actual goal, by season (near 1.0 is well calibrated):\n%s",
        xg_per_goal(enriched).to_string(float_format="{:.2f}".format),
    )
    log.info(
        "xG coverage by season (%% of matches):\n%s",
        xg_coverage(enriched).to_string(float_format="{:.0f}".format),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    log.info("Wrote %s", args.output)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Understat xG into the match table."
    )
    parser.add_argument("--understat-dir", type=Path, default=UNDERSTAT_DIR)
    parser.add_argument("--matches", type=Path, default=MATCHES_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--xg-output", type=Path, default=XG_OUTPUT)
    parser.add_argument("--output", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument(
        "--xg-source",
        choices=("understat", "shots"),
        default="understat",
        help="understat: real xG from the saved Understat files; shots: a proxy "
        "priced from football-data shot counts, for when Understat is unavailable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build ``matches_enriched.parquet`` and return the process exit code.

    Returns ``0`` on success, ``1`` when data is missing, unresolved or fails
    validation, and ``2`` when the team registry is missing (Understat mode only).
    """
    args = _parse_args(argv)
    log = setup_logging("enrich")
    if args.xg_source == "shots":
        return _run_shots(args, log)
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
