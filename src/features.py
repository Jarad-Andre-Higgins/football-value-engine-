"""Leak-free pre-match features for the xG models.

Every feature of a match is computed from information available strictly
before kick-off. Three mechanisms enforce this:

1. Team form is a time-decayed average (EWMA, half-life measured in days) over
   the team's own history, then shifted one match with ``.shift(1)``, so a row
   never sees its own result.
2. Elo ratings are recorded *before* they are updated with the result.
3. League-wide scoring averages are aggregated per calendar day and shifted
   one match day, so matches played the same day never see each other.

Policy: bookmaker odds are never features, because the model has to be
independent of the market it is later compared with. Columns starting with
``target_`` are outcomes and must never be used as inputs.

``python -m src.features`` builds ``data/processed/features.parquet`` and, by
default, runs a leakage audit: it scrambles every outcome from a cut-off date
onward and confirms that no feature of an earlier match moves.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.elo import EloConfig, compute_elo
from src.enrich import ENRICHED_OUTPUT
from src.leagues import season_label
from src.logging_config import setup_logging

logger = logging.getLogger("src.features")

FEATURES_OUTPUT: Final[Path] = PROCESSED_DIR / "features.parquet"
HALF_LIVES_DAYS: Final[tuple[int, int]] = (45, 180)  # about 6 and 25 matches
LONG: Final[int] = HALF_LIVES_DAYS[-1]
VENUE_HALF_LIFE_DAYS: Final[int] = 270
LEAGUE_HALF_LIFE_DAYS: Final[int] = 365
REST_CAP_DAYS: Final[int] = 14

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "league",
    "season",
    "date",
    "home_team_id",
    "away_team_id",
    "home_goals",
    "away_goals",
)
# long-format metric -> (home-side match column, away-side match column)
_PAIRS: Final[dict[str, tuple[str, str]]] = {
    "goals": ("home_goals", "away_goals"),
    "xg": ("home_xg", "away_xg"),
    "shots": ("home_shots", "away_shots"),
    "sot": ("home_shots_on_target", "away_shots_on_target"),
    "deep": ("home_deep", "away_deep"),
    "ppda": ("home_ppda", "away_ppda"),
}
FORM_METRICS: Final[tuple[str, ...]] = (
    *(f"{metric}_{way}" for metric in _PAIRS for way in ("for", "against")),
    "points",
)
#: Features computed per team; the match table carries them as home_* and away_*.
TEAM_FEATURES: Final[tuple[str, ...]] = (
    *(
        f"{metric}_ew{half_life}"
        for metric in FORM_METRICS
        for half_life in HALF_LIVES_DAYS
    ),
    f"finishing_ew{LONG}",  # goals scored minus xG created: luck or finishing skill
    f"keeping_ew{LONG}",  # goals conceded minus xG faced: luck or goalkeeping
    f"venue_gd_ew{VENUE_HALF_LIFE_DAYS}",  # goal difference at this venue type only
    f"venue_xgd_ew{VENUE_HALF_LIFE_DAYS}",
    "rest_days",
    "prior_matches",
    "season_matches",
)
MATCH_FEATURES: Final[tuple[str, ...]] = (
    "home_elo",
    "away_elo",
    "elo_win_prob_home",
    "league_avg_goals_home",
    "league_avg_goals_away",
    "league_avg_xg",
    "home_xg_exp",
    "away_xg_exp",
    "rest_diff",
)
META_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "league",
    "season",
    "date",
    "home_team_id",
    "away_team_id",
)
TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "target_home_goals",
    "target_away_goals",
    "target_home_xg",
    "target_away_xg",
    "target_result",
)


def feature_columns() -> list[str]:
    """Names of every model input, in output order (never includes targets)."""
    team = [f"{side}_{name}" for side in ("home", "away") for name in TEAM_FEATURES]
    return [*team, *MATCH_FEATURES]


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _numeric(matches: pd.DataFrame, column: str) -> pd.Series:
    """A float column, or an all-missing one when the column is absent."""
    if column not in matches.columns:
        return pd.Series(np.nan, index=matches.index, dtype="float64")
    return pd.to_numeric(matches[column], errors="coerce").astype("float64")


def _points(home_goals: pd.Series, away_goals: pd.Series, side: str) -> pd.Series:
    """League points earned by ``side`` (3 win, 1 draw, 0 loss)."""
    own, opp = (home_goals, away_goals) if side == "home" else (away_goals, home_goals)
    return pd.Series(
        np.where(own > opp, 3.0, np.where(own == opp, 1.0, 0.0)), index=own.index
    )


def to_team_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match, from that team's point of view.

    Columns ``<metric>_for`` hold the team's own figure and ``<metric>_against``
    the opponent's. Rows are sorted by team and date.
    """
    frames = []
    for side in ("home", "away"):
        columns: dict[str, object] = {
            "match_id": matches["match_id"],
            "side": side,
            "team_id": matches[f"{side}_team_id"],
            "league": matches["league"],
            "season": matches["season"],
            "date": matches["date"],
        }
        for metric, (home_col, away_col) in _PAIRS.items():
            own, opp = (home_col, away_col) if side == "home" else (away_col, home_col)
            columns[f"{metric}_for"] = _numeric(matches, own)
            columns[f"{metric}_against"] = _numeric(matches, opp)
        columns["points"] = _points(
            _numeric(matches, "home_goals"), _numeric(matches, "away_goals"), side
        )
        frames.append(pd.DataFrame(columns))
    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["team_id", "date"], kind="stable").reset_index(drop=True)


def _ewma_before(
    values: pd.Series, dates: pd.Series, half_life_days: float
) -> pd.Series:
    """Time-decayed mean of the *previous* rows: ``ewm`` followed by ``shift(1)``."""
    smoothed = values.ewm(
        halflife=pd.Timedelta(half_life_days, unit="D"), times=dates.loc[values.index]
    ).mean()
    return smoothed.shift(1)


def add_form_features(long: pd.DataFrame) -> pd.DataFrame:
    """Add time-decayed pre-match form of every metric, per team and window."""
    dates = long["date"]
    by_team = long.groupby("team_id", sort=False)
    added: dict[str, pd.Series] = {}
    for metric in FORM_METRICS:
        for half_life in HALF_LIVES_DAYS:
            added[f"{metric}_ew{half_life}"] = by_team[metric].transform(
                lambda s, h=half_life: _ewma_before(s, dates, h)
            )
    added[f"finishing_ew{LONG}"] = (
        added[f"goals_for_ew{LONG}"] - added[f"xg_for_ew{LONG}"]
    )
    added[f"keeping_ew{LONG}"] = (
        added[f"goals_against_ew{LONG}"] - added[f"xg_against_ew{LONG}"]
    )
    return pd.concat([long, pd.DataFrame(added, index=long.index)], axis=1)


def add_venue_features(long: pd.DataFrame) -> pd.DataFrame:
    """Add form measured only in matches at the same venue type (home or away)."""
    dates = long["date"]
    helper = pd.DataFrame(
        {
            "gd": long["goals_for"] - long["goals_against"],
            "xgd": long["xg_for"] - long["xg_against"],
            "team_id": long["team_id"],
            "side": long["side"],
        }
    )
    by_venue = helper.groupby(["team_id", "side"], sort=False)
    added = {
        f"venue_{name}_ew{VENUE_HALF_LIFE_DAYS}": by_venue[name].transform(
            lambda s: _ewma_before(s, dates, VENUE_HALF_LIFE_DAYS)
        )
        for name in ("gd", "xgd")
    }
    return pd.concat([long, pd.DataFrame(added, index=long.index)], axis=1)


def add_schedule_features(long: pd.DataFrame) -> pd.DataFrame:
    """Add days of rest (capped) and counts of earlier matches for each team."""
    by_team = long.groupby("team_id", sort=False)
    added = pd.DataFrame(
        {
            "rest_days": by_team["date"].diff().dt.days.clip(upper=REST_CAP_DAYS),
            "prior_matches": by_team.cumcount(),
            "season_matches": long.groupby(
                ["team_id", "season"], sort=False
            ).cumcount(),
        },
        index=long.index,
    )
    return pd.concat([long, added], axis=1)


def league_context(matches: pd.DataFrame) -> pd.DataFrame:
    """League scoring averages as known before each match day.

    Matches are summed per league and calendar day, smoothed with a one-year
    half-life, and shifted one match day, so a match never sees any match
    played on its own date.

    Returns:
        ``league``, ``date`` and ``league_avg_goals_home``, ``league_avg_goals_away``,
        ``league_avg_xg`` (per team per match).
    """
    xg_total = _numeric(matches, "home_xg") + _numeric(matches, "away_xg")
    frame = pd.DataFrame(
        {
            "league": matches["league"],
            "date": matches["date"],
            "n": 1.0,
            "home_goals": _numeric(matches, "home_goals"),
            "away_goals": _numeric(matches, "away_goals"),
            "xg_total": xg_total.fillna(0.0),
            "xg_n": 2.0 * xg_total.notna(),
        }
    )
    daily = frame.groupby(["league", "date"], as_index=False, sort=True).sum()
    pieces = []
    for _, group in daily.groupby("league", sort=False):
        smooth = (
            group[["n", "home_goals", "away_goals", "xg_total", "xg_n"]]
            .ewm(
                halflife=pd.Timedelta(LEAGUE_HALF_LIFE_DAYS, unit="D"),
                times=group["date"],
            )
            .mean()
        )
        averages = pd.DataFrame(
            {
                "league_avg_goals_home": smooth["home_goals"] / smooth["n"],
                "league_avg_goals_away": smooth["away_goals"] / smooth["n"],
                "league_avg_xg": (smooth["xg_total"] / smooth["xg_n"]).where(
                    smooth["xg_n"] > 0
                ),
            }
        ).shift(1)
        pieces.append(pd.concat([group[["league", "date"]], averages], axis=1))
    return pd.concat(pieces, ignore_index=True)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_features(
    matches: pd.DataFrame, elo_config: EloConfig | None = None
) -> pd.DataFrame:
    """Build the leak-free feature table, one row per match.

    Args:
        matches: The enriched match table (``matches_enriched.parquet``).
        elo_config: Elo constants; defaults to :class:`~src.elo.EloConfig`.

    Returns:
        ``META_COLUMNS``, then :func:`feature_columns`, then ``TARGET_COLUMNS``.

    Raises:
        ValueError: On missing columns, repeated match ids, or a team playing
            twice on one date.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing columns: {missing}")
    data = matches.sort_values(
        ["date", "league", "home_team_id"], kind="stable"
    ).reset_index(drop=True)
    if data["match_id"].duplicated().any():
        raise ValueError("match_id must be unique")

    long = to_team_matches(data)
    repeats = long.duplicated(["team_id", "date"], keep=False)
    if repeats.any():
        example = long.loc[repeats, ["team_id", "date"]].iloc[0]
        raise ValueError(
            f"a team plays twice on one date, e.g. {example['team_id']} on {example['date']:%Y-%m-%d}"
        )
    long = add_schedule_features(add_venue_features(add_form_features(long)))

    per_team = list(TEAM_FEATURES)
    home = (
        long[long["side"] == "home"].set_index("match_id")[per_team].add_prefix("home_")
    )
    away = (
        long[long["side"] == "away"].set_index("match_id")[per_team].add_prefix("away_")
    )
    wide = data.set_index("match_id").join(home).join(away)

    elo = compute_elo(data, elo_config).set_index("match_id")
    wide = wide.join(elo)
    wide = wide.reset_index().merge(
        league_context(data), on=["league", "date"], how="left"
    )
    wide["rest_diff"] = wide["home_rest_days"] - wide["away_rest_days"]
    wide["home_xg_exp"] = (
        wide[f"home_xg_for_ew{LONG}"]
        * wide[f"away_xg_against_ew{LONG}"]
        / wide["league_avg_xg"]
    )
    wide["away_xg_exp"] = (
        wide[f"away_xg_for_ew{LONG}"]
        * wide[f"home_xg_against_ew{LONG}"]
        / wide["league_avg_xg"]
    )

    home_goals, away_goals = _numeric(wide, "home_goals"), _numeric(wide, "away_goals")
    wide["target_home_goals"] = home_goals
    wide["target_away_goals"] = away_goals
    wide["target_home_xg"] = _numeric(wide, "home_xg")
    wide["target_away_xg"] = _numeric(wide, "away_xg")
    wide["target_result"] = np.where(
        home_goals > away_goals, "H", np.where(home_goals == away_goals, "D", "A")
    )
    return wide.loc[:, [*META_COLUMNS, *feature_columns(), *TARGET_COLUMNS]]


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
_OUTCOME_COLUMNS: Final[tuple[str, ...]] = (
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_npxg",
    "away_npxg",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_corners",
    "away_corners",
    "home_ppda",
    "away_ppda",
    "home_deep",
    "away_deep",
)


def scramble_outcomes(
    matches: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 0
) -> pd.DataFrame:
    """Shuffle every outcome column of matches dated on or after ``cutoff``."""
    rng = np.random.default_rng(seed)
    scrambled = matches.copy()
    future = scrambled["date"] >= cutoff
    for column in _OUTCOME_COLUMNS:
        if column in scrambled.columns:
            scrambled.loc[future, column] = rng.permutation(
                scrambled.loc[future, column].to_numpy()
            )
    return scrambled


def leakage_audit(matches: pd.DataFrame, cutoff_fraction: float = 0.7) -> list[str]:
    """Check that no feature depends on outcomes at or after a cut-off date.

    Builds the features twice: once as given and once after scrambling every
    outcome from the cut-off date onward. Features of matches played on or
    before the cut-off must be identical, because they may only use matches
    strictly before their own date.

    Args:
        matches: The enriched match table.
        cutoff_fraction: Where to cut, as a share of the matches in date order.

    Returns:
        Names of features that changed (an empty list means no leakage found).
    """
    ordered = matches.sort_values("date", kind="stable")
    cutoff = ordered["date"].iloc[int(len(ordered) * cutoff_fraction)]
    before = build_features(matches)
    after = build_features(scramble_outcomes(matches, cutoff))
    mask = (before["date"] <= cutoff).to_numpy()
    leaking = []
    for column in feature_columns():
        if not np.allclose(
            before.loc[mask, column].to_numpy(dtype="float64"),
            after.loc[mask, column].to_numpy(dtype="float64"),
            equal_nan=True,
            rtol=1e-9,
            atol=1e-9,
        ):
            leaking.append(column)
    return leaking


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def feature_coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Share of rows per season with each key feature available."""
    keys = {
        "home_elo": "elo",
        f"home_xg_for_ew{LONG}": "xg_form",
        f"home_shots_for_ew{LONG}": "shots_form",
        f"home_ppda_for_ew{LONG}": "ppda_form",
        "home_xg_exp": "xg_exp",
    }
    present = features[list(keys)].notna().astype(float) * 100.0
    coverage = present.groupby(features["season"]).mean()
    coverage.columns = list(keys.values())
    coverage.insert(0, "matches", features.groupby("season").size())
    return coverage


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leak-free pre-match features.")
    parser.add_argument("--input", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument("--output", type=Path, default=FEATURES_OUTPUT)
    parser.add_argument(
        "--no-audit", action="store_true", help="skip the leakage audit"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build ``features.parquet`` and return the process exit code.

    Returns ``0`` on success and ``1`` when the input is missing or invalid, or
    when the leakage audit finds a feature that depends on the future.
    """
    args = _parse_args(argv)
    log = setup_logging("features")
    if not args.input.exists():
        log.error("Input not found: %s; run python -m src.enrich first", args.input)
        return 1
    matches = pd.read_parquet(args.input)
    try:
        features = build_features(matches)
    except ValueError as exc:
        log.error("Cannot build features: %s", exc)
        return 1
    log.info("Built %d rows with %d features", len(features), len(feature_columns()))
    log.info(
        "Feature availability by season (%% of matches):\n%s",
        feature_coverage(features).to_string(float_format="{:.0f}".format),
    )

    if not args.no_audit:
        leaking = leakage_audit(matches)
        if leaking:
            log.error(
                "LEAKAGE: %d feature(s) change when future outcomes change: %s",
                len(leaking),
                ", ".join(leaking),
            )
            return 1
        log.info(
            "Leakage audit passed: no feature moved after scrambling outcomes from the 70% mark"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    log.info(
        "Wrote %s (%s to %s)",
        args.output,
        season_label(int(features["season"].min())),
        season_label(int(features["season"].max())),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
