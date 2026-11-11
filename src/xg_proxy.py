"""A shot-based expected-goals proxy, for when Understat is unavailable.

Understat's xG values every shot by where and how it was taken. The
football-data.co.uk files only count shots, so this module prices them
instead: a shot on target is worth ``on_target`` goals and any other shot
``off_target`` goals. The two prices come from a Poisson regression of goals on
the two shot counts, fitted on a single calibration season: by default the first
season in the data, which is the warm-up season the models never train on, so no
later outcome can leak into the proxy.

The proxy is smoother than raw goals because it ignores finishing luck, but it
knows nothing about shot quality. It is a weaker signal than real xG, so
everything built on it must be labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from src.leagues import season_label

SHOT_COLUMNS: Final[tuple[str, ...]] = (
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
)
#: Fewest team-matches a fit is trusted with; one full season has thousands.
MIN_FIT_OBSERVATIONS: Final[int] = 400
MAX_ITERATIONS: Final[int] = 20_000
TOLERANCE: Final[float] = 1e-12


@dataclass(frozen=True)
class ShotModel:
    """The fitted prices of a shot.

    Attributes:
        on_target: Expected goals per shot on target.
        off_target: Expected goals per shot that missed or was blocked.
        season: Calibration season (start year).
        n_observations: Team-matches the prices were fitted on.
    """

    on_target: float
    off_target: float
    season: int
    n_observations: int


def _counts(matches: pd.DataFrame, side: str) -> tuple[pd.Series, pd.Series]:
    """Shots on target and other shots of one side; missing where either count is.

    A match that lists fewer shots than shots on target is a data error, so it
    is read as having no other shots rather than a negative number of them.
    """
    shots = pd.to_numeric(matches[f"{side}_shots"], errors="coerce")
    on_target = pd.to_numeric(matches[f"{side}_shots_on_target"], errors="coerce")
    return on_target, (shots - on_target).clip(lower=0.0)


def _team_rows(matches: pd.DataFrame, side: str) -> pd.DataFrame:
    """Goals, shots on target and other shots of one side, one row per match."""
    on_target, off_target = _counts(matches, side)
    return pd.DataFrame(
        {
            "goals": pd.to_numeric(matches[f"{side}_goals"], errors="coerce"),
            "on_target": on_target,
            "off_target": off_target,
        }
    )


def _fit_prices(
    goals: np.ndarray, on_target: np.ndarray, off_target: np.ndarray
) -> tuple[float, float]:
    """Maximum-likelihood prices for ``goals ~ Poisson(a * on_target + b * off_target)``.

    The likelihood is concave and both prices must stay non-negative, so this
    uses the multiplicative (expectation-maximisation) update of Poisson
    regression: it keeps the prices positive and never lowers the likelihood.

    Raises:
        ValueError: If the iteration has not settled after ``MAX_ITERATIONS``.
    """
    total_on, total_off = on_target.sum(), off_target.sum()
    on_price = off_price = goals.sum() / (total_on + total_off)  # equal prices to start
    for _ in range(MAX_ITERATIONS):
        ratio = goals / (on_price * on_target + off_price * off_target)
        new_on = on_price * (ratio * on_target).sum() / total_on
        new_off = off_price * (ratio * off_target).sum() / total_off
        settled = (
            abs(new_on - on_price) < TOLERANCE and abs(new_off - off_price) < TOLERANCE
        )
        on_price, off_price = new_on, new_off
        if settled:
            return float(on_price), float(off_price)
    raise ValueError("the shot prices did not settle")


def fit_shot_model(
    matches: pd.DataFrame,
    *,
    season: int | None = None,
    min_observations: int = MIN_FIT_OBSERVATIONS,
) -> ShotModel:
    """Fit the price of a shot on target and of any other shot.

    Args:
        matches: Match table with ``season``, goals, shots and shots on target
            for both sides (the columns of ``matches.parquet``).
        season: Calibration season (start year); defaults to the first season in
            ``matches``. Only that season's matches are used.
        min_observations: Fewest team-matches with complete shot data accepted.

    Returns:
        The fitted :class:`ShotModel`.

    Raises:
        ValueError: If columns are missing, the table is empty, the calibration
            season has too few matches with complete shot data, or it has no
            goals, no shots on target or no other shots to price.
    """
    needed = ("season", "home_goals", "away_goals", *SHOT_COLUMNS)
    missing = [column for column in needed if column not in matches.columns]
    if missing:
        raise ValueError(f"the match table is missing columns: {missing}")
    if matches.empty:
        raise ValueError("the match table is empty")

    calibration = int(matches["season"].min()) if season is None else season
    sample = matches[matches["season"] == calibration]
    rows = pd.concat(
        [_team_rows(sample, "home"), _team_rows(sample, "away")], ignore_index=True
    ).dropna()
    rows = rows[rows["on_target"] + rows["off_target"] > 0]
    label = season_label(calibration)
    if len(rows) < min_observations:
        raise ValueError(
            f"only {len(rows)} team-matches with complete shot data in {label}; "
            f"need at least {min_observations}"
        )
    goals, on_target, off_target = (
        rows[column].to_numpy(dtype="float64")
        for column in ("goals", "on_target", "off_target")
    )
    if goals.sum() <= 0 or on_target.sum() <= 0 or off_target.sum() <= 0:
        raise ValueError(
            f"{label} has no goals, no shots on target or no other shots to price"
        )
    on_price, off_price = _fit_prices(goals, on_target, off_target)
    return ShotModel(on_price, off_price, calibration, len(rows))


def shot_xg(matches: pd.DataFrame, model: ShotModel) -> pd.DataFrame:
    """Expected goals of each side, priced from its shot counts.

    Args:
        matches: Match table with the :data:`SHOT_COLUMNS`.
        model: Prices from :func:`fit_shot_model`.

    Returns:
        ``home_xg`` and ``away_xg``, indexed like ``matches`` and missing where
        that side's shot counts are.

    Raises:
        ValueError: If a shot column is missing.
    """
    missing = [column for column in SHOT_COLUMNS if column not in matches.columns]
    if missing:
        raise ValueError(f"the match table is missing columns: {missing}")
    priced = {}
    for side in ("home", "away"):
        on_target, off_target = _counts(matches, side)
        priced[f"{side}_xg"] = (
            model.on_target * on_target + model.off_target * off_target
        )
    return pd.DataFrame(priced, index=matches.index)


def xg_per_goal(matches: pd.DataFrame) -> pd.Series:
    """Total xG divided by total goals for each season, over matches with both xG values.

    A value near 1.0 means the xG is as large as the goals it stands in for. A
    drift across seasons points at a change in how shots are recorded.

    Args:
        matches: Table with ``season``, goals and ``home_xg`` / ``away_xg``.
    """
    usable = matches[matches["home_xg"].notna() & matches["away_xg"].notna()]
    by_season = usable.groupby("season")
    xg = by_season["home_xg"].sum() + by_season["away_xg"].sum()
    goals = by_season["home_goals"].sum() + by_season["away_goals"].sum()
    return (xg / goals).rename("xg_per_goal")
