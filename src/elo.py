"""Elo ratings with goal-difference scaling, in the style of World Football Elo.

Each club has one rating per league. Before a match, the expected score of the
home side is ``1 / (1 + 10 ** (-(home + home_advantage - away) / 400))``. After
the match both ratings move by ``K * G * (actual - expected)``, where ``G``
grows with the winning margin, so a 4-0 moves ratings more than a 1-0.

The ratings returned for a match are the ones held *before* its result is
applied, which is what keeps them leak-free. Between seasons every rating
moves part of the way back to the league average; clubs seen for the first
time after the league's opening season start below average, as newly promoted
sides tend to be.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    """Tuning constants for :func:`compute_elo`.

    Attributes:
        initial: Rating of every club in a league's first season in the data.
        newcomer: Rating of a club first seen in a later season (promoted side).
        k: Base update size in rating points.
        home_advantage: Rating points added to the home side for the expectation.
        season_regression: Share of the gap to the league mean erased each summer.
    """

    initial: float = 1500.0
    newcomer: float = 1450.0
    k: float = 20.0
    home_advantage: float = 65.0
    season_regression: float = 0.20


def goal_difference_multiplier(goal_difference: int) -> float:
    """Scale an update by the winning margin: 1, 1, 1.5, then ``(11 + n) / 8``."""
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def expected_score(rating_difference: float) -> float:
    """Expected score (win probability plus half the draw probability)."""
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def compute_elo(matches: pd.DataFrame, config: EloConfig | None = None) -> pd.DataFrame:
    """Compute pre-match Elo ratings for every match.

    Args:
        matches: Needs ``match_id``, ``league``, ``season``, ``date``,
            ``home_team_id``, ``away_team_id``, ``home_goals`` and ``away_goals``.
        config: Rating constants; defaults to :class:`EloConfig`.

    Returns:
        One row per match with ``match_id``, ``home_elo`` and ``away_elo`` (both
        before the match) and ``elo_win_prob_home`` (the home expected score,
        including home advantage).
    """
    config = config or EloConfig()
    ordered = matches.sort_values(["date", "match_id"], kind="stable")
    ratings: dict[tuple[str, str], float] = {}
    current_season: dict[str, int] = {}
    opening_season: dict[str, int] = {}
    rows: list[tuple[str, float, float, float]] = []

    for match in ordered.itertuples(index=False):
        league, season = match.league, int(match.season)
        if league not in opening_season:
            opening_season[league] = season
        if current_season.get(league) != season:
            _regress_to_mean(ratings, league, config.season_regression)
            current_season[league] = season
        start = config.initial if season == opening_season[league] else config.newcomer
        home_key = (league, match.home_team_id)
        away_key = (league, match.away_team_id)
        home = ratings.setdefault(home_key, start)
        away = ratings.setdefault(away_key, start)

        expected = expected_score(home + config.home_advantage - away)
        rows.append((match.match_id, home, away, expected))

        goal_difference = int(match.home_goals) - int(match.away_goals)
        actual = 1.0 if goal_difference > 0 else 0.5 if goal_difference == 0 else 0.0
        change = (
            config.k * goal_difference_multiplier(goal_difference) * (actual - expected)
        )
        ratings[home_key] = home + change
        ratings[away_key] = away - change

    return pd.DataFrame(
        rows, columns=["match_id", "home_elo", "away_elo", "elo_win_prob_home"]
    )


def _regress_to_mean(
    ratings: dict[tuple[str, str], float], league: str, share: float
) -> None:
    """Move every rating in ``league`` part of the way back to the league mean."""
    keys = [key for key in ratings if key[0] == league]
    if not keys:
        return
    mean = sum(ratings[key] for key in keys) / len(keys)
    for key in keys:
        ratings[key] += share * (mean - ratings[key])
