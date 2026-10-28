"""Tests for the Elo rating module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.elo import EloConfig, compute_elo, expected_score, goal_difference_multiplier


def match(
    match_id: str,
    date: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    *,
    league: str = "E0",
    season: int = 2024,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "league": league,
        "season": season,
        "date": pd.Timestamp(date),
        "home_team_id": home,
        "away_team_id": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def elo(rows: list[dict[str, object]], config: EloConfig | None = None) -> pd.DataFrame:
    return compute_elo(pd.DataFrame(rows), config).set_index("match_id")


@pytest.mark.parametrize(
    ("goal_difference", "expected"),
    [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (-2, 1.5), (3, 1.75), (5, 2.0)],
)
def test_goal_difference_multiplier(goal_difference: int, expected: float) -> None:
    assert goal_difference_multiplier(goal_difference) == pytest.approx(expected)


def test_expected_score() -> None:
    assert expected_score(0) == pytest.approx(0.5)
    assert expected_score(400) == pytest.approx(1 / 1.1)
    assert expected_score(-400) == pytest.approx(1 - 1 / 1.1)


def test_first_match_uses_the_initial_rating_and_home_advantage() -> None:
    result = elo([match("m1", "2024-08-10", "a", "b", 1, 0)])
    assert result.loc["m1", "home_elo"] == 1500.0
    assert result.loc["m1", "away_elo"] == 1500.0
    assert result.loc["m1", "elo_win_prob_home"] == pytest.approx(expected_score(65.0))


def test_ratings_are_recorded_before_the_result_is_applied() -> None:
    rows = [
        match("m1", "2024-08-10", "a", "b", 1, 0),
        match("m2", "2024-08-17", "a", "c", 0, 0),
    ]
    result = elo(rows)
    change = 20.0 * 1.0 * (1.0 - expected_score(65.0))  # K * G * (actual - expected)
    assert result.loc["m1", "home_elo"] == 1500.0  # not yet moved by its own result
    assert result.loc["m2", "home_elo"] == pytest.approx(1500.0 + change)


def test_updates_are_zero_sum() -> None:
    rows = [
        match("m1", "2024-08-10", "a", "b", 3, 0),
        match("m2", "2024-08-17", "b", "a", 2, 2),
    ]
    result = elo(rows)
    assert result.loc["m2", "home_elo"] + result.loc["m2", "away_elo"] == pytest.approx(
        3000.0
    )


def test_bigger_wins_move_ratings_more() -> None:
    narrow = elo(
        [
            match("m1", "2024-08-10", "a", "b", 1, 0),
            match("m2", "2024-08-17", "a", "c", 0, 0),
        ]
    )
    wide = elo(
        [
            match("m1", "2024-08-10", "a", "b", 4, 0),
            match("m2", "2024-08-17", "a", "c", 0, 0),
        ]
    )
    assert wide.loc["m2", "home_elo"] > narrow.loc["m2", "home_elo"]


def test_a_draw_between_equals_favours_the_away_side() -> None:
    result = elo(
        [
            match("m1", "2024-08-10", "a", "b", 1, 1),
            match("m2", "2024-08-17", "a", "b", 0, 0),
        ]
    )
    assert result.loc["m2", "home_elo"] < 1500.0 < result.loc["m2", "away_elo"]


def test_newcomers_in_later_seasons_start_lower() -> None:
    rows = [
        match("m1", "2023-08-10", "a", "b", 1, 0, season=2023),
        match(
            "m2", "2024-08-10", "a", "c", 1, 0, season=2024
        ),  # c appears for the first time
    ]
    result = elo(rows, EloConfig(season_regression=0.0))
    assert result.loc["m2", "away_elo"] == 1450.0


def test_ratings_regress_towards_the_mean_between_seasons() -> None:
    rows = [
        match("m1", "2023-08-10", "a", "b", 4, 0, season=2023),
        match("m2", "2023-08-17", "a", "b", 4, 0, season=2023),
        match("m3", "2024-08-10", "a", "b", 0, 0, season=2024),
    ]
    kept = elo(rows, EloConfig(season_regression=0.0))
    shrunk = elo(rows, EloConfig(season_regression=0.5))
    assert kept.loc["m3", "home_elo"] > shrunk.loc["m3", "home_elo"] > 1500.0
    assert shrunk.loc["m3", "away_elo"] < 1500.0 < shrunk.loc["m3", "away_elo"] + 100


def test_leagues_keep_separate_ratings() -> None:
    rows = [
        match("m1", "2024-08-10", "a", "b", 5, 0, league="E0"),
        match(
            "m2", "2024-08-17", "a", "b", 0, 0, league="SP1"
        ),  # same ids, another league
    ]
    assert elo(rows).loc["m2", "home_elo"] == 1500.0


def test_input_order_does_not_matter() -> None:
    rows = [
        match("m1", "2024-08-10", "a", "b", 2, 0),
        match("m2", "2024-08-10", "c", "d", 1, 1),
        match("m3", "2024-08-17", "a", "c", 0, 3),
    ]
    forward = elo(rows).sort_index()
    backward = elo(rows[::-1]).sort_index()
    pd.testing.assert_frame_equal(forward, backward)


def test_one_row_per_match() -> None:
    rows = [
        match(f"m{i}", f"2024-08-{10 + i:02d}", "a", "b", i % 3, 1) for i in range(5)
    ]
    result = compute_elo(pd.DataFrame(rows))
    assert list(result.columns) == [
        "match_id",
        "home_elo",
        "away_elo",
        "elo_win_prob_home",
    ]
    assert len(result) == 5
    assert result["elo_win_prob_home"].between(0, 1).all()
