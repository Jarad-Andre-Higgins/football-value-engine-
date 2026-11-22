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


# --------------------------------------------------------------------------- #
# Matches not played yet
# --------------------------------------------------------------------------- #
def fixture(
    match_id: str,
    date: str,
    home: str,
    away: str,
    *,
    league: str = "E0",
    season: int = 2024,
) -> dict[str, object]:
    """A match still to be played: no score."""
    return {
        **match(match_id, date, home, away, 0, 0, league=league, season=season),
        "home_goals": float("nan"),
        "away_goals": float("nan"),
    }


def test_a_match_without_a_score_is_rated_but_moves_nobody() -> None:
    played = [
        match("m1", "2024-08-10", "a", "b", 3, 0),
        match("m3", "2024-08-24", "a", "b", 0, 1),
    ]
    plain = elo(played)
    with_fixture = elo([*played, fixture("m2", "2024-08-17", "a", "c")])
    columns = ["home_elo", "away_elo", "elo_win_prob_home"]
    # the later match is rated exactly as if the fixture were not in the table
    assert with_fixture.loc["m3", columns].tolist() == plain.loc["m3", columns].tolist()
    # the fixture carries the ratings held before it: a after beating b, c untouched
    assert with_fixture.loc["m2", "home_elo"] == pytest.approx(
        plain.loc["m3", "home_elo"]
    )
    assert with_fixture.loc["m2", "away_elo"] == 1500.0
    assert with_fixture.notna().all().all()


def test_a_fixture_never_changes_a_rating_however_many_there_are() -> None:
    played = [match("m1", "2024-08-10", "a", "b", 2, 2)]
    ahead = [fixture(f"f{i}", f"2024-08-{17 + 7 * i:02d}", "a", "b") for i in range(3)]
    result = elo([*played, *ahead])
    after_draw = result.loc["f0", ["home_elo", "away_elo"]].tolist()
    for i in (1, 2):
        assert result.loc[f"f{i}", ["home_elo", "away_elo"]].tolist() == after_draw


def test_a_promoted_club_is_rated_as_a_newcomer_on_its_first_fixture() -> None:
    result = elo(
        [
            match("m1", "2024-08-10", "a", "b", 1, 0, season=2024),
            fixture("m2", "2025-08-09", "a", "z", season=2025),
        ]
    )
    assert result.loc["m2", "away_elo"] == EloConfig().newcomer


def test_a_table_of_fixtures_only_gives_every_club_the_starting_rating() -> None:
    result = elo([fixture("m1", "2024-08-10", "a", "b")])
    assert result.loc["m1", ["home_elo", "away_elo"]].tolist() == [1500.0, 1500.0]
    assert result.loc["m1", "elo_win_prob_home"] == pytest.approx(expected_score(65.0))
