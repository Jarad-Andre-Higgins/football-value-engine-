"""Tests for the closing-line-value autopsy."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import autopsy as module
from src import logging_config
from src.autopsy import (
    AUTOPSY_FOLDER,
    BET_REQUIRED,
    BREAKDOWN_VALUES,
    CONTROL_WIDTH,
    ODDS_LABELS,
    PART_COLUMNS,
    PART_LABELS,
    RESPONSE_COLUMNS,
    Autopsy,
    attach_matches,
    autopsy,
    best_bets,
    breakdown,
    concentration,
    control_population,
    decomposition,
    format_autopsy,
    gap_distribution,
    gap_ranges,
    main,
    odds_band,
    quote_table,
    reproduces,
    response_table,
    save_autopsy,
    similar_drift,
    strategy_bets,
    verdict,
    without_bets,
)
from src.backtest import BACKTEST_DIR, BacktestResult, cluster_bootstrap, run_backtest
from src.demargin import remove_margin
from src.odds_history import MARKETS
from src.value import ValueConfig

KEY = ["match_id", "market_id", "selection"]
#: What src.backtest writes for each bet: the audit may not ask for less or more.
BET_COLUMNS = [
    "match_id",
    "season",
    "date",
    "batch",
    "bookmaker",
    "market_id",
    "selection",
    "odds",
    "p_reference",
    "p_close",
    "stake",
    "weight",
    "won",
    "clv",
]
#: What the match table needs for settling the sharp book's quotes.
MATCH_NEEDED = ["match_id", "season", "date", "home_goals", "away_goals"]
WEEKS = pd.DatetimeIndex(["2023-09-01", "2023-09-08", "2023-09-15", "2024-09-13"])


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


# --------------------------------------------------------------------------- #
# Tables to test on
# --------------------------------------------------------------------------- #
def friday_of(stamp: pd.Timestamp) -> pd.Timestamp:
    """The Friday on or before a date (a coupon's start)."""
    stamp = stamp.normalize()
    return stamp - pd.Timedelta((stamp.weekday() - 4) % 7, unit="D")


def bet_row(
    match_id: str,
    date: str,
    market_id: str,
    selection: str,
    odds: float,
    p_reference: float,
    p_close: float,
    stake: float,
    won: bool,
    *,
    season: int,
    book: str = "b365",
    weight: float = 0.0,
) -> dict[str, object]:
    stamp = pd.Timestamp(date)
    return {
        "match_id": match_id,
        "season": season,
        "date": stamp,
        "batch": friday_of(stamp),
        "bookmaker": book,
        "market_id": market_id,
        "selection": selection,
        "odds": odds,
        "p_reference": p_reference,
        "p_close": p_close,
        "stake": stake,
        "weight": weight,
        "won": won,
        "clv": odds * p_close - 1.0,
    }


def hand_bets() -> pd.DataFrame:
    """Four Bet365 bets at weight 0 and two bets of other strategies.

    Bet365 at weight 0 (price gap, drift, CLV in brackets)::

        m1 1X2 home   4.0 at 0.30 -> 0.35, stake 10  (+0.20, +0.20, +0.40)  Saturday 2 Sep 2023
        m2 O/U over   2.5 at 0.50 -> 0.40, stake 30  (+0.25, -0.25,  0.00)  Sunday 3 Sep 2023
        m3 1X2 draw   3.0 at 0.40 -> 0.40, stake 20  (+0.20,  0.00, +0.20)  Saturday 9 Sep 2023
        m4 1X2 away   8.0 at 0.15 -> 0.175, stake 40 (+0.20, +0.20, +0.40)  Saturday 14 Sep 2024

    ``m1`` and ``m2`` are in the coupon of Friday 1 September 2023, ``m3`` in that of
    the 8th and ``m4`` in that of 13 September 2024. Per unit staked: price gap
    0.215, drift 0.025, CLV 0.24. Per bet: 0.2125, 0.0375 and 0.25. The other bets
    (``m5`` at weight 0.5, ``m6`` at the market average) add the weeks of 15
    September 2023 and, again, 1 September.
    """
    rows = [
        bet_row("m1", "2023-09-02", "1X2", "home", 4.0, 0.30, 0.35, 10.0, False, season=2023),
        bet_row("m2", "2023-09-03", "OU:2.5", "over", 2.5, 0.50, 0.40, 30.0, True, season=2023),
        bet_row("m3", "2023-09-09", "1X2", "draw", 3.0, 0.40, 0.40, 20.0, False, season=2023),
        bet_row("m4", "2024-09-14", "1X2", "away", 8.0, 0.15, 0.175, 40.0, True, season=2024),
        bet_row(
            "m5", "2023-09-16", "1X2", "home", 2.0, 0.55, 0.50, 25.0, False,
            season=2023, weight=0.5,
        ),
        bet_row(
            "m6", "2023-09-02", "OU:2.5", "under", 2.2, 0.50, 0.45, 15.0, True,
            season=2023, book="avg",
        ),
    ]  # fmt: skip
    return pd.DataFrame(rows)


def control_row(
    match_id: str,
    market_id: str,
    selection: str,
    p_early: float,
    move: float,
    batch: str,
    season: int = 2023,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "market_id": market_id,
        "selection": selection,
        "season": season,
        "batch": pd.Timestamp(batch),
        "p_early": p_early,
        "p_close": p_early + move,
    }


def hand_population() -> pd.DataFrame:
    """The sharp book's quotes: the four bets' own, controls near each, and decoys.

    The controls within 0.02 of each bet's probability drift by (mean, in
    probability): 0.02 near ``m1``, 0 near ``m3``, 0.01 near ``m4`` and -0.01 near
    ``m2`` (over/under), so ``typical`` is 0.08, -0.025, 0 and 0.08 (odds times
    that). The decoys are too far away or in the other market. Ten quotes are not
    bets.
    """
    own = [
        control_row("m1", "1X2", "home", 0.30, 0.05, "2023-09-01"),
        control_row("m2", "OU:2.5", "over", 0.50, -0.10, "2023-09-01"),
        control_row("m3", "1X2", "draw", 0.40, 0.00, "2023-09-08"),
        control_row("m4", "1X2", "away", 0.15, 0.025, "2024-09-13", season=2024),
    ]
    others = [
        control_row("c1", "1X2", "home", 0.29, 0.01, "2023-09-22"),
        control_row("c2", "1X2", "home", 0.31, 0.03, "2023-09-22"),
        control_row("c3", "1X2", "draw", 0.39, -0.01, "2023-09-22"),
        control_row("c4", "1X2", "draw", 0.41, 0.01, "2023-09-22"),
        control_row("c5", "1X2", "away", 0.14, 0.01, "2023-09-22"),
        control_row("c6", "1X2", "away", 0.16, 0.01, "2023-09-22"),
        control_row("c7", "OU:2.5", "over", 0.49, -0.02, "2023-09-22"),
        control_row("c8", "OU:2.5", "under", 0.51, 0.00, "2023-09-22"),
        control_row("d1", "1X2", "home", 0.60, 0.50, "2023-09-22"),
        control_row("d2", "OU:2.5", "over", 0.30, 0.30, "2023-09-22"),
    ]
    return pd.DataFrame([*own, *others])


def hand_names() -> pd.DataFrame:
    """League and team names of the matches (``m1`` twice, as in a messy table)."""
    return pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m4", "m5", "m6", "m1"],
            "league": [
                "Premier League",
                "La Liga",
                "Serie A",
                "Premier League",
                "La Liga",
                "Serie A",
                "Premier League",
            ],
            "home_team": ["Arsenal", "Real", "Roma", "Spurs", "Girona", "Lazio", "Arsenal"],
            "away_team": ["Chelsea", "Sevilla", "Milan", "Everton", "Betis", "Napoli", "Chelsea"],
        }
    )  # fmt: skip


def parts_of(entries: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    """A table of parts, per unit staked: ``{part: (value, low, high)}``."""
    rows = [
        {"weighting": "stake", "part": part, "value": value, "low": low, "high": high}
        for part, (value, low, high) in entries.items()
    ]
    return pd.DataFrame(rows, columns=PART_COLUMNS)


def hand_result(**options: object) -> Autopsy:
    """The autopsy of :func:`hand_bets` (Bet365 at weight 0), by default alone."""
    return autopsy(hand_bets(), n_resamples=50, **options)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Which bets, and their price gap and drift
# --------------------------------------------------------------------------- #
def test_a_strategy_is_the_bets_of_one_book_at_one_weight() -> None:
    chosen = strategy_bets(hand_bets(), "b365", 0.0)
    assert chosen["match_id"].tolist() == ["m1", "m2", "m3", "m4"]
    assert chosen.index.tolist() == [0, 1, 2, 3]
    other_weight = strategy_bets(hand_bets(), "b365", 0.5)
    assert other_weight["match_id"].tolist() == ["m5"]
    other_book = strategy_bets(hand_bets(), "avg", 0.0)
    assert other_book["match_id"].tolist() == ["m6"]


def test_a_bets_closing_line_value_is_its_price_gap_plus_the_drift() -> None:
    chosen = strategy_bets(hand_bets(), "b365", 0.0)
    np.testing.assert_allclose(chosen["gap"], [0.20, 0.25, 0.20, 0.20])
    np.testing.assert_allclose(chosen["drift"], [0.20, -0.25, 0.0, 0.20], atol=1e-12)
    np.testing.assert_allclose(chosen["clv"], [0.40, 0.0, 0.20, 0.40], atol=1e-12)
    np.testing.assert_allclose(
        chosen["gap"] + chosen["drift"], chosen["clv"], atol=1e-12
    )


def test_the_price_gap_is_the_edge_against_the_sharp_books_early_probability() -> None:
    chosen = strategy_bets(hand_bets(), "b365", 0.0)
    np.testing.assert_allclose(
        chosen["gap"], chosen["odds"] * chosen["p_reference"] - 1.0
    )
    np.testing.assert_allclose(
        chosen["drift"], chosen["odds"] * (chosen["p_close"] - chosen["p_reference"])
    )


def test_the_weight_is_matched_as_a_number_not_as_text() -> None:
    assert len(strategy_bets(hand_bets(), "b365", 1e-12)) == 4
    assert len(strategy_bets(hand_bets(), "b365", 0.5000000001)) == 1


def test_a_strategy_without_bets_names_what_the_file_has() -> None:
    with pytest.raises(ValueError) as caught:
        strategy_bets(hand_bets(), "max", 0.0)
    assert str(caught.value) == (
        "no bets for max at weight 0; the file has "
        "avg at weight 0, b365 at weight 0, b365 at weight 0.5"
    )


def test_an_empty_file_has_no_strategy() -> None:
    with pytest.raises(ValueError, match="the file has no bets at all"):
        strategy_bets(hand_bets().iloc[0:0], "b365", 0.0)


def test_the_audit_asks_for_the_columns_the_backtest_writes() -> None:
    assert list(BET_REQUIRED) == BET_COLUMNS
    assert list(module.MATCH_REQUIRED) == MATCH_NEEDED


@pytest.mark.parametrize("column", BET_COLUMNS)
def test_a_bets_table_needs_every_column(column: str) -> None:
    with pytest.raises(
        ValueError, match=f"the bets table is missing columns: \\['{column}'\\]"
    ):
        strategy_bets(hand_bets().drop(columns=column), "b365", 0.0)


def test_a_table_that_is_not_a_backtests_is_refused() -> None:
    bets = hand_bets()
    bets["clv"] = bets["clv"] + 0.01
    with pytest.raises(ValueError, match="the stored clv is not odds \\* p_close - 1"):
        strategy_bets(bets, "b365", 0.0)


def test_other_strategies_clv_does_not_matter_to_this_one() -> None:
    bets = hand_bets()
    bets.loc[bets["bookmaker"] == "avg", "clv"] = 99.0
    assert len(strategy_bets(bets, "b365", 0.0)) == 4


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def test_matches_give_each_bet_its_league_and_teams() -> None:
    named = attach_matches(strategy_bets(hand_bets(), "b365", 0.0), hand_names())
    assert named["match_id"].tolist() == ["m1", "m2", "m3", "m4"]
    assert named["league"].tolist() == [
        "Premier League",
        "La Liga",
        "Serie A",
        "Premier League",
    ]
    assert named["home_team"].tolist() == ["Arsenal", "Real", "Roma", "Spurs"]
    assert named["away_team"].tolist() == ["Chelsea", "Sevilla", "Milan", "Everton"]


def test_the_strategys_bets_are_numbered_from_zero() -> None:
    chosen = strategy_bets(hand_bets().iloc[::-1], "b365", 0.0)
    assert chosen.index.tolist() == [0, 1, 2, 3]
    assert chosen["match_id"].tolist() == ["m4", "m3", "m2", "m1"]


def test_a_match_listed_twice_does_not_double_a_bet() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    assert len(attach_matches(bets, hand_names())) == len(bets)


def test_a_bet_whose_match_is_not_in_the_table_has_no_names() -> None:
    names = hand_names()
    named = attach_matches(
        strategy_bets(hand_bets(), "b365", 0.0), names[names["match_id"] != "m3"]
    )
    assert pd.isna(named["league"].iloc[2])
    assert named["league"].iloc[0] == "Premier League"


def test_columns_the_bets_already_have_are_not_overwritten() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0).assign(league="mine")
    named = attach_matches(bets, hand_names())
    assert named["league"].tolist() == ["mine"] * 4
    assert named["home_team"].tolist() == ["Arsenal", "Real", "Roma", "Spurs"]


def test_a_table_without_names_leaves_the_bets_alone() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    assert attach_matches(bets, hand_names()[["match_id"]]).equals(bets)
    assert attach_matches(bets, hand_names()[["league"]]).equals(bets)


# --------------------------------------------------------------------------- #
# Weeks and price ranges
# --------------------------------------------------------------------------- #
def test_weeks_are_the_distinct_dates_in_time_order_whatever_the_resolution() -> None:
    coarse = pd.Series(pd.to_datetime(["2023-09-08", "2023-09-01"])).astype(
        "datetime64[s]"
    )
    text = pd.Series(["2023-09-08", "2023-09-01", "2023-09-08"])
    expected = [pd.Timestamp("2023-09-01"), pd.Timestamp("2023-09-08")]
    assert list(module._weeks(coarse)) == expected
    assert list(module._weeks(text)) == expected


def test_a_price_is_in_the_range_that_includes_its_upper_end() -> None:
    odds = pd.Series([1.3, 2.0, 2.01, 3.0, 4.0, 5.99, 6.0, 8.0, 8.01, 20.0])
    ranges = odds_band(odds)
    assert ranges.astype(str).tolist() == [
        "1.0-2.0",
        "1.0-2.0",
        "2.0-3.0",
        "2.0-3.0",
        "3.0-4.0",
        "5.0-6.0",
        "5.0-6.0",
        "6.0-8.0",
        "8.0+",
        "8.0+",
    ]


@pytest.mark.parametrize(
    ("odds", "label"),
    [
        (1.0, "1.0-2.0"),
        (1.5, "1.0-2.0"),
        (2.0, "1.0-2.0"),
        (2.0001, "2.0-3.0"),
        (3.0, "2.0-3.0"),
        (3.0001, "3.0-4.0"),
        (4.0, "3.0-4.0"),
        (4.0001, "4.0-5.0"),
        (5.0, "4.0-5.0"),
        (5.0001, "5.0-6.0"),
        (6.0, "5.0-6.0"),
        (6.0001, "6.0-8.0"),
        (8.0, "6.0-8.0"),
        (8.0001, "8.0+"),
        (50.0, "8.0+"),
    ],
)
def test_every_price_range_ends_where_its_label_says(odds: float, label: str) -> None:
    assert str(odds_band(pd.Series([odds])).iloc[0]) == label


def test_price_ranges_keep_their_natural_order() -> None:
    ranges = odds_band(pd.Series([8.5, 1.5]))
    assert list(ranges.cat.categories) == list(ODDS_LABELS)
    assert ranges.cat.ordered
    assert ODDS_LABELS == (
        "1.0-2.0",
        "2.0-3.0",
        "3.0-4.0",
        "4.0-5.0",
        "5.0-6.0",
        "6.0-8.0",
        "8.0+",
    )


@pytest.mark.parametrize(
    ("gap", "min_edge", "label"),
    [
        (-0.5, 0.04, "below -10%"),
        (-0.10, 0.04, "-10% to -5%"),
        (-0.0501, 0.04, "-10% to -5%"),
        (-0.05, 0.04, "-5% to -2%"),
        (-0.02, 0.04, "-2% to 0%"),
        (-0.0001, 0.04, "-2% to 0%"),
        (0.0, 0.04, "0% to 2%"),
        (0.02, 0.04, "2% to 4%"),
        (0.0399, 0.04, "2% to 4%"),
        (0.04, 0.04, "4% or more"),
        (0.9, 0.04, "4% or more"),
        (0.02, 0.02, "2% or more"),
        (0.0199, 0.02, "0% to 2%"),
        (0.045, 0.07, "4% to 7%"),
        (0.07, 0.07, "7% or more"),
        (0.0249, 0.025, "2% to 2.5%"),
        (0.025, 0.025, "2.5% or more"),
        (-0.001, 0.0, "-2% to 0%"),
        (0.0, 0.0, "0% or more"),
        (0.005, 0.01, "0% to 1%"),
        (0.01, 0.01, "1% or more"),
    ],
)
def test_a_price_gap_is_in_the_range_that_includes_its_lower_end(
    gap: float, min_edge: float, label: str
) -> None:
    assert str(gap_ranges(pd.Series([gap]), min_edge).iloc[0]) == label


@pytest.mark.parametrize(
    ("min_edge", "labels"),
    [
        (
            0.04,
            [
                "below -10%",
                "-10% to -5%",
                "-5% to -2%",
                "-2% to 0%",
                "0% to 2%",
                "2% to 4%",
                "4% or more",
            ],
        ),
        (0.07, ["below -10%", "-10% to -5%", "-5% to -2%", "-2% to 0%", "0% to 2%", "2% to 4%", "4% to 7%", "7% or more"]),
        (0.02, ["below -10%", "-10% to -5%", "-5% to -2%", "-2% to 0%", "0% to 2%", "2% or more"]),
        (0.0, ["below -10%", "-10% to -5%", "-5% to -2%", "-2% to 0%", "0% or more"]),
    ],
)  # fmt: skip
def test_the_last_range_starts_at_the_minimum_edge(
    min_edge: float, labels: list[str]
) -> None:
    ranges = gap_ranges(pd.Series([0.0]), min_edge)
    assert list(ranges.cat.categories) == labels
    assert ranges.cat.ordered


# --------------------------------------------------------------------------- #
# Quotes priced like the bets
# --------------------------------------------------------------------------- #
def test_similar_quotes_are_those_within_the_width_in_the_same_market() -> None:
    population = pd.DataFrame(
        {
            "market_id": ["1X2"] * 5 + ["OU:2.5"],
            "p_early": [0.125, 0.25, 0.375, 0.5, 0.625, 0.375],
            "p_close": [0.135, 0.27, 0.345, 0.54, 0.625, 0.875],
        }
    )
    bets = pd.DataFrame(
        {
            "market_id": ["1X2", "1X2", "1X2", "OU:2.5"],
            "p_reference": [0.375, 0.0625, 0.9, 0.375],
        }
    )
    drifts = similar_drift(bets, population, width=0.125)
    # 0.25, 0.375 and 0.5, both ends included: (0.02 - 0.03 + 0.04) / 3
    assert drifts[0] == pytest.approx(0.01)
    # -0.0625 to 0.1875 holds only 0.125, which moved by 0.01
    assert drifts[1] == pytest.approx(0.01)
    assert np.isnan(drifts[2])
    # the other market has its own quote
    assert drifts[3] == pytest.approx(0.5)


def test_each_end_of_the_window_counts() -> None:
    population = pd.DataFrame(
        {
            "market_id": ["1X2"] * 3,
            "p_early": [0.25, 0.375, 0.5],
            "p_close": [0.25 + 0.03, 0.375 + 0.0, 0.5 - 0.03],
        }
    )
    bets = pd.DataFrame({"market_id": ["1X2"] * 3, "p_reference": [0.375, 0.5, 0.25]})
    low_and_high = similar_drift(bets.iloc[:1], population, width=0.125)[0]
    assert low_and_high == pytest.approx(0.0)
    only_upper = similar_drift(bets.iloc[1:2], population, width=0.125)[0]
    assert only_upper == pytest.approx(-0.015)
    only_lower = similar_drift(bets.iloc[2:], population, width=0.125)[0]
    assert only_lower == pytest.approx(0.015)


def test_the_window_is_two_points_of_probability_unless_told_otherwise() -> None:
    assert CONTROL_WIDTH == 0.02
    population = pd.DataFrame(
        {
            "market_id": ["1X2"] * 4,
            "p_early": [0.26, 0.285, 0.315, 0.34],
            "p_close": [0.36, 0.295, 0.325, 0.44],
        }
    )
    bets = pd.DataFrame({"market_id": ["1X2"], "p_reference": [0.30]})
    assert similar_drift(bets, population)[0] == pytest.approx(0.01)


def test_similar_quotes_do_not_depend_on_the_order_of_either_table() -> None:
    population = hand_population()
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    others = without_bets(population, bets)
    expected = similar_drift(bets, others)
    shuffled_population = others.sample(frac=1.0, random_state=4)
    shuffled_bets = bets.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    drifts = similar_drift(shuffled_bets, shuffled_population)
    np.testing.assert_allclose(drifts, expected[[2, 0, 3, 1]])


def test_every_bet_gets_the_drift_of_the_quotes_near_its_own_probability() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    others = without_bets(hand_population(), bets)
    np.testing.assert_allclose(
        similar_drift(bets, others), [0.02, -0.01, 0.0, 0.01], atol=1e-12
    )


def test_a_market_without_any_quote_has_no_similar_drift() -> None:
    bets = pd.DataFrame({"market_id": ["BTTS"], "p_reference": [0.5]})
    population = pd.DataFrame(
        {"market_id": ["1X2"], "p_early": [0.5], "p_close": [0.6]}
    )
    assert np.isnan(similar_drift(bets, population)[0])


def test_the_bets_own_quotes_are_not_their_own_controls() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    population = hand_population()
    others = without_bets(population, bets)
    assert len(others) == len(population) - 4
    assert set(map(tuple, others[KEY].to_numpy())).isdisjoint(
        set(map(tuple, bets[KEY].to_numpy()))
    )
    assert others.index.tolist() == list(range(len(others)))
    assert list(others.columns) == list(population.columns)


def test_taking_bets_out_leaves_quotes_of_the_same_match_that_were_not_bet() -> None:
    population = pd.DataFrame(
        {
            "match_id": ["m1", "m1", "m1"],
            "market_id": ["1X2"] * 3,
            "selection": ["home", "draw", "away"],
            "p_early": [0.5, 0.3, 0.2],
            "p_close": [0.5, 0.3, 0.2],
        }
    )
    bets = pd.DataFrame(
        {"match_id": ["m1", "m1"], "market_id": ["1X2"] * 2, "selection": ["draw"] * 2}
    )
    assert without_bets(population, bets)["selection"].tolist() == ["home", "away"]


# --------------------------------------------------------------------------- #
# Does the match table reproduce the backtest?
# --------------------------------------------------------------------------- #
def test_a_population_that_holds_the_bets_probabilities_reproduces_them() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    assert reproduces(bets, hand_population())


def test_the_check_tolerates_rounding_and_nothing_more() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    near = hand_population()
    near["p_early"] += 1e-12
    near["p_close"] -= 1e-12
    assert reproduces(bets, near)
    off = hand_population()
    off.loc[0, "p_early"] += 1e-6
    assert not reproduces(bets, off)
    late = hand_population()
    late.loc[2, "p_close"] += 1e-6
    assert not reproduces(bets, late)


def test_a_difference_of_exactly_the_tolerance_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "PROBABILITY_TOLERANCE", 2.0**-20)
    bets = pd.DataFrame(
        {
            "match_id": ["a"],
            "market_id": ["1X2"],
            "selection": ["home"],
            "p_reference": [0.5],
            "p_close": [0.25],
        }
    )
    population = pd.DataFrame(
        {
            "match_id": ["a"],
            "market_id": ["1X2"],
            "selection": ["home"],
            "p_early": [0.5 + 2.0**-20],
            "p_close": [0.25 - 2.0**-20],
        }
    )
    assert reproduces(bets, population)
    assert not reproduces(bets, population.assign(p_early=0.5 + 2.0**-19))
    assert not reproduces(bets, population.assign(p_close=0.25 - 2.0**-19))


def test_a_population_without_one_of_the_bets_does_not_reproduce_them() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    population = hand_population()
    assert not reproduces(bets, population[population["match_id"] != "m3"])
    assert not reproduces(bets, population.iloc[0:0])


def test_only_the_strategys_own_bets_need_to_be_in_the_population() -> None:
    bets = strategy_bets(hand_bets(), "b365", 0.0)
    assert reproduces(bets, hand_population())
    other = strategy_bets(hand_bets(), "avg", 0.0)
    assert not reproduces(other, hand_population())


# --------------------------------------------------------------------------- #
# The sharp book's quotes from a match table
# --------------------------------------------------------------------------- #
def match_table() -> pd.DataFrame:
    """Four matches with Pinnacle's 1X2 quotes, early and at the close."""
    return pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "season": [2022, 2022, 2023, 2023],
            "date": pd.to_datetime(
                ["2022-09-02", "2022-09-03", "2023-09-01", "2023-09-09"]
            ),
            "home_goals": [2, 0, 1, 1],
            "away_goals": [1, 0, 3, 0],
            "ps_h": [2.00, 2.60, 3.40, 1.50],
            "ps_d": [3.50, 3.10, 3.60, 4.20],
            "ps_a": [4.00, 2.90, 2.10, 7.00],
            "psc_h": [1.90, 2.70, 3.30, 1.45],
            "psc_d": [3.60, 3.00, 3.70, 4.40],
            "psc_a": [4.60, 2.95, 2.15, 7.50],
            "b365_h": [1.95, 2.50, 3.60, 1.25],
            "b365_d": [3.40, 3.00, 3.40, 4.40],
            "b365_a": [3.90, 2.80, 2.10, 9.00],
        }
    )


def proportional(odds: list[float]) -> np.ndarray:
    implied = 1.0 / np.array(odds)
    return implied / implied.sum()


def test_the_population_holds_each_selection_early_and_at_the_close() -> None:
    population = control_population(
        match_table(), book="ps", demargin="proportional", first_season=2023
    )
    assert list(population.columns) == [
        "match_id",
        "market_id",
        "selection",
        "season",
        "batch",
        "p_early",
        "p_close",
    ]
    assert population.groupby("match_id").size().to_dict() == {"c": 3, "d": 3}
    for _, rows in population.groupby("match_id"):
        assert sorted(rows["selection"]) == ["away", "draw", "home"]
    assert set(population["market_id"]) == {"1X2"}

    def of(match: str, column: str) -> np.ndarray:
        rows = population[population["match_id"] == match].set_index("selection")
        return rows.loc[["home", "draw", "away"], column].to_numpy()

    np.testing.assert_allclose(of("c", "p_early"), proportional([3.40, 3.60, 2.10]))
    np.testing.assert_allclose(of("c", "p_close"), proportional([3.30, 3.70, 2.15]))
    np.testing.assert_allclose(of("d", "p_early"), proportional([1.50, 4.20, 7.00]))
    np.testing.assert_allclose(of("d", "p_close"), proportional([1.45, 4.40, 7.50]))


def test_the_population_starts_at_the_first_season_and_keeps_each_matchs_coupon() -> (
    None
):
    both = control_population(
        match_table(), book="ps", demargin="proportional", first_season=2022
    )
    assert sorted(set(both["season"])) == [2022, 2023]
    later = control_population(
        match_table(), book="ps", demargin="proportional", first_season=2023
    )
    assert sorted(set(later["season"])) == [2023]
    batches = later.groupby("match_id")["batch"].first()
    assert batches["c"] == pd.Timestamp("2023-09-01")
    assert batches["d"] == pd.Timestamp("2023-09-08")
    assert control_population(
        match_table(), book="ps", demargin="proportional", first_season=2024
    ).empty


@pytest.mark.parametrize("method", ["proportional", "power", "shin"])
def test_the_population_takes_the_margin_out_as_the_backtest_did(method: str) -> None:
    population = control_population(
        match_table(), book="ps", demargin=method, first_season=2023
    )
    row = population[population["match_id"] == "d"]
    expected = remove_margin(np.array([[1.50, 4.20, 7.00]]), method)[0]
    np.testing.assert_allclose(row["p_early"], expected)


def test_the_methods_disagree_on_the_long_shot() -> None:
    by_method = {
        method: control_population(
            match_table(), book="ps", demargin=method, first_season=2023
        )
        .query("match_id == 'd' and selection == 'away'")["p_early"]
        .iloc[0]
        for method in ("proportional", "power", "shin")
    }
    assert len({round(value, 6) for value in by_method.values()}) == 3


def test_an_unknown_margin_removal_is_refused() -> None:
    with pytest.raises(ValueError) as caught:
        control_population(
            match_table(), book="ps", demargin="wrong", first_season=2023
        )
    assert str(caught.value) == (
        "unknown margin removal 'wrong'; choose from ['proportional', 'power', 'shin']"
    )


@pytest.mark.parametrize("column", MATCH_NEEDED)
def test_the_match_table_must_have_what_settling_needs(column: str) -> None:
    with pytest.raises(
        ValueError, match=f"the match table is missing columns: \\['{column}'\\]"
    ):
        control_population(
            match_table().drop(columns=column),
            book="ps",
            demargin="proportional",
            first_season=2023,
        )


def test_a_book_without_quotes_has_an_empty_population() -> None:
    population = control_population(
        match_table(), book="avg", demargin="proportional", first_season=2023
    )
    assert population.empty
    assert list(population.columns) == [
        "match_id",
        "market_id",
        "selection",
        "season",
        "batch",
        "p_early",
        "p_close",
    ]


# --------------------------------------------------------------------------- #
# Every price of a book
# --------------------------------------------------------------------------- #
def price_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nine Bet365 prices at the edges of the engine's limits, and the sharp book's view.

    Dropped by the default limits: ``m1`` away (odds above 8), ``m2`` home (odds
    below 1.30) and ``m3`` home (probability just under 5%).
    """
    enriched = pd.DataFrame(
        {
            "match_id": ["m1", "m2", "m3", "m9"],
            "b365_h": [1.30, 1.29, 2.0, 2.0],
            "b365_d": [8.0, 3.0, 2.0, 2.0],
            "b365_a": [8.1, 3.0, 2.0, 2.0],
        }
    )
    rows = []
    for match_id, early, close in (
        ("m1", (0.50, 0.125, 0.125), (0.55, 0.10, 0.15)),
        ("m2", (0.70, 0.15, 0.05), (0.60, 0.20, 0.05)),
        ("m3", (0.04999, 0.50, 0.45), (0.05, 0.45, 0.50)),
    ):
        for selection, p_early, p_close in zip(("home", "draw", "away"), early, close):
            rows.append(
                {
                    "match_id": match_id,
                    "market_id": "1X2",
                    "selection": selection,
                    "season": 2023,
                    "batch": pd.Timestamp("2023-09-01"),
                    "p_early": p_early,
                    "p_close": p_close,
                }
            )
    rows.append(control_row("m4", "1X2", "home", 0.5, 0.0, "2023-09-01"))
    return enriched, pd.DataFrame(rows)


def kept(quotes: pd.DataFrame) -> list[str]:
    return [f"{m}:{s}" for m, s in zip(quotes["match_id"], quotes["selection"])]


def test_every_price_within_the_engines_limits_is_a_quote() -> None:
    enriched, population = price_tables()
    quotes = quote_table(enriched, population, "b365")
    assert sorted(kept(quotes)) == [
        "m1:draw",
        "m1:home",
        "m2:away",
        "m2:draw",
        "m3:away",
        "m3:draw",
    ]


def test_the_limits_include_their_ends() -> None:
    enriched, population = price_tables()
    quotes = quote_table(enriched, population, "b365")
    assert (
        quotes.loc[
            quotes["selection"].eq("home") & quotes["match_id"].eq("m1"), "odds"
        ].iloc[0]
        == 1.30
    )
    assert quotes["odds"].max() == 8.0
    assert quotes["p_early"].min() == 0.05


def test_the_engines_limits_are_the_ones_given() -> None:
    enriched, population = price_tables()
    cheap = quote_table(enriched, population, "b365", ValueConfig(max_odds=3.0))
    assert sorted(kept(cheap)) == [
        "m1:home",
        "m2:away",
        "m2:draw",
        "m3:away",
        "m3:draw",
    ]
    likely = quote_table(enriched, population, "b365", ValueConfig(min_probability=0.2))
    assert sorted(kept(likely)) == ["m1:home", "m3:away", "m3:draw"]
    narrow = quote_table(
        enriched, population, "b365", ValueConfig(min_odds=1.5, max_odds=2.5)
    )
    assert sorted(kept(narrow)) == ["m3:away", "m3:draw"]


def test_a_quote_carries_its_gap_drift_and_closing_line_value() -> None:
    enriched, population = price_tables()
    quotes = quote_table(enriched, population, "b365").set_index(
        ["match_id", "selection"]
    )
    row = quotes.loc[("m1", "home")]
    assert row["odds"] == 1.30
    assert row["gap"] == pytest.approx(1.30 * 0.50 - 1.0)
    assert row["drift"] == pytest.approx(1.30 * 0.05)
    assert row["clv"] == pytest.approx(1.30 * 0.55 - 1.0)
    assert row["clv"] == pytest.approx(row["gap"] + row["drift"])
    row = quotes.loc[("m3", "draw")]
    assert row["gap"] == pytest.approx(2.0 * 0.50 - 1.0)
    assert row["drift"] == pytest.approx(2.0 * -0.05)
    for column in ("season", "batch", "p_early", "p_close"):
        assert column in quotes.columns


def test_prices_the_sharp_book_did_not_quote_are_left_out() -> None:
    enriched, population = price_tables()
    quotes = quote_table(enriched, population, "b365")
    assert "m9" not in set(quotes["match_id"])
    assert "m4" not in set(quotes["match_id"])


def test_a_book_without_prices_has_no_quotes() -> None:
    enriched, population = price_tables()
    assert quote_table(enriched, population, "avg").empty
    assert quote_table(enriched[["match_id"]], population, "b365").empty


# --------------------------------------------------------------------------- #
# The parts of the closing-line value
# --------------------------------------------------------------------------- #
def strategy(extra: bool = False) -> pd.DataFrame:
    chosen = strategy_bets(hand_bets(), "b365", 0.0)
    if extra:
        bets = hand_bets()
        others = without_bets(hand_population(), strategy_bets(bets, "b365", 0.0))
        typical = chosen["odds"] * similar_drift(chosen, others)
        chosen = chosen.assign(typical=typical, specific=chosen["drift"] - typical)
    return chosen


def parts_by(table: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    return {(row["weighting"], row["part"]): row for _, row in table.iterrows()}


def test_each_part_is_exact_per_unit_staked_and_per_bet() -> None:
    parts = parts_by(decomposition(strategy(), WEEKS, n_resamples=100))
    assert parts[("stake", "gap")]["value"] == pytest.approx(0.215)
    assert parts[("stake", "drift")]["value"] == pytest.approx(0.025)
    assert parts[("stake", "clv")]["value"] == pytest.approx(0.24)
    assert parts[("bet", "gap")]["value"] == pytest.approx(0.2125)
    assert parts[("bet", "drift")]["value"] == pytest.approx(0.0375)
    assert parts[("bet", "clv")]["value"] == pytest.approx(0.25)


def test_the_parts_come_in_a_fixed_order_with_the_stake_weighting_first() -> None:
    table = decomposition(strategy(), WEEKS, n_resamples=20)
    assert list(table.columns) == PART_COLUMNS
    assert table["weighting"].tolist() == ["stake"] * 3 + ["bet"] * 3
    assert table["part"].tolist() == ["gap", "drift", "clv"] * 2
    full = decomposition(strategy(extra=True), WEEKS, n_resamples=20)
    assert full["part"].tolist() == ["gap", "drift", "typical", "specific", "clv"] * 2
    assert list(PART_LABELS) == ["gap", "drift", "typical", "specific", "clv"]


def test_the_interval_resamples_the_weeks_of_the_calendar() -> None:
    table = decomposition(strategy(), WEEKS, n_resamples=200, seed=3)
    by = parts_by(table)
    # per week of the calendar: 1 Sep 2023, 8 Sep 2023, 15 Sep 2023 (no bet), 13 Sep 2024
    stake = np.array([40.0, 20.0, 0.0, 40.0])
    for part, per_week in (
        ("gap", [9.5, 4.0, 0.0, 8.0]),
        ("drift", [-5.5, 0.0, 0.0, 8.0]),
        ("clv", [4.0, 4.0, 0.0, 16.0]),
    ):
        low, high = cluster_bootstrap(
            np.array(per_week), stake, n_resamples=200, seed=3
        )
        assert by[("stake", part)]["low"] == pytest.approx(low)
        assert by[("stake", part)]["high"] == pytest.approx(high)
    count = np.array([2.0, 1.0, 0.0, 1.0])
    for part, per_week in (
        ("gap", [0.45, 0.2, 0.0, 0.2]),
        ("drift", [-0.05, 0.0, 0.0, 0.2]),
        ("clv", [0.4, 0.2, 0.0, 0.4]),
    ):
        low, high = cluster_bootstrap(
            np.array(per_week), count, n_resamples=200, seed=3
        )
        assert by[("bet", part)]["low"] == pytest.approx(low)
        assert by[("bet", part)]["high"] == pytest.approx(high)


def test_weeks_without_a_bet_widen_the_interval_not_the_estimate() -> None:
    bets = strategy()
    bet_weeks = pd.DatetimeIndex(["2023-09-01", "2023-09-08", "2024-09-13"])
    narrow = parts_by(decomposition(bets, bet_weeks, n_resamples=500, seed=1))
    wide = parts_by(
        decomposition(bets, bet_weeks.append(pd.DatetimeIndex(["2023-09-15", "2023-09-22"])).sort_values(), n_resamples=500, seed=1)
    )  # fmt: skip
    for part in ("gap", "drift", "clv"):
        assert wide[("stake", part)]["value"] == pytest.approx(
            narrow[("stake", part)]["value"]
        )
        assert (wide[("stake", part)]["high"] - wide[("stake", part)]["low"]) > (
            narrow[("stake", part)]["high"] - narrow[("stake", part)]["low"]
        )


def test_every_part_is_resampled_with_the_same_weeks() -> None:
    seen: list[tuple[int, int, int]] = []
    real = module.cluster_bootstrap

    def spy(
        numerator: np.ndarray, denominator: np.ndarray, **options: int
    ) -> tuple[float, float]:
        seen.append((len(numerator), options["n_resamples"], options["seed"]))
        return real(numerator, denominator, **options)

    original = module.cluster_bootstrap
    try:
        module.cluster_bootstrap = spy  # type: ignore[assignment]
        decomposition(strategy(extra=True), WEEKS, n_resamples=77, seed=5)
    finally:
        module.cluster_bootstrap = original
    # gap, drift, specific and clv, per unit staked and per bet; typical has none
    assert seen == [(4, 77, 5)] * 8


def test_the_similar_quotes_drift_has_no_interval_of_its_own() -> None:
    parts = parts_by(decomposition(strategy(extra=True), WEEKS, n_resamples=50))
    for weighting in ("stake", "bet"):
        typical = parts[(weighting, "typical")]
        assert np.isnan(typical["low"]) and np.isnan(typical["high"])
        assert np.isfinite(typical["value"])
        assert np.isfinite(parts[(weighting, "specific")]["low"])


def test_the_drift_splits_into_what_similar_quotes_do_and_what_is_left() -> None:
    parts = parts_by(decomposition(strategy(extra=True), WEEKS, n_resamples=50))
    # odds times 0.02, -0.01, 0 and 0.01: 0.08, -0.025, 0 and 0.08
    assert parts[("stake", "typical")]["value"] == pytest.approx(0.0325)
    assert parts[("stake", "specific")]["value"] == pytest.approx(-0.0075)
    assert parts[("bet", "typical")]["value"] == pytest.approx(0.03375)
    assert parts[("bet", "specific")]["value"] == pytest.approx(0.00375)
    for weighting in ("stake", "bet"):
        assert parts[(weighting, "typical")]["value"] + parts[(weighting, "specific")][
            "value"
        ] == pytest.approx(parts[(weighting, "drift")]["value"])
        assert parts[(weighting, "gap")]["value"] + parts[(weighting, "drift")][
            "value"
        ] == pytest.approx(parts[(weighting, "clv")]["value"])


def test_identical_weeks_give_a_point_interval() -> None:
    bets = pd.concat([strategy().iloc[[0]]] * 3, ignore_index=True)
    bets["batch"] = pd.to_datetime(["2023-09-01", "2023-09-08", "2023-09-15"])
    parts = parts_by(decomposition(bets, WEEKS[:3], n_resamples=100))
    gap = parts[("stake", "gap")]
    assert gap["low"] == pytest.approx(gap["value"])
    assert gap["high"] == pytest.approx(gap["value"])


def test_a_bet_in_a_week_outside_the_calendar_is_refused() -> None:
    with pytest.raises(ValueError, match="1 bets fall in weeks outside the calendar"):
        decomposition(strategy(), WEEKS[:3], n_resamples=10)


def test_batches_stored_at_another_resolution_are_still_found() -> None:
    bets = strategy().assign(
        batch=lambda frame: frame["batch"].astype("datetime64[us]")
    )
    reference = parts_by(decomposition(strategy(), WEEKS, n_resamples=50))
    coarse = parts_by(decomposition(bets, WEEKS, n_resamples=50))
    assert coarse[("stake", "clv")]["value"] == pytest.approx(
        reference[("stake", "clv")]["value"]
    )
    assert coarse[("stake", "clv")]["low"] == pytest.approx(
        reference[("stake", "clv")]["low"]
    )


def test_a_strategy_that_staked_nothing_has_no_stake_weighted_value() -> None:
    bets = strategy().assign(stake=0.0)
    with np.errstate(divide="raise", invalid="raise"):
        table = decomposition(bets, WEEKS, n_resamples=10)
    stake = table[table["weighting"] == "stake"]
    assert stake["value"].isna().all()
    assert stake["low"].isna().all() and stake["high"].isna().all()
    assert table[table["weighting"] == "bet"]["value"].notna().all()


# --------------------------------------------------------------------------- #
# Do a few bets carry it?
# --------------------------------------------------------------------------- #
def test_taking_the_best_bets_away_shows_what_is_left() -> None:
    table = concentration(strategy())
    assert table["removed"].tolist() == [0, 1, 3]
    assert table["left"].tolist() == [4, 3, 1]
    # CLV 0.4, 0.4, 0.2, 0.0
    np.testing.assert_allclose(table["mean_clv"], [0.25, 0.2, 0.0], atol=1e-12)
    np.testing.assert_allclose(table["median_clv"], [0.3, 0.2, 0.0], atol=1e-12)
    assert np.isnan(table["share"].iloc[0])
    np.testing.assert_allclose(table["share"].iloc[1:], [0.4, 1.0], atol=1e-12)


def test_the_best_bets_are_those_with_the_highest_clv_whatever_the_order() -> None:
    bets = strategy().iloc[[3, 1, 0, 2]].reset_index(drop=True)
    np.testing.assert_allclose(
        concentration(bets, trims=(0, 2))["mean_clv"], [0.25, 0.1], atol=1e-12
    )


def test_a_trim_that_would_leave_nothing_is_left_out() -> None:
    assert concentration(strategy(), trims=(0, 4, 5, 9))["removed"].tolist() == [0]
    assert concentration(strategy().iloc[:1])["removed"].tolist() == [0]


def test_the_share_of_a_total_that_is_not_positive_is_not_defined() -> None:
    bets = strategy().assign(clv=[-0.1, -0.2, 0.05, -0.3])
    table = concentration(bets, trims=(0, 1, 2))
    assert table["share"].isna().all()
    np.testing.assert_allclose(table["mean_clv"], [-0.1375, -0.2, -0.25], atol=1e-12)


def test_a_total_of_exactly_zero_has_no_share_and_no_division() -> None:
    bets = strategy().assign(clv=[0.1, -0.1, 0.0, 0.0])
    with np.errstate(divide="raise", invalid="raise"):
        table = concentration(bets, trims=(0, 1, 2))
    assert table["share"].isna().all()
    np.testing.assert_allclose(table["mean_clv"], [0.0, -0.1 / 3, -0.05], atol=1e-12)


def test_every_trim_of_the_verdict_table_is_there_with_enough_bets() -> None:
    bets = (
        strategy()
        .iloc[[0] * 12]
        .reset_index(drop=True)
        .assign(clv=np.linspace(0.6, 0.05, 12))
    )
    table = concentration(bets)
    assert table["removed"].tolist() == [0, 1, 3, 5, 10]
    assert table["left"].tolist() == [12, 11, 9, 7, 2]


def test_the_share_can_exceed_the_whole_when_the_rest_loses() -> None:
    bets = strategy().assign(clv=[0.5, -0.1, -0.1, 0.0])
    share = concentration(bets, trims=(1,))["share"].iloc[0]
    assert share == pytest.approx(0.5 / 0.3)


# --------------------------------------------------------------------------- #
# Does it hold everywhere?
# --------------------------------------------------------------------------- #
def test_a_breakdown_gives_means_per_group_and_the_hit_rate() -> None:
    table = breakdown(strategy(), "season")
    assert list(table.columns) == ["season", *BREAKDOWN_VALUES]
    assert table["season"].tolist() == [2023, 2024]
    assert table["bets"].tolist() == [3, 1]
    np.testing.assert_allclose(table["gap"], [(0.2 + 0.25 + 0.2) / 3, 0.2])
    np.testing.assert_allclose(
        table["drift"], [(0.2 - 0.25 + 0.0) / 3, 0.2], atol=1e-12
    )
    np.testing.assert_allclose(table["clv"], [0.2, 0.4], atol=1e-12)
    np.testing.assert_allclose(table["hit"], [1 / 3, 1.0])
    np.testing.assert_allclose(table["expected"], [(0.35 + 0.4 + 0.4) / 3, 0.175])


def test_a_groups_clv_is_its_mean_not_its_median() -> None:
    bets = strategy().assign(
        clv=[0.0, 0.0, 0.3, 0.0], gap=[0.0, 0.0, 0.6, 0.0], season=2023
    )
    table = breakdown(bets, "season")
    assert table["clv"].iloc[0] == pytest.approx(0.075)
    assert table["gap"].iloc[0] == pytest.approx(0.15)


def test_a_breakdown_by_market() -> None:
    table = breakdown(strategy(), "market_id")
    assert table["market_id"].tolist() == ["1X2", "OU:2.5"]
    assert table["bets"].tolist() == [3, 1]
    np.testing.assert_allclose(table["hit"], [1 / 3, 1.0])


def test_price_ranges_come_in_their_natural_order_and_only_those_with_bets() -> None:
    bets = strategy().assign(odds_range=odds_band(strategy()["odds"]))
    table = breakdown(bets, "odds_range")
    # 2.5 and 3.0 are in 2.0-3.0, 4.0 in 3.0-4.0, 8.0 in 6.0-8.0
    assert table["odds_range"].astype(str).tolist() == ["2.0-3.0", "3.0-4.0", "6.0-8.0"]
    assert table["bets"].tolist() == [2, 1, 1]


def test_the_largest_and_smallest_gaps_and_the_quartiles() -> None:
    table = gap_distribution(strategy())
    assert table["statistic"].tolist() == [
        "lowest",
        "25%",
        "median",
        "75%",
        "90%",
        "highest",
    ]
    np.testing.assert_allclose(table["gap"], [0.2, 0.2, 0.2, 0.2125, 0.235, 0.25])


def test_each_statistic_is_the_quantile_it_names() -> None:
    bets = (
        strategy()
        .iloc[[0] * 11]
        .reset_index(drop=True)
        .assign(gap=np.linspace(0.0, 1.0, 11))
    )
    table = gap_distribution(bets)
    np.testing.assert_allclose(table["gap"], [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])


def test_the_best_bets_are_listed_with_their_match() -> None:
    named = attach_matches(strategy(), hand_names())
    table = best_bets(named, top=3)
    assert table["match"].tolist() == [
        "Arsenal v Chelsea",
        "Spurs v Everton",
        "Roma v Milan",
    ]
    assert table["date"].tolist() == ["2023-09-02", "2024-09-14", "2023-09-09"]
    assert table["league"].tolist() == ["Premier League", "Premier League", "Serie A"]
    assert table["selection"].tolist() == ["home", "away", "draw"]
    assert table["won"].tolist() == [False, True, False]
    assert list(table.columns) == module.BEST_COLUMNS
    assert table["clv"].is_monotonic_decreasing


def test_without_names_a_match_is_its_id_and_the_league_is_blank() -> None:
    table = best_bets(strategy(), top=2)
    assert table["match"].tolist() == ["m1", "m4"]
    assert table["league"].tolist() == ["", ""]


def test_ties_keep_the_order_of_the_bets() -> None:
    # m1 and m4 both have a CLV of 0.4
    assert best_bets(strategy(), top=2)["match"].tolist() == ["m1", "m4"]
    assert best_bets(strategy().iloc[[3, 0, 1, 2]].reset_index(drop=True), top=2)[
        "match"
    ].tolist() == ["m4", "m1"]


def test_many_ties_keep_the_order_of_the_bets() -> None:
    ids = [f"m{i:02d}" for i in range(60)]
    clv = np.tile([0.2, 0.1, 0.3], 20)
    bets = (
        strategy().iloc[[0] * 60].reset_index(drop=True).assign(match_id=ids, clv=clv)
    )
    # Python's own sort is stable: equal CLV keeps the order of the bets
    expected = [ids[i] for i in sorted(range(60), key=lambda i: -clv[i])]
    assert best_bets(bets, top=60)["match"].tolist() == expected


def test_a_match_is_named_only_when_both_teams_are_known() -> None:
    home_only = strategy().assign(home_team="Arsenal")
    away_only = strategy().assign(away_team="Chelsea")
    assert best_bets(home_only, top=2)["match"].tolist() == ["m1", "m4"]
    assert best_bets(away_only, top=2)["match"].tolist() == ["m1", "m4"]


def test_more_best_bets_than_bets_lists_every_bet() -> None:
    assert len(best_bets(strategy(), top=50)) == 4


def test_dates_stored_as_text_are_listed_as_dates() -> None:
    bets = strategy().assign(
        date=["2023-09-02", "2023-09-03", "2023-09-09", "2024-09-14"]
    )
    assert best_bets(bets, top=1)["date"].tolist() == ["2023-09-02"]


# --------------------------------------------------------------------------- #
# What the sharp book did next, for every price
# --------------------------------------------------------------------------- #
def quote_rows() -> pd.DataFrame:
    """Five prices in four weeks: gap, drift and the week of each."""
    rows = [
        (-0.20, 0.00, "2023-09-01"),
        (-0.04, 0.01, "2023-09-01"),
        (0.05, 0.02, "2023-09-08"),
        (0.08, -0.04, "2023-09-08"),
        (0.045, 0.03, "2023-09-15"),
    ]
    frame = pd.DataFrame(rows, columns=["gap", "drift", "batch"])
    frame["batch"] = pd.to_datetime(frame["batch"])
    frame["clv"] = frame["gap"] + frame["drift"]
    return frame


def test_each_range_has_its_prices_and_means() -> None:
    table = response_table(quote_rows(), WEEKS, min_edge=0.04, n_resamples=50)
    assert list(table.columns) == RESPONSE_COLUMNS
    assert table["gap_range"].tolist() == ["below -10%", "-5% to -2%", "4% or more"]
    assert table["quotes"].tolist() == [1, 1, 3]
    np.testing.assert_allclose(table["gap"], [-0.20, -0.04, (0.05 + 0.08 + 0.045) / 3])
    np.testing.assert_allclose(
        table["drift"], [0.0, 0.01, (0.02 - 0.04 + 0.03) / 3], atol=1e-12
    )
    np.testing.assert_allclose(table["clv"], [-0.2, -0.03, (0.07 + 0.04 + 0.075) / 3])


def test_only_the_last_range_is_the_region() -> None:
    table = response_table(quote_rows(), WEEKS, min_edge=0.04, n_resamples=50)
    assert table["region"].tolist() == [False, False, True]
    higher = response_table(quote_rows(), WEEKS, min_edge=0.07, n_resamples=50)
    assert higher["gap_range"].tolist() == [
        "below -10%",
        "-5% to -2%",
        "4% to 7%",
        "7% or more",
    ]
    assert higher["quotes"].tolist() == [1, 1, 2, 1]
    assert higher["region"].tolist() == [False, False, False, True]


def test_when_no_price_reaches_the_minimum_edge_there_is_no_region() -> None:
    quotes = quote_rows().iloc[:2]
    table = response_table(quotes, WEEKS, min_edge=0.04, n_resamples=50)
    assert table["region"].tolist() == [False, False]


def test_every_range_has_week_clustered_intervals_for_drift_and_clv() -> None:
    table = response_table(quote_rows(), WEEKS, min_edge=0.04, n_resamples=100, seed=2)
    region = table[table["region"]].iloc[0]
    count = np.array([0.0, 2.0, 1.0, 0.0])
    low, high = cluster_bootstrap(
        np.array([0.0, 0.02 - 0.04, 0.03, 0.0]), count, n_resamples=100, seed=2
    )
    assert region["drift_low"] == pytest.approx(low)
    assert region["drift_high"] == pytest.approx(high)
    low, high = cluster_bootstrap(
        np.array([0.0, 0.07 + 0.04, 0.075, 0.0]), count, n_resamples=100, seed=2
    )
    assert region["clv_low"] == pytest.approx(low)
    assert region["clv_high"] == pytest.approx(high)


def many_quotes() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """120 prices spread over 30 weeks: enough for the intervals to depend on the seed."""
    rng = np.random.default_rng(5)
    calendar = pd.date_range("2023-09-01", periods=30, freq="7D")
    gap = rng.uniform(0.04, 0.2, 120)
    drift = rng.normal(0.0, 0.1, 120)
    frame = pd.DataFrame(
        {
            "gap": gap,
            "drift": drift,
            "clv": gap + drift,
            "batch": calendar[rng.integers(0, 30, 120)],
        }
    )
    return frame, calendar


def weekly_sums(
    values: pd.Series, batch: pd.Series, calendar: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {"top": values.to_numpy(), "bottom": 1.0, "batch": batch.to_numpy()}
    )
    sums = frame.groupby("batch").sum().reindex(calendar, fill_value=0.0)
    return sums["top"].to_numpy(), sums["bottom"].to_numpy()


def test_the_seed_and_the_number_of_resamples_are_those_asked_for() -> None:
    quotes, calendar = many_quotes()
    table = response_table(quotes, calendar, min_edge=0.04, n_resamples=100, seed=2)
    row = table.iloc[0]
    for column in ("drift", "clv"):
        top, bottom = weekly_sums(quotes[column], quotes["batch"], calendar)
        low, high = cluster_bootstrap(top, bottom, n_resamples=100, seed=2)
        assert row[f"{column}_low"] == pytest.approx(low)
        assert row[f"{column}_high"] == pytest.approx(high)
        other_seed = cluster_bootstrap(top, bottom, n_resamples=100, seed=0)
        other_count = cluster_bootstrap(top, bottom, n_resamples=2000, seed=2)
        assert (low, high) != other_seed
        assert (low, high) != other_count


def test_a_price_in_a_week_outside_the_calendar_cannot_be_clustered() -> None:
    quotes = quote_rows().assign(batch=pd.Timestamp("2030-01-04"))
    with pytest.raises(ValueError, match="5 prices fall in weeks outside the calendar"):
        response_table(quotes, WEEKS, min_edge=0.04, n_resamples=10)
    one = quote_rows().assign(
        batch=lambda frame: frame["batch"].where(
            frame["gap"] != 0.08, pd.Timestamp("2030-01-04")
        )
    )
    with pytest.raises(ValueError, match="1 prices fall in weeks outside the calendar"):
        response_table(one, WEEKS, min_edge=0.04, n_resamples=10)


# --------------------------------------------------------------------------- #
# The whole autopsy
# --------------------------------------------------------------------------- #
def test_the_autopsy_of_a_strategy_alone() -> None:
    result = hand_result()
    assert result.book == "b365" and result.weight == 0.0
    assert len(result.bets) == 4
    assert result.weeks == 4
    assert not result.compared and result.similar == 0
    assert result.response.empty
    assert list(result.response.columns) == RESPONSE_COLUMNS
    assert result.config == ValueConfig()
    assert result.parts["part"].tolist() == ["gap", "drift", "clv"] * 2
    assert result.seasons["season"].tolist() == [2023, 2024]
    assert result.markets["market_id"].tolist() == ["1X2", "OU:2.5"]
    assert result.leagues.empty
    assert list(result.leagues.columns) == ["league", *BREAKDOWN_VALUES]
    assert list(result.bets["odds_range"].cat.categories) == list(ODDS_LABELS)
    assert result.best["match"].tolist() == ["m1", "m4", "m3", "m2"]


def test_the_calendar_covers_every_strategys_weeks() -> None:
    # 1, 8 and 15 September 2023 and 13 September 2024: m5 and m6 add no week of their own
    assert hand_result().weeks == 4
    alone = autopsy(
        hand_bets()[lambda frame: frame["bookmaker"].eq("b365") & frame["weight"].eq(0.0)],
        n_resamples=10,
    )  # fmt: skip
    assert alone.weeks == 3


def test_the_strategy_is_chosen_by_book_and_weight() -> None:
    other = autopsy(hand_bets(), book="b365", weight=0.5, n_resamples=10)
    assert other.book == "b365" and other.weight == 0.5
    assert other.bets["match_id"].tolist() == ["m5"]
    assert other.weeks == 4


def test_names_give_each_bet_a_league() -> None:
    result = hand_result(matches=hand_names())
    assert result.leagues["league"].tolist() == ["La Liga", "Premier League", "Serie A"]
    assert result.leagues["bets"].tolist() == [1, 2, 1]
    assert result.best["match"].iloc[0] == "Arsenal v Chelsea"


def test_a_bet_whose_match_is_unknown_is_in_the_league_unknown() -> None:
    names = hand_names()
    result = hand_result(matches=names[names["match_id"] != "m3"])
    assert "unknown" in result.leagues["league"].tolist()
    assert result.leagues.set_index("league").loc["unknown", "bets"] == 1


def test_the_drift_is_compared_with_similar_quotes_when_the_population_matches() -> (
    None
):
    result = hand_result(population=hand_population())
    assert result.compared
    assert result.similar == 10
    assert result.weeks == 5  # the controls are quoted in the week of 22 September
    parts = parts_by(result.parts)
    assert parts[("stake", "typical")]["value"] == pytest.approx(0.0325)
    assert parts[("stake", "specific")]["value"] == pytest.approx(-0.0075)
    assert result.bets["typical"].tolist() == pytest.approx([0.08, -0.025, 0.0, 0.08])
    assert result.bets["specific"].tolist() == pytest.approx([0.12, -0.225, 0.0, 0.12])
    assert result.response.empty  # no prices to look at without the match table


def test_a_population_that_does_not_match_is_ignored_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    population = hand_population()
    population.loc[0, "p_early"] += 0.01
    with caplog.at_level(logging.WARNING, logger="src.autopsy"):
        result = hand_result(population=population)
    assert not result.compared
    assert "typical" not in result.bets.columns
    assert "does not reproduce the bets' probabilities" in caplog.text
    assert "--reference and --demargin of the backtest that made them" in caplog.text
    assert result.weeks == 4  # its weeks are not borrowed either
    assert result.parts["part"].tolist() == ["gap", "drift", "clv"] * 2


def test_a_bet_with_no_similar_quote_leaves_the_comparison_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    population = hand_population()
    population = population[~population["match_id"].isin(["c5", "c6"])]
    with caplog.at_level(logging.WARNING, logger="src.autopsy"):
        result = hand_result(population=population)
    assert not result.compared and result.similar == 0
    assert "typical" not in result.bets.columns
    assert "Some bets have no similarly priced quote" in caplog.text
    assert result.weeks == 5  # the population still matches the bets


def test_an_empty_population_is_a_population_that_does_not_match(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="src.autopsy"):
        result = hand_result(population=hand_population().iloc[0:0])
    assert not result.compared
    assert "does not reproduce" in caplog.text


def test_names_without_prices_leave_the_price_table_empty() -> None:
    result = hand_result(matches=hand_names(), population=hand_population())
    assert result.compared
    assert result.response.empty
    assert result.leagues["bets"].sum() == 4


def test_the_engines_limits_are_kept_with_the_result() -> None:
    config = ValueConfig(min_edge=0.07)
    assert hand_result(config=config).config is config


def test_an_unknown_strategy_cannot_be_taken_apart() -> None:
    with pytest.raises(ValueError, match="no bets for max at weight 0"):
        autopsy(hand_bets(), book="max", weight=0.0)


# --------------------------------------------------------------------------- #
# Simulated markets whose truth is known
# --------------------------------------------------------------------------- #
def tilt(p: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """A noisy view of the probabilities (log-normal noise, renormalised)."""
    view = p * np.exp(rng.normal(0.0, sigma, p.shape))
    return view / view.sum(axis=1, keepdims=True)


def jitter(p: np.ndarray, concentration: float, rng: np.random.Generator) -> np.ndarray:
    """A Dirichlet draw whose mean is ``p``: a martingale step."""
    draws = rng.gamma(concentration * p)
    return draws / draws.sum(axis=1, keepdims=True)


def put_quotes(
    frame: pd.DataFrame, prefix: str, p_1x2: np.ndarray, p_ou: np.ndarray, margin: float
) -> None:
    for market_id, probabilities in (("1X2", p_1x2), ("OU:2.5", p_ou)):
        odds = 1.0 / (probabilities * (1.0 + margin))
        for position, (_, suffix) in enumerate(MARKETS[market_id]):
            frame[f"{prefix}_{suffix}"] = odds[:, position]


def make_market(
    *, seed: int, follow: float = 0.0, structural: float = 0.0, per_season: int = 1_500
) -> pd.DataFrame:
    """Matches where Bet365 prices a noisy copy of Pinnacle's early view.

    Pinnacle's early fair probability is ``early`` and Bet365 prices ``soft``, a noisy
    copy of it. Between the early and the closing price the market moves to ``early +
    follow * (soft - early)``, raised to ``1 + structural`` and renormalised, with
    martingale noise on top, and the results come from the closing probabilities.

    * ``follow = 0``: Bet365's deviations are noise the sharp book ignores.
    * ``follow > 0``: the sharp book follows Bet365, so its price carried information.
    * ``follow < 0``: the sharp book moves away from Bet365: a slow price.
    * ``structural > 0``: favourites gain and long shots lose probability for every
      book alike, whoever quotes them.
    """
    seasons = (2022, 2023, 2024)
    rng = np.random.default_rng(seed)
    n = per_season * len(seasons)
    early = {
        "1X2": rng.dirichlet([4.0, 3.0, 3.0], n),
        "OU": rng.dirichlet([5.0, 5.0], n),
    }
    soft = {name: tilt(p, 0.06, rng) for name, p in early.items()}
    close = {}
    for name, before in early.items():
        view = before + follow * (soft[name] - before)
        view = view ** (1.0 + structural)
        close[name] = jitter(view / view.sum(axis=1, keepdims=True), 300.0, rng)
    result = (rng.random(n)[:, None] > close["1X2"].cumsum(axis=1)).sum(axis=1)
    over = rng.random(n) < close["OU"][:, 0]
    draw_goals = np.where(over, 2, 0)
    home = np.where(
        result == 0, np.where(over, 3, 1), np.where(result == 1, draw_goals, 0)
    )
    away = np.where(
        result == 2, np.where(over, 3, 1), np.where(result == 1, draw_goals, 0)
    )
    season = np.repeat(seasons, per_season)
    offsets = np.concatenate(
        [np.sort(rng.integers(0, 280, per_season)) for _ in seasons]
    )
    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "season": season,
            "date": pd.to_datetime([f"{s}-08-10" for s in season])
            + pd.to_timedelta(offsets, unit="D"),
            "home_goals": home,
            "away_goals": away,
            "league": rng.choice(["Premier League", "La Liga", "Serie A"], n),
            "home_team": [f"Home {i % 20}" for i in range(n)],
            "away_team": [f"Away {i % 20}" for i in range(n)],
        }
    )  # fmt: skip
    put_quotes(frame, "ps", early["1X2"], early["OU"], 0.03)
    put_quotes(frame, "psc", close["1X2"], close["OU"], 0.027)
    put_quotes(frame, "b365", soft["1X2"], soft["OU"], 0.055)
    return frame


def expected_goals(frame: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward predictions with noise: enough to price every match."""
    rng = np.random.default_rng(0)
    n = len(frame)
    return pd.DataFrame(
        {
            "match_id": frame["match_id"],
            "date": frame["date"],
            "season": frame["season"],
            "pred_home_xg": 1.4 * np.exp(rng.normal(0, 0.2, n)),
            "pred_away_xg": 1.1 * np.exp(rng.normal(0, 0.2, n)),
        }
    )


@dataclass(frozen=True)
class Run:
    frame: pd.DataFrame
    backtest: BacktestResult
    population: pd.DataFrame
    result: Autopsy


def run_market(*, follow: float = 0.0, structural: float = 0.0) -> Run:
    frame = make_market(seed=1, follow=follow, structural=structural)
    backtest = run_backtest(
        frame,
        expected_goals(frame),
        books=("b365",),
        weights=(0.0,),
        n_resamples=300,
        min_calibration_matches=500,
    )
    population = control_population(
        frame, book="ps", demargin="proportional", first_season=2023
    )
    result = autopsy(
        backtest.bets,
        book="b365",
        weight=0.0,
        matches=frame,
        population=population,
        n_resamples=300,
    )
    return Run(frame, backtest, population, result)


@pytest.fixture(scope="module")
def noise() -> Run:
    """Bet365's price gaps are noise the sharp book ignores."""
    return run_market()


@pytest.fixture(scope="module")
def followed() -> Run:
    """The sharp book follows Bet365: the price gap was information."""
    return run_market(follow=0.5)


@pytest.fixture(scope="module")
def slow() -> Run:
    """The sharp book moves away from Bet365's price: a slow price."""
    return run_market(follow=-0.5)


@pytest.fixture(scope="module")
def pattern() -> Run:
    """Long shots lose probability towards the close for every book alike."""
    return run_market(structural=0.15)


def stake_part(run: Run, part: str) -> pd.Series:
    return parts_by(run.result.parts)[("stake", part)]


def test_the_autopsys_clv_is_the_backtests_own_with_its_interval(noise: Run) -> None:
    summary = noise.backtest.summary
    row = summary[(summary["bookmaker"] == "b365") & (summary["weight"] == 0.0)].iloc[0]
    clv = stake_part(noise, "clv")
    assert noise.result.bets.shape[0] == row["bets"] > 100
    assert clv["value"] == pytest.approx(row["clv"], abs=1e-12)
    assert clv["low"] == pytest.approx(row["clv_low"], abs=1e-12)
    assert clv["high"] == pytest.approx(row["clv_high"], abs=1e-12)


def test_at_weight_zero_the_price_gap_is_the_edge_the_engine_saw(noise: Run) -> None:
    bets = noise.backtest.bets
    chosen = bets[(bets["bookmaker"] == "b365") & (bets["weight"] == 0.0)]
    np.testing.assert_allclose(noise.result.bets["gap"], chosen["edge"])
    assert noise.result.bets["gap"].min() >= ValueConfig().min_edge - 1e-12


def test_the_match_table_reproduces_the_backtests_probabilities(noise: Run) -> None:
    assert noise.result.compared
    assert reproduces(noise.result.bets, noise.population)
    assert noise.result.similar == len(noise.population) - len(noise.result.bets)


def test_the_options_reach_every_table(noise: Run) -> None:
    config = ValueConfig(min_edge=0.07, min_odds=1.5, max_odds=6.0, min_probability=0.1)
    result = autopsy(
        noise.backtest.bets,
        book="b365",
        weight=0.0,
        matches=noise.frame,
        population=noise.population,
        config=config,
        n_resamples=60,
        seed=3,
    )
    calendar = module._weeks(noise.backtest.bets["batch"]).union(
        module._weeks(noise.population["batch"])
    )
    quotes = quote_table(noise.frame, noise.population, "b365", config)
    expected = response_table(quotes, calendar, min_edge=0.07, n_resamples=60, seed=3)
    pd.testing.assert_frame_equal(result.response, expected)
    assert result.response["gap_range"].iloc[-1] == "7% or more"
    assert result.config == config
    parts = decomposition(result.bets, calendar, n_resamples=60, seed=3)
    pd.testing.assert_frame_equal(result.parts, parts)
    # the same tables with other options are not the same: the check above can fail
    other = response_table(quotes, calendar, min_edge=0.07, n_resamples=60, seed=0)
    assert not result.response.equals(other)
    by_seed = decomposition(result.bets, calendar, n_resamples=60, seed=0)
    by_count = decomposition(result.bets, calendar, n_resamples=300, seed=3)
    assert not result.parts.equals(by_seed)
    assert not result.parts.equals(by_count)


def test_the_calendar_is_the_backtests(noise: Run) -> None:
    assert noise.result.weeks == noise.population["batch"].nunique()
    assert noise.result.weeks >= noise.result.bets["batch"].nunique()


def test_a_population_made_the_wrong_way_is_noticed(
    noise: Run, caplog: pytest.LogCaptureFixture
) -> None:
    power = control_population(
        noise.frame, book="ps", demargin="power", first_season=2023
    )
    with caplog.at_level(logging.WARNING, logger="src.autopsy"):
        result = autopsy(
            noise.backtest.bets,
            book="b365",
            weight=0.0,
            population=power,
            n_resamples=50,
        )
    assert not result.compared
    assert "does not reproduce the bets' probabilities" in caplog.text


def test_the_parts_add_up(noise: Run) -> None:
    for weighting in ("stake", "bet"):
        by = parts_by(noise.result.parts)
        assert by[(weighting, "gap")]["value"] + by[(weighting, "drift")][
            "value"
        ] == pytest.approx(by[(weighting, "clv")]["value"])
        assert by[(weighting, "typical")]["value"] + by[(weighting, "specific")][
            "value"
        ] == pytest.approx(by[(weighting, "drift")]["value"])


def test_every_bet_is_among_the_prices_the_engine_could_have_bet(noise: Run) -> None:
    region = noise.result.response[noise.result.response["region"]].iloc[0]
    assert region["quotes"] >= len(noise.result.bets)
    assert region["gap_range"] == "4% or more"
    assert noise.result.response["gap_range"].iloc[0] == "below -10%"


def test_bets_carry_their_league_and_teams(noise: Run) -> None:
    bets = noise.result.bets
    assert set(bets["league"]) <= {"Premier League", "La Liga", "Serie A"}
    assert noise.result.best["match"].str.match(r"Home \d+ v Away \d+").all()
    assert noise.result.leagues["bets"].sum() == len(bets)
    assert noise.result.seasons["season"].tolist() == [2023, 2024]


def test_when_the_sharp_book_ignores_the_price_the_clv_is_the_price_gap(
    noise: Run,
) -> None:
    gap, drift, clv = (stake_part(noise, part) for part in ("gap", "drift", "clv"))
    specific = stake_part(noise, "specific")
    assert drift["low"] < 0.0 < drift["high"]
    assert specific["low"] < 0.0 < specific["high"]
    assert abs(clv["value"] - gap["value"]) < 0.02
    region = noise.result.response[noise.result.response["region"]].iloc[0]
    assert region["drift_low"] < 0.0 < region["drift_high"]


def test_when_the_sharp_book_follows_the_price_the_gap_was_information(
    followed: Run,
) -> None:
    gap, drift, clv = (stake_part(followed, part) for part in ("gap", "drift", "clv"))
    assert drift["high"] < 0.0
    assert stake_part(followed, "specific")["high"] < 0.0
    assert clv["value"] < gap["value"] - 0.03
    region = followed.result.response[followed.result.response["region"]].iloc[0]
    assert region["drift_high"] < 0.0


def test_when_the_price_is_slow_the_sharp_book_keeps_moving_in_the_bettors_favour(
    slow: Run,
) -> None:
    gap, drift, clv = (stake_part(slow, part) for part in ("gap", "drift", "clv"))
    assert drift["low"] > 0.0
    assert stake_part(slow, "specific")["low"] > 0.0
    assert clv["value"] > gap["value"] + 0.03
    region = slow.result.response[slow.result.response["region"]].iloc[0]
    assert region["drift_low"] > 0.0


def test_a_pattern_of_the_whole_market_is_not_blamed_on_the_bets(pattern: Run) -> None:
    drift, typical = stake_part(pattern, "drift"), stake_part(pattern, "typical")
    specific = stake_part(pattern, "specific")
    assert drift["high"] < 0.0
    assert typical["value"] < -0.02
    assert specific["low"] < 0.0 < specific["high"]
    assert abs(specific["value"]) < abs(typical["value"]) / 4


def test_the_verdict_reads_each_market_correctly(
    noise: Run, followed: Run, slow: Run, pattern: Run
) -> None:
    quiet = "\n".join(verdict(noise.result))
    assert "did not move for or against the bettor" in quiet
    assert "the price gap behaves like noise the sharp book ignored" in quiet
    assert "the sharp book's move cannot be told from zero" in quiet
    informed = "\n".join(verdict(followed.result))
    assert (
        "the sharp book moved against the bettor after the price was taken" in informed
    )
    assert "part of the price gap was information rather than mispricing" in informed
    assert "the sharp book moved against the bettor:" not in informed.split("\n")[1]
    late = "\n".join(verdict(slow.result))
    assert "the sharp book kept moving in the bettor's favour after the price" in late
    assert "That is how a slow Bet365 price looks" in late
    assert "the sharp book moved in the bettor's favour" in late
    structural = "\n".join(verdict(pattern.result))
    assert "moved against the bettor after the price was taken" in structural
    assert "the price gap behaves like noise the sharp book ignored" in structural


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #
def test_text_is_aligned_with_names_flush_left_and_numbers_flush_right() -> None:
    table = pd.DataFrame(
        {"name": ["a", "long name"], "n": [5, 1234], "share": ["+1.0%", "-12.5%"]}
    )
    lines = module._render(table, left=["name"]).split("\n")
    assert lines == [
        "name          n   share",
        "a             5   +1.0%",
        "long name  1234  -12.5%",
    ]


def test_a_header_wider_than_its_cells_sets_the_column_width() -> None:
    table = pd.DataFrame({"a long header": ["x"], "b": [1]})
    assert module._render(table, left=["a long header"]).split("\n") == [
        "a long header  b",
        "x              1",
    ]
    assert module._render(table).split("\n") == [
        "a long header  b",
        "            x  1",
    ]


def test_aligned_text_has_no_trailing_spaces_and_keeps_indentation() -> None:
    table = pd.DataFrame({"part": ["gap", "  of it"], "value": ["+1%", "-2%"]})
    lines = module._render(table, left=["part"]).split("\n")
    assert all(line == line.rstrip() for line in lines)
    assert lines[2].startswith("  of it")
    assert lines[1].startswith("gap")


def test_a_last_column_flush_left_leaves_no_trailing_spaces() -> None:
    table = pd.DataFrame({"n": [1, 22], "result": ["won", "lost this one"]})
    assert module._render(table, left=["result"]).split("\n") == [
        " n  result",
        " 1  won",
        "22  lost this one",
    ]


def test_a_part_is_read_for_the_weighting_asked() -> None:
    parts = pd.DataFrame(
        [
            {"weighting": "bet", "part": "clv", "value": 0.5, "low": 0.4, "high": 0.6},
            {
                "weighting": "stake",
                "part": "clv",
                "value": 0.1,
                "low": 0.0,
                "high": 0.2,
            },
        ],
        columns=PART_COLUMNS,
    )
    stake = module._part(parts, "stake", "clv")
    bet = module._part(parts, "bet", "clv")
    assert stake is not None and bet is not None
    assert (stake["value"], bet["value"]) == (0.1, 0.5)
    assert module._part(parts, "stake", "gap") is None


def test_a_table_without_rows_is_just_its_header() -> None:
    table = pd.DataFrame({"a": [], "bb": []})
    assert module._render(table) == "a  bb"


def test_percentages_are_signed_when_asked_and_missing_values_are_dashes() -> None:
    assert module._percent(0.1234) == "12.3%"
    assert module._percent(0.1234, signed=True) == "+12.3%"
    assert module._percent(-0.1234, signed=True) == "-12.3%"
    assert module._percent(0.1234, digits=0) == "12%"
    assert module._percent(float("nan")) == "-"
    assert module._percent(float("inf"), signed=True) == "-"
    assert module._interval(-0.05, 0.1) == "[-5.0%, +10.0%]"
    assert module._interval(float("nan"), 0.1) == "-"
    assert module._interval(0.1, float("nan")) == "-"
    assert module._share(0.04) == "4%"
    assert module._share(-0.1) == "-10%"
    assert module._share(0.025) == "2.5%"


def tokens(text: str, start: str) -> list[str]:
    """The words of the one line whose own words begin with those of ``start``."""
    head = start.split()
    lines = [
        line.split() for line in text.split("\n") if line.split()[: len(head)] == head
    ]
    assert len(lines) == 1, (start, lines)
    return lines[0]


def test_the_text_names_the_strategy_and_what_it_covers() -> None:
    text = format_autopsy(hand_result())
    assert text.split("\n")[0] == (
        "Autopsy of Bet365 at model weight 0.00: 4 bets, seasons 2023-2024, over 4 weeks"
    )
    other = format_autopsy(autopsy(hand_bets(), book="avg", weight=0.0, n_resamples=10))
    assert other.split("\n")[0].startswith(
        "Autopsy of market average at model weight 0.00: 1 bets"
    )


def test_the_text_counts_the_weeks_of_the_calendar_not_the_bets() -> None:
    first = format_autopsy(replace(hand_result(), weeks=7)).split("\n")[0]
    assert first == (
        "Autopsy of Bet365 at model weight 0.00: 4 bets, seasons 2023-2024, over 7 weeks"
    )


def test_a_book_without_a_name_is_called_by_its_code() -> None:
    bets = hand_bets()
    bets.loc[bets["bookmaker"] == "b365", "bookmaker"] = "zz"
    result = autopsy(bets, book="zz", weight=0.0, n_resamples=20)
    assert (
        format_autopsy(result)
        .split("\n")[0]
        .startswith("Autopsy of zz at model weight 0.00: 4 bets")
    )
    assert verdict(result)[0].startswith("zz at weight 0.00: CLV per unit staked")


def test_the_text_explains_what_the_parts_are() -> None:
    text = format_autopsy(hand_result())
    assert "Price gap: odds x the sharp book's early fair probability - 1" in text
    assert "Market drift: odds x the change in that probability by the close" in text
    assert "'weighted by stake' is the backtest's CLV" in text
    assert "weighted by  part" in text


def test_the_parts_table_shows_every_part_with_its_interval() -> None:
    text = format_autopsy(hand_result())
    gap = tokens(text, "stake price gap")
    assert gap[:4] == ["stake", "price", "gap", "+21.5%"]
    assert gap[4].startswith("[+") and gap[-1].endswith("]")
    assert tokens(text, "stake market drift")[:4] == [
        "stake",
        "market",
        "drift",
        "+2.5%",
    ]
    assert tokens(text, "stake CLV")[:3] == ["stake", "CLV", "+24.0%"]
    # 21.25% is a tie between two roundings
    assert tokens(text, "bet price gap")[3] in ("+21.2%", "+21.3%")
    assert tokens(text, "bet market drift")[3:5] == ["+3.7%", "[-2.4%,"]
    assert tokens(text, "bet CLV")[:3] == ["bet", "CLV", "+25.0%"]


def test_the_comparison_with_similar_quotes_is_explained_only_when_made() -> None:
    plain = format_autopsy(hand_result())
    assert "Similar quotes:" not in plain
    assert "similar quotes drift by" not in plain
    compared = format_autopsy(hand_result(population=hand_population()))
    assert (
        "Similar quotes: for each bet, the other quotes of the same market" in compared
    )
    assert "within 2 points of the bet's, from 10 quotes of the sharp book" in compared
    assert "Their average drift is taken as known." in compared
    assert "of it, what similar quotes drift by" in compared
    assert "of it, specific to these bets" in compared
    typical = [
        line for line in compared.split("\n") if "what similar quotes drift by" in line
    ]
    assert len(typical) == 2 and all(
        line.startswith(("stake", "bet")) for line in typical
    )
    assert all(line.endswith("-") for line in typical)  # no interval of its own


def test_the_concentration_table_has_a_row_per_trim() -> None:
    text = format_autopsy(hand_result())
    assert "Do a few bets carry it? Per bet, with the best bets taken away" in text
    assert "(chosen after the fact, so no interval)." in text
    assert (
        "best bets removed  bets left  mean CLV  median CLV  share of all CLV in them"
        in text
    )
    assert tokens(text, "0 4") == ["0", "4", "+25.0%", "+30.0%", "-"]
    assert tokens(text, "1 3") == ["1", "3", "+20.0%", "+20.0%", "40%"]
    assert tokens(text, "3 1") == ["3", "1", "+0.0%", "+0.0%", "100%"]


def test_each_breakdown_has_its_own_title_and_groups() -> None:
    text = format_autopsy(hand_result(matches=hand_names()))
    for title in (
        "By season: means per bet (no intervals, the groups are small)",
        "By league: means per bet (no intervals, the groups are small)",
        "By market: means per bet (no intervals, the groups are small)",
        "By price range: means per bet (no intervals, the groups are small)",
    ):
        assert title in text
    assert tokens(text, "2023 3")[:5] == ["2023", "3", "+21.7%", "-1.7%", "+20.0%"]
    assert tokens(text, "Premier League") == [
        "Premier", "League", "2", "+20.0%", "+20.0%", "+40.0%", "50.0%", "26.2%",
    ]  # fmt: skip
    assert tokens(text, "OU:2.5")[:2] == ["OU:2.5", "1"]
    assert tokens(text, "3.0-4.0")[:2] == ["3.0-4.0", "1"]
    header = ["bets", "price", "gap", "drift", "CLV", "won", "expected"]
    heads = [line.split() for line in text.split("\n")]
    for label in (["season"], ["league"], ["market"], ["price", "range"]):
        assert [*label, *header] in heads


def test_a_breakdown_with_nothing_to_show_is_left_out() -> None:
    text = format_autopsy(hand_result())
    assert "By league" not in text
    assert "By season" in text


def test_the_gap_table_warns_about_gaps_that_are_too_good() -> None:
    text = format_autopsy(hand_result())
    assert (
        "A gap above 50% against a sharp book is more often a data or timing error"
        in text
    )
    assert tokens(text, "statistic price gap") == ["statistic", "price", "gap"]
    assert tokens(text, "highest") == ["highest", "+25.0%"]
    assert tokens(text, "lowest") == ["lowest", "+20.0%"]
    assert tokens(text, "median") == ["median", "+20.0%"]
    assert any(line.startswith("lowest") for line in text.split("\n"))  # flush left


def test_the_best_bets_are_listed_with_their_teams_and_results() -> None:
    text = format_autopsy(hand_result(matches=hand_names(), top=3))
    assert (
        "The 3 bets with the highest CLV (p early, p close: the sharp book's fair probabilities)"
        in text
    )
    line = tokens(text, "2023 2023-09-02")
    assert line[:5] == ["2023", "2023-09-02", "Premier", "League", "Arsenal"]
    assert line[-1] == "lost"
    assert line.count("v") == 1
    assert "p early  p close" in text
    last = tokens(text, "2024 2024-09-14")
    assert last[-1] == "won"
    assert last.count("v") == 1
    assert tokens(text, "2023 2023-09-09")[-1] == "lost"
    assert "Real v Sevilla" not in text  # the fourth bet is not among the best three


def test_without_leagues_the_best_bets_have_no_league_column() -> None:
    text = format_autopsy(hand_result())
    header = [
        line.split()
        for line in text.split("\n")
        if line.split()[:2] == ["season", "date"]
    ]
    assert header == [
        [
            "season",
            "date",
            "match",
            "market",
            "selection",
            "odds",
            "p",
            "early",
            "p",
            "close",
        ]
        + ["price", "gap", "CLV", "result"]
    ]
    assert tokens(text, "2023 2023-09-02")[2] == "m1"


def test_the_price_table_is_shown_when_there_are_prices(noise: Run) -> None:
    text = format_autopsy(noise.result)
    assert (
        "Every Bet365 price the engine could have bet, not only the bets "
        "(odds 1.30 to 8.00, the sharp book's probability at least 5%)"
    ) in text
    assert "by how far it sat above the sharp book's early price." in text
    assert "The last range is where the engine bets at weight 0." in text
    header = [line for line in text.split("\n") if line.startswith("price gap range")]
    assert len(header) == 1
    for column in ("prices", "price gap", "drift 95% interval", "CLV 95% interval"):
        assert column in header[0]
    last = tokens(text, "4% or more")
    assert last[:3] == ["4%", "or", "more"]
    assert int(last[3]) == noise.result.response["quotes"].iloc[-1]


def test_the_price_table_is_absent_without_prices() -> None:
    text = format_autopsy(hand_result(population=hand_population()))
    assert "Every Bet365 price the engine could have bet" not in text


def test_the_text_follows_the_engines_limits(noise: Run) -> None:
    config = ValueConfig(min_odds=1.5, max_odds=6.0, min_probability=0.1)
    text = format_autopsy(replace(noise.result, config=config))
    assert "(odds 1.50 to 6.00, the sharp book's probability at least 10%)" in text


def test_the_sections_come_in_the_order_of_the_questions(noise: Run) -> None:
    text = format_autopsy(noise.result)
    marks = [
        "What the closing-line value is made of.",
        "Similar quotes:",
        "Do a few bets carry it?",
        "By season:",
        "By league:",
        "By market:",
        "By price range:",
        "How large the price gaps are.",
        "Every Bet365 price the engine could have bet",
        "bets with the highest CLV",
    ]
    positions = [text.index(mark) for mark in marks]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #
def with_parts(
    entries: dict[str, tuple[float, float, float]], **changes: object
) -> Autopsy:
    return replace(hand_result(), parts=parts_of(entries), **changes)  # type: ignore[arg-type]


BASE_PARTS = {
    "gap": (0.08, 0.07, 0.09),
    "drift": (0.02, -0.01, 0.05),
    "clv": (0.10, 0.05, 0.15),
}


def test_the_first_line_gives_the_clv_and_what_it_is_made_of() -> None:
    first = verdict(with_parts(BASE_PARTS))[0]
    assert first == (
        "Bet365 at weight 0.00: CLV per unit staked is +10.0% [+5.0%, +15.0%], "
        "a price gap of +8.0% plus a market drift of +2.0% [-1.0%, +5.0%]"
    )


def test_the_real_strategy_is_described_by_its_own_numbers() -> None:
    lines = verdict(hand_result())
    assert lines[0].startswith("Bet365 at weight 0.00: CLV per unit staked is +24.0% [")
    assert " a price gap of +21.5% plus a market drift of +2.5% [" in lines[0]


NO_MOVE = (
    "the sharp book did not move for or against the bettor by more than the data can "
    "tell from zero, so the CLV is mostly the price gap itself"
)


@pytest.mark.parametrize(
    ("drift", "phrase"),
    [
        ((0.02, 0.01, 0.03), "the sharp book kept moving in the bettor's favour after the price was taken"),
        ((-0.02, -0.03, -0.01), "the sharp book moved against the bettor after the price was taken, so the price gap overstates what the close confirms"),
        ((0.0, -0.01, 0.01), NO_MOVE),
        ((0.02, 0.0, 0.03), NO_MOVE),  # an interval that only touches zero does not exclude it
        ((-0.02, -0.03, 0.0), NO_MOVE),
    ],
)  # fmt: skip
def test_the_drift_is_read_against_zero_by_its_interval(
    drift: tuple[float, float, float], phrase: str
) -> None:
    lines = verdict(with_parts({**BASE_PARTS, "drift": drift}))
    assert lines[1] == f"Drift: {phrase}"


def compared_parts(
    typical: float, specific: tuple[float, float, float]
) -> dict[str, tuple[float, float, float]]:
    return {
        **BASE_PARTS,
        "typical": (typical, float("nan"), float("nan")),
        "specific": specific,
    }


def test_without_a_comparison_the_verdict_says_what_cannot_be_told() -> None:
    lines = verdict(with_parts(BASE_PARTS))
    assert lines[2] == (
        "The drift was not compared with similar quotes (no match table, or one that "
        "does not match this backtest's settings), so a pattern of the whole market, "
        "such as long shots losing probability towards the close, cannot be told from "
        "a property of the bets"
    )


@pytest.mark.parametrize(
    "entries",
    [
        {**BASE_PARTS, "typical": (0.01, float("nan"), float("nan"))},
        {**BASE_PARTS, "specific": (0.01, -0.01, 0.03)},
    ],
)
def test_a_comparison_needs_both_its_parts(
    entries: dict[str, tuple[float, float, float]],
) -> None:
    line = verdict(with_parts(entries))[2]
    assert line.startswith("The drift was not compared with similar quotes")


NOISE_LIKE = (
    "nothing the data can tell apart from the pattern of similar quotes: the price "
    "gap behaves like noise the sharp book ignored, which is how a price-shopping "
    "edge looks"
)


@pytest.mark.parametrize(
    "specific",
    [
        (0.02, 0.0, 0.04),  # an interval that only touches zero does not exclude it
        (-0.02, -0.04, 0.0),
        (0.03, -0.01, 0.05),
        (-0.03, -0.05, 0.01),
    ],
)
def test_a_specific_drift_whose_interval_holds_zero_is_not_told_from_the_pattern(
    specific: tuple[float, float, float],
) -> None:
    line = verdict(with_parts(compared_parts(0.0, specific)))[2]
    assert line.endswith(f"to these bets: {NOISE_LIKE}")


def test_a_specific_drift_above_zero_is_a_slow_price_or_a_different_moment() -> None:
    line = verdict(with_parts(compared_parts(-0.01, (0.03, 0.01, 0.05))))[2]
    assert line == (
        "Quotes priced like these drifted -1.0% on average, so +3.0% [+1.0%, +5.0%] "
        "of the drift is specific to these bets: more than the pattern of similar "
        "quotes explains: the sharp book kept moving in the bettor's favour. That is "
        "how a slow Bet365 price looks, or two prices recorded at different moments; "
        "the data cannot tell which"
    )


def test_a_specific_drift_below_zero_means_part_of_the_gap_was_information() -> None:
    line = verdict(with_parts(compared_parts(0.01, (-0.03, -0.05, -0.01))))[2]
    assert line == (
        "Quotes priced like these drifted +1.0% on average, so -3.0% [-5.0%, -1.0%] "
        "of the drift is specific to these bets: less than the pattern of similar "
        "quotes: the sharp book moved towards Bet365's view, so part of the price "
        "gap was information rather than mispricing"
    )


def test_a_specific_drift_that_includes_zero_is_a_gap_the_sharp_book_ignored() -> None:
    line = verdict(with_parts(compared_parts(0.0, (0.0, -0.02, 0.02))))[2]
    assert line == (
        "Quotes priced like these drifted +0.0% on average, so +0.0% [-2.0%, +2.0%] "
        "of the drift is specific to these bets: nothing the data can tell apart "
        "from the pattern of similar quotes: the price gap behaves like noise the "
        "sharp book ignored, which is how a price-shopping edge looks"
    )


def response_of(
    drift: tuple[float, float, float],
    clv: tuple[float, float, float],
    quotes: int = 120,
) -> pd.DataFrame:
    row = {
        "gap_range": "4% or more",
        "region": True,
        "quotes": quotes,
        "gap": 0.07,
        "drift": drift[0],
        "drift_low": drift[1],
        "drift_high": drift[2],
        "clv": clv[0],
        "clv_low": clv[1],
        "clv_high": clv[2],
    }
    below = {**row, "gap_range": "2% to 4%", "region": False}
    return pd.DataFrame([below, row], columns=RESPONSE_COLUMNS)


@pytest.mark.parametrize(
    ("drift", "phrase"),
    [
        ((0.03, 0.01, 0.05), "the sharp book moved in the bettor's favour"),
        ((-0.03, -0.05, -0.01), "the sharp book moved against the bettor"),
        ((0.0, -0.02, 0.02), "the sharp book's move cannot be told from zero"),
    ],
)
def test_the_prices_the_engine_could_have_bet_are_read_the_same_way(
    drift: tuple[float, float, float], phrase: str
) -> None:
    result = with_parts(BASE_PARTS, response=response_of(drift, (0.07, 0.05, 0.09)))
    line = next(line for line in verdict(result) if line.startswith("Across all"))
    assert line == (
        "Across all 120 Bet365 prices the engine could have bet that sat at least 4% "
        "above the sharp book's early price (not only the 4 bets), the mean CLV per "
        "price is +7.0% [+5.0%, +9.0%] and the mean drift "
        f"{drift[0] * 100:+.1f}% [{drift[1] * 100:+.1f}%, {drift[2] * 100:+.1f}%]: {phrase}"
    )


@pytest.mark.parametrize(
    "drift",
    [
        (0.02, 0.0, 0.04),  # an interval that only touches zero does not exclude it
        (-0.02, -0.04, 0.0),
        (0.03, -0.01, 0.05),
        (-0.03, -0.05, 0.01),
    ],
)
def test_a_price_drift_whose_interval_holds_zero_cannot_be_told_from_it(
    drift: tuple[float, float, float],
) -> None:
    result = with_parts(BASE_PARTS, response=response_of(drift, (0.07, 0.05, 0.09)))
    line = next(line for line in verdict(result) if line.startswith("Across all"))
    assert line.endswith(": the sharp book's move cannot be told from zero")


def test_the_price_line_names_the_minimum_edge_it_was_run_with() -> None:
    result = with_parts(
        BASE_PARTS,
        response=response_of((0.0, -0.02, 0.02), (0.07, 0.05, 0.09)),
        config=ValueConfig(min_edge=0.025),
    )
    line = next(line for line in verdict(result) if line.startswith("Across all"))
    assert "that sat at least 2.5% above the sharp book's early price" in line


def test_the_price_line_is_left_out_without_a_region() -> None:
    table = response_of((0.0, -0.02, 0.02), (0.07, 0.05, 0.09))
    table["region"] = False
    result = with_parts(BASE_PARTS, response=table)
    assert not any(line.startswith("Across all") for line in verdict(result))
    assert not any(
        line.startswith("Across all") for line in verdict(with_parts(BASE_PARTS))
    )


def lines_of(result: Autopsy) -> list[str]:
    return verdict(result)


def test_the_best_bets_taken_away_are_reported_when_there_are_enough_bets() -> None:
    clv = np.linspace(0.30, 0.0, 12)
    bets = hand_result().bets.iloc[[0] * 12].reset_index(drop=True).assign(clv=clv)
    table = concentration(bets)
    result = replace(with_parts(BASE_PARTS), concentration=table)
    line = next(line for line in verdict(result) if line.startswith("Without its 5"))
    # 12 bets from +30% to 0: the best five hold 0.30 + ... of 1.65
    top_five = clv[:5].sum() / clv.sum()
    rest = clv[5:].mean()
    assert line == (
        f"Without its 5 best bets the mean CLV per bet is {rest * 100:+.1f}% "
        f"(all bets: {clv.mean() * 100:+.1f}%); those 5 bets hold {top_five * 100:.0f}% "
        "of all CLV: it stays positive"
    )


def test_a_result_that_needs_its_best_bets_is_called_that() -> None:
    table = pd.DataFrame(
        {
            "removed": [0, 5],
            "left": [12, 7],
            "mean_clv": [0.05, -0.01],
            "median_clv": [0.02, -0.02],
            "share": [float("nan"), 1.4],
        }
    )
    line = next(
        line
        for line in verdict(replace(with_parts(BASE_PARTS), concentration=table))
        if line.startswith("Without its 5")
    )
    assert line == (
        "Without its 5 best bets the mean CLV per bet is -1.0% (all bets: +5.0%); "
        "those 5 bets hold 140% of all CLV: it does not survive"
    )


def test_a_mean_of_exactly_zero_without_the_best_bets_does_not_survive() -> None:
    table = pd.DataFrame(
        {
            "removed": [0, 5],
            "left": [12, 7],
            "mean_clv": [0.05, 0.0],
            "median_clv": [0.02, 0.0],
            "share": [float("nan"), 1.0],
        }
    )
    line = next(
        line
        for line in verdict(replace(with_parts(BASE_PARTS), concentration=table))
        if line.startswith("Without its 5")
    )
    assert line.endswith("of all CLV: it does not survive")


def test_a_share_that_is_not_defined_is_not_reported() -> None:
    table = pd.DataFrame(
        {
            "removed": [0, 5],
            "left": [12, 7],
            "mean_clv": [0.05, 0.03],
            "median_clv": [0.02, 0.02],
            "share": [float("nan"), float("nan")],
        }
    )
    line = next(
        line
        for line in verdict(replace(with_parts(BASE_PARTS), concentration=table))
        if line.startswith("Without its 5")
    )
    assert line == (
        "Without its 5 best bets the mean CLV per bet is +3.0% (all bets: +5.0%): "
        "it stays positive"
    )


def test_too_few_bets_to_take_the_best_away() -> None:
    lines = verdict(hand_result())
    assert "Only 4 bets: too few to take the best 5 away" in lines


def test_the_seasons_are_counted_and_the_weakest_named() -> None:
    lines = verdict(hand_result())
    assert (
        "CLV per bet is positive in 2 of 2 seasons; the weakest is 2023 with +20.0% on 3 bets"
        in lines
    )
    table = pd.DataFrame(
        {
            "season": [2023, 2024, 2025],
            "bets": [30, 25, 20],
            "gap": [0.07] * 3,
            "drift": [0.0] * 3,
            "clv": [0.08, -0.02, 0.0],
            "hit": [0.3] * 3,
            "expected": [0.3] * 3,
        }
    )
    line = next(
        line
        for line in verdict(replace(with_parts(BASE_PARTS), seasons=table))
        if line.startswith("CLV per bet is positive")
    )
    assert line == (
        "CLV per bet is positive in 1 of 3 seasons; the weakest is 2024 with -2.0% on 25 bets"
    )


def test_the_weakest_season_of_a_tie_is_the_earlier() -> None:
    table = pd.DataFrame(
        {
            "season": [2023, 2024],
            "bets": [30, 25],
            "gap": [0.07] * 2,
            "drift": [0.0] * 2,
            "clv": [-0.02, -0.02],
            "hit": [0.3] * 2,
            "expected": [0.3] * 2,
        }
    )
    line = next(
        line
        for line in verdict(replace(with_parts(BASE_PARTS), seasons=table))
        if line.startswith("CLV per bet is positive")
    )
    assert "the weakest is 2023" in line


def test_a_gap_too_good_to_be_true_is_flagged() -> None:
    result = hand_result()
    bets = result.bets.assign(gap=[0.2, 0.55, 0.9, 0.2])
    lines = verdict(replace(result, bets=bets))
    assert (
        "2 of the bets have a price gap above 50% (the largest is +90.0%): against a "
        "sharp book that is more often a data or timing error than an opportunity, "
        "so check them in the list of best bets"
    ) in lines
    assert not any(line.startswith("No bet has a price gap above") for line in lines)


def test_a_gap_of_exactly_fifty_percent_is_not_flagged() -> None:
    result = hand_result()
    lines = verdict(replace(result, bets=result.bets.assign(gap=[0.5, 0.2, 0.2, 0.2])))
    assert "No bet has a price gap above 50%; the largest is +50.0%" in lines


def test_no_implausible_gap() -> None:
    assert "No bet has a price gap above 50%; the largest is +25.0%" in verdict(
        hand_result()
    )


def test_the_wins_are_compared_with_what_the_closing_prices_expected() -> None:
    lines = verdict(hand_result())
    # 2 wins against 0.35 + 0.4 + 0.4 + 0.175 = 1.325; spread = sqrt(0.851875)
    assert (
        "The bets won 2 times against 1.3 expected from the closing prices (z = +0.7)"
        in lines
    )


def test_the_z_score_counts_standard_deviations_of_the_wins() -> None:
    result = hand_result()
    bets = (
        result.bets.iloc[[0] * 100]
        .reset_index(drop=True)
        .assign(p_close=0.5, won=[True] * 60 + [False] * 40)
    )
    line = next(
        line
        for line in verdict(replace(result, bets=bets))
        if line.startswith("The bets won")
    )
    # 60 wins against 50, and a standard deviation of sqrt(100 * 0.25) = 5
    assert line == (
        "The bets won 60 times against 50.0 expected from the closing prices (z = +2.0)"
    )
    fewer = bets.assign(won=[True] * 30 + [False] * 70)
    line = next(
        line
        for line in verdict(replace(result, bets=fewer))
        if line.startswith("The bets won")
    )
    assert line.endswith("(z = -4.0)")


def test_with_certain_outcomes_there_is_no_z_score() -> None:
    result = hand_result()
    certain = result.bets.assign(p_close=[1.0, 0.0, 1.0, 0.0])
    assert not any(
        line.startswith("The bets won")
        for line in verdict(replace(result, bets=certain))
    )


def test_a_small_sample_is_said_to_be_small() -> None:
    lines = verdict(hand_result())
    assert (
        "4 bets is a small sample: the intervals allow for that, but the splits by "
        "season, league and price range rest on a handful of bets each"
    ) in lines
    big = hand_result()
    many = big.bets.iloc[[0] * 100].reset_index(drop=True)
    assert not any(
        "is a small sample" in line for line in verdict(replace(big, bets=many))
    )
    few = big.bets.iloc[[0] * 99].reset_index(drop=True)
    assert any(
        "99 bets is a small sample" in line for line in verdict(replace(big, bets=few))
    )


def test_a_result_without_the_parts_has_nothing_to_say() -> None:
    assert (
        verdict(replace(hand_result(), parts=parts_of({"gap": (0.1, 0.0, 0.2)}))) == []
    )
    assert verdict(replace(hand_result(), parts=parts_of({}))) == []


def test_the_verdict_never_recommends_a_bet() -> None:
    for run in (hand_result(), hand_result(population=hand_population())):
        text = " ".join(verdict(run)).lower()
        for word in ("should bet", "recommend", "you should", "place a bet"):
            assert word not in text


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def test_every_table_is_saved_as_a_csv(tmp_path: Path) -> None:
    result = hand_result(matches=hand_names(), population=hand_population())
    directory = tmp_path / "deep" / "out"
    written = save_autopsy(result, directory)
    names = [path.name for path in written]
    assert names == [
        "bets.csv",
        "parts.csv",
        "concentration.csv",
        "seasons.csv",
        "leagues.csv",
        "markets.csv",
        "odds_ranges.csv",
        "gaps.csv",
        "response.csv",
        "best.csv",
    ]
    assert all(path.parent == directory and path.exists() for path in written)
    bets = pd.read_csv(directory / "bets.csv")
    assert bets["match_id"].tolist() == ["m1", "m2", "m3", "m4"]
    assert {"gap", "drift", "typical", "specific", "odds_range", "league"} <= set(
        bets.columns
    )
    parts = pd.read_csv(directory / "parts.csv")
    assert list(parts.columns) == PART_COLUMNS
    assert len(parts) == 10
    assert pd.read_csv(directory / "response.csv").empty
    assert list(pd.read_csv(directory / "response.csv").columns) == RESPONSE_COLUMNS
    assert pd.read_csv(directory / "best.csv")["match"].iloc[0] == "Arsenal v Chelsea"
    assert pd.read_csv(directory / "odds_ranges.csv")["odds_range"].tolist() == [
        "2.0-3.0",
        "3.0-4.0",
        "6.0-8.0",
    ]
    tables = {
        "bets": result.bets,
        "parts": result.parts,
        "concentration": result.concentration,
        "seasons": result.seasons,
        "leagues": result.leagues,
        "markets": result.markets,
        "odds_ranges": result.odds_ranges,
        "gaps": result.gaps,
        "response": result.response,
        "best": result.best,
    }
    for written_path in written:
        saved = pd.read_csv(written_path)
        table = tables[written_path.stem]
        assert list(saved.columns) == list(table.columns), written_path.name
        assert len(saved) == len(table), written_path.name


def test_saving_twice_overwrites(tmp_path: Path) -> None:
    result = hand_result()
    save_autopsy(result, tmp_path)
    again = save_autopsy(result, tmp_path)
    assert len(again) == 10
    assert len(pd.read_csv(tmp_path / "bets.csv")) == 4


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def write_run(tmp_path: Path, run: Run, *, matches: bool = True) -> tuple[Path, Path]:
    folder = tmp_path / "backtest"
    folder.mkdir()
    run.backtest.bets.to_parquet(folder / "bets.parquet", index=False)
    table = tmp_path / "enriched.parquet"
    if matches:
        run.frame.to_parquet(table, index=False)
    return folder, table


def command(folder: Path, table: Path, *extra: str) -> list[str]:
    return [
        "--backtest-dir",
        str(folder),
        "--enriched",
        str(table),
        "--resamples",
        "50",
        *extra,
    ]


def test_the_command_defaults() -> None:
    args = module._parse_args([])
    assert args.backtest_dir == BACKTEST_DIR
    assert args.enriched == module.ENRICHED_OUTPUT
    assert (args.book, args.weight) == ("b365", 0.0)
    assert (args.reference, args.demargin) == ("ps", "proportional")
    assert args.min_edge == ValueConfig().min_edge == 0.04
    assert (args.top, args.resamples, args.seed) == (10, 2000, 0)
    assert args.output_dir is None
    assert AUTOPSY_FOLDER == "autopsy"


def test_the_command_takes_the_options_of_the_backtest_that_made_the_bets() -> None:
    args = module._parse_args(
        ["--book", "max", "--weight", "0.25", "--reference", "avg", "--demargin", "shin",
         "--min-edge", "0.07", "--top", "3", "--resamples", "9", "--seed", "4"]
    )  # fmt: skip
    assert (args.book, args.weight, args.reference, args.demargin) == (
        "max",
        0.25,
        "avg",
        "shin",
    )
    assert (args.min_edge, args.top, args.resamples, args.seed) == (0.07, 3, 9, 4)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--weight", "1.5"],
        ["--weight", "-0.1"],
        ["--weight", "abc"],
        ["--book", "nobody"],
        ["--reference", "max"],
        ["--demargin", "magic"],
        ["--resamples", "0"],
        ["--top", "0"],
    ],
)
def test_the_command_refuses_options_that_make_no_sense(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        module._parse_args(arguments)


def test_the_weight_includes_both_ends_and_names_what_is_wrong() -> None:
    assert module._weight("0") == 0.0
    assert module._weight("1") == 1.0
    assert module._weight("0.25") == 0.25
    with pytest.raises(argparse.ArgumentTypeError, match="^'abc' is not a number$"):
        module._weight("abc")
    with pytest.raises(
        argparse.ArgumentTypeError, match="^a weight must be between 0 and 1, got 1.01$"
    ):
        module._weight("1.01")


def test_one_resample_and_one_best_bet_are_the_least_that_makes_sense() -> None:
    args = module._parse_args(["--resamples", "1", "--top", "1"])
    assert (args.resamples, args.top) == (1, 1)


def test_the_command_says_why(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        module._parse_args(["--weight", "1.5"])
    assert "a weight must be between 0 and 1, got 1.5" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        module._parse_args(["--top", "0"])
    assert "--top must be at least 1" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        module._parse_args(["--resamples", "0"])
    assert "--resamples must be at least 1" in capsys.readouterr().err


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_prints_the_autopsy_the_verdict_and_saves_the_tables(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise)
    with caplog.at_level(logging.INFO):
        assert main(command(folder, table, "--top", "3")) == 0
    assert "Autopsy of Bet365 at model weight 0.00" in caplog.text
    assert "Similar quotes: for each bet" in caplog.text
    assert "Every Bet365 price the engine could have bet" in caplog.text
    assert "The 3 bets with the highest CLV" in caplog.text
    assert "Bet365 at weight 0.00: CLV per unit staked is" in caplog.text
    messages = caplog.messages
    heading = messages.index("Verdict")
    assert any("The 3 bets with the highest CLV" in line for line in messages[:heading])
    assert messages[heading + 1].startswith(
        "Bet365 at weight 0.00: CLV per unit staked is"
    )
    folder_out = folder / "autopsy" / "b365_weight0"
    assert (folder_out / "bets.csv").exists()
    assert f"Wrote {folder_out / 'parts.csv'}" in caplog.text
    assert len(list(folder_out.glob("*.csv"))) == 10
    assert len(pd.read_csv(folder_out / "best.csv")) == 3


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_writes_where_it_is_told_to(noise: Run, tmp_path: Path) -> None:
    folder, table = write_run(tmp_path, noise)
    target = tmp_path / "elsewhere"
    assert main(command(folder, table, "--output-dir", str(target))) == 0
    assert (target / "parts.csv").exists()
    assert not (folder / "autopsy").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_names_another_strategys_folder_by_its_weight(
    noise: Run, tmp_path: Path
) -> None:
    folder, table = write_run(tmp_path, noise)
    bets = pd.read_parquet(folder / "bets.parquet")
    pd.concat([bets, bets.assign(weight=0.25)], ignore_index=True).to_parquet(
        folder / "bets.parquet", index=False
    )
    assert main(command(folder, table, "--weight", "0.25")) == 0
    assert (folder / "autopsy" / "b365_weight0.25" / "bets.csv").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_reproduces_the_backtests_interval(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise)
    arguments = [
        "--backtest-dir", str(folder), "--enriched", str(table), "--resamples", "300",
    ]  # fmt: skip
    with caplog.at_level(logging.INFO):
        assert main(arguments) == 0
    summary = noise.backtest.summary
    row = summary[(summary["bookmaker"] == "b365") & (summary["weight"] == 0.0)].iloc[0]
    expected = (
        f"CLV per unit staked is {row['clv'] * 100:+.1f}% "
        f"[{row['clv_low'] * 100:+.1f}%, {row['clv_high'] * 100:+.1f}%]"
    )
    assert expected in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_still_works_without_a_match_table(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise, matches=False)
    with caplog.at_level(logging.INFO):
        assert main(command(folder, table)) == 0
    assert f"No match table at {table}" in caplog.text
    assert "the comparison with similar quotes is left out" in caplog.text
    assert "Autopsy of Bet365 at model weight 0.00" in caplog.text
    assert "Similar quotes:" not in caplog.text
    assert "Every Bet365 price" not in caplog.text
    assert "The drift was not compared with similar quotes" in caplog.text
    assert "Home" not in caplog.text  # matches are shown by id


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_warns_when_the_backtest_used_another_margin_removal(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise)
    with caplog.at_level(logging.INFO):
        assert main(command(folder, table, "--demargin", "power")) == 0
    assert "does not reproduce the bets' probabilities" in caplog.text
    assert "Similar quotes:" not in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_passes_every_option_on(
    noise: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, table = write_run(tmp_path, noise)
    seen: dict[str, object] = {}

    def fake(bets: pd.DataFrame, **options: object) -> Autopsy:
        seen.update(options)
        seen["bets"] = bets
        raise ValueError("stop here")

    monkeypatch.setattr(module, "autopsy", fake)
    arguments = command(folder, table, "--book", "b365", "--weight", "0", "--top", "4")
    arguments += ["--seed", "5", "--min-edge", "0.07", "--resamples", "77"]
    assert main(arguments) == 1
    config = seen.pop("config")
    assert isinstance(config, ValueConfig) and config.min_edge == 0.07
    matches = seen.pop("matches")
    population = seen.pop("population")
    assert isinstance(matches, pd.DataFrame) and len(matches) == len(noise.frame)
    assert isinstance(population, pd.DataFrame) and not population.empty
    assert set(population["season"]) == {2023, 2024}
    assert isinstance(seen.pop("bets"), pd.DataFrame)
    assert seen == {
        "book": "b365",
        "weight": 0.0,
        "top": 4,
        "n_resamples": 77,
        "seed": 5,
    }


@pytest.mark.usefixtures("isolated_logging")
def test_the_population_starts_with_the_backtests_first_season(
    noise: Run, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder, table = write_run(tmp_path, noise)
    seen: dict[str, object] = {}
    real = module.control_population

    def spy(frame: pd.DataFrame, **options: object) -> pd.DataFrame:
        seen.update(options)
        return real(frame, **options)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "control_population", spy)
    assert main(command(folder, table, "--reference", "ps", "--demargin", "shin")) == 0
    assert seen == {"book": "ps", "demargin": "shin", "first_season": 2023}


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_names_the_step_to_run_when_there_are_no_bets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        assert main(["--backtest-dir", str(tmp_path)]) == 1
    assert (
        f"Not found: {tmp_path / 'bets.parquet'}; run python -m src.backtest first"
        in caplog.text
    )


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_strategy_without_bets(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise)
    with caplog.at_level(logging.ERROR):
        assert main(command(folder, table, "--book", "max")) == 1
    assert (
        "Cannot take the strategy apart: no bets for max at weight 0; the file has "
        "b365 at weight 0"
    ) in caplog.text


@pytest.mark.usefixtures("isolated_logging")
@pytest.mark.parametrize("column", ["p_close", "season"])
def test_the_command_fails_cleanly_on_bets_it_cannot_read(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture, column: str
) -> None:
    folder, table = write_run(tmp_path, noise)
    bets = pd.read_parquet(folder / "bets.parquet").drop(columns=column)
    bets.to_parquet(folder / "bets.parquet", index=False)
    with caplog.at_level(logging.ERROR):
        assert main(command(folder, table)) == 1
    assert (
        f"Cannot take the strategy apart: the bets table is missing columns: ['{column}']"
        in caplog.text
    )


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_match_table_it_cannot_use(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise)
    pd.read_parquet(table).drop(columns="date").to_parquet(table, index=False)
    with caplog.at_level(logging.ERROR):
        assert main(command(folder, table)) == 1
    assert (
        "Cannot take the strategy apart: the match table is missing columns: ['date']"
        in caplog.text
    )


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_key_that_is_missing(
    noise: Run,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    folder, table = write_run(tmp_path, noise)

    def fake(bets: pd.DataFrame, **options: object) -> Autopsy:
        raise KeyError("won")

    monkeypatch.setattr(module, "autopsy", fake)
    with caplog.at_level(logging.ERROR):
        assert main(command(folder, table)) == 1
    assert "Cannot take the strategy apart: 'won'" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_minimum_edge_that_makes_no_sense(
    noise: Run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    folder, table = write_run(tmp_path, noise)
    with caplog.at_level(logging.ERROR):
        assert main(command(folder, table, "--min-edge", "-0.1")) == 1
    assert "Cannot take the strategy apart: min_edge cannot be negative" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_does_not_write_when_the_autopsy_fails(
    noise: Run, tmp_path: Path
) -> None:
    folder, table = write_run(tmp_path, noise)
    assert main(command(folder, table, "--book", "max")) == 1
    assert not (folder / "autopsy").exists()
