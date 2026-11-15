"""Tests for the margin-removal audit."""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import logging_config
from src import margin_audit as module
from src.demargin import METHODS, remove_margin
from src.margin_audit import (
    BAND_LABELS,
    MARGIN_AUDIT_DIR,
    PROBABILITY_COLUMNS,
    RELIABILITY_COLUMNS,
    SCORE_COLUMNS,
    SELECTION_COLUMNS,
    MarginAudit,
    audit_margin,
    decide,
    format_margin_audit,
    main,
    reliability_table,
    save_margin_audit,
    score_methods,
    selection_table,
    verdict,
)
from src.odds_history import MARKETS

SELECTIONS = {market: [name for name, _ in MARKETS[market]] for market in MARKETS}


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
def hand_frame() -> pd.DataFrame:
    """Four matches with Pinnacle prices, early and at the close.

    ``a`` (Friday 2 September 2022) ended 2-1 and ``b`` (the next day, the same
    week) 0-0. ``c`` (Friday 1 September 2023) ended 1-3 and ``d`` (Saturday the
    9th, the following week) 1-0.
    """
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
            "ps_over25": [1.80, 2.20, 1.60, 1.95],
            "ps_under25": [2.10, 1.70, 2.30, 1.95],
            "psc_h": [1.90, 2.70, 3.30, 1.45],
            "psc_d": [3.60, 3.00, 3.70, 4.40],
            "psc_a": [4.60, 2.95, 2.15, 7.50],
            "psc_over25": [1.75, 2.30, 1.55, 2.00],
            "psc_under25": [2.15, 1.65, 2.40, 1.90],
        }
    )


def simulated_world(
    rule: str,
    *,
    seasons: Sequence[int] = tuple(range(2019, 2026)),
    per_season: int = 800,
    seed: int = 0,
    k: float = 1.5,
    margin: float = 0.15,
    book: str = "ps",
) -> pd.DataFrame:
    """Matches whose results come from known probabilities, priced by a known rule.

    ``power``: the book's implied probability is ``p ** (1 / k)``, which is exactly
    what the power method undoes. ``proportional``: it is ``p * (1 + margin)``,
    which is exactly what proportional scaling undoes. Early and closing prices are
    the same, so every season is priced alike.
    """
    rng = np.random.default_rng(seed)
    n = per_season * len(seasons)
    home_draw_away = np.clip(rng.dirichlet([4.0, 3.0, 3.0], n), 0.02, 0.9)
    home_draw_away /= home_draw_away.sum(axis=1, keepdims=True)
    over_under = rng.dirichlet([5.0, 5.0], n)
    result = (rng.random(n)[:, None] > home_draw_away.cumsum(axis=1)).sum(axis=1)
    over = rng.random(n) < over_under[:, 0]
    draw_goals = np.where(over, 2, 0)
    home = np.where(
        result == 0, np.where(over, 3, 1), np.where(result == 1, draw_goals, 0)
    )
    away = np.where(
        result == 2, np.where(over, 3, 1), np.where(result == 1, draw_goals, 0)
    )
    season = np.repeat(list(seasons), per_season)
    days = np.concatenate([np.sort(rng.integers(0, 280, per_season)) for _ in seasons])
    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "season": season,
            "date": pd.to_datetime([f"{s}-08-10" for s in season])
            + pd.to_timedelta(days, unit="D"),
            "home_goals": home,
            "away_goals": away,
        }
    )
    prefixes = {"ps": ("ps", "psc"), "avg": ("avg", "avgc"), "b365": ("b365", "b365c")}
    for prefix in prefixes[book]:
        for market_id, probabilities in (
            ("1X2", home_draw_away),
            ("OU:2.5", over_under),
        ):
            implied = (
                probabilities ** (1.0 / k)
                if rule == "power"
                else probabilities * (1.0 + margin)
            )
            odds = 1.0 / np.minimum(implied, 0.995)
            for position, (_, suffix) in enumerate(MARKETS[market_id]):
                frame[f"{prefix}_{suffix}"] = odds[:, position]
    return frame


def quote_table(
    quotes: Sequence[tuple[str, int, str, str, str, dict[str, Sequence[float]]]],
) -> pd.DataFrame:
    """A selection table from hand-written quotes.

    A quote is ``(match, season, week, market, winner, {method: probabilities})``
    with the probabilities in the order of the market's selections.
    """
    rows = []
    for match_id, season, week, market_id, winner, probabilities in quotes:
        for position, selection in enumerate(SELECTIONS[market_id]):
            rows.append(
                {
                    "match_id": match_id,
                    "season": season,
                    "batch": pd.Timestamp(week),
                    "market_id": market_id,
                    "selection": selection,
                    "won": selection == winner,
                    **{
                        PROBABILITY_COLUMNS[method]: probabilities[method][position]
                        for method in METHODS
                    },
                }
            )
    return pd.DataFrame(rows, columns=SELECTION_COLUMNS)


def same(*probabilities: float) -> dict[str, Sequence[float]]:
    return {method: probabilities for method in METHODS}


def scores_of(
    cells: dict[str, dict[str, tuple[float, float, float]]], n: int = 1000
) -> pd.DataFrame:
    """Pooled rows of a scores table.

    ``cells`` maps ``"close_selection"``, ``"close_test"``, ``"early_selection"`` or
    ``"early_test"`` to ``{method: (gain, low, high)}``. Every cell also gets its
    baseline row.
    """
    rows = []
    for name, by_method in cells.items():
        snapshot, window = name.split("_")
        rows.append((snapshot, window, "proportional", 0.9, 0.0, np.nan, np.nan))
        for method, (gain, low, high) in by_method.items():
            rows.append((snapshot, window, method, 0.9 - gain, gain, low, high))
    return pd.DataFrame(
        [
            {
                "snapshot": snapshot,
                "window": window,
                "market_id": "all",
                "method": method,
                "n": n,
                "log_loss": loss,
                "gain": gain,
                "gain_low": low,
                "gain_high": high,
            }
            for snapshot, window, method, loss, gain, low, high in rows
        ],
        columns=SCORE_COLUMNS,
    )


CLEAR = (0.0009, 0.0004, 0.0014)  # the whole interval above zero
NEARLY = (0.0012, -0.0001, 0.0025)  # the largest gain, but the interval holds zero
FLAT = (0.0001, -0.0004, 0.0006)
WORSE = (-0.0006, -0.0011, -0.0001)  # the whole interval below zero


def audit_of(
    scores: pd.DataFrame,
    *,
    reliability: pd.DataFrame | None = None,
    chosen: str = "proportional",
    recommended: str = "proportional",
    book: str = "ps",
    first_season: int = 2023,
) -> MarginAudit:
    empty = pd.DataFrame(columns=RELIABILITY_COLUMNS)
    return MarginAudit(
        book=book,
        first_season=first_season,
        scores=scores,
        reliability=empty if reliability is None else reliability,
        chosen=chosen,
        recommended=recommended,
    )


def tokens(text: str, containing: str) -> list[str]:
    """The words of the one line of ``text`` that contains ``containing``."""
    lines = [line for line in text.splitlines() if containing in line]
    assert len(lines) == 1, f"{containing!r} is on {len(lines)} lines"
    return lines[0].split()


# --------------------------------------------------------------------------- #
# One row per selection
# --------------------------------------------------------------------------- #
def test_every_selection_of_every_complete_quote_is_a_row() -> None:
    table = selection_table(hand_frame(), "ps", "early")
    assert list(table.columns) == SELECTION_COLUMNS
    assert len(table) == 4 * (3 + 2)
    counts = table.groupby(["match_id", "market_id"]).size()
    assert len(counts) == 8
    assert counts.xs("1X2", level="market_id").eq(3).all()
    assert counts.xs("OU:2.5", level="market_id").eq(2).all()
    seasons = table.drop_duplicates("match_id").set_index("match_id")["season"]
    assert seasons.to_dict() == {"a": 2022, "b": 2022, "c": 2023, "d": 2023}
    assert table.index.equals(pd.RangeIndex(len(table)))


def test_exactly_one_selection_of_each_quote_won() -> None:
    table = selection_table(hand_frame(), "ps", "close")
    assert table.groupby(["match_id", "market_id"])["won"].sum().eq(1).all()
    winners = table[table["won"]].set_index(["match_id", "market_id"])["selection"]
    assert winners.to_dict() == {
        ("a", "1X2"): "home",
        ("a", "OU:2.5"): "over",
        ("b", "1X2"): "draw",
        ("b", "OU:2.5"): "under",
        ("c", "1X2"): "away",
        ("c", "OU:2.5"): "over",
        ("d", "1X2"): "home",
        ("d", "OU:2.5"): "under",
    }


@pytest.mark.parametrize("snapshot", ["early", "close"])
def test_each_method_gives_its_own_probability_from_the_snapshots_prices(
    snapshot: str,
) -> None:
    frame = hand_frame()
    table = selection_table(frame, "ps", snapshot)
    prefix = "ps" if snapshot == "early" else "psc"
    for method in METHODS:
        column = PROBABILITY_COLUMNS[method]
        for index, match_id in enumerate(frame["match_id"]):
            for market_id, selections in MARKETS.items():
                prices = frame.loc[
                    index, [f"{prefix}_{suffix}" for _, suffix in selections]
                ].to_numpy(dtype=float)
                expected = remove_margin(prices, method)
                for position, (selection, _) in enumerate(selections):
                    row = table[
                        (table["match_id"] == match_id)
                        & (table["market_id"] == market_id)
                        & (table["selection"] == selection)
                    ]
                    assert row[column].iloc[0] == pytest.approx(
                        expected[position], rel=1e-12
                    )


def test_the_proportional_column_is_the_scaled_inverse_odds() -> None:
    table = selection_table(hand_frame(), "ps", "early")
    home = table[
        (table["match_id"] == "a")
        & (table["market_id"] == "1X2")
        & (table["selection"] == "home")
    ]
    assert home["p_proportional"].iloc[0] == pytest.approx(14 / 29)
    for column in PROBABILITY_COLUMNS.values():
        totals = table.groupby(["match_id", "market_id"])[column].sum()
        np.testing.assert_allclose(totals, 1.0, atol=1e-12)


def test_a_week_is_the_friday_on_or_before_the_match() -> None:
    table = selection_table(hand_frame(), "ps", "early")
    weeks = table.drop_duplicates("match_id").set_index("match_id")["batch"]
    assert weeks.to_dict() == {
        "a": pd.Timestamp("2022-09-02"),
        "b": pd.Timestamp("2022-09-02"),
        "c": pd.Timestamp("2023-09-01"),
        "d": pd.Timestamp("2023-09-08"),
    }


def test_a_quote_with_a_missing_price_is_left_out_whole() -> None:
    frame = hand_frame()
    frame.loc[1, "psc_d"] = np.nan
    close = selection_table(frame, "ps", "close")
    broken = close[(close["match_id"] == "b") & (close["market_id"] == "1X2")]
    assert broken.empty
    assert (
        len(close[(close["match_id"] == "b") & (close["market_id"] == "OU:2.5")]) == 2
    )
    assert len(selection_table(frame, "ps", "early")) == 20


def test_matches_without_a_usable_score_are_left_out() -> None:
    frame = hand_frame()
    frame.loc[0, "home_goals"] = np.nan
    frame.loc[1, "home_goals"] = 12  # off the score grid
    table = selection_table(frame, "ps", "early")
    assert sorted(table["match_id"].unique()) == ["c", "d"]


def test_matches_without_a_date_are_left_out_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    frame = hand_frame()
    frame.loc[2, "date"] = pd.NaT
    with caplog.at_level(logging.WARNING, logger="src.margin_audit"):
        table = selection_table(frame, "ps", "early")
    assert sorted(table["match_id"].unique()) == ["a", "b", "d"]
    assert "5 prices have no match date" in caplog.text
    assert table.index.equals(pd.RangeIndex(len(table)))


def test_dates_written_as_text_are_read_as_dates() -> None:
    frame = hand_frame()
    as_text = frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d"))
    pd.testing.assert_frame_equal(
        selection_table(as_text, "ps", "early"),
        selection_table(frame, "ps", "early"),
    )


def test_a_quote_that_comes_through_incomplete_is_left_out_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = module.odds_long

    def with_a_price_missing(
        enriched: pd.DataFrame, snapshot: str, books: Sequence[str]
    ) -> pd.DataFrame:
        offers = real(enriched, snapshot, books)
        lost = (
            (offers["match_id"] == "b")
            & (offers["market_id"] == "1X2")
            & (offers["selection"] == "draw")
        )
        return offers[~lost].reset_index(drop=True)

    monkeypatch.setattr(module, "odds_long", with_a_price_missing)
    table = selection_table(hand_frame(), "ps", "close")
    assert table[(table["match_id"] == "b") & (table["market_id"] == "1X2")].empty
    assert len(table) == 4 * (3 + 2) - 3
    for column in PROBABILITY_COLUMNS.values():
        assert table[column].notna().all()


def test_a_book_with_no_prices_gives_an_empty_table_with_the_right_columns() -> None:
    table = selection_table(hand_frame(), "avg", "early")
    assert table.empty
    assert list(table.columns) == SELECTION_COLUMNS


def test_an_unknown_snapshot_is_an_error() -> None:
    with pytest.raises(ValueError, match="snapshot"):
        selection_table(hand_frame(), "ps", "midweek")


def test_the_table_does_not_touch_the_match_table() -> None:
    frame = hand_frame()
    before = frame.copy()
    selection_table(frame, "ps", "close")
    pd.testing.assert_frame_equal(frame, before)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
HAND_QUOTES = [
    ("q1", 2022, "2022-09-02", "1X2", "home", {
        "proportional": (0.5, 0.3, 0.2), "power": (0.55, 0.28, 0.17), "shin": (0.52, 0.29, 0.19)}),
    ("q2", 2022, "2022-09-02", "1X2", "draw", {
        "proportional": (0.4, 0.35, 0.25), "power": (0.42, 0.34, 0.24), "shin": (0.41, 0.345, 0.245)}),
    ("q3", 2022, "2022-09-09", "OU:2.5", "under", {
        "proportional": (0.55, 0.45), "power": (0.5, 0.5), "shin": (0.52, 0.48)}),
    ("q4", 2023, "2023-09-01", "1X2", "away", {
        "proportional": (0.45, 0.3, 0.25), "power": (0.47, 0.3, 0.23), "shin": (0.46, 0.3, 0.24)}),
    ("q5", 2023, "2023-09-01", "OU:2.5", "over", {
        "proportional": (0.6, 0.4), "power": (0.62, 0.38), "shin": (0.61, 0.39)}),
]  # fmt: skip

#: (window, market) -> the quotes in it, as (method -> probability of the winner)
WINNER_PROBABILITY = {
    "q1": {"proportional": 0.5, "power": 0.55, "shin": 0.52},
    "q2": {"proportional": 0.35, "power": 0.34, "shin": 0.345},
    "q3": {"proportional": 0.45, "power": 0.5, "shin": 0.48},
    "q4": {"proportional": 0.25, "power": 0.23, "shin": 0.24},
    "q5": {"proportional": 0.6, "power": 0.62, "shin": 0.61},
}
GROUPS = {
    ("selection", "1X2"): ["q1", "q2"],
    ("selection", "OU:2.5"): ["q3"],
    ("selection", "all"): ["q1", "q2", "q3"],
    ("test", "1X2"): ["q4"],
    ("test", "OU:2.5"): ["q5"],
    ("test", "all"): ["q4", "q5"],
}


def cell(scores: pd.DataFrame, window: str, market_id: str, method: str) -> pd.Series:
    rows = scores[
        (scores["window"] == window)
        & (scores["market_id"] == market_id)
        & (scores["method"] == method)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


def test_log_loss_is_the_negative_log_of_the_winners_probability() -> None:
    scores = score_methods(quote_table(HAND_QUOTES), first_season=2023, n_resamples=50)
    assert list(scores.columns) == [c for c in SCORE_COLUMNS if c != "snapshot"]
    assert len(scores) == 6 * 3
    for (window, market_id), quotes in GROUPS.items():
        for method in METHODS:
            losses = [-math.log(WINNER_PROBABILITY[q][method]) for q in quotes]
            row = cell(scores, window, market_id, method)
            assert row["log_loss"] == pytest.approx(
                sum(losses) / len(losses), rel=1e-12
            )
            assert row["n"] == len(quotes)
    assert cell(scores, "selection", "1X2", "proportional")[
        "log_loss"
    ] == pytest.approx((-math.log(0.5) - math.log(0.35)) / 2)


def test_the_gain_is_the_proportional_loss_less_the_methods_loss_quote_by_quote() -> (
    None
):
    scores = score_methods(quote_table(HAND_QUOTES), first_season=2023, n_resamples=50)
    for (window, market_id), quotes in GROUPS.items():
        for method in METHODS:
            gains = [
                math.log(
                    WINNER_PROBABILITY[q][method]
                    / WINNER_PROBABILITY[q]["proportional"]
                )
                for q in quotes
            ]
            row = cell(scores, window, market_id, method)
            assert row["gain"] == pytest.approx(sum(gains) / len(gains), abs=1e-12)
    power = cell(scores, "test", "all", "power")
    assert power["gain"] == pytest.approx(
        (math.log(0.23 / 0.25) + math.log(0.62 / 0.6)) / 2
    )


def test_the_proportional_method_has_no_gain_and_no_interval() -> None:
    scores = score_methods(quote_table(HAND_QUOTES), first_season=2023, n_resamples=50)
    baseline = scores[scores["method"] == "proportional"]
    assert len(baseline) == 6
    assert (baseline["gain"] == 0.0).all()
    assert baseline[["gain_low", "gain_high"]].isna().all().all()
    others = scores[scores["method"] != "proportional"]
    assert others[["gain_low", "gain_high"]].notna().all().all()
    assert (others["gain_low"] <= others["gain"] + 1e-12).all()
    assert (others["gain"] <= others["gain_high"] + 1e-12).all()


def test_seasons_before_the_first_one_are_the_selection_window() -> None:
    table = quote_table(HAND_QUOTES)
    for first_season, expected in (
        (2023, {"selection": 3, "test": 2}),
        (2022, {"test": 5}),
        (2024, {"selection": 5}),
    ):
        scores = score_methods(table, first_season=first_season, n_resamples=20)
        pooled = scores[
            (scores["market_id"] == "all") & (scores["method"] == "proportional")
        ]
        assert pooled.set_index("window")["n"].to_dict() == expected


def test_markets_without_a_quote_are_absent() -> None:
    table = quote_table([q for q in HAND_QUOTES if q[3] == "1X2"])
    scores = score_methods(table, first_season=2023, n_resamples=20)
    assert set(scores["market_id"]) == {"1X2", "all"}


def test_a_table_with_no_winner_scores_nothing() -> None:
    table = quote_table(HAND_QUOTES).assign(won=False)
    scores = score_methods(table, first_season=2023, n_resamples=20)
    assert scores.empty
    assert list(scores.columns) == [c for c in SCORE_COLUMNS if c != "snapshot"]


def week_table(per_week: Sequence[tuple[str, int, float]]) -> pd.DataFrame:
    """Quotes whose winner is 0.2 under proportional and ``0.2 * exp(gain)`` otherwise."""
    quotes = []
    for number, (week, count, gain) in enumerate(per_week):
        for index in range(count):
            winner = 0.2 * math.exp(gain)
            quotes.append(
                (f"w{number}-{index}", 2023, week, "OU:2.5", "over", {
                    "proportional": (0.2, 0.8), "power": (winner, 1 - winner), "shin": (winner, 1 - winner)})
            )  # fmt: skip
    return quote_table(quotes)


def test_the_interval_resamples_whole_weeks_not_single_quotes() -> None:
    # Week one: three quotes that each gain 1. Week two: one that loses 1. Resampling
    # the two weeks gives a ratio of 1, 0.5 or -1, so the interval is exactly
    # [-1, 1]. Resampling the four quotes one by one would give a narrower one.
    table = week_table([("2023-09-01", 3, 1.0), ("2023-09-08", 1, -1.0)])
    row = cell(score_methods(table, n_resamples=2000), "test", "all", "power")
    assert row["gain"] == pytest.approx(0.5)
    assert row["gain_low"] == pytest.approx(-1.0)
    assert row["gain_high"] == pytest.approx(1.0)


def test_quotes_of_one_week_cannot_be_told_apart_by_resampling() -> None:
    table = week_table([("2023-09-01", 3, 1.0), ("2023-09-01", 1, -1.0)])
    row = cell(score_methods(table, n_resamples=200), "test", "all", "power")
    assert row["gain"] == pytest.approx(0.5)
    assert row["gain_low"] == pytest.approx(0.5)
    assert row["gain_high"] == pytest.approx(0.5)


def test_one_resample_collapses_the_interval_and_more_widen_it() -> None:
    table = week_table(
        [("2023-09-01", 2, 1.0), ("2023-09-08", 2, -0.5), ("2023-09-15", 2, 0.3)]
    )
    one = cell(score_methods(table, n_resamples=1), "test", "all", "power")
    many = cell(score_methods(table, n_resamples=500), "test", "all", "power")
    assert one["gain_low"] == one["gain_high"]
    assert many["gain_low"] < many["gain_high"]


def test_the_seed_fixes_the_resampling() -> None:
    table = week_table(
        [("2023-09-01", 2, 1.0), ("2023-09-08", 2, -0.5), ("2023-09-15", 2, 0.3)]
    )
    first = score_methods(table, n_resamples=300, seed=1)
    pd.testing.assert_frame_equal(first, score_methods(table, n_resamples=300, seed=1))
    other = score_methods(table, n_resamples=300, seed=2)
    assert not first[["gain_low", "gain_high"]].equals(other[["gain_low", "gain_high"]])


def test_scoring_leaves_the_table_alone() -> None:
    table = quote_table(HAND_QUOTES)
    before = table.copy()
    score_methods(table, first_season=2023, n_resamples=20)
    pd.testing.assert_frame_equal(table, before)


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #
def reliability_input() -> pd.DataFrame:
    """Selections on the edges of the bands, won or lost as written."""
    rows = [
        # market, p_proportional, p_power, p_shin, won
        ("1X2", 0.08, 0.06, 0.07, True),
        ("1X2", 0.0801, 0.07, 0.08, False),
        ("1X2", 0.02, 0.01, 0.015, False),
        ("1X2", 0.06, 0.045, 0.05, False),
        ("1X2", 0.15, 0.14, 0.145, False),
        ("1X2", 0.65, 0.66, 0.67, True),
        ("1X2", 0.9, 0.92, 0.91, True),
        ("OU:2.5", 0.55, 0.5, 0.52, True),
        ("OU:2.5", 0.45, 0.5, 0.48, False),
    ]
    return pd.DataFrame(
        {
            "market_id": [r[0] for r in rows],
            "p_proportional": [r[1] for r in rows],
            "p_power": [r[2] for r in rows],
            "p_shin": [r[3] for r in rows],
            "won": [r[4] for r in rows],
        }
    )


def test_bands_include_their_upper_end_and_are_set_by_the_proportional_probability() -> (
    None
):
    table = reliability_table(reliability_input())
    assert list(table.columns) == [c for c in RELIABILITY_COLUMNS if c != "snapshot"]
    first = table[(table["market_id"] == "1X2") & (table["band"] == "0-8%")].iloc[0]
    assert first["n"] == 3  # 0.02, 0.06 and exactly 0.08
    assert first["observed"] == pytest.approx(1 / 3)
    assert first["proportional"] == pytest.approx((0.08 + 0.02 + 0.06) / 3)
    assert first["power"] == pytest.approx((0.06 + 0.01 + 0.045) / 3)
    assert first["shin"] == pytest.approx((0.07 + 0.015 + 0.05) / 3)
    second = table[(table["market_id"] == "1X2") & (table["band"] == "8-15%")].iloc[0]
    assert second["n"] == 2  # 0.0801 and exactly 0.15
    assert second["observed"] == 0.0
    top = table[(table["market_id"] == "1X2") & (table["band"] == "65-100%")].iloc[0]
    assert top["n"] == 1  # 0.65 is in 50-65%, 0.9 is here
    assert top["observed"] == 1.0
    assert top["power"] == pytest.approx(0.92)


def test_reliability_lists_markets_then_bands_and_skips_empty_ones() -> None:
    table = reliability_input()
    result = reliability_table(table)
    assert result["market_id"].tolist() == ["1X2"] * 4 + ["OU:2.5"] * 2
    assert result["band"].tolist() == [
        "0-8%",
        "8-15%",
        "50-65%",
        "65-100%",
        "35-50%",
        "50-65%",
    ]
    assert result["n"].sum() == len(table)
    only_over_under = reliability_table(table[table["market_id"] == "OU:2.5"])
    assert set(only_over_under["market_id"]) == {"OU:2.5"}


def test_the_band_labels_name_the_documented_edges() -> None:
    assert BAND_LABELS == (
        "0-8%",
        "8-15%",
        "15-25%",
        "25-35%",
        "35-50%",
        "50-65%",
        "65-100%",
    )
    assert module.BAND_EDGES == (0.0, 0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 1.0)


def test_a_probability_of_zero_belongs_to_the_lowest_band() -> None:
    table = pd.DataFrame(
        {
            "market_id": "1X2",
            "p_proportional": [0.0],
            "p_power": [0.0],
            "p_shin": [0.0],
            "won": [False],
        }
    )
    assert reliability_table(table)["band"].tolist() == ["0-8%"]


@pytest.mark.parametrize("index", range(1, 8))
def test_every_band_includes_its_upper_end_and_the_next_starts_just_above(
    index: int,
) -> None:
    edge = module.BAND_EDGES[index]
    probabilities = [edge, edge + 1e-6] if index < 7 else [edge]
    table = pd.DataFrame(
        {
            "market_id": "1X2",
            "p_proportional": probabilities,
            "p_power": probabilities,
            "p_shin": probabilities,
            "won": True,
        }
    )
    bands = reliability_table(table)["band"].tolist()
    assert bands == list(BAND_LABELS[index - 1 : index + 1])[: len(probabilities)]


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_a_clear_gain_before_the_first_season_that_holds_after_it_is_recommended() -> (
    None
):
    scores = scores_of(
        {
            "close_selection": {"shin": CLEAR},
            "close_test": {"shin": CLEAR},
            "early_test": {"shin": FLAT},
        }
    )
    assert decide(scores) == ("shin", "shin")


def test_the_larger_gain_wins_among_the_clear_ones() -> None:
    bigger = (0.0020, 0.0010, 0.0030)
    for first, second in (("power", "shin"), ("shin", "power")):
        cells = {
            "close_selection": {first: bigger, second: CLEAR},
            "close_test": {first: CLEAR, second: CLEAR},
            "early_test": {first: FLAT, second: FLAT},
        }
        assert decide(scores_of(cells)) == (first, first)


def test_a_tie_goes_to_the_method_listed_first() -> None:
    cells = {
        "close_selection": {"shin": CLEAR, "power": CLEAR},
        "close_test": {"shin": CLEAR, "power": CLEAR},
        "early_test": {"shin": FLAT, "power": FLAT},
    }
    assert decide(scores_of(cells)) == ("power", "power")


def test_an_interval_holding_zero_is_not_enough_however_large_the_gain() -> None:
    cells = {
        "close_selection": {"power": NEARLY, "shin": CLEAR},
        "close_test": {"power": CLEAR, "shin": CLEAR},
        "early_test": {"power": FLAT, "shin": FLAT},
    }
    assert decide(scores_of(cells)) == ("shin", "shin")


@pytest.mark.parametrize("low", [0.0, -1e-9])
def test_an_interval_that_only_reaches_zero_is_not_clearly_better(low: float) -> None:
    edge = (0.0005, low, 0.0010)
    cells = {"close_selection": {"shin": edge}, "close_test": {"shin": CLEAR}}
    assert decide(scores_of(cells)) == ("proportional", "proportional")


def test_nothing_clearly_better_keeps_proportional_scaling() -> None:
    scores = scores_of(
        {
            "close_selection": {"power": NEARLY, "shin": FLAT},
            "close_test": {"power": CLEAR},
        }
    )
    assert decide(scores) == ("proportional", "proportional")


def test_a_worse_method_is_never_chosen() -> None:
    scores = scores_of({"close_selection": {"power": WORSE, "shin": WORSE}})
    assert decide(scores) == ("proportional", "proportional")


def test_only_the_closing_line_of_the_pooled_markets_chooses() -> None:
    early = scores_of(
        {"early_selection": {"shin": CLEAR}, "early_test": {"shin": CLEAR}}
    )
    assert decide(early) == ("proportional", "proportional")
    one_market = scores_of({"close_selection": {"shin": CLEAR}}).assign(market_id="1X2")
    assert decide(one_market) == ("proportional", "proportional")


def test_the_choice_needs_the_closing_line_to_hold_in_the_later_seasons() -> None:
    for held in (FLAT, WORSE, (0.0005, 0.0, 0.001)):
        scores = scores_of(
            {
                "close_selection": {"shin": CLEAR},
                "close_test": {"shin": held},
                "early_test": {"shin": CLEAR},
            }
        )
        assert decide(scores) == ("shin", "proportional")


def test_the_choice_is_vetoed_by_an_early_price_that_is_clearly_worse() -> None:
    scores = scores_of(
        {
            "close_selection": {"shin": CLEAR},
            "close_test": {"shin": CLEAR},
            "early_test": {"shin": WORSE},
        }
    )
    assert decide(scores) == ("shin", "proportional")


@pytest.mark.parametrize("high", [0.0, 0.0001, 0.001])
def test_an_early_price_that_is_not_clearly_worse_does_not_veto(high: float) -> None:
    early = (-0.0004, -0.0010, high)
    scores = scores_of(
        {
            "close_selection": {"shin": CLEAR},
            "close_test": {"shin": CLEAR},
            "early_test": {"shin": early},
        }
    )
    assert decide(scores) == ("shin", "shin")


def test_the_choice_cannot_be_confirmed_without_both_later_rows() -> None:
    no_early = scores_of(
        {"close_selection": {"shin": CLEAR}, "close_test": {"shin": CLEAR}}
    )
    assert decide(no_early) == ("shin", "proportional")
    no_close = scores_of(
        {"close_selection": {"shin": CLEAR}, "early_test": {"shin": FLAT}}
    )
    assert decide(no_close) == ("shin", "proportional")


def test_the_decision_is_made_without_rows_to_read() -> None:
    assert decide(scores_of({})) == ("proportional", "proportional")


def test_a_row_without_an_interval_is_never_clearly_better_or_worse() -> None:
    row = pd.Series({"gain": 0.0, "gain_low": np.nan, "gain_high": np.nan})
    assert not module._clearly_better(row)
    assert not module._clearly_worse(row)


# --------------------------------------------------------------------------- #
# Whole audits on simulated books
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def power_priced() -> pd.DataFrame:
    return simulated_world("power", seed=0)


@pytest.fixture(scope="module")
def power_audit(power_priced: pd.DataFrame) -> MarginAudit:
    return audit_margin(power_priced, n_resamples=200)


@pytest.fixture(scope="module")
def scaled_audit() -> MarginAudit:
    return audit_margin(simulated_world("proportional", seed=0), n_resamples=200)


def test_a_book_that_shades_long_shots_gets_a_method_that_undoes_it(
    power_audit: MarginAudit,
) -> None:
    assert power_audit.chosen in ("power", "shin")
    assert power_audit.recommended == power_audit.chosen
    for method in ("power", "shin"):
        row = cell(
            power_audit.scores[power_audit.scores["snapshot"] == "close"],
            "selection",
            "all",
            method,
        )
        assert row["gain_low"] > 0.0
    # the power method undoes exactly the rule that priced the book
    shares = power_audit.scores[
        (power_audit.scores["market_id"] == "all")
        & (power_audit.scores["snapshot"] == "close")
        & (power_audit.scores["window"] == "test")
    ].set_index("method")["gain"]
    assert shares["power"] > shares["shin"] > 0.0


def test_a_book_that_scales_proportionally_keeps_proportional_scaling(
    scaled_audit: MarginAudit,
) -> None:
    assert scaled_audit.chosen == "proportional"
    assert scaled_audit.recommended == "proportional"
    close = scaled_audit.scores[scaled_audit.scores["snapshot"] == "close"]
    for method in ("power", "shin"):
        assert cell(close, "selection", "all", method)["gain"] < 0.0


def test_the_audit_scores_every_snapshot_window_market_and_method(
    power_audit: MarginAudit,
) -> None:
    scores = power_audit.scores
    assert list(scores.columns) == SCORE_COLUMNS
    assert len(scores) == 2 * 2 * 3 * 3
    assert scores["snapshot"].tolist()[:18] == ["close"] * 18
    cells = scores[["snapshot", "window", "market_id", "method"]].to_numpy().tolist()
    assert cells == [
        [snapshot, window, market_id, method]
        for snapshot in ("close", "early")
        for window in ("selection", "test")
        for market_id in ("1X2", "OU:2.5", "all")
        for method in ("proportional", "power", "shin")
    ]
    pooled = scores[
        (scores["market_id"] == "all") & (scores["method"] == "proportional")
    ]
    sizes = pooled.set_index(["snapshot", "window"])["n"].to_dict()
    assert sizes == {
        ("close", "selection"): 2 * 4 * 800,
        ("close", "test"): 2 * 3 * 800,
        ("early", "selection"): 2 * 4 * 800,
        ("early", "test"): 2 * 3 * 800,
    }
    assert power_audit.book == "ps"
    assert power_audit.first_season == 2023


@pytest.fixture(scope="module")
def regime_change() -> MarginAudit:
    """A book that shaded long shots before 2023 and scaled proportionally after."""
    before = simulated_world("power", seasons=(2019, 2020, 2021, 2022), seed=0)
    after = simulated_world("proportional", seasons=(2023, 2024, 2025), seed=1)
    after["match_id"] = "later-" + after["match_id"]
    return audit_margin(pd.concat([before, after], ignore_index=True), n_resamples=200)


def test_a_choice_the_later_seasons_do_not_confirm_is_kept_apart_from_the_recommendation(
    regime_change: MarginAudit,
) -> None:
    assert regime_change.chosen in ("power", "shin")
    assert regime_change.recommended == "proportional"
    lines = verdict(regime_change)
    assert lines[0].startswith("Chosen on seasons before 2023")
    assert lines[1].endswith("proportional scaling stays")
    assert "not confirmed" in lines[1]
    assert lines[-1] == "Recommended: keep the default, python -m src.backtest"


def test_the_audit_reports_reliability_for_both_snapshots(
    power_audit: MarginAudit,
) -> None:
    reliability = power_audit.reliability
    assert list(reliability.columns) == RELIABILITY_COLUMNS
    assert set(reliability["snapshot"]) == {"close", "early"}
    close = reliability[reliability["snapshot"] == "close"]
    assert close["n"].sum() == 7 * 800 * (3 + 2)
    lowest = close[(close["market_id"] == "1X2") & (close["band"] == "0-8%")].iloc[0]
    # a book that shades long shots: proportional scaling says more than happened
    assert lowest["proportional"] > lowest["observed"]
    assert lowest["power"] < lowest["proportional"]


def test_the_first_season_moves_the_windows(power_priced: pd.DataFrame) -> None:
    late = audit_margin(power_priced, first_season=2025, n_resamples=20)
    pooled = late.scores[
        (late.scores["market_id"] == "all")
        & (late.scores["method"] == "proportional")
        & (late.scores["snapshot"] == "close")
    ]
    assert pooled.set_index("window")["n"].to_dict() == {
        "selection": 2 * 6 * 800,
        "test": 2 * 800,
    }
    assert late.first_season == 2025


def test_with_no_earlier_season_nothing_can_be_chosen(
    power_priced: pd.DataFrame,
) -> None:
    audit = audit_margin(power_priced, first_season=2000, n_resamples=20)
    assert set(audit.scores["window"]) == {"test"}
    assert (audit.chosen, audit.recommended) == ("proportional", "proportional")
    assert "No season before 2000 has Pinnacle closing quotes" in " ".join(
        verdict(audit)
    )


def test_without_closing_prices_the_early_ones_are_still_audited(
    power_priced: pd.DataFrame,
) -> None:
    early_only = power_priced.drop(
        columns=[c for c in power_priced if c.startswith("psc_")]
    )
    audit = audit_margin(early_only, n_resamples=20)
    assert set(audit.scores["snapshot"]) == {"early"}
    assert set(audit.reliability["snapshot"]) == {"early"}
    assert (audit.chosen, audit.recommended) == ("proportional", "proportional")
    assert "0 closing and 11200 early quotes scored" in format_margin_audit(audit)


def test_another_book_can_be_audited() -> None:
    frame = simulated_world(
        "power", per_season=300, seasons=(2021, 2022, 2023, 2024), book="avg"
    )
    audit = audit_margin(frame, book="avg", n_resamples=20)
    assert audit.book == "avg"
    assert "market average" in " ".join(verdict(audit))
    with pytest.raises(
        ValueError, match="no complete 1X2 or over/under Pinnacle quote"
    ):
        audit_margin(frame, book="ps", n_resamples=20)


def test_the_resamples_and_the_seed_reach_the_intervals(
    power_priced: pd.DataFrame,
) -> None:
    one = audit_margin(power_priced, n_resamples=1)
    others = one.scores[one.scores["method"] != "proportional"]
    assert (others["gain_low"] == others["gain_high"]).all()
    first = audit_margin(power_priced, n_resamples=100, seed=1)
    again = audit_margin(power_priced, n_resamples=100, seed=1)
    other = audit_margin(power_priced, n_resamples=100, seed=2)
    pd.testing.assert_frame_equal(first.scores, again.scores)
    assert not first.scores["gain_low"].equals(other.scores["gain_low"])


def test_the_audit_does_not_touch_the_match_table(power_priced: pd.DataFrame) -> None:
    before = power_priced.copy()
    audit_margin(power_priced, n_resamples=20)
    pd.testing.assert_frame_equal(power_priced, before)


def test_the_audit_refuses_a_table_it_cannot_read(power_priced: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError, match="the match table is missing columns: \\['date'\\]"
    ):
        audit_margin(power_priced.drop(columns="date"))
    repeated = pd.concat([power_priced, power_priced.iloc[[0]]])
    with pytest.raises(ValueError, match="repeats a match"):
        audit_margin(repeated)
    with pytest.raises(ValueError, match="the book must be one of"):
        audit_margin(power_priced, book="max")
    bare = power_priced[["match_id", "season", "date", "home_goals", "away_goals"]]
    with pytest.raises(
        ValueError, match="no complete 1X2 or over/under Pinnacle quote"
    ):
        audit_margin(bare)


def test_a_table_where_no_quote_has_a_result_cannot_be_audited(
    power_priced: pd.DataFrame,
) -> None:
    unplayed = power_priced.assign(home_goals=np.nan, away_goals=np.nan)
    with pytest.raises(ValueError, match="with a result"):
        audit_margin(unplayed)


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #
def full_scores() -> pd.DataFrame:
    """Scores with every row the report prints, and numbers that are easy to find."""
    rows = []
    for snapshot, window, n in (
        ("close", "selection", 1200),
        ("close", "test", 600),
        ("early", "selection", 1100),
        ("early", "test", 500),
    ):
        for market_id, share in (("1X2", 0.5), ("OU:2.5", 0.5), ("all", 1.0)):
            size = int(n * share)
            rows.append(
                (
                    snapshot,
                    window,
                    market_id,
                    "proportional",
                    size,
                    0.98765,
                    0.0,
                    np.nan,
                    np.nan,
                )
            )
            rows.append(
                (
                    snapshot,
                    window,
                    market_id,
                    "power",
                    size,
                    0.98675,
                    0.0009,
                    0.0004,
                    0.0014,
                )
            )
            rows.append(
                (
                    snapshot,
                    window,
                    market_id,
                    "shin",
                    size,
                    0.98715,
                    0.0005,
                    -0.0001,
                    0.0011,
                )
            )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def reliability_rows(low_band_n: int = 578) -> pd.DataFrame:
    rows = [
        ("close", "1X2", "0-8%", low_band_n, 0.052, 0.059, 0.050, 0.047),
        ("close", "1X2", "8-15%", 2753, 0.09, 0.12, 0.109, 0.11),
        ("close", "OU:2.5", "35-50%", 8749, 0.421, 0.431, 0.425, 0.427),
        ("early", "1X2", "0-8%", 600, 0.05, 0.06, 0.05, 0.05),
    ]
    return pd.DataFrame(rows, columns=RELIABILITY_COLUMNS)


def test_the_report_headline_counts_the_quotes_of_each_snapshot() -> None:
    text = format_margin_audit(audit_of(full_scores()))
    assert text.splitlines()[0] == (
        "Margin audit of Pinnacle prices: 1800 closing and 1600 early quotes "
        "scored. Seasons before 2023 choose the method; seasons from 2023 (the "
        "backtest) only confirm it"
    )


def test_the_report_names_the_book_and_the_first_season() -> None:
    text = format_margin_audit(audit_of(full_scores(), book="b365", first_season=2021))
    assert text.splitlines()[0].startswith("Margin audit of Bet365 prices:")
    assert "Seasons before 2021 choose the method; seasons from 2021" in text
    windows = {
        " ".join(line.split()[1:3])
        for line in text.splitlines()
        if line.split()[:1] in (["close"], ["early"])
    }
    assert windows == {"before 2021", "from 2021"}


def test_the_log_loss_table_has_a_row_per_snapshot_window_and_market() -> None:
    text = format_margin_audit(audit_of(full_scores()))
    assert (
        "Log-loss of each method on the outcome that happened, lower is better." in text
    )
    assert "positive = the method is more accurate" in text
    header = tokens(text, "snapshot")
    assert header[:7] == [
        "snapshot",
        "window",
        "market",
        "n",
        "proportional",
        "power",
        "shin",
    ]
    assert header[7:] == [
        "power",
        "gain",
        "[95%",
        "interval]",
        "shin",
        "gain",
        "[95%",
        "interval]",
    ]
    rows = [
        line.split()
        for line in text.splitlines()
        if line.split()[:1] in (["close"], ["early"])
    ]
    assert len(rows) == 12
    assert sum(row[0] == "close" for row in rows) == 6
    assert [row[:4] for row in rows[:3]] == [
        ["close", "before", "2023", "1X2"],
        ["close", "before", "2023", "OU:2.5"],
        ["close", "before", "2023", "both"],
    ]


def test_each_row_shows_the_log_losses_and_the_gains_with_their_intervals() -> None:
    text = format_margin_audit(audit_of(full_scores()))
    selection = [
        line for line in text.splitlines()
        if line.split()[:3] == ["close", "before", "2023"] and line.split()[3] == "1X2"
    ]  # fmt: skip
    assert len(selection) == 1
    assert selection[0].split() == [
        "close", "before", "2023", "1X2", "600",
        "0.98765", "0.98675", "0.98715",
        "+0.00090", "[+0.00040,", "+0.00140]",
        "+0.00050", "[-0.00010,", "+0.00110]",
    ]  # fmt: skip
    pooled = [
        line for line in text.splitlines()
        if line.split()[:3] == ["early", "from", "2023"] and line.split()[3] == "both"
    ]  # fmt: skip
    assert len(pooled) == 1
    assert pooled[0].split()[4] == "500"


def test_the_markets_are_titled_by_their_ids_and_both_markets_together() -> None:
    text = format_margin_audit(audit_of(full_scores()))
    markets = {
        line.split()[3]
        for line in text.splitlines()
        if line.split()[:3] == ["close", "from", "2023"]
    }
    assert markets == {"1X2", "OU:2.5", "both"}


def test_the_reliability_table_shows_the_closing_line_in_percent() -> None:
    audit = audit_of(full_scores(), reliability=reliability_rows())
    text = format_margin_audit(audit)
    assert (
        "Pinnacle closing line: how often selections won, against what each method said."
        in text
    )
    header = tokens(text, "won %")
    assert header == [
        "market",
        "band",
        "n",
        "won",
        "%",
        "proportional",
        "%",
        "power",
        "%",
        "shin",
        "%",
    ]
    assert tokens(text, "0-8%") == ["1X2", "0-8%", "578", "5.2", "5.9", "5.0", "4.7"]
    assert tokens(text, "8-15%") == [
        "1X2",
        "8-15%",
        "2753",
        "9.0",
        "12.0",
        "10.9",
        "11.0",
    ]
    assert tokens(text, "35-50%") == [
        "OU:2.5",
        "35-50%",
        "8749",
        "42.1",
        "43.1",
        "42.5",
        "42.7",
    ]
    assert (
        "600" not in text.split("how often selections won")[1]
    )  # the early row is not shown


def test_without_closing_reliability_the_report_has_no_such_table() -> None:
    early_only = reliability_rows().query("snapshot == 'early'")
    text = format_margin_audit(audit_of(full_scores(), reliability=early_only))
    assert "how often selections won" not in text
    assert "how often selections won" not in format_margin_audit(
        audit_of(full_scores())
    )


def verdict_of(
    chosen: str, recommended: str, scores: pd.DataFrame | None = None, **kwargs: object
) -> list[str]:
    audit = audit_of(
        full_scores() if scores is None else scores,
        reliability=reliability_rows(),
        chosen=chosen,
        recommended=recommended,
        **kwargs,  # type: ignore[arg-type]
    )
    return verdict(audit)


def test_a_confirmed_choice_is_recommended_with_the_command_to_run() -> None:
    lines = verdict_of("power", "power")
    assert lines[0] == (
        "Chosen on seasons before 2023 (Pinnacle closing line, both markets): the "
        "power method is more accurate than proportional scaling by +0.00090 "
        "[+0.00040, +0.00140] log-loss per quote"
    )
    assert lines[1] == (
        "Seasons from 2023: closing line +0.00090 [+0.00040, +0.00140], early "
        "price +0.00090 [+0.00040, +0.00140]: confirmed"
    )
    assert lines[-1] == "Recommended: python -m src.backtest --demargin power"


def test_a_choice_the_later_seasons_do_not_confirm_is_not_recommended() -> None:
    lines = verdict_of("power", "proportional")
    assert lines[0].startswith("Chosen on seasons before 2023")
    assert lines[1].endswith(
        "not confirmed (the closing-line gain needs an interval above zero and the "
        "early price must not be clearly worse); proportional scaling stays"
    )
    assert lines[-1] == "Recommended: keep the default, python -m src.backtest"


def test_when_nothing_is_clearly_better_the_best_candidate_is_named() -> None:
    scores = scores_of(
        {
            "close_selection": {
                "power": (0.0003, -0.0002, 0.0008),
                "shin": (0.0006, -0.0001, 0.0013),
            },
        }
    )
    lines = verdict_of("proportional", "proportional", scores)
    assert lines[0] == (
        "No method is clearly more accurate than proportional scaling on the "
        "Pinnacle closing line before season 2023; the best, Shin's method, gains "
        "+0.00060 [-0.00010, +0.00130] log-loss per quote; proportional scaling stays"
    )
    assert lines[-1] == "Recommended: keep the default, python -m src.backtest"


def test_without_earlier_closing_quotes_the_choice_cannot_be_made() -> None:
    scores = scores_of({"close_test": {"power": CLEAR}})
    lines = verdict_of("proportional", "proportional", scores)
    assert lines[0] == (
        "No season before 2023 has Pinnacle closing quotes with a result, so a "
        "method cannot be chosen out of sample; proportional scaling stays"
    )


@pytest.mark.parametrize(
    "later", [[], ["close_test"], ["early_test"]], ids=["neither", "closing", "early"]
)
def test_a_choice_without_both_later_rows_cannot_be_confirmed(
    later: list[str],
) -> None:
    scores = scores_of(
        {
            "close_selection": {"power": CLEAR},
            **{name: {"power": CLEAR} for name in later},
        }
    )
    lines = verdict_of("power", "proportional", scores)
    assert lines[1] == (
        "Seasons from 2023 lack closing or early quotes, so the choice cannot be "
        "confirmed; proportional scaling stays"
    )


def test_the_verdict_quotes_the_lowest_band_of_the_closing_1x2_line() -> None:
    lines = verdict_of("proportional", "proportional")
    quoted = [line for line in lines if "below 8%" in line]
    expected = (
        "Pinnacle closing 1X2 selections that proportional scaling puts below 8% "
        "(578 of them) won 5.2% of the time; proportional scaling says 5.9%, the "
        "power method 5.0%, Shin's method 4.7%"
    )
    assert quoted == [expected]


def test_a_thin_lowest_band_is_not_quoted() -> None:
    audit = audit_of(full_scores(), reliability=reliability_rows(low_band_n=49))
    assert not any("below 8%" in line for line in verdict(audit))
    audit = audit_of(full_scores(), reliability=reliability_rows(low_band_n=50))
    assert any("below 8%" in line for line in verdict(audit))
    assert not any("below 8%" in line for line in verdict(audit_of(full_scores())))
    early_only = reliability_rows().query("snapshot == 'early'")
    audit = audit_of(full_scores(), reliability=early_only)
    assert not any("below 8%" in line for line in verdict(audit))


def test_the_verdict_names_the_book() -> None:
    lines = verdict_of("power", "power", book="avg")
    assert "market average closing line" in lines[0]
    assert lines[-1] == "Recommended: python -m src.backtest --demargin power"


# --------------------------------------------------------------------------- #
# Saving and the command line
# --------------------------------------------------------------------------- #
def test_the_tables_are_saved_as_csv(power_audit: MarginAudit, tmp_path: Path) -> None:
    written = save_margin_audit(power_audit, tmp_path / "deep" / "audit")
    assert [path.name for path in written] == ["scores.csv", "reliability.csv"]
    scores = pd.read_csv(written[0])
    assert list(scores.columns) == SCORE_COLUMNS
    assert len(scores) == len(power_audit.scores)
    reliability = pd.read_csv(written[1])
    assert list(reliability.columns) == RELIABILITY_COLUMNS
    assert len(reliability) == len(power_audit.reliability)
    np.testing.assert_allclose(scores["gain"], power_audit.scores["gain"])


def test_the_default_folder_is_inside_the_processed_data() -> None:
    assert MARGIN_AUDIT_DIR.name == "margin_audit"
    assert MARGIN_AUDIT_DIR.parent.name == "processed"


def write_world(tmp_path: Path, **options: object) -> Path:
    path = tmp_path / "enriched.parquet"
    simulated_world(
        "power", per_season=300, seasons=(2021, 2022, 2023, 2024), **options
    ).to_parquet(  # type: ignore[arg-type]
        path, index=False
    )
    return path


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_prints_the_audit_the_verdict_and_saves_the_tables(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enriched = write_world(tmp_path)
    output = tmp_path / "audit"
    arguments = [
        "--enriched",
        str(enriched),
        "--resamples",
        "50",
        "--output-dir",
        str(output),
    ]
    with caplog.at_level(logging.INFO):
        assert main(arguments) == 0
    assert (
        "Margin audit of Pinnacle prices: 2400 closing and 2400 early quotes"
        in caplog.text
    )
    assert "Log-loss of each method on the outcome that happened" in caplog.text
    assert "closing line: how often selections won" in caplog.text
    assert "Recommended: " in caplog.text
    assert f"Wrote {output / 'scores.csv'}" in caplog.text
    assert f"Wrote {output / 'reliability.csv'}" in caplog.text
    assert (output / "scores.csv").exists()
    assert (output / "reliability.csv").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_can_audit_another_book_and_season(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "enriched.parquet"
    frame = simulated_world(
        "power", per_season=300, seasons=(2021, 2022, 2023, 2024), book="avg"
    )
    frame.to_parquet(path, index=False)
    arguments = ["--enriched", str(path), "--book", "avg", "--first-season", "2024"]
    arguments += [
        "--resamples",
        "30",
        "--seed",
        "3",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    with caplog.at_level(logging.INFO):
        assert main(arguments) == 0
    assert "Margin audit of market average prices" in caplog.text
    assert "Seasons before 2024 choose the method" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_passes_every_option_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_world(tmp_path)
    seen: dict[str, object] = {}

    def fake(frame: pd.DataFrame, **options: object) -> MarginAudit:
        seen.update(options)
        raise ValueError("stop here")

    monkeypatch.setattr(module, "audit_margin", fake)
    arguments = ["--enriched", str(path), "--book", "b365", "--first-season", "2022"]
    arguments += ["--resamples", "77", "--seed", "5"]
    assert main(arguments) == 1
    assert seen == {"book": "b365", "first_season": 2022, "n_resamples": 77, "seed": 5}


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_names_the_step_to_run_when_the_table_is_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(tmp_path / "nope.parquet")]) == 1
    assert "python -m src.enrich" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_table_it_cannot_read(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "enriched.parquet"
    simulated_world("power", per_season=50, seasons=(2022, 2023)).drop(
        columns="date"
    ).to_parquet(path, index=False)
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(path)]) == 1
    assert (
        "Cannot audit the margin removal: the match table is missing columns"
        in caplog.text
    )


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_when_the_odds_are_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "enriched.parquet"
    frame = simulated_world("power", per_season=50, seasons=(2022, 2023))
    frame[["match_id", "season", "date", "home_goals", "away_goals"]].to_parquet(
        path, index=False
    )
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(path)]) == 1
    assert "no complete 1X2 or over/under Pinnacle quote" in caplog.text


def test_the_command_defaults() -> None:
    args = module._parse_args([])
    assert args.book == "ps"
    assert args.first_season == 2023
    assert args.resamples == 2000
    assert args.seed == 0
    assert args.output_dir == MARGIN_AUDIT_DIR
    assert args.enriched == module.ENRICHED_OUTPUT


def test_the_command_only_audits_books_that_give_a_fair_view() -> None:
    with pytest.raises(SystemExit):
        main(["--book", "max"])
    for book in ("ps", "avg", "b365"):
        assert module._parse_args(["--book", book]).book == book


def test_the_command_needs_at_least_one_resample(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--resamples", "0"])
    assert "--resamples must be at least 1" in capsys.readouterr().err
    assert module._parse_args(["--resamples", "1"]).resamples == 1


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_missing_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = write_world(tmp_path)

    def fake(frame: pd.DataFrame, **options: object) -> MarginAudit:
        raise KeyError("home_goals")

    monkeypatch.setattr(module, "audit_margin", fake)
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(path)]) == 1
    assert "Cannot audit the margin removal: 'home_goals'" in caplog.text
