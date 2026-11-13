"""Tests for the historical odds layer: long offers, fair prices, settling, the audit."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import logging_config
from src.odds_history import (
    BOOKS,
    FAIR_BOOKS,
    MARKETS,
    OFFER_COLUMNS,
    SNAPSHOTS,
    OddsAudit,
    audit_odds,
    coverage_table,
    fair_probabilities,
    fair_view,
    format_audit,
    line_movement,
    main,
    margin_table,
    market_learning,
    odds_long,
    price_column,
    settle,
    verdict,
)


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
def hand() -> pd.DataFrame:
    """Four matches with round numbers: easy to check by eye.

    Match ``a``: Pinnacle early 1.9/3.8/3.8 is a fair 0.5/0.25/0.25 plus a margin;
    ``b`` is a flat market; ``c`` mirrors ``a``; ``d`` ended 11-0, off the score grid.
    """
    return pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "season": [2022, 2022, 2023, 2023],
            "home_goals": [2, 0, 1, 11],
            "away_goals": [1, 0, 3, 0],
            "ps_h": [1.9, 2.85, 3.8, 1.5],
            "ps_d": [3.8, 2.85, 3.8, 4.0],
            "ps_a": [3.8, 2.85, 1.9, 6.0],
            "psc_h": [1.8, 2.85, 3.8, 1.5],
            "psc_d": [4.0, 2.85, 3.8, 4.0],
            "psc_a": [4.0, 2.85, 1.9, 6.0],
            "ps_over25": [1.9, 2.0, 1.8, 1.7],
            "ps_under25": [1.9, 1.8, 2.0, 2.2],
            "psc_over25": [1.85, 2.0, 1.8, 1.7],
            "psc_under25": [2.0, 1.8, 2.0, 2.2],
            "b365_h": [1.85, np.nan, 3.6, 1.5],
            "b365_d": [3.7, 2.8, 3.6, 4.0],
            "b365_a": [3.6, 2.8, 1.8, 6.0],
        }
    )


def priced(probabilities: np.ndarray, margin: float) -> np.ndarray:
    """Odds whose inverses add up to ``1 + margin``, in proportion to the probabilities."""
    return 1.0 / (probabilities * (1.0 + margin))


def add_quotes(
    frame: pd.DataFrame,
    book: str,
    snapshot: str,
    p_1x2: np.ndarray | None,
    p_over_under: np.ndarray | None,
    margin: float,
) -> pd.DataFrame:
    """Write one book's wide columns for a snapshot, from fair probabilities."""
    for market_id, probabilities in (("1X2", p_1x2), ("OU:2.5", p_over_under)):
        if probabilities is None:
            continue
        odds = priced(probabilities, margin)
        for position, (_, suffix) in enumerate(MARKETS[market_id]):
            frame[price_column(book, snapshot, suffix)] = odds[:, position]
    return frame


def base(n: int, seasons: tuple[int, ...] = (2022, 2023)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "season": [seasons[i % len(seasons)] for i in range(n)],
            "home_goals": 1,
            "away_goals": 0,
        }
    )


def exact_table(n: int = 8, seed: int = 0) -> pd.DataFrame:
    """Every book in both snapshots, each with a known margin."""
    rng = np.random.default_rng(seed)
    frame = base(n)
    for book, margin in (("ps", 0.02), ("b365", 0.05), ("avg", 0.04), ("max", -0.01)):
        for snapshot in SNAPSHOTS:
            add_quotes(
                frame,
                book,
                snapshot,
                rng.dirichlet([4.0, 3.0, 3.0], n),
                rng.dirichlet([5.0, 5.0], n),
                margin,
            )
    return frame


def tilt(
    probabilities: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """A noisy view of the probabilities (log-normal noise, renormalised)."""
    view = probabilities * np.exp(rng.normal(0.0, sigma, probabilities.shape))
    return view / view.sum(axis=1, keepdims=True)


def simulated(
    n: int = 6_000,
    *,
    move_noise: float = 0.25,
    reverse: bool = False,
    seed: int = 0,
    seasons: tuple[int, ...] = (2022, 2023),
) -> pd.DataFrame:
    """Matches whose closing price is close to the truth and whose early price is noisier.

    The truth draws the result and the over/under outcome; goals are chosen to
    agree with both. With ``reverse`` the early price is the better one.
    """
    rng = np.random.default_rng(seed)
    truth_1x2 = rng.dirichlet([4.0, 3.0, 3.0], n)
    truth_ou = rng.dirichlet([5.0, 5.0], n)
    result = (rng.random(n)[:, None] > truth_1x2.cumsum(axis=1)).sum(axis=1)
    over = rng.random(n) < truth_ou[:, 0]
    home = np.where(
        result == 0,
        np.where(over, 3, 1),
        np.where(result == 1, np.where(over, 2, 0), 0),
    )
    away = np.where(result == 2, np.where(over, 3, 1), np.where(result == 1, np.where(over, 2, 0), 0))  # fmt: skip
    frame = base(n, seasons).assign(home_goals=home, away_goals=away)
    better_1x2 = tilt(truth_1x2, 0.05, rng)
    better_ou = tilt(truth_ou, 0.05, rng)
    worse_1x2 = tilt(better_1x2, move_noise, rng)
    worse_ou = tilt(better_ou, move_noise, rng)
    early_1x2, close_1x2 = (
        (better_1x2, worse_1x2) if reverse else (worse_1x2, better_1x2)
    )
    early_ou, close_ou = (better_ou, worse_ou) if reverse else (worse_ou, better_ou)
    add_quotes(frame, "ps", "early", early_1x2, early_ou, 0.02)
    add_quotes(frame, "ps", "close", close_1x2, close_ou, 0.02)
    return frame


# --------------------------------------------------------------------------- #
# Column names
# --------------------------------------------------------------------------- #
def test_price_columns_follow_the_schema() -> None:
    assert price_column("ps", "early", "h") == "ps_h"
    assert price_column("ps", "close", "h") == "psc_h"
    assert price_column("b365", "close", "over25") == "b365c_over25"
    assert price_column("avg", "early", "under25") == "avg_under25"


def test_an_unknown_snapshot_is_rejected() -> None:
    with pytest.raises(ValueError, match="snapshot"):
        price_column("ps", "late", "h")
    with pytest.raises(ValueError, match="snapshot"):
        odds_long(hand(), "late")
    with pytest.raises(ValueError, match="snapshot"):
        odds_long(hand(), "late", ())  # checked even when there is nothing to read


# --------------------------------------------------------------------------- #
# Wide columns to long offers
# --------------------------------------------------------------------------- #
def price(frame: pd.DataFrame, match: str, market: str, selection: str) -> float:
    chosen = frame[
        (frame["match_id"] == match)
        & (frame["market_id"] == market)
        & (frame["selection"] == selection)
    ]
    assert len(chosen) == 1
    return float(chosen["odds"].iloc[0])


def test_early_reads_the_plain_columns_and_close_the_c_columns() -> None:
    early = odds_long(hand(), "early", ("ps",))
    close = odds_long(hand(), "close", ("ps",))
    assert price(early, "a", "1X2", "home") == 1.9
    assert price(close, "a", "1X2", "home") == 1.8
    assert price(early, "a", "1X2", "draw") == 3.8
    assert price(close, "a", "1X2", "draw") == 4.0
    assert price(early, "a", "OU:2.5", "over") == 1.9
    assert price(close, "a", "OU:2.5", "over") == 1.85
    assert price(early, "c", "1X2", "away") == 1.9
    assert price(early, "c", "OU:2.5", "under") == 2.0


def test_offers_have_the_columns_the_engine_reads() -> None:
    early = odds_long(hand(), "early", ("ps",))
    assert list(early.columns) == OFFER_COLUMNS
    assert len(early) == 4 * 3 + 4 * 2
    assert set(early["bookmaker"]) == {"ps"}
    assert set(early["market_id"]) == {"1X2", "OU:2.5"}
    assert early["odds"].dtype == "float64"


def test_only_the_requested_books_are_read() -> None:
    bet365 = odds_long(hand(), "early", ("b365",))
    assert set(bet365["bookmaker"]) == {"b365"}
    assert set(bet365["market_id"]) == {"1X2"}  # the table has no b365 over/under
    both = odds_long(hand(), "early", ("ps", "b365"))
    assert set(both["bookmaker"]) == {"ps", "b365"}


def test_books_and_markets_without_columns_are_skipped() -> None:
    assert set(odds_long(hand(), "early")["bookmaker"]) == {"ps", "b365"}
    nothing = odds_long(hand(), "close", ("b365",))  # there are no b365c columns
    assert nothing.empty
    assert list(nothing.columns) == OFFER_COLUMNS
    assert nothing["odds"].dtype == "float64"
    assert odds_long(hand()[["match_id"]], "early").empty


def test_a_market_with_a_missing_column_is_skipped_not_half_read() -> None:
    frame = hand().drop(columns="ps_d")  # Pinnacle's early draw price has no column
    early = odds_long(frame, "early", ("ps",))
    assert set(early["market_id"]) == {"OU:2.5"}
    assert len(early) == 4 * 2
    assert len(odds_long(frame, "close", ("ps",))) == 4 * 3 + 4 * 2  # close is intact


def test_a_quote_missing_a_selection_is_dropped_whole() -> None:
    bet365 = odds_long(hand(), "early", ("b365",))  # match b has no home price
    assert set(bet365["match_id"]) == {"a", "c", "d"}
    assert (bet365.groupby("match_id").size() == 3).all()


@pytest.mark.parametrize("bad", [1.0, 0.9, 0.0, -2.0, np.inf, np.nan, "n/a"])
def test_an_impossible_price_drops_only_that_quote(bad: object) -> None:
    frame = hand()
    frame["ps_d"] = frame["ps_d"].astype(object)
    frame.loc[0, "ps_d"] = bad
    early = odds_long(frame, "early", ("ps",))
    one_x_two = early[early["market_id"] == "1X2"]
    assert set(one_x_two["match_id"]) == {"b", "c", "d"}
    over_under = early[early["market_id"] == "OU:2.5"]
    assert set(over_under["match_id"]) == {"a", "b", "c", "d"}  # untouched


def test_the_order_of_rows_in_the_table_does_not_matter() -> None:
    frame = hand()
    shuffled = frame.iloc[[3, 1, 0, 2]].set_axis([30, 10, 20, 0])
    keys = ["match_id", "market_id", "selection"]
    first = odds_long(frame, "early").sort_values(keys).reset_index(drop=True)
    second = odds_long(shuffled, "early").sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Taking the margin out
# --------------------------------------------------------------------------- #
def fair_by_key(offers: pd.DataFrame) -> pd.Series:
    fair = fair_probabilities(offers)
    return offers.assign(p=fair).set_index(["match_id", "market_id", "selection"])["p"]


def test_fair_probabilities_remove_the_margin_in_proportion() -> None:
    fair = fair_by_key(odds_long(hand(), "early", ("ps",)))
    assert fair[("a", "1X2", "home")] == pytest.approx(0.5)
    assert fair[("a", "1X2", "draw")] == pytest.approx(0.25)
    assert fair[("a", "1X2", "away")] == pytest.approx(0.25)
    assert fair[("a", "OU:2.5", "over")] == pytest.approx(0.5)
    assert fair[("b", "1X2", "draw")] == pytest.approx(1 / 3)
    assert fair[("c", "1X2", "away")] == pytest.approx(0.5)
    # 1/1.5, 1/4 and 1/6 add up to 1.0833, and are scaled by that
    assert fair[("d", "1X2", "home")] == pytest.approx(
        (1 / 1.5) / (1 / 1.5 + 0.25 + 1 / 6)
    )
    assert fair[("c", "OU:2.5", "over")] == pytest.approx((1 / 1.8) / (1 / 1.8 + 0.5))


def test_every_market_of_every_book_adds_up_to_one() -> None:
    offers = odds_long(exact_table(), "early")
    fair = offers.assign(p=fair_probabilities(offers))
    totals = fair.groupby(["match_id", "bookmaker", "market_id"])["p"].sum()
    np.testing.assert_allclose(totals, 1.0)
    assert len(totals) == 8 * 4 * 2


def test_fair_probabilities_match_the_probabilities_the_odds_were_made_from() -> None:
    rng = np.random.default_rng(1)
    truth = rng.dirichlet([4.0, 3.0, 3.0], 5)
    frame = add_quotes(base(5), "ps", "early", truth, None, 0.07)
    offers = odds_long(frame, "early", ("ps",))
    fair = offers.assign(p=fair_probabilities(offers))
    wide = fair.pivot(index="match_id", columns="selection", values="p")
    np.testing.assert_allclose(wide[["home", "draw", "away"]].to_numpy(), truth)


def test_fair_probabilities_keep_the_offers_index_and_flag_incomplete_markets() -> None:
    offers = odds_long(hand(), "early", ("ps",))
    assert fair_probabilities(offers).index.equals(offers.index)
    lonely = offers[~((offers["match_id"] == "a") & (offers["selection"] == "draw"))]
    fair = lonely.assign(p=fair_probabilities(lonely))
    broken = fair[(fair["match_id"] == "a") & (fair["market_id"] == "1X2")]
    assert broken["p"].isna().all()  # a market with a selection missing is not guessed
    assert fair.loc[fair["match_id"] == "b", "p"].notna().all()


def test_the_fair_view_takes_the_sharpest_book_that_has_a_quote() -> None:
    frame = base(4)
    truth = np.tile([0.5, 0.25, 0.25], (4, 1))
    tilted = truth * [1.0, 1.1, 0.9]
    tilted = tilted / tilted.sum(
        axis=1, keepdims=True
    )  # a different view of the same match
    add_quotes(frame, "ps", "close", truth, None, 0.02)
    add_quotes(frame, "avg", "close", tilted, None, 0.05)
    add_quotes(frame, "b365", "close", truth, None, 0.06)
    frame.loc[1, ["psc_h", "psc_d", "psc_a"]] = np.nan  # no Pinnacle for match m1
    frame.loc[2, ["psc_h", "psc_d", "psc_a"]] = np.nan
    frame.loc[2, ["avgc_h", "avgc_d", "avgc_a"]] = np.nan  # nor average for m2
    frame.loc[3, ["psc_h", "psc_d", "psc_a", "avgc_h", "avgc_d", "avgc_a"]] = np.nan
    frame.loc[3, ["b365c_h", "b365c_d", "b365c_a"]] = np.nan  # nothing at all for m3
    view = fair_view(frame, "close")
    sources = view.drop_duplicates("match_id").set_index("match_id")["source"]
    assert sources.to_dict() == {"m0": "ps", "m1": "avg", "m2": "b365"}
    assert "m3" not in set(view["match_id"])
    assert list(view.columns) == [
        "match_id",
        "market_id",
        "selection",
        "source",
        "p_fair",
    ]
    # a match's selections come from one book, so they still add up to one
    assert (view.groupby("match_id")["source"].nunique() == 1).all()
    np.testing.assert_allclose(view.groupby("match_id")["p_fair"].sum(), 1.0)
    pinnacle = view[view["match_id"] == "m0"].set_index("selection")["p_fair"]
    assert pinnacle["home"] == pytest.approx(0.5)


def test_the_fair_view_never_uses_the_best_price_by_default() -> None:
    frame = base(3)
    add_quotes(frame, "max", "close", np.tile([0.5, 0.25, 0.25], (3, 1)), None, -0.01)
    assert "max" not in FAIR_BOOKS
    assert fair_view(frame, "close").empty
    assert len(fair_view(frame, "close", ("max",))) == 3 * 3  # unless asked for it


def test_the_order_of_books_decides_which_is_used() -> None:
    frame = base(2)
    truth = np.tile([0.5, 0.25, 0.25], (2, 1))
    add_quotes(frame, "ps", "early", truth, None, 0.02)
    add_quotes(frame, "b365", "early", truth, None, 0.06)
    assert set(fair_view(frame, "early")["source"]) == {"ps"}
    assert set(fair_view(frame, "early", ("b365", "ps"))["source"]) == {"b365"}


def test_the_fair_view_of_nothing_is_empty_with_the_right_columns() -> None:
    view = fair_view(hand()[["match_id"]], "close")
    assert view.empty
    assert list(view.columns) == [
        "match_id",
        "market_id",
        "selection",
        "source",
        "p_fair",
    ]


# --------------------------------------------------------------------------- #
# Settling bets
# --------------------------------------------------------------------------- #
def settled(
    bets: list[tuple[str, str, str, float]], scores: list[tuple[str, float, float]]
) -> pd.DataFrame:
    offers = pd.DataFrame(bets, columns=["match_id", "market_id", "selection", "odds"])
    results = pd.DataFrame(scores, columns=["match_id", "home_goals", "away_goals"])
    return settle(offers, results)


def test_a_one_x_two_bet_wins_on_the_result() -> None:
    bets = [("m", "1X2", s, 2.5) for s in ("home", "draw", "away")]
    won = settled(bets, [("m", 2, 1)])
    assert list(won["won"]) == [True, False, False]
    assert list(won["settled"]) == [True, True, True]
    assert list(won["push"]) == [False, False, False]
    assert list(won["profit"]) == [1.5, -1.0, -1.0]
    assert list(settled(bets, [("m", 0, 0)])["won"]) == [False, True, False]
    assert list(settled(bets, [("m", 0, 4)])["won"]) == [False, False, True]


def test_over_under_pivots_on_three_goals() -> None:
    bets = [("m", "OU:2.5", s, 2.0) for s in ("over", "under")]
    for home, away, over_wins in (
        (1, 1, False),
        (2, 0, False),
        (2, 1, True),
        (3, 0, True),
        (0, 0, False),
    ):
        won = settled(bets, [("m", home, away)])
        assert list(won["won"]) == [over_wins, not over_wins], (home, away)


def test_a_refunded_stake_is_neither_a_win_nor_a_loss() -> None:
    draw_no_bet = settled(
        [("m", "DNB", "home", 1.8), ("m", "DNB", "away", 2.2)], [("m", 1, 1)]
    )
    assert list(draw_no_bet["push"]) == [True, True]
    assert list(draw_no_bet["won"]) == [False, False]
    assert list(draw_no_bet["profit"]) == [0.0, 0.0]
    handicap = settled(
        [
            ("m", "AH:-1", "home", 2.0),
            ("m", "AH:-1.5", "home", 2.4),
            ("m", "AH:-0.5", "home", 1.7),
        ],
        [("m", 1, 0)],
    )
    assert list(handicap["push"]) == [True, False, False]
    assert list(handicap["won"]) == [False, False, True]
    assert list(handicap["profit"]) == [0.0, -1.0, 0.7]


def test_the_edge_of_the_score_grid_is_settled_and_beyond_it_is_not() -> None:
    bets = [("m", "1X2", "home", 2.0)]
    assert bool(settled(bets, [("m", 10, 0)])["settled"].iloc[0])
    beyond = settled(bets, [("m", 11, 0)])
    assert not beyond["settled"].iloc[0]
    assert not beyond["won"].iloc[0] and not beyond["push"].iloc[0]
    assert np.isnan(beyond["profit"].iloc[0])


@pytest.mark.parametrize(
    "home, away",
    [
        (np.nan, 1.0),
        (1.0, np.nan),
        (-1.0, 0.0),
        (0.0, -2.0),
        (1.5, 0.0),
        (0.0, 0.5),
        (0.0, 11.0),
    ],
)
def test_a_score_that_cannot_be_used_leaves_the_bet_unsettled(
    home: float, away: float
) -> None:
    result = settled([("m", "1X2", "draw", 3.0)], [("m", home, away)])
    assert not result["settled"].iloc[0]
    assert not result["won"].iloc[0] and not result["push"].iloc[0]
    assert np.isnan(result["profit"].iloc[0])


def test_unknown_markets_selections_and_matches_are_left_unsettled() -> None:
    result = settled(
        [
            ("m", "NOPE", "home", 2.0),
            ("m", "1X2", "nobody", 2.0),
            ("ghost", "1X2", "home", 2.0),
            ("m", "1X2", "home", 2.0),
        ],
        [("m", 1, 0)],
    )
    assert list(result["settled"]) == [False, False, False, True]
    assert list(result["won"]) == [False, False, False, True]
    assert result["profit"].isna().tolist() == [True, True, True, False]


def test_settling_keeps_the_offers_untouched_and_in_order() -> None:
    offers = pd.DataFrame(
        {
            "match_id": ["b", "a", "b"],
            "market_id": ["1X2", "OU:2.5", "OU:2.5"],
            "selection": ["home", "over", "under"],
            "odds": [2.0, 1.9, 1.95],
            "bookmaker": ["x", "y", "z"],
        },
        index=[7, 8, 9],
    )
    before = offers.copy()
    results = pd.DataFrame(
        {"match_id": ["a", "b"], "home_goals": [3, 0], "away_goals": [1, 2]}
    )
    result = settle(offers, results)
    pd.testing.assert_frame_equal(offers, before)  # no side effects
    assert result.index.tolist() == [7, 8, 9]
    assert result["bookmaker"].tolist() == ["x", "y", "z"]
    assert result["won"].tolist() == [
        False,
        True,
        True,
    ]  # b lost 0-2 on home; a was over; b was under


def test_settling_nothing_gives_an_empty_result() -> None:
    result = settled([], [("m", 1, 0)])
    assert result.empty
    assert {"settled", "won", "push", "profit"} <= set(result.columns)


def test_settling_rejects_bad_inputs() -> None:
    offers = pd.DataFrame(
        {"match_id": ["m"], "market_id": ["1X2"], "selection": ["home"], "odds": [2.0]}
    )
    results = pd.DataFrame({"match_id": ["m"], "home_goals": [1], "away_goals": [0]})
    with pytest.raises(ValueError, match="offers is missing"):
        settle(offers.drop(columns="odds"), results)
    with pytest.raises(ValueError, match="results is missing"):
        settle(offers, results.drop(columns="away_goals"))
    with pytest.raises(ValueError, match="more than one row"):
        settle(offers, pd.concat([results, results]))


# --------------------------------------------------------------------------- #
# The audit: coverage, margins, movement, learning
# --------------------------------------------------------------------------- #
def test_coverage_counts_complete_quotes_per_season() -> None:
    frame = exact_table(8)
    frame.loc[0, "psc_d"] = np.nan  # m0 (season 2022) loses its closing 1X2 quote
    frame = frame.drop(
        columns=["max_over25", "max_under25", "maxc_over25", "maxc_under25"]
    )
    table = coverage_table(frame)
    assert len(table) == 2 * len(MARKETS) * len(BOOKS) * len(SNAPSHOTS)
    assert list(table.columns) == [
        "season",
        "market_id",
        "bookmaker",
        "snapshot",
        "matches",
        "share",
    ]

    def share(season: int, market: str, book: str, snapshot: str) -> float:
        row = table[
            (table["season"] == season)
            & (table["market_id"] == market)
            & (table["bookmaker"] == book)
            & (table["snapshot"] == snapshot)
        ]
        assert len(row) == 1
        return float(row["share"].iloc[0])

    assert share(2022, "1X2", "ps", "close") == 0.75  # 3 of 4
    assert share(2023, "1X2", "ps", "close") == 1.0
    assert share(2022, "1X2", "ps", "early") == 1.0
    assert (
        share(2022, "OU:2.5", "ps", "close") == 1.0
    )  # the over/under quote is separate
    assert share(2022, "OU:2.5", "max", "early") == 0.0  # columns gone: shown as zero
    assert share(2023, "OU:2.5", "max", "close") == 0.0
    assert share(2022, "1X2", "max", "close") == 1.0
    lost = table[
        (table["season"] == 2022)
        & (table["market_id"] == "1X2")
        & (table["bookmaker"] == "ps")
        & (table["snapshot"] == "close")
    ]
    assert lost["matches"].iloc[0] == 3


def test_coverage_of_a_table_without_odds_is_all_zeros() -> None:
    table = coverage_table(base(6))
    assert len(table) == 2 * len(MARKETS) * len(BOOKS) * len(SNAPSHOTS)
    assert (table["matches"] == 0).all() and (table["share"] == 0.0).all()


def test_margins_are_what_the_prices_were_built_with() -> None:
    table = margin_table(exact_table(8))
    assert len(table) == len(BOOKS) * len(MARKETS) * len(SNAPSHOTS)
    expected = {"ps": 0.02, "b365": 0.05, "avg": 0.04, "max": -0.01}
    for row in table.itertuples():
        assert row.margin == pytest.approx(expected[row.bookmaker]), row
        assert row.n == 8
    assert list(table.columns) == ["snapshot", "bookmaker", "market_id", "n", "margin"]


def test_margins_average_over_matches() -> None:
    frame = base(3)
    truth = np.tile([0.5, 0.25, 0.25], (3, 1))
    odds = np.vstack(
        [priced(truth[:1], 0.10), priced(truth[1:2], 0.0), priced(truth[2:], 0.0)]
    )
    frame["ps_h"], frame["ps_d"], frame["ps_a"] = odds.T  # margins of 10%, 0% and 0%
    table = margin_table(frame)
    assert len(table) == 1  # only Pinnacle's early 1X2 exists
    assert table["n"].iloc[0] == 3
    assert table["margin"].iloc[0] == pytest.approx(
        0.10 / 3
    )  # the mean, not the median


def test_margins_of_a_table_without_odds_are_empty() -> None:
    table = margin_table(base(3))
    assert table.empty
    assert list(table.columns) == ["snapshot", "bookmaker", "market_id", "n", "margin"]


def test_movement_is_the_change_in_the_fair_probability_not_in_the_odds() -> None:
    frame = base(4)
    add_quotes(
        frame,
        "ps",
        "early",
        np.tile([0.50, 0.30, 0.20], (4, 1)),
        np.tile([0.55, 0.45], (4, 1)),
        0.02,
    )
    add_quotes(
        frame,
        "ps",
        "close",
        np.tile([0.52, 0.29, 0.19], (4, 1)),
        np.tile([0.50, 0.50], (4, 1)),
        0.03,
    )
    table = line_movement(frame)
    assert list(table["selection"]) == ["home", "draw", "away", "over", "under"]
    assert list(table["market_id"]) == ["1X2"] * 3 + ["OU:2.5"] * 2
    np.testing.assert_allclose(table["mean_abs_move"], [2.0, 1.0, 1.0, 5.0, 5.0])
    assert (table["n"] == 4).all()
    assert (table["unchanged"] == 0.0).all()


def test_a_price_that_does_not_move_is_reported_as_unchanged() -> None:
    frame = base(5)
    truth = np.tile([0.5, 0.3, 0.2], (5, 1))
    add_quotes(frame, "ps", "early", truth, None, 0.02)
    add_quotes(frame, "ps", "close", truth, None, 0.02)
    frame.loc[0, "psc_h"] *= 1.1  # one price of one match moves
    table = line_movement(frame)
    by_selection = table.set_index("selection")
    assert by_selection.loc["home", "unchanged"] == pytest.approx(0.8)
    # The draw price itself never moved, but the market was scaled back to one, so
    # its fair probability moved a little in that match: 4 of 5 matches stayed put.
    assert by_selection.loc["draw", "unchanged"] == pytest.approx(1.0)
    implied = np.array([0.5 / 1.1, 0.3, 0.2]) * 1.02  # the moved market's prices
    fair = implied / implied.sum()
    assert by_selection.loc["home", "mean_abs_move"] == pytest.approx(
        abs(fair[0] - 0.5) * 100 / 5
    )
    assert by_selection.loc["draw", "mean_abs_move"] == pytest.approx(
        abs(fair[1] - 0.3) * 100 / 5
    )


def test_movement_needs_both_snapshots_from_the_same_book() -> None:
    frame = base(3)
    truth = np.tile([0.5, 0.3, 0.2], (3, 1))
    add_quotes(frame, "ps", "early", truth, None, 0.02)
    add_quotes(frame, "b365", "close", truth, None, 0.05)
    table = line_movement(frame)
    assert table.empty
    assert list(table.columns) == [
        "market_id",
        "selection",
        "n",
        "mean_abs_move",
        "unchanged",
    ]
    assert line_movement(frame, "b365").empty  # Bet365 has a close but no early price


def two_match_table() -> pd.DataFrame:
    """m0: home win 1-0 (under); m1: away win 0-2 (under)."""
    frame = base(2).assign(home_goals=[1, 0], away_goals=[0, 2])
    add_quotes(
        frame, "ps", "early",
        np.array([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]]), np.array([[0.5, 0.5], [0.45, 0.55]]), 0.02,
    )  # fmt: skip
    add_quotes(
        frame, "ps", "close",
        np.array([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]]), np.array([[0.4, 0.6], [0.45, 0.55]]), 0.02,
    )  # fmt: skip
    return frame


def test_learning_scores_the_winning_selection_of_each_snapshot() -> None:
    table = market_learning(two_match_table()).set_index("market_id")
    early_1x2 = -(np.log(0.5) + np.log(0.3)) / 2  # winners: home (m0), away (m1)
    close_1x2 = -(np.log(0.6) + np.log(0.5)) / 2
    assert table.loc["1X2", "n"] == 2
    assert table.loc["1X2", "early"] == pytest.approx(early_1x2)
    assert table.loc["1X2", "close"] == pytest.approx(close_1x2)
    assert table.loc["1X2", "gain"] == pytest.approx(early_1x2 - close_1x2)
    early_ou = -(np.log(0.5) + np.log(0.55)) / 2  # both matches ended under 2.5
    close_ou = -(np.log(0.6) + np.log(0.55)) / 2
    assert table.loc["OU:2.5", "early"] == pytest.approx(early_ou)
    assert table.loc["OU:2.5", "close"] == pytest.approx(close_ou)
    assert table.loc["OU:2.5", "gain"] == pytest.approx(early_ou - close_ou)
    assert (table["gain_low"] <= table["gain"]).all() and (
        table["gain"] <= table["gain_high"]
    ).all()


def test_a_match_without_a_usable_score_is_left_out_of_the_learning() -> None:
    frame = two_match_table()
    frame.loc[1, "home_goals"] = 12
    table = market_learning(frame).set_index("market_id")
    assert table.loc["1X2", "n"] == 1
    assert table.loc["1X2", "early"] == pytest.approx(-np.log(0.5))


def test_the_market_learns_when_the_close_is_the_better_view() -> None:
    table = market_learning(simulated(seed=1)).set_index("market_id")
    assert (table["n"] == 6_000).all()
    assert (table["gain"] > 0.003).all()
    assert (table["gain_low"] > 0.0).all()
    assert (table["early"] > table["close"]).all()


def test_a_closing_price_no_better_than_the_early_one_shows_no_gain() -> None:
    frame = simulated(n=3_000, seed=2)
    for column in [c for c in frame.columns if c.startswith("ps_")]:
        frame["psc_" + column[3:]] = frame[column]  # the close is just the early price
    table = market_learning(frame).set_index("market_id")
    np.testing.assert_allclose(table["gain"], 0.0, atol=1e-12)
    np.testing.assert_allclose(table["gain_low"], 0.0, atol=1e-12)
    np.testing.assert_allclose(table["gain_high"], 0.0, atol=1e-12)


def test_an_early_price_better_than_the_close_gives_a_negative_gain() -> None:
    table = market_learning(simulated(reverse=True, seed=3)).set_index("market_id")
    assert (table["gain"] < -0.003).all()
    assert (table["gain_high"] < 0.0).all()


def test_the_bootstrap_seed_is_passed_on() -> None:
    frame = simulated(n=500, seed=11)
    first = market_learning(frame, seed=1)
    pd.testing.assert_frame_equal(first, market_learning(frame, seed=1))
    other = market_learning(frame, seed=2)
    assert (first["gain"] == other["gain"]).all()  # the data decide the gain...
    assert (
        first["gain_low"] != other["gain_low"]
    ).any()  # ...the seed only the interval


def test_learning_needs_the_scores() -> None:
    with pytest.raises(ValueError, match="home_goals"):
        market_learning(two_match_table().drop(columns="home_goals"))


def test_learning_of_a_table_without_odds_is_empty() -> None:
    table = market_learning(base(4))
    assert table.empty
    assert list(table.columns) == [
        "market_id",
        "n",
        "early",
        "close",
        "gain",
        "gain_low",
        "gain_high",
    ]


# --------------------------------------------------------------------------- #
# The whole audit
# --------------------------------------------------------------------------- #
def test_the_audit_covers_all_seasons_but_compares_prices_only_in_the_window() -> None:
    frame = simulated(n=3_000, seed=4, seasons=(2021, 2022, 2023))
    audit = audit_odds(frame, from_season=2022)
    assert isinstance(audit, OddsAudit)
    assert audit.matches == 3_000
    assert audit.window_matches == 2_000
    assert audit.from_season == 2022 and audit.reference_book == "ps"
    assert set(audit.coverage["season"]) == {2021, 2022, 2023}
    assert set(audit.margins["bookmaker"]) == {"ps"}
    assert (audit.movement["n"] == 2_000).all()
    assert (audit.learning["n"] == 2_000).all()


def test_the_audit_can_use_another_book() -> None:
    frame = simulated(n=1_000, seed=5)
    for snapshot in SNAPSHOTS:
        for suffix in ("h", "d", "a", "over25", "under25"):
            frame[price_column("avg", snapshot, suffix)] = frame[
                price_column("ps", snapshot, suffix)
            ]
    only_avg = frame.drop(columns=[c for c in frame.columns if c.startswith("ps")])
    assert audit_odds(only_avg, reference_book="ps").learning.empty
    assert not audit_odds(only_avg, reference_book="avg").learning.empty


def test_the_audit_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="season"):
        audit_odds(simulated(n=100).drop(columns="season"))


def test_the_report_names_every_table() -> None:
    audit = audit_odds(simulated(n=1_500, seed=6), from_season=2022)
    text = format_audit(audit)
    for phrase in (
        "Odds audit: 1500 matches",
        "1X2: complete quotes",
        "Over/under 2.5 goals: complete quotes",
        "Average bookmaker margin",
        "Pinnacle: movement from the early to the closing price",
        "Pinnacle: log-loss of the early and the closing price",
        "mean_abs_move",
        "gain_low",
    ):
        assert phrase in text, phrase


def test_the_report_of_a_table_without_odds_still_renders() -> None:
    audit = audit_odds(base(10), from_season=2022)
    text = format_audit(audit)
    assert "Odds audit: 10 matches" in text
    assert "no selection has both an early and a closing price" in text
    assert "Average bookmaker margin" not in text
    assert "log-loss of the early" not in text


def test_the_verdict_says_whether_the_close_beats_the_early_price() -> None:
    learning = "\n".join(verdict(audit_odds(simulated(seed=7))))
    assert "6000 of 6000 matches (100%)" in learning
    assert "closing price beats the early one" in learning
    assert "not clearly better" not in learning
    flat = simulated(n=2_000, seed=8)
    for column in [c for c in flat.columns if c.startswith("ps_")]:
        flat["psc_" + column[3:]] = flat[column]
    assert "not clearly better" in "\n".join(verdict(audit_odds(flat)))


def test_the_verdict_explains_when_nothing_can_be_compared() -> None:
    (line,) = verdict(audit_odds(base(10), from_season=2022))
    assert "cannot be measured" in line and "--book avg" in line
    assert "season 2022" in line


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def write_table(tmp_path: Path, frame: pd.DataFrame) -> str:
    path = tmp_path / "enriched.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_prints_the_audit_and_the_verdict(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_table(tmp_path, simulated(n=2_000, seed=9))
    with caplog.at_level(logging.INFO):
        assert main(["--enriched", path]) == 0
    assert "Odds audit: 2000 matches" in caplog.text
    assert "Pinnacle: log-loss of the early and the closing price" in caplog.text
    assert "closing price beats the early one" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_honours_the_window_and_the_book(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_table(tmp_path, simulated(n=1_000, seed=10))
    with caplog.at_level(logging.INFO):
        assert main(["--enriched", path, "--from-season", "2099"]) == 0
    assert "No match since season 2099" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert main(["--enriched", path, "--book", "avg"]) == 0
    assert (
        "No match since season 2022 has both an early and a closing market average"
        in caplog.text
    )


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_names_the_step_to_run_when_the_table_is_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(tmp_path / "nope.parquet")]) == 1
    assert "python -m src.enrich" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_table_without_odds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", write_table(tmp_path, base(10))]) == 1
    assert "No complete 1X2 or over/under quote" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_table_missing_columns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_table(tmp_path, simulated(n=100).drop(columns="home_goals"))
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", path]) == 1
    assert "Cannot audit the odds" in caplog.text


def test_the_best_price_cannot_be_the_reference_book() -> None:
    with pytest.raises(SystemExit):
        main(["--book", "max"])
