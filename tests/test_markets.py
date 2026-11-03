"""Tests for the market engine."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import logging_config
from src.dc_baseline import outcome_probabilities
from src.dixon_coles import score_matrix
from src.markets import (
    MIN_GRID_GOALS,
    fixture_markets,
    main,
    market_set,
    market_table,
    selection_probabilities,
)

N = 11  # a 10-goal grid


def delta(home_goals: int, away_goals: int) -> np.ndarray:
    """A matrix where the final score is certain."""
    matrix = np.zeros((N, N))
    matrix[home_goals, away_goals] = 1.0
    return matrix


def price(matrix: np.ndarray) -> dict[tuple[str, str], tuple[float, float]]:
    table = market_table(matrix)
    return {
        (row.market_id, row.selection): (row.probability, row.push_probability)
        for row in table.itertuples(index=False)
    }


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
# The catalog
# --------------------------------------------------------------------------- #
def test_at_least_fifty_markets_and_no_duplicates() -> None:
    catalog = market_set().catalog
    assert catalog["market_id"].nunique() >= 50
    assert not catalog.duplicated(["market_id", "selection"]).any()
    assert len(catalog) > 150
    expected = {
        "1X2",
        "DC",
        "DNB",
        "OU:1.5",
        "OU:2.5",
        "OU:3.5",
        "OU:4.5",
        "BTTS",
        "CS",
        "EH:-1",
        "EH:1",
        "AH:-0.5",
        "WM",
    }
    assert expected <= set(catalog["market_id"])


def test_the_market_set_is_cached_and_read_only() -> None:
    assert market_set(10) is market_set(10)
    with pytest.raises(ValueError):
        market_set(10).win[0, 0] = 5.0


def test_small_grids_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_goals must be"):
        market_set(MIN_GRID_GOALS - 1)
    assert market_set(MIN_GRID_GOALS).catalog["market_id"].nunique() >= 50


# --------------------------------------------------------------------------- #
# Certain scores give exact answers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("market", "selection", "win", "push"),
    [
        ("1X2", "home", 1, 0), ("1X2", "away", 0, 0),
        ("DC", "home_draw", 1, 0), ("DC", "home_away", 1, 0), ("DC", "draw_away", 0, 0),
        ("DNB", "home", 1, 0),
        ("OU:2.5", "over", 1, 0), ("OU:3.5", "under", 1, 0), ("OU:2.5", "under", 0, 0),
        ("BTTS", "yes", 1, 0), ("OE", "odd", 1, 0), ("TG", "3", 1, 0),
        ("GB4", "2-3", 1, 0), ("GB3", "2-4", 1, 0),
        ("TT_HOME:1.5", "over", 1, 0), ("TT_AWAY:0.5", "over", 1, 0), ("TT_AWAY:1.5", "under", 1, 0),
        ("TG_HOME", "2", 1, 0), ("TG_AWAY", "1", 1, 0),
        ("TS_HOME", "yes", 1, 0), ("TS_AWAY", "yes", 1, 0),
        ("WTN_HOME", "yes", 0, 0), ("WTN_HOME", "no", 1, 0),
        ("EH:-1", "draw", 1, 0), ("EH:-2", "away", 1, 0), ("EH:1", "home", 1, 0),
        ("AH:-1", "home", 0, 1), ("AH:-1", "away", 0, 1), ("AH:-0.5", "home", 1, 0), ("AH:0.5", "home", 1, 0),
        ("AH:-1.5", "away", 1, 0),
        ("WM", "home_by_1", 1, 0),
        ("1X2_BTTS", "home_yes", 1, 0), ("1X2_OU:2.5", "home_over", 1, 0),
        ("DC_OU:2.5", "home_draw_over", 1, 0), ("DC_BTTS", "home_draw_yes", 1, 0),
        ("BTTS_OU:2.5", "yes_over", 1, 0),
        ("CS", "2-1", 1, 0), ("CS", "other", 0, 0),
    ],
)  # fmt: skip
def test_a_certain_2_1_win(market: str, selection: str, win: int, push: int) -> None:
    probabilities = price(delta(2, 1))
    assert probabilities[(market, selection)] == (win, push)


def test_a_certain_7_0_lands_in_the_open_ended_selections() -> None:
    probabilities = price(delta(7, 0))
    for key in (
        ("CS", "other"),
        ("TG", "6+"),
        ("GB4", "6+"),
        ("GB3", "5+"),
        ("WM", "home_by_3plus"),
        ("TG_HOME", "3+"),
        ("WTN_HOME", "yes"),
    ):
        assert probabilities[key][0] == 1, key


def test_a_certain_goalless_draw() -> None:
    probabilities = price(delta(0, 0))
    for key in (
        ("1X2", "draw"),
        ("OU:0.5", "under"),
        ("BTTS", "no"),
        ("OE", "even"),
        ("TG", "0"),
        ("CS", "0-0"),
        ("AH:0", "home"),
        ("WM", "draw"),
    ):
        assert probabilities[key][0] + probabilities[key][1] == 1, key


# --------------------------------------------------------------------------- #
# Identities on a realistic matrix
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def matrix() -> np.ndarray:
    return score_matrix(1.7, 1.1, -0.07)


def test_exclusive_markets_cover_every_outcome_once(matrix: np.ndarray) -> None:
    table = market_table(matrix)
    for market_id, group in table[table["exclusive"]].groupby("market_id"):
        refunded = group["push_probability"].max()
        assert group["probability"].sum() + refunded == pytest.approx(1.0), market_id


def test_double_chance_style_markets_count_each_outcome_twice(
    matrix: np.ndarray,
) -> None:
    table = market_table(matrix)
    for market_id, group in table[~table["exclusive"]].groupby("market_id"):
        assert group["probability"].sum() == pytest.approx(2.0), market_id


def test_markets_agree_with_each_other_and_with_the_baseline(
    matrix: np.ndarray,
) -> None:
    p = {key: value[0] for key, value in price(matrix).items()}
    home, draw, away = outcome_probabilities(matrix)
    assert (
        p[("1X2", "home")],
        p[("1X2", "draw")],
        p[("1X2", "away")],
    ) == pytest.approx((home, draw, away))
    assert p[("DC", "home_draw")] == pytest.approx(home + draw)
    assert p[("AH:-0.5", "home")] == pytest.approx(home)  # winning outright
    assert p[("AH:0.5", "home")] == pytest.approx(home + draw)
    assert p[("AH:-1.5", "home")] == pytest.approx(
        p[("EH:-1", "home")]
    )  # winning by two or more
    assert p[("OU:2.5", "over")] == pytest.approx(
        1 - sum(p[("TG", str(k))] for k in range(3))
    )
    home_scoreless, away_scoreless = matrix[0].sum(), matrix[:, 0].sum()
    assert p[("BTTS", "yes")] == pytest.approx(
        1 - home_scoreless - away_scoreless + matrix[0, 0]
    )
    assert (
        p[("TT_HOME:0.5", "over")]
        == pytest.approx(p[("TS_HOME", "yes")])
        == pytest.approx(1 - home_scoreless)
    )
    assert p[("EH:-1", "draw")] == pytest.approx(p[("WM", "home_by_1")])
    assert sum(p[("CS", f"{i}-{j}")] for i in range(6) for j in range(6)) + p[
        ("CS", "other")
    ] == pytest.approx(1.0)


def test_draw_no_bet_odds_account_for_the_refund(matrix: np.ndarray) -> None:
    table = market_table(matrix).set_index(["market_id", "selection"])
    home, draw, away = outcome_probabilities(matrix)
    assert table.loc[("DNB", "home"), "fair_odds"] == pytest.approx(
        (home + away) / home
    )
    assert table.loc[("DNB", "home"), "push_probability"] == pytest.approx(draw)
    assert table.loc[("1X2", "home"), "fair_odds"] == pytest.approx(1 / home)


def test_impossible_selections_have_infinite_fair_odds() -> None:
    table = market_table(delta(2, 1)).set_index(["market_id", "selection"])
    assert np.isinf(table.loc[("1X2", "away"), "fair_odds"])
    assert table.loc[("1X2", "home"), "fair_odds"] == 1.0


# --------------------------------------------------------------------------- #
# Batch pricing and fixtures
# --------------------------------------------------------------------------- #
def test_batch_pricing_equals_pricing_one_matrix_at_a_time() -> None:
    matrices = np.stack(
        [score_matrix(1.2 + 0.2 * k, 0.9 + 0.1 * k, -0.05) for k in range(4)]
    )
    win, push = selection_probabilities(matrices)
    assert win.shape == (4, len(market_set().catalog))
    for k in range(4):
        single_win, single_push = selection_probabilities(matrices[k])
        np.testing.assert_allclose(win[k], single_win[0])
        np.testing.assert_allclose(push[k], single_push[0])


@pytest.mark.parametrize("bad", [np.zeros(5), np.zeros((4, 5)), np.zeros((2, 3, 4))])
def test_non_square_input_is_rejected(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="square"):
        selection_probabilities(bad)


def test_fixture_markets_layout_and_values() -> None:
    rates = pd.DataFrame(
        {"lam_home": [1.8, 0.9], "lam_away": [0.9, 1.4]}, index=["m1", "m2"]
    )
    table = fixture_markets(rates, rho=-0.05)
    size = len(market_set().catalog)
    assert len(table) == 2 * size
    assert list(table["match_id"].unique()) == ["m1", "m2"]
    one = table[table["match_id"] == "m1"].reset_index(drop=True)
    direct = market_table(score_matrix(1.8, 0.9, -0.05))
    np.testing.assert_allclose(one["probability"], direct["probability"])
    assert one["selection"].tolist() == direct["selection"].tolist()
    assert table["probability"].between(0, 1).all()
    assert table.loc[table["market_id"] == "1X2", "probability"].sum() == pytest.approx(
        2.0
    )  # one per fixture


def test_fixture_markets_accepts_explicit_ids_and_clips_extreme_inputs() -> None:
    rates = pd.DataFrame({"lam_home": [0.0, 50.0], "lam_away": [-1.0, 50.0]})
    table = fixture_markets(rates, rho=-0.9, match_ids=["a", "b"])
    assert set(table["match_id"]) == {"a", "b"}
    assert np.isfinite(table["probability"]).all()
    with pytest.raises(ValueError, match="one entry per row"):
        fixture_markets(rates, rho=0.0, match_ids=["only-one"])


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("isolated_logging")
def test_cli_prints_a_summary_and_the_full_table() -> None:
    assert main([]) == 0
    assert main(["--lam", "2.1", "--mu", "0.7", "--rho", "-0.1", "--all"]) == 0
