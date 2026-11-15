"""Tests for the walk-forward backtest: no look-ahead, exact bookkeeping, honest measuring."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import backtest as bt
from src import logging_config
from src.backtest import (
    BACKTEST_DIR,
    BASELINE,
    BET_COLUMNS,
    BacktestResult,
    batch_start,
    build_offers,
    build_universe,
    cluster_bootstrap,
    flat_baseline,
    format_backtest,
    main,
    max_drawdown,
    odds_range,
    place_bets,
    run_backtest,
    save_backtest,
    settle_bets,
    strategy_label,
    summarise,
    verdict,
    walk_forward_probabilities,
)
from src.demargin import METHOD_NAMES, METHODS, remove_margin
from src.evaluate import fit_calibration
from src.markets import fixture_markets
from src.odds_history import KEY, MARKETS, fair_view
from src.value import ValueConfig


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
def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def tilt(
    probabilities: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """A noisy view of the probabilities (log-normal noise, renormalised)."""
    view = probabilities * np.exp(rng.normal(0.0, sigma, probabilities.shape))
    return view / view.sum(axis=1, keepdims=True)


def jitter(
    probabilities: np.ndarray, concentration: float, rng: np.random.Generator
) -> np.ndarray:
    """A Dirichlet draw whose mean is ``probabilities``: a martingale step."""
    draws = rng.gamma(concentration * probabilities)
    return draws / draws.sum(axis=1, keepdims=True)


def put_quotes(
    frame: pd.DataFrame,
    prefix: str,
    p_1x2: np.ndarray,
    p_over_under: np.ndarray,
    margin: float,
) -> None:
    """Write one book's wide odds columns from fair probabilities and a margin."""
    for market_id, probabilities in (("1X2", p_1x2), ("OU:2.5", p_over_under)):
        odds = 1.0 / (probabilities * (1.0 + margin))
        for position, (_, suffix) in enumerate(MARKETS[market_id]):
            frame[f"{prefix}_{suffix}"] = odds[:, position]


@dataclass(frozen=True)
class World:
    """Matches from a market that learns: the closing price is the truth.

    The early fair probability is what the market believes before the news;
    the closing one is a draw whose mean is the early one (so the early price is
    right on average) and the result is drawn from the closing one. Pinnacle quotes
    both; Bet365 quotes a noisy copy of the early view with a bigger margin.
    """

    frame: pd.DataFrame
    early_1x2: np.ndarray
    close_1x2: np.ndarray
    early_ou: np.ndarray
    close_ou: np.ndarray

    @property
    def results(self) -> pd.DataFrame:
        return self.frame[["match_id", "home_goals", "away_goals"]]


def make_world(
    per_season: int = 2_000,
    seasons: tuple[int, ...] = (2022, 2023, 2024),
    *,
    seed: int = 0,
    concentration: float = 300.0,
    soft_noise: float = 0.06,
) -> World:
    rng = np.random.default_rng(seed)
    n = per_season * len(seasons)
    early_1x2 = rng.dirichlet([4.0, 3.0, 3.0], n)
    early_ou = rng.dirichlet([5.0, 5.0], n)
    close_1x2 = jitter(early_1x2, concentration, rng)
    close_ou = jitter(early_ou, concentration, rng)
    result = (rng.random(n)[:, None] > close_1x2.cumsum(axis=1)).sum(axis=1)
    over = rng.random(n) < close_ou[:, 0]
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
    start = pd.to_datetime([f"{s}-08-10" for s in season])
    frame = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "season": season,
            "date": start + pd.to_timedelta(offsets, unit="D"),
            "home_goals": home,
            "away_goals": away,
        }
    )
    put_quotes(frame, "ps", early_1x2, early_ou, 0.03)
    put_quotes(frame, "psc", close_1x2, close_ou, 0.027)
    put_quotes(
        frame,
        "b365",
        tilt(early_1x2, soft_noise, rng),
        tilt(early_ou, soft_noise, rng),
        0.055,
    )
    put_quotes(
        frame, "avg", tilt(early_1x2, 0.02, rng), tilt(early_ou, 0.02, rng), 0.05
    )
    return World(frame, early_1x2, close_1x2, early_ou, close_ou)


def model_long(
    frame: pd.DataFrame, p_1x2: np.ndarray, p_over_under: np.ndarray
) -> pd.DataFrame:
    """The long probability table :func:`build_universe` reads."""
    pieces = []
    for market_id, probabilities in (("1X2", p_1x2), ("OU:2.5", p_over_under)):
        for position, (selection, _) in enumerate(MARKETS[market_id]):
            pieces.append(
                pd.DataFrame(
                    {
                        "match_id": frame["match_id"].to_numpy(),
                        "market_id": market_id,
                        "selection": selection,
                        "probability": probabilities[:, position],
                        "push_probability": 0.0,
                    }
                )
            )
    return pd.concat(pieces, ignore_index=True)


def play(
    frame: pd.DataFrame,
    model: pd.DataFrame,
    *,
    book: str = "b365",
    weight: float = 0.0,
    config: ValueConfig | None = None,
    reference: str = "ps",
) -> pd.DataFrame:
    """Universe, offers, bets and settlement for one strategy."""
    universe = build_universe(frame, model, reference)
    offers = build_offers(frame, universe, book)
    settings = ValueConfig(model_weight=weight) if config is None else config
    bets = place_bets(offers, settings)
    return settle_bets(bets, frame[["match_id", "home_goals", "away_goals"]])


@pytest.fixture(scope="module")
def world() -> World:
    return make_world(seed=11)


@pytest.fixture(scope="module")
def flat_model(world: World) -> pd.DataFrame:
    n = len(world.frame)
    return model_long(world.frame, np.full((n, 3), 1.0 / 3.0), np.full((n, 2), 0.5))


@pytest.fixture(scope="module")
def shopped(world: World, flat_model: pd.DataFrame) -> pd.DataFrame:
    """Weight 0 at Bet365: only the sharp book's early price is trusted."""
    return play(world.frame, flat_model, book="b365", weight=0.0)


def hand_table() -> pd.DataFrame:
    """Four matches with round numbers and deliberate gaps.

    ``a`` is complete. ``b`` has no Pinnacle closing home price (no 1X2) and ``c``
    no early over price (no over/under). ``d`` has no Bet365 home price.
    """
    return pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "season": 2023,
            "date": pd.to_datetime(
                ["2024-01-05", "2024-01-07", "2024-01-11", "2024-01-12"]
            ),
            "home_goals": [2, 0, 1, 3],
            "away_goals": [1, 0, 1, 0],
            "ps_h": [1.9, 2.85, 3.8, 1.5],
            "ps_d": [3.8, 2.85, 3.8, 4.0],
            "ps_a": [3.8, 2.85, 1.9, 6.0],
            "psc_h": [1.8, np.nan, 3.8, 1.5],
            "psc_d": [4.0, 2.85, 3.8, 4.0],
            "psc_a": [4.0, 2.85, 1.9, 6.0],
            "ps_over25": [1.9, 2.0, np.nan, 1.7],
            "ps_under25": [1.9, 1.8, 2.0, 2.2],
            "psc_over25": [1.85, 2.0, 1.8, 1.7],
            "psc_under25": [2.0, 1.8, 2.0, 2.2],
            "b365_h": [1.85, 2.8, 3.6, np.nan],
            "b365_d": [3.7, 2.8, 3.6, 4.0],
            "b365_a": [3.6, 2.8, 1.8, 6.0],
            "b365_over25": [1.95, 1.95, 1.75, 1.7],
            "b365_under25": [1.85, 1.85, 2.05, 2.2],
        }
    )


def hand_model() -> pd.DataFrame:
    """A different model probability for every match, so a mismatched join shows."""
    frame = hand_table()
    k = np.arange(4, dtype="float64")
    p_1x2 = np.c_[0.2 + 0.1 * k, np.full(4, 0.3), 0.5 - 0.1 * k]
    p_ou = np.c_[0.4 + 0.05 * k, 0.6 - 0.05 * k]
    return model_long(frame, p_1x2, p_ou)


def expected_row(
    table: pd.DataFrame, match: str, market: str, selection: str
) -> pd.Series:
    chosen = table[
        (table["match_id"] == match)
        & (table["market_id"] == market)
        & (table["selection"] == selection)
    ]
    assert len(chosen) == 1, (match, market, selection)
    return chosen.iloc[0]


def offer_table(
    n_batches: int,
    per_batch: int,
    *,
    odds: float = 3.0,
    p_reference: float = 0.5,
    probability: float = 0.5,
    p_close: float = 0.5,
) -> pd.DataFrame:
    """Offers with one 1X2 home price per match, ``per_batch`` matches a week."""
    rows = []
    for week in range(n_batches):
        friday = day("2024-01-05") + pd.Timedelta(7 * week, unit="D")
        for k in range(per_batch):
            rows.append(
                {
                    "match_id": f"w{week}m{k}",
                    "bookmaker": "b365",
                    "market_id": "1X2",
                    "selection": "home",
                    "odds": odds,
                    "probability": probability,
                    "push_probability": 0.0,
                    "p_reference": p_reference,
                    "p_close": p_close,
                    "season": 2023,
                    "date": friday + pd.Timedelta(1, unit="D"),
                    "batch": friday,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Weeks
# --------------------------------------------------------------------------- #
def test_a_match_belongs_to_the_coupon_of_the_friday_on_or_before_it() -> None:
    dates = pd.Series(
        pd.to_datetime(
            [
                "2024-01-05",  # Friday
                "2024-01-06",
                "2024-01-07",
                "2024-01-08",  # Monday
                "2024-01-11",  # Thursday
                "2024-01-12",  # the next Friday
            ]
        )
    )
    expected = ["2024-01-05"] * 5 + ["2024-01-12"]
    assert batch_start(dates).dt.strftime("%Y-%m-%d").tolist() == expected


def test_batches_ignore_the_time_of_day_and_accept_text_dates() -> None:
    late = pd.Series(["2024-01-06 20:45", "2024-01-05 00:00", "2024-01-05 23:59"])
    assert batch_start(late).tolist() == [day("2024-01-05")] * 3


def test_every_batch_is_a_friday_within_a_week_of_its_match() -> None:
    dates = pd.Series(pd.date_range("2023-08-01", periods=400, freq="D"))
    batches = batch_start(dates)
    assert (batches.dt.weekday == 4).all()
    gap = (dates.dt.normalize() - batches).dt.days
    assert gap.between(0, 6).all()
    assert set(gap) == set(range(7))


def test_a_missing_date_has_no_batch() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-06", None]))
    batches = batch_start(dates)
    assert batches.iloc[0] == day("2024-01-05")
    assert pd.isna(batches.iloc[1])


# --------------------------------------------------------------------------- #
# What the model thought, without hindsight
# --------------------------------------------------------------------------- #
def oof_frame(
    seasons: tuple[int, ...] = (2022, 2023, 2024), per_season: int = 300, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward predictions and the results that followed them."""
    rng = np.random.default_rng(seed)
    n = per_season * len(seasons)
    lam, mu = 1.4 * np.exp(rng.normal(0, 0.25, n)), 1.1 * np.exp(rng.normal(0, 0.25, n))
    oof = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(n)],
            "season": np.repeat(seasons, per_season),
            "pred_home_xg": lam,
            "pred_away_xg": mu,
        }
    )
    results = pd.DataFrame(
        {
            "match_id": oof["match_id"],
            "home_goals": rng.poisson(lam * 1.05),
            "away_goals": rng.poisson(mu * 0.95),
        }
    )
    return oof, results


def probabilities_of(
    oof: pd.DataFrame, results: pd.DataFrame, **kwargs: object
) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings: dict[str, object] = {"first_season": 2023, "min_calibration_matches": 100}
    settings.update(kwargs)
    return walk_forward_probabilities(oof, results, **settings)  # type: ignore[arg-type]


def test_each_season_is_calibrated_on_the_seasons_before_it() -> None:
    oof, results = oof_frame()
    probabilities, calibrations = probabilities_of(oof, results)
    assert calibrations["season"].tolist() == [2023, 2024]
    assert calibrations["calibrated_on"].tolist() == [300, 600]
    assert calibrations["matches"].tolist() == [300, 300]
    merged = oof.merge(results, on="match_id")
    for season in (2023, 2024):
        past = merged[merged["season"] < season]
        fitted = fit_calibration(
            past["pred_home_xg"].to_numpy(),
            past["pred_away_xg"].to_numpy(),
            past["home_goals"].to_numpy(dtype="float64"),
            past["away_goals"].to_numpy(dtype="float64"),
        )
        row = calibrations[calibrations["season"] == season].iloc[0]
        assert row["scale_home"] == pytest.approx(fitted.scale_home)
        assert row["scale_away"] == pytest.approx(fitted.scale_away)
        assert row["rho"] == pytest.approx(fitted.rho)
        block = merged[merged["season"] == season]
        rates = pd.DataFrame(
            {
                "lam_home": fitted.scale_home * block["pred_home_xg"].to_numpy(),
                "lam_away": fitted.scale_away * block["pred_away_xg"].to_numpy(),
            }
        )
        live = fixture_markets(rates, fitted.rho, match_ids=block["match_id"].tolist())
        ours = probabilities[probabilities["match_id"].isin(block["match_id"])]
        both = ours.merge(live, on=KEY, suffixes=("", "_live"))
        assert len(both) == len(ours) == 5 * len(block)
        np.testing.assert_allclose(both["probability"], both["probability_live"])
        np.testing.assert_allclose(
            both["push_probability"], both["push_probability_live"]
        )


def test_probabilities_cover_every_price_and_add_up() -> None:
    oof, results = oof_frame()
    probabilities, _ = probabilities_of(oof, results)
    assert len(probabilities) == 5 * 600
    assert set(zip(probabilities["market_id"], probabilities["selection"])) == {
        ("1X2", "home"),
        ("1X2", "draw"),
        ("1X2", "away"),
        ("OU:2.5", "over"),
        ("OU:2.5", "under"),
    }
    sums = probabilities.groupby(["match_id", "market_id"])["probability"].sum()
    np.testing.assert_allclose(sums, 1.0)
    assert (probabilities["push_probability"] == 0.0).all()
    assert probabilities["probability"].between(0.0, 1.0).all()


def test_a_season_never_leaks_into_its_own_calibration() -> None:
    oof, results = oof_frame()
    base, _ = probabilities_of(oof, results)
    ids = {s: set(oof.loc[oof["season"] == s, "match_id"]) for s in (2022, 2023, 2024)}

    def after_changing(season: int) -> pd.DataFrame:
        changed = results.copy()
        rows = changed["match_id"].isin(ids[season])
        changed.loc[rows, "home_goals"] += 3
        changed.loc[rows, "away_goals"] += 1
        return probabilities_of(oof, changed)[0]

    def same(left: pd.DataFrame, right: pd.DataFrame, season: int) -> bool:
        rows = left["match_id"].isin(ids[season])
        return bool(
            np.array_equal(
                left.loc[rows, "probability"], right.loc[rows, "probability"]
            )
        )

    # the goals of a season price nothing from that season or the ones before it
    changed_2024 = after_changing(2024)
    assert same(base, changed_2024, 2023) and same(base, changed_2024, 2024)
    changed_2023 = after_changing(2023)
    assert same(base, changed_2023, 2023)
    assert not same(base, changed_2023, 2024)
    changed_2022 = after_changing(2022)
    assert not same(base, changed_2022, 2023)
    assert not same(base, changed_2022, 2024)


def test_a_seasons_own_predictions_do_not_move_its_calibration() -> None:
    oof, results = oof_frame()
    _, base = probabilities_of(oof, results)
    changed = oof.copy()
    changed.loc[changed["season"] == 2024, "pred_home_xg"] *= 2.0
    _, calibrations = probabilities_of(changed, results)
    assert calibrations["scale_home"].tolist() == base["scale_home"].tolist()
    assert calibrations["rho"].tolist() == base["rho"].tolist()


def test_matches_without_a_result_or_a_prediction_are_left_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    oof, results = oof_frame()
    oof.loc[oof["match_id"] == "m400", "pred_home_xg"] = np.nan
    results.loc[results["match_id"] == "m401", "home_goals"] = np.nan
    with caplog.at_level(logging.WARNING):
        probabilities, calibrations = probabilities_of(oof, results)
    assert "2 predicted matches have no result or no prediction" in caplog.text
    assert not probabilities["match_id"].isin({"m400", "m401"}).any()
    assert calibrations["matches"].tolist() == [298, 300]  # m400 and m401 are 2023
    assert calibrations["calibrated_on"].tolist() == [300, 598]


def test_a_result_missing_from_the_table_counts_as_no_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    oof, results = oof_frame()
    with caplog.at_level(logging.WARNING):
        probabilities, calibrations = probabilities_of(oof, results.iloc[:-5])
    assert "5 predicted matches have no result" in caplog.text
    assert calibrations["matches"].tolist() == [300, 295]
    assert len(probabilities) == 5 * 595


def test_a_calibration_may_rest_on_exactly_the_minimum() -> None:
    oof, results = oof_frame()
    _, calibrations = probabilities_of(oof, results, min_calibration_matches=300)
    assert calibrations["calibrated_on"].tolist() == [300, 600]
    with pytest.raises(ValueError, match="needs at least 301 earlier"):
        probabilities_of(oof, results, min_calibration_matches=301)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda oof, res: (oof.drop(columns="pred_away_xg"), res), "out-of-fold"),
        (lambda oof, res: (oof, res.drop(columns="home_goals")), "results is missing"),
        (lambda oof, res: (pd.concat([oof, oof.iloc[:1]]), res), "repeat a match"),
        (lambda oof, res: (oof, pd.concat([res, res.iloc[:1]])), "results repeats"),
    ],
)
def test_unusable_inputs_are_named(change: object, message: str) -> None:
    oof, results = oof_frame()
    bad_oof, bad_results = change(oof, results)  # type: ignore[operator]
    with pytest.raises(ValueError, match=message):
        probabilities_of(bad_oof, bad_results)


def test_a_season_without_enough_history_cannot_be_priced() -> None:
    oof, results = oof_frame()
    with pytest.raises(ValueError, match="needs at least 100 earlier .* found 0"):
        probabilities_of(oof, results, first_season=2022)
    with pytest.raises(ValueError, match="needs at least 5000 earlier"):
        probabilities_of(oof, results, min_calibration_matches=5000)
    with pytest.raises(
        ValueError, match="no walk-forward predictions from season 2030"
    ):
        probabilities_of(oof, results, first_season=2030)


@pytest.mark.parametrize("column", ["pred_home_xg", "pred_away_xg"])
def test_predictions_that_add_up_to_nothing_cannot_calibrate(column: str) -> None:
    oof, results = oof_frame()
    oof.loc[oof["season"] == 2022, column] = 0.0
    with pytest.raises(ValueError, match="no expected goals"):
        probabilities_of(oof, results)


def test_a_calibration_needs_five_hundred_earlier_matches_unless_told_otherwise() -> (
    None
):
    oof, results = oof_frame()  # 300 matches before 2023 and 600 before 2024
    with pytest.raises(ValueError, match="needs at least 500 earlier .* found 300"):
        walk_forward_probabilities(oof, results, first_season=2023)
    _, calibrations = walk_forward_probabilities(oof, results, first_season=2024)
    assert calibrations["calibrated_on"].tolist() == [600]
    assert bt.MIN_CALIBRATION_MATCHES == 500


def test_the_score_grid_has_the_requested_size() -> None:
    oof, results = oof_frame()
    small, calibrations = probabilities_of(oof, results, max_goals=6)
    large, _ = probabilities_of(oof, results)
    fitted = calibrations[calibrations["season"] == 2023].iloc[0]
    block = oof[oof["season"] == 2023]
    rates = pd.DataFrame(
        {
            "lam_home": fitted["scale_home"] * block["pred_home_xg"].to_numpy(),
            "lam_away": fitted["scale_away"] * block["pred_away_xg"].to_numpy(),
        }
    )
    live = fixture_markets(
        rates, fitted["rho"], match_ids=block["match_id"].tolist(), max_goals=6
    )
    ours = small[small["match_id"].isin(block["match_id"])]
    both = ours.merge(live, on=KEY, suffixes=("", "_live"))
    assert len(both) == len(ours) == 5 * len(block)
    np.testing.assert_allclose(both["probability"], both["probability_live"])
    # a wider grid really is a different (more exact) grid
    wide = large[large["match_id"].isin(block["match_id"])].merge(
        ours, on=KEY, suffixes=("_wide", "_small")
    )
    assert not np.allclose(
        wide["probability_wide"], wide["probability_small"], rtol=1e-6, atol=0.0
    )


# --------------------------------------------------------------------------- #
# What can be bet on, and at what price
# --------------------------------------------------------------------------- #
def test_the_universe_keeps_only_prices_the_reference_quoted_early_and_at_the_close() -> (
    None
):
    universe = build_universe(hand_table(), hand_model())
    markets = universe.groupby("match_id")["market_id"].agg(lambda m: sorted(set(m)))
    assert markets.to_dict() == {
        "a": ["1X2", "OU:2.5"],
        "b": ["OU:2.5"],  # no closing home price: the whole 1X2 quote is unusable
        "c": ["1X2"],  # no early over price
        "d": ["1X2", "OU:2.5"],
    }
    assert len(universe) == 5 + 2 + 3 + 5


def test_the_universe_carries_the_references_early_and_closing_probabilities() -> None:
    universe = build_universe(hand_table(), hand_model())
    early = {
        sel: expected_row(universe, "a", "1X2", sel)["p_reference"]
        for sel in ("home", "draw", "away")
    }
    assert early == pytest.approx({"home": 0.5, "draw": 0.25, "away": 0.25})
    close = {
        sel: expected_row(universe, "a", "1X2", sel)["p_close"]
        for sel in ("home", "draw", "away")
    }
    total = 1 / 1.8 + 1 / 4.0 + 1 / 4.0
    assert close == pytest.approx(
        {"home": 1 / 1.8 / total, "draw": 0.25 / total, "away": 0.25 / total}
    )
    over = expected_row(universe, "a", "OU:2.5", "over")
    assert over["p_reference"] == pytest.approx(0.5)  # 1.9 against 1.9
    assert over["p_close"] == pytest.approx((1 / 1.85) / (1 / 1.85 + 1 / 2.0))


@pytest.mark.parametrize("method", METHODS)
def test_the_universe_takes_the_margin_out_by_the_chosen_method(method: str) -> None:
    frame = hand_table()
    universe = build_universe(frame, hand_model(), "ps", method)
    quotes = {
        ("1X2", "early"): ["ps_h", "ps_d", "ps_a"],
        ("1X2", "close"): ["psc_h", "psc_d", "psc_a"],
        ("OU:2.5", "early"): ["ps_over25", "ps_under25"],
        ("OU:2.5", "close"): ["psc_over25", "psc_under25"],
    }
    names = {
        market_id: [name for name, _ in MARKETS[market_id]] for market_id in MARKETS
    }
    for match, match_id in ((3, "d"), (0, "a")):
        for (market_id, snapshot), columns in quotes.items():
            fair = remove_margin(
                frame.loc[match, columns].to_numpy(dtype=float), method
            )
            column = "p_reference" if snapshot == "early" else "p_close"
            for position, selection in enumerate(names[market_id]):
                row = expected_row(universe, match_id, market_id, selection)
                assert row[column] == pytest.approx(fair[position], rel=1e-12)


def test_other_methods_move_the_probability_off_the_long_shot() -> None:
    frame, model = hand_table(), hand_model()
    scaled = build_universe(frame, model)
    for method in ("power", "shin"):
        other = build_universe(frame, model, "ps", method)
        longest = expected_row(other, "d", "1X2", "away")  # priced at 6.0
        proportional = expected_row(scaled, "d", "1X2", "away")
        assert longest["p_reference"] < proportional["p_reference"] - 1e-4
        assert longest["p_close"] < proportional["p_close"] - 1e-4
        favourite = expected_row(other, "d", "1X2", "home")  # priced at 1.5
        plain = expected_row(scaled, "d", "1X2", "home")
        assert favourite["p_reference"] > plain["p_reference"] + 1e-4


def test_the_universe_is_proportional_unless_told_otherwise() -> None:
    explicit = build_universe(hand_table(), hand_model(), "ps", "proportional")
    pd.testing.assert_frame_equal(build_universe(hand_table(), hand_model()), explicit)
    assert bt.DEMARGIN == "proportional"


def test_an_unknown_method_cannot_build_a_universe() -> None:
    with pytest.raises(ValueError, match="unknown margin-removal method 'additive'"):
        build_universe(hand_table(), hand_model(), "ps", "additive")


def test_the_universe_pairs_each_selection_with_its_own_model_probability() -> None:
    universe = build_universe(hand_table(), hand_model())
    assert expected_row(universe, "a", "1X2", "home")["probability"] == pytest.approx(
        0.2
    )
    assert expected_row(universe, "c", "1X2", "home")["probability"] == pytest.approx(
        0.4
    )
    assert expected_row(universe, "c", "1X2", "away")["probability"] == pytest.approx(
        0.3
    )
    assert expected_row(universe, "d", "OU:2.5", "over")[
        "probability"
    ] == pytest.approx(0.55)
    assert expected_row(universe, "b", "OU:2.5", "under")[
        "probability"
    ] == pytest.approx(0.55)
    assert (universe["push_probability"] == 0.0).all()


def test_the_universe_dates_each_price_and_puts_it_in_a_weekly_coupon() -> None:
    universe = build_universe(hand_table(), hand_model())
    batches = universe.groupby("match_id")["batch"].first()
    assert batches.tolist() == [
        day("2024-01-05"),
        day("2024-01-05"),  # a Sunday
        day("2024-01-05"),  # a Thursday
        day("2024-01-12"),
    ]
    assert (universe["season"] == 2023).all()
    assert universe["date"].is_monotonic_increasing


def test_a_price_without_a_date_is_left_out(caplog: pytest.LogCaptureFixture) -> None:
    frame = hand_table()
    frame.loc[frame["match_id"] == "a", "date"] = pd.NaT
    with caplog.at_level(logging.WARNING):
        universe = build_universe(frame, hand_model())
    assert "5 prices have no match date" in caplog.text
    assert "a" not in set(universe["match_id"])
    assert universe["batch"].notna().all()


def test_text_dates_become_dates() -> None:
    frame = hand_table().assign(
        date=lambda table: table["date"].dt.strftime("%Y-%m-%d")
    )
    universe = build_universe(frame, hand_model())
    assert pd.api.types.is_datetime64_any_dtype(universe["date"])
    assert universe["date"].iloc[0] == day("2024-01-05")


def test_the_universe_is_sorted_by_date_whatever_the_order_of_its_inputs() -> None:
    # the ids run against the calendar, so only the date can put the rows in order
    ids = {"a": "d", "b": "c", "c": "b", "d": "a"}
    frame = hand_table().assign(match_id=lambda t: t["match_id"].map(ids))
    model = hand_model().assign(match_id=lambda t: t["match_id"].map(ids))
    frame = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    model = model.sample(frac=1.0, random_state=4).reset_index(drop=True)
    universe = build_universe(frame, model)
    order = ["date", "match_id", "market_id", "selection"]
    pd.testing.assert_frame_equal(
        universe, universe.sort_values(order).reset_index(drop=True)
    )
    assert universe["match_id"].iloc[0] == "d"  # the match of 5 January
    assert universe["match_id"].iloc[-1] == "a"  # the match of 12 January
    assert universe["date"].is_monotonic_increasing


def test_prices_without_a_model_probability_are_not_in_the_universe() -> None:
    model = hand_model()
    universe = build_universe(hand_table(), model[model["match_id"] != "a"])
    assert "a" not in set(universe["match_id"])
    assert set(universe["match_id"]) == {"b", "c", "d"}


def test_another_book_can_be_the_reference() -> None:
    frame = hand_table()
    frame["avg_h"] = [2.0, 2.9, 3.7, 1.6]
    frame["avg_d"] = [3.6, 2.9, 3.7, 3.9]
    frame["avg_a"] = [3.7, 2.9, 1.95, 5.5]
    frame["avgc_h"] = [1.9, 2.9, 3.6, 1.55]
    frame["avgc_d"] = [3.8, 2.9, 3.6, 3.9]
    frame["avgc_a"] = [3.9, 2.9, 2.0, 5.8]
    universe = build_universe(frame, hand_model(), "avg")
    assert set(universe["market_id"]) == {"1X2"}
    assert set(universe["match_id"]) == {"a", "b", "c", "d"}  # b is complete here
    early = 1 / 2.0 / (1 / 2.0 + 1 / 3.6 + 1 / 3.7)
    assert expected_row(universe, "a", "1X2", "home")["p_reference"] == pytest.approx(
        early
    )


def test_a_table_without_the_reference_book_has_an_empty_universe() -> None:
    frame = hand_table().drop(columns=["ps_h", "psc_h"])
    universe = build_universe(frame[["match_id", "season", "date"]], hand_model())
    assert universe.empty
    assert {"p_reference", "p_close", "batch"} <= set(universe.columns)


def test_offers_are_one_books_early_prices_inside_the_universe() -> None:
    frame = hand_table()
    universe = build_universe(frame, hand_model())
    offers = build_offers(frame, universe, "b365")
    keys = set(zip(offers["match_id"], offers["market_id"], offers["selection"]))
    assert len(offers) == 12  # a: 3 + 2, b: 2, c: 3, d: 2 (no Bet365 home price)
    assert {m for m, _, _ in keys} == {"a", "b", "c", "d"}
    assert ("b", "1X2", "home") not in keys  # outside the universe
    assert ("c", "OU:2.5", "over") not in keys
    assert ("d", "1X2", "home") not in keys  # Bet365 has no complete quote
    assert set(offers["bookmaker"]) == {"b365"}
    assert expected_row(offers, "a", "1X2", "home")["odds"] == 1.85
    assert expected_row(offers, "b", "OU:2.5", "under")["odds"] == 1.85
    assert expected_row(offers, "c", "1X2", "away")["odds"] == 1.8
    assert expected_row(offers, "d", "OU:2.5", "over")["odds"] == 1.7
    paired = expected_row(offers, "a", "1X2", "draw")
    assert paired["p_reference"] == pytest.approx(0.25)
    assert paired["probability"] == pytest.approx(0.3)
    assert offers["date"].is_monotonic_increasing


def test_offers_from_the_reference_book_itself_are_possible() -> None:
    frame = hand_table()
    universe = build_universe(frame, hand_model())
    offers = build_offers(frame, universe, "ps")
    assert len(offers) == len(universe)
    assert (offers["bookmaker"] == "ps").all()


def test_coverage_counts_the_matches_with_a_complete_quote_in_each_market() -> None:
    universe = build_universe(hand_table(), hand_model())
    calibrations = pd.DataFrame({"season": [2023, 2024], "matches": [4, 2]})
    coverage = bt._coverage(universe, calibrations)
    assert coverage.columns.tolist() == ["season", "matches", "1X2", "OU:2.5"]
    # a, c and d have a complete 1X2 quote; a, b and d a complete over/under one
    assert coverage.to_numpy().tolist() == [[2023, 4, 3, 3], [2024, 2, 0, 0]]
    without_totals = hand_table().drop(
        columns=[c for c in hand_table().columns if "25" in c]
    )
    only_1x2 = bt._coverage(build_universe(without_totals, hand_model()), calibrations)
    assert only_1x2.to_numpy().tolist() == [[2023, 4, 3, 0], [2024, 2, 0, 0]]


# --------------------------------------------------------------------------- #
# Placing bets
# --------------------------------------------------------------------------- #
def test_the_engine_sees_the_price_and_the_probabilities_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []
    real = bt.find_value_bets

    def spy(offers: pd.DataFrame, config: ValueConfig) -> pd.DataFrame:
        seen.append(list(offers.columns))
        return real(offers, config)

    monkeypatch.setattr(bt, "find_value_bets", spy)
    place_bets(offer_table(3, 2), ValueConfig(model_weight=0.0))
    assert len(seen) == 3
    assert all(columns == list(bt.DECISION_COLUMNS) for columns in seen)
    assert "p_close" not in bt.DECISION_COLUMNS
    assert "season" not in bt.DECISION_COLUMNS


def test_each_weekly_coupon_has_its_own_exposure_cap() -> None:
    # edge 50% at odds 3: a quarter Kelly is far above the 2% cap, so every
    # coupon of 12 bets is scaled down to the 10% weekly cap, 100 on a bankroll
    # of 1,000. One run over both weeks would be capped at 100 in all.
    offers = offer_table(2, 12, odds=3.0, p_reference=0.5)
    bets = place_bets(offers, ValueConfig(model_weight=0.0))
    assert len(bets) == 24
    assert bets.groupby("batch")["stake"].sum().tolist() == pytest.approx(
        [100.0, 100.0]
    )
    assert bets["stake"].tolist() == pytest.approx([100.0 / 12] * 24)
    assert bets["stake_fraction"].tolist() == pytest.approx([0.1 / 12] * 24)


def test_bets_come_with_the_columns_the_measuring_needs() -> None:
    offers = offer_table(2, 2, odds=2.5, p_reference=0.5, p_close=0.47)
    bets = place_bets(offers, ValueConfig(model_weight=0.0))
    assert list(bets.columns) == list(BET_COLUMNS)
    assert (bets["p_close"] == 0.47).all()
    assert (bets["season"] == 2023).all()
    assert set(bets["batch"]) == {day("2024-01-05"), day("2024-01-12")}
    assert (bets["date"] == bets["batch"] + pd.Timedelta(1, unit="D")).all()
    assert (bets["bookmaker"] == "b365").all()
    assert bets["edge"].tolist() == pytest.approx([2.5 * 0.5 - 1.0] * 4)


@pytest.mark.parametrize(
    ("weight", "final"), [(0.0, 0.5), (0.25, 0.575), (0.5, 0.65), (1.0, 0.8)]
)
def test_the_model_weight_blends_the_model_into_the_reference(
    weight: float, final: float
) -> None:
    offers = offer_table(1, 1, odds=2.5, p_reference=0.5, probability=0.8)
    bets = place_bets(offers, ValueConfig(model_weight=weight))
    assert bets["p_final"].tolist() == pytest.approx([final])
    assert bets["edge"].tolist() == pytest.approx([2.5 * final - 1.0])
    assert bets["probability"].tolist() == [0.8]
    assert bets["p_reference"].tolist() == [0.5]


def test_a_stake_is_the_capped_fraction_of_the_bankroll() -> None:
    offers = offer_table(1, 1, odds=2.0, p_reference=0.55)  # edge 10%, kelly 10%
    small = place_bets(offers, ValueConfig(model_weight=0.0, bankroll=1_000.0))
    large = place_bets(offers, ValueConfig(model_weight=0.0, bankroll=5_000.0))
    assert small["stake_fraction"].tolist() == pytest.approx([0.025 * 0.8])  # 2% cap
    assert small["stake"].tolist() == pytest.approx([20.0])
    assert large["stake"].tolist() == pytest.approx([100.0])


def test_no_bets_is_an_empty_table_with_the_right_columns() -> None:
    nothing = place_bets(offer_table(2, 2).iloc[0:0], ValueConfig(model_weight=0.0))
    assert nothing.empty
    assert list(nothing.columns) == list(BET_COLUMNS)
    losing = place_bets(
        offer_table(2, 2, odds=1.5, p_reference=0.5), ValueConfig(model_weight=0.0)
    )
    assert losing.empty  # 1.5 against a fair 2.0 has no edge
    assert list(losing.columns) == list(BET_COLUMNS)
    assert losing["stake"].dtype == "float64"


def test_one_bet_per_match_even_when_several_prices_have_value() -> None:
    offers = pd.concat(
        [
            offer_table(1, 3, odds=3.0, p_reference=0.5),
            offer_table(1, 3, odds=3.0, p_reference=0.5).assign(selection="draw"),
        ],
        ignore_index=True,
    )
    bets = place_bets(offers, ValueConfig(model_weight=0.0))
    assert len(offers) == 6
    assert bets["match_id"].is_unique
    assert len(bets) == 3


# --------------------------------------------------------------------------- #
# Settling and measuring against the close
# --------------------------------------------------------------------------- #
def placed(rows: list[tuple[str, str, str, float, float, float]]) -> pd.DataFrame:
    """Bets from ``(match, market, selection, odds, stake, p_close)`` tuples."""
    return pd.DataFrame(
        [
            {
                "match_id": match,
                "market_id": market,
                "selection": selection,
                "odds": odds,
                "stake": stake,
                "p_close": p_close,
            }
            for match, market, selection, odds, stake, p_close in rows
        ]
    )


def scores(table: dict[str, tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": list(table),
            "home_goals": [home for home, _ in table.values()],
            "away_goals": [away for _, away in table.values()],
        }
    )


def test_settling_adds_the_outcome_the_money_and_the_closing_line_value() -> None:
    bets = placed(
        [
            ("m1", "1X2", "home", 2.0, 10.0, 0.55),
            ("m2", "1X2", "draw", 3.5, 20.0, 0.30),
            ("m3", "OU:2.5", "over", 1.8, 5.0, 0.50),
            ("m4", "OU:2.5", "under", 2.1, 8.0, 0.52),
        ]
    )
    results = scores({"m1": (2, 1), "m2": (1, 0), "m3": (0, 0), "m4": (1, 1)})
    settled = settle_bets(bets, results)
    assert settled["won"].tolist() == [True, False, False, True]
    assert settled["profit"].tolist() == pytest.approx([1.0, -1.0, -1.0, 1.1])
    assert settled["pnl"].tolist() == pytest.approx([10.0, -20.0, -5.0, 8.8])
    assert settled["clv"].tolist() == pytest.approx([0.10, 0.05, -0.10, 0.092])
    assert "settled" not in settled.columns
    assert not settled["push"].any()


def test_a_bet_on_a_match_without_a_score_is_dropped_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bets = placed(
        [
            ("m1", "1X2", "home", 2.0, 10.0, 0.55),
            ("m9", "1X2", "home", 2.0, 10.0, 0.55),
        ]
    )
    with caplog.at_level(logging.WARNING):
        settled = settle_bets(bets, scores({"m1": (2, 1)}))
    assert settled["match_id"].tolist() == ["m1"]
    assert "1 bets cannot be settled" in caplog.text


def test_closing_line_value_is_odds_times_closing_probability_minus_one() -> None:
    bets = placed([("m1", "1X2", "away", 4.0, 1.0, 0.30)])
    settled = settle_bets(bets, scores({"m1": (0, 0)}))
    assert settled["clv"].tolist() == pytest.approx([0.2])  # 4.0 * 0.30 - 1


def test_the_yardstick_is_one_unit_on_every_price() -> None:
    frame = hand_table()
    universe = build_universe(frame, hand_model())
    offers = build_offers(frame, universe, "b365")
    table = flat_baseline(offers, frame[["match_id", "home_goals", "away_goals"]])
    assert len(table) == len(offers)
    assert (table["stake"] == 1.0).all()
    assert table["pnl"].tolist() == table["profit"].tolist()
    assert table["p_final"].tolist() == table["p_reference"].tolist()
    np.testing.assert_allclose(
        table["edge"], table["odds"] * table["p_reference"] - 1.0
    )
    np.testing.assert_allclose(table["clv"], table["odds"] * table["p_close"] - 1.0)
    home = expected_row(table, "a", "1X2", "home")  # a ended 2-1 and Bet365 paid 1.85
    assert home["won"] and home["pnl"] == pytest.approx(0.85)
    draw = expected_row(table, "a", "1X2", "draw")
    assert not draw["won"] and draw["pnl"] == -1.0
    assert expected_row(table, "a", "OU:2.5", "over")["won"]  # 3 goals


# --------------------------------------------------------------------------- #
# Intervals and risk
# --------------------------------------------------------------------------- #
def test_the_interval_of_identical_weeks_is_a_point() -> None:
    low, high = cluster_bootstrap(np.full(20, 3.0), np.full(20, 40.0))
    assert low == pytest.approx(0.075)
    assert high == pytest.approx(0.075)


def test_resampling_two_weeks_reaches_the_extreme_ratios() -> None:
    # Weeks with ratios 1 and 3: a resample is 1, 2 (either order) or 3, the
    # last two quartiles of the draws, so the interval is exactly [1, 3].
    low, high = cluster_bootstrap(np.array([1.0, 3.0]), np.array([1.0, 1.0]))
    assert (low, high) == (1.0, 3.0)
    narrow = cluster_bootstrap(np.array([1.0, 3.0]), np.array([1.0, 1.0]), level=0.4)
    assert narrow == (2.0, 2.0)  # the middle 40% of the draws: the mixed pair


def test_the_interval_is_around_the_ratio_and_depends_on_the_data() -> None:
    top = np.arange(1.0, 41.0)
    bottom = np.full(40, 10.0)
    low, high = cluster_bootstrap(top, bottom, seed=5)
    assert low < top.sum() / bottom.sum() < high
    assert 1.0 < low and high < 3.5  # ratio of weeks 0.1 to 4.0 averaging 2.05
    wide = cluster_bootstrap(top, bottom, level=0.99, seed=5)
    assert wide[0] < low and wide[1] > high
    assert cluster_bootstrap(top, bottom, seed=5) == (low, high)  # reproducible
    assert cluster_bootstrap(top, bottom, seed=6) != (low, high)


def test_the_interval_is_the_percentile_of_the_resampled_ratios() -> None:
    top = np.array([4.0, -2.0, 5.0, 1.0, 3.0, 0.5, 2.0])
    bottom = np.array([10.0, 8.0, 12.0, 5.0, 9.0, 7.0, 6.0])
    picks = np.random.default_rng(9).integers(0, 7, size=(400, 7))
    ratios = top[picks].sum(axis=1) / bottom[picks].sum(axis=1)
    for level in (0.95, 0.8):
        tail = (1.0 - level) / 2.0
        expected = np.quantile(ratios, [tail, 1.0 - tail])
        got = cluster_bootstrap(top, bottom, n_resamples=400, level=level, seed=9)
        assert got[0] < got[1]
        assert got == pytest.approx(tuple(expected))


def test_weeks_with_nothing_staked_widen_the_interval_not_the_estimate() -> None:
    top = np.array([4.0, -2.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    bottom = np.array([10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    dense = cluster_bootstrap(top[:4], bottom[:4], seed=1)
    padded = cluster_bootstrap(top, bottom, seed=1)
    assert dense[0] < 0.2 < dense[1]
    assert padded != dense
    assert np.isfinite(
        padded
    ).all()  # resamples that skip every betting week are dropped


def test_without_any_stake_there_is_no_interval() -> None:
    low, high = cluster_bootstrap(np.zeros(5), np.zeros(5))
    assert np.isnan(low) and np.isnan(high)


@pytest.mark.parametrize(
    ("top", "bottom", "kwargs", "message"),
    [
        (np.array([]), np.array([]), {}, "non-empty"),
        (np.ones(3), np.ones(4), {}, "equal length"),
        (np.ones((2, 2)), np.ones((2, 2)), {}, "one-dimensional"),
        (np.ones(3), np.ones(3), {"level": 1.0}, "level"),
        (np.ones(3), np.ones(3), {"level": 0.0}, "level"),
        (np.ones(3), np.ones(3), {"n_resamples": 0}, "n_resamples"),
    ],
)
def test_the_interval_rejects_unusable_input(
    top: np.ndarray, bottom: np.ndarray, kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        cluster_bootstrap(top, bottom, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("budget", [7 * 30, 10])  # seven resamples a block, or one
def test_the_interval_is_the_same_however_the_resamples_are_blocked(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    top, bottom = np.arange(1.0, 31.0), np.full(30, 5.0)
    whole = cluster_bootstrap(top, bottom, n_resamples=500, seed=2)
    monkeypatch.setattr(bt, "BOOTSTRAP_BUDGET", budget)
    blocked = cluster_bootstrap(top, bottom, n_resamples=500, seed=2)
    assert blocked[0] == pytest.approx(whole[0], abs=0.05)
    assert blocked[1] == pytest.approx(whole[1], abs=0.05)


def test_a_single_resample_is_enough_for_an_interval() -> None:
    low, high = cluster_bootstrap(
        np.array([1.0, 3.0, 2.0]), np.ones(3), n_resamples=1, seed=3
    )
    assert np.isfinite(low) and low == high


def test_the_worst_fall_is_measured_from_the_running_peak() -> None:
    assert max_drawdown(np.array([1.0, -2.0, -1.0, 3.0, -4.0])) == 4.0
    assert max_drawdown(np.array([2.0, 3.0, 1.0])) == 0.0
    assert max_drawdown(np.array([-1.0, -1.0])) == 2.0  # the start is the peak
    assert max_drawdown(np.array([5.0, -2.0, 1.0, -2.0, 4.0])) == 3.0
    assert max_drawdown(np.array([])) == 0.0


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summary_bets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "market_id": ["1X2", "OU:2.5", "1X2", "1X2"],
            "batch": pd.to_datetime(["2024-01-05"] * 2 + ["2024-01-12"] * 2),
            "stake": [10.0, 20.0, 10.0, 40.0],
            "odds": [2.0, 3.0, 4.0, 1.5],
            "won": [True, False, False, True],
            "pnl": [10.0, -20.0, -10.0, 20.0],
            "clv": [0.10, -0.05, 0.02, -0.01],
            "p_final": [0.55, 0.38, 0.42, 0.70],
            "edge": [0.10, 0.14, 0.05, 0.05],
        }
    )


def test_summaries_are_exact_on_a_hand_table() -> None:
    row = summarise(summary_bets()).iloc[0]
    assert row["bets"] == 4
    assert row["staked"] == 80.0
    assert row["profit"] == 0.0
    assert row["roi"] == 0.0
    assert row["clv"] == pytest.approx(-0.2 / 80.0)  # stake-weighted
    assert row["clv_mean"] == pytest.approx(0.015)  # every bet counts the same
    assert row["beat_close"] == 0.5
    assert row["hit_rate"] == 0.5
    assert row["expected_hit_rate"] == pytest.approx(0.5125)
    assert row["mean_odds"] == pytest.approx(2.625)  # the median would be 2.5
    assert row["mean_edge"] == pytest.approx(0.085)
    assert row["max_drawdown"] == 10.0  # -10 after the first week, back to 0


def test_summaries_can_be_grouped() -> None:
    by_season = summarise(summary_bets(), ["season"])
    assert by_season["season"].tolist() == [2023, 2024]
    assert by_season["bets"].tolist() == [2, 2]
    assert by_season["staked"].tolist() == [30.0, 50.0]
    assert by_season["profit"].tolist() == [-10.0, 10.0]
    assert by_season["roi"].tolist() == pytest.approx([-1 / 3, 0.2])
    by_both = summarise(summary_bets(), ["season", "market_id"])
    assert by_both[["season", "market_id", "bets"]].values.tolist() == [
        [2023, "1X2", 1],
        [2023, "OU:2.5", 1],
        [2024, "1X2", 2],
    ]
    assert by_both["bets"].sum() == 4


def test_a_group_without_a_stake_has_no_return() -> None:
    bets = summary_bets().assign(stake=0.0, pnl=0.0)
    row = summarise(bets).iloc[0]
    assert np.isnan(row["roi"]) and np.isnan(row["clv"])
    assert row["bets"] == 4 and row["staked"] == 0.0
    assert row["clv_mean"] == pytest.approx(0.015)


def test_a_bet_that_only_matches_the_close_did_not_beat_it() -> None:
    bets = summary_bets().iloc[:2].assign(clv=[0.0, 0.1])
    assert summarise(bets).iloc[0]["beat_close"] == 0.5


def test_the_order_of_the_bets_does_not_change_the_summary() -> None:
    bets = summary_bets()
    shuffled = bets.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(summarise(shuffled), summarise(bets))
    assert summarise(shuffled).iloc[0]["max_drawdown"] == 10.0


def test_the_worst_fall_follows_the_weeks_not_the_order_of_the_rows() -> None:
    weeks = pd.to_datetime(["2024-01-19", "2024-01-05", "2024-01-26", "2024-01-12"])
    bets = summary_bets().assign(batch=weeks, pnl=[3.0, 10.0, -3.0, -10.0])
    # week by week: +10, -10, +3, -3 (the rows come in another order)
    assert summarise(bets).iloc[0]["max_drawdown"] == 10.0


def test_a_summary_row_has_intervals_for_profit_and_for_closing_line_value() -> None:
    rng = np.random.default_rng(4)
    weeks = pd.Index(pd.date_range("2024-01-05", periods=10, freq="7D"))
    n = 40
    bets = pd.DataFrame(
        {
            "batch": rng.choice(weeks[:8], n),  # no bets in the last two weeks
            "stake": rng.uniform(5.0, 20.0, n),
            "odds": rng.uniform(1.5, 4.0, n),
            "won": rng.random(n) < 0.4,
            "pnl": rng.normal(0.0, 15.0, n),
            "clv": rng.normal(0.01, 0.05, n),
            "p_final": rng.uniform(0.2, 0.6, n),
            "edge": rng.uniform(0.04, 0.2, n),
        }
    )
    row = bt._summary_row(
        bets, weeks, book="b365", weight=0.25, n_resamples=300, seed=5
    )
    stake, pnl, weighted = bt._cluster_totals(bets, weeks)
    roi = cluster_bootstrap(pnl, stake, n_resamples=300, seed=5)
    clv = cluster_bootstrap(weighted, stake, n_resamples=300, seed=5)
    assert (row["roi_low"], row["roi_high"]) == roi
    assert (row["clv_low"], row["clv_high"]) == clv
    assert roi != clv
    assert (row["bookmaker"], row["strategy"], row["weight"]) == (
        "b365",
        "weight 0.25",
        0.25,
    )
    assert row["bets"] == n
    assert row["roi"] == pytest.approx(bets["pnl"].sum() / bets["stake"].sum())


def test_a_missing_group_key_is_its_own_group() -> None:
    bets = summary_bets().assign(season=[2023.0, np.nan, 2024.0, 2024.0])
    grouped = summarise(bets, ["season"])
    assert len(grouped) == 3
    assert grouped["bets"].tolist() == [1, 2, 1]
    assert np.isnan(grouped["season"].iloc[-1])


def test_every_week_of_the_calendar_has_a_total_even_without_bets() -> None:
    weeks = pd.to_datetime(["2024-01-05", "2024-01-05", "2024-01-19", "2024-01-19"])
    bets = summary_bets().assign(batch=weeks)
    calendar = pd.Index(
        pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"])
    )
    stake, pnl, weighted = bt._cluster_totals(bets, calendar)
    assert stake.tolist() == [30.0, 0.0, 50.0, 0.0]
    assert pnl.tolist() == [-10.0, 0.0, 10.0, 0.0]
    assert weighted.tolist() == pytest.approx([0.0, 0.0, -0.2, 0.0])


def test_no_bets_summarise_to_an_empty_row_or_no_rows() -> None:
    nothing = summary_bets().iloc[0:0]
    row = summarise(nothing).iloc[0]
    assert row["bets"] == 0 and row["staked"] == 0.0 and row["profit"] == 0.0
    assert np.isnan(row["roi"]) and np.isnan(row["beat_close"])
    assert row["max_drawdown"] == 0.0
    grouped = summarise(nothing, ["season"])
    assert grouped.empty
    assert list(grouped.columns) == ["season", *bt.METRIC_COLUMNS]


def test_odds_ranges_include_their_upper_end() -> None:
    odds = pd.Series([1.0, 1.5, 2.0, 2.0001, 3.0, 3.01, 5.0, 5.01, 50.0, np.nan])
    assert odds_range(odds).dtype == "object"  # plain labels: no empty groups appear
    labels = odds_range(odds).tolist()
    assert labels[:9] == [
        "1.0-2.0",
        "1.0-2.0",
        "1.0-2.0",
        "2.0-3.0",
        "2.0-3.0",
        "3.0-5.0",
        "3.0-5.0",
        "5.0+",
        "5.0+",
    ]
    assert pd.isna(labels[9])


def test_strategies_are_named_by_their_weight() -> None:
    assert strategy_label(0.0) == "weight 0.00"
    assert strategy_label(0.25) == "weight 0.25"
    assert strategy_label(float("nan")) == BASELINE == "every price"


def test_books_are_named_for_people_and_unknown_codes_are_kept() -> None:
    assert bt._book_name("b365") == "Bet365"
    assert bt._book_name("zzz") == "zzz"


def test_the_command_defaults_are_the_documented_ones() -> None:
    args = bt._parse_args([])
    assert args.books == ["b365", "avg", "max", "ps"]
    assert args.weights == [0.0, 0.25, 0.5, 1.0]
    assert args.reference == "ps"
    assert args.demargin == "proportional"
    assert args.first_season == 2023
    assert args.min_edge == ValueConfig().min_edge > 0.0
    assert args.resamples == 2_000
    assert args.seed == 0
    assert args.enriched == bt.ENRICHED_OUTPUT and args.oof == bt.OOF_OUTPUT
    assert args.output_dir == BACKTEST_DIR


# --------------------------------------------------------------------------- #
# A market that learns
# --------------------------------------------------------------------------- #
def test_price_shopping_earns_the_closing_line_value_it_claims(
    shopped: pd.DataFrame,
) -> None:
    assert len(shopped) > 250
    # Pinnacle's early price is right on average, so a bet's closing-line value is
    # its edge over that price, whatever the soft book's quirks were.
    assert shopped["clv"].mean() == pytest.approx(shopped["edge"].mean(), abs=0.015)
    assert shopped["clv"].mean() > 0.035  # at least the 4% minimum edge, on average
    assert (shopped["p_final"] == shopped["p_reference"]).all()  # weight 0
    assert (shopped["edge"] >= 0.04).all()
    assert (
        shopped["beat_close"] if "beat_close" in shopped else shopped["clv"] > 0
    ).mean() > 0.6


def test_a_bet_on_every_price_costs_the_bookmakers_margin(
    world: World, flat_model: pd.DataFrame
) -> None:
    universe = build_universe(world.frame, flat_model)
    offers = build_offers(world.frame, universe, "b365")
    table = flat_baseline(offers, world.results)
    assert len(table) == 5 * len(world.frame)
    assert -0.062 < table["clv"].mean() < -0.045  # Bet365's 5.5% margin
    assert -0.09 < table["pnl"].mean() < -0.02  # profit is noisier than the line
    sharp = flat_baseline(build_offers(world.frame, universe, "ps"), world.results)
    assert -0.04 < sharp["clv"].mean() < -0.02  # Pinnacle's 3% margin


def test_a_model_with_no_information_trails_the_close(
    world: World,
) -> None:
    rng = np.random.default_rng(3)
    noisy = model_long(
        world.frame,
        tilt(world.early_1x2, 0.2, rng),
        tilt(world.early_ou, 0.2, rng),
    )
    bets = play(world.frame, noisy, book="b365", weight=1.0)
    assert len(bets) > 1_000
    weighted = (bets["stake"] * bets["clv"]).sum() / bets["stake"].sum()
    assert weighted < -0.02
    assert bets["clv"].mean() < -0.02
    assert (bets["edge"] > bets["clv"]).mean() > 0.8  # it expected more than it got


def test_a_model_that_knows_the_close_beats_it(world: World) -> None:
    oracle = model_long(world.frame, world.close_1x2, world.close_ou)
    bets = play(world.frame, oracle, book="b365", weight=1.0)
    assert len(bets) > 1_000
    # the engine's edge is odds x closing probability - 1, which is the definition
    np.testing.assert_allclose(bets["edge"], bets["clv"])
    assert (bets["clv"] >= 0.04 - 1e-9).all()
    assert bets["pnl"].sum() > 0.0


def test_the_sharp_book_cannot_be_beaten_at_its_own_price_without_a_model(
    world: World, flat_model: pd.DataFrame
) -> None:
    bets = play(world.frame, flat_model, book="ps", weight=0.0)
    assert bets.empty  # its own price has the margin as its edge


# --------------------------------------------------------------------------- #
# Nothing from the future reaches a decision
# --------------------------------------------------------------------------- #
def decisions(bets: pd.DataFrame) -> list[tuple[str, str, str, float]]:
    return list(
        zip(bets["match_id"], bets["market_id"], bets["selection"], bets["stake"])
    )


def test_changing_the_closing_prices_changes_the_measure_not_the_bets(
    world: World, flat_model: pd.DataFrame, shopped: pd.DataFrame
) -> None:
    rng = np.random.default_rng(5)
    frame = world.frame.copy()
    for column in [c for c in frame.columns if c.startswith("psc_")]:
        # stretch the margin over evens, so every price stays a usable quote
        frame[column] = 1.0 + (frame[column] - 1.0) * rng.uniform(0.8, 1.2, len(frame))
    again = play(frame, flat_model, book="b365", weight=0.0)
    assert decisions(again) == decisions(shopped)
    assert not np.allclose(again["clv"], shopped["clv"])
    assert not np.allclose(again["p_close"], shopped["p_close"])


def test_changing_the_results_changes_the_profit_not_the_bets(
    world: World, flat_model: pd.DataFrame, shopped: pd.DataFrame
) -> None:
    frame = world.frame.copy()
    frame["home_goals"] = (frame["home_goals"] + 2) % 5
    frame["away_goals"] = (frame["away_goals"] + 1) % 4
    again = play(frame, flat_model, book="b365", weight=0.0)
    assert decisions(again) == decisions(shopped)
    assert not np.allclose(again["pnl"], shopped["pnl"])
    np.testing.assert_allclose(again["clv"], shopped["clv"])  # the close is unchanged


def test_prices_of_later_seasons_do_not_move_earlier_decisions(
    world: World, flat_model: pd.DataFrame, shopped: pd.DataFrame
) -> None:
    rng = np.random.default_rng(8)
    frame = world.frame.copy()
    later = frame["season"] >= 2024
    for column in [c for c in frame.columns if c.startswith(("ps_", "b365_"))]:
        frame.loc[later, column] = frame.loc[later, column] * rng.uniform(
            0.8, 1.2, int(later.sum())
        )
    again = play(frame, flat_model, book="b365", weight=0.0)
    before = shopped[shopped["season"] < 2024]
    assert decisions(again[again["season"] < 2024]) == decisions(before)
    assert decisions(again) != decisions(shopped)


def test_only_the_reference_books_early_price_decides_at_weight_zero(
    world: World, flat_model: pd.DataFrame
) -> None:
    frame = world.frame.copy()
    for column in [c for c in frame.columns if c.startswith("avg_")]:
        frame[column] = 1.5  # nonsense prices from a book that is not used
    first = play(world.frame, flat_model, book="b365", weight=0.0)
    second = play(frame, flat_model, book="b365", weight=0.0)
    assert decisions(first) == decisions(second)


# --------------------------------------------------------------------------- #
# The whole run
# --------------------------------------------------------------------------- #
def oof_for(world: World, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(world.frame)
    return pd.DataFrame(
        {
            "match_id": world.frame["match_id"],
            "date": world.frame["date"],
            "season": world.frame["season"],
            "pred_home_xg": 1.4 * np.exp(rng.normal(0, 0.2, n)),
            "pred_away_xg": 1.1 * np.exp(rng.normal(0, 0.2, n)),
        }
    )


@pytest.fixture(scope="module")
def small_world() -> World:
    return make_world(per_season=1_000, seed=21)


@pytest.fixture(scope="module")
def result(small_world: World) -> BacktestResult:
    return run_backtest(
        small_world.frame,
        oof_for(small_world),
        books=("b365", "avg"),
        weights=(0.0, 1.0),
        n_resamples=200,
        min_calibration_matches=500,
    )


def test_one_row_per_strategy_and_a_yardstick_for_every_book(
    result: BacktestResult,
) -> None:
    summary = result.summary
    assert summary["bookmaker"].tolist() == ["b365"] * 3 + ["avg"] * 3
    assert (
        summary["strategy"].tolist()
        == [
            "every price",
            "weight 0.00",
            "weight 1.00",
        ]
        * 2
    )
    assert summary["weight"].isna().tolist() == [True, False, False] * 2
    assert list(summary.columns) == list(bt.SUMMARY_COLUMNS)
    assert result.reference_book == "ps"
    assert result.bankroll == 1_000.0


def test_bets_start_with_the_first_priced_season_and_carry_their_weight(
    result: BacktestResult,
) -> None:
    bets = result.bets
    assert bets["season"].min() == 2023
    assert set(bets["season"]) == {2023, 2024}
    assert set(bets["weight"]) == {0.0, 1.0}
    assert set(bets["bookmaker"]) == {"b365", "avg"}
    assert {"won", "pnl", "clv", "profit"} <= set(bets.columns)
    assert not bets.duplicated(["bookmaker", "weight", *KEY]).any()
    trusting_the_sharp_book = bets[bets["weight"] == 0.0]
    trusting_the_model = bets[bets["weight"] == 1.0]
    assert len(trusting_the_sharp_book) > 0 and len(trusting_the_model) > 0
    np.testing.assert_allclose(
        trusting_the_sharp_book["p_final"], trusting_the_sharp_book["p_reference"]
    )
    np.testing.assert_allclose(
        trusting_the_model["p_final"], trusting_the_model["probability"]
    )


def test_the_breakdowns_add_up_to_the_summary(result: BacktestResult) -> None:
    strategies = result.summary[result.summary["weight"].notna()]
    for table in (result.by_season, result.by_market, result.by_odds):
        totals = table.groupby(["bookmaker", "weight"])[
            ["bets", "staked", "profit"]
        ].sum()
        merged = strategies.set_index(["bookmaker", "weight"]).loc[
            totals.index, ["bets", "staked", "profit"]
        ]
        np.testing.assert_allclose(totals.to_numpy(), merged.to_numpy())
    assert set(result.by_season["season"]) <= {2023, 2024}
    assert set(result.by_market["market_id"]) <= {"1X2", "OU:2.5"}
    assert set(result.by_odds["odds_range"]) <= set(bt.ODDS_LABELS)


def test_every_odds_range_holds_the_bets_it_names(result: BacktestResult) -> None:
    bets = result.bets
    for label, (low, high) in {
        "1.0-2.0": (1.0, 2.0),
        "2.0-3.0": (2.0, 3.0),
        "3.0-5.0": (3.0, 5.0),
        "5.0+": (5.0, np.inf),
    }.items():
        inside = int(((bets["odds"] > low) & (bets["odds"] <= high)).sum())
        listed = int(
            result.by_odds.loc[result.by_odds["odds_range"] == label, "bets"].sum()
        )
        assert listed == inside


def test_the_summary_matches_the_bets(result: BacktestResult) -> None:
    for row in result.summary[result.summary["weight"].notna()].itertuples():
        mine = result.bets[
            (result.bets["bookmaker"] == row.bookmaker)
            & (result.bets["weight"] == row.weight)
        ]
        assert row.bets == len(mine)
        assert row.staked == pytest.approx(mine["stake"].sum())
        assert row.profit == pytest.approx(mine["pnl"].sum())
        if row.bets:
            assert row.clv == pytest.approx(
                (mine["stake"] * mine["clv"]).sum() / mine["stake"].sum()
            )
            assert row.roi_low <= row.roi <= row.roi_high
            assert row.clv_low <= row.clv <= row.clv_high


def test_the_yardstick_row_is_one_unit_on_every_price(
    result: BacktestResult, small_world: World
) -> None:
    row = result.summary.iloc[0]
    assert row["strategy"] == "every price"
    assert row["bets"] == 5 * 2 * 1_000  # two priced seasons, five prices a match
    assert row["staked"] == row["bets"]
    assert row["roi"] == pytest.approx(row["profit"] / row["bets"])
    assert -0.065 < row["clv"] < -0.04


def test_coverage_and_calibrations_describe_the_priced_seasons(
    result: BacktestResult,
) -> None:
    assert result.coverage.columns.tolist() == ["season", "matches", "1X2", "OU:2.5"]
    assert result.coverage["season"].tolist() == [2023, 2024]
    assert result.coverage["matches"].tolist() == [1_000, 1_000]
    assert result.coverage["1X2"].tolist() == [1_000, 1_000]
    assert result.calibrations["calibrated_on"].tolist() == [1_000, 2_000]


def test_a_run_is_reproducible(small_world: World, result: BacktestResult) -> None:
    again = run_backtest(
        small_world.frame,
        oof_for(small_world),
        books=("b365", "avg"),
        weights=(0.0, 1.0),
        n_resamples=200,
        min_calibration_matches=500,
    )
    pd.testing.assert_frame_equal(again.summary, result.summary)
    pd.testing.assert_frame_equal(again.bets, result.bets)


def test_repeated_books_and_weights_are_run_once(small_world: World) -> None:
    run = run_backtest(
        small_world.frame,
        oof_for(small_world),
        books=("b365", "b365"),
        weights=(1.0, 1.0, 0),
        n_resamples=50,
        min_calibration_matches=500,
    )
    assert run.summary["strategy"].tolist() == [
        "every price",
        "weight 1.00",
        "weight 0.00",
    ]


def test_the_betting_rules_are_passed_on(small_world: World) -> None:
    common = {
        "books": ("b365",),
        "weights": (1.0,),
        "n_resamples": 50,
        "min_calibration_matches": 500,
    }
    default = run_backtest(small_world.frame, oof_for(small_world), **common)  # type: ignore[arg-type]
    rich = run_backtest(
        small_world.frame,
        oof_for(small_world),
        config=ValueConfig(bankroll=5_000.0, min_edge=0.2),
        **common,  # type: ignore[arg-type]
    )
    assert rich.bankroll == 5_000.0
    assert 0 < rich.bets["stake"].sum()
    assert rich.bets["edge"].min() >= 0.2
    assert len(rich.bets) < len(default.bets)
    assert rich.bets["stake"].max() > default.bets["stake"].max()


def test_a_run_without_bets_still_reports(small_world: World) -> None:
    quiet = run_backtest(
        small_world.frame,
        oof_for(small_world),
        books=("b365",),
        weights=(0.0,),
        config=ValueConfig(min_edge=5.0),
        n_resamples=50,
        min_calibration_matches=500,
    )
    assert quiet.bets.empty
    row = quiet.summary.iloc[1]
    assert row["bets"] == 0 and row["staked"] == 0.0
    assert np.isnan(row["roi"]) and np.isnan(row["clv_low"])
    assert quiet.by_season.empty and quiet.by_market.empty and quiet.by_odds.empty
    assert "no bets" in " ".join(verdict(quiet))
    assert "Strategies" in format_backtest(quiet)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"books": ()}, "at least one bookmaker"),
        ({"weights": ()}, "at least one bookmaker"),
        ({"books": ("b365", "zzz")}, "unknown bookmakers"),
        ({"reference_book": "max"}, "the reference must be one of"),
        ({"weights": (1.5,)}, "model_weight"),
        ({"first_season": 2022}, "earlier predicted matches"),
        ({"first_season": 2040}, "no walk-forward predictions"),
    ],
)
def test_unusable_settings_are_named(
    small_world: World, kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_backtest(
            small_world.frame,
            oof_for(small_world),
            n_resamples=10,
            min_calibration_matches=500,
            **kwargs,  # type: ignore[arg-type]
        )


def test_a_match_table_that_repeats_a_match_is_rejected(small_world: World) -> None:
    frame = pd.concat([small_world.frame, small_world.frame.iloc[:1]])
    with pytest.raises(ValueError, match="the match table repeats a match"):
        run_backtest(frame, oof_for(small_world), n_resamples=10)


def test_the_calibration_minimum_is_passed_on(small_world: World) -> None:
    with pytest.raises(ValueError, match="needs at least 1500 earlier .* found 1000"):
        run_backtest(
            small_world.frame,
            oof_for(small_world),
            min_calibration_matches=1_500,
            n_resamples=10,
        )


def test_a_match_table_without_a_date_is_rejected(small_world: World) -> None:
    with pytest.raises(ValueError, match="the match table is missing columns"):
        run_backtest(
            small_world.frame.drop(columns="date"), oof_for(small_world), n_resamples=10
        )


def test_a_table_without_the_sharp_book_says_what_to_try(small_world: World) -> None:
    frame = small_world.frame.drop(
        columns=[c for c in small_world.frame.columns if c.startswith(("ps_", "psc_"))]
    )
    with pytest.raises(ValueError, match="complete early and closing Pinnacle quote"):
        run_backtest(
            frame, oof_for(small_world), n_resamples=10, min_calibration_matches=500
        )


def test_the_market_average_can_be_the_sharp_view(small_world: World) -> None:
    frame = small_world.frame.copy()
    for suffix in ("h", "d", "a", "over25", "under25"):
        frame[f"avgc_{suffix}"] = frame[f"psc_{suffix}"]
    run = run_backtest(
        frame,
        oof_for(small_world),
        books=("b365",),
        weights=(0.0,),
        reference_book="avg",
        n_resamples=50,
        min_calibration_matches=500,
    )
    assert run.reference_book == "avg"
    assert run.coverage["1X2"].tolist() == [1_000, 1_000]
    assert "Market average" not in format_backtest(run)
    assert "market average close" in format_backtest(run)


def test_the_margin_removal_method_is_checked_before_any_work(
    small_world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model was priced before the method was checked")

    monkeypatch.setattr(bt, "walk_forward_probabilities", never)
    with pytest.raises(ValueError, match="unknown margin-removal method 'additive'"):
        run_backtest(small_world.frame, oof_for(small_world), demargin="additive")


def test_the_backtest_uses_and_records_the_margin_removal_method(
    small_world: World,
) -> None:
    frame, oof = small_world.frame, oof_for(small_world)
    options = {
        "books": ("b365",),
        "weights": (0.0,),
        "n_resamples": 20,
        "min_calibration_matches": 500,
    }
    run = run_backtest(frame, oof, demargin="shin", **options)  # type: ignore[arg-type]
    assert run.demargin == "shin"
    assert len(run.bets) > 0
    early = fair_view(frame, "early", ("ps",), "shin")[[*KEY, "p_fair"]]
    close = fair_view(frame, "close", ("ps",), "shin")[[*KEY, "p_fair"]]
    joined = run.bets.merge(early, on=KEY).merge(
        close, on=KEY, suffixes=("_early", "_close")
    )
    assert len(joined) == len(run.bets)
    np.testing.assert_allclose(joined["p_reference"], joined["p_fair_early"])
    np.testing.assert_allclose(joined["p_close"], joined["p_fair_close"])
    scaled = fair_view(frame, "close", ("ps",))[[*KEY, "p_fair"]]
    against = joined.merge(scaled, on=KEY)  # the same selections, scaled by hand
    assert len(against) == len(joined)
    assert not np.allclose(against["p_close"], against["p_fair"], atol=1e-4)
    default = run_backtest(frame, oof, **options)  # type: ignore[arg-type]
    assert default.demargin == "proportional"


# --------------------------------------------------------------------------- #
# Reading and saving the result
# --------------------------------------------------------------------------- #
def test_the_report_shows_every_strategy_and_the_yardstick(
    result: BacktestResult,
) -> None:
    text = format_backtest(result)
    assert "Matches priced per season" in text
    assert "complete early and closing Pinnacle quote" in text
    assert "Calibration of the model" in text
    assert "Strategies, early prices, measured against the Pinnacle close" in text
    assert "margin taken out by proportional scaling" in text
    assert "What the bets looked like" in text
    assert "constant bankroll of 1,000" in text
    assert text.count("every price") >= 2 * 2  # two books, two tables
    assert "Bet365" in text and "market average" in text
    assert "weight 0.00" in text and "weight 1.00" in text
    assert "ROI 95% interval" in text and "CLV 95% interval" in text


@pytest.mark.parametrize("method", METHODS)
def test_the_report_names_how_the_margin_was_taken_out(method: str) -> None:
    text = format_backtest(replace(hand_result(), demargin=method))
    name = METHOD_NAMES[method]
    sentence = (
        f"close (margin taken out by {name}). 95% intervals resample whole weeks."
    )
    assert sentence in text


def test_a_method_without_a_readable_name_is_shown_by_its_code() -> None:
    text = format_backtest(replace(hand_result(), demargin="zzz"))
    assert "(margin taken out by zzz). 95% intervals" in text


def test_a_result_made_without_a_method_took_the_margin_out_proportionally() -> None:
    assert hand_result().demargin == "proportional"


def test_the_report_leaves_out_the_money_columns_for_the_yardstick(
    result: BacktestResult,
) -> None:
    text = format_backtest(result)
    strategies, diagnostics = text.split("What the bets looked like")
    for section in (strategies, diagnostics):
        rows = [
            line.split()
            for line in section.splitlines()
            if re.match(r"\s*(Bet365|market average) every price", line)
        ]
        assert len(rows) == 2  # one yardstick per book
    yardstick = next(
        line.split()
        for line in strategies.splitlines()
        if re.match(r"\s*Bet365 every price", line)
    )
    assert yardstick[:3] == ["Bet365", "every", "price"]
    assert yardstick[4] == "-"  # no amount staked: it is one unit a price
    money = next(
        line
        for line in diagnostics.splitlines()
        if re.match(r"\s*Bet365 every price", line)
    )
    assert money.split()[-1] == "-" and " - " in money  # no profit, no worst fall


def hand_result() -> BacktestResult:
    """A result with round numbers, so the report can be checked number by number."""
    nan = float("nan")

    def strategy(**fields: object) -> dict[str, object]:
        row: dict[str, object] = {name: nan for name in bt.SUMMARY_COLUMNS}
        row.update(bookmaker="b365", **fields)
        return row

    rows = [
        strategy(
            strategy="every price", weight=nan, bets=10, staked=10.0, profit=-0.5,
            roi=-0.05, roi_low=-0.12, roi_high=0.02,
            clv=-0.0512, clv_low=-0.06, clv_high=-0.04, clv_mean=-0.05,
            beat_close=0.2, hit_rate=0.4, expected_hit_rate=0.41,
            mean_odds=2.5, mean_edge=-0.05, max_drawdown=1.0,
        ),
        strategy(
            strategy="weight 0.25", weight=0.25, bets=20, staked=345.6, profit=12.4,
            roi=0.0359, roi_low=-0.08, roi_high=0.15,
            clv=0.021, clv_low=0.004, clv_high=0.038, clv_mean=0.02,
            beat_close=0.55, hit_rate=0.45, expected_hit_rate=0.43,
            mean_odds=2.31, mean_edge=0.067, max_drawdown=80.0,
        ),
        strategy(
            strategy="weight 1.00", weight=1.0, bets=1, staked=2500.0, profit=-1234.6,
            roi=-0.49384, roi_low=-0.49384, roi_high=-0.49384,
            clv=0.1, clv_low=0.1, clv_high=0.1, clv_mean=0.1,
            beat_close=1.0, hit_rate=0.0, expected_hit_rate=0.5,
            mean_odds=3.0, mean_edge=0.1, max_drawdown=2000.0,
        ),
        strategy(
            strategy="weight 0.00", weight=0.0, bets=0, staked=0.0, profit=0.0,
            max_drawdown=0.0,
        ),
    ]  # fmt: skip
    return BacktestResult(
        bets=pd.DataFrame(),
        summary=pd.DataFrame(rows, columns=list(bt.SUMMARY_COLUMNS)),
        by_season=pd.DataFrame(),
        by_market=pd.DataFrame(),
        by_odds=pd.DataFrame(),
        calibrations=pd.DataFrame(
            {
                "season": [2023],
                "calibrated_on": [1000],
                "matches": [500],
                "scale_home": [1.0512],
                "scale_away": [0.9871],
                "rho": [-0.0436],
            }
        ),
        coverage=pd.DataFrame(
            {"season": [2023], "matches": [500], "1X2": [497], "OU:2.5": [490]}
        ),
        reference_book="ps",
        bankroll=2_000.0,
    )


def tokens(section: str, label: str) -> list[str]:
    """The cells of the one line of ``section`` that starts with ``label``."""
    lines = [
        line.split() for line in section.splitlines() if line.strip().startswith(label)
    ]
    assert len(lines) == 1, (label, lines)
    return lines[0]


def test_the_report_prints_each_number_in_its_own_column() -> None:
    sections = format_backtest(hand_result()).split("\n\n")
    assert len(sections) == 4
    coverage, calibration, strategies, diagnostics = sections
    assert coverage.startswith("Matches priced per season")
    assert calibration.startswith("Calibration of the model")
    assert strategies.startswith("Strategies, early prices")
    assert diagnostics.startswith("What the bets looked like")
    assert "constant bankroll of 2,000" in diagnostics
    assert tokens(coverage, "2023") == ["2023", "500", "497", "490"]
    assert "fitted on" in calibration
    assert tokens(calibration, "2023") == [
        "2023",
        "1000",
        "500",
        "1.051",
        "0.987",
        "-0.044",
    ]
    assert tokens(strategies, "Bet365 weight 0.25") == [
        *("Bet365", "weight", "0.25", "20", "346"),
        *("+3.6", "[-8.0,", "+15.0]", "+2.1", "[+0.4,", "+3.8]", "55"),
    ]
    assert tokens(strategies, "Bet365 every price") == [
        *("Bet365", "every", "price", "10", "-"),  # no amount staked: a unit a price
        *("-5.0", "[-12.0,", "+2.0]", "-5.1", "[-6.0,", "-4.0]", "20"),
    ]
    assert tokens(strategies, "Bet365 weight 1.00") == [
        *("Bet365", "weight", "1.00", "1", "2,500"),
        *("-49.4", "[-49.4,", "-49.4]", "+10.0", "[+10.0,", "+10.0]", "100"),
    ]
    assert tokens(strategies, "Bet365 weight 0.00") == [
        *("Bet365", "weight", "0.00", "0", "0"),
        *("-", "-", "-", "-", "-"),
    ]
    assert tokens(diagnostics, "Bet365 weight 0.25") == [
        *("Bet365", "weight", "0.25", "45.0", "43.0", "2.31", "+6.7"),
        *("+12", "4.0"),  # worst fall: 80 on a bankroll of 2,000
    ]
    assert tokens(diagnostics, "Bet365 every price") == [
        *("Bet365", "every", "price", "40.0", "41.0", "2.50", "-5.0", "-", "-"),
    ]
    assert tokens(diagnostics, "Bet365 weight 1.00") == [
        *("Bet365", "weight", "1.00", "0.0", "50.0", "3.00", "+10.0"),
        *("-1,235", "100.0"),
    ]
    assert tokens(diagnostics, "Bet365 weight 0.00") == [
        *("Bet365", "weight", "0.00", "-", "-", "-", "-", "+0", "0.0"),
    ]


def test_the_verdict_reads_every_strategy_of_a_hand_result() -> None:
    mixed = (
        "Bet365, weight 0.25: 20 bets, CLV +2.1% (95%: [+0.4, +3.8]) beats the "
        "closing line; ROI +3.6% (95%: [-8.0, +15.0]), profit cannot be told "
        "apart from luck"
    )
    single = (
        "Bet365, weight 1.00: 1 bets, CLV +10.0% (95%: [+10.0, +10.0]) beats the "
        "closing line; ROI -49.4% (95%: [-49.4, -49.4]), the loss is unlikely to "
        "be bad luck"
    )
    assert verdict(hand_result()) == [mixed, single, "Bet365, weight 0.00: no bets"]


def test_an_interval_with_a_missing_end_is_not_shown() -> None:
    assert bt._interval(np.nan, 0.1) == "-"
    assert bt._interval(0.1, np.nan) == "-"
    assert bt._interval(-0.05, 0.1) == "[-5.0, +10.0]"
    assert bt._percent(np.nan) == "-"
    assert bt._percent(0.0512) == "5.1" and bt._percent(0.0512, signed=True) == "+5.1"


def test_the_verdict_has_a_line_for_each_strategy_but_not_the_yardstick(
    result: BacktestResult,
) -> None:
    lines = verdict(result)
    assert len(lines) == 4
    assert all("every price" not in line for line in lines)
    assert lines[0].startswith("Bet365, weight 0.00: ")
    assert lines[-1].startswith("market average, weight 1.00: ")
    for line in lines:
        assert "no bets" in line or ("CLV" in line and "ROI" in line)
    assert any("CLV" in line for line in lines)


@pytest.mark.parametrize(
    ("low", "high", "clv", "roi"),
    [
        (0.01, 0.03, "beats the closing line", "profit is unlikely to be luck"),
        (
            -0.03,
            -0.01,
            "trails the closing line",
            "the loss is unlikely to be bad luck",
        ),
        (-0.01, 0.02, "cannot be told apart", "cannot be told apart from luck"),
        (0.0, 0.02, "cannot be told apart", "cannot be told apart from luck"),
        (-0.02, 0.0, "cannot be told apart", "cannot be told apart from luck"),
        (np.nan, np.nan, "no interval", "no interval"),
        (np.nan, 0.02, "no interval", "no interval"),
        (0.01, np.nan, "no interval", "no interval"),
    ],
)
def test_intervals_are_read_in_plain_words(
    low: float, high: float, clv: str, roi: str
) -> None:
    assert clv in bt._clv_phrase(low, high)
    assert roi in bt._roi_phrase(low, high)


def test_the_verdict_names_the_numbers(result: BacktestResult) -> None:
    row = result.summary.iloc[1]
    line = verdict(result)[0]
    assert f"{row['bets']} bets" in line
    assert f"CLV {100 * row['clv']:+.1f}%" in line
    assert f"ROI {100 * row['roi']:+.1f}%" in line


def test_the_tables_are_saved_and_read_back(
    result: BacktestResult, tmp_path: Path
) -> None:
    written = save_backtest(result, tmp_path / "out")
    assert [p.name for p in written] == [
        "bets.parquet",
        "summary.csv",
        "by_season.csv",
        "by_market.csv",
        "by_odds.csv",
        "calibrations.csv",
        "coverage.csv",
    ]
    assert all(p.exists() for p in written)
    again = pd.read_parquet(tmp_path / "out" / "bets.parquet")
    assert len(again) == len(result.bets)
    np.testing.assert_allclose(again["clv"], result.bets["clv"])
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert summary["bets"].tolist() == result.summary["bets"].tolist()
    assert summary["strategy"].tolist() == result.summary["strategy"].tolist()
    for name in (
        "summary",
        "by_season",
        "by_market",
        "by_odds",
        "calibrations",
        "coverage",
    ):  # no stray index column: the files read back as the tables they came from
        saved = pd.read_csv(tmp_path / "out" / f"{name}.csv")
        assert saved.columns.tolist() == getattr(result, name).columns.tolist()
        assert len(saved) == len(getattr(result, name))
    assert BACKTEST_DIR.name == "backtest"


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
@pytest.fixture
def inputs(small_world: World, tmp_path: Path) -> tuple[Path, Path]:
    enriched, oof = tmp_path / "enriched.parquet", tmp_path / "oof.parquet"
    small_world.frame.to_parquet(enriched, index=False)
    oof_for(small_world).to_parquet(oof, index=False)
    return enriched, oof


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_reports_and_saves(
    inputs: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    out = tmp_path / "results"
    with caplog.at_level(logging.INFO):
        code = main(
            [
                "--enriched",
                str(enriched),
                "--oof",
                str(oof),
                "--books",
                "b365",
                "--weights",
                "0",
                "1",
                "--resamples",
                "50",
                "--output-dir",
                str(out),
            ]
        )
    assert code == 0
    assert (
        "Strategies, early prices, measured against the Pinnacle close" in caplog.text
    )
    assert "Bet365, weight 1.00:" in caplog.text
    assert "Wrote" in caplog.text
    assert (out / "summary.csv").exists() and (out / "bets.parquet").exists()
    summary = pd.read_csv(out / "summary.csv")
    assert summary["strategy"].tolist() == ["every price", "weight 0.00", "weight 1.00"]


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_honours_the_reference_the_season_and_the_edge(
    inputs: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    with caplog.at_level(logging.INFO):
        code = main(
            [
                "--enriched",
                str(enriched),
                "--oof",
                str(oof),
                "--books",
                "b365",
                "--weights",
                "1",
                "--first-season",
                "2024",
                "--min-edge",
                "5",
                "--resamples",
                "20",
                "--seed",
                "3",
                "--output-dir",
                str(tmp_path / "other"),
            ]
        )
    assert code == 0
    assert "Bet365, weight 1.00: no bets" in caplog.text
    coverage = pd.read_csv(tmp_path / "other" / "coverage.csv")
    assert coverage["season"].tolist() == [2024]


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_passes_the_seed_to_the_intervals(
    inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    enriched, oof = inputs
    lows = []
    for seed in ("1", "2"):
        out = tmp_path / f"seed{seed}"
        arguments = ["--enriched", str(enriched), "--oof", str(oof)]
        arguments += ["--books", "b365", "--weights", "1", "--resamples", "50"]
        arguments += ["--seed", seed, "--output-dir", str(out)]
        assert main(arguments) == 0
        lows.append(pd.read_csv(out / "summary.csv")["roi_low"].tolist())
    assert lows[0] != lows[1]


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_can_use_another_book_as_the_sharp_view(
    small_world: World, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frame = small_world.frame.copy()
    for suffix in ("h", "d", "a", "over25", "under25"):
        frame[f"avgc_{suffix}"] = frame[f"psc_{suffix}"]
    enriched, oof = tmp_path / "avg.parquet", tmp_path / "avg_oof.parquet"
    frame.to_parquet(enriched, index=False)
    oof_for(small_world).to_parquet(oof, index=False)
    arguments = ["--enriched", str(enriched), "--oof", str(oof)]
    arguments += ["--books", "b365", "--weights", "1", "--resamples", "20"]
    arguments += ["--reference", "avg", "--output-dir", str(tmp_path / "avg_out")]
    with caplog.at_level(logging.INFO):
        assert main(arguments) == 0
    assert "measured against the market average close" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_can_take_the_margin_out_another_way(
    inputs: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    arguments = ["--enriched", str(enriched), "--oof", str(oof), "--books", "b365"]
    arguments += ["--weights", "0", "--resamples", "20", "--demargin", "power"]
    arguments += ["--output-dir", str(tmp_path / "power")]
    with caplog.at_level(logging.INFO):
        assert main(arguments) == 0
    assert "margin taken out by the power method" in caplog.text
    bets = pd.read_parquet(tmp_path / "power" / "bets.parquet")
    close = fair_view(pd.read_parquet(enriched), "close", ("ps",), "power")
    joined = bets.merge(close, on=KEY)
    assert len(joined) == len(bets) > 0
    np.testing.assert_allclose(joined["p_close"], joined["p_fair"])


def test_the_command_only_accepts_the_known_methods(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for method in METHODS:
        assert bt._parse_args(["--demargin", method]).demargin == method
    with pytest.raises(SystemExit):
        bt._parse_args(["--demargin", "additive"])
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_names_the_step_to_run_when_an_input_is_missing(
    inputs: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    with caplog.at_level(logging.ERROR):
        assert (
            main(["--enriched", str(tmp_path / "no.parquet"), "--oof", str(oof)]) == 1
        )
    assert "python -m src.enrich" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert (
            main(["--enriched", str(enriched), "--oof", str(tmp_path / "no.parquet")])
            == 1
        )
    assert "python -m src.train" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_unusable_data(
    inputs: tuple[Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    pd.read_parquet(enriched).drop(columns="date").to_parquet(enriched, index=False)
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(enriched), "--oof", str(oof)]) == 1
    assert "Cannot run the backtest" in caplog.text
    assert "date" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_passes_every_option_on(
    inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def spy(
        enriched: pd.DataFrame, oof: pd.DataFrame, **kwargs: object
    ) -> BacktestResult:
        seen.update(kwargs, rows=(len(enriched), len(oof)))
        raise ValueError("stop here")

    monkeypatch.setattr(bt, "run_backtest", spy)
    enriched, oof = inputs
    options = ["--books", "avg", "max", "--weights", "0.5", "1", "--reference", "avg"]
    options += ["--first-season", "2024", "--min-edge", "0.07", "--resamples", "123"]
    options += ["--seed", "9", "--demargin", "shin"]
    assert main(["--enriched", str(enriched), "--oof", str(oof), *options]) == 1
    assert seen == {
        "books": ["avg", "max"],
        "weights": [0.5, 1.0],
        "reference_book": "avg",
        "demargin": "shin",
        "config": ValueConfig(min_edge=0.07),
        "first_season": 2024,
        "n_resamples": 123,
        "seed": 9,
        "rows": (len(pd.read_parquet(enriched)), len(pd.read_parquet(oof))),
    }


@pytest.mark.usefixtures("isolated_logging")
@pytest.mark.parametrize("error", [ValueError("bad table"), KeyError("odd column")])
def test_the_command_reports_either_kind_of_failed_run_and_exits_one(
    inputs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    def fail(*args: object, **kwargs: object) -> BacktestResult:
        raise error

    monkeypatch.setattr(bt, "run_backtest", fail)
    enriched, oof = inputs
    with caplog.at_level(logging.ERROR):
        assert main(["--enriched", str(enriched), "--oof", str(oof)]) == 1
    assert "Cannot run the backtest" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_on_a_season_it_cannot_calibrate(
    inputs: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    with caplog.at_level(logging.ERROR):
        assert (
            main(
                [
                    "--enriched",
                    str(enriched),
                    "--oof",
                    str(oof),
                    "--first-season",
                    "2022",
                ]
            )
            == 1
        )
    assert "Cannot run the backtest" in caplog.text
    assert "earlier predicted matches" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_rejects_an_edge_the_engine_cannot_use(
    inputs: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    enriched, oof = inputs
    with caplog.at_level(logging.ERROR):
        assert (
            main(["--enriched", str(enriched), "--oof", str(oof), "--min-edge", "-1"])
            == 1
        )
    assert "min_edge cannot be negative" in caplog.text


@pytest.mark.parametrize(
    "arguments",
    [
        ["--weights", "1.5"],
        ["--weights", "-0.1"],
        ["--weights", "high"],
        ["--books", "zzz"],
        ["--reference", "max"],
        ["--resamples", "0"],
    ],
)
def test_the_command_rejects_unusable_options(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        main(arguments)


def test_the_command_accepts_the_edges_of_its_ranges() -> None:
    args = bt._parse_args(["--resamples", "1", "--weights", "0", "1"])
    assert args.resamples == 1
    assert args.weights == [0.0, 1.0]
