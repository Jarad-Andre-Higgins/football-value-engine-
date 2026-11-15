"""Walk-forward backtest of the value engine: profit and closing-line value.

The backtest replays the weekly routine on past seasons. Every Friday it takes
the prices on offer (the *early* snapshot of the odds table), asks the value
engine which to bet and how much, and settles the bets from the final scores.
Each bet is then judged against the price the sharp book settled on at kick-off:
closing-line value (CLV) is ``odds_taken * closing_probability - 1``, the profit
per unit staked the bet would be expected to make if the closing price is right.
A bet with positive CLV was taken at a better price than the market finally
agreed on, whether or not it happened to win.

Nothing in a decision may use information from after the decision:

* **Model probabilities** come from the walk-forward predictions of
  ``python -m src.train``. Their calibration (goals per predicted xG, and the
  Dixon-Coles ``rho``) is fitted on the seasons *before* the one being priced.
* **The reference price** is the sharp book's early price with its margin removed.
* **Results and the closing price** are read only after the bets are placed.

The sharp book's margin is taken out of both its early price (the reference) and
its closing price (the yardstick). How that is done is a choice that moves both
the bets selected and the closing-line value measured: ``proportional`` scaling
trusts a long shot's price as much as a favourite's, and the ``power`` and
``shin`` methods do not. ``python -m src.margin_audit`` compares them on the
results and ``--demargin`` picks one.

A strategy is a bookmaker to bet at and a model weight, the share of the final
probability taken from the model instead of the sharp book's early price. Weight
0 never trusts the model: it only shops for prices where a softer book is out of
line with the sharp one. Weight 1 trusts the model alone. Next to every book sits
a yardstick, a unit bet on every price it quotes: its CLV is the margin you pay
before any skill, and any strategy has to beat that.

Limits worth knowing:

* Only 1X2 and over/under 2.5 goals have price history, so only those are tested.
* The early snapshot has no timestamp. The backtest bets at the first listed
  price, which may be fresher or staler than a real Friday coupon would see.
* Prices are taken as quoted: no stake limits, account restrictions or
  commissions. ``max`` (the best price among many bookmakers) is an upper bound,
  not a price anyone can take on every outcome.
* Stakes are sized on a constant bankroll, so a bad week does not shrink the
  stakes of the next and results stay comparable between strategies.

Usage::

    python -m src.backtest
    python -m src.backtest --books b365 --weights 0 0.25 --first-season 2024
    python -m src.backtest --reference avg    # use the market average as the sharp view
    python -m src.backtest --demargin shin --output-dir data/processed/backtest_shin
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.blend import BOOTSTRAP_BUDGET
from src.config import PROCESSED_DIR
from src.demargin import DEFAULT_METHOD, METHOD_NAMES, METHODS, check_method
from src.dixon_coles import DEFAULT_MAX_GOALS, FloatArray
from src.enrich import ENRICHED_OUTPUT
from src.evaluate import Calibration, fit_calibration
from src.logging_config import setup_logging
from src.markets import market_set, score_matrices, selection_probabilities
from src.odds_history import (
    BOOK_NAMES,
    BOOKS,
    FAIR_BOOKS,
    KEY,
    MARKETS,
    fair_view,
    odds_long,
    settle,
)
from src.train import OOF_OUTPUT
from src.value import ValueConfig, find_value_bets, selected_bets

logger = logging.getLogger("src.backtest")

#: First season to bet on: the first with a whole earlier season to calibrate on.
FIRST_SEASON: Final[int] = 2023
MIN_CALIBRATION_MATCHES: Final[int] = 500
#: Weekday of the weekly coupon (Monday is 0), so Friday.
BATCH_WEEKDAY: Final[int] = 4
DEFAULT_BOOKS: Final[tuple[str, ...]] = ("b365", "avg", "max", "ps")
DEFAULT_WEIGHTS: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0)
REFERENCE_BOOK: Final[str] = "ps"
#: How the margin is taken out of the sharp book's prices (see :mod:`src.demargin`).
DEMARGIN: Final[str] = DEFAULT_METHOD
DEFAULT_RESAMPLES: Final[int] = 2000
BACKTEST_DIR: Final[Path] = PROCESSED_DIR / "backtest"

#: ``(market_id, selection)`` of every price the backtest can test.
SELECTIONS: Final[tuple[tuple[str, str], ...]] = tuple(
    (market_id, name) for market_id, pairs in MARKETS.items() for name, _ in pairs
)
ENRICHED_REQUIRED: Final[tuple[str, ...]] = (
    "match_id",
    "season",
    "date",
    "home_goals",
    "away_goals",
)
OOF_REQUIRED: Final[tuple[str, ...]] = (
    "match_id",
    "season",
    "pred_home_xg",
    "pred_away_xg",
)
#: What the value engine may look at. Everything else (the closing price, the
#: result) is attached after the decision.
DECISION_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "bookmaker",
    "market_id",
    "selection",
    "odds",
    "probability",
    "push_probability",
    "p_reference",
)
BET_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "season",
    "date",
    "batch",
    "bookmaker",
    "market_id",
    "selection",
    "odds",
    "probability",
    "p_reference",
    "p_final",
    "edge",
    "stake_fraction",
    "stake",
    "p_close",
)
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "bets",
    "staked",
    "profit",
    "roi",
    "clv",
    "clv_mean",
    "beat_close",
    "hit_rate",
    "expected_hit_rate",
    "mean_odds",
    "mean_edge",
    "max_drawdown",
)
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "bookmaker",
    "strategy",
    "weight",
    "bets",
    "staked",
    "profit",
    "roi",
    "roi_low",
    "roi_high",
    "clv",
    "clv_low",
    "clv_high",
    "clv_mean",
    "beat_close",
    "hit_rate",
    "expected_hit_rate",
    "mean_odds",
    "mean_edge",
    "max_drawdown",
)
ODDS_EDGES: Final[tuple[float, ...]] = (1.0, 2.0, 3.0, 5.0, float("inf"))
ODDS_LABELS: Final[tuple[str, ...]] = ("1.0-2.0", "2.0-3.0", "3.0-5.0", "5.0+")
BASELINE: Final[str] = "every price"


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def odds_range(odds: pd.Series) -> pd.Series:
    """Label each price with its range: ``1.0-2.0``, ``2.0-3.0``, ``3.0-5.0`` or ``5.0+``.

    Ranges include their upper end, so a price of exactly 2.0 is in ``1.0-2.0``.
    """
    ranges = pd.cut(
        odds, bins=list(ODDS_EDGES), labels=list(ODDS_LABELS), include_lowest=True
    )
    return ranges.astype("object")


def _book_name(book: str) -> str:
    return BOOK_NAMES.get(book, book)


def strategy_label(weight: float) -> str:
    """Name of a strategy: the model weight, or the every-price yardstick (NaN)."""
    return BASELINE if np.isnan(weight) else f"weight {weight:.2f}"


# --------------------------------------------------------------------------- #
# Weeks
# --------------------------------------------------------------------------- #
def batch_start(dates: pd.Series) -> pd.Series:
    """The Friday on or before each date: the day the coupon for that match is made.

    A coupon made on a Friday covers the matches until the next Thursday, and a
    match is only ever bet on once, in the coupon of its own week.
    """
    days = pd.to_datetime(dates).dt.normalize()
    return days - pd.to_timedelta((days.dt.weekday - BATCH_WEEKDAY) % 7, unit="D")


# --------------------------------------------------------------------------- #
# What the model thought, without hindsight
# --------------------------------------------------------------------------- #
def _season_probabilities(
    block: pd.DataFrame, calibration: Calibration, max_goals: int
) -> pd.DataFrame:
    """1X2 and over/under 2.5 probabilities for one season's predicted matches."""
    lam = calibration.scale_home * block["pred_home_xg"].to_numpy(dtype="float64")
    mu = calibration.scale_away * block["pred_away_xg"].to_numpy(dtype="float64")
    win, push = selection_probabilities(
        score_matrices(lam, mu, calibration.rho, max_goals)
    )
    catalog = market_set(max_goals).catalog
    position = {
        (market_id, selection): row
        for row, (market_id, selection) in enumerate(
            zip(catalog["market_id"], catalog["selection"])
        )
    }
    match_ids = block["match_id"].to_numpy()
    pieces = []
    for market_id, selection in SELECTIONS:
        row = position[(market_id, selection)]
        pieces.append(
            pd.DataFrame(
                {
                    "match_id": match_ids,
                    "market_id": market_id,
                    "selection": selection,
                    "probability": win[:, row],
                    "push_probability": push[:, row],
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def walk_forward_probabilities(
    oof: pd.DataFrame,
    results: pd.DataFrame,
    *,
    first_season: int = FIRST_SEASON,
    min_calibration_matches: int = MIN_CALIBRATION_MATCHES,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrated model probabilities, one season at a time, with no look-ahead.

    The walk-forward predictions are honest forecasts of xG, but xG is not goals.
    Before a season is priced, the goals per predicted xG and the Dixon-Coles
    ``rho`` are fitted on the predictions of the *earlier* seasons and the goals
    that followed them. The season being priced, and every later one, never
    touches its own calibration.

    Args:
        oof: Walk-forward predictions: ``match_id``, ``season``,
            ``pred_home_xg`` and ``pred_away_xg``.
        results: ``match_id``, ``home_goals`` and ``away_goals``.
        first_season: First season to price.
        min_calibration_matches: Fewest earlier matches a calibration may rest on.
        max_goals: Size of the score grid.

    Returns:
        ``(probabilities, calibrations)``. The first has ``match_id``,
        ``market_id``, ``selection``, ``probability`` and ``push_probability`` for
        every price the backtest tests. The second has one row per priced season:
        ``season``, ``calibrated_on`` (matches the calibration used), ``matches``
        (matches priced), ``scale_home``, ``scale_away`` and ``rho``.

    Raises:
        ValueError: If columns are missing, a match is repeated, no season from
            ``first_season`` on has predictions, or a season has too few earlier
            matches to be calibrated on.
    """
    _require(oof, OOF_REQUIRED, "the out-of-fold predictions")
    _require(results, ("match_id", "home_goals", "away_goals"), "results")
    if oof["match_id"].duplicated().any():
        raise ValueError("the out-of-fold predictions repeat a match")
    if results["match_id"].duplicated().any():
        raise ValueError("results repeats a match")
    frame = oof[list(OOF_REQUIRED)].merge(
        results[["match_id", "home_goals", "away_goals"]], on="match_id", how="left"
    )
    usable = frame.dropna(
        subset=["pred_home_xg", "pred_away_xg", "home_goals", "away_goals"]
    )
    if len(usable) < len(frame):
        logger.warning(
            "%d predicted matches have no result or no prediction and are left out",
            len(frame) - len(usable),
        )
    seasons = sorted(int(s) for s in usable["season"].unique() if s >= first_season)
    if not seasons:
        raise ValueError(f"no walk-forward predictions from season {first_season} on")
    pieces, rows = [], []
    for season in seasons:
        past = usable[usable["season"] < season]
        if len(past) < min_calibration_matches:
            raise ValueError(
                f"season {season} needs at least {min_calibration_matches} earlier "
                f"predicted matches to calibrate on, found {len(past)}; start the "
                "backtest at a later season"
            )
        if past["pred_home_xg"].sum() <= 0 or past["pred_away_xg"].sum() <= 0:
            raise ValueError("the earlier predictions add up to no expected goals")
        calibration = fit_calibration(
            past["pred_home_xg"].to_numpy(dtype="float64"),
            past["pred_away_xg"].to_numpy(dtype="float64"),
            past["home_goals"].to_numpy(dtype="float64"),
            past["away_goals"].to_numpy(dtype="float64"),
        )
        block = usable[usable["season"] == season]
        pieces.append(_season_probabilities(block, calibration, max_goals))
        rows.append(
            {
                "season": season,
                "calibrated_on": len(past),
                "matches": len(block),
                "scale_home": calibration.scale_home,
                "scale_away": calibration.scale_away,
                "rho": calibration.rho,
            }
        )
    return pd.concat(pieces, ignore_index=True), pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# What can be bet on, and at what price
# --------------------------------------------------------------------------- #
def build_universe(
    enriched: pd.DataFrame,
    model: pd.DataFrame,
    reference_book: str = REFERENCE_BOOK,
    demargin: str = DEMARGIN,
) -> pd.DataFrame:
    """Every selection that can be bet on and measured.

    A selection is in the universe when the model priced it and the reference
    book quoted its whole market both early (the price a decision may use) and at
    the close (the yardstick the bet is judged against).

    Args:
        enriched: The match table with ``match_id``, ``season``, ``date`` and the
            wide odds columns.
        model: Probabilities from :func:`walk_forward_probabilities`.
        reference_book: The sharp book (one of :data:`src.odds_history.FAIR_BOOKS`).
        demargin: How the margin is taken out of the sharp book's early and
            closing prices (one of :data:`src.demargin.METHODS`).

    Returns:
        ``match_id``, ``market_id``, ``selection``, ``probability``,
        ``push_probability``, ``p_reference`` (the book's early fair probability),
        ``p_close`` (its closing fair probability), ``season``, ``date`` and
        ``batch`` (the Friday of the coupon the match belongs to).

    Raises:
        ValueError: For an unknown ``demargin`` method.
    """
    early = fair_view(enriched, "early", (reference_book,), demargin)[[*KEY, "p_fair"]]
    close = fair_view(enriched, "close", (reference_book,), demargin)[[*KEY, "p_fair"]]
    universe = (
        early.rename(columns={"p_fair": "p_reference"})
        .merge(close.rename(columns={"p_fair": "p_close"}), on=KEY)
        .merge(model, on=KEY)
        .merge(enriched[["match_id", "season", "date"]], on="match_id", how="left")
    )
    universe["date"] = pd.to_datetime(universe["date"])
    dated = universe["date"].notna()
    if not dated.all():
        logger.warning(
            "%d prices have no match date and are left out", int((~dated).sum())
        )
        universe = universe[dated]
    universe["batch"] = batch_start(universe["date"])
    order = ["date", "match_id", "market_id", "selection"]
    return universe.sort_values(order).reset_index(drop=True)


def build_offers(
    enriched: pd.DataFrame, universe: pd.DataFrame, book: str
) -> pd.DataFrame:
    """One book's early prices for every selection of the universe.

    Returns:
        The universe's columns plus ``bookmaker`` and ``odds``, one row per price.
    """
    prices = odds_long(enriched, "early", (book,))
    offers = prices.merge(universe, on=KEY, how="inner")
    order = ["date", "match_id", "market_id", "selection"]
    return offers.sort_values(order).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Placing and settling bets
# --------------------------------------------------------------------------- #
def place_bets(offers: pd.DataFrame, config: ValueConfig) -> pd.DataFrame:
    """The bets the value engine places, one weekly coupon at a time.

    The engine sees only :data:`DECISION_COLUMNS`: the price, the model's
    probability and the reference's early probability. Each batch (week) is a
    separate run, so the per-run exposure cap applies to a coupon as it would
    live. The closing price and the season are attached afterwards.

    Args:
        offers: From :func:`build_offers`.
        config: Betting rules, including the model weight.

    Returns:
        One row per bet with :data:`BET_COLUMNS`.
    """
    decisions = offers.loc[:, list(DECISION_COLUMNS)]
    pieces = []
    for _, coupon in decisions.groupby(offers["batch"], sort=True):
        chosen = selected_bets(find_value_bets(coupon, config))
        if not chosen.empty:
            pieces.append(chosen)
    if pieces:
        bets = pd.concat(pieces, ignore_index=True)
    else:
        bets = selected_bets(find_value_bets(decisions, config))
    after = offers[[*KEY, "season", "date", "batch", "p_close"]]
    bets = bets.merge(after, on=KEY, how="left", validate="one_to_one")
    return bets[list(BET_COLUMNS)]


def _score(table: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Settle a table of priced bets and add its closing-line value."""
    scored = settle(table, results)
    unsettled = ~scored["settled"]
    if unsettled.any():
        logger.warning(
            "%d bets cannot be settled (no usable score) and are left out",
            int(unsettled.sum()),
        )
        scored = scored[~unsettled]
    scored = scored.drop(columns="settled")
    scored["clv"] = scored["odds"] * scored["p_close"] - 1.0
    return scored.reset_index(drop=True)


def settle_bets(bets: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Add each bet's outcome, money won or lost (``pnl``) and closing-line value.

    ``profit`` is per unit staked and ``pnl`` is that times the stake. ``clv`` is
    ``odds * p_close - 1``.

    Args:
        bets: From :func:`place_bets`.
        results: ``match_id``, ``home_goals`` and ``away_goals``.
    """
    scored = _score(bets, results)
    scored["pnl"] = scored["stake"] * scored["profit"]
    return scored


def flat_baseline(offers: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """One unit on every price a book quotes: the cost of its margin, with no skill.

    ``p_final`` and ``edge`` describe the bet as the reference sees it, so the
    yardstick's expected hit rate and edge sit in the same columns as a strategy's.
    """
    table = offers.assign(stake=1.0)
    scored = _score(table, results)
    scored["pnl"] = scored["profit"]
    scored["p_final"] = scored["p_reference"]
    scored["edge"] = scored["odds"] * scored["p_reference"] - 1.0
    return scored


# --------------------------------------------------------------------------- #
# Measuring
# --------------------------------------------------------------------------- #
def max_drawdown(pnl: FloatArray) -> float:
    """Largest fall from a running peak of cumulative profit (zero is the start).

    Args:
        pnl: Profit or loss per period, in time order.
    """
    path = np.concatenate([[0.0], np.cumsum(np.asarray(pnl, dtype="float64"))])
    return float((np.maximum.accumulate(path) - path).max())


def cluster_bootstrap(
    numerator: FloatArray,
    denominator: FloatArray,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval of ``sum(numerator) / sum(denominator)`` over resampled clusters.

    Bets in the same week share news, weather and market moves, so they are not
    independent. Whole weeks are resampled with replacement instead of single
    bets, which keeps the interval honest about that. A resample that stakes
    nothing has no ratio and is left out.

    Args:
        numerator: Per cluster (week), for example the profit.
        denominator: Per cluster, for example the amount staked.
        n_resamples: Number of resamples.
        level: Width of the interval.
        seed: Seed of the resampling.

    Returns:
        ``(low, high)``; both NaN when no resample has a positive denominator.

    Raises:
        ValueError: For empty or mismatched arrays, a ``level`` outside (0, 1) or
            fewer than one resample.
    """
    top = np.asarray(numerator, dtype="float64")
    bottom = np.asarray(denominator, dtype="float64")
    if top.ndim != 1 or len(top) == 0 or top.shape != bottom.shape:
        raise ValueError("need two non-empty one-dimensional arrays of equal length")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rng = np.random.default_rng(seed)
    n = len(top)
    block = max(1, min(n_resamples, BOOTSTRAP_BUDGET // n))
    ratios = np.empty(n_resamples, dtype="float64")
    for start in range(0, n_resamples, block):
        size = min(block, n_resamples - start)
        picks = rng.integers(0, n, size=(size, n))
        above, below = top[picks].sum(axis=1), bottom[picks].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios[start : start + size] = np.where(below > 0.0, above / below, np.nan)
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(ratios, [tail, 1.0 - tail])
    return float(low), float(high)


def _metrics(bets: pd.DataFrame) -> dict[str, float]:
    """Profit and closing-line value of one group of bets."""
    n = len(bets)
    if n == 0:
        nothing = float("nan")
        return {
            "bets": 0,
            "staked": 0.0,
            "profit": 0.0,
            "roi": nothing,
            "clv": nothing,
            "clv_mean": nothing,
            "beat_close": nothing,
            "hit_rate": nothing,
            "expected_hit_rate": nothing,
            "mean_odds": nothing,
            "mean_edge": nothing,
            "max_drawdown": 0.0,
        }
    staked = float(bets["stake"].sum())
    profit = float(bets["pnl"].sum())
    weighted = float((bets["stake"] * bets["clv"]).sum())
    by_week = bets.groupby("batch", sort=True)["pnl"].sum().to_numpy(dtype="float64")
    return {
        "bets": n,
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked > 0.0 else float("nan"),
        "clv": weighted / staked if staked > 0.0 else float("nan"),
        "clv_mean": float(bets["clv"].mean()),
        "beat_close": float((bets["clv"] > 0.0).mean()),
        "hit_rate": float(bets["won"].astype("float64").mean()),
        "expected_hit_rate": float(bets["p_final"].mean()),
        "mean_odds": float(bets["odds"].mean()),
        "mean_edge": float(bets["edge"].mean()),
        "max_drawdown": max_drawdown(by_week),
    }


def summarise(bets: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Profit and closing-line value of settled bets, overall or per group.

    Args:
        bets: Settled bets with ``stake``, ``pnl``, ``clv``, ``won``, ``p_final``,
            ``odds``, ``edge`` and ``batch``.
        by: Columns to group by; none gives a single row.

    Returns:
        The ``by`` columns, then :data:`METRIC_COLUMNS`: ``bets``; ``staked`` and
        ``profit`` in money; ``roi`` (profit over staked); ``clv`` (stake-weighted)
        and ``clv_mean`` (every bet counts the same); ``beat_close`` (share of bets
        with positive closing-line value); ``hit_rate`` against
        ``expected_hit_rate``; ``mean_odds``; ``mean_edge`` (what the engine
        expected to earn per unit staked); ``max_drawdown`` (largest fall of
        cumulative profit, in money).
    """
    columns = [*by, *METRIC_COLUMNS]
    if not by:
        return pd.DataFrame([_metrics(bets)], columns=columns)
    rows = []
    for key, group in bets.groupby(list(by), sort=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, keys)), **_metrics(group)})
    return pd.DataFrame(rows, columns=columns)


def _cluster_totals(
    bets: pd.DataFrame, batches: pd.Index
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Stake, profit and stake-weighted closing-line value of each week.

    Every week of ``batches`` is present, with zeros where nothing was bet.
    """
    totals = (
        bets.assign(_weighted=bets["stake"] * bets["clv"])
        .groupby("batch", sort=True)[["stake", "pnl", "_weighted"]]
        .sum()
        .reindex(batches, fill_value=0.0)
    )
    return (
        totals["stake"].to_numpy(dtype="float64"),
        totals["pnl"].to_numpy(dtype="float64"),
        totals["_weighted"].to_numpy(dtype="float64"),
    )


def _summary_row(
    bets: pd.DataFrame,
    batches: pd.Index,
    *,
    book: str,
    weight: float,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """One strategy's metrics with 95% intervals for ROI and closing-line value."""
    stake, pnl, weighted = _cluster_totals(bets, batches)
    roi_low, roi_high = cluster_bootstrap(
        pnl, stake, n_resamples=n_resamples, seed=seed
    )
    clv_low, clv_high = cluster_bootstrap(
        weighted, stake, n_resamples=n_resamples, seed=seed
    )
    return {
        "bookmaker": book,
        "strategy": strategy_label(weight),
        "weight": weight,
        **_metrics(bets),
        "roi_low": roi_low,
        "roi_high": roi_high,
        "clv_low": clv_low,
        "clv_high": clv_high,
    }


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BacktestResult:
    """What a backtest found.

    Attributes:
        bets: Every bet of every strategy, settled, with ``weight`` and
            :data:`BET_COLUMNS` plus ``won``, ``push``, ``profit`` (per unit),
            ``pnl`` (money) and ``clv``.
        summary: One row per strategy and per every-price yardstick (``weight``
            NaN): :data:`METRIC_COLUMNS` plus 95% week-resampled intervals for ROI
            and closing-line value.
        by_season: The metrics per bookmaker, weight and season.
        by_market: The same per market.
        by_odds: The same per odds range (a long-shot check).
        calibrations: See :func:`walk_forward_probabilities`.
        coverage: Per season, matches priced and matches in the universe per market.
        reference_book: The sharp book the bets were measured against.
        bankroll: Bankroll the stakes were sized on.
        demargin: How the margin was taken out of the sharp book's prices.
    """

    bets: pd.DataFrame
    summary: pd.DataFrame
    by_season: pd.DataFrame
    by_market: pd.DataFrame
    by_odds: pd.DataFrame
    calibrations: pd.DataFrame
    coverage: pd.DataFrame
    reference_book: str
    bankroll: float
    demargin: str = DEMARGIN


def _coverage(universe: pd.DataFrame, calibrations: pd.DataFrame) -> pd.DataFrame:
    counts = (
        universe.groupby(["season", "market_id"])["match_id"]
        .nunique()
        .unstack("market_id")
        .reindex(columns=list(MARKETS))
        .rename_axis(columns=None)
        .reset_index()
    )
    coverage = calibrations[["season", "matches"]].merge(
        counts, on="season", how="left"
    )
    for market_id in MARKETS:
        coverage[market_id] = coverage[market_id].fillna(0).astype("int64")
    return coverage


def run_backtest(
    enriched: pd.DataFrame,
    oof: pd.DataFrame,
    *,
    books: Sequence[str] = DEFAULT_BOOKS,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    reference_book: str = REFERENCE_BOOK,
    demargin: str = DEMARGIN,
    config: ValueConfig | None = None,
    first_season: int = FIRST_SEASON,
    min_calibration_matches: int = MIN_CALIBRATION_MATCHES,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> BacktestResult:
    """Replay the weekly betting routine on past seasons.

    Args:
        enriched: The enriched match table (``match_id``, ``season``, ``date``,
            ``home_goals``, ``away_goals`` and the wide odds columns).
        oof: Walk-forward xG predictions (``python -m src.train``).
        books: Bookmakers to bet at (codes of :data:`src.odds_history.BOOKS`).
        weights: Model weights to try, each between 0 and 1.
        reference_book: The sharp book whose early price is the reference and
            whose closing price is the yardstick.
        demargin: How the margin is taken out of the sharp book's early and
            closing prices (one of :data:`src.demargin.METHODS`).
        config: Betting rules; its ``model_weight`` is replaced by each weight.
        first_season: First season to bet on.
        min_calibration_matches: Fewest earlier matches a calibration may use.
        n_resamples: Bootstrap resamples for the intervals.
        seed: Seed of the bootstrap.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        ValueError: For unusable settings or inputs, including a table in which
            no match has the reference book's early and closing quotes.
    """
    config = config or ValueConfig()
    books = tuple(dict.fromkeys(books))
    weights = tuple(dict.fromkeys(float(weight) for weight in weights))
    if not books or not weights:
        raise ValueError("choose at least one bookmaker and one model weight")
    unknown = [book for book in books if book not in BOOKS]
    if unknown:
        raise ValueError(f"unknown bookmakers {unknown}; choose from {list(BOOKS)}")
    if reference_book not in FAIR_BOOKS:
        raise ValueError(
            f"the reference must be one of {list(FAIR_BOOKS)}, got {reference_book!r}"
        )
    check_method(demargin)
    configs = {weight: replace(config, model_weight=weight) for weight in weights}
    _require(enriched, ENRICHED_REQUIRED, "the match table")
    if enriched["match_id"].duplicated().any():
        raise ValueError("the match table repeats a match")
    results = enriched[["match_id", "home_goals", "away_goals"]]

    model, calibrations = walk_forward_probabilities(
        oof,
        results,
        first_season=first_season,
        min_calibration_matches=min_calibration_matches,
    )
    universe = build_universe(enriched, model, reference_book, demargin)
    if universe.empty:
        raise ValueError(
            f"no match from season {first_season} on has a model probability and "
            f"a complete early and closing {_book_name(reference_book)} quote; "
            "try --reference avg, or check the odds with python -m src.odds_history"
        )
    coverage = _coverage(universe, calibrations)
    batches = pd.Index(np.sort(universe["batch"].unique()))
    logger.info(
        "Universe: %d prices in %d weekly coupons",
        len(universe),
        len(batches),
    )

    rows: list[dict[str, object]] = []
    pieces: list[pd.DataFrame] = []
    last = pd.DataFrame()
    for book in books:
        offers = build_offers(enriched, universe, book)
        logger.info("%s: %d early prices to judge", _book_name(book), len(offers))
        yardstick = flat_baseline(offers, results)
        rows.append(
            _summary_row(
                yardstick,
                batches,
                book=book,
                weight=float("nan"),
                n_resamples=n_resamples,
                seed=seed,
            )
        )
        for weight in weights:
            bets = settle_bets(place_bets(offers, configs[weight]), results)
            bets = bets.assign(weight=weight)
            last = bets
            if not bets.empty:
                pieces.append(bets)
            rows.append(
                _summary_row(
                    bets,
                    batches,
                    book=book,
                    weight=weight,
                    n_resamples=n_resamples,
                    seed=seed,
                )
            )
            logger.info(
                "%s, %s: %d bets", _book_name(book), strategy_label(weight), len(bets)
            )
    all_bets = pd.concat(pieces, ignore_index=True) if pieces else last
    ranged = all_bets.assign(odds_range=odds_range(all_bets["odds"]))
    keys = ["bookmaker", "weight"]
    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    return BacktestResult(
        bets=all_bets,
        summary=summary,
        by_season=summarise(all_bets, [*keys, "season"]),
        by_market=summarise(all_bets, [*keys, "market_id"]),
        by_odds=summarise(ranged, [*keys, "odds_range"]),
        calibrations=calibrations,
        coverage=coverage,
        reference_book=reference_book,
        bankroll=config.bankroll,
        demargin=demargin,
    )


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #
def _percent(value: float, *, signed: bool = False, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{100.0 * value:+.{digits}f}" if signed else f"{100.0 * value:.{digits}f}"


def _interval(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "-"
    return f"[{100.0 * low:+.1f}, {100.0 * high:+.1f}]"


def format_backtest(result: BacktestResult) -> str:
    """Render the result as aligned text tables."""
    sharp = _book_name(result.reference_book)
    summary = result.summary
    yardstick = summary["weight"].isna()
    money = summary["staked"].map("{:,.0f}".format).where(~yardstick, "-")
    strategies = pd.DataFrame(
        {
            "book": summary["bookmaker"].map(_book_name),
            "strategy": summary["strategy"],
            "bets": summary["bets"],
            "staked": money,
            "ROI %": summary["roi"].map(lambda v: _percent(v, signed=True)),
            "ROI 95% interval": [
                _interval(a, b) for a, b in zip(summary["roi_low"], summary["roi_high"])
            ],
            "CLV %": summary["clv"].map(lambda v: _percent(v, signed=True)),
            "CLV 95% interval": [
                _interval(a, b) for a, b in zip(summary["clv_low"], summary["clv_high"])
            ],
            "beat close %": summary["beat_close"].map(lambda v: _percent(v, digits=0)),
        }
    )
    diagnostics = pd.DataFrame(
        {
            "book": summary["bookmaker"].map(_book_name),
            "strategy": summary["strategy"],
            "hit %": summary["hit_rate"].map(lambda v: _percent(v)),
            "expected hit %": summary["expected_hit_rate"].map(lambda v: _percent(v)),
            "mean odds": summary["mean_odds"].map(
                lambda v: "-" if not np.isfinite(v) else f"{v:.2f}"
            ),
            "mean edge %": summary["mean_edge"].map(lambda v: _percent(v, signed=True)),
            "profit": summary["profit"].map("{:+,.0f}".format).where(~yardstick, "-"),
            "worst fall, % of bankroll": (
                100.0 * summary["max_drawdown"] / result.bankroll
            )
            .map("{:.1f}".format)
            .where(~yardstick, "-"),
        }
    )
    fitted = result.calibrations.rename(columns={"calibrated_on": "fitted on"})
    coverage_title = (
        "Matches priced per season, and prices with a complete early and closing "
        f"{sharp} quote"
    )
    calibration_title = (
        "Calibration of the model, fitted on the seasons before each one"
    )
    method = METHOD_NAMES.get(result.demargin, result.demargin)
    strategy_title = (
        f"Strategies, early prices, measured against the {sharp} close (margin "
        f"taken out by {method}). 95% intervals resample whole weeks.\n"
        "  ROI: profit over amount staked; CLV: expected profit per unit if the "
        "closing price is right (positive = beat the close)\n"
        "  every price: one unit on every price the book quotes, the margin "
        "paid before any skill"
    )
    diagnostic_title = (
        "What the bets looked like (hit % against the engine's own expectation).\n"
        f"  Stakes are sized on a constant bankroll of {result.bankroll:,.0f}, so a "
        "worst fall above 100% would have wiped it out"
    )
    fitted_text = fitted.to_string(index=False, float_format="{:.3f}".format)
    sections = [
        f"{coverage_title}\n{result.coverage.to_string(index=False)}",
        f"{calibration_title}\n{fitted_text}",
        f"{strategy_title}\n{strategies.to_string(index=False)}",
        f"{diagnostic_title}\n{diagnostics.to_string(index=False)}",
    ]
    return "\n\n".join(sections)


def _clv_phrase(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "no interval"
    if low > 0.0:
        return "beats the closing line"
    if high < 0.0:
        return "trails the closing line"
    return "cannot be told apart from the closing line"


def _roi_phrase(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "no interval"
    if low > 0.0:
        return "profit is unlikely to be luck"
    if high < 0.0:
        return "the loss is unlikely to be bad luck"
    return "profit cannot be told apart from luck"


def verdict(result: BacktestResult) -> list[str]:
    """Plain-language reading of every strategy's result."""
    lines = []
    for row in result.summary.itertuples():
        if np.isnan(row.weight):
            continue
        label = f"{_book_name(row.bookmaker)}, {row.strategy}"
        if row.bets == 0:
            lines.append(f"{label}: no bets")
            continue
        lines.append(
            f"{label}: {row.bets} bets, CLV {_percent(row.clv, signed=True)}% "
            f"(95%: {_interval(row.clv_low, row.clv_high)}) "
            f"{_clv_phrase(row.clv_low, row.clv_high)}; ROI "
            f"{_percent(row.roi, signed=True)}% "
            f"(95%: {_interval(row.roi_low, row.roi_high)}), "
            f"{_roi_phrase(row.roi_low, row.roi_high)}"
        )
    return lines


def save_backtest(result: BacktestResult, directory: Path = BACKTEST_DIR) -> list[Path]:
    """Write the result's tables to ``directory`` and return the files written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    result.bets.to_parquet(directory / "bets.parquet", index=False)
    written.append(directory / "bets.parquet")
    for name in (
        "summary",
        "by_season",
        "by_market",
        "by_odds",
        "calibrations",
        "coverage",
    ):
        path = directory / f"{name}.csv"
        getattr(result, name).to_csv(path, index=False)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _weight(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"a weight must be between 0 and 1, got {text}"
        )
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest: profit and closing-line value."
    )
    parser.add_argument("--enriched", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument("--oof", type=Path, default=OOF_OUTPUT)
    parser.add_argument(
        "--books", nargs="+", choices=BOOKS, default=list(DEFAULT_BOOKS)
    )
    parser.add_argument(
        "--weights", nargs="+", type=_weight, default=list(DEFAULT_WEIGHTS)
    )
    parser.add_argument(
        "--reference",
        choices=FAIR_BOOKS,
        default=REFERENCE_BOOK,
        help="sharp book: early price is the reference, close is the yardstick",
    )
    parser.add_argument(
        "--demargin",
        choices=METHODS,
        default=DEMARGIN,
        help="how the margin is taken out of the sharp book's prices "
        "(python -m src.margin_audit compares the methods; default: %(default)s)",
    )
    parser.add_argument("--first-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--min-edge", type=float, default=ValueConfig().min_edge)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=BACKTEST_DIR)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the backtest and return the process exit code.

    Returns ``0`` on success and ``1`` when an input is missing or unusable.
    """
    args = _parse_args(argv)
    log = setup_logging("backtest")
    for path, command in (
        (args.enriched, "python -m src.enrich"),
        (args.oof, "python -m src.train"),
    ):
        if not path.exists():
            log.error("Not found: %s; run %s first", path, command)
            return 1
    try:
        result = run_backtest(
            pd.read_parquet(args.enriched),
            pd.read_parquet(args.oof),
            books=args.books,
            weights=args.weights,
            reference_book=args.reference,
            demargin=args.demargin,
            config=ValueConfig(min_edge=args.min_edge),
            first_season=args.first_season,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except (ValueError, KeyError) as exc:
        log.error("Cannot run the backtest: %s", exc)
        return 1
    log.info("\n%s", format_backtest(result))
    for line in verdict(result):
        log.info(line)
    for path in save_backtest(result, args.output_dir):
        log.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
