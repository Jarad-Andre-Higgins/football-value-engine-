"""Take a strategy's closing-line value apart: where does it come from, and does it hold?

The backtest gives each strategy one number, closing-line value (CLV): the profit
per unit staked a bet would earn if the sharp book's closing price is right. One
number can hide a lot, so this audit reads the bets the backtest saved and asks
four questions of one strategy.

* **What is the CLV made of?** For a bet at ``odds``, with the sharp book's fair
  probability before the bet (``p_early``, the price the engine judged against)
  and at the close (``p_close``)::

      CLV = (odds * p_early - 1)  +  odds * (p_close - p_early)
            price gap                market drift

  The price gap is what the bettor saw: how far the price on offer sat above the
  sharp book's own. A strategy that shops for prices (model weight 0) has a large
  one by construction. The drift is what the sharp book did next, the part that
  says whether it later agreed. Quotes priced alike drift alike, so the drift is
  split again into what *similar* quotes drifted by on average (a property of the
  market, not of these bets) and what is left, which belongs to the bets. That
  split needs the match table, and the same sharp book and margin removal as the
  backtest that made the bets.
* **Do a few bets carry it?** The mean CLV with the best bets taken away, and the
  share of the total they hold.
* **Does it hold everywhere?** By season, league, market and price range.
* **Are the bets plausible?** How large the price gaps get. Against a sharp book a
  gap of 50% or more is more often a data or timing error than an opportunity.

Intervals resample whole weeks, as in the backtest, over the same calendar: every
week of the backtest, also the weeks without a bet. The weeks come from the match
table, or failing that from the weeks in which any strategy bet. With the
backtest's ``--resamples`` and ``--seed`` the CLV interval is the backtest's own.
Every row of the first table shares the same resamples, so the parts add up inside
each one.

Usage::

    python -m src.autopsy                              # Bet365 at model weight 0
    python -m src.autopsy --book max --weight 0 --top 15
    python -m src.autopsy --backtest-dir data/processed/backtest_shin --demargin shin
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.backtest import (
    BACKTEST_DIR,
    DEFAULT_RESAMPLES,
    DEMARGIN,
    REFERENCE_BOOK,
    cluster_bootstrap,
)
from src.demargin import METHODS
from src.enrich import ENRICHED_OUTPUT
from src.logging_config import setup_logging
from src.margin_audit import PROBABILITY_COLUMNS, selection_table
from src.odds_history import BOOK_NAMES, BOOKS, FAIR_BOOKS, odds_long
from src.value import ValueConfig

logger = logging.getLogger("src.autopsy")

FloatArray = NDArray[np.float64]

DEFAULT_BOOK: Final[str] = "b365"
DEFAULT_WEIGHT: Final[float] = 0.0
DEFAULT_TOP: Final[int] = 10
BET_REQUIRED: Final[tuple[str, ...]] = (
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
)
#: What the match table must have for the comparison with similar quotes.
MATCH_REQUIRED: Final[tuple[str, ...]] = (
    "match_id",
    "season",
    "date",
    "home_goals",
    "away_goals",
)
#: Facts about a match that make the bets readable, taken from the match table.
MATCH_COLUMNS: Final[tuple[str, ...]] = ("league", "home_team", "away_team")
#: Key of one selection of one match.
KEY: Final[list[str]] = ["match_id", "market_id", "selection"]
#: Number of best bets taken away to see whether a few carry the result.
TRIMS: Final[tuple[int, ...]] = (0, 1, 3, 5, 10)
#: The trim the verdict reports.
VERDICT_TRIM: Final[int] = 5
#: A price gap above this against a sharp book is more often a mistake than an opportunity.
IMPLAUSIBLE_GAP: Final[float] = 0.5
#: Fewer bets than this is a thin sample.
THIN_SAMPLE: Final[int] = 100
#: Half-width, in probability, of the window of quotes priced like a bet.
CONTROL_WIDTH: Final[float] = 0.02
#: Tolerance when checking that the match table reproduces the bets' probabilities.
PROBABILITY_TOLERANCE: Final[float] = 1e-9
ODDS_EDGES: Final[tuple[float, ...]] = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, float("inf"))
ODDS_LABELS: Final[tuple[str, ...]] = (
    "1.0-2.0",
    "2.0-3.0",
    "3.0-4.0",
    "4.0-5.0",
    "5.0-6.0",
    "6.0-8.0",
    "8.0+",
)
#: Price-gap ranges below the minimum edge; the range from the minimum edge up is the
#: last one.
GAP_BREAKS: Final[tuple[float, ...]] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.04)
#: Parts of the CLV in the order they are shown, with their labels.
PART_LABELS: Final[dict[str, str]] = {
    "gap": "price gap",
    "drift": "market drift",
    "typical": "  of it, what similar quotes drift by",
    "specific": "  of it, specific to these bets",
    "clv": "CLV",
}
#: ``(label, quantile)`` of the price-gap distribution.
GAP_STATISTICS: Final[tuple[tuple[str, float], ...]] = (
    ("lowest", 0.0),
    ("25%", 0.25),
    ("median", 0.5),
    ("75%", 0.75),
    ("90%", 0.9),
    ("highest", 1.0),
)
PART_COLUMNS: Final[list[str]] = ["weighting", "part", "value", "low", "high"]
CONCENTRATION_COLUMNS: Final[list[str]] = [
    "removed",
    "left",
    "mean_clv",
    "median_clv",
    "share",
]
BREAKDOWN_VALUES: Final[list[str]] = [
    "bets",
    "gap",
    "drift",
    "clv",
    "hit",
    "expected",
]
GAP_COLUMNS: Final[list[str]] = ["statistic", "gap"]
RESPONSE_COLUMNS: Final[list[str]] = [
    "gap_range",
    "region",
    "quotes",
    "gap",
    "drift",
    "drift_low",
    "drift_high",
    "clv",
    "clv_low",
    "clv_high",
]
BEST_COLUMNS: Final[list[str]] = [
    "season",
    "date",
    "league",
    "match",
    "market_id",
    "selection",
    "odds",
    "p_reference",
    "p_close",
    "gap",
    "clv",
    "won",
]
#: Folder of a run's tables, inside the backtest's.
AUTOPSY_FOLDER: Final[str] = "autopsy"


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _book_name(book: str) -> str:
    return BOOK_NAMES.get(book, book)


def odds_band(odds: pd.Series) -> pd.Series:
    """Label each price with its range, long shots in finer steps than the backtest.

    Ranges include their upper end, so a price of exactly 2.0 is in ``1.0-2.0``.
    The result is an ordered categorical, so tables keep the ranges in their
    natural order.
    """
    return pd.cut(
        odds, bins=list(ODDS_EDGES), labels=list(ODDS_LABELS), include_lowest=True
    )


def _stamps(values: pd.Series) -> pd.DatetimeIndex:
    """Dates as nanosecond timestamps, whatever resolution the table stored."""
    return pd.DatetimeIndex(pd.to_datetime(values)).astype("datetime64[ns]")


def _weeks(values: pd.Series) -> pd.DatetimeIndex:
    """The distinct week starts of a column of dates, in time order."""
    return pd.DatetimeIndex(_stamps(values).unique().sort_values())


# --------------------------------------------------------------------------- #
# The strategy's bets
# --------------------------------------------------------------------------- #
def strategy_bets(bets: pd.DataFrame, book: str, weight: float) -> pd.DataFrame:
    """The bets of one strategy, each with its price gap and market drift.

    Args:
        bets: The ``bets.parquet`` of a backtest run (see :data:`BET_REQUIRED`).
        book: The bookmaker the bets were placed at.
        weight: The model weight of the strategy.

    Returns:
        The strategy's bets with ``gap`` (``odds * p_reference - 1``) and ``drift``
        (``odds * (p_close - p_reference)``), which add up to ``clv``.

    Raises:
        ValueError: If a column is missing, the strategy has no bets, or the stored
            ``clv`` is not ``odds * p_close - 1``, which means the table was not
            made by :mod:`src.backtest`.
    """
    _require(bets, BET_REQUIRED, "the bets table")
    weights = bets["weight"].to_numpy(dtype="float64")
    wanted = (bets["bookmaker"] == book).to_numpy() & np.isclose(weights, weight)
    if not wanted.any():
        pairs = sorted({(str(b), float(w)) for b, w in zip(bets["bookmaker"], weights)})
        listed = ", ".join(f"{b} at weight {w:g}" for b, w in pairs) or "no bets at all"
        raise ValueError(
            f"no bets for {book} at weight {weight:g}; the file has {listed}"
        )
    chosen = bets.loc[wanted].reset_index(drop=True)
    odds = chosen["odds"].to_numpy(dtype="float64")
    before = chosen["p_reference"].to_numpy(dtype="float64")
    after = chosen["p_close"].to_numpy(dtype="float64")
    gap = odds * before - 1.0
    drift = odds * (after - before)
    if not np.allclose(gap + drift, chosen["clv"].to_numpy(dtype="float64"), atol=1e-9):
        raise ValueError(
            "the stored clv is not odds * p_close - 1; was this table made by "
            "src.backtest?"
        )
    return chosen.assign(gap=gap, drift=drift)


def attach_matches(bets: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Add the league and team names of each bet's match, when the table has them."""
    wanted = [
        column
        for column in MATCH_COLUMNS
        if column in matches.columns and column not in bets.columns
    ]
    if "match_id" not in matches.columns or not wanted:
        return bets
    names = matches[["match_id", *wanted]].drop_duplicates("match_id")
    return bets.merge(names, on="match_id", how="left", validate="many_to_one")


# --------------------------------------------------------------------------- #
# Quotes priced like the bets
# --------------------------------------------------------------------------- #
def control_population(
    enriched: pd.DataFrame, *, book: str, demargin: str, first_season: int
) -> pd.DataFrame:
    """Every complete quote of the sharp book, early and at the close, from ``first_season`` on.

    Args:
        enriched: The match table (:data:`MATCH_REQUIRED` and the wide odds columns).
        book: The sharp book the backtest took its fair probabilities from.
        demargin: The margin removal the backtest used.
        first_season: First season to include.

    Returns:
        One row per selection with ``match_id``, ``market_id``, ``selection``,
        ``season``, ``batch`` (the Friday of its week), ``p_early`` and
        ``p_close``; empty when the book has no complete quote.

    Raises:
        ValueError: If the match table lacks a column or ``demargin`` is unknown.
    """
    _require(enriched, MATCH_REQUIRED, "the match table")
    if demargin not in PROBABILITY_COLUMNS:
        raise ValueError(
            f"unknown margin removal {demargin!r}; choose from {list(METHODS)}"
        )
    column = PROBABILITY_COLUMNS[demargin]
    early = selection_table(enriched, book, "early")
    close = selection_table(enriched, book, "close")
    both = early[[*KEY, "season", "batch", column]].merge(
        close[[*KEY, column]],
        on=KEY,
        suffixes=("_early", "_close"),
        validate="one_to_one",
    )
    both = both.rename(
        columns={f"{column}_early": "p_early", f"{column}_close": "p_close"}
    )
    return both[both["season"] >= first_season].reset_index(drop=True)


def reproduces(bets: pd.DataFrame, population: pd.DataFrame) -> bool:
    """Whether the population holds every bet's selection with the same probabilities.

    If not, the margin removal or the sharp book differs from the backtest's and
    the population says nothing about these bets.
    """
    joined = bets[[*KEY, "p_reference", "p_close"]].merge(
        population[[*KEY, "p_early", "p_close"]],
        on=KEY,
        how="left",
        suffixes=("", "_population"),
    )
    if joined["p_early"].isna().any():
        return False
    early = (joined["p_reference"] - joined["p_early"]).abs().max()
    close = (joined["p_close"] - joined["p_close_population"]).abs().max()
    return bool(max(early, close) <= PROBABILITY_TOLERANCE)


def without_bets(population: pd.DataFrame, bets: pd.DataFrame) -> pd.DataFrame:
    """The population without the selections that were bet on."""
    # a selection bet twice would repeat only rows that are dropped below anyway
    taken = bets[KEY].assign(_taken=True)
    joined = population.merge(taken, on=KEY, how="left")
    return joined[joined["_taken"].isna()].drop(columns="_taken").reset_index(drop=True)


def similar_drift(
    bets: pd.DataFrame, population: pd.DataFrame, width: float = CONTROL_WIDTH
) -> FloatArray:
    """Mean probability drift of the quotes priced like each bet.

    For each bet: the average ``p_close - p_early`` over the population's
    selections of the same market whose early probability is within ``width`` of
    the bet's reference probability (both ends included). NaN where no selection
    is that close.
    """
    drifts = np.full(len(bets), np.nan)
    wanted = bets["p_reference"].to_numpy(dtype="float64")
    for market_id, rows in bets.groupby("market_id").indices.items():
        pool = population[population["market_id"] == market_id]
        if pool.empty:
            continue
        early = pool["p_early"].to_numpy(dtype="float64")
        order = np.argsort(early, kind="stable")
        moves = (pool["p_close"] - pool["p_early"]).to_numpy(dtype="float64")[order]
        cumulative = np.concatenate([[0.0], np.cumsum(moves)])
        target = wanted[rows]
        low = np.searchsorted(early[order], target - width, side="left")
        high = np.searchsorted(early[order], target + width, side="right")
        count = high - low
        drifts[rows] = np.where(
            count > 0,
            (cumulative[high] - cumulative[low]) / np.maximum(count, 1),
            np.nan,
        )
    return drifts


def quote_table(
    enriched: pd.DataFrame,
    population: pd.DataFrame,
    book: str,
    config: ValueConfig | None = None,
) -> pd.DataFrame:
    """Every early price of ``book`` the engine could have bet, with its gap and drift.

    These are all the prices within the engine's limits (odds, and a sharp-book
    probability of at least ``min_probability``), not only the ones it bet on. A
    price is kept when the sharp book also quoted its market early and at the
    close.

    Args:
        enriched: The match table with the wide odds columns.
        population: From :func:`control_population`.
        book: The bookmaker whose prices are read.
        config: The engine's limits; defaults to :class:`src.value.ValueConfig`.

    Returns:
        :data:`KEY`, ``season``, ``batch``, ``odds``, ``p_early``, ``p_close`` and,
        per price, ``gap`` (``odds * p_early - 1``), ``drift``
        (``odds * (p_close - p_early)``) and ``clv`` (their sum).
    """
    config = config or ValueConfig()
    prices = odds_long(enriched, "early", (book,))[[*KEY, "odds"]]
    quotes = population.merge(prices, on=KEY, how="inner", validate="one_to_one")
    odds = quotes["odds"].to_numpy(dtype="float64")
    before = quotes["p_early"].to_numpy(dtype="float64")
    after = quotes["p_close"].to_numpy(dtype="float64")
    within = (
        (odds >= config.min_odds)
        & (odds <= config.max_odds)
        & (before >= config.min_probability)
    )
    gap = odds[within] * before[within] - 1.0
    clv = odds[within] * after[within] - 1.0
    return (
        quotes.loc[within]
        .reset_index(drop=True)
        .assign(gap=gap, drift=clv - gap, clv=clv)
    )


# --------------------------------------------------------------------------- #
# The tables
# --------------------------------------------------------------------------- #
def _weekly(
    values: FloatArray,
    weights: FloatArray,
    batches: pd.Series,
    calendar: pd.DatetimeIndex,
) -> tuple[FloatArray, FloatArray]:
    """Per week of the calendar: the sum of ``values * weights`` and of ``weights``."""
    totals = (
        pd.DataFrame(
            {"batch": _stamps(batches), "top": values * weights, "bottom": weights}
        )
        .groupby("batch", sort=True)[["top", "bottom"]]
        .sum()
        .reindex(calendar, fill_value=0.0)
    )
    return (
        totals["top"].to_numpy(dtype="float64"),
        totals["bottom"].to_numpy(dtype="float64"),
    )


def _within(batches: pd.Series, calendar: pd.DatetimeIndex, what: str) -> None:
    """Refuse ``what`` in weeks outside the calendar: the intervals would leave them out."""
    outside = ~_stamps(batches).isin(calendar)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} {what} fall in weeks outside the calendar"
        )


def decomposition(
    bets: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Each part of the CLV, per unit staked and per bet, with week-clustered intervals.

    Args:
        bets: From :func:`strategy_bets`, with ``typical`` and ``specific`` when the
            comparison with similar quotes was possible.
        calendar: Every week of the backtest (see :func:`_weeks`); each week with a
            bet must be in it.
        n_resamples: Number of resamples of weeks.
        seed: Seed of the resampling; every row uses the same weeks.

    Returns:
        :data:`PART_COLUMNS`; ``weighting`` is ``"stake"`` or ``"bet"`` and ``part``
        a key of :data:`PART_LABELS`. The similar quotes' drift (``typical``) has no
        interval: it is estimated on all the quotes of the population and taken
        as known.

    Raises:
        ValueError: If a bet falls in a week outside the calendar.
    """
    _within(bets["batch"], calendar, "bets")
    rows = []
    for weighting, weights in (
        ("stake", bets["stake"].to_numpy(dtype="float64")),
        ("bet", np.ones(len(bets))),
    ):
        for part in PART_LABELS:
            if part not in bets.columns:
                continue
            values = bets[part].to_numpy(dtype="float64")
            top, bottom = _weekly(values, weights, bets["batch"], calendar)
            total = float(bottom.sum())
            value = float(top.sum() / total) if total > 0.0 else float("nan")
            if part == "typical":
                low = high = float("nan")
            else:
                low, high = cluster_bootstrap(
                    top, bottom, n_resamples=n_resamples, seed=seed
                )
            rows.append(
                {
                    "weighting": weighting,
                    "part": part,
                    "value": value,
                    "low": low,
                    "high": high,
                }
            )
    return pd.DataFrame(rows, columns=PART_COLUMNS)


def concentration(bets: pd.DataFrame, trims: Sequence[int] = TRIMS) -> pd.DataFrame:
    """The mean and median CLV per bet with the best bets taken away.

    ``share`` is the part of the total CLV held by the bets removed; it is NaN when
    nothing is removed or when the total is not positive. Trims that would leave
    no bet are left out. These are descriptive: which bets to remove is chosen after
    the fact, so no interval is given.
    """
    clv = np.sort(bets["clv"].to_numpy(dtype="float64"))[::-1]
    total = float(clv.sum())
    rows = []
    for removed in trims:
        if removed >= len(clv):
            continue
        left = clv[removed:]
        share = (
            float(clv[:removed].sum() / total)
            if removed > 0 and total > 0.0
            else float("nan")
        )
        rows.append(
            {
                "removed": removed,
                "left": len(left),
                "mean_clv": float(left.mean()),
                "median_clv": float(np.median(left)),
                "share": share,
            }
        )
    return pd.DataFrame(rows, columns=CONCENTRATION_COLUMNS)


def breakdown(bets: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per group of ``by``: bets, mean price gap, drift and CLV, hit rate and expected hit rate.

    ``hit`` is the share of bets that won and ``expected`` the mean closing
    probability of the selections bet on. Groups come in the order of ``by``
    (price ranges in their natural order). No intervals: groups are small.
    """
    grouped = bets.groupby(by, sort=True, observed=True)
    table = grouped.agg(
        bets=("clv", "size"),
        gap=("gap", "mean"),
        drift=("drift", "mean"),
        clv=("clv", "mean"),
        hit=("won", "mean"),
        expected=("p_close", "mean"),
    ).reset_index()
    return table[[by, *BREAKDOWN_VALUES]]


def gap_distribution(bets: pd.DataFrame) -> pd.DataFrame:
    """Lowest, quartiles, 90th percentile and highest price gap."""
    gaps = bets["gap"].to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            "statistic": [label for label, _ in GAP_STATISTICS],
            "gap": [float(np.quantile(gaps, q)) for _, q in GAP_STATISTICS],
        },
        columns=GAP_COLUMNS,
    )


def _share(value: float) -> str:
    """A fraction as a short percentage: ``-0.1`` is ``-10%`` and ``0.025`` is ``2.5%``."""
    return f"{100.0 * value:g}%"


def gap_ranges(gap: pd.Series, min_edge: float) -> pd.Series:
    """Label each price gap with its range; the last range starts at ``min_edge``.

    A range includes its lower end. The ranges below ``min_edge`` are those of
    :data:`GAP_BREAKS` that lie under it, so the last range, ``"<min_edge> or
    more"``, is exactly the prices the engine may bet on at model weight 0. The
    result is an ordered categorical.
    """
    edges = [edge for edge in GAP_BREAKS if edge < min_edge] + [min_edge]
    labels = [f"below {_share(edges[0])}"]
    labels += [
        f"{_share(low)} to {_share(high)}" for low, high in itertools.pairwise(edges)
    ]
    labels.append(f"{_share(edges[-1])} or more")
    return pd.cut(gap, bins=[-np.inf, *edges, np.inf], labels=labels, right=False)


def response_table(
    quotes: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    min_edge: float,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """What the sharp book did next, for every price of a book, by how far the price sat above it.

    Args:
        quotes: From :func:`quote_table`.
        calendar: Every week of the backtest (see :func:`_weeks`).
        min_edge: The engine's minimum edge, where the last range starts.
        n_resamples: Number of resamples of weeks for the intervals.
        seed: Seed of the resampling.

    Returns:
        :data:`RESPONSE_COLUMNS`: per gap range with prices, the number of
        ``quotes``, the mean price ``gap``, and the mean ``drift`` and ``clv`` per
        price with week-clustered 95% intervals. ``region`` marks the last range.

    Raises:
        ValueError: If a price falls in a week outside the calendar.
    """
    _within(quotes["batch"], calendar, "prices")
    ranges = gap_ranges(quotes["gap"], min_edge)
    labels = list(ranges.cat.categories)
    rows = []
    for position, label in enumerate(labels):
        chosen = quotes[(ranges == label).to_numpy()]
        if chosen.empty:
            continue
        ones = np.ones(len(chosen))
        row: dict[str, object] = {
            "gap_range": label,
            "region": position == len(labels) - 1,
            "quotes": len(chosen),
            "gap": float(chosen["gap"].mean()),
        }
        for column in ("drift", "clv"):
            values = chosen[column].to_numpy(dtype="float64")
            top, bottom = _weekly(values, ones, chosen["batch"], calendar)
            low, high = cluster_bootstrap(
                top, bottom, n_resamples=n_resamples, seed=seed
            )
            row.update(
                {
                    column: float(values.mean()),
                    f"{column}_low": low,
                    f"{column}_high": high,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=RESPONSE_COLUMNS)


def best_bets(bets: pd.DataFrame, top: int = DEFAULT_TOP) -> pd.DataFrame:
    """The ``top`` bets with the highest CLV, with the match named when the names are known."""
    ordered = bets.sort_values("clv", ascending=False, kind="stable").head(top)
    if {"home_team", "away_team"} <= set(bets.columns):
        match = (
            ordered["home_team"].astype(str) + " v " + ordered["away_team"].astype(str)
        )
    else:
        match = ordered["match_id"].astype(str)
    table = ordered.assign(
        match=match,
        date=pd.to_datetime(ordered["date"]).dt.strftime("%Y-%m-%d"),
    )
    if "league" not in table.columns:
        table = table.assign(league="")
    return table[BEST_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Autopsy:
    """A strategy's closing-line value taken apart.

    Attributes:
        book: The bookmaker the strategy bets at.
        weight: The strategy's model weight.
        bets: The strategy's bets with ``gap``, ``drift`` and, when compared,
            ``typical`` and ``specific``, ``odds_range`` and, when known, the
            league and team names.
        weeks: Number of weeks of the calendar the intervals resample.
        compared: Whether the drift was compared with similar quotes.
        similar: Number of quotes the comparison draws on (0 when not compared).
        config: The engine's limits, which also bound the prices of ``response``.
        parts: See :func:`decomposition`.
        concentration: See :func:`concentration`.
        seasons: Means per season (see :func:`breakdown`); likewise for ``leagues``,
            ``markets`` and ``odds_ranges``.
        gaps: See :func:`gap_distribution`.
        response: See :func:`response_table`; empty when the match table was not
            given or did not match the backtest.
        best: See :func:`best_bets`.
    """

    book: str
    weight: float
    bets: pd.DataFrame
    weeks: int
    compared: bool
    similar: int
    config: ValueConfig
    parts: pd.DataFrame
    concentration: pd.DataFrame
    seasons: pd.DataFrame
    leagues: pd.DataFrame
    markets: pd.DataFrame
    odds_ranges: pd.DataFrame
    gaps: pd.DataFrame
    response: pd.DataFrame
    best: pd.DataFrame


def autopsy(
    bets: pd.DataFrame,
    *,
    book: str = DEFAULT_BOOK,
    weight: float = DEFAULT_WEIGHT,
    matches: pd.DataFrame | None = None,
    population: pd.DataFrame | None = None,
    config: ValueConfig | None = None,
    top: int = DEFAULT_TOP,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Autopsy:
    """Take apart the CLV of one strategy.

    Args:
        bets: The ``bets.parquet`` of a backtest run, every strategy in it.
        book: The bookmaker the strategy bets at.
        weight: The strategy's model weight.
        matches: The match table; names the matches and gives each its league.
        population: From :func:`control_population`; splits the drift into what
            similar quotes drifted by and what is left, and gives the calendar of
            the backtest. Skipped, with a warning, when it does not reproduce the
            bets' probabilities.
        config: The engine's limits as the backtest ran them; defaults to
            :class:`src.value.ValueConfig`.
        top: Number of best bets to list.
        n_resamples: Number of resamples of weeks for the intervals.
        seed: Seed of the resampling.

    Returns:
        The :class:`Autopsy`.

    Raises:
        ValueError: If the strategy has no bets or the table is not a backtest's.
    """
    config = config or ValueConfig()
    chosen = strategy_bets(bets, book, weight)
    if matches is not None:
        chosen = attach_matches(chosen, matches)
    if "league" in chosen.columns:
        chosen = chosen.assign(league=chosen["league"].fillna("unknown"))
    calendar = _weeks(bets["batch"])
    compared, similar = False, 0
    response = pd.DataFrame(columns=RESPONSE_COLUMNS)
    if population is not None:
        if reproduces(chosen, population):
            calendar = calendar.union(_weeks(population["batch"]))
            others = without_bets(population, chosen)
            typical = chosen["odds"].to_numpy(dtype="float64") * similar_drift(
                chosen, others
            )
            if np.isfinite(typical).all():
                chosen = chosen.assign(
                    typical=typical, specific=chosen["drift"] - typical
                )
                compared, similar = True, len(others)
            else:
                logger.warning(
                    "Some bets have no similarly priced quote; the comparison is "
                    "left out"
                )
            if matches is not None:
                quotes = quote_table(matches, population, book, config)
                if not quotes.empty:
                    response = response_table(
                        quotes,
                        calendar,
                        min_edge=config.min_edge,
                        n_resamples=n_resamples,
                        seed=seed,
                    )
        else:
            logger.warning(
                "The match table does not reproduce the bets' probabilities, so the "
                "comparison with similar quotes is left out; pass the --reference "
                "and --demargin of the backtest that made them"
            )
    chosen = chosen.assign(odds_range=odds_band(chosen["odds"]))
    leagues = (
        breakdown(chosen, "league")
        if "league" in chosen.columns
        else pd.DataFrame(columns=["league", *BREAKDOWN_VALUES])
    )
    return Autopsy(
        book=book,
        weight=weight,
        bets=chosen,
        weeks=len(calendar),
        compared=compared,
        similar=similar,
        config=config,
        parts=decomposition(chosen, calendar, n_resamples=n_resamples, seed=seed),
        concentration=concentration(chosen),
        seasons=breakdown(chosen, "season"),
        leagues=leagues,
        markets=breakdown(chosen, "market_id"),
        odds_ranges=breakdown(chosen, "odds_range"),
        gaps=gap_distribution(chosen),
        response=response,
        best=best_bets(chosen, top),
    )


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #
def _percent(value: float, *, signed: bool = False, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "-"
    text = f"{100.0 * value:+.{digits}f}" if signed else f"{100.0 * value:.{digits}f}"
    return f"{text}%"


def _interval(low: float, high: float) -> str:
    if not (np.isfinite(low) and np.isfinite(high)):
        return "-"
    return f"[{_percent(low, signed=True)}, {_percent(high, signed=True)}]"


def _render(table: pd.DataFrame, left: Sequence[str] = ()) -> str:
    """Aligned text: the columns named in ``left`` flush left, the others flush right."""
    cells = table.astype(str)
    names = [str(column) for column in cells.columns]
    widths = [
        max([len(name), *(len(cell) for cell in cells.iloc[:, position])])
        for position, name in enumerate(names)
    ]

    def row(values: Sequence[str]) -> str:
        padded = [
            value.ljust(width) if name in left else value.rjust(width)
            for name, value, width in zip(names, values, widths)
        ]
        return "  ".join(padded).rstrip()

    lines = [row(names)]
    lines += [row(values) for values in cells.itertuples(index=False, name=None)]
    return "\n".join(lines)


def _part(parts: pd.DataFrame, weighting: str, key: str) -> pd.Series | None:
    rows = parts[(parts["weighting"] == weighting) & (parts["part"] == key)]
    return rows.iloc[0] if len(rows) else None


def _shown(table: pd.DataFrame, label: str, key: str) -> str:
    """A breakdown with percentages ready to print."""
    return _render(
        pd.DataFrame(
            {
                label: table[key].astype(str),
                "bets": table["bets"],
                "price gap": table["gap"].map(lambda v: _percent(v, signed=True)),
                "drift": table["drift"].map(lambda v: _percent(v, signed=True)),
                "CLV": table["clv"].map(lambda v: _percent(v, signed=True)),
                "won": table["hit"].map(_percent),
                "expected": table["expected"].map(_percent),
            }
        ),
        left=[label],
    )


def format_autopsy(result: Autopsy) -> str:
    """The autopsy as text: the parts of the CLV, concentration, breakdowns and best bets."""
    bets = result.bets
    first, last = int(bets["season"].min()), int(bets["season"].max())
    parts = result.parts
    shown_parts = pd.DataFrame(
        {
            "weighted by": parts["weighting"],
            "part": parts["part"].map(PART_LABELS),
            "value": parts["value"].map(lambda v: _percent(v, signed=True)),
            "95% interval": [
                _interval(a, b) for a, b in zip(parts["low"], parts["high"])
            ],
        }
    )
    kept = result.concentration
    shown_kept = pd.DataFrame(
        {
            "best bets removed": kept["removed"],
            "bets left": kept["left"],
            "mean CLV": kept["mean_clv"].map(lambda v: _percent(v, signed=True)),
            "median CLV": kept["median_clv"].map(lambda v: _percent(v, signed=True)),
            "share of all CLV in them": kept["share"].map(
                lambda v: _percent(v, digits=0)
            ),
        }
    )
    gaps = pd.DataFrame(
        {
            "statistic": result.gaps["statistic"],
            "price gap": result.gaps["gap"].map(lambda v: _percent(v, signed=True)),
        }
    )
    best = result.best
    shown_best = pd.DataFrame(
        {
            "season": best["season"],
            "date": best["date"],
            "league": best["league"],
            "match": best["match"],
            "market": best["market_id"],
            "selection": best["selection"],
            "odds": best["odds"].map("{:.2f}".format),
            "p early": best["p_reference"].map(_percent),
            "p close": best["p_close"].map(_percent),
            "price gap": best["gap"].map(lambda v: _percent(v, signed=True)),
            "CLV": best["clv"].map(lambda v: _percent(v, signed=True)),
            "result": best["won"].map(lambda won: "won" if won else "lost"),
        }
    )
    if not best["league"].astype(str).str.len().any():
        shown_best = shown_best.drop(columns="league")
    lines = [
        (
            f"Autopsy of {_book_name(result.book)} at model weight {result.weight:.2f}: "
            f"{len(bets)} bets, seasons {first}-{last}, over {result.weeks} weeks"
        ),
        "",
        (
            "What the closing-line value is made of. Price gap: odds x the sharp book's "
            "early fair probability - 1, what you saw when you bet. Market drift: odds x "
            "the change in that probability by the close. CLV is their sum. 'weighted by "
            "stake' is the backtest's CLV; 'weighted by bet' counts every bet alike."
        ),
        _render(shown_parts, left=["weighted by", "part"]),
    ]
    if result.compared:
        lines += [
            "",
            (
                "Similar quotes: for each bet, the other quotes of the same market whose "
                f"early probability is within {100 * CONTROL_WIDTH:.0f} points of the "
                f"bet's, from {result.similar} quotes of the sharp book in the same "
                "seasons. Their average drift is taken as known."
            ),
        ]
    lines += [
        "",
        (
            "Do a few bets carry it? Per bet, with the best bets taken away "
            "(chosen after the fact, so no interval)."
        ),
        _render(shown_kept),
    ]
    for title, label, table, key in (
        ("By season", "season", result.seasons, "season"),
        ("By league", "league", result.leagues, "league"),
        ("By market", "market", result.markets, "market_id"),
        ("By price range", "price range", result.odds_ranges, "odds_range"),
    ):
        if table.empty:
            continue
        lines += ["", f"{title}: means per bet (no intervals, the groups are small)"]
        lines.append(_shown(table, label, key))
    lines += [
        "",
        (
            "How large the price gaps are. A gap above 50% against a sharp book is more "
            "often a data or timing error than an opportunity."
        ),
        _render(gaps, left=["statistic"]),
    ]
    response = result.response
    if not response.empty:
        config = result.config
        shown_response = pd.DataFrame(
            {
                "price gap range": response["gap_range"],
                "prices": response["quotes"],
                "price gap": response["gap"].map(lambda v: _percent(v, signed=True)),
                "drift": response["drift"].map(lambda v: _percent(v, signed=True)),
                "drift 95% interval": [
                    _interval(a, b)
                    for a, b in zip(response["drift_low"], response["drift_high"])
                ],
                "CLV": response["clv"].map(lambda v: _percent(v, signed=True)),
                "CLV 95% interval": [
                    _interval(a, b)
                    for a, b in zip(response["clv_low"], response["clv_high"])
                ],
            }
        )
        lines += [
            "",
            (
                f"Every {_book_name(result.book)} price the engine could have bet, not "
                f"only the bets (odds {config.min_odds:.2f} to {config.max_odds:.2f}, the "
                f"sharp book's probability at least {config.min_probability:.0%}), by how "
                "far it sat above the sharp book's early price. Drift and CLV are means "
                "per price. The last range is where the engine bets at weight 0."
            ),
            _render(shown_response, left=["price gap range"]),
        ]
    lines += [
        "",
        (
            f"The {len(best)} bets with the highest CLV (p early, p close: the sharp "
            "book's fair probabilities)"
        ),
        _render(shown_best, left=["league", "match", "market", "selection", "result"]),
    ]
    return "\n".join(lines)


def verdict(result: Autopsy) -> list[str]:
    """What the autopsy shows, one finding per line, without recommending a bet."""
    lines: list[str] = []
    bets = result.bets
    count = len(bets)
    name = _book_name(result.book)
    clv = _part(result.parts, "stake", "clv")
    gap = _part(result.parts, "stake", "gap")
    drift = _part(result.parts, "stake", "drift")
    if clv is None or gap is None or drift is None:
        return lines
    lines.append(
        f"{name} at weight {result.weight:.2f}: CLV per unit staked is "
        f"{_percent(clv['value'], signed=True)} {_interval(clv['low'], clv['high'])}, "
        f"a price gap of {_percent(gap['value'], signed=True)} plus a market drift of "
        f"{_percent(drift['value'], signed=True)} {_interval(drift['low'], drift['high'])}"
    )
    if drift["low"] > 0.0:
        reading = "the sharp book kept moving in the bettor's favour after the price was taken"
    elif drift["high"] < 0.0:
        reading = (
            "the sharp book moved against the bettor after the price was taken, so "
            "the price gap overstates what the close confirms"
        )
    else:
        reading = (
            "the sharp book did not move for or against the bettor by more than the "
            "data can tell from zero, so the CLV is mostly the price gap itself"
        )
    lines.append(f"Drift: {reading}")
    typical = _part(result.parts, "stake", "typical")
    specific = _part(result.parts, "stake", "specific")
    if typical is not None and specific is not None:
        if specific["low"] > 0.0:
            beyond = (
                "more than the pattern of similar quotes explains: the sharp book "
                "kept moving in the bettor's favour. That is how a slow "
                f"{name} price looks, or two prices recorded at different moments; "
                "the data cannot tell which"
            )
        elif specific["high"] < 0.0:
            beyond = (
                "less than the pattern of similar quotes: the sharp book moved "
                f"towards {name}'s view, so part of the price gap was information "
                "rather than mispricing"
            )
        else:
            beyond = (
                "nothing the data can tell apart from the pattern of similar "
                "quotes: the price gap behaves like noise the sharp book ignored, "
                "which is how a price-shopping edge looks"
            )
        lines.append(
            f"Quotes priced like these drifted {_percent(typical['value'], signed=True)} "
            f"on average, so {_percent(specific['value'], signed=True)} "
            f"{_interval(specific['low'], specific['high'])} of the drift is specific "
            f"to these bets: {beyond}"
        )
    else:
        lines.append(
            "The drift was not compared with similar quotes (no match table, or one "
            "that does not match this backtest's settings), so a pattern of the whole "
            "market, such as long shots losing probability towards the close, cannot "
            "be told from a property of the bets"
        )
    response = result.response
    region = response[response["region"].astype(bool)] if len(response) else response
    if len(region):
        top = region.iloc[0]
        if top["drift_low"] > 0.0:
            moved = "the sharp book moved in the bettor's favour"
        elif top["drift_high"] < 0.0:
            moved = "the sharp book moved against the bettor"
        else:
            moved = "the sharp book's move cannot be told from zero"
        lines.append(
            f"Across all {int(top['quotes'])} {name} prices the engine could have bet "
            f"that sat at least {_share(result.config.min_edge)} above the sharp "
            f"book's early price (not only the {count} bets), the mean CLV per price "
            f"is {_percent(top['clv'], signed=True)} "
            f"{_interval(top['clv_low'], top['clv_high'])} and the mean drift "
            f"{_percent(top['drift'], signed=True)} "
            f"{_interval(top['drift_low'], top['drift_high'])}: {moved}"
        )
    kept = result.concentration
    trimmed = kept[kept["removed"] == VERDICT_TRIM]
    if len(trimmed):
        row = trimmed.iloc[0]
        everything = kept[kept["removed"] == 0].iloc[0]
        held = (
            f"; those {VERDICT_TRIM} bets hold {_percent(row['share'], digits=0)} of "
            "all CLV"
            if np.isfinite(row["share"])
            else ""
        )
        outcome = (
            "it stays positive" if row["mean_clv"] > 0.0 else "it does not survive"
        )
        lines.append(
            f"Without its {VERDICT_TRIM} best bets the mean CLV per bet is "
            f"{_percent(row['mean_clv'], signed=True)} (all bets: "
            f"{_percent(everything['mean_clv'], signed=True)}){held}: {outcome}"
        )
    else:
        lines.append(f"Only {count} bets: too few to take the best {VERDICT_TRIM} away")
    seasons = result.seasons
    positive = int((seasons["clv"] > 0.0).sum())
    weakest = seasons.iloc[int(np.argmin(seasons["clv"].to_numpy(dtype="float64")))]
    lines.append(
        f"CLV per bet is positive in {positive} of {len(seasons)} seasons; the weakest "
        f"is {int(weakest['season'])} with {_percent(weakest['clv'], signed=True)} on "
        f"{int(weakest['bets'])} bets"
    )
    largest = float(bets["gap"].max())
    implausible = int((bets["gap"] > IMPLAUSIBLE_GAP).sum())
    if implausible:
        lines.append(
            f"{implausible} of the bets have a price gap above {IMPLAUSIBLE_GAP:.0%} "
            f"(the largest is {_percent(largest, signed=True)}): against a sharp "
            "book that is more often a data or timing error than an opportunity, so "
            "check them in the list of best bets"
        )
    else:
        lines.append(
            f"No bet has a price gap above {IMPLAUSIBLE_GAP:.0%}; the largest is "
            f"{_percent(largest, signed=True)}"
        )
    wins = int(bets["won"].sum())
    probabilities = bets["p_close"].to_numpy(dtype="float64")
    spread = float(np.sqrt((probabilities * (1.0 - probabilities)).sum()))
    if spread > 0.0:
        expected = float(probabilities.sum())
        lines.append(
            f"The bets won {wins} times against {expected:.1f} expected from the "
            f"closing prices (z = {(wins - expected) / spread:+.1f})"
        )
    if count < THIN_SAMPLE:
        lines.append(
            f"{count} bets is a small sample: the intervals allow for that, but the "
            "splits by season, league and price range rest on a handful of bets each"
        )
    return lines


def save_autopsy(result: Autopsy, directory: Path) -> list[Path]:
    """Write the autopsy's tables to ``directory`` and return the files written."""
    directory.mkdir(parents=True, exist_ok=True)
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
    written = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path, index=False)
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
        description="Take a strategy's closing-line value apart."
    )
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=BACKTEST_DIR,
        help="folder of the backtest run, with bets.parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--enriched",
        type=Path,
        default=ENRICHED_OUTPUT,
        help="match table: names the matches and gives the comparison with similar "
        "quotes (optional)",
    )
    parser.add_argument("--book", choices=BOOKS, default=DEFAULT_BOOK)
    parser.add_argument("--weight", type=_weight, default=DEFAULT_WEIGHT)
    parser.add_argument(
        "--reference",
        choices=FAIR_BOOKS,
        default=REFERENCE_BOOK,
        help="sharp book of the backtest that made the bets (default: %(default)s)",
    )
    parser.add_argument(
        "--demargin",
        choices=METHODS,
        default=DEMARGIN,
        help="margin removal of the backtest that made the bets (default: %(default)s)",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=ValueConfig().min_edge,
        help="minimum edge of the backtest that made the bets (default: %(default)s)",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the tables (default: <backtest-dir>/autopsy/<book>_weight<w>)",
    )
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")
    if args.top < 1:
        parser.error("--top must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the autopsy and return the process exit code.

    Returns ``0`` on success and ``1`` when the backtest's bets are missing or
    unusable or the strategy has no bets.
    """
    args = _parse_args(argv)
    log = setup_logging("autopsy")
    path = args.backtest_dir / "bets.parquet"
    if not path.exists():
        log.error("Not found: %s; run python -m src.backtest first", path)
        return 1
    try:
        bets = pd.read_parquet(path)
        _require(bets, BET_REQUIRED, "the bets table")
        matches = None
        population = None
        if args.enriched.exists():
            matches = pd.read_parquet(args.enriched)
            population = control_population(
                matches,
                book=args.reference,
                demargin=args.demargin,
                first_season=int(bets["season"].min()),
            )
        else:
            log.info(
                "No match table at %s: matches are shown by id and the comparison "
                "with similar quotes is left out",
                args.enriched,
            )
        result = autopsy(
            bets,
            book=args.book,
            weight=args.weight,
            matches=matches,
            population=population,
            config=ValueConfig(min_edge=args.min_edge),
            top=args.top,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except (ValueError, KeyError) as exc:
        log.error("Cannot take the strategy apart: %s", exc)
        return 1
    log.info("\n%s", format_autopsy(result))
    log.info("Verdict")
    for line in verdict(result):
        log.info(line)
    directory = args.output_dir or (
        args.backtest_dir / AUTOPSY_FOLDER / f"{args.book}_weight{args.weight:g}"
    )
    for written in save_autopsy(result, directory):
        log.info("Wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
