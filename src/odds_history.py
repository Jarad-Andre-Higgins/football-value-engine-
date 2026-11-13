"""Historical bookmaker odds, in the shape the betting engine reads.

football-data.co.uk stores every bookmaker's prices as wide columns: ``ps_h`` is
Pinnacle's early price for a home win and ``psc_h`` its closing price (names in
:mod:`src.schema`). The betting engine reads long tables instead, one row per
price. This module converts the one into the other, takes the margin out of a
price, settles a bet from the final score, and audits what the data can test.

Every match has two snapshots. ``early`` is the price recorded before the closing
one (the files do not say when, so how far ahead of kick-off it was cannot be
checked here) and ``close`` is the last price before kick-off. The gap between
them is where closing-line value is measured: a bet taken at the early price was
a good one if the market later moved toward it.

Only two markets have history: 1X2 and over/under 2.5 goals. A backtest can
price those two, even though the engine prices 55 markets live.

Usage::

    python -m src.odds_history
    python -m src.odds_history --from-season 2023 --book avg
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from src.blend import paired_bootstrap_interval
from src.enrich import ENRICHED_OUTPUT
from src.logging_config import setup_logging
from src.markets import market_set
from src.value import reference_probabilities

logger = logging.getLogger("src.odds_history")

SNAPSHOTS: Final[tuple[str, ...]] = ("early", "close")
BOOKS: Final[tuple[str, ...]] = ("ps", "b365", "avg", "max")
#: Books that give a margin-free view of the market, sharpest first. "max" is the
#: best price among many bookmakers, so its margin is sometimes negative and it
#: is never a fair view.
FAIR_BOOKS: Final[tuple[str, ...]] = ("ps", "avg", "b365")
BOOK_NAMES: Final[dict[str, str]] = {
    "ps": "Pinnacle",
    "b365": "Bet365",
    "avg": "market average",
    "max": "market best",
}
#: market id -> (selection, column suffix) pairs; the ids are those of :mod:`src.markets`.
MARKETS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "1X2": (("home", "h"), ("draw", "d"), ("away", "a")),
    "OU:2.5": (("over", "over25"), ("under", "under25")),
}
KEY: Final[list[str]] = ["match_id", "market_id", "selection"]
OFFER_COLUMNS: Final[list[str]] = [
    "match_id",
    "bookmaker",
    "market_id",
    "selection",
    "odds",
]
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "season",
    "home_goals",
    "away_goals",
)
DEFAULT_FROM_SEASON: Final[int] = 2022  # the first season with walk-forward predictions


# --------------------------------------------------------------------------- #
# Wide columns to long offers
# --------------------------------------------------------------------------- #
def _check_snapshot(snapshot: str) -> None:
    if snapshot not in SNAPSHOTS:
        raise ValueError(f"snapshot must be one of {SNAPSHOTS}, got {snapshot!r}")


def price_column(book: str, snapshot: str, suffix: str) -> str:
    """Name of one price column, for example ``psc_h`` (Pinnacle, close, home win).

    Raises:
        ValueError: If ``snapshot`` is neither ``"early"`` nor ``"close"``.
    """
    _check_snapshot(snapshot)
    prefix = f"{book}c" if snapshot == "close" else book
    return f"{prefix}_{suffix}"


def _empty_offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": pd.Series(dtype="object"),
            "bookmaker": pd.Series(dtype="object"),
            "market_id": pd.Series(dtype="object"),
            "selection": pd.Series(dtype="object"),
            "odds": pd.Series(dtype="float64"),
        }
    )


def odds_long(
    enriched: pd.DataFrame, snapshot: str, books: Sequence[str] = BOOKS
) -> pd.DataFrame:
    """One row per bookmaker price, the shape :func:`src.value.find_value_bets` reads.

    A bookmaker's quote of a market is kept only when every selection of it is
    present and above 1.0. A market with a missing or impossible price cannot be
    de-margined, so it is left out whole rather than half used. A book or market
    with no columns in ``enriched`` is skipped.

    Args:
        enriched: The enriched match table, with ``match_id`` and wide odds columns.
        snapshot: ``"early"`` or ``"close"``.
        books: Bookmaker codes to read (see :data:`BOOKS`).

    Returns:
        ``match_id``, ``bookmaker``, ``market_id``, ``selection`` and ``odds``.

    Raises:
        ValueError: If ``snapshot`` is not one of :data:`SNAPSHOTS`.
    """
    _check_snapshot(snapshot)
    frames: list[pd.DataFrame] = []
    for book in books:
        for market_id, selections in MARKETS.items():
            columns = [price_column(book, snapshot, suffix) for _, suffix in selections]
            if not all(column in enriched.columns for column in columns):
                continue
            quotes = (
                enriched[columns]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype="float64")
            )
            with np.errstate(invalid="ignore"):
                complete = np.isfinite(quotes).all(axis=1) & (quotes > 1.0).all(axis=1)
            for position, (selection, _) in enumerate(selections):
                frames.append(
                    pd.DataFrame(
                        {
                            "match_id": enriched.loc[complete, "match_id"].to_numpy(),
                            "bookmaker": book,
                            "market_id": market_id,
                            "selection": selection,
                            "odds": quotes[complete, position],
                        }
                    )
                )
    if not frames:
        return _empty_offers()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Taking the margin out
# --------------------------------------------------------------------------- #
def fair_probabilities(offers: pd.DataFrame) -> pd.Series:
    """Margin-free probability of every offer.

    Each bookmaker's market is scaled so its inverse odds add up to one, which
    removes the margin in proportion to each price (the same rule the value
    engine uses, :func:`src.value.reference_probabilities`).

    Args:
        offers: Prices from :func:`odds_long`.

    Returns:
        A probability per offer, aligned to ``offers``.
    """
    return reference_probabilities(offers.assign(push_probability=0.0))


def _fair_quotes(
    enriched: pd.DataFrame, snapshot: str, books: Sequence[str]
) -> pd.DataFrame:
    """Complete quotes of the given books, each with its margin-free probability."""
    quotes = odds_long(enriched, snapshot, books)
    if quotes.empty:
        return quotes.assign(p_fair=pd.Series(dtype="float64"))
    quotes["p_fair"] = fair_probabilities(quotes).to_numpy(dtype="float64")
    return quotes.dropna(subset=["p_fair"]).reset_index(drop=True)


def fair_view(
    enriched: pd.DataFrame, snapshot: str, books: Sequence[str] = FAIR_BOOKS
) -> pd.DataFrame:
    """The sharpest margin-free probability available for every selection.

    Books are tried in the order given (Pinnacle first by default). A bookmaker's
    quote is complete or absent, so every selection of a match's market comes
    from the same book and the probabilities still add up to one.

    Returns:
        ``match_id``, ``market_id``, ``selection``, ``source`` (the book used) and
        ``p_fair``, sorted by those keys.
    """
    quotes = _fair_quotes(enriched, snapshot, books)
    rank = {book: position for position, book in enumerate(books)}
    best = (
        quotes.assign(_rank=quotes["bookmaker"].map(rank))
        .sort_values("_rank", kind="stable")
        .drop_duplicates(KEY)
        .rename(columns={"bookmaker": "source"})
    )
    return best[[*KEY, "source", "p_fair"]].sort_values(KEY).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Settling bets
# --------------------------------------------------------------------------- #
def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def settle(offers: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Work out from the final score whether each offer won, was refunded or lost.

    The winning squares are the ones that priced the market (:mod:`src.markets`),
    so a bet is settled by the same rules it was priced with. An offer is left
    unsettled when its match has no usable score (missing, negative, fractional
    or beyond the score grid) or its market is unknown.

    Args:
        offers: Needs ``match_id``, ``market_id``, ``selection`` and ``odds``.
        results: One row per match with ``match_id``, ``home_goals`` and
            ``away_goals``.

    Returns:
        A copy of ``offers`` with ``settled``, ``won``, ``push`` (stake refunded)
        and ``profit`` per unit staked (NaN when unsettled).

    Raises:
        ValueError: If columns are missing or a match has several results.
    """
    _require(offers, ("match_id", "market_id", "selection", "odds"), "offers")
    _require(results, ("match_id", "home_goals", "away_goals"), "results")
    if results["match_id"].duplicated().any():
        raise ValueError("results has more than one row for the same match")

    markets = market_set()
    size = markets.max_goals + 1
    rows = markets.catalog[["market_id", "selection"]].assign(
        _row=np.arange(len(markets.catalog), dtype="float64")
    )
    row = offers[["market_id", "selection"]].merge(
        rows, how="left", on=["market_id", "selection"]
    )["_row"]
    goals = offers[["match_id"]].merge(
        results[["match_id", "home_goals", "away_goals"]], how="left", on="match_id"
    )
    line = row.to_numpy(dtype="float64")
    home = goals["home_goals"].to_numpy(dtype="float64")
    away = goals["away_goals"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        settled = (
            np.isfinite(line)
            & np.isfinite(home)
            & np.isfinite(away)
            & (home >= 0)
            & (away >= 0)
            & (home <= markets.max_goals)
            & (away <= markets.max_goals)
            & (home == np.floor(home))
            & (away == np.floor(away))
        )
    cell = np.where(settled, home * size + away, 0).astype(np.intp)
    selection = np.where(settled, line, 0).astype(np.intp)
    won = settled & (markets.win[selection, cell] > 0.5)
    push = settled & (markets.push[selection, cell] > 0.5)
    odds = offers["odds"].to_numpy(dtype="float64")
    profit = np.where(won, odds - 1.0, np.where(push, 0.0, -1.0))
    return offers.assign(
        settled=settled,
        won=won,
        push=push,
        profit=np.where(settled, profit, np.nan),
    )


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #
def coverage_table(
    enriched: pd.DataFrame, books: Sequence[str] = BOOKS
) -> pd.DataFrame:
    """Share of each season's matches with a complete quote, per market, book and snapshot.

    Returns:
        ``season``, ``market_id``, ``bookmaker``, ``snapshot``, ``matches`` (with a
        complete quote) and ``share`` (of all matches that season). Combinations
        with no quote at all are present, with zero.
    """
    seasons = enriched.drop_duplicates("match_id").groupby("season").size()
    season_of = enriched.drop_duplicates("match_id").set_index("match_id")["season"]
    counted: list[pd.Series] = []
    for snapshot in SNAPSHOTS:
        quotes = odds_long(enriched, snapshot, books).drop_duplicates(
            ["match_id", "bookmaker", "market_id"]
        )
        if quotes.empty:
            continue
        found = quotes.assign(
            season=quotes["match_id"].map(season_of), snapshot=snapshot
        )
        counted.append(
            found.groupby(["season", "market_id", "bookmaker", "snapshot"]).size()
        )
    grid = pd.MultiIndex.from_product(
        [seasons.index, list(MARKETS), list(books), list(SNAPSHOTS)],
        names=["season", "market_id", "bookmaker", "snapshot"],
    )
    found_all = (
        pd.concat(counted) if counted else pd.Series(dtype="int64", index=grid[:0])
    )
    table = found_all.reindex(grid, fill_value=0).rename("matches").reset_index()
    table["share"] = table["matches"] / table["season"].map(seasons)
    return table


def margin_table(enriched: pd.DataFrame, books: Sequence[str] = BOOKS) -> pd.DataFrame:
    """Average bookmaker margin (inverse odds added up, minus one) per quote.

    Returns:
        ``snapshot``, ``bookmaker``, ``market_id``, ``n`` (quotes) and ``margin``
        as a fraction. A negative margin means the prices added up to less than
        one: the best price per selection (``max``) is not a bookmaker you can
        bet at.
    """
    parts: list[pd.DataFrame] = []
    for snapshot in SNAPSHOTS:
        quotes = odds_long(enriched, snapshot, books)
        if quotes.empty:
            continue
        overround = (1.0 / quotes["odds"]).groupby(
            [quotes["match_id"], quotes["bookmaker"], quotes["market_id"]]
        ).sum() - 1.0
        summary = (
            overround.groupby(level=["bookmaker", "market_id"])
            .agg(["size", "mean"])
            .rename(columns={"size": "n", "mean": "margin"})
            .reset_index()
        )
        parts.append(summary.assign(snapshot=snapshot))
    if not parts:
        return pd.DataFrame(
            columns=["snapshot", "bookmaker", "market_id", "n", "margin"]
        )
    return pd.concat(parts, ignore_index=True)[
        ["snapshot", "bookmaker", "market_id", "n", "margin"]
    ]


def _paired(enriched: pd.DataFrame, book: str) -> pd.DataFrame:
    """One book's early and closing quote of the same selection, side by side."""
    early = _fair_quotes(enriched, "early", (book,))
    close = _fair_quotes(enriched, "close", (book,))
    wanted = [*KEY, "odds", "p_fair"]
    return early[wanted].merge(close[wanted], on=KEY, suffixes=("_early", "_close"))


def _selection_order() -> dict[tuple[str, str], int]:
    return {
        (market_id, selection): position
        for position, (market_id, selection) in enumerate(
            (market_id, selection)
            for market_id, selections in MARKETS.items()
            for selection, _ in selections
        )
    }


def line_movement(enriched: pd.DataFrame, book: str = "ps") -> pd.DataFrame:
    """How far one bookmaker's price moves between the early and the closing snapshot.

    Returns:
        Per ``market_id`` and ``selection``: ``n`` (selections quoted in both
        snapshots), ``mean_abs_move`` (average absolute change of the margin-free
        probability, in percentage points) and ``unchanged`` (share of prices
        identical in both snapshots; a high value would mean the "early" price is
        really the closing one and closing-line value cannot be measured).
    """
    columns = ["market_id", "selection", "n", "mean_abs_move", "unchanged"]
    pairs = _paired(enriched, book)
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    frame = pairs.assign(
        move=(pairs["p_fair_close"] - pairs["p_fair_early"]).abs() * 100.0,
        same=pairs["odds_early"] == pairs["odds_close"],
    )
    table = (
        frame.groupby(["market_id", "selection"], sort=False)
        .agg(
            n=("move", "size"),
            mean_abs_move=("move", "mean"),
            unchanged=("same", "mean"),
        )
        .reset_index()
    )
    order = _selection_order()
    rank = [order[(m, s)] for m, s in zip(table["market_id"], table["selection"])]
    return (
        table.assign(_rank=rank)
        .sort_values("_rank")
        .drop(columns="_rank")[columns]
        .reset_index(drop=True)
    )


def market_learning(
    enriched: pd.DataFrame, book: str = "ps", *, seed: int = 0
) -> pd.DataFrame:
    """What the market learns between the early and the closing price.

    Scores one bookmaker's margin-free early and closing probabilities by
    log-loss on the same matches. The closing price knows more, so its loss is
    lower; the difference is the information that arrives in that window, and it
    is the room closing-line value has to work with.

    Returns:
        Per ``market_id``: ``n`` matches, ``early`` and ``close`` log-loss,
        ``gain`` (early minus close, positive when the close is better) and its 95%
        paired-bootstrap interval ``gain_low`` and ``gain_high``.

    Raises:
        ValueError: If ``enriched`` lacks ``home_goals`` or ``away_goals``.
    """
    _require(enriched, ("match_id", "home_goals", "away_goals"), "enriched")
    columns = ["market_id", "n", "early", "close", "gain", "gain_low", "gain_high"]
    pairs = _paired(enriched, book)
    results = enriched[["match_id", "home_goals", "away_goals"]]
    settled = settle(pairs.rename(columns={"odds_early": "odds"}), results)
    winners = settled[settled["settled"] & settled["won"]]
    rows: list[dict[str, object]] = []
    for market_id in MARKETS:
        chosen = winners[winners["market_id"] == market_id]
        if chosen.empty:
            continue
        early = -np.log(chosen["p_fair_early"].to_numpy(dtype="float64"))
        close = -np.log(chosen["p_fair_close"].to_numpy(dtype="float64"))
        low, high = paired_bootstrap_interval(early - close, seed=seed)
        rows.append(
            {
                "market_id": market_id,
                "n": len(chosen),
                "early": float(early.mean()),
                "close": float(close.mean()),
                "gain": float((early - close).mean()),
                "gain_low": low,
                "gain_high": high,
            }
        )
    return pd.DataFrame(rows, columns=columns)


@dataclass(frozen=True)
class OddsAudit:
    """What the odds columns can and cannot test.

    Attributes:
        matches: Matches in the table.
        window_matches: Matches from ``from_season`` on (the backtest window).
        from_season: First season of the window.
        reference_book: Book whose early and closing prices were compared.
        coverage: See :func:`coverage_table` (all seasons).
        margins: See :func:`margin_table` (all seasons).
        movement: See :func:`line_movement` (window only).
        learning: See :func:`market_learning` (window only).
    """

    matches: int
    window_matches: int
    from_season: int
    reference_book: str
    coverage: pd.DataFrame
    margins: pd.DataFrame
    movement: pd.DataFrame
    learning: pd.DataFrame


def audit_odds(
    enriched: pd.DataFrame,
    *,
    reference_book: str = "ps",
    from_season: int = DEFAULT_FROM_SEASON,
    seed: int = 0,
) -> OddsAudit:
    """Audit the odds in the enriched match table.

    Args:
        enriched: The enriched match table (needs ``match_id``, ``season``,
            ``home_goals``, ``away_goals`` and the wide odds columns).
        reference_book: Book whose early and closing prices are compared.
        from_season: First season of the movement and learning tables.
        seed: Seed of the bootstrap.

    Raises:
        ValueError: If required columns are missing.
    """
    _require(enriched, REQUIRED_COLUMNS, "the match table")
    window = enriched[enriched["season"] >= from_season]
    return OddsAudit(
        matches=len(enriched),
        window_matches=len(window),
        from_season=from_season,
        reference_book=reference_book,
        coverage=coverage_table(enriched),
        margins=margin_table(enriched),
        movement=line_movement(window, reference_book),
        learning=market_learning(window, reference_book, seed=seed),
    )


def _market_title(market_id: str) -> str:
    return "1X2" if market_id == "1X2" else "Over/under 2.5 goals"


def _coverage_view(coverage: pd.DataFrame, market_id: str) -> pd.DataFrame:
    view = coverage[coverage["market_id"] == market_id].pivot(
        index="season", columns=["bookmaker", "snapshot"], values="share"
    )
    order = [
        (book, snapshot)
        for book in BOOKS
        for snapshot in SNAPSHOTS
        if (book, snapshot) in view.columns
    ]
    return view[order] * 100.0


def format_audit(audit: OddsAudit) -> str:
    """Render the audit as aligned text tables."""
    book = BOOK_NAMES.get(audit.reference_book, audit.reference_book)
    headline = (
        f"Odds audit: {audit.matches} matches; backtest window = "
        f"{audit.window_matches} matches from season {audit.from_season}"
    )
    sections = [headline]
    for market_id in MARKETS:
        view = _coverage_view(audit.coverage, market_id)
        sections.append(
            f"{_market_title(market_id)}: complete quotes, % of each season's "
            f"matches (early = before the closing price)\n"
            f"{view.to_string(float_format='{:.0f}'.format, na_rep='-')}"
        )
    if not audit.margins.empty:
        margins = audit.margins.assign(margin=audit.margins["margin"] * 100.0).pivot(
            index="bookmaker", columns=["market_id", "snapshot"], values="margin"
        )
        order = [
            (market_id, snapshot)
            for market_id in MARKETS
            for snapshot in SNAPSHOTS
            if (market_id, snapshot) in margins.columns
        ]
        sections.append(
            "Average bookmaker margin, % (inverse odds added up, minus one; "
            "negative = not a real bookmaker)\n"
            f"{margins[order].reindex([b for b in BOOKS if b in margins.index]).to_string(float_format='{:.1f}'.format, na_rep='-')}"
        )
    if audit.movement.empty:
        sections.append(
            f"{book}: no selection has both an early and a closing price in the window"
        )
    else:
        moves = audit.movement.assign(unchanged=audit.movement["unchanged"] * 100.0)
        sections.append(
            f"{book}: movement from the early to the closing price\n"
            "  mean_abs_move: average change of the fair probability, in "
            "percentage points\n"
            "  unchanged: % of prices identical in both snapshots\n"
            f"{moves.to_string(index=False, float_format='{:.2f}'.format)}"
        )
    if not audit.learning.empty:
        sections.append(
            f"{book}: log-loss of the early and the closing price, lower is better\n"
            "  gain: early minus close (positive = the close is better), with a "
            "95% bootstrap interval\n"
            f"{audit.learning.to_string(index=False, float_format='{:.4f}'.format)}"
        )
    return "\n\n".join(sections)


def verdict(audit: OddsAudit) -> list[str]:
    """Plain-language reading of the audit."""
    book = BOOK_NAMES.get(audit.reference_book, audit.reference_book)
    if audit.learning.empty:
        message = (
            f"No match since season {audit.from_season} has both an early and a "
            f"closing {book} quote, so closing-line value cannot be measured "
            "against that book; try --book avg or a lower --from-season"
        )
        return [message]
    lines = []
    for row in audit.learning.itertuples():
        title = _market_title(row.market_id)
        share = row.n / audit.window_matches if audit.window_matches else 0.0
        lines.append(
            f"{title}: {row.n} of {audit.window_matches} matches ({share:.0%}) have "
            f"an early and a closing {book} quote"
        )
        if row.gain_low > 0.0:
            lines.append(
                f"{title}: the closing price beats the early one by {row.gain:.4f} "
                "log-loss, so the market learns in that window and a bet's "
                "closing-line value can be measured"
            )
        else:
            lines.append(
                f"{title}: the closing price is not clearly better than the early "
                "one; closing-line value on this market would be mostly noise"
            )
    return lines


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the bookmaker odds: coverage, margins, line movement."
    )
    parser.add_argument("--enriched", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument(
        "--book",
        choices=FAIR_BOOKS,
        default="ps",
        help="book whose early and closing prices are compared (default: ps)",
    )
    parser.add_argument("--from-season", type=int, default=DEFAULT_FROM_SEASON)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Audit the odds and return the process exit code.

    Returns ``0`` on success and ``1`` when the match table is missing, unusable
    or holds no complete quote at all.
    """
    args = _parse_args(argv)
    log = setup_logging("odds_history")
    if not args.enriched.exists():
        log.error("Not found: %s; run python -m src.enrich first", args.enriched)
        return 1
    try:
        audit = audit_odds(
            pd.read_parquet(args.enriched),
            reference_book=args.book,
            from_season=args.from_season,
            seed=args.seed,
        )
    except (ValueError, KeyError) as exc:
        log.error("Cannot audit the odds: %s", exc)
        return 1
    if audit.coverage["matches"].sum() == 0:
        log.error(
            "No complete 1X2 or over/under quote found in %s; the odds columns "
            "are empty, so there is nothing to audit (check the football-data "
            "download, then re-run python -m src.pipeline and python -m src.enrich)",
            args.enriched,
        )
        return 1
    log.info("\n%s", format_audit(audit))
    for line in verdict(audit):
        log.info(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
