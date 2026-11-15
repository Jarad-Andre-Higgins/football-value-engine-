"""Which way of taking the margin out of a price is the most accurate?

The methods of :mod:`src.demargin` all turn a bookmaker's prices into
probabilities that add up to one, and they differ in how they share the margin
out between favourites and long shots. This audit settles the question the way
any forecast is judged, from what happened. For every quote it takes the
probability each method gave the outcome that occurred and scores it by
log-loss, the negative log of that probability. A more accurate method gives the
winner more probability on average.

The differences are small, so the methods are compared on the very same quotes,
and the intervals resample whole weeks, because the matches of one week share
news and moves in the market.

The choice is made out of sample, so the backtest is not fitted to its own
results:

* The method is **chosen** on the seasons before the backtest's first season,
  from the closing line (the yardstick that closing-line value is measured
  against), with both markets pooled. A method is only chosen if its gain over
  proportional scaling has a 95% interval above zero; otherwise proportional
  scaling stays.
* The backtest seasons only get a **veto**. The choice is confirmed if the gain
  on the closing line there also has an interval above zero, and the early price
  (the reference the bets are chosen with) is not clearly worse.

A reliability table shows the mechanism: how often selections of a given
probability really won, against what each method said, so a method that
overstates long shots shows up as predicted above observed in the lowest bands.

Usage::

    python -m src.margin_audit
    python -m src.margin_audit --book avg --first-season 2023
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

from src.backtest import (
    DEFAULT_RESAMPLES,
    FIRST_SEASON,
    REFERENCE_BOOK,
    batch_start,
    cluster_bootstrap,
)
from src.config import PROCESSED_DIR
from src.demargin import METHOD_NAMES, METHODS
from src.enrich import ENRICHED_OUTPUT
from src.logging_config import setup_logging
from src.odds_history import (
    BOOK_NAMES,
    FAIR_BOOKS,
    MARKETS,
    fair_probabilities,
    odds_long,
    settle,
)

logger = logging.getLogger("src.margin_audit")

DEFAULT_BOOK: Final[str] = REFERENCE_BOOK
MARGIN_AUDIT_DIR: Final[Path] = PROCESSED_DIR / "margin_audit"
#: The method every other is measured against.
BASELINE: Final[str] = "proportional"
#: The closing line decides the choice; the early price can only veto it.
SNAPSHOT_ORDER: Final[tuple[str, ...]] = ("close", "early")
#: ``selection``: seasons before the first backtest season. ``test``: from it on.
WINDOWS: Final[tuple[str, ...]] = ("selection", "test")
#: The ``market_id`` of the rows that pool every market.
POOLED: Final[str] = "all"
#: Probability bands of the reliability table; each includes its upper end.
BAND_EDGES: Final[tuple[float, ...]] = (0.0, 0.08, 0.15, 0.25, 0.35, 0.50, 0.65, 1.0)
BAND_LABELS: Final[tuple[str, ...]] = (
    "0-8%",
    "8-15%",
    "15-25%",
    "25-35%",
    "35-50%",
    "50-65%",
    "65-100%",
)
#: Fewest selections in the lowest band for the verdict to quote it.
MIN_BAND_SELECTIONS: Final[int] = 50

ENRICHED_REQUIRED: Final[tuple[str, ...]] = (
    "match_id",
    "season",
    "date",
    "home_goals",
    "away_goals",
)
PROBABILITY_COLUMNS: Final[dict[str, str]] = {
    method: f"p_{method}" for method in METHODS
}
SELECTION_COLUMNS: Final[list[str]] = [
    "match_id",
    "season",
    "batch",
    "market_id",
    "selection",
    "won",
    *PROBABILITY_COLUMNS.values(),
]
SCORE_COLUMNS: Final[list[str]] = [
    "snapshot",
    "window",
    "market_id",
    "method",
    "n",
    "log_loss",
    "gain",
    "gain_low",
    "gain_high",
]
RELIABILITY_COLUMNS: Final[list[str]] = [
    "snapshot",
    "market_id",
    "band",
    "n",
    "observed",
    *METHODS,
]


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


# --------------------------------------------------------------------------- #
# One row per selection of every quote, with what each method said
# --------------------------------------------------------------------------- #
def selection_table(enriched: pd.DataFrame, book: str, snapshot: str) -> pd.DataFrame:
    """Every selection of one book's complete quotes, settled, with each method's probability.

    Quotes without a usable score or a match date are left out, and so is any
    quote a method cannot price, so the methods are always compared on exactly
    the same selections.

    Args:
        enriched: The match table with ``match_id``, ``season``, ``date``, the
            goals and the wide odds columns.
        book: The bookmaker code (see :data:`src.odds_history.BOOKS`).
        snapshot: ``"early"`` or ``"close"``.

    Returns:
        :data:`SELECTION_COLUMNS`: the match, its ``season`` and ``batch`` (the
        Friday of its week), the selection, whether it ``won`` and one
        probability column per method (``p_proportional``, ``p_power``, ...).
    """
    quotes = odds_long(enriched, snapshot, (book,))
    if quotes.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)
    for method, column in PROBABILITY_COLUMNS.items():
        quotes[column] = fair_probabilities(quotes, method).to_numpy(dtype="float64")
    priced = quotes.dropna(subset=list(PROBABILITY_COLUMNS.values()))
    results = enriched[["match_id", "home_goals", "away_goals"]]
    settled = settle(priced, results)
    settled = settled[settled["settled"]]
    table = settled.merge(
        enriched[["match_id", "season", "date"]], on="match_id", how="left"
    )
    undated = table["date"].isna()
    if undated.any():
        logger.warning(
            "%d prices have no match date and are left out", int(undated.sum())
        )
        table = table[~undated]
    table["batch"] = batch_start(table["date"])
    return table[SELECTION_COLUMNS].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Scoring the methods
# --------------------------------------------------------------------------- #
def score_methods(
    table: pd.DataFrame,
    *,
    first_season: int = FIRST_SEASON,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """Log-loss of every method on the outcomes that happened, with paired gains.

    Each quote contributes the negative log of the probability a method gave its
    winning selection. The ``gain`` of a method is the proportional method's
    log-loss minus its own, quote by quote: positive means more accurate. Its
    interval resamples whole weeks.

    Args:
        table: From :func:`selection_table`.
        first_season: Seasons before it are the ``selection`` window, the rest the
            ``test`` window.
        n_resamples: Bootstrap resamples for the intervals.
        seed: Seed of the resampling.

    Returns:
        One row per window, market (each of :data:`src.odds_history.MARKETS` and
        the pooled :data:`POOLED`) and method with the columns of
        :data:`SCORE_COLUMNS` except ``snapshot``. The baseline's ``gain`` is zero
        and has no interval. Windows or markets with no quote are absent.
    """
    winners = table[table["won"].astype(bool)]
    seasons = winners["season"].to_numpy()
    markets = winners["market_id"].to_numpy()
    rows: list[dict[str, object]] = []
    for window in WINDOWS:
        in_window = (
            seasons < first_season if window == "selection" else seasons >= first_season
        )
        for market_id in (*MARKETS, POOLED):
            chosen = (
                in_window if market_id == POOLED else in_window & (markets == market_id)
            )
            if not chosen.any():
                continue
            part = winners[chosen]
            weeks = part["batch"].to_numpy()
            baseline = -np.log(
                part[PROBABILITY_COLUMNS[BASELINE]].to_numpy(dtype="float64")
            )
            for method in METHODS:
                loss = -np.log(
                    part[PROBABILITY_COLUMNS[method]].to_numpy(dtype="float64")
                )
                gain = baseline - loss
                low = high = float("nan")
                if method != BASELINE:
                    by_week = pd.Series(gain).groupby(weeks).agg(["sum", "size"])
                    low, high = cluster_bootstrap(
                        by_week["sum"].to_numpy(dtype="float64"),
                        by_week["size"].to_numpy(dtype="float64"),
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                rows.append(
                    {
                        "window": window,
                        "market_id": market_id,
                        "method": method,
                        "n": len(part),
                        "log_loss": float(loss.mean()),
                        "gain": float(gain.mean()),
                        "gain_low": low,
                        "gain_high": high,
                    }
                )
    return pd.DataFrame(rows, columns=[c for c in SCORE_COLUMNS if c != "snapshot"])


def reliability_table(table: pd.DataFrame) -> pd.DataFrame:
    """How often selections won, against what each method said, band by band.

    Selections are put in bands by their proportional probability, the same for
    every method, so each method is read against the same outcomes.

    Returns:
        Per market and band (bands with no selection are absent): ``n``,
        ``observed`` (share that won) and each method's average probability.
    """
    bands = pd.cut(
        table[PROBABILITY_COLUMNS[BASELINE]],
        bins=list(BAND_EDGES),
        labels=list(BAND_LABELS),
        include_lowest=True,
    ).astype("object")
    rows: list[dict[str, object]] = []
    for market_id in MARKETS:
        in_market = table["market_id"] == market_id
        for label in BAND_LABELS:
            part = table[in_market & (bands == label)]
            if part.empty:
                continue
            rows.append(
                {
                    "market_id": market_id,
                    "band": label,
                    "n": len(part),
                    "observed": float(part["won"].astype("float64").mean()),
                    **{
                        method: float(part[column].mean())
                        for method, column in PROBABILITY_COLUMNS.items()
                    },
                }
            )
    return pd.DataFrame(
        rows, columns=[c for c in RELIABILITY_COLUMNS if c != "snapshot"]
    )


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def _pooled(
    scores: pd.DataFrame, snapshot: str, window: str, method: str
) -> pd.Series | None:
    """The row of both markets pooled, or None when there is none."""
    rows = scores[
        (scores["snapshot"] == snapshot)
        & (scores["window"] == window)
        & (scores["market_id"] == POOLED)
        & (scores["method"] == method)
    ]
    return rows.iloc[0] if len(rows) else None


def _clearly_better(row: pd.Series) -> bool:
    """The whole 95% interval of the gain lies above zero (no interval: never)."""
    return bool(row["gain_low"] > 0.0)


def _clearly_worse(row: pd.Series) -> bool:
    """The whole 95% interval of the gain lies below zero (no interval: never)."""
    return bool(row["gain_high"] < 0.0)


def decide(scores: pd.DataFrame) -> tuple[str, str]:
    """Choose a method on the earlier seasons and let the backtest seasons veto it.

    The choice is the method with the largest gain on the closing line of the
    ``selection`` window, among those whose interval lies above zero; if there is
    none, proportional scaling. It is confirmed when its closing-line gain on the
    ``test`` window also lies above zero and its early-price gain there is not
    clearly below zero (and both exist).

    Args:
        scores: From :func:`audit_margin`, with a ``snapshot`` column.

    Returns:
        ``(chosen, recommended)``: ``recommended`` is ``chosen`` when confirmed
        and proportional scaling otherwise.
    """
    chosen, best = BASELINE, 0.0
    for method in METHODS:
        if method == BASELINE:
            continue
        row = _pooled(scores, "close", "selection", method)
        if row is not None and _clearly_better(row) and row["gain"] > best:
            chosen, best = method, float(row["gain"])
    if chosen == BASELINE:
        return chosen, BASELINE
    close = _pooled(scores, "close", "test", chosen)
    early = _pooled(scores, "early", "test", chosen)
    confirmed = (
        close is not None
        and _clearly_better(close)
        and early is not None
        and not _clearly_worse(early)
    )
    return chosen, chosen if confirmed else BASELINE


@dataclass(frozen=True)
class MarginAudit:
    """What the audit found.

    Attributes:
        book: The bookmaker whose prices were audited.
        first_season: First season of the ``test`` window.
        scores: Log-loss and gain per snapshot, window, market and method
            (:data:`SCORE_COLUMNS`).
        reliability: Observed frequency against each method's probability per
            snapshot, market and band (:data:`RELIABILITY_COLUMNS`).
        chosen: The method picked on the ``selection`` window; proportional
            scaling when none was clearly better.
        recommended: ``chosen`` if the ``test`` window confirmed it, otherwise
            proportional scaling.
    """

    book: str
    first_season: int
    scores: pd.DataFrame
    reliability: pd.DataFrame
    chosen: str
    recommended: str


def audit_margin(
    enriched: pd.DataFrame,
    *,
    book: str = DEFAULT_BOOK,
    first_season: int = FIRST_SEASON,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> MarginAudit:
    """Compare the margin-removal methods on one book's early and closing prices.

    Args:
        enriched: The enriched match table (``match_id``, ``season``, ``date``,
            ``home_goals``, ``away_goals`` and the wide odds columns).
        book: The bookmaker to audit (one of :data:`src.odds_history.FAIR_BOOKS`).
        first_season: First season of the ``test`` window; earlier seasons choose.
        n_resamples: Bootstrap resamples for the intervals.
        seed: Seed of the resampling.

    Returns:
        A :class:`MarginAudit`.

    Raises:
        ValueError: For missing columns, a repeated match, an unknown book, or a
            table with no complete quote of the book that has a result.
    """
    _require(enriched, ENRICHED_REQUIRED, "the match table")
    if enriched["match_id"].duplicated().any():
        raise ValueError("the match table repeats a match")
    if book not in FAIR_BOOKS:
        raise ValueError(f"the book must be one of {list(FAIR_BOOKS)}, got {book!r}")
    scored: list[pd.DataFrame] = []
    reliable: list[pd.DataFrame] = []
    for snapshot in SNAPSHOT_ORDER:
        table = selection_table(enriched, book, snapshot)
        found = score_methods(
            table, first_season=first_season, n_resamples=n_resamples, seed=seed
        )
        if found.empty:
            continue
        scored.append(found.assign(snapshot=snapshot))
        reliable.append(reliability_table(table).assign(snapshot=snapshot))
    if not scored:
        raise ValueError(
            f"no complete 1X2 or over/under {BOOK_NAMES.get(book, book)} quote with "
            "a result in the match table; check the odds with python -m src.odds_history"
        )
    scores = pd.concat(scored, ignore_index=True)[SCORE_COLUMNS]
    reliability = pd.concat(reliable, ignore_index=True)[RELIABILITY_COLUMNS]
    chosen, recommended = decide(scores)
    return MarginAudit(
        book=book,
        first_season=first_season,
        scores=scores,
        reliability=reliability,
        chosen=chosen,
        recommended=recommended,
    )


# --------------------------------------------------------------------------- #
# Reading the result
# --------------------------------------------------------------------------- #
def _name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def _gain_text(row: pd.Series) -> str:
    return f"{row['gain']:+.5f} [{row['gain_low']:+.5f}, {row['gain_high']:+.5f}]"


def _market_title(market_id: str) -> str:
    return "both" if market_id == POOLED else market_id


def _quotes(scores: pd.DataFrame, snapshot: str) -> int:
    rows = scores[
        (scores["snapshot"] == snapshot)
        & (scores["market_id"] == POOLED)
        & (scores["method"] == BASELINE)
    ]
    return int(rows["n"].sum())


def format_margin_audit(audit: MarginAudit) -> str:
    """Render the audit as aligned text tables."""
    book = BOOK_NAMES.get(audit.book, audit.book)
    first = audit.first_season
    scores = audit.scores
    headline = (
        f"Margin audit of {book} prices: {_quotes(scores, 'close')} closing and "
        f"{_quotes(scores, 'early')} early quotes scored. Seasons before {first} "
        f"choose the method; seasons from {first} (the backtest) only confirm it"
    )
    rows = []
    for (snapshot, window, market_id), group in scores.groupby(
        ["snapshot", "window", "market_id"], sort=False
    ):
        by_method = group.set_index("method")
        row: dict[str, object] = {
            "snapshot": snapshot,
            "window": f"before {first}" if window == "selection" else f"from {first}",
            "market": _market_title(str(market_id)),
            "n": int(by_method["n"].iloc[0]),
        }
        for method in METHODS:
            row[method] = f"{by_method.loc[method, 'log_loss']:.5f}"
        for method in METHODS:
            if method != BASELINE:
                row[f"{method} gain [95% interval]"] = _gain_text(by_method.loc[method])
        rows.append(row)
    loss_title = (
        "Log-loss of each method on the outcome that happened, lower is better.\n"
        "  gain: proportional minus the method, per quote; positive = the method "
        "is more accurate. The 95% interval resamples whole weeks"
    )
    sections = [
        headline,
        f"{loss_title}\n{pd.DataFrame(rows).to_string(index=False)}",
    ]
    close = audit.reliability[audit.reliability["snapshot"] == "close"]
    if not close.empty:
        view = pd.DataFrame(
            {
                "market": close["market_id"],
                "band": close["band"],
                "n": close["n"],
                "won %": (100.0 * close["observed"]).map("{:.1f}".format),
                **{
                    f"{method} %": (100.0 * close[method]).map("{:.1f}".format)
                    for method in METHODS
                },
            }
        )
        reliability_title = (
            f"{book} closing line: how often selections won, against what each "
            "method said.\n  Bands are by the proportional probability. A method "
            "that overstates long shots says more than the observed share in the "
            "lowest bands"
        )
        sections.append(f"{reliability_title}\n{view.to_string(index=False)}")
    return "\n\n".join(sections)


def verdict(audit: MarginAudit) -> list[str]:
    """Plain-language reading of the audit."""
    book = BOOK_NAMES.get(audit.book, audit.book)
    first = audit.first_season
    scores = audit.scores
    lines = []
    candidates = {
        method: _pooled(scores, "close", "selection", method)
        for method in METHODS
        if method != BASELINE
    }
    present = {m: row for m, row in candidates.items() if row is not None}
    if not present:
        lines.append(
            f"No season before {first} has {book} closing quotes with a result, so "
            "a method cannot be chosen out of sample; proportional scaling stays"
        )
    elif audit.chosen == BASELINE:
        best = max(present, key=lambda method: float(present[method]["gain"]))
        lines.append(
            f"No method is clearly more accurate than proportional scaling on the "
            f"{book} closing line before season {first}; the best, {_name(best)}, "
            f"gains {_gain_text(present[best])} log-loss per quote; proportional "
            "scaling stays"
        )
    else:
        chosen = present[audit.chosen]
        lines.append(
            f"Chosen on seasons before {first} ({book} closing line, both markets): "
            f"{_name(audit.chosen)} is more accurate than proportional scaling by "
            f"{_gain_text(chosen)} log-loss per quote"
        )
        close = _pooled(scores, "close", "test", audit.chosen)
        early = _pooled(scores, "early", "test", audit.chosen)
        if close is None or early is None:
            lines.append(
                f"Seasons from {first} lack closing or early quotes, so the choice "
                "cannot be confirmed; proportional scaling stays"
            )
        else:
            outcome = (
                "confirmed"
                if audit.recommended == audit.chosen
                else "not confirmed (the closing-line gain needs an interval above "
                "zero and the early price must not be clearly worse); proportional "
                "scaling stays"
            )
            lines.append(
                f"Seasons from {first}: closing line {_gain_text(close)}, early "
                f"price {_gain_text(early)}: {outcome}"
            )
    close_rel = audit.reliability[
        (audit.reliability["snapshot"] == "close")
        & (audit.reliability["market_id"] == "1X2")
        & (audit.reliability["band"] == BAND_LABELS[0])
    ]
    if len(close_rel) and int(close_rel["n"].iloc[0]) >= MIN_BAND_SELECTIONS:
        low = close_rel.iloc[0]
        lines.append(
            f"{book} closing 1X2 selections that proportional scaling puts below "
            f"{BAND_EDGES[1]:.0%} ({int(low['n'])} of them) won "
            f"{100.0 * low['observed']:.1f}% of the time; proportional scaling says "
            f"{100.0 * low[BASELINE]:.1f}%, the power method "
            f"{100.0 * low['power']:.1f}%, Shin's method {100.0 * low['shin']:.1f}%"
        )
    if audit.recommended == BASELINE:
        lines.append("Recommended: keep the default, python -m src.backtest")
    else:
        lines.append(
            f"Recommended: python -m src.backtest --demargin {audit.recommended}"
        )
    return lines


def save_margin_audit(
    audit: MarginAudit, directory: Path = MARGIN_AUDIT_DIR
) -> list[Path]:
    """Write the audit's tables to ``directory`` and return the files written."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("scores", "reliability"):
        path = directory / f"{name}.csv"
        getattr(audit, name).to_csv(path, index=False)
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the ways of taking the margin out of a price."
    )
    parser.add_argument("--enriched", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument(
        "--book",
        choices=FAIR_BOOKS,
        default=DEFAULT_BOOK,
        help="bookmaker whose prices are audited (default: %(default)s)",
    )
    parser.add_argument(
        "--first-season",
        type=int,
        default=FIRST_SEASON,
        help="first season of the confirming window; earlier seasons choose "
        "(default: %(default)s, the backtest's first season)",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=MARGIN_AUDIT_DIR)
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the audit and return the process exit code.

    Returns ``0`` on success and ``1`` when the match table is missing or unusable.
    """
    args = _parse_args(argv)
    log = setup_logging("margin_audit")
    if not args.enriched.exists():
        log.error("Not found: %s; run python -m src.enrich first", args.enriched)
        return 1
    try:
        audit = audit_margin(
            pd.read_parquet(args.enriched),
            book=args.book,
            first_season=args.first_season,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    except (ValueError, KeyError) as exc:
        log.error("Cannot audit the margin removal: %s", exc)
        return 1
    log.info("\n%s", format_margin_audit(audit))
    for line in verdict(audit):
        log.info(line)
    for path in save_margin_audit(audit, args.output_dir):
        log.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
