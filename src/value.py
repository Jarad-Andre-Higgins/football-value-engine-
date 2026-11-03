"""Value detection and stake sizing.

A bet has value when the chance we give it beats the bookmaker's price. Our
model is slightly less sharp than the betting market, though, so its raw
disagreement with a price is partly noise. The engine protects against that in
three ways:

* **Margin out of the price.** A bookmaker's odds include a margin, so the
  inverse odds add up to more than one. When every selection of a market is
  quoted, the inverse odds are scaled back to add up exactly, leaving the
  market's own *reference* probability. Without a complete market, an assumed
  margin is removed instead (see :class:`ValueConfig`).
* **Shrinkage.** The probability used for a bet is a blend of that reference
  and the model's: ``(1 - w) * reference + w * model``. The weight ``w`` should
  come from data (``python -m src.blend``).
* **Guard rails.** Minimum edge, odds and probability limits, one bet per match,
  fractional Kelly staking, a cap per bet and a cap on total exposure.

Every offer is returned with a ``status`` that explains why it was or was not
selected, so a skipped bet is never a mystery.

Kelly: with win probability ``p``, refund probability ``r`` (draw no bet and
whole-number Asian handicaps) and decimal odds ``o``, long-run growth is
maximised by staking ``f = EV / ((o - 1) * (1 - r))`` of the bankroll, where
``EV = p * (o - 1) - (1 - p - r)`` is the expected profit per unit staked.
Kelly assumes independent bets; bets on different matches are close to that,
while several selections of one match are not, hence the per-match limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.markets import market_set

FloatArray = NDArray[np.float64]

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "market_id",
    "selection",
    "odds",
    "probability",
    "push_probability",
)
#: Statuses from best to worst; results are sorted in this order.
STATUS_ORDER: Final[tuple[str, ...]] = (
    "bet",
    "below_min_stake",
    "match_limit",
    "better_price",
    "below_min_edge",
    "low_probability",
    "odds_out_of_range",
    "invalid_probability",
    "invalid_odds",
)
PROBABILITY_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True)
class ValueConfig:
    """Betting rules.

    Attributes:
        model_weight: Share of the final probability taken from the model (0 to
            1); the rest comes from the reference. The default is a cautious
            placeholder: use the value fitted by ``python -m src.blend``
            (see :func:`src.blend.load_model_weight`).
        min_edge: Minimum expected profit per unit staked after shrinkage. An
            edge of zero or less is never bet, whatever this is set to.
        min_odds: Shortest price considered.
        max_odds: Longest price considered (model error dominates long shots).
        min_probability: Smallest final win probability considered.
        kelly_fraction: Share of the full Kelly stake to use (a quarter is a
            common cautious choice: it gives up little growth for much less
            variance and tolerates errors in the probabilities).
        max_stake_fraction: Largest stake on one bet, as a share of the bankroll.
        max_total_fraction: Largest total stake in one run, as a share of the
            bankroll; stakes are scaled down together when it is exceeded.
        max_bets_per_match: Bets allowed per match; correlated selections of one
            match are not stacked.
        min_stake: Smallest stake worth placing, in currency units.
        bankroll: Current bankroll, in currency units.
        fallback_margin: Bookmaker margin assumed when a price cannot be
            de-margined (an incomplete market and no reference supplied).
    """

    model_weight: float = 0.25
    min_edge: float = 0.04
    min_odds: float = 1.30
    max_odds: float = 8.0
    min_probability: float = 0.05
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.02
    max_total_fraction: float = 0.10
    max_bets_per_match: int = 1
    min_stake: float = 0.0
    bankroll: float = 1000.0
    fallback_margin: float = 0.05

    def __post_init__(self) -> None:
        """Reject settings that cannot describe a sensible strategy."""
        for name in ("model_weight", "kelly_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        for name in ("max_stake_fraction", "max_total_fraction"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if not 1.0 < self.min_odds < self.max_odds:
            raise ValueError("need 1 < min_odds < max_odds")
        if self.max_bets_per_match < 1:
            raise ValueError("max_bets_per_match must be at least 1")
        if not self.bankroll > 0.0:
            raise ValueError(f"bankroll must be positive, got {self.bankroll}")
        for name in ("min_stake", "min_edge"):
            value = getattr(self, name)
            if not value >= 0.0:
                raise ValueError(f"{name} cannot be negative, got {value}")
        for name in ("min_probability", "fallback_margin"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")


# --------------------------------------------------------------------------- #
# The arithmetic of a bet
# --------------------------------------------------------------------------- #
def expected_value(
    p_win: FloatArray, p_push: FloatArray, odds: FloatArray
) -> FloatArray:
    """Expected profit per unit staked; a refunded stake neither wins nor loses."""
    return p_win * (odds - 1.0) - (1.0 - p_win - p_push)


def full_kelly(p_win: FloatArray, p_push: FloatArray, odds: FloatArray) -> FloatArray:
    """Growth-optimal share of the bankroll to stake (zero when there is no edge)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = expected_value(p_win, p_push, odds) / ((odds - 1.0) * (1.0 - p_push))
    return np.where(
        np.isfinite(fraction) & (fraction > 0.0), np.minimum(fraction, 1.0), 0.0
    )


def blend_probability(
    p_model: FloatArray, p_reference: FloatArray, weight: float
) -> FloatArray:
    """Shrink the model's probability toward the reference: ``(1 - w) * ref + w * model``."""
    return (1.0 - weight) * p_reference + weight * p_model


# --------------------------------------------------------------------------- #
# Reference probabilities
# --------------------------------------------------------------------------- #
@cache
def _market_shapes() -> pd.DataFrame:
    """How many selections each market has, and how many win in every square.

    Most markets split the score grid: exactly one selection wins in each
    square, so fair probabilities add up to one. Double-chance markets cover each
    square twice, so theirs add up to two. A refunded square counts as covered
    for the whole market, so draw no bet and Asian handicaps still total one.

    Returns:
        ``n_selections`` and ``cover`` per ``market_id`` (``cover`` is NaN for a
        market whose squares are not covered evenly).
    """
    markets = market_set()
    rows = []
    for market_id, positions in markets.catalog.groupby(
        "market_id", sort=False
    ).indices.items():
        cover = markets.win[positions].sum(axis=0) + markets.push[positions].max(axis=0)
        even = bool(cover.min() == cover.max())
        rows.append((market_id, len(positions), float(cover[0]) if even else np.nan))
    return pd.DataFrame(rows, columns=["market_id", "n_selections", "cover"]).set_index(
        "market_id"
    )


def _require(offers: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in offers.columns]
    if missing:
        raise ValueError(f"offers is missing columns: {missing}")


def _group_keys(offers: pd.DataFrame) -> list[str]:
    """Columns that identify one bookmaker's quote of one market of one match."""
    keys = ["match_id", "market_id"]
    return [*keys, "bookmaker"] if "bookmaker" in offers.columns else keys


def _require_unique(offers: pd.DataFrame) -> None:
    keys = [*_group_keys(offers), "selection"]
    if offers.duplicated(subset=keys).any():
        raise ValueError(
            "offers has more than one price for the same match, market and selection"
            + (" per bookmaker" if "bookmaker" in offers.columns else "")
            + "; keep one"
        )


def reference_probabilities(offers: pd.DataFrame) -> pd.Series:
    """Fair win probabilities implied by a bookmaker's own prices, where possible.

    A market can be de-margined when the bookmaker quotes every selection of it:
    the inverse odds are scaled to add up to what the selections of that market
    always add up to (one winner per match; two for double chance), which removes
    the margin in proportion to each price. Markets that refund a stake give the
    win probability *given no refund*, so the result is scaled by the model's
    chance of no refund.

    Args:
        offers: Prices with ``match_id``, ``market_id``, ``selection``, ``odds``
            and ``push_probability``; a ``bookmaker`` column, when present,
            keeps bookmakers apart.

    Returns:
        A probability per offer, aligned to ``offers``; NaN when the market is
        incomplete, a price is not above 1, or the market is unknown.

    Raises:
        ValueError: If columns are missing or a selection is priced twice.
    """
    _require(offers, ("match_id", "market_id", "selection", "odds", "push_probability"))
    _require_unique(offers)
    keys = _group_keys(offers)
    odds = offers["odds"].to_numpy(dtype="float64")
    quoted = np.isfinite(odds) & (odds > 1.0)
    work = offers[keys].copy()
    work["implied"] = np.where(quoted, 1.0 / np.where(quoted, odds, 2.0), 0.0)
    work["quoted"] = quoted.astype("int64")
    grouped = work.groupby(keys, sort=False)
    total = grouped["implied"].transform("sum")
    count = grouped["quoted"].transform("sum")
    shapes = _market_shapes()
    size = offers["market_id"].map(shapes["n_selections"])
    cover = offers["market_id"].map(shapes["cover"])
    complete = (count == size) & cover.notna()
    p_push = offers["push_probability"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        reference = work["implied"] / total * cover * (1.0 - p_push)
    return reference.where(complete)


def reference_from_bookmaker(offers: pd.DataFrame, bookmaker: str) -> pd.Series:
    """One bookmaker's de-margined probabilities, looked up for every offer.

    Use it to judge a soft bookmaker's prices against a sharp one's: the sharp
    book's margin-free view of each selection becomes the reference for all
    books' prices of that selection.

    Returns:
        A probability per offer, aligned to ``offers``; NaN where that bookmaker
        has no complete market for the selection.

    Raises:
        ValueError: If ``offers`` has no ``bookmaker`` column.
    """
    if "bookmaker" not in offers.columns:
        raise ValueError("offers needs a 'bookmaker' column to pick a reference book")
    sharp = offers[offers["bookmaker"] == bookmaker]
    if sharp.empty:
        return pd.Series(np.nan, index=offers.index, dtype="float64")
    keys = ["match_id", "market_id", "selection"]
    known = (
        sharp[keys]
        .assign(p_reference=reference_probabilities(sharp).to_numpy(dtype="float64"))
        .dropna(subset=["p_reference"])
    )
    merged = offers[keys].merge(known, on=keys, how="left")
    return pd.Series(
        merged["p_reference"].to_numpy(dtype="float64"),
        index=offers.index,
        dtype="float64",
    )


def _choose_reference(
    table: pd.DataFrame, config: ValueConfig, reference_bookmaker: str | None
) -> tuple[pd.Series, pd.Series]:
    """Pick each offer's reference probability, best source first.

    The order is a supplied ``p_reference`` column, the reference bookmaker's
    de-margined view, the offer's own bookmaker de-margined, and finally the
    price with an assumed margin removed. The last choice sets the reference to
    ``(1 - refund) / (odds * (1 + margin))``, so a price that matches the model
    still shows a loss equal to the margin it carries: the model has to beat the
    margin before a bet shows any edge.
    """
    nothing = pd.Series(np.nan, index=table.index, dtype="float64")
    supplied = (
        table["p_reference"].astype("float64")
        if "p_reference" in table.columns
        else nothing
    )
    shared = (
        reference_from_bookmaker(table, reference_bookmaker)
        if reference_bookmaker is not None
        else nothing
    )
    own = reference_probabilities(table)
    odds = table["odds"].to_numpy(dtype="float64")
    p_push = table["push_probability"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        assumed = pd.Series(
            (1.0 - p_push) / (odds * (1.0 + config.fallback_margin)),
            index=table.index,
            dtype="float64",
        )
    reference = supplied.fillna(shared).fillna(own).fillna(assumed)
    source = np.select(
        [supplied.notna(), shared.notna(), own.notna()],
        ["supplied", "reference book", "own prices"],
        default="assumed margin",
    )
    return reference, pd.Series(source, index=table.index)


# --------------------------------------------------------------------------- #
# Choosing and sizing bets
# --------------------------------------------------------------------------- #
def find_value_bets(
    offers: pd.DataFrame,
    config: ValueConfig | None = None,
    *,
    reference_bookmaker: str | None = None,
) -> pd.DataFrame:
    """Decide which offers to bet on, and how much.

    Args:
        offers: One row per bookmaker price, with ``match_id``, ``market_id``,
            ``selection``, ``odds`` (decimal), the model's ``probability`` and
            ``push_probability`` (see :func:`src.markets.fixture_markets`).
            Optional: ``bookmaker``, and ``p_reference``, a fair reference
            probability that overrides everything derived below. It means the
            same as ``probability``: the chance the selection wins, where a
            refunded stake is neither a win nor a loss.
        config: Betting rules; defaults to :class:`ValueConfig`.
        reference_bookmaker: Take the reference from this bookmaker's de-margined
            prices (a sharp book) instead of each offer's own bookmaker.

    Returns:
        Every offer plus ``p_reference``, ``reference_source``, ``p_final`` (after
        shrinkage), ``raw_edge`` (the model alone), ``edge`` (after shrinkage),
        ``kelly_full``, ``stake_fraction``, ``stake`` and ``status``. Rows are
        ordered best status first, then by edge. The statuses are ``bet`` and,
        for offers left out: ``below_min_stake``, ``match_limit`` (a better-sized
        bet on the same match was taken), ``better_price`` (another bookmaker
        pays more for the same selection), ``below_min_edge`` (including edges
        of zero or less), ``low_probability``, ``odds_out_of_range``,
        ``invalid_probability`` and ``invalid_odds`` (these two leave the edge
        columns empty).

    Raises:
        ValueError: If required columns are missing or a selection is priced
            twice by one bookmaker.
    """
    config = config or ValueConfig()
    _require(offers, REQUIRED_COLUMNS)
    _require_unique(offers)
    table = offers.copy().reset_index(drop=True)
    odds = table["odds"].to_numpy(dtype="float64")
    p_model = table["probability"].to_numpy(dtype="float64")
    p_push = table["push_probability"].to_numpy(dtype="float64")
    reference, source = _choose_reference(table, config, reference_bookmaker)
    p_ref = reference.to_numpy(dtype="float64")

    odds_ok = np.isfinite(odds) & (odds > 1.0)
    with np.errstate(invalid="ignore"):
        probabilities_ok = (
            np.isfinite(p_model)
            & np.isfinite(p_push)
            & np.isfinite(p_ref)
            & (p_model >= 0.0)
            & (p_push >= 0.0)
            & (p_ref >= 0.0)
            & (p_ref <= 1.0)
            & (p_model + p_push <= 1.0 + PROBABILITY_TOLERANCE)
        )
    usable = odds_ok & probabilities_ok

    # Rows that cannot be priced get harmless stand-ins, then NaN results.
    price = np.where(usable, odds, 2.0)
    model = np.where(usable, p_model, 0.5)
    refund = np.where(usable, p_push, 0.0)
    anchor = np.where(usable, p_ref, 0.5)
    final = np.minimum(
        blend_probability(model, anchor, config.model_weight), 1.0 - refund
    )
    raw_edge = expected_value(model, refund, price)
    edge = expected_value(final, refund, price)
    kelly = full_kelly(final, refund, price)

    status = np.select(
        [
            ~odds_ok,
            ~probabilities_ok,
            (odds < config.min_odds) | (odds > config.max_odds),
            final < config.min_probability,
            ~(edge > 0.0) | (edge < config.min_edge),
        ],
        [
            "invalid_odds",
            "invalid_probability",
            "odds_out_of_range",
            "low_probability",
            "below_min_edge",
        ],
        default="candidate",
    ).astype(object)

    # A selection is bet once, at its best price; a match gets one bet by default,
    # the selection Kelly would stake most on.
    candidates = np.flatnonzero(status == "candidate")
    pool = pd.DataFrame(
        {
            "match_id": table["match_id"].to_numpy()[candidates],
            "market_id": table["market_id"].to_numpy()[candidates],
            "selection": table["selection"].to_numpy()[candidates],
            "kelly": kelly[candidates],
            "edge": edge[candidates],
        },
        index=candidates,
    ).sort_values(["kelly", "edge"], ascending=False, kind="stable")
    repeated = pool.duplicated(subset=["match_id", "market_id", "selection"])
    status[pool.index[repeated.to_numpy()]] = "better_price"
    pool = pool[~repeated.to_numpy()]
    place = pool.groupby("match_id", sort=False).cumcount()
    status[place.index[place.to_numpy() >= config.max_bets_per_match]] = "match_limit"

    chosen = status == "candidate"
    fraction = np.where(
        chosen,
        np.minimum(config.kelly_fraction * kelly, config.max_stake_fraction),
        0.0,
    )
    total = float(fraction.sum())
    if total > config.max_total_fraction:
        fraction = fraction * (config.max_total_fraction / total)
    stake = fraction * config.bankroll
    too_small = chosen & ((stake <= 0.0) | (stake < config.min_stake))
    status[too_small] = "below_min_stake"
    status[status == "candidate"] = "bet"

    table["p_reference"] = reference
    table["reference_source"] = source
    table["p_final"] = np.where(usable, final, np.nan)
    table["raw_edge"] = np.where(usable, raw_edge, np.nan)
    table["edge"] = np.where(usable, edge, np.nan)
    table["kelly_full"] = np.where(usable, kelly, np.nan)
    table["stake_fraction"] = np.where(too_small, 0.0, fraction)
    table["stake"] = np.where(too_small, 0.0, stake)
    table["status"] = status
    rank = {name: position for position, name in enumerate(STATUS_ORDER)}
    return (
        table.assign(_order=table["status"].map(rank))
        .sort_values(
            ["_order", "edge"],
            ascending=[True, False],
            na_position="last",
            kind="stable",
        )
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def selected_bets(result: pd.DataFrame) -> pd.DataFrame:
    """The rows of :func:`find_value_bets` that are actual bets."""
    return result[result["status"] == "bet"].reset_index(drop=True)
