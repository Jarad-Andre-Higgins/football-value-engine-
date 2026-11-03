"""From one exact-score matrix to 50+ betting markets.

The score matrix is a chessboard: square ``(i, j)`` holds the probability that
the home side scores ``i`` goals and the away side ``j``. Every betting market
is just a different set of squares to highlight. "Over 2.5" highlights each
square where ``i + j > 2``; "both teams to score" each square where both are at
least 1. A selection's probability is the sum of its squares, so one matrix
prices every market consistently.

A selection has a *win* set of squares and, for markets that refund the stake
(draw no bet, whole-number Asian handicaps), a *push* set. Its fair odds are
``(1 - push) / win``. Half-time, first-goal and timing markets cannot be derived
from a full-time score grid, so they are not included.

Run ``python -m src.markets`` for a demo.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.dixon_coles import DEFAULT_MAX_GOALS, FloatArray, rho_bounds, score_matrix
from src.evaluate import MAX_RATE, MIN_RATE, RHO_MARGIN
from src.logging_config import setup_logging

logger = logging.getLogger("src.markets")

BoolGrid = NDArray[np.bool_]
MIN_GRID_GOALS: Final[int] = 6  # the correct-score market needs scores up to 5-5
OU_LINES: Final[tuple[float, ...]] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
TEAM_TOTAL_LINES: Final[tuple[float, ...]] = (0.5, 1.5, 2.5)
EH_LINES: Final[tuple[int, ...]] = (-3, -2, -1, 1, 2, 3)
AH_LINES: Final[tuple[float, ...]] = tuple(x / 2 for x in range(-5, 6))
COMBO_OU_LINES: Final[tuple[float, ...]] = (1.5, 2.5, 3.5, 4.5)
CORRECT_SCORE_MAX: Final[int] = 5


@dataclass(frozen=True)
class _Selection:
    name: str
    win: BoolGrid
    push: BoolGrid | None = None


@dataclass(frozen=True)
class _Market:
    market_id: str
    family: str
    label: str
    line: float | None
    exclusive: bool  # True when exactly one selection (or a push) always happens
    selections: list[_Selection]


@dataclass(frozen=True)
class MarketSet:
    """Every market as matrices, so any number of fixtures price in one product.

    Attributes:
        catalog: One row per selection: ``market_id``, ``family``, ``label``,
            ``line``, ``selection`` and ``exclusive``.
        win: ``(selections, cells)`` 0/1 matrix of winning squares.
        push: The same for squares that refund the stake (mostly zeros).
        max_goals: Largest goal count on the grid.
    """

    catalog: pd.DataFrame
    win: FloatArray
    push: FloatArray
    max_goals: int


def _definitions(n: int) -> list[_Market]:
    """Define every market as boolean masks over an ``n`` by ``n`` score grid."""
    home, away = np.indices((n, n))
    total, margin = home + away, home - away
    markets: list[_Market] = []

    def add(
        market_id: str,
        family: str,
        label: str,
        line: float | None,
        exclusive: bool,
        **sels: BoolGrid,
    ) -> None:
        markets.append(
            _Market(
                market_id,
                family,
                label,
                line,
                exclusive,
                [_Selection(k, v) for k, v in sels.items()],
            )
        )

    def lined(
        family: str, label: str, line: float, exclusive: bool, **sels: BoolGrid
    ) -> None:
        add(
            f"{family}:{line:g}",
            family,
            f"{label} {line:+g}" if family in ("EH", "AH") else f"{label} {line:g}",
            line,
            exclusive,
            **sels,
        )

    add(
        "1X2",
        "1X2",
        "Match result",
        None,
        True,
        home=margin > 0,
        draw=margin == 0,
        away=margin < 0,
    )
    add(
        "DC",
        "DC",
        "Double chance",
        None,
        False,
        home_draw=margin >= 0,
        home_away=margin != 0,
        draw_away=margin <= 0,
    )
    markets.append(
        _Market(
            "DNB", "DNB", "Draw no bet", None, True,
            [_Selection("home", margin > 0, margin == 0), _Selection("away", margin < 0, margin == 0)],
        )
    )  # fmt: skip
    for line in OU_LINES:
        lined(
            "OU",
            "Total goals over/under",
            line,
            True,
            over=total > line,
            under=total < line,
        )
    add(
        "BTTS",
        "BTTS",
        "Both teams to score",
        None,
        True,
        yes=(home >= 1) & (away >= 1),
        no=(home == 0) | (away == 0),
    )
    add(
        "OE",
        "OE",
        "Total goals odd/even",
        None,
        True,
        odd=total % 2 == 1,
        even=total % 2 == 0,
    )
    add(
        "TG",
        "TG",
        "Exact total goals",
        None,
        True,
        **{str(k): total == k for k in range(6)},
        **{"6+": total >= 6},
    )
    add(
        "GB4",
        "GB",
        "Goal brackets (4)",
        None,
        True,
        **{
            "0-1": total <= 1,
            "2-3": (total >= 2) & (total <= 3),
            "4-5": (total >= 4) & (total <= 5),
            "6+": total >= 6,
        },
    )
    add(
        "GB3",
        "GB",
        "Goal brackets (3)",
        None,
        True,
        **{"0-1": total <= 1, "2-4": (total >= 2) & (total <= 4), "5+": total >= 5},
    )
    for side, goals in (("home", home), ("away", away)):
        for line in TEAM_TOTAL_LINES:
            lined(
                f"TT_{side.upper()}",
                f"{side.capitalize()} team goals over/under",
                line,
                True,
                over=goals > line,
                under=goals < line,
            )
        add(
            f"TG_{side.upper()}",
            "TG",
            f"Exact {side} team goals",
            None,
            True,
            **{str(k): goals == k for k in range(3)},
            **{"3+": goals >= 3},
        )
        add(
            f"TS_{side.upper()}",
            "TS",
            f"{side.capitalize()} team to score",
            None,
            True,
            yes=goals >= 1,
            no=goals == 0,
        )
    add(
        "WTN_HOME",
        "WTN",
        "Home win to nil",
        None,
        True,
        yes=(home > 0) & (away == 0),
        no=~((home > 0) & (away == 0)),
    )
    add(
        "WTN_AWAY",
        "WTN",
        "Away win to nil",
        None,
        True,
        yes=(away > 0) & (home == 0),
        no=~((away > 0) & (home == 0)),
    )
    for handicap in EH_LINES:
        shifted = margin + handicap
        lined(
            "EH",
            "European handicap (home)",
            float(handicap),
            True,
            home=shifted > 0,
            draw=shifted == 0,
            away=shifted < 0,
        )
    for handicap_line in AH_LINES:
        adjusted = margin + handicap_line
        push = adjusted == 0
        markets.append(
            _Market(
                f"AH:{handicap_line:g}", "AH", f"Asian handicap (home) {handicap_line:+g}", handicap_line, True,
                [_Selection("home", adjusted > 0, push if push.any() else None), _Selection("away", adjusted < 0, push if push.any() else None)],
            )
        )  # fmt: skip
    add("WM", "WM", "Winning margin", None, True,
        home_by_1=margin == 1, home_by_2=margin == 2, home_by_3plus=margin >= 3, draw=margin == 0,
        away_by_1=margin == -1, away_by_2=margin == -2, away_by_3plus=margin <= -3)  # fmt: skip

    result = {"home": margin > 0, "draw": margin == 0, "away": margin < 0}
    double_chance = {
        "home_draw": margin >= 0,
        "home_away": margin != 0,
        "draw_away": margin <= 0,
    }
    btts = {"yes": (home >= 1) & (away >= 1), "no": (home == 0) | (away == 0)}
    add("1X2_BTTS", "1X2_BTTS", "Result and both teams to score", None, True,
        **{f"{r}_{b}": rm & bm for r, rm in result.items() for b, bm in btts.items()})  # fmt: skip
    for line in COMBO_OU_LINES:
        lined("1X2_OU", "Result and total goals", line, True,
              **{f"{r}_{o}": rm & om for r, rm in result.items() for o, om in (("over", total > line), ("under", total < line))})  # fmt: skip
    for line in (1.5, 2.5, 3.5):
        lined("DC_OU", "Double chance and total goals", line, False,
              **{f"{d}_{o}": dm & om for d, dm in double_chance.items() for o, om in (("over", total > line), ("under", total < line))})  # fmt: skip
    add("DC_BTTS", "DC_BTTS", "Double chance and both teams to score", None, False,
        **{f"{d}_{b}": dm & bm for d, dm in double_chance.items() for b, bm in btts.items()})  # fmt: skip
    over_25 = total > 2.5
    add("BTTS_OU:2.5", "BTTS_OU", "Both teams to score and over/under 2.5", 2.5, True,
        yes_over=btts["yes"] & over_25, yes_under=btts["yes"] & ~over_25, no_over=btts["no"] & over_25, no_under=btts["no"] & ~over_25)  # fmt: skip

    scores = {
        f"{i}-{j}": (home == i) & (away == j)
        for i in range(CORRECT_SCORE_MAX + 1)
        for j in range(CORRECT_SCORE_MAX + 1)
    }
    listed = np.zeros((n, n), dtype=bool)
    for mask in scores.values():
        listed |= mask
    add("CS", "CS", "Correct score", None, True, **scores, other=~listed)
    return markets


@cache
def market_set(max_goals: int = DEFAULT_MAX_GOALS) -> MarketSet:
    """Build (once) the catalog and winning-square matrices for a grid size.

    Raises:
        ValueError: If the grid is too small to hold the correct-score market.
    """
    if max_goals < MIN_GRID_GOALS:
        raise ValueError(f"max_goals must be >= {MIN_GRID_GOALS}, got {max_goals}")
    n = max_goals + 1
    rows, wins, pushes = [], [], []
    for market in _definitions(n):
        for selection in market.selections:
            rows.append(
                (
                    market.market_id,
                    market.family,
                    market.label,
                    market.line,
                    selection.name,
                    market.exclusive,
                )
            )
            wins.append(selection.win.ravel())
            pushes.append(
                np.zeros(n * n, dtype=bool)
                if selection.push is None
                else selection.push.ravel()
            )
    catalog = pd.DataFrame(
        rows, columns=["market_id", "family", "label", "line", "selection", "exclusive"]
    )
    win, push = np.array(wins, dtype="float64"), np.array(pushes, dtype="float64")
    win.flags.writeable = False
    push.flags.writeable = False
    return MarketSet(catalog, win, push, max_goals)


def selection_probabilities(matrices: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Price every selection for one matrix or a stack of matrices.

    Args:
        matrices: Score matrices of shape ``(n, n)`` or ``(fixtures, n, n)``.

    Returns:
        ``(win, push)`` probabilities, each of shape ``(fixtures, selections)``
        in the order of ``market_set(n - 1).catalog``.

    Raises:
        ValueError: If the grid is not square or is too small.
    """
    grid = np.asarray(matrices, dtype="float64")
    if grid.ndim == 2:
        grid = grid[np.newaxis]
    if grid.ndim != 3 or grid.shape[1] != grid.shape[2]:
        raise ValueError(f"expected square score matrices, got shape {grid.shape}")
    markets = market_set(grid.shape[1] - 1)
    flat = grid.reshape(grid.shape[0], -1)
    return flat @ markets.win.T, flat @ markets.push.T


def _fair_odds(win: FloatArray, push: FloatArray) -> FloatArray:
    """Odds at which a selection is break-even, allowing for refunded stakes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(win > 0, (1.0 - push) / win, np.inf)


def market_table(matrix: FloatArray) -> pd.DataFrame:
    """Every market for one score matrix, with fair odds.

    Returns:
        The catalog plus ``probability``, ``push_probability`` and ``fair_odds``.
    """
    win, push = selection_probabilities(matrix)
    table = market_set(np.asarray(matrix).shape[-1] - 1).catalog.copy()
    table["probability"] = win[0]
    table["push_probability"] = push[0]
    table["fair_odds"] = _fair_odds(win[0], push[0])
    return table


def _matrix_for(lam: float, mu: float, rho: float, max_goals: int) -> FloatArray:
    """Score matrix with rates and ``rho`` clipped into their valid ranges."""
    lam = min(max(float(lam), MIN_RATE), MAX_RATE)
    mu = min(max(float(mu), MIN_RATE), MAX_RATE)
    lower, upper = rho_bounds(lam, mu)
    return score_matrix(
        lam, mu, min(max(rho, lower + RHO_MARGIN), upper - RHO_MARGIN), max_goals
    )


def fixture_markets(
    rates: pd.DataFrame,
    rho: float,
    match_ids: Sequence[str] | None = None,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> pd.DataFrame:
    """Price every market for many fixtures at once.

    Args:
        rates: Goal rates with columns ``lam_home`` and ``lam_away`` (for example
            from :func:`src.bundle.predict_rates`).
        rho: Dixon-Coles dependence parameter.
        match_ids: Identifier per row; defaults to ``rates.index``.
        max_goals: Size of the score grid.

    Returns:
        Long table with ``match_id``, the catalog columns, ``probability``,
        ``push_probability`` and ``fair_odds``.
    """
    ids = list(rates.index if match_ids is None else match_ids)
    if len(ids) != len(rates):
        raise ValueError("match_ids must have one entry per row of rates")
    matrices = np.stack(
        [
            _matrix_for(a, b, rho, max_goals)
            for a, b in zip(rates["lam_home"], rates["lam_away"])
        ]
    )
    win, push = selection_probabilities(matrices)
    catalog = market_set(max_goals).catalog
    table = pd.concat([catalog] * len(ids), ignore_index=True)
    table.insert(0, "match_id", np.repeat(ids, len(catalog)))
    table["probability"] = win.ravel()
    table["push_probability"] = push.ravel()
    table["fair_odds"] = _fair_odds(win.ravel(), push.ravel())
    return table


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Price every market for one fixture.")
    parser.add_argument(
        "--lam", type=float, default=1.60, help="home goal rate (default 1.60)"
    )
    parser.add_argument(
        "--mu", type=float, default=1.10, help="away goal rate (default 1.10)"
    )
    parser.add_argument(
        "--rho", type=float, default=-0.044, help="Dixon-Coles rho (default -0.044)"
    )
    parser.add_argument(
        "--all", action="store_true", help="print every selection, not a summary"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the markets for one fixture and return the exit code (always 0)."""
    args = _parse_args(argv)
    log = setup_logging("markets")
    table = market_table(_matrix_for(args.lam, args.mu, args.rho, DEFAULT_MAX_GOALS))
    log.info(
        "Fixture rates: home %.2f, away %.2f, rho %.3f -> %d markets, %d selections",
        args.lam, args.mu, args.rho, table["market_id"].nunique(), len(table),
    )  # fmt: skip
    if args.all:
        shown = table
    else:
        keys = [
            "1X2",
            "DC",
            "DNB",
            "OU:2.5",
            "BTTS",
            "EH:-1",
            "AH:-0.5",
            "TT_HOME:1.5",
            "WM",
        ]
        top_scores = table[table["market_id"] == "CS"].nlargest(5, "probability")
        shown = pd.concat([table[table["market_id"].isin(keys)], top_scores])
    columns = ["market_id", "selection", "probability", "fair_odds"]
    log.info(
        "\n%s", shown[columns].to_string(index=False, float_format="{:.3f}".format)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
