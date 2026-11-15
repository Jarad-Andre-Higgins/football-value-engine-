"""Taking the bookmaker's margin out of its prices.

A bookmaker's decimal odds ``o`` imply a probability ``1 / o`` for each outcome,
but the implied probabilities of a market add up to more than one. The excess,
``booksum - 1``, is the margin: the fee the bookmaker charges. To read the
market's own view of the outcomes, that fee has to be taken back out, and how it
is shared between the outcomes is a modelling choice. Three choices are
implemented; each returns probabilities that add up to exactly one.

``proportional``
    Divide every implied probability by the booksum. Every outcome pays the same
    share of its own price, so a long shot is trusted as much as a favourite.

``power``
    Raise every implied probability to one common power ``k >= 1`` chosen so that
    the results add up to one (Clarke, Kovalchik and Ingram, 2017). Powers shrink
    small numbers proportionally more than large ones, so long shots lose more of
    their implied probability than favourites do.

``shin``
    Shin's model (1992, 1993): a share ``z`` of the bettors are insiders who know
    the result, and the bookmaker shades its prices against them, mostly on long
    shots. ``z`` is solved so that the probabilities add up to one (Jullien and
    Salanie, 1994; Strumbelj, 2014). For a market with two outcomes it equals
    taking half of the margin off each side.

The last two are the usual answer to the favourite-longshot bias, the tendency of
prices to overstate the chances of long shots. Whether that bias matters in the
books and markets used here is an empirical question, which
:mod:`src.margin_audit` answers from the results.

A market whose prices add up to one or less has no margin to remove, so every
method falls back to proportional scaling there (which changes nothing at exactly
one). A market with a missing price, or one that is not above 1, has no fair
probabilities at all and comes back as NaN.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

METHODS: Final[tuple[str, ...]] = ("proportional", "power", "shin")
DEFAULT_METHOD: Final[str] = "proportional"
METHOD_NAMES: Final[dict[str, str]] = {
    "proportional": "proportional scaling",
    "power": "the power method",
    "shin": "Shin's method",
}
#: Times the bracket around the answer is halved. 64 halvings leave it far below
#: what a double can tell apart, so the answer is exact to machine precision.
_HALVINGS: Final[int] = 64


def check_method(method: str) -> None:
    """Raise ``ValueError`` unless ``method`` is one of :data:`METHODS`."""
    if method not in METHODS:
        raise ValueError(
            f"unknown margin-removal method {method!r}; choose from {list(METHODS)}"
        )


# --------------------------------------------------------------------------- #
# The three methods, on implied probabilities
# --------------------------------------------------------------------------- #
def _proportional(implied: FloatArray) -> FloatArray:
    """Scale each market (row) so it adds up to one."""
    return implied / implied.sum(axis=1, keepdims=True)


def _power_exponent(implied: FloatArray) -> FloatArray:
    """The ``k`` with ``sum(implied ** k) == 1``, one per market with a margin.

    The sum falls as ``k`` grows. At ``k = 1`` it is the booksum, above one. At
    ``ln(n) / -ln(largest)`` even ``n`` copies of the largest term add up to only
    one, so the answer lies between the two.
    """
    logs = np.log(implied)
    low = np.ones(len(implied))
    high = np.log(implied.shape[1]) / -logs.max(axis=1)
    for _ in range(_HALVINGS):
        middle = 0.5 * (low + high)
        above = np.exp(middle[:, None] * logs).sum(axis=1) > 1.0
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    return 0.5 * (low + high)


def _power(implied: FloatArray) -> FloatArray:
    fair = _proportional(implied)
    margin = implied.sum(axis=1) > 1.0
    if margin.any():
        exponent = _power_exponent(implied[margin])
        fair[margin] = _proportional(implied[margin] ** exponent[:, None])
    return fair


def _shin_probabilities(implied: FloatArray, z: FloatArray) -> FloatArray:
    """Shin's probabilities for insider share ``z``, before any renormalising.

    This is ``(sqrt(z^2 + 4 (1 - z) a) - z) / (2 (1 - z))`` with ``a`` the squared
    implied probability over the booksum, written as ``2 a / (z + sqrt(...))``.
    The two are equal, and the second does not lose digits when ``z`` is small or
    close to one.
    """
    booksum = implied.sum(axis=1, keepdims=True)
    squared = implied**2 / booksum
    share = z[:, None]
    return 2.0 * squared / (share + np.sqrt(share**2 + 4.0 * (1.0 - share) * squared))


def _shin_z(implied: FloatArray) -> FloatArray:
    """Shin's insider share ``z`` for markets whose prices add up to more than one.

    The probabilities fall as ``z`` grows. At ``z = 0`` they add up to the square
    root of the booksum, above one; at ``z = 1`` to ``sum(implied ** 2) / booksum``,
    below one for any price above 1. So the ``z`` that makes them add up to one lies
    in between and is found by halving the bracket.
    """
    low = np.zeros(len(implied))
    high = np.ones(len(implied))
    for _ in range(_HALVINGS):
        middle = 0.5 * (low + high)
        above = _shin_probabilities(implied, middle).sum(axis=1) > 1.0
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    return 0.5 * (low + high)


def _shin(implied: FloatArray) -> FloatArray:
    fair = _proportional(implied)
    margin = implied.sum(axis=1) > 1.0
    if margin.any():
        chosen = implied[margin]
        fair[margin] = _proportional(_shin_probabilities(chosen, _shin_z(chosen)))
    return fair


_SOLVERS: Final[dict[str, Callable[[FloatArray], FloatArray]]] = {
    "proportional": _proportional,
    "power": _power,
    "shin": _shin,
}


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
def remove_margin(odds: ArrayLike, method: str = DEFAULT_METHOD) -> FloatArray:
    """Fair probabilities of one market, or of many, from decimal odds.

    Args:
        odds: Decimal odds of every selection of a market, with at least two
            selections: one market as a 1-D array, or many as a 2-D array with a
            market per row. A market is complete or it is not: a row needs a
            price above 1 for every selection.
        method: One of :data:`METHODS`.

    Returns:
        An array shaped like ``odds``. Each market's probabilities add up to one;
        a row with a missing price, or a price that is not above 1, is NaN.

    Raises:
        ValueError: For an unknown ``method``, odds that are not 1-D or 2-D, or a
            market with fewer than two selections.
    """
    check_method(method)
    quoted = np.asarray(odds, dtype="float64")
    if quoted.ndim not in (1, 2) or quoted.shape[-1] < 2:
        raise ValueError(
            "odds must be one market or a table with a market per row, and a "
            "market needs at least two selections"
        )
    table = quoted.reshape(1, -1) if quoted.ndim == 1 else quoted
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(table).all(axis=1) & (table > 1.0).all(axis=1)
    fair = np.full(table.shape, np.nan)
    if usable.any():
        fair[usable] = _SOLVERS[method](1.0 / table[usable])
    return fair[0] if quoted.ndim == 1 else fair
