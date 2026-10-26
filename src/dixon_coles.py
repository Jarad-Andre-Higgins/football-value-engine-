"""Dixon-Coles adjusted bivariate Poisson score model.

Given the expected goals of the home side (``lam``) and the away side (``mu``),
this module builds the exact-score probability matrix::

    matrix[i, j] = P(home scores i goals, away scores j goals)

Independent Poisson goals under-predict low-scoring draws (0-0, 1-1) and
over-predict narrow wins (1-0, 0-1). Dixon & Coles (1997) fix this by
multiplying those four cells by a correction factor `tau` controlled by a
single dependence parameter `rho` (usually a small negative number).

The correction only moves probability between the four corner cells: both goal
marginals, and therefore the expected goals `lam` and `mu`, are unchanged.

Run the smoke demo with `python -m src.dixon_coles`.
"""

from __future__ import annotations

import logging
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_GOALS: Final[int] = 10

FloatArray = NDArray[np.float64]


def poisson_pmf(rate: float, max_goals: int = DEFAULT_MAX_GOALS) -> FloatArray:
    """Poisson probabilities `P(K = k)` for `k = 0 .. max_goals`.

    Uses the recurrence `p[k] = p[k-1] * rate / k` which is exact and avoids
    factorial overflow.

    Args:
        rate: Expected number of goals (> 0).
        max_goals: Highest goal count to include.

    Returns:
        Array of length `max_goals + 1`.

    Raises:
        ValueError: If `rate` is not a positive finite number or
            `max_goals` is negative.
    """
    _validate_rate(rate, "rate")
    if max_goals < 0:
        raise ValueError(f"max_goals must be >= 0, got {max_goals}")
    pmf = np.empty(max_goals + 1, dtype=np.float64)
    pmf[0] = math.exp(-rate)
    pmf[1:] = pmf[0] * np.cumprod(rate / np.arange(1, max_goals + 1))
    return pmf


def rho_bounds(lam: float, mu: float) -> tuple[float, float]:
    """Open interval of `rho` for which every `tau` factor stays positive.

    The four correction factors are `1 - lam*mu*rho`, `1 + lam*rho`,
    `1 + mu*rho` and `1 - rho`; a non-positive factor would produce a
    negative probability.

    Args:
        lam: Home expected goals (> 0).
        mu: Away expected goals (> 0).

    Returns:
        `(lower, upper)`; valid values satisfy `lower < rho < upper`.
    """
    _validate_rate(lam, "lam")
    _validate_rate(mu, "mu")
    lower = max(-1.0 / lam, -1.0 / mu)
    upper = min(1.0, 1.0 / (lam * mu))
    return lower, upper


def score_matrix(
    lam: float,
    mu: float,
    rho: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> FloatArray:
    """Build the Dixon-Coles exact-score probability matrix.

    Args:
        lam: Expected goals of the home team (> 0).
        mu: Expected goals of the away team (> 0).
        rho: Dixon-Coles dependence parameter; `0` means independent Poisson.
        max_goals: Highest goal count modelled per team (default 10).

    Returns:
        Array of shape `(max_goals + 1, max_goals + 1)`. Rows index home
        goals, columns index away goals, and all cells sum to 1.

    Raises:
        ValueError: If a rate is not positive and finite, `rho` lies outside
            :func:`rho_bounds`, or `max_goals` is below 1.
    """
    if max_goals < 1:
        raise ValueError(f"max_goals must be >= 1, got {max_goals}")
    lower, upper = rho_bounds(lam, mu)
    if not lower < rho < upper:
        raise ValueError(
            f"rho={rho} is outside the valid range ({lower:.4f}, {upper:.4f}) "
            f"for lam={lam}, mu={mu}"
        )

    matrix = np.outer(poisson_pmf(lam, max_goals), poisson_pmf(mu, max_goals))
    matrix[0, 0] *= 1.0 - lam * mu * rho
    matrix[0, 1] *= 1.0 + lam * rho
    matrix[1, 0] *= 1.0 + mu * rho
    matrix[1, 1] *= 1.0 - rho
    # The correction conserves probability mass; only the truncation tail
    # (> max_goals) is missing, so renormalise for numerical hygiene.
    matrix /= matrix.sum()
    return matrix


def _validate_rate(value: float, label: str) -> None:
	"""Raise `ValueError` unless `value` is a positive finite number."""
	if not math.isfinite(value) or value <= 0:
		raise ValueError(f"{label} must be a positive finite number, got {value}")


def main() -> None:
    """Smoke-test the engine on a hypothetical fixture and log a summary.

    The full 50+ market parser arrives in Phase 5; this only checks that the
    matrix behaves sensibly.
    """
    log = setup_logging("dixon_coles_demo")
    lam, mu, rho = 1.80, 0.90, -0.10
    matrix = score_matrix(lam, mu, rho)

    goals = np.arange(matrix.shape[0])
    total_goals = np.add.outer(goals, goals)
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    over_25 = float(matrix[total_goals >= 3].sum())
    btts = float(matrix[1:, 1:].sum())

    log.info("Fixture xG: home=%.2f away=%.2f rho=%.2f", lam, mu, rho)
    log.info("Matrix shape=%s sum=%.12f", matrix.shape, matrix.sum())
    log.info(
        "1X2: home=%.1f%% draw=%.1f%% away=%.1f%%",
        100 * home_win,
        100 * draw,
        100 * away_win,
    )
    log.info(
        "Over/Under 2.5: over=%.1f%% under=%.1f%%", 100 * over_25, 100 * (1 - over_25)
    )
    log.info("BTTS: yes=%.1f%% no=%.1f%%", 100 * btts, 100 * (1 - btts))
    for flat_index in np.argsort(matrix, axis=None)[::-1][:5]:
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        log.info(
            "Likely score %d-%d: %.2f%%",
            home_goals,
            away_goals,
            100 * matrix[home_goals, away_goals],
        )


if __name__ == "__main__":
    main()
