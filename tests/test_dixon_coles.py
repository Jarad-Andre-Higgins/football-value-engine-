"""Unit tests for the Dixon-Coles score-matrix engine."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from src.dixon_coles import poisson_pmf, rho_bounds, score_matrix


@pytest.mark.parametrize(
    ("lam", "mu", "rho"),
    [(1.6, 1.1, -0.10), (0.4, 0.3, 0.0), (3.2, 2.5, -0.05)],
)
def test_matrix_is_a_valid_probability_distribution(
    lam: float, mu: float, rho: float
) -> None:
    matrix = score_matrix(lam, mu, rho)
    assert matrix.shape == (11, 11)
    assert (matrix >= 0).all()
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)


def test_poisson_pmf_matches_scipy_reference() -> None:
    np.testing.assert_allclose(
        poisson_pmf(1.7, 10), poisson.pmf(np.arange(11), 1.7), rtol=1e-12
    )


def test_zero_rho_reduces_to_independent_poisson() -> None:
    matrix = score_matrix(1.5, 1.2, rho=0.0)
    independent = np.outer(poisson_pmf(1.5), poisson_pmf(1.2))
    np.testing.assert_allclose(matrix, independent / independent.sum(), atol=1e-12)


def test_negative_rho_shifts_mass_toward_low_scoring_draws() -> None:
    base = score_matrix(1.4, 1.1, rho=0.0)
    adjusted = score_matrix(1.4, 1.1, rho=-0.10)
    assert adjusted[0, 0] > base[0, 0]
    assert adjusted[1, 1] > base[1, 1]
    assert adjusted[0, 1] < base[0, 1]
    assert adjusted[1, 0] < base[1, 0]


def test_correction_leaves_goal_marginals_unchanged() -> None:
    base = score_matrix(1.7, 1.2, rho=0.0)
    adjusted = score_matrix(1.7, 1.2, rho=-0.12)
    np.testing.assert_allclose(adjusted.sum(axis=1), base.sum(axis=1), atol=1e-12)
    np.testing.assert_allclose(adjusted.sum(axis=0), base.sum(axis=0), atol=1e-12)


def test_swapping_teams_transposes_the_matrix() -> None:
    forward = score_matrix(1.9, 0.8, rho=-0.08)
    swapped = score_matrix(0.8, 1.9, rho=-0.08)
    np.testing.assert_allclose(forward, swapped.T, atol=1e-12)


def test_custom_grid_size() -> None:
    matrix = score_matrix(1.2, 1.0, rho=-0.05, max_goals=6)
    assert matrix.shape == (7, 7)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)


def test_rho_bounds_values() -> None:
    lower, upper = rho_bounds(1.5, 1.0)
    assert lower == pytest.approx(-2 / 3)
    assert upper == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("lam", "mu", "rho", "max_goals"),
    [
        (0.0, 1.0, 0.0, 10),
        (-1.0, 1.0, 0.0, 10),
        (1.0, float("nan"), 0.0, 10),
        (1.0, float("inf"), 0.0, 10),
        (1.5, 1.0, -0.70, 10),  # below the lower bound of -2/3
        (1.5, 1.0, 0.70, 10),  # above the upper bound of 2/3
        (1.5, 1.0, -0.10, 0),  # grid too small for the correction
    ],
)
def test_invalid_inputs_raise_value_error(
    lam: float, mu: float, rho: float, max_goals: int
) -> None:
    with pytest.raises(ValueError):
        score_matrix(lam, mu, rho, max_goals)
