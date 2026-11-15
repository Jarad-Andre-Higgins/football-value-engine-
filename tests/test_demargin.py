"""Tests for the margin-removal methods."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import brentq

from src.demargin import (
    DEFAULT_METHOD,
    METHOD_NAMES,
    METHODS,
    _shin_z,
    check_method,
    remove_margin,
)

FloatArray = NDArray[np.float64]


def random_markets(selections: int, rows: int, seed: int) -> FloatArray:
    """Decimal odds of markets with a margin of 0.5% to 15%, every price above 1."""
    rng = np.random.default_rng(seed)
    true = np.clip(rng.dirichlet(np.full(selections, 2.0), size=rows), 0.03, 0.85)
    true = true / true.sum(axis=1, keepdims=True)
    margin = rng.uniform(0.005, 0.15, size=(rows, 1))
    implied = true * (1.0 + margin)
    return np.asarray(1.0 / implied[implied.max(axis=1) < 0.98], dtype="float64")


def shin_by_root_finder(implied: FloatArray) -> FloatArray:
    """Shin's probabilities from the textbook formula and a root finder."""
    booksum = implied.sum()

    def share(z: float) -> FloatArray:
        root = np.sqrt(z**2 + 4.0 * (1.0 - z) * implied**2 / booksum)
        return np.asarray((root - z) / (2.0 * (1.0 - z)), dtype="float64")

    z = brentq(lambda value: share(value).sum() - 1.0, 0.0, 0.999, xtol=1e-15)
    return share(z)


def power_by_root_finder(implied: FloatArray) -> FloatArray:
    """The power method's probabilities from a root finder."""
    k = brentq(lambda value: (implied**value).sum() - 1.0, 1.0, 2000.0, xtol=1e-14)
    return np.asarray(implied**k, dtype="float64")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #
def test_the_three_methods_and_their_default() -> None:
    assert METHODS == ("proportional", "power", "shin")
    assert DEFAULT_METHOD == "proportional"


def test_every_method_has_its_own_readable_name() -> None:
    assert list(METHOD_NAMES) == list(METHODS)
    assert len(set(METHOD_NAMES.values())) == len(METHODS)
    assert all(name.strip() for name in METHOD_NAMES.values())


def test_check_method_accepts_the_known_methods() -> None:
    for method in METHODS:
        check_method(method)


def test_check_method_rejects_others_and_lists_the_choices() -> None:
    with pytest.raises(ValueError, match="unknown margin-removal method 'additive'"):
        check_method("additive")
    with pytest.raises(ValueError, match=r"\['proportional', 'power', 'shin'\]"):
        check_method("Shin")


def test_remove_margin_checks_the_method_before_anything_else() -> None:
    with pytest.raises(ValueError, match="unknown margin-removal method"):
        remove_margin([], "additive")


# --------------------------------------------------------------------------- #
# Known answers
# --------------------------------------------------------------------------- #
def test_proportional_divides_the_inverse_odds_by_their_sum() -> None:
    # 1/2 + 2/7 + 1/4 = 29/28, so the probabilities are 14/29, 8/29 and 7/29.
    fair = remove_margin([2.0, 3.5, 4.0], "proportional")
    assert fair == pytest.approx([14 / 29, 8 / 29, 7 / 29], rel=1e-12)


def test_the_default_method_is_proportional() -> None:
    odds = [2.6, 2.4, 4.3]
    assert remove_margin(odds) == pytest.approx(remove_margin(odds, "proportional"))


# Solved to 60 digits with plain bisection on the textbook equations, independently
# of the code under test.
GOLDEN = [
    (
        (2.6, 2.4, 4.3),
        {
            "proportional": (0.372025955299, 0.403028118241, 0.224945926460),
            "power": (0.372984404069, 0.405107167457, 0.221908428474),
            "shin": (0.372994060332, 0.404779410920, 0.222226528748),
        },
    ),
    (
        (1.5, 2.74),
        {
            "proportional": (0.646226415094, 0.353773584906),
            "power": (0.653152279805, 0.346847720195),
            "shin": (0.650851581509, 0.349148418491),
        },
    ),
    (
        (1.25, 6.5, 13.0),
        {
            "proportional": (0.776119402985, 0.149253731343, 0.074626865672),
            "power": (0.791443750047, 0.140576813299, 0.067979436655),
            "shin": (0.786285360898, 0.144938228801, 0.068776410301),
        },
    ),
]


@pytest.mark.parametrize(("odds", "expected"), GOLDEN)
@pytest.mark.parametrize("method", METHODS)
def test_golden_values(
    odds: tuple[float, ...], expected: dict[str, tuple[float, ...]], method: str
) -> None:
    fair = remove_margin(np.array(odds), method)
    assert fair == pytest.approx(expected[method], abs=1e-10)


def test_shin_insider_share_matches_the_published_reference_values() -> None:
    # The values the ``shin`` package on PyPI reports for these two markets. Its
    # three-outcome value comes from an iteration stopped at a tolerance of 1e-12.
    three = _shin_z(1.0 / np.array([[2.6, 2.4, 4.3]]))
    two = _shin_z(1.0 / np.array([[1.5, 2.74]]))
    assert three[0] == pytest.approx(0.01694251276407055, abs=1e-11)
    assert two[0] == pytest.approx(0.03172728540646625, abs=1e-13)


def test_shin_with_two_outcomes_takes_half_the_margin_off_each_side() -> None:
    odds = random_markets(2, 300, seed=1)
    implied = 1.0 / odds
    half_margin = (implied.sum(axis=1, keepdims=True) - 1.0) / 2.0
    assert remove_margin(odds, "shin") == pytest.approx(
        implied - half_margin, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("selections", [2, 3, 4, 6])
@pytest.mark.parametrize("method", METHODS)
def test_probabilities_add_up_to_one(method: str, selections: int) -> None:
    fair = remove_margin(random_markets(selections, 200, seed=selections), method)
    assert fair.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    assert ((fair > 0.0) & (fair < 1.0)).all()


@pytest.mark.parametrize("selections", [2, 3, 5])
def test_the_solvers_agree_with_a_root_finder(selections: int) -> None:
    odds = random_markets(selections, 60, seed=10 + selections)
    shin = remove_margin(odds, "shin")
    power = remove_margin(odds, "power")
    for row, quote in enumerate(odds):
        implied = 1.0 / quote
        assert shin[row] == pytest.approx(shin_by_root_finder(implied), abs=1e-10)
        assert power[row] == pytest.approx(power_by_root_finder(implied), abs=1e-10)


@pytest.mark.parametrize("method", ["power", "shin"])
def test_long_shots_lose_more_and_favourites_less_than_under_proportional(
    method: str,
) -> None:
    odds = random_markets(3, 500, seed=7)
    rows = np.arange(len(odds))
    shortest = odds.argmin(axis=1)
    longest = odds.argmax(axis=1)
    fair = remove_margin(odds, method)
    proportional = remove_margin(odds, "proportional")
    assert (fair[rows, shortest] >= proportional[rows, shortest] - 1e-12).all()
    assert (fair[rows, longest] <= proportional[rows, longest] + 1e-12).all()
    # and strictly so on a market with a clear favourite and a clear long shot
    clear = remove_margin([1.25, 6.5, 13.0], method)
    plain = remove_margin([1.25, 6.5, 13.0], "proportional")
    assert clear[0] > plain[0]
    assert clear[2] < plain[2]


@pytest.mark.parametrize("method", METHODS)
def test_the_shortest_price_keeps_the_highest_probability(method: str) -> None:
    odds = random_markets(4, 300, seed=3)
    fair = remove_margin(odds, method)
    assert (np.argsort(-fair, axis=1) == np.argsort(odds, axis=1)).all()


@pytest.mark.parametrize("method", METHODS)
def test_equal_prices_share_the_probability_equally(method: str) -> None:
    assert remove_margin([2.8, 2.8, 2.8], method) == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_a_market_with_no_margin_is_returned_unchanged(method: str) -> None:
    # 1/2 + 1/4 + 1/4 is exactly one.
    assert remove_margin([2.0, 4.0, 4.0], method) == pytest.approx(
        [0.5, 0.25, 0.25], abs=1e-12
    )


@pytest.mark.parametrize("method", METHODS)
def test_a_negative_margin_falls_back_to_proportional_scaling(method: str) -> None:
    odds = [2.2, 3.5, 4.5]  # the inverse odds add up to less than one
    implied = 1.0 / np.array(odds)
    assert implied.sum() < 1.0
    fair = remove_margin(odds, method)
    assert fair == pytest.approx(implied / implied.sum(), abs=1e-12)
    assert fair.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("method", ["power", "shin"])
def test_a_tiny_margin_changes_the_prices_only_a_little(method: str) -> None:
    odds = np.array([2.0, 4.0, 3.9999999])
    implied = 1.0 / odds
    assert implied.sum() > 1.0
    fair = remove_margin(odds, method)
    assert fair == pytest.approx(implied, abs=1e-7)
    assert fair.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("method", ["power", "shin"])
def test_prices_far_below_fair_value_still_solve(method: str) -> None:
    # The inverse odds add up to almost three; the power needed is about 100.
    odds = np.array([1.01, 1.02, 1.03])
    implied = 1.0 / odds
    expected = (
        power_by_root_finder(implied)
        if method == "power"
        else shin_by_root_finder(implied)
    )
    assert remove_margin(odds, method) == pytest.approx(expected, abs=1e-9)
    equal = remove_margin([1.01, 1.01, 1.01], method)
    assert equal == pytest.approx(1 / 3, abs=1e-9)


# --------------------------------------------------------------------------- #
# Shapes and bad input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", METHODS)
def test_the_result_has_the_shape_of_the_input(method: str) -> None:
    assert remove_margin([2.0, 3.5, 4.0], method).shape == (3,)
    assert remove_margin(random_markets(3, 5, seed=0), method).shape[1] == 3
    assert remove_margin([[2.0, 3.5, 4.0], [1.5, 2.74, 9.0]], method).shape == (2, 3)
    assert remove_margin(np.empty((0, 3)), method).shape == (0, 3)


@pytest.mark.parametrize("method", METHODS)
def test_unusable_markets_are_nan_and_leave_the_others_alone(method: str) -> None:
    good = [2.6, 2.4, 4.3]
    other = [1.5, 2.74, 9.0]
    table = np.array(
        [
            good,
            [2.0, np.nan, 4.0],
            other,
            [1.0, 3.0, 4.0],
            [0.8, 3.0, 4.0],
            [2.0, np.inf, 4.0],
            [-2.0, 3.0, 4.0],
        ]
    )
    fair = remove_margin(table, method)
    assert fair[0] == pytest.approx(remove_margin(good, method), rel=1e-12)
    assert fair[2] == pytest.approx(remove_margin(other, method), rel=1e-12)
    assert np.isnan(fair[[1, 3, 4, 5, 6]]).all()
    assert np.isfinite(fair[[0, 2]]).all()


@pytest.mark.parametrize("method", METHODS)
def test_a_batch_gives_what_each_market_gives_alone(method: str) -> None:
    odds = random_markets(3, 40, seed=5)
    batch = remove_margin(odds, method)
    for row, quote in enumerate(odds):
        assert batch[row] == pytest.approx(remove_margin(quote, method), rel=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_the_input_is_left_untouched(method: str) -> None:
    odds = random_markets(3, 10, seed=2)
    before = odds.copy()
    remove_margin(odds, method)
    assert (odds == before).all()


def test_a_single_market_may_be_a_plain_list() -> None:
    fair = remove_margin([2.0, 3.5, 4.0])
    assert isinstance(fair, np.ndarray)
    assert fair.dtype == np.float64


@pytest.mark.parametrize(
    "odds",
    [[2.0], [[2.0], [3.0]], [], np.ones((2, 2, 2))],
    ids=["one price", "one column", "empty", "three dimensions"],
)
def test_odds_that_are_not_markets_are_rejected(odds: object) -> None:
    with pytest.raises(ValueError, match="at least two selections"):
        remove_margin(odds)  # type: ignore[arg-type]
