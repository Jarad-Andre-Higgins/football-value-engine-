"""Tests for value detection and stake sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize_scalar

from src.markets import fixture_markets, market_set
from src.value import (
    STATUS_ORDER,
    ValueConfig,
    _market_shapes,
    blend_probability,
    expected_value,
    find_value_bets,
    full_kelly,
    reference_from_bookmaker,
    reference_probabilities,
    selected_bets,
)

COLUMNS = [
    "match_id",
    "market_id",
    "selection",
    "odds",
    "probability",
    "push_probability",
]


def offers(*rows: tuple[object, ...]) -> pd.DataFrame:
    """Rows are ``(match_id, market_id, selection, odds, probability[, push])``."""
    return pd.DataFrame(
        [(*row, 0.0) if len(row) == 5 else row for row in rows], columns=COLUMNS
    )


def arr(*values: float) -> np.ndarray:
    return np.array(values, dtype="float64")


# --------------------------------------------------------------------------- #
# Expected value and Kelly
# --------------------------------------------------------------------------- #
def test_expected_value_with_and_without_a_refund() -> None:
    assert expected_value(arr(0.6), arr(0.0), arr(2.0))[0] == pytest.approx(0.2)
    assert expected_value(arr(0.4), arr(0.3), arr(2.5))[0] == pytest.approx(0.3)
    assert expected_value(arr(0.5), arr(0.0), arr(2.0))[0] == pytest.approx(0.0)


def test_full_kelly_known_cases() -> None:
    assert full_kelly(arr(0.6), arr(0.0), arr(2.0))[0] == pytest.approx(0.2)
    assert full_kelly(arr(0.4), arr(0.3), arr(2.5))[0] == pytest.approx(
        0.3 / (1.5 * 0.7)
    )
    assert full_kelly(arr(0.4), arr(0.0), arr(2.0))[0] == 0.0  # no edge, no stake
    assert full_kelly(arr(0.5), arr(0.0), arr(1.0))[0] == 0.0  # nothing to win
    assert full_kelly(arr(np.nan), arr(0.0), arr(2.0))[0] == 0.0


def test_full_kelly_maximises_long_run_growth() -> None:
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        push = float(rng.choice([0.0, rng.uniform(0.0, 0.3)]))
        win = float(rng.uniform(0.05, 0.9 - push))
        odds = float(rng.uniform(1.2, 9.0))
        fraction = float(full_kelly(arr(win), arr(push), arr(odds))[0])
        if fraction == 0.0:
            continue
        lose = 1.0 - win - push

        def negative_growth(
            f: float, win: float = win, lose: float = lose, odds: float = odds
        ) -> float:
            return -(win * np.log1p(f * (odds - 1.0)) + lose * np.log1p(-f))

        best = minimize_scalar(
            negative_growth,
            bounds=(0.0, 0.999),
            method="bounded",
            options={"xatol": 1e-10},
        ).x
        assert fraction == pytest.approx(best, abs=1e-6)
        checked += 1
    assert checked > 100


def test_blend_probability_endpoints_and_midpoint() -> None:
    model, reference = arr(0.6, 0.2), arr(0.4, 0.3)
    np.testing.assert_allclose(blend_probability(model, reference, 0.0), reference)
    np.testing.assert_allclose(blend_probability(model, reference, 1.0), model)
    np.testing.assert_allclose(blend_probability(model, reference, 0.5), arr(0.5, 0.25))


# --------------------------------------------------------------------------- #
# How a market is shaped
# --------------------------------------------------------------------------- #
def test_market_shapes_agree_with_the_market_catalog() -> None:
    shapes = _market_shapes()
    counts = market_set().catalog.groupby("market_id", sort=False).size()
    assert (shapes["n_selections"] == counts.loc[shapes.index]).all()
    assert shapes["cover"].notna().all()
    double = shapes.index.to_series().str.startswith("DC")
    assert (
        shapes.loc[double, "cover"] == 2.0
    ).all()  # each square is in two selections
    assert (
        shapes.loc[~double, "cover"] == 1.0
    ).all()  # one winner (or a refund) per square
    assert set(shapes.index) == set(counts.index)


# --------------------------------------------------------------------------- #
# Reference probabilities
# --------------------------------------------------------------------------- #
def test_a_complete_market_is_de_margined() -> None:
    table = offers(
        ("m1", "1X2", "home", 2.0, 0.5),
        ("m1", "1X2", "draw", 4.0, 0.25),
        ("m1", "1X2", "away", 4.0, 0.25),
    )
    reference = reference_probabilities(table)
    assert reference.sum() == pytest.approx(1.0)
    assert reference.iloc[0] == pytest.approx(0.5)
    margin_added = offers(
        ("m1", "1X2", "home", 1.8, 0.5),
        ("m1", "1X2", "draw", 3.6, 0.25),
        ("m1", "1X2", "away", 3.6, 0.25),
    )
    shrunk = reference_probabilities(margin_added)
    assert shrunk.sum() == pytest.approx(1.0)
    assert shrunk.iloc[0] == pytest.approx(0.5)  # the margin is taken out evenly


def test_double_chance_selections_add_up_to_two() -> None:
    table = offers(
        ("m1", "DC", "home_draw", 1.30, 0.7),
        ("m1", "DC", "home_away", 1.35, 0.7),
        ("m1", "DC", "draw_away", 1.90, 0.5),
    )
    reference = reference_probabilities(table)
    assert reference.sum() == pytest.approx(2.0)
    implied = 1.0 / table["odds"]
    np.testing.assert_allclose(reference, 2.0 * implied / implied.sum())


def test_incomplete_or_unknown_markets_have_no_reference() -> None:
    partial = offers(("m1", "1X2", "home", 2.0, 0.5), ("m1", "1X2", "draw", 4.0, 0.25))
    assert reference_probabilities(partial).isna().all()
    unknown = offers(("m1", "NOPE", "a", 2.0, 0.5), ("m1", "NOPE", "b", 2.0, 0.5))
    assert reference_probabilities(unknown).isna().all()


@pytest.mark.parametrize("bad_price", [1.0, 0.8, float("nan"), float("inf")])
def test_one_unusable_price_makes_the_market_incomplete(bad_price: float) -> None:
    table = offers(
        ("m1", "1X2", "home", 2.0, 0.5),
        ("m1", "1X2", "draw", bad_price, 0.25),
        ("m1", "1X2", "away", 4.0, 0.25),
    )
    assert reference_probabilities(table).isna().all()


def test_refund_markets_are_conditional_on_no_refund() -> None:
    table = offers(
        ("m1", "DNB", "home", 1.6, 0.45, 0.25),
        ("m1", "DNB", "away", 2.6, 0.30, 0.25),
    )
    reference = reference_probabilities(table)
    conditional_home = (1 / 1.6) / (1 / 1.6 + 1 / 2.6)
    assert reference.iloc[0] == pytest.approx(conditional_home * 0.75)


def test_each_bookmaker_is_de_margined_on_its_own() -> None:
    table = offers(
        ("m1", "OU:2.5", "over", 1.9, 0.5),
        ("m1", "OU:2.5", "under", 1.9, 0.5),
        ("m1", "OU:2.5", "over", 2.2, 0.5),
    ).assign(bookmaker=["a", "a", "b"])
    reference = reference_probabilities(table)
    assert reference.iloc[:2].sum() == pytest.approx(1.0)
    assert np.isnan(reference.iloc[2])  # bookmaker b quotes only one side


def test_a_selection_priced_twice_is_rejected() -> None:
    table = offers(
        ("m1", "OU:2.5", "over", 1.9, 0.5), ("m1", "OU:2.5", "over", 2.0, 0.5)
    )
    with pytest.raises(ValueError, match="more than one price"):
        reference_probabilities(table)
    with pytest.raises(ValueError, match="more than one price"):
        find_value_bets(table)


def test_a_book_with_a_uniform_margin_gives_back_the_models_own_probabilities() -> None:
    """Across all 55 markets, including refunds and double chance."""
    rates = pd.DataFrame(
        {"lam_home": [1.7, 0.9, 2.4], "lam_away": [1.0, 1.3, 0.7]},
        index=["a", "b", "c"],
    )
    table = fixture_markets(rates, rho=-0.05)
    table["odds"] = table["fair_odds"] / 1.07
    reference = reference_probabilities(table)
    # A near-certain outcome priced at 1.0 or less cannot be bet, so its market is skipped.
    quotable = (
        (table["odds"] > 1.0)
        .groupby([table["match_id"], table["market_id"]])
        .transform("all")
    )
    assert reference.notna().equals(quotable)
    assert quotable.mean() > 0.9  # almost every market is complete
    np.testing.assert_allclose(
        reference[quotable], table.loc[quotable, "probability"], atol=1e-9
    )


def test_a_reference_bookmaker_supplies_the_view_for_every_book() -> None:
    sharp = offers(
        ("m1", "1X2", "home", 2.0, 0.5),
        ("m1", "1X2", "draw", 4.0, 0.25),
        ("m1", "1X2", "away", 4.0, 0.25),
    ).assign(bookmaker="sharp")
    soft = offers(("m1", "1X2", "home", 2.4, 0.5)).assign(bookmaker="soft")
    table = pd.concat([sharp, soft], ignore_index=True)
    view = reference_from_bookmaker(table, "sharp")
    np.testing.assert_allclose(
        view, [0.5, 0.25, 0.25, 0.5]
    )  # the soft price is judged by sharp's view
    assert reference_from_bookmaker(table, "nobody").isna().all()
    with pytest.raises(ValueError, match="bookmaker"):
        reference_from_bookmaker(table.drop(columns="bookmaker"), "sharp")


# --------------------------------------------------------------------------- #
# Choosing and sizing bets
# --------------------------------------------------------------------------- #
def one_bet(**overrides: object) -> pd.DataFrame:
    table = offers(("m1", "1X2", "home", 2.2, 0.55)).assign(p_reference=0.45)
    return find_value_bets(table, ValueConfig(**overrides))  # type: ignore[arg-type]


def test_a_clear_value_bet_is_sized_by_fractional_kelly() -> None:
    row = one_bet(model_weight=0.5).iloc[0]
    assert row["status"] == "bet"
    assert row["reference_source"] == "supplied"
    assert row["p_final"] == pytest.approx(0.5)  # halfway between 0.45 and 0.55
    assert row["raw_edge"] == pytest.approx(0.55 * 2.2 - 1)
    assert row["edge"] == pytest.approx(0.5 * 2.2 - 1)
    full = (0.5 * 2.2 - 1) / 1.2
    assert row["kelly_full"] == pytest.approx(full)
    assert row["stake_fraction"] == pytest.approx(
        min(0.25 * full, 0.02)
    )  # the cap binds
    assert row["stake"] == pytest.approx(row["stake_fraction"] * 1000)


def test_uncapped_stakes_follow_the_kelly_fraction() -> None:
    row = one_bet(
        model_weight=0.5, max_stake_fraction=0.5, kelly_fraction=0.1, bankroll=500.0
    ).iloc[0]
    assert row["stake_fraction"] == pytest.approx(0.1 * row["kelly_full"])
    assert row["stake"] == pytest.approx(row["stake_fraction"] * 500.0)


@pytest.mark.parametrize("refund", [0.0, 0.2])
def test_shrinkage_keeps_the_margin_penalty_of_the_assumed_reference(
    refund: float,
) -> None:
    """With no market to learn from, a price at the model's view still carries its margin."""
    margin = 0.06
    table = offers(("m1", "AH:-1", "home", 2.3, 0.55 - refund, refund))
    for weight in (0.2, 0.5, 0.8):
        config = ValueConfig(model_weight=weight, min_edge=0.0, fallback_margin=margin)
        row = find_value_bets(table, config).iloc[0]
        assert row["reference_source"] == "assumed margin"
        priced_in = (
            -(1.0 - refund) * margin / (1.0 + margin)
        )  # edge of the reference itself
        assert row["edge"] == pytest.approx(
            (1 - weight) * priced_in + weight * row["raw_edge"]
        )


def test_with_no_margin_assumed_the_edge_scales_with_the_weight() -> None:
    table = offers(("m1", "OU:2.5", "over", 2.0, 0.60))
    for weight in (0.2, 0.5, 0.8):
        config = ValueConfig(model_weight=weight, min_edge=0.0, fallback_margin=0.0)
        row = find_value_bets(table, config).iloc[0]
        assert row["edge"] == pytest.approx(weight * row["raw_edge"])


def test_trusting_the_model_more_raises_the_edge_when_it_disagrees_upward() -> None:
    edges = [
        one_bet(model_weight=w).iloc[0]["edge"] for w in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert edges == sorted(edges)
    assert edges[0] == pytest.approx(0.45 * 2.2 - 1)  # full trust in the reference


@pytest.mark.parametrize(
    ("odds", "probability", "status"),
    [
        (1.15, 0.9, "odds_out_of_range"),
        (12.0, 0.2, "odds_out_of_range"),
        (3.0, 0.04, "low_probability"),
        (2.0, 0.49, "below_min_edge"),
        (2.0, 0.50, "below_min_edge"),  # a fair price: no edge at all
        (0.9, 0.9, "invalid_odds"),
        (float("nan"), 0.5, "invalid_odds"),
        (2.0, float("nan"), "invalid_probability"),
        (2.0, 1.2, "invalid_probability"),
        (2.0, -0.1, "invalid_probability"),
    ],
)
def test_rejection_reasons(odds: float, probability: float, status: str) -> None:
    table = offers(("m1", "OU:2.5", "over", odds, probability)).assign(p_reference=0.5)
    result = find_value_bets(table, ValueConfig(model_weight=1.0))
    row = result.iloc[0]
    assert row["status"] == status
    assert row["stake"] == 0.0 and row["stake_fraction"] == 0.0
    if status.startswith("invalid"):
        assert np.isnan(row["edge"]) and np.isnan(row["kelly_full"])


def test_a_refund_that_leaves_no_room_for_the_probability_is_invalid() -> None:
    table = offers(("m1", "AH:-1", "home", 2.5, 0.8, 0.4)).assign(p_reference=0.4)
    assert find_value_bets(table).iloc[0]["status"] == "invalid_probability"


def test_zero_edge_is_never_a_bet_even_when_no_minimum_is_set() -> None:
    table = offers(("m1", "OU:2.5", "over", 2.0, 0.5)).assign(p_reference=0.5)
    row = find_value_bets(table, ValueConfig(min_edge=0.0)).iloc[0]
    assert row["edge"] == pytest.approx(0.0)
    assert row["status"] == "below_min_edge"


def test_only_the_best_selection_per_match_is_taken() -> None:
    table = offers(
        ("m1", "1X2", "home", 2.2, 0.55),
        ("m1", "OU:2.5", "over", 2.4, 0.55),
        ("m2", "1X2", "away", 3.0, 0.40),
    ).assign(p_reference=[0.45, 0.45, 0.30])
    config = ValueConfig(model_weight=1.0, min_edge=0.0)
    result = find_value_bets(table, config)
    status = result.set_index(["match_id", "selection"])["status"]
    assert result["status"].tolist().count("bet") == 2
    assert status.loc[("m1", "over")] == "bet"  # 2.4 beats 2.2
    assert status.loc[("m1", "home")] == "match_limit"
    both = find_value_bets(
        table, ValueConfig(model_weight=1.0, min_edge=0.0, max_bets_per_match=2)
    )
    assert both["status"].tolist().count("bet") == 3


def test_a_selection_is_bet_once_at_its_best_price() -> None:
    table = offers(
        ("m1", "1X2", "home", 2.1, 0.55), ("m1", "1X2", "home", 2.3, 0.55)
    ).assign(bookmaker=["a", "b"], p_reference=0.45)
    for limit in (
        1,
        2,
    ):  # allowing more bets per match must not double up the same selection
        config = ValueConfig(model_weight=1.0, max_bets_per_match=limit)
        result = find_value_bets(table, config)
        assert result.iloc[0]["bookmaker"] == "b" and result.iloc[0]["status"] == "bet"
        assert result.iloc[1]["status"] == "better_price"
        assert len(selected_bets(result)) == 1


def test_a_reference_bookmaker_turns_a_soft_price_into_value() -> None:
    sharp = offers(
        ("m1", "1X2", "home", 2.0, 0.5),
        ("m1", "1X2", "draw", 4.0, 0.25),
        ("m1", "1X2", "away", 4.0, 0.25),
    ).assign(bookmaker="sharp")
    soft = offers(
        ("m1", "1X2", "home", 2.4, 0.5),
        ("m1", "1X2", "draw", 3.5, 0.25),
        ("m1", "1X2", "away", 3.5, 0.25),
    ).assign(bookmaker="soft")
    table = pd.concat([sharp, soft], ignore_index=True)
    # The model agrees with the sharp book, so only the soft book's price is out of line.
    result = find_value_bets(
        table,
        ValueConfig(model_weight=0.0, min_edge=0.04),
        reference_bookmaker="sharp",
    )
    bets = selected_bets(result)
    assert len(bets) == 1
    assert bets.iloc[0]["bookmaker"] == "soft" and bets.iloc[0]["selection"] == "home"
    assert bets.iloc[0]["reference_source"] == "reference book"
    assert bets.iloc[0]["edge"] == pytest.approx(0.5 * 2.4 - 1.0)
    # Judged by its own prices instead, no bookmaker looks like value.
    own = find_value_bets(table, ValueConfig(model_weight=0.0, min_edge=0.04))
    assert selected_bets(own).empty


def test_total_exposure_is_capped_proportionally() -> None:
    rows = [(f"m{i}", "1X2", "home", 2.5, 0.6) for i in range(10)]
    table = offers(*rows).assign(p_reference=0.45)
    config = ValueConfig(
        model_weight=1.0, max_stake_fraction=0.05, max_total_fraction=0.10
    )
    result = find_value_bets(table, config)
    assert result["stake_fraction"].sum() == pytest.approx(0.10)
    assert result["stake_fraction"].nunique() == 1  # identical bets are scaled alike
    assert result["stake"].sum() == pytest.approx(100.0)


def test_tiny_stakes_are_dropped() -> None:
    result = one_bet(model_weight=0.5, min_stake=50.0)
    assert result.iloc[0]["status"] == "below_min_stake"
    assert result.iloc[0]["stake"] == 0.0 and result.iloc[0]["stake_fraction"] == 0.0
    assert selected_bets(result).empty
    assert (
        one_bet(model_weight=0.5, kelly_fraction=0.0).iloc[0]["status"]
        == "below_min_stake"
    )


def test_refunded_stakes_change_the_edge_and_the_stake() -> None:
    plain = offers(("m1", "AH:-1", "home", 2.5, 0.40, 0.0)).assign(p_reference=0.40)
    refund = offers(("m1", "AH:-1", "home", 2.5, 0.40, 0.30)).assign(p_reference=0.40)
    config = ValueConfig(model_weight=1.0, min_edge=0.0)
    assert find_value_bets(refund, config).iloc[0]["edge"] == pytest.approx(0.30)
    assert find_value_bets(plain, config).iloc[0]["edge"] == pytest.approx(0.0)
    assert find_value_bets(refund, config).iloc[0]["kelly_full"] == pytest.approx(
        0.3 / (1.5 * 0.7)
    )


def test_results_are_ordered_and_the_input_is_untouched() -> None:
    table = offers(
        ("m1", "1X2", "home", 2.2, 0.55),
        ("m2", "1X2", "home", 2.0, 0.40),
        ("m3", "1X2", "home", 3.0, 0.50),
        ("m4", "1X2", "home", 1.1, 0.95),
        ("m5", "1X2", "home", 0.9, 0.95),
    ).assign(p_reference=[0.45, 0.40, 0.30, 0.9, 0.9])
    snapshot = table.copy()
    result = find_value_bets(table, ValueConfig(model_weight=1.0))
    pd.testing.assert_frame_equal(table, snapshot)
    assert result["status"].tolist()[:2] == ["bet", "bet"]
    assert result["edge"].iloc[0] >= result["edge"].iloc[1]  # best edge first
    assert result["status"].tolist()[-2:] == ["odds_out_of_range", "invalid_odds"]
    ranks = result["status"].map(STATUS_ORDER.index)
    assert ranks.is_monotonic_increasing
    assert len(selected_bets(result)) == 2
    assert list(result.index) == list(range(len(result)))


def test_no_offers_gives_an_empty_result_instead_of_an_error() -> None:
    result = find_value_bets(offers())
    assert result.empty
    assert {"p_reference", "p_final", "edge", "stake", "status"} <= set(result.columns)
    assert selected_bets(result).empty


def test_invalid_inputs_and_configs_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        find_value_bets(pd.DataFrame({"match_id": ["m1"]}))
    bad_settings: list[dict[str, float]] = [
        {"model_weight": 1.5},
        {"kelly_fraction": -0.1},
        {"max_stake_fraction": 0.0},
        {"max_total_fraction": 1.5},
        {"min_odds": 1.0},
        {"max_odds": 1.2},
        {"max_bets_per_match": 0},
        {"bankroll": 0.0},
        {"min_stake": -1.0},
        {"min_edge": -0.01},
        {"min_probability": 1.0},
        {"fallback_margin": -0.1},
    ]
    for bad in bad_settings:
        with pytest.raises(ValueError):
            ValueConfig(**bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# From the score matrix to a bet
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def priced() -> pd.DataFrame:
    """Two fixtures, every market, at the model's fair odds less a 6% margin."""
    rates = pd.DataFrame(
        {"lam_home": [1.8, 1.0], "lam_away": [0.9, 1.4]}, index=["A", "B"]
    )
    table = fixture_markets(rates, rho=-0.05)
    table["odds"] = table["fair_odds"] / 1.06
    return table


def test_a_fairly_priced_book_offers_no_value(priced: pd.DataFrame) -> None:
    result = find_value_bets(priced)
    assert selected_bets(result).empty
    assert result["p_reference"].notna().all()
    assert (result["edge"].dropna() < 0).all()  # every price costs about its margin
    unusable = int(
        (priced["odds"] <= 1.0).sum()
    )  # near-certain outcomes priced below 1
    assert unusable > 0
    assert (result["status"] == "invalid_odds").sum() == unusable
    # A market is de-margined on its own prices only when every selection is quoted.
    keys = ["match_id", "market_id", "selection"]
    complete = (
        (priced["odds"] > 1.0)
        .groupby([priced["match_id"], priced["market_id"]])
        .transform("all")
    )
    expected = pd.Series(
        np.where(complete, "own prices", "assumed margin"),
        index=pd.MultiIndex.from_frame(priced[keys]),
    )
    actual = result.set_index(keys)["reference_source"]
    assert (actual.loc[expected.index] == expected).all()


def test_a_generous_price_is_found_among_all_markets(priced: pd.DataFrame) -> None:
    table = priced.copy()
    target = (
        (table["match_id"] == "A")
        & (table["market_id"] == "OU:2.5")
        & (table["selection"] == "over")
    )
    table.loc[target, "odds"] *= 1.35
    result = find_value_bets(table, ValueConfig(model_weight=1.0))
    bets = selected_bets(result)
    assert len(bets) == 1
    bet = bets.iloc[0]
    assert (bet["match_id"], bet["market_id"], bet["selection"]) == (
        "A",
        "OU:2.5",
        "over",
    )
    assert bet["edge"] == pytest.approx(1.35 / 1.06 - 1.0)
    assert bet["stake"] == pytest.approx(
        20.0
    )  # a quarter Kelly is far above the 2% cap here
