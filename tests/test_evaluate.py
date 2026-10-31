"""Tests for evaluation metrics, calibration, baselines and the report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.dixon_coles import score_matrix
from src.evaluate import (
    OUTCOMES,
    accuracy_1x2,
    baseline_predictions,
    error_metrics,
    evaluation_report,
    fit_calibration,
    fit_rho,
    format_report,
    log_loss_1x2,
    market_probabilities,
    outcome_table,
    poisson_nll,
)
from src.features import build_features
from src.model import ModelParams, walk_forward_predict
from tests.test_features import make_matches

FAST = ModelParams(n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=1)


def simulate_scores(
    lam: float, mu: float, rho: float, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Draw scores from a Dixon-Coles distribution with known parameters."""
    rng = np.random.default_rng(seed)
    matrix = score_matrix(lam, mu, rho)
    cells = rng.choice(matrix.size, size=n, p=matrix.ravel())
    home, away = np.divmod(cells, matrix.shape[1])
    return home.astype("float64"), away.astype("float64")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_poisson_nll_matches_scipy() -> None:
    rate = np.array([0.8, 1.5, 2.2, 1.1])
    goals = np.array([0.0, 2.0, 3.0, 1.0])
    expected = -poisson.logpmf(goals, rate).mean()
    assert poisson_nll(rate, goals) == pytest.approx(expected)


def test_poisson_nll_is_lowest_at_the_true_rate() -> None:
    goals = np.random.default_rng(1).poisson(1.4, 5000).astype("float64")
    losses = {rate: poisson_nll(np.full(5000, rate), goals) for rate in (0.8, 1.4, 2.4)}
    assert min(losses, key=losses.__getitem__) == 1.4


def test_error_metrics_ignore_unknown_actuals() -> None:
    mae, rmse = error_metrics(np.array([1.0, 2.0, 9.0]), np.array([2.0, 4.0, np.nan]))
    assert mae == pytest.approx(1.5)
    assert rmse == pytest.approx(np.sqrt((1 + 4) / 2))


def test_log_loss_and_accuracy() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    outcome = np.array([0, 2])
    assert log_loss_1x2(probabilities, outcome) == pytest.approx(
        -(np.log(0.7) + np.log(0.6)) / 2
    )
    assert accuracy_1x2(probabilities, outcome) == 1.0
    assert accuracy_1x2(probabilities, np.array([1, 2])) == 0.5


# --------------------------------------------------------------------------- #
# rho and calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("true_rho", [-0.12, 0.0])
def test_fit_rho_recovers_the_dependence(true_rho: float) -> None:
    home, away = simulate_scores(1.5, 1.1, true_rho, n=40_000, seed=3)
    estimate = fit_rho(home, away, np.full(len(home), 1.5), np.full(len(home), 1.1))
    assert estimate == pytest.approx(true_rho, abs=0.03)


def test_calibration_scales_rates_to_the_goals_scored() -> None:
    pred_home, pred_away = np.array([1.0, 2.0, 1.0]), np.array([0.5, 1.0, 0.5])
    goals_home, goals_away = np.array([1.0, 3.0, 2.0]), np.array([0.0, 1.0, 1.0])
    calibration = fit_calibration(pred_home, pred_away, goals_home, goals_away)
    assert calibration.scale_home == pytest.approx(6.0 / 4.0)
    assert calibration.scale_away == pytest.approx(2.0 / 2.0)
    assert -1.0 < calibration.rho < 1.0
    with pytest.raises(ValueError, match="No development rows"):
        fit_calibration(np.array([]), np.array([]), np.array([]), np.array([]))


# --------------------------------------------------------------------------- #
# Outcome probabilities
# --------------------------------------------------------------------------- #
def test_outcome_table_probabilities_and_coding() -> None:
    lam, mu = np.array([1.8, 0.9, 1.2]), np.array([0.9, 1.4, 1.2])
    goals_home, goals_away = np.array([2.0, 0.0, 1.0]), np.array([0.0, 3.0, 1.0])
    table = outcome_table(lam, mu, goals_home, goals_away, rho=-0.1)
    assert np.allclose(table[list(OUTCOMES)].sum(axis=1), 1.0)
    assert list(table["outcome"]) == [0, 2, 1]
    assert (table["scoreline_nll"] > 0).all()
    assert table.loc[0, "p_home"] > table.loc[0, "p_away"]


def test_scores_outside_the_grid_get_a_finite_penalty() -> None:
    table = outcome_table(
        np.array([1.5]), np.array([1.5]), np.array([12.0]), np.array([0.0]), rho=0.0
    )
    assert np.isfinite(table.loc[0, "scoreline_nll"])
    assert table.loc[0, "outcome"] == 0


def test_extreme_rates_are_clipped_rather_than_crashing() -> None:
    table = outcome_table(
        np.array([0.0, 40.0]),
        np.array([-1.0, 40.0]),
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        rho=-0.2,
    )
    assert np.isfinite(table.to_numpy(dtype="float64")).all()


# --------------------------------------------------------------------------- #
# Baselines and market
# --------------------------------------------------------------------------- #
def test_baselines_have_the_expected_shape_and_fallbacks() -> None:
    features = pd.DataFrame(
        {
            "league_avg_goals_home": [1.6, 1.5, np.nan],
            "league_avg_goals_away": [1.2, 1.1, np.nan],
            "league_avg_xg": [1.4, np.nan, np.nan],
            "home_xg_exp": [1.9, np.nan, np.nan],
            "away_xg_exp": [0.8, np.nan, np.nan],
        }
    )
    baselines = baseline_predictions(features)
    assert set(baselines) == {"league_avg", "strength_only"}
    league, strength = baselines["league_avg"], baselines["strength_only"]
    assert (
        league.loc[0, "pred_home_xg"] > league.loc[0, "pred_away_xg"]
    )  # home advantage
    assert league.loc[1, "pred_home_xg"] == 1.5  # no league xG yet: use the goals level
    assert strength.loc[0, "pred_home_xg"] == 1.9
    assert (
        strength.loc[1, "pred_home_xg"] == league.loc[1, "pred_home_xg"]
    )  # falls back


def test_market_probabilities_remove_the_margin_and_prefer_sharp_books() -> None:
    enriched = pd.DataFrame(
        {
            "match_id": ["a", "b", "c"],
            "psc_h": [2.0, np.nan, np.nan],
            "psc_d": [4.0, np.nan, np.nan],
            "psc_a": [4.0, np.nan, np.nan],
            "b365c_h": [1.5, 2.0, np.nan],
            "b365c_d": [4.0, 3.0, np.nan],
            "b365c_a": [8.0, 4.0, np.nan],
        }
    )
    market = market_probabilities(enriched)
    assert np.allclose(market.loc[["a", "b"]].sum(axis=1), 1.0)
    assert market.loc["a", "p_home"] == pytest.approx(
        0.5 / 1.0
    )  # Pinnacle: 0.5 + 0.25 + 0.25 = 1.0
    assert market.loc["b", "p_home"] == pytest.approx(
        (1 / 2.0) / (1 / 2.0 + 1 / 3.0 + 1 / 4.0)
    )  # fell back
    assert market.loc["c"].isna().all()


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def setup() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(make_matches(n_teams=16, seasons=7, seed=3))
    oof = walk_forward_predict(features, FAST, first_eval_season=2022, refit_days=90)
    odds = pd.DataFrame(
        {"match_id": features["match_id"], "psc_h": 2.4, "psc_d": 3.4, "psc_a": 3.3}
    )
    return features, oof, odds


def test_report_covers_every_model_and_slice(
    setup: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof, odds = setup
    report = evaluation_report(oof, features, odds, test_from_season=2025)
    assert set(report["model"]) == {
        "lightgbm",
        "league_avg",
        "strength_only",
        "market_closing",
    }
    assert set(report["slice"]) == {"dev", "test"}
    model_rows = report[report["model"] != "market_closing"]
    assert np.isfinite(
        model_rows[
            [
                "mae_xg",
                "rmse_xg",
                "goals_nll",
                "scoreline_nll",
                "logloss_1x2",
                "acc_1x2",
            ]
        ].to_numpy(dtype="float64")
    ).all()
    assert (model_rows["rmse_xg"] >= model_rows["mae_xg"]).all()
    assert (
        report.loc[report["slice"] == "test", "n"].nunique() == 1
    )  # identical matches for every model
    assert (
        "mae_xg"
        not in report.loc[report["model"] == "market_closing"]
        .dropna(axis=1, how="all")
        .columns
    )


def test_the_model_beats_the_naive_baseline_on_data_with_real_structure(
    setup: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof, _ = setup
    report = evaluation_report(oof, features, None, test_from_season=2025).set_index(
        ["model", "slice"]
    )
    assert (
        report.loc[("lightgbm", "test"), "goals_nll"]
        < report.loc[("league_avg", "test"), "goals_nll"]
    )
    assert (
        report.loc[("lightgbm", "test"), "mae_xg"]
        < report.loc[("league_avg", "test"), "mae_xg"]
    )
    assert "market_closing" not in report.index.get_level_values("model")


def test_calibration_uses_only_the_development_slice(
    setup: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof, _ = setup
    report = evaluation_report(oof, features, None, test_from_season=2025)
    rows = report[report["model"] == "lightgbm"]
    assert rows["rho"].nunique() == 1  # one rho, fitted on dev, applied to both slices


def test_an_empty_development_slice_is_rejected(
    setup: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof, _ = setup
    with pytest.raises(ValueError, match="development slice is empty"):
        evaluation_report(oof, features, None, test_from_season=2022)


def test_format_report_is_readable(
    setup: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof, odds = setup
    text = format_report(evaluation_report(oof, features, odds, test_from_season=2025))
    assert "lightgbm" in text and "market_closing" in text
    assert "rho" not in text
