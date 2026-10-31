"""Tests for hyper-parameter tuning on development seasons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.model import MIN_TRAIN_ROWS, ModelParams
from src.tuning import (
    BAGGING_FRACTION,
    FEATURE_FRACTION,
    LEARNING_RATE,
    NUM_LEAVES,
    load_params,
    make_folds,
    poisson_deviance,
    save_params,
    tune_params,
)
from tests.test_features import make_matches


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(make_matches(n_teams=16, seasons=7, seed=3))


def tune(data: pd.DataFrame, **overrides: object) -> object:
    settings = {
        "fold_seasons": [2022, 2023],
        "n_trials": 4,
        "threads": 1,
        "max_estimators": 30,
    }
    return tune_params(data, **{**settings, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def test_deviance_is_zero_for_perfect_predictions_and_grows_with_error() -> None:
    actual = np.array([0.0, 0.5, 1.2, 2.4])
    # predictions are floored at 1e-6, so a perfect forecast of an exact zero costs a hair
    assert poisson_deviance(actual, actual) == pytest.approx(0.0, abs=1e-5)
    assert poisson_deviance(actual + 0.5, actual) > 0.0
    assert poisson_deviance(actual * 3, actual) > poisson_deviance(actual * 1.2, actual)


def test_deviance_matches_the_formula_including_zero_actuals() -> None:
    predicted, actual = np.array([1.0, 2.0]), np.array([0.0, 3.0])
    expected = np.mean([2 * (0 - (0 - 1.0)), 2 * (3 * np.log(3 / 2) - (3 - 2.0))])
    assert poisson_deviance(predicted, actual) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #
def test_folds_train_strictly_before_the_season_they_validate(
    features: pd.DataFrame,
) -> None:
    folds = make_folds(features, [2022, 2023])
    assert [fold.season for fold in folds] == [2022, 2023]
    for fold in folds:
        start = features.loc[features["season"] == fold.season, "date"].min()
        assert (
            fold.train_age_days >= 1
        ).all()  # every training match is before the season
        assert len(fold.train_inputs) >= MIN_TRAIN_ROWS
        assert len(fold.valid_inputs) == len(fold.valid_targets["home"]) > 0
        assert start > features.loc[features["season"] < fold.season, "date"].max()
    assert len(folds[1].train_inputs) > len(folds[0].train_inputs)  # expanding window


def test_unusable_fold_seasons_are_rejected(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No matches in season"):
        make_folds(features, [2099])
    with pytest.raises(ValueError, match="at least"):
        make_folds(features, [2020])  # nothing earlier to train on


# --------------------------------------------------------------------------- #
# The search
# --------------------------------------------------------------------------- #
def test_the_defaults_are_the_first_trial_so_tuning_cannot_end_worse(
    features: pd.DataFrame,
) -> None:
    result = tune(features)
    assert result.n_trials == 4  # type: ignore[attr-defined]
    assert result.best_value <= result.default_value  # type: ignore[attr-defined]
    assert np.isfinite(result.default_value)  # type: ignore[attr-defined]
    assert result.fold_seasons == (2022, 2023)  # type: ignore[attr-defined]


def test_best_settings_stay_inside_the_search_ranges(features: pd.DataFrame) -> None:
    params = tune(features).best_params  # type: ignore[attr-defined]
    assert isinstance(params, ModelParams)
    assert LEARNING_RATE[0] <= params.learning_rate <= LEARNING_RATE[1]
    assert NUM_LEAVES[0] <= params.num_leaves <= NUM_LEAVES[1]
    assert FEATURE_FRACTION[0] <= params.feature_fraction <= FEATURE_FRACTION[1]
    assert BAGGING_FRACTION[0] <= params.bagging_fraction <= BAGGING_FRACTION[1]
    assert params.n_estimators <= 30
    assert params.seed == 42 and params.num_threads == 1


def test_the_search_is_reproducible_for_a_seed(features: pd.DataFrame) -> None:
    first, second = tune(features, seed=5), tune(features, seed=5)
    assert first.best_params == second.best_params  # type: ignore[attr-defined]
    assert first.best_value == second.best_value  # type: ignore[attr-defined]


def test_test_seasons_never_influence_tuning(features: pd.DataFrame) -> None:
    baseline = tune(features)
    tampered = features.copy()
    held_out = tampered["season"] >= 2024  # the folds use 2022 and 2023 only
    for column in (
        "target_home_xg",
        "target_away_xg",
        "target_home_goals",
        "target_away_goals",
    ):
        tampered.loc[held_out, column] = 9.0
    changed = tune(tampered)
    assert changed.best_params == baseline.best_params  # type: ignore[attr-defined]
    assert changed.best_value == baseline.best_value  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Saved settings
# --------------------------------------------------------------------------- #
def test_settings_round_trip_through_json(tmp_path: Path) -> None:
    params = ModelParams(
        n_estimators=123, learning_rate=0.05, num_leaves=9, sample_half_life_days=400.0
    )
    path = tmp_path / "models" / "best_params.json"
    save_params(path, params)
    assert load_params(path) == params
    assert '"n_estimators": 123' in path.read_text(encoding="utf-8")


def test_unknown_saved_settings_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "params.json"
    path.write_text('{"n_estimators": 10, "turbo_mode": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="turbo_mode"):
        load_params(path)
