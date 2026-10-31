"""Tests for the LightGBM xG models and walk-forward prediction."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from src import model
from src.features import build_features, feature_columns
from src.model import (
    MIN_TRAIN_ROWS,
    ModelParams,
    feature_importance,
    fit_boosters,
    predict_xg,
    refit_windows,
    training_mask,
    walk_forward_predict,
)
from tests.test_features import make_matches

FAST = ModelParams(n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=1)
PREDICTIONS = ["pred_home_xg", "pred_away_xg"]


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(make_matches(n_teams=16, seasons=7, seed=3))


@pytest.fixture(scope="module")
def oof(features: pd.DataFrame) -> pd.DataFrame:
    return walk_forward_predict(features, FAST, first_eval_season=2022, refit_days=90)


# --------------------------------------------------------------------------- #
# Parameters and training rows
# --------------------------------------------------------------------------- #
def test_lightgbm_parameters_use_the_poisson_objective_and_are_reproducible() -> None:
    config = ModelParams(
        learning_rate=0.05, num_leaves=6, min_child_samples=40
    ).lightgbm()
    assert config["objective"] == "poisson"
    assert config["learning_rate"] == 0.05
    assert config["num_leaves"] == 6
    assert config["min_data_in_leaf"] == 40
    assert config["deterministic"] is True
    assert "sample_half_life_days" not in config  # ours, not LightGBM's


def test_training_mask_drops_warm_up_seasons_and_missing_targets(
    features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    mask = training_mask(features)
    assert mask.sum() < len(features)  # about one match in ten has no xG
    assert features.loc[mask, "target_home_xg"].notna().all()
    monkeypatch.setattr(model, "FIRST_TRAIN_SEASON", 2021)
    stricter = training_mask(features)
    assert stricter.sum() < mask.sum()
    assert features.loc[stricter, "season"].min() == 2021


def test_fit_rejects_rows_from_the_future_and_too_few_rows(
    features: pd.DataFrame,
) -> None:
    usable = training_mask(features)
    as_of = features["date"].sort_values().iloc[len(features) // 2]
    with pytest.raises(ValueError, match="before as_of"):
        fit_boosters(features, usable, FAST, as_of=as_of)  # includes later matches
    tiny = usable & (features.index < MIN_TRAIN_ROWS // 2)
    with pytest.raises(ValueError, match="at least"):
        fit_boosters(
            features,
            tiny,
            FAST,
            as_of=features["date"].max() + pd.Timedelta(1, unit="D"),
        )


def test_predictions_are_positive_and_aligned(features: pd.DataFrame) -> None:
    as_of = features["date"].sort_values().iloc[int(len(features) * 0.6)]
    train = training_mask(features) & (features["date"] < as_of)
    boosters = fit_boosters(features, train, FAST, as_of=as_of)
    predicted = predict_xg(boosters, features)
    assert predicted.index.equals(features.index)
    assert list(predicted.columns) == PREDICTIONS
    assert (predicted > 0).all().all()
    assert np.isfinite(predicted.to_numpy()).all()


def test_recency_weights_change_the_fit(features: pd.DataFrame) -> None:
    as_of = features["date"].sort_values().iloc[int(len(features) * 0.6)]
    train = training_mask(features) & (features["date"] < as_of)
    flat = predict_xg(
        fit_boosters(
            features,
            train,
            ModelParams(**{**FAST.__dict__, "sample_half_life_days": 1e9}),
            as_of=as_of,
        ),
        features,
    )
    recent = predict_xg(
        fit_boosters(
            features,
            train,
            ModelParams(**{**FAST.__dict__, "sample_half_life_days": 60.0}),
            as_of=as_of,
        ),
        features,
    )
    assert not np.allclose(flat["pred_home_xg"], recent["pred_home_xg"])


# --------------------------------------------------------------------------- #
# Refit windows
# --------------------------------------------------------------------------- #
def test_windows_are_contiguous_and_cover_the_evaluation_period(
    features: pd.DataFrame,
) -> None:
    windows = refit_windows(features, first_eval_season=2022, step_days=60)
    assert windows[0].start == features.loc[features["season"] >= 2022, "date"].min()
    assert windows[-1].end > features["date"].max()
    for earlier, later in pairwise(windows):
        assert earlier.end == later.start
        assert earlier.start < earlier.end
    assert all((w.end - w.start).days <= 60 for w in windows[:-1])


@pytest.mark.parametrize(("season", "step"), [(2022, 0), (2022, -5), (2099, 30)])
def test_invalid_window_arguments_are_rejected(
    features: pd.DataFrame, season: int, step: int
) -> None:
    with pytest.raises(ValueError):
        refit_windows(features, season, step)


# --------------------------------------------------------------------------- #
# Walk-forward predictions
# --------------------------------------------------------------------------- #
def test_every_evaluation_match_is_predicted_exactly_once(
    features: pd.DataFrame, oof: pd.DataFrame
) -> None:
    expected = features.loc[features["season"] >= 2022, "match_id"]
    assert sorted(oof["match_id"]) == sorted(expected)
    assert oof["match_id"].is_unique
    assert oof["season"].min() == 2022
    assert (oof["window_start"] <= oof["date"]).all()
    assert (oof[PREDICTIONS] > 0).all().all()


def test_each_refit_trains_only_on_matches_before_its_window(
    features: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = model.fit_boosters
    checked = []

    def spy(
        data: pd.DataFrame, mask: pd.Series, params: ModelParams, *, as_of: pd.Timestamp
    ) -> model.Boosters:
        assert data.loc[mask, "date"].max() < as_of
        checked.append(as_of)
        return original(data, mask, params, as_of=as_of)

    monkeypatch.setattr(model, "fit_boosters", spy)
    walk_forward_predict(features, FAST, first_eval_season=2022, refit_days=120)
    assert len(checked) >= 5


def test_predictions_never_depend_on_outcomes_from_the_future(
    features: pd.DataFrame, oof: pd.DataFrame
) -> None:
    cutoff = features["date"].sort_values().iloc[int(len(features) * 0.8)]
    tampered = features.copy()
    future = tampered["date"] >= cutoff
    for column in (
        "target_home_xg",
        "target_away_xg",
        "target_home_goals",
        "target_away_goals",
    ):
        tampered.loc[future, column] = 9.0
    changed = walk_forward_predict(
        tampered, FAST, first_eval_season=2022, refit_days=90
    )
    early = (oof["window_start"] <= cutoff).to_numpy()
    np.testing.assert_allclose(
        oof.loc[early, PREDICTIONS].to_numpy(),
        changed.loc[early, PREDICTIONS].to_numpy(),
    )
    assert not np.allclose(
        oof.loc[~early, PREDICTIONS].to_numpy(),
        changed.loc[~early, PREDICTIONS].to_numpy(),
    )


def test_training_is_reproducible(features: pd.DataFrame, oof: pd.DataFrame) -> None:
    again = walk_forward_predict(features, FAST, first_eval_season=2022, refit_days=90)
    pd.testing.assert_frame_equal(oof, again)


def test_the_model_learns_something_beyond_the_league_average(
    features: pd.DataFrame, oof: pd.DataFrame
) -> None:
    data = oof.merge(
        features[["match_id", "target_home_xg", "league_avg_xg"]], on="match_id"
    ).dropna()
    model_error = (data["pred_home_xg"] - data["target_home_xg"]).abs().mean()
    naive_error = (data["league_avg_xg"] - data["target_home_xg"]).abs().mean()
    assert model_error < naive_error


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def test_feature_importance_is_a_sorted_percentage_table(
    features: pd.DataFrame,
) -> None:
    as_of = features["date"].max() + pd.Timedelta(1, unit="D")
    table = feature_importance(
        fit_boosters(features, training_mask(features), FAST, as_of=as_of)
    )
    assert set(table["feature"]) == set(feature_columns())
    assert table["home_gain"].sum() == pytest.approx(100.0)
    assert table["away_gain"].sum() == pytest.approx(100.0)
    assert table["mean_gain"].is_monotonic_decreasing
