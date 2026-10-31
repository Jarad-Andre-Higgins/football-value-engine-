"""Tests for the saved model bundle."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.bundle import (
    SCHEMA_VERSION,
    ModelBundle,
    build_bundle,
    current_versions,
    load_bundle,
    predict_rates,
    save_bundle,
    summary,
)
from src.evaluate import MAX_RATE, MIN_RATE, Calibration
from src.features import build_features, feature_columns
from src.model import ModelParams, fit_boosters, training_mask
from tests.test_features import make_matches

FAST = ModelParams(n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=1)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(make_matches(n_teams=16, seasons=7, seed=3))


@pytest.fixture(scope="module")
def bundle(features: pd.DataFrame) -> ModelBundle:
    mask = training_mask(features)
    last = features.loc[mask, "date"].max()
    boosters = fit_boosters(
        features, mask, FAST, as_of=last + pd.Timedelta(1, unit="D")
    )
    return build_bundle(
        boosters,
        Calibration(scale_home=1.05, scale_away=0.95, rho=-0.08),
        FAST,
        trained_through=last,
        n_train_rows=int(mask.sum()),
        metrics={"logloss_1x2": 1.0, "n": 100.0},
    )


def test_bundle_records_what_it_is(bundle: ModelBundle) -> None:
    assert bundle.feature_names == tuple(feature_columns())
    assert bundle.schema_version == SCHEMA_VERSION
    assert bundle.versions == current_versions()
    assert bundle.created_at.endswith("+00:00")
    assert bundle.n_train_rows > 300


def test_saving_writes_a_pickle_and_a_readable_summary(
    bundle: ModelBundle, tmp_path: Path
) -> None:
    path = save_bundle(bundle, tmp_path / "models" / "xg_model.pkl")
    assert path.exists()
    described = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert described["n_features"] == len(feature_columns())
    assert described["calibration"] == {
        "scale_home": 1.05,
        "scale_away": 0.95,
        "rho": -0.08,
    }
    assert described["trained_through"] == bundle.trained_through.strftime("%Y-%m-%d")
    assert described["params"]["n_estimators"] == 25
    assert summary(bundle)["metrics"] == {"logloss_1x2": 1.0, "n": 100.0}


def test_a_loaded_bundle_predicts_exactly_like_the_original(
    bundle: ModelBundle, features: pd.DataFrame, tmp_path: Path
) -> None:
    loaded = load_bundle(save_bundle(bundle, tmp_path / "xg_model.pkl"))
    assert loaded.calibration == bundle.calibration
    assert loaded.params == bundle.params
    assert loaded.trained_through == bundle.trained_through
    pd.testing.assert_frame_equal(
        predict_rates(loaded, features), predict_rates(bundle, features)
    )


def test_rates_are_calibrated_positive_and_clipped(
    bundle: ModelBundle, features: pd.DataFrame
) -> None:
    rates = predict_rates(bundle, features)
    assert rates.index.equals(features.index)
    assert list(rates.columns) == ["lam_home", "lam_away"]
    assert rates.to_numpy().min() >= MIN_RATE and rates.to_numpy().max() <= MAX_RATE
    doubled = replace(
        bundle, calibration=Calibration(scale_home=2.1, scale_away=1.9, rho=-0.08)
    )
    inside = (rates["lam_home"] > 0.2) & (rates["lam_home"] < 2.5)
    np.testing.assert_allclose(
        predict_rates(doubled, features).loc[inside, "lam_home"],
        2.0 * rates.loc[inside, "lam_home"],
    )


def test_missing_feature_columns_are_reported(
    bundle: ModelBundle, features: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="missing 2 columns"):
        predict_rates(bundle, features.drop(columns=["home_elo", "away_elo"]))


def test_loading_rejects_foreign_files_and_old_layouts(
    bundle: ModelBundle, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.pkl")
    stranger = tmp_path / "stranger.pkl"
    stranger.write_bytes(pickle.dumps({"not": "a bundle"}))
    with pytest.raises(TypeError, match="not a model bundle"):
        load_bundle(stranger)
    old = save_bundle(
        replace(bundle, schema_version=SCHEMA_VERSION + 1), tmp_path / "old.pkl"
    )
    with pytest.raises(ValueError, match="rebuild the model"):
        load_bundle(old)


def test_a_library_version_mismatch_is_a_warning_not_an_error(
    bundle: ModelBundle, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    other = replace(bundle, versions={**bundle.versions, "lightgbm": "0.0.1"})
    path = save_bundle(other, tmp_path / "xg_model.pkl")
    with caplog.at_level(logging.WARNING, logger="src.bundle"):
        loaded = load_bundle(path)
    assert loaded.versions["lightgbm"] == "0.0.1"
    assert "built with lightgbm 0.0.1" in caplog.text
