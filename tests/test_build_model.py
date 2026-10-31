"""Tests for the command that tunes, evaluates, fits and saves the model."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from src import build_model, logging_config
from src.build_model import _combine, _test_metrics, main
from src.bundle import load_bundle, predict_rates
from src.features import build_features
from src.model import ModelParams
from src.tuning import load_params, save_params
from tests.test_features import make_matches


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(make_matches(n_teams=16, seasons=7, seed=3))


@pytest.fixture(autouse=True)
def fast_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small default models, so the command's logic is tested quickly."""
    monkeypatch.setattr(
        build_model,
        "ModelParams",
        lambda num_threads=0: ModelParams(
            n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=num_threads
        ),
    )


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def write_inputs(tmp_path: Path, features: pd.DataFrame) -> list[str]:
    features.to_parquet(tmp_path / "features.parquet", index=False)
    odds = pd.DataFrame(
        {"match_id": features["match_id"], "psc_h": 2.4, "psc_d": 3.4, "psc_a": 3.3}
    )
    odds.to_parquet(tmp_path / "enriched.parquet", index=False)
    return [
        "--features", str(tmp_path / "features.parquet"),
        "--enriched", str(tmp_path / "enriched.parquet"),
        "--oof-output", str(tmp_path / "out" / "oof.parquet"),
        "--model-output", str(tmp_path / "models" / "xg_model.pkl"),
        "--params-file", str(tmp_path / "models" / "best_params.json"),
        "--refit-days", "150",
        "--trials", "3",
        "--max-estimators", "30",
        "--threads", "1",
    ]  # fmt: skip


@pytest.mark.usefixtures("isolated_logging")
def test_full_run_tunes_saves_settings_and_a_working_model(
    tmp_path: Path, features: pd.DataFrame
) -> None:
    assert main(write_inputs(tmp_path, features)) == 0

    tuned = load_params(tmp_path / "models" / "best_params.json")
    assert tuned.n_estimators <= 30

    model_path = tmp_path / "models" / "xg_model.pkl"
    bundle = load_bundle(model_path)
    assert bundle.params == tuned
    assert bundle.calibration.scale_home > 0 and -1 < bundle.calibration.rho < 1
    assert (
        bundle.trained_through == features["date"].max()
        or bundle.trained_through <= features["date"].max()
    )
    assert {"logloss_1x2", "acc_1x2", "n", "market_logloss_1x2"} <= set(bundle.metrics)

    rates = predict_rates(bundle, features.tail(20))
    assert len(rates) == 20 and (rates > 0).all().all()
    described = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert described["n_train_rows"] == bundle.n_train_rows

    oof = pd.read_parquet(tmp_path / "out" / "oof.parquet")
    assert oof["match_id"].is_unique and oof["season"].min() == 2022


@pytest.mark.usefixtures("isolated_logging")
def test_no_tune_reuses_saved_settings(tmp_path: Path, features: pd.DataFrame) -> None:
    args = write_inputs(tmp_path, features)
    saved = ModelParams(
        n_estimators=21,
        num_leaves=5,
        min_child_samples=12,
        learning_rate=0.07,
        num_threads=3,
    )
    save_params(tmp_path / "models" / "best_params.json", saved)
    assert main([*args, "--no-tune", "--no-compare"]) == 0
    bundle = load_bundle(tmp_path / "models" / "xg_model.pkl")
    assert bundle.params.n_estimators == 21 and bundle.params.learning_rate == 0.07
    assert (
        bundle.params.num_threads == 1
    )  # the command-line thread count wins over the saved one


@pytest.mark.usefixtures("isolated_logging")
def test_no_tune_without_saved_settings_falls_back_to_defaults(
    tmp_path: Path, features: pd.DataFrame
) -> None:
    args = write_inputs(tmp_path, features)
    assert main([*args, "--no-tune", "--no-compare"]) == 0
    assert (
        load_bundle(tmp_path / "models" / "xg_model.pkl").params.n_estimators == 25
    )  # the fast default


@pytest.mark.usefixtures("isolated_logging")
def test_missing_or_outdated_inputs_are_reported(
    tmp_path: Path, features: pd.DataFrame
) -> None:
    assert main(["--features", str(tmp_path / "nope.parquet")]) == 1
    outdated = features.drop(columns=["home_elo"])
    assert main(write_inputs(tmp_path, outdated)) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_unusable_development_seasons_are_reported(
    tmp_path: Path, features: pd.DataFrame
) -> None:
    args = [
        *write_inputs(tmp_path, features),
        "--first-eval-season",
        "2030",
        "--test-from-season",
        "2031",
    ]
    assert main(args) == 1
    assert not (tmp_path / "models" / "xg_model.pkl").exists()


def test_test_slice_must_follow_the_first_evaluated_season() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--first-eval-season", "2024", "--test-from-season", "2024"])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def make_report() -> pd.DataFrame:
    rows = []
    for model, loss in (
        ("lightgbm", 1.00),
        ("league_avg", 1.07),
        ("market_closing", 0.98),
    ):
        for slice_name in ("dev", "test"):
            rows.append(
                {
                    "model": model,
                    "slice": slice_name,
                    "n": 100,
                    "mae_xg": 0.4,
                    "rmse_xg": 0.5,
                    "goals_nll": 1.5,
                    "scoreline_nll": 3.0,
                    "logloss_1x2": loss,
                    "acc_1x2": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_combine_puts_default_and_tuned_rows_side_by_side() -> None:
    tuned = make_report()
    default = make_report().assign(logloss_1x2=1.02)
    combined = _combine(tuned, default)
    assert list(dict.fromkeys(combined["model"])) == [
        "lgbm_default",
        "lgbm_tuned",
        "league_avg",
        "market_closing",
    ]
    assert (
        combined.loc[combined["model"] == "lgbm_default", "logloss_1x2"].eq(1.02).all()
    )
    assert list(dict.fromkeys(_combine(tuned, None)["model"])) == [
        "lgbm_tuned",
        "league_avg",
        "market_closing",
    ]


def test_bundle_metrics_come_from_the_test_slice() -> None:
    metrics = _test_metrics(make_report())
    assert metrics["logloss_1x2"] == 1.00
    assert metrics["market_logloss_1x2"] == 0.98
    assert set(metrics) >= {
        "n",
        "mae_xg",
        "rmse_xg",
        "goals_nll",
        "scoreline_nll",
        "acc_1x2",
    }
    assert _test_metrics(make_report()[lambda frame: frame["model"] != "lightgbm"]) == {
        "market_logloss_1x2": 0.98
    }
