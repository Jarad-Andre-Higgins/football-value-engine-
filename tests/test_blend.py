"""Tests for the blend test: does the model add information beyond the market?"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import blend, logging_config
from src.blend import (
    BlendResult,
    blend_report,
    fit_model_weight,
    load_model_weight,
    main,
    match_losses,
    paired_bootstrap_interval,
    save_blend,
)
from src.evaluate import OUTCOMES, Calibration, outcome_table
from src.features import build_features
from src.model import ModelParams, walk_forward_predict
from tests.test_features import make_matches

FAST = ModelParams(n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=1)
TEST_FROM = 2024


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


# --------------------------------------------------------------------------- #
# Fitting the weight
# --------------------------------------------------------------------------- #
def noisy_view(truth: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """A distorted copy of the true probabilities (log-normal noise, renormalised)."""
    view = truth * np.exp(rng.normal(0.0, noise, truth.shape))
    return view / view.sum(axis=1, keepdims=True)


def simulated(
    model_noise: float, market_noise: float, n: int = 20_000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True probabilities, outcomes drawn from them, and two noisy views of them."""
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet([4.0, 3.0, 3.5], size=n)
    cumulative = truth.cumsum(axis=1)
    outcome = (rng.random(n)[:, None] > cumulative).sum(axis=1).astype(np.intp)
    return (
        noisy_view(truth, model_noise, rng),
        noisy_view(truth, market_noise, rng),
        outcome,
    )


def test_match_losses_are_the_log_probability_of_what_happened() -> None:
    probabilities = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.0, 0.5, 0.5]])
    losses = match_losses(probabilities, np.array([0, 2, 0], dtype=np.intp))
    np.testing.assert_allclose(losses[:2], [-np.log(0.7), -np.log(0.6)])
    assert np.isfinite(losses[2])  # a probability of zero is floored, not infinite


def test_the_weight_goes_to_the_better_informed_side() -> None:
    model, market, outcome = simulated(model_noise=0.0, market_noise=0.5)
    assert fit_model_weight(model, market, outcome) > 0.9  # the model knows the truth
    model, market, outcome = simulated(model_noise=0.5, market_noise=0.0)
    assert fit_model_weight(model, market, outcome) < 0.1  # the market knows the truth
    model, market, outcome = simulated(model_noise=0.4, market_noise=0.4)
    assert fit_model_weight(model, market, outcome) == pytest.approx(0.5, abs=0.1)


def test_the_fitted_weight_beats_either_side_alone() -> None:
    model, market, outcome = simulated(model_noise=0.35, market_noise=0.25, seed=4)
    weight = fit_model_weight(model, market, outcome)
    blended = match_losses((1 - weight) * market + weight * model, outcome).mean()
    assert blended < match_losses(market, outcome).mean()
    assert blended < match_losses(model, outcome).mean()
    assert 0.0 < weight < 1.0


def test_weight_inputs_are_validated() -> None:
    probabilities = np.full((4, 3), 1 / 3)
    outcome = np.zeros(4, dtype=np.intp)
    with pytest.raises(ValueError, match="same shape"):
        fit_model_weight(probabilities, probabilities[:3], outcome)
    with pytest.raises(ValueError, match="at least one match"):
        fit_model_weight(probabilities[:0], probabilities[:0], outcome[:0])
    with pytest.raises(ValueError, match="one result per match"):
        fit_model_weight(probabilities, probabilities, outcome[:3])


# --------------------------------------------------------------------------- #
# The bootstrap
# --------------------------------------------------------------------------- #
def test_the_interval_brackets_the_mean_and_follows_the_standard_error() -> None:
    values = np.random.default_rng(1).normal(0.02, 0.5, 4_000)
    low, high = paired_bootstrap_interval(values, n_resamples=1_000)
    assert low < values.mean() < high
    expected_width = 2 * 1.96 * values.std() / np.sqrt(len(values))
    assert high - low == pytest.approx(expected_width, rel=0.15)


def test_more_matches_make_the_interval_narrower() -> None:
    rng = np.random.default_rng(2)
    small = paired_bootstrap_interval(rng.normal(0, 0.5, 250), n_resamples=500)
    large = paired_bootstrap_interval(rng.normal(0, 0.5, 4_000), n_resamples=500)
    assert (large[1] - large[0]) < 0.5 * (small[1] - small[0])


def test_the_bootstrap_is_reproducible_and_seeded() -> None:
    values = np.random.default_rng(3).normal(0.0, 1.0, 300)
    assert paired_bootstrap_interval(values, seed=7) == paired_bootstrap_interval(
        values, seed=7
    )
    assert paired_bootstrap_interval(values, seed=7) != paired_bootstrap_interval(
        values, seed=8
    )


def test_resampling_in_blocks_gives_a_valid_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.random.default_rng(4).normal(0.1, 1.0, 400)
    monkeypatch.setattr(blend, "BOOTSTRAP_BUDGET", 1_000)  # two resamples per block
    low, high = paired_bootstrap_interval(
        values, n_resamples=301
    )  # an uneven last block
    assert low < values.mean() < high
    assert (high - low) == pytest.approx(2 * 1.96 * values.std() / 20.0, rel=0.25)


def test_a_constant_difference_has_a_degenerate_interval() -> None:
    assert paired_bootstrap_interval(np.full(50, 0.3), n_resamples=20) == (
        pytest.approx(0.3),
        pytest.approx(0.3),
    )


def test_bootstrap_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_interval(np.array([]))
    with pytest.raises(ValueError, match="level"):
        paired_bootstrap_interval(np.ones(5), level=1.0)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap_interval(np.ones(5), n_resamples=0)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def setup() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_features(make_matches(n_teams=16, seasons=7, seed=3))
    oof = walk_forward_predict(features, FAST, first_eval_season=2022, refit_days=90)
    return features, oof


def flat_odds(features: pd.DataFrame, home: float = 3.0) -> pd.DataFrame:
    """The same prices for every match: a market that knows nothing."""
    return pd.DataFrame(
        {
            "match_id": features["match_id"],
            "psc_h": home,
            "psc_d": 3.0,
            "psc_a": 3.0,
        }
    )


def model_prices(
    features: pd.DataFrame, oof: pd.DataFrame, calibration: Calibration
) -> pd.DataFrame:
    """Margin-free odds equal to the model's own calibrated 1X2 probabilities."""
    data = oof.merge(
        features[["match_id", "target_home_goals", "target_away_goals"]],
        on="match_id",
        how="left",
    )
    table = outcome_table(
        calibration.scale_home * data["pred_home_xg"].to_numpy(dtype="float64"),
        calibration.scale_away * data["pred_away_xg"].to_numpy(dtype="float64"),
        data["target_home_goals"].to_numpy(dtype="float64"),
        data["target_away_goals"].to_numpy(dtype="float64"),
        calibration.rho,
    )
    odds = 1.0 / table[list(OUTCOMES)].to_numpy()
    return pd.DataFrame(
        {
            "match_id": data["match_id"],
            "psc_h": odds[:, 0],
            "psc_d": odds[:, 1],
            "psc_a": odds[:, 2],
        }
    )


def test_a_model_with_real_structure_beats_a_market_that_knows_nothing(
    setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof = setup
    result = blend_report(
        oof, features, flat_odds(features), test_from_season=TEST_FROM
    )
    assert isinstance(result, BlendResult)
    assert result.weight > 0.5
    assert list(result.table["slice"]) == ["dev", "test"]
    test = result.table.set_index("slice").loc["test"]
    assert test["model"] < test["market"]
    assert test["gain"] > 0 and test["gain_low"] < test["gain"] < test["gain_high"]
    assert test["blend"] <= test["market"]
    assert result.test_from_season == TEST_FROM
    assert result.adds_information()


def test_a_market_that_equals_the_model_leaves_nothing_to_add(
    setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof = setup
    first = blend_report(oof, features, flat_odds(features), test_from_season=TEST_FROM)
    same = model_prices(features, oof, first.calibration)
    result = blend_report(oof, features, same, test_from_season=TEST_FROM)
    assert result.calibration == first.calibration  # odds never touch the calibration
    np.testing.assert_allclose(result.table["gain"], 0.0, atol=1e-9)
    np.testing.assert_allclose(result.table["market"], result.table["model"], atol=1e-9)
    assert (result.table["gain_low"] <= 1e-9).all() and (
        result.table["gain_high"] >= -1e-9
    ).all()
    assert not result.adds_information()


def dev_row(result: BlendResult) -> pd.Series:
    return result.table.set_index("slice").loc["dev"]


def test_the_weight_and_calibration_never_see_the_test_odds(
    setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof = setup
    base = blend_report(oof, features, flat_odds(features), test_from_season=TEST_FROM)
    odds = flat_odds(features)
    in_test = features["season"] >= TEST_FROM
    scrambled = odds.assign(psc_h=np.where(in_test, 1.3, odds["psc_h"]))
    again = blend_report(oof, features, scrambled, test_from_season=TEST_FROM)
    assert again.weight == base.weight
    assert again.calibration == base.calibration
    assert dev_row(again)["blend"] == pytest.approx(dev_row(base)["blend"])
    assert (
        again.table.set_index("slice").loc["test", "market"]
        != base.table.set_index("slice").loc["test", "market"]
    )


def test_matches_without_odds_are_left_out(
    setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof = setup
    everything = blend_report(
        oof, features, flat_odds(features), test_from_season=TEST_FROM
    )
    odds = flat_odds(features)
    odds.loc[::5, ["psc_h", "psc_d", "psc_a"]] = np.nan
    fewer = blend_report(oof, features, odds, test_from_season=TEST_FROM)
    assert (fewer.table["n"] < everything.table["n"]).all()
    assert fewer.table["n"].sum() > 0.7 * everything.table["n"].sum()


def test_an_empty_development_slice_is_rejected(
    setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof = setup
    with pytest.raises(ValueError, match="No development matches"):
        blend_report(oof, features, flat_odds(features), test_from_season=2022)


def test_a_missing_test_slice_is_reported_not_faked(
    setup: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    features, oof = setup
    result = blend_report(oof, features, flat_odds(features), test_from_season=2099)
    assert list(result.table["slice"]) == ["dev"]
    assert not result.adds_information()


# --------------------------------------------------------------------------- #
# Saving and loading the weight
# --------------------------------------------------------------------------- #
def make_result(weight: float = 0.2371) -> BlendResult:
    table = pd.DataFrame(
        [
            {"slice": "dev", "n": 900, "market": 0.99, "model": 1.0, "blend": 0.985,
             "gain": 0.005, "gain_low": 0.001, "gain_high": 0.009},
            {"slice": "test", "n": 400, "market": 0.97, "model": 0.98, "blend": 0.969,
             "gain": 0.001, "gain_low": -0.002, "gain_high": 0.004},
        ]
    )  # fmt: skip
    return BlendResult(weight, Calibration(1.02, 0.97, -0.05), table, 2025)


def test_the_weight_survives_a_round_trip(tmp_path: Path) -> None:
    path = save_blend(make_result(), tmp_path / "models" / "blend.json")
    assert load_model_weight(path) == pytest.approx(0.2371)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["adds_information"] is False  # the test interval includes zero
    assert saved["test_from_season"] == 2025
    assert saved["calibration"]["rho"] == pytest.approx(-0.05)
    assert [row["slice"] for row in saved["slices"]] == ["dev", "test"]


def test_loading_rejects_a_missing_or_damaged_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m src.blend"):
        load_model_weight(tmp_path / "nope.json")
    path = tmp_path / "blend.json"
    cases = {
        "not json at all": "valid JSON",
        json.dumps([1, 2, 3]): "unsupported layout",
        json.dumps({"schema_version": 99, "model_weight": 0.2}): "unsupported layout",
        json.dumps({"schema_version": 1}): "no model_weight",
        json.dumps({"schema_version": 1, "model_weight": "0.2"}): "non-numeric",
        json.dumps({"schema_version": 1, "model_weight": True}): "non-numeric",
        json.dumps({"schema_version": 1, "model_weight": 1.5}): "between 0 and 1",
        json.dumps({"schema_version": 1, "model_weight": -0.1}): "between 0 and 1",
    }
    for text, message in cases.items():
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_model_weight(path)


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def write_inputs(tmp_path: Path, setup: tuple[pd.DataFrame, pd.DataFrame]) -> list[str]:
    features, oof = setup
    features.to_parquet(tmp_path / "features.parquet", index=False)
    oof.to_parquet(tmp_path / "oof.parquet", index=False)
    flat_odds(features).to_parquet(tmp_path / "enriched.parquet", index=False)
    return [
        "--oof", str(tmp_path / "oof.parquet"),
        "--features", str(tmp_path / "features.parquet"),
        "--enriched", str(tmp_path / "enriched.parquet"),
        "--test-from-season", str(TEST_FROM),
        "--output", str(tmp_path / "models" / "blend.json"),
    ]  # fmt: skip


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_saves_a_weight_the_engine_can_load(
    tmp_path: Path, setup: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    assert main(write_inputs(tmp_path, setup)) == 0
    weight = load_model_weight(tmp_path / "models" / "blend.json")
    assert (
        0.5 < weight <= 1.0
    )  # the flat market knows nothing, so the model gets the weight


@pytest.mark.usefixtures("isolated_logging")
def test_no_save_prints_without_writing(
    tmp_path: Path, setup: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    assert main([*write_inputs(tmp_path, setup), "--no-save"]) == 0
    assert not (tmp_path / "models" / "blend.json").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_names_the_step_to_run_when_an_input_is_missing(
    tmp_path: Path,
    setup: tuple[pd.DataFrame, pd.DataFrame],
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = write_inputs(tmp_path, setup)
    (tmp_path / "oof.parquet").unlink()
    with caplog.at_level(logging.ERROR):
        assert main(args) == 1
    assert "python -m src.build_model" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_cleanly_when_there_is_nothing_to_fit(
    tmp_path: Path, setup: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    args = [*write_inputs(tmp_path, setup), "--test-from-season", "2022"]
    assert main(args) == 1
    assert not (tmp_path / "models" / "blend.json").exists()
