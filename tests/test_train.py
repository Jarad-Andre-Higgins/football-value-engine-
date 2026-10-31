"""Tests for the walk-forward training command."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from src import logging_config, train
from src.features import build_features
from src.model import ModelParams
from src.train import _verdict, main
from tests.test_features import make_matches


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return build_features(make_matches(n_teams=16, seasons=7, seed=3))


@pytest.fixture(autouse=True)
def fast_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Train small models so the command's logic is tested quickly."""
    monkeypatch.setattr(
        train,
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


def write_inputs(
    tmp_path: Path, features: pd.DataFrame, *, with_odds: bool = True
) -> list[str]:
    features.to_parquet(tmp_path / "features.parquet", index=False)
    args = [
        "--features", str(tmp_path / "features.parquet"),
        "--enriched", str(tmp_path / "enriched.parquet"),
        "--oof-output", str(tmp_path / "out" / "oof.parquet"),
        "--refit-days", "150",
        "--threads", "1",
    ]  # fmt: skip
    if with_odds:
        odds = pd.DataFrame(
            {"match_id": features["match_id"], "psc_h": 2.4, "psc_d": 3.4, "psc_a": 3.3}
        )
        odds.to_parquet(tmp_path / "enriched.parquet", index=False)
    return args


@pytest.mark.usefixtures("isolated_logging")
def test_cli_evaluates_and_saves_out_of_sample_predictions(
    tmp_path: Path, features: pd.DataFrame
) -> None:
    assert main(write_inputs(tmp_path, features)) == 0
    oof = pd.read_parquet(tmp_path / "out" / "oof.parquet")
    assert oof["match_id"].is_unique
    assert oof["season"].min() == 2022
    assert {"pred_home_xg", "pred_away_xg", "window_start"} <= set(oof.columns)


@pytest.mark.usefixtures("isolated_logging")
def test_cli_runs_without_an_odds_table(tmp_path: Path, features: pd.DataFrame) -> None:
    assert main(write_inputs(tmp_path, features, with_odds=False)) == 0


@pytest.mark.usefixtures("isolated_logging")
def test_cli_reports_missing_and_outdated_inputs(
    tmp_path: Path, features: pd.DataFrame
) -> None:
    assert main(["--features", str(tmp_path / "nope.parquet")]) == 1
    outdated = features.drop(columns=["home_elo"])
    assert main(write_inputs(tmp_path, outdated)) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_cli_reports_unusable_evaluation_seasons(
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


def test_cli_requires_the_test_slice_to_follow_the_first_evaluated_season() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--first-eval-season", "2024", "--test-from-season", "2023"])
    assert excinfo.value.code == 2


def test_verdict_describes_the_comparison_in_plain_words() -> None:
    report = pd.DataFrame(
        {
            "model": ["lightgbm", "league_avg", "strength_only", "market_closing"],
            "slice": ["test"] * 4,
            "logloss_1x2": [1.00, 1.07, 1.01, 0.98],
        }
    )
    lines = _verdict(report)
    assert "1X2 log-loss vs league_avg: better by 0.070" in lines
    assert "1X2 log-loss vs strength_only: better by 0.010" in lines
    assert any("+0.020" in line for line in lines)  # 1.00 - 0.98
    assert any("noise" in line for line in lines)
    worse = report.assign(logloss_1x2=[1.10, 1.07, 1.01, 0.98])
    assert "1X2 log-loss vs league_avg: WORSE by 0.030" in _verdict(worse)
    assert _verdict(report[report["model"] != "lightgbm"]) == []
