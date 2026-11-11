"""Tests for the shot-based xG proxy and its use as a fallback for Understat."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import logging_config, xg_proxy
from src.enrich import (
    EXTRA_COLUMNS,
    SHOT_PROXY_SOURCE,
    _parse_args,
    enrich_with_shots,
    main,
    xg_coverage,
)
from src.features import LONG, build_features, leakage_audit
from src.model import ModelParams, fit_boosters, predict_xg, training_mask
from src.xg_proxy import (
    MIN_FIT_OBSERVATIONS,
    ShotModel,
    fit_shot_model,
    shot_xg,
    xg_per_goal,
)
from tests.test_features import make_matches


def simulated_shots(
    n: int, on_price: float, off_price: float, *, season: int = 2019, seed: int = 7
) -> pd.DataFrame:
    """``n`` matches whose goals follow the pricing model: goals ~ Poisson(price x shots)."""
    rng = np.random.default_rng(seed)
    columns: dict[str, object] = {"season": season}
    for side in ("home", "away"):
        on_target = rng.poisson(4.5, n)
        other = rng.poisson(8.0, n)
        columns[f"{side}_shots_on_target"] = on_target
        columns[f"{side}_shots"] = on_target + other
        columns[f"{side}_goals"] = rng.poisson(on_price * on_target + off_price * other)
    return pd.DataFrame(columns)


def league_table() -> pd.DataFrame:
    """A pipeline-style match table (shots but no xG): 20 teams, 3 seasons."""
    table = make_matches(n_teams=20, seasons=3)
    return table.drop(
        columns=[
            "home_xg",
            "away_xg",
            "home_deep",
            "away_deep",
            "home_ppda",
            "away_ppda",
        ]
    )


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return league_table()


@pytest.fixture(scope="module")
def enriched(table: pd.DataFrame) -> pd.DataFrame:
    return enrich_with_shots(table)[0]


@pytest.fixture(scope="module")
def built(enriched: pd.DataFrame) -> pd.DataFrame:
    return build_features(enriched)


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
# Fitting the prices
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("on_price", "off_price"), [(0.30, 0.04), (0.10, 0.25)])
def test_fit_recovers_the_prices_that_generated_the_goals(
    on_price: float, off_price: float
) -> None:
    model = fit_shot_model(simulated_shots(20_000, on_price, off_price))
    assert model.on_target == pytest.approx(on_price, abs=0.01)
    assert model.off_target == pytest.approx(off_price, abs=0.01)
    assert (model.season, model.n_observations) == (2019, 40_000)


def test_other_shots_that_never_score_are_priced_at_zero() -> None:
    model = fit_shot_model(simulated_shots(20_000, 0.30, 0.0))
    assert 0.0 <= model.off_target < 1e-6
    assert model.on_target == pytest.approx(0.30, abs=0.01)


def test_prices_stay_non_negative_when_other_shots_look_harmful() -> None:
    rng = np.random.default_rng(5)
    on_target = rng.poisson(4.5, 5_000)
    other = np.clip(20 - 2 * on_target + rng.poisson(2, 5_000), 0, None)
    goals = rng.poisson(0.3 * on_target)
    frame = pd.DataFrame(
        {
            "season": 2019,
            "home_goals": goals,
            "away_goals": goals,
            "home_shots_on_target": on_target,
            "away_shots_on_target": on_target,
            "home_shots": on_target + other,
            "away_shots": on_target + other,
        }
    )
    model = fit_shot_model(frame)
    assert 0.0 <= model.off_target < 1e-6
    # With nothing left to explain by other shots, the price is goals per shot on target.
    assert model.on_target == pytest.approx(goals.sum() / on_target.sum(), rel=1e-6)


def test_only_the_calibration_season_is_used() -> None:
    first = simulated_shots(2_000, 0.30, 0.04, season=2019, seed=1)
    later = simulated_shots(2_000, 0.10, 0.10, season=2020, seed=2)
    both = pd.concat([first, later], ignore_index=True)
    assert fit_shot_model(both) == fit_shot_model(first)  # later seasons cannot leak in
    assert fit_shot_model(both, season=2020) == fit_shot_model(later)


def test_the_calibration_season_defaults_to_the_earliest_one() -> None:
    later = simulated_shots(2_000, 0.10, 0.10, season=2021, seed=2)
    first = simulated_shots(2_000, 0.30, 0.04, season=2019, seed=1)
    shuffled = pd.concat(
        [later, first], ignore_index=True
    )  # earliest season is not first
    assert fit_shot_model(shuffled).season == 2019


def test_incomplete_rows_are_left_out_rather_than_read_as_zero() -> None:
    clean = simulated_shots(3_000, 0.30, 0.04)
    holey = clean.astype(
        {
            "home_shots_on_target": "float64",
            "away_shots": "float64",
            "home_goals": "float64",
        }
    )
    holey.loc[:99, "home_shots_on_target"] = np.nan
    holey.loc[100:199, "away_shots"] = np.nan
    holey.loc[200:249, "home_goals"] = np.nan
    model = fit_shot_model(holey)
    assert model.n_observations == 6_000 - 250
    reference = fit_shot_model(clean)
    assert model.on_target == pytest.approx(reference.on_target, abs=0.01)
    assert model.off_target == pytest.approx(reference.off_target, abs=0.01)


def test_team_matches_without_any_shots_are_ignored() -> None:
    silent = simulated_shots(2_000, 0.30, 0.04)
    silent.loc[:49, ["home_shots", "home_shots_on_target", "home_goals"]] = 0
    model = fit_shot_model(silent)
    assert model.n_observations == 4_000 - 50
    assert model.on_target == pytest.approx(0.30, abs=0.02)


def test_a_fit_that_does_not_settle_is_an_error_not_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xg_proxy, "MAX_ITERATIONS", 3)
    with pytest.raises(ValueError, match="did not settle"):
        fit_shot_model(simulated_shots(2_000, 0.30, 0.04))


# --------------------------------------------------------------------------- #
# Refusing bad input
# --------------------------------------------------------------------------- #
def test_fit_reports_missing_columns() -> None:
    frame = simulated_shots(300, 0.30, 0.04).drop(columns=["away_shots"])
    with pytest.raises(ValueError, match=r"missing columns.*away_shots"):
        fit_shot_model(frame)


def test_fit_needs_matches() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_shot_model(simulated_shots(300, 0.30, 0.04).iloc[:0])


def test_fit_needs_enough_team_matches() -> None:
    frame = simulated_shots(250, 0.30, 0.04)  # 500 team-matches
    assert fit_shot_model(frame, min_observations=500).n_observations == 500
    with pytest.raises(
        ValueError, match=r"only 500 team-matches.*2019/20.*at least 501"
    ):
        fit_shot_model(frame, min_observations=501)


def test_the_default_minimum_is_four_hundred_team_matches() -> None:
    assert MIN_FIT_OBSERVATIONS == 400
    assert fit_shot_model(simulated_shots(200, 0.30, 0.04)).n_observations == 400
    with pytest.raises(ValueError, match=r"only 398 team-matches.*at least 400"):
        fit_shot_model(simulated_shots(199, 0.30, 0.04))


def test_an_unknown_calibration_season_is_refused() -> None:
    with pytest.raises(ValueError, match=r"only 0 team-matches.*2030/31"):
        fit_shot_model(simulated_shots(500, 0.30, 0.04), season=2030)


@pytest.mark.parametrize("spoil", ["goals", "on_target", "other_shots"])
def test_fit_refuses_when_there_is_nothing_to_price(spoil: str) -> None:
    frame = simulated_shots(500, 0.30, 0.04)
    if spoil == "goals":
        frame[["home_goals", "away_goals"]] = 0
    elif spoil == "on_target":
        frame[["home_shots_on_target", "away_shots_on_target"]] = 0
    else:
        frame["home_shots"] = frame["home_shots_on_target"]
        frame["away_shots"] = frame["away_shots_on_target"]
    with pytest.raises(
        ValueError, match="no goals, no shots on target or no other shots"
    ):
        fit_shot_model(frame)


# --------------------------------------------------------------------------- #
# Pricing a match
# --------------------------------------------------------------------------- #
def test_proxy_is_the_priced_sum_of_the_shots() -> None:
    model = ShotModel(0.30, 0.05, 2019, 1_000)
    frame = pd.DataFrame(
        {
            "home_shots": [12, 3],
            "home_shots_on_target": [4, 5],
            "away_shots": [8, 10],
            "away_shots_on_target": [2, 1],
        },
        index=[10, 20],
    )
    xg = shot_xg(frame, model)
    assert list(xg.columns) == ["home_xg", "away_xg"]
    assert list(xg.index) == [10, 20]
    # Second home row lists fewer shots (3) than shots on target (5): no other shots.
    assert xg["home_xg"].tolist() == pytest.approx([0.30 * 4 + 0.05 * 8, 0.30 * 5])
    assert xg["away_xg"].tolist() == pytest.approx(
        [0.30 * 2 + 0.05 * 6, 0.30 * 1 + 0.05 * 9]
    )


def test_proxy_is_missing_where_the_shot_counts_are() -> None:
    model = ShotModel(0.30, 0.05, 2019, 1_000)
    frame = pd.DataFrame(
        {
            "home_shots": [np.nan, 10.0],
            "home_shots_on_target": [4.0, 3.0],
            "away_shots": [8.0, 9.0],
            "away_shots_on_target": [2.0, np.nan],
        }
    )
    xg = shot_xg(frame, model)
    assert np.isnan(xg.loc[0, "home_xg"]) and not np.isnan(xg.loc[0, "away_xg"])
    assert np.isnan(xg.loc[1, "away_xg"]) and not np.isnan(xg.loc[1, "home_xg"])


def test_pricing_reports_missing_columns() -> None:
    frame = simulated_shots(10, 0.30, 0.04).drop(columns=["home_shots_on_target"])
    with pytest.raises(ValueError, match=r"missing columns.*home_shots_on_target"):
        shot_xg(frame, ShotModel(0.30, 0.05, 2019, 1_000))


def test_xg_per_goal_compares_xg_with_goals_season_by_season() -> None:
    matches = pd.DataFrame(
        {
            "season": [2019, 2019, 2019, 2020],
            "home_goals": [2, 1, 3, 1],
            "away_goals": [0, 1, 3, 1],
            "home_xg": [1.0, 1.5, np.nan, 3.0],
            "away_xg": [0.5, 0.5, 2.0, 1.0],
        }
    )
    ratio = xg_per_goal(matches)
    # 2019 leaves out the match whose home xG is missing: xG 3.5 against 4 goals.
    assert ratio.loc[2019] == pytest.approx(3.5 / 4)
    assert ratio.loc[2020] == pytest.approx(4.0 / 2)


# --------------------------------------------------------------------------- #
# The enriched table
# --------------------------------------------------------------------------- #
def test_enriched_table_keeps_the_matches_and_uses_the_understat_layout(
    table: pd.DataFrame, enriched: pd.DataFrame
) -> None:
    assert list(enriched.columns) == [*table.columns, *EXTRA_COLUMNS, "xg_source"]
    pd.testing.assert_frame_equal(enriched[list(table.columns)], table)
    assert (enriched["xg_source"] == SHOT_PROXY_SOURCE).all()
    assert SHOT_PROXY_SOURCE == "shots_proxy"
    understat_only = [c for c in EXTRA_COLUMNS if c not in ("home_xg", "away_xg")]
    assert enriched[understat_only].isna().all().all()


def test_enriched_xg_is_the_priced_proxy(
    table: pd.DataFrame, enriched: pd.DataFrame
) -> None:
    _, model = enrich_with_shots(table)
    assert model.season == int(table["season"].min())
    proxy = shot_xg(table, model)
    pd.testing.assert_series_equal(enriched["home_xg"], proxy["home_xg"])
    pd.testing.assert_series_equal(enriched["away_xg"], proxy["away_xg"])
    assert enriched["home_xg"].notna().all()


def test_enrichment_needs_shot_data(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        enrich_with_shots(table.drop(columns=["home_shots"]))


def test_coverage_shows_xg_everywhere_and_no_ppda_or_deep(
    enriched: pd.DataFrame,
) -> None:
    coverage = xg_coverage(enriched)
    assert (coverage["xg"] == 100.0).all()
    assert (coverage["ppda"] == 0.0).all()
    assert (coverage["deep"] == 0.0).all()


def test_the_feature_builder_accepts_the_fallback_table(
    enriched: pd.DataFrame, built: pd.DataFrame
) -> None:
    joined = built.merge(enriched[["match_id", "home_xg", "away_xg"]], on="match_id")
    assert len(joined) == len(enriched)
    assert (joined["target_home_xg"] == joined["home_xg"]).all()
    assert (joined["target_away_xg"] == joined["away_xg"]).all()
    assert built[f"home_xg_for_ew{LONG}"].notna().any()
    assert built[f"home_ppda_for_ew{LONG}"].isna().all()  # no Understat PPDA


def test_features_built_on_the_proxy_never_see_the_future(
    enriched: pd.DataFrame,
) -> None:
    assert leakage_audit(enriched) == []


def test_the_models_train_on_the_fallback_features(built: pd.DataFrame) -> None:
    usable = training_mask(built)
    assert usable.sum() >= 300
    as_of = built.loc[usable, "date"].max() + pd.Timedelta(1, unit="D")
    boosters = fit_boosters(built, usable, ModelParams(n_estimators=20), as_of=as_of)
    predictions = predict_xg(boosters, built).to_numpy()
    assert np.isfinite(predictions).all()
    assert (predictions > 0).all()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def cli_args(tmp_path: Path, matches: pd.DataFrame | None = None) -> list[str]:
    path = tmp_path / "matches.parquet"
    (league_table() if matches is None else matches).to_parquet(path, index=False)
    return [
        "--xg-source", "shots",
        "--matches", str(path),
        "--output", str(tmp_path / "out" / "matches_enriched.parquet"),
        "--xg-output", str(tmp_path / "out" / "xg_matches.parquet"),
        # Neither of these exists: the fallback must never need them.
        "--registry", str(tmp_path / "no-registry.yaml"),
        "--understat-dir", str(tmp_path / "no-understat"),
    ]  # fmt: skip


@pytest.mark.usefixtures("isolated_logging")
def test_cli_writes_the_enriched_table_without_understat_or_a_registry(
    tmp_path: Path, table: pd.DataFrame
) -> None:
    assert main(cli_args(tmp_path)) == 0
    result = pd.read_parquet(tmp_path / "out" / "matches_enriched.parquet")
    assert len(result) == len(table)
    assert set(EXTRA_COLUMNS) <= set(result.columns)
    assert (result["xg_source"] == SHOT_PROXY_SOURCE).all()
    assert result["home_xg"].notna().all()
    assert not (
        tmp_path / "out" / "xg_matches.parquet"
    ).exists()  # an Understat-only file


@pytest.mark.usefixtures("isolated_logging")
def test_cli_says_loudly_that_the_xg_is_a_proxy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert main(cli_args(tmp_path)) == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    fallback = [m for m in warnings if "FALLBACK" in m]
    assert len(fallback) == 1
    assert "NOT Understat xG" in fallback[0]
    assert "2020/21" in fallback[0]  # the calibration season
    assert "--xg-source shots" in fallback[0]  # how to undo it


@pytest.mark.usefixtures("isolated_logging")
def test_cli_counts_matches_without_shot_data(
    tmp_path: Path, table: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    gaps = table.astype({"home_shots": "float64"})
    gaps.loc[[5, 700, 701], "home_shots"] = np.nan  # seasons 2020 and 2021
    with caplog.at_level(logging.WARNING):
        assert main(cli_args(tmp_path, gaps)) == 0
    assert f"3 of {len(table)} matches have no shot data" in caplog.text
    result = pd.read_parquet(tmp_path / "out" / "matches_enriched.parquet")
    assert result["home_xg"].isna().sum() == 3
    assert result["away_xg"].notna().all()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_stays_quiet_about_gaps_when_there_are_none(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert main(cli_args(tmp_path)) == 0
    assert "no shot data" not in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_cli_refuses_a_match_table_without_shots(
    tmp_path: Path, table: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    args = cli_args(tmp_path, table.drop(columns=["home_shots_on_target"]))
    with caplog.at_level(logging.ERROR):
        assert main(args) == 1
    assert "Cannot build the shot-based xG proxy" in caplog.text
    assert not (tmp_path / "out" / "matches_enriched.parquet").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_refuses_a_calibration_season_that_is_too_small(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    small = make_matches(n_teams=8, seasons=3)  # 112 team-matches in the first season
    with caplog.at_level(logging.ERROR):
        assert main(cli_args(tmp_path, small)) == 1
    assert "only 112 team-matches" in caplog.text
    assert not (tmp_path / "out" / "matches_enriched.parquet").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_needs_the_match_table(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    args = cli_args(tmp_path)
    (tmp_path / "matches.parquet").unlink()
    with caplog.at_level(logging.ERROR):
        assert main(args) == 1
    assert "python -m src.pipeline" in caplog.text


def test_real_xg_stays_the_default_source() -> None:
    assert _parse_args([]).xg_source == "understat"
    assert _parse_args(["--xg-source", "shots"]).xg_source == "shots"
    with pytest.raises(SystemExit):
        _parse_args(["--xg-source", "guess"])
