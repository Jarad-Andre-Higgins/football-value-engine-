"""Tests for the Dixon-Coles team-strength model, the loader and the backtest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dc_baseline import (
    DixonColesModel,
    UnknownTeamError,
    load_football_data_csv,
    load_matches,
    outcome_probabilities,
    time_decay_weights,
)
from src.dixon_coles import score_matrix
from src.poc_dixon_coles import walk_forward_backtest

TRUE_BASE = 0.15
TRUE_HOME_ADV = 0.25
TRUE_RHO = -0.10


def simulate_league(
    n_teams: int = 20, seasons: int = 2, seed: int = 11, spread: float = 0.30
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Simulate double round-robin seasons from known Dixon-Coles strengths."""
    rng = np.random.default_rng(seed)
    attack = rng.normal(0.0, spread, n_teams)
    attack -= attack.mean()
    defence = rng.normal(0.0, spread, n_teams)
    defence -= defence.mean()
    start = pd.Timestamp("2024-08-10")
    rows = []
    for season in range(seasons):
        pairs = [(h, a) for h in range(n_teams) for a in range(n_teams) if h != a]
        rng.shuffle(pairs)
        for k, (h, a) in enumerate(pairs):
            lam = np.exp(TRUE_BASE + TRUE_HOME_ADV + attack[h] - defence[a])
            mu = np.exp(TRUE_BASE + attack[a] - defence[h])
            matrix = score_matrix(lam, mu, TRUE_RHO)
            home_goals, away_goals = divmod(
                rng.choice(matrix.size, p=matrix.ravel()), 11
            )
            day = 365 * season + int(k * 280 / len(pairs))
            rows.append(
                (
                    start + pd.Timedelta(day, unit="D"),
                    f"T{h:02d}",
                    f"T{a:02d}",
                    home_goals,
                    away_goals,
                )
            )
    columns = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    league = pd.DataFrame(rows, columns=columns)
    return (
        league.sort_values("date", kind="stable").reset_index(drop=True),
        attack,
        defence,
    )


@pytest.fixture(scope="module")
def league() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return simulate_league()


@pytest.fixture(scope="module")
def fitted(league: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> DixonColesModel:
    return DixonColesModel().fit(league[0])


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def test_fit_recovers_known_parameters(
    league: tuple[pd.DataFrame, np.ndarray, np.ndarray], fitted: DixonColesModel
) -> None:
    _, true_attack, true_defence = league
    params = fitted.params
    assert params is not None
    assert fitted.converged
    order = [params.teams.index(f"T{i:02d}") for i in range(len(true_attack))]
    assert np.corrcoef(true_attack, params.attack[order])[0, 1] > 0.75
    assert np.corrcoef(true_defence, params.defence[order])[0, 1] > 0.75
    assert params.home_adv == pytest.approx(TRUE_HOME_ADV, abs=0.10)
    assert params.base == pytest.approx(TRUE_BASE, abs=0.10)
    assert -0.30 < params.rho < 0.10


def test_ratings_are_centred_on_zero(fitted: DixonColesModel) -> None:
    params = fitted.params
    assert params is not None
    assert params.attack.mean() == pytest.approx(0.0, abs=1e-9)
    assert params.defence.mean() == pytest.approx(0.0, abs=1e-9)


def test_fit_never_sees_matches_on_or_after_as_of(
    league: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    matches = league[0]
    cutoff = matches["date"].iloc[len(matches) // 2]
    honest = DixonColesModel().fit(matches, as_of=cutoff)

    tampered = matches.copy()
    future = tampered["date"] >= cutoff
    tampered.loc[future, ["home_goals", "away_goals"]] = 9  # absurd future results
    same = DixonColesModel().fit(tampered, as_of=cutoff)

    assert honest.n_matches == int((matches["date"] < cutoff).sum())
    assert honest.params is not None and same.params is not None
    np.testing.assert_allclose(honest.params.attack, same.params.attack, atol=1e-12)
    np.testing.assert_allclose(honest.params.defence, same.params.defence, atol=1e-12)
    assert honest.params.rho == pytest.approx(same.params.rho, abs=1e-12)


def test_fit_rejects_too_little_data(
	league: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="at least"):
        DixonColesModel().fit(league[0].head(10))


def test_fit_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        DixonColesModel().fit(pd.DataFrame({"date": []}))


def test_invalid_half_life_is_rejected() -> None:
    with pytest.raises(ValueError):
        DixonColesModel(half_life_days=0)


def test_time_decay_halves_every_half_life() -> None:
    weights = time_decay_weights(np.array([0.0, 100.0, 200.0]), half_life_days=100.0)
    np.testing.assert_allclose(weights, [1.0, 0.5, 0.25])


# --------------------------------------------------------------------------- #
# Predicting
# --------------------------------------------------------------------------- #
def test_predict_matrix_is_a_valid_distribution(fitted: DixonColesModel) -> None:
    matrix = fitted.predict_matrix("T00", "T01")
    assert matrix.shape == (11, 11)
    assert matrix.sum() == pytest.approx(1.0, abs=1e-12)


def test_best_team_is_favourite_at_home_against_worst(fitted: DixonColesModel) -> None:
    table = fitted.ratings()
    best, worst = str(table.loc[0, "team"]), str(table.loc[len(table) - 1, "team"])
    lam, mu = fitted.expected_goals(best, worst)
    assert lam > mu
    home_win, _, away_win = outcome_probabilities(fitted.predict_matrix(best, worst))
    assert home_win > away_win


def test_unknown_team_error_suggests_close_names(fitted: DixonColesModel) -> None:
    with pytest.raises(UnknownTeamError, match="Did you mean: T00"):
        fitted.predict_matrix("t00", "T01")


def test_team_cannot_play_itself(fitted: DixonColesModel) -> None:
    with pytest.raises(ValueError, match="itself"):
        fitted.expected_goals("T00", "T00")


def test_predicting_before_fitting_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        DixonColesModel().predict_matrix("A", "B")


def test_rho_override_gives_plain_poisson(fitted: DixonColesModel) -> None:
    poisson = fitted.predict_matrix("T00", "T01", rho=0.0)
    lam, mu = fitted.expected_goals("T00", "T01")
    np.testing.assert_allclose(poisson, score_matrix(lam, mu, 0.0), atol=1e-12)


def test_outcome_probabilities_orientation() -> None:
    matrix = np.zeros((11, 11))
    matrix[2, 0] = 1.0  # home won 2-0
    assert outcome_probabilities(matrix) == (1.0, 0.0, 0.0)
    matrix = np.zeros((11, 11))
    matrix[1, 1] = 1.0
    assert outcome_probabilities(matrix) == (0.0, 1.0, 0.0)
    matrix = np.zeros((11, 11))
    matrix[0, 3] = 1.0
    assert outcome_probabilities(matrix) == (0.0, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_loader_handles_real_world_csv_quirks(tmp_path: Path) -> None:
    csv = tmp_path / "E0.csv"
    csv.write_text(
        "\ufeffDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H\n"
        "E0,15/08/2025,20:00,Arsenal,Chelsea,2,1,H,1.90\n"
        "E0,16/08/25,12:30, Everton ,Fulham,0,0,D,2.50\n"
        "E0,17/08/2025,15:00,Leeds,Wolves,,,,2.10\n"
        ",,,,,,,,\n",
        encoding="utf-8",
    )
    matches = load_football_data_csv(csv)
    assert list(matches.columns) == [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]
    assert len(matches) == 2  # unplayed and blank rows are dropped
    assert list(matches["home_team"]) == ["Arsenal", "Everton"]  # whitespace stripped
    assert list(matches["date"]) == [
        pd.Timestamp("2025-08-15"),
        pd.Timestamp("2025-08-16"),
    ]
    assert list(matches["home_goals"]) == [2, 0]
    assert pd.api.types.is_integer_dtype(matches["home_goals"])
    
def test_loader_rejects_files_without_the_expected_columns(tmp_path: Path) -> None:
    csv = tmp_path / "other.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="other.csv"):
        load_football_data_csv(csv)


def test_load_matches_combines_files_and_drops_duplicates(tmp_path: Path) -> None:
    text = "Date,HomeTeam,AwayTeam,FTHG,FTAG\n15/08/2025,Arsenal,Chelsea,2,1\n"
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    first.write_text(text, encoding="utf-8")
    second.write_text(
        text + "22/08/2025,Chelsea,Arsenal,0,0\n",
        encoding="utf-8",
    )
    matches = load_matches([first, second])
    assert len(matches) == 2
    assert matches["date"].is_monotonic_increasing


def test_load_matches_requires_at_least_one_file() -> None:
    with pytest.raises(ValueError, match="No results files"):
        load_matches([])


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
def test_walk_forward_backtest_beats_the_naive_baseline() -> None:
    matches, _, _ = simulate_league(n_teams=14, seasons=2, seed=5, spread=0.40)
    result = walk_forward_backtest(matches, test_fraction=0.25)
    assert result.n_predicted > 50
    assert result.n_refits > 5
    assert result.log_loss_model < result.log_loss_baseline
    assert 0.0 < result.accuracy < 1.0
    assert np.isfinite(result.scoreline_loss_dc)
    assert np.isfinite(result.scoreline_loss_poisson)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5])
def test_walk_forward_rejects_invalid_test_fraction(fraction: float) -> None:
    matches, _, _ = simulate_league(n_teams=6, seasons=1, seed=2)
    with pytest.raises(ValueError, match="test_fraction"):
        walk_forward_backtest(matches, test_fraction=fraction)
