"""Tests for the leak-free feature pipeline."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features, logging_config
from src.elo import compute_elo
from src.features import (
    HALF_LIVES_DAYS,
    LONG,
    TARGET_COLUMNS,
    TEAM_FEATURES,
    add_form_features,
    add_schedule_features,
    add_venue_features,
    build_features,
    feature_columns,
    feature_coverage,
    league_context,
    leakage_audit,
    main,
    to_team_matches,
)
from src.schema import ODDS_COLUMNS


def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Double round robin by the circle method: every team plays once per round."""
    n = len(teams)
    rotating = teams[1:]
    first_leg = []
    for round_number in range(n - 1):
        order = [teams[0], *rotating]
        pairs = []
        for i in range(n // 2):
            a, b = order[i], order[n - 1 - i]
            pairs.append((a, b) if (round_number + i) % 2 == 0 else (b, a))
        first_leg.append(pairs)
        rotating = rotating[-1:] + rotating[:-1]
    return first_leg + [[(b, a) for a, b in pairs] for pairs in first_leg]


def make_matches(
    n_teams: int = 8,
    seasons: int = 3,
    leagues: tuple[str, ...] = ("E0",),
    seed: int = 1,
) -> pd.DataFrame:
    """Synthetic enriched matches with realistic scheduling, gaps and noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for league in leagues:
        teams = [f"{league.lower()}-{i:02d}" for i in range(n_teams)]
        attack = rng.normal(0.0, 0.25, n_teams)
        for s in range(seasons):
            season = 2020 + s
            start = pd.Timestamp(f"{season}-08-14")
            for round_number, fixtures in enumerate(round_robin(teams)):
                for home, away in fixtures:
                    date = start + pd.Timedelta(
                        7 * round_number + int(rng.integers(0, 2)), unit="D"
                    )
                    lam, mu = (
                        np.exp(0.3 + attack[teams.index(home)]),
                        np.exp(attack[teams.index(away)]),
                    )
                    has_xg = rng.random() > 0.1  # about one match in ten lacks xG data
                    rows.append(
                        {
                            "match_id": f"{league}-{season}-{date:%Y%m%d}-{home}-{away}",
                            "league": league,
                            "season": season,
                            "date": date,
                            "home_team_id": home,
                            "away_team_id": away,
                            "home_goals": int(rng.poisson(lam)),
                            "away_goals": int(rng.poisson(mu)),
                            "home_xg": float(rng.gamma(4, lam / 4))
                            if has_xg
                            else np.nan,
                            "away_xg": float(rng.gamma(4, mu / 4))
                            if has_xg
                            else np.nan,
                            "home_shots": int(rng.integers(5, 20)),
                            "away_shots": int(rng.integers(5, 20)),
                            "home_shots_on_target": int(rng.integers(1, 9)),
                            "away_shots_on_target": int(rng.integers(1, 9)),
                            "home_deep": float(rng.integers(2, 14))
                            if has_xg
                            else np.nan,
                            "away_deep": float(rng.integers(2, 14))
                            if has_xg
                            else np.nan,
                            "home_ppda": float(rng.uniform(6, 16))
                            if has_xg
                            else np.nan,
                            "away_ppda": float(rng.uniform(6, 16))
                            if has_xg
                            else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def reference_ewma(
    values: pd.Series, dates: pd.Series, half_life: float
) -> list[float]:
    """Weighted mean of strictly earlier non-missing values: weight 0.5 ** (age / half_life)."""
    out = []
    for k in range(len(values)):
        weights, total = 0.0, 0.0
        for i in range(k):
            if np.isnan(values.iloc[i]):
                continue
            weight = 0.5 ** ((dates.iloc[k] - dates.iloc[i]).days / half_life)
            weights += weight
            total += weight * values.iloc[i]
        out.append(total / weights if weights else np.nan)
    return out


@pytest.fixture(scope="module")
def matches() -> pd.DataFrame:
    return make_matches()


@pytest.fixture(scope="module")
def built(matches: pd.DataFrame) -> pd.DataFrame:
    return build_features(matches)


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
# Layout and policy
# --------------------------------------------------------------------------- #
def test_inputs_never_include_targets_or_odds() -> None:
    columns = feature_columns()
    assert len(columns) == len(set(columns))
    assert not [c for c in columns if c.startswith("target_")]
    odds = set(ODDS_COLUMNS.values())
    assert not [
        c
        for c in columns
        if c in odds or c.removeprefix("home_").removeprefix("away_") in odds
    ]


def test_output_layout_and_targets(matches: pd.DataFrame, built: pd.DataFrame) -> None:
    assert len(built) == len(matches)
    assert built["match_id"].is_unique
    assert list(built.columns[-len(TARGET_COLUMNS) :]) == list(TARGET_COLUMNS)
    assert built["date"].is_monotonic_increasing
    joined = built.merge(matches, on="match_id", suffixes=("", "_raw"))
    assert (joined["target_home_goals"] == joined["home_goals"]).all()
    expected = np.where(
        joined["home_goals"] > joined["away_goals"],
        "H",
        np.where(joined["home_goals"] == joined["away_goals"], "D", "A"),
    )
    assert (joined["target_result"] == expected).all()
    assert not np.isinf(built[feature_columns()].to_numpy(dtype="float64")).any()


def test_team_matches_mirror_home_and_away(matches: pd.DataFrame) -> None:
    long = to_team_matches(matches)
    assert len(long) == 2 * len(matches)
    first = matches.iloc[0]
    home = long[
        (long["match_id"] == first["match_id"]) & (long["side"] == "home")
    ].iloc[0]
    away = long[
        (long["match_id"] == first["match_id"]) & (long["side"] == "away")
    ].iloc[0]
    assert home["goals_for"] == away["goals_against"] == first["home_goals"]
    assert home["goals_against"] == away["goals_for"] == first["away_goals"]
    assert home["points"] + away["points"] in (2.0, 3.0)


# --------------------------------------------------------------------------- #
# Form features against a brute-force reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("metric", "half_life"),
    [
        ("xg_for", 45),
        ("xg_against", 180),
        ("goals_for", 45),
        ("shots_for", 180),
        ("ppda_for", 45),
        ("points", 180),
    ],
)
def test_form_equals_the_weighted_mean_of_strictly_earlier_matches(
    matches: pd.DataFrame, metric: str, half_life: int
) -> None:
    long = add_form_features(to_team_matches(matches))
    for _, team in long.groupby("team_id"):
        expected = reference_ewma(team[metric], team["date"], half_life)
        np.testing.assert_allclose(
            team[f"{metric}_ew{half_life}"].to_numpy(),
            expected,
            equal_nan=True,
            atol=1e-9,
        )


def test_first_match_has_no_form_and_second_sees_only_the_first(
    matches: pd.DataFrame,
) -> None:
    long = add_form_features(to_team_matches(matches))
    team = long[long["team_id"] == "e0-00"].reset_index(drop=True)
    assert team.loc[0, ["goals_for_ew45", "points_ew180"]].isna().all()
    assert team.loc[1, "goals_for_ew45"] == pytest.approx(team.loc[0, "goals_for"])
    assert team.loc[1, "points_ew180"] == pytest.approx(team.loc[0, "points"])


def test_finishing_and_keeping_are_goal_minus_xg_form(matches: pd.DataFrame) -> None:
    long = add_form_features(to_team_matches(matches))
    pd.testing.assert_series_equal(
        long[f"finishing_ew{LONG}"],
        long[f"goals_for_ew{LONG}"] - long[f"xg_for_ew{LONG}"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        long[f"keeping_ew{LONG}"],
        long[f"goals_against_ew{LONG}"] - long[f"xg_against_ew{LONG}"],
        check_names=False,
    )


def test_venue_form_uses_only_matches_at_the_same_venue(matches: pd.DataFrame) -> None:
    long = add_venue_features(to_team_matches(matches))
    column = f"venue_gd_ew{features.VENUE_HALF_LIFE_DAYS}"
    for (_, _), group in long.groupby(["team_id", "side"]):
        goal_difference = group["goals_for"] - group["goals_against"]
        expected = reference_ewma(
            goal_difference, group["date"], features.VENUE_HALF_LIFE_DAYS
        )
        np.testing.assert_allclose(
            group[column].to_numpy(), expected, equal_nan=True, atol=1e-9
        )


def test_schedule_features(matches: pd.DataFrame) -> None:
    long = add_schedule_features(to_team_matches(matches))
    team = long[long["team_id"] == "e0-00"].reset_index(drop=True)
    assert pd.isna(team.loc[0, "rest_days"])
    assert team.loc[1, "rest_days"] == (team.loc[1, "date"] - team.loc[0, "date"]).days
    assert (
        team["rest_days"].max() <= features.REST_CAP_DAYS
    )  # the summer break is capped
    assert list(team["prior_matches"][:3]) == [0, 1, 2]
    first_of_next_season = team[team["season"] == 2021].iloc[0]
    assert first_of_next_season["season_matches"] == 0
    assert first_of_next_season["prior_matches"] == 14  # 14 matches in the 2020 season


# --------------------------------------------------------------------------- #
# League context and derived features
# --------------------------------------------------------------------------- #
def test_league_context_never_sees_its_own_match_day(matches: pd.DataFrame) -> None:
    day = matches["date"].sort_values().iloc[len(matches) // 2]
    changed = matches.copy()
    changed.loc[changed["date"] == day, ["home_goals", "away_goals"]] = 9
    before, after = league_context(matches), league_context(changed)
    pd.testing.assert_frame_equal(
        before[before["date"] <= day], after[after["date"] <= day]
    )
    later = before["date"] > day
    assert not np.allclose(
        before.loc[later, "league_avg_goals_home"],
        after.loc[later, "league_avg_goals_home"],
    )


def test_league_context_is_missing_on_the_first_day_and_non_negative_after(
    matches: pd.DataFrame,
) -> None:
    context = league_context(matches)
    assert context.iloc[0][["league_avg_goals_home", "league_avg_xg"]].isna().all()
    assert (context["league_avg_goals_home"].dropna() >= 0).all()


def test_interaction_features_follow_their_formula(built: pd.DataFrame) -> None:
    expected = (
        built[f"home_xg_for_ew{LONG}"]
        * built[f"away_xg_against_ew{LONG}"]
        / built["league_avg_xg"]
    )
    pd.testing.assert_series_equal(built["home_xg_exp"], expected, check_names=False)
    assert built["rest_diff"].equals(built["home_rest_days"] - built["away_rest_days"])


def test_elo_columns_come_from_the_elo_module(
    matches: pd.DataFrame, built: pd.DataFrame
) -> None:
    expected = (
        compute_elo(matches)
        .set_index("match_id")
        .loc[built["match_id"], "home_elo"]
        .to_numpy()
    )
    np.testing.assert_allclose(built["home_elo"].to_numpy(), expected)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_missing_columns_and_duplicate_ids_are_rejected(matches: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_features(matches.drop(columns=["home_goals"]))
    with pytest.raises(ValueError, match="unique"):
        build_features(pd.concat([matches, matches.iloc[[0]]], ignore_index=True))


def test_a_team_playing_twice_on_one_date_is_rejected(matches: pd.DataFrame) -> None:
    broken = matches.copy()
    broken.loc[1, "date"] = broken.loc[0, "date"]
    broken.loc[1, "home_team_id"] = broken.loc[0, "home_team_id"]
    with pytest.raises(ValueError, match="twice on one date"):
        build_features(broken)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_features_of_earlier_matches_do_not_depend_on_later_outcomes(
    matches: pd.DataFrame, built: pd.DataFrame
) -> None:
    cutoff = matches["date"].sort_values().iloc[int(len(matches) * 0.6)]
    tampered = matches.copy()
    future = tampered["date"] >= cutoff
    tampered.loc[future, ["home_goals", "away_goals"]] = 7
    tampered.loc[future, ["home_xg", "away_xg", "home_deep", "away_deep"]] = 9.9
    tampered.loc[future, ["home_shots", "away_shots", "home_ppda", "away_ppda"]] = 40
    rebuilt = build_features(tampered)
    early = (built["date"] <= cutoff).to_numpy()
    for column in feature_columns():
        np.testing.assert_allclose(
            built.loc[early, column].to_numpy(dtype="float64"),
            rebuilt.loc[early, column].to_numpy(dtype="float64"),
            equal_nan=True,
            err_msg=column,
        )


def test_features_do_move_when_an_earlier_match_changes(
    matches: pd.DataFrame, built: pd.DataFrame
) -> None:
    changed = matches.copy()
    changed.loc[changed.index[3], ["home_goals", "away_goals"]] = [9, 0]
    rebuilt = build_features(changed)
    assert not np.allclose(built["home_elo"], rebuilt["home_elo"]) or not np.allclose(
        built["away_elo"], rebuilt["away_elo"]
    )


def test_leakage_audit_passes_on_the_real_pipeline(matches: pd.DataFrame) -> None:
    assert leakage_audit(matches) == []


def test_leakage_audit_catches_a_forgotten_shift(
    matches: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unshifted(
        values: pd.Series, dates: pd.Series, half_life_days: float
    ) -> pd.Series:
        return values.ewm(
            halflife=pd.Timedelta(half_life_days, unit="D"),
            times=dates.loc[values.index],
        ).mean()

    monkeypatch.setattr(features, "_ewma_before", unshifted)
    leaking = leakage_audit(matches)
    assert "home_goals_for_ew45" in leaking
    assert f"away_xg_for_ew{HALF_LIVES_DAYS[-1]}" in leaking


def test_leakage_audit_catches_an_elo_that_peeks_at_the_result(
    matches: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def peeking(data: pd.DataFrame, config: object = None) -> pd.DataFrame:
        elo = compute_elo(data, config)  # type: ignore[arg-type]
        margin = (data["home_goals"] - data["away_goals"]).to_numpy()
        elo = elo.set_index("match_id").loc[data["match_id"]].reset_index()
        elo["home_elo"] = elo["home_elo"] + margin
        return elo

    monkeypatch.setattr(features, "compute_elo", peeking)
    assert "home_elo" in leakage_audit(matches)


def test_two_leagues_do_not_contaminate_each_other() -> None:
    both = make_matches(leagues=("E0", "SP1"))
    alone = make_matches(leagues=("E0",))
    # make_matches draws random numbers league by league, so E0 is identical in both tables
    combined = build_features(both)
    single = build_features(alone)
    e0 = combined[combined["league"] == "E0"].reset_index(drop=True)
    for column in (
        "home_elo",
        "home_xg_for_ew45",
        "league_avg_goals_home",
        "home_rest_days",
    ):
        np.testing.assert_allclose(
            e0[column].to_numpy(dtype="float64"),
            single[column].to_numpy(dtype="float64"),
            equal_nan=True,
        )


# --------------------------------------------------------------------------- #
# Coverage and command line
# --------------------------------------------------------------------------- #
def test_feature_coverage_reports_availability_per_season(built: pd.DataFrame) -> None:
    coverage = feature_coverage(built)
    assert list(coverage.index) == [2020, 2021, 2022]
    assert coverage["elo"].eq(100.0).all()  # Elo exists from the very first match
    assert (
        coverage.loc[2020, "xg_form"] < coverage.loc[2022, "xg_form"]
    )  # history builds up


def test_every_team_feature_is_emitted_for_both_sides(built: pd.DataFrame) -> None:
    for name in TEAM_FEATURES:
        assert f"home_{name}" in built.columns and f"away_{name}" in built.columns


@pytest.mark.usefixtures("isolated_logging")
def test_cli_writes_features_after_a_clean_audit(
    tmp_path: Path, matches: pd.DataFrame
) -> None:
    matches.to_parquet(tmp_path / "enriched.parquet", index=False)
    args = [
        "--input",
        str(tmp_path / "enriched.parquet"),
        "--output",
        str(tmp_path / "out" / "features.parquet"),
    ]
    assert main(args) == 0
    written = pd.read_parquet(tmp_path / "out" / "features.parquet")
    assert len(written) == len(matches)
    assert set(feature_columns()) <= set(written.columns)


@pytest.mark.usefixtures("isolated_logging")
def test_cli_refuses_to_write_when_the_audit_finds_leakage(
    tmp_path: Path, matches: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(features, "leakage_audit", lambda _matches: ["home_elo"])
    matches.to_parquet(tmp_path / "enriched.parquet", index=False)
    args = [
        "--input",
        str(tmp_path / "enriched.parquet"),
        "--output",
        str(tmp_path / "out" / "features.parquet"),
    ]
    assert main(args) == 1
    assert not (tmp_path / "out" / "features.parquet").exists()
    assert main([*args, "--no-audit"]) == 0  # the flag skips the check


@pytest.mark.usefixtures("isolated_logging")
def test_cli_reports_missing_and_invalid_input(
    tmp_path: Path, matches: pd.DataFrame
) -> None:
    out = ["--output", str(tmp_path / "out.parquet")]
    assert main(["--input", str(tmp_path / "missing.parquet"), *out]) == 1
    matches.drop(columns=["home_goals"]).to_parquet(
        tmp_path / "bad.parquet", index=False
    )
    assert main(["--input", str(tmp_path / "bad.parquet"), *out]) == 1
