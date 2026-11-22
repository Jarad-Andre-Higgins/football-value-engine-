"""Tests for upcoming matches: fixtures, prices and pre-match features."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src import coupon, logging_config
from src.bundle import ModelBundle, build_bundle, save_bundle
from src.entities import Team, TeamRegistry
from src.evaluate import Calibration
from src.features import (
    META_COLUMNS,
    RESULT_COLUMNS,
    TARGET_COLUMNS,
    build_features,
    feature_columns,
)
from src.model import ModelParams, fit_boosters, training_mask
from src.upcoming import (
    BOOK_LABELS,
    Upcoming,
    build_upcoming,
    compare_with_market,
    download_fixtures,
    drop_started,
    format_comparison,
    format_schedule,
    history_notes,
    kickoff_label,
    kickoff_moment,
    looks_like_fixtures_csv,
    main,
    odds_table,
    read_fixtures,
    resolve_fixtures,
    schedule_notes,
    select_window,
    upcoming_features,
)
from tests.test_features import make_matches

FAST = ModelParams(n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=1)
LONDON = ZoneInfo("Europe/London")
HEADER = (
    "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,PSH,PSD,PSA,MaxH,MaxD,MaxA,"
    "AvgH,AvgD,AvgA,B365>2.5,B365<2.5,P>2.5,P<2.5,Max>2.5,Max<2.5,Avg>2.5,Avg<2.5"
)
TIMES = ("12:30", "15:00", "15:00", "17:30", "20:00")


def club(team_id: str) -> str:
    """The name a data source would use for a simulated team."""
    return f"Club {team_id.split('-')[1]}"


def douala_clock(day: pd.Timestamp, clock: str) -> str:
    """The Douala wall clock of a UK kick-off, worked out with the standard library only."""
    uk = datetime.combine(day.date(), time.fromisoformat(clock), tzinfo=LONDON)
    return f"{uk.astimezone(ZoneInfo('Africa/Douala')):%H:%M}"


def quotes(rng: np.random.Generator) -> list[float]:
    """Twenty prices in the order of HEADER: Bet365, Pinnacle, best, average; 1X2 then O/U."""
    one = rng.dirichlet([5.0, 3.0, 3.0])
    two = rng.dirichlet([6.0, 5.0])

    def price(p: np.ndarray, margin: float, noise: float) -> list[float]:
        shaded = p * np.exp(rng.normal(0.0, noise, len(p)))
        return [round(float(x), 2) for x in 1.0 / (shaded * (1.0 + margin))]

    return [
        *price(one, 0.06, 0.04),
        *price(one, 0.025, 0.0),
        *price(one, 0.0, 0.03),
        *price(one, 0.05, 0.01),
        *price(two, 0.06, 0.04),
        *price(two, 0.025, 0.0),
        *price(two, 0.0, 0.03),
        *price(two, 0.05, 0.01),
    ]


def row(
    division: str, day: str, clock: str, home: str, away: str, prices: Sequence[float]
) -> str:
    return ",".join([division, day, clock, home, away, *(str(p) for p in prices)])


def write_fixtures(
    path: Path,
    matches: pd.DataFrame,
    *,
    seed: int = 0,
    extra: Sequence[str] = (),
) -> Path:
    """A football-data style fixtures file for ``matches`` (their results are not written)."""
    rng = np.random.default_rng(seed)
    lines = [HEADER]
    for position, match in enumerate(matches.itertuples(index=False)):
        lines.append(
            row(
                match.league,
                f"{match.date:%d/%m/%Y}",
                TIMES[position % len(TIMES)],
                club(match.home_team_id),
                club(match.away_team_id),
                quotes(rng),
            )
        )
    lines.extend(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def registry_for(matches: pd.DataFrame) -> TeamRegistry:
    ids = sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    return TeamRegistry([Team(team, club(team), "England") for team in ids])


def write_registry(path: Path, matches: pd.DataFrame) -> Path:
    ids = sorted(set(matches["home_team_id"]) | set(matches["away_team_id"]))
    lines = ["teams:"] + [
        f"  - {{id: {team}, name: {club(team)}, country: England}}" for team in ids
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class World:
    """A simulated league with one matchday still to play."""

    full: pd.DataFrame  # every match, results included
    history: pd.DataFrame  # the matches before the window
    window: pd.DataFrame  # the matches on the first day of the window
    day: pd.Timestamp
    registry: TeamRegistry
    bundle: ModelBundle
    model_path: Path


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    full = make_matches(n_teams=16, seasons=7, seed=3)
    dates = sorted(full.loc[full["season"] == 2026, "date"].unique())
    day = pd.Timestamp(dates[10])
    history = full[full["date"] < day].reset_index(drop=True)
    window = full[full["date"] == day].reset_index(drop=True)
    features = build_features(history)
    mask = training_mask(features)
    last = features.loc[mask, "date"].max()
    boosters = fit_boosters(
        features, mask, FAST, as_of=last + pd.Timedelta(1, unit="D")
    )
    bundle = build_bundle(
        boosters,
        Calibration(scale_home=1.02, scale_away=0.97, rho=-0.06),
        FAST,
        trained_through=last,
        n_train_rows=int(mask.sum()),
        metrics={},
    )
    model_path = tmp_path_factory.mktemp("model") / "xg_model.pkl"
    save_bundle(bundle, model_path)
    return World(full, history, window, day, registry_for(full), bundle, model_path)


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def raw_for(
    world: World, matches: pd.DataFrame, tmp_path: Path, **kwargs: Any
) -> pd.DataFrame:
    """The fixtures of ``matches`` as read from a file."""
    table, _ = read_fixtures(
        write_fixtures(tmp_path / "fixtures.csv", matches, **kwargs)
    )
    return table


def resolved_for(world: World, matches: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    resolved, notes = resolve_fixtures(
        raw_for(world, matches, tmp_path), world.registry
    )
    assert notes == []
    return resolved


# --------------------------------------------------------------------------- #
# Reading the file
# --------------------------------------------------------------------------- #
def test_read_fixtures_gives_standard_names_and_keeps_only_the_five_leagues(
    tmp_path: Path,
) -> None:
    one = [1.4, 5.0, 8.0, 1.42, 4.9, 8.5, 1.45, 5.2, 9.0, 1.41, 4.95, 8.4]
    two = [1.3, 3.6, 1.31, 3.55, 1.33, 3.8, 1.3, 3.5]
    path = tmp_path / "fixtures.csv"
    path.write_text(
        "\n".join(
            [
                HEADER,
                row("SP1", "22/11/26", "20:00", "Club 04", "Club 05", [2.0] * 20),
                row("E0", "21/11/2026", "15:00", "Club 00", " Club 01 ", one + two),
                "," * 24,
                row("E1", "21/11/2026", "15:00", "Club 02", "Club 03", [2.0] * 20),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    table, notes = read_fixtures(path)
    assert notes == []
    assert table["league"].tolist() == ["E0", "SP1"]  # sorted by date, E1 left out
    assert table["date"].tolist() == [
        pd.Timestamp("2026-11-21"),
        pd.Timestamp("2026-11-22"),
    ]
    assert table["kickoff_time"].tolist() == ["15:00", "20:00"]
    first = table.iloc[0]
    assert first["home_team_raw"] == "Club 00"
    assert first["away_team_raw"] == "Club 01"  # whitespace stripped
    expected = {
        "b365_h": 1.4, "b365_d": 5.0, "b365_a": 8.0,
        "ps_h": 1.42, "ps_d": 4.9, "ps_a": 8.5,
        "max_h": 1.45, "max_d": 5.2, "max_a": 9.0,
        "avg_h": 1.41, "avg_d": 4.95, "avg_a": 8.4,
        "b365_over25": 1.3, "b365_under25": 3.6,
        "ps_over25": 1.31, "ps_under25": 3.55,
        "max_over25": 1.33, "max_under25": 3.8,
        "avg_over25": 1.3, "avg_under25": 3.5,
    }  # fmt: skip
    for name, value in expected.items():
        assert first[name] == value, name
    assert not any(
        column.startswith(("b365c", "psc", "avgc", "maxc")) for column in table
    )


def test_read_fixtures_leaves_closing_prices_out_and_copes_without_time_or_prices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bare.csv"
    path.write_text(
        "Div,Date,HomeTeam,AwayTeam,B365CH\nE0,21/11/2026,Club 00,Club 01,1.5\n",
        encoding="utf-8",
    )
    table, notes = read_fixtures(path)
    assert notes == [] and len(table) == 1
    assert pd.isna(table.loc[0, "kickoff_time"])
    assert list(table.columns) == [
        "league", "date", "kickoff_time", "home_team_raw", "away_team_raw",
    ]  # fmt: skip


def test_read_fixtures_reports_unreadable_dates_and_requires_the_identity_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "f.csv"
    lines = [
        "Div,Date,Time,HomeTeam,AwayTeam",
        "E0,not a date,15:00,Club 00,Club 01",
        "E0,21/11/2026,15:00,Club 02,Club 03",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table, notes = read_fixtures(path)
    assert len(table) == 1
    assert notes == ["1 fixtures ignored: unreadable date"]
    broken = tmp_path / "broken.csv"
    broken.write_text("Div,Date,HomeTeam\nE0,21/11/2026,Club 00\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing required columns \['AwayTeam'\]"):
        read_fixtures(broken)


def test_looks_like_fixtures_csv() -> None:
    assert looks_like_fixtures_csv(HEADER.encode() + b"\nE0,...")
    assert looks_like_fixtures_csv(b"\xef\xbb\xbfDiv,Date,Time,HomeTeam,AwayTeam\n")
    assert not looks_like_fixtures_csv(b"<html><body>Error 500</body></html>")
    assert not looks_like_fixtures_csv(b"")


class StubResponse:
    def __init__(self, content: bytes, status: int = 200) -> None:
        self.content = content
        self.status_code = status

    @property
    def ok(self) -> bool:
        return self.status_code < 400


class StubSession:
    def __init__(self, response: StubResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, tuple[float, float]]] = []

    def get(self, url: str, timeout: tuple[float, float]) -> StubResponse:
        self.calls.append((url, timeout))
        return self.response


def test_download_fixtures_saves_the_file_atomically(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "fixtures.csv"
    session = StubSession(StubResponse((HEADER + "\n").encode()))
    assert download_fixtures(path, "http://x/f.csv", session=session) == path  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == HEADER + "\n"
    assert not list(path.parent.glob("*.part"))
    assert session.calls == [("http://x/f.csv", (10.0, 30.0))]


def test_download_fixtures_failures_keep_the_old_file(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(OSError, match="HTTP 503"):
        download_fixtures(path, "http://x", session=StubSession(StubResponse(b"", 503)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a football-data fixtures file"):
        download_fixtures(
            path, "http://x", session=StubSession(StubResponse(b"<html>"))
        )  # type: ignore[arg-type]
    assert path.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.part"))


# --------------------------------------------------------------------------- #
# Identity: ids, seasons, windows
# --------------------------------------------------------------------------- #
def test_resolve_fixtures_gives_ids_names_season_and_the_match_tables_match_id(
    world: World, tmp_path: Path
) -> None:
    resolved = resolved_for(world, world.window, tmp_path)
    assert set(resolved["match_id"]) == set(
        world.window["match_id"]
    )  # pipeline's format
    first = resolved.iloc[0]
    assert first["home_team"] == club(first["home_team_id"])
    assert first["season"] == 2026
    assert {"home_team_id", "away_team_id", "home_team", "away_team"} <= set(resolved)


def test_a_season_starts_on_the_first_of_july(world: World, tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "league": ["E0", "E0"],
            "date": [pd.Timestamp("2027-06-30"), pd.Timestamp("2027-07-01")],
            "kickoff_time": ["15:00", "15:00"],
            "home_team_raw": ["Club 00", "Club 02"],
            "away_team_raw": ["Club 01", "Club 03"],
        }
    )
    resolved, _ = resolve_fixtures(raw, world.registry)
    assert resolved["season"].tolist() == [2026, 2027]
    assert resolved["match_id"].tolist() == [
        "E0-2026-20270630-e0-00-e0-01",
        "E0-2027-20270701-e0-02-e0-03",
    ]


def test_an_unknown_team_costs_one_fixture_not_the_coupon(
    world: World, tmp_path: Path
) -> None:
    raw = raw_for(world, world.window.head(3), tmp_path)
    raw.loc[0, "home_team_raw"] = "Clubb 0"  # a typo no alias covers
    resolved, notes = resolve_fixtures(raw, world.registry)
    assert len(resolved) == 2
    assert len(notes) == 1
    assert "1 fixtures left out" in notes[0]
    assert "E0 'Clubb 0': closest" in notes[0]
    assert "Club 0" in notes[0]  # a suggestion is offered
    assert "data/reference/teams.yaml" in notes[0]


def test_a_team_is_resolved_in_the_country_of_its_league(
    world: World, tmp_path: Path
) -> None:
    raw = raw_for(world, world.window.head(1), tmp_path).assign(league="SP1")
    resolved, notes = resolve_fixtures(raw, world.registry)  # the teams are English
    assert resolved.empty and len(notes) == 1


def test_a_fixture_listed_twice_is_kept_once(world: World, tmp_path: Path) -> None:
    raw = raw_for(world, world.window.head(2), tmp_path)
    doubled = pd.concat([raw, raw.head(1)], ignore_index=True)
    resolved, notes = resolve_fixtures(doubled, world.registry)
    assert len(resolved) == 2
    assert notes == ["1 fixtures appear twice in the file: kept once"]


def test_select_window_includes_the_first_day_and_excludes_the_day_after_the_last() -> (
    None
):
    dates = pd.to_datetime(["2026-11-20", "2026-11-21", "2026-11-22", "2026-11-23"])
    fixtures = pd.DataFrame({"date": dates, "n": range(4)})
    one = select_window(fixtures, pd.Timestamp("2026-11-21 13:00"), 1)
    assert one["n"].tolist() == [1]
    assert select_window(fixtures, pd.Timestamp("2026-11-21"), 2)["n"].tolist() == [
        1,
        2,
    ]
    assert select_window(fixtures, pd.Timestamp("2027-01-01"), 3).empty
    with pytest.raises(ValueError, match="days must be at least 1"):
        select_window(fixtures, pd.Timestamp("2026-11-21"), 0)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def assert_same_features(got: pd.DataFrame, want: pd.DataFrame) -> None:
    columns = [*META_COLUMNS, *feature_columns()]
    pd.testing.assert_frame_equal(
        got.sort_values("match_id")[columns].reset_index(drop=True),
        want.sort_values("match_id")[columns].reset_index(drop=True),
        check_dtype=False,
        rtol=1e-9,
    )


def test_a_fixtures_features_equal_what_the_training_pipeline_gave_it_once_played(
    world: World, tmp_path: Path
) -> None:
    fixtures = resolved_for(world, world.window, tmp_path)
    got = upcoming_features(world.history, fixtures)
    full = build_features(
        world.full
    )  # here the matches are played, and later ones exist
    want = full[full["match_id"].isin(set(world.window["match_id"]))]
    assert len(got) == len(world.window) > 3
    assert_same_features(got, want)


def test_the_features_of_a_longer_window_never_see_the_windows_own_results(
    world: World, tmp_path: Path
) -> None:
    ahead = world.full[
        (world.full["date"] >= world.day)
        & (world.full["date"] < world.day + pd.Timedelta(3, unit="D"))
    ]
    assert ahead["date"].nunique() >= 2
    got = upcoming_features(world.history, resolved_for(world, ahead, tmp_path))
    hidden = world.full.copy()
    columns = list(RESULT_COLUMNS)
    hidden[columns] = hidden[columns].astype("float64")
    hidden.loc[hidden["match_id"].isin(set(ahead["match_id"])), columns] = np.nan
    want = build_features(hidden)
    assert_same_features(got, want[want["match_id"].isin(set(ahead["match_id"]))])


def test_a_fixtures_targets_are_empty(world: World, tmp_path: Path) -> None:
    got = upcoming_features(world.history, resolved_for(world, world.window, tmp_path))
    assert list(got.columns) == [*META_COLUMNS, *feature_columns(), *TARGET_COLUMNS]
    assert got[list(TARGET_COLUMNS)].isna().all().all()
    assert got[feature_columns()].notna().any(axis=1).all()


def test_upcoming_features_refuse_a_match_that_is_already_in_the_history(
    world: World, tmp_path: Path
) -> None:
    fixtures = resolved_for(world, world.window, tmp_path)
    played = pd.concat([world.history, world.window], ignore_index=True)
    with pytest.raises(ValueError, match="already in the history"):
        upcoming_features(played, fixtures)


def test_upcoming_features_name_a_missing_column(world: World, tmp_path: Path) -> None:
    fixtures = resolved_for(world, world.window, tmp_path).drop(columns=["season"])
    with pytest.raises(ValueError, match=r"missing columns: \['season'\]"):
        upcoming_features(world.history, fixtures)


def test_upcoming_features_work_when_the_history_lacks_optional_columns(
    world: World, tmp_path: Path
) -> None:
    thin = world.history.drop(columns=["home_deep", "away_deep"])
    got = upcoming_features(thin, resolved_for(world, world.window, tmp_path))
    assert len(got) == len(world.window)
    assert got["home_deep_for_ew180"].isna().all()
    assert got["away_deep_against_ew45"].isna().all()
    assert (
        got["home_goals_for_ew180"].notna().all()
    )  # what the history does have is used


# --------------------------------------------------------------------------- #
# Notes
# --------------------------------------------------------------------------- #
def build(
    world: World,
    tmp_path: Path,
    *,
    history: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
    **kwargs: Any,
) -> Upcoming:
    raw = raw_for(world, world.window if matches is None else matches, tmp_path)
    return build_upcoming(
        world.history if history is None else history,
        raw,
        world.registry,
        start=kwargs.pop("start", world.day),
        **kwargs,
    )


def test_build_upcoming_returns_fixtures_features_and_prices(
    world: World, tmp_path: Path
) -> None:
    result = build(world, tmp_path)
    n = len(world.window)
    assert len(result.fixtures) == len(result.features) == n
    assert set(result.fixtures["match_id"]) == set(result.features["match_id"])
    assert result.notes == ()
    assert len(result.odds) == n * len(BOOK_LABELS) * 5  # 3 + 2 selections each


def test_a_stale_match_table_is_reported(world: World, tmp_path: Path) -> None:
    old = world.history[world.history["date"] < world.day - pd.Timedelta(30, unit="D")]
    notes = build(world, tmp_path, history=old).notes
    assert any(
        "The match table ends on" in note and "days before" in note for note in notes
    )
    assert not any("ends on" in note for note in build(world, tmp_path).notes)


def test_missing_recent_xg_is_reported(world: World, tmp_path: Path) -> None:
    history = world.history.copy()
    recent = history["date"] >= history["date"].max() - pd.Timedelta(30, unit="D")
    history.loc[recent, ["home_xg", "away_xg"]] = np.nan
    notes = build(world, tmp_path, history=history).notes
    assert any(
        note.startswith("Only 0% of the") and "have xG" in note for note in notes
    )


def test_a_team_with_little_history_is_reported(world: World, tmp_path: Path) -> None:
    team = world.window.iloc[0]["home_team_id"]
    mine = (world.history["home_team_id"] == team) | (
        world.history["away_team_id"] == team
    )
    history = pd.concat(
        [world.history[~mine], world.history[mine].head(2)], ignore_index=True
    )
    notes = build(world, tmp_path, history=history).notes
    assert any(f"fewer than 5 matches in the table: {team}" in note for note in notes)


def test_a_team_playing_twice_is_reported(world: World, tmp_path: Path) -> None:
    ahead = world.full[
        (world.full["date"] >= world.day)
        & (world.full["date"] < world.day + pd.Timedelta(9, unit="D"))
    ]
    result = build(world, tmp_path, matches=ahead, days=9)
    assert any("play more than once in the window" in note for note in result.notes)
    assert len(result.features) == len(ahead)


def test_schedule_notes_name_every_repeated_team() -> None:
    fixtures = pd.DataFrame(
        {"home_team_id": ["a", "c", "a"], "away_team_id": ["b", "b", "d"]}
    )
    assert schedule_notes(fixtures) == [
        (
            "a, b play more than once in the window: the later fixture is built as if "
            "the earlier one had not been played"
        )
    ]
    assert schedule_notes(fixtures.head(1)) == []


def test_history_notes_are_empty_for_an_empty_table(world: World) -> None:
    assert history_notes(world.history.head(0), world.window, world.day) == [
        "The match table has no matches"
    ]


def test_matches_already_in_the_table_are_left_out_with_a_note(
    world: World, tmp_path: Path
) -> None:
    one = world.window.iloc[:1]
    history = pd.concat([world.history, one], ignore_index=True)
    result = build(world, tmp_path, history=history)
    assert len(result.fixtures) == len(world.window) - 1
    assert one.iloc[0]["match_id"] not in set(result.features["match_id"])
    assert any("1 fixtures are already in the match table" in n for n in result.notes)
    everything = pd.concat([world.history, world.window], ignore_index=True)
    with pytest.raises(ValueError, match="already in the match table"):
        build(world, tmp_path, history=everything)


def test_scoreless_rows_of_the_history_are_ignored(
    world: World, tmp_path: Path
) -> None:
    history = world.history.copy()
    history[list(RESULT_COLUMNS)] = history[list(RESULT_COLUMNS)].astype("float64")
    history.loc[history.index[-3:], ["home_goals", "away_goals"]] = np.nan
    notes = build(world, tmp_path, history=history).notes
    assert "3 matches without a score were ignored" in notes


def test_a_fixture_without_complete_prices_is_reported_and_has_no_odds(
    world: World, tmp_path: Path
) -> None:
    raw = raw_for(world, world.window, tmp_path)
    prices = [
        column
        for column in raw
        if column.split("_", 1)[0] in {"b365", "ps", "avg", "max"}
    ]
    raw.loc[0, prices] = np.nan
    result = build_upcoming(world.history, raw, world.registry, start=world.day)
    assert f"No complete prices for 1 of {len(world.window)} fixtures" in result.notes
    assert raw.loc[0, "home_team_raw"] not in set(result.odds["match_id"])
    assert len(result.fixtures) == len(world.window)  # still a fixture, just unpriced


def test_no_fixtures_in_the_window_says_what_the_file_covers(
    world: World, tmp_path: Path
) -> None:
    far = world.day + pd.Timedelta(400, unit="D")
    with pytest.raises(
        ValueError, match=r"no fixtures on .*; it covers 2026-.* to 2026-"
    ):
        build(world, tmp_path, start=far)
    empty = raw_for(world, world.window, tmp_path).head(0)
    with pytest.raises(ValueError, match="no fixtures of the top five leagues"):
        build_upcoming(world.history, empty, world.registry, start=world.day)


def test_a_window_whose_teams_are_all_unknown_explains_why(
    world: World, tmp_path: Path
) -> None:
    raw = raw_for(world, world.window, tmp_path).assign(home_team_raw="Nobody FC")
    with pytest.raises(ValueError, match="left out, team name not in the registry"):
        build_upcoming(world.history, raw, world.registry, start=world.day)


# --------------------------------------------------------------------------- #
# Matches that have started
# --------------------------------------------------------------------------- #
def test_kickoff_moment_is_in_uk_time_summer_and_winter() -> None:
    winter = kickoff_moment(pd.Timestamp("2026-11-21"), "15:00")
    summer = kickoff_moment(pd.Timestamp("2026-07-18"), "15:00")
    assert winter is not None and summer is not None
    assert winter.astimezone(UTC).hour == 15
    assert summer.astimezone(UTC).hour == 14  # British summer time
    assert kickoff_moment(pd.Timestamp("2026-11-21"), pd.NA) is None
    assert kickoff_moment(pd.Timestamp("2026-11-21"), "TBC") is None
    assert kickoff_moment(pd.Timestamp("2026-11-21"), "") is None
    assert kickoff_moment(pd.Timestamp("2026-11-21"), None) is None


def test_drop_started_keeps_what_has_not_kicked_off_and_what_has_no_time() -> None:
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-11-21"] * 5 + ["2026-11-22"]),
            "kickoff_time": ["12:30", "15:00", "17:30", "TBC", None, "12:30"],
            "n": range(6),
        }
    )
    now = datetime(2026, 11, 21, 15, 0, tzinfo=LONDON)  # exactly at the 15:00 kick-off
    left, started = drop_started(fixtures, now)
    assert started == 2  # 12:30 and 15:00; a match at this very minute has started
    assert left["n"].tolist() == [2, 3, 4, 5]
    assert drop_started(fixtures, datetime(2026, 11, 21, 12, 29, tzinfo=LONDON))[1] == 0
    again, none = drop_started(fixtures.head(0), now)
    assert none == 0 and again.empty


def test_started_matches_are_left_out_and_said_so(world: World, tmp_path: Path) -> None:
    clock = datetime(
        world.day.year, world.day.month, world.day.day, 16, 0, tzinfo=LONDON
    )
    times = [TIMES[i % len(TIMES)] for i in range(len(world.window))]
    expected_left = sum(t in {"17:30", "20:00"} for t in times)
    result = build(world, tmp_path, now=clock)
    assert len(result.fixtures) == expected_left
    assert set(result.fixtures["kickoff_time"]) <= {"17:30", "20:00"}
    assert (
        f"{len(world.window) - expected_left} fixtures have already kicked off: left out"
        in (result.notes)
    )
    assert len(build(world, tmp_path).fixtures) == len(
        world.window
    )  # no clock, no filter


def test_when_everything_has_started_the_error_says_so(
    world: World, tmp_path: Path
) -> None:
    late = datetime(
        world.day.year, world.day.month, world.day.day, 23, 0, tzinfo=LONDON
    )
    with pytest.raises(
        ValueError, match=r"all \d+ fixtures of the window have already kicked off"
    ):
        build(world, tmp_path, now=late)


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def test_odds_table_names_the_bookmakers_and_covers_every_selection(
    world: World, tmp_path: Path
) -> None:
    fixtures = resolved_for(world, world.window, tmp_path)
    odds = odds_table(fixtures)
    assert list(odds.columns) == [
        "match_id",
        "bookmaker",
        "market_id",
        "selection",
        "odds",
    ]
    assert set(odds["bookmaker"]) == set(BOOK_LABELS.values())
    one = odds[
        (odds["match_id"] == fixtures.loc[0, "match_id"])
        & (odds["bookmaker"] == "bet365")
    ]
    assert sorted(zip(one["market_id"], one["selection"])) == [
        ("1X2", "away"), ("1X2", "draw"), ("1X2", "home"), ("OU:2.5", "over"), ("OU:2.5", "under"),
    ]  # fmt: skip
    pinnacle = odds[odds["bookmaker"] == "pinnacle"].set_index(
        ["match_id", "market_id", "selection"]
    )
    home = fixtures.loc[0, "ps_h"]
    assert pinnacle.loc[(fixtures.loc[0, "match_id"], "1X2", "home"), "odds"] == home


def test_a_market_missing_one_selection_is_dropped_whole(
    world: World, tmp_path: Path
) -> None:
    fixtures = resolved_for(world, world.window, tmp_path)
    fixtures.loc[0, "b365_d"] = np.nan  # a quote that cannot be de-margined
    odds = odds_table(fixtures)
    mine = odds[
        (odds["match_id"] == fixtures.loc[0, "match_id"])
        & (odds["bookmaker"] == "bet365")
    ]
    assert set(mine["market_id"]) == {
        "OU:2.5"
    }  # the 1X2 quote is gone, over/under stays


def test_the_odds_table_is_what_the_coupon_reads(world: World, tmp_path: Path) -> None:
    fixtures = resolved_for(world, world.window, tmp_path)
    odds = odds_table(fixtures)
    cleaned, notes = coupon.clean_odds(odds, fixtures["match_id"])
    assert notes == []  # nothing the coupon would have to throw away
    assert len(cleaned) == len(odds)


# --------------------------------------------------------------------------- #
# The model against the market
# --------------------------------------------------------------------------- #
def test_compare_with_market_lines_up_model_market_and_bet365(
    world: World, tmp_path: Path
) -> None:
    result = build(world, tmp_path)
    table = compare_with_market(result, world.bundle)
    n = len(world.window)
    assert len(table) == n * 5
    assert table["match_id"].tolist()[:5] == [result.fixtures.loc[0, "match_id"]] * 5
    first = table.head(5)
    assert first["market_id"].tolist() == ["1X2"] * 3 + ["OU:2.5"] * 2
    assert first["selection"].tolist() == ["home", "draw", "away", "over", "under"]
    sums = table.groupby(["match_id", "market_id"])["p_model"].sum()
    assert np.allclose(sums, 1.0, atol=1e-6)
    # the market's probability is Pinnacle's 1/odds scaled to sum to one
    fixture = result.fixtures.iloc[0]
    inverse = 1.0 / np.array([fixture["ps_h"], fixture["ps_d"], fixture["ps_a"]])
    assert np.allclose(first["p_market"].to_numpy()[:3], inverse / inverse.sum())
    assert set(first["source"]) == {"ps"}
    bet365 = np.array([fixture["b365_h"], fixture["b365_d"], fixture["b365_a"]])
    assert np.allclose(first["odds"].to_numpy()[:3], bet365)
    assert np.allclose(first["price_gap"], first["odds"] * first["p_market"] - 1.0)
    assert np.allclose(first["model_edge"], first["odds"] * first["p_model"] - 1.0)
    assert first["lam_home"].nunique() == 1 and (first["lam_home"] > 0).all()


def test_the_market_view_falls_back_to_the_next_sharpest_book(
    world: World, tmp_path: Path
) -> None:
    raw = raw_for(world, world.window, tmp_path)
    no_pinnacle = raw.drop(columns=[c for c in raw if c.startswith("ps_")])
    result = build_upcoming(world.history, no_pinnacle, world.registry, start=world.day)
    table = compare_with_market(result, world.bundle)
    assert set(table["source"]) == {"avg"}
    assert table["p_market"].notna().all()


def test_a_fixture_without_prices_still_shows_the_model(
    world: World, tmp_path: Path
) -> None:
    raw = raw_for(world, world.window, tmp_path)
    prices = [c for c in raw if c.split("_", 1)[0] in {"b365", "ps", "avg", "max"}]
    raw[prices] = np.nan
    result = build_upcoming(world.history, raw, world.registry, start=world.day)
    table = compare_with_market(result, world.bundle)
    assert table["p_model"].notna().all() and table["p_market"].isna().all()
    text = format_comparison(result, table)
    assert "market -" in text
    assert "Where the model differs most" not in text  # nothing to compare with


def test_format_comparison_reads_like_a_briefing(world: World, tmp_path: Path) -> None:
    result = build(world, tmp_path)
    text = format_comparison(
        result, compare_with_market(result, world.bundle), "Africa/Douala"
    )
    first = result.fixtures.iloc[0]
    assert f"Premier League · {first['home_team']} v {first['away_team']} · " in text
    assert f"12:30 UK ({douala_clock(first['date'], '12:30')} Douala)" in text
    assert "model goal rates" in text
    assert text.count("1X2") >= len(world.window)
    assert "O/U 2.5" in text and "Bet365" in text
    assert "Where the model differs most from the market (percentage points):" in text
    assert "Bet365 prices furthest above the market" in text
    top = text.split("Where the model differs most")[1].split("Bet365 prices furthest")[
        0
    ]
    lines = [line for line in top.splitlines() if line.startswith("   ")]
    assert len(lines) == 5
    # the two sides of one over/under market are one disagreement: listed once
    totals = [
        line.rsplit("(", 1)[1].split(")")[0] for line in lines if "2.5 goals" in line
    ]
    assert len(totals) == len(set(totals))
    gaps = [abs(float(line.split()[0])) for line in lines]
    assert gaps == sorted(gaps, reverse=True)


@pytest.mark.parametrize(
    ("day", "clock", "zone", "expected"),
    [
        ("2026-11-21", "15:00", "Africa/Douala", "15:00 UK (16:00 Douala)"),
        ("2026-07-18", "15:00", "Africa/Douala", "15:00 UK (15:00 Douala)"),
        ("2026-11-21", "15:00", "America/New_York", "15:00 UK (10:00 New York)"),
        ("2026-11-21", "15:00", None, "15:00 UK"),
        ("2026-11-21", "15:00", "Nowhere/Land", "15:00 UK"),
        ("2026-11-21", pd.NA, "Africa/Douala", "time not given"),
        ("2026-11-21", " ", "Africa/Douala", "time not given"),
        ("2026-11-21", "TBC", "Africa/Douala", "TBC"),
    ],
)
def test_kickoff_label(
    day: str, clock: object, zone: str | None, expected: str
) -> None:
    assert kickoff_label(pd.Timestamp(day), clock, zone) == expected


def test_format_schedule_lists_every_fixture_with_its_bookmakers(
    world: World, tmp_path: Path
) -> None:
    result = build(world, tmp_path)
    text = format_schedule(result.fixtures, result.odds, "Africa/Douala")
    lines = text.splitlines()
    assert len(lines) == len(world.window)
    assert lines[0].startswith("E0   ")
    assert (
        f"12:30 UK ({douala_clock(result.fixtures.loc[0, 'date'], '12:30')} Douala)"
        in lines[0]
    )
    assert "[prices: bet365 market_avg market_best pinnacle]" in lines[0]
    unpriced = format_schedule(result.fixtures, result.odds.head(0), None)
    assert "[prices: none]" in unpriced


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
def make_inputs(world: World, tmp_path: Path) -> dict[str, Path]:
    fixtures = write_fixtures(tmp_path / "fixtures.csv", world.window)
    history = tmp_path / "history.parquet"
    world.history.to_parquet(history, index=False)
    return {
        "fixtures": fixtures,
        "history": history,
        "registry": write_registry(tmp_path / "teams.yaml", world.full),
        "output": tmp_path / "out" / "upcoming.parquet",
        "odds": tmp_path / "out" / "upcoming_odds.csv",
    }


def cli(
    world: World, paths: dict[str, Path], *extra: str, model: bool = True
) -> list[str]:
    before = f"{world.day - pd.Timedelta(1, unit='D'):%Y-%m-%d}T12:00"
    return [
        "--fixtures", str(paths["fixtures"]),
        "--history", str(paths["history"]),
        "--registry", str(paths["registry"]),
        "--model", str(world.model_path if model else paths["history"].parent / "none.pkl"),
        "--output", str(paths["output"]),
        "--odds-output", str(paths["odds"]),
        "--date", f"{world.day:%Y-%m-%d}",
        "--now", before,
        *extra,
    ]  # fmt: skip


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_builds_both_files_and_the_coupon_can_use_them(
    world: World,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = make_inputs(world, tmp_path)
    with caplog.at_level(logging.INFO):
        assert main(cli(world, paths)) == 0
    assert "The model against the market:" in caplog.text
    assert "Schedule (kick-off in UK time, then Africa/Douala)" in caplog.text
    assert "Next: python -m src.coupon --fixtures" in caplog.text
    features = pd.read_parquet(paths["output"])
    assert len(features) == len(world.window)
    assert {"kickoff_time", *feature_columns()} <= set(features.columns)
    assert features["kickoff_time"].isin(TIMES).all()
    odds = pd.read_csv(paths["odds"])
    assert set(odds["bookmaker"]) == set(BOOK_LABELS.values())

    # Bet365 is slow to cut one price: the engine should bet it, at Bet365 only
    pool = odds[
        (odds["bookmaker"] == "bet365")
        & (odds["market_id"] == "1X2")
        & (odds["selection"] == "home")
        & odds["odds"].between(1.6, 4.0)
    ]
    stale = pool.index[0]
    stale_match = odds.loc[stale, "match_id"]
    odds.loc[stale, "odds"] = round(odds.loc[stale, "odds"] * 1.6, 2)
    odds.to_csv(paths["odds"], index=False)
    out = tmp_path / "coupons"
    args = [
        "--fixtures", str(paths["output"]),
        "--odds", str(paths["odds"]),
        "--model", str(world.model_path),
        "--blend", str(tmp_path / "blend.json"),
        "--registry", str(paths["registry"]),
        "--output-dir", str(out),
        "--weight", "0",
        "--bet-at", "bet365",
        "--reference-bookmaker", "pinnacle",
    ]  # fmt: skip
    assert coupon.main(args) == 0
    printed = capsys.readouterr().out
    assert "VALUE COUPON" in printed
    record = pd.read_csv(next(out.glob("**/bets_*.csv")))
    assert stale_match in set(record["match_id"])
    assert set(record["bookmaker"]) == {"bet365"}  # never Pinnacle, the reference
    assert (record["reference_source"] == "reference book").all()
    row_ = record[record["match_id"] == stale_match].iloc[0]
    assert row_["selection"] == "home" and row_["edge"] > 0.2


@pytest.mark.usefixtures("isolated_logging")
def test_the_next_command_names_the_sharpest_reference_there_is(
    world: World, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = make_inputs(world, tmp_path)
    with caplog.at_level(logging.INFO):
        assert main(cli(world, paths, model=False)) == 0
    assert "--bet-at bet365 --reference-bookmaker pinnacle" in caplog.text
    caplog.clear()
    table = pd.read_csv(paths["fixtures"])
    table.drop(columns=["PSH", "PSD", "PSA", "P>2.5", "P<2.5"]).to_csv(
        paths["fixtures"], index=False
    )  # a file without Pinnacle's prices
    with caplog.at_level(logging.INFO):
        assert main(cli(world, paths, model=False)) == 0
    assert "--bet-at bet365 --reference-bookmaker market_avg" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_still_writes_the_files_without_a_model(
    world: World, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = make_inputs(world, tmp_path)
    with caplog.at_level(logging.WARNING):
        assert main(cli(world, paths, model=False)) == 0
    assert "No model view: cannot load" in caplog.text
    assert paths["output"].exists() and paths["odds"].exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_logs_every_data_note(
    world: World, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = make_inputs(world, tmp_path)
    old = world.history[world.history["date"] < world.day - pd.Timedelta(30, unit="D")]
    old.to_parquet(paths["history"], index=False)
    with caplog.at_level(logging.WARNING):
        assert main(cli(world, paths, model=False)) == 0
    assert "Data note: The match table ends on" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
@pytest.mark.parametrize("missing", ["fixtures", "history", "registry"])
def test_the_command_names_a_missing_input(
    world: World, tmp_path: Path, caplog: pytest.LogCaptureFixture, missing: str
) -> None:
    paths = make_inputs(world, tmp_path)
    paths[missing].unlink()
    with caplog.at_level(logging.ERROR):
        assert main(cli(world, paths)) == 1
    assert "not found" in caplog.text and str(paths[missing]) in caplog.text
    assert ("(use --download)" in caplog.text) == (missing == "fixtures")
    assert not paths["output"].exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_fails_clearly_when_nothing_is_left(
    world: World, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = make_inputs(world, tmp_path)
    with caplog.at_level(logging.ERROR):
        late = f"{world.day:%Y-%m-%d}T23:30"
        args = cli(world, paths)
        args[args.index("--now") + 1] = late
        assert main(args) == 1
    assert "Cannot build the upcoming matches: all" in caplog.text
    assert "already kicked off" in caplog.text
    assert not paths["output"].exists()
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        assert main([*cli(world, paths), "--date", "2030-01-01"]) == 1
    assert "no fixtures on 2030-01-01" in caplog.text


@pytest.mark.usefixtures("isolated_logging")
def test_include_started_keeps_every_match(world: World, tmp_path: Path) -> None:
    paths = make_inputs(world, tmp_path)
    args = cli(world, paths, "--include-started")
    args[args.index("--now") + 1] = f"{world.day:%Y-%m-%d}T23:30"
    assert main(args) == 0
    assert len(pd.read_parquet(paths["output"])) == len(world.window)


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_reports_a_failed_download(
    world: World,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(path: Path, url: str) -> Path:
        raise OSError("HTTP 503")

    monkeypatch.setattr("src.upcoming.download_fixtures", refuse)
    paths = make_inputs(world, tmp_path)
    with caplog.at_level(logging.ERROR):
        assert main(cli(world, paths, "--download")) == 1
    assert "Cannot download the fixtures: HTTP 503" in caplog.text
    assert not paths["output"].exists()


@pytest.mark.usefixtures("isolated_logging")
def test_the_command_downloads_then_builds(
    world: World,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_inputs(world, tmp_path)
    fetched: list[tuple[Path, str]] = []

    def fake(path: Path, url: str) -> Path:
        fetched.append((path, url))
        return path

    monkeypatch.setattr("src.upcoming.download_fixtures", fake)
    with caplog.at_level(logging.INFO):
        assert (
            main(cli(world, paths, "--download", "--url", "http://example/f.csv")) == 0
        )
    assert fetched == [(paths["fixtures"], "http://example/f.csv")]
    assert f"Downloaded {paths['fixtures']}" in caplog.text


@pytest.mark.parametrize(
    "bad",
    [["--days", "0"], ["--date", "tomorrow-ish"], ["--now", "half past six"]],
)
def test_the_command_rejects_bad_options(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        main(bad)
    assert exit_.value.code == 2
