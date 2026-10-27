"""Tests for the match-table pipeline: reading, resolution, validation and CLI."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import logging_config, pipeline
from src.entities import Team, TeamRegistry
from src.pipeline import (
    BuildResult,
    Severity,
    UnresolvedNamesError,
    build_match_table,
    load_matches_table,
    main,
    odds_coverage,
    parse_file_name,
    read_raw_file,
    resolve_teams,
    validate_matches,
)
from src.schema import MATCH_COLUMNS, ODDS_COLUMNS

TEAMS = [
    Team("arsenal", "Arsenal", "England", ("Arsenal FC",)),
    Team("chelsea", "Chelsea", "England"),
    Team("liverpool", "Liverpool", "England"),
    Team("everton", "Everton", "England"),
]
ENGLISH = ("Arsenal", "Chelsea", "Liverpool", "Everton")
TODAY = pd.Timestamp("2026-10-26")


@pytest.fixture
def registry() -> TeamRegistry:
    return TeamRegistry(TEAMS)


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep CLI tests from writing to the real logs folder."""
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def season_frame(
    teams: tuple[str, ...] = ENGLISH, start: str = "2024-08-10"
) -> pd.DataFrame:
    """A deterministic double round robin in football-data's raw layout."""
    rows = []
    first = pd.Timestamp(start)
    for index, (home, away) in enumerate(permutations(teams, 2)):
        home_goals, away_goals = (index * 7) % 4, (index * 5) % 3
        result = (
            "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"
        )
        rows.append(
            {
                "Div": "E0",
                "Date": (first + pd.Timedelta(index * 3, unit="D")).strftime(
                    "%d/%m/%Y"
                ),
                "Time": "15:00",
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "FTR": result,
                "HS": 10 + index,
                "AS": 8,
                "B365H": 2.0,
                "B365D": 3.4,
                "B365A": 3.8,
                "PSCH": 1.95,
                "PSCD": 3.5,
                "PSCA": 4.1,
                "P>2.5": 1.9,
                "P<2.5": 2.0,
            }
        )
    return pd.DataFrame(rows)


def write_season(directory: Path, name: str = "E0_2425.csv", **kwargs: object) -> Path:
    path = directory / name
    season_frame(**kwargs).to_csv(path, index=False)  # type: ignore[arg-type]
    return path


def resolved(directory: Path, registry: TeamRegistry) -> pd.DataFrame:
    frame, _ = read_raw_file(write_season(directory))
    return resolve_teams(frame, registry)


def codes(issues: list[pipeline.Issue]) -> set[str]:
    return {issue.code for issue in issues}


# --------------------------------------------------------------------------- #
# File names and reading
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("E0_2526.csv", ("E0", 2025)),
        ("SP1_1920.csv", ("SP1", 2019)),
        ("F1_9900.csv", ("F1", 1999)),
    ],
)
def test_parse_file_name(name: str, expected: tuple[str, int]) -> None:
    assert parse_file_name(Path(name)) == expected


@pytest.mark.parametrize(
    "name", ["notes.csv", "E0_2527.csv", "XX_2526.csv", "E0_25.csv"]
)
def test_parse_file_name_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        parse_file_name(Path(name))


def test_read_raw_file_standardises_columns(tmp_path: Path) -> None:
    frame, found = read_raw_file(write_season(tmp_path))
    assert len(frame) == 12
    assert set(frame["league"]) == {"E0"}
    assert set(frame["season"]) == {2024}
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame["home_goals"].dtype == "int64"
    assert frame["home_shots"].dtype == "Int64"
    assert frame["home_corners"].isna().all()  # column absent in the file
    assert frame["b365_h"].dtype == "float64"
    assert frame["psc_h"].eq(1.95).all()
    assert frame["ps_over25"].eq(1.9).all()
    assert frame["avgc_h"].isna().all()
    assert found == {
        "HS",
        "AS",
        "B365H",
        "B365D",
        "B365A",
        "PSCH",
        "PSCD",
        "PSCA",
        "P>2.5",
        "P<2.5",
    }


def test_read_raw_file_drops_unplayed_and_blank_rows(tmp_path: Path) -> None:
    path = tmp_path / "E0_2425.csv"
    path.write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
        "E0,10/08/2024, Arsenal ,Chelsea,2,1\n"
        "E0,17/08/2024,Liverpool,Everton,,\n"
        ",,,,,\n",
        encoding="utf-8",
    )
    frame, _ = read_raw_file(path)
    assert len(frame) == 1
    assert frame.loc[0, "home_team_raw"] == "Arsenal"  # whitespace stripped
    assert pd.isna(frame.loc[0, "kickoff_time"])


def test_read_raw_file_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "E0_2425.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam\n10/08/2024,Arsenal,Chelsea\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="E0_2425.csv"):
        read_raw_file(path)


def test_read_raw_file_rejects_non_numeric_goals(tmp_path: Path) -> None:
    path = tmp_path / "E0_2425.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n10/08/2024,Arsenal,Chelsea,two,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-numeric"):
        read_raw_file(path)


def test_read_raw_file_rejects_fractional_stats(tmp_path: Path) -> None:
    path = tmp_path / "E0_2425.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,HS\n10/08/2024,Arsenal,Chelsea,2,1,10.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-integer"):
        read_raw_file(path)


# --------------------------------------------------------------------------- #
# Team resolution
# --------------------------------------------------------------------------- #
def test_resolve_teams_adds_canonical_ids_and_names(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    frame = resolved(tmp_path, registry)
    assert set(frame["home_team_id"]) == {"arsenal", "chelsea", "liverpool", "everton"}
    assert set(frame["home_team"]) == {"Arsenal", "Chelsea", "Liverpool", "Everton"}


def test_resolve_teams_reports_every_unknown_spelling(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    path = tmp_path / "E0_2425.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
        "10/08/2024,Arsenl,Chelsea,2,1\n"
        "17/08/2024,Liverpool,Evrton,0,0\n",
        encoding="utf-8",
    )
    frame, _ = read_raw_file(path)
    with pytest.raises(UnresolvedNamesError) as excinfo:
        resolve_teams(frame, registry)
    problems = excinfo.value.problems
    assert len(problems) == 2
    assert any("Arsenl" in p and "Arsenal" in p for p in problems)
    assert any("Evrton" in p and "Everton" in p for p in problems)


# --------------------------------------------------------------------------- #
# Building the table
# --------------------------------------------------------------------------- #
def test_build_match_table_end_to_end(tmp_path: Path, registry: TeamRegistry) -> None:
    write_season(tmp_path)
    result = build_match_table(tmp_path, registry, today=TODAY, current_season=2026)
    table = result.matches
    assert list(table.columns) == list(MATCH_COLUMNS)
    assert len(table) == 12
    assert table["date"].is_monotonic_increasing
    assert table["match_id"].is_unique
    first = table.iloc[0]
    assert first["match_id"] == "E0-2024-20240810-arsenal-chelsea"
    assert first["home_team"] == "Arsenal"
    assert set(table["result"]) <= {"H", "D", "A"}
    assert not result.errors
    assert codes(result.issues) == {
        "unusual_team_count"
    }  # four teams is not a real league


def test_build_match_table_requires_files(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    with pytest.raises(FileNotFoundError, match="python -m src.ingest"):
        build_match_table(tmp_path, registry)


def test_build_match_table_combines_leagues_and_seasons(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    write_season(tmp_path, "E0_2324.csv", start="2023-08-10")
    write_season(tmp_path, "E0_2425.csv")
    table = build_match_table(
        tmp_path, registry, today=TODAY, current_season=2026
    ).matches
    assert len(table) == 24
    assert sorted(table["season"].unique()) == [2023, 2024]


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #
def validate(
    frame: pd.DataFrame, current_season: int = 2026
) -> tuple[pd.DataFrame, list[pipeline.Issue]]:
    return validate_matches(frame, today=TODAY, current_season=current_season)


def test_clean_data_has_no_errors(tmp_path: Path, registry: TeamRegistry) -> None:
    cleaned, issues = validate(resolved(tmp_path, registry))
    assert len(cleaned) == 12
    assert not [i for i in issues if i.severity is Severity.ERROR]


def test_exact_duplicates_are_dropped_with_a_warning(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    frame = resolved(tmp_path, registry)
    cleaned, issues = validate(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    assert len(cleaned) == 12
    assert "duplicate_rows" in codes(issues)


def test_conflicting_duplicates_are_an_error(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    frame = resolved(tmp_path, registry)
    clash = frame.iloc[[0]].copy()
    clash["home_goals"] = clash["home_goals"] + 3
    _, issues = validate(pd.concat([frame, clash], ignore_index=True))
    assert "conflicting_duplicates" in codes(issues)


@pytest.mark.parametrize(
    ("column", "value", "code", "severity"),
    [
        ("home_goals", -1, "negative_goals", Severity.ERROR),
        ("home_goals", 12, "very_high_score", Severity.WARNING),
        ("away_team_id", "arsenal", "team_plays_itself", Severity.ERROR),
        ("date", pd.Timestamp("2027-01-01"), "future_dates", Severity.ERROR),
        ("ftr_raw", "A", "result_mismatch", Severity.ERROR),
    ],
)
def test_row_level_checks(
    tmp_path: Path,
    registry: TeamRegistry,
    column: str,
    value: object,
    code: str,
    severity: Severity,
) -> None:
    frame = resolved(tmp_path, registry)
    frame.loc[0, "home_team_id"] = "arsenal"
    frame.loc[0, "ftr_raw"] = "H"
    frame.loc[0, "away_goals"] = 0
    frame.loc[0, "home_goals"] = 1
    frame.loc[0, column] = value
    _, issues = validate(frame)
    matching = [i for i in issues if i.code == code]
    assert matching and matching[0].severity is severity


def test_impossible_odds_are_set_to_missing(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    frame = resolved(tmp_path, registry)
    frame.loc[0, "b365_h"] = 0.9
    cleaned, issues = validate(frame)
    assert pd.isna(cleaned.loc[cleaned.index[0], "b365_h"])
    assert "odds_not_above_one" in codes(issues)


def test_implausible_overround_is_flagged_but_kept(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    frame = resolved(tmp_path, registry)
    frame.loc[0, ["b365_h", "b365_d", "b365_a"]] = [1.1, 1.1, 1.1]
    cleaned, issues = validate(frame)
    assert "odd_overround" in codes(issues)
    assert cleaned.loc[cleaned.index[0], "b365_h"] == 1.1


def test_finished_season_with_missing_matches_is_flagged(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    frame = resolved(tmp_path, registry).iloc[1:]
    _, finished = validate(frame, current_season=2025)
    assert "incomplete_season" in codes(finished)
    _, in_progress = validate(frame, current_season=2024)
    assert "incomplete_season" not in codes(in_progress)


def test_known_short_seasons_are_not_flagged(
    tmp_path: Path, registry: TeamRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline, "KNOWN_SHORT_SEASONS", {("E0", 2024): 11})
    frame = resolved(tmp_path, registry)
    _, ok = validate(frame.iloc[1:], current_season=2025)
    assert "incomplete_season" not in codes(ok)
    _, flagged = validate(frame.iloc[2:], current_season=2025)
    assert "incomplete_season" in codes(flagged)


# --------------------------------------------------------------------------- #
# Coverage report
# --------------------------------------------------------------------------- #
def test_odds_coverage_percentages() -> None:
    table = pd.DataFrame(
        {
            "season": [2023, 2023, 2024, 2024],
            "ps_h": [2.0, np.nan, 2.0, 2.0],
            "psc_h": [np.nan] * 4,
            "b365_h": [2.0] * 4,
            "b365c_h": [2.0, 2.0, np.nan, np.nan],
            "avgc_h": [np.nan] * 4,
            "psc_over25": [1.9, 1.9, 1.9, 1.9],
        }
    )
    coverage = odds_coverage(table)
    assert list(coverage.index) == [2023, 2024]
    assert list(coverage["matches"]) == [2, 2]
    assert coverage.loc[2023, "ps"] == 50.0
    assert coverage.loc[2024, "ps"] == 100.0
    assert coverage.loc[2024, "b365c"] == 0.0
    assert coverage["psc"].eq(0.0).all()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def write_registry(path: Path) -> Path:
    lines = ["teams:"]
    for team in TEAMS:
        aliases = ", ".join(f'"{alias}"' for alias in team.aliases)
        lines.append(
            f'  - {{id: {team.id}, country: {team.country}, name: "{team.name}", aliases: [{aliases}]}}'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cli_args(tmp_path: Path, raw_dir: Path) -> list[str]:
    return [
        "--registry",
        str(tmp_path / "teams.yaml"),
        "--raw-dir",
        str(raw_dir),
        "--output",
        str(tmp_path / "out" / "matches.parquet"),
    ]


@pytest.mark.usefixtures("isolated_logging")
def test_cli_writes_the_parquet_table(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    write_season(raw)
    assert main(cli_args(tmp_path, raw)) == 0
    loaded = load_matches_table(tmp_path / "out" / "matches.parquet")
    assert list(loaded.columns) == list(MATCH_COLUMNS)
    assert len(loaded) == 12
    assert loaded["home_shots"].dtype == "Int64"
    assert loaded["date"].min() == pd.Timestamp("2024-08-10")


@pytest.mark.usefixtures("isolated_logging")
def test_cli_refuses_to_write_when_validation_finds_errors(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    write_season(raw, start="2099-08-10")  # dated in the future
    assert main(cli_args(tmp_path, raw)) == 1
    assert not (tmp_path / "out" / "matches.parquet").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_refuses_to_write_when_a_team_is_unknown(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "E0_2425.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n10/08/2024,Arsenal,Chelsae,2,1\n",
        encoding="utf-8",
    )
    assert main(cli_args(tmp_path, raw)) == 1
    assert not (tmp_path / "out" / "matches.parquet").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_reports_malformed_files_without_crashing(tmp_path: Path) -> None:
    write_registry(tmp_path / "teams.yaml")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "E0_2425.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    assert main(cli_args(tmp_path, raw)) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_cli_returns_one_without_data_and_two_without_registry(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    assert main(cli_args(tmp_path, raw)) == 2  # registry missing
    write_registry(tmp_path / "teams.yaml")
    assert main(cli_args(tmp_path, raw)) == 1  # no CSV files


def test_every_standard_odds_column_is_in_the_table_schema() -> None:
    assert set(ODDS_COLUMNS.values()) <= set(MATCH_COLUMNS)
    assert isinstance(BuildResult, type)
