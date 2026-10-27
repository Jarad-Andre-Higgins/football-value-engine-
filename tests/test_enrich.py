"""Tests for merging Understat xG into the match table."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from itertools import permutations
from pathlib import Path

import pandas as pd
import pytest

from src import logging_config
from src.enrich import EXTRA_COLUMNS, build_xg_table, main, merge_xg, xg_coverage
from src.entities import Team, TeamRegistry
from src.pipeline import Issue, Severity, UnresolvedNamesError
from src.understat import parse_league_payload

N_TEAMS = 12
NAMES = tuple(f"Team {i:02d}" for i in range(N_TEAMS))
REGISTRY_TEAMS = [
    Team(f"team-{i:02d}", name, "England") for i, name in enumerate(NAMES)
]
START = pd.Timestamp("2024-08-10")


def season_fixtures() -> list[dict[str, object]]:
    """A deterministic double round robin shared by both fake sources."""
    rows = []
    for index, (home, away) in enumerate(permutations(range(N_TEAMS), 2)):
        rows.append(
            {
                "index": index,
                "home": NAMES[home],
                "away": NAMES[away],
                "home_id": f"team-{home:02d}",
                "away_id": f"team-{away:02d}",
                "date": START
                + pd.Timedelta(index * 2, unit="D")
                + pd.Timedelta(15, unit="h"),
                "hg": (index * 7) % 4,
                "ag": (index * 5) % 3,
            }
        )
    return rows


def base_matches() -> pd.DataFrame:
    rows = [
        {
            "match_id": f"E0-2024-{r['date']:%Y%m%d}-{r['home_id']}-{r['away_id']}",
            "league": "E0",
            "season": 2024,
            "date": r["date"].normalize(),
            "home_team_id": r["home_id"],
            "away_team_id": r["away_id"],
            "home_goals": r["hg"],
            "away_goals": r["ag"],
        }
        for r in season_fixtures()
    ]
    return pd.DataFrame(rows)


def understat_payload() -> dict[str, object]:
    ids = {name: str(200 + i) for i, name in enumerate(NAMES)}
    history: dict[str, list[dict[str, object]]] = {name: [] for name in NAMES}
    fixtures = []
    for r in season_fixtures():
        stamp = f"{r['date']:%Y-%m-%d %H:%M:%S}"
        index = int(r["index"])  # type: ignore[call-overload]
        xg_home, xg_away = 1.0 + (index % 9) * 0.1, 0.7 + (index % 7) * 0.1
        fixtures.append(
            {
                "id": str(5000 + index),
                "isResult": True,
                "h": {"id": ids[str(r["home"])], "title": r["home"]},
                "a": {"id": ids[str(r["away"])], "title": r["away"]},
                "goals": {"h": str(r["hg"]), "a": str(r["ag"])},
                "xG": {"h": f"{xg_home:.4f}", "a": f"{xg_away:.4f}"},
                "datetime": stamp,
            }
        )
        for team, side, xg in ((r["home"], "h", xg_home), (r["away"], "a", xg_away)):
            history[str(team)].append(
                {
                    "h_a": side,
                    "npxG": xg - 0.1,
                    "deep": 4 + index % 5,
                    "ppda": {"att": 120, "def": 12},
                    "date": stamp,
                }
            )
    return {
        "teams": {
            ids[n]: {"id": ids[n], "title": n, "history": history[n]} for n in NAMES
        },
        "players": [],
        "dates": fixtures,
    }


def xg_frame() -> pd.DataFrame:
    """The parsed-and-resolved Understat table, as build_xg_table would return."""
    frame = parse_league_payload(understat_payload(), "E0", 2024)
    ids = {name: f"team-{i:02d}" for i, name in enumerate(NAMES)}
    frame["home_team_id"] = frame["home_team_raw"].map(ids)
    frame["away_team_id"] = frame["away_team_raw"].map(ids)
    return frame


@pytest.fixture
def registry() -> TeamRegistry:
    return TeamRegistry(REGISTRY_TEAMS)


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def codes(issues: list[Issue]) -> set[str]:
    return {issue.code for issue in issues}


def severity_of(issues: list[Issue], code: str) -> Severity:
    return next(issue.severity for issue in issues if issue.code == code)


# --------------------------------------------------------------------------- #
# Building the xG table
# --------------------------------------------------------------------------- #
def test_build_xg_table_resolves_ids(tmp_path: Path, registry: TeamRegistry) -> None:
    (tmp_path / "E0_2425.json").write_text(
        json.dumps(understat_payload()), encoding="utf-8"
    )
    table = build_xg_table(tmp_path, registry)
    assert len(table) == N_TEAMS * (N_TEAMS - 1)
    assert set(table["home_team_id"]) == {t.id for t in REGISTRY_TEAMS}


def test_build_xg_table_reports_unknown_spellings(
    tmp_path: Path, registry: TeamRegistry
) -> None:
    payload = understat_payload()
    payload["dates"][0]["h"]["title"] = "Tean 00"  # type: ignore[index]
    (tmp_path / "E0_2425.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UnresolvedNamesError) as excinfo:
        build_xg_table(tmp_path, registry)
    assert any("Tean 00" in p and "Team 00" in p for p in excinfo.value.problems)


def test_build_xg_table_needs_files(tmp_path: Path, registry: TeamRegistry) -> None:
    with pytest.raises(FileNotFoundError, match="python -m src.understat"):
        build_xg_table(tmp_path, registry)


# --------------------------------------------------------------------------- #
# Merging and cross-checks
# --------------------------------------------------------------------------- #
def test_merge_adds_xg_columns_and_finds_no_problems() -> None:
    matches = base_matches()
    enriched, issues = merge_xg(matches, xg_frame(), current_season=2026)
    assert list(enriched.columns) == [*matches.columns, *EXTRA_COLUMNS]
    assert len(enriched) == len(matches)
    assert enriched[EXTRA_COLUMNS].notna().all().all()
    assert not issues
    first = enriched.iloc[0]
    assert first["home_xg"] == pytest.approx(1.0)
    assert first["home_ppda"] == pytest.approx(10.0)


def test_a_few_score_disagreements_warn_but_many_are_an_error() -> None:
    xg = xg_frame()
    xg.loc[0, "home_goals_u"] = 9
    _, one = merge_xg(base_matches(), xg, current_season=2026)  # 1 of 132 is below 1%
    assert severity_of(one, "score_disagreement") is Severity.WARNING

    xg.loc[:5, "home_goals_u"] = 9
    _, many = merge_xg(base_matches(), xg, current_season=2026)
    assert severity_of(many, "score_disagreement") is Severity.ERROR


def test_swapped_home_and_away_is_caught_as_score_disagreement() -> None:
    xg = xg_frame()
    swapped = xg.rename(
        columns={"home_team_id": "away_team_id", "away_team_id": "home_team_id"}
    )
    _, issues = merge_xg(base_matches(), swapped, current_season=2026)
    assert severity_of(issues, "score_disagreement") is Severity.ERROR


def test_duplicate_understat_fixtures_are_an_error() -> None:
    xg = xg_frame()
    _, issues = merge_xg(
        base_matches(),
        pd.concat([xg, xg.iloc[[0]]], ignore_index=True),
        current_season=2026,
    )
    assert severity_of(issues, "duplicate_xg_fixtures") is Severity.ERROR


def test_xg_value_checks() -> None:
    xg = xg_frame()
    xg.loc[0, "home_xg"] = -0.5
    xg.loc[1, "away_xg"] = 9.5
    _, issues = merge_xg(base_matches(), xg, current_season=2026)
    assert severity_of(issues, "negative_xg") is Severity.ERROR
    assert severity_of(issues, "very_high_xg") is Severity.WARNING


def test_date_gap_is_a_warning() -> None:
    xg = xg_frame()
    xg.loc[0, "date"] = xg.loc[0, "date"] + pd.Timedelta(10, unit="D")
    _, issues = merge_xg(base_matches(), xg, current_season=2026)
    assert severity_of(issues, "date_gap") is Severity.WARNING


def test_missing_and_orphan_matches() -> None:
    matches = base_matches()
    _, finished = merge_xg(matches, xg_frame().iloc[2:], current_season=2026)
    assert severity_of(finished, "match_without_xg") is Severity.WARNING
    _, in_progress = merge_xg(matches, xg_frame().iloc[2:], current_season=2024)
    assert "match_without_xg" not in codes(in_progress)

    _, orphaned = merge_xg(matches.iloc[2:], xg_frame(), current_season=2026)
    assert severity_of(orphaned, "xg_without_match") is Severity.WARNING


def test_coverage_percentages() -> None:
    enriched, _ = merge_xg(base_matches(), xg_frame().iloc[66:], current_season=2024)
    coverage = xg_coverage(enriched)
    assert coverage.loc[2024, "matches"] == 132
    assert coverage.loc[2024, "xg"] == pytest.approx(50.0)
    assert coverage.loc[2024, "ppda"] == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def write_inputs(
    tmp_path: Path, *, payload: dict[str, object] | None = None
) -> list[str]:
    lines = ["teams:"] + [
        f'  - {{id: {t.id}, country: England, name: "{t.name}", aliases: []}}'
        for t in REGISTRY_TEAMS
    ]
    (tmp_path / "teams.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    raw = tmp_path / "understat"
    raw.mkdir()
    (raw / "E0_2425.json").write_text(
        json.dumps(payload or understat_payload()), encoding="utf-8"
    )
    base_matches().to_parquet(tmp_path / "matches.parquet", index=False)
    return [
        "--registry", str(tmp_path / "teams.yaml"),
        "--understat-dir", str(raw),
        "--matches", str(tmp_path / "matches.parquet"),
        "--xg-output", str(tmp_path / "out" / "xg_matches.parquet"),
        "--output", str(tmp_path / "out" / "matches_enriched.parquet"),
    ]  # fmt: skip


@pytest.mark.usefixtures("isolated_logging")
def test_cli_writes_both_outputs(tmp_path: Path) -> None:
    assert main(write_inputs(tmp_path)) == 0
    enriched = pd.read_parquet(tmp_path / "out" / "matches_enriched.parquet")
    xg = pd.read_parquet(tmp_path / "out" / "xg_matches.parquet")
    assert len(enriched) == len(xg) == N_TEAMS * (N_TEAMS - 1)
    assert set(EXTRA_COLUMNS) <= set(enriched.columns)
    assert enriched["home_xg"].notna().all()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_refuses_to_write_when_the_sources_disagree(tmp_path: Path) -> None:
    payload = understat_payload()
    for fixture in payload["dates"][:20]:  # type: ignore[index]
        fixture["goals"]["h"] = "9"
    args = write_inputs(tmp_path, payload=payload)
    assert main(args) == 1
    assert not (tmp_path / "out" / "matches_enriched.parquet").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_stops_on_unknown_team_names(tmp_path: Path) -> None:
    payload = understat_payload()
    payload["dates"][0]["a"]["title"] = "Nonexistent FC"  # type: ignore[index]
    assert main(write_inputs(tmp_path, payload=payload)) == 1
    assert not (tmp_path / "out" / "matches_enriched.parquet").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_exit_codes_for_missing_inputs(tmp_path: Path) -> None:
    args = write_inputs(tmp_path)
    (tmp_path / "matches.parquet").unlink()
    assert main(args) == 1  # match table missing
    (tmp_path / "teams.yaml").unlink()
    assert main(args) == 2  # registry missing
