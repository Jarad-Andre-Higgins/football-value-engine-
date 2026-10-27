"""Tests for the Understat downloader and parser, run against a fake local server."""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import permutations
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src import logging_config
from src.ingest import FetchStatus
from src.understat import (
    XG_COLUMNS,
    UnderstatDownloader,
    describe_file,
    main,
    parse_file_name,
    parse_league_payload,
    payload_problem,
    read_league_file,
)

ENGLISH = ("Arsenal", "Chelsea", "Liverpool", "Everton")


def make_payload(
    teams: tuple[str, ...] = ENGLISH, start: str = "2024-08-10", unplayed: int = 1
) -> dict[str, Any]:
    """A getLeagueData-shaped payload: a double round robin with team histories."""
    ids = {team: str(100 + i) for i, team in enumerate(teams)}
    history: dict[str, list[dict[str, Any]]] = {team: [] for team in teams}
    fixtures = []
    pairs = list(permutations(teams, 2))
    for index, (home, away) in enumerate(pairs):
        when = (
            pd.Timestamp(start)
            + pd.Timedelta(index * 3, unit="D")
            + pd.Timedelta(15, unit="h")
        )
        stamp = when.strftime("%Y-%m-%d %H:%M:%S")
        played = index < len(pairs) - unplayed
        home_goals, away_goals = (index * 7) % 4, (index * 5) % 3
        home_xg, away_xg = 1.0 + index * 0.1, 0.8 + index * 0.05
        fixtures.append(
            {
                "id": str(1000 + index),
                "isResult": played,
                "h": {"id": ids[home], "title": home, "short_title": home[:3].upper()},
                "a": {"id": ids[away], "title": away, "short_title": away[:3].upper()},
                "goals": {
                    "h": str(home_goals) if played else None,
                    "a": str(away_goals) if played else None,
                },
                "xG": {
                    "h": f"{home_xg:.5f}" if played else None,
                    "a": f"{away_xg:.5f}" if played else None,
                },
                "datetime": stamp,
                "forecast": {"w": "0.4", "d": "0.3", "l": "0.3"},
            }
        )
        if played:
            for team, side, xg in ((home, "h", home_xg), (away, "a", away_xg)):
                history[team].append(
                    {
                        "h_a": side,
                        "xG": xg,
                        "npxG": xg - 0.1,
                        "deep": 5 + index,
                        "ppda": {"att": 100 + index, "def": 10},
                        "date": stamp,
                    }
                )
    return {
        "teams": {
            ids[t]: {"id": ids[t], "title": t, "history": history[t]} for t in teams
        },
        "players": [],
        "dates": fixtures,
    }


def encode(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_returns_played_matches_with_team_stats() -> None:
    frame = parse_league_payload(make_payload(), "E0", 2024)
    assert list(frame.columns) == list(XG_COLUMNS)
    assert len(frame) == 11  # twelve fixtures, one not yet played
    first = frame.iloc[0]
    assert (first["home_team_raw"], first["away_team_raw"]) == ("Arsenal", "Chelsea")
    assert first["understat_id"] == "1000"
    assert (first["home_goals_u"], first["away_goals_u"]) == (0, 0)
    assert first["home_xg"] == pytest.approx(1.0)
    assert first["away_xg"] == pytest.approx(0.8)
    assert first["home_npxg"] == pytest.approx(0.9)
    assert first["home_ppda"] == pytest.approx(
        10.0
    )  # 100 passes / 10 defensive actions
    assert first["home_deep"] == 5
    assert frame["home_goals_u"].dtype == "int64"
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_history_is_matched_by_calendar_day_not_exact_time() -> None:
    payload = make_payload()
    for team in payload["teams"].values():
        for row in team["history"]:
            row["date"] = (
                row["date"][:10] + " 00:00:00"
            )  # same day, different clock time
    frame = parse_league_payload(payload, "E0", 2024)
    assert frame["home_ppda"].notna().all()


def test_matches_without_team_history_are_kept_with_missing_stats() -> None:
    payload = make_payload()
    payload["teams"]["100"]["history"] = []  # Arsenal has no history rows
    frame = parse_league_payload(payload, "E0", 2024)
    assert len(frame) == 11
    arsenal_home = frame[frame["home_team_raw"] == "Arsenal"]
    assert arsenal_home["home_ppda"].isna().all()
    assert arsenal_home["away_ppda"].notna().all()


def test_zero_defensive_actions_give_missing_ppda() -> None:
    payload = make_payload()
    payload["teams"]["100"]["history"][0]["ppda"] = {"att": 80, "def": 0}
    frame = parse_league_payload(payload, "E0", 2024)
    assert pd.isna(frame.iloc[0]["home_ppda"])


def test_string_false_is_not_treated_as_played() -> None:
    payload = make_payload()
    payload["dates"][0]["isResult"] = "false"
    assert len(parse_league_payload(payload, "E0", 2024)) == 10


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {
            "teams": {},
            "dates": [{"id": "1", "isResult": True, "goals": {"h": "1", "a": "0"}}],
        },
    ],
)
def test_unexpected_payload_shape_raises_a_helpful_error(
    broken: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="--peek"):
        parse_league_payload(broken, "E0", 2024)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("E0_2526.json", ("E0", 2025)), ("SP1_1920.json", ("SP1", 2019))],
)
def test_parse_file_name(name: str, expected: tuple[str, int]) -> None:
    assert parse_file_name(Path(name)) == expected


@pytest.mark.parametrize(
    "name", ["E0_2526.csv", "XX_2526.json", "E0_2527.json", "notes.json"]
)
def test_parse_file_name_rejects_bad_names(name: str) -> None:
    with pytest.raises(ValueError):
        parse_file_name(Path(name))


def test_read_league_file_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "E0_2425.json"
    path.write_text("<html>shell</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_league_file(path)


def test_describe_file_shows_the_structure(tmp_path: Path) -> None:
    path = tmp_path / "E0_2425.json"
    path.write_bytes(encode(make_payload()))
    text = describe_file(path)
    assert "top-level keys ['dates', 'players', 'teams']" in text
    assert "first played match" in text
    assert '"title": "Arsenal"' in text


@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        (b"<html></html>", "not JSON"),
        (b'{"teams": {}}', "unexpected payload shape"),
        (b"[]", "unexpected payload shape"),
        (b'{"teams": {}, "dates": []}', "no fixtures"),
    ],
)
def test_payload_problem_detects_bad_responses(content: bytes, fragment: str) -> None:
    assert fragment in (payload_problem(content) or "")


def test_payload_problem_accepts_a_real_shape() -> None:
    assert payload_problem(encode(make_payload())) is None


# --------------------------------------------------------------------------- #
# Downloading (a fake server that, like Understat, 404s without the header)
# --------------------------------------------------------------------------- #
@dataclass
class FakeServer:
    base_url: str
    routes: dict[str, tuple[int, bytes]]
    hits: list[str]


@pytest.fixture
def server() -> Iterator[FakeServer]:
    routes: dict[str, tuple[int, bytes]] = {}
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            if self.headers.get("X-Requested-With") != "XMLHttpRequest":
                status, body = 404, b"not found"
            else:
                status, body = routes.get(self.path, (404, b"not found"))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Silence the default per-request console output."""

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield FakeServer(f"http://127.0.0.1:{httpd.server_address[1]}", routes, hits)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def make_downloader(server: FakeServer, raw_dir: Path) -> UnderstatDownloader:
    return UnderstatDownloader(
        raw_dir, server.base_url, retries=1, backoff=0.0, delay=0.0, timeout=(2.0, 2.0)
    )


def test_download_sends_the_required_header_and_saves_raw_bytes(
    server: FakeServer, tmp_path: Path
) -> None:
    body = encode(make_payload())
    server.routes["/getLeagueData/EPL/2025"] = (200, body)
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.DOWNLOADED
    assert (tmp_path / "E0_2526.json").read_bytes() == body
    assert not list(tmp_path.glob("*.part"))


def test_completed_seasons_are_cached_and_the_current_one_is_refreshed(
    server: FakeServer, tmp_path: Path
) -> None:
    body = encode(make_payload())
    server.routes["/getLeagueData/EPL/2025"] = (200, body)
    server.routes["/getLeagueData/EPL/2026"] = (200, body)
    with make_downloader(server, tmp_path) as downloader:
        downloader.fetch("E0", 2025, current_season=2026)
        cached = downloader.fetch("E0", 2025, current_season=2026)
        downloader.fetch("E0", 2026, current_season=2026)
        downloader.fetch("E0", 2026, current_season=2026)
    assert cached.status is FetchStatus.CACHED
    assert server.hits.count("/getLeagueData/EPL/2025") == 1
    assert server.hits.count("/getLeagueData/EPL/2026") == 2


def test_each_league_uses_its_understat_slug(
    server: FakeServer, tmp_path: Path
) -> None:
    body = encode(make_payload())
    for slug in ("La_liga", "Serie_A", "Bundesliga", "Ligue_1"):
        server.routes[f"/getLeagueData/{slug}/2025"] = (200, body)
    with make_downloader(server, tmp_path) as downloader:
        results = downloader.download_all(
            ["SP1", "I1", "D1", "F1"], [2025], current_season=2026
        )
    assert [r.status for r in results] == [FetchStatus.DOWNLOADED] * 4


def test_404_is_missing_and_a_shell_page_is_a_loud_failure(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/getLeagueData/EPL/2025"] = (200, b"<html>empty shell</html>")
    with make_downloader(server, tmp_path) as downloader:
        shell = downloader.fetch("E0", 2025, current_season=2026)
        missing = downloader.fetch("E0", 2026, current_season=2026)
    assert shell.status is FetchStatus.FAILED
    assert "not JSON" in shell.detail
    assert missing.status is FetchStatus.MISSING
    assert not list(tmp_path.iterdir())


def test_wrong_shape_and_server_errors_are_failures(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/getLeagueData/EPL/2025"] = (200, b'{"unexpected": true}')
    server.routes["/getLeagueData/EPL/2024"] = (500, b"boom")
    with make_downloader(server, tmp_path) as downloader:
        shape = downloader.fetch("E0", 2025, current_season=2026)
        error = downloader.fetch("E0", 2024, current_season=2026)
    assert (
        shape.status is FetchStatus.FAILED
        and "unexpected payload shape" in shape.detail
    )
    assert error.status is FetchStatus.FAILED
    assert server.hits.count("/getLeagueData/EPL/2024") == 2  # one retry


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def cli_args(server: FakeServer, raw_dir: Path) -> list[str]:
    return [
        "--leagues",
        "E0",
        "--from-season",
        "2025",
        "--to-season",
        "2025",
        "--delay",
        "0",
        "--raw-dir",
        str(raw_dir),
        "--base-url",
        server.base_url,
    ]


@pytest.mark.usefixtures("isolated_logging")
def test_cli_exit_codes(server: FakeServer, tmp_path: Path) -> None:
    server.routes["/getLeagueData/EPL/2025"] = (200, encode(make_payload()))
    assert main(cli_args(server, tmp_path / "raw")) == 0
    assert (tmp_path / "raw" / "E0_2526.json").exists()
    server.routes["/getLeagueData/EPL/2025"] = (200, b"<html></html>")
    assert main([*cli_args(server, tmp_path / "other")]) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_cli_peek(server: FakeServer, tmp_path: Path) -> None:
    (tmp_path / "E0_2526.json").write_bytes(encode(make_payload()))
    assert main(["--raw-dir", str(tmp_path), "--peek", "E0", "2025"]) == 0
    assert main(["--raw-dir", str(tmp_path), "--peek", "E0", "2019"]) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_cli_rejects_a_reversed_season_range(
    server: FakeServer, tmp_path: Path
) -> None:
    args = cli_args(server, tmp_path)
    args[args.index("--to-season") + 1] = "2020"
    with pytest.raises(SystemExit) as excinfo:
        main(args)
    assert excinfo.value.code == 2
