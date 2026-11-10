"""Tests for the Understat downloader and parser, run against a fake local server."""

from __future__ import annotations

import json
import logging
import sys
import threading
import zipfile
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
    describe_response,
    import_files,
    main,
    parse_file_name,
    parse_league_payload,
    payload_problem,
    read_league_file,
    unusable_reason,
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
# Saying what a bad response is (Understat's "We will come back soon" page)
# --------------------------------------------------------------------------- #
MAINTENANCE = (
    b"<!DOCTYPE html><html><head><title>Understat</title>"
    b"<style>body { background: #333; }</style></head>"
    b'<body><script>var note = "just a moment";</script>'
    b"<h1>We will come back soon</h1></body></html>"
)


@pytest.mark.parametrize(
    ("status", "headers", "content", "fragments"),
    [
        (
            200,
            {},
            MAINTENANCE,
            ["maintenance page", '"We will come back soon"', "try again later"],
        ),
        (
            200,
            {},
            b"<html><head><title>Site under maintenance</title></head><body><img></body></html>",
            ["maintenance page", '"Site under maintenance"'],  # only the title says so
        ),
        (
            200,
            {},
            b"<html><title>Just a moment...</title><body>Checking your browser</body></html>",
            ["firewall or bot-check"],
        ),
        (
            403,
            {"CF-RAY": "8a1b"},
            b"<html><body>Forbidden</body></html>",
            ["firewall or bot-check"],
        ),
        (
            200,
            {"CF-RAY": "8a1b"},
            b"<html><head><title>Understat</title></head><body>Hello</body></html>",
            ['title "Understat"', "endpoint changed"],
        ),
        (502, {}, b"<html><body>Bad gateway</body></html>", ["site may be down"]),
        (
            200,
            {},
            b"<html><head><title> EPL\n  table </title></head></html>",
            ['title "EPL table"'],
        ),
        (
            200,
            {},
            b"<html><body><script>var a = 'back soon';</script><p>Hello there</p></body></html>",
            ['text "Hello there"', "endpoint changed"],  # script text is not page text
        ),
        (200, {}, b"  \n", ["the response is empty"]),
        (200, {}, b"\x1f\x8bgarbage", ["not data", "\\x1f\\x8b", "endpoint changed"]),
    ],
)
def test_describe_response_says_what_came_back(
    status: int, headers: dict[str, str], content: bytes, fragments: list[str]
) -> None:
    text = describe_response(status, headers, content)
    for fragment in fragments:
        assert fragment in text, (fragment, text)


def test_words_inside_scripts_and_styles_are_not_mistaken_for_the_page() -> None:
    page = (
        b"<html><head><style>.back-soon { color: red }</style></head><body>"
        b"<script>var a = 'just a moment, back soon';</script><p>Hello there</p>"
        b"</body></html>"
    )
    text = describe_response(200, {}, page)
    assert "maintenance" not in text and "firewall" not in text
    assert 'text "Hello there"' in text


def test_unusable_reason_explains_only_what_is_not_json() -> None:
    assert unusable_reason(encode(make_payload())) is None
    assert unusable_reason(b'{"teams": {}}') == payload_problem(b'{"teams": {}}')
    reason = unusable_reason(MAINTENANCE)
    assert reason is not None
    assert reason.startswith("response is not JSON: Understat is showing a maintenance")


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
            else:  # "*" is the answer for every path that has no route of its own
                fallback = routes.get("*", (404, b"not found"))
                status, body = routes.get(self.path, fallback)
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


def test_a_maintenance_page_is_reported_as_what_it_is(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/getLeagueData/EPL/2025"] = (200, MAINTENANCE)
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.FAILED
    assert result.detail.startswith("response is not JSON: Understat is showing")
    assert '"We will come back soon"' in result.detail
    assert result.detail.endswith(
        f"[HTTP 200, no content type, {len(MAINTENANCE)} bytes]"
    )
    assert not list(tmp_path.iterdir())


def test_a_blocked_request_says_so_and_keeps_its_status(
    server: FakeServer, tmp_path: Path
) -> None:
    page = b"<html><title>Just a moment...</title><body>Checking your browser</body></html>"
    server.routes["/getLeagueData/EPL/2025"] = (403, page)
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.FAILED
    assert result.detail.startswith("HTTP 403: a firewall or bot-check page")
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# Stopping when the site is down
# --------------------------------------------------------------------------- #
def test_the_run_stops_after_three_failures_in_a_row(
    server: FakeServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    with (
        make_downloader(server, tmp_path) as downloader,
        caplog.at_level(logging.ERROR, logger="src.understat"),
    ):
        results = downloader.download_all(
            ["E0", "SP1"], [2019, 2020, 2021], current_season=2026
        )
    assert [r.status for r in results] == [FetchStatus.FAILED] * 3
    assert len(server.hits) == 3  # three of the six files were tried
    assert "Stopping after 3 failures in a row; 3 files were not tried" in caplog.text
    assert not list(tmp_path.iterdir())


def test_a_limit_of_zero_never_stops(server: FakeServer, tmp_path: Path) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    with make_downloader(server, tmp_path) as downloader:
        results = downloader.download_all(
            ["E0", "SP1"], [2019, 2020, 2021], current_season=2026, max_failures=0
        )
    assert len(results) == 6 and len(server.hits) == 6


def test_the_limit_is_configurable(server: FakeServer, tmp_path: Path) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    with make_downloader(server, tmp_path) as downloader:
        results = downloader.download_all(
            ["E0"], [2019, 2020, 2021, 2022], current_season=2026, max_failures=2
        )
    assert len(results) == 2 and len(server.hits) == 2


@pytest.mark.parametrize(
    "answer",
    [(200, encode(make_payload())), (404, b"not found")],
    ids=["a download", "a 404"],
)
def test_any_answer_from_the_site_resets_the_count(
    server: FakeServer, tmp_path: Path, answer: tuple[int, bytes]
) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    server.routes["/getLeagueData/EPL/2021"] = answer
    with make_downloader(server, tmp_path) as downloader:
        results = downloader.download_all(
            ["E0"], [2019, 2020, 2021, 2022, 2023], current_season=2026
        )
    assert len(results) == 5 and len(server.hits) == 5  # two, a reset, two: never three


def test_cached_files_neither_count_nor_reset(
    server: FakeServer, tmp_path: Path
) -> None:
    (tmp_path / "E0_2021.json").write_bytes(encode(make_payload()))  # season 2020
    server.routes["*"] = (200, MAINTENANCE)
    with make_downloader(server, tmp_path) as downloader:
        results = downloader.download_all(
            ["E0"], [2019, 2020, 2021, 2022, 2023, 2024], current_season=2026
        )
    statuses = [r.status for r in results]
    assert statuses == [
        FetchStatus.FAILED,
        FetchStatus.CACHED,
        FetchStatus.FAILED,
        FetchStatus.FAILED,
    ]
    assert len(server.hits) == 3


def test_no_stop_message_when_the_last_file_trips_the_limit(
    server: FakeServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    with (
        make_downloader(server, tmp_path) as downloader,
        caplog.at_level(logging.ERROR, logger="src.understat"),
    ):
        results = downloader.download_all(
            ["E0"], [2019, 2020, 2021], current_season=2026
        )
    assert len(results) == 3 and "Stopping after" not in caplog.text


# --------------------------------------------------------------------------- #
# Restoring a cache made elsewhere
# --------------------------------------------------------------------------- #
def write_saved_cache(source: Path) -> bytes:
    """A folder as an earlier session or a Kaggle dataset would leave it."""
    good = encode(make_payload())
    (source / "nested").mkdir(parents=True)
    (source / "E0_2526.json").write_bytes(good)
    (source / "nested" / "SP1_2425.json").write_bytes(good)
    (source / "E0_2425.json").write_bytes(MAINTENANCE)  # an error page saved by mistake
    (source / "dataset-metadata.json").write_text("{}", encoding="utf-8")
    (source / "notes.txt").write_text("not data", encoding="utf-8")
    return good


def test_import_copies_valid_files_and_reports_the_rest(tmp_path: Path) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    good = write_saved_cache(source)
    report = import_files(source, raw)
    assert sorted(report.imported) == ["E0_2526.json", "SP1_2425.json"]
    assert report.present == ()
    assert list(report.rejected) == ["E0_2425.json"]
    assert report.rejected["E0_2425.json"].startswith(
        "response is not JSON: Understat is showing a maintenance page"
    )
    assert report.ignored == 1  # dataset-metadata.json
    assert sorted(p.name for p in raw.iterdir()) == ["E0_2526.json", "SP1_2425.json"]
    assert (raw / "E0_2526.json").read_bytes() == good


def test_import_keeps_files_already_present_unless_forced(tmp_path: Path) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    write_saved_cache(source)
    import_files(source, raw)
    newer = encode(make_payload(start="2025-08-09"))
    (source / "E0_2526.json").write_bytes(newer)
    kept = import_files(source, raw)
    assert kept.imported == () and sorted(kept.present) == [
        "E0_2526.json",
        "SP1_2425.json",
    ]
    assert (raw / "E0_2526.json").read_bytes() != newer
    forced = import_files(source, raw, force=True)
    assert sorted(forced.imported) == ["E0_2526.json", "SP1_2425.json"]
    assert (raw / "E0_2526.json").read_bytes() == newer
    assert not list(raw.glob("*.part"))


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_import_opens_zip_archives_like_folders(tmp_path: Path) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    good = encode(make_payload())
    make_zip(
        source / "understat_cache.zip",
        {
            "data/raw/understat/E0_2526.json": good,  # folders inside the zip do not matter
            "data/raw/understat/SP1_2425.json": good,
            "data/raw/understat/E0_2425.json": MAINTENANCE,
            "data/raw/understat/dataset-metadata.json": b"{}",
            "data/raw/understat/readme.txt": b"not json",
        },
    )
    report = import_files(source, raw)
    assert sorted(report.imported) == ["E0_2526.json", "SP1_2425.json"]
    assert list(report.rejected) == ["E0_2425.json"] and report.ignored == 1
    assert (raw / "E0_2526.json").read_bytes() == good
    assert sorted(p.name for p in raw.iterdir()) == ["E0_2526.json", "SP1_2425.json"]


def test_import_reports_an_unreadable_zip_and_keeps_going(tmp_path: Path) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    (source / "broken.zip").write_bytes(b"this is not a zip archive")
    (source / "E0_2526.json").write_bytes(encode(make_payload()))
    report = import_files(source, raw)
    assert report.rejected == {"broken.zip": "not a readable zip archive"}
    assert report.imported == ("E0_2526.json",)


def test_the_same_file_in_a_folder_and_a_zip_is_imported_once(tmp_path: Path) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    good = encode(make_payload())
    (source / "E0_2526.json").write_bytes(good)
    make_zip(source / "again.zip", {"E0_2526.json": good})
    report = import_files(source, raw)
    assert report.imported == ("E0_2526.json",) and report.present == ("E0_2526.json",)


def test_import_needs_a_folder_that_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        import_files(tmp_path / "nowhere", tmp_path / "raw")
    (tmp_path / "file.json").write_text("{}", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="not a folder"):
        import_files(tmp_path / "file.json", tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_imported_files_read_like_downloaded_ones(tmp_path: Path) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    write_saved_cache(source)
    import_files(source, raw)
    frame = read_league_file(raw / "E0_2526.json")
    assert list(frame.columns) == list(XG_COLUMNS) and len(frame) > 0


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


@pytest.mark.usefixtures("isolated_logging")
def test_cli_stops_early_when_the_site_is_down(
    server: FakeServer, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    args = [*cli_args(server, tmp_path / "raw"), "--from-season", "2019"]
    with caplog.at_level(logging.INFO):
        assert main(args) == 1
    assert len(server.hits) == 3  # seven seasons were asked for
    assert "failed=3 not_tried=4" in caplog.text
    assert "Stopping after 3 failures in a row; 4 files were not tried" in caplog.text
    assert not (tmp_path / "raw").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_max_failures_zero_tries_everything(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["*"] = (200, MAINTENANCE)
    args = [*cli_args(server, tmp_path / "raw"), "--from-season", "2019"]
    assert main([*args, "--max-failures", "0"]) == 1
    assert len(server.hits) == 7


def test_cli_rejects_a_negative_failure_limit(
    server: FakeServer, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([*cli_args(server, tmp_path), "--max-failures", "-1"])
    assert excinfo.value.code == 2


@pytest.mark.usefixtures("isolated_logging")
def test_cli_import_from_restores_a_clean_cache(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    (source / "E0_2526.json").write_bytes(encode(make_payload()))
    with caplog.at_level(logging.INFO):
        assert main(["--import-from", str(source), "--raw-dir", str(raw)]) == 0
    assert "imported=1 already_present=0 rejected=0 other_json_ignored=0" in caplog.text
    assert (raw / "E0_2526.json").exists()
    assert main(["--import-from", str(source), "--raw-dir", str(raw)]) == 0  # again


@pytest.mark.usefixtures("isolated_logging")
def test_cli_import_from_fails_when_a_file_is_refused_but_keeps_the_good_ones(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source, raw = tmp_path / "source", tmp_path / "raw"
    write_saved_cache(source)
    with caplog.at_level(logging.INFO):
        assert main(["--import-from", str(source), "--raw-dir", str(raw)]) == 1
    assert "E0_2425.json: rejected - response is not JSON" in caplog.text
    assert (raw / "E0_2526.json").exists() and not (raw / "E0_2425.json").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_import_from_needs_something_to_import(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    empty, raw = tmp_path / "empty", tmp_path / "raw"
    empty.mkdir()
    with caplog.at_level(logging.ERROR):
        assert main(["--import-from", str(empty), "--raw-dir", str(raw)]) == 1
        assert "No Understat league files found" in caplog.text
        caplog.clear()
        assert (
            main(["--import-from", str(tmp_path / "gone"), "--raw-dir", str(raw)]) == 1
        )
        assert "Cannot import from" in caplog.text


def test_cli_import_from_cannot_be_combined_with_peek(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--import-from", str(tmp_path), "--peek", "E0", "2025"])
    assert excinfo.value.code == 2
