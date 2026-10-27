"""Tests for the football-data.co.uk downloader, run against a local fake server."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src import logging_config
from src.ingest import (
    FetchStatus,
    FootballDataDownloader,
    looks_like_results_csv,
    main,
)
from src.leagues import current_season_start, season_code, season_label

CSV = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,15/08/2025,Arsenal,Chelsea,2,1\n"


@dataclass
class FakeServer:
    """A local HTTP server whose responses each test controls."""

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
        port = httpd.server_address[1]
        yield FakeServer(f"http://127.0.0.1:{port}/mmz4281", routes, hits)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


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


def make_downloader(
    server: FakeServer, raw_dir: Path, *, base_url: str | None = None
) -> FootballDataDownloader:
    return FootballDataDownloader(
        raw_dir,
        base_url or server.base_url,
        retries=1,
        backoff=0.0,
        delay=0.0,
        timeout=(2.0, 2.0),
    )


# --------------------------------------------------------------------------- #
# Season helpers
# --------------------------------------------------------------------------- #
def test_season_helpers() -> None:
    assert season_code(2025) == "2526"
    assert season_code(1999) == "9900"
    assert season_label(2025) == "2025/26"
    assert season_label(1999) == "1999/00"


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 26), 2026),
        (date(2027, 3, 1), 2026),
        (date(2027, 6, 30), 2026),
        (date(2027, 7, 1), 2027),
    ],
)
def test_current_season_rolls_over_on_1_july(today: date, expected: int) -> None:
    assert current_season_start(today) == expected


def test_csv_header_check() -> None:
    assert looks_like_results_csv(CSV)
    assert looks_like_results_csv(b"\xef\xbb\xbf" + CSV)  # byte-order mark
    assert not looks_like_results_csv(b"<html><body>Not found</body></html>")
    assert not looks_like_results_csv(b"")


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #
def test_downloads_a_valid_file_atomically(server: FakeServer, tmp_path: Path) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (200, CSV)
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.DOWNLOADED
    assert result.path == tmp_path / "E0_2526.csv"
    assert (tmp_path / "E0_2526.csv").read_bytes() == CSV
    assert not list(tmp_path.glob("*.part"))


def test_completed_seasons_are_cached(server: FakeServer, tmp_path: Path) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (200, CSV)
    with make_downloader(server, tmp_path) as downloader:
        first = downloader.fetch("E0", 2025, current_season=2026)
        second = downloader.fetch("E0", 2025, current_season=2026)
    assert first.status is FetchStatus.DOWNLOADED
    assert second.status is FetchStatus.CACHED
    assert server.hits == ["/mmz4281/2526/E0.csv"]


def test_current_season_is_always_refreshed(server: FakeServer, tmp_path: Path) -> None:
    server.routes["/mmz4281/2627/E0.csv"] = (200, CSV)
    with make_downloader(server, tmp_path) as downloader:
        downloader.fetch("E0", 2026, current_season=2026)
        again = downloader.fetch("E0", 2026, current_season=2026)
    assert again.status is FetchStatus.DOWNLOADED
    assert server.hits.count("/mmz4281/2627/E0.csv") == 2


def test_force_bypasses_the_cache(server: FakeServer, tmp_path: Path) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (200, CSV)
    with make_downloader(server, tmp_path) as downloader:
        downloader.fetch("E0", 2025, current_season=2026)
        forced = downloader.fetch("E0", 2025, current_season=2026, force=True)
    assert forced.status is FetchStatus.DOWNLOADED
    assert len(server.hits) == 2


def test_404_is_reported_as_missing_not_failed(
    server: FakeServer, tmp_path: Path
) -> None:
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2026, current_season=2026)
    assert result.status is FetchStatus.MISSING
    assert not list(tmp_path.iterdir())


def test_html_error_page_is_rejected_and_not_saved(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (200, b"<html>Maintenance</html>")
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.FAILED
    assert "not a football-data results CSV" in result.detail
    assert not list(tmp_path.iterdir())


def test_server_errors_are_retried_then_reported(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (500, b"boom")
    with make_downloader(server, tmp_path) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.FAILED
    assert len(server.hits) == 2  # first attempt plus one retry
    assert not list(tmp_path.iterdir())


def test_unreachable_server_is_reported_not_raised(
    server: FakeServer, tmp_path: Path
) -> None:
    dead_url = "http://127.0.0.1:9/mmz4281"  # nothing listens on the discard port
    with make_downloader(server, tmp_path, base_url=dead_url) as downloader:
        result = downloader.fetch("E0", 2025, current_season=2026)
    assert result.status is FetchStatus.FAILED
    assert "Error" in result.detail


def test_download_all_covers_every_league_and_season(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/mmz4281/2425/E0.csv"] = (200, CSV)
    server.routes["/mmz4281/2526/E0.csv"] = (200, CSV)
    with make_downloader(server, tmp_path) as downloader:
        results = downloader.download_all(
            ["E0", "SP1"], [2024, 2025], current_season=2026
        )
    statuses = {(r.league, r.season): r.status for r in results}
    assert statuses == {
        ("E0", 2024): FetchStatus.DOWNLOADED,
        ("E0", 2025): FetchStatus.DOWNLOADED,
        ("SP1", 2024): FetchStatus.MISSING,
        ("SP1", 2025): FetchStatus.MISSING,
    }


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def cli_args(server: FakeServer, raw_dir: Path) -> list[str]:
    return [
        "--leagues",
        "E0","--from-season",
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
def test_cli_exit_code_is_zero_on_success(server: FakeServer, tmp_path: Path) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (200, CSV)
    assert main(cli_args(server, tmp_path / "raw")) == 0
    assert (tmp_path / "raw" / "E0_2526.csv").exists()


@pytest.mark.usefixtures("isolated_logging")
def test_cli_exit_code_is_one_when_a_file_fails(
    server: FakeServer, tmp_path: Path
) -> None:
    server.routes["/mmz4281/2526/E0.csv"] = (200, b"<html>oops</html>")
    assert main(cli_args(server, tmp_path / "raw")) == 1


@pytest.mark.usefixtures("isolated_logging")
def test_cli_rejects_reversed_season_range(server: FakeServer, tmp_path: Path) -> None:
    args = cli_args(server, tmp_path)
    args[args.index("--to-season") + 1] = "2020"
    with pytest.raises(SystemExit) as excinfo:
        main(args)
    assert excinfo.value.code == 2
