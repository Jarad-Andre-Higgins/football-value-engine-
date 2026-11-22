"""Upcoming matches: fixtures, prices and pre-match features for the coupon.

The coupon needs three things about matches that have not been played: who plays
whom and when, what the bookmakers charge, and the features the model reads.
This step builds all three from ``fixtures.csv``, the file football-data.co.uk
publishes for the coming days. Its prices are the opening prices that fill the
``b365_*`` and ``ps_*`` columns of the match table, so a price here means what it
meant in the backtest::

    fixtures.csv --(team registry)--> team ids and match ids
    match table + fixtures --(src.features)--> features of every fixture
    fixtures.csv --> one row per price: Bet365, Pinnacle, market average, best

Nothing from the future is used. A fixture's features come from the matches
played before it, exactly as they did when the model was trained, and a fixture
changes no other match's features. Two limits come with that. Results that are
not in the match table yet (last weekend's, until ``python -m src.ingest`` has
been run again) are not seen. And a team that plays twice in the window has its
second fixture computed as if the first had not been played. Both are reported
as notes.

Usage::

    python -m src.upcoming --download                    # the matches of today
    python -m src.upcoming --download --date 2026-11-21 --days 3
    python -m src.upcoming --fixtures my_fixtures.csv    # a file you already have

Matches that have already kicked off are left out (``--include-started`` keeps
them), because they can no longer be bet on.

then::

    python -m src.coupon --fixtures data/processed/upcoming.parquet \\
        --odds data/processed/upcoming_odds.csv --bet-at bet365 \\
        --reference-bookmaker pinnacle
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.bundle import DEFAULT_MODEL_PATH, ModelBundle, load_bundle, predict_rates
from src.config import PROCESSED_DIR, RAW_DIR
from src.dixon_coles import DEFAULT_MAX_GOALS
from src.elo import EloConfig
from src.enrich import ENRICHED_OUTPUT
from src.entities import DEFAULT_REGISTRY_PATH, TeamRegistry
from src.features import REQUIRED_COLUMNS, RESULT_COLUMNS, build_features
from src.ingest import USER_AGENT
from src.labels import describe_selection
from src.leagues import LEAGUES, current_season_start
from src.logging_config import setup_logging
from src.markets import fixture_markets
from src.odds_history import BOOKS, FAIR_BOOKS, fair_view, odds_long
from src.schema import ODDS_COLUMNS

logger = logging.getLogger("src.upcoming")

FIXTURES_URL: Final[str] = "https://www.football-data.co.uk/fixtures.csv"
# Not in data/raw/football_data: python -m src.pipeline reads every file in that folder.
FIXTURES_PATH: Final[Path] = RAW_DIR / "fixtures" / "fixtures.csv"
UPCOMING_OUTPUT: Final[Path] = PROCESSED_DIR / "upcoming.parquet"
UPCOMING_ODDS_OUTPUT: Final[Path] = PROCESSED_DIR / "upcoming_odds.csv"
IDENTITY_RAW: Final[tuple[str, ...]] = ("Div", "Date", "Time", "HomeTeam", "AwayTeam")
REQUIRED_RAW: Final[tuple[str, ...]] = ("Div", "Date", "HomeTeam", "AwayTeam")
#: Raw price column -> standard name, opening prices only: a fixture has no closing price.
OPENING_ODDS: Final[dict[str, str]] = {
    raw: name for raw, name in ODDS_COLUMNS.items() if name.split("_", 1)[0] in BOOKS
}
#: Bookmaker codes of the match table -> the names the coupon shows.
BOOK_LABELS: Final[dict[str, str]] = {
    "b365": "bet365",
    "ps": "pinnacle",
    "avg": "market_avg",
    "max": "market_best",
}
KEY: Final[list[str]] = ["match_id", "market_id", "selection"]
#: The markets the fixtures file has prices for, with their selections in display order.
COMPARED_MARKETS: Final[dict[str, tuple[str, ...]]] = {
    "1X2": ("home", "draw", "away"),
    "OU:2.5": ("over", "under"),
}
MARKET_TITLES: Final[dict[str, str]] = {"1X2": "1X2", "OU:2.5": "O/U 2.5"}
#: football-data.co.uk lists kick-off times in UK time.
SOURCE_ZONE: Final[str] = "Europe/London"
DEFAULT_LOCAL_ZONE: Final[str] = "Africa/Douala"
STALE_DAYS: Final[int] = 10  # the match table may end this long before the fixtures
RECENT_DAYS: Final[int] = 30  # window in which xG coverage is checked
MIN_RECENT_XG: Final[float] = 0.8
THIN_HISTORY: Final[int] = 5  # matches a team needs in the table to have form
TOP_DIFFERENCES: Final[int] = 5


@dataclass(frozen=True)
class Upcoming:
    """Everything the coupon needs about the matches of one window.

    Attributes:
        fixtures: One row per fixture, in the layout of the match table (names,
            ids, ``kickoff_time`` and the opening prices).
        features: One row per fixture in the layout of ``features.parquet``; its
            targets are empty because nothing has been played.
        odds: One row per price (``match_id``, ``bookmaker``, ``market_id``,
            ``selection``, ``odds``), the layout of :mod:`src.coupon`.
        notes: Things the operator should know, in words.
    """

    fixtures: pd.DataFrame
    features: pd.DataFrame
    odds: pd.DataFrame
    notes: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Getting and reading the fixtures file
# --------------------------------------------------------------------------- #
def looks_like_fixtures_csv(content: bytes) -> bool:
    """Whether ``content`` starts with the header of football-data's fixtures file.

    Guards against saving an HTML error page as the fixtures.
    """
    header = content.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace")
    columns = {column.strip() for column in header.split(",")}
    return set(REQUIRED_RAW) <= columns


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_fixtures(
    path: Path = FIXTURES_PATH,
    url: str = FIXTURES_URL,
    *,
    session: requests.Session | None = None,
    timeout: tuple[float, float] = (10.0, 30.0),
) -> Path:
    """Download the fixtures file and save it (a half-written file is never left).

    Args:
        path: Where to save it.
        url: Where to get it.
        session: An HTTP session; one with retries is made when not given.
        timeout: ``(connect, read)`` timeouts in seconds.

    Returns:
        ``path``.

    Raises:
        OSError: If the request fails or the server answers with an error.
        ValueError: If the answer is not a fixtures file.
    """
    own = session is None
    http = session or _session()
    try:
        response = http.get(url, timeout=timeout)
    finally:
        if own:
            http.close()
    if not response.ok:
        raise OSError(f"{url}: HTTP {response.status_code}")
    if not looks_like_fixtures_csv(response.content):
        raise ValueError(f"{url}: the answer is not a football-data fixtures file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_bytes(response.content)
    os.replace(temp, path)
    return path


def read_fixtures(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read ``fixtures.csv`` into the match table's names, for the five leagues.

    Args:
        path: The fixtures file.

    Returns:
        The fixtures sorted by date and kick-off, with ``league``, ``date``,
        ``kickoff_time``, the raw team names and the opening prices under their
        standard names (a price the file does not have is left out), and one note
        per kind of row that was left out.

    Raises:
        ValueError: If the file has no usable header or lacks a required column.
    """
    wanted = {*IDENTITY_RAW, *OPENING_ODDS}
    raw = pd.read_csv(
        path,
        usecols=lambda column: str(column).strip() in wanted,
        encoding="utf-8-sig",
        encoding_errors="replace",
        dtype=str,
    )
    raw.columns = [str(column).strip() for column in raw.columns]
    missing = [column for column in REQUIRED_RAW if column not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")
    for column in IDENTITY_RAW:
        if column in raw.columns:
            raw[column] = raw[column].astype("string").str.strip()
    notes: list[str] = []
    rows = raw.dropna(subset=list(REQUIRED_RAW))
    rows = rows[rows["Div"].str.upper().isin(set(LEAGUES))]
    dates = pd.to_datetime(rows["Date"], dayfirst=True, format="mixed", errors="coerce")
    unreadable = dates.isna().to_numpy()
    if unreadable.any():
        notes.append(f"{int(unreadable.sum())} fixtures ignored: unreadable date")
    rows, dates = rows[~unreadable], dates[~unreadable]
    table = pd.DataFrame(
        {
            "league": rows["Div"].str.upper().to_numpy(dtype=object),
            "date": dates.dt.normalize().to_numpy(),
            "kickoff_time": rows["Time"].to_numpy(dtype=object)
            if "Time" in rows.columns
            else np.full(len(rows), pd.NA, dtype=object),
            "home_team_raw": rows["HomeTeam"].to_numpy(dtype=object),
            "away_team_raw": rows["AwayTeam"].to_numpy(dtype=object),
        }
    )
    for source, name in OPENING_ODDS.items():
        if source in rows.columns:
            table[name] = pd.to_numeric(rows[source], errors="coerce").to_numpy(
                dtype="float64"
            )
    ordered = table.sort_values(
        ["date", "kickoff_time", "league", "home_team_raw"], kind="stable"
    )
    return ordered.reset_index(drop=True), notes


def select_window(
    fixtures: pd.DataFrame, start: pd.Timestamp, days: int
) -> pd.DataFrame:
    """The fixtures from ``start`` (a date) on, for ``days`` days.

    Raises:
        ValueError: If ``days`` is less than 1.
    """
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    first = pd.Timestamp(start).normalize()
    end = first + pd.Timedelta(days, unit="D")
    inside = (fixtures["date"] >= first) & (fixtures["date"] < end)
    return fixtures[inside].reset_index(drop=True)


def resolve_fixtures(
    fixtures: pd.DataFrame, registry: TeamRegistry
) -> tuple[pd.DataFrame, list[str]]:
    """Give each fixture team ids, names, a season and the ``match_id`` of the match table.

    A fixture with a team the registry does not know is left out and reported,
    because one unknown spelling must not cost the whole coupon. Fix it by adding
    the spelling as an alias in ``data/reference/teams.yaml``.

    Returns:
        The resolved fixtures and the notes about those left out or repeated.
    """
    countries = {code: league.country for code, league in LEAGUES.items()}
    spellings = pd.concat(
        [
            fixtures[["league", "home_team_raw"]].rename(
                columns={"home_team_raw": "raw"}
            ),
            fixtures[["league", "away_team_raw"]].rename(
                columns={"away_team_raw": "raw"}
            ),
        ]
    ).drop_duplicates()
    ids: dict[tuple[str, str], str | None] = {}
    problems: list[str] = []
    for league, raw in spellings.itertuples(index=False):
        team_id = registry.resolve(raw, country=countries.get(league))
        ids[(league, raw)] = team_id
        if team_id is None:
            close = registry.suggest(raw, country=countries.get(league))
            hint = (
                "; ".join(f"{s.name} ({s.score:.2f})" for s in close)
                or "no close match"
            )
            problems.append(f"{league} {raw!r}: closest {hint}")
    home = [
        ids[(lg, raw)] for lg, raw in zip(fixtures["league"], fixtures["home_team_raw"])
    ]
    away = [
        ids[(lg, raw)] for lg, raw in zip(fixtures["league"], fixtures["away_team_raw"])
    ]
    frame = fixtures.assign(
        home_team_id=pd.Series(home, index=fixtures.index, dtype="object"),
        away_team_id=pd.Series(away, index=fixtures.index, dtype="object"),
    )
    known = frame["home_team_id"].notna() & frame["away_team_id"].notna()
    notes: list[str] = []
    if not known.all():
        notes.append(
            f"{int((~known).sum())} fixtures left out, team name not in the registry "
            "(add it as an alias in data/reference/teams.yaml): " + " | ".join(problems)
        )
    frame = frame[known].copy()
    frame["home_team"] = [registry.name_of(team) for team in frame["home_team_id"]]
    frame["away_team"] = [registry.name_of(team) for team in frame["away_team_id"]]
    frame["season"] = [current_season_start(day.date()) for day in frame["date"]]
    frame["match_id"] = [
        f"{league}-{season}-{day:%Y%m%d}-{home}-{away}"
        for league, season, day, home, away in zip(
            frame["league"],
            frame["season"],
            frame["date"],
            frame["home_team_id"],
            frame["away_team_id"],
        )
    ]  # built row by row so that a window with no known team (no rows) still works
    repeated = frame["match_id"].duplicated().to_numpy()
    if repeated.any():
        notes.append(
            f"{int(repeated.sum())} fixtures appear twice in the file: kept once"
        )
        frame = frame[~repeated]
    return frame.reset_index(drop=True), notes


# --------------------------------------------------------------------------- #
# Features and prices
# --------------------------------------------------------------------------- #
def upcoming_features(
    history: pd.DataFrame,
    fixtures: pd.DataFrame,
    elo_config: EloConfig | None = None,
) -> pd.DataFrame:
    """Pre-match features of fixtures, from the played matches before them.

    The fixtures are appended to the played matches without a result and the
    ordinary feature pipeline is run, so they are built by the very code that built
    the training rows. A fixture's features equal those the model would have seen at
    kick-off, and appending fixtures leaves every played match's features as they
    were.

    Args:
        history: The enriched match table, played matches only.
        fixtures: Fixtures with ``match_id``, ``league``, ``season``, ``date``,
            ``home_team_id`` and ``away_team_id``.
        elo_config: Elo constants; defaults to :class:`~src.elo.EloConfig`.

    Returns:
        One row per fixture in the layout of ``features.parquet``; the targets are
        empty.

    Raises:
        ValueError: If a column is missing, a ``match_id`` is already in
            ``history`` or a team would play twice on one date.
    """
    identity = [column for column in REQUIRED_COLUMNS if column not in RESULT_COLUMNS]
    missing = [column for column in identity if column not in fixtures.columns]
    if missing:
        raise ValueError(f"fixtures is missing columns: {missing}")
    known = set(history["match_id"])
    already = [match for match in fixtures["match_id"] if match in known]
    if already:
        raise ValueError(
            f"{len(already)} fixtures are already in the history, e.g. {already[0]}"
        )
    kept = [
        column
        for column in dict.fromkeys([*identity, *RESULT_COLUMNS])
        if column in history.columns
    ]
    past = history.loc[:, kept].copy()
    results = [column for column in RESULT_COLUMNS if column in past.columns]
    past[results] = past[results].astype("float64")  # fixtures hold missing results
    ahead = fixtures.loc[:, identity].reindex(columns=kept)
    ahead[results] = np.nan
    combined = pd.concat([past, ahead], ignore_index=True)
    table = build_features(combined, elo_config)
    wanted = table["match_id"].isin(set(fixtures["match_id"]))
    return table[wanted].reset_index(drop=True)


def odds_table(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Every complete opening quote of the fixtures, one row per price.

    A bookmaker's quote of a market is kept only when every selection of it is
    present and above 1.0, so the reference prices can be de-margined. Bookmakers
    carry the names of :data:`BOOK_LABELS`.
    """
    offers = odds_long(fixtures, "early", tuple(BOOK_LABELS))
    offers = offers.assign(bookmaker=offers["bookmaker"].map(BOOK_LABELS))
    return offers.sort_values(
        ["match_id", "bookmaker", "market_id", "selection"], kind="stable"
    ).reset_index(drop=True)


def history_notes(
    history: pd.DataFrame, fixtures: pd.DataFrame, start: pd.Timestamp
) -> list[str]:
    """Reasons to doubt the features: a stale match table, missing xG, new teams."""
    notes: list[str] = []
    last = history["date"].max()
    if pd.isna(last):
        return ["The match table has no matches"]
    gap = (pd.Timestamp(start) - last).days
    if gap > STALE_DAYS:
        notes.append(
            f"The match table ends on {last:%Y-%m-%d}, {gap} days before the first "
            "fixture: later results are missing, so form and Elo are out of date. "
            "Refresh with python -m src.ingest, python -m src.pipeline and "
            "python -m src.enrich"
        )
    recent = history[history["date"] >= last - pd.Timedelta(RECENT_DAYS, unit="D")]
    if {"home_xg", "away_xg"} <= set(recent.columns) and len(recent):
        share = float(recent[["home_xg", "away_xg"]].notna().all(axis=1).mean())
        if share < MIN_RECENT_XG:
            notes.append(
                f"Only {share:.0%} of the {len(recent)} matches of the last "
                f"{RECENT_DAYS} days in the table have xG, so recent form leans on "
                "older matches"
            )
    played = pd.concat(
        [history["home_team_id"], history["away_team_id"]]
    ).value_counts()
    teams = pd.concat([fixtures["home_team_id"], fixtures["away_team_id"]]).unique()
    thin = sorted(str(team) for team in teams if played.get(team, 0) < THIN_HISTORY)
    if thin:
        notes.append(
            f"Teams with fewer than {THIN_HISTORY} matches in the table: "
            f"{', '.join(thin)}; their features are unreliable"
        )
    return notes


def schedule_notes(fixtures: pd.DataFrame) -> list[str]:
    """A note for each team that plays more than once in the window."""
    teams = pd.concat([fixtures["home_team_id"], fixtures["away_team_id"]])
    counts = teams.value_counts()
    twice = sorted(str(team) for team in counts[counts > 1].index)
    if not twice:
        return []
    names = ", ".join(twice)
    return [
        (
            f"{names} play more than once in the window: the later fixture is "
            "built as if the earlier one had not been played"
        )
    ]


def build_upcoming(
    history: pd.DataFrame,
    raw_fixtures: pd.DataFrame,
    registry: TeamRegistry,
    *,
    start: pd.Timestamp,
    days: int = 1,
    now: datetime | None = None,
    elo_config: EloConfig | None = None,
) -> Upcoming:
    """Fixtures, features and prices for the matches from ``start`` on, for ``days`` days.

    Args:
        history: The enriched match table (``matches_enriched.parquet``).
        raw_fixtures: From :func:`read_fixtures`.
        registry: The team registry.
        start: First date of the window.
        days: Length of the window in days.
        now: When given (with a time zone), fixtures that have already kicked off
            at that moment are left out, because they can no longer be bet on.
        elo_config: Elo constants; defaults to :class:`~src.elo.EloConfig`.

    Returns:
        The :class:`Upcoming` matches.

    Raises:
        ValueError: If no fixture is left to price, or the features cannot be built.
    """
    first = pd.Timestamp(start).normalize()
    window = select_window(raw_fixtures, first, days)
    if window.empty:
        covered = (
            f"it covers {raw_fixtures['date'].min():%Y-%m-%d} to "
            f"{raw_fixtures['date'].max():%Y-%m-%d}"
            if len(raw_fixtures)
            else "it has no fixtures of the top five leagues"
        )
        raise ValueError(
            f"no fixtures on {first:%Y-%m-%d} (for {days} day(s)); {covered}"
        )
    started = 0
    if now is not None:
        window, started = drop_started(window, now)
        if window.empty:
            raise ValueError(
                f"all {started} fixtures of the window have already kicked off "
                f"(it is {now:%Y-%m-%d %H:%M %Z}); try --days 2 or a later --date"
            )
    resolved, notes = resolve_fixtures(window, registry)
    if started:
        notes.append(f"{started} fixtures have already kicked off: left out")
    if resolved.empty:
        raise ValueError("; ".join(notes) or "no fixture has two known teams")
    played = history[history["home_goals"].notna() & history["away_goals"].notna()]
    if len(played) < len(history):
        notes.append(
            f"{len(history) - len(played)} matches without a score were ignored"
        )
    done = resolved["match_id"].isin(set(played["match_id"])).to_numpy()
    if done.any():
        notes.append(
            f"{int(done.sum())} fixtures are already in the match table, so they have "
            "been played: left out"
        )
        resolved = resolved[~done].reset_index(drop=True)
    if resolved.empty:
        raise ValueError("every fixture of the window is already in the match table")
    notes.extend(history_notes(played, resolved, first))
    notes.extend(schedule_notes(resolved))
    features = upcoming_features(played, resolved, elo_config)
    odds = odds_table(resolved)
    unpriced = ~resolved["match_id"].isin(set(odds["match_id"]))
    if unpriced.any():
        notes.append(
            f"No complete prices for {int(unpriced.sum())} of {len(resolved)} fixtures"
        )
    return Upcoming(resolved, features, odds, tuple(notes))


# --------------------------------------------------------------------------- #
# The model against the market
# --------------------------------------------------------------------------- #
def kickoff_moment(day: pd.Timestamp, clock: object) -> datetime | None:
    """A kick-off as a moment in time; ``None`` when the file gives no readable time.

    The file's times are UK times, so the result carries the UK zone (summer time
    included).
    """
    if clock is None or pd.isna(clock) or not str(clock).strip():
        return None
    try:
        hour, minute = (int(part) for part in str(clock).strip().split(":")[:2])
        return datetime(
            day.year, day.month, day.day, hour, minute, tzinfo=ZoneInfo(SOURCE_ZONE)
        )
    except (ValueError, ZoneInfoNotFoundError):
        return None


def drop_started(fixtures: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, int]:
    """The fixtures that have not kicked off at ``now``, and how many had.

    A fixture whose time is missing or unreadable is kept: it cannot be shown to
    have started.

    Args:
        fixtures: From :func:`read_fixtures`.
        now: The present moment; must carry a time zone.

    Returns:
        The fixtures still to be played and the number left out.
    """
    started = np.array(
        [
            moment is not None and moment <= now
            for moment in (
                kickoff_moment(day, clock)
                for day, clock in zip(fixtures["date"], fixtures["kickoff_time"])
            )
        ],
        dtype=bool,
    )
    return fixtures[~started].reset_index(drop=True), int(started.sum())


def kickoff_label(day: pd.Timestamp, clock: object, local_zone: str | None) -> str:
    """A kick-off as ``15:00 UK (16:00 Douala)``; the file's time is UK time."""
    moment = kickoff_moment(day, clock)
    if moment is None:
        if clock is None or pd.isna(clock) or not str(clock).strip():
            return "time not given"
        return str(clock).strip()
    text = str(clock).strip()
    label = f"{text} UK"
    if local_zone:
        try:
            local = moment.astimezone(ZoneInfo(local_zone))
        except ZoneInfoNotFoundError:
            return label
        place = local_zone.rsplit("/", 1)[-1].replace("_", " ")
        label += f" ({local:%H:%M} {place})"
    return label


def compare_with_market(upcoming: Upcoming, bundle: ModelBundle) -> pd.DataFrame:
    """The model's probability of every 1X2 and over/under 2.5 selection beside the market's.

    Args:
        upcoming: From :func:`build_upcoming`.
        bundle: The deployable model.

    Returns:
        One row per fixture and selection: ``match_id``, ``market_id``,
        ``selection``, ``p_model``, ``p_market`` (the sharpest book's margin-free
        probability, from ``source``), ``odds`` (Bet365's price), ``price_gap``
        (Bet365's price against the market, ``odds * p_market - 1``),
        ``model_edge`` (``odds * p_model - 1``) and the fixture's goal rates
        ``lam_home`` and ``lam_away``.
    """
    features = upcoming.features
    rates = predict_rates(bundle, features)
    priced = fixture_markets(
        rates,
        bundle.calibration.rho,
        match_ids=features["match_id"],
        max_goals=DEFAULT_MAX_GOALS,
    )
    shown = priced[priced["market_id"].isin(set(COMPARED_MARKETS))]
    table = shown[KEY + ["probability"]].rename(columns={"probability": "p_model"})
    fair = fair_view(upcoming.fixtures, "early", FAIR_BOOKS).rename(
        columns={"p_fair": "p_market"}
    )
    bet365 = odds_long(upcoming.fixtures, "early", ("b365",))[[*KEY, "odds"]]
    table = table.merge(fair, on=KEY, how="left").merge(bet365, on=KEY, how="left")
    table["price_gap"] = table["odds"] * table["p_market"] - 1.0
    table["model_edge"] = table["odds"] * table["p_model"] - 1.0
    goals = pd.DataFrame(
        {
            "match_id": features["match_id"].to_numpy(),
            "lam_home": rates["lam_home"].to_numpy(),
            "lam_away": rates["lam_away"].to_numpy(),
        }
    )
    order = {match: place for place, match in enumerate(upcoming.fixtures["match_id"])}
    table = table.merge(goals, on="match_id", how="left")
    table["_place"] = table["match_id"].map(order)
    table["_slot"] = [
        COMPARED_MARKETS[market].index(selection)
        for market, selection in zip(table["market_id"], table["selection"])
    ]
    table["_market"] = table["market_id"].map(
        {market: place for place, market in enumerate(COMPARED_MARKETS)}
    )
    ordered = table.sort_values(["_place", "_market", "_slot"], kind="stable")
    return ordered.drop(columns=["_place", "_slot", "_market"]).reset_index(drop=True)


def _join(values: Sequence[float], template: str) -> str:
    return " / ".join(
        "-" if np.isnan(value) else template.format(value) for value in values
    )


def format_comparison(
    upcoming: Upcoming, comparison: pd.DataFrame, local_zone: str | None = None
) -> str:
    """The model's view of every fixture beside the market's, as plain text.

    Percentages are chances of each selection in the order shown (home / draw /
    away, or over / under). Bet365 prices are decimal odds.
    """
    lines: list[str] = []
    for fixture in upcoming.fixtures.itertuples(index=False):
        rows = comparison[comparison["match_id"] == fixture.match_id]
        if rows.empty:
            continue
        league = LEAGUES[fixture.league].name
        when = kickoff_label(fixture.date, fixture.kickoff_time, local_zone)
        goals = rows.iloc[0]
        lines.append(f"{league} · {fixture.home_team} v {fixture.away_team} · {when}")
        lines.append(
            f"   model goal rates {goals['lam_home']:.2f} - {goals['lam_away']:.2f}"
        )
        for market, selections in COMPARED_MARKETS.items():
            part = rows[rows["market_id"] == market].set_index("selection")
            part = part.reindex(list(selections))
            if part["p_model"].isna().all():
                continue
            source = part["p_market"].notna().any()
            lines.append(
                f"   {MARKET_TITLES[market]:<8} model "
                f"{_join((100 * part['p_model']).tolist(), '{:.0f}')}"
                + (
                    f"   market {_join((100 * part['p_market']).tolist(), '{:.0f}')}"
                    if source
                    else "   market -"
                )
                + f"   Bet365 {_join(part['odds'].tolist(), '{:.2f}')}"
            )
        lines.append("")
    lines.extend(_top_lines(upcoming, comparison))
    return "\n".join(lines).rstrip()


def _top_lines(upcoming: Upcoming, comparison: pd.DataFrame) -> list[str]:
    """Where the model differs most from the market, and where Bet365 pays most."""
    names = upcoming.fixtures.set_index("match_id")[["home_team", "away_team"]]
    table = comparison.dropna(subset=["p_market"]).copy()
    if table.empty:
        return []

    def words(row: pd.Series) -> str:
        home, away = names.loc[row["match_id"], ["home_team", "away_team"]]
        return f"{describe_selection(row['market_id'], row['selection'], home, away)} ({home} v {away})"

    lines: list[str] = []
    table["difference"] = table["p_model"] - table["p_market"]
    # One line per match and market: in a two-way market, "over 2.5 is 33 points
    # lower" and "under 2.5 is 33 points higher" are one disagreement, not two.
    widest = table["difference"].abs().sort_values(ascending=False, kind="stable").index
    far = (
        table.loc[widest]
        .drop_duplicates(["match_id", "market_id"])
        .head(TOP_DIFFERENCES)
    )
    lines.append("Where the model differs most from the market (percentage points):")
    for _, row in far.iterrows():
        lines.append(
            f"   {row['difference'] * 100:+.1f}  {words(row)}: model {row['p_model']:.0%}, "
            f"market {row['p_market']:.0%}"
        )
    priced = table.dropna(subset=["odds"])
    if not priced.empty:
        top = priced.sort_values("price_gap", ascending=False, kind="stable").head(
            TOP_DIFFERENCES
        )
        lines.append("")
        lines.append(
            "Bet365 prices furthest above the market (the engine bets from +4% at model weight 0):"
        )
        for _, row in top.iterrows():
            lines.append(
                f"   {row['price_gap'] * 100:+.1f}%  {words(row)} at {row['odds']:.2f}"
            )
    return lines


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def format_schedule(
    fixtures: pd.DataFrame, odds: pd.DataFrame, local_zone: str | None
) -> str:
    """One line per fixture: league, kick-off, teams and the bookmakers with prices."""
    books = odds.groupby("match_id")["bookmaker"].agg(
        lambda s: " ".join(sorted(set(s)))
    )
    lines = []
    for fixture in fixtures.itertuples(index=False):
        when = kickoff_label(fixture.date, fixture.kickoff_time, local_zone)
        lines.append(
            f"{fixture.league:<4} {fixture.date:%Y-%m-%d} {when:<26} "
            f"{fixture.home_team} v {fixture.away_team}  "
            f"[prices: {books.get(fixture.match_id, 'none')}]"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixtures, prices and features of the coming matches, for the coupon."
    )
    parser.add_argument(
        "--download", action="store_true", help="fetch the latest fixtures.csv first"
    )
    parser.add_argument("--fixtures", type=Path, default=FIXTURES_PATH)
    parser.add_argument("--url", default=FIXTURES_URL)
    parser.add_argument(
        "--date",
        help="first day of the window, YYYY-MM-DD (default: today, UTC)",
    )
    parser.add_argument(
        "--days", type=int, default=1, help="length of the window in days (default: 1)"
    )
    parser.add_argument(
        "--include-started",
        action="store_true",
        help="keep matches that have already kicked off (they are left out by default)",
    )
    parser.add_argument(
        "--now",
        help="pretend it is this moment, YYYY-MM-DDTHH:MM in UTC (default: the clock)",
    )
    parser.add_argument("--history", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=UPCOMING_OUTPUT)
    parser.add_argument("--odds-output", type=Path, default=UPCOMING_ODDS_OUTPUT)
    parser.add_argument(
        "--local-zone",
        default=DEFAULT_LOCAL_ZONE,
        help="time zone for kick-off times next to UK time (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")
    if args.date is not None:
        try:
            pd.Timestamp(args.date)
        except ValueError:
            parser.error(f"--date must look like 2026-11-21, got {args.date!r}")
    if args.now is not None:
        try:
            pd.Timestamp(args.now)
        except ValueError:
            parser.error(f"--now must look like 2026-11-21T18:50, got {args.now!r}")
    return args


def _clock(text: str | None) -> datetime:
    """``--now`` as a moment in UTC, or the present moment when not given."""
    if text is None:
        return datetime.now(UTC)
    moment = pd.Timestamp(text)
    if moment.tzinfo is None:
        moment = moment.tz_localize("UTC")
    return moment.to_pydatetime()


def main(argv: list[str] | None = None) -> int:
    """Build the upcoming matches and return the process exit code.

    Writes ``upcoming.parquet`` and ``upcoming_odds.csv``, the inputs of
    ``python -m src.coupon``, and logs the schedule and the model's view of it.

    Returns ``0`` on success and ``1`` when an input is missing or unusable, or
    when no fixture is left to price.
    """
    args = _parse_args(argv)
    log = setup_logging("upcoming")
    if args.download:
        try:
            download_fixtures(args.fixtures, args.url)
        except (OSError, ValueError) as exc:
            log.error("Cannot download the fixtures: %s", exc)
            return 1
        log.info("Downloaded %s", args.fixtures)
    for label, path in (
        ("Fixtures file", args.fixtures),
        ("Match table", args.history),
        ("Team registry", args.registry),
    ):
        if not path.exists():
            hint = " (use --download)" if path == args.fixtures else ""
            log.error("%s not found: %s%s", label, path, hint)
            return 1
    now = _clock(args.now)
    start = pd.Timestamp(args.date) if args.date else pd.Timestamp(now.date())
    try:
        registry = TeamRegistry.from_yaml(args.registry)
        raw, read_notes = read_fixtures(args.fixtures)
        history = pd.read_parquet(args.history)
        upcoming = build_upcoming(
            history,
            raw,
            registry,
            start=start,
            days=args.days,
            now=None if args.include_started else now,
        )
    except (OSError, ValueError, KeyError) as exc:
        log.error("Cannot build the upcoming matches: %s", exc)
        return 1
    notes = [*read_notes, *upcoming.notes]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    upcoming.features.merge(
        upcoming.fixtures[["match_id", "kickoff_time"]], on="match_id", how="left"
    ).to_parquet(args.output, index=False)
    args.odds_output.parent.mkdir(parents=True, exist_ok=True)
    upcoming.odds.to_csv(args.odds_output, index=False)
    log.info(
        "%d fixtures from %s for %d day(s) (clock %s UTC); %d prices from %s",
        len(upcoming.fixtures),
        f"{start:%Y-%m-%d}",
        args.days,
        f"{now:%Y-%m-%d %H:%M}",
        len(upcoming.odds),
        ", ".join(sorted(set(upcoming.odds["bookmaker"]))) or "no bookmaker",
    )
    log.info(
        "Schedule (kick-off in UK time, then %s):\n%s",
        args.local_zone,
        format_schedule(upcoming.fixtures, upcoming.odds, args.local_zone),
    )
    try:
        bundle = load_bundle(args.model)
    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        EOFError,
        AttributeError,
        ImportError,
    ) as exc:
        log.warning("No model view: cannot load %s (%s)", args.model, exc)
    else:
        comparison = compare_with_market(upcoming, bundle)
        log.info(
            "The model against the market:\n%s",
            format_comparison(upcoming, comparison, args.local_zone),
        )
    for note in notes:
        log.warning("Data note: %s", note)
    log.info("Wrote %s and %s", args.output, args.odds_output)
    books = set(upcoming.odds["bookmaker"])
    reference = next(
        (name for name in ("pinnacle", "market_avg") if name in books), None
    )
    log.info(
        "Next: python -m src.coupon --fixtures %s --odds %s --bet-at bet365%s",
        args.output,
        args.odds_output,
        f" --reference-bookmaker {reference}" if reference else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
