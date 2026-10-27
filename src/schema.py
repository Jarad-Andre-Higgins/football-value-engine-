"""Column definitions for the unified match table.

football-data.co.uk names its columns tersely (``PSCH`` is Pinnacle's closing
home-win odds). This module is the one place that maps those raw names to the
project's standard names, so a change in the source only touches this file.

Standard odds names read ``<book>_<outcome>`` before kick-off and
``<book>c_<outcome>`` at close, for example ``ps_h`` and ``psc_h``. Books are
``b365`` (Bet365), ``ps`` (Pinnacle), ``avg`` (market average) and ``max``
(market best price). Over/under 2.5 goals uses ``over25`` and ``under25``.
"""

from __future__ import annotations

from typing import Final

REQUIRED_RAW: Final[tuple[str, ...]] = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")

# Optional per-match facts: raw name -> standard name.
STAT_COLUMNS: Final[dict[str, str]] = {
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellows",
    "AY": "away_yellows",
    "HR": "home_reds",
    "AR": "away_reds",
}

# book -> (raw prefix before kick-off, raw prefix at close) for 1X2 odds.
_ONE_X_TWO_PREFIXES: Final[dict[str, tuple[str, str]]] = {
    "b365": ("B365", "B365C"),
    "ps": ("PS", "PSC"),
    "avg": ("Avg", "AvgC"),
    "max": ("Max", "MaxC"),
}
# The same for over/under 2.5 goals, where Pinnacle uses "P" and "PC".
_OVER_UNDER_PREFIXES: Final[dict[str, tuple[str, str]]] = {
    "b365": ("B365", "B365C"),
    "ps": ("P", "PC"),
    "avg": ("Avg", "AvgC"),
    "max": ("Max", "MaxC"),
}


def _build_odds_columns() -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Build the raw -> standard odds mapping and the list of 1X2 triples."""
    mapping: dict[str, str] = {}
    triples: list[tuple[str, str, str]] = []
    for book, (opening, closing) in _ONE_X_TWO_PREFIXES.items():
        for raw_prefix, name in ((opening, book), (closing, f"{book}c")):
            for outcome in "HDA":
                mapping[f"{raw_prefix}{outcome}"] = f"{name}_{outcome.lower()}"
            triples.append((f"{name}_h", f"{name}_d", f"{name}_a"))
    for book, (opening, closing) in _OVER_UNDER_PREFIXES.items():
        for raw_prefix, name in ((opening, book), (closing, f"{book}c")):
            mapping[f"{raw_prefix}>2.5"] = f"{name}_over25"
            mapping[f"{raw_prefix}<2.5"] = f"{name}_under25"
    return mapping, triples


_ODDS_MAPPING, _ONE_X_TWO_TRIPLES = _build_odds_columns()

#: Raw odds column -> standard column, for every book and market.
ODDS_COLUMNS: Final[dict[str, str]] = _ODDS_MAPPING
#: Standard ``(home, draw, away)`` column triples, used for overround checks.
ONE_X_TWO_SETS: Final[list[tuple[str, str, str]]] = _ONE_X_TWO_TRIPLES

IDENTITY_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "league",
    "season",
    "date",
    "kickoff_time",
    "home_team_id",
    "away_team_id",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
)

#: Every column of the unified match table, in output order.
MATCH_COLUMNS: Final[tuple[str, ...]] = (
    *IDENTITY_COLUMNS,
    *STAT_COLUMNS.values(),
    *ODDS_COLUMNS.values(),
)
