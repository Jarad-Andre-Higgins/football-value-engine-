"""League registry and season helpers for football-data.co.uk.

football-data.co.uk publishes one CSV per league and season at
`<base>/<season code>/<division code>.csv`, for example `2526/E0.csv` for
the 2025/26 Premier League. A season is identified everywhere in this project
by its *start year* (2025 means 2025/26).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final


@dataclass(frozen=True)
class League:
    """A competition available from football-data.co.uk.

    Attributes:
        code: football-data.co.uk division code, such as `E0`.
        name: Canonical league name used throughout the project.
        country: Country the league is played in.
    """

    code: str
    name: str
    country: str


LEAGUES: Final[dict[str, League]] = {
    "E0": League("E0", "Premier League", "England"),
    "SP1": League("SP1", "La Liga", "Spain"),
    "I1": League("I1", "Serie A", "Italy"),
    "D1": League("D1", "Bundesliga", "Germany"),
    "F1": League("F1", "Ligue 1", "France"),
}

TOP_FIVE: Final[tuple[str, ...]] = tuple(LEAGUES)


def season_code(start_year: int) -> str:
    """Return the football-data.co.uk season folder, e.g. `2025` -> `"2526"`."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """Return a human-readable season, e.g. `2025` -> `"2025/26"`."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    """Return the start year of the season in progress.

    Seasons roll over on 1 July, so October 2026 belongs to the 2026/27 season
    and March 2027 still belongs to it.

    Args:
        today: Date to evaluate; defaults to today's date.
    """
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1
