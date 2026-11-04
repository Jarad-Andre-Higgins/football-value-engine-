"""Tests for the plain-English selection names."""

from __future__ import annotations

import re

import pytest

from src.labels import describe_selection
from src.markets import market_set

HOME, AWAY = "Arsenal", "Chelsea"


def test_every_selection_has_its_own_specific_description() -> None:
    texts: dict[str, list[str]] = {}
    for row in market_set().catalog.itertuples(index=False):
        text = describe_selection(row.market_id, row.selection, HOME, AWAY)
        assert text != f"{row.market_id}: {row.selection}", (
            row.market_id,
            row.selection,
        )
        assert text.strip()
        assert "_" not in text, text  # no raw ids leak into the words
        texts.setdefault(row.market_id, []).append(text)
    for market_id, described in texts.items():
        assert len(set(described)) == len(described), market_id


@pytest.mark.parametrize(
    ("market", "selection", "expected"),
    [
        ("1X2", "home", "Arsenal to win"),
        ("1X2", "draw", "Draw"),
        ("1X2", "away", "Chelsea to win"),
        ("DC", "home_draw", "Arsenal or draw"),
        ("DC", "home_away", "Arsenal or Chelsea"),
        ("DC", "draw_away", "Draw or Chelsea"),
        ("DNB", "away", "Chelsea (draw no bet)"),
        ("OU:2.5", "over", "Over 2.5 goals"),
        ("OU:0.5", "under", "Under 0.5 goals"),
        ("BTTS", "yes", "Both teams to score: Yes"),
        ("OE", "odd", "Total goals: odd"),
        ("TG", "3", "Total goals: exactly 3"),
        ("TG", "6+", "Total goals: 6 or more"),
        ("GB4", "2-3", "Total goals: 2-3"),
        ("TT_HOME:1.5", "over", "Arsenal over 1.5 goals"),
        ("TT_AWAY:0.5", "under", "Chelsea under 0.5 goals"),
        ("TG_AWAY", "3+", "Chelsea goals: 3 or more"),
        ("TS_AWAY", "no", "Chelsea to score: No"),
        ("WTN_HOME", "yes", "Arsenal to win to nil: Yes"),
        ("EH:-1", "home", "Arsenal -1 (3-way handicap)"),
        ("EH:-1", "draw", "Draw after Arsenal -1 (3-way handicap)"),
        ("EH:-1", "away", "Chelsea +1 (3-way handicap)"),
        ("AH:-1.5", "home", "Arsenal -1.5 (Asian handicap)"),
        ("AH:-1.5", "away", "Chelsea +1.5 (Asian handicap)"),
        ("AH:0", "home", "Arsenal 0 (Asian handicap)"),
        ("WM", "home_by_2", "Arsenal to win by 2"),
        ("WM", "away_by_3plus", "Chelsea to win by 3 or more"),
        ("WM", "draw", "Draw"),
        ("1X2_BTTS", "draw_yes", "Draw & both teams to score: Yes"),
        ("1X2_OU:2.5", "away_under", "Chelsea to win & under 2.5 goals"),
        ("DC_OU:1.5", "home_draw_over", "Arsenal or draw & over 1.5 goals"),
        ("DC_BTTS", "draw_away_no", "Draw or Chelsea & both teams to score: No"),
        ("BTTS_OU:2.5", "no_over", "Both teams to score: No & over 2.5 goals"),
        ("CS", "2-1", "Correct score: Arsenal 2-1 Chelsea"),
        ("CS", "other", "Correct score: any other score"),
    ],
)
def test_known_descriptions(market: str, selection: str, expected: str) -> None:
    assert describe_selection(market, selection, HOME, AWAY) == expected


def test_asian_handicaps_mirror_each_other() -> None:
    for line in (-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5):
        home = describe_selection(f"AH:{line:g}", "home", HOME, AWAY)
        away = describe_selection(f"AH:{line:g}", "away", HOME, AWAY)
        assert home.startswith(HOME) and away.startswith(AWAY)
        assert not re.search(r"[+-]0(?![\d.])", home + away)  # zero is written plainly


def test_team_names_pass_through_untouched() -> None:
    name = "Brighton & Hove Albion"
    assert (
        describe_selection("1X2", "home", name, AWAY) == "Brighton & Hove Albion to win"
    )


@pytest.mark.parametrize(
    ("market", "selection"),
    [("NOPE:1", "x"), ("1X2", "banana"), ("AH:wide", "home"), ("EH:x", "home")],
)
def test_unknown_names_fall_back_instead_of_failing(
    market: str, selection: str
) -> None:
    assert describe_selection(market, selection, HOME, AWAY) == f"{market}: {selection}"
