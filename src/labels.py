"""Plain-English names for betting selections, for coupons.

``describe_selection("OU:2.5", "over", "Arsenal", "Chelsea")`` gives
``"Over 2.5 goals"``. The ids come from :mod:`src.markets`; this module only turns
them into words a person can look for on a bookmaker's site.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

Describer = Callable[[str, str, str, str], str]  # (line, selection, home, away) -> text


def _yes_no(token: str) -> str:
    return {"yes": "Yes", "no": "No"}[token]


def _signed(value: float) -> str:
    """``+1``, ``-0.5``, ``0``: whole numbers without decimals, never ``-0``."""
    return "0" if value == 0 else f"{value:+g}"


def _result(token: str, home: str, away: str) -> str:
    """A 1X2 or double-chance outcome in words."""
    return {
        "home": f"{home} to win",
        "draw": "Draw",
        "away": f"{away} to win",
        "home_draw": f"{home} or draw",
        "home_away": f"{home} or {away}",
        "draw_away": f"Draw or {away}",
    }[token]


def _total(line: str, token: str) -> str:
    """``over``/``under`` of a goal line in words."""
    return f"{token.capitalize()} {line} goals"


def _exact(token: str) -> str:
    """An exact-goals bucket: ``3`` is "exactly 3", ``6+`` is "6 or more"."""
    return f"{token[:-1]} or more" if token.endswith("+") else f"exactly {token}"


def _team(side: str, home: str, away: str) -> str:
    return home if side == "HOME" else away


def _split_last(selection: str) -> tuple[str, str]:
    head, _, tail = selection.rpartition("_")
    return head, tail


def _margin(selection: str, home: str, away: str) -> str:
    if selection == "draw":
        return "Draw"
    side, _, size = selection.partition("_by_")
    team = home if side == "home" else away
    return f"{team} to win by {'3 or more' if size == '3plus' else size}"


def _european_handicap(line: str, selection: str, home: str, away: str) -> str:
    handicap = int(line)
    return {
        "home": f"{home} {_signed(handicap)} (3-way handicap)",
        "draw": f"Draw after {home} {_signed(handicap)} (3-way handicap)",
        "away": f"{away} {_signed(-handicap)} (3-way handicap)",
    }[selection]


def _asian_handicap(line: str, selection: str, home: str, away: str) -> str:
    handicap = float(line)
    if selection == "home":
        return f"{home} {_signed(handicap)} (Asian handicap)"
    return f"{away} {_signed(-handicap)} (Asian handicap)"


def _correct_score(_line: str, selection: str, home: str, away: str) -> str:
    if selection == "other":
        return "Correct score: any other score"
    home_goals, _, away_goals = selection.partition("-")
    return f"Correct score: {home} {home_goals}-{away_goals} {away}"


def _result_and_total(line: str, selection: str, home: str, away: str) -> str:
    result, tail = _split_last(selection)
    return f"{_result(result, home, away)} & {_total(line, tail).lower()}"


def _result_and_btts(_line: str, selection: str, home: str, away: str) -> str:
    result, tail = _split_last(selection)
    return f"{_result(result, home, away)} & both teams to score: {_yes_no(tail)}"


def _btts_and_total(line: str, selection: str, _home: str, _away: str) -> str:
    btts, tail = _split_last(selection)
    return f"Both teams to score: {_yes_no(btts)} & {_total(line, tail).lower()}"


_DESCRIBERS: Final[dict[str, Describer]] = {
    "1X2": lambda line, s, h, a: _result(s, h, a),
    "DC": lambda line, s, h, a: _result(s, h, a),
    "DNB": lambda line, s, h, a: f"{_team(s.upper(), h, a)} (draw no bet)",
    "OU": lambda line, s, h, a: _total(line, s),
    "BTTS": lambda line, s, h, a: f"Both teams to score: {_yes_no(s)}",
    "OE": lambda line, s, h, a: f"Total goals: {s}",
    "TG": lambda line, s, h, a: f"Total goals: {_exact(s)}",
    "GB4": lambda line, s, h, a: f"Total goals: {s}",
    "GB3": lambda line, s, h, a: f"Total goals: {s}",
    "TT_HOME": lambda line, s, h, a: f"{h} {_total(line, s).lower()}",
    "TT_AWAY": lambda line, s, h, a: f"{a} {_total(line, s).lower()}",
    "TG_HOME": lambda line, s, h, a: f"{h} goals: {_exact(s)}",
    "TG_AWAY": lambda line, s, h, a: f"{a} goals: {_exact(s)}",
    "TS_HOME": lambda line, s, h, a: f"{h} to score: {_yes_no(s)}",
    "TS_AWAY": lambda line, s, h, a: f"{a} to score: {_yes_no(s)}",
    "WTN_HOME": lambda line, s, h, a: f"{h} to win to nil: {_yes_no(s)}",
    "WTN_AWAY": lambda line, s, h, a: f"{a} to win to nil: {_yes_no(s)}",
    "EH": _european_handicap,
    "AH": _asian_handicap,
    "WM": lambda line, s, h, a: _margin(s, h, a),
    "1X2_BTTS": _result_and_btts,
    "DC_BTTS": _result_and_btts,
    "1X2_OU": _result_and_total,
    "DC_OU": _result_and_total,
    "BTTS_OU": _btts_and_total,
    "CS": _correct_score,
}


def describe_selection(market_id: str, selection: str, home: str, away: str) -> str:
    """A selection in words, with the team names filled in.

    Args:
        market_id: Market id from :mod:`src.markets`, such as ``"OU:2.5"``.
        selection: Selection name within that market, such as ``"over"``.
        home: Home team name.
        away: Away team name.

    Returns:
        A readable description. A market or selection this module does not know
        comes back as ``"<market_id>: <selection>"`` rather than an error, so a
        coupon is never lost to a naming gap.
    """
    family, _, line = market_id.partition(":")
    describer = _DESCRIBERS.get(family)
    if describer is None:
        return f"{market_id}: {selection}"
    try:
        return describer(line, selection, home, away)
    except (KeyError, ValueError):
        return f"{market_id}: {selection}"
