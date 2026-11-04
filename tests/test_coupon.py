"""Tests for the weekly coupon."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import logging_config
from src.bundle import build_bundle, load_bundle, predict_rates, save_bundle
from src.coupon import (
    BET_COLUMNS,
    DEMO_BOOKS,
    MIN_QUOTED_ODDS,
    Coupon,
    _add_names,
    _display,
    build_offers,
    clean_odds,
    format_coupon,
    main,
    make_coupon,
    pretty_team,
    price_fixtures,
    round_stakes,
    save_coupon,
    simulated_odds,
    split_message,
)
from src.evaluate import Calibration
from src.features import build_features
from src.markets import fixture_markets
from src.model import ModelParams, fit_boosters, training_mask
from src.value import ValueConfig
from tests.test_features import make_matches

FAST = ModelParams(n_estimators=25, num_leaves=4, min_child_samples=10, num_threads=1)
RHO = -0.05
STAMP = datetime(2026, 11, 6, 9, 0, tzinfo=UTC)  # a Friday
TRUST_MODEL = ValueConfig(model_weight=1.0)


@pytest.fixture
def isolated_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    root = logging.getLogger()
    original_level, original_hook = root.level, sys.excepthook
    monkeypatch.setattr(logging_config, "LOG_DIR", tmp_path / "logs")
    yield
    logging_config._remove_engine_handlers(root)
    root.setLevel(original_level)
    sys.excepthook = original_hook


def fixtures_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": ["M1", "M2", "M3"],
            "lam_home": [1.8, 1.1, 1.4],
            "lam_away": [0.9, 1.2, 1.0],
            "home_team_id": ["manchester-city", "arsenal", "chelsea"],
            "away_team_id": ["fulham", "everton", "brighton"],
            "league": "E0",
            "date": ["2026-11-07", "2026-11-08", "2026-11-07"],
        }
    )


def book(
    fixtures: pd.DataFrame, name: str = "betpawa", margin: float = 0.06
) -> pd.DataFrame:
    """A bookmaker pricing every selection at the model's fair odds, less ``margin``."""
    priced = price_fixtures(fixtures, RHO)
    odds = np.maximum(priced["fair_odds"].to_numpy() / (1.0 + margin), MIN_QUOTED_ODDS)
    return pd.DataFrame(
        {
            "match_id": priced["match_id"],
            "bookmaker": name,
            "market_id": priced["market_id"],
            "selection": priced["selection"],
            "odds": odds,
        }
    )


def generous(
    odds: pd.DataFrame, match: str, market: str, selection: str, factor: float
) -> pd.DataFrame:
    """The same prices, except one selection pays ``factor`` times as much."""
    target = (
        (odds["match_id"] == match)
        & (odds["market_id"] == market)
        & (odds["selection"] == selection)
    )
    assert target.sum() == 1
    return odds.assign(odds=np.where(target, odds["odds"] * factor, odds["odds"]))


def priced_at(
    odds: pd.DataFrame, match: str, market: str, selection: str, price: float
) -> pd.DataFrame:
    """The same prices, except one selection is offered at exactly ``price``."""
    target = (
        (odds["match_id"] == match)
        & (odds["market_id"] == market)
        & (odds["selection"] == selection)
    )
    assert target.sum() == 1
    return odds.assign(odds=np.where(target, price, odds["odds"]))


# --------------------------------------------------------------------------- #
# Odds
# --------------------------------------------------------------------------- #
def test_clean_odds_keeps_good_rows_and_reports_the_rest() -> None:
    odds = pd.DataFrame(
        {
            "match_id": ["M1", "M1", "M1", "M9", "M1", "M1", "M1", "M1"],
            "bookmaker": [" BetPawa ", "betpawa", "betpawa", "betpawa", "betpawa", "betpawa", "betpawa", "other"],
            "market_id": ["1X2", "1X2", "1X2", "1X2", "NOPE", "OU:2.5", "OU:2.5", "OU:2.5"],
            "selection": ["home", "draw", "away", "home", "x", "over", "over", "over"],
            "odds": [2.0, "abc", 0.9, 2.0, 2.0, 1.9, 2.1, 2.2],
        }
    )  # fmt: skip
    clean, notes = clean_odds(odds, ["M1", "M2"])
    assert list(clean.columns) == [
        "match_id",
        "bookmaker",
        "market_id",
        "selection",
        "odds",
    ]
    assert set(clean["bookmaker"]) == {"betpawa", "other"}  # trimmed and lower-cased
    kept = clean.set_index(["bookmaker", "market_id", "selection"])["odds"]
    assert kept.loc[("betpawa", "1X2", "home")] == 2.0
    assert kept.loc[("betpawa", "OU:2.5", "over")] == 2.1  # the later of two repeats
    assert kept.loc[("other", "OU:2.5", "over")] == 2.2
    assert len(clean) == 3
    joined = "\n".join(notes)
    assert "2 odds rows ignored: price is not a number above 1" in joined
    assert "1 odds rows ignored: match is not among the fixtures" in joined
    assert "1 odds rows ignored: unknown market or selection (e.g. NOPE/x)" in joined
    assert "1 odds rows ignored: the same price appears twice" in joined


def test_clean_odds_says_nothing_when_everything_is_fine() -> None:
    clean, notes = clean_odds(book(fixtures_frame()), ["M1", "M2", "M3"])
    assert notes == [] and len(clean) == len(book(fixtures_frame()))


def test_a_price_of_exactly_one_is_not_a_price() -> None:
    odds = pd.DataFrame(
        {
            "match_id": ["M1", "M1"],
            "bookmaker": ["betpawa", "betpawa"],
            "market_id": ["1X2", "1X2"],
            "selection": ["home", "draw"],
            "odds": [1.0, 1.01],
        }
    )
    clean, notes = clean_odds(odds, ["M1"])
    assert clean["selection"].tolist() == ["draw"]
    assert notes == ["1 odds rows ignored: price is not a number above 1"]


def test_clean_odds_requires_its_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        clean_odds(pd.DataFrame({"match_id": ["M1"], "odds": [2.0]}), ["M1"])


def test_offers_join_the_models_probabilities_to_the_prices() -> None:
    fixtures = fixtures_frame()
    odds = book(fixtures).query("market_id in ['1X2', 'OU:2.5']")
    offers = build_offers(fixtures, odds, RHO)
    assert len(offers) == len(odds) == 3 * (3 + 2)
    expected = fixture_markets(
        fixtures[["lam_home", "lam_away"]], RHO, match_ids=fixtures["match_id"]
    ).set_index(["match_id", "market_id", "selection"])["probability"]
    keyed = offers.set_index(["match_id", "market_id", "selection"])
    np.testing.assert_allclose(keyed["probability"], expected.loc[keyed.index])
    assert {"push_probability", "fair_odds", "bookmaker", "odds"} <= set(offers.columns)


def test_simulated_odds_are_reproducible_and_quotable() -> None:
    priced = price_fixtures(fixtures_frame(), RHO)
    first, again = simulated_odds(priced, seed=1), simulated_odds(priced, seed=1)
    pd.testing.assert_frame_equal(first, again)
    assert not first["odds"].equals(simulated_odds(priced, seed=2)["odds"])
    assert set(first["bookmaker"]) == set(DEMO_BOOKS)
    assert len(first) == len(DEMO_BOOKS) * len(priced)
    assert (first["odds"] >= MIN_QUOTED_ODDS).all() and np.isfinite(first["odds"]).all()
    clean, notes = clean_odds(first, ["M1", "M2", "M3"])
    assert notes == [] and len(clean) == len(first)  # every simulated price is usable


# --------------------------------------------------------------------------- #
# Making the coupon
# --------------------------------------------------------------------------- #
def test_a_fairly_priced_book_gives_an_empty_coupon() -> None:
    fixtures = fixtures_frame()
    coupon = make_coupon(
        fixtures, book(fixtures), rho=RHO, config=TRUST_MODEL, generated_at=STAMP
    )
    assert coupon.bets.empty and list(coupon.bets.columns) == list(BET_COLUMNS)
    assert len(coupon.offers) == len(book(fixtures)) and coupon.n_fixtures == 3
    assert coupon.notes == ()


def test_a_planted_value_bet_comes_out_with_names_sizes_and_words() -> None:
    fixtures = fixtures_frame()
    odds = generous(book(fixtures), "M1", "OU:2.5", "over", 1.5)
    coupon = make_coupon(
        fixtures, odds, rho=RHO, config=TRUST_MODEL, generated_at=STAMP
    )
    assert len(coupon.bets) == 1
    bet = coupon.bets.iloc[0]
    assert (bet["match_id"], bet["market_id"], bet["selection"]) == (
        "M1",
        "OU:2.5",
        "over",
    )
    assert bet["description"] == "Over 2.5 goals" and bet["bookmaker"] == "betpawa"
    assert (bet["home"], bet["away"]) == ("Manchester City", "Fulham")
    assert (bet["league"], bet["date"]) == ("Premier League", "2026-11-07")
    assert bet["edge"] == pytest.approx(1.5 / 1.06 - 1.0)
    assert bet["stake"] == 20.0 and bet["stake_fraction"] == pytest.approx(
        0.02
    )  # the 2% cap
    assert bet["model_weight"] == 1.0
    assert bet["min_odds_to_bet"] <= bet["odds"]


def test_the_lowest_worthwhile_price_keeps_the_minimum_edge() -> None:
    fixtures = fixtures_frame()
    sharp, soft = book(fixtures, "sharp", 0.02), book(fixtures, "soft", 0.10)
    odds = pd.concat(
        [sharp, generous(soft, "M1", "OU:2.5", "over", 1.3)], ignore_index=True
    )
    config = ValueConfig(model_weight=0.0, min_edge=0.05, min_odds=1.05)
    coupon = make_coupon(
        fixtures,
        odds,
        rho=RHO,
        config=config,
        reference_bookmaker="Sharp",
        stake_step=0,
    )
    bet = coupon.bets.iloc[0]
    assert bet["bookmaker"] == "soft" and bet["reference_source"] == "reference book"
    assert (
        bet["p_final"] * bet["min_odds_to_bet"] - 1.0 >= 0.05 - 1e-9
    )  # still worth it
    assert (
        bet["p_final"] * (bet["min_odds_to_bet"] - 0.01) - 1.0 < 0.05
    )  # one tick lower is not


def test_the_lowest_worthwhile_price_is_never_below_the_engines_minimum_odds() -> None:
    fixtures = fixtures_frame()
    odds = generous(book(fixtures), "M1", "OU:0.5", "over", 1.5)  # a near-certain goal
    coupon = make_coupon(
        fixtures, odds, rho=RHO, config=TRUST_MODEL, generated_at=STAMP
    )
    bet = coupon.bets.iloc[0]
    assert (bet["market_id"], bet["selection"]) == ("OU:0.5", "over")
    assert bet["p_final"] > 0.9
    needed = (1.0 + TRUST_MODEL.min_edge) / bet["p_final"]
    assert (
        needed < TRUST_MODEL.min_odds
    )  # the edge alone would allow a price the engine refuses
    assert bet["min_odds_to_bet"] == TRUST_MODEL.min_odds


def test_the_lowest_worthwhile_price_never_exceeds_the_price_on_offer() -> None:
    fixtures = fixtures_frame()
    base = book(fixtures)
    probability = (
        price_fixtures(fixtures, RHO)
        .set_index(["match_id", "market_id", "selection"])
        .loc[("M1", "OU:2.5", "over"), "probability"]
    )
    # Just over the minimum edge, and not on a whole cent: rounding the break-even price up
    # to the next cent would land above the price actually offered.
    offered = (1.0 + TRUST_MODEL.min_edge) / probability + 0.0004
    coupon = make_coupon(
        fixtures,
        priced_at(base, "M1", "OU:2.5", "over", offered),
        rho=RHO,
        config=TRUST_MODEL,
        generated_at=STAMP,
    )
    bet = coupon.bets.iloc[0]
    assert bet["odds"] == pytest.approx(offered)
    assert bet["min_odds_to_bet"] == pytest.approx(offered)
    assert round(offered * 100) != offered * 100  # the price is not on a whole cent


def test_the_reference_bookmaker_is_used_or_its_absence_reported() -> None:
    fixtures = fixtures_frame()
    odds = pd.concat(
        [book(fixtures, "sharp", 0.02), generous(book(fixtures, "soft", 0.10), "M2", "BTTS", "yes", 1.35)],
        ignore_index=True,
    )  # fmt: skip
    config = ValueConfig(model_weight=0.0)
    found = make_coupon(
        fixtures, odds, rho=RHO, config=config, reference_bookmaker="sharp"
    )
    assert found.bets["selection"].tolist() == ["yes"] and found.notes == ()
    own = make_coupon(
        fixtures, odds, rho=RHO, config=config, reference_bookmaker="nobody"
    )
    assert (
        own.bets.empty
    )  # judged by its own prices, a margin-laden book offers nothing
    assert any(
        "Reference bookmaker 'nobody' has no prices" in note for note in own.notes
    )


def test_bets_come_in_match_date_order_then_best_edge_first() -> None:
    fixtures = fixtures_frame()
    odds = book(fixtures)
    for match, factor in (("M1", 1.4), ("M2", 1.6), ("M3", 1.8)):
        odds = generous(odds, match, "BTTS", "yes", factor)
    coupon = make_coupon(fixtures, odds, rho=RHO, config=TRUST_MODEL)
    assert coupon.bets["match_id"].tolist() == [
        "M3",
        "M1",
        "M2",
    ]  # 7 Nov (best edge first), then 8 Nov
    assert coupon.bets["date"].tolist() == ["2026-11-07", "2026-11-07", "2026-11-08"]


def test_stakes_are_rounded_down_to_the_step() -> None:
    fixtures = fixtures_frame()
    odds = generous(book(fixtures), "M1", "OU:2.5", "over", 1.5)
    config = ValueConfig(
        model_weight=1.0, max_stake_fraction=0.0183
    )  # 18.3 before rounding
    exact = make_coupon(fixtures, odds, rho=RHO, config=config, stake_step=0)
    stepped = make_coupon(fixtures, odds, rho=RHO, config=config, stake_step=5)
    assert exact.bets.iloc[0]["stake"] == pytest.approx(18.3)
    assert stepped.bets.iloc[0]["stake"] == 15.0
    assert stepped.bets.iloc[0]["stake_fraction"] == pytest.approx(0.015)
    too_big = make_coupon(fixtures, odds, rho=RHO, config=config, stake_step=50)
    assert too_big.bets.empty  # not worth a stake of zero
    assert "below_min_stake" in set(too_big.offers["status"])


def test_round_stakes_leaves_other_rows_alone() -> None:
    result = pd.DataFrame(
        {
            "status": ["bet", "bet", "bet", "below_min_edge"],
            "stake": [19.9, 0.4, 7.0, 0.0],
            "stake_fraction": [0.0199, 0.0004, 0.007, 0.0],
        }
    )
    rounded = round_stakes(result, bankroll=1000.0, step=5.0)
    assert rounded["stake"].tolist() == [15.0, 0.0, 5.0, 0.0]
    assert rounded["status"].tolist() == [
        "bet",
        "below_min_stake",
        "bet",
        "below_min_edge",
    ]
    assert rounded["stake_fraction"].tolist() == pytest.approx([0.015, 0.0, 0.005, 0.0])
    assert result["stake"].tolist() == [19.9, 0.4, 7.0, 0.0]  # the input is untouched
    pd.testing.assert_frame_equal(round_stakes(result, 1000.0, 0.0), result)


def test_a_week_with_nothing_to_price_gives_an_empty_coupon_not_an_error() -> None:
    fixtures = fixtures_frame()
    none_left = make_coupon(fixtures.iloc[:0], book(fixtures), rho=RHO)
    assert none_left.bets.empty and none_left.n_fixtures == 0
    assert "No fixtures to price" in none_left.notes
    no_odds = make_coupon(fixtures, book(fixtures).iloc[:0], rho=RHO)
    assert no_odds.bets.empty and no_odds.n_fixtures == 3
    assert "No usable odds for 3 of 3 fixtures" in no_odds.notes
    other_matches = book(fixtures).assign(match_id="Z9")
    unmatched = make_coupon(fixtures, other_matches, rho=RHO)
    assert unmatched.bets.empty
    assert any("match is not among the fixtures" in note for note in unmatched.notes)
    for coupon in (none_left, no_odds, unmatched):
        assert "No value bets" in format_coupon(coupon)


def test_fixtures_without_usable_rates_are_skipped_and_reported() -> None:
    fixtures = fixtures_frame()
    fixtures.loc[1, "lam_home"] = np.nan
    fixtures.loc[2, "lam_away"] = 0.0
    coupon = make_coupon(fixtures, book(fixtures_frame()), rho=RHO, config=TRUST_MODEL)
    assert coupon.n_fixtures == 1
    assert "2 fixtures skipped: no usable goal rates" in coupon.notes
    assert set(coupon.offers["match_id"]) == {"M1"}


def test_fixtures_with_missing_odds_are_counted() -> None:
    fixtures = fixtures_frame()
    odds = book(fixtures)
    coupon = make_coupon(fixtures, odds[odds["match_id"] != "M2"], rho=RHO)
    assert "No usable odds for 1 of 3 fixtures" in coupon.notes


def test_invalid_inputs_are_rejected() -> None:
    fixtures = fixtures_frame()
    with pytest.raises(ValueError, match="fixtures is missing columns"):
        make_coupon(fixtures.drop(columns="lam_away"), book(fixtures), rho=RHO)
    with pytest.raises(ValueError, match="repeated match_id"):
        make_coupon(pd.concat([fixtures, fixtures]), book(fixtures), rho=RHO)
    with pytest.raises(ValueError, match="stake_step"):
        make_coupon(fixtures, book(fixtures), rho=RHO, stake_step=-1.0)
    with pytest.raises(ValueError, match="odds is missing columns"):
        make_coupon(fixtures, book(fixtures).drop(columns="bookmaker"), rho=RHO)


def test_display_names_fall_back_gracefully() -> None:
    bare = pd.DataFrame({"match_id": ["a", "b"]})
    shown = _display(bare)
    assert shown["home"].tolist() == ["Home", "Home"] and shown["away"].tolist() == [
        "Away",
        "Away",
    ]
    assert shown["league"].tolist() == ["", ""] and shown["date"].tolist() == ["", ""]
    named = pd.DataFrame(
        {"match_id": ["a"], "home_name": ["Real Madrid"], "away_team_id": ["atletico-madrid"],
         "league": ["XX"], "date": [pd.Timestamp("2026-11-07")]}
    )  # fmt: skip
    shown = _display(named)
    assert (shown.loc[0, "home"], shown.loc[0, "away"]) == (
        "Real Madrid",
        "Atletico Madrid",
    )
    assert (shown.loc[0, "league"], shown.loc[0, "date"]) == ("XX", "2026-11-07")
    assert (
        pretty_team("manchester-city") == "Manchester City"
        and pretty_team("a_b") == "A B"
    )


def test_the_team_registry_supplies_display_names(tmp_path: Path) -> None:
    registry = tmp_path / "teams.yaml"
    registry.write_text(
        "teams:\n"
        '  - {id: manchester-city, country: England, name: "Manchester City", aliases: ["Man City"]}\n'
        '  - {id: brighton, country: England, name: "Brighton & Hove Albion", aliases: []}\n',
        encoding="utf-8",
    )
    fixtures = pd.DataFrame(
        {"match_id": ["a"], "home_team_id": ["brighton"], "away_team_id": ["new-club"]}
    )
    named = _add_names(fixtures, registry)
    assert named.loc[0, "home_name"] == "Brighton & Hove Albion"
    assert named.loc[0, "away_name"] == "New Club"  # unknown ids are prettified
    without_file = _add_names(fixtures, tmp_path / "missing.yaml")
    assert without_file.loc[0, "home_name"] == "Brighton"  # no registry: prettified ids
    assert "home_name" not in _add_names(pd.DataFrame({"match_id": ["a"]}), registry)


TWO_CLUBS_ONE_SPELLING = """\
teams:
  - {id: a, name: A, country: England, aliases: ["Foo"]}
  - {id: b, name: B, country: England, aliases: ["Foo"]}
"""


@pytest.mark.parametrize(
    "contents",
    [
        "teams: [unclosed\n  - {id: x",  # not even YAML
        "teams: 5\n",  # wrong shape
        "teams:\n  - {id: x}\n",  # an entry without its name
        "",
        TWO_CLUBS_ONE_SPELLING,
        None,  # the path is a folder, not a file
    ],
)
def test_a_broken_registry_never_costs_the_coupon(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, contents: str | None
) -> None:
    path = tmp_path / "teams.yaml"
    if contents is None:
        path.mkdir()
    else:
        path.write_text(contents, encoding="utf-8")
    fixtures = pd.DataFrame(
        {"match_id": ["a"], "home_team_id": ["brighton"], "away_team_id": ["new-club"]}
    )
    with caplog.at_level(logging.WARNING, logger="src.coupon"):
        named = _add_names(fixtures, path)
    assert (named.loc[0, "home_name"], named.loc[0, "away_name"]) == (
        "Brighton",
        "New Club",
    )
    assert "Team registry unusable" in caplog.text


# --------------------------------------------------------------------------- #
# Text and files
# --------------------------------------------------------------------------- #
def planted_coupon(weight_source: str = "default") -> Coupon:
    """A coupon with exactly one bet: Over 2.5 goals in match M1, priced 50% too high."""
    fixtures = fixtures_frame()
    odds = generous(book(fixtures), "M1", "OU:2.5", "over", 1.5)
    return make_coupon(
        fixtures,
        odds,
        rho=RHO,
        config=TRUST_MODEL,
        weight_source=weight_source,
        generated_at=STAMP,
    )


def test_the_text_tells_a_person_what_to_bet() -> None:
    text = format_coupon(planted_coupon(weight_source="fitted, blend.json"))
    lines = text.split("\n")
    assert lines[0] == "⚽ VALUE COUPON · Fri 06 Nov 2026"
    assert "1 bet · total stake 20 units (2.0% of a 1,000 bankroll)" in lines
    assert "Model weight 100% (fitted, blend.json) · 0.25× Kelly · min edge 4%" in lines
    assert "1) Manchester City vs Fulham" in lines
    assert "   Premier League · 2026-11-07" in lines
    assert any(
        line.startswith("   ➜ Over 2.5 goals @ ") and line.endswith("(betpawa)")
        for line in lines
    )
    assert any(
        "edge +41.5%" in line and "Stake 20 units (2.0%)" in line for line in lines
    )
    assert any(line.startswith("   Worth taking down to odds ") for line in lines)
    assert "DEMO" not in text and "Data notes" not in text


def test_the_text_marks_demos_and_lists_data_notes() -> None:
    fixtures = fixtures_frame()
    odds = pd.concat(
        [
            generous(book(fixtures), "M1", "OU:2.5", "over", 1.5),
            book(fixtures).iloc[:1].assign(odds=0.5),
        ]
    )
    coupon = make_coupon(
        fixtures,
        odds,
        rho=RHO,
        config=TRUST_MODEL,
        generated_at=STAMP,
        demo=True,
        currency="XAF",
    )
    text = format_coupon(coupon)
    assert "🧪 DEMO: simulated fixtures and odds, not real bets" in text
    assert "total stake 20 XAF" in text and "Stake 20 XAF" in text
    assert "Data notes:\n• 1 odds rows ignored: price is not a number above 1" in text


def test_stakes_show_decimals_when_the_step_is_fractional() -> None:
    fixtures = fixtures_frame()
    odds = generous(book(fixtures), "M1", "OU:2.5", "over", 1.5)
    config = ValueConfig(model_weight=1.0, max_stake_fraction=0.0183)
    coupon = make_coupon(
        fixtures, odds, rho=RHO, config=config, stake_step=0.5, generated_at=STAMP
    )
    assert "Stake 18.00 units" in format_coupon(coupon)


def test_several_bets_are_numbered_and_pluralised() -> None:
    fixtures = fixtures_frame()
    odds = generous(
        generous(book(fixtures), "M1", "BTTS", "yes", 1.4), "M2", "BTTS", "yes", 1.4
    )
    text = format_coupon(
        make_coupon(fixtures, odds, rho=RHO, config=TRUST_MODEL, generated_at=STAMP)
    )
    assert "2 bets · total stake 40 units" in text
    assert "1) Manchester City vs Fulham" in text and "2) Arsenal vs Everton" in text


def test_a_message_that_fits_is_not_split() -> None:
    assert split_message("a short message") == ["a short message"]
    assert split_message("") == []


def test_a_message_exactly_at_the_limit_still_fits() -> None:
    assert split_message("a" * 10, limit=10) == ["a" * 10]
    assert split_message("aaaa\n\nbbbb", limit=10) == [
        "aaaa\n\nbbbb"
    ]  # both blocks, 10 characters
    assert split_message("aaaa\n\nbbbbb", limit=10) == [
        "aaaa",
        "bbbbb",
    ]  # one over: cut between blocks


def test_messages_are_cut_between_bets() -> None:
    blocks = [f"{i}) bet\n   line one\n   line two" for i in range(1, 11)]
    text = "\n\n".join(blocks)
    parts = split_message(text, limit=100)
    assert len(parts) > 1 and all(len(part) <= 100 for part in parts)
    assert "\n\n".join(parts) == text  # nothing lost, and cuts fell on block boundaries
    assert all(
        part.split("\n")[0].endswith("bet") for part in parts
    )  # no bet torn in half


def test_an_oversized_block_is_cut_at_line_breaks_then_by_characters() -> None:
    lines = [f"line {i:02d} " + "x" * 20 for i in range(12)]
    parts = split_message("\n".join(lines), limit=70)
    assert all(len(part) <= 70 for part in parts)
    assert "\n".join(parts).split("\n") == lines
    long_line = "y" * 250
    cut = split_message(long_line, limit=100)
    assert [len(part) for part in cut] == [100, 100, 50] and "".join(cut) == long_line


def test_a_full_coupon_survives_the_telegram_limit() -> None:
    fixtures = fixtures_frame()
    odds = book(fixtures)
    for match in ("M1", "M2", "M3"):
        odds = generous(odds, match, "BTTS", "yes", 1.4)
    text = format_coupon(
        make_coupon(fixtures, odds, rho=RHO, config=TRUST_MODEL, generated_at=STAMP)
    )
    parts = split_message(text, limit=300)
    assert len(parts) > 1 and all(len(part) <= 300 for part in parts)
    assert "\n\n".join(parts) == text


def test_split_message_rejects_a_silly_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        split_message("text", limit=0)


def test_saving_writes_the_text_and_a_csv_record(tmp_path: Path) -> None:
    coupon = planted_coupon()
    text_path, csv_path = save_coupon(coupon, tmp_path)
    assert text_path == tmp_path / "coupon_2026-11-06_0900.txt"
    assert csv_path == tmp_path / "bets_2026-11-06_0900.csv"
    assert text_path.read_text(encoding="utf-8").rstrip("\n") == format_coupon(coupon)
    record = pd.read_csv(csv_path)
    assert list(record.columns) == ["generated_at", "demo", *BET_COLUMNS]
    assert (
        len(record) == 1
        and record.loc[0, "generated_at"] == "2026-11-06T09:00:00+00:00"
    )
    assert not record.loc[0, "demo"] and record.loc[0, "market_id"] == "OU:2.5"


def test_demo_coupons_are_kept_apart_and_empty_ones_still_leave_a_record(
    tmp_path: Path,
) -> None:
    fixtures = fixtures_frame()
    empty = make_coupon(
        fixtures,
        book(fixtures),
        rho=RHO,
        config=TRUST_MODEL,
        generated_at=STAMP,
        demo=True,
    )
    text_path, csv_path = save_coupon(empty, tmp_path)
    assert text_path.parent == tmp_path / "demo" == csv_path.parent
    record = pd.read_csv(csv_path)
    assert record.empty and list(record.columns) == [
        "generated_at",
        "demo",
        *BET_COLUMNS,
    ]


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("coupon")
    features = build_features(make_matches(n_teams=16, seasons=7, seed=3))
    mask = training_mask(features)
    last = features.loc[mask, "date"].max()
    boosters = fit_boosters(
        features, mask, FAST, as_of=last + pd.Timedelta(1, unit="D")
    )
    bundle = build_bundle(
        boosters,
        Calibration(scale_home=1.02, scale_away=0.97, rho=-0.06),
        FAST,
        trained_through=last,
        n_train_rows=int(mask.sum()),
        metrics={},
    )
    save_bundle(bundle, root / "xg_model.pkl")
    features.to_parquet(root / "features.parquet", index=False)
    return {"model": root / "xg_model.pkl", "features": root / "features.parquet"}


def common(tmp_path: Path, art: dict[str, Path]) -> list[str]:
    return [
        "--model", str(art["model"]),
        "--blend", str(tmp_path / "blend.json"),
        "--registry", str(tmp_path / "no_registry.yaml"),
        "--output-dir", str(tmp_path / "out"),
    ]  # fmt: skip


def real_inputs(tmp_path: Path, art: dict[str, Path]) -> list[str]:
    """Upcoming fixtures and a CSV of bookmaker odds for them."""
    upcoming = pd.read_parquet(art["features"]).tail(6).reset_index(drop=True)
    bundle = load_bundle(art["model"])
    rates = predict_rates(bundle, upcoming)
    fixtures = upcoming.assign(lam_home=rates["lam_home"], lam_away=rates["lam_away"])
    odds = simulated_odds(price_fixtures(fixtures, bundle.calibration.rho), seed=3)
    upcoming.to_parquet(tmp_path / "upcoming.parquet", index=False)
    odds.to_csv(tmp_path / "odds.csv", index=False)
    return [
        "--fixtures",
        str(tmp_path / "upcoming.parquet"),
        "--odds",
        str(tmp_path / "odds.csv"),
    ]


@pytest.mark.usefixtures("isolated_logging")
def test_the_demo_runs_end_to_end(
    tmp_path: Path,
    artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    args = [
        "--demo",
        "--features",
        str(artifacts["features"]),
        "--demo-matches",
        "6",
        *common(tmp_path, artifacts),
    ]
    with caplog.at_level(logging.WARNING):
        assert main(args) == 0
    printed = capsys.readouterr().out
    assert "VALUE COUPON" in printed and "DEMO" in printed
    assert "Model weight 25% (default, not fitted)" in printed
    assert "No fitted model weight" in caplog.text  # the missing blend file is flagged
    folder = tmp_path / "out" / "demo"
    assert len(list(folder.glob("coupon_*.txt"))) == 1
    record = pd.read_csv(next(folder.glob("bets_*.csv")))
    assert (
        len(record) >= 1
    )  # 6 matches against a noisy soft book always leave something
    assert record["demo"].all() and record["bookmaker"].isin(list(DEMO_BOOKS)).all()
    assert (record["stake"] % 1 == 0).all()  # whole-unit stakes by default
    assert record["stake_fraction"].sum() <= 0.10 + 1e-9  # the exposure cap holds
    assert record["match_id"].is_unique  # one bet per match


@pytest.mark.usefixtures("isolated_logging")
def test_a_real_run_uses_the_fitted_weight_and_the_chosen_currency(
    tmp_path: Path, artifacts: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "blend.json").write_text(
        json.dumps({"schema_version": 1, "model_weight": 0.3}), encoding="utf-8"
    )
    args = [
        *real_inputs(tmp_path, artifacts), *common(tmp_path, artifacts),
        "--reference-bookmaker", "sim_sharp", "--currency", "XAF", "--bankroll", "100000", "--stake-step", "100",
    ]  # fmt: skip
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "Model weight 30% (fitted, blend.json)" in printed and "DEMO" not in printed
    folder = tmp_path / "out"
    assert (
        len(list(folder.glob("coupon_*.txt"))) == 1 and not (folder / "demo").exists()
    )
    record = pd.read_csv(next(folder.glob("bets_*.csv")))
    assert len(record) >= 1
    assert not record["demo"].any() and (record["model_weight"] == 0.3).all()
    assert (record["stake"] % 100 == 0).all() and (record["stake"] >= 100).all()
    assert (record["stake"] <= 0.02 * 100_000 + 1e-6).all()
    assert "Stake" in printed and " XAF " in printed


@pytest.mark.usefixtures("isolated_logging")
def test_an_unusable_blend_file_falls_back_to_the_default_with_a_warning(
    tmp_path: Path,
    artifacts: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "blend.json").write_text("not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert (
            main([*real_inputs(tmp_path, artifacts), *common(tmp_path, artifacts)]) == 0
        )
    assert "Unusable model weight file" in caplog.text
    assert "default, not fitted" in capsys.readouterr().out


@pytest.mark.usefixtures("isolated_logging")
def test_the_weight_flag_overrides_the_file(
    tmp_path: Path, artifacts: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "blend.json").write_text(
        json.dumps({"schema_version": 1, "model_weight": 0.3}), encoding="utf-8"
    )
    assert (
        main(
            [
                *real_inputs(tmp_path, artifacts),
                *common(tmp_path, artifacts),
                "--weight",
                "0.6",
            ]
        )
        == 0
    )
    assert "Model weight 60% (set on the command line)" in capsys.readouterr().out


@pytest.mark.usefixtures("isolated_logging")
def test_problems_with_inputs_give_exit_code_one_and_a_clear_message(
    tmp_path: Path, artifacts: dict[str, Path], caplog: pytest.LogCaptureFixture
) -> None:
    inputs = real_inputs(tmp_path, artifacts)
    base = common(tmp_path, artifacts)
    missing_model = [*inputs, *base, "--model", str(tmp_path / "nope.pkl")]
    (tmp_path / "broken.pkl").write_bytes(b"not a pickle")
    broken_model = [*inputs, *base, "--model", str(tmp_path / "broken.pkl")]
    no_odds = [inputs[0], inputs[1], "--odds", str(tmp_path / "gone.csv"), *base]
    wrong_type = [inputs[0], inputs[1], "--odds", str(tmp_path / "odds.txt"), *base]
    (tmp_path / "odds.txt").write_text("x", encoding="utf-8")
    bad_rule = [*inputs, *base, "--kelly-fraction", "2"]
    missing_columns = [
        inputs[0],
        inputs[1],
        "--odds",
        str(tmp_path / "thin.csv"),
        *base,
    ]
    pd.DataFrame({"match_id": ["a"]}).to_csv(tmp_path / "thin.csv", index=False)
    with caplog.at_level(logging.ERROR):
        for args, message in (
            (missing_model, "python -m src.build_model"),
            (broken_model, "Cannot load the model"),
            (no_odds, "Cannot build the coupon"),
            (wrong_type, "expected a .parquet or .csv file"),
            (bad_rule, "kelly_fraction"),
            (missing_columns, "odds is missing columns"),
        ):
            caplog.clear()
            assert main(args) == 1, message
            assert message in caplog.text
    assert not (tmp_path / "out").exists()  # nothing half-written


def test_the_command_line_refuses_confusing_combinations(
    tmp_path: Path, artifacts: dict[str, Path]
) -> None:
    base = common(tmp_path, artifacts)
    for args in (
        ["--demo", "--odds", "x.csv", *base],  # never mix simulated and real inputs
        ["--fixtures", "f.parquet", *base],  # odds are needed too
        base,  # nothing to work on
        ["--demo", "--demo-matches", "0", *base],
    ):
        with pytest.raises(SystemExit) as stopped:
            main(args)
        assert stopped.value.code == 2
