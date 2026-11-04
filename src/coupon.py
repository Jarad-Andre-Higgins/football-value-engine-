"""The weekly coupon: from goal rates and bookmaker odds to a list of bets.

This is the part of the Friday job that turns predictions into something a
person can act on::

    features of upcoming matches --(model)--> goal rates        src.bundle
    goal rates --> probability of every selection               src.markets
    + bookmaker odds --> offers                                 this module
    offers --> value bets with stakes                           src.value
    bets --> Telegram-sized text and a CSV record               this module

Odds file (CSV or parquet), one row per price::

    match_id,bookmaker,market_id,selection,odds
    E0-2026-20261107-arsenal-chelsea,betpawa,OU:2.5,over,2.10

``match_id`` must be the id of a fixture; ``market_id`` and ``selection`` use the
names of :mod:`src.markets` (``python -m src.markets --all`` lists them). Rows
that do not fit are dropped and counted in the coupon's notes. Turning a
bookmaker's own feed into this layout is the job of the API step (Phase 7).

Fixtures file: the feature rows of the upcoming matches, in the layout of
``features.parquet`` (``match_id`` and every model feature; ``league``, ``date``,
``home_team_id`` and ``away_team_id`` are used for display).

Usage::

    python -m src.coupon --demo
    python -m src.coupon --fixtures upcoming.parquet --odds odds.csv --bankroll 50000
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml

from src.blend import BLEND_PATH, load_model_weight
from src.bundle import DEFAULT_MODEL_PATH, load_bundle, predict_rates
from src.config import PROCESSED_DIR
from src.dixon_coles import DEFAULT_MAX_GOALS
from src.entities import DEFAULT_REGISTRY_PATH, TeamRegistry
from src.features import FEATURES_OUTPUT
from src.labels import describe_selection
from src.leagues import LEAGUES
from src.logging_config import setup_logging
from src.markets import fixture_markets, market_set
from src.value import ValueConfig, find_value_bets

logger = logging.getLogger("src.coupon")

ODDS_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "bookmaker",
    "market_id",
    "selection",
    "odds",
)
BET_COLUMNS: Final[tuple[str, ...]] = (
    "match_id",
    "league",
    "date",
    "home",
    "away",
    "bookmaker",
    "market_id",
    "selection",
    "description",
    "odds",
    "min_odds_to_bet",
    "probability",
    "p_reference",
    "reference_source",
    "p_final",
    "raw_edge",
    "edge",
    "kelly_full",
    "stake_fraction",
    "stake",
    "model_weight",
)
COUPON_DIR: Final[Path] = PROCESSED_DIR / "coupons"
TELEGRAM_LIMIT: Final[int] = 4096  # characters in one Telegram message
DEFAULT_WEIGHT: Final[float] = ValueConfig().model_weight
MIN_QUOTED_ODDS: Final[float] = 1.01  # bookmakers do not quote below this
#: Made-up bookmakers for the demo: (margin, noise in the price). Nothing real.
DEMO_BOOKS: Final[dict[str, tuple[float, float]]] = {
    "sim_sharp": (0.03, 0.02),
    "sim_soft": (0.08, 0.07),
}


@dataclass(frozen=True)
class Coupon:
    """A finished coupon.

    Attributes:
        bets: The bets to place, soonest match first (``BET_COLUMNS``).
        offers: Every priced offer with its ``status``, for audit.
        config: The betting rules that were used.
        n_fixtures: Fixtures that were priced.
        generated_at: UTC time of the run.
        weight_source: Where the model weight came from, in words.
        currency: Label for stakes.
        stake_step: Stakes are rounded down to a multiple of this.
        notes: Data problems found on the way, in words.
        demo: True when the odds are simulated.
    """

    bets: pd.DataFrame
    offers: pd.DataFrame
    config: ValueConfig
    n_fixtures: int
    generated_at: datetime
    weight_source: str = "default"
    currency: str = "units"
    stake_step: float = 1.0
    notes: tuple[str, ...] = ()
    demo: bool = False


# --------------------------------------------------------------------------- #
# Odds
# --------------------------------------------------------------------------- #
def clean_odds(
    odds: pd.DataFrame, match_ids: Iterable[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Keep the usable rows of an odds table and say what was dropped.

    Bookmaker names are lower-cased. A row is dropped when its price is not a
    number above 1, its match is not among ``match_ids``, its market or selection
    is unknown, or it repeats an earlier price (the last one is kept).

    Returns:
        The clean table (``ODDS_COLUMNS``) and one note per kind of problem.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [column for column in ODDS_COLUMNS if column not in odds.columns]
    if missing:
        raise ValueError(f"odds is missing columns: {missing}")
    table = odds.loc[:, list(ODDS_COLUMNS)].copy()
    for column in ("match_id", "market_id", "selection"):
        table[column] = table[column].astype(str).str.strip()
    table["bookmaker"] = table["bookmaker"].astype(str).str.strip().str.lower()
    table["odds"] = pd.to_numeric(table["odds"], errors="coerce")
    notes: list[str] = []

    def drop(keep: pd.Series, reason: str, examples: pd.Series | None = None) -> None:
        nonlocal table
        lost = int((~keep).sum())
        if lost:
            shown = ""
            if examples is not None:
                sample = ", ".join(map(str, examples[~keep].drop_duplicates().head(3)))
                shown = f" (e.g. {sample})"
            notes.append(f"{lost} odds rows ignored: {reason}{shown}")
        table = table[keep]

    drop(
        np.isfinite(table["odds"]) & (table["odds"] > 1.0),
        "price is not a number above 1",
    )
    drop(table["match_id"].isin(set(match_ids)), "match is not among the fixtures")
    catalog = market_set().catalog
    known = pd.MultiIndex.from_frame(catalog[["market_id", "selection"]])
    asked = pd.MultiIndex.from_frame(table[["market_id", "selection"]])
    drop(
        pd.Series(asked.isin(known), index=table.index),
        "unknown market or selection",
        table["market_id"] + "/" + table["selection"],
    )
    keys = ["match_id", "bookmaker", "market_id", "selection"]
    drop(~table.duplicated(keys, keep="last"), "the same price appears twice")
    return table.reset_index(drop=True), notes


def price_fixtures(fixtures: pd.DataFrame, rho: float) -> pd.DataFrame:
    """Probability of every selection of every fixture (see :func:`fixture_markets`)."""
    return fixture_markets(
        fixtures[["lam_home", "lam_away"]],
        rho,
        match_ids=fixtures["match_id"],
        max_goals=DEFAULT_MAX_GOALS,
    )


def build_offers(
    fixtures: pd.DataFrame, odds: pd.DataFrame, rho: float
) -> pd.DataFrame:
    """Join the model's probabilities to the bookmakers' prices.

    Args:
        fixtures: ``match_id``, ``lam_home`` and ``lam_away`` per fixture.
        odds: A clean odds table (see :func:`clean_odds`).
        rho: Dixon-Coles dependence parameter.

    Returns:
        One row per price that has a model probability: the market catalog
        columns, ``probability``, ``push_probability``, ``fair_odds``,
        ``bookmaker`` and ``odds``.
    """
    priced = price_fixtures(fixtures, rho)
    return priced.merge(odds, on=["match_id", "market_id", "selection"], how="inner")


def simulated_odds(priced: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Two made-up bookmakers for the demo: a sharp one and a soft one.

    Each prices every selection at the model's fair odds, less a margin, times
    random noise; the soft book is both greedier and noisier. Nothing here is
    real: it exists so the whole pipeline can run without a live odds feed.
    """
    rng = np.random.default_rng(seed)
    fair = priced["fair_odds"].to_numpy(dtype="float64")
    frames = []
    for name, (margin, noise) in DEMO_BOOKS.items():
        factor = np.exp(rng.normal(0.0, noise, len(priced)))
        quoted = np.maximum(fair / (1.0 + margin) * factor, MIN_QUOTED_ODDS)
        frames.append(
            pd.DataFrame(
                {
                    "match_id": priced["match_id"].to_numpy(),
                    "bookmaker": name,
                    "market_id": priced["market_id"].to_numpy(),
                    "selection": priced["selection"].to_numpy(),
                    "odds": np.round(quoted, 2),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Building the coupon
# --------------------------------------------------------------------------- #
def pretty_team(team_id: object) -> str:
    """A readable name from a team id slug: ``manchester-city`` -> ``Manchester City``."""
    return str(team_id).replace("-", " ").replace("_", " ").title()


def _league_name(code: object) -> str:
    league = LEAGUES.get(str(code))
    return league.name if league else str(code)


def _display(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Names, league and date of each fixture, with fallbacks for missing columns."""
    frame = pd.DataFrame({"match_id": fixtures["match_id"].to_numpy()})
    for side, default in (("home", "Home"), ("away", "Away")):
        if f"{side}_name" in fixtures.columns:
            names = fixtures[f"{side}_name"].astype(str)
        elif f"{side}_team_id" in fixtures.columns:
            names = fixtures[f"{side}_team_id"].map(pretty_team)
        else:
            names = pd.Series(default, index=fixtures.index)
        frame[side] = names.to_numpy()
    frame["league"] = (
        fixtures["league"].map(_league_name).to_numpy()
        if "league" in fixtures.columns
        else ""
    )
    if "date" in fixtures.columns:
        parsed = pd.to_datetime(fixtures["date"], errors="coerce")
        frame["date"] = parsed.dt.strftime("%Y-%m-%d").fillna("").to_numpy()
    else:
        frame["date"] = ""
    return frame


def _no_bets() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in BET_COLUMNS})


def round_stakes(result: pd.DataFrame, bankroll: float, step: float) -> pd.DataFrame:
    """Round the stake of every bet down to a multiple of ``step``.

    Rounding down keeps every cap respected. A bet whose stake rounds to zero is
    not worth placing and becomes ``below_min_stake``. A ``step`` of zero leaves
    the table unchanged.
    """
    if step <= 0.0:
        return result
    table = result.copy()
    bet = (table["status"] == "bet").to_numpy()
    rounded = np.floor(table["stake"].to_numpy(dtype="float64") / step + 1e-9) * step
    table["stake"] = np.where(bet, rounded, table["stake"])
    table["stake_fraction"] = np.where(
        bet, table["stake"] / bankroll, table["stake_fraction"]
    )
    table.loc[bet & (rounded <= 0.0), "status"] = "below_min_stake"
    return table


def _present(
    bets: pd.DataFrame, fixtures: pd.DataFrame, config: ValueConfig
) -> pd.DataFrame:
    """Add names, a plain-English description and the lowest worthwhile price."""
    if bets.empty:
        return _no_bets()
    table = bets.merge(_display(fixtures), on="match_id", how="left")
    table["description"] = [
        describe_selection(market, selection, home, away)
        for market, selection, home, away in zip(
            table["market_id"], table["selection"], table["home"], table["away"]
        )
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        needed = (1.0 - table["push_probability"] + config.min_edge) / table["p_final"]
    rounded_up = np.ceil(needed.to_numpy(dtype="float64") * 100.0 - 1e-9) / 100.0
    lowest = np.maximum(
        rounded_up, config.min_odds
    )  # the engine never bets below min_odds
    table["min_odds_to_bet"] = np.minimum(lowest, table["odds"].to_numpy())
    table["model_weight"] = config.model_weight
    ordered = table.sort_values(
        ["date", "edge"], ascending=[True, False], kind="stable"
    )
    return ordered.loc[:, list(BET_COLUMNS)].reset_index(drop=True)


def make_coupon(
    fixtures: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    rho: float,
    config: ValueConfig | None = None,
    reference_bookmaker: str | None = None,
    stake_step: float = 1.0,
    currency: str = "units",
    weight_source: str = "default",
    generated_at: datetime | None = None,
    demo: bool = False,
) -> Coupon:
    """Turn predicted goal rates and bookmaker odds into a coupon.

    Args:
        fixtures: One row per fixture with ``match_id``, ``lam_home`` and
            ``lam_away``; optionally ``home_name``/``away_name`` (or the team ids),
            ``league`` and ``date`` for display.
        odds: Bookmaker prices (see the module docstring for the layout).
        rho: Dixon-Coles dependence parameter of the model.
        config: Betting rules; defaults to :class:`~src.value.ValueConfig`.
        reference_bookmaker: Judge every price against this (sharp) bookmaker's
            de-margined prices instead of each bookmaker's own.
        stake_step: Stakes are rounded down to a multiple of this (0 for none).
        currency: Label for stakes in the text.
        weight_source: Words saying where ``config.model_weight`` came from.
        generated_at: Time of the run (UTC); defaults to now.
        demo: Mark the coupon as built on simulated odds.

    Returns:
        The coupon. A week with nothing to bet on, or without usable odds, gives
        an empty coupon with notes rather than an error.

    Raises:
        ValueError: If columns are missing, a match id repeats, or ``stake_step``
            is negative.
    """
    config = config or ValueConfig()
    if stake_step < 0.0:
        raise ValueError(f"stake_step cannot be negative, got {stake_step}")
    missing = [
        c for c in ("match_id", "lam_home", "lam_away") if c not in fixtures.columns
    ]
    if missing:
        raise ValueError(f"fixtures is missing columns: {missing}")
    if fixtures["match_id"].duplicated().any():
        raise ValueError("fixtures has repeated match_id values")
    stamp = generated_at or datetime.now(UTC)
    notes: list[str] = []

    rates = fixtures[["lam_home", "lam_away"]].to_numpy(dtype="float64")
    usable = np.isfinite(rates).all(axis=1) & (rates > 0.0).all(axis=1)
    if not usable.all():
        notes.append(f"{int((~usable).sum())} fixtures skipped: no usable goal rates")
    fixtures = fixtures.loc[usable].reset_index(drop=True)

    def finish(bets: pd.DataFrame, offers: pd.DataFrame) -> Coupon:
        return Coupon(
            bets=bets,
            offers=offers,
            config=config,
            n_fixtures=len(fixtures),
            generated_at=stamp,
            weight_source=weight_source,
            currency=currency,
            stake_step=stake_step,
            notes=tuple(notes),
            demo=demo,
        )

    if fixtures.empty:
        notes.append("No fixtures to price")
        return finish(_no_bets(), pd.DataFrame())
    clean, odds_notes = clean_odds(odds, fixtures["match_id"])
    notes.extend(odds_notes)
    offers = build_offers(fixtures, clean, rho)
    without_odds = int((~fixtures["match_id"].isin(set(offers["match_id"]))).sum())
    if without_odds:
        notes.append(f"No usable odds for {without_odds} of {len(fixtures)} fixtures")
    if offers.empty:
        return finish(_no_bets(), offers)

    reference = reference_bookmaker.strip().lower() if reference_bookmaker else None
    if reference is not None and reference not in set(clean["bookmaker"]):
        notes.append(
            f"Reference bookmaker '{reference}' has no prices: "
            "each bookmaker is judged by its own prices"
        )
        reference = None
    result = find_value_bets(offers, config, reference_bookmaker=reference)
    result = round_stakes(result, config.bankroll, stake_step)
    bets = _present(result[result["status"] == "bet"], fixtures, config)
    return finish(bets, result)


# --------------------------------------------------------------------------- #
# Text and files
# --------------------------------------------------------------------------- #
def _money(value: float, step: float) -> str:
    """Whole numbers when stakes come in whole steps, otherwise two decimals."""
    decimals = 0 if step >= 1.0 and float(step).is_integer() else 2
    return f"{value:,.{decimals}f}"


def _bet_lines(number: int, bet: Any, coupon: Coupon) -> list[str]:
    when = " · ".join(part for part in (bet.league, bet.date) if part)
    lines = [f"{number}) {bet.home} vs {bet.away}"]
    if when:
        lines.append(f"   {when}")
    stake = f"Stake {_money(bet.stake, coupon.stake_step)} {coupon.currency}"
    summary = f"({bet.stake_fraction:.1%}) · edge {bet.edge:+.1%} · win chance {bet.p_final:.0%}"
    lines.append(f"   ➜ {bet.description} @ {bet.odds:.2f} ({bet.bookmaker})")
    lines.append(f"   {stake} {summary}")
    lines.append(f"   Worth taking down to odds {bet.min_odds_to_bet:.2f}")
    return lines


def format_coupon(coupon: Coupon) -> str:
    """The coupon as plain text, ready for Telegram (no markup to escape)."""
    config = coupon.config
    lines = [f"⚽ VALUE COUPON · {coupon.generated_at:%a %d %b %Y}"]
    if coupon.demo:
        lines.append("🧪 DEMO: simulated fixtures and odds, not real bets")
    bets = coupon.bets
    lines.append("")
    if bets.empty:
        lines.append(
            f"No value bets: no price cleared the minimum edge of {config.min_edge:.0%} "
            "after shrinking the model toward the market."
        )
        lines.append(
            f"Checked {len(coupon.offers):,} prices across {coupon.n_fixtures} fixtures."
        )
    else:
        total = float(bets["stake"].sum())
        step = coupon.stake_step
        lines.append(
            f"{len(bets)} bet{'s' if len(bets) != 1 else ''} · total stake "
            f"{_money(total, step)} {coupon.currency} "
            f"({total / config.bankroll:.1%} of a {_money(config.bankroll, step)} bankroll)"
        )
        lines.append(
            f"Model weight {config.model_weight:.0%} ({coupon.weight_source}) · "
            f"{config.kelly_fraction:g}× Kelly · min edge {config.min_edge:.0%}"
        )
        for number, bet in enumerate(bets.itertuples(index=False), start=1):
            lines.extend(["", *_bet_lines(number, bet, coupon)])
        lines.append("")
        lines.append(
            "Edge is the expected profit per unit staked, after shrinking the model "
            "toward the market."
        )
        lines.append(
            "Prices move: check the odds before you bet and skip any bet whose price "
            "has fallen below its 'worth taking down to' level."
        )
    if coupon.notes:
        lines += ["", "Data notes:", *[f"• {note}" for note in coupon.notes]]
    return "\n".join(lines)


def _fit_block(block: str, limit: int) -> list[str]:
    """Cut one block into pieces of at most ``limit`` characters, at line breaks if possible."""
    if len(block) <= limit:
        return [block]
    pieces: list[str] = []
    current = ""
    for line in block.split("\n"):
        while len(line) > limit:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
        else:
            pieces.append(current)
            current = line
    if current:
        pieces.append(current)
    return pieces


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split ``text`` into messages of at most ``limit`` characters.

    Cuts fall between blank-line separated blocks (one bet each) whenever
    possible, so a bet is never torn in half unless it alone exceeds the limit.

    Raises:
        ValueError: If ``limit`` is below 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    messages: list[str] = []
    current = ""
    for block in text.split("\n\n"):
        for piece in _fit_block(block, limit):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    messages.append(current)
                current = piece
    if current:
        messages.append(current)
    return messages


def save_coupon(coupon: Coupon, directory: Path = COUPON_DIR) -> tuple[Path, Path]:
    """Write the coupon text and a CSV record of its bets; return both paths.

    The CSV is the permanent record that later steps compare with closing odds
    (closing line value). A run with no bets still writes a header-only CSV.
    Demo coupons go to a ``demo`` subfolder so they are never mistaken for real ones.
    """
    folder = directory / "demo" if coupon.demo else directory
    folder.mkdir(parents=True, exist_ok=True)
    stamp = coupon.generated_at.strftime("%Y-%m-%d_%H%M")
    text_path = folder / f"coupon_{stamp}.txt"
    csv_path = folder / f"bets_{stamp}.csv"
    text_path.write_text(format_coupon(coupon) + "\n", encoding="utf-8")
    record = coupon.bets.copy()
    record.insert(0, "generated_at", coupon.generated_at.isoformat(timespec="seconds"))
    record.insert(1, "demo", coupon.demo)
    record.to_csv(csv_path, index=False)
    return text_path, csv_path


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV file by its extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"{path.name}: expected a .parquet or .csv file")


def _add_names(fixtures: pd.DataFrame, registry_path: Path) -> pd.DataFrame:
    """Add ``home_name`` and ``away_name`` from the team registry where it knows the team."""
    if "home_team_id" not in fixtures.columns or "away_team_id" not in fixtures.columns:
        return fixtures
    names: dict[str, str] = {}
    if registry_path.exists():
        try:
            registry = TeamRegistry.from_yaml(registry_path)
            names = {team.id: team.name for team in registry.teams}
        except (OSError, ValueError, yaml.YAMLError) as exc:
            # Names are cosmetic: a broken registry must never cost the coupon.
            logger.warning("Team registry unusable, showing team ids instead: %s", exc)
    return fixtures.assign(
        home_name=fixtures["home_team_id"].map(lambda t: names.get(t, pretty_team(t))),
        away_name=fixtures["away_team_id"].map(lambda t: names.get(t, pretty_team(t))),
    )


def _model_weight(
    override: float | None, blend_path: Path, log: logging.Logger
) -> tuple[float, str]:
    """The model weight to use and a few words on where it came from."""
    if override is not None:
        return override, "set on the command line"
    try:
        return load_model_weight(blend_path), f"fitted, {blend_path.name}"
    except FileNotFoundError:
        log.warning(
            "No fitted model weight at %s; using the cautious default %.2f "
            "(run python -m src.blend to fit it)",
            blend_path,
            DEFAULT_WEIGHT,
        )
    except ValueError as exc:
        log.warning(
            "Unusable model weight file, using the default %.2f: %s",
            DEFAULT_WEIGHT,
            exc,
        )
    return DEFAULT_WEIGHT, "default, not fitted"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the weekly value-bet coupon.")
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument(
        "--demo",
        action="store_true",
        help="use the latest matches of the features table and simulated bookmakers",
    )
    inputs.add_argument(
        "--fixtures", type=Path, help="features of the upcoming matches"
    )
    inputs.add_argument("--odds", type=Path, help="bookmaker odds, one row per price")
    inputs.add_argument(
        "--features", type=Path, default=FEATURES_OUTPUT, help="for --demo"
    )
    inputs.add_argument(
        "--demo-matches", type=int, default=10, help="fixtures in the demo"
    )
    inputs.add_argument(
        "--seed", type=int, default=0, help="seed of the simulated odds"
    )
    inputs.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    inputs.add_argument("--blend", type=Path, default=BLEND_PATH)
    inputs.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    rules = parser.add_argument_group("betting rules")
    rules.add_argument(
        "--weight", type=float, help="model weight 0-1 (default: fitted)"
    )
    rules.add_argument("--bankroll", type=float, default=1000.0)
    rules.add_argument("--currency", default="units")
    rules.add_argument(
        "--stake-step", type=float, default=1.0, help="round stakes down to this"
    )
    rules.add_argument("--kelly-fraction", type=float, default=0.25)
    rules.add_argument("--min-edge", type=float, default=0.04)
    rules.add_argument("--max-stake-fraction", type=float, default=0.02)
    rules.add_argument("--max-total-fraction", type=float, default=0.10)
    rules.add_argument(
        "--reference-bookmaker", help="a sharp bookmaker to judge prices by"
    )
    parser.add_argument("--output-dir", type=Path, default=COUPON_DIR)
    args = parser.parse_args(argv)
    if args.demo and (args.fixtures or args.odds):
        parser.error("--demo cannot be combined with --fixtures or --odds")
    if not args.demo and not (args.fixtures and args.odds):
        parser.error("give --fixtures and --odds, or use --demo")
    if args.demo_matches < 1:
        parser.error("--demo-matches must be at least 1")
    return args


def _echo(text: str) -> None:
    """Print without failing on consoles that cannot show emoji."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def main(argv: list[str] | None = None) -> int:
    """Build, print and save the coupon; return the process exit code.

    Returns ``0`` when a coupon was produced (even one without bets) and ``1``
    when an input is missing or unusable.
    """
    args = _parse_args(argv)
    log = setup_logging("coupon")
    try:
        bundle = load_bundle(args.model)
    except FileNotFoundError:
        log.error(
            "Model not found: %s; run python -m src.build_model first", args.model
        )
        return 1
    except (
        TypeError,
        ValueError,
        EOFError,
        AttributeError,
        ImportError,
        pickle.UnpicklingError,
    ) as exc:
        log.error("Cannot load the model %s: %s", args.model, exc)
        return 1

    weight, weight_source = _model_weight(args.weight, args.blend, log)
    try:
        config = ValueConfig(
            model_weight=weight,
            bankroll=args.bankroll,
            kelly_fraction=args.kelly_fraction,
            min_edge=args.min_edge,
            max_stake_fraction=args.max_stake_fraction,
            max_total_fraction=args.max_total_fraction,
            min_stake=max(args.stake_step, 0.0),
        )
        source = args.features if args.demo else args.fixtures
        fixtures = _read_table(source)
        if args.demo:
            fixtures = fixtures.sort_values(["date", "match_id"], kind="stable").tail(
                args.demo_matches
            )
        fixtures = fixtures.reset_index(drop=True)
        rates = predict_rates(bundle, fixtures)
        fixtures = _add_names(
            fixtures.assign(lam_home=rates["lam_home"], lam_away=rates["lam_away"]),
            args.registry,
        )
        rho = bundle.calibration.rho
        if args.demo:
            odds = simulated_odds(price_fixtures(fixtures, rho), seed=args.seed)
            reference = args.reference_bookmaker or "sim_sharp"
        else:
            odds = _read_table(args.odds)
            reference = args.reference_bookmaker
        coupon = make_coupon(
            fixtures,
            odds,
            rho=rho,
            config=config,
            reference_bookmaker=reference,
            stake_step=args.stake_step,
            currency=args.currency,
            weight_source=weight_source,
            demo=args.demo,
        )
    except (OSError, KeyError, ValueError) as exc:
        log.error("Cannot build the coupon: %s", exc)
        return 1

    text_path, csv_path = save_coupon(coupon, args.output_dir)
    _echo(format_coupon(coupon))
    log.info(
        "Coupon: %d bets from %d fixtures; saved %s and %s",
        len(coupon.bets),
        coupon.n_fixtures,
        text_path,
        csv_path,
    )
    for note in coupon.notes:
        log.warning("Data note: %s", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
