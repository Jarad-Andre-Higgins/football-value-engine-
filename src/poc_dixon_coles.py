"""Phase 1 proof of concept: Dixon-Coles on real results, backtested walk-forward.

Download one or more football-data.co.uk results files into `data/sample/`,
then run::

    python -m src.poc_dixon_coles
    python -m src.poc_dixon_coles --home Arsenal --away Chelsea

The backtest refits the model every matchweek using only earlier matches, just
as the Friday cron job will in production, and compares it with a naive
baseline that always predicts the historical home/draw/away frequencies.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SAMPLE_DIR
from src.dc_baseline import (
    DixonColesModel,
    UnknownTeamError,
    load_matches,
    outcome_probabilities,
)
from src.dixon_coles import FloatArray
from src.logging_config import setup_logging

_PROB_FLOOR = 1e-12


@dataclass(frozen=True)
class BacktestResult:
    """Out-of-sample scores from :func:`walk_forward_backtest` (lower loss is better).

    Attributes:
        n_predicted: Matches that were predicted and scored.
        n_skipped: Matches skipped because a team was unseen before the cut-off.
        n_refits: Number of weekly refits performed.
        log_loss_model: Mean 1X2 log-loss of the Dixon-Coles model.
        log_loss_baseline: Mean 1X2 log-loss of the frequency baseline.
        accuracy: Share of matches where the most likely 1X2 outcome happened.
        scoreline_loss_dc: Mean `-log P(actual score)` with the fitted rho.
        scoreline_loss_poisson: The same with rho forced to 0 (plain Poisson).
    """

    n_predicted: int
    n_skipped: int
    n_refits: int
    log_loss_model: float
    log_loss_baseline: float
    accuracy: float
    scoreline_loss_dc: float
    scoreline_loss_poisson: float


def walk_forward_backtest(
    matches: pd.DataFrame,
    test_fraction: float = 0.3,
    half_life_days: float = 365.0,
) -> BacktestResult:
    """Refit weekly on past matches only, then score the matches that follow.

    Args:
        matches: Standard match schema, in any order.
        test_fraction: Share of the most recent matches used as the test period.
        half_life_days: Time-decay half-life passed to the model.

    Returns:
        Aggregated out-of-sample scores.

    Raises:
        ValueError: If `test_fraction` is not in (0, 1) or no match could be
            predicted.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    ordered = matches.sort_values("date", kind="stable").reset_index(drop=True)
    test_start = ordered["date"].iloc[int(len(ordered) * (1.0 - test_fraction))]
    test = ordered.loc[ordered["date"] >= test_start]

    model_losses: list[float] = []
    baseline_losses: list[float] = []
    hits: list[bool] = []
    dc_scoreline: list[float] = []
    poisson_scoreline: list[float] = []
    skipped = 0
    refits = 0

    for _, week in test.groupby(test["date"].dt.to_period("W"), sort=True):
        as_of = week["date"].min()
        model = DixonColesModel(half_life_days).fit(ordered, as_of=as_of)
        refits += 1
        baseline = _outcome_frequencies(ordered.loc[ordered["date"] < as_of])

        for row in week.itertuples(index=False):
            try:
                matrix = model.predict_matrix(row.home_team, row.away_team)
            except UnknownTeamError:
                skipped += 1
                continue
            poisson = model.predict_matrix(row.home_team, row.away_team, rho=0.0)
            probs = outcome_probabilities(matrix)
            actual = _outcome_index(row.home_goals, row.away_goals)

            model_losses.append(-math.log(max(probs[actual], _PROB_FLOOR)))
            baseline_losses.append(-math.log(max(baseline[actual], _PROB_FLOOR)))
            hits.append(int(np.argmax(probs)) == actual)
            dc_scoreline.append(_scoreline_loss(matrix, row.home_goals, row.away_goals))
            poisson_scoreline.append(_scoreline_loss(poisson, row.home_goals, row.away_goals)
            )

    if not model_losses:
        raise ValueError("No test match could be predicted")
    return BacktestResult(
        n_predicted=len(model_losses),
        n_skipped=skipped,
        n_refits=refits,
        log_loss_model=float(np.mean(model_losses)),
        log_loss_baseline=float(np.mean(baseline_losses)),
        accuracy=float(np.mean(hits)),
        scoreline_loss_dc=float(np.mean(dc_scoreline)),
        scoreline_loss_poisson=float(np.mean(poisson_scoreline)),
    )


def _outcome_index(home_goals: int, away_goals: int) -> int:
    """Return 0 for a home win, 1 for a draw and 2 for an away win."""
    if home_goals > away_goals:
        return 0
    return 1 if home_goals == away_goals else 2


def _outcome_frequencies(train: pd.DataFrame) -> tuple[float, float, float]:
    """Historical `(home win, draw, away win)` frequencies of `train`."""
    home = float((train["home_goals"] > train["away_goals"]).mean())
    draw = float((train["home_goals"] == train["away_goals"]).mean())
    return home, draw, 1.0 - home - draw


def _scoreline_loss(matrix: FloatArray, home_goals: int, away_goals: int) -> float:
    """``-log`` of the probability the matrix gave to the score that happened."""
    size = matrix.shape[0]
    if home_goals >= size or away_goals >= size:
        return -math.log(_PROB_FLOOR)
    return -math.log(max(float(matrix[home_goals, away_goals]), _PROB_FLOOR))


def _log_backtest(log: logging.Logger, result: BacktestResult) -> None:
    """Write the backtest summary to the log."""
    log.info(
        "Walk-forward backtest: %d matches predicted (%d skipped) over %d weekly refits",
        result.n_predicted,
        result.n_skipped,
        result.n_refits,
    )
    log.info(
        "  1X2 log-loss (lower is better): model=%.4f baseline=%.4f",
        result.log_loss_model,
        result.log_loss_baseline,
    )
    log.info("  1X2 accuracy: %.1f%%", 100 * result.accuracy)
    log.info(
        "  Scoreline log-loss (lower is better): dixon-coles=%.4f plain-poisson=%.4f",
        result.scoreline_loss_dc,
        result.scoreline_loss_poisson,
    )
    if result.log_loss_model >= result.log_loss_baseline:
        log.warning(
            "The model did NOT beat the naive baseline; investigate before going on"
        )
    else:
        log.info("The model beats the naive baseline out of sample")


def _log_ratings(log: logging.Logger, model: DixonColesModel) -> None:
    """Write the fitted parameters and best/worst team ratings to the log."""
    params = model.params
    if params is None:
        raise RuntimeError("Model is not fitted")
    log.info(
        "Final fit on %d matches before %s: home advantage x%.2f goals, rho=%.4f, converged=%s",
        model.n_matches,
        model.as_of.date() if model.as_of is not None else "n/a",
        math.exp(params.home_adv),
        params.rho,
        model.converged,
    )
    table = model.ratings()
    fmt = "{:+.3f}".format
    log.info("Top 5 teams:\n%s", table.head(5).to_string(index=False, float_format=fmt))
    log.info(
        "Bottom 5 teams:\n%s", table.tail(5).to_string(index=False, float_format=fmt)
    )


def _log_fixture(
    log: logging.Logger, model: DixonColesModel, home: str | None, away: str | None
) -> None:
    """Predict one fixture (default: best-rated team hosts the second best)."""
    if home is None or away is None:
        ratings = model.ratings()
        home, away = str(ratings.loc[0, "team"]), str(ratings.loc[1, "team"])
    lam, mu = model.expected_goals(home, away)
    matrix = model.predict_matrix(home, away)
    home_win, draw, away_win = outcome_probabilities(matrix)
    goals = np.arange(matrix.shape[0])
    over_25 = float(matrix[np.add.outer(goals, goals) >= 3].sum())
    btts = float(matrix[1:, 1:].sum())

    log.info("Fixture: %s v %s (expected goals %.2f - %.2f)", home, away, lam, mu)
    log.info(
        "  1X2: home=%.1f%% draw=%.1f%% away=%.1f%%",
        100 * home_win,
        100 * draw,100 * away_win,
    )
    log.info("  Over 2.5: %.1f%% | BTTS yes: %.1f%%", 100 * over_25, 100 * btts)
    for flat_index in np.argsort(matrix, axis=None)[::-1][:5]:
        home_goals, away_goals = np.unravel_index(flat_index, matrix.shape)
        log.info(
            "  Likely score %d-%d: %.2f%%",
            home_goals,
            away_goals,
            100 * matrix[home_goals, away_goals],
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dixon-Coles proof of concept on football-data.co.uk results."
    )
    parser.add_argument("--home", help="Home team of the demo fixture")
    parser.add_argument("--away", help="Away team of the demo fixture")
    parser.add_argument(
        "--half-life",
        type=float,
        default=365.0,
        help="time-decay half-life in days (default: 365)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="share of the most recent matches used for the backtest (default: 0.3)",
    )
    args = parser.parse_args(argv)
    if (args.home is None) != (args.away is None):
        parser.error("--home and --away must be given together")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the proof of concept and return the process exit code."""
    args = _parse_args(argv)
    log = setup_logging("poc_dixon_coles")
    paths = sorted(SAMPLE_DIR.glob("*.csv"))
    if not paths:
        log.error("No CSV files found in %s; download the results first", SAMPLE_DIR)
        return 1

    matches = load_matches(paths)
    teams = set(matches["home_team"]) | set(matches["away_team"])
    log.info(
        "Dataset: %d matches from %s to %s, %d teams",
        len(matches),
        matches["date"].min().date(),
        matches["date"].max().date(),
        len(teams),
    )

    _log_backtest(
        log, walk_forward_backtest(matches, args.test_fraction, args.half_life)
    )
    model = DixonColesModel(args.half_life).fit(matches)
    _log_ratings(log, model)
    try:
        _log_fixture(log, model, args.home, args.away)
    except UnknownTeamError as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
