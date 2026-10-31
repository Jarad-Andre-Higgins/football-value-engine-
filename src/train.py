"""Walk-forward evaluation of the LightGBM xG model.

Refits the two xG models on a schedule using only matches played before each
refit date, predicts the matches that follow, and scores those out-of-sample
predictions against two simple baselines and the bookmakers' closing odds.
The predictions are saved too, because the backtest of a later phase needs
honest out-of-sample probabilities for past matches.

Usage::

    python -m src.train
    python -m src.train --refit-days 56      # coarser refits, faster
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.enrich import ENRICHED_OUTPUT
from src.evaluate import evaluation_report, format_report
from src.features import FEATURES_OUTPUT, TARGET_COLUMNS, feature_columns
from src.logging_config import setup_logging
from src.model import (
    FIRST_TRAIN_SEASON,
    ModelParams,
    feature_importance,
    fit_boosters,
    refit_windows,
    training_mask,
    walk_forward_predict,
)

logger = logging.getLogger("src.train")

OOF_OUTPUT = PROCESSED_DIR / "oof_predictions.parquet"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluation of the xG model."
    )
    parser.add_argument("--features", type=Path, default=FEATURES_OUTPUT)
    parser.add_argument(
        "--enriched", type=Path, default=ENRICHED_OUTPUT, help="for bookmaker odds"
    )
    parser.add_argument("--oof-output", type=Path, default=OOF_OUTPUT)
    parser.add_argument("--first-eval-season", type=int, default=2022)
    parser.add_argument("--test-from-season", type=int, default=2025)
    parser.add_argument("--refit-days", type=int, default=28)
    parser.add_argument(
        "--threads", type=int, default=0, help="CPU threads (0 = automatic)"
    )
    args = parser.parse_args(argv)
    if args.test_from_season <= args.first_eval_season:
        parser.error("--test-from-season must be later than --first-eval-season")
    return args


def _verdict(report: pd.DataFrame) -> list[str]:
    """Plain-language comparison of the model with each baseline on the test slice."""
    test = report[report["slice"] == "test"].set_index("model")
    if "lightgbm" not in test.index:
        return []
    ours = test.loc["lightgbm", "logloss_1x2"]
    lines = []
    for name in ("league_avg", "strength_only"):
        if name in test.index:
            gap = test.loc[name, "logloss_1x2"] - ours
            word = "better" if gap > 0 else "WORSE"
            lines.append(f"1X2 log-loss vs {name}: {word} by {abs(gap):.3f}")
    if "market_closing" in test.index:
        gap = ours - test.loc["market_closing", "logloss_1x2"]
        lines.append(
            f"1X2 log-loss gap to the closing market: {gap:+.3f} (negative would beat it)"
        )
    if lines:
        lines.append("Gaps below about 0.003 are noise at this sample size")
    return lines


def main(argv: list[str] | None = None) -> int:
    """Evaluate the model walk-forward and return the process exit code.

    Returns ``0`` on success and ``1`` when inputs are missing or unusable.
    """
    args = _parse_args(argv)
    log = setup_logging("train")
    if not args.features.exists():
        log.error(
            "Features not found: %s; run python -m src.features first", args.features
        )
        return 1
    features = pd.read_parquet(args.features)
    missing = [
        c for c in (*feature_columns(), *TARGET_COLUMNS) if c not in features.columns
    ]
    if missing:
        log.error("Features table is missing columns (rebuild it): %s", missing[:5])
        return 1
    enriched = None
    if args.enriched.exists():
        enriched = pd.read_parquet(args.enriched)
    else:
        log.warning(
            "No odds table at %s; the market benchmark is skipped", args.enriched
        )

    params = ModelParams(num_threads=args.threads)
    try:
        windows = refit_windows(features, args.first_eval_season, args.refit_days)
        usable = int(training_mask(features).sum())
        log.info(
            "Protocol: %d usable training rows (seasons from %d), %d refits every %d days, "
            "predicting from season %d; test slice starts at season %d",
            usable,
            FIRST_TRAIN_SEASON,
            len(windows),
            args.refit_days,
            args.first_eval_season,
            args.test_from_season,
        )
        started = time.time()
        oof = walk_forward_predict(
            features,
            params,
            first_eval_season=args.first_eval_season,
            refit_days=args.refit_days,
        )
        log.info(
            "Walk-forward predictions: %d matches in %.0f s",
            len(oof),
            time.time() - started,
        )
        args.oof_output.parent.mkdir(parents=True, exist_ok=True)
        oof.to_parquet(args.oof_output, index=False)

        report = evaluation_report(
            oof, features, enriched, test_from_season=args.test_from_season
        )
    except ValueError as exc:
        log.error("Cannot evaluate: %s", exc)
        return 1
    log.info(
        "Results (lower is better except accuracy); calibrated on the dev slice:\n%s",
        format_report(report),
    )
    for line in _verdict(report):
        log.info(line)

    final = fit_boosters(
        features,
        training_mask(features),
        params,
        as_of=features["date"].max() + pd.Timedelta(1, unit="D"),
    )
    top = feature_importance(final).head(12)
    log.info(
        "Most useful features (share of total gain, %%):\n%s",
        top.to_string(index=False, float_format="{:.1f}".format),
    )
    log.info("Wrote %s", args.oof_output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
