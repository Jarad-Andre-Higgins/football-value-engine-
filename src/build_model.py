"""Tune, evaluate, fit and save the xG model (the end of Phase 4).

1. Tune the settings on development seasons only (``src.tuning``).
2. Evaluate default and tuned settings walk-forward against the baselines and
   the closing market, so the table shows what tuning actually bought.
3. Calibrate on the out-of-sample predictions, fit the final models on every
   usable match, and save everything to ``models/xg_model.pkl``.

Usage::

    python -m src.build_model                 # full run: tune, compare, save
    python -m src.build_model --trials 20     # quicker search
    python -m src.build_model --no-tune       # reuse models/best_params.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.bundle import DEFAULT_MODEL_PATH, build_bundle, save_bundle, summary
from src.config import MODELS_DIR
from src.enrich import ENRICHED_OUTPUT
from src.evaluate import evaluation_report, fit_calibration, format_report
from src.features import FEATURES_OUTPUT, TARGET_COLUMNS, feature_columns
from src.logging_config import setup_logging
from src.model import (
    ModelParams,
    feature_importance,
    fit_boosters,
    training_mask,
    walk_forward_predict,
)
from src.train import OOF_OUTPUT
from src.train import _verdict as verdict_lines
from src.tuning import load_params, save_params, tune_params

logger = logging.getLogger("src.build_model")

PARAMS_PATH = MODELS_DIR / "best_params.json"
METRIC_COLUMNS = (
    "n",
    "mae_xg",
    "rmse_xg",
    "goals_nll",
    "scoreline_nll",
    "logloss_1x2",
    "acc_1x2",
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune, evaluate and save the xG model."
    )
    parser.add_argument("--features", type=Path, default=FEATURES_OUTPUT)
    parser.add_argument(
        "--enriched", type=Path, default=ENRICHED_OUTPUT, help="for bookmaker odds"
    )
    parser.add_argument("--oof-output", type=Path, default=OOF_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--params-file", type=Path, default=PARAMS_PATH)
    parser.add_argument("--first-eval-season", type=int, default=2022)
    parser.add_argument("--test-from-season", type=int, default=2025)
    parser.add_argument("--refit-days", type=int, default=28)
    parser.add_argument(
        "--trials", type=int, default=40, help="settings to try while tuning"
    )
    parser.add_argument(
        "--max-estimators",
        type=int,
        default=700,
        help="cap on boosting rounds in the search",
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="CPU threads (0 = automatic)"
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="reuse the saved settings (or defaults)"
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="skip the default-settings comparison run",
    )
    args = parser.parse_args(argv)
    if args.test_from_season <= args.first_eval_season:
        parser.error("--test-from-season must be later than --first-eval-season")
    return args


def _combine(tuned: pd.DataFrame, default: pd.DataFrame | None) -> pd.DataFrame:
    """One table with default and tuned LightGBM rows next to the baselines."""
    tuned_rows = tuned.assign(model=tuned["model"].replace({"lightgbm": "lgbm_tuned"}))
    if default is None:
        return tuned_rows
    default_rows = default[default["model"] == "lightgbm"].assign(model="lgbm_default")
    return pd.concat([default_rows, tuned_rows], ignore_index=True)


def _test_metrics(report: pd.DataFrame) -> dict[str, float]:
    """Test-slice scores of the model (and the market's log-loss) for the bundle."""
    test = report[report["slice"] == "test"].set_index("model")
    metrics: dict[str, float] = {}
    if "lightgbm" in test.index:
        metrics = {name: float(test.loc["lightgbm", name]) for name in METRIC_COLUMNS}
    if "market_closing" in test.index:
        metrics["market_logloss_1x2"] = float(test.loc["market_closing", "logloss_1x2"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Run the full model build and return the process exit code.

    Returns ``0`` on success and ``1`` when inputs are missing or unusable.
    """
    args = _parse_args(argv)
    log = setup_logging("build_model")
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
    enriched = pd.read_parquet(args.enriched) if args.enriched.exists() else None
    if enriched is None:
        log.warning(
            "No odds table at %s; the market benchmark is skipped", args.enriched
        )

    started = time.time()
    try:
        if args.no_tune:
            saved = args.params_file.exists()
            params = load_params(args.params_file) if saved else ModelParams()
            params = replace(params, num_threads=args.threads)
            log.info(
                "Using %s settings (tuning skipped)", "saved" if saved else "default"
            )
        else:
            seasons = list(range(args.first_eval_season, args.test_from_season))
            log.info(
                "Tuning %d trials on development seasons %s (test seasons stay unseen)",
                args.trials,
                seasons,
            )
            tuned = tune_params(
                features,
                fold_seasons=seasons,
                n_trials=args.trials,
                threads=args.threads,
                max_estimators=args.max_estimators,
            )
            params = tuned.best_params
            save_params(args.params_file, params)
            log.info(
                "Tuning finished: %d trials (%d stopped early); deviance %.4f with default settings, %.4f tuned",
                tuned.n_trials,
                tuned.n_pruned,
                tuned.default_value,
                tuned.best_value,
            )
            shown = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in asdict(params).items()
            }
            log.info("Best settings (saved to %s): %s", args.params_file, shown)

        evaluate = {
            "first_eval_season": args.first_eval_season,
            "refit_days": args.refit_days,
        }
        oof = walk_forward_predict(features, params, **evaluate)
        report = evaluation_report(
            oof, features, enriched, test_from_season=args.test_from_season
        )
        default_report = None
        if not args.no_compare:
            default_oof = walk_forward_predict(
                features, ModelParams(num_threads=args.threads), **evaluate
            )
            default_report = evaluation_report(
                default_oof, features, enriched, test_from_season=args.test_from_season
            )
    except ValueError as exc:
        log.error("Cannot build the model: %s", exc)
        return 1

    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.oof_output, index=False)
    log.info(
        "Results (lower is better except acc_1x2; calibrated on the dev slice, which tuning also saw, "
        "so judge by the test rows):\n%s",
        format_report(_combine(report, default_report)),
    )
    for line in verdict_lines(report):
        log.info(line)

    # Calibration for deployment uses every out-of-sample prediction made above.
    data = oof.merge(
        features[["match_id", "target_home_goals", "target_away_goals"]], on="match_id"
    )
    data = data[np.isfinite(data[["pred_home_xg", "pred_away_xg"]]).all(axis=1)]
    calibration = fit_calibration(
        data["pred_home_xg"].to_numpy(dtype="float64"),
        data["pred_away_xg"].to_numpy(dtype="float64"),
        data["target_home_goals"].to_numpy(dtype="float64"),
        data["target_away_goals"].to_numpy(dtype="float64"),
    )
    log.info(
        "Calibration: home x%.3f, away x%.3f goals per predicted xG, Dixon-Coles rho %.3f",
        calibration.scale_home,
        calibration.scale_away,
        calibration.rho,
    )

    mask = training_mask(features)
    last_match = features.loc[mask, "date"].max()
    boosters = fit_boosters(
        features, mask, params, as_of=last_match + pd.Timedelta(1, unit="D")
    )
    bundle = build_bundle(
        boosters,
        calibration,
        params,
        trained_through=last_match,
        n_train_rows=int(mask.sum()),
        metrics=_test_metrics(report),
    )
    path = save_bundle(bundle, args.model_output)
    log.info(
        "Final models trained on %d matches through %s",
        bundle.n_train_rows,
        last_match.date(),
    )
    log.info(
        "Most useful features (share of total gain, %%):\n%s",
        feature_importance(boosters)
        .head(10)
        .to_string(index=False, float_format="{:.1f}".format),
    )
    log.info(
        "Saved %s and %s (%.0f s total)",
        path,
        path.with_suffix(".json"),
        time.time() - started,
    )
    log.debug("Summary: %s", summary(bundle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
