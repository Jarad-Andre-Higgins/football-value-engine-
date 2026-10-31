"""Hyper-parameter tuning for the xG models, using development seasons only.

Tuning picks the settings that predict *future seasons* best. Each fold trains
on every usable match before a development season and scores the predictions
on that season, so the search never sees the test seasons. The score is the
Poisson deviance of the predicted xG, averaged over both sides and all folds.

The default settings are always the first trial, so tuning can only match or
improve on them (on the development folds), and weak trials are stopped early
by a median pruner to save time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.features import feature_columns
from src.model import MIN_TRAIN_ROWS, ModelParams, training_mask

logger = logging.getLogger("src.tuning")

# Search ranges: (low, high), sampled on a log scale where marked in ``tune_params``.
LEARNING_RATE: Final[tuple[float, float]] = (0.01, 0.1)
NUM_LEAVES: Final[tuple[int, int]] = (4, 31)
MIN_CHILD_SAMPLES: Final[tuple[int, int]] = (20, 200)
FEATURE_FRACTION: Final[tuple[float, float]] = (0.4, 1.0)
BAGGING_FRACTION: Final[tuple[float, float]] = (0.6, 1.0)
LAMBDA_L2: Final[tuple[float, float]] = (0.01, 50.0)
LAMBDA_L1: Final[tuple[float, float]] = (0.001, 10.0)
HALF_LIFE_DAYS: Final[tuple[float, float]] = (180.0, 3650.0)


def poisson_deviance(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Poisson unit deviance (0 for a perfect prediction; lower is better)."""
    mean = np.clip(predicted, 1e-6, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(actual > 0, actual * np.log(actual / mean), 0.0)
    return float(np.mean(2.0 * (log_ratio - (actual - mean))))


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold: train before a season, validate on that season."""

    season: int
    train_inputs: pd.DataFrame
    train_age_days: np.ndarray
    train_targets: dict[str, np.ndarray]
    valid_inputs: pd.DataFrame
    valid_targets: dict[str, np.ndarray]


@dataclass(frozen=True)
class TuningResult:
    """Outcome of :func:`tune_params`.

    Attributes:
        best_params: The best settings found.
        best_value: Their mean validation deviance.
        default_value: The deviance of the default settings (the first trial).
        n_trials: Trials started.
        n_pruned: Trials stopped early.
        fold_seasons: The seasons used as validation folds.
    """

    best_params: ModelParams
    best_value: float
    default_value: float
    n_trials: int
    n_pruned: int
    fold_seasons: tuple[int, ...]


def make_folds(features: pd.DataFrame, fold_seasons: Sequence[int]) -> list[Fold]:
    """Build one fold per season: train strictly before it, validate inside it.

    Raises:
        ValueError: If a season is absent or its training set is too small.
    """
    usable = training_mask(features)
    folds = []
    for season in fold_seasons:
        in_season = features["season"] == season
        if not in_season.any():
            raise ValueError(f"No matches in season {season}")
        start = features.loc[in_season, "date"].min()
        train = features.loc[usable & (features["date"] < start)]
        valid = features.loc[usable & in_season]
        if len(train) < MIN_TRAIN_ROWS or valid.empty:
            raise ValueError(
                f"Season {season}: need at least {MIN_TRAIN_ROWS} earlier training rows "
                f"and some validation rows (found {len(train)} and {len(valid)})"
            )
        folds.append(
            Fold(
                season=season,
                train_inputs=train[feature_columns()].astype("float64"),
                train_age_days=(start - train["date"]).dt.days.to_numpy(
                    dtype="float64"
                ),
                train_targets={
                    s: train[f"target_{s}_xg"].to_numpy(dtype="float64")
                    for s in ("home", "away")
                },
                valid_inputs=valid[feature_columns()].astype("float64"),
                valid_targets={
                    s: valid[f"target_{s}_xg"].to_numpy(dtype="float64")
                    for s in ("home", "away")
                },
            )
        )
    return folds


def score_fold(fold: Fold, params: ModelParams) -> float:
    """Train both sides on the fold's past and return the mean validation deviance."""
    weights = np.exp2(-fold.train_age_days / params.sample_half_life_days)
    deviances = []
    for side in ("home", "away"):
        dataset = lgb.Dataset(
            fold.train_inputs, label=fold.train_targets[side], weight=weights
        )
        booster = lgb.train(
            params.lightgbm(), dataset, num_boost_round=params.n_estimators
        )
        predicted = np.asarray(booster.predict(fold.valid_inputs), dtype="float64")
        deviances.append(poisson_deviance(predicted, fold.valid_targets[side]))
    return float(np.mean(deviances))


def tune_params(
    features: pd.DataFrame,
    *,
    fold_seasons: Sequence[int],
    n_trials: int = 40,
    seed: int = 42,
    threads: int = 0,
    max_estimators: int = 700,
) -> TuningResult:
    """Search for the settings that predict future seasons best.

    Args:
        features: The feature table.
        fold_seasons: Development seasons used as validation folds. Keep the test
            seasons out of this list, so they stay unseen.
        n_trials: Number of settings to try (the defaults count as the first).
        seed: Seed for the sampler and the models, for reproducibility.
        threads: CPU threads for LightGBM (``0`` = automatic).
        max_estimators: Upper limit on boosting rounds in the search.

    Returns:
        The best settings with their score and the defaults' score.
    """
    folds = make_folds(features, fold_seasons)
    low_trees = max(10, max_estimators // 7)
    defaults = ModelParams(seed=seed, num_threads=threads)

    def clip(value: float, bounds: tuple[float, float]) -> float:
        return float(min(max(value, bounds[0]), bounds[1]))

    def suggest(trial: optuna.Trial) -> ModelParams:
        return ModelParams(
            n_estimators=trial.suggest_int("n_estimators", low_trees, max_estimators),
            learning_rate=trial.suggest_float(
                "learning_rate", *LEARNING_RATE, log=True
            ),
            num_leaves=trial.suggest_int("num_leaves", *NUM_LEAVES),
            min_child_samples=trial.suggest_int(
                "min_child_samples", *MIN_CHILD_SAMPLES
            ),
            feature_fraction=trial.suggest_float("feature_fraction", *FEATURE_FRACTION),
            bagging_fraction=trial.suggest_float("bagging_fraction", *BAGGING_FRACTION),
            lambda_l2=trial.suggest_float("lambda_l2", *LAMBDA_L2, log=True),
            lambda_l1=trial.suggest_float("lambda_l1", *LAMBDA_L1, log=True),
            sample_half_life_days=trial.suggest_float(
                "sample_half_life_days", *HALF_LIFE_DAYS, log=True
            ),
            seed=seed,
            num_threads=threads,
        )

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        scores: list[float] = []
        for step, fold in enumerate(folds):
            scores.append(score_fold(fold, params))
            trial.report(float(np.mean(scores)), step)
            if step < len(folds) - 1 and trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    def log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            logger.info(
                "Trial %2d: deviance %.4f (best so far %.4f)",
                trial.number,
                trial.value,
                study.best_value,
            )
        else:
            logger.info("Trial %2d: stopped early", trial.number)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.enqueue_trial(
        {
            "n_estimators": int(
                min(max(defaults.n_estimators, low_trees), max_estimators)
            ),
            "learning_rate": clip(defaults.learning_rate, LEARNING_RATE),
            "num_leaves": int(clip(defaults.num_leaves, NUM_LEAVES)),
            "min_child_samples": int(
                clip(defaults.min_child_samples, MIN_CHILD_SAMPLES)
            ),
            "feature_fraction": clip(defaults.feature_fraction, FEATURE_FRACTION),
            "bagging_fraction": clip(defaults.bagging_fraction, BAGGING_FRACTION),
            "lambda_l2": clip(defaults.lambda_l2, LAMBDA_L2),
            "lambda_l1": clip(defaults.lambda_l1, LAMBDA_L1),
            "sample_half_life_days": clip(
                defaults.sample_half_life_days, HALF_LIFE_DAYS
            ),
        }
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[log_trial])

    pruned = sum(
        trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials
    )
    first = study.trials[0]
    return TuningResult(
        best_params=ModelParams(**study.best_params, seed=seed, num_threads=threads),
        best_value=float(study.best_value),
        default_value=float(first.value) if first.value is not None else float("nan"),
        n_trials=len(study.trials),
        n_pruned=int(pruned),
        fold_seasons=tuple(int(season) for season in fold_seasons),
    )


def save_params(path: Path, params: ModelParams) -> None:
    """Write ``params`` to a readable JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(params), indent=2) + "\n", encoding="utf-8")


def load_params(path: Path) -> ModelParams:
    """Read settings written by :func:`save_params`.

    Raises:
        ValueError: If the file holds unknown setting names.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {field.name for field in fields(ModelParams)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{path.name}: unknown settings {unknown}")
    return ModelParams(**data)
