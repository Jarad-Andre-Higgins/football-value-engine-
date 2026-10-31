"""LightGBM expected-goals models and walk-forward prediction.

Two gradient-boosted regressors, one per side, predict how much xG each team
will create from the leak-free features of :mod:`src.features`. The Poisson
objective suits a non-negative rate: predictions are always positive and
effects are multiplicative.

Honest evaluation needs predictions made as they would have been in real
life, so :func:`walk_forward_predict` refits on a schedule using only matches
played before each refit date and predicts the matches that follow. Those
out-of-sample predictions also feed the backtest in a later phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import feature_columns

logger = logging.getLogger("src.model")

FIRST_TRAIN_SEASON: Final[int] = 2020  # 2019/20 is warm-up: form and Elo start cold
MIN_TRAIN_ROWS: Final[int] = 300


@dataclass(frozen=True)
class ModelParams:
    """Hyper-parameters for both xG models.

    Attributes:
        n_estimators: Boosting rounds.
        learning_rate: Shrinkage per round.
        num_leaves: Leaves per tree (small trees resist overfitting).
        min_child_samples: Minimum rows in a leaf.
        feature_fraction: Share of features tried per tree.
        bagging_fraction: Share of rows sampled per round.
        lambda_l2: L2 regularisation.
        lambda_l1: L1 regularisation.
        sample_half_life_days: Training rows lose half their weight this many
            days before the refit date (football drifts, so recent matches count more).
        seed: Random seed (fixed, so training is reproducible).
        num_threads: CPU threads; ``0`` lets LightGBM decide.
    """

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 8
    min_child_samples: int = 50
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.8
    lambda_l2: float = 10.0
    lambda_l1: float = 0.0
    sample_half_life_days: float = 730.0
    seed: int = 42
    num_threads: int = 0

    def lightgbm(self) -> dict[str, Any]:
        """The parameter dictionary handed to ``lightgbm.train``."""
        return {
            "objective": "poisson",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "lambda_l2": self.lambda_l2,
            "lambda_l1": self.lambda_l1,
            "seed": self.seed,
            "num_threads": self.num_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }


@dataclass(frozen=True)
class Boosters:
    """The fitted home-side and away-side models."""

    home: lgb.Booster
    away: lgb.Booster


@dataclass(frozen=True)
class Window:
    """One refit: train on matches before ``start``, predict ``start <= date < end``."""

    start: pd.Timestamp
    end: pd.Timestamp


def training_mask(features: pd.DataFrame) -> pd.Series:
    """Rows usable for training: after the warm-up season, with both xG targets known."""
    return (
        (features["season"] >= FIRST_TRAIN_SEASON)
        & features["target_home_xg"].notna()
        & features["target_away_xg"].notna()
    )


def fit_boosters(
    features: pd.DataFrame,
    train_mask: pd.Series,
    params: ModelParams,
    *,
    as_of: pd.Timestamp,
) -> Boosters:
    """Fit the home and away xG models on the selected rows.

    Args:
        features: The feature table from ``python -m src.features``.
        train_mask: Which rows to train on; every one must be dated before ``as_of``.
        params: Hyper-parameters.
        as_of: Reference date for the recency weights.

    Raises:
        ValueError: If a training row is not before ``as_of`` or there are too few rows.
    """
    train = features.loc[train_mask]
    if len(train) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_TRAIN_ROWS} training rows, found {len(train)}"
        )
    if (train["date"] >= as_of).any():
        raise ValueError("training rows must all be dated before as_of")
    inputs = train[feature_columns()].astype("float64")
    age_days = (as_of - train["date"]).dt.days.to_numpy(dtype="float64")
    weights = np.exp2(-age_days / params.sample_half_life_days)
    fitted = {}
    for side in ("home", "away"):
        dataset = lgb.Dataset(
            inputs,
            label=train[f"target_{side}_xg"].to_numpy(dtype="float64"),
            weight=weights,
        )
        fitted[side] = lgb.train(
            params.lightgbm(), dataset, num_boost_round=params.n_estimators
        )
    return Boosters(fitted["home"], fitted["away"])


def predict_xg(boosters: Boosters, features: pd.DataFrame) -> pd.DataFrame:
    """Predict the xG of both sides for every row of ``features``."""
    inputs = features[feature_columns()].astype("float64")
    return pd.DataFrame(
        {
            "pred_home_xg": boosters.home.predict(inputs),
            "pred_away_xg": boosters.away.predict(inputs),
        },
        index=features.index,
    )


def refit_windows(
    features: pd.DataFrame, first_eval_season: int, step_days: int
) -> list[Window]:
    """Split the evaluation period into consecutive refit windows.

    Args:
        features: The feature table.
        first_eval_season: Start year of the first season to predict.
        step_days: Days between refits (production refits weekly; evaluation can be coarser).

    Raises:
        ValueError: If ``step_days`` is not positive or there is nothing to evaluate.
    """
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days}")
    evaluated = features.loc[features["season"] >= first_eval_season, "date"]
    if evaluated.empty:
        raise ValueError(f"No matches from season {first_eval_season} onward")
    start, stop = evaluated.min(), features["date"].max() + pd.Timedelta(1, unit="D")
    boundaries = list(pd.date_range(start, stop, freq=f"{step_days}D"))
    if boundaries[-1] < stop:
        boundaries.append(stop)
    return [Window(a, b) for a, b in pairwise(boundaries)]


def walk_forward_predict(
    features: pd.DataFrame,
    params: ModelParams,
    *,
    first_eval_season: int,
    refit_days: int,
) -> pd.DataFrame:
    """Out-of-sample xG predictions from models refitted on a schedule.

    For each window the models are trained only on usable matches played
    before the window starts, then predict every match inside the window, so
    no prediction ever uses a result from its own future.

    Returns:
        One row per predicted match: ``match_id``, ``date``, ``season``,
        ``league``, ``pred_home_xg``, ``pred_away_xg`` and ``window_start``.
    """
    usable = training_mask(features)
    pieces = []
    for window in refit_windows(features, first_eval_season, refit_days):
        in_window = (features["date"] >= window.start) & (features["date"] < window.end)
        if not in_window.any():
            continue
        boosters = fit_boosters(
            features,
            usable & (features["date"] < window.start),
            params,
            as_of=window.start,
        )
        predictions = predict_xg(boosters, features.loc[in_window])
        meta = features.loc[in_window, ["match_id", "date", "season", "league"]]
        pieces.append(
            pd.concat([meta, predictions], axis=1).assign(window_start=window.start)
        )
        logger.debug(
            "Refit for %s: %d matches predicted",
            window.start.date(),
            int(in_window.sum()),
        )
    return pd.concat(pieces, ignore_index=True)


def feature_importance(boosters: Boosters) -> pd.DataFrame:
    """Gain-based importance of every feature for each side's model, best first."""
    names = boosters.home.feature_name()
    table = pd.DataFrame(
        {
            "feature": names,
            "home_gain": boosters.home.feature_importance(importance_type="gain"),
            "away_gain": boosters.away.feature_importance(importance_type="gain"),
        }
    )
    for column in ("home_gain", "away_gain"):
        table[column] = 100.0 * table[column] / table[column].sum()
    table["mean_gain"] = table[["home_gain", "away_gain"]].mean(axis=1)
    return table.sort_values("mean_gain", ascending=False).reset_index(drop=True)
