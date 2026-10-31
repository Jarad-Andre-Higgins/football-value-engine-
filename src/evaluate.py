"""Evaluation of xG models against baselines and the betting market.

A model is judged on three levels, all on matches it never saw:

* **xG accuracy**: MAE and RMSE against the xG actually created.
* **Goal likelihood**: the Poisson log-loss of the goals that were scored,
  given each side's predicted rate.
* **Match outcomes**: the predicted rates go through the Dixon-Coles score
  matrix, and the resulting 1X2 probabilities are scored by log-loss.

Every predictor is calibrated the same way before the goal and outcome
scores: a scale factor turns xG into goals and one Dixon-Coles ``rho`` is
fitted, both on the development period only, so the test period stays clean.
Bookmaker closing odds, with the margin removed, are the benchmark to beat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from src.dc_baseline import outcome_probabilities
from src.dixon_coles import rho_bounds, score_matrix

MIN_RATE: Final[float] = 0.05
MAX_RATE: Final[float] = 6.0
RHO_MARGIN: Final[float] = 1e-6
MARKET_BOOKS: Final[tuple[str, ...]] = ("psc", "avgc", "b365c", "ps", "avg", "b365")
OUTCOMES: Final[tuple[str, ...]] = ("p_home", "p_draw", "p_away")


def poisson_nll(rate: np.ndarray, goals: np.ndarray) -> float:
    """Mean Poisson negative log-likelihood of ``goals`` given ``rate`` (lower is better)."""
    rate = np.clip(rate, MIN_RATE, MAX_RATE)
    return float(np.mean(rate - goals * np.log(rate) + gammaln(goals + 1.0)))


def error_metrics(predicted: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return ``(MAE, RMSE)`` over the entries where ``actual`` is known."""
    known = ~np.isnan(actual)
    error = predicted[known] - actual[known]
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def fit_rho(
    goals_home: np.ndarray, goals_away: np.ndarray, lam: np.ndarray, mu: np.ndarray
) -> float:
    """Maximum-likelihood Dixon-Coles ``rho`` for predicted rates and observed scores.

    Only the 0-0, 0-1, 1-0 and 1-1 cells depend on ``rho``, so the likelihood is
    maximised over those matches within the interval where every correction
    factor stays positive. With rates clipped to ``[MIN_RATE, MAX_RATE]`` that
    interval always contains zero, so it is never empty.
    """
    lam, mu = np.clip(lam, MIN_RATE, MAX_RATE), np.clip(mu, MIN_RATE, MAX_RATE)
    lower = float(np.max(np.maximum(-1.0 / lam, -1.0 / mu)))
    upper = float(np.min(np.minimum(1.0, 1.0 / (lam * mu))))
    m00 = (goals_home == 0) & (goals_away == 0)
    m01 = (goals_home == 0) & (goals_away == 1)
    m10 = (goals_home == 1) & (goals_away == 0)
    m11 = (goals_home == 1) & (goals_away == 1)

    def negative_log_likelihood(rho: float) -> float:
        tau = np.ones_like(lam)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        return -float(np.log(np.maximum(tau, 1e-10)).sum())

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(lower + RHO_MARGIN, upper - RHO_MARGIN),
        method="bounded",
    )
    return float(result.x)


@dataclass(frozen=True)
class Calibration:
    """Turns predicted xG into goal rates and score dependence.

    Attributes:
        scale_home: Goals per predicted xG for the home side.
        scale_away: Goals per predicted xG for the away side.
        rho: Dixon-Coles dependence fitted on the scaled rates.
    """

    scale_home: float
    scale_away: float
    rho: float


def fit_calibration(
    pred_home: np.ndarray,
    pred_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
) -> Calibration:
    """Fit the scale factors (ratio of total goals to total predicted xG) and ``rho``.

    Raises:
        ValueError: If there are no rows to fit on.
    """
    if len(pred_home) == 0:
        raise ValueError("No development rows to calibrate on")
    scale_home = float(goals_home.sum() / pred_home.sum())
    scale_away = float(goals_away.sum() / pred_away.sum())
    rho = fit_rho(
        goals_home, goals_away, scale_home * pred_home, scale_away * pred_away
    )
    return Calibration(scale_home, scale_away, rho)


def outcome_table(
    lam: np.ndarray,
    mu: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
    rho: float,
) -> pd.DataFrame:
    """Score-matrix probabilities for each match.

    Returns:
        Columns ``p_home``, ``p_draw``, ``p_away``, ``scoreline_nll`` (negative log
        of the probability given to the exact score) and ``outcome``
        (0 home win, 1 draw, 2 away win).
    """
    rows = []
    for rate_home, rate_away, home_goals, away_goals in zip(
        lam, mu, goals_home, goals_away
    ):
        rate_home = min(max(float(rate_home), MIN_RATE), MAX_RATE)
        rate_away = min(max(float(rate_away), MIN_RATE), MAX_RATE)
        lower, upper = rho_bounds(rate_home, rate_away)
        usable_rho = min(max(rho, lower + RHO_MARGIN), upper - RHO_MARGIN)
        matrix = score_matrix(rate_home, rate_away, usable_rho)
        home_win, draw, away_win = outcome_probabilities(matrix)
        size = matrix.shape[0]
        in_grid = home_goals < size and away_goals < size
        score_probability = (
            float(matrix[int(home_goals), int(away_goals)]) if in_grid else 1e-12
        )
        rows.append(
            (home_win, draw, away_win, -math.log(max(score_probability, 1e-12)))
        )
    table = pd.DataFrame(rows, columns=[*OUTCOMES, "scoreline_nll"])
    table["outcome"] = np.where(
        goals_home > goals_away, 0, np.where(goals_home == goals_away, 1, 2)
    )
    return table


def log_loss_1x2(probabilities: np.ndarray, outcome: np.ndarray) -> float:
    """Mean negative log-probability given to the 1X2 result that happened."""
    chosen = probabilities[np.arange(len(outcome)), outcome]
    return float(-np.mean(np.log(np.maximum(chosen, 1e-12))))


def accuracy_1x2(probabilities: np.ndarray, outcome: np.ndarray) -> float:
    """Share of matches where the most probable result happened."""
    return float(np.mean(probabilities.argmax(axis=1) == outcome))


def baseline_predictions(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Simple predictors the model must beat, from leak-free features only.

    ``league_avg`` predicts the league's recent scoring level with home
    advantage; ``strength_only`` is the classic multiplicative model (own
    attack times opposing defence over league average) without any learning.
    """
    home_avg, away_avg = (
        features["league_avg_goals_home"],
        features["league_avg_goals_away"],
    )
    mean_avg = (home_avg + away_avg) / 2.0
    league = pd.DataFrame(
        {
            "pred_home_xg": features["league_avg_xg"] * home_avg / mean_avg,
            "pred_away_xg": features["league_avg_xg"] * away_avg / mean_avg,
        },
        index=features.index,
    )
    goals_level = pd.DataFrame({"pred_home_xg": home_avg, "pred_away_xg": away_avg})
    league = league.fillna(goals_level)
    strength = pd.DataFrame(
        {
            "pred_home_xg": features["home_xg_exp"],
            "pred_away_xg": features["away_xg_exp"],
        }
    ).fillna(league)
    return {"league_avg": league, "strength_only": strength}


def market_probabilities(enriched: pd.DataFrame) -> pd.DataFrame:
    """Margin-free 1X2 probabilities from the best available bookmaker odds.

    Books are tried in order of sharpness (Pinnacle closing first); each match
    uses the first book that quotes all three outcomes. Probabilities are the
    inverse odds scaled to sum to one.

    Returns:
        ``p_home``, ``p_draw``, ``p_away`` indexed by ``match_id`` (NaN if no odds).
    """
    result = pd.DataFrame(np.nan, index=enriched["match_id"], columns=list(OUTCOMES))
    for book in MARKET_BOOKS:
        columns = [f"{book}_h", f"{book}_d", f"{book}_a"]
        if not all(column in enriched.columns for column in columns):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = 1.0 / enriched[columns].to_numpy(dtype="float64")
            probabilities = implied / implied.sum(axis=1, keepdims=True)
        candidate = pd.DataFrame(
            probabilities, index=enriched["match_id"], columns=list(OUTCOMES)
        )
        result = result.fillna(candidate.replace([np.inf, -np.inf], np.nan))
    return result


def _evaluate(
    name: str,
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    is_dev: np.ndarray,
    is_test: np.ndarray,
) -> list[dict[str, object]]:
    """Score one predictor on the development and test slices."""
    pred_home = predictions["pred_home_xg"].to_numpy(dtype="float64")
    pred_away = predictions["pred_away_xg"].to_numpy(dtype="float64")
    goals_home = data["target_home_goals"].to_numpy(dtype="float64")
    goals_away = data["target_away_goals"].to_numpy(dtype="float64")
    calibration = fit_calibration(
        pred_home[is_dev], pred_away[is_dev], goals_home[is_dev], goals_away[is_dev]
    )
    rows: list[dict[str, object]] = []
    for label, selected in (("dev", is_dev), ("test", is_test)):
        if not selected.any():
            continue
        ph, pa = pred_home[selected], pred_away[selected]
        gh, ga = goals_home[selected], goals_away[selected]
        mae_h, rmse_h = error_metrics(
            ph, data.loc[selected, "target_home_xg"].to_numpy(dtype="float64")
        )
        mae_a, rmse_a = error_metrics(
            pa, data.loc[selected, "target_away_xg"].to_numpy(dtype="float64")
        )
        lam, mu = calibration.scale_home * ph, calibration.scale_away * pa
        table = outcome_table(lam, mu, gh, ga, calibration.rho)
        probabilities = table[list(OUTCOMES)].to_numpy()
        outcome = table["outcome"].to_numpy()
        rows.append(
            {
                "model": name,
                "slice": label,
                "n": int(selected.sum()),
                "mae_xg": (mae_h + mae_a) / 2,
                "rmse_xg": (rmse_h + rmse_a) / 2,
                "goals_nll": poisson_nll(
                    np.concatenate([lam, mu]), np.concatenate([gh, ga])
                ),
                "scoreline_nll": float(table["scoreline_nll"].mean()),
                "logloss_1x2": log_loss_1x2(probabilities, outcome),
                "acc_1x2": accuracy_1x2(probabilities, outcome),
                "rho": calibration.rho,
            }
        )
    return rows


def evaluation_report(
    oof: pd.DataFrame,
    features: pd.DataFrame,
    enriched: pd.DataFrame | None,
    *,
    test_from_season: int,
) -> pd.DataFrame:
    """Compare the model with baselines (and the market) on identical matches.

    Args:
        oof: Walk-forward predictions from :func:`src.model.walk_forward_predict`.
        features: The feature table (targets and baseline inputs).
        enriched: The enriched match table, for bookmaker odds; ``None`` skips the market.
        test_from_season: First season of the test slice; earlier predicted seasons are
            the development slice used for calibration.

    Returns:
        One row per model and slice; the market row only has outcome metrics.

    Raises:
        ValueError: If the development slice is empty.
    """
    wanted = [
        "match_id",
        "target_home_goals",
        "target_away_goals",
        "target_home_xg",
        "target_away_xg",
    ]
    data = oof[["match_id", "season", "pred_home_xg", "pred_away_xg"]].merge(
        features[wanted], on="match_id", how="left"
    )
    baselines = {
        name: frame.assign(match_id=features["match_id"])
        .set_index("match_id")
        .loc[data["match_id"]]
        .reset_index(drop=True)
        for name, frame in baseline_predictions(features).items()
    }
    models = {"lightgbm": data[["pred_home_xg", "pred_away_xg"]], **baselines}
    finite = np.logical_and.reduce(
        [
            np.isfinite(frame.to_numpy(dtype="float64")).all(axis=1)
            for frame in models.values()
        ]
    )
    data = data.loc[finite].reset_index(drop=True)
    models = {
        name: frame.loc[finite].reset_index(drop=True) for name, frame in models.items()
    }

    is_test = (data["season"] >= test_from_season).to_numpy()
    is_dev = ~is_test
    if not is_dev.any():
        raise ValueError(
            "The development slice is empty; lower test_from_season or first_eval_season"
        )
    rows: list[dict[str, object]] = []
    for name, frame in models.items():
        rows.extend(_evaluate(name, frame, data, is_dev, is_test))

    if enriched is not None:
        market = market_probabilities(enriched).reindex(data["match_id"])
        has_odds = market.notna().all(axis=1).to_numpy()
        goals_home = data["target_home_goals"].to_numpy()
        goals_away = data["target_away_goals"].to_numpy()
        outcome = np.where(
            goals_home > goals_away, 0, np.where(goals_home == goals_away, 1, 2)
        )
        for label, selected in (
            ("dev", is_dev & has_odds),
            ("test", is_test & has_odds),
        ):
            if selected.any():
                probabilities = market.to_numpy()[selected]
                rows.append(
                    {
                        "model": "market_closing",
                        "slice": label,
                        "n": int(selected.sum()),
                        "logloss_1x2": log_loss_1x2(probabilities, outcome[selected]),
                        "acc_1x2": accuracy_1x2(probabilities, outcome[selected]),
                    }
                )
    return pd.DataFrame(rows)


def format_report(report: pd.DataFrame) -> str:
    """Render the report as an aligned text table."""
    shown = report.drop(columns=["rho"], errors="ignore")
    return shown.to_string(index=False, float_format="{:.3f}".format, na_rep="-")
