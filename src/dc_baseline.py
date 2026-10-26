"""Classic Dixon-Coles team-strength model: the statistical baseline.

Every team gets an attack and a defence rating. For home team `h` against
away team `a`::

    log(lam) = base + home_adv + attack[h] - defence[a]   # home expected goals
    log(mu)  = base            + attack[a] - defence[h]   # away expected goals

Higher attack means more goals scored; higher defence means fewer conceded.
The ratings, `home_adv` and the Dixon-Coles dependence `rho` are fitted
jointly by maximum likelihood on past results. Each match is weighted by
`0.5 ** (age / half_life)` so recent form counts more than old form. Attack
and defence ratings each average 0, so `base` is the log of the league's
average away scoring rate.

The fitted `(lam, mu, rho)` feed :func:`src.dixon_coles.score_matrix`. In the
final hybrid system LightGBM replaces the rating formula for `lam` and
`mu`; this model stays as the benchmark to beat.
"""

from __future__ import annotations

import difflib
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.dixon_coles import DEFAULT_MAX_GOALS, FloatArray, rho_bounds, score_matrix

logger = logging.getLogger(__name__)

SOURCE_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
MATCH_COLUMNS = ("date", "home_team", "away_team", "home_goals", "away_goals")
MIN_TRAINING_MATCHES = 50

_TAU_FLOOR = 1e-10  # keeps log(tau) finite if the optimiser probes bad rho values
_RHO_MARGIN = 1e-6


class UnknownTeamError(ValueError):
    """Raised when a team name was not seen in the training window."""


@dataclass(frozen=True, eq=False)
class DCParams:
    """Fitted Dixon-Coles parameters.

    Attributes:
        teams: Team names, aligned with `attack` and `defence`.
        attack: Attack ratings (mean 0); higher scores more.
        defence: Defence ratings (mean 0); higher concedes less.
        base: Log of the league-average away scoring rate.
        home_adv: Log of the home/away scoring ratio.
        rho: Dixon-Coles low-score dependence parameter.
    """

    teams: tuple[str, ...]
    attack: FloatArray
    defence: FloatArray
    base: float
    home_adv: float
    rho: float


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_football_data_csv(path: Path) -> pd.DataFrame:
    """Load one football-data.co.uk results file into the standard schema.

    Only played matches are kept (rows without a full-time score are dropped).

    Args:
        path: CSV file downloaded from football-data.co.uk.

    Returns:
        DataFrame with columns `date`, `home_team`, `away_team`,
        `home_goals` and `away_goals`, sorted by date.

    Raises:
        ValueError: If the file does not have the expected columns.
    """
    try:
        raw = pd.read_csv(
            path,
            usecols=list(SOURCE_COLUMNS),
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
    except ValueError as exc:
        raise ValueError(
            f"{path.name} is not a football-data.co.uk results file: {exc}"
        ) from exc
    raw = raw.dropna(subset=list(SOURCE_COLUMNS))
    matches = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
            "home_team": raw["HomeTeam"].astype(str).str.strip(),
            "away_team": raw["AwayTeam"].astype(str).str.strip(),
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
        }
    )
    return matches.sort_values("date", kind="stable").reset_index(drop=True)


def load_matches(paths: Iterable[Path]) -> pd.DataFrame:
    """Load and concatenate several results files (for example one per season).

    Raises:
        ValueError: If no paths are given or no played matches are found.
    """
    frames = []
    for path in paths:
      frame = load_football_data_csv(path)
      logger.info("Loaded %d matches from %s", len(frame), path.name)
      frames.append(frame)
    if not frames:
        raise ValueError("No results files were given")
    matches = pd.concat(frames, ignore_index=True)
    matches = matches.drop_duplicates(subset=["date", "home_team", "away_team"])
    if matches.empty:
        raise ValueError("The results files contain no played matches")
    return matches.sort_values("date", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #
def time_decay_weights(age_days: FloatArray, half_life_days: float) -> FloatArray:
    """Exponential decay weights: 1 today, 0.5 after one half-life, 0.25 after two.

    Raises:
        ValueError: If `half_life_days` is not positive.
    """
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days}")
    return np.asarray(np.exp2(-age_days / half_life_days), dtype=np.float64)


def outcome_probabilities(matrix: FloatArray) -> tuple[float, float, float]:
    """Return `(P(home win), P(draw), P(away win))` from a score matrix."""
    home_win = float(np.tril(matrix, k=-1).sum())  # rows (home goals) > columns
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    return home_win, draw, away_win


# --------------------------------------------------------------------------- #
# Likelihood
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _FitData:
    """Pre-computed arrays shared by every likelihood evaluation."""

    home: NDArray[np.intp]
    away: NDArray[np.intp]
    home_goals: FloatArray
    away_goals: FloatArray
    weights: FloatArray
    n_teams: int
    is_00: NDArray[np.bool_]
    is_01: NDArray[np.bool_]
    is_10: NDArray[np.bool_]
    is_11: NDArray[np.bool_]


def _unpack(
    theta: FloatArray, n_teams: int
) -> tuple[float, float, float, FloatArray, FloatArray]:
    """Split the optimiser vector into named parameters.

    Layout: `[base, home_adv, rho, attack[:-1], defence[:-1]]`. The last
    attack/defence rating is implied by the sum-to-zero constraint.
    """
    base, home_adv, rho = float(theta[0]), float(theta[1]), float(theta[2])
    free_attack = theta[3 : 3 + n_teams - 1]
    free_defence = theta[3 + n_teams - 1 :]
    attack = np.append(free_attack, -free_attack.sum())
    defence = np.append(free_defence, -free_defence.sum())
    return base, home_adv, rho, attack, defence


def _neg_log_likelihood(theta: FloatArray, data: _FitData) -> float:
    """Time-weighted negative Dixon-Coles log-likelihood per unit of weight."""
    base, home_adv, rho, attack, defence = _unpack(theta, data.n_teams)
    eta_home = base + home_adv + attack[data.home] - defence[data.away]
    eta_away = base + attack[data.away] - defence[data.home]
    lam = np.exp(eta_home)
    mu = np.exp(eta_away)

    log_tau = np.zeros_like(lam)
    tau_00 = 1.0 - lam[data.is_00] * mu[data.is_00] * rho
    tau_01 = 1.0 + lam[data.is_01] * rho
    tau_10 = 1.0 + mu[data.is_10] * rho
    log_tau[data.is_00] = np.log(np.maximum(tau_00, _TAU_FLOOR))
    log_tau[data.is_01] = np.log(np.maximum(tau_01, _TAU_FLOOR))
    log_tau[data.is_10] = np.log(np.maximum(tau_10, _TAU_FLOOR))
    log_tau[data.is_11] = math.log(max(1.0 - rho, _TAU_FLOOR))

    # Poisson terms; the constant log(k!) does not depend on theta, so it is dropped.
    log_lik = (
        log_tau + data.home_goals * eta_home - lam + data.away_goals * eta_away - mu
    )
    return float(-(data.weights * log_lik).sum() / data.weights.sum())


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class DixonColesModel:
    """Time-decayed Dixon-Coles model fitted by maximum likelihood.

    Attributes:half_life_days: Days after which a match counts half as much.
        max_goals: Goal cap for the score matrices it produces.
        params: Fitted parameters, or `None` before :meth:`fit`.
        as_of: Cut-off date of the last fit (only earlier matches were used).
        n_matches: Number of matches used in the last fit.
        converged: Whether the optimiser reported success.
    """

    def __init__(
        self, half_life_days: float = 365.0, max_goals: int = DEFAULT_MAX_GOALS
    ) -> None:
        if half_life_days <= 0:
            raise ValueError(f"half_life_days must be > 0, got {half_life_days}")
        self.half_life_days = half_life_days
        self.max_goals = max_goals
        self.params: DCParams | None = None
        self.as_of: pd.Timestamp | None = None
        self.n_matches = 0
        self.converged = False

    def fit(
        self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None
    ) -> DixonColesModel:
        """Fit on every match played strictly before `as_of`.

        Matches on or after `as_of` are ignored, so the model can never see
        the future. This is the no-leakage guarantee the backtests rely on.

        Args:
            matches: Standard match schema (see :func:`load_football_data_csv`).
            as_of: Cut-off date. Defaults to the day after the last match.

        Returns:
            `self`, to allow chaining.

        Raises:
            ValueError: On a bad schema or too few matches before `as_of`.
        """
        missing = [c for c in MATCH_COLUMNS if c not in matches.columns]
        if missing:
            raise ValueError(f"matches is missing columns: {missing}")
        if as_of is None:
            as_of = matches["date"].max() + pd.Timedelta(1, unit="D")
        past = matches.loc[matches["date"] < as_of]
        if len(past) < MIN_TRAINING_MATCHES:
            raise ValueError(
                f"Need at least {MIN_TRAINING_MATCHES} matches before "
                f"{as_of.date()}, found {len(past)}"
            )

        teams = tuple(sorted(set(past["home_team"]) | set(past["away_team"])))
        team_index = {team: i for i, team in enumerate(teams)}
        home_goals = past["home_goals"].to_numpy(dtype=np.float64)
        away_goals = past["away_goals"].to_numpy(dtype=np.float64)
        age_days = (as_of - past["date"]).dt.days.to_numpy(dtype=np.float64)
        n_teams = len(teams)
        data = _FitData(
            home=past["home_team"].map(team_index).to_numpy(dtype=np.intp),
            away=past["away_team"].map(team_index).to_numpy(dtype=np.intp),
            home_goals=home_goals,
            away_goals=away_goals,
            weights=time_decay_weights(age_days, self.half_life_days),
            n_teams=n_teams,
            is_00=(home_goals == 0) & (away_goals == 0),
            is_01=(home_goals == 0) & (away_goals == 1),
            is_10=(home_goals == 1) & (away_goals == 0),
            is_11=(home_goals == 1) & (away_goals == 1),
        )

        n_free = 2 * (n_teams - 1)
        theta0 = np.zeros(3 + n_free)
        theta0[0] = math.log(max(float(away_goals.mean()), 0.05))
        theta0[1] = 0.2
        bounds = [(-3.0, 3.0), (-1.5, 1.5), (-0.5, 0.5)] + [(-3.0, 3.0)] * n_free
        result = minimize(
            _neg_log_likelihood,
            theta0,
            args=(data,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000},
        )
        if not np.all(np.isfinite(result.x)):
            raise RuntimeError("Dixon-Coles fit produced non-finite parameters")

        base, home_adv, rho, attack, defence = _unpack(result.x, n_teams)
        self.params = DCParams(teams, attack, defence, base, home_adv, rho)
        self.as_of = as_of
        self.n_matches = len(past)
        self.converged = bool(result.success)
        if not self.converged:
            logger.warning("Optimiser did not fully converge: %s", result.message)
        logger.debug(
            "Fitted %d teams on %d matches before %s (rho=%.4f, home_adv=%.4f)",
            n_teams,len(past),
            as_of.date(),
            rho,
            home_adv,
        )
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals `(lam, mu)` for `home` hosting `away`.

        Raises:
            RuntimeError: If the model is not fitted.
            UnknownTeamError: If a team was not in the training window.
            ValueError: If both names are the same team.
        """
        params = self._fitted()
        h, a = self._index(home), self._index(away)
        if h == a:
            raise ValueError("A team cannot play itself")
        lam = math.exp(
            params.base + params.home_adv + params.attack[h] - params.defence[a]
        )
        mu = math.exp(params.base + params.attack[a] - params.defence[h])
        return lam, mu

    def predict_matrix(
        self, home: str, away: str, *, rho: float | None = None
    ) -> FloatArray:
        """Exact-score probability matrix for `home` hosting `away`.

        Args:
            home: Home team name (as spelled in the training data).
            away: Away team name.
            rho: Override the fitted `rho` (``0.0`` gives plain Poisson).

        Returns:
            Matrix from :func:`src.dixon_coles.score_matrix`.
        """
        params = self._fitted()
        lam, mu = self.expected_goals(home, away)
        wanted = params.rho if rho is None else rho
        lower, upper = rho_bounds(lam, mu)
        usable = min(max(wanted, lower + _RHO_MARGIN), upper - _RHO_MARGIN)
        if usable != wanted:
            logger.warning(
                "rho=%.4f outside the valid range for %s v %s; using %.4f",
                wanted,
                home,
                away,
                usable,
            )
        return score_matrix(lam, mu, usable, self.max_goals)

    def ratings(self) -> pd.DataFrame:
        """Team ratings table, best team first.

        Columns: `team`, `attack`, `defence` and `overall` (their sum).
        """
        params = self._fitted()
        table = pd.DataFrame(
            {
                "team": list(params.teams),
                "attack": params.attack,
                "defence": params.defence,
            }
        )
        table["overall"] = table["attack"] + table["defence"]
        return table.sort_values("overall", ascending=False).reset_index(drop=True)

    def _fitted(self) -> DCParams:
        if self.params is None:
            raise RuntimeError("Model is not fitted yet; call fit() first")
        return self.params

    def _index(self, team: str) -> int:
        teams = self._fitted().teams
        if team in teams:
            return teams.index(team)
        hints = difflib.get_close_matches(team, teams, n=3, cutoff=0.5)
        hint = f" Did you mean: {', '.join(hints)}?" if hints else ""
        raise UnknownTeamError(f"Unknown team '{team}'.{hint}")
