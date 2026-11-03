"""Does the model know something the closing market does not?

A model can be a little worse than the betting market on its own and still be
worth listening to, if it sometimes knows something the market has not priced.
The test is a blend: ``(1 - w) * market + w * model``. The weight ``w`` is chosen
to minimise 1X2 log-loss on the development seasons. The untouched test seasons
then show whether that blend beats the market alone, and a paired bootstrap
shows how much of any gain could be luck.

The weight is what :class:`src.value.ValueConfig` uses to shrink the model's
disagreement with a bookmaker's price before a bet is chosen. A weight near zero
means the model adds nothing to the market on this evidence; the engine then
only finds bets where a bookmaker's price is out of line with a sharper
bookmaker's.

Read the result with two limits in mind:

* It is measured on the 1X2 market against the *closing* line, the sharpest and
  hardest price to beat. Softer bookmakers, earlier prices and other markets
  (goals, both teams to score, handicaps) are not tested here, so the weight is
  a cautious proxy for how far to trust the model, not proof of an edge.
  Phase 6 measures real returns and closing-line value.
* The bootstrap treats matches as independent, which they only roughly are, so
  its interval is somewhat optimistic.

Usage::

    python -m src.blend
    python -m src.blend --test-from-season 2024 --no-save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from src.config import MODELS_DIR
from src.enrich import ENRICHED_OUTPUT
from src.evaluate import (
    OUTCOMES,
    Calibration,
    fit_calibration,
    market_probabilities,
    outcome_table,
)
from src.features import FEATURES_OUTPUT
from src.logging_config import setup_logging
from src.train import OOF_OUTPUT

logger = logging.getLogger("src.blend")

FloatArray = NDArray[np.float64]

SCHEMA_VERSION: Final[int] = 1
BLEND_PATH: Final[Path] = MODELS_DIR / "blend.json"
PROB_FLOOR: Final[float] = 1e-12
#: Bootstrap index draws held in memory at once (about 8 bytes each).
BOOTSTRAP_BUDGET: Final[int] = 4_000_000
#: A fitted weight below this is reported as "the model adds nothing".
NEGLIGIBLE_WEIGHT: Final[float] = 0.02


def match_losses(probabilities: FloatArray, outcome: NDArray[np.intp]) -> FloatArray:
    """Negative log-probability given to the result that happened, per match."""
    chosen = probabilities[np.arange(len(outcome)), outcome]
    return -np.log(np.maximum(chosen, PROB_FLOOR))


def fit_model_weight(
    p_model: FloatArray, p_market: FloatArray, outcome: NDArray[np.intp]
) -> float:
    """The weight on the model that minimises the 1X2 log-loss of the blend.

    The log-loss of a mixture is convex in the weight, so the bounded search
    finds the one global optimum.

    Args:
        p_model: Model probabilities, shape ``(matches, 3)``.
        p_market: Market probabilities, same shape.
        outcome: Result index per match (0 home, 1 draw, 2 away).

    Returns:
        A weight in ``[0, 1]``: 0 trusts only the market, 1 only the model.

    Raises:
        ValueError: If the inputs are empty or their shapes disagree.
    """
    if p_model.shape != p_market.shape or p_model.ndim != 2:
        raise ValueError("p_model and p_market must be matrices of the same shape")
    if len(outcome) != len(p_model) or len(outcome) == 0:
        raise ValueError("need one result per match, and at least one match")

    def loss(weight: float) -> float:
        mixed = (1.0 - weight) * p_market + weight * p_model
        return float(match_losses(mixed, outcome).mean())

    search = minimize_scalar(
        loss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6}
    )
    return float(search.x)


def paired_bootstrap_interval(
    differences: FloatArray,
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-match differences.

    Matches are resampled with replacement, which treats them as independent.
    Resampling is done in blocks, so memory stays flat however many matches
    there are.

    Raises:
        ValueError: For no differences, a ``level`` outside (0, 1) or fewer than
            one resample.
    """
    values = np.asarray(differences, dtype="float64")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("need a non-empty one-dimensional array of differences")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be between 0 and 1, got {level}")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    rng = np.random.default_rng(seed)
    n = len(values)
    block = max(1, min(n_resamples, BOOTSTRAP_BUDGET // n))
    means = np.empty(n_resamples, dtype="float64")
    for start in range(0, n_resamples, block):
        size = min(block, n_resamples - start)
        picks = rng.integers(0, n, size=(size, n))
        means[start : start + size] = values[picks].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


@dataclass(frozen=True)
class BlendResult:
    """Outcome of :func:`blend_report`.

    Attributes:
        weight: Model weight fitted on the development slice.
        calibration: xG-to-goals scales and ``rho`` fitted on the development slice.
        table: One row per slice with ``n``, the market, model and blend
            log-losses, and ``gain`` (market minus blend) with its 95% interval.
            The weight was fitted on the development slice, so only the test
            row is an honest test.
        test_from_season: First season of the test slice.
    """

    weight: float
    calibration: Calibration
    table: pd.DataFrame
    test_from_season: int

    def adds_information(self) -> bool:
        """True when the test-slice gain over the market is clearly above zero."""
        test = self.table[self.table["slice"] == "test"]
        return bool(not test.empty and test["gain_low"].iloc[0] > 0.0)


def blend_report(
    oof: pd.DataFrame,
    features: pd.DataFrame,
    enriched: pd.DataFrame,
    *,
    test_from_season: int,
    seed: int = 0,
) -> BlendResult:
    """Fit the blend on development seasons and test it on the later ones.

    Args:
        oof: Walk-forward predictions from :mod:`src.model`.
        features: The feature table (for the goals scored).
        enriched: The enriched match table (for bookmaker odds).
        test_from_season: First season of the test slice.
        seed: Seed of the bootstrap.

    Returns:
        The fitted weight, the calibration and the per-slice scores. Matches
        without odds or without a finite prediction are left out.

    Raises:
        ValueError: If the development slice has no matches with odds.
    """
    goals = features[["match_id", "target_home_goals", "target_away_goals"]]
    data = oof[["match_id", "season", "pred_home_xg", "pred_away_xg"]].merge(
        goals, on="match_id", how="left"
    )
    numeric = ["pred_home_xg", "pred_away_xg", "target_home_goals", "target_away_goals"]
    market = market_probabilities(enriched).reindex(data["match_id"])
    keep = market.notna().all(axis=1).to_numpy() & np.isfinite(
        data[numeric].to_numpy(dtype="float64")
    ).all(axis=1)
    data = data.loc[keep].reset_index(drop=True)
    p_market = market.loc[keep, list(OUTCOMES)].to_numpy(dtype="float64")
    is_test = (data["season"] >= test_from_season).to_numpy()
    is_dev = ~is_test
    if not is_dev.any():
        raise ValueError("No development matches with odds; lower test_from_season")

    pred_home = data["pred_home_xg"].to_numpy(dtype="float64")
    pred_away = data["pred_away_xg"].to_numpy(dtype="float64")
    goals_home = data["target_home_goals"].to_numpy(dtype="float64")
    goals_away = data["target_away_goals"].to_numpy(dtype="float64")
    calibration = fit_calibration(
        pred_home[is_dev], pred_away[is_dev], goals_home[is_dev], goals_away[is_dev]
    )
    scored = outcome_table(
        calibration.scale_home * pred_home,
        calibration.scale_away * pred_away,
        goals_home,
        goals_away,
        calibration.rho,
    )
    p_model = scored[list(OUTCOMES)].to_numpy(dtype="float64")
    outcome = scored["outcome"].to_numpy(dtype=np.intp)

    weight = fit_model_weight(p_model[is_dev], p_market[is_dev], outcome[is_dev])
    blended = (1.0 - weight) * p_market + weight * p_model
    rows: list[dict[str, object]] = []
    for label, selected in (("dev", is_dev), ("test", is_test)):
        if not selected.any():
            continue
        market_loss = match_losses(p_market[selected], outcome[selected])
        model_loss = match_losses(p_model[selected], outcome[selected])
        blend_loss = match_losses(blended[selected], outcome[selected])
        gain = market_loss - blend_loss
        low, high = paired_bootstrap_interval(gain, seed=seed)
        rows.append(
            {
                "slice": label,
                "n": int(selected.sum()),
                "market": float(market_loss.mean()),
                "model": float(model_loss.mean()),
                "blend": float(blend_loss.mean()),
                "gain": float(gain.mean()),
                "gain_low": low,
                "gain_high": high,
            }
        )
    return BlendResult(weight, calibration, pd.DataFrame(rows), test_from_season)


def save_blend(result: BlendResult, path: Path = BLEND_PATH) -> Path:
    """Write the fitted weight and the evidence behind it as JSON; return the path."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_weight": round(result.weight, 4),
        "fitted_against": "closing 1X2 market, development seasons only",
        "test_from_season": result.test_from_season,
        "adds_information": result.adds_information(),
        "calibration": asdict(result.calibration),
        "slices": json.loads(result.table.to_json(orient="records")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_model_weight(path: Path = BLEND_PATH) -> float:
    """Read the weight saved by :func:`save_blend`.

    Raises:
        FileNotFoundError: If the file does not exist (run ``python -m src.blend``).
        ValueError: If the file is not valid JSON, has an unsupported layout
            version, or holds a weight outside ``[0, 1]``.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python -m src.blend first")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name} has an unsupported layout; rerun python -m src.blend"
        )
    try:
        raw = payload["model_weight"]
    except KeyError as exc:
        raise ValueError(f"{path.name} has no model_weight") from exc
    if type(raw) not in (int, float):  # also rules out True and False
        raise ValueError(f"{path.name} has a non-numeric model_weight")
    weight = float(raw)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"model_weight must be between 0 and 1, got {weight}")
    return weight


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether the model adds information beyond the closing market."
    )
    parser.add_argument("--oof", type=Path, default=OOF_OUTPUT)
    parser.add_argument("--features", type=Path, default=FEATURES_OUTPUT)
    parser.add_argument("--enriched", type=Path, default=ENRICHED_OUTPUT)
    parser.add_argument("--test-from-season", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=BLEND_PATH)
    parser.add_argument(
        "--no-save", action="store_true", help="print the result without saving it"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the blend test, save the weight, and return the process exit code.

    Returns ``0`` on success and ``1`` when an input is missing or unusable.
    """
    args = _parse_args(argv)
    log = setup_logging("blend")
    needed = (
        (args.oof, "python -m src.build_model"),
        (args.features, "python -m src.features"),
        (args.enriched, "python -m src.enrich"),
    )
    for path, hint in needed:
        if not path.exists():
            log.error("Not found: %s; run %s first", path, hint)
            return 1
    try:
        result = blend_report(
            pd.read_parquet(args.oof),
            pd.read_parquet(args.features),
            pd.read_parquet(args.enriched),
            test_from_season=args.test_from_season,
        )
    except (ValueError, KeyError) as exc:
        log.error("Cannot run the blend test: %s", exc)
        return 1

    log.info(
        "Model weight fitted on the development seasons: %.3f "
        "(0 = trust the market only, 1 = trust the model only)",
        result.weight,
    )
    log.info(
        "1X2 log-loss, lower is better. gain = market minus blend, with a 95%% "
        "bootstrap interval; the weight was fitted on 'dev', so judge by 'test':\n%s",
        result.table.to_string(index=False, float_format="{:.4f}".format),
    )
    if result.table[result.table["slice"] == "test"].empty:
        log.warning("There is no test slice to judge by; lower --test-from-season")
    elif result.adds_information():
        log.info(
            "The blend beats the closing market on the test seasons: the model adds "
            "information the 1X2 market lacks"
        )
    else:
        log.info(
            "No clear gain over the closing market on the test seasons: no evidence "
            "yet that the model adds information to 1X2 prices"
        )
    if result.weight < NEGLIGIBLE_WEIGHT:
        log.warning(
            "The fitted weight is almost zero, so the engine will trust the market "
            "and only bet where bookmakers disagree with a sharper bookmaker"
        )
    if not args.no_save:
        log.info("Saved %s", save_blend(result, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
