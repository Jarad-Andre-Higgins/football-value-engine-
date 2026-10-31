"""The deployable model: boosters, calibration and metadata in one pickle.

The weekly job needs more than two boosters. It needs the calibration that
turns predicted xG into goal rates and the Dixon-Coles ``rho``, the exact
feature list, and enough metadata to know what it is running. All of that is
saved as one ``.pkl`` plus a readable ``.json`` summary beside it.

Security: a pickle can run code when loaded. Only load model files that you
created yourself, never one from an untrusted source.
"""

from __future__ import annotations

import json
import logging
import pickle
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.evaluate import MAX_RATE, MIN_RATE, Calibration
from src.features import feature_columns
from src.model import Boosters, ModelParams

logger = logging.getLogger("src.bundle")

SCHEMA_VERSION: Final[int] = 1
DEFAULT_MODEL_PATH: Final[Path] = MODELS_DIR / "xg_model.pkl"


@dataclass(frozen=True)
class ModelBundle:
    """Everything needed to turn features into goal rates.

    Attributes:
        boosters: The fitted home and away xG models.
        feature_names: The exact inputs, in training order.
        calibration: xG-to-goals scales and the Dixon-Coles ``rho``.
        params: The settings the models were trained with.
        trained_through: Date of the last match the models saw (plus one day).
        n_train_rows: Number of matches trained on.
        metrics: Out-of-sample test scores recorded at build time.
        created_at: UTC timestamp of the build.
        versions: Library versions used, to spot mismatches when loading.
        schema_version: Layout version of this class.
    """

    boosters: Boosters
    feature_names: tuple[str, ...]
    calibration: Calibration
    params: ModelParams
    trained_through: pd.Timestamp
    n_train_rows: int
    metrics: dict[str, float]
    created_at: str
    versions: dict[str, str]
    schema_version: int = SCHEMA_VERSION


def current_versions() -> dict[str, str]:
    """Versions of the libraries a saved model depends on."""
    return {
        "python": platform.python_version(),
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def build_bundle(
    boosters: Boosters,
    calibration: Calibration,
    params: ModelParams,
    *,
    trained_through: pd.Timestamp,
    n_train_rows: int,
    metrics: dict[str, float],
) -> ModelBundle:
    """Assemble a bundle, stamping it with the time and library versions."""
    return ModelBundle(
        boosters=boosters,
        feature_names=tuple(feature_columns()),
        calibration=calibration,
        params=params,
        trained_through=trained_through,
        n_train_rows=n_train_rows,
        metrics=metrics,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        versions=current_versions(),
    )


def summary(bundle: ModelBundle) -> dict[str, object]:
    """A JSON-friendly description of the bundle (no model internals)."""
    return {
        "schema_version": bundle.schema_version,
        "created_at": bundle.created_at,
        "trained_through": bundle.trained_through.strftime("%Y-%m-%d"),
        "n_train_rows": bundle.n_train_rows,
        "n_features": len(bundle.feature_names),
        "calibration": asdict(bundle.calibration),
        "params": asdict(bundle.params),
        "metrics": bundle.metrics,
        "versions": bundle.versions,
        "feature_names": list(bundle.feature_names),
    }


def save_bundle(bundle: ModelBundle, path: Path = DEFAULT_MODEL_PATH) -> Path:
    """Write the pickle and a ``.json`` summary beside it; return the pickle path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    path.with_suffix(".json").write_text(
        json.dumps(summary(bundle), indent=2) + "\n", encoding="utf-8"
    )
    return path


def load_bundle(path: Path = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load a bundle saved by :func:`save_bundle` (trusted files only).

    Raises:
        FileNotFoundError: If the file does not exist.
        TypeError: If the file does not hold a model bundle.
        ValueError: If the bundle has an unsupported layout version.
    """
    with path.open("rb") as handle:
        bundle = pickle.load(handle)  # trusted files only, see the module docstring
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path.name} is not a model bundle")
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{path.name} has schema {bundle.schema_version}, expected {SCHEMA_VERSION}; rebuild the model"
        )
    now = current_versions()
    for library in ("lightgbm", "numpy", "pandas"):
        if bundle.versions.get(library) != now[library]:
            logger.warning(
                "Model was built with %s %s but %s is installed; results may differ",
                library,
                bundle.versions.get(library),
                now[library],
            )
    return bundle


def predict_rates(bundle: ModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    """Calibrated goal rates for the home and away side of each row.

    Args:
        bundle: A loaded model bundle.
        features: A feature table containing every column in ``bundle.feature_names``.

    Returns:
        ``lam_home`` and ``lam_away``, ready for the Dixon-Coles score matrix
        together with ``bundle.calibration.rho``.

    Raises:
        ValueError: If feature columns are missing.
    """
    missing = [name for name in bundle.feature_names if name not in features.columns]
    if missing:
        raise ValueError(
            f"features are missing {len(missing)} columns, e.g. {missing[:3]}"
        )
    inputs = features[list(bundle.feature_names)].astype("float64")
    calibration = bundle.calibration
    return pd.DataFrame(
        {
            "lam_home": np.clip(
                calibration.scale_home
                * np.asarray(bundle.boosters.home.predict(inputs), dtype="float64"),
                MIN_RATE,
                MAX_RATE,
            ),
            "lam_away": np.clip(
                calibration.scale_away
                * np.asarray(bundle.boosters.away.predict(inputs), dtype="float64"),
                MIN_RATE,
                MAX_RATE,
            ),
        },
        index=features.index,
    )
