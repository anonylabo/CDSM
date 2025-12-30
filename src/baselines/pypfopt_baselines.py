"""Wrappers around PyPortfolioOpt risk models for rolling-window baselines."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency during unit tests
    from pypfopt import risk_models
except ImportError as exc:  # pragma: no cover - makes failure explicit to users
    raise ImportError(
        "PyPortfolioOpt is required for pypfopt_baselines. "
        "Install it via `pip install PyPortfolioOpt` or initialise the"
        " submodule under external/baselines/PyPortfolioOpt."
    ) from exc


ArrayLike = Sequence[Sequence[float]] | np.ndarray | pd.DataFrame


def _to_dataframe(data: ArrayLike, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input must be 2-D (time × assets).")
    if columns is None:
        columns = [f"asset_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=list(columns))


def sample_covariance(
    returns_window: ArrayLike,
    columns: Iterable[str] | None = None,
    frequency: int = 1,
) -> pd.DataFrame:
    """Plain sample covariance computed via PyPortfolioOpt.

    Parameters
    ----------
    returns_window: array-like
        Standardised returns of shape (window, assets).
    columns: optional iterable of str
        Asset names for readability.
    """

    df = _to_dataframe(returns_window, columns)
    # PyPortfolioOpt defaults to frequency=252; set to 1 to avoid annualising covariances.
    return risk_models.sample_cov(df, returns_data=True, frequency=frequency)


def exp_covariance(
    returns_window: ArrayLike,
    span: int = 60,
    adjust: bool = True,
    frequency: int = 1,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Exponentially weighted covariance (RiskMetrics style)."""

    df = _to_dataframe(returns_window, columns)
    return risk_models.exp_cov(df, span=span, returns_data=True, adjust=adjust, frequency=frequency)


def ledoit_wolf_covariance(
    returns_window: ArrayLike,
    columns: Iterable[str] | None = None,
    frequency: int = 1,
    shrinkage_target: str = "constant_variance",
) -> pd.DataFrame:
    """Ledoit--Wolf shrinkage estimator implemented by PyPortfolioOpt."""

    df = _to_dataframe(returns_window, columns)
    shrinker = risk_models.CovarianceShrinkage(df, returns_data=True, frequency=frequency)
    if shrinkage_target == "constant_variance":
        return shrinker.ledoit_wolf()
    if shrinkage_target == "single_index":
        return shrinker.single_index()
    if shrinkage_target == "oracle_approx" or shrinkage_target == "oas":
        return shrinker.oracle_approximating()
    raise ValueError(f"Unsupported shrinkage_target: {shrinkage_target}")


def oracle_shrinkage_covariance(
    returns_window: ArrayLike,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convenience alias for PyPortfolioOpt's oracle-approximating shrinkage."""

    df = _to_dataframe(returns_window, columns)
    shrinker = risk_models.CovarianceShrinkage(df, returns_data=True)
    return shrinker.oracle_approximating()
