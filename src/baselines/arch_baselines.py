"""ARCH-based rolling window covariance baselines."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:  # pragma: no cover - optional dependency
    from arch.univariate import arch_model
except ImportError as exc:  # pragma: no cover - explicit guidance to the user
    raise ImportError(
        "The `arch` package is required for arch_baselines. Install via `pip"
        " install arch` or initialise the git submodule under"
        " external/baselines/arch."
    ) from exc


ArrayLike = np.ndarray | pd.DataFrame


def _to_dataframe(data: ArrayLike, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input must be 2-D (time × assets).")
    if columns is None:
        columns = [f"asset_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=list(columns))


def riskmetrics_ewma_covariance(
    returns_window: ArrayLike,
    lam: float = 0.94,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """RiskMetrics-style EWMA covariance.

    Parameters
    ----------
    returns_window: array-like
        Standardised returns of shape (window, assets).
    lam: float
        Decay factor (0 < lam < 1). Default 0.94 matches RiskMetrics.
    """

    df = _to_dataframe(returns_window, columns)
    r = df.to_numpy()
    cov = np.cov(r, rowvar=False)
    for t in range(r.shape[0]):
        xt = r[-(t + 1)]  # iterate backwards so newest has largest weight
        cov = lam * cov + (1 - lam) * np.outer(xt, xt)
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def garch_vol_forecast(
    returns_window: ArrayLike,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    columns: Iterable[str] | None = None,
) -> pd.Series:
    """Fit univariate GARCH(p, q) for each asset and forecast variance."""

    df = _to_dataframe(returns_window, columns)
    forecasts = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < max(p, q) + 5:
            forecasts[col] = float(np.var(series))
            continue
        model = arch_model(series, p=p, q=q, dist=dist, mean="Zero")
        res = model.fit(disp="off")
        vol = res.forecast(horizon=horizon).variance.iloc[-1, -1]
        forecasts[col] = float(vol)
    return pd.Series(forecasts)


def garch_covariance_forecast(
    returns_window: ArrayLike,
    horizon: int = 1,
    correlation_method: str = "pearson",
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a simple GARCH-based covariance forecast.

    Variances are obtained per asset using :func:`garch_vol_forecast`, while
    correlations are estimated from the latest window snapshot.
    """

    df = _to_dataframe(returns_window, columns)
    vols = garch_vol_forecast(df, horizon=horizon, columns=df.columns)
    if correlation_method == "pearson":
        corr = df.corr().to_numpy()
    elif correlation_method == "spearman":
        corr = df.corr(method="spearman").to_numpy()
    else:
        raise ValueError(f"Unsupported correlation_method: {correlation_method}")

    sigma = np.diag(np.sqrt(vols.to_numpy() + 1e-12))
    cov = sigma @ corr @ sigma
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def ccc_garch_covariance_forecast(
    returns_window: ArrayLike,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    lam_corr: float | None = None,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Constant-correlation GARCH (CCC-GARCH) covariance forecast.

    Parameters
    ----------
    returns_window: array-like
        Standardised returns of shape (window, assets).
    horizon: int
        Number of steps to forecast. When ``horizon>1`` we follow the paper's
        Eq. (13) and average the per-step covariance forecasts.
    p, q: int
        ARCH/GARCH orders.
    dist: str
        Innovation distribution passed to :func:`arch.univariate.arch_model`.
    lam_corr: optional float
        Optional decay applied to the residual correlation estimate to improve
        numerical stability. When ``None`` (default) the plain sample
        correlation of standardised residuals is used.
    """

    df = _to_dataframe(returns_window, columns)
    T, N = df.shape
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    vols: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    eps = 1e-12

    for col in df.columns:
        series = df[col].dropna()
        if len(series) < max(p, q) + 5:
            var_fallback = float(np.var(series)) if len(series) else 1.0
            vols.append(np.repeat(np.sqrt(var_fallback + eps), horizon))
            resid = (series - series.mean()) / (series.std(ddof=1) + eps)
            residuals.append(resid.to_numpy())
            continue

        model = arch_model(series, p=p, q=q, dist=dist, mean="Zero")
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=horizon, reindex=False)
        var = forecast.variance.values[-1]
        vols.append(np.sqrt(np.clip(var, eps, None)))
        std_resid = res.resid / res.conditional_volatility
        residuals.append(std_resid.to_numpy())

    # Align residual lengths by truncating to the shortest series
    min_len = min(len(r) for r in residuals) if residuals else 0
    if min_len <= 1:
        corr = df.corr().to_numpy()
    else:
        resid_mat = np.stack([r[-min_len:] for r in residuals], axis=1)
        corr = np.corrcoef(resid_mat, rowvar=False)

    if lam_corr is not None:
        corr = lam_corr * corr + (1 - lam_corr) * np.eye(corr.shape[0])
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)

    avg_cov = np.zeros((N, N), dtype=float)
    for step in range(horizon):
        step_vols = np.array([v[step] if step < len(v) else v[-1] for v in vols])
        D = np.diag(step_vols + eps)
        avg_cov += D @ corr @ D
    avg_cov /= float(horizon)

    return pd.DataFrame(avg_cov, index=df.columns, columns=df.columns)


def dcc_garch_covariance_forecast(
    returns_window: ArrayLike,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    lookback_window: int | None = None,
    a: float = 0.05,
    b: float = 0.9,
    estimate_ab: bool = False,
    return_corr_only: bool = False,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Dynamic conditional correlation (DCC-GARCH) covariance forecast.

    This follows the standard two-step DCC(1,1): univariate GARCH for
    volatilities, then a dynamic correlation recursion. For multi-step horizons
    we propagate the last conditional correlation toward its long-run mean
    (Q_bar) using the DCC decay (a+b). The per-step covariances are averaged.
    """

    df = _to_dataframe(returns_window, columns)
    if lookback_window is not None and lookback_window > 0:
        df = df.tail(int(lookback_window))
    T, N = df.shape
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if T < max(p, q) + 5:
        raise ValueError(f"Not enough observations ({T}) to fit DCC-GARCH.")

    eps = 1e-12
    vols: list[np.ndarray] = []
    std_resids: list[np.ndarray] = []

    # 1) Fit univariate GARCH and collect standardised residuals + variance forecasts
    for col in df.columns:
        series = df[col].dropna()
        model = arch_model(series, p=p, q=q, dist=dist, mean="Zero")
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=horizon, reindex=False)
        var = forecast.variance.values[-1]
        vols.append(np.sqrt(np.clip(var, eps, None)))
        std_resid = res.resid / res.conditional_volatility
        std_resids.append(std_resid.to_numpy())

    min_len = min(len(r) for r in std_resids)
    resid_mat = np.stack([r[-min_len:] for r in std_resids], axis=1)
    Q_bar = np.cov(resid_mat, rowvar=False)

    def _dcc_loglike(params: np.ndarray, resid: np.ndarray, Q0: np.ndarray) -> float:
        a_, b_ = params
        if a_ < 0 or b_ < 0 or a_ + b_ >= 0.999:
            return 1e10
        Qt = Q0.copy()
        ll = 0.0
        for t in range(1, resid.shape[0]):
            e_tm1 = resid[t - 1 : t].T
            Qt = (1 - a_ - b_) * Q_bar + a_ * (e_tm1 @ e_tm1.T) + b_ * Qt
            d_inv = np.diag(1.0 / np.sqrt(np.diag(Qt) + eps))
            Rt = d_inv @ Qt @ d_inv
            try:
                R_inv = np.linalg.inv(Rt)
                logdet = np.log(np.linalg.det(Rt) + eps)
                e_t = resid[t : t + 1].T
                ll += 0.5 * (logdet + float(e_t.T @ R_inv @ e_t))
            except np.linalg.LinAlgError:
                return 1e10
        return ll

    if estimate_ab:
        init = np.array([max(a, eps), max(b, eps)])
        bounds = [(1e-5, 0.999), (1e-5, 0.999)]
        res_opt = minimize(_dcc_loglike, init, args=(resid_mat, Q_bar), method="SLSQP", bounds=bounds)
        if res_opt.success:
            a, b = res_opt.x

    # Reconstruct last Qt with observed residuals
    Qt = Q_bar.copy()
    for t in range(1, resid_mat.shape[0]):
        e_tm1 = resid_mat[t - 1 : t].T
        Qt = (1 - a - b) * Q_bar + a * (e_tm1 @ e_tm1.T) + b * Qt
    last_resid = resid_mat[-1]

    avg_cov = np.zeros((N, N), dtype=float)
    avg_corr = np.zeros((N, N), dtype=float)
    Qt_fore = Qt.copy()
    shock = np.outer(last_resid, last_resid)
    for step in range(horizon):
        # First step uses last residual shock; subsequent steps revert toward Q_bar
        if step == 0:
            Qt_fore = (1 - a - b) * Q_bar + a * shock + b * Qt_fore
        else:
            decay = a + b
            Qt_fore = (1 - decay) * Q_bar + decay * Qt_fore

        d_inv = np.diag(1.0 / np.sqrt(np.diag(Qt_fore) + eps))
        Rt = d_inv @ Qt_fore @ d_inv
        if return_corr_only:
            avg_corr += Rt
        else:
            vol_step = np.array([v[step] if step < len(v) else v[-1] for v in vols])
            D = np.diag(vol_step + eps)
            avg_cov += D @ Rt @ D

    if return_corr_only:
        avg_corr /= float(horizon)
        return pd.DataFrame(avg_corr, index=df.columns, columns=df.columns)
    avg_cov /= float(horizon)
    return pd.DataFrame(avg_cov, index=df.columns, columns=df.columns)
