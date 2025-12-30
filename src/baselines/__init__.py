"""Utility baselines built on top of classical covariance estimators.

This package provides thin wrappers over PyPortfolioOpt and ARCH so that
traditional sliding-window estimators can be invoked directly from the main
codebase without editing the upstream projects.
"""

def _missing_pypfopt(*_args, **_kwargs):
    raise ImportError(
        "PyPortfolioOpt is required for this baseline. "
        "Install it via `pip install PyPortfolioOpt` or initialize the submodule "
        "under external/baselines/PyPortfolioOpt."
    )


try:
    from .pypfopt_baselines import (
        sample_covariance,
        exp_covariance,
        ledoit_wolf_covariance,
        oracle_shrinkage_covariance,
    )
except ImportError:
    sample_covariance = _missing_pypfopt
    exp_covariance = _missing_pypfopt
    ledoit_wolf_covariance = _missing_pypfopt
    oracle_shrinkage_covariance = _missing_pypfopt
def _missing_arch(*_args, **_kwargs):
    raise ImportError(
        "The `arch` package is required for this baseline. "
        "Install via `pip install arch` or initialize the git submodule "
        "under external/baselines/arch."
    )


try:
    from .arch_baselines import (
        riskmetrics_ewma_covariance,
        garch_vol_forecast,
        garch_covariance_forecast,
        ccc_garch_covariance_forecast,
        dcc_garch_covariance_forecast,
    )
except ImportError:
    riskmetrics_ewma_covariance = _missing_arch
    garch_vol_forecast = _missing_arch
    garch_covariance_forecast = _missing_arch
    ccc_garch_covariance_forecast = _missing_arch
    dcc_garch_covariance_forecast = _missing_arch
from .deep_baselines import CABModel, CNNBiLSTMCovarianceBaseline

__all__ = [
    "sample_covariance",
    "exp_covariance",
    "ledoit_wolf_covariance",
    "oracle_shrinkage_covariance",
    "riskmetrics_ewma_covariance",
    "garch_vol_forecast",
    "garch_covariance_forecast",
    "ccc_garch_covariance_forecast",
    "dcc_garch_covariance_forecast",
    "CABModel",
    "CNNBiLSTMCovarianceBaseline",
]
