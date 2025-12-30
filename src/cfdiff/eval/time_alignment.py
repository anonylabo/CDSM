from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class TimeAlignedMetrics:
    """Container for time-aligned evaluation statistics."""

    mae: np.ndarray  # (pred_len, assets)
    rmse: np.ndarray  # (pred_len, assets)
    corr_per_asset: np.ndarray  # (assets,)
    corr_time_asset: np.ndarray  # (pred_len, assets)


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Compute correlation while avoiding NaNs for constant inputs."""
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_time_aligned_metrics(preds: torch.Tensor, truth: torch.Tensor) -> TimeAlignedMetrics:
    """Compute element-wise metrics for time-aligned predictions.

    Args:
        preds: Tensor of shape (num_windows, pred_len, num_assets).
        truth: Tensor of identical shape.

    Returns:
        TimeAlignedMetrics dataclass.
    """
    if preds.shape != truth.shape:
        raise ValueError(f"preds and truth must share shape, got {preds.shape} vs {truth.shape}")

    # Ensure CPU tensors for numpy conversion
    preds_cpu = preds.detach().cpu()
    truth_cpu = truth.detach().cpu()

    diff = preds_cpu - truth_cpu
    mae = diff.abs().mean(dim=0).numpy()
    rmse = torch.sqrt((diff**2).mean(dim=0)).numpy()

    num_steps, num_assets = mae.shape
    corr_time_asset = np.zeros((num_steps, num_assets), dtype=np.float64)

    for t in range(num_steps):
        for a in range(num_assets):
            corr_time_asset[t, a] = _safe_corrcoef(
                preds_cpu[:, t, a].numpy(),
                truth_cpu[:, t, a].numpy(),
            )

    flat_preds = preds_cpu.reshape(-1, num_assets).numpy()
    flat_truth = truth_cpu.reshape(-1, num_assets).numpy()
    corr_per_asset = np.array(
        [_safe_corrcoef(flat_preds[:, a], flat_truth[:, a]) for a in range(num_assets)],
        dtype=np.float64,
    )

    return TimeAlignedMetrics(mae=mae, rmse=rmse, corr_per_asset=corr_per_asset, corr_time_asset=corr_time_asset)


def flatten_metrics_to_dataframe(metrics: TimeAlignedMetrics) -> Dict[str, np.ndarray]:
    """Convert metrics to plain numpy arrays for easy serialization."""

    return {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "corr_per_asset": metrics.corr_per_asset,
        "corr_time_asset": metrics.corr_time_asset,
    }
