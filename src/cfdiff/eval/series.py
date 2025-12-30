from __future__ import annotations

from typing import Dict

import torch


def _flatten(truth: torch.Tensor, preds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if truth.shape != preds.shape:
        raise ValueError(f"truth and preds must share shape; got {truth.shape} vs {preds.shape}")
    # combine batch and horizon dimensions
    b, t, c = truth.shape
    truth_flat = truth.reshape(b * t, c)
    preds_flat = preds.reshape(b * t, c)
    return truth_flat, preds_flat


def compute_series_metrics(truth: torch.Tensor, preds: torch.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    """
    Aggregate single-series metrics comparing diffusion samples against ground truth.

    Args:
        truth: Tensor shaped (batch, pred_len, num_assets)
        preds: Tensor shaped (batch, pred_len, num_assets)

    Returns:
        Dict containing mae, mse, rmse, mape, smape, cover80, cover95.
    """

    truth_f, preds_f = _flatten(truth, preds)

    diff = preds_f - truth_f
    mse = torch.mean(diff**2)
    mae = torch.mean(diff.abs())
    rmse = torch.sqrt(mse)

    denom_mape = truth_f.abs().clamp_min(eps)
    mape = torch.mean((diff.abs() / denom_mape)) * 100.0

    denom_smape = (truth_f.abs() + preds_f.abs()).clamp_min(eps)
    smape = torch.mean((diff.abs() * 2.0 / denom_smape)) * 100.0

    # Coverage metrics: compute quantiles across batch dimension (per horizon, per asset)
    if preds.shape[0] >= 2:
        q_lo_80 = torch.quantile(preds, 0.10, dim=0)
        q_hi_80 = torch.quantile(preds, 0.90, dim=0)
        cover80 = torch.mean(((truth >= q_lo_80) & (truth <= q_hi_80)).float())

        q_lo_95 = torch.quantile(preds, 0.025, dim=0)
        q_hi_95 = torch.quantile(preds, 0.975, dim=0)
        cover95 = torch.mean(((truth >= q_lo_95) & (truth <= q_hi_95)).float())
    else:
        cover80 = torch.tensor(float("nan"), device=truth.device)
        cover95 = torch.tensor(float("nan"), device=truth.device)

    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "mape": float(mape.item()),
        "smape": float(smape.item()),
        "cover80": float(cover80.item()),
        "cover95": float(cover95.item()),
    }
