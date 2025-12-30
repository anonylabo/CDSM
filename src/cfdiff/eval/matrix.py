from __future__ import annotations

from typing import Dict

import numpy as np
import torch

_EPS = 1e-12


def _to_np(a: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _ensure_square(C: np.ndarray, name: str) -> np.ndarray:
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"{name} must be square; got shape={C.shape}")
    return C.astype(np.float64, copy=False)


def _covariance(x: torch.Tensor, ddof: int = 1) -> np.ndarray:
    # x: (N, channels)
    x_np = _to_np(x)
    return np.cov(x_np, rowvar=False, ddof=ddof)


def _cov_to_corr(C: np.ndarray) -> np.ndarray:
    C = _ensure_square(C, "covariance")
    d = np.sqrt(np.clip(np.diag(C), _EPS, None))
    R = (C / d).T / d
    R = np.where(np.isfinite(R), R, 0.0)
    R = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R, 1.0)
    return R


def _flat_offdiag(A: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx]


def compute_covariance_metrics(
    truth: torch.Tensor,
    preds: torch.Tensor,
) -> Dict[str, float]:
    """
    Compare covariance/ correlation structure between predicted and true windows.

    Args:
        truth: Tensor (batch, pred_len, channels)
        preds: Tensor (batch, pred_len, channels)
    """

    if truth.shape != preds.shape:
        raise ValueError(f"truth and preds must share shape; got {truth.shape} vs {preds.shape}")

    b, t, c = truth.shape
    if b == 0:
        return {
            "cov_fro": 0.0,
            "cov_fro_rel": 0.0,
            "cov_mse": 0.0,
            "cov_mae": 0.0,
            "cov_diag_mape": 0.0,
            "corr_fro": 0.0,
            "corr_fro_rel": 0.0,
            "corr_offdiag_pearson": 0.0,
            "corr_offdiag_spearman": 0.0,
            "corr_cross_mse": 0.0,
            "corr_sign_rate": 0.0,
        }

    accum = {
        "cov_fro": 0.0,
        "cov_fro_rel": 0.0,
        "cov_mse": 0.0,
        "cov_mae": 0.0,
        "cov_diag_mape": 0.0,
        "corr_fro": 0.0,
        "corr_fro_rel": 0.0,
        "corr_offdiag_pearson": 0.0,
        "corr_offdiag_spearman": 0.0,
        "corr_cross_mse": 0.0,
        "corr_sign_rate": 0.0,
    }

    for i in range(b):
        truth_flat = truth[i].reshape(t, c)
        preds_flat = preds[i].reshape(t, c)

        cov_true = _covariance(truth_flat)
        cov_pred = _covariance(preds_flat)

        Ct = _ensure_square(cov_true, "cov_true")
        Cp = _ensure_square(cov_pred, "cov_pred")

        diff = Cp - Ct
        frob = float(np.linalg.norm(diff, ord="fro"))
        frob_rel = float(frob / (np.linalg.norm(Ct, ord="fro") + _EPS))
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        diag_mape = float(np.mean(np.abs((np.diag(Cp) - np.diag(Ct)) / (np.abs(np.diag(Ct)) + _EPS))))

        Rt = _cov_to_corr(Ct)
        Rp = _cov_to_corr(Cp)
        corr_diff = Rp - Rt
        corr_fro = float(np.linalg.norm(corr_diff, ord="fro"))
        corr_fro_rel = float(corr_fro / (np.linalg.norm(Rt, ord="fro") + _EPS))

        x = _flat_offdiag(Rp)
        y = _flat_offdiag(Rt)
        if x.size and y.size:
            pearson = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0
            # spearman via rank transformation
            rx = _rank_avg_ties(x)
            ry = _rank_avg_ties(y)
            spearman = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else 0.0
            cross_mse = float(np.mean((x - y) ** 2))
            sign_rate = float(np.mean(np.sign(x) == np.sign(y)))
        else:
            pearson = spearman = cross_mse = sign_rate = 0.0

        accum["cov_fro"] += frob
        accum["cov_fro_rel"] += frob_rel
        accum["cov_mse"] += mse
        accum["cov_mae"] += mae
        accum["cov_diag_mape"] += diag_mape
        accum["corr_fro"] += corr_fro
        accum["corr_fro_rel"] += corr_fro_rel
        accum["corr_offdiag_pearson"] += pearson
        accum["corr_offdiag_spearman"] += spearman
        accum["corr_cross_mse"] += cross_mse
        accum["corr_sign_rate"] += sign_rate

    return {k: v / b for k, v in accum.items()}


def _rank_avg_ties(z: np.ndarray) -> np.ndarray:
    order = z.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(z), dtype=float)
    _, inv, cnt = np.unique(z, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    return (sums / cnt)[inv]
