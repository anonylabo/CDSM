from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.stats import wasserstein_distance
from typing import Dict


def _ensure_symmetric_unit(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(sym, 1.0)
    return np.clip(sym, -1.0, 1.0)


def _gini_coefficient(values: np.ndarray) -> float:
    if np.allclose(values, 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return float(gini)


def _cophenetic_corr(matrix: np.ndarray, method: str) -> float:
    dist = np.sqrt(np.clip(2.0 * (1.0 - matrix), 0.0, None))
    iu = np.triu_indices_from(matrix, k=1)
    compressed = dist[iu]
    if compressed.size < 2:
        return 0.0
    Z = linkage(compressed, method=method)
    corr, _ = cophenet(Z, compressed)
    return float(corr)


def _perron_frobenius_neg_sum(eigenvecs: np.ndarray) -> float:
    first_vec = eigenvecs[:, 0]
    neg_entries = first_vec[first_vec < 0]
    return float(np.abs(neg_entries).sum())


def _power_law_exponent(eigenvals: np.ndarray) -> float:
    vals = np.clip(eigenvals, 1e-8, None)
    ranks = np.arange(1, len(vals) + 1)
    log_r = np.log(ranks)
    log_v = np.log(vals)
    slope, _ = np.polyfit(log_r, log_v, 1)
    return float(-slope)


def _stylised_metrics(matrix: np.ndarray) -> Dict[str, float]:
    mean_corr = float(matrix[np.triu_indices_from(matrix, k=1)].mean())
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    order = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]
    metrics = {
        "mean_correl": mean_corr,
        "eigen_gini": _gini_coefficient(eigenvals),
        "coph_single": _cophenetic_corr(matrix, "single"),
        "coph_ward": _cophenetic_corr(matrix, "ward"),
        "perron_frob": _perron_frobenius_neg_sum(eigenvecs),
        "power_eigen": _power_law_exponent(eigenvals),
    }
    return metrics


def _batch_correlation(data: np.ndarray) -> np.ndarray:
    # data shape (B, T, C)
    B, T, C = data.shape
    corrs = np.zeros((B, C, C), dtype=np.float64)
    for i in range(B):
        sample = data[i]
        if T < 2:
            corrs[i] = np.eye(C, dtype=np.float64)
            continue
        sample = sample - sample.mean(axis=0, keepdims=True)
        cov = sample.T @ sample / max(T - 1, 1)
        std = np.sqrt(np.clip(np.diag(cov), 1e-8, None))
        outer = np.outer(std, std)
        corr = np.divide(cov, outer, out=np.eye(C, dtype=np.float64), where=outer > 0)
        corrs[i] = _ensure_symmetric_unit(corr)
    return corrs


def compute_corr_structure_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Compute correlation-structure metrics between predicted and true sequences.

    Parameters
    ----------
    pred : np.ndarray
        Array with shape (B, T, C) containing predicted samples.
    truth : np.ndarray
        Array with shape (B, T, C) containing the corresponding targets.
    """

    pred_corrs = _batch_correlation(pred)
    truth_corrs = _batch_correlation(truth)

    diag_diffs = []
    symmetry_errors = []
    wasserstein_dists = []
    stylised_diffs = {k: [] for k in [
        "mean_correl",
        "eigen_gini",
        "coph_single",
        "coph_ward",
        "perron_frob",
        "power_eigen",
    ]}

    for pred_corr, truth_corr in zip(pred_corrs, truth_corrs):
        diag_pred = np.diag(pred_corr)
        diag_truth = np.diag(truth_corr)
        diag_diffs.append(np.abs(diag_pred - diag_truth))

        sym_error = pred_corr - pred_corr.T
        idx = np.tril_indices_from(sym_error, k=-1)
        symmetry_errors.append(np.abs(sym_error[idx]))

        wasserstein_dists.append(
            wasserstein_distance(pred_corr.flatten(), truth_corr.flatten())
        )

        pred_metrics = _stylised_metrics(_ensure_symmetric_unit(pred_corr))
        truth_metrics = _stylised_metrics(_ensure_symmetric_unit(truth_corr))
        for key in stylised_diffs:
            stylised_diffs[key].append(abs(pred_metrics[key] - truth_metrics[key]))

    diag_diffs = np.concatenate(diag_diffs) if diag_diffs else np.array([0.0])
    symmetry_errors = np.concatenate(symmetry_errors) if symmetry_errors else np.array([0.0])
    wasserstein_dists = np.array(wasserstein_dists) if wasserstein_dists else np.array([0.0])

    aggregated: Dict[str, float] = {
        "corr_diag_abs_mean": float(diag_diffs.mean()),
        "corr_diag_abs_std": float(diag_diffs.std()),
        "corr_symmetry_abs_mean": float(symmetry_errors.mean()),
        "corr_symmetry_abs_std": float(symmetry_errors.std()),
        "corr_wasserstein_flat": float(wasserstein_dists.mean()),
    }

    for key, values in stylised_diffs.items():
        arr = np.array(values) if values else np.array([0.0])
        aggregated[f"corr_{key}_abs_diff"] = float(arr.mean())

    return aggregated
