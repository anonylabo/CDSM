from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().reshape(-1)
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _quantile_interp(sorted_values: np.ndarray, num_points: int) -> np.ndarray:
    ranks = (np.arange(sorted_values.size) + 0.5) / sorted_values.size
    target = (np.arange(num_points) + 0.5) / num_points
    return np.interp(target, ranks, sorted_values)


def wasserstein_1d_squared(orig: ArrayLike, other: ArrayLike) -> float:
    """Closed-form 2-Wasserstein distance squared for 1-D empirical measures."""

    x = np.sort(_to_numpy(orig).astype(np.float64, copy=False))
    y = np.sort(_to_numpy(other).astype(np.float64, copy=False))
    if x.size == 0 or y.size == 0:
        raise ValueError("Input measures must contain at least one sample.")
    num_points = max(x.size, y.size)
    x_q = _quantile_interp(x, num_points)
    y_q = _quantile_interp(y, num_points)
    diff = x_q - y_q
    return float(np.mean(diff * diff))


__all__ = ["wasserstein_1d_squared"]
