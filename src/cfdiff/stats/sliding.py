from __future__ import annotations

from typing import Tuple

import torch


def compute_sliding_covariances(
    sequence: torch.Tensor,
    window: int,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate per-step covariance matrices using a sliding window.

    Parameters
    ----------
    sequence:
        Tensor of shape (T, C) containing the time steps to analyse.
    window:
        Number of past points (including the current one) to use when estimating
        each covariance matrix.
    eps:
        Diagonal jitter to guarantee positive definiteness.

    Returns
    -------
    covariances:
        Tensor of shape (T, C, C) with the covariance estimate for each step.
    mask:
        Tensor of shape (T,) where 1 marks a well-estimated covariance
        (window >= 2) and 0 indicates the covariance should be ignored.
    """

    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence tensor of shape (T, C); received {tuple(sequence.shape)}")
    if window <= 0:
        raise ValueError("window must be positive")

    T, C = sequence.shape
    device = sequence.device
    dtype = sequence.dtype
    eye = torch.eye(C, device=device, dtype=dtype)

    covariances = []
    mask = torch.zeros(T, device=device, dtype=torch.bool)

    for t in range(T):
        start = max(0, t - window + 1)
        segment = sequence[start : t + 1]
        if segment.size(0) < 2:
            covariances.append(eye)
            continue

        segment_centered = segment - segment.mean(dim=0, keepdim=True)
        denom = max(segment_centered.size(0) - 1, 1)
        cov = torch.matmul(segment_centered.transpose(0, 1), segment_centered) / denom
        cov = (cov + cov.transpose(0, 1)) * 0.5  # ensure symmetry
        cov = cov + eps * eye

        covariances.append(cov)
        mask[t] = True

    return torch.stack(covariances), mask
