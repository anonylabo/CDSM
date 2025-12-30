from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import torch


@dataclass(frozen=True)
class DistributionMetrics:
    """Summary statistics comparing generated samples against reference targets."""

    cov_fro_error: float
    cov_mean_abs: float
    var_mean_abs: float
    corr_fro_error: float
    corr_mean_abs: float
    kl_gaussian: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _flatten(samples: torch.Tensor) -> torch.Tensor:
    if samples.ndim != 3:
        raise ValueError(f"Expected samples shaped (batch, length, channels); received {samples.shape}")
    return samples.reshape(-1, samples.shape[-1]).to(dtype=torch.float64)


def _covariance(flat: torch.Tensor, eps: float) -> torch.Tensor:
    centered = flat - flat.mean(dim=0, keepdim=True)
    denom = max(flat.shape[0] - 1, 1)
    cov = centered.transpose(0, 1) @ centered / float(denom)
    cov = 0.5 * (cov + cov.transpose(0, 1))  # ensure symmetry
    return cov + eps * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)


def _correlation(cov: torch.Tensor, eps: float) -> torch.Tensor:
    std = torch.sqrt(torch.diagonal(cov).clamp_min(eps))
    outer = std.unsqueeze(-1) * std.unsqueeze(-2)
    corr = cov / outer
    corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr


def _gaussian_kl(mean_p: torch.Tensor, cov_p: torch.Tensor, mean_q: torch.Tensor, cov_q: torch.Tensor) -> torch.Tensor:
    """KL divergence KL(P || Q) between multivariate Gaussians."""

    chol_q = torch.linalg.cholesky(cov_q)
    chol_p = torch.linalg.cholesky(cov_p)

    inv_cov_q_cov_p = torch.cholesky_solve(cov_p, chol_q)
    trace_term = torch.diagonal(inv_cov_q_cov_p).sum()

    mean_diff = (mean_q - mean_p).unsqueeze(-1)
    quad_term = torch.matmul(mean_diff.transpose(0, 1), torch.cholesky_solve(mean_diff, chol_q)).squeeze()

    log_det_q = 2.0 * torch.log(torch.diagonal(chol_q)).sum()
    log_det_p = 2.0 * torch.log(torch.diagonal(chol_p)).sum()

    dim = mean_p.numel()
    kl = 0.5 * (trace_term + quad_term - dim + log_det_q - log_det_p)
    return kl


def compute_distribution_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-6,
) -> DistributionMetrics:
    """Compute covariance, correlation, and Gaussian KL metrics between two sample sets."""

    if generated.shape != reference.shape:
        raise ValueError(f"generated and reference must have identical shape, got {generated.shape} vs {reference.shape}")

    gen_flat = _flatten(generated)
    ref_flat = _flatten(reference)

    mean_gen = gen_flat.mean(dim=0)
    mean_ref = ref_flat.mean(dim=0)

    cov_gen = _covariance(gen_flat, eps)
    cov_ref = _covariance(ref_flat, eps)

    cov_diff = cov_gen - cov_ref
    cov_fro_error = torch.linalg.matrix_norm(cov_diff, ord="fro")
    cov_mean_abs = cov_diff.abs().mean()
    var_mean_abs = (torch.diagonal(cov_gen) - torch.diagonal(cov_ref)).abs().mean()

    corr_gen = _correlation(cov_gen, eps)
    corr_ref = _correlation(cov_ref, eps)
    corr_diff = corr_gen - corr_ref
    corr_fro_error = torch.linalg.matrix_norm(corr_diff, ord="fro")
    corr_mean_abs = corr_diff.abs().mean()

    kl_div = _gaussian_kl(mean_gen, cov_gen, mean_ref, cov_ref)

    return DistributionMetrics(
        cov_fro_error=float(cov_fro_error.item()),
        cov_mean_abs=float(cov_mean_abs.item()),
        var_mean_abs=float(var_mean_abs.item()),
        corr_fro_error=float(corr_fro_error.item()),
        corr_mean_abs=float(corr_mean_abs.item()),
        kl_gaussian=float(kl_div.item()),
    )
