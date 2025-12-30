from typing import Callable

import torch
import torch.nn as nn

from fdiff.schedulers.sde import SDE
from fdiff.utils.dataclasses import DiffusableBatch


def get_sde_loss_fn(
    scheduler: SDE,
    train: bool,
    reduce_mean: bool = True,
    likelihood_weighting: bool = False,
    # Timestep sampling controls
    t_sampling_mode: str = "uniform",  # one of: "uniform", "beta", "power"
    t_beta_alpha: float = 2.0,
    t_beta_beta: float = 5.0,
    t_power_gamma: float = 2.0,
    t_importance_correction: bool = False,
) -> Callable[[nn.Module, DiffusableBatch], torch.Tensor]:
    """Construct unconditional score-matching loss for the fdiff batch type."""

    reduce_op = torch.mean if reduce_mean else (lambda x, dim=-1: 0.5 * torch.sum(x, dim=dim))  # type: ignore

    def _masked_reduce(per_elem: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return reduce_op(per_elem.reshape(per_elem.shape[0], -1), dim=-1)  # type: ignore[arg-type]
        mask_f = mask.to(device=per_elem.device, dtype=per_elem.dtype)
        masked = per_elem * mask_f
        num = masked.reshape(masked.shape[0], -1).sum(dim=-1)
        denom = mask_f.reshape(mask_f.shape[0], -1).sum(dim=-1).clamp_min(1.0)
        if reduce_mean:
            return num / denom
        return 0.5 * num / denom

    def loss_fn(model: nn.Module, batch: DiffusableBatch) -> torch.Tensor:
        if train:
            model.train()
        else:
            model.eval()

        target_clean = batch.X
        if target_clean is None:
            raise ValueError("DiffusableBatch.X is required for loss computation.")

        timesteps = batch.timesteps
        if timesteps is None:
            device = target_clean.device
            B = target_clean.shape[0]
            mode = (t_sampling_mode or "uniform").lower()
            if mode == "beta":
                beta_dist = torch.distributions.Beta(float(t_beta_alpha), float(t_beta_beta))
                u = beta_dist.sample((B,)).to(device)
            elif mode in {"power", "pow"}:
                u = torch.rand(B, device=device).pow(float(t_power_gamma))
            else:
                u = torch.rand(B, device=device)
            timesteps = u * (scheduler.T - scheduler.eps) + scheduler.eps

        z = torch.randn_like(target_clean)
        _, std = scheduler.marginal_prob(target_clean, timesteps)
        var = std**2

        std_matrix = torch.diag_embed(std)
        inv_std_matrix = torch.diag_embed(1.0 / std)

        noise = torch.matmul(std_matrix, z)
        target_noise = torch.matmul(inv_std_matrix, z)

        target_noisy = scheduler.add_noise(original_samples=target_clean, noise=noise, timesteps=timesteps)
        noisy_batch = DiffusableBatch(
            X=target_noisy,
            timesteps=timesteps,
            target_time=batch.target_time,
            X_mask=batch.X_mask,
        )

        score = model(noisy_batch)

        if not likelihood_weighting:
            weighting = 1.0 / torch.sum(1.0 / var, dim=1)
            losses = weighting.view(-1, 1, 1) * torch.square(score + target_noise)
            losses = _masked_reduce(losses, batch.X_mask)
        else:
            diff = score + target_noise
            scaled = torch.matmul(std_matrix, diff)
            losses = _masked_reduce(torch.square(scaled), batch.X_mask)

        # Optional importance correction for non-uniform t sampling
        if timesteps is not None and t_importance_correction:
            mode = (t_sampling_mode or "uniform").lower()
            if mode == "beta":
                alpha = torch.tensor(float(t_beta_alpha), device=losses.device, dtype=losses.dtype)
                beta = torch.tensor(float(t_beta_beta), device=losses.device, dtype=losses.dtype)
                beta_dist = torch.distributions.Beta(alpha, beta)
                u = (timesteps - scheduler.eps) / (scheduler.T - scheduler.eps)
                u = u.clamp(1e-8, 1 - 1e-8).to(losses.device, losses.dtype)
                pdf = torch.exp(beta_dist.log_prob(u))
                iw = 1.0 / (pdf + 1e-12)
            elif mode in {"power", "pow"}:
                gamma = float(t_power_gamma)
                u = (timesteps - scheduler.eps) / (scheduler.T - scheduler.eps)
                u = u.clamp(1e-12, 1.0).to(losses.device, losses.dtype)
                pdf = (1.0 / gamma) * torch.pow(u, (1.0 / gamma) - 1.0)
                iw = 1.0 / (pdf + 1e-12)
            else:
                iw = None
            if 'iw' in locals() and iw is not None:
                return torch.sum(losses * iw) / torch.sum(iw)
        return torch.mean(losses)

    return loss_fn
