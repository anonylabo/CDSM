from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from cfdiff.utils.dataclasses import DiffusionBatch
from cfdiff.utils.sde import SDE


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
    temporal_loss_weighting: str | None = None,
    temporal_loss_max: float = 1.0,
) -> Callable[[nn.Module, DiffusionBatch], torch.Tensor]:
    """Create the standard unconditional score-matching loss for arbitrary SDEs."""

    reduce_op = torch.mean if reduce_mean else (lambda x, dim=-1: 0.5 * torch.sum(x, dim=dim))  # type: ignore

    def _masked_reduce(per_elem: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Reduce (B, T, C) losses to (B,) optionally masking missing entries."""

        if mask is None:
            return reduce_op(per_elem.reshape(per_elem.shape[0], -1), dim=-1)  # type: ignore[arg-type]

        mask_f = mask.to(device=per_elem.device, dtype=per_elem.dtype)
        masked = per_elem * mask_f
        num = masked.reshape(masked.shape[0], -1).sum(dim=-1)
        denom = mask_f.reshape(mask_f.shape[0], -1).sum(dim=-1).clamp_min(1.0)
        if reduce_mean:
            return num / denom
        return 0.5 * num / denom

    def loss_fn(model: nn.Module, batch: DiffusionBatch) -> torch.Tensor:
        if train:
            model.train()
        else:
            model.eval()

        if batch.context is None or batch.target is None:
            raise ValueError("DiffusionBatch must contain context and target for loss computation.")

        context = batch.context
        target_clean = batch.target
        timesteps = batch.timesteps

        if timesteps is None:
            batch_size = target_clean.shape[0]
            device = target_clean.device
            # sample u in (0, 1), then scale to [eps, T]
            mode = (t_sampling_mode or "uniform").lower()
            if mode == "beta":
                beta_dist = torch.distributions.Beta(float(t_beta_alpha), float(t_beta_beta))
                u = beta_dist.sample((batch_size,)).to(device)
            elif mode in {"power", "pow"}:
                u = torch.rand(batch_size, device=device).pow(float(t_power_gamma))
            else:  # "uniform"
                u = torch.rand(batch_size, device=device)
            timesteps = u * (scheduler.T - scheduler.eps) + scheduler.eps

        z = torch.randn_like(target_clean)
        _, std = scheduler.marginal_prob(target_clean, timesteps)
        var = std**2

        std_matrix = torch.diag_embed(std)
        inv_std_matrix = torch.diag_embed(1.0 / std)

        noise = torch.matmul(std_matrix, z)
        target_noise = torch.matmul(inv_std_matrix, z)

        target_noisy = scheduler.add_noise(original_samples=target_clean, noise=noise, timesteps=timesteps)
        extras = {
            "context": context,
            "timesteps": timesteps,
        }
        if batch.context_time is not None:
            extras["context_time"] = batch.context_time
        if batch.target_time is not None:
            extras["target_time"] = batch.target_time
        if batch.target_clean is not None:
            extras["target_clean"] = batch.target_clean
        else:
            extras["target_clean"] = target_clean
        if batch.context_mask is not None:
            extras["context_mask"] = batch.context_mask
        if batch.target_mask is not None:
            extras["target_mask"] = batch.target_mask
        if batch.context_mask_time is not None:
            extras["context_mask_time"] = batch.context_mask_time
        if batch.target_mask_time is not None:
            extras["target_mask_time"] = batch.target_mask_time
        if batch.cov is not None:
            extras["cov"] = batch.cov
        if batch.cov_mask is not None:
            extras["cov_mask"] = batch.cov_mask
        noisy_batch = DiffusionBatch(target=target_noisy, **extras)

        score = model(noisy_batch)

        # Optional per-time-step weights (e.g., mid-horizon emphasis). Normalised to mean=1.
        time_weights = None
        if temporal_loss_weighting:
            mode = str(temporal_loss_weighting).lower()
            max_scale = max(1.0, float(temporal_loss_max))
            T = target_clean.shape[1]
            idx = torch.arange(T, device=target_clean.device, dtype=target_clean.dtype)
            if T > 1:
                center = (T - 1) / 2.0
                if mode in {"mid", "mid_peak"}:
                    ramp = 1.0 + (max_scale - 1.0) * (1.0 - 2.0 * torch.abs(idx - center) / (T - 1))
                    ramp = torch.clamp(ramp, min=1.0)
                elif mode in {"back", "tail"}:
                    ramp = torch.linspace(1.0, max_scale, steps=T, device=target_clean.device, dtype=target_clean.dtype)
                else:
                    raise ValueError(f"Unsupported temporal_loss_weighting='{temporal_loss_weighting}'")
            else:
                ramp = torch.ones_like(idx)
            ramp = ramp / ramp.mean()
            time_weights = ramp.view(1, -1, 1)  # (1, T, 1) for broadcasting

        if not likelihood_weighting:
            weighting = 1.0 / torch.sum(1.0 / var, dim=1)
            losses = weighting.view(-1, 1, 1) * torch.square(score + target_noise)
            if time_weights is not None:
                losses = losses * time_weights
            losses = _masked_reduce(losses, batch.target_mask)
        else:
            diff = score + target_noise
            scaled = torch.matmul(std_matrix, diff)
            losses = torch.square(scaled)
            if time_weights is not None:
                losses = losses * time_weights
            losses = _masked_reduce(losses, batch.target_mask)

        # Optional importance correction when using non-uniform t sampling
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
                # normalize to keep loss scale stable
                base_loss = torch.sum(losses * iw) / torch.sum(iw)
            else:
                base_loss = torch.mean(losses)
        else:
            base_loss = torch.mean(losses)

        total_loss = base_loss
        model._last_mean_loss = None
        model._last_cov_loss = None
        if hasattr(model, "_last_corr_loss"):
            model._last_corr_loss = None
        if hasattr(model, "_last_spec_loss"):
            model._last_spec_loss = None
        if hasattr(model, "_last_sliding_cov_loss"):
            model._last_sliding_cov_loss = None
        if hasattr(model, "lambda_mean") and getattr(model, "lambda_mean", 0.0) > 0 and batch.target_time is not None:
            if getattr(model, "_last_pred_mean", None) is not None:
                pred_mean = model._last_pred_mean
                target_tail = batch.target_time.to(pred_mean.device, pred_mean.dtype)
                tail_mask = batch.target_mask_time
                if tail_mask is None:
                    tail_mask = torch.isfinite(target_tail)
                tail_mask = tail_mask.to(device=pred_mean.device)
                # masked MSE normalised by number of observed entries
                diff = (pred_mean - target_tail) ** 2
                mask_f = tail_mask.to(diff.dtype)
                num = (diff * mask_f).sum()
                denom = mask_f.sum().clamp_min(1.0)
                mean_loss = num / denom
                total_loss = total_loss + model.lambda_mean * mean_loss
                model._last_mean_loss = mean_loss.detach()
        if hasattr(model, "lambda_cov") and getattr(model, "lambda_cov", 0.0) > 0 and batch.target_time is not None:
            if getattr(model, "_last_pred_chol", None) is not None:
                pred_cov = model.chol_to_cov(model._last_pred_chol)
                target_tail = batch.target_time.to(pred_cov.device, pred_cov.dtype)
                tail_mask = batch.target_mask_time
                if tail_mask is None:
                    tail_mask = torch.isfinite(target_tail)
                # Drop any rows with missing entries to avoid contaminating the covariance estimate.
                cov_targets = []
                for b in range(target_tail.size(0)):
                    row_mask = tail_mask[b].all(dim=-1)
                    rows = target_tail[b][row_mask]
                    if rows.size(0) <= 1:
                        cov_targets.append(torch.zeros_like(pred_cov[b]))
                        continue
                    centered = rows - rows.mean(dim=0, keepdim=True)
                    denom = float(max(centered.size(0) - 1, 1))
                    cov = centered.t().matmul(centered) / denom
                    cov_targets.append(cov)
                target_cov = torch.stack(cov_targets, dim=0)
                cov_loss = torch.square(pred_cov - target_cov).mean()
                total_loss = total_loss + model.lambda_cov * cov_loss
                model._last_cov_loss = cov_loss.detach()

        if hasattr(model, "lambda_corr") and getattr(model, "lambda_corr", 0.0) > 0 and batch.target_time is not None:
            pred_cov_for_corr: torch.Tensor | None = None
            # Preferred: use covariance predicted via Cholesky head, if available
            if getattr(model, "_last_pred_chol", None) is not None:
                pred_cov_for_corr = model.chol_to_cov(model._last_pred_chol)
                device = pred_cov_for_corr.device
                dtype = pred_cov_for_corr.dtype
                target_tail = batch.target_time.to(device, dtype)
                centered = target_tail - target_tail.mean(dim=1, keepdim=True)
                denom = float(max(centered.size(1) - 1, 1))
                true_cov = torch.matmul(centered.transpose(1, 2), centered) / denom
            # Fallback: approximate with sample covariance from mean head outputs (legacy behaviour)
            elif getattr(model, "_last_pred_mean", None) is not None:
                pred_mean = model._last_pred_mean
                target_tail = batch.target_time.to(pred_mean.device, pred_mean.dtype)
                pred_center = pred_mean - pred_mean.mean(dim=1, keepdim=True)
                true_center = target_tail - target_tail.mean(dim=1, keepdim=True)
                pred_den = float(max(pred_center.size(1) - 1, 1))
                true_den = float(max(true_center.size(1) - 1, 1))
                pred_cov_for_corr = torch.matmul(pred_center.transpose(1, 2), pred_center) / pred_den
                true_cov = torch.matmul(true_center.transpose(1, 2), true_center) / true_den
            else:
                pred_cov_for_corr = None

            if pred_cov_for_corr is not None:
                eps = 1e-8
                pred_std = torch.sqrt(torch.diagonal(pred_cov_for_corr, dim1=-2, dim2=-1).clamp_min(eps))
                true_std = torch.sqrt(torch.diagonal(true_cov, dim1=-2, dim2=-1).clamp_min(eps))
                pred_corr = pred_cov_for_corr / (pred_std.unsqueeze(-1) * pred_std.unsqueeze(-2) + eps)
                true_corr = true_cov / (true_std.unsqueeze(-1) * true_std.unsqueeze(-2) + eps)
                corr_loss = torch.square(pred_corr - true_corr).mean()
                total_loss = total_loss + model.lambda_corr * corr_loss
                model._last_corr_loss = corr_loss.detach()

        if hasattr(model, "lambda_spectral") and getattr(model, "lambda_spectral", 0.0) > 0 and batch.target_time is not None:
            if getattr(model, "_last_pred_mean", None) is not None:
                pred_mean = model._last_pred_mean
                target_tail = batch.target_time.to(pred_mean.device, pred_mean.dtype)
                tail_mask = batch.target_mask_time
                if tail_mask is None:
                    tail_mask = torch.isfinite(target_tail)
                # If there is missingness in the target tail, skip the spectral penalty to avoid bias.
                if not bool(tail_mask.all()):
                    spec_loss = torch.zeros((), device=pred_mean.device, dtype=pred_mean.dtype)
                    model._last_spec_loss = spec_loss.detach()
                    total_loss = total_loss + model.lambda_spectral * spec_loss
                else:
                    pred_fft = torch.fft.rfft(pred_mean, dim=1)
                    target_fft = torch.fft.rfft(target_tail, dim=1)
                    spec_loss = torch.abs(torch.abs(pred_fft) ** 2 - torch.abs(target_fft) ** 2).mean()
                    total_loss = total_loss + model.lambda_spectral * spec_loss
                    model._last_spec_loss = spec_loss.detach()

        if hasattr(model, "lambda_sliding_cov") and getattr(model, "lambda_sliding_cov", 0.0) > 0:
            if getattr(model, "_last_pred_chol", None) is not None and batch.cov is not None and batch.cov_mask is not None:
                tail_mask = batch.target_mask_time
                if tail_mask is not None and not bool(tail_mask.all()):
                    return total_loss
                pred_cov = model.chol_to_cov(model._last_pred_chol)
                cov = batch.cov.to(pred_cov.device, pred_cov.dtype)
                mask = batch.cov_mask.to(cov.device, cov.dtype).unsqueeze(-1).unsqueeze(-1)
                summed = (cov * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp_min(1.0)
                target_sliding_cov = summed / counts
                sliding_cov_loss = torch.square(pred_cov - target_sliding_cov).mean()
                total_loss = total_loss + model.lambda_sliding_cov * sliding_cov_loss
                if hasattr(model, "_last_sliding_cov_loss"):
                    model._last_sliding_cov_loss = sliding_cov_loss.detach()

        return total_loss

    return loss_fn
