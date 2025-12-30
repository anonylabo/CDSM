from __future__ import annotations

from typing import Optional, Tuple

import torch
from tqdm import tqdm

from cfdiff.models import ScoreModel
from cfdiff.utils.dataclasses import DiffusionBatch
from cfdiff.utils.fourier import tensor_ifft_realimag
from cfdiff.utils.sde import SDE


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModel,
        noise_scheduler: SDE,
        context_len: int,
        target_len: int,
        target_time_len: int,
        sample_batch_size: int,
        fourier_transform: bool = False,
    ) -> None:
        self.score_model = score_model
        self.noise_scheduler = noise_scheduler
        self.context_len = context_len
        self.target_len = target_len
        self.target_time_len = target_time_len
        self.sample_batch_size = sample_batch_size
        self.fourier_transform = fourier_transform
        if self.noise_scheduler.G is None:
            self.noise_scheduler.set_noise_scaling(self.target_len)

    def _run_diffusion(
        self,
        context: torch.Tensor,
        context_time: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        num_diffusion_steps: Optional[int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        device = self.score_model.device
        context = context.to(device)
        context_time = context_time.to(device) if context_time is not None else None
        context_mask = context_mask.to(device) if context_mask is not None else None
        batch_size = context.size(0)
        n_channels = self.score_model.n_channels

        steps = num_diffusion_steps or 50
        self.noise_scheduler.set_timesteps(steps)
        samples = self.noise_scheduler.prior_sampling((batch_size, self.target_len, n_channels)).to(device)
        target_mask = None
        if getattr(self.score_model, "add_missing_mask", False):
            target_mask = torch.ones_like(samples, dtype=samples.dtype, device=samples.device)

        with torch.no_grad():
            for t in tqdm(self.noise_scheduler.timesteps, desc="Diffusion", leave=False):
                timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
                batch = DiffusionBatch(
                    context=context,
                    target=samples,
                    timesteps=timesteps,
                    context_time=context_time,
                    context_mask=context_mask,
                    target_mask=target_mask,
                )
                score = self.score_model(batch)
                out = self.noise_scheduler.step(score, float(t), samples)
                samples = out.prev_sample

        # one final pass at t=0 to extract mean/cov predictions conditioned on context
        with torch.no_grad():
            timesteps = torch.zeros(batch_size, device=device)
            batch = DiffusionBatch(
                context=context,
                target=samples,
                timesteps=timesteps,
                context_time=context_time,
                context_mask=context_mask,
                target_mask=target_mask,
            )
            _ = self.score_model(batch)
            pred_mean = self.score_model._last_pred_mean
            pred_chol = self.score_model._last_pred_chol

        return samples, pred_mean, pred_chol

    def sample(
        self,
        context: torch.Tensor,
        context_time: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_diffusion_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        samples, pred_mean, pred_chol = self._run_diffusion(context, context_time, context_mask, num_diffusion_steps)
        samples = samples.cpu()
        if pred_mean is not None:
            pred_mean = pred_mean.cpu()
        if pred_chol is not None:
            pred_chol = pred_chol.cpu()
        if self.fourier_transform:
            samples = tensor_ifft_realimag(samples, signal_len=self.target_time_len)
        return samples, pred_mean, pred_chol

    def predict_loader(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_diffusion_steps: Optional[int] = None,
        max_batches: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds: list[torch.Tensor] = []
        truths: list[torch.Tensor] = []

        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if not isinstance(batch, DiffusionBatch):
                raise TypeError("Dataloader must yield DiffusionBatch instances.")

            generated, pred_mean, pred_chol = self.sample(
                context=batch.context,
                context_time=batch.context_time,
                context_mask=getattr(batch, "context_mask", None),
                num_diffusion_steps=num_diffusion_steps,
            )
            preds.append(generated)
            truth = batch.target_time.cpu() if batch.target_time is not None else batch.target.cpu()
            truths.append(truth)

        if not preds:
            raise ValueError("Dataloader produced no batches; ensure it is not empty.")

        return torch.cat(preds, dim=0), torch.cat(truths, dim=0)
