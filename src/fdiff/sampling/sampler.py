from __future__ import annotations

from typing import Optional

import torch
from tqdm import tqdm

from fdiff.models.score_model import ScoreModel
from fdiff.schedulers.sde import SDE
from fdiff.utils.dataclasses import DiffusableBatch


class DiffusionSampler:
    def __init__(
        self,
        score_model: ScoreModel,
        noise_scheduler: SDE,
        sample_batch_size: int,
    ) -> None:
        self.score_model = score_model
        self.noise_scheduler = noise_scheduler
        self.sample_batch_size = int(sample_batch_size)
        self.n_channels = score_model.n_channels
        self.max_len = score_model.max_len

    def sample(
        self,
        num_samples: int,
        num_diffusion_steps: Optional[int] = None,
    ) -> torch.Tensor:
        self.score_model.eval()
        num_diffusion_steps = num_diffusion_steps or self.score_model.num_training_steps
        self.noise_scheduler.set_timesteps(num_diffusion_steps)

        batches = []
        num_batches = (num_samples + self.sample_batch_size - 1) // self.sample_batch_size
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Sampling", unit="batch", leave=False):
                current_bs = min(num_samples - batch_idx * self.sample_batch_size, self.sample_batch_size)
                samples = self.sample_prior(current_bs)
                for t in self.noise_scheduler.timesteps:
                    timesteps = torch.full(
                        (current_bs,),
                        t,
                        device=self.score_model.device,
                        dtype=torch.float32,
                    )
                    batch = DiffusableBatch(X=samples, timesteps=timesteps)
                    score = self.score_model(batch)
                    out = self.noise_scheduler.step(score, float(t), samples)
                    samples = out.prev_sample
                batches.append(samples.cpu())
        return torch.cat(batches, dim=0)

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        if isinstance(self.noise_scheduler, SDE):
            return self.noise_scheduler.prior_sampling(
                (batch_size, self.max_len, self.n_channels)
            ).to(device=self.score_model.device)
        raise NotImplementedError("Unknown scheduler type")
