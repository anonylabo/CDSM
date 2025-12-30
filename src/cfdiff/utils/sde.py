from __future__ import annotations

import abc
import math
from collections import namedtuple
from typing import Optional

import torch

SamplingOutput = namedtuple("SamplingOutput", ["prev_sample"])


class SDE(abc.ABC):
    """Abstract SDE class for mini-batch processing."""

    def __init__(
        self,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
        temporal_noise_ramp: str | None = None,
        temporal_ramp_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.noise_scaling = fourier_noise_scaling
        self.eps = eps
        self.temporal_noise_ramp = temporal_noise_ramp
        self.temporal_ramp_max = float(temporal_ramp_max)
        self.G: Optional[torch.Tensor] = None

    @property
    def T(self) -> float:
        return 1.0

    @abc.abstractmethod
    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @abc.abstractmethod
    def step(self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor) -> SamplingOutput:
        ...

    def set_noise_scaling(self, length: int) -> None:
        G = torch.ones(length)
        if self.noise_scaling:
            G = 1 / math.sqrt(2) * G
            G[0] *= math.sqrt(2)
            if length % 2 == 0:
                G[length // 2] *= math.sqrt(2)

        # Optional temporal ramp: increase noise for later timesteps in the prediction block.
        if self.temporal_noise_ramp:
            ramp_mode = str(self.temporal_noise_ramp).lower()
            max_scale = max(1.0, float(self.temporal_ramp_max))
            if ramp_mode in {"linear", "lin"}:
                ramp = torch.linspace(1.0, max_scale, steps=length)
            elif ramp_mode in {"exp", "exponential"}:
                ramp = torch.logspace(0.0, math.log10(max_scale), steps=length)
            else:
                raise ValueError(f"Unsupported temporal_noise_ramp='{self.temporal_noise_ramp}'")
            # Normalise ramp to preserve average noise level ~1.
            ramp = ramp / ramp.mean()
            G = G * ramp

        self.G = G
        self.G_matrix = torch.diag(G)

    def set_timesteps(self, num_diffusion_steps: int) -> None:
        self.timesteps = torch.linspace(1.0, self.eps, num_diffusion_steps)
        self.step_size = float(self.timesteps[0] - self.timesteps[1])

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        mean, _ = self.marginal_prob(original_samples, timesteps)
        return mean + noise

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        device = self.G_matrix.device
        scaling_matrix = self.G_matrix.to(device).view(1, self.G_matrix.shape[0], self.G_matrix.shape[1])
        z = torch.randn(*shape, device=device)
        return torch.matmul(scaling_matrix, z)


class VEScheduler(SDE):
    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
        temporal_noise_ramp: str | None = None,
        temporal_ramp_max: float = 1.0,
    ) -> None:
        super().__init__(
            fourier_noise_scaling=fourier_noise_scaling,
            eps=eps,
            temporal_noise_ramp=temporal_noise_ramp,
            temporal_ramp_max=temporal_ramp_max,
        )
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None

        sigma_min = torch.tensor(self.sigma_min).type_as(t)
        sigma_max = torch.tensor(self.sigma_max).type_as(t)
        std = (sigma_min * (sigma_max / sigma_min) ** t).view(-1, 1) * self.G.to(x.device)
        mean = x
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        return self.sigma_max * super().prior_sampling(shape)

    def step(self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor) -> SamplingOutput:
        if not hasattr(self, "step_size"):
            raise RuntimeError("Call set_timesteps() before sampling.")

        sqrt_derivative = (
            self.sigma_min
            * math.sqrt(2 * math.log(self.sigma_max / self.sigma_min))
            * (self.sigma_max / self.sigma_min) ** timestep
        )
        diffusion = torch.diag_embed(sqrt_derivative * self.G.to(sample.device))

        drift = -(diffusion * diffusion) @ model_output
        z = torch.randn_like(sample)
        step_tensor = torch.tensor(self.step_size, dtype=sample.dtype, device=sample.device)
        x = sample - drift * step_tensor + torch.sqrt(step_tensor) * torch.matmul(diffusion, z)
        return SamplingOutput(prev_sample=x)


class VPScheduler(SDE):
    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        fourier_noise_scaling: bool = False,
        eps: float = 1e-5,
        temporal_noise_ramp: str | None = None,
        temporal_ramp_max: float = 1.0,
    ) -> None:
        super().__init__(
            fourier_noise_scaling=fourier_noise_scaling,
            eps=eps,
            temporal_noise_ramp=temporal_noise_ramp,
            temporal_ramp_max=temporal_ramp_max,
        )
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def marginal_prob(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.G is None:
            self.set_noise_scaling(x.shape[1])
        assert self.G is not None

        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean_coeff = torch.exp(log_mean_coeff).view(-1, 1, 1)
        mean = mean_coeff * x

        std_coeff = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff)).view(-1, 1)
        G = self.G.to(x.device).view(1, -1)
        std = std_coeff * G  # (batch, pred_len)
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...]) -> torch.Tensor:
        return super().prior_sampling(shape)

    def step(self, model_output: torch.Tensor, timestep: float, sample: torch.Tensor) -> SamplingOutput:
        if not hasattr(self, "step_size"):
            raise RuntimeError("Call set_timesteps() before sampling.")

        timestep_tensor = torch.as_tensor(timestep, dtype=torch.float32, device=sample.device)
        beta_t = self.beta_0 + timestep_tensor * (self.beta_1 - self.beta_0)
        diffusion = torch.sqrt(beta_t) * torch.diag_embed(self.G.to(sample.device))

        score_term = (diffusion * diffusion) @ model_output
        forward_drift = 0.5 * beta_t * sample
        drift = forward_drift + score_term
        z = torch.randn_like(sample)
        step_tensor = torch.tensor(self.step_size, dtype=sample.dtype, device=sample.device)
        x = sample - drift * step_tensor + torch.sqrt(step_tensor) * torch.matmul(diffusion, z)
        return SamplingOutput(prev_sample=x)
