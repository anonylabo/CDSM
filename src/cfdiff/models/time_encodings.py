from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Learnable positional embeddings added to token representations."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
            max_norm=math.sqrt(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected tensor of shape (B, T, D); received {tuple(x.shape)}")
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        encoding = self.embedding(positions)
        return x + encoding


class TimeEncoding(nn.Module):
    """Learnable embeddings for discrete diffusion timesteps."""

    def __init__(self, d_model: int, max_time: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max_time,
            embedding_dim=d_model,
            max_norm=math.sqrt(d_model),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.LongTensor) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.size(0) != x.size(0):
            raise ValueError("timesteps must be shape (batch,)")
        encoding = self.embedding(timesteps).unsqueeze(1)
        return x + encoding


class GaussianFourierProjection(nn.Module):
    """Random Fourier features for continuous diffusion timesteps."""

    def __init__(self, d_model: int, scale: float = 30.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.W = nn.Parameter(torch.randn((d_model + 1) // 2) * scale, requires_grad=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.size(0) != x.size(0):
            raise ValueError("timesteps must be shape (batch,)")
        projected = timesteps[:, None] * self.W[None, :] * 2 * np.pi
        embeddings = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        embeddings = embeddings[:, : self.d_model]
        return x + self.proj(embeddings.unsqueeze(1))
