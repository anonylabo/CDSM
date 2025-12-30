import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model,
            max_norm=math.sqrt(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pe = self.embedding(position)
        return x + pe


class GaussianFourierProjection(nn.Module):
    def __init__(self, d_model: int, scale: float = 30.0):
        super().__init__()
        self.d_model = d_model
        self.W = nn.Parameter(torch.randn((d_model + 1) // 2) * scale, requires_grad=False)
        self.dense = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        use_time_axis: bool = True,
    ) -> torch.Tensor:
        time_proj = timesteps[:, None] * self.W[None, :] * 2 * np.pi
        embeddings = torch.cat([torch.sin(time_proj), torch.cos(time_proj)], dim=-1)
        t_emb = embeddings[:, : self.d_model]
        if use_time_axis:
            t_emb = t_emb.unsqueeze(1)
        projected = self.dense(t_emb)
        return x + projected
