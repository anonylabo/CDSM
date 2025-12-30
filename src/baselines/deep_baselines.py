"""Trainable deep-learning baselines implemented in PyTorch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cfdiff.stats import compute_sliding_covariances


def _to_tensor(array: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        tensor = array
    else:
        tensor = torch.as_tensor(array, dtype=torch.float32)
    return tensor.to(device=device, dtype=torch.float32)


def _covariance_matrix(sequence: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute an empirical covariance matrix with diagonal jitter."""

    if sequence.ndim != 2:
        raise ValueError(f"Expected 2-D sequence, got shape {tuple(sequence.shape)}")
    centered = sequence - sequence.mean(dim=0, keepdim=True)
    denom = max(centered.size(0) - 1, 1)
    cov = centered.transpose(0, 1) @ centered / denom
    cov = (cov + cov.transpose(0, 1)) * 0.5
    eye = torch.eye(cov.size(-1), device=cov.device, dtype=cov.dtype)
    return cov + eps * eye


class CABModel(nn.Module):
    """CNN-BiLSTM baseline mirroring the CAB architecture from the paper."""

    def __init__(
        self,
        n_assets: int,
        lookback_window: int,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        shrinkage: float = 0.8,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback_window
        self.shrinkage = float(shrinkage)

        # Input: (B, 1, L, N, N) -> (B, 1, L, N, N)
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.lstm_input_dim = n_assets * n_assets
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * lstm_hidden,
            num_heads=n_heads,
            batch_first=True,
        )
        self.fc = nn.Linear(2 * lstm_hidden, n_assets * n_assets)

    def forward(self, cov_seq: torch.Tensor, historical_cov: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cov_seq:
            Tensor of shape (B, L, N, N) with rolling covariance matrices.
        historical_cov:
            Tensor of shape (B, N, N) representing the latest covariance used
            for shrinkage blending.
        """

        batch_size, seq_len, n, _ = cov_seq.shape
        x = cov_seq.unsqueeze(1)  # (B, 1, L, N, N)
        conv_out = self.conv3d(x).squeeze(1)  # (B, L, N, N)
        conv_flat = conv_out.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(conv_flat)
        lstm_out = self.dropout(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)

        y = self.fc(pooled).view(batch_size, n, n)
        y_sym = 0.5 * (y + y.transpose(-2, -1))

        eigvals, eigvecs = torch.linalg.eigh(y_sym)
        eigvals = torch.clamp(eigvals, min=0.0)
        y_psd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)

        if historical_cov is not None:
            return y_psd * self.shrinkage + historical_cov * (1.0 - self.shrinkage)
        return y_psd


@dataclass
class CNNBiLSTMCovarianceBaseline:
    """Trainable CAB baseline wrapped for the baseline runner."""

    lookback_window: Optional[int] = None
    cov_window: Optional[int] = None
    cov_eps: float = 1e-5
    lstm_hidden: int = 128
    lstm_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.2
    shrinkage: float = 0.8
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    device: str = "cpu"
    verbose: bool = True

    def __post_init__(self) -> None:
        self._model: Optional[CABModel] = None
        self.asset_names: List[str] = []
        self._lookback: Optional[int] = None

    @property
    def model(self) -> CABModel:
        if self._model is None:
            raise RuntimeError("CAB baseline has not been fitted yet.")
        return self._model

    def _build_cov_sequence(self, context_batch: torch.Tensor) -> torch.Tensor:
        # context_batch: (B, T, A)
        if context_batch.ndim != 3:
            raise ValueError(f"Expected context batch of shape (B, T, A); got {tuple(context_batch.shape)}")
        covs = []
        window = int(self.cov_window or context_batch.size(1))
        lookback = int(self._lookback or context_batch.size(1))
        for sample in context_batch:
            seq, _ = compute_sliding_covariances(sample, window=window, eps=self.cov_eps)
            covs.append(seq[-lookback:, :, :])
        return torch.stack(covs, dim=0)

    def _target_covariance(self, target_batch: torch.Tensor) -> torch.Tensor:
        covs = [_covariance_matrix(target, eps=self.cov_eps) for target in target_batch]
        return torch.stack(covs, dim=0)

    def fit(
        self,
        datamodule,
        asset_names: Sequence[str],
        context_len: int,
        pred_len: int,
    ) -> None:
        """Fit the CAB model on the datamodule's training split."""

        self.asset_names = list(asset_names)
        self._lookback = int(self.lookback_window or context_len)
        device = torch.device(self.device)

        train_loader = datamodule.train_dataloader()
        sample_batch = next(iter(train_loader))
        if not hasattr(sample_batch, "context_time") or getattr(sample_batch, "context_time") is None:
            raise ValueError("CAB baseline requires time-domain context_time from the datamodule.")

        n_assets = sample_batch.context_time.shape[-1]
        self._model = CABModel(
            n_assets=n_assets,
            lookback_window=self._lookback,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers,
            n_heads=self.n_heads,
            dropout=self.dropout,
            shrinkage=self.shrinkage,
        ).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        num_samples = 0

        for epoch in range(int(self.epochs)):
            epoch_loss = 0.0
            self.model.train()
            for batch in train_loader:
                context = batch.context_time.to(device)
                target = batch.target_time.to(device)
                cov_seq = self._build_cov_sequence(context)
                hist_cov = cov_seq[:, -1, :, :]
                pred_cov = self.model(cov_seq, historical_cov=hist_cov)
                target_cov = self._target_covariance(target)
                loss = F.mse_loss(pred_cov, target_cov)

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

                batch_size = context.size(0)
                epoch_loss += float(loss.detach()) * batch_size
                num_samples += batch_size

            if self.verbose and num_samples > 0:
                avg_loss = epoch_loss / num_samples
                print(f"[CAB] epoch {epoch + 1}/{self.epochs} - mse={avg_loss:.6f}")

        self.model.eval()

    def predict(self, context_window: np.ndarray, columns: Iterable[str] | None = None) -> pd.DataFrame:
        """Predict a covariance matrix given a single context window."""

        if self._model is None:
            raise RuntimeError("CAB baseline must be fitted before calling predict().")

        device = next(self.model.parameters()).device
        ctx = _to_tensor(context_window, device=device)
        if ctx.ndim != 2:
            raise ValueError(f"Expected context window of shape (T, A); got {tuple(ctx.shape)}")

        ctx = ctx.unsqueeze(0)  # (1, T, A)
        cov_seq = self._build_cov_sequence(ctx)
        hist_cov = cov_seq[:, -1, :, :]
        with torch.no_grad():
            pred_cov = self.model(cov_seq, historical_cov=hist_cov)[0].cpu().numpy()

        if columns is None:
            if self.asset_names:
                columns = self.asset_names
            else:
                columns = [f"asset_{i}" for i in range(pred_cov.shape[0])]
        return pd.DataFrame(pred_cov, index=list(columns), columns=list(columns))
