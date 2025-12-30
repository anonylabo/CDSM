from typing import Any, Callable, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import math

from fdiff.models.transformer import GaussianFourierProjection, PositionalEncoding
from fdiff.schedulers.sde import SDE
from fdiff.utils.dataclasses import DiffusableBatch
from fdiff.utils.losses import get_sde_loss_fn


class ScoreModel(pl.LightningModule):
    """Transformer-based unconditional score model."""

    def __init__(
        self,
        n_channels: int,
        max_len: int,
        noise_scheduler: SDE,
        add_missing_mask: bool = False,
        fourier_noise_scaling: bool = True,
        d_model: int = 60,
        num_layers: int = 3,
        n_head: int = 12,
        num_training_steps: int = 1000,
        lr_max: float = 1e-3,
        lr_scheduler_type: str = "cosine_warmup",
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        likelihood_weighting: bool = False,
        # timestep sampling controls
        t_sampling_mode: str = "uniform",
        t_beta_alpha: float = 2.0,
        t_beta_beta: float = 5.0,
        t_power_gamma: float = 2.0,
        t_importance_correction: bool = False,
        backbone: str = "transformer",
        mlp_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.n_channels = n_channels
        self.add_missing_mask = bool(add_missing_mask)
        self.input_channels = n_channels * (2 if self.add_missing_mask else 1)
        self.noise_scheduler = noise_scheduler
        self.num_training_steps = num_training_steps
        self.lr_max = lr_max
        self.lr_scheduler_type = str(lr_scheduler_type).lower()
        self.lr_scheduler_kwargs = dict(lr_scheduler_kwargs or {})
        self.scale_noise = fourier_noise_scaling
        self.likelihood_weighting = likelihood_weighting
        # store t sampling controls
        self.t_sampling_mode = str(t_sampling_mode)
        self.t_beta_alpha = float(t_beta_alpha)
        self.t_beta_beta = float(t_beta_beta)
        self.t_power_gamma = float(t_power_gamma)
        self.t_importance_correction = bool(t_importance_correction)
        self.backbone_type = backbone.lower()
        if self.backbone_type not in {"transformer", "mlp", "lstm"}:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from transformer/mlp/lstm.")
        self.mlp_hidden_dim = mlp_hidden_dim

        self.training_loss_fn, self.validation_loss_fn = self.set_loss_fn()

        self.time_encoder = GaussianFourierProjection(d_model=d_model)
        if self.backbone_type == "transformer":
            self.embedder = nn.Linear(self.input_channels, d_model)
            self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
            self.unembedder = nn.Linear(d_model, n_channels)
        elif self.backbone_type == "mlp":
            self.embedder = nn.Linear(max_len * self.input_channels, d_model)
            self.pos_encoder = None
            self.backbone = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, mlp_hidden_dim),
                        nn.GELU(),
                        nn.Linear(mlp_hidden_dim, d_model),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.unembedder = nn.Linear(d_model, max_len * n_channels)
        else:  # lstm
            self.embedder = nn.Linear(self.input_channels, d_model)
            self.pos_encoder = None
            self.backbone = nn.ModuleList(
                [
                    nn.LSTM(
                        input_size=d_model,
                        hidden_size=d_model,
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.unembedder = nn.Linear(d_model, n_channels)
        self.save_hyperparameters()

    def forward(self, batch: DiffusableBatch) -> torch.Tensor:
        X = batch.X
        if X.size(1) != self.max_len or X.size(2) != self.n_channels:
            raise ValueError(
                f"Expected input shape (batch,{self.max_len},{self.n_channels}) but got {tuple(X.size())}"
            )
        if self.add_missing_mask:
            mask = batch.X_mask
            if mask is None:
                mask = torch.ones_like(X, dtype=X.dtype, device=X.device)
            else:
                mask = mask.to(device=X.device, dtype=X.dtype)
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).expand(X.size(0), -1, -1)
            X_in = torch.cat([X, mask], dim=-1)
        else:
            X_in = X
        timesteps = batch.timesteps
        if timesteps is None or timesteps.size(0) != len(batch):
            raise ValueError("DiffusableBatch must include timesteps")

        if self.backbone_type == "transformer":
            X = self.embedder(X_in)
            if self.pos_encoder is not None:
                X = self.pos_encoder(X)
            X = self.time_encoder(X, timesteps)
            X = self.backbone(X)
            return self.unembedder(X)

        if self.backbone_type == "mlp":
            batch_size = X_in.size(0)
            X = X_in.reshape(batch_size, -1)
            X = self.embedder(X)
            X = self.time_encoder(X, timesteps, use_time_axis=False)
            for layer in self.backbone:
                X = X + layer(X)
            X = self.unembedder(X)
            return X.view(batch_size, self.max_len, self.n_channels)

        # lstm backbone
        X = self.embedder(X_in)
        X = self.time_encoder(X, timesteps)
        for lstm in self.backbone:
            out, _ = lstm(X)
            X = X + out
        return self.unembedder(X)

    def training_step(
        self,
        batch: DiffusableBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)
        self.log_dict(
            {"train/loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )
        return loss

    def validation_step(
        self,
        batch: DiffusableBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        loss = self.validation_loss_fn(self, batch)
        self.log_dict(
            {"val/loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr_max)
        scheduler = self._build_lr_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_lr_scheduler(self, optimizer: optim.Optimizer) -> Optional[dict[str, Any]]:
        sched_type = (self.lr_scheduler_type or "none").lower()
        if sched_type in {"", "none"}:
            return None

        kwargs = dict(self.lr_scheduler_kwargs)
        interval = str(kwargs.pop("interval", "epoch"))

        if sched_type == "cosine":
            T_max = int(kwargs.pop("T_max", getattr(self.trainer, "max_epochs", 1)))
            eta_min = float(kwargs.pop("eta_min", 0.0))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, T_max), eta_min=eta_min)
        elif sched_type in {"cosine_warmup", "warmup_cosine"}:
            total_steps = int(kwargs.pop("total_steps", self.num_training_steps))
            if total_steps <= 0:
                total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
            if total_steps <= 0:
                raise ValueError("cosine_warmup scheduler requires 'total_steps' > 0.")
            warmup_cfg = kwargs.pop("warmup_steps", None)
            if warmup_cfg is None:
                warmup_steps = max(1, total_steps // 10)
            else:
                warmup_steps = int(warmup_cfg)

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = min(1.0, (step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            interval = "step"
        else:
            raise ValueError(f"Unsupported lr_scheduler_type '{self.lr_scheduler_type}'.")

        return {"scheduler": scheduler, "interval": interval, "frequency": 1}

    def set_loss_fn(
        self,
    ) -> tuple[
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
        Callable[[nn.Module, DiffusableBatch], torch.Tensor],
    ]:
        train_fn = get_sde_loss_fn(
            scheduler=self.noise_scheduler,
            train=True,
            likelihood_weighting=self.likelihood_weighting,
            t_sampling_mode=self.t_sampling_mode,
            t_beta_alpha=self.t_beta_alpha,
            t_beta_beta=self.t_beta_beta,
            t_power_gamma=self.t_power_gamma,
            t_importance_correction=self.t_importance_correction,
        )
        val_fn = get_sde_loss_fn(
            scheduler=self.noise_scheduler,
            train=False,
            likelihood_weighting=self.likelihood_weighting,
            t_sampling_mode=self.t_sampling_mode,
            t_beta_alpha=self.t_beta_alpha,
            t_beta_beta=self.t_beta_beta,
            t_power_gamma=self.t_power_gamma,
            t_importance_correction=self.t_importance_correction,
        )
        return train_fn, val_fn
