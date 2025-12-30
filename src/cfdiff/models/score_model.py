from __future__ import annotations

import warnings
from typing import Any, Callable, Optional

import torch.nn.functional as F

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRScheduler

from cfdiff.models.time_encodings import GaussianFourierProjection, PositionalEncoding
from cfdiff.utils.dataclasses import DiffusionBatch
from cfdiff.utils.losses import get_sde_loss_fn
from cfdiff.utils.sde import SDE
import math


class MeanHead(nn.Module):
    def __init__(self, d_in: int, pred_len: int, n_channels: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in * 4),
            nn.SiLU(),
            nn.Linear(d_in * 4, pred_len * n_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.net(features)
        return out.view(-1, self.pred_len, self.n_channels)


class CovHead(nn.Module):
    def __init__(self, d_in: int, n_channels: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.out_dim = (n_channels * (n_channels + 1)) // 2
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in * 4),
            nn.SiLU(),
            nn.Linear(d_in * 4, self.out_dim),
        )
        self.register_buffer("tri_rows", torch.tril_indices(n_channels, n_channels)[0], persistent=False)
        self.register_buffer("tri_cols", torch.tril_indices(n_channels, n_channels)[1], persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(features)
        B = raw.size(0)
        chol = torch.zeros(B, self.n_channels, self.n_channels, device=raw.device, dtype=raw.dtype)
        chol[:, self.tri_rows, self.tri_cols] = raw
        diag = torch.diagonal(chol, dim1=-2, dim2=-1)
        diag_pos = torch.nn.functional.softplus(diag) + 1e-6
        chol = chol - torch.diag_embed(diag) + torch.diag_embed(diag_pos)
        return chol


class AssetEncoder(nn.Module):
    def __init__(self, context_len_time: int, d_model: int, num_layers: int, n_head: int) -> None:
        super().__init__()
        self.proj = nn.Linear(context_len_time, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

    def forward(self, context_time: torch.Tensor) -> torch.Tensor:
        x = context_time.transpose(1, 2)  # (B, assets, context_len)
        x = self.proj(x)
        return self.encoder(x)


class ScoreModel(pl.LightningModule):
    """Transformer-based conditional score network with context conditioning."""

    def __init__(
        self,
        n_channels: int,
        sequence_len: int,
        noise_scheduler: SDE,
        pred_len: int,
        pred_len_time: int,
        context_len_time: int,
        target_channels: int,
        add_missing_mask: bool = False,
        d_model: int = 128,
        num_layers: int = 4,
        n_head: int = 8,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        lr_scheduler_type: str = "none",
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        likelihood_weighting: bool = False,
        # timestep sampling controls
        t_sampling_mode: str = "uniform",
        t_beta_alpha: float = 2.0,
        t_beta_beta: float = 5.0,
        t_power_gamma: float = 2.0,
        t_importance_correction: bool = False,
        temporal_loss_weighting: str | None = None,
        temporal_loss_max: float = 1.0,
        lambda_mean: float = 0.0,
        lambda_cov: float = 0.0,
        lambda_corr: float = 0.0,
        lambda_spectral: float = 0.0,
        lambda_sliding_cov: float = 0.0,
        use_asset_encoder: bool = True,
        backbone: str = "transformer",
        mlp_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["noise_scheduler"])

        self.n_channels = n_channels
        self.add_missing_mask = bool(add_missing_mask)
        self.input_channels = n_channels * (2 if self.add_missing_mask else 1)
        self.sequence_len = sequence_len
        self.pred_len = pred_len
        self.pred_len_time = pred_len_time
        self.context_len_time = context_len_time
        self.target_channels = target_channels
        self.context_len = sequence_len - pred_len
        self.d_model = d_model
        self.backbone_type = backbone.lower()
        if self.backbone_type not in {"transformer", "mlp", "lstm"}:
            raise ValueError(f"Unsupported backbone '{backbone}'. Choose from transformer, mlp, or lstm.")
        self.mlp_hidden_dim = mlp_hidden_dim
        self.noise_scheduler = noise_scheduler
        if self.noise_scheduler.G is None:
            self.noise_scheduler.set_noise_scaling(self.pred_len)

        self.time_encoding = GaussianFourierProjection(d_model=d_model)

        # store t sampling controls
        self.t_sampling_mode = str(t_sampling_mode)
        self.t_beta_alpha = float(t_beta_alpha)
        self.t_beta_beta = float(t_beta_beta)
        self.t_power_gamma = float(t_power_gamma)
        self.t_importance_correction = bool(t_importance_correction)
        self.lr_scheduler_type = str(lr_scheduler_type).lower()
        self.lr_scheduler_kwargs = dict(lr_scheduler_kwargs or {})

        # Initialise optional modules to avoid attribute errors
        self.context_embed: Optional[nn.Module] = None
        self.context_pos: Optional[nn.Module] = None
        self.context_encoder: Optional[nn.Module] = None
        self.asset_encoder: Optional[nn.Module] = None
        self.asset_norm: Optional[nn.Module] = None
        self.global_proj: Optional[nn.Module] = None
        self.target_embed: Optional[nn.Module] = None
        self.target_pos: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.projection: Optional[nn.Module] = None
        self.context_flat: Optional[nn.Module] = None
        self.target_flat: Optional[nn.Module] = None
        self.mlp_fusion: Optional[nn.Module] = None
        self.score_head: Optional[nn.Module] = None
        self.context_lstm: Optional[nn.Module] = None
        self.target_lstm: Optional[nn.Module] = None

        if self.backbone_type == "transformer":
            self.context_embed = nn.Linear(self.input_channels, d_model)
            self.context_pos = PositionalEncoding(d_model=d_model, max_len=self.context_len)
            context_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.context_encoder = nn.TransformerEncoder(
                encoder_layer=context_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )

            self.use_asset_encoder = bool(use_asset_encoder)
            if self.use_asset_encoder:
                self.asset_encoder = AssetEncoder(self.context_len_time, d_model, num_layers, n_head)
                self.asset_norm = nn.LayerNorm(d_model)
                self.global_proj = nn.Linear(2 * d_model, d_model)
            else:
                self.asset_encoder = None
                self.asset_norm = None
                self.global_proj = None

            self.target_embed = nn.Linear(self.input_channels, d_model)
            self.target_pos = PositionalEncoding(d_model=d_model, max_len=pred_len)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )
            self.projection = nn.Linear(d_model, n_channels)
        elif self.backbone_type == "mlp":
            if use_asset_encoder:
                warnings.warn(
                    "Asset encoder is only available with the transformer backbone; disabling it for MLP backbone.",
                    stacklevel=2,
                )
            self.use_asset_encoder = False
            self.context_flat = nn.Linear(self.context_len * self.input_channels, d_model)
            self.target_flat = nn.Linear(self.pred_len * self.input_channels, d_model)
            self.mlp_fusion = nn.Sequential(
                nn.Linear(3 * d_model, mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(mlp_hidden_dim, d_model),
                nn.GELU(),
            )
            self.score_head = nn.Linear(d_model, self.pred_len * self.n_channels)
        else:  # lstm backbone
            if use_asset_encoder:
                warnings.warn(
                    "Asset encoder is only available with the transformer backbone; disabling it for LSTM backbone.",
                    stacklevel=2,
                )
            self.use_asset_encoder = False
            self.context_embed = nn.Linear(self.input_channels, d_model)
            self.context_lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
            )
            self.target_embed = nn.Linear(self.input_channels, d_model)
            self.target_lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
            )
            self.projection = nn.Linear(d_model, n_channels)

        self.lambda_mean = float(lambda_mean)
        self.lambda_cov = float(lambda_cov)
        self.lambda_corr = float(lambda_corr)
        self.lambda_spectral = float(lambda_spectral)
        self.lambda_sliding_cov = float(lambda_sliding_cov)
        self.temporal_loss_weighting = temporal_loss_weighting
        self.temporal_loss_max = float(temporal_loss_max)
        self.mean_head: Optional[MeanHead] = None
        self.cov_head: Optional[CovHead] = None
        if self.lambda_mean > 0:
            self.mean_head = MeanHead(d_model, pred_len_time, target_channels)
        if self.lambda_cov > 0:
            self.cov_head = CovHead(d_model, target_channels)
        self._last_pred_mean: Optional[torch.Tensor] = None
        self._last_pred_chol: Optional[torch.Tensor] = None
        self._last_mean_loss: Optional[torch.Tensor] = None
        self._last_cov_loss: Optional[torch.Tensor] = None
        self._last_corr_loss: Optional[torch.Tensor] = None
        self._last_spec_loss: Optional[torch.Tensor] = None
        self._last_sliding_cov_loss: Optional[torch.Tensor] = None

        self.training_loss_fn: Callable[[nn.Module, DiffusionBatch], torch.Tensor] = get_sde_loss_fn(
            scheduler=self.noise_scheduler,
            train=True,
            likelihood_weighting=likelihood_weighting,
            t_sampling_mode=self.t_sampling_mode,
            t_beta_alpha=self.t_beta_alpha,
            t_beta_beta=self.t_beta_beta,
            t_power_gamma=self.t_power_gamma,
            t_importance_correction=self.t_importance_correction,
            temporal_loss_weighting=self.temporal_loss_weighting,
            temporal_loss_max=self.temporal_loss_max,
        )
        self.validation_loss_fn: Callable[[nn.Module, DiffusionBatch], torch.Tensor] = get_sde_loss_fn(
            scheduler=self.noise_scheduler,
            train=False,
            likelihood_weighting=likelihood_weighting,
            t_sampling_mode=self.t_sampling_mode,
            t_beta_alpha=self.t_beta_alpha,
            t_beta_beta=self.t_beta_beta,
            t_power_gamma=self.t_power_gamma,
            t_importance_correction=self.t_importance_correction,
            temporal_loss_weighting=self.temporal_loss_weighting,
            temporal_loss_max=self.temporal_loss_max,
        )

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch: DiffusionBatch) -> torch.Tensor:
        if batch.context is None or batch.target is None:
            raise ValueError("DiffusionBatch must include context and target tensors.")

        context = batch.context
        target = batch.target
        context_time = batch.context_time

        if self.add_missing_mask:
            ctx_mask = batch.context_mask
            if ctx_mask is None:
                ctx_mask = torch.ones_like(context, dtype=context.dtype, device=context.device)
            else:
                ctx_mask = ctx_mask.to(device=context.device, dtype=context.dtype)
                if ctx_mask.ndim == 2:
                    ctx_mask = ctx_mask.unsqueeze(0).expand(context.size(0), -1, -1)
            tgt_mask = batch.target_mask
            if tgt_mask is None:
                tgt_mask = torch.ones_like(target, dtype=target.dtype, device=target.device)
            else:
                tgt_mask = tgt_mask.to(device=target.device, dtype=target.dtype)
                if tgt_mask.ndim == 2:
                    tgt_mask = tgt_mask.unsqueeze(0).expand(target.size(0), -1, -1)
            context_in = torch.cat([context, ctx_mask], dim=-1)
            target_in = torch.cat([target, tgt_mask], dim=-1)
        else:
            context_in = context
            target_in = target

        if context.shape[1] != self.context_len:
            raise ValueError(f"Expected context length {self.context_len}; received {context.shape[1]}")
        if target.shape[1] != self.pred_len:
            raise ValueError(f"Expected target length {self.pred_len}; received {target.shape[1]}")

        timesteps: Optional[torch.Tensor] = batch.timesteps
        if timesteps is None:
            raise ValueError("DiffusionBatch must include timesteps during score evaluation.")
        if self.backbone_type == "transformer":
            if self.context_embed is None or self.context_pos is None or self.context_encoder is None:
                raise RuntimeError("Transformer backbone modules were not initialised.")
            context_emb = self.context_embed(context_in)
            context_emb = self.context_pos(context_emb)
            context_memory = self.context_encoder(context_emb)

            asset_global: Optional[torch.Tensor] = None
            if self.use_asset_encoder and context_time is not None:
                if context_time.shape[1] != self.context_len_time:
                    raise ValueError(
                        f"Expected context_time length {self.context_len_time}; received {context_time.shape[1]}"
                    )
                asset_memory = self.asset_encoder(context_time)
                asset_memory = self.asset_norm(asset_memory)
                context_memory = torch.cat([context_memory, asset_memory], dim=1)
                asset_global = asset_memory.mean(dim=1)

            if self.target_embed is None or self.target_pos is None or self.decoder is None or self.projection is None:
                raise RuntimeError("Transformer backbone target modules were not initialised.")
            target_emb = self.target_embed(target_in)
            target_emb = self.target_pos(target_emb)
            target_emb = self.time_encoding(target_emb, timesteps)
            decoded = self.decoder(tgt=target_emb, memory=context_memory)
            score = self.projection(decoded)

            global_feature = context_memory.mean(dim=1)
            if self.use_asset_encoder and self.global_proj is not None and asset_global is not None:
                fusion = torch.cat([global_feature, asset_global], dim=-1)
                global_feature = self.global_proj(fusion)
        elif self.backbone_type == "mlp":
            if self.context_flat is None or self.target_flat is None or self.mlp_fusion is None or self.score_head is None:
                raise RuntimeError("MLP backbone modules were not initialised.")
            batch_size = context.size(0)
            context_flat = context_in.reshape(batch_size, -1)
            target_flat = target_in.reshape(batch_size, -1)
            context_feat = F.gelu(self.context_flat(context_flat))
            target_feat = F.gelu(self.target_flat(target_flat))
            time_feat = self._time_embedding_vector(timesteps)
            fusion_input = torch.cat([context_feat, target_feat, time_feat], dim=-1)
            global_feature = self.mlp_fusion(fusion_input)
            score_flat = self.score_head(global_feature)
            score = score_flat.view(batch_size, self.pred_len, self.n_channels)
        else:  # lstm backbone
            if self.context_embed is None or self.context_lstm is None or self.target_embed is None or self.target_lstm is None or self.projection is None:
                raise RuntimeError("LSTM backbone modules were not initialised.")
            context_emb = self.context_embed(context_in)
            context_seq, (h_ctx, c_ctx) = self.context_lstm(context_emb)
            target_emb = self.target_embed(target_in)
            target_emb = self.time_encoding(target_emb, timesteps)
            target_out, _ = self.target_lstm(target_emb, (h_ctx, c_ctx))
            score = self.projection(target_out)
            global_feature = h_ctx[-1]

        self._last_pred_mean = None
        self._last_pred_chol = None
        self._last_mean_loss = None
        self._last_cov_loss = None
        self._last_corr_loss = None
        self._last_spec_loss = None
        self._last_sliding_cov_loss = None
        if self.mean_head is not None:
            self._last_pred_mean = self.mean_head(global_feature)
        if self.cov_head is not None:
            self._last_pred_chol = self.cov_head(global_feature)

        return score

    def _time_embedding_vector(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() != 1:
            raise ValueError("Timesteps must be a 1-D tensor for MLP/LSTM backbones.")
        dtype = self.time_encoding.dense.weight.dtype
        device = timesteps.device
        dummy = torch.zeros((timesteps.size(0), 1, self.d_model), device=device, dtype=dtype)
        return self.time_encoding(dummy, timesteps).squeeze(1)

    def chol_to_cov(self, chol: torch.Tensor) -> torch.Tensor:
        return chol @ chol.transpose(-1, -2)

    def training_step(self, batch: DiffusionBatch, batch_idx: int) -> torch.Tensor:
        loss = self.training_loss_fn(self, batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch))
        if self.lambda_mean > 0 and getattr(self, "_last_mean_loss", None) is not None:
            self.log(
                "train/mean_penalty",
                self._last_mean_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch),
            )
        if self.lambda_cov > 0 and getattr(self, "_last_cov_loss", None) is not None:
            self.log(
                "train/cov_penalty",
                self._last_cov_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch),
            )
        if self.lambda_corr > 0 and getattr(self, "_last_corr_loss", None) is not None:
            self.log(
                "train/corr_penalty",
                self._last_corr_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch),
            )
        if self.lambda_spectral > 0 and getattr(self, "_last_spec_loss", None) is not None:
            self.log(
                "train/spectral_penalty",
                self._last_spec_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch),
            )
        if getattr(self, "lambda_sliding_cov", 0.0) > 0 and getattr(self, "_last_sliding_cov_loss", None) is not None:
            self.log(
                "train/sliding_cov_penalty",
                self._last_sliding_cov_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=len(batch),
            )
        return loss

    def validation_step(self, batch: DiffusionBatch, batch_idx: int) -> None:
        loss = self.validation_loss_fn(self, batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=len(batch))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = self._build_lr_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_lr_scheduler(self, optimizer: optim.Optimizer) -> Optional[dict[str, Any]]:
        sched_type = (self.lr_scheduler_type or "none").lower()
        if sched_type in {"", "none", "constant"}:
            return None

        kwargs = dict(self.lr_scheduler_kwargs)
        interval = str(kwargs.pop("interval", "epoch"))

        if sched_type == "cosine":
            T_max = int(kwargs.pop("T_max", getattr(self.trainer, "max_epochs", 1)))
            eta_min = float(kwargs.pop("eta_min", 0.0))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, T_max), eta_min=eta_min)
        elif sched_type in {"cosine_warmup", "warmup_cosine"}:
            total_steps = int(kwargs.pop("total_steps", getattr(self.trainer, "estimated_stepping_batches", 0)))
            if total_steps <= 0:
                raise ValueError("cosine_warmup scheduler requires 'total_steps' > 0.")
            warmup_steps = int(kwargs.pop("warmup_steps", max(1, total_steps // 10)))

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = min(1.0, (step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            interval = "step"
        elif sched_type in {"warmup_linear", "linear"}:
            total_steps = int(kwargs.pop("total_steps", getattr(self.trainer, "estimated_stepping_batches", 0)))
            if total_steps <= 0:
                raise ValueError("warmup_linear scheduler requires 'total_steps' > 0.")
            warmup_steps = int(kwargs.pop("warmup_steps", max(1, total_steps // 10)))
            min_lr_scale = float(kwargs.pop("min_lr_scale", 0.0))  # final lr factor at the end (0=to zero)

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                progress = min(1.0, (step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
                return (1.0 - progress) * (1.0 - min_lr_scale) + min_lr_scale

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            interval = "step"
        else:
            raise ValueError(f"Unsupported lr_scheduler_type '{self.lr_scheduler_type}'.")

        return {"scheduler": scheduler, "interval": interval, "frequency": 1}
