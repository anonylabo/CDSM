from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningModule, Trainer

from cfdiff.eval import (
    compute_corr_structure_metrics,
    compute_covariance_metrics,
    compute_distribution_metrics,
    compute_metric_collection,
    compute_series_metrics,
    compute_time_aligned_metrics,
)
from cfdiff.sampling import DiffusionSampler
from cfdiff.utils.dataclasses import DiffusionBatch

log = logging.getLogger(__name__)


@dataclass
class EpochMetricsConfig:
    max_batches: int = 1
    max_windows: int = 32
    log_prefix: str = "epoch"
    compute_fourier: bool = False
    sampler_num_diffusion_steps: Optional[int] = None
    sampler_sample_batch_size: Optional[int] = None
    num_mc_samples: int = 1
    save_mc_artifacts: bool = False


class EpochMetricsLogger(Callback):
    """Lightning callback to record extra metrics at the end of each validation epoch."""

    def __init__(
        self,
        datamodule,
        dataset_params: Dict[str, int],
        scheduler_cfg,
        sampler_cfg,
        fourier_transform: bool,
        config: EpochMetricsConfig,
    ) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.dataset_params = dataset_params
        self.scheduler_cfg = OmegaConf.create(OmegaConf.to_container(scheduler_cfg, resolve=True))
        self.base_sampler_cfg = OmegaConf.create(OmegaConf.to_container(sampler_cfg, resolve=True))
        self.fourier_transform = fourier_transform
        self.cfg = config

    def _prepare_sampler(self, pl_module: LightningModule) -> Tuple[DiffusionSampler, DictConfig]:
        scheduler = instantiate(self.scheduler_cfg)
        if self.fourier_transform:
            scheduler.noise_scaling = True
        scheduler.set_noise_scaling(self.dataset_params["pred_len"])

        sampler_cfg = OmegaConf.create(
            OmegaConf.to_container(self.base_sampler_cfg, resolve=True)
        )
        if self.cfg.sampler_sample_batch_size is not None:
            sampler_cfg.sample_batch_size = self.cfg.sampler_sample_batch_size

        sampler = DiffusionSampler(
            score_model=pl_module,
            noise_scheduler=scheduler,
            context_len=self.dataset_params["context_len"],
            target_len=self.dataset_params["pred_len"],
            target_time_len=self.dataset_params["pred_len_time"],
            sample_batch_size=sampler_cfg.sample_batch_size,
            fourier_transform=self.fourier_transform,
        )
        return sampler, sampler_cfg

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not self.cfg:
            return

        try:
            sampler, sampler_cfg = self._prepare_sampler(pl_module)
        except Exception as exc:
            log.warning("EpochMetricsLogger failed to prepare sampler: %s", exc, exc_info=True)
            return

        val_loader = self.datamodule.val_dataloader()
        if val_loader is None:
            log.warning("EpochMetricsLogger: val_dataloader is None.")
            return

        samples_list = []
        samples_mc_list = [] if self.cfg.num_mc_samples and self.cfg.num_mc_samples > 1 else None
        truth_list = []
        context_time_list = []

        max_batches = max(1, int(self.cfg.max_batches))
        device = pl_module.device

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                if not isinstance(batch, DiffusionBatch):
                    batch = DiffusionBatch(**batch)

                batch_size = min(len(batch), sampler.sample_batch_size)
                context = batch.context[:batch_size].to(device)
                context_time = (
                    batch.context_time[:batch_size].to(device)
                    if batch.context_time is not None
                    else None
                )

                try:
                    if samples_mc_list is not None:
                        mc_samples = []
                        for _ in range(self.cfg.num_mc_samples):
                            generated, _, _ = sampler.sample(
                                context=context,
                                context_time=context_time,
                                context_mask=getattr(batch, "context_mask", None)[:batch_size].to(device)
                                if getattr(batch, "context_mask", None) is not None
                                else None,
                                num_diffusion_steps=self.cfg.sampler_num_diffusion_steps
                                or sampler_cfg.num_diffusion_steps,
                            )
                            mc_samples.append(generated.cpu())
                        mc_stack = torch.stack(mc_samples, dim=0)  # (S, B, T, A)
                        generated = mc_samples[0]
                        samples_mc_list.append(mc_stack)
                    else:
                        generated, _, _ = sampler.sample(
                            context=context,
                            context_time=context_time,
                            context_mask=getattr(batch, "context_mask", None)[:batch_size].to(device)
                            if getattr(batch, "context_mask", None) is not None
                            else None,
                            num_diffusion_steps=self.cfg.sampler_num_diffusion_steps
                            or sampler_cfg.num_diffusion_steps,
                        )
                except Exception as exc:
                    log.warning("EpochMetricsLogger sampling failed: %s", exc, exc_info=True)
                    continue

                target = (
                    batch.target_time[:batch_size]
                    if batch.target_time is not None
                    else batch.target[:batch_size]
                )

                samples_list.append(generated.cpu())
                truth_list.append(target.cpu())
                if context_time is not None:
                    context_time_list.append(context_time.cpu())

        if not samples_list:
            log.warning("EpochMetricsLogger: no samples collected for metrics.")
            return

        samples = torch.cat(samples_list, dim=0)
        truth = torch.cat(truth_list, dim=0)
        samples_mc = None
        if samples_mc_list:
            samples_mc = torch.cat(samples_mc_list, dim=1)  # (S, total_B, T, A)
        if self.cfg.max_windows > 0:
            limit = min(self.cfg.max_windows, samples.size(0))
            samples = samples[:limit]
            truth = truth[:limit]
            if samples_mc is not None:
                samples_mc = samples_mc[:, :limit]

        metrics = self._compute_metrics(samples, truth, samples_mc)
        for name, value in metrics.items():
            pl_module.log(
                f"{self.cfg.log_prefix}/{name}",
                float(value),
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        if self.cfg.save_mc_artifacts and samples_mc is not None and samples_mc.numel() > 0:
            # Persist MC samples and per-sample metrics for post-hoc aggregation.
            log_dir = None
            if trainer is not None and getattr(trainer, "logger", None) is not None:
                log_dir = getattr(trainer.logger, "log_dir", None)
            base = Path(log_dir) if log_dir else Path.cwd() / "mc_epoch_artifacts"
            base.mkdir(parents=True, exist_ok=True)

            # Save raw MC samples (already truncated to max_windows).
            torch.save(
                {"samples_mc": samples_mc, "truth": truth},
                base / f"epoch{trainer.current_epoch:04d}_mc.pt",
            )

            def _metrics_for_single(mc_slice: torch.Tensor, truth_slice: torch.Tensor) -> Dict[str, float]:
                dist = compute_distribution_metrics(mc_slice, truth_slice).as_dict()
                series = compute_series_metrics(truth_slice, mc_slice)
                matrix = compute_covariance_metrics(truth_slice, mc_slice)
                corr_struct = compute_corr_structure_metrics(
                    mc_slice.detach().cpu().numpy(), truth_slice.detach().cpu().numpy()
                )
                out = {
                    **{f"distribution_{k}": float(v) for k, v in dist.items()},
                    **{f"series_{k}": float(v) for k, v in series.items()},
                    **{f"matrix_{k}": float(v) for k, v in matrix.items()},
                }
                out.update({f"{k}": float(v) for k, v in corr_struct.items()})
                if self.cfg.compute_fourier and mc_slice.size(0) >= 2:
                    try:
                        fourier = compute_metric_collection(truth_slice, mc_slice)
                        out.update({f"fourier_{k}": float(torch.as_tensor(v).float().mean().item()) for k, v in fourier.items()})
                    except Exception as exc:  # pragma: no cover - diagnostic aid
                        log.warning("EpochMetricsLogger Fourier metrics failed for MC slice: %s", exc, exc_info=True)
                return out

            metrics_mc: Dict[str, List[float]] = {}
            for i in range(samples_mc.size(0)):
                m = _metrics_for_single(samples_mc[i], truth)
                for k, v in m.items():
                    metrics_mc.setdefault(k, []).append(float(v))

            out_path = base / f"epoch{trainer.current_epoch:04d}_mc_metrics.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({"epoch": int(trainer.current_epoch), "mc": int(samples_mc.size(0)), "metrics": metrics_mc}, f, ensure_ascii=False, indent=2)
            log.info("Saved MC artifacts to %s", out_path)

    def _compute_metrics(
        self,
        samples: torch.Tensor,
        truth: torch.Tensor,
        samples_mc: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        metrics = compute_time_aligned_metrics(samples, truth)
        time_metrics = {}
        for key, value in metrics.__dict__.items():
            tensor = torch.as_tensor(value)
            time_metrics[key] = tensor.float().mean().item()
        dist = compute_distribution_metrics(samples, truth).as_dict()
        series = compute_series_metrics(truth, samples)
        matrix = compute_covariance_metrics(truth, samples)
        aggregated: Dict[str, float] = {}
        aggregated.update({f"time_aligned_{k}": float(v) for k, v in time_metrics.items()})
        aggregated.update({f"distribution_{k}": float(v) for k, v in dist.items()})
        aggregated.update({f"series_{k}": float(v) for k, v in series.items()})
        aggregated.update({f"matrix_{k}": float(v) for k, v in matrix.items()})

        corr_struct = compute_corr_structure_metrics(
            samples.detach().cpu().numpy(), truth.detach().cpu().numpy()
        )
        aggregated.update({k: float(v) for k, v in corr_struct.items()})

        # --- additional probabilistic metrics (CRPS / ND / NRMSE) ---
        # mean prediction for ND/NRMSE
        pred_mean = samples.unsqueeze(0) if samples_mc is None else samples_mc.mean(dim=0, keepdim=True)
        pred_mean = pred_mean.mean(dim=0)  # (B, T, A)
        eps = 1e-8
        diff = pred_mean - truth
        nd = torch.sum(torch.abs(diff)) / (torch.sum(torch.abs(truth)) + eps)
        nrmse = torch.sqrt(torch.sum(diff ** 2) / (torch.sum(truth ** 2) + eps))

        aggregated["series_nd"] = float(nd.item())
        aggregated["series_nrmse"] = float(nrmse.item())

        truth_sum = truth.sum(dim=-1)
        pred_sum_mean = pred_mean.sum(dim=-1)
        diff_sum = pred_sum_mean - truth_sum
        nd_sum = torch.sum(torch.abs(diff_sum)) / (torch.sum(torch.abs(truth_sum)) + eps)
        nrmse_sum = torch.sqrt(torch.sum(diff_sum ** 2) / (torch.sum(truth_sum ** 2) + eps))
        aggregated["series_nd_sum"] = float(nd_sum.item())
        aggregated["series_nrmse_sum"] = float(nrmse_sum.item())

        if samples_mc is not None and samples_mc.size(0) >= 2:
            # empirical CRPS for each dimension
            truth_exp = truth.unsqueeze(0)
            term1 = torch.mean(torch.abs(samples_mc - truth_exp), dim=0)
            pairwise = torch.abs(samples_mc.unsqueeze(0) - samples_mc.unsqueeze(1)).mean(dim=(0, 1))
            crps = term1 - 0.5 * pairwise
            aggregated["series_crps"] = float(crps.mean().item())

            samples_sum = samples_mc.sum(dim=-1)  # (S, B, T)
            truth_sum_exp = truth_sum.unsqueeze(0)
            term1_sum = torch.mean(torch.abs(samples_sum - truth_sum_exp), dim=0)
            pairwise_sum = torch.abs(samples_sum.unsqueeze(0) - samples_sum.unsqueeze(1)).mean(dim=(0, 1))
            crps_sum = term1_sum - 0.5 * pairwise_sum
            aggregated["series_crps_sum"] = float(crps_sum.mean().item())

        if self.cfg.compute_fourier and samples.size(0) >= 2:
            try:
                fourier = compute_metric_collection(truth, samples)
                for key, value in fourier.items():
                    if isinstance(value, (list, tuple)):
                        if len(value) == 0:
                            continue
                        tensor = torch.as_tensor(value, dtype=torch.float32)
                        log_val = float(tensor.mean().item())
                    elif isinstance(value, (float, int)):
                        log_val = float(value)
                    else:
                        tensor = torch.as_tensor(value)
                        log_val = float(tensor.float().mean().item())
                    aggregated[f"fourier_{key}"] = log_val
            except Exception as exc:
                log.warning("EpochMetricsLogger Fourier metrics failed: %s", exc, exc_info=True)

        spec_samples = torch.fft.rfft(samples, dim=1)
        spec_truth = torch.fft.rfft(truth, dim=1)
        power_diff = (torch.abs(spec_samples) ** 2 - torch.abs(spec_truth) ** 2).abs().mean().item()
        aggregated["spectral_power_abs_error"] = float(power_diff)

        return aggregated
