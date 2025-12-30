from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

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
from cfdiff.utils.fourier import tensor_ifft_realimag
from fdiff.sampling import DiffusionSampler
from fdiff.schedulers import SDE
from fdiff.utils.dataclasses import DiffusableBatch

log = logging.getLogger(__name__)


@dataclass
class EpochMetricsConfig:
    enabled: bool = False
    max_batches: int = 1
    max_windows: int = 32
    log_prefix: str = "epoch"
    compute_fourier: bool = False
    sampler_num_diffusion_steps: Optional[int] = None
    sampler_sample_batch_size: Optional[int] = None
    num_mc_samples: int = 1


class EpochMetricsLogger(Callback):
    """Log unconditional metrics at validation epoch end."""

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

    def _prepare_sampler(self, pl_module: LightningModule) -> tuple[DiffusionSampler, DictConfig]:
        scheduler: SDE = instantiate(self.scheduler_cfg)
        scheduler.set_noise_scaling(self.dataset_params["sequence_len"])

        sampler_cfg = OmegaConf.create(OmegaConf.to_container(self.base_sampler_cfg, resolve=True))
        if self.cfg.sampler_sample_batch_size is not None:
            sampler_cfg.sample_batch_size = int(self.cfg.sampler_sample_batch_size)

        sampler = DiffusionSampler(
            score_model=pl_module,
            noise_scheduler=scheduler,
            sample_batch_size=int(sampler_cfg.sample_batch_size),
        )
        return sampler, sampler_cfg

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not self.cfg or not self.cfg.enabled:
            return

        try:
            sampler, sampler_cfg = self._prepare_sampler(pl_module)
        except Exception as exc:
            log.warning("EpochMetricsLogger: unable to build sampler: %s", exc, exc_info=True)
            return

        val_loader = self.datamodule.val_dataloader()
        if val_loader is None:
            log.warning("EpochMetricsLogger: val_dataloader is None.")
            return

        samples_list: list[torch.Tensor] = []
        samples_mc_list: list[torch.Tensor] | None = [] if self.cfg.num_mc_samples and self.cfg.num_mc_samples > 1 else None
        truth_list: list[torch.Tensor] = []

        max_batches = max(1, int(self.cfg.max_batches))

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                if not isinstance(batch, DiffusableBatch):
                    batch = DiffusableBatch(**batch)

                batch_len = len(batch)
                if batch_len == 0:
                    continue
                request = min(batch_len, sampler.sample_batch_size)
                try:
                    if samples_mc_list is not None:
                        mc_samples = []
                        for _ in range(int(self.cfg.num_mc_samples)):
                            generated = sampler.sample(
                                num_samples=request,
                                num_diffusion_steps=self.cfg.sampler_num_diffusion_steps
                                or sampler_cfg.num_diffusion_steps,
                            )
                            if self.fourier_transform:
                                generated = tensor_ifft_realimag(
                                    generated,
                                    signal_len=self.dataset_params["pred_len_time"],
                                    dim=1,
                                )
                            mc_samples.append(generated.cpu())
                        mc_stack = torch.stack(mc_samples, dim=0)
                        generated = mc_samples[0]
                        samples_mc_list.append(mc_stack)
                    else:
                        generated = sampler.sample(
                            num_samples=request,
                            num_diffusion_steps=self.cfg.sampler_num_diffusion_steps
                            or sampler_cfg.num_diffusion_steps,
                        )
                        if self.fourier_transform:
                            generated = tensor_ifft_realimag(
                                generated,
                                signal_len=self.dataset_params["pred_len_time"],
                                dim=1,
                            )
                except Exception as exc:
                    log.warning("EpochMetricsLogger: sampling failed on batch %d: %s", batch_idx, exc, exc_info=True)
                    continue
                target = (
                    batch.target_time[:request]
                    if batch.target_time is not None
                    else batch.X[:request]
                )

                samples_list.append(generated.cpu())
                truth_list.append(target.cpu())

        if not samples_list:
            log.warning("EpochMetricsLogger: collected 0 samples.")
            return

        samples = torch.cat(samples_list, dim=0)
        truth = torch.cat(truth_list, dim=0)
        samples_mc = None
        if samples_mc_list:
            samples_mc = torch.cat(samples_mc_list, dim=1)

        if self.cfg.max_windows > 0:
            cap = min(int(self.cfg.max_windows), samples.size(0))
            samples = samples[:cap]
            truth = truth[:cap]
            if samples_mc is not None:
                samples_mc = samples_mc[:, :cap]

        metrics = self._collect_metrics(samples, truth, samples_mc)
        for key, value in metrics.items():
            pl_module.log(
                f"{self.cfg.log_prefix}/{key}",
                float(value),
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

    def _collect_metrics(
        self,
        samples: torch.Tensor,
        truth: torch.Tensor,
        samples_mc: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        time_metrics = compute_time_aligned_metrics(samples, truth)
        aggregated: Dict[str, float] = {
            f"time_aligned_{k}": float(torch.as_tensor(v).float().mean().item())
            for k, v in time_metrics.__dict__.items()
        }
        dist = compute_distribution_metrics(samples, truth).as_dict()
        aggregated.update({f"distribution_{k}": float(v) for k, v in dist.items()})
        series = compute_series_metrics(truth, samples)
        aggregated.update({f"series_{k}": float(v) for k, v in series.items()})
        matrix = compute_covariance_metrics(truth, samples)
        aggregated.update({f"matrix_{k}": float(v) for k, v in matrix.items()})

        corr_struct = compute_corr_structure_metrics(samples.cpu().numpy(), truth.cpu().numpy())
        aggregated.update({k: float(v) for k, v in corr_struct.items()})

        # Additional probabilistic metrics (ND/NRMSE/CRPS)
        eps = 1e-8
        pred_mean = samples.unsqueeze(0) if samples_mc is None else samples_mc.mean(dim=0, keepdim=True)
        pred_mean = pred_mean.mean(dim=0)
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
            truth_exp = truth.unsqueeze(0)
            term1 = torch.mean(torch.abs(samples_mc - truth_exp), dim=0)
            pairwise = torch.abs(samples_mc.unsqueeze(0) - samples_mc.unsqueeze(1)).mean(dim=(0, 1))
            crps = term1 - 0.5 * pairwise
            aggregated["series_crps"] = float(crps.mean().item())

            samples_sum = samples_mc.sum(dim=-1)
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
                        aggregated[f"fourier_{key}"] = float(tensor.mean().item())
                    elif isinstance(value, (float, int)):
                        aggregated[f"fourier_{key}"] = float(value)
                    else:
                        tensor = torch.as_tensor(value)
                        aggregated[f"fourier_{key}"] = float(tensor.float().mean().item())
            except Exception as exc:
                log.warning("EpochMetricsLogger: Fourier metrics failed: %s", exc, exc_info=True)

        spec_samples = torch.fft.rfft(samples, dim=1)
        spec_truth = torch.fft.rfft(truth, dim=1)
        power_diff = (torch.abs(spec_samples) ** 2 - torch.abs(spec_truth) ** 2).abs().mean().item()
        aggregated["spectral_power_abs_error"] = float(power_diff)
        return aggregated
