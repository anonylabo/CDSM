import json
import logging
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

from cfdiff.callbacks import EpochMetricsConfig, EpochMetricsLogger, PeriodicCheckpoint
from cfdiff.dataloaders.conditional_gluonts import ConditionalGluonTSJsonDatamodule
from cfdiff.eval import (
    compute_metric_collection,
    compute_covariance_metrics,
    compute_distribution_metrics,
    compute_series_metrics,
    compute_time_aligned_metrics,
    compute_corr_structure_metrics,
)
from cfdiff.models import ScoreModel
from cfdiff.sampling import DiffusionSampler


log = logging.getLogger(__name__)


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        if "datamodule" not in cfg and "experiment" in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, cfg.experiment)
            cfg.pop("experiment", None)
            OmegaConf.set_struct(cfg, True)

        pl.seed_everything(cfg.random_seed)

        self.cfg = cfg

        self.run_mode = str(getattr(cfg, "run_mode", "train_sample")).lower()
        if self.run_mode not in {"train", "sample", "train_sample"}:
            raise ValueError(f"Unsupported run_mode: {self.run_mode}")
        requested_checkpoint = getattr(cfg, "checkpoint_path", None)
        requested_output_dir = getattr(cfg, "output_dir", None)

        self.datamodule: ConditionalGluonTSJsonDatamodule = instantiate(cfg.datamodule)
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")
        self.fourier_transform = bool(getattr(self.datamodule, "fourier_transform", False))

        self.dataset_params = self.datamodule.dataset_parameters
        params = self.dataset_params
        noise_scheduler = instantiate(cfg.scheduler)
        if self.fourier_transform:
            noise_scheduler.noise_scaling = True
        noise_scheduler.set_noise_scaling(params["pred_len"])

        model_partial: partial = instantiate(cfg.model)
        add_missing_mask = bool(getattr(self.datamodule, "add_missing_mask", False))
        self.model: ScoreModel = model_partial(
            n_channels=params["n_channels"],
            sequence_len=params["sequence_len"],
            noise_scheduler=noise_scheduler,
            pred_len=params["pred_len"],
            pred_len_time=params["pred_len_time"],
            context_len_time=params["context_len_time"],
            target_channels=params["target_channels"],
            add_missing_mask=add_missing_mask,
        )

        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.noise_scheduler = noise_scheduler
        self.sampler_cfg = cfg.sampler
        self._best_metric_cfgs: list[dict] = []
        self._best_metric_checkpoints: list[ModelCheckpoint] = []
        self._configure_epoch_metrics_callback()

        if requested_output_dir:
            candidate = Path(requested_output_dir)
            candidate.mkdir(parents=True, exist_ok=True)
        else:
            domain_folder = "fourier" if self.fourier_transform else "time"
            base_output_dir = Path.cwd() / "outputs" / domain_folder / "conditional"
            base_output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            candidate = base_output_dir / timestamp
            attempt = 1
            while candidate.exists():
                candidate = base_output_dir / f"{timestamp}_{attempt:02d}"
                attempt += 1

        self.output_dir = candidate
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "train_config.yaml"
        if self.run_mode in {"train", "train_sample"} or not config_path.exists():
            OmegaConf.save(cfg, config_path)

        log_save_dir = self.output_dir / "lightning_logs"
        log_save_dir.mkdir(parents=True, exist_ok=True)
        self.trainer.logger = CSVLogger(save_dir=str(self.output_dir), name="lightning_logs")
        self._attach_best_metric_checkpoint()
        self._attach_periodic_checkpoint_callback()

        self.checkpoint_path = Path(requested_checkpoint) if requested_checkpoint else None
        self._loaded_checkpoint = False
        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)
        elif self.run_mode == "sample":
            raise ValueError("checkpoint_path must be provided when run_mode='sample'")

    def _configure_epoch_metrics_callback(self) -> None:
        callbacks_cfg = getattr(self.cfg, "callbacks", None)
        if callbacks_cfg is None or "epoch_metrics" not in callbacks_cfg:
            return
        epoch_cfg = callbacks_cfg.epoch_metrics
        if not epoch_cfg or not bool(epoch_cfg.get("enabled", False)):
            return

        sampler_override = epoch_cfg.get("sampler", {})
        config = EpochMetricsConfig(
            max_batches=int(epoch_cfg.get("max_batches", 1) or 1),
            max_windows=int(epoch_cfg.get("max_windows", 32) or 32),
            log_prefix=str(epoch_cfg.get("log_prefix", "epoch")),
            compute_fourier=bool(epoch_cfg.get("compute_fourier", False)),
            num_mc_samples=int(epoch_cfg.get("num_mc_samples", 1) or 1),
            save_mc_artifacts=bool(epoch_cfg.get("save_mc_artifacts", False)),
            sampler_num_diffusion_steps=(
                int(sampler_override.get("num_diffusion_steps"))
                if sampler_override and sampler_override.get("num_diffusion_steps") is not None
                else None
            ),
            sampler_sample_batch_size=(
                int(sampler_override.get("sample_batch_size"))
                if sampler_override and sampler_override.get("sample_batch_size") is not None
                else None
            ),
        )

        callback = EpochMetricsLogger(
            datamodule=self.datamodule,
            dataset_params=self.dataset_params,
            scheduler_cfg=self.cfg.scheduler,
            sampler_cfg=self.cfg.sampler,
            fourier_transform=self.fourier_transform,
            config=config,
        )
        self.trainer.callbacks.append(callback)
        log.info(
            "Epoch metrics logging enabled: prefix=%s max_batches=%d max_windows=%d compute_fourier=%s num_mc_samples=%d",
            config.log_prefix,
            config.max_batches,
            config.max_windows,
            config.compute_fourier,
            config.num_mc_samples,
        )

        self._best_metric_cfgs = []

        def _normalize_entry(entry: dict) -> dict:
            metric_name = str(entry.get("metric"))
            if not metric_name:
                raise ValueError("Each best_metrics entry must specify a 'metric'.")
            mode = str(entry.get("mode", entry.get("best_metric_mode", "min"))).lower()
            filename = str(entry.get("filename", entry.get("best_metric_filename", f"best-{metric_name}")))
            save_last = bool(entry.get("save_last", entry.get("best_metric_save_last", True)))
            return {
                "metric": metric_name,
                "mode": mode,
                "filename": filename,
                "log_prefix": config.log_prefix,
                "save_last": save_last,
            }

        def _expand_best_metrics(raw_list) -> list[dict]:
            expanded: list[dict] = []
            if raw_list is None:
                return expanded

            if isinstance(raw_list, (str, dict, DictConfig)):
                iterable = [raw_list]
            else:
                iterable = [item for item in raw_list]

            for raw in iterable:
                if isinstance(raw, str):
                    stripped = raw.strip()
                    path_candidate: Path | None = None
                    if stripped.startswith("@"):  # Hydra-style reference
                        rel_path = stripped[1:]
                        path_candidate = Path(rel_path) if Path(rel_path).is_absolute() else Path(to_absolute_path(rel_path))
                    else:
                        candidate = Path(stripped)
                        if not candidate.is_absolute():
                            candidate = Path(to_absolute_path(stripped))
                        if candidate.exists():
                            path_candidate = candidate

                    if path_candidate is not None:
                        if not path_candidate.exists() or path_candidate.is_dir():
                            raise ValueError(f"best_metrics reference points to a directory: {path_candidate}")
                        loaded = OmegaConf.load(path_candidate)
                        if isinstance(loaded, dict) and "best_metrics" in loaded:
                            candidates = loaded["best_metrics"]
                        else:
                            candidates = loaded
                        if not OmegaConf.is_list(candidates):
                            raise ValueError(f"Unsupported best_metrics format in {path_candidate}")
                        for item in candidates:
                            expanded.append(_normalize_entry(OmegaConf.to_container(item, resolve=True)))
                        continue

                if isinstance(raw, (dict, DictConfig)):
                    raw_dict = OmegaConf.to_container(raw, resolve=True)
                else:
                    raw_dict = raw
                if not isinstance(raw_dict, dict):
                    raise ValueError(f"Unsupported best_metrics entry: {raw}")
                expanded.append(_normalize_entry(raw_dict))
            return expanded

        if epoch_cfg.get("best_metrics"):
            expanded = _expand_best_metrics(epoch_cfg.best_metrics)
            for cfg_entry in expanded:
                self._best_metric_cfgs.append(cfg_entry)
                log.info(
                    "Best-metric tracking requested: metric=%s mode=%s filename=%s",
                    cfg_entry["metric"],
                    cfg_entry["mode"],
                    cfg_entry["filename"],
                )
        else:
            best_metric = epoch_cfg.get("best_metric")
            if best_metric:
                cfg_entry = _normalize_entry(epoch_cfg)
                self._best_metric_cfgs.append(cfg_entry)
                log.info(
                    "Best-metric tracking requested: metric=%s mode=%s filename=%s",
                    cfg_entry["metric"],
                    cfg_entry["mode"],
                    cfg_entry["filename"],
                )

    def _attach_best_metric_checkpoint(self) -> None:
        if not self._best_metric_cfgs:
            return

        dirpath = self.output_dir / "checkpoints"
        dirpath.mkdir(parents=True, exist_ok=True)

        for cfg in self._best_metric_cfgs:
            log_prefix = cfg["log_prefix"]
            metric_name = cfg["metric"]
            monitor = metric_name if "/" in metric_name else f"{log_prefix}/{metric_name}"

            checkpoint_cb = ModelCheckpoint(
                dirpath=str(dirpath),
                filename=cfg["filename"],
                monitor=monitor,
                mode=cfg["mode"],
                save_last=cfg["save_last"],
            )
            self.trainer.callbacks.append(checkpoint_cb)
            self._best_metric_checkpoints.append(checkpoint_cb)
            log.info(
                "Best-metric checkpoint enabled: monitor=%s mode=%s filename=%s",
                monitor,
                cfg["mode"],
                cfg["filename"],
            )

    def _attach_periodic_checkpoint_callback(self) -> None:
        callbacks_cfg = getattr(self.cfg, "callbacks", None)
        pc_cfg = None
        if callbacks_cfg and "periodic_checkpoint" in callbacks_cfg:
            pc_cfg = callbacks_cfg.periodic_checkpoint
            if pc_cfg and not bool(pc_cfg.get("enabled", False)):
                return
        # default behavior: enable every 20 epochs when not configured
        try:
            every_n = (pc_cfg.get("every_n_epochs") if pc_cfg else None) or 20
            epochs = pc_cfg.get("epochs", None) if pc_cfg else None
            filename = str((pc_cfg.get("filename_pattern") if pc_cfg else None) or "epoch{epoch:03d}.ckpt")
            subdir = str((pc_cfg.get("dir_name") if pc_cfg else None) or "checkpoints_periodic")
            dirpath = self.output_dir / subdir
            callback = PeriodicCheckpoint(
                dirpath=dirpath,
                every_n_epochs=int(every_n) if every_n is not None else None,
                epochs=list(epochs) if epochs is not None else None,
                filename_pattern=filename,
            )
            self.trainer.callbacks.append(callback)
            log.info(
                "Periodic checkpoints enabled: dir=%s every_n=%s epochs=%s pattern=%s",
                dirpath,
                every_n,
                epochs,
                filename,
            )
        except Exception as exc:
            log.warning("Failed to attach periodic checkpoint callback: %s", exc, exc_info=True)

    def train(self) -> None:
        self.trainer.fit(self.model, datamodule=self.datamodule)
        ckpt_path = self.output_dir / "model.ckpt"
        self.trainer.save_checkpoint(str(ckpt_path))
        best_summaries: list[dict] = []
        for cfg, checkpoint_cb in zip(self._best_metric_cfgs, self._best_metric_checkpoints):
            best_path = None
            best_score = None
            best_epoch = None
            if checkpoint_cb.best_model_path:
                try:
                    candidate = Path(checkpoint_cb.best_model_path)
                    if candidate.is_file():
                        best_path = candidate
                        score = checkpoint_cb.best_model_score
                        if score is not None:
                            best_score = float(score.item())
                        checkpoint = torch.load(best_path, map_location="cpu")
                        best_epoch = int(checkpoint.get("epoch", -1))
                except Exception as exc:
                    log.warning(
                        "Failed to read best checkpoint for metric %s: %s",
                        cfg["metric"],
                        exc,
                        exc_info=True,
                    )
            best_summaries.append(
                {
                    "metric": cfg["metric"],
                    "mode": cfg["mode"],
                    "filename": cfg["filename"],
                    "best_model_path": str(best_path) if best_path else None,
                    "best_score": best_score,
                    "best_epoch": best_epoch,
                }
            )

        summary = {
            "output_dir": str(self.output_dir),
            "best_checkpoints": best_summaries,
        }
        with (self.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        for entry in best_summaries:
            if entry["best_model_path"]:
                log.info(
                    "Checkpoint for metric %s saved at %s (score=%s epoch=%s)",
                    entry["metric"],
                    entry["best_model_path"],
                    entry["best_score"],
                    entry["best_epoch"],
                )
            else:
                log.info(
                    "No checkpoint produced for metric %s (mode=%s filename=%s)",
                    entry["metric"],
                    entry["mode"],
                    entry["filename"],
                )

        preferred = next((s for s in best_summaries if s["best_model_path"]), None)
        best_path = Path(preferred["best_model_path"]) if preferred and preferred["best_model_path"] else None
        self.checkpoint_path = best_path if best_path else ckpt_path
        self._loaded_checkpoint = True
        self.model.eval()
        log.info("Saved checkpoint to %s", ckpt_path)
        if not preferred:
            log.warning(
                "No monitored metric produced a checkpoint; falling back to final model."
            )

    def _collect_stage_samples(
        self,
        loader: Optional[DataLoader],
        stage_name: str,
        all_samples: list[torch.Tensor],
        all_truth: list[torch.Tensor],
        all_context_time: list[torch.Tensor],
        all_pred_mean: list[torch.Tensor],
        all_pred_chol: list[torch.Tensor],
        all_samples_mc: Optional[list[torch.Tensor]] = None,
    ) -> int:
        if loader is None:
            log.info("No %s dataloader available; skipping sampling for this stage.", stage_name)
            return 0

        stage_sample_count = 0
        stage_batches = 0
        for batch_idx, batch in enumerate(loader):
            batch_len = len(batch)
            if batch_len == 0:
                continue
            effective_bs = min(batch_len, self.sampler_cfg.sample_batch_size)
            context = batch.context[:effective_bs]
            context_time = batch.context_time[:effective_bs] if batch.context_time is not None else None
            context_mask = getattr(batch, "context_mask", None)
            context_mask = context_mask[:effective_bs] if context_mask is not None else None
            sampler = DiffusionSampler(
                score_model=self.model,
                noise_scheduler=self.noise_scheduler,
                context_len=self.dataset_params["context_len"],
                target_len=self.dataset_params["pred_len"],
                target_time_len=self.dataset_params["pred_len_time"],
                sample_batch_size=effective_bs,
                fourier_transform=self.fourier_transform,
            )
            # Support multi-sample per window via sampler.mc_samples or sampler.num_mc_samples
            mc = int(getattr(self.sampler_cfg, "mc_samples", None) or getattr(self.sampler_cfg, "num_mc_samples", 1) or 1)
            if mc <= 1:
                samples_chunk, pred_mean_chunk, pred_chol_chunk = sampler.sample(
                    context=context,
                    context_time=context_time,
                    context_mask=context_mask,
                    num_diffusion_steps=self.sampler_cfg.num_diffusion_steps,
                )
            else:
                stacks: list[torch.Tensor] = []
                pred_mean_chunk = None
                pred_chol_chunk = None
                for _ in range(mc):
                    s_i, pm_i, pc_i = sampler.sample(
                        context=context,
                        context_time=context_time,
                        context_mask=context_mask,
                        num_diffusion_steps=self.sampler_cfg.num_diffusion_steps,
                    )
                    stacks.append(s_i)
                    # Heads depend only on context/time; keep the first
                    if pred_mean_chunk is None:
                        pred_mean_chunk = pm_i
                    if pred_chol_chunk is None:
                        pred_chol_chunk = pc_i
                mc_tensor = torch.stack(stacks, dim=0)  # (mc, bs, P, C)
                samples_chunk = mc_tensor.mean(dim=0)
                if all_samples_mc is not None:
                    all_samples_mc.append(mc_tensor.cpu())
            truth_chunk = (
                batch.target_time[:effective_bs]
                if batch.target_time is not None
                else batch.target[:effective_bs]
            )

            all_samples.append(samples_chunk.cpu())
            all_truth.append(truth_chunk.cpu())
            if context_time is not None:
                all_context_time.append(context_time.cpu())
            if pred_mean_chunk is not None:
                all_pred_mean.append(pred_mean_chunk.cpu())
            if pred_chol_chunk is not None:
                all_pred_chol.append(pred_chol_chunk.cpu())

            stage_sample_count += effective_bs
            stage_batches += 1
            log.info(
                "Processed %s batch %d: requested=%d, effective=%d, samples shape=%s",
                stage_name,
                batch_idx,
                self.sampler_cfg.sample_batch_size,
                effective_bs,
                tuple(samples_chunk.shape),
            )

        log.info(
            "Sampling finished for %s dataloader: batches=%d, windows=%d",
            stage_name,
            stage_batches,
            stage_sample_count,
        )
        return stage_sample_count

    def sample(self) -> None:
        if not self._loaded_checkpoint and self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)

        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()
        if val_loader is None and test_loader is None:
            log.warning("Both validation and test dataloaders are None; skipping sampling.")
            return

        sample_repeats = int(getattr(self.sampler_cfg, "sample_repeats", 1) or 1)
        base_seed = int(getattr(self.cfg, "random_seed", 0) or 0)
        last_history_dir: Path | None = None
        history_root = self.output_dir / "samples_history"
        batch_root: Path | None = None
        if sample_repeats > 1:
            batch_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            batch_root = history_root / f"batch-{batch_stamp}"
            batch_root.mkdir(parents=True, exist_ok=True)
            log.info("Sampling repeats=%d; grouping outputs under %s", sample_repeats, batch_root)

        for repeat_idx in range(sample_repeats):
            if sample_repeats > 1:
                pl.seed_everything(base_seed + repeat_idx)
                log.info("Sampling repeat %d/%d with seed=%d", repeat_idx + 1, sample_repeats, base_seed + repeat_idx)

            all_samples: list[torch.Tensor] = []
            all_truth: list[torch.Tensor] = []
            all_context_time: list[torch.Tensor] = []
            all_pred_mean: list[torch.Tensor] = []
            all_pred_chol: list[torch.Tensor] = []
            all_samples_mc: list[torch.Tensor] = []

            stage_counts: Dict[str, int] = {}
            stage_counts["val"] = self._collect_stage_samples(
                val_loader, "val", all_samples, all_truth, all_context_time, all_pred_mean, all_pred_chol, all_samples_mc
            )
            stage_counts["test"] = self._collect_stage_samples(
                test_loader, "test", all_samples, all_truth, all_context_time, all_pred_mean, all_pred_chol, all_samples_mc
            )

            if not all_samples:
                log.warning("Sampling yielded no usable batches from either stage; skipping save.")
                continue

            samples = torch.cat(all_samples, dim=0)
            truth_tail = torch.cat(all_truth, dim=0)
            context_tail = torch.cat(all_context_time, dim=0) if all_context_time else None
            pred_mean = torch.cat(all_pred_mean, dim=0) if all_pred_mean else None
            pred_chol = torch.cat(all_pred_chol, dim=0) if all_pred_chol else None

            log.info(
                "Sampling summary: repeat=%d val windows=%d, test windows=%d, total=%d",
                repeat_idx + 1,
                stage_counts.get("val", 0),
                stage_counts.get("test", 0),
                samples.size(0),
            )

            log.info(
                "Computing metrics: time_aligned on batch=%d, pred_len=%d, channels=%d",
                samples.size(0),
                samples.size(1),
                samples.size(2),
            )
            metrics = compute_time_aligned_metrics(samples, truth_tail)
            samples_payload = {
                "samples": samples,
                "truth": truth_tail,
                "context": context_tail,
                "pred_mean": pred_mean if pred_mean is not None else None,
                "pred_chol": pred_chol if pred_chol is not None else None,
                "fourier_transform": self.fourier_transform,
                "window_stage_counts": stage_counts,
            }

            mc_tensor_all = None
            # Prefer ensemble for distribution metrics when MC>1
            mc = int(getattr(self.sampler_cfg, "mc_samples", None) or getattr(self.sampler_cfg, "num_mc_samples", 1) or 1)
            save_mc_artifacts = bool(getattr(self.sampler_cfg, "save_mc_artifacts", False))
            if mc > 1 and all_samples_mc:
                mc_tensor_all = torch.cat(all_samples_mc, dim=1)  # (mc, total, P, C)
                gen_for_dist = mc_tensor_all.reshape(mc_tensor_all.size(0) * mc_tensor_all.size(1), samples.size(1), samples.size(2))
                ref_for_dist = truth_tail.repeat(mc, 1, 1)
                dist_metrics = compute_distribution_metrics(gen_for_dist, ref_for_dist).as_dict()
            else:
                dist_metrics = compute_distribution_metrics(samples, truth_tail).as_dict()
            log.info("Distribution metrics computed: keys=%s", list(dist_metrics.keys()))
            series_metrics = compute_series_metrics(truth_tail, samples)
            log.info("Series metrics computed: keys=%s", list(series_metrics.keys()))
            matrix_metrics = compute_covariance_metrics(truth_tail, samples)
            log.info("Covariance metrics computed: keys=%s", list(matrix_metrics.keys()))
            corr_struct_metrics = compute_corr_structure_metrics(
                samples.detach().cpu().numpy(),
                truth_tail.detach().cpu().numpy(),
            )
            log.info("Correlation-structure metrics computed: keys=%s", list(corr_struct_metrics.keys()))
            # ND/NRMSE style metrics (align with epoch metrics callback)
            pred_mean_for_nd = samples if mc_tensor_all is None else mc_tensor_all.mean(dim=0)
            eps = 1e-8
            diff = pred_mean_for_nd - truth_tail
            nd = torch.sum(torch.abs(diff)) / (torch.sum(torch.abs(truth_tail)) + eps)
            nrmse = torch.sqrt(torch.sum(diff ** 2) / (torch.sum(truth_tail ** 2) + eps))
            truth_sum = truth_tail.sum(dim=-1)
            pred_sum_mean = pred_mean_for_nd.sum(dim=-1)
            diff_sum = pred_sum_mean - truth_sum
            nd_sum = torch.sum(torch.abs(diff_sum)) / (torch.sum(torch.abs(truth_sum)) + eps)
            nrmse_sum = torch.sqrt(torch.sum(diff_sum ** 2) / (torch.sum(truth_sum ** 2) + eps))

            disable_fourier = os.environ.get("CFDIFF_DISABLE_FOURIER", "0") == "1"
            if disable_fourier or samples.size(0) < 2:
                if disable_fourier:
                    log.info("Skipping Fourier metrics because CFDIFF_DISABLE_FOURIER is set.")
                else:
                    log.info("Skipping Fourier metrics: requires at least two samples (got %d).", samples.size(0))
                fourier_metrics = {}
            else:
                try:
                    fourier_metrics = compute_metric_collection(truth_tail, samples)
                    log.info(
                        "Fourier metrics computed (directions=%s): keys=%s",
                        os.environ.get("CFDIFF_SW_DIRECTIONS", "128"),
                        list(fourier_metrics.keys()),
                    )
                except Exception as exc:  # pragma: no cover - debug logging aid
                    log.error("Fourier metric computation failed: %s", exc, exc_info=True)
                    raise

            spec_samples = torch.fft.rfft(samples, dim=1)
            spec_truth = torch.fft.rfft(truth_tail, dim=1)
            power_diff = (torch.abs(spec_samples) ** 2 - torch.abs(spec_truth) ** 2).abs().mean().item()
            # Optional CRPS from MC ensemble
            crps_mc = None
            if mc > 1 and all_samples_mc:
                mc_tensor = mc_tensor_all if mc_tensor_all is not None else torch.cat(all_samples_mc, dim=1)  # (mc, total, P, C)
                truth_exp = truth_tail.unsqueeze(0).expand_as(mc_tensor)
                term1 = (mc_tensor - truth_exp).abs().mean(dim=0)
                pairwise = (mc_tensor.unsqueeze(0) - mc_tensor.unsqueeze(1)).abs().mean(dim=(0, 1))
                crps_mc = float((term1 - 0.5 * pairwise).mean().item())

            extra_metrics = {
                **{f"distribution_{k}": v for k, v in dist_metrics.items()},
                **{f"series_{k}": v for k, v in series_metrics.items()},
                **{f"matrix_{k}": v for k, v in matrix_metrics.items()},
                **{f"fourier_{k}": v for k, v in fourier_metrics.items()},
                "spectral_power_abs_error": power_diff,
            }
            extra_metrics["series_nd"] = float(nd.item())
            extra_metrics["series_nrmse"] = float(nrmse.item())
            extra_metrics["series_nd_sum"] = float(nd_sum.item())
            extra_metrics["series_nrmse_sum"] = float(nrmse_sum.item())
            extra_metrics.update({f"{k}": v for k, v in corr_struct_metrics.items()})
            if crps_mc is not None:
                extra_metrics["series_crps_mc"] = crps_mc
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # Derive a human‑readable tag from the checkpoint filename (e.g., e020 or best-metric)
            cp = self.checkpoint_path if hasattr(self, "checkpoint_path") else None
            tag: str | None = None
            if cp is None:
                cp_cfg = getattr(self.cfg, "checkpoint_path", None)
                cp = Path(cp_cfg) if cp_cfg else None
            if isinstance(cp, Path):
                name = cp.name
                if name.startswith("epoch") and name.endswith(".ckpt"):
                    # epochXYZ.ckpt -> eXYZ
                    stem = name.removeprefix("epoch").removesuffix(".ckpt")
                    tag = f"e{stem}"
                elif name.endswith(".ckpt"):
                    tag = name.removesuffix(".ckpt")
            parent_dir = batch_root if batch_root is not None else history_root
            suffix = f"-r{repeat_idx+1:02d}" if sample_repeats > 1 else ""
            history_dir = parent_dir / (f"{timestamp}{suffix}-{tag}" if tag else f"{timestamp}{suffix}")
            history_dir.mkdir(parents=True, exist_ok=True)
            last_history_dir = history_dir
            timestamped_samples = history_dir / "samples.pt"
            timestamped_metrics = history_dir / "metrics.pt"
            timestamped_extra = history_dir / "metrics_extra.json"

            torch.save(samples_payload, timestamped_samples)
            torch.save(
                {
                    "time_aligned": metrics,
                    "distribution": dist_metrics,
                    "series": series_metrics,
                    "matrix": matrix_metrics,
                },
                timestamped_metrics,
            )
            with timestamped_extra.open("w", encoding="utf-8") as f:
                json.dump(extra_metrics, f, ensure_ascii=False, indent=2)
            log.info("Saved sampling artifacts to %s (timestamp %s)", history_dir, timestamp)

            if mc > 1 and save_mc_artifacts and mc_tensor_all is not None:
                torch.save({"samples_mc": mc_tensor_all, "truth": truth_tail}, history_dir / "samples_mc.pt")

                def _metrics_single(mc_slice: torch.Tensor, truth_slice: torch.Tensor) -> dict:
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
                    if not disable_fourier and mc_slice.size(0) >= 2:
                        try:
                            fourier = compute_metric_collection(truth_slice, mc_slice)
                            out.update(
                                {
                                    f"fourier_{k}": float(torch.as_tensor(v).float().mean().item())
                                    for k, v in fourier.items()
                                }
                            )
                        except Exception:
                            pass
                    return out

                metrics_mc: dict[str, list[float]] = {}
                for i in range(mc_tensor_all.size(0)):
                    m = _metrics_single(mc_tensor_all[i], truth_tail)
                    for k, v in m.items():
                        metrics_mc.setdefault(k, []).append(float(v))
                with (history_dir / "metrics_mc.json").open("w", encoding="utf-8") as f:
                    json.dump({"mc": mc_tensor_all.size(0), "metrics": metrics_mc}, f, ensure_ascii=False, indent=2)

            # Stage-wise metrics (val/test separated)
            def _stage_metrics(offset: int, count: int) -> dict:
                if count <= 0:
                    return {}
                s_slice = samples[offset : offset + count]
                t_slice = truth_tail[offset : offset + count]
                ctx_slice = context_tail[offset : offset + count] if context_tail is not None else None
                if mc > 1 and mc_tensor_all is not None:
                    mc_slice = mc_tensor_all[:, offset : offset + count]
                    gen_for_dist = mc_slice.reshape(mc_slice.size(0) * mc_slice.size(1), s_slice.size(1), s_slice.size(2))
                    ref_for_dist = t_slice.repeat(mc, 1, 1)
                    dist = compute_distribution_metrics(gen_for_dist, ref_for_dist).as_dict()
                else:
                    dist = compute_distribution_metrics(s_slice, t_slice).as_dict()
                series = compute_series_metrics(t_slice, s_slice)
                matrix = compute_covariance_metrics(t_slice, s_slice)
                corr_struct = compute_corr_structure_metrics(s_slice.detach().cpu().numpy(), t_slice.detach().cpu().numpy())
                if disable_fourier or s_slice.size(0) < 2:
                    fourier = {}
                else:
                    try:
                        fourier = compute_metric_collection(t_slice, s_slice)
                    except Exception:
                        fourier = {}
                crps_val = None
                if mc > 1 and mc_tensor_all is not None:
                    mc_slice = mc_tensor_all[:, offset : offset + count]
                    truth_exp = t_slice.unsqueeze(0).expand_as(mc_slice)
                    term1 = (mc_slice - truth_exp).abs().mean(dim=0)
                    pairwise = (mc_slice.unsqueeze(0) - mc_slice.unsqueeze(1)).abs().mean(dim=(0, 1))
                    crps_val = float((term1 - 0.5 * pairwise).mean().item())
                # ND/NRMSE style metrics per stage
                pred_mean_nd = s_slice if mc_tensor_all is None else mc_tensor_all.mean(dim=0)[offset : offset + count]
                eps = 1e-8
                diff = pred_mean_nd - t_slice
                nd = torch.sum(torch.abs(diff)) / (torch.sum(torch.abs(t_slice)) + eps)
                nrmse = torch.sqrt(torch.sum(diff ** 2) / (torch.sum(t_slice ** 2) + eps))
                truth_sum = t_slice.sum(dim=-1)
                pred_sum_mean = pred_mean_nd.sum(dim=-1)
                diff_sum = pred_sum_mean - truth_sum
                nd_sum = torch.sum(torch.abs(diff_sum)) / (torch.sum(torch.abs(truth_sum)) + eps)
                nrmse_sum = torch.sqrt(torch.sum(diff_sum ** 2) / (torch.sum(truth_sum ** 2) + eps))

                out = {
                    **{f"distribution_{k}": v for k, v in dist.items()},
                    **{f"series_{k}": v for k, v in series.items()},
                    **{f"matrix_{k}": v for k, v in matrix.items()},
                    **{f"fourier_{k}": v for k, v in fourier.items()},
                }
                out["series_nd"] = float(nd.item())
                out["series_nrmse"] = float(nrmse.item())
                out["series_nd_sum"] = float(nd_sum.item())
                out["series_nrmse_sum"] = float(nrmse_sum.item())
                out.update({f"{k}": v for k, v in corr_struct.items()})
                if crps_val is not None:
                    out["series_crps_mc"] = crps_val
                    out.setdefault("series_crps", crps_val)
                return out

            stage_metrics: dict = {}
            stage_mc_metrics: dict[str, dict[str, list[float]]] = {}
            offset = 0
            for stage_name in ("val", "test"):
                n = int(stage_counts.get(stage_name, 0) or 0)
                if n > 0:
                    stage_metrics[stage_name] = _stage_metrics(offset, n)
                    if mc > 1 and save_mc_artifacts and mc_tensor_all is not None:
                        metrics_mc_stage: dict[str, list[float]] = {}
                        truth_stage = truth_tail[offset : offset + n]
                        for i in range(mc_tensor_all.size(0)):
                            m = _metrics_single(mc_tensor_all[i, offset : offset + n], truth_stage)
                            for k, v in m.items():
                                metrics_mc_stage.setdefault(k, []).append(float(v))
                        stage_mc_metrics[stage_name] = metrics_mc_stage
                offset += n

            for stage_name, metrics_dict in stage_metrics.items():
                path = history_dir / f"metrics_extra_{stage_name}.json"
                with path.open("w", encoding="utf-8") as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
                log.info("Saved %s metrics to %s", stage_name, path)
            if stage_mc_metrics:
                for stage_name, metrics_dict in stage_mc_metrics.items():
                    path = history_dir / f"metrics_mc_{stage_name}.json"
                    with path.open("w", encoding="utf-8") as f:
                        json.dump({"mc": mc_tensor_all.size(0), "metrics": metrics_dict}, f, ensure_ascii=False, indent=2)
                    log.info("Saved MC metrics for %s to %s", stage_name, path)

        if self.run_mode in {"sample", "train_sample"}:
            sample_config_path = self.output_dir / "sample_config.yaml"
            cfg_to_save = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            OmegaConf.save(cfg_to_save, sample_config_path)
            # Also persist a copy of the sampling config alongside the artifacts for traceability
            if last_history_dir:
                OmegaConf.save(cfg_to_save, last_history_dir / "sample_config.yaml")
            log.info("Saved sample config to %s", sample_config_path)

    def _load_checkpoint(self, path: Path) -> None:
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(path, map_location=target_device)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning("Missing keys when loading checkpoint %s: %s", path, missing)
        if unexpected:
            log.warning("Unexpected keys when loading checkpoint %s: %s", path, unexpected)
        self._loaded_checkpoint = True
        self.model.eval()
        log.info("Loaded checkpoint from %s", path)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg)
    if runner.run_mode in {"train", "train_sample"}:
        runner.train()
    if runner.run_mode in {"sample", "train_sample"}:
        runner.sample()


if __name__ == "__main__":
    main()
