import json
import logging
import math
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

from cfdiff.eval import (
    compute_covariance_metrics,
    compute_distribution_metrics,
    compute_metric_collection,
    compute_series_metrics,
    compute_time_aligned_metrics,
    compute_corr_structure_metrics,
)
from cfdiff.utils.fourier import tensor_ifft_realimag
from fdiff.callbacks import EpochMetricsConfig, EpochMetricsLogger, PeriodicCheckpoint
from fdiff.dataloaders import UnconditionalGluonTSJsonDatamodule
from fdiff.models import ScoreModel
from fdiff.sampling import DiffusionSampler
from fdiff.schedulers import SDE
from fdiff.utils.dataclasses import DiffusableBatch

log = logging.getLogger(__name__)


class TrainingRunner:
    def __init__(self, cfg: DictConfig) -> None:
        if "datamodule" not in cfg and "experiment" in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, cfg.experiment)
            cfg.pop("experiment", None)
            OmegaConf.set_struct(cfg, True)

        pl.seed_everything(int(cfg.random_seed))
        self.cfg = cfg
        self.run_mode = str(getattr(cfg, "run_mode", "train_sample")).lower()
        if self.run_mode not in {"train", "sample", "train_sample"}:
            raise ValueError(f"Unsupported run_mode: {self.run_mode}")

        requested_checkpoint = getattr(cfg, "checkpoint_path", None)
        requested_output_dir = getattr(cfg, "output_dir", None)

        self.datamodule: UnconditionalGluonTSJsonDatamodule = instantiate(cfg.datamodule)
        self.datamodule.prepare_data()
        self.datamodule.setup("fit")

        self.fourier_transform = bool(getattr(self.datamodule, "fourier_transform", False))
        self.dataset_params = self.datamodule.dataset_parameters
        noise_scheduler: SDE = instantiate(cfg.scheduler)
        noise_scheduler.set_noise_scaling(self.dataset_params["sequence_len"])

        model_partial: partial = instantiate(cfg.model)
        sequence_len = self.dataset_params["sequence_len"]
        add_missing_mask = bool(getattr(self.datamodule, "add_missing_mask", False))
        self.model: ScoreModel = model_partial(
            n_channels=self.dataset_params["n_channels"],
            max_len=sequence_len,
            noise_scheduler=noise_scheduler,
            add_missing_mask=add_missing_mask,
        )

        self.noise_scheduler = noise_scheduler
        self.trainer: pl.Trainer = instantiate(cfg.trainer)
        self.sampler_cfg = cfg.sampler
        self._best_metric_cfgs: list[dict] = []
        self._best_metric_checkpoints: list[ModelCheckpoint] = []
        self._configure_epoch_metrics_callback()

        if requested_output_dir:
            candidate = Path(requested_output_dir)
        else:
            domain_folder = "fourier" if bool(getattr(self.datamodule, "fourier_transform", False)) else "time"
            base_output = Path.cwd() / "outputs" / domain_folder / "unconditional"
            base_output.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            candidate = base_output / timestamp
            suffix = 1
            while candidate.exists():
                candidate = base_output / f"{timestamp}_{suffix:02d}"
                suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        self.output_dir = candidate

        config_path = self.output_dir / "train_config.yaml"
        if self.run_mode in {"train", "train_sample"} or not config_path.exists():
            OmegaConf.save(cfg, config_path)

        log_dir = self.output_dir / "lightning_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
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
            enabled=True,
            max_batches=int(epoch_cfg.get("max_batches", 1) or 1),
            max_windows=int(epoch_cfg.get("max_windows", 32) or 32),
            log_prefix=str(epoch_cfg.get("log_prefix", "epoch")),
            compute_fourier=bool(epoch_cfg.get("compute_fourier", False)),
            num_mc_samples=int(epoch_cfg.get("num_mc_samples", 1) or 1),
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

        best_metrics_cfg = epoch_cfg.get("best_metrics")
        if best_metrics_cfg:
            expanded = _expand_best_metrics(best_metrics_cfg)
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

        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for cfg in self._best_metric_cfgs:
            monitor = cfg["metric"] if "/" in cfg["metric"] else f"{cfg['log_prefix']}/{cfg['metric']}"
            checkpoint_cb = ModelCheckpoint(
                dirpath=str(ckpt_dir),
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

        with (self.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"output_dir": str(self.output_dir), "best_checkpoints": best_summaries},
                f,
                ensure_ascii=False,
                indent=2,
            )

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
        sampler: DiffusionSampler,
        all_samples: list[torch.Tensor],
        all_samples_mc: list[torch.Tensor],
        all_samples_fourier: list[torch.Tensor],
        all_truth: list[torch.Tensor],
        mc: int,
    ) -> int:
        if loader is None:
            log.info("No %s dataloader available; skipping sampling.", stage_name)
            return 0

        stage_total = 0
        for batch_idx, batch in enumerate(loader):
            assert isinstance(batch, DiffusableBatch)
            batch_len = len(batch)
            if batch_len == 0:
                continue
            effective_bs = min(batch_len, self.sampler_cfg.sample_batch_size)
            num_to_generate = effective_bs * mc
            generated = sampler.sample(
                num_samples=num_to_generate,
                num_diffusion_steps=self.sampler_cfg.num_diffusion_steps,
            )
            generated_mean = generated
            generated_mc = None
            if mc > 1:
                generated_mc = generated.view(mc, effective_bs, *generated.shape[1:])  # (mc, bs, T, C)
                generated_mean = generated_mc.mean(dim=0)  # (bs, T, C)
            samples_chunk = generated_mean
            fourier_chunk = None
            if self.fourier_transform:
                if mc > 1 and generated_mc is not None:
                    generated_time_mc = tensor_ifft_realimag(
                        generated_mc,
                        signal_len=self.dataset_params["pred_len_time"],
                        dim=2,
                    )  # (mc, bs, T_time, C_time)
                    all_samples_mc.append(generated_time_mc.cpu())
                    samples_chunk = generated_time_mc.mean(dim=0)  # already in time domain
                    fourier_chunk = generated_time_mc
                else:
                    samples_chunk = tensor_ifft_realimag(
                        generated_mean,
                        signal_len=self.dataset_params["pred_len_time"],
                        dim=1,
                    )
                    fourier_chunk = generated_mean
            else:
                if mc > 1 and generated_mc is not None:
                    all_samples_mc.append(generated_mc.cpu())
            truth = batch.target_time[:effective_bs] if batch.target_time is not None else batch.X[:effective_bs]

            all_samples.append(samples_chunk.cpu())
            if fourier_chunk is not None:
                all_samples_fourier.append(fourier_chunk.detach().cpu())
            all_truth.append(truth.cpu())
            stage_total += effective_bs
            log.info("Processed %s batch %d: generated shape=%s", stage_name, batch_idx, tuple(samples_chunk.shape))

        log.info("Completed sampling for %s dataloader: windows=%d", stage_name, stage_total)
        return stage_total

    def sample(self) -> None:
        if not self._loaded_checkpoint and self.checkpoint_path is not None:
            self._load_checkpoint(self.checkpoint_path)

        val_loader = self.datamodule.val_dataloader()
        test_loader = self.datamodule.test_dataloader()
        if val_loader is None and test_loader is None:
            log.warning("Both validation and test dataloaders are None; skipping sampling.")
            return

        sample_repeats = int(getattr(self.sampler_cfg, "sample_repeats", 1) or 1)
        save_mc_artifacts = bool(getattr(self.sampler_cfg, "save_mc_artifacts", False))
        base_seed = int(getattr(self.cfg, "random_seed", 0) or 0)

        sampler = DiffusionSampler(
            score_model=self.model,
            noise_scheduler=self.noise_scheduler,
            sample_batch_size=self.sampler_cfg.sample_batch_size,
        )

        cp = self.checkpoint_path if hasattr(self, "checkpoint_path") else None
        if cp is None:
            cp_cfg = getattr(self.cfg, "checkpoint_path", None)
            cp = Path(cp_cfg) if cp_cfg else None
        tag: str | None = None
        if isinstance(cp, Path):
            name = cp.name
            if name.startswith("epoch") and name.endswith(".ckpt"):
                stem = name.removeprefix("epoch").removesuffix(".ckpt")
                tag = f"e{stem}"
            elif name.endswith(".ckpt"):
                tag = name.removesuffix(".ckpt")

        history_root = self.output_dir / "samples_history"
        batch_root: Optional[Path] = None
        last_history_dir: Optional[Path] = None
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
            all_samples_fourier: list[torch.Tensor] = []
            all_truth: list[torch.Tensor] = []
            all_samples_mc: list[torch.Tensor] = []

            stage_counts: Dict[str, int] = {}
            mc = int(getattr(self.sampler_cfg, "num_mc_samples", 1) or getattr(self.sampler_cfg, "mc_samples", 1) or 1)
            stage_counts["val"] = self._collect_stage_samples(
                val_loader, "val", sampler, all_samples, all_samples_mc, all_samples_fourier, all_truth, mc
            )
            stage_counts["test"] = self._collect_stage_samples(
                test_loader, "test", sampler, all_samples, all_samples_mc, all_samples_fourier, all_truth, mc
            )

            if not all_samples:
                log.warning("No samples collected; skipping metrics for repeat %d.", repeat_idx + 1)
                continue

            samples = torch.cat(all_samples, dim=0)
            truth = torch.cat(all_truth, dim=0)
            mc_tensor = None
            if mc > 1 and all_samples_mc:
                mc_tensor = torch.cat(all_samples_mc, dim=1)  # (mc, total, P, A)

            # Align channel counts if they mismatch (e.g., Fourier transform cases)
            if samples.shape[-1] != truth.shape[-1]:
                min_c = min(samples.shape[-1], truth.shape[-1])
                log.warning(
                    "Channel mismatch in samples vs truth (%d vs %d); truncating to %d channels for metrics.",
                    samples.shape[-1],
                    truth.shape[-1],
                    min_c,
                )
                samples = samples[..., :min_c]
                truth = truth[..., :min_c]
                if mc_tensor is not None:
                    mc_tensor = mc_tensor[..., :min_c]

            metrics = compute_time_aligned_metrics(samples, truth)
            if mc_tensor is not None:
                gen_for_dist = mc_tensor.reshape(mc_tensor.size(0) * mc_tensor.size(1), samples.size(1), samples.size(2))
                ref_for_dist = truth.repeat(mc, 1, 1)
                dist = compute_distribution_metrics(gen_for_dist, ref_for_dist).as_dict()
            else:
                dist = compute_distribution_metrics(samples, truth).as_dict()
            series = compute_series_metrics(truth, samples)
            matrix = compute_covariance_metrics(truth, samples)

            # Additional ND/NRMSE style metrics (align with EpochMetricsLogger)
            pred_mean = samples if mc_tensor is None else mc_tensor.mean(dim=0)
            eps = 1e-8
            diff = pred_mean - truth
            nd = torch.sum(torch.abs(diff)) / (torch.sum(torch.abs(truth)) + eps)
            nrmse = torch.sqrt(torch.sum(diff ** 2) / (torch.sum(truth ** 2) + eps))
            truth_sum = truth.sum(dim=-1)
            pred_sum_mean = pred_mean.sum(dim=-1)
            diff_sum = pred_sum_mean - truth_sum
            nd_sum = torch.sum(torch.abs(diff_sum)) / (torch.sum(torch.abs(truth_sum)) + eps)
            nrmse_sum = torch.sqrt(torch.sum(diff_sum ** 2) / (torch.sum(truth_sum ** 2) + eps))

            payload = {
                "samples": samples,
                "truth": truth,
                "window_stage_counts": stage_counts,
            }
            if all_samples_fourier:
                if mc > 1 and self.fourier_transform:
                    payload["samples_fourier"] = torch.cat(all_samples_fourier, dim=1)
                else:
                    payload["samples_fourier"] = torch.cat(all_samples_fourier, dim=0)
            metrics_payload = {
                "time_aligned": metrics,
                "distribution": dist,
                "series": series,
                "matrix": matrix,
            }

            extras = {}
            for key, value in metrics.__dict__.items():
                tensor = torch.as_tensor(value)
                extras[f"time_aligned_{key}"] = float(tensor.float().mean().item())
            extras.update({f"distribution_{k}": float(v) for k, v in dist.items()})
            extras.update({f"series_{k}": float(v) for k, v in series.items()})
            extras.update({f"matrix_{k}": float(v) for k, v in matrix.items()})
            extras["series_nd"] = float(nd.item())
            extras["series_nrmse"] = float(nrmse.item())
            extras["series_nd_sum"] = float(nd_sum.item())
            extras["series_nrmse_sum"] = float(nrmse_sum.item())
            corr_struct = compute_corr_structure_metrics(
                samples.detach().cpu().numpy(),
                truth.detach().cpu().numpy(),
            )
            extras.update({k: float(v) for k, v in corr_struct.items()})

            # CRPS (MC) when multiple trajectories are available
            if mc_tensor is not None:
                term1 = (mc_tensor - truth.unsqueeze(0)).abs().mean(dim=0)
                pairwise = (mc_tensor.unsqueeze(0) - mc_tensor.unsqueeze(1)).abs().mean(dim=(0, 1))
                crps = (term1 - 0.5 * pairwise).mean().item()
                extras["series_crps_mc"] = float(crps)
                extras.setdefault("series_crps", float(crps))

            disable_fourier = os.environ.get("CFDIFF_DISABLE_FOURIER", "0") == "1"
            if not disable_fourier and samples.size(0) >= 2:
                try:
                    fourier = compute_metric_collection(truth, samples)
                    for key, value in fourier.items():
                        if isinstance(value, (list, tuple)):
                            if len(value) == 0:
                                continue
                            tensor = torch.as_tensor(value, dtype=torch.float32)
                            extras[f"fourier_{key}"] = float(tensor.mean().item())
                        elif isinstance(value, (float, int)):
                            extras[f"fourier_{key}"] = float(value)
                        else:
                            tensor = torch.as_tensor(value)
                            extras[f"fourier_{key}"] = float(tensor.float().mean().item())
                except Exception as exc:
                    log.warning("Fourier metric computation failed: %s", exc)

            spec_samples = torch.fft.rfft(samples, dim=1)
            spec_truth = torch.fft.rfft(truth, dim=1)
            power_diff = (torch.abs(spec_samples) ** 2 - torch.abs(spec_truth) ** 2).abs().mean().item()
            extras["spectral_power_abs_error"] = power_diff

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            parent_dir = batch_root if batch_root is not None else history_root
            suffix = f"-r{repeat_idx+1:02d}" if sample_repeats > 1 else ""
            history_dir = parent_dir / (f"{timestamp}{suffix}-{tag}" if tag else f"{timestamp}{suffix}")
            history_dir.mkdir(parents=True, exist_ok=True)
            last_history_dir = history_dir
            timestamped_samples = history_dir / "samples.pt"
            timestamped_metrics = history_dir / "metrics.pt"
            timestamped_extra = history_dir / "metrics_extra.json"
            timestamped_extra_val = history_dir / "metrics_extra_val.json"
            timestamped_extra_test = history_dir / "metrics_extra_test.json"

            torch.save(payload, timestamped_samples)
            shutil.copyfile(timestamped_samples, self.output_dir / "samples.pt")

            torch.save(metrics_payload, timestamped_metrics)
            shutil.copyfile(timestamped_metrics, self.output_dir / "metrics.pt")

            with timestamped_extra.open("w", encoding="utf-8") as f:
                json.dump(extras, f, ensure_ascii=False, indent=2)
            shutil.copyfile(timestamped_extra, self.output_dir / "metrics_extra.json")
            log.info("Saved unconditional metrics to %s (timestamp %s)", history_dir, timestamp)

            if mc > 1 and save_mc_artifacts and mc_tensor is not None:
                torch.save({"samples_mc": mc_tensor, "truth": truth}, history_dir / "samples_mc.pt")

                def _metrics_single(mc_slice: torch.Tensor, truth_slice: torch.Tensor) -> dict:
                    dist_single = compute_distribution_metrics(mc_slice, truth_slice).as_dict()
                    series_single = compute_series_metrics(truth_slice, mc_slice)
                    matrix_single = compute_covariance_metrics(truth_slice, mc_slice)
                    corr_struct_single = compute_corr_structure_metrics(
                        mc_slice.detach().cpu().numpy(),
                        truth_slice.detach().cpu().numpy(),
                    )
                    out_single = {
                        **{f"distribution_{k}": float(v) for k, v in dist_single.items()},
                        **{f"series_{k}": float(v) for k, v in series_single.items()},
                        **{f"matrix_{k}": float(v) for k, v in matrix_single.items()},
                    }
                    out_single.update({f"{k}": float(v) for k, v in corr_struct_single.items()})
                    if not disable_fourier and mc_slice.size(0) >= 2:
                        try:
                            fourier_single = compute_metric_collection(truth_slice, mc_slice)
                            out_single.update(
                                {
                                    f"fourier_{k}": float(torch.as_tensor(v).float().mean().item())
                                    for k, v in fourier_single.items()
                                }
                            )
                        except Exception:
                            pass
                    return out_single

                metrics_mc: dict[str, list[float]] = {}
                for i in range(mc_tensor.size(0)):
                    m_single = _metrics_single(mc_tensor[i], truth)
                    for k, v in m_single.items():
                        metrics_mc.setdefault(k, []).append(float(v))
                with (history_dir / "metrics_mc.json").open("w", encoding="utf-8") as f:
                    json.dump({"mc": mc_tensor.size(0), "metrics": metrics_mc}, f, ensure_ascii=False, indent=2)

            # Stage-wise metrics (val/test separated)
            def _stage_metrics(offset: int, count: int) -> dict:
                if count <= 0:
                    return {}
                s_slice = samples[offset : offset + count]
                t_slice = truth[offset : offset + count]
                if s_slice.shape[-1] != t_slice.shape[-1]:
                    min_c = min(s_slice.shape[-1], t_slice.shape[-1])
                    s_slice = s_slice[..., :min_c]
                    t_slice = t_slice[..., :min_c]
                if mc_tensor is not None:
                    mc_slice = mc_tensor[:, offset : offset + count]
                    gen_for_dist = mc_slice.reshape(mc_slice.size(0) * mc_slice.size(1), s_slice.size(1), s_slice.size(2))
                    ref_for_dist = t_slice.repeat(mc_slice.size(0), 1, 1)
                    dist_stage = compute_distribution_metrics(gen_for_dist, ref_for_dist).as_dict()
                else:
                    dist_stage = compute_distribution_metrics(s_slice, t_slice).as_dict()
                series_stage = compute_series_metrics(t_slice, s_slice)
                matrix_stage = compute_covariance_metrics(t_slice, s_slice)
                corr_struct_stage = compute_corr_structure_metrics(
                    s_slice.detach().cpu().numpy(),
                    t_slice.detach().cpu().numpy(),
                )
                # ND/NRMSE style metrics per stage (match overall extras)
                pred_mean_stage = s_slice if mc_tensor is None else mc_tensor.mean(dim=0)[offset : offset + count]
                eps_stage = 1e-8
                diff_stage = pred_mean_stage - t_slice
                nd_stage = torch.sum(torch.abs(diff_stage)) / (torch.sum(torch.abs(t_slice)) + eps_stage)
                nrmse_stage = torch.sqrt(torch.sum(diff_stage ** 2) / (torch.sum(t_slice ** 2) + eps_stage))
                truth_sum_stage = t_slice.sum(dim=-1)
                pred_sum_mean_stage = pred_mean_stage.sum(dim=-1)
                diff_sum_stage = pred_sum_mean_stage - truth_sum_stage
                nd_sum_stage = torch.sum(torch.abs(diff_sum_stage)) / (torch.sum(torch.abs(truth_sum_stage)) + eps_stage)
                nrmse_sum_stage = torch.sqrt(torch.sum(diff_sum_stage ** 2) / (torch.sum(truth_sum_stage ** 2) + eps_stage))
                if disable_fourier or s_slice.size(0) < 2:
                    fourier_stage = {}
                else:
                    try:
                        fourier_stage = compute_metric_collection(t_slice, s_slice)
                    except Exception:
                        fourier_stage = {}
                crps_val = None
                if mc_tensor is not None:
                    truth_exp = t_slice.unsqueeze(0).expand_as(mc_slice)
                    term1 = (mc_slice - truth_exp).abs().mean(dim=0)
                    pairwise = (mc_slice.unsqueeze(0) - mc_slice.unsqueeze(1)).abs().mean(dim=(0, 1))
                    crps_val = float((term1 - 0.5 * pairwise).mean().item())

                out_stage = {
                    **{f"distribution_{k}": float(v) for k, v in dist_stage.items()},
                    **{f"series_{k}": float(v) for k, v in series_stage.items()},
                    **{f"matrix_{k}": float(v) for k, v in matrix_stage.items()},
                    **{
                        f"fourier_{k}": (float(torch.as_tensor(v).float().mean().item()) if isinstance(v, (list, tuple)) else float(v))
                        for k, v in fourier_stage.items()
                    },
                }
                out_stage["series_nd"] = float(nd_stage.item())
                out_stage["series_nrmse"] = float(nrmse_stage.item())
                out_stage["series_nd_sum"] = float(nd_sum_stage.item())
                out_stage["series_nrmse_sum"] = float(nrmse_sum_stage.item())
                out_stage.update({k: float(v) for k, v in corr_struct_stage.items()})
                if crps_val is not None:
                    out_stage["series_crps_mc"] = crps_val
                    out_stage.setdefault("series_crps", crps_val)
                return out_stage

            stage_metrics: dict = {}
            offset = 0
            for stage_name in ("val", "test"):
                n = int(stage_counts.get(stage_name, 0) or 0)
                if n > 0:
                    stage_metrics[stage_name] = _stage_metrics(offset, n)
                offset += n

            for stage_name, metrics_dict in stage_metrics.items():
                path = history_dir / f"metrics_extra_{stage_name}.json"
                with path.open("w", encoding="utf-8") as f:
                    json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
                log.info("Saved %s metrics to %s", stage_name, path)
                # Also copy to root for convenience
                if stage_name == "val":
                    if path != timestamped_extra_val:
                        shutil.copyfile(path, timestamped_extra_val)
                    shutil.copyfile(path, self.output_dir / "metrics_extra_val.json")
                if stage_name == "test":
                    if path != timestamped_extra_test:
                        shutil.copyfile(path, timestamped_extra_test)
                    shutil.copyfile(path, self.output_dir / "metrics_extra_test.json")

        if self.run_mode in {"sample", "train_sample"}:
            cfg_to_save = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
            OmegaConf.save(cfg_to_save, self.output_dir / "sample_config.yaml")
            if last_history_dir:
                OmegaConf.save(cfg_to_save, last_history_dir / "sample_config.yaml")

    def _load_checkpoint(self, path: Path) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(path, map_location=device)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
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
