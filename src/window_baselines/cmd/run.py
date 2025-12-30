from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import copy

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    return arr


def _stack_target_rows(windows: np.ndarray, pred_len: int) -> np.ndarray:
    # windows shape: (N, L, A); target rows are the last pred_len entries
    if windows.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    targets = windows[:, -pred_len:, :]  # (N, P, A)
    return targets.reshape(-1, targets.shape[-1])  # ((N*P), A)


def _cov_from_rows(rows: np.ndarray) -> np.ndarray:
    rows = _ensure_2d(rows)
    # Drop any time rows that contain NaNs to avoid contaminating the covariance.
    clean_rows = rows[~np.isnan(rows).any(axis=1)]
    n_assets = rows.shape[1]
    if clean_rows.shape[0] <= 1:
        return np.zeros((n_assets, n_assets), dtype=np.float64)
    cov = np.cov(clean_rows, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    return cov.astype(np.float64, copy=False)


def _cov_to_corr(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(cov, dtype=np.float64)
    diag = np.sqrt(np.clip(np.diag(a), eps, None))
    denom = np.outer(diag, diag)
    corr = np.zeros_like(a, dtype=np.float64)
    mask = denom > 0
    corr[mask] = a[mask] / denom[mask]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _flat_offdiag(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(matrix.shape[0], k=1)
    return matrix[idx]


def _rank_avg_ties(z: np.ndarray) -> np.ndarray:
    order = z.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(z), dtype=float)
    _, inv, cnt = np.unique(z, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    return (sums / cnt)[inv]


def compute_metrics(est: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    diff = est - ref
    ref_corr = _cov_to_corr(ref)
    est_corr = _cov_to_corr(est)
    ref_off = _flat_offdiag(ref_corr)
    est_off = _flat_offdiag(est_corr)

    metrics = {
        "matrix_cov_fro": float(np.linalg.norm(diff, ord="fro")),
        "matrix_corr_fro": float(np.linalg.norm(est_corr - ref_corr, ord="fro")),
        "matrix_corr_offdiag_pearson": 0.0,
        "matrix_corr_offdiag_spearman": 0.0,
    }
    if ref_off.size > 1 and est_off.size > 1:
        if np.std(est_off) > 1e-12 and np.std(ref_off) > 1e-12:
            metrics["matrix_corr_offdiag_pearson"] = float(np.corrcoef(est_off, ref_off)[0, 1])
        est_rank = _rank_avg_ties(est_off)
        ref_rank = _rank_avg_ties(ref_off)
        if np.std(est_rank) > 1e-12 and np.std(ref_rank) > 1e-12:
            metrics["matrix_corr_offdiag_spearman"] = float(np.corrcoef(est_rank, ref_rank)[0, 1])
    return metrics


class WindowBaselineRunner:
    def __init__(self, cfg: DictConfig) -> None:
        if "experiment" in cfg:
            OmegaConf.set_struct(cfg, False)
            cfg = OmegaConf.merge(cfg, cfg.experiment)
            cfg.pop("experiment", None)
            OmegaConf.set_struct(cfg, True)

        pl.seed_everything(cfg.random_seed)
        self.cfg = cfg
        self.datamodule = instantiate(cfg.datamodule)
        self.datamodule.prepare_data()
        self.datamodule.setup()

        if getattr(self.datamodule, "ds_test", None) is None:
            raise ValueError("Datamodule did not create a test split; baselines require ds_test.")
        if getattr(self.datamodule, "fourier_transform", False):
            raise ValueError("window-space baselines expect time-domain windows (fourier_transform=false).")

        self.context_len = int(self.datamodule.context_len_time)
        self.pred_len = int(self.datamodule.pred_len_time)

        include_val = bool(getattr(cfg.baseline, "include_val", False))
        self.val_count = len(self.datamodule.ds_val) if (include_val and self.datamodule.ds_val is not None) else 0
        self.test_count = len(self.datamodule.ds_test)

        dm_name = self.datamodule.__class__.__name__.lower()
        self.mode = "unconditional" if "unconditional" in dm_name else "conditional"

        requested_output = getattr(cfg, "output_dir", None)
        if requested_output:
            output_dir = Path(str(requested_output)).expanduser().resolve()
        else:
            base = Path.cwd() / "outputs" / "baselines" / cfg.baseline.method / self.mode
            if cfg.baseline.tag:
                base = base / str(cfg.baseline.tag)
            output_dir = (base / datetime.now().strftime("%Y%m%d-%H%M%S")).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        OmegaConf.save(cfg, output_dir / "baseline_config.yaml")
        stage_info = {"val": self.val_count, "test": self.test_count}
        with (output_dir / "stage_counts.json").open("w", encoding="utf-8") as f:
            json.dump(stage_info, f, indent=2)

    # ---------------------- Baseline estimators ---------------------- #
    def _train_unconditional(self, train_windows: np.ndarray) -> np.ndarray:
        rows = _stack_target_rows(train_windows, self.pred_len)
        return _cov_from_rows(rows)

    def _factor_cov(self, train_windows: np.ndarray, rank: int) -> np.ndarray:
        base_cov = self._train_unconditional(train_windows)
        eigvals, eigvecs = np.linalg.eigh(base_cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        k = max(1, min(rank, eigvals.size))
        top_vals = eigvals[:k]
        top_vecs = eigvecs[:, :k]
        residual_mean = float(np.mean(eigvals[k:])) if eigvals.size > k else 0.0
        sigma2 = max(residual_mean, 0.0)
        cov_lowrank = top_vecs @ np.diag(top_vals) @ top_vecs.T
        cov = cov_lowrank + sigma2 * np.eye(base_cov.shape[0], dtype=np.float64)
        return cov

    # ---------------------- Runner ---------------------- #
    def run(self) -> None:
        cfg = self.cfg.baseline
        method = str(cfg.method)
        method_kwargs = dict(cfg.method_kwargs) if cfg.method_kwargs else {}

        train_windows = self.datamodule.ds_train.windows
        val_windows = self.datamodule.ds_val.windows if self.datamodule.ds_val is not None else None
        use_val_for_cov = bool(getattr(cfg, "use_val_for_cov", False))
        if use_val_for_cov and val_windows is not None:
            # Optionally include validation windows when fitting the covariance
            cov_windows = np.concatenate([train_windows, val_windows], axis=0)
        else:
            cov_windows = train_windows
        test_windows = self.datamodule.ds_test.windows

        # Prepare evaluation window list (val then test)
        eval_windows: List[Tuple[str, np.ndarray]] = []
        if cfg.include_val and val_windows is not None:
            eval_windows.append(("val", val_windows))
        eval_windows.append(("test", test_windows))

        constant_cov: Optional[np.ndarray] = None
        factor_rank = int(method_kwargs.get("factor_rank", 4))
        if method == "window_uncond":
            constant_cov = self._train_unconditional(cov_windows)
        elif method == "window_factor":
            constant_cov = self._factor_cov(cov_windows, factor_rank)
        elif method == "series_uncond":
            # Raw-series covariance using full standardized train (optionally val) sequences
            train_series = getattr(self.datamodule, "train_series", None)
            val_series = getattr(self.datamodule, "val_series", None) if use_val_for_cov else None
            if train_series is None:
                raise ValueError("Datamodule does not expose train_series for series_uncond baseline.")
            segments = [train_series]
            if val_series is not None:
                segments.append(val_series)
            full_series = np.concatenate(segments, axis=0)
            constant_cov = _cov_from_rows(full_series)
        elif method in {"window_local", "window_context"}:
            pass  # handled per-window below
        else:
            raise ValueError(f"Unknown baseline method '{method}'.")

        summaries: List[Dict[str, object]] = []
        save_csv = bool(cfg.save_csv)
        save_pt = bool(cfg.save_pt)
        metric_names = (
            "matrix_cov_fro",
            "matrix_corr_fro",
            "matrix_corr_offdiag_pearson",
            "matrix_corr_offdiag_spearman",
        )

        history_rows = _stack_target_rows(cov_windows, self.pred_len)
        requested_index = int(cfg.window_index)
        allowed_indices = None if requested_index < 0 else {requested_index}
        window_global_idx = 0

        for stage, windows in eval_windows:
            for local_idx in range(windows.shape[0]):
                window = windows[local_idx]
                # Skip windows not requested, but keep history causal updates for local baseline
                if allowed_indices is not None and window_global_idx not in allowed_indices:
                    if method == "window_local":
                        history_rows = np.concatenate([history_rows, window[-self.pred_len :, :]], axis=0)
                    window_global_idx += 1
                    continue

                context = window[: self.context_len, :]
                target = window[-self.pred_len :, :]

                if method == "window_local":
                    est_cov = _cov_from_rows(history_rows)
                elif method == "window_context":
                    # Use covariance of the current context block directly as the prediction.
                    est_cov = _cov_from_rows(context)
                else:
                    est_cov = constant_cov

                truth_cov = _cov_from_rows(target)
                metrics = compute_metrics(est_cov, truth_cov)

                prefix = f"{method}_win{window_global_idx:04d}"
                cols = [f"asset_{i}" for i in range(target.shape[1])]
                if save_csv:
                    pd.DataFrame(est_cov, index=cols, columns=cols).to_csv(
                        self.output_dir / f"{prefix}_est.csv"
                    )
                    pd.DataFrame(truth_cov, index=cols, columns=cols).to_csv(
                        self.output_dir / f"{prefix}_truth.csv"
                    )
                    pd.DataFrame(context, columns=cols).to_csv(
                        self.output_dir / f"{prefix}_context_series.csv"
                    )
                    pd.DataFrame(target, columns=cols).to_csv(
                        self.output_dir / f"{prefix}_target_series.csv"
                    )
                if save_pt:
                    torch.save(torch.from_numpy(est_cov), self.output_dir / f"{prefix}_est.pt")
                    torch.save(torch.from_numpy(truth_cov), self.output_dir / f"{prefix}_truth.pt")
                    torch.save(torch.from_numpy(context), self.output_dir / f"{prefix}_context_series.pt")
                    torch.save(torch.from_numpy(target), self.output_dir / f"{prefix}_target_series.pt")

                for name, value in metrics.items():
                    summaries.append(
                        {
                            "window_index": window_global_idx,
                            "stage": stage,
                            "method": method,
                            "metric": name,
                            "metric_value": float(value),
                            "num_assets": int(target.shape[1]),
                        }
                    )
                log.info(
                    "Baseline %s window %d (%s) -> cov_fro=%.6f corr_fro=%.6f",
                    method,
                    window_global_idx,
                    stage,
                    metrics["matrix_cov_fro"],
                    metrics["matrix_corr_fro"],
                )

                # Advance history after predicting (causal roll forward) for local baseline
                if method == "window_local":
                    history_rows = np.concatenate([history_rows, target], axis=0)
                window_global_idx += 1

        # Persist summaries in per-metric rows for compatibility with plotting scripts
        with (self.output_dir / f"{method}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        log.info("Saved %d baseline metric rows to %s", len(summaries), self.output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    repeats = int(getattr(cfg.baseline, "repeats", 1) or 1)
    base_seed = int(getattr(cfg, "random_seed", 0) or 0)

    if repeats <= 1:
        runner = WindowBaselineRunner(cfg)
        runner.run()
        return

    dm = instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup()
    mode = "unconditional" if "unconditional" in dm.__class__.__name__.lower() else "conditional"
    method = str(cfg.baseline.method)
    tag = cfg.baseline.tag
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if cfg.output_dir:
        base = Path(str(cfg.output_dir)).expanduser().resolve()
    else:
        base = Path.cwd() / "outputs" / "baselines" / method / mode
        if tag:
            base = base / str(tag)
    batch_root = base / f"batch-{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)
    log.info("Running %d window baselines under %s", repeats, batch_root)

    for i in range(repeats):
        cfg_rep = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_rep.random_seed = base_seed + i
        cfg_rep.output_dir = str(batch_root / f"{stamp}-r{i+1:02d}")
        runner = WindowBaselineRunner(cfg_rep)
        runner.run()


if __name__ == "__main__":
    main()
