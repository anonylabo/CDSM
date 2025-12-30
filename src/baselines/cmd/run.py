from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence
import inspect
import copy

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset

from scipy.cluster.hierarchy import cophenet, linkage
from scipy.stats import wasserstein_distance

from baselines import (
    exp_covariance,
    ccc_garch_covariance_forecast,
    dcc_garch_covariance_forecast,
    garch_covariance_forecast,
    ledoit_wolf_covariance,
    oracle_shrinkage_covariance,
    riskmetrics_ewma_covariance,
    sample_covariance,
    CNNBiLSTMCovarianceBaseline,
)

log = logging.getLogger(__name__)

BaselineFn = Callable[..., pd.DataFrame]

BASELINE_REGISTRY: Dict[str, BaselineFn] = {
    "sample_cov": sample_covariance,
    "exp_cov": exp_covariance,
    "ledoit_wolf": ledoit_wolf_covariance,
    "oracle_shrinkage": oracle_shrinkage_covariance,
    "riskmetrics": riskmetrics_ewma_covariance,
    "ewma": riskmetrics_ewma_covariance,
    "garch_cov": garch_covariance_forecast,
    "ccc_garch": ccc_garch_covariance_forecast,
    "dcc_garch": dcc_garch_covariance_forecast,
    "cab": CNNBiLSTMCovarianceBaseline,
}


_EPS = 1e-12

DEFAULT_METRICS: List[str] = [
    "matrix_cov_fro",
    "matrix_corr_fro",
    "matrix_cov_fro_rel",
    "matrix_cov_mse",
    "matrix_cov_mae",
    "matrix_cov_diag_mape",
    "matrix_corr_offdiag_pearson",
    "matrix_corr_offdiag_spearman",
    "matrix_corr_cross_mse",
    "matrix_corr_sign_rate",
    "corr_wasserstein_flat",
    "corr_mean_correl_abs_diff",
    "corr_diag_abs_mean",
    "corr_diag_abs_std",
    "corr_symmetry_abs_mean",
    "corr_symmetry_abs_std",
    "corr_eigen_gini_abs_diff",
    "corr_coph_single_abs_diff",
    "corr_coph_ward_abs_diff",
    "corr_perron_frob_abs_diff",
    "corr_power_eigen_abs_diff",
]


def _default_asset_names(num_assets: int) -> List[str]:
    return [f"asset_{i}" for i in range(num_assets)]


def _cov_from_rows(rows: np.ndarray, num_assets: int, nan_policy: str = "raise") -> np.ndarray:
    """Compute a covariance matrix from (time, assets) rows with NaN handling."""
    arr = np.asarray(rows)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array for window; got {arr.shape}")
    policy = str(nan_policy or "raise").lower()
    non_finite = ~np.isfinite(arr)
    if non_finite.any():
        if policy == "raise":
            bad = int(non_finite.sum())
            raise ValueError(f"Window contains {bad} non-finite values but nan_policy='{nan_policy}'.")
        if policy == "drop":
            arr = arr[np.isfinite(arr).all(axis=1)]
        else:
            raise ValueError(f"Unsupported nan_policy='{nan_policy}'")
    if arr.shape[0] <= 1:
        return np.zeros((num_assets, num_assets), dtype=np.float64)
    cov = np.cov(arr, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    # np.cov can return scalars for single-asset cases; ensure 2-D.
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    if np.ndim(cov) == 1:
        cov = np.array([[float(cov[0])]], dtype=np.float64)
    return cov.astype(np.float64, copy=False)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    diag = np.sqrt(np.clip(np.diag(cov), _EPS, None))
    denom = np.outer(diag, diag)
    corr = np.zeros_like(cov)
    mask = denom > 0
    corr[mask] = cov[mask] / denom[mask]
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


def _wasserstein_1d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    a = np.sort(x)
    b = np.sort(y)
    n = min(a.size, b.size)
    if a.size != b.size:
        grid = np.linspace(0.0, 1.0, n, endpoint=False)
        a = np.interp(grid, np.linspace(0.0, 1.0, a.size, endpoint=False), a)
        b = np.interp(grid, np.linspace(0.0, 1.0, b.size, endpoint=False), b)
    return float(np.mean(np.abs(a - b)))


def _ensure_symmetric_unit(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(sym, 1.0)
    return np.clip(sym, -1.0, 1.0)


def _gini_coefficient(values: np.ndarray) -> float:
    if np.allclose(values, 0):
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cumvals = np.cumsum(sorted_vals)
    gini = (n + 1 - 2 * np.sum(cumvals) / cumvals[-1]) / n
    return float(gini)


def _cophenetic_corr(matrix: np.ndarray, method: str) -> float:
    dist = np.sqrt(np.clip(2.0 * (1.0 - matrix), 0.0, None))
    iu = np.triu_indices_from(matrix, k=1)
    compressed = dist[iu]
    if compressed.size < 2:
        return 0.0
    Z = linkage(compressed, method=method)
    corr, _ = cophenet(Z, compressed)
    return float(corr)


def _perron_frobenius_neg_sum(eigenvecs: np.ndarray) -> float:
    first_vec = eigenvecs[:, 0]
    neg_entries = first_vec[first_vec < 0]
    return float(np.abs(neg_entries).sum())


def _power_law_exponent(eigenvals: np.ndarray) -> float:
    vals = np.clip(eigenvals, 1e-8, None)
    ranks = np.arange(1, len(vals) + 1)
    log_r = np.log(ranks)
    log_v = np.log(vals)
    slope, _ = np.polyfit(log_r, log_v, 1)
    return float(-slope)


def _structure_metrics(est_corr: np.ndarray, ref_corr: np.ndarray) -> Dict[str, float]:
    est = _ensure_symmetric_unit(est_corr)
    ref = _ensure_symmetric_unit(ref_corr)
    diag_diff = np.abs(np.diag(est) - np.diag(ref))
    sym_error = np.abs((est - est.T)[np.tril_indices_from(est, k=-1)])
    flat_est = est.flatten()
    flat_ref = ref.flatten()
    stylised_keys = [
        "mean_correl",
        "eigen_gini",
        "coph_single",
        "coph_ward",
        "perron_frob",
        "power_eigen",
    ]
    def _stylised(mat: np.ndarray) -> Dict[str, float]:
        eigvals, eigvecs = np.linalg.eigh(mat)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        return {
            "mean_correl": float(mat[np.triu_indices_from(mat, k=1)].mean()),
            "eigen_gini": _gini_coefficient(eigvals),
            "coph_single": _cophenetic_corr(mat, "single"),
            "coph_ward": _cophenetic_corr(mat, "ward"),
            "perron_frob": _perron_frobenius_neg_sum(eigvecs),
            "power_eigen": _power_law_exponent(eigvals),
        }
    est_stylised = _stylised(est)
    ref_stylised = _stylised(ref)
    metrics = {
        "corr_diag_abs_mean": float(diag_diff.mean()),
        "corr_diag_abs_std": float(diag_diff.std()),
        "corr_symmetry_abs_mean": float(sym_error.mean()) if sym_error.size else 0.0,
        "corr_symmetry_abs_std": float(sym_error.std()) if sym_error.size else 0.0,
        "corr_wasserstein_flat": float(wasserstein_distance(flat_est, flat_ref)),
    }
    for key in stylised_keys:
        metrics[f"corr_{key}_abs_diff"] = abs(est_stylised[key] - ref_stylised[key])
    return metrics


def _compute_metric(metric_name: str, estimate: np.ndarray, reference: np.ndarray) -> float:
    diff = estimate - reference
    if metric_name == "matrix_cov_fro":
        return float(np.linalg.norm(diff, ord="fro"))
    if metric_name == "matrix_cov_fro_rel":
        denom = np.linalg.norm(reference, ord="fro") + 1e-12
        return float(np.linalg.norm(diff, ord="fro") / denom)
    if metric_name == "matrix_cov_mse":
        return float(np.mean(diff * diff))
    if metric_name == "matrix_cov_mae":
        return float(np.mean(np.abs(diff)))
    if metric_name == "matrix_cov_diag_mape":
        ref_diag = np.diag(reference)
        est_diag = np.diag(estimate)
        return float(np.mean(np.abs((est_diag - ref_diag) / (np.abs(ref_diag) + 1e-12))))
    if metric_name in {"matrix_corr_fro", "matrix_corr_fro_rel"}:
        ref_corr = _cov_to_corr(reference)
        est_corr = _cov_to_corr(estimate)
        corr_diff = est_corr - ref_corr
        frob = float(np.linalg.norm(corr_diff, ord="fro"))
        if metric_name == "matrix_corr_fro":
            return frob
        denom = np.linalg.norm(ref_corr, ord="fro") + _EPS
        return frob / denom
    if metric_name in {
        "matrix_corr_offdiag_pearson",
        "matrix_corr_offdiag_spearman",
        "matrix_corr_cross_mse",
        "matrix_corr_sign_rate",
    }:
        ref_corr = _cov_to_corr(reference)
        est_corr = _cov_to_corr(estimate)
        ref_off = _flat_offdiag(ref_corr)
        est_off = _flat_offdiag(est_corr)
        if metric_name == "matrix_corr_cross_mse":
            return float(np.mean((est_off - ref_off) ** 2))
        if metric_name == "matrix_corr_sign_rate":
            return float(np.mean(np.sign(est_off) == np.sign(ref_off)))
        if metric_name == "corr_wasserstein_flat":
            return _wasserstein_1d(est_off, ref_off)
        if metric_name == "corr_mean_correl_abs_diff":
            ref_mean = float(ref_off.mean()) if ref_off.size else 0.0
            est_mean = float(est_off.mean()) if est_off.size else 0.0
            return abs(est_mean - ref_mean)
        if metric_name == "matrix_corr_offdiag_pearson":
            if est_off.std() <= _EPS or ref_off.std() <= _EPS:
                return 0.0
            return float(np.corrcoef(est_off, ref_off)[0, 1])
        if metric_name == "matrix_corr_offdiag_spearman":
            est_rank = _rank_avg_ties(est_off)
            ref_rank = _rank_avg_ties(ref_off)
            if est_rank.std() <= _EPS or ref_rank.std() <= _EPS:
                return 0.0
            return float(np.corrcoef(est_rank, ref_rank)[0, 1])
        if metric_name == "corr_wasserstein_flat" or metric_name == "corr_mean_correl_abs_diff":
            struct = _structure_metrics(est_corr, ref_corr)
            return struct[metric_name]
    if metric_name in {
        "corr_diag_abs_mean",
        "corr_diag_abs_std",
        "corr_symmetry_abs_mean",
        "corr_symmetry_abs_std",
        "corr_mean_correl_abs_diff",
        "corr_eigen_gini_abs_diff",
        "corr_coph_single_abs_diff",
        "corr_coph_ward_abs_diff",
        "corr_perron_frob_abs_diff",
        "corr_power_eigen_abs_diff",
        "corr_wasserstein_flat",
    }:
        ref_corr = _cov_to_corr(reference)
        est_corr = _cov_to_corr(estimate)
        struct = _structure_metrics(est_corr, ref_corr)
        return struct[metric_name]
    raise ValueError(f"Unsupported metric: {metric_name}")


class BaselineRunner:
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

        include_val = bool(getattr(cfg.baseline, "include_val", False))
        self.val_count = 0
        self.test_count = len(self.datamodule.ds_test)
        if include_val:
            ds_val = getattr(self.datamodule, "ds_val", None)
            if ds_val is None:
                raise ValueError("include_val=true but datamodule does not expose ds_val.")
            self.val_count = len(ds_val)
            self.dataset = ConcatDataset([ds_val, self.datamodule.ds_test])
        else:
            self.dataset = self.datamodule.ds_test
        self.context_len = int(getattr(self.datamodule, "context_len_time", cfg.datamodule.context_len))
        self.pred_len = int(getattr(self.datamodule, "pred_len_time", cfg.datamodule.pred_len))
        self.asset_count = getattr(self.datamodule, "original_num_assets", None)
        if self.asset_count is None:
            sample = self.dataset[0]["target_time"]
            self.asset_count = sample.shape[1]

        self.fourier = bool(getattr(self.datamodule, "fourier_transform", False))
        self.method_name = str(cfg.baseline.method)
        dm_name = self.datamodule.__class__.__name__.lower()
        if "unconditional" in dm_name:
            self.mode = "unconditional"
        else:
            self.mode = "conditional"

        requested_output = getattr(cfg, "output_dir", None)
        if requested_output:
            output_dir = Path(to_absolute_path(str(requested_output)))
        else:
            base = Path.cwd() / "outputs" / "baselines" / self.method_name / self.mode
            if cfg.baseline.tag:
                base = base / str(cfg.baseline.tag)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = base / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        OmegaConf.save(cfg, output_dir / "baseline_config.yaml")
        stage_info = {"val": self.val_count, "test": self.test_count}
        with (output_dir / "stage_counts.json").open("w", encoding="utf-8") as f:
            json.dump(stage_info, f, indent=2)

    def run(self) -> None:
        cfg = self.cfg.baseline
        method_name = self.method_name
        if method_name not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline method '{method_name}'. Valid options: {sorted(BASELINE_REGISTRY)}")
        method = BASELINE_REGISTRY[method_name]
        method_label = method_name

        total = len(self.dataset)
        window_index = int(cfg.window_index)
        if window_index < 0:
            indices: Iterable[int] = range(total)
        else:
            if window_index >= total:
                raise IndexError(f"window_index {window_index} exceeds available test windows ({total})")
            indices = [window_index]

        metric_names_cfg = getattr(cfg, "metric_names", None)
        metric_names: List[str]
        if metric_names_cfg:
            metric_names = [str(m) for m in metric_names_cfg]
        else:
            metric_name = getattr(cfg, "metric_name", None)
            if metric_name and str(metric_name).lower() not in {"", "all", "default"}:
                metric_names = [str(metric_name)]
            else:
                metric_names = DEFAULT_METRICS
        summaries = []
        method_kwargs = dict(cfg.method_kwargs) if cfg.method_kwargs else {}
        show_progress = bool(getattr(cfg, "show_progress", False))
        if "return_corr_only" in method_kwargs:
            method_label = f"{method_name}_{'corr' if method_kwargs['return_corr_only'] else 'cov'}"

        # Build the estimator once so deep baselines can train on the training split.
        trainable = None
        if inspect.isclass(method) and hasattr(method, "fit") and hasattr(method, "predict"):
            trainable = method(**method_kwargs)
        elif hasattr(method, "fit") and hasattr(method, "predict"):
            trainable = method

        default_cols = _default_asset_names(self.asset_count)
        if trainable is not None:
            log.info("Fitting trainable baseline '%s' on training data", method_label)
            trainable.fit(
                datamodule=self.datamodule,
                asset_names=default_cols,
                context_len=self.context_len,
                pred_len=self.pred_len,
            )

            def _predict_fn(context_window, colnames):
                return trainable.predict(context_window, columns=colnames)

        else:
            sig = inspect.signature(method)
            if "horizon" in sig.parameters and "horizon" not in method_kwargs:
                method_kwargs["horizon"] = self.pred_len
            if "pred_len" in sig.parameters and "pred_len" not in method_kwargs:
                method_kwargs["pred_len"] = self.pred_len

            def _predict_fn(context_window, colnames):
                return method(context_window, columns=colnames, **method_kwargs)

        iterator = indices
        if show_progress:
            iterator = tqdm(indices, desc=f"{method_label} windows", leave=False)
        nan_policy = str(getattr(cfg, "nan_policy", "raise") or "raise").lower()
        for idx in iterator:
            sample = self.dataset[idx]
            context_raw = sample["context_time"].detach().cpu().numpy()
            target_raw = sample["target_time"].detach().cpu().numpy()
            n_assets = target_raw.shape[1]

            cols = _default_asset_names(n_assets)
            # Do not drop NaN rows before prediction; only handle NaNs when computing covariances.
            baseline_df = _predict_fn(context_raw, cols)
            if not isinstance(baseline_df, pd.DataFrame):
                baseline_df = pd.DataFrame(baseline_df, index=cols, columns=cols)

            truth_cov = _cov_from_rows(target_raw, n_assets, nan_policy=nan_policy)
            est_mat = np.nan_to_num(baseline_df.to_numpy(), nan=0.0)
            metrics_for_window = {
                metric: _compute_metric(metric, est_mat, truth_cov) for metric in metric_names
            }

            prefix = f"{method_label}_win{idx:04d}"
            if cfg.save_csv:
                pd.DataFrame(est_mat, index=cols, columns=cols).to_csv(self.output_dir / f"{prefix}_est.csv")
                pd.DataFrame(truth_cov, index=cols, columns=cols).to_csv(
                    self.output_dir / f"{prefix}_truth.csv"
                )
                pd.DataFrame(context_raw, columns=cols).to_csv(
                    self.output_dir / f"{prefix}_context_series.csv"
                )
                pd.DataFrame(target_raw, columns=cols).to_csv(
                    self.output_dir / f"{prefix}_target_series.csv"
                )
            if cfg.save_pt:
                torch.save(torch.from_numpy(est_mat), self.output_dir / f"{prefix}_est.pt")
                torch.save(torch.from_numpy(truth_cov), self.output_dir / f"{prefix}_truth.pt")
                torch.save(torch.from_numpy(context_raw), self.output_dir / f"{prefix}_context_series.pt")
                torch.save(torch.from_numpy(target_raw), self.output_dir / f"{prefix}_target_series.pt")

            for metric_name, metric_value in metrics_for_window.items():
                summaries.append(
                    {
                        "window_index": idx,
                        "method": method_label,
                        "metric": metric_name,
                        "metric_value": metric_value,
                        "num_assets": n_assets,
                    }
                )
                log.info(
                    "Baseline %s window %d -> %s=%.6f",
                    method_label,
                    idx,
                    metric_name,
                    metric_value,
                )

        summary_path = self.output_dir / f"{method_label}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        log.info("Saved %d baseline windows to %s", len(summaries), self.output_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    repeats = int(getattr(cfg.baseline, "repeats", 1) or 1)
    base_seed = int(getattr(cfg, "random_seed", 0) or 0)

    if repeats <= 1:
        runner = BaselineRunner(cfg)
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
        base = Path(to_absolute_path(str(cfg.output_dir)))
    else:
        base = Path.cwd() / "outputs" / "baselines" / method / mode
        if tag:
            base = base / str(tag)
    batch_root = base / f"batch-{stamp}"
    batch_root.mkdir(parents=True, exist_ok=True)
    log.info("Running %d baseline repeats under %s", repeats, batch_root)

    for i in range(repeats):
        cfg_rep = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg_rep.random_seed = base_seed + i
        cfg_rep.output_dir = str(batch_root / f"{stamp}-r{i+1:02d}")
        runner = BaselineRunner(cfg_rep)
        runner.run()


if __name__ == "__main__":
    main()
