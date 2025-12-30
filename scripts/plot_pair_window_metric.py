#!/usr/bin/env python3
"""
Plot per-window pairwise correlation/covariance for two assets, comparing
per-model predictions against the ground truth.

Supports per-pair outputs and an optional combined grid for multiple pairs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cfdiff.dataloaders.conditional_gluonts import _resolve_split_file

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pred-runs",
        nargs="+",
        required=True,
        help="Run directories containing samples.pt (time-cond/fourier-cond).",
    )
    p.add_argument(
        "--pred-sample-tags",
        nargs="+",
        default=None,
        help="Optional sample tags, same order as --pred-runs (use '-' to select latest).",
    )
    p.add_argument(
        "--pred-labels",
        nargs="+",
        default=None,
        help="Optional labels for pred-runs (same order). Used when plotting separate model lines.",
    )
    p.add_argument(
        "--metric-kind",
        choices=["corr", "cov"],
        default="corr",
        help="Pairwise metric to plot: corr or cov (ignored when --metric-name is set).",
    )
    p.add_argument(
        "--metric-name",
        choices=[
            "matrix_corr_fro",
            "matrix_cov_fro",
            "matrix_corr_fro_rel",
            "matrix_cov_fro_rel",
            "matrix_cov_mse",
            "matrix_cov_mae",
            "matrix_cov_diag_mape",
            "matrix_corr_cross_mse",
            "matrix_corr_offdiag_pearson",
            "matrix_corr_offdiag_spearman",
            "matrix_corr_sign_rate",
            "corr_wasserstein",
            "eigen_wasserstein",
        ],
        help="Optional per-window metric between prediction and truth matrices (overrides --metric-kind).",
    )
    p.add_argument(
        "--matrix-kind",
        choices=["auto", "corr", "cov"],
        default="auto",
        help="Matrix kind for --metric-name (auto picks based on metric).",
    )
    p.add_argument("--stage", choices=["all", "val", "test"], default="test", help="Stage to plot (val/test indexing).")
    p.add_argument(
        "--use-augmented-cov",
        action="store_true",
        help="If set, compute predicted metric on [context; pred] instead of pred-only.",
    )
    p.add_argument(
        "--augmented-run-substr",
        action="append",
        help="Enable augmented cov if any pred-run path contains these substrings.",
    )
    p.add_argument(
        "--baseline",
        action="append",
        help="Optional baselines in format RUN|PREFIX|LABEL[|KIND]. KIND can be 'cov' or 'corr'.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        help="Dataset name or path (for resolving item_id labels).",
    )
    p.add_argument(
        "--dataset-split",
        choices=["train", "test"],
        default="train",
        help="Split to read item_id labels from when --dataset is set.",
    )
    p.add_argument(
        "--asset-ids",
        nargs="+",
        help="Two or more item_id labels to plot (requires --dataset).",
    )
    p.add_argument(
        "--asset-indices",
        nargs="+",
        type=int,
        help="Two or more asset indices to plot (0-based).",
    )
    p.add_argument("--title", type=str, default=None, help="Optional plot title.")
    p.add_argument(
        "--legend",
        choices=["inside", "outside", "none"],
        default="inside",
        help="Legend placement for per-pair plots.",
    )
    p.add_argument(
        "--legend-loc",
        type=str,
        default="best",
        help="Legend location for inside legends.",
    )
    p.add_argument(
        "--legend-columns",
        type=int,
        default=1,
        help="Legend column count.",
    )
    p.add_argument(
        "--legend-fontsize",
        type=int,
        default=9,
        help="Legend font size.",
    )
    p.add_argument(
        "--legend-alpha",
        type=float,
        default=0.75,
        help="Legend transparency for text/handles.",
    )
    p.add_argument(
        "--show-test-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --stage all, draw a vertical line at the val/test boundary.",
    )
    p.add_argument(
        "--x-axis",
        choices=["global", "stage"],
        default="global",
        help="X-axis index: global uses val+test window indices; stage reindexes from 0 within the chosen stage.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the plot (PNG). If multiple pairs are requested, you can include "
        "'{pair}' to substitute the pair label (e.g., assets/pair_{pair}.png).",
    )
    p.add_argument(
        "--combined-output",
        type=Path,
        help="Optional output path for a combined grid of all pairs.",
    )
    p.add_argument(
        "--combined-cols",
        type=int,
        default=2,
        help="Number of columns in the combined grid.",
    )
    p.add_argument(
        "--combined-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(4.6, 2.8),
        help="Per-subplot size (width, height) for the combined grid.",
    )
    p.add_argument(
        "--combined-legend",
        choices=["first", "outside", "none"],
        default="first",
        help="Where to show the legend in the combined grid.",
    )
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    return p.parse_args()


def _iter_jsonl(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def resolve_dataset_path(dataset: str) -> Path:
    p = Path(dataset).expanduser()
    if p.exists():
        return p
    candidate = Path.home() / ".gluonts" / "datasets" / dataset
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Dataset path not found: {dataset}")


def load_item_ids(data_dir: Path, split: str) -> List[str]:
    split_file = _resolve_split_file(data_dir, split)
    item_ids: List[str] = []
    for idx, obj in enumerate(_iter_jsonl(split_file)):
        item_id = obj.get("item_id")
        if item_id is None:
            item_id = f"series_{idx}"
        item_ids.append(str(item_id))
    return item_ids


def resolve_artifact_path(run_dir: Path, filename: str, sample_tag: Optional[str]) -> Path:
    history_root = run_dir / "samples_history"
    if history_root.is_dir():
        candidates = []
        for p in sorted(history_root.rglob(filename)):
            if sample_tag and sample_tag not in str(p.parent):
                continue
            candidates.append(p)
        if candidates:
            return candidates[-1]
    fallback = run_dir / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Unable to locate {filename} in {run_dir}.")


def load_samples(run_dir: Path, sample_tag: Optional[str]) -> Dict[str, np.ndarray]:
    path = resolve_artifact_path(run_dir, "samples.pt", sample_tag)
    data = torch.load(path, map_location="cpu", weights_only=False)
    payload: Dict[str, np.ndarray] = {
        "pred": data["samples"].cpu().numpy(),
        "truth": data["truth"].cpu().numpy(),
        "context": data.get("context").cpu().numpy() if data.get("context") is not None else None,
    }
    if data.get("window_stage_counts") is not None:
        payload["stage_counts"] = {k: int(v) for k, v in data["window_stage_counts"].items()}
    return payload


def stage_indices(stage_counts: Dict[str, int], stage: str, total: int) -> List[int]:
    val_n = int(stage_counts.get("val", 0))
    test_n = int(stage_counts.get("test", 0))
    if stage == "val":
        return list(range(0, val_n))
    if stage == "test":
        return list(range(val_n, val_n + test_n))
    if val_n + test_n > 0:
        return list(range(0, val_n + test_n))
    return list(range(0, total))


def safe_cov(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    n_assets = arr.shape[1] if arr.size else 0
    clean = arr[~np.isnan(arr).any(axis=1)]
    if clean.shape[0] <= 1:
        cov = np.zeros((n_assets, n_assets), dtype=float)
    else:
        cov = np.cov(clean, rowvar=False)
        cov = np.nan_to_num(cov, nan=0.0)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    if np.ndim(cov) == 1:
        cov = np.array([[float(cov[0])]], dtype=float)
    return cov


def safe_corr(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
    n_assets = arr.shape[1] if arr.size else 0
    clean = arr[~np.isnan(arr).any(axis=1)]
    if clean.shape[0] <= 1:
        corr = np.eye(n_assets, dtype=float)
    else:
        corr = np.corrcoef(clean, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
    if np.ndim(corr) == 0:
        corr = np.array([[float(corr)]], dtype=float)
    if np.ndim(corr) == 1:
        corr = np.array([[float(corr[0])]], dtype=float)
    corr = np.clip(corr, -1.0, 1.0)
    if corr.size:
        np.fill_diagonal(corr, 1.0)
    return corr


def flatten_pred(pred_seq: np.ndarray) -> np.ndarray:
    arr = pred_seq
    if arr.ndim == 4:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    if arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"Unexpected pred_seq shape {pred_seq.shape}")
    return arr


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


def compute_metric(metric_name: str, estimate: np.ndarray, reference: np.ndarray) -> float:
    diff = estimate - reference

    if metric_name in {"matrix_corr_fro", "matrix_cov_fro"}:
        return float(np.linalg.norm(diff, ord="fro"))
    if metric_name in {"matrix_corr_fro_rel", "matrix_cov_fro_rel"}:
        denom = np.linalg.norm(reference, ord="fro") + EPS
        return float(np.linalg.norm(diff, ord="fro") / denom)
    if metric_name == "matrix_cov_mse":
        return float(np.mean(diff * diff))
    if metric_name == "matrix_cov_mae":
        return float(np.mean(np.abs(diff)))
    if metric_name == "matrix_cov_diag_mape":
        ref_diag = np.diag(reference)
        est_diag = np.diag(estimate)
        return float(np.mean(np.abs((est_diag - ref_diag) / (np.abs(ref_diag) + EPS))))
    if metric_name == "matrix_corr_cross_mse":
        values = _flat_offdiag(diff)
        return float(np.mean(values ** 2))
    if metric_name == "matrix_corr_offdiag_pearson":
        x = _flat_offdiag(estimate)
        y = _flat_offdiag(reference)
        if x.size == 0 or y.size == 0 or np.std(x) < EPS or np.std(y) < EPS:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    if metric_name == "matrix_corr_offdiag_spearman":
        x = _flat_offdiag(estimate)
        y = _flat_offdiag(reference)
        if x.size == 0 or y.size == 0:
            return 0.0
        rx = _rank_avg_ties(x)
        ry = _rank_avg_ties(y)
        if np.std(rx) < EPS or np.std(ry) < EPS:
            return 0.0
        return float(np.corrcoef(rx, ry)[0, 1])
    if metric_name == "matrix_corr_sign_rate":
        x = np.sign(_flat_offdiag(estimate))
        y = np.sign(_flat_offdiag(reference))
        if x.size == 0 or y.size == 0:
            return 0.0
        return float(np.mean(x == y))
    if metric_name == "corr_wasserstein":
        x = _flat_offdiag(estimate)
        y = _flat_offdiag(reference)
        if x.size == 0 or y.size == 0:
            return 0.0
        x_sorted = np.sort(x)
        y_sorted = np.sort(y)
        min_len = min(x_sorted.size, y_sorted.size)
        return float(np.mean(np.abs(x_sorted[:min_len] - y_sorted[:min_len])))
    if metric_name == "eigen_wasserstein":
        est_sym = 0.5 * (estimate + estimate.T)
        ref_sym = 0.5 * (reference + reference.T)
        eig_est = np.linalg.eigvalsh(est_sym)
        eig_ref = np.linalg.eigvalsh(ref_sym)
        eig_est.sort()
        eig_ref.sort()
        min_len = min(eig_est.size, eig_ref.size)
        if min_len == 0:
            return 0.0
        return float(np.mean(np.abs(eig_est[:min_len] - eig_ref[:min_len])))
    raise ValueError(f"Unsupported metric: {metric_name}")


def _compact_metric_label(metric_name: str) -> str:
    return {
        "matrix_corr_fro": "F(corr Δ)",
        "matrix_cov_fro": "F(cov Δ)",
        "matrix_corr_fro_rel": "RelF(corr Δ)",
        "matrix_cov_fro_rel": "RelF(cov Δ)",
        "matrix_cov_mse": "MSE(cov Δ)",
        "matrix_cov_mae": "MAE(cov Δ)",
        "matrix_cov_diag_mape": "MAPE(diag)",
        "matrix_corr_cross_mse": "MSE(offdiag)",
        "matrix_corr_offdiag_pearson": "Pearson(offdiag)",
        "matrix_corr_offdiag_spearman": "Spearman(offdiag)",
        "matrix_corr_sign_rate": "SignRate(offdiag)",
        "corr_wasserstein": "W1(offdiag corr)",
        "eigen_wasserstein": "W1(eigvals)",
    }.get(metric_name, metric_name)


def _resolve_matrix_kind(metric_name: str, requested: str) -> str:
    if requested == "corr":
        return "corr"
    if requested == "cov":
        return "cov"
    return "cov" if metric_name.startswith("matrix_cov") else "corr"


def cov_to_corr(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    diag = np.sqrt(np.clip(np.diag(cov), eps, None))
    denom = np.outer(diag, diag)
    corr = np.zeros_like(cov, dtype=float)
    mask = denom > 0
    corr[mask] = cov[mask] / denom[mask]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def compute_window_matrices(sequence: np.ndarray, use_correlation: bool) -> np.ndarray:
    mat = safe_corr(sequence) if use_correlation else safe_cov(sequence)
    mat = np.nan_to_num(mat, nan=0.0)
    if np.ndim(mat) == 0:
        mat = np.array([[float(mat)]], dtype=float)
    if np.ndim(mat) == 1:
        mat = np.array([[float(mat[0])]], dtype=float)
    if use_correlation and mat.size:
        np.fill_diagonal(mat, 1.0)
    return mat


def _load_array(pt_path: Path, csv_path: Path) -> np.ndarray:
    if pt_path.exists():
        tensor = torch.load(pt_path, map_location="cpu", weights_only=False)
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return np.asarray(tensor)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, index_col=0)
            return df.to_numpy()
        except Exception:
            return np.loadtxt(csv_path, delimiter=",")
    raise FileNotFoundError(f"Neither {pt_path} nor {csv_path} exists.")


def _baseline_has_files(run_dir: Path, prefix: str) -> bool:
    return bool(list(run_dir.glob(f"{prefix}_win*_est.pt"))) or bool(list(run_dir.glob(f"{prefix}_win*_est.csv")))


def _resolve_baseline_runs(run_dir: Path, prefix: str) -> List[Path]:
    if _baseline_has_files(run_dir, prefix):
        return [run_dir]
    if not run_dir.is_dir():
        return []
    candidates = [p for p in run_dir.iterdir() if p.is_dir()]
    subruns = [p for p in candidates if _baseline_has_files(p, prefix)]
    return sorted(subruns, key=lambda p: p.name)


def _load_baseline_matrix(run_dir: Path, prefix: str, window_idx: int) -> np.ndarray:
    stub = f"{prefix}_win{window_idx:04d}"
    return _load_array(run_dir / f"{stub}_est.pt", run_dir / f"{stub}_est.csv")


def _baseline_matrix_for_window(run_dirs: Sequence[Path], prefix: str, window_idx: int) -> Optional[np.ndarray]:
    matrices = []
    for run_dir in run_dirs:
        try:
            matrices.append(_load_baseline_matrix(run_dir, prefix, window_idx))
        except FileNotFoundError:
            continue
    if not matrices:
        return None
    return np.mean(np.stack(matrices, axis=0), axis=0)


def _normalize_baseline_matrix(raw: np.ndarray, baseline_kind: str, matrix_kind: str) -> Optional[np.ndarray]:
    mat = np.asarray(raw, dtype=float)
    if baseline_kind == matrix_kind:
        if matrix_kind == "corr":
            mat = 0.5 * (mat + mat.T)
            np.fill_diagonal(mat, 1.0)
            mat = np.clip(mat, -1.0, 1.0)
        return mat
    if matrix_kind == "corr" and baseline_kind == "cov":
        return cov_to_corr(mat)
    return None


def pair_metric(seq: np.ndarray, idx_a: int, idx_b: int, kind: str) -> float:
    mat = safe_corr(seq) if kind == "corr" else safe_cov(seq)
    return float(mat[idx_a, idx_b])


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("_") or "pair"


def _pair_output_path(base: Path, pair_label: str) -> Path:
    if "{pair}" in str(base):
        return Path(str(base).replace("{pair}", _sanitize_filename(pair_label)))
    if base.suffix:
        return base.with_name(f"{base.stem}_{_sanitize_filename(pair_label)}{base.suffix}")
    return base.with_name(f"{base.name}_{_sanitize_filename(pair_label)}")


def _apply_legend_alpha(legend: Optional[plt.Legend], alpha: float) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_alpha(alpha)
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(alpha)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(p).expanduser().resolve() for p in args.pred_runs]
    if args.pred_sample_tags and len(args.pred_sample_tags) != len(run_dirs):
        raise SystemExit("--pred-sample-tags must match the number of --pred-runs.")
    if args.pred_labels and len(args.pred_labels) != len(run_dirs):
        raise SystemExit("--pred-labels must match the number of --pred-runs.")
    sample_tags = args.pred_sample_tags or [None] * len(run_dirs)
    sample_tags = [None if tag in {None, "-", ""} else tag for tag in sample_tags]
    run_labels = args.pred_labels or [p.name for p in run_dirs]

    baseline_specs = []
    if args.baseline:
        for spec in args.baseline:
            parts = [p for p in spec.split("|")]
            if len(parts) < 2:
                raise SystemExit("--baseline entries must be RUN|PREFIX|LABEL[|KIND].")
            run = Path(parts[0]).expanduser().resolve()
            prefix = parts[1]
            label = parts[2] if len(parts) > 2 and parts[2] else prefix
            kind = parts[3] if len(parts) > 3 and parts[3] else None
            if kind not in {None, "cov", "corr"}:
                raise SystemExit(f"Unsupported baseline kind '{kind}' (use 'cov' or 'corr').")
            baseline_specs.append((run, prefix, label, kind))

    asset_indices: Optional[List[int]] = None
    asset_labels: Optional[List[str]] = None
    if args.asset_indices:
        asset_indices = list(args.asset_indices)
    elif args.asset_ids:
        if not args.dataset:
            raise SystemExit("--asset-ids requires --dataset so item_id labels can be loaded.")
        ds_path = resolve_dataset_path(args.dataset)
        item_ids = load_item_ids(ds_path, args.dataset_split)
        asset_labels = item_ids
        index_map = {name: idx for idx, name in enumerate(item_ids)}
        missing = [name for name in args.asset_ids if name not in index_map]
        if missing:
            raise SystemExit(f"Unknown item_id labels: {', '.join(missing)}")
        asset_indices = [index_map[name] for name in args.asset_ids]
    else:
        raise SystemExit("Provide either --asset-indices or --asset-ids.")

    if len(asset_indices) < 2:
        raise SystemExit("Provide at least two assets for pairwise plotting.")
    pred_payloads = [load_samples(run_dir, tag) for run_dir, tag in zip(run_dirs, sample_tags)]

    # Use first run as reference for truth/context and stage counts.
    ref = pred_payloads[0]
    stage_counts = ref.get("stage_counts", {})
    total_windows = ref["pred"].shape[0]
    indices = stage_indices(stage_counts, args.stage, total_windows)
    val_n = int(stage_counts.get("val", 0))

    run_aug_flags = []
    for run_dir in run_dirs:
        run_aug = args.use_augmented_cov or (
            args.augmented_run_substr
            and any(sub in str(run_dir) for sub in args.augmented_run_substr)
        )
        run_aug_flags.append(bool(run_aug))

    use_metric_name = args.metric_name is not None
    if use_metric_name:
        matrix_kind = _resolve_matrix_kind(args.metric_name, args.matrix_kind)
        metric_label = _compact_metric_label(args.metric_name)
    else:
        matrix_kind = args.metric_kind
        metric_label = "corr" if args.metric_kind == "corr" else "cov"
    use_correlation = matrix_kind == "corr"

    baseline_entries = []
    for run, prefix, label, kind in baseline_specs:
        run_paths = _resolve_baseline_runs(run, prefix)
        if not run_paths:
            raise SystemExit(f"No baseline windows found under {run} with prefix '{prefix}'.")
        baseline_kind = kind or ("corr" if prefix.endswith("_corr") else "cov")
        baseline_entries.append(
            {
                "run_paths": run_paths,
                "prefix": prefix,
                "label": label,
                "kind": baseline_kind,
                "key": f"{run}|{prefix}|{label}",
            }
        )

    pairs = []
    for i in range(len(asset_indices)):
        for j in range(i + 1, len(asset_indices)):
            pairs.append((asset_indices[i], asset_indices[j]))

    x_label = "Window index (val+test)" if args.x_axis == "global" else f"Window index ({args.stage})"
    pair_results = []
    warned_baseline_kind = set()
    for idx_a, idx_b in pairs:
        x_vals = []
        truth_series = [] if not use_metric_name else None
        pred_series_list = [[] for _ in pred_payloads]
        baseline_series_list = [[] for _ in baseline_entries]
        for local_i, idx in enumerate(indices):
            if idx >= total_windows:
                break
            x_vals.append(local_i if args.x_axis == "stage" else idx)
            ctx_seq = ref["context"][idx] if ref.get("context") is not None else None
            fallback_idx = local_i if args.stage != "all" else idx
            if use_metric_name:
                truth_seq = ref["truth"][idx][:, [idx_a, idx_b]]
                truth_mat = compute_window_matrices(truth_seq, use_correlation)
                for run_idx, (payload, run_aug) in enumerate(zip(pred_payloads, run_aug_flags)):
                    pred_seq = payload["pred"][idx]
                    pred_flat = flatten_pred(pred_seq)
                    if run_aug:
                        if ctx_seq is None:
                            raise RuntimeError("No context available for augmented cov.")
                        pred_flat = np.concatenate([ctx_seq, pred_flat], axis=0)
                    pred_flat = pred_flat[:, [idx_a, idx_b]]
                    pred_mat = compute_window_matrices(pred_flat, use_correlation)
                    pred_series_list[run_idx].append(compute_metric(args.metric_name, pred_mat, truth_mat))
                for b_idx, baseline in enumerate(baseline_entries):
                    raw = _baseline_matrix_for_window(baseline["run_paths"], baseline["prefix"], idx)
                    if raw is None and fallback_idx != idx:
                        raw = _baseline_matrix_for_window(baseline["run_paths"], baseline["prefix"], fallback_idx)
                    value = np.nan
                    if raw is not None:
                        norm = _normalize_baseline_matrix(raw, baseline["kind"], matrix_kind)
                        if norm is None:
                            if baseline["key"] not in warned_baseline_kind:
                                print(
                                    f"[WARN] Baseline {baseline['label']} stored as {baseline['kind']} "
                                    f"cannot be converted to {matrix_kind}."
                                )
                                warned_baseline_kind.add(baseline["key"])
                        else:
                            sub = norm[np.ix_([idx_a, idx_b], [idx_a, idx_b])]
                            value = compute_metric(args.metric_name, sub, truth_mat)
                    baseline_series_list[b_idx].append(value)
            else:
                truth_seq = ref["truth"][idx]
                truth_series.append(pair_metric(truth_seq, idx_a, idx_b, args.metric_kind))
                for run_idx, (payload, run_aug) in enumerate(zip(pred_payloads, run_aug_flags)):
                    pred_seq = payload["pred"][idx]
                    pred_flat = flatten_pred(pred_seq)
                    if run_aug:
                        if ctx_seq is None:
                            raise RuntimeError("No context available for augmented cov.")
                        pred_flat = np.concatenate([ctx_seq, pred_flat], axis=0)
                    pred_series_list[run_idx].append(pair_metric(pred_flat, idx_a, idx_b, args.metric_kind))
                for b_idx, baseline in enumerate(baseline_entries):
                    raw = _baseline_matrix_for_window(baseline["run_paths"], baseline["prefix"], idx)
                    if raw is None and fallback_idx != idx:
                        raw = _baseline_matrix_for_window(baseline["run_paths"], baseline["prefix"], fallback_idx)
                    value = np.nan
                    if raw is not None:
                        norm = _normalize_baseline_matrix(raw, baseline["kind"], matrix_kind)
                        if norm is None:
                            if baseline["key"] not in warned_baseline_kind:
                                print(
                                    f"[WARN] Baseline {baseline['label']} stored as {baseline['kind']} "
                                    f"cannot be converted to {matrix_kind}."
                                )
                                warned_baseline_kind.add(baseline["key"])
                        else:
                            value = float(norm[idx_a, idx_b])
                    baseline_series_list[b_idx].append(value)

        label_a = asset_labels[idx_a] if asset_labels else f"asset_{idx_a}"
        label_b = asset_labels[idx_b] if asset_labels else f"asset_{idx_b}"
        pair_label = f"{label_a}_{label_b}"
        pair_results.append(
            {
                "pair": pair_label,
                "x": x_vals,
                "truth": truth_series,
                "preds": list(zip(run_labels, pred_series_list)),
                "baselines": list(zip([b["label"] for b in baseline_entries], baseline_series_list)),
            }
        )

        fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=args.dpi)
        if truth_series is not None:
            ax.plot(
                x_vals,
                truth_series,
                color="black",
                linewidth=1.6,
                label=f"Truth {metric_label}({label_a},{label_b})",
            )
        for run_label, series in zip(run_labels, pred_series_list):
            ax.plot(x_vals, series, linewidth=1.2, alpha=0.85, label=run_label)
        for baseline_label, series in zip([b["label"] for b in baseline_entries], baseline_series_list):
            ax.plot(x_vals, series, linewidth=1.2, alpha=0.85, label=baseline_label)

        if args.stage == "all" and args.show_test_boundary and val_n > 0:
            ax.axvline(val_n, color="gray", linestyle="--", linewidth=1.0)

        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label)
        if args.legend == "inside":
            leg = ax.legend(
                loc=args.legend_loc,
                frameon=False,
                ncol=args.legend_columns,
                fontsize=args.legend_fontsize,
            )
            _apply_legend_alpha(leg, args.legend_alpha)
        elif args.legend == "outside":
            leg = ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
                ncol=args.legend_columns,
                fontsize=args.legend_fontsize,
            )
            _apply_legend_alpha(leg, args.legend_alpha)
        if args.title:
            ax.set_title(args.title)

        out_path = _pair_output_path(args.output, pair_label)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight" if args.legend == "outside" else None)
        plt.close(fig)
        print(f"Saved plot to {out_path}")

    if args.combined_output and pair_results:
        cols = max(1, int(args.combined_cols))
        rows = int(np.ceil(len(pair_results) / cols))
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=(args.combined_size[0] * cols, args.combined_size[1] * rows),
            squeeze=False,
            dpi=args.dpi,
        )
        for idx, result in enumerate(pair_results):
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]
            if result["truth"] is not None:
                ax.plot(result["x"], result["truth"], color="black", linewidth=1.4, label="Truth")
            for run_label, series in result["preds"]:
                ax.plot(result["x"], series, linewidth=1.1, alpha=0.85, label=run_label)
            for baseline_label, series in result["baselines"]:
                ax.plot(result["x"], series, linewidth=1.1, alpha=0.85, label=baseline_label)
            ax.set_title(result["pair"])
            if r == rows - 1:
                ax.set_xlabel(x_label)
            if c == 0:
                ax.set_ylabel(metric_label)
            if args.combined_legend == "first" and idx == 0:
                leg = ax.legend(
                    loc=args.legend_loc,
                    frameon=False,
                    ncol=args.legend_columns,
                    fontsize=args.legend_fontsize,
                )
                _apply_legend_alpha(leg, args.legend_alpha)
        for idx in range(len(pair_results), rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis("off")
        args.combined_output.parent.mkdir(parents=True, exist_ok=True)
        if args.combined_legend == "outside":
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                leg = fig.legend(
                    handles,
                    labels,
                    loc="center right",
                    frameon=False,
                    ncol=args.legend_columns,
                    fontsize=args.legend_fontsize,
                )
                _apply_legend_alpha(leg, args.legend_alpha)
                fig.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            fig.tight_layout()
        fig.savefig(
            args.combined_output,
            dpi=args.dpi,
            bbox_inches="tight" if args.combined_legend == "outside" else None,
        )
        plt.close(fig)
        print(f"Saved combined plot to {args.combined_output}")


if __name__ == "__main__":
    main()
