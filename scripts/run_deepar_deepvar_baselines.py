#!/usr/bin/env python3
"""
Train GluonTS DeepAR / DeepVAR baselines on the FX panel and
export window-level covariance estimates aligned with the diffusion outputs.

Design:
- Reuse ConditionalGluonTSJsonDatamodule to generate (context, target) windows.
- Train DeepAR (per-asset) and DeepVAR (multivariate) on train windows.
- For each test window, draw Monte-Carlo samples, compute:
    * per-window forecast mean (for series metrics),
    * per-window asset–asset covariance from samples (for structural metrics).
- Save outputs under:
    outputs/baselines/{deepar,deepvar}/conditional/{timestamp}/*
  with filenames compatible with plot_correlation_mean.py / plot_correlation_table.py,
  e.g. deepar_win0000_est.pt, deepar_win0000_truth.pt.

Note:
- This script requires GluonTS with MXNet back-end (gluonts.mx + mxnet).
  If imports fail, install appropriate versions before running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import warnings

import numpy as np

# Compatibility shims for older MXNet/gluonts code paths on NumPy>=1.24
if not hasattr(np, "bool"):  # NumPy 1.24+ removed np.bool alias
    np.bool = np.bool_  # type: ignore[attr-defined]
if not hasattr(np, "object"):  # defensive: some MXNet builds use np.object
    np.object = np.object_  # type: ignore[attr-defined]
if not hasattr(np, "int"):  # defensive: legacy alias
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):  # defensive: legacy alias
    np.float = float  # type: ignore[attr-defined]
import torch

# Silence noisy pandas / GluonTS FutureWarnings that do not affect results
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure local src/ is on the Python path so that cfdiff imports work
_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cfdiff.dataloaders.conditional_gluonts import (
    ConditionalGluonTSJsonDatamodule,
    _load_gluonts_like,
    _resolve_split_file,
)
from cfdiff.eval.matrix import compute_covariance_metrics
from cfdiff.eval.series import compute_series_metrics


def _drop_nan_rows(
    arr: np.ndarray,
    fill_assets: int | None = None,
    nan_policy: str = "drop",
    context: str = "",
) -> np.ndarray:
    """Remove rows containing non-finite values; optionally return a single zero row if empty."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D array; got shape {arr.shape}")
    nan_policy = str(nan_policy or "drop").lower()
    mask = np.isfinite(arr).all(axis=1)
    if mask.all():
        return arr
    if nan_policy in {"raise", "error"}:
        bad = int((~np.isfinite(arr)).sum())
        prefix = f"{context} " if context else ""
        raise ValueError(f"{prefix}contains {bad} non-finite values but nan_policy='{nan_policy}'.")
    if nan_policy != "drop":
        raise ValueError(f"Unsupported nan_policy='{nan_policy}'")
    cleaned = arr[mask]
    if cleaned.size == 0 and fill_assets is not None:
        cleaned = np.zeros((1, fill_assets), dtype=float)
    return cleaned
def _default_data_dir() -> Path:
    env = os.environ.get("CFDIFF_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "data"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Root of cleaned GluonTS dataset (defaults to CFDIFF_DATA_DIR or ./data).",
    )
    ap.add_argument("--context-len", type=int, default=30, help="Context length C for windows.")
    ap.add_argument("--pred-len", type=int, default=15, help="Prediction length P for windows.")
    ap.add_argument("--stride", type=int, default=1, help="Stride for sliding windows.")
    ap.add_argument("--val-ratio", type=float, default=0.3, help="Validation ratio for datamodule.")
    ap.add_argument(
        "--train-val-gap",
        type=int,
        default=-1,
        help="Gap (in windows) between train and val splits; -1 uses auto no-overlap gap.",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        choices=["deepar", "deepvar", "both"],
        default=["deepar", "deepvar"],
        help="Which baselines to train/evaluate.",
    )
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs for DeepAR/DeepVAR.")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for GluonTS Trainer.")
    ap.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of Monte-Carlo samples per forecast for covariance estimation.",
    )
    ap.add_argument(
        "--nan-policy",
        choices=["drop", "raise"],
        default="drop",
        help="How to handle non-finite values during covariance export (default: drop).",
    )
    ap.add_argument(
        "--include-val",
        dest="include_val",
        action="store_true",
        default=True,
        help="Include validation windows in evaluation outputs (default).",
    )
    ap.add_argument(
        "--no-include-val",
        dest="include_val",
        action="store_false",
        help="Exclude validation windows; only test windows are saved.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "baselines",
        help="Root directory for baseline outputs.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag inserted into output path (for bookkeeping).",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs. If >1, results are grouped under batch-<timestamp>.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; if repeats>1, seed is incremented per repeat.",
    )
    return ap.parse_args()


def _prepare_datamodule(
    data_dir: Path,
    context_len: int,
    pred_len: int,
    stride: int,
    val_ratio: float,
    train_val_gap: int,
) -> ConditionalGluonTSJsonDatamodule:
    dm = ConditionalGluonTSJsonDatamodule(
        data_dir=str(data_dir),
        batch_size=32,
        context_len=context_len,
        pred_len=pred_len,
        stride=stride,
        val_ratio=val_ratio,
        train_val_gap=train_val_gap,
        val_test_gap=-1,
        standardize=True,
        num_workers=0,
        pin_memory=False,
        fourier_transform=False,
        estimate_sliding_cov=False,
        align_tail_windows=True,
    )
    dm.prepare_data()
    dm.setup()
    return dm


def _build_training_windows(dm: ConditionalGluonTSJsonDatamodule) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate context+target for train windows into a (N, L, A) array."""
    ds = dm.ds_train
    assert ds is not None
    full_windows: List[np.ndarray] = []
    for sample in ds:
        ctx = sample["context_time"].detach().cpu().numpy()  # (C, A)
        tgt = sample["target_time"].detach().cpu().numpy()  # (P, A)
        full = np.concatenate([ctx, tgt], axis=0)  # (L, A)
        full_windows.append(full.astype(np.float32))
    arr = np.stack(full_windows, axis=0)  # (N, L, A)
    return arr, arr[0].shape  # (L, A)


def _require_gluonts_mx():
    try:
        from gluonts.mx import Trainer  # noqa: F401
        from gluonts.mx.model.deepar import DeepAREstimator  # noqa: F401
        from gluonts.mx.model.deepvar import DeepVAREstimator  # noqa: F401
        from gluonts.dataset.common import ListDataset  # noqa: F401
        import pandas as pd  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "This script requires GluonTS MXNet models and mxnet.\n"
            "Install compatible versions, e.g.:\n"
            "  pip install \"gluonts[mxnet]\" mxnet\n"
            f"Import error: {exc}"
        )


def _train_deepar(
    train_windows: np.ndarray,
    freq: str,
    prediction_length: int,
    context_length: int,
    epochs: int,
    batch_size: int,
):
    from gluonts.mx import Trainer
    from gluonts.mx.model.deepar import DeepAREstimator
    from gluonts.dataset.common import ListDataset
    import pandas as pd

    N, L, A = train_windows.shape
    records = []
    start = pd.Timestamp("2000-01-01")
    for i in range(N):
        window = train_windows[i]  # (L, A)
        for a in range(A):
            series = window[:, a].astype("float32")
            records.append({"start": start, "target": series})

    train_ds = ListDataset(records, freq=freq)
    # GluonTS MX Trainer in 0.16.x does not accept batch_size; that is handled
    # inside the data loader. We only control the number of epochs here.
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        trainer=Trainer(epochs=epochs),
    )
    predictor = estimator.train(train_ds)
    return predictor


def _train_deepvar(
    data_dir: Path,
    freq: str,
    prediction_length: int,
    context_length: int,
    epochs: int,
):
    from gluonts.mx import Trainer
    from gluonts.mx.model.deepvar import DeepVAREstimator
    from gluonts.dataset.common import ListDataset
    import pandas as pd
    from gluonts.dataset.multivariate_grouper import MultivariateGrouper

    # Load full training panel and construct multivariate dataset
    train_path = _resolve_split_file(data_dir, "train")
    Xtr = _load_gluonts_like(train_path)  # (T, A)
    T, A = Xtr.shape
    records = []
    start = pd.Timestamp("2000-01-01")

    # Univariate dataset: one series per asset
    for a in range(A):
        series = Xtr[:, a].astype("float32")
        records.append({"start": start, "target": series})

    univar_ds = ListDataset(records, freq=freq)
    grouper = MultivariateGrouper(max_target_dim=A)
    train_ds = grouper(univar_ds)

    estimator = DeepVAREstimator(
        target_dim=A,
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        trainer=Trainer(epochs=epochs),
    )
    predictor = estimator.train(train_ds)
    return predictor


def _evaluate_deepar(
    predictor,
    ds,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (samples_all, truth_all, context_all).

    samples_all: (num_windows, num_samples, P, A)
    truth_all:   (num_windows, P, A)
    context_all: (num_windows, C, A)
    """
    from gluonts.dataset.common import ListDataset
    import pandas as pd

    assert ds is not None
    num_windows = len(ds)

    # Determine asset count and pred_len from first sample
    first = ds[0]
    ctx0 = first["context_time"].detach().cpu().numpy()
    tgt0 = first["target_time"].detach().cpu().numpy()
    C, A = ctx0.shape
    P = tgt0.shape[0]

    records = []
    index_map: List[Tuple[int, int]] = []
    start = pd.Timestamp("2000-01-01")
    for w in range(num_windows):
        sample = ds[w]
        ctx = sample["context_time"].detach().cpu().numpy()  # (C, A)
        for a in range(A):
            series = ctx[:, a].astype("float32")
            records.append({"start": start, "target": series})
            index_map.append((w, a))

    test_ds = ListDataset(records, freq="B")
    forecasts = list(predictor.predict(test_ds, num_samples=num_samples))

    samples_all = np.zeros((num_windows, num_samples, P, A), dtype=np.float32)
    truth_all = np.zeros((num_windows, P, A), dtype=np.float32)
    context_all = np.zeros((num_windows, C, A), dtype=np.float32)
    for w in range(num_windows):
        sample = ds[w]
        truth_all[w] = sample["target_time"].detach().cpu().numpy()
        context_all[w] = sample["context_time"].detach().cpu().numpy()

    for rec_idx, fc in enumerate(forecasts):
        w, a = index_map[rec_idx]
        s = fc.samples  # (num_samples, P)
        samples_all[w, :, :, a] = s.astype(np.float32)

    return samples_all, truth_all, context_all


def _evaluate_deepvar(
    predictor,
    ds,
    num_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (samples_all, truth_all, context_all) for DeepVAR.

    samples_all: (num_windows, num_samples, P, A)
    truth_all:   (num_windows, P, A)
    context_all: (num_windows, C, A)
    """
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.multivariate_grouper import MultivariateGrouper
    import pandas as pd

    assert ds is not None
    num_windows = len(ds)

    first = ds[0]
    ctx0 = first["context_time"].detach().cpu().numpy()
    tgt0 = first["target_time"].detach().cpu().numpy()
    C, A = ctx0.shape
    P = tgt0.shape[0]

    samples_all = np.zeros((num_windows, num_samples, P, A), dtype=np.float32)
    truth_all = np.zeros((num_windows, P, A), dtype=np.float32)
    context_all = np.zeros((num_windows, C, A), dtype=np.float32)

    start = pd.Timestamp("2000-01-01")

    for w in range(num_windows):
        sample = ds[w]
        ctx = sample["context_time"].detach().cpu().numpy()  # (C, A)
        truth_all[w] = sample["target_time"].detach().cpu().numpy()
        context_all[w] = ctx

        # Build univariate dataset for this window then group to multivariate
        records = []
        for a in range(A):
            series = ctx[:, a].astype("float32")
            records.append({"start": start, "target": series})

        univar_ds = ListDataset(records, freq="B")
        grouper = MultivariateGrouper(max_target_dim=A)
        test_ds = grouper(univar_ds)

        fc = next(predictor.predict(test_ds, num_samples=num_samples))
        s = fc.samples.astype(np.float32)
        # Expected shapes: (num_samples, P, A) or (num_samples, A, P)
        if s.ndim == 3 and s.shape[1] == P and s.shape[2] == A:
            samples_all[w] = s
        elif s.ndim == 3 and s.shape[1] == A and s.shape[2] == P:
            samples_all[w] = np.transpose(s, (0, 2, 1))
        else:
            raise RuntimeError(f"Unexpected DeepVAR samples shape: {s.shape}")

    return samples_all, truth_all, context_all


def _save_window_covariances(
    method_name: str,
    samples_all: np.ndarray,
    truth_all: np.ndarray,
    context_all: np.ndarray,
    run_dir: Path,
    stage_counts: Dict[str, int],
    nan_policy: str = "drop",
) -> Dict[str, float]:
    """Save per-window covariance estimates and return aggregated matrix/series metrics."""
    nan_policy = str(nan_policy or "drop").lower()
    if nan_policy not in {"drop", "raise", "error"}:
        raise ValueError(f"Unsupported nan_policy='{nan_policy}'")
    if nan_policy in {"raise", "error"}:
        for label, arr in (("samples", samples_all), ("truth", truth_all)):
            if not np.isfinite(arr).all():
                bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"{label} contains {bad} non-finite values but nan_policy='{nan_policy}'."
                )

    num_windows, num_samples, P, A = samples_all.shape
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save stage_counts for plotting scripts
    with (run_dir / "stage_counts.json").open("w", encoding="utf-8") as f:
        json.dump(stage_counts, f, indent=2)

    def _compute_metrics(samples_slice: np.ndarray, truth_slice: np.ndarray) -> Dict[str, Dict[str, float]]:
        pred_mean = samples_slice.mean(axis=1)  # (n, P, A)
        truth_tensor = torch.from_numpy(truth_slice.astype(np.float32))
        pred_tensor = torch.from_numpy(pred_mean.astype(np.float32))

        series_metrics = compute_series_metrics(truth_tensor, pred_tensor)
        matrix_metrics = compute_covariance_metrics(truth_tensor, pred_tensor)

        # Additional series-level metrics: ND and CRPS
        eps = 1e-8
        diff = pred_tensor - truth_tensor
        nd = torch.sum(diff.abs()) / (torch.sum(truth_tensor.abs()) + eps)
        series_metrics["nd"] = float(nd.item())

        if samples_slice.shape[1] >= 2:
            samples_mc = torch.from_numpy(samples_slice.astype(np.float32))  # (B, S, P, A)
            samples_mc = samples_mc.permute(1, 0, 2, 3)  # (S, B, P, A)
            truth_exp = truth_tensor.unsqueeze(0)
            term1 = torch.mean(torch.abs(samples_mc - truth_exp), dim=0)
            pairwise = torch.abs(samples_mc.unsqueeze(0) - samples_mc.unsqueeze(1)).mean(dim=(0, 1))
            crps = term1 - 0.5 * pairwise
            series_metrics["crps"] = float(crps.mean().item())

        return series_metrics, matrix_metrics, pred_mean

    # Overall metrics
    series_metrics, matrix_metrics, pred_mean = _compute_metrics(samples_all, truth_all)

    # Per-window covariance objects (used by correlation plotting scripts)
    for w in range(num_windows):
        flat_samples = samples_all[w].reshape(num_samples * P, A)
        flat_samples = _drop_nan_rows(
            flat_samples, fill_assets=A, nan_policy=nan_policy, context="Sample rows"
        )
        truth_rows = _drop_nan_rows(
            truth_all[w], fill_assets=A, nan_policy=nan_policy, context="Truth rows"
        )
        if flat_samples.shape[0] <= 1:
            cov_est = np.zeros((A, A), dtype=np.float64)
        else:
            cov_est = np.cov(flat_samples, rowvar=False)
        if truth_rows.shape[0] <= 1:
            cov_truth = np.zeros((A, A), dtype=np.float64)
        else:
            cov_truth = np.cov(truth_rows, rowvar=False)
        if nan_policy == "drop":
            cov_est = np.nan_to_num(cov_est, nan=0.0, posinf=0.0, neginf=0.0)
            cov_truth = np.nan_to_num(cov_truth, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            if not np.isfinite(cov_est).all():
                raise ValueError(
                    f"cov_est contains non-finite values but nan_policy='{nan_policy}'."
                )
            if not np.isfinite(cov_truth).all():
                raise ValueError(
                    f"cov_truth contains non-finite values but nan_policy='{nan_policy}'."
                )

        stub = f"{method_name}_win{w:04d}"
        torch.save(torch.from_numpy(cov_est.astype(np.float32)), run_dir / f"{stub}_est.pt")
        torch.save(torch.from_numpy(cov_truth.astype(np.float32)), run_dir / f"{stub}_truth.pt")

    # Also save aggregated series/matrix metrics as a compact JSON for tables
    agg = {"series": series_metrics, "matrix": matrix_metrics}
    with (run_dir / f"{method_name}_aggregate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    # Stage-wise metrics (val/test)
    offset = 0
    for stage in ("val", "test"):
        n = int(stage_counts.get(stage, 0) or 0)
        if n <= 0:
            continue
        s_slice = samples_all[offset : offset + n]
        t_slice = truth_all[offset : offset + n]
        stage_series, stage_matrix, _ = _compute_metrics(s_slice, t_slice)
        stage_agg = {"series": stage_series, "matrix": stage_matrix}
        with (run_dir / f"{method_name}_aggregate_metrics_{stage}.json").open("w", encoding="utf-8") as f:
            json.dump(stage_agg, f, indent=2)
        offset += n

    # Persist mean predictions/samples for downstream panel plots
    pred_mean_np = pred_mean.astype(np.float32)
    torch.save(
        {
            "samples": pred_mean_np,  # (num_windows, P, A)
            "truth": truth_all.astype(np.float32),
            "context": context_all.astype(np.float32),
            "fourier_transform": False,
            "window_stage_counts": stage_counts,
        },
        run_dir / "samples.pt",
    )

    return {**{f"series_{k}": v for k, v in series_metrics.items()}, **{f"matrix_{k}": v for k, v in matrix_metrics.items()}}


def main() -> None:
    args = _parse_args()

    # Normalise methods argument
    methods: Sequence[str]
    if "both" in args.methods:
        methods = ["deepar", "deepvar"]
    else:
        methods = list(dict.fromkeys(args.methods))

    # Data root
    if args.data_dir is not None:
        data_dir = args.data_dir.expanduser().resolve()
    else:
        data_dir = _default_data_dir().resolve()
    if not data_dir.exists():
        raise SystemExit(f"data_dir does not exist: {data_dir}")

    # Ensure GluonTS MXNet is available
    _require_gluonts_mx()

    # Prepare datamodule and training windows
    dm = _prepare_datamodule(
        data_dir=data_dir,
        context_len=args.context_len,
        pred_len=args.pred_len,
        stride=args.stride,
        val_ratio=args.val_ratio,
        train_val_gap=args.train_val_gap,
    )
    train_windows, _ = _build_training_windows(dm)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    freq = "B"
    base_seed = int(args.seed or 0)

    for method in methods:
        print(f"[INFO] Training and evaluating {method} baseline...")
        # Output directory aligned with other baselines:
        # outputs/baselines/{method}/conditional/{tag}/[batch-<ts>/]run-<ts>-rXX
        base = args.output_root / method / "conditional"
        if args.tag:
            base = base / str(args.tag)
        batch_dir = None
        if args.repeats and args.repeats > 1:
            batch_dir = (base / f"batch-{timestamp}").resolve()
            batch_dir.mkdir(parents=True, exist_ok=True)

        def _run_dir_for(idx: int) -> Path:
            if batch_dir:
                return batch_dir / f"{timestamp}-r{idx+1:02d}"
            return (base / timestamp).resolve()

        for rep in range(max(1, args.repeats)):
            # re-seed per repeat
            seed_val = base_seed + rep
            np.random.seed(seed_val)
            try:
                import random

                random.seed(seed_val)
            except Exception:
                pass
            try:
                import mxnet as mx

                mx.random.seed(seed_val)
            except Exception:
                pass

            if method == "deepar":
                predictor = _train_deepar(
                    train_windows=train_windows,
                    freq=freq,
                    prediction_length=args.pred_len,
                    context_length=args.context_len,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                )
                samples_test, truth_test, context_test = _evaluate_deepar(
                    predictor, dm.ds_test, num_samples=args.num_samples
                )
            elif method == "deepvar":
                predictor = _train_deepvar(
                    data_dir=data_dir,
                    freq=freq,
                    prediction_length=args.pred_len,
                    context_length=args.context_len,
                    epochs=args.epochs,
                )
                samples_test, truth_test, context_test = _evaluate_deepvar(
                    predictor, dm.ds_test, num_samples=args.num_samples
                )
            else:
                raise ValueError(f"Unknown method {method}")

            if args.include_val:
                if method == "deepar":
                    samples_val, truth_val, context_val = _evaluate_deepar(
                        predictor, dm.ds_val, num_samples=args.num_samples
                    )
                else:
                    samples_val, truth_val, context_val = _evaluate_deepvar(
                        predictor, dm.ds_val, num_samples=args.num_samples
                    )
                samples_all = np.concatenate([samples_val, samples_test], axis=0)
                truth_all = np.concatenate([truth_val, truth_test], axis=0)
                context_all = np.concatenate([context_val, context_test], axis=0)
                stage_counts = {"val": samples_val.shape[0], "test": samples_test.shape[0]}
            else:
                samples_all = samples_test
                truth_all = truth_test
                context_all = context_test
                stage_counts = {"val": 0, "test": samples_test.shape[0]}

            run_dir = _run_dir_for(rep)
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics = _save_window_covariances(
                method,
                samples_all,
                truth_all,
                context_all,
                run_dir,
                stage_counts,
                nan_policy=args.nan_policy,
            )
            print(f"[INFO] Saved {method} baseline to {run_dir}")
            print("[INFO] Aggregate metrics:")
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
