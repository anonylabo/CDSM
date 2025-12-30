#!/usr/bin/env python3
"""
Compute cross-signal energy spectra for a GluonTS-style dataset.

This mirrors the notebook logic:
1) Build sliding windows from a standardized multivariate series.
2) Compute per-window power spectra (|FFT|^2).
3) For each pair of windows (i, j), compute the cross-energy spectrum as
   PSD_i * PSD_j, normalize by total cross-energy, and then normalize by
   sqrt(PSD_i_norm * PSD_j_norm).
4) Average across all sampled pairs to obtain the mean cross-signal spectrum.

Modes:
- window: cross-energy across window pairs (original notebook logic).
- window-mean: average normalized per-window power spectrum (no pairs).
- asset-pair: per-asset-pair cross-energy curves (averaged over windows).
- asset-matrix: per-frequency asset-by-asset matrices (averaged over windows).
- asset-mean: average curve across asset pairs (averaged over windows).

Outputs a CSV (and optionally a plot) under the project assets directory by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from cfdiff.dataloaders.conditional_gluonts import (
    _forward_fill_numpy,
    _load_gluonts_like,
    _resolve_split_file,
)
from cfdiff.utils.windowing import compute_window_positions


EPS = 1e-15
X_LABEL = r"Frequency $\omega_k/\omega_{\mathrm{Nyq}}$"
Y_LABEL = "Spectral Density"


def _format_mode_label(mode: str, source: Optional[str] = None) -> str:
    if source:
        return f"Mode: {mode} | {source}"
    return f"Mode: {mode}"


def _default_data_dir() -> Path:
    env = os.environ.get("CFDIFF_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".gluonts" / "datasets" / "exchange_rate_clean"


def _normalize_source_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    if text.lower().startswith("source"):
        return text
    return f"Source: {text}"


def _infer_source_text(args: argparse.Namespace, data_dir: Optional[Path]) -> Optional[str]:
    if not args.plot_source:
        return None
    if args.source_text is not None:
        return _normalize_source_text(args.source_text)
    if args.data_array is not None:
        data_path = args.data_array.expanduser().resolve()
        if args.data_array_key:
            return _normalize_source_text(f"{data_path.name} [{args.data_array_key}]")
        return _normalize_source_text(data_path.name)
    if data_dir is not None:
        return _normalize_source_text(f"{data_dir.name} ({args.split})")
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cross-mode",
        choices=[
            "window",
            "window-mean",
            "window-pair",
            "asset-pair",
            "asset-matrix",
            "asset-mean",
        ],
        default="window",
        help="Cross-signal mode: window pairs, window-mean, or asset-level summaries.",
    )
    ap.add_argument(
        "--data-array",
        type=Path,
        default=None,
        help=(
            "Optional array/tensor file (.npy/.npz/.pt) containing data shaped (T, A) or (B, T, A). "
            "If provided, this overrides --data-dir."
        ),
    )
    ap.add_argument(
        "--data-array-key",
        type=str,
        default=None,
        help=(
            "Optional key for .npz or dict-like .pt inputs. If omitted, uses --split when available "
            "or the only array in the file."
        ),
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root (GluonTS JSONL). Defaults to CFDIFF_DATA_DIR or ~/.gluonts/datasets/exchange_rate_clean.",
    )
    ap.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Split to load for windowing.",
    )
    ap.add_argument("--context-len", type=int, default=60, help="Context length for windows.")
    ap.add_argument("--pred-len", type=int, default=30, help="Prediction length for windows.")
    ap.add_argument(
        "--window-len",
        type=int,
        default=None,
        help="Optional explicit window length (overrides context_len + pred_len).",
    )
    ap.add_argument("--stride", type=int, default=1, help="Stride between windows.")
    ap.add_argument(
        "--align-tail-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align the final window to the end of the series.",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help=(
            "If <0, drop the last |val_ratio| of windows (plus an optional gap) to mirror datamodule "
            "train/val splitting. Uses the remaining windows for analysis."
        ),
    )
    ap.add_argument(
        "--train-val-gap",
        type=int,
        default=-1,
        help="Gap (in window indices) between train and val when val_ratio < 0. -1 uses ceil(L/stride)-1.",
    )
    ap.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize per-asset before windowing.",
    )
    ap.add_argument(
        "--nan-policy",
        choices=["raise", "ffill"],
        default="raise",
        help="How to handle non-finite values in the series.",
    )
    ap.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap on the number of windows (randomly sampled).",
    )
    ap.add_argument("--window-seed", type=int, default=0, help="Random seed for window sampling.")
    ap.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional cap on the number of window pairs. <=0 uses all pairs (window/window-pair).",
    )
    ap.add_argument("--pair-seed", type=int, default=0, help="Random seed for pair sampling.")
    ap.add_argument(
        "--pair-batch-size",
        type=int,
        default=256,
        help="Batch size for pairwise accumulation.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for FFTs (cpu or cuda).",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (mode-dependent default under assets/).",
    )
    ap.add_argument(
        "--output-matrix",
        type=Path,
        default=None,
        help="Output NPZ path for asset-matrix mode (default: assets/cross_signal_energy_asset_matrix.npz).",
    )
    ap.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Output plot path (mode-dependent default under assets/).",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot output.",
    )
    ap.add_argument(
        "--include-zero-freq",
        action="store_true",
        help="Include the zero-frequency bin in outputs (default: drop it).",
    )
    ap.add_argument(
        "--log-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot the y-axis on a log scale.",
    )
    ap.add_argument(
        "--plot-pairs",
        nargs="+",
        default=None,
        help="Pairs to plot for asset-pair/window-pair mode, format i,j (0-based).",
    )
    ap.add_argument(
        "--plot-max-pairs",
        type=int,
        default=0,
        help="If >0 and --plot-pairs not set, randomly sample this many pairs for plotting.",
    )
    ap.add_argument("--plot-pair-seed", type=int, default=0, help="Random seed for plot pair sampling.")
    ap.add_argument(
        "--plot-legend",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show legend for asset-pair plots (default: auto).",
    )
    ap.add_argument(
        "--pair-plot-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Figure size for asset-pair plot in inches (e.g., 10 8).",
    )
    ap.add_argument(
        "--skip-pair-plot",
        action="store_true",
        help="Skip the overlay line plot for pair modes (still allows grids/other plots).",
    )
    ap.add_argument(
        "--plot-pair-grid",
        action="store_true",
        help="Plot a grid of small multiples (one subplot per asset pair).",
    )
    ap.add_argument(
        "--pair-grid-cols",
        type=int,
        default=7,
        help="Number of columns in the asset-pair grid.",
    )
    ap.add_argument(
        "--pair-grid-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1.6, 1.2),
        help="Per-panel size (width, height) for the asset-pair grid.",
    )
    ap.add_argument(
        "--pair-grid-output",
        type=Path,
        default=None,
        help="Output path for the asset-pair grid plot.",
    )
    ap.add_argument(
        "--plot-heatmap",
        action="store_true",
        help="Plot asset-by-asset heatmaps for selected frequency bands.",
    )
    ap.add_argument(
        "--heatmap-bands",
        nargs="+",
        type=float,
        default=[0.0, 0.33, 0.66, 1.0],
        help="Frequency band boundaries for summaries/heatmaps (e.g., 0 0.33 0.66 1.0).",
    )
    ap.add_argument(
        "--heatmap-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use log color scale for heatmaps.",
    )
    ap.add_argument(
        "--heatmap-shared-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Share the color scale across heatmaps.",
    )
    ap.add_argument(
        "--heatmap-output-prefix",
        type=Path,
        default=None,
        help="Output prefix for heatmap PNGs (default: assets/cross_signal_energy_heatmap).",
    )
    ap.add_argument(
        "--plot-quantile-band",
        action="store_true",
        help="Plot quantile band across all asset pairs.",
    )
    ap.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 0.9],
        help="Quantiles to plot for the band (e.g., 0.1 0.5 0.9).",
    )
    ap.add_argument(
        "--quantile-output",
        type=Path,
        default=None,
        help="Output path for the quantile band plot.",
    )
    ap.add_argument(
        "--plot-topk",
        action="store_true",
        help="Plot top-K asset pairs by a summary metric.",
    )
    ap.add_argument("--topk-k", type=int, default=10, help="Number of pairs to plot for top-K.")
    ap.add_argument(
        "--topk-metric",
        choices=["mean", "max", "integral"],
        default="mean",
        help="Metric used to rank top-K pairs.",
    )
    ap.add_argument(
        "--topk-output",
        type=Path,
        default=None,
        help="Output path for the top-K plot.",
    )
    ap.add_argument(
        "--topk-legend",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show legend for top-K plot (default: auto).",
    )
    ap.add_argument(
        "--plot-asset-panels",
        action="store_true",
        help="Plot per-asset panels with top partners.",
    )
    ap.add_argument(
        "--asset-panels-topk",
        type=int,
        default=3,
        help="Number of partners to plot per asset.",
    )
    ap.add_argument(
        "--asset-panels-metric",
        choices=["mean", "max", "integral"],
        default="mean",
        help="Metric used to select top partners per asset.",
    )
    ap.add_argument(
        "--asset-panels-cols",
        type=int,
        default=4,
        help="Number of columns in the asset panel grid.",
    )
    ap.add_argument(
        "--asset-panels-size",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(3.0, 2.2),
        help="Per-panel size (width, height) for asset panels.",
    )
    ap.add_argument(
        "--asset-panels-output",
        type=Path,
        default=None,
        help="Output path for the asset panel plot.",
    )
    ap.add_argument(
        "--asset-panels-legend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show legends in asset panels.",
    )
    ap.add_argument(
        "--plot-band-summary",
        action="store_true",
        help="Plot average cross-energy per frequency band.",
    )
    ap.add_argument(
        "--band-summary-output",
        type=Path,
        default=None,
        help="Output path for the band summary plot.",
    )
    ap.add_argument(
        "--band-summary-error",
        choices=["none", "std"],
        default="none",
        help="Error bar type for band summary.",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Plot DPI.")
    ap.add_argument(
        "--plot-source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate plots with data source (default: true).",
    )
    ap.add_argument(
        "--source-text",
        type=str,
        default=None,
        help=(
            "Override the auto source label for plots. "
            "Use an empty string to suppress the annotation."
        ),
    )
    return ap.parse_args()


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (X - mu) / sigma


def _build_windows(X: np.ndarray, window_len: int, stride: int, align_end: bool) -> np.ndarray:
    idx = compute_window_positions(X.shape[0], window_len, stride, align_end=align_end)
    return np.stack([X[s:e, :] for (s, e) in idx], axis=0).astype(np.float32, copy=False)


def _sample_windows(
    windows: np.ndarray,
    max_windows: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = windows.shape[0]
    if max_windows is None or max_windows <= 0 or n <= max_windows:
        return windows, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_windows, replace=False)
    return windows[idx], idx


def _power_spectrum(windows: torch.Tensor) -> torch.Tensor:
    freq = torch.fft.rfft(windows, dim=1, norm="ortho")
    return freq.real.square() + freq.imag.square()


def _pair_batches_all(n: int, batch_size: int) -> Iterable[List[Tuple[int, int]]]:
    batch: List[Tuple[int, int]] = []
    for j in range(1, n):
        for i in range(j):
            batch.append((i, j))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _sample_pairs(n: int, max_pairs: int, seed: int) -> Tuple[Optional[List[Tuple[int, int]]], int]:
    total = n * (n - 1) // 2
    if max_pairs <= 0 or max_pairs >= total:
        return None, total
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < max_pairs:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        a, b = (i, j) if i < j else (j, i)
        pairs.add((a, b))
    pairs_list = sorted(pairs)
    return pairs_list, len(pairs_list)


def _parse_pair(text: str) -> Tuple[int, int]:
    for sep in (",", ":", "-"):
        if sep in text:
            left, right = text.split(sep, 1)
            return int(left.strip()), int(right.strip())
    raise ValueError(f"Invalid pair '{text}', expected format i,j (0-based).")


def _pair_score(values: np.ndarray, metric: str, freq_norm: np.ndarray) -> float:
    if metric == "mean":
        return float(values.mean())
    if metric == "max":
        return float(values.max())
    if metric == "integral":
        return float(np.trapezoid(values, freq_norm))
    raise ValueError(f"Unsupported metric: {metric}")


def _format_band_label(lo: float, hi: float) -> str:
    return f"{lo:.2f}-{hi:.2f}".replace(".", "p")


def _validate_bands(bands: Sequence[float]) -> List[Tuple[float, float]]:
    if len(bands) < 2:
        raise ValueError("heatmap-bands must have at least two values.")
    ordered = sorted(float(v) for v in bands)
    if ordered[0] < 0.0 or ordered[-1] > 1.0:
        raise ValueError("heatmap-bands must be within [0, 1].")
    return [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]


def _extract_array_payload(
    obj,
    *,
    key: Optional[str],
    split: str,
) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        return obj
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        if key is not None and key in obj:
            return _extract_array_payload(obj[key], key=None, split=split)
        if split in obj:
            return _extract_array_payload(obj[split], key=None, split=split)
        if len(obj) == 1:
            return _extract_array_payload(next(iter(obj.values())), key=None, split=split)
        available = ", ".join(map(str, obj.keys()))
        raise ValueError(f"Unable to infer array key from dict. Available keys: {available}")
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj)
    raise ValueError(f"Unsupported data container type: {type(obj)}")


def _load_array_input(path: Path, *, key: Optional[str], split: str) -> np.ndarray:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"data-array path does not exist: {path}")
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        with np.load(path) as data:
            if key is not None:
                if key not in data:
                    raise ValueError(f"Key '{key}' not found in {path} (available: {data.files}).")
                arr = data[key]
            elif split in data:
                arr = data[split]
            elif len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                raise ValueError(
                    f"Multiple arrays in {path}; pass --data-array-key (available: {data.files})."
                )
    elif ext in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        arr = _extract_array_payload(payload, key=key, split=split)
    else:
        raise ValueError(f"Unsupported data-array extension '{ext}'. Use .npy, .npz, .pt, or .pth.")

    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected data-array with 2 or 3 dims (T,A) or (B,T,A), got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _standardize_batch(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=(0, 1))
    sigma = X.std(axis=(0, 1), ddof=1)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (X - mu) / sigma


def _apply_nan_policy(X: np.ndarray, nan_policy: str) -> np.ndarray:
    if nan_policy == "raise":
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).sum())
            raise SystemExit(f"Found {bad} non-finite values in input array.")
        return X
    if nan_policy == "ffill":
        if X.ndim == 2:
            filled, _ = _forward_fill_numpy(X, initial_last=0.0)
            return filled
        if X.ndim == 3:
            filled = []
            for b in range(X.shape[0]):
                xb, _ = _forward_fill_numpy(X[b], initial_last=0.0)
                filled.append(xb)
            return np.stack(filled, axis=0)
    raise SystemExit(f"Unsupported nan_policy: {nan_policy}")


def _select_train_windows(
    windows: np.ndarray,
    *,
    val_ratio: float,
    window_len: int,
    stride: int,
    train_val_gap: int,
) -> np.ndarray:
    if val_ratio >= 0:
        return windows
    total_train = int(windows.shape[0])
    if total_train <= 0:
        raise SystemExit("No training windows generated; cannot split train/val.")

    abs_ratio = abs(val_ratio)
    val_count = int(round(total_train * abs_ratio))
    min_val_windows = 2 if total_train >= 2 else 1
    val_count = max(min_val_windows, val_count)
    if val_count >= total_train:
        raise SystemExit(
            f"val_ratio={val_ratio} requests {val_count} validation windows but only {total_train} exist. "
            "Reduce |val_ratio| or increase data length."
        )
    if train_val_gap < 0:
        gap = max(0, (window_len + stride - 1) // stride - 1)
    else:
        gap = max(0, int(train_val_gap))

    train_end = total_train - val_count - gap
    if train_end <= 0:
        raise SystemExit(
            f"No training windows remain after applying train_val_gap (val_count={val_count}, "
            f"gap={gap}, total_train_windows={total_train}). Adjust val_ratio/gap or stride/length."
        )
    return windows[:train_end]


def _build_windows_any(
    X: np.ndarray,
    *,
    window_len: int,
    stride: int,
    align_end: bool,
    val_ratio: float,
    train_val_gap: int,
) -> np.ndarray:
    if X.ndim == 2:
        windows = _build_windows(X, window_len, stride, align_end)
        return _select_train_windows(
            windows,
            val_ratio=val_ratio,
            window_len=window_len,
            stride=stride,
            train_val_gap=train_val_gap,
        )
    windows_list = []
    for b in range(X.shape[0]):
        win = _build_windows(X[b], window_len, stride, align_end)
        win = _select_train_windows(
            win,
            val_ratio=val_ratio,
            window_len=window_len,
            stride=stride,
            train_val_gap=train_val_gap,
        )
        windows_list.append(win)
    if not windows_list:
        raise SystemExit("No windows generated from batch input.")
    return np.concatenate(windows_list, axis=0)


def _window_pair_curves(
    psd: torch.Tensor,
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    num_pairs = len(pairs)
    num_freq = psd.size(1)
    cross = torch.zeros((num_pairs, num_freq), dtype=torch.float64, device=psd.device)
    xy_norm = torch.zeros_like(cross)
    for idx, (i, j) in enumerate(pairs):
        psd_i = psd[i]
        psd_j = psd[j]
        xy = psd_i * psd_j
        xy_sum = xy.sum(dim=1)
        xy_total = xy_sum.sum()
        xy_norm_i = xy_sum / (EPS + xy_total)
        x_sum = psd_i.sum(dim=1)
        y_sum = psd_j.sum(dim=1)
        x_norm = x_sum / (EPS + x_sum.sum())
        y_norm = y_sum / (EPS + y_sum.sum())
        cross_i = xy_norm_i / torch.sqrt(torch.clamp(x_norm * y_norm, min=EPS))
        cross[idx] = cross_i
        xy_norm[idx] = xy_norm_i
    return xy_norm.cpu().numpy(), cross.cpu().numpy()


def _plot_band_summary(
    curves: np.ndarray,
    freq_norm: np.ndarray,
    bands: Sequence[Tuple[float, float]],
    out_path: Path,
    *,
    log_y: bool,
    error: str,
    title: Optional[str] = None,
) -> None:
    if curves.ndim == 1:
        curves = curves[None, :]
    if curves.ndim != 2:
        raise ValueError("curves must be 1D or 2D for band summary.")

    labels = []
    means = []
    stds = []
    for lo, hi in bands:
        if hi == bands[-1][1]:
            mask = (freq_norm >= lo) & (freq_norm <= hi)
        else:
            mask = (freq_norm >= lo) & (freq_norm < hi)
        if not mask.any():
            continue
        band_vals = curves[:, mask].mean(axis=1)
        labels.append(f"{lo:.2f}-{hi:.2f}")
        means.append(float(band_vals.mean()))
        stds.append(float(band_vals.std(ddof=0)))

    if not labels:
        raise ValueError("No frequency bands overlap with the provided frequency grid.")

    x = np.arange(len(labels))
    plt.figure(figsize=(4.6, 3.2))
    if error == "std":
        plt.bar(x, means, yerr=stds, capsize=3, color="tab:blue", alpha=0.75)
    else:
        plt.bar(x, means, color="tab:blue", alpha=0.75)
    plt.xticks(x, labels, rotation=0)
    if title:
        plt.title(title)
    plt.xlabel("Frequency Band")
    plt.ylabel(Y_LABEL)
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _accumulate_cross_energy(
    psd: torch.Tensor,
    psd_norm: torch.Tensor,
    pairs: Optional[List[Tuple[int, int]]],
    pair_count: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    n_freq = psd.size(1)
    accum_xy = torch.zeros(n_freq, dtype=torch.float64, device=device)
    accum_cross = torch.zeros(n_freq, dtype=torch.float64, device=device)

    if pairs is None:
        pair_batches = _pair_batches_all(psd.size(0), batch_size)
    else:
        pair_batches = (pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size))

    seen = 0
    for batch in pair_batches:
        if not batch:
            continue
        idx_i = torch.tensor([p[0] for p in batch], device=device)
        idx_j = torch.tensor([p[1] for p in batch], device=device)

        psd_i = psd.index_select(0, idx_i)
        psd_j = psd.index_select(0, idx_j)

        xy = psd_i * psd_j
        xy_sum = xy.sum(dim=2)
        xy_total = xy_sum.sum(dim=1, keepdim=True)
        xy_norm = xy_sum / (EPS + xy_total)

        x_norm = psd_norm.index_select(0, idx_i)
        y_norm = psd_norm.index_select(0, idx_j)
        denom = torch.sqrt(torch.clamp(x_norm * y_norm, min=EPS))
        cross = xy_norm / denom

        accum_xy += xy_norm.to(torch.float64).sum(dim=0)
        accum_cross += cross.to(torch.float64).sum(dim=0)
        seen += xy_norm.size(0)

    if seen != pair_count:
        raise RuntimeError(f"Pair count mismatch: expected {pair_count}, saw {seen}.")

    mean_xy = (accum_xy / pair_count).cpu().numpy()
    mean_cross = (accum_cross / pair_count).cpu().numpy()
    return mean_xy, mean_cross


def _asset_cross_means(
    psd: torch.Tensor,
    psd_norm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_assets = psd.size(2)
    num_freq = psd.size(1)
    device = psd.device
    cross_mean = torch.zeros((num_freq, num_assets, num_assets), dtype=torch.float64, device=device)
    xy_mean = torch.zeros_like(cross_mean)

    for a in range(num_assets):
        psd_a = psd[:, :, a]
        norm_a = psd_norm[:, :, a]
        for b in range(a, num_assets):
            psd_b = psd[:, :, b]
            norm_b = psd_norm[:, :, b]

            xy = psd_a * psd_b
            xy_total = xy.sum(dim=1, keepdim=True)
            xy_norm = xy / (EPS + xy_total)
            denom = torch.sqrt(torch.clamp(norm_a * norm_b, min=EPS))
            cross = xy_norm / denom

            cross_mean[:, a, b] = cross.to(torch.float64).mean(dim=0)
            xy_mean[:, a, b] = xy_norm.to(torch.float64).mean(dim=0)
            if b != a:
                cross_mean[:, b, a] = cross_mean[:, a, b]
                xy_mean[:, b, a] = xy_mean[:, a, b]

    return xy_mean, cross_mean


def main() -> None:
    args = parse_args()

    data_dir = None
    if args.data_array is not None:
        X = _load_array_input(args.data_array, key=args.data_array_key, split=args.split)
    else:
        data_dir = (args.data_dir or _default_data_dir()).expanduser().resolve()
        if not data_dir.exists():
            raise SystemExit(f"data_dir does not exist: {data_dir}")
        split_file = _resolve_split_file(data_dir, args.split)
        X = _load_gluonts_like(split_file)  # shape (T, A)

    source_label = _infer_source_text(args, data_dir)
    mode_label = _format_mode_label(args.cross_mode, source_label)

    X = _apply_nan_policy(X, args.nan_policy)
    if args.standardize:
        if X.ndim == 2:
            X = _standardize(X)
        else:
            X = _standardize_batch(X)

    window_len = args.window_len or (args.context_len + args.pred_len)
    windows = _build_windows_any(
        X,
        window_len=window_len,
        stride=args.stride,
        align_end=args.align_tail_windows,
        val_ratio=args.val_ratio,
        train_val_gap=args.train_val_gap,
    )
    if windows.shape[0] < 1:
        raise SystemExit("Need at least one window to compute spectral summaries.")
    if windows.shape[0] < 2 and args.cross_mode in {"window", "window-pair"}:
        raise SystemExit("Need at least two windows to compute cross-signal energy.")

    windows, sampled_idx = _sample_windows(windows, args.max_windows, args.window_seed)
    num_windows = windows.shape[0]

    device = torch.device(args.device)
    windows_t = torch.from_numpy(windows).to(device)

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        psd = _power_spectrum(windows_t)

    if args.cross_mode == "window-mean":
        with torch.no_grad():
            psd_sum = psd.sum(dim=2)
            psd_total = psd_sum.sum(dim=1, keepdim=True)
            psd_norm = psd_sum / (EPS + psd_total)
            mean_psd = psd_norm.mean(dim=0).cpu().numpy()

        n_freq = mean_psd.shape[0]
        freq_norm = np.linspace(0.0, 1.0, num=n_freq)
        if not args.include_zero_freq:
            freq_norm = freq_norm[1:]
            mean_psd = mean_psd[1:]

        out_csv = args.output_csv or (assets_dir / "cross_signal_energy_window_mean.csv")
        out_plot = args.output_plot or (assets_dir / "cross_signal_energy_window_mean.png")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        df = np.column_stack([freq_norm, mean_psd])
        header = "normalized_frequency,normalized_spectral_density"
        np.savetxt(out_csv, df, delimiter=",", header=header, comments="")

        if not args.no_plot:
            plt.figure(figsize=(4.0, 3.0))
            plot_vals = mean_psd
            if args.log_y:
                plot_vals = np.clip(plot_vals, EPS, None)
                plt.yscale("log")
            plt.plot(freq_norm, plot_vals, marker="o")
            plt.title(mode_label)
            plt.xlabel(X_LABEL)
            plt.ylabel(Y_LABEL)
            plt.tight_layout()
            plt.savefig(out_plot, dpi=args.dpi)
            plt.close()

            if args.plot_band_summary:
                bands = _validate_bands(args.heatmap_bands)
                summary_out = args.band_summary_output or (
                    assets_dir / f"cross_signal_energy_band_summary_{args.cross_mode}.png"
                )
                _plot_band_summary(
                    mean_psd,
                    freq_norm,
                    bands,
                    summary_out,
                    log_y=args.log_y,
                    error=args.band_summary_error,
                    title=mode_label,
                )
                print(f"[INFO] Saved band summary plot to {summary_out}")

        print(f"[INFO] Windows used: {num_windows} (sampled {len(sampled_idx)}).")
        print(f"[INFO] Saved CSV to {out_csv}")
        if not args.no_plot:
            print(f"[INFO] Saved plot to {out_plot}")
        return

    if args.cross_mode == "window":
        total_pairs = num_windows * (num_windows - 1) // 2
        pairs, pair_count = _sample_pairs(num_windows, args.max_pairs, args.pair_seed)
        if pairs is None and total_pairs > 1_000_000:
            print(f"[WARN] Computing all {total_pairs} pairs; consider --max-pairs for speed.")

        with torch.no_grad():
            psd_sum = psd.sum(dim=2)
            psd_total = psd_sum.sum(dim=1, keepdim=True)
            psd_norm = psd_sum / (EPS + psd_total)

            mean_xy, mean_cross = _accumulate_cross_energy(
                psd=psd,
                psd_norm=psd_norm,
                pairs=pairs,
                pair_count=pair_count,
                batch_size=args.pair_batch_size,
                device=device,
            )

        n_freq = mean_cross.shape[0]
        freq_norm = np.linspace(0.0, 1.0, num=n_freq)
        if not args.include_zero_freq:
            freq_norm = freq_norm[1:]
            mean_cross = mean_cross[1:]
            mean_xy = mean_xy[1:]

        out_csv = args.output_csv or (assets_dir / "cross_signal_energy.csv")
        out_plot = args.output_plot or (assets_dir / "cross_signal_energy.png")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        df = np.column_stack([freq_norm, mean_cross, mean_xy])
        header = "normalized_frequency,normalized_cross_energy,normalized_xy_density"
        np.savetxt(out_csv, df, delimiter=",", header=header, comments="")

        if not args.no_plot:
            plt.figure(figsize=(4.0, 3.0))
            plot_vals = mean_cross
            if args.log_y:
                plot_vals = np.clip(plot_vals, EPS, None)
                plt.yscale("log")
            plt.plot(freq_norm, plot_vals, marker="o")
            plt.title(mode_label)
            plt.xlabel(X_LABEL)
            plt.ylabel(Y_LABEL)
            plt.tight_layout()
            plt.savefig(out_plot, dpi=args.dpi)
            plt.close()

            if args.plot_band_summary:
                bands = _validate_bands(args.heatmap_bands)
                summary_out = args.band_summary_output or (
                    assets_dir / f"cross_signal_energy_band_summary_{args.cross_mode}.png"
                )
                _plot_band_summary(
                    mean_cross,
                    freq_norm,
                    bands,
                    summary_out,
                    log_y=args.log_y,
                    error=args.band_summary_error,
                    title=mode_label,
                )
                print(f"[INFO] Saved band summary plot to {summary_out}")

        print(f"[INFO] Windows used: {num_windows} (sampled {len(sampled_idx)}).")
        print(f"[INFO] Pairs used: {pair_count} of {total_pairs}.")
        print(f"[INFO] Saved CSV to {out_csv}")
        if not args.no_plot:
            print(f"[INFO] Saved plot to {out_plot}")
        return

    if args.cross_mode == "window-pair":
        total_pairs = num_windows * (num_windows - 1) // 2
        if args.plot_pairs:
            pairs_compute = []
            for text in args.plot_pairs:
                i, j = _parse_pair(text)
                if i == j:
                    raise SystemExit("plot-pairs requires distinct window indices.")
                if i < 0 or j < 0 or i >= num_windows or j >= num_windows:
                    raise SystemExit(f"plot-pairs window index out of range: {text}.")
                if i > j:
                    i, j = j, i
                pairs_compute.append((i, j))
        else:
            if args.max_pairs <= 0 and total_pairs > 5000:
                raise SystemExit(
                    "window-pair mode can be very large; provide --max-pairs or --plot-pairs."
                )
            if args.max_pairs <= 0 or args.max_pairs >= total_pairs:
                pairs_compute = [(i, j) for j in range(1, num_windows) for i in range(j)]
            else:
                pairs_compute, _ = _sample_pairs(num_windows, args.max_pairs, args.pair_seed)
                if pairs_compute is None:
                    pairs_compute = [(i, j) for j in range(1, num_windows) for i in range(j)]

        pairs_compute = list(pairs_compute)
        if not pairs_compute:
            raise SystemExit("No window pairs selected for window-pair mode.")

        xy_pairs, cross_pairs = _window_pair_curves(psd, pairs_compute)
        n_freq = cross_pairs.shape[1]
        freq_norm = np.linspace(0.0, 1.0, num=n_freq)
        if not args.include_zero_freq:
            freq_norm = freq_norm[1:]
            cross_pairs = cross_pairs[:, 1:]
            xy_pairs = xy_pairs[:, 1:]

        out_csv = args.output_csv or (assets_dir / "cross_signal_energy_window_pairs.csv")
        out_plot = args.output_plot or (assets_dir / "cross_signal_energy_window_pairs.png")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Tuple[float, int, int, float, float]] = []
        for idx, (i, j) in enumerate(pairs_compute):
            for k, freq in enumerate(freq_norm):
                rows.append((freq, i, j, cross_pairs[idx, k], xy_pairs[idx, k]))

        header = "normalized_frequency,window_i,window_j,normalized_cross_energy,normalized_xy_density"
        np.savetxt(
            out_csv,
            np.array(rows, dtype=np.float64),
            delimiter=",",
            header=header,
            comments="",
            fmt=["%.10g", "%d", "%d", "%.10g", "%.10g"],
        )

        if not args.no_plot:
            if args.plot_max_pairs and len(pairs_compute) > args.plot_max_pairs:
                rng = np.random.default_rng(args.plot_pair_seed)
                idx = rng.choice(len(pairs_compute), size=args.plot_max_pairs, replace=False)
                plot_pairs = [pairs_compute[i] for i in sorted(idx)]
            else:
                plot_pairs = list(pairs_compute)

            pair_index = {pair: idx for idx, pair in enumerate(pairs_compute)}
            if not args.skip_pair_plot:
                fig_size = tuple(args.pair_plot_size) if args.pair_plot_size else (5.0, 3.5)
                plt.figure(figsize=fig_size)
                if args.log_y:
                    plt.yscale("log")
                for i, j in plot_pairs:
                    vals = cross_pairs[pair_index[(i, j)]]
                    if args.log_y:
                        vals = np.clip(vals, EPS, None)
                    plt.plot(freq_norm, vals, linewidth=1.0, alpha=0.7, label=f"{i}-{j}")
                plt.title(mode_label)
                plt.xlabel(X_LABEL)
                plt.ylabel(Y_LABEL)
                show_legend = args.plot_legend if args.plot_legend is not None else len(plot_pairs) <= 10
                if show_legend:
                    plt.legend(title="Window Pair", fontsize=7)
                plt.tight_layout()
                plt.savefig(out_plot, dpi=args.dpi)
                plt.close()

            if args.plot_pair_grid:
                cols = max(1, args.pair_grid_cols)
                rows_grid = int(np.ceil(len(plot_pairs) / cols))
                panel_w, panel_h = args.pair_grid_size
                fig, axes = plt.subplots(
                    rows_grid,
                    cols,
                    figsize=(panel_w * cols, panel_h * rows_grid),
                    sharex=True,
                    sharey=True,
                    squeeze=False,
                )
                for idx, (i, j) in enumerate(plot_pairs):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r][c]
                    vals = cross_pairs[pair_index[(i, j)]]
                    if args.log_y:
                        vals = np.clip(vals, EPS, None)
                        ax.set_yscale("log")
                    ax.plot(freq_norm, vals, linewidth=0.8)
                    ax.set_title(f"{i}-{j}", fontsize=7)
                    ax.tick_params(
                        labelbottom=(r == rows_grid - 1),
                        labelleft=(c == 0),
                        length=2,
                        width=0.5,
                    )
                    if r == rows_grid - 1:
                        ax.set_xlabel(X_LABEL, fontsize=7)
                    if c == 0:
                        ax.set_ylabel(Y_LABEL, fontsize=7)
                for idx in range(len(plot_pairs), rows_grid * cols):
                    r = idx // cols
                    c = idx % cols
                    axes[r][c].axis("off")
                grid_out = args.pair_grid_output or (assets_dir / "cross_signal_energy_window_pair_grid.png")
                grid_out.parent.mkdir(parents=True, exist_ok=True)
                fig.suptitle(mode_label, y=0.995, fontsize=9)
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
                fig.savefig(grid_out, dpi=args.dpi)
                plt.close(fig)
                print(f"[INFO] Saved window-pair grid plot to {grid_out}")

            if args.plot_band_summary:
                bands = _validate_bands(args.heatmap_bands)
                summary_out = args.band_summary_output or (
                    assets_dir / f"cross_signal_energy_band_summary_{args.cross_mode}.png"
                )
                _plot_band_summary(
                    cross_pairs,
                    freq_norm,
                    bands,
                    summary_out,
                    log_y=args.log_y,
                    error=args.band_summary_error,
                    title=mode_label,
                )
                print(f"[INFO] Saved band summary plot to {summary_out}")

        print(f"[INFO] Windows used: {num_windows} (sampled {len(sampled_idx)}).")
        print(f"[INFO] Window pairs computed: {len(pairs_compute)} of {total_pairs}.")
        print(f"[INFO] Saved window-pair CSV to {out_csv}")
        if not args.no_plot:
            print(f"[INFO] Saved window-pair plot to {out_plot}")
        return

    with torch.no_grad():
        psd_freq_sum = psd.sum(dim=1, keepdim=True)
        psd_norm_asset = psd / (EPS + psd_freq_sum)
        xy_mean, cross_mean = _asset_cross_means(psd, psd_norm_asset)

    n_freq = cross_mean.shape[0]
    freq_norm = np.linspace(0.0, 1.0, num=n_freq)
    if not args.include_zero_freq:
        freq_norm = freq_norm[1:]
        cross_mean = cross_mean[1:]
        xy_mean = xy_mean[1:]

    num_assets = cross_mean.shape[1]
    if args.cross_mode == "asset-matrix":
        out_matrix = args.output_matrix or (assets_dir / "cross_signal_energy_asset_matrix.npz")
        out_matrix.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_matrix,
            normalized_frequency=freq_norm,
            normalized_cross_energy=cross_mean.cpu().numpy(),
            normalized_xy_density=xy_mean.cpu().numpy(),
        )
        print(f"[INFO] Windows used: {num_windows} (sampled {len(sampled_idx)}).")
        print(f"[INFO] Saved asset-matrix NPZ to {out_matrix}")
        if args.no_plot:
            return

    if args.cross_mode == "asset-mean":
        triu = torch.triu_indices(num_assets, num_assets, offset=1, device=cross_mean.device)
        cross_pairs = cross_mean[:, triu[0], triu[1]]
        xy_pairs = xy_mean[:, triu[0], triu[1]]
        mean_cross = cross_pairs.mean(dim=1).cpu().numpy()
        mean_xy = xy_pairs.mean(dim=1).cpu().numpy()

        out_csv = args.output_csv or (assets_dir / "cross_signal_energy_asset_mean.csv")
        out_plot = args.output_plot or (assets_dir / "cross_signal_energy_asset_mean.png")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        df = np.column_stack([freq_norm, mean_cross, mean_xy])
        header = "normalized_frequency,normalized_cross_energy,normalized_xy_density"
        np.savetxt(out_csv, df, delimiter=",", header=header, comments="")

        if not args.no_plot:
            plt.figure(figsize=(4.0, 3.0))
            plot_vals = mean_cross
            if args.log_y:
                plot_vals = np.clip(plot_vals, EPS, None)
                plt.yscale("log")
            plt.plot(freq_norm, plot_vals, marker="o")
            plt.title(mode_label)
            plt.xlabel(X_LABEL)
            plt.ylabel(Y_LABEL)
            plt.tight_layout()
            plt.savefig(out_plot, dpi=args.dpi)
            plt.close()

        print(f"[INFO] Windows used: {num_windows} (sampled {len(sampled_idx)}).")
        print(f"[INFO] Asset pairs averaged: {cross_pairs.shape[1]}.")
        print(f"[INFO] Saved CSV to {out_csv}")
        if not args.no_plot:
            print(f"[INFO] Saved plot to {out_plot}")
        if args.no_plot:
            return

    if args.cross_mode == "asset-pair":
        out_csv = args.output_csv or (assets_dir / "cross_signal_energy_asset_pairs.csv")
        out_plot = args.output_plot or (assets_dir / "cross_signal_energy_asset_pairs.png")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        rows: List[Tuple[float, int, int, float, float]] = []
        cross_np = cross_mean.cpu().numpy()
        xy_np = xy_mean.cpu().numpy()
        for a in range(num_assets):
            for b in range(a + 1, num_assets):
                for k, freq in enumerate(freq_norm):
                    rows.append((freq, a, b, cross_np[k, a, b], xy_np[k, a, b]))

        header = "normalized_frequency,asset_i,asset_j,normalized_cross_energy,normalized_xy_density"
        np.savetxt(
            out_csv,
            np.array(rows, dtype=np.float64),
            delimiter=",",
            header=header,
            comments="",
            fmt=["%.10g", "%d", "%d", "%.10g", "%.10g"],
        )

        if not args.no_plot:
            if args.plot_pairs:
                pairs_to_plot = []
                for text in args.plot_pairs:
                    a, b = _parse_pair(text)
                    if a == b:
                        raise SystemExit("plot-pairs requires distinct asset indices.")
                    if a < 0 or b < 0 or a >= num_assets or b >= num_assets:
                        raise SystemExit(f"plot-pairs index out of range: {text}.")
                    if a > b:
                        a, b = b, a
                    pairs_to_plot.append((a, b))
            else:
                pairs_to_plot = [(a, b) for a in range(num_assets) for b in range(a + 1, num_assets)]
                if args.plot_max_pairs and len(pairs_to_plot) > args.plot_max_pairs:
                    rng = np.random.default_rng(args.plot_pair_seed)
                    idx = rng.choice(len(pairs_to_plot), size=args.plot_max_pairs, replace=False)
                    pairs_to_plot = [pairs_to_plot[i] for i in sorted(idx)]

            if not args.skip_pair_plot:
                fig_size = tuple(args.pair_plot_size) if args.pair_plot_size else (5.0, 3.5)
                plt.figure(figsize=fig_size)
                if args.log_y:
                    plt.yscale("log")
                for a, b in pairs_to_plot:
                    vals = cross_np[:, a, b]
                    if args.log_y:
                        vals = np.clip(vals, EPS, None)
                    plt.plot(freq_norm, vals, linewidth=1.0, alpha=0.7, label=f"{a}-{b}")
                plt.title(mode_label)
                plt.xlabel(X_LABEL)
                plt.ylabel(Y_LABEL)
                show_legend = args.plot_legend if args.plot_legend is not None else len(pairs_to_plot) <= 10
                if show_legend:
                    plt.legend(title="Asset Pair", fontsize=7)
                plt.tight_layout()
                plt.savefig(out_plot, dpi=args.dpi)
                plt.close()

            if args.plot_pair_grid:
                cols = max(1, args.pair_grid_cols)
                rows = int(np.ceil(len(pairs_to_plot) / cols))
                panel_w, panel_h = args.pair_grid_size
                fig, axes = plt.subplots(
                    rows,
                    cols,
                    figsize=(panel_w * cols, panel_h * rows),
                    sharex=True,
                    sharey=True,
                    squeeze=False,
                )
                for idx, (a, b) in enumerate(pairs_to_plot):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r][c]
                    vals = cross_np[:, a, b]
                    if args.log_y:
                        vals = np.clip(vals, EPS, None)
                        ax.set_yscale("log")
                    ax.plot(freq_norm, vals, linewidth=0.8)
                    ax.set_title(f"{a}-{b}", fontsize=7)
                    ax.tick_params(
                        labelbottom=(r == rows - 1),
                        labelleft=(c == 0),
                        length=2,
                        width=0.5,
                    )
                    if r == rows - 1:
                        ax.set_xlabel(X_LABEL, fontsize=7)
                    if c == 0:
                        ax.set_ylabel(Y_LABEL, fontsize=7)
                for idx in range(len(pairs_to_plot), rows * cols):
                    r = idx // cols
                    c = idx % cols
                    axes[r][c].axis("off")
                grid_out = args.pair_grid_output or (assets_dir / "cross_signal_energy_pair_grid.png")
                grid_out.parent.mkdir(parents=True, exist_ok=True)
                fig.suptitle(mode_label, y=0.995, fontsize=9)
                fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
                fig.savefig(grid_out, dpi=args.dpi)
                plt.close(fig)
                print(f"[INFO] Saved asset-pair grid plot to {grid_out}")

        print(f"[INFO] Windows used: {num_windows} (sampled {len(sampled_idx)}).")
        print(f"[INFO] Saved asset-pair CSV to {out_csv}")
        if not args.no_plot:
            print(f"[INFO] Saved asset-pair plot to {out_plot}")
        if args.no_plot:
            return

    if args.no_plot:
        return

    cross_np = cross_mean.cpu().numpy()
    xy_np = xy_mean.cpu().numpy()

    if args.plot_heatmap:
        bands = _validate_bands(args.heatmap_bands)
        prefix = args.heatmap_output_prefix or (assets_dir / "cross_signal_energy_heatmap")
        prefix.parent.mkdir(parents=True, exist_ok=True)

        band_mats = []
        for lo, hi in bands:
            if hi == bands[-1][1]:
                mask = (freq_norm >= lo) & (freq_norm <= hi)
            else:
                mask = (freq_norm >= lo) & (freq_norm < hi)
            if not mask.any():
                print(f"[WARN] No frequencies in band {lo}-{hi}; skipping.")
                continue
            band_mats.append((lo, hi, cross_np[mask].mean(axis=0)))

        if band_mats:
            if args.heatmap_shared_scale:
                global_min = min(mat.min() for (_, _, mat) in band_mats)
                global_max = max(mat.max() for (_, _, mat) in band_mats)
            else:
                global_min = global_max = None

            for lo, hi, mat in band_mats:
                plt.figure(figsize=(4.0, 3.6))
                if args.heatmap_log:
                    mat_plot = np.clip(mat, EPS, None)
                    norm = LogNorm(vmin=global_min or mat_plot.min(), vmax=global_max or mat_plot.max())
                    im = plt.imshow(mat_plot, cmap="viridis", norm=norm)
                else:
                    vmin = global_min if global_min is not None else None
                    vmax = global_max if global_max is not None else None
                    im = plt.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
                plt.title(f"{mode_label} | {lo:.2f}-{hi:.2f}")
                plt.xlabel("Asset")
                plt.ylabel("Asset")
                plt.colorbar(im, fraction=0.046, pad=0.04)
                plt.tight_layout()
                band_label = _format_band_label(lo, hi)
                out_path = Path(f"{prefix}_band_{band_label}.png")
                plt.savefig(out_path, dpi=args.dpi)
                plt.close()
                print(f"[INFO] Saved heatmap to {out_path}")

    if args.plot_band_summary:
        bands = _validate_bands(args.heatmap_bands)
        summary_out = args.band_summary_output or (
            assets_dir / f"cross_signal_energy_band_summary_{args.cross_mode}.png"
        )
        triu = torch.triu_indices(num_assets, num_assets, offset=1)
        cross_pairs = cross_np[:, triu[0].numpy(), triu[1].numpy()].T
        _plot_band_summary(
            cross_pairs,
            freq_norm,
            bands,
            summary_out,
            log_y=args.log_y,
            error=args.band_summary_error,
            title=mode_label,
        )
        print(f"[INFO] Saved band summary plot to {summary_out}")

    if args.plot_quantile_band:
        triu = torch.triu_indices(num_assets, num_assets, offset=1)
        cross_pairs = cross_np[:, triu[0].numpy(), triu[1].numpy()]
        quantiles = np.array(sorted(args.quantiles))
        if quantiles.size < 2:
            raise SystemExit("quantiles must contain at least two values.")
        q_vals = np.quantile(cross_pairs, quantiles, axis=1)
        q_low = q_vals[0]
        q_high = q_vals[-1]
        if 0.5 in quantiles:
            q_med = q_vals[list(quantiles).index(0.5)]
        else:
            q_med = np.quantile(cross_pairs, 0.5, axis=1)

        out_path = args.quantile_output or (assets_dir / "cross_signal_energy_quantiles.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(4.6, 3.2))
        if args.log_y:
            q_low = np.clip(q_low, EPS, None)
            q_high = np.clip(q_high, EPS, None)
            q_med = np.clip(q_med, EPS, None)
            plt.yscale("log")
        plt.fill_between(freq_norm, q_low, q_high, color="tab:blue", alpha=0.2, label="Quantile band")
        plt.plot(freq_norm, q_med, color="tab:blue", linewidth=1.6, label="Median")
        plt.title(mode_label)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.tight_layout()
        plt.savefig(out_path, dpi=args.dpi)
        plt.close()
        print(f"[INFO] Saved quantile band plot to {out_path}")

    if args.plot_topk:
        scores = []
        for a in range(num_assets):
            for b in range(a + 1, num_assets):
                vals = cross_np[:, a, b]
                score = _pair_score(vals, args.topk_metric, freq_norm)
                scores.append((score, a, b))
        scores.sort(reverse=True)
        topk = scores[: max(1, args.topk_k)]

        out_path = args.topk_output or (assets_dir / "cross_signal_energy_topk.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(5.0, 3.5))
        if args.log_y:
            plt.yscale("log")
        for score, a, b in topk:
            vals = cross_np[:, a, b]
            if args.log_y:
                vals = np.clip(vals, EPS, None)
            plt.plot(freq_norm, vals, linewidth=1.2, alpha=0.8, label=f"{a}-{b}")
        plt.title(mode_label)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        show_legend = args.topk_legend if args.topk_legend is not None else len(topk) <= 10
        if show_legend:
            plt.legend(title="Asset Pair", fontsize=7)
        plt.tight_layout()
        plt.savefig(out_path, dpi=args.dpi)
        plt.close()
        print(f"[INFO] Saved top-K plot to {out_path}")

    if args.plot_asset_panels:
        cols = max(1, args.asset_panels_cols)
        rows = int(np.ceil(num_assets / cols))
        fig_w = args.asset_panels_size[0] * cols
        fig_h = args.asset_panels_size[1] * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
        for a in range(num_assets):
            r = a // cols
            c = a % cols
            ax = axes[r][c]
            partner_scores = []
            for b in range(num_assets):
                if a == b:
                    continue
                vals = cross_np[:, a, b]
                partner_scores.append((_pair_score(vals, args.asset_panels_metric, freq_norm), b))
            partner_scores.sort(reverse=True)
            partners = [b for (_, b) in partner_scores[: max(1, args.asset_panels_topk)]]
            for b in partners:
                vals = cross_np[:, a, b]
                if args.log_y:
                    vals = np.clip(vals, EPS, None)
                ax.plot(freq_norm, vals, linewidth=1.0, alpha=0.8, label=f"{a}-{b}")
            ax.set_title(f"Asset {a}")
            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(Y_LABEL)
            if args.log_y:
                ax.set_yscale("log")
            if args.asset_panels_legend:
                ax.legend(fontsize=6)
        # Hide unused axes
        for idx in range(num_assets, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis("off")
        fig.suptitle(mode_label, y=0.995, fontsize=9)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        out_path = args.asset_panels_output or (assets_dir / "cross_signal_energy_asset_panels.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"[INFO] Saved asset panel plot to {out_path}")

    return


if __name__ == "__main__":
    main()
