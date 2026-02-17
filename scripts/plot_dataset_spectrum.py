#!/usr/bin/env python3
"""
Plot dataset spectral and temporal energy densities for the FX panel.

This reproduces the style of FourierDiffusion's spectral interpretation plots:
- Normalised spectral density vs. normalised frequency.
- Normalised energy density vs. normalised time.

The script operates directly on the cleaned GluonTS dataset (e.g. exchange_rate_clean),
using the same standardisation as the ConditionalGluonTSJsonDatamodule, and saves
figures under ./assets by default.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import string
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfdiff.dataloaders.conditional_gluonts import _load_gluonts_like, _resolve_split_file
from cfdiff.eval.fourier_metrics import _spectral_density


EPS = 1e-15
DISPLAY_LABELS = {
    "exchange_rate_clean": "Exchange",
    "fx30_ecb": "FX29",
    "industry49_clean": "Industry49",
    "industry49_clean_0.85": "Industry49",
    "ishares14_clean": "iShares14",
}


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def _default_data_dir() -> Path:
    env = os.environ.get("CFDIFF_DATA_DIR")
    if env:
        return Path(env).expanduser()
    # Fall back to the standard exchange_rate_clean location
    return Path.home() / ".gluonts" / "datasets" / "exchange_rate_clean"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="One or more dataset roots (GluonTS format). If omitted, uses CFDIFF_DATA_DIR or ~/.gluonts/datasets/exchange_rate_clean.",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use for computing spectra.",
    )
    ap.add_argument(
        "--output-spectral",
        type=Path,
        default=None,
        help="Output path for the spectral-density plot (default: assets/dataset_spectral_density.<output-format>).",
    )
    ap.add_argument(
        "--output-energy",
        type=Path,
        default=None,
        help="Output path for the time-domain energy plot (default: assets/dataset_energy_density.<output-format>).",
    )
    ap.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="Optional single figure with spectral (left) and energy (right) subplots (default: assets/dataset_spectrum_energy.<output-format>).",
    )
    ap.add_argument(
        "--combined-layout",
        type=str,
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Layout for the combined figure if requested.",
    )
    ap.add_argument(
        "--combined-stack-datasets",
        action="store_true",
        help="Stack datasets vertically in the combined figure (each row is one dataset; spectral left, energy right).",
    )
    ap.add_argument(
        "--per-dataset-dir",
        type=Path,
        default=None,
        help="If set, saves a separate two-panel (spectral + energy) figure for each dataset into this directory.",
    )
    ap.add_argument(
        "--output-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format (default: png).",
    )
    return ap.parse_args()


def _standardise(X: np.ndarray) -> np.ndarray:
    """Standardise per-asset, matching the datamodule."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma < 1e-8] = 1.0
    return (X - mu) / sigma


def _compute_density(data_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    split_file = _resolve_split_file(data_dir, split)
    X = _load_gluonts_like(split_file)  # shape (T, A)
    X_std = _standardise(X)
    X_tensor = torch.from_numpy(X_std[None, :, :])  # (1, T, A)
    T = X_tensor.size(1)

    X_spec = _spectral_density(X_tensor)  # (1, n_freq, A)
    num = X_spec.sum(dim=2, keepdim=True)
    den = X_spec.sum(dim=(1, 2), keepdim=True)
    spec_norm = num / (EPS + den)
    spec_mean = spec_norm.mean(dim=(0, 2)).cpu().numpy()
    n_freq = spec_mean.shape[0]
    freq_norm = np.linspace(0.0, 1.0, num=n_freq)

    energy_num = (X_tensor ** 2).sum(dim=2, keepdim=True)
    energy_den = (X_tensor ** 2).sum(dim=(1, 2), keepdim=True)
    energy_norm = energy_num / (EPS + energy_den)
    energy_mean = energy_norm.mean(dim=(0, 2)).cpu().numpy()
    time_norm = np.linspace(0.0, 1.0, num=T)
    return freq_norm, spec_mean, time_norm, energy_mean, data_dir.name


def main() -> None:
    args = parse_args()
    fmt = args.output_format.lstrip(".").lower()

    if args.data_dirs is None:
        data_dirs = [_default_data_dir()]
    else:
        data_dirs = [p.expanduser().resolve() for p in args.data_dirs]
    for d in data_dirs:
        if not d.exists():
            raise SystemExit(f"data_dir does not exist: {d}")

    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    if args.output_spectral is None:
        out_spec = assets_dir / "dataset_spectral_density.png"
    else:
        out_spec = args.output_spectral
    out_spec = out_spec.with_suffix(f".{fmt}")
    out_spec.parent.mkdir(parents=True, exist_ok=True)
    if args.output_energy is None:
        out_energy = assets_dir / "dataset_energy_density.png"
    else:
        out_energy = args.output_energy
    out_energy = out_energy.with_suffix(f".{fmt}")
    out_energy.parent.mkdir(parents=True, exist_ok=True)
    if args.combined_output is None:
        out_combined = assets_dir / "dataset_spectrum_energy.png"
    else:
        out_combined = args.combined_output
    out_combined = out_combined.with_suffix(f".{fmt}")
    out_combined.parent.mkdir(parents=True, exist_ok=True)

    results = [
        _compute_density(d, args.split) for d in data_dirs
    ]  # list of (freq_norm, spec_mean, time_norm, energy_mean, label)

    # Plot spectral density
    plt.figure(figsize=(4.0, 3.0))
    for freq_norm, spec_mean, _, _, label in results:
        plt.plot(freq_norm[1:], spec_mean[1:], label=_display_label(label))
    plt.yscale("log")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Spectral Density (normalised)")
    plt.legend(
        title="Dataset",
        framealpha=0.35,  # lighter box so it does not obscure the curves
        facecolor="white",
        edgecolor="lightgray",
    )
    plt.tight_layout()
    plt.savefig(out_spec, dpi=300)
    plt.close()

    # Plot time-domain energy density
    plt.figure(figsize=(4.0, 3.0))
    for _, _, time_norm, energy_mean, label in results:
        plt.plot(time_norm, energy_mean, label=_display_label(label))
    plt.yscale("log")
    plt.xlabel("Normalized Time")
    plt.ylabel("Energy Density (normalised)")
    plt.legend(
        title="Dataset",
        framealpha=0.35,
        facecolor="white",
        edgecolor="lightgray",
    )
    plt.tight_layout()
    plt.savefig(out_energy, dpi=300)
    plt.close()

    # Optional combined figure
    if args.combined_stack_datasets:
        nrows = len(data_dirs)
        ncols = 2
        data_height = 2.6
        label_height = 0.35
        figsize = (8.0, max(2.6, (data_height + label_height) * nrows))
        fig = plt.figure(figsize=figsize)
        height_ratios = [1.0, 0.18] * nrows
        grid = fig.add_gridspec(nrows=nrows * 2, ncols=ncols, height_ratios=height_ratios, hspace=0.28, wspace=0.3)

        letters = string.ascii_lowercase
        total_panels = nrows * ncols
        labels = [f"({letters[i]})" if i < len(letters) else f"({i + 1})" for i in range(total_panels)]

        for idx, (freq_norm, spec_mean, time_norm, energy_mean, label) in enumerate(results):
            label_display = _display_label(label)
            ax_spec = fig.add_subplot(grid[2 * idx, 0])
            ax_energy = fig.add_subplot(grid[2 * idx, 1])
            ax_spec.plot(freq_norm[1:], spec_mean[1:], color="tab:blue")
            ax_energy.plot(time_norm, energy_mean, color="tab:orange")
            ax_spec.set_yscale("log")
            ax_energy.set_yscale("log")
            ax_spec.set_ylabel("Spectral Density (normalised)")
            ax_energy.set_ylabel("Energy Density (normalised)")
            ax_spec.set_title(f"{label_display} | Spectral")
            ax_energy.set_title(f"{label_display} | Temporal")
            if idx == nrows - 1:
                ax_spec.set_xlabel("Normalized Frequency")
                ax_energy.set_xlabel("Normalized Time")
            else:
                ax_spec.set_xlabel("")
                ax_energy.set_xlabel("")
            ax_spec.tick_params(axis="x", labelbottom=True)
            ax_energy.tick_params(axis="x", labelbottom=True)

            ax_label_left = fig.add_subplot(grid[2 * idx + 1, 0])
            ax_label_right = fig.add_subplot(grid[2 * idx + 1, 1])
            ax_label_left.axis("off")
            ax_label_right.axis("off")
            ax_label_left.text(0.5, 0.45, labels[2 * idx], ha="center", va="center", fontsize=12, fontweight="bold")
            ax_label_right.text(0.5, 0.45, labels[2 * idx + 1], ha="center", va="center", fontsize=12, fontweight="bold")

        fig.subplots_adjust(top=0.965, bottom=0.03)
        fig.savefig(out_combined, dpi=300)
        plt.close(fig)
    else:
        ncols = 2 if args.combined_layout == "horizontal" else 1
        nrows = 1 if args.combined_layout == "horizontal" else 2
        figsize = (8.0, 3.0) if args.combined_layout == "horizontal" else (4.0, 6.0)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        ax_spec = axes[0, 0]
        for freq_norm, spec_mean, _, _, label in results:
            ax_spec.plot(freq_norm[1:], spec_mean[1:], label=_display_label(label))
        ax_spec.set_yscale("log")
        ax_spec.set_xlabel("Normalized Frequency")
        ax_spec.set_ylabel("Spectral Density (normalised)")
        ax_spec.legend(title="Dataset")

        ax_energy = axes[0, 1] if args.combined_layout == "horizontal" else axes[1, 0]
        for _, _, time_norm, energy_mean, label in results:
            ax_energy.plot(time_norm, energy_mean, label=_display_label(label))
        ax_energy.set_yscale("log")
        ax_energy.set_xlabel("Normalized Time")
        ax_energy.set_ylabel("Energy Density (normalised)")
        ax_energy.legend(title="Dataset")

        plt.tight_layout()
        plt.savefig(out_combined, dpi=300)
        plt.close()

    # Per-dataset figures
    if args.per_dataset_dir is not None:
        per_dir = args.per_dataset_dir.expanduser().resolve()
        per_dir.mkdir(parents=True, exist_ok=True)
        for freq_norm, spec_mean, time_norm, energy_mean, label in results:
            label_display = _display_label(label)
            safe_label = label.replace("/", "_")
            fig, (ax_spec, ax_energy) = plt.subplots(nrows=1, ncols=2, figsize=(8.0, 3.0), squeeze=True)
            ax_spec.plot(freq_norm[1:], spec_mean[1:], color="tab:blue")
            ax_energy.plot(time_norm, energy_mean, color="tab:orange")
            ax_spec.set_yscale("log")
            ax_energy.set_yscale("log")
            ax_spec.set_xlabel("Normalized Frequency")
            ax_spec.set_ylabel("Spectral Density (normalised)")
            ax_energy.set_xlabel("Normalized Time")
            ax_energy.set_ylabel("Energy Density (normalised)")
            ax_spec.set_title(f"{label_display} | Spectral")
            ax_energy.set_title(f"{label_display} | Temporal")
            plt.tight_layout()
            per_path = per_dir / f"{safe_label}_{args.split}_spectrum_energy.png"
            per_path = per_path.with_suffix(f".{fmt}")
            plt.savefig(per_path, dpi=300)
            plt.close()
            print(f"[INFO] Saved per-dataset figure for {label} to {per_path}")

    print(f"[INFO] Saved spectral density plot to {out_spec}")
    print(f"[INFO] Saved time-domain energy plot to {out_energy}")
    print(f"[INFO] Saved combined figure to {out_combined}")


if __name__ == "__main__":
    main()
