#!/usr/bin/env python
"""
Plot volatility clustering (|r_t|) overlays for multiple datasets in a grid.

Each panel overlays Ground Truth (test) with CDSM-Spectral and CDSM-Temporal
predictions on the test split. The y-axis range is unified across panels.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_stylized_facts import load_panel
from plot_stylized_facts_with_pred import _resolve_samples_pt, _stage_indices


def _to_np(arr):
    return arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)


def _load_samples_series(samples_pt: str, stage: str, asset_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    data = torch.load(samples_pt, map_location="cpu", weights_only=False)
    samples = _to_np(data["samples"])
    truth = _to_np(data["truth"])

    if samples.ndim == 4:
        # [W, M, P, A] or [M, W, P, A] -> pick trajectory 0
        if samples.shape[0] == truth.shape[0]:
            samples = samples[:, 0, :, :]
        elif samples.shape[1] == truth.shape[0]:
            samples = samples[0]
        else:
            samples = samples[:, 0, :, :]

    total = samples.shape[0]
    indices = _stage_indices(data.get("window_stage_counts", {}), stage, total)
    samples = samples[indices]
    truth = truth[indices]

    pred_series = np.abs(samples[..., asset_idx]).reshape(-1)
    truth_series = np.abs(truth[..., asset_idx]).reshape(-1)
    return truth_series, pred_series


def _parse_entry(entry: str) -> Tuple[str, str, str, str, List[str]]:
    parts = [p.strip() for p in entry.split("|")]
    if len(parts) < 4:
        raise ValueError("Entry must be: TITLE|DATA_DIR|SPECTRAL_RUN|TEMPORAL_RUN[|EXCLUDE_ASSETS]")
    title, data_dir, spectral_run, temporal_run = parts[:4]
    exclude = []
    if len(parts) >= 5 and parts[4]:
        exclude = [e.strip() for e in parts[4].split(",") if e.strip()]
    return title, data_dir, spectral_run, temporal_run, exclude


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Format: TITLE|DATA_DIR|SPECTRAL_RUN|TEMPORAL_RUN[|EXCLUDE_ASSETS]",
    )
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--max-assets", type=int, default=8)
    p.add_argument("--pred-repeat", default="r01", help="Repeat tag to select (e.g., r01).")
    p.add_argument("--pred-stage", default="test", choices=["train", "test", "all"])
    p.add_argument("--cols", type=int, default=2, help="Number of columns in grid.")
    p.add_argument("--stride", type=int, default=2, help="Plot every Nth point to reduce density.")
    p.add_argument("--unify-ylim", action="store_true", help="Use a shared y-axis range across panels.")
    p.add_argument("--output", type=str, required=True, help="Output PDF path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    entries = [_parse_entry(e) for e in args.entry]

    series_list = []
    meta = []

    for title, data_dir, spectral_run, temporal_run, exclude in entries:
        df = load_panel(data_dir, args.split, args.max_assets, exclude)
        returns = df.values.T.astype(float)
        stds = np.nanstd(returns, axis=1)
        asset_idx = int(np.nanargmax(stds))

        spectral_pt = _resolve_samples_pt(spectral_run, None, args.pred_repeat)
        temporal_pt = _resolve_samples_pt(temporal_run, None, args.pred_repeat)

        truth_series, spectral_series = _load_samples_series(spectral_pt, args.pred_stage, asset_idx)
        _, temporal_series = _load_samples_series(temporal_pt, args.pred_stage, asset_idx)

        series_list.append((truth_series, spectral_series, temporal_series))
        meta.append(title)

    # allow per-panel y-range for clarity
    ymax_global = max(float(np.nanmax(s)) for triplet in series_list for s in triplet if s.size)
    ymax_global = ymax_global * 1.25 if ymax_global > 0 else 1.0

    cols = max(1, args.cols)
    rows = int(np.ceil(len(series_list) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows), squeeze=False)

    # High-contrast palette for readability
    color_gt = "#1f77b4"    # blue
    color_spec = "#ff7f0e"  # orange
    color_temp = "#d62728"  # red (high contrast vs. blue/orange)

    for idx, (title, (truth_series, spectral_series, temporal_series)) in enumerate(zip(meta, series_list)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        stride = max(1, int(args.stride))
        truth_series = truth_series[::stride]
        spectral_series = spectral_series[::stride]
        temporal_series = temporal_series[::stride]
        x = np.arange(truth_series.size)
        ax.plot(
            x,
            temporal_series,
            lw=1.0,
            color=color_temp,
            alpha=0.55,
            linestyle="-",
            label="CDSM-Temporal",
            zorder=1,
        )
        ax.plot(
            x,
            truth_series,
            lw=1.2,
            color=color_gt,
            alpha=0.70,
            label="Ground Truth (test)",
            zorder=2,
        )
        ax.plot(
            x,
            spectral_series,
            lw=1.4,
            color=color_spec,
            alpha=0.85,
            linestyle="-",
            label="CDSM-Spectral",
            zorder=3,
        )
        ax.set_title(title, fontsize=11, pad=4)
        ax.set_xlabel("Windowed step")
        ax.set_ylabel("|Log Return|")
        if args.unify_ylim:
            ax.set_ylim(0, ymax_global)
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.35)
        ax.legend(fontsize=7, frameon=False, ncol=1, loc="upper right")

    # hide unused axes
    for idx in range(len(series_list), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
