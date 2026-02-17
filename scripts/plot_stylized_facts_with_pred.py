#!/usr/bin/env python
"""
Stylized facts plot with predicted vs. ground-truth distribution overlay.

Matches the 2x2 layout of plot_stylized_facts.py, but the top-left panel
shows test-split ground-truth returns and overlays predicted returns.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, norm

from plot_stylized_facts import (
    _compute_cumulative_returns,
    _infer_returns_mode,
    _plot_full_series_with_split,
    compute_acf_mean,
    load_full_panel_for_split,
    load_panel,
)


def _stage_indices(stage_counts: dict, stage: str, total: int) -> List[int]:
    if not stage_counts or stage == "all":
        return list(range(total))
    val_n = int(stage_counts.get("val", 0))
    test_n = int(stage_counts.get("test", 0))
    if stage == "val":
        return list(range(0, val_n))
    if stage == "test":
        return list(range(val_n, val_n + test_n))
    return list(range(total))


def _resolve_samples_pt(run_dir: Path, sample_tag: str | None, repeat_tag: str) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    hist = run_dir / "samples_history"
    if hist.is_dir():
        candidates = [p for p in hist.iterdir() if p.is_dir() and p.name.startswith("batch-")]
        if sample_tag:
            candidates = [p for p in candidates if sample_tag in p.name]
        if not candidates:
            raise FileNotFoundError(f"No batch directories under {hist} (sample_tag={sample_tag})")
        batch = sorted(candidates)[-1]
        # prefer mc0 if present
        mc0 = batch.with_name(f"{batch.name}-mc0")
        if mc0.is_dir():
            batch = mc0
        # select repeat subdir
        repeat_dirs = [p for p in batch.iterdir() if p.is_dir()]
        if repeat_dirs:
            match = [p for p in repeat_dirs if repeat_tag in p.name]
            chosen = sorted(match)[0] if match else sorted(repeat_dirs)[0]
            samples_pt = chosen / "samples.pt"
            if samples_pt.exists():
                return samples_pt
        # fallback
        samples_pt = batch / "samples.pt"
        if samples_pt.exists():
            return samples_pt
    # handle baseline style: run_dir contains repeat subdirs (r01, r02, ...)
    if run_dir.is_dir():
        repeat_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
        if repeat_dirs:
            match = [p for p in repeat_dirs if repeat_tag in p.name]
            chosen = sorted(match)[0] if match else sorted(repeat_dirs)[0]
            samples_pt = chosen / "samples.pt"
            if samples_pt.exists():
                return samples_pt
    # fallback to run_dir/samples.pt
    samples_pt = run_dir / "samples.pt"
    if samples_pt.exists():
        return samples_pt
    raise FileNotFoundError(f"No samples.pt found under {run_dir}")


def _load_pred_returns(
    samples_pt: Path,
    stage: str,
    trajectory_index: int,
) -> np.ndarray:
    data = torch.load(samples_pt, map_location="cpu", weights_only=False)
    def _to_np(arr):
        return arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)
    samples = _to_np(data["samples"])
    truth = _to_np(data["truth"])
    if samples.ndim == 4:
        if samples.shape[0] == truth.shape[0]:
            samples = samples[:, trajectory_index % samples.shape[1], :, :]
        elif samples.shape[1] == truth.shape[0]:
            samples = samples[trajectory_index % samples.shape[0]]
        else:
            samples = samples[:, trajectory_index % samples.shape[1], :, :]
    total = samples.shape[0]
    indices = _stage_indices(data.get("window_stage_counts", {}), stage, total)
    sel = samples[indices]
    return sel.reshape(-1)


def _plot_heavy_tails_panel(
    ax: plt.Axes,
    flattened: np.ndarray,
    pred_series: List[Tuple[str, np.ndarray]],
    bins: int,
    xlim_quantiles: Tuple[float, float] | None,
    xlim_mode: str = "quantile",
    xlim_sigma: float = 4.0,
    log_density: bool = False,
    normal_ref: bool = False,
    show_legend: bool = True,
    legend_loc: str = "best",
    legend_cols: int = 3,
    legend_fontsize: int = 8,
    legend_bbox: Tuple[float, float] | None = None,
) -> None:
    color_map = {
        "Ground Truth (test)": "#1f77b4",  # blue
        "CDSM-Temporal": "#2ca02c",        # green
        "CDSM-Spectral": "#ff7f0e",        # orange
    }
    mu, sigma = np.mean(flattened), np.std(flattened)
    k = kurtosis(flattened, fisher=False, bias=False)
    series_list = [flattened]
    for _label, series in pred_series:
        if series is None:
            continue
        series = series[np.isfinite(series)]
        if series.size:
            series_list.append(series)
    all_vals = np.concatenate(series_list) if series_list else flattened
    if xlim_mode == "sigma" and np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        qlo, qhi = mu - xlim_sigma * sigma, mu + xlim_sigma * sigma
    elif xlim_quantiles is not None:
        lo_q, hi_q = xlim_quantiles
        qlo, qhi = np.percentile(all_vals, [lo_q, hi_q])
        if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo == qhi:
            qlo, qhi = np.min(all_vals), np.max(all_vals)
    else:
        qlo, qhi = np.min(all_vals), np.max(all_vals)
    if not np.isfinite(qlo) or not np.isfinite(qhi) or qlo == qhi:
        qlo, qhi = -1.0, 1.0
    bin_edges = np.linspace(qlo, qhi, bins + 1)

    ax.hist(
        flattened,
        bins=bin_edges,
        density=True,
        alpha=0.55,
        color=color_map.get("Ground Truth (test)", "tab:blue"),
        label="Ground Truth (test)",
    )
    for label, series in pred_series:
        series = series[np.isfinite(series)]
        if series.size == 0:
            continue
        color = color_map.get(label, ax._get_lines.get_next_color())
        ax.hist(
            series,
            bins=bin_edges,
            density=True,
            histtype="stepfilled",
            alpha=0.18,
            linewidth=1.0,
            edgecolor=color,
            color=color,
            label=label,
        )
    if normal_ref and np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        xs = np.linspace(qlo, qhi, 300)
        ax.plot(
            xs,
            norm.pdf(xs, mu, sigma),
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.2,
            label=f"Normal ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})",
        )
    ax.set_title(f"Heavy tails (kurtosis={k:.2f}; normal=3)", fontsize=11, pad=2)
    ax.set_xlabel("Log Return", labelpad=1)
    ax.set_ylabel("Density")
    if log_density:
        ax.set_yscale("log")
    ax.set_xlim(qlo, qhi)
    if show_legend:
        ax.legend(
            loc=legend_loc,
            ncol=max(1, legend_cols),
            fontsize=legend_fontsize,
            frameon=False,
            bbox_to_anchor=legend_bbox,
        )


def _plot_stylized_facts_with_pred(
    axes: np.ndarray,
    df: pd.DataFrame,
    dataset_name: str,
    acf_lags: int,
    rolling_window: int,
    pred_series: List[Tuple[str, np.ndarray]],
    full_df: pd.DataFrame | None = None,
    split_time: pd.Timestamp | None = None,
    full_df_is_levels: bool = False,
    full_series_mode: str = "levels",
    full_series_returns_mode: str | None = None,
    full_series_labels: str = "none",
    legend_cols: int = 3,
    legend_fontsize: int = 7,
    bins: int = 60,
    xlim_quantiles: Tuple[float, float] | None = (0.5, 99.5),
) -> None:
    returns = df.values.T.astype(float)  # [A, T]
    stds = np.nanstd(returns, axis=1)
    keep = np.isfinite(stds) & (stds > 1e-12)
    returns = returns[keep]
    flattened = returns.flatten()
    flattened = flattened[np.isfinite(flattened)]
    axes = np.asarray(axes)
    if axes.shape != (2, 2):
        raise ValueError(f"Expected axes shape (2, 2), got {axes.shape}")

    # 1) Heavy tails: ground truth + predictions
    _plot_heavy_tails_panel(
        axes[0, 0],
        flattened,
        pred_series,
        bins=bins,
        xlim_quantiles=xlim_quantiles,
    )

    # 2) Volatility clustering (|r|)
    ax = axes[0, 1]
    sample_idx = int(np.nanargmax(np.nanstd(returns, axis=1)))
    sample = np.abs(returns[sample_idx])
    ax.plot(df.index, sample, lw=0.8, color="tab:orange")
    ax.set_title("|r_t| (volatility clustering)", fontsize=11, pad=2)
    ax.set_xlabel("Time", labelpad=1)
    ax.set_ylabel("|Return|")
    ax.tick_params(axis="x", rotation=30)

    # 3) ACF of r and r^2
    ax = axes[1, 0]
    acf_r = compute_acf_mean(returns, acf_lags)
    acf_r2 = compute_acf_mean(returns ** 2, acf_lags)
    lags = np.arange(acf_lags + 1)
    ax.stem(lags, acf_r, linefmt="tab:blue", markerfmt=" ", basefmt="k-", label="ACF(r)")
    ax.stem(lags + 0.1, acf_r2, linefmt="tab:red", markerfmt=" ", basefmt="k-", label="ACF(r$^2$)")
    ax.set_xlim(0, acf_lags)
    ax.set_title("ACF of returns vs. squared returns", fontsize=11, pad=2)
    ax.set_xlabel("Lag", labelpad=1)
    ax.set_ylabel("Correlation")
    ax.legend()

    # 4) Full panel series with train/test split marker
    ax = axes[1, 1]
    if full_df is None or full_df.empty:
        ax.text(0.5, 0.5, "No full panel data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        plot_df = full_df
        title = "Cumulative returns with train/test split" if full_series_mode == "cum_returns" else "All series with train/test split"
        ylabel = "Cumulative return" if full_series_mode == "cum_returns" else "Value"
        if full_series_mode == "cum_returns":
            plot_df = _compute_cumulative_returns(
                full_df,
                returns_mode=full_series_returns_mode,
                is_levels=full_df_is_levels,
            )
        _plot_full_series_with_split(
            ax,
            plot_df,
            split_time,
            title=title,
            ylabel=ylabel,
            show_legend=full_series_labels == "legend",
            legend_cols=legend_cols,
            legend_fontsize=legend_fontsize,
        )
        ax.set_title(title, fontsize=11, pad=2)
        ax.set_ylabel(ylabel)


def parse_args():
    p = argparse.ArgumentParser(description="Stylized facts with predicted distribution overlay.")
    p.add_argument("--data-dirs", nargs="+", required=True)
    p.add_argument("--dataset-names", nargs="+", default=None)
    p.add_argument("--split", default="test", choices=["train", "test", "all"])
    p.add_argument("--max-assets", type=int, default=8)
    p.add_argument("--exclude-assets", nargs="+", default=None)
    p.add_argument("--acf-lags", type=int, default=50)
    p.add_argument("--rolling-window", type=int, default=20)
    p.add_argument("--full-series-mode", choices=["levels", "cum_returns"], default="levels")
    p.add_argument("--full-series-labels", choices=["none", "legend"], default="none")
    p.add_argument("--legend-cols", type=int, default=3)
    p.add_argument("--legend-fontsize", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="assets")
    p.add_argument("--format", type=str, default="pdf")
    p.add_argument("--output-path", type=str, default=None, help="Optional full output path to save the figure.")
    p.add_argument("--heavy-only", action="store_true", help="Only plot the heavy-tails panel.")
    p.add_argument("--bins", type=int, default=60, help="Histogram bins for heavy-tails panel.")
    p.add_argument(
        "--xlim-quantiles",
        type=float,
        nargs=2,
        default=(0.5, 99.5),
        help="Quantiles for heavy-tails x-axis limits (e.g., 0.5 99.5).",
    )
    p.add_argument(
        "--xlim-mode",
        choices=["quantile", "sigma"],
        default="quantile",
        help="How to set heavy-tails x-axis limits (quantile or mu±k*sigma).",
    )
    p.add_argument(
        "--xlim-sigma",
        type=float,
        default=4.0,
        help="Sigma multiplier when --xlim-mode sigma is used (default: 4).",
    )
    p.add_argument("--log-density", action="store_true", help="Use log scale on the density axis.")
    p.add_argument("--normal-ref", action="store_true", help="Overlay a fitted normal PDF.")
    p.add_argument("--legend-location", default="best", help="Legend location for heavy-tails panel.")
    p.add_argument(
        "--legend-top",
        action="store_true",
        help="Place legend above the plot (outside the axes).",
    )
    # prediction overlays
    p.add_argument("--pred-runs", nargs="*", default=[])
    p.add_argument("--pred-labels", nargs="*", default=[])
    p.add_argument("--pred-sample-tags", nargs="*", default=[])
    p.add_argument("--pred-repeat", default="r01", help="Repeat tag to select (e.g., r01).")
    p.add_argument("--pred-stage", default="test", choices=["train", "test", "all"])
    p.add_argument("--trajectory-index", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.pred_runs and len(args.data_dirs) > 1:
        raise SystemExit("Predicted distribution overlay currently supports a single dataset.")

    if args.pred_labels and len(args.pred_labels) != len(args.pred_runs):
        raise SystemExit("--pred-labels must match --pred-runs length.")
    if args.pred_sample_tags and len(args.pred_sample_tags) != len(args.pred_runs):
        raise SystemExit("--pred-sample-tags must match --pred-runs length.")

    pred_series: List[Tuple[str, np.ndarray]] = []
    for i, run in enumerate(args.pred_runs):
        label = args.pred_labels[i] if args.pred_labels else f"Pred {i + 1}"
        tag = args.pred_sample_tags[i] if args.pred_sample_tags else None
        samples_pt = _resolve_samples_pt(Path(run), tag, args.pred_repeat)
        series = _load_pred_returns(samples_pt, args.pred_stage, args.trajectory_index)
        pred_series.append((label, series))

    for idx, data_dir in enumerate(args.data_dirs):
        dataset_name = (
            args.dataset_names[idx]
            if args.dataset_names and idx < len(args.dataset_names)
            else os.path.basename(os.path.abspath(data_dir))
        )
        df = load_panel(data_dir, args.split, args.max_assets, args.exclude_assets)
        full_df, split_time, full_df_is_levels = load_full_panel_for_split(
            data_dir,
            args.split,
            args.max_assets,
            args.exclude_assets,
            df,
        )
        returns_mode = _infer_returns_mode(data_dir)

        if args.heavy_only:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
            fig.suptitle(f"{dataset_name}", fontsize=13, y=0.985)
            legend_bbox = (0.5, 0.915) if args.legend_top else None
            legend_loc = "upper center" if args.legend_top else args.legend_location
            _plot_heavy_tails_panel(
                ax,
                df.values.T.astype(float).flatten(),
                pred_series,
                bins=args.bins,
                xlim_quantiles=tuple(args.xlim_quantiles) if args.xlim_quantiles else None,
                xlim_mode=args.xlim_mode,
                xlim_sigma=args.xlim_sigma,
                log_density=args.log_density,
                normal_ref=args.normal_ref,
                show_legend=not args.legend_top,
                legend_loc=legend_loc,
                legend_cols=args.legend_cols,
                legend_fontsize=args.legend_fontsize,
                legend_bbox=legend_bbox,
            )
            if args.legend_top:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 0.915),
                        ncol=max(1, args.legend_cols),
                        fontsize=args.legend_fontsize,
                        frameon=False,
                    )
                fig.subplots_adjust(top=0.78, bottom=0.14, left=0.12, right=0.98)
            else:
                plt.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Stylized Facts: {dataset_name}", fontsize=14)
            _plot_stylized_facts_with_pred(
                axes,
                df,
                dataset_name=dataset_name,
                acf_lags=args.acf_lags,
                rolling_window=args.rolling_window,
                pred_series=pred_series,
                full_df=full_df,
                split_time=split_time,
                full_df_is_levels=full_df_is_levels,
                full_series_mode=args.full_series_mode,
                full_series_returns_mode=returns_mode,
                full_series_labels=args.full_series_labels,
                legend_cols=args.legend_cols,
                legend_fontsize=args.legend_fontsize,
                bins=args.bins,
                xlim_quantiles=tuple(args.xlim_quantiles) if args.xlim_quantiles else None,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = args.output_path
        if out_path is None:
            out_path = os.path.join(args.output_dir, f"stylized_facts_{dataset_name}_{args.split}_pred.{args.format}")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
