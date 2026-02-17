#!/usr/bin/env python
"""
Plot a small set of stylized facts for GluonTS-format datasets:
1) Heavy tails: return histogram vs. normal PDF (kurtosis reported).
2) Volatility clustering: |r_t| for a sample asset.
3) ACF of returns and squared returns.
4) Full-panel time series with train/test split marker (prefers raw levels if available).
   Optionally plot cumulative returns instead of levels.

Supports multiple datasets in one run; saves one figure per dataset.
"""

import argparse
import gzip
import json
import os
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm
from statsmodels.tsa.stattools import acf


def _load_series(path: str) -> List[pd.Series]:
    """Load all series from a GluonTS dataset split file (json or json.gz)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    # Be defensive: some datasets may ship a `.gz` filename that is not actually
    # gzip-compressed.
    def _open_text(p: str):
        with open(p, "rb") as fb:
            magic = fb.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(p, "rt", encoding="utf-8")
        return open(p, "rt", encoding="utf-8")

    series = []
    with _open_text(path) as fh:
        for line in fh:
            obj = json.loads(line)
            target = pd.to_numeric(pd.Series(obj["target"]), errors="coerce")
            start = pd.to_datetime(obj["start"])
            idx = pd.date_range(start=start, periods=len(target), freq="B")
            s = pd.Series(target.values, index=idx, name=obj.get("item_id"))
            series.append(s)
    if not series:
        raise ValueError(f"No series loaded from {path}")
    return series


def _dedupe_names(names: List[str]) -> List[str]:
    seen: dict[str, int] = {}
    out: List[str] = []
    for name in names:
        if name in seen:
            seen[name] += 1
            out.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _align_panel(series: List[pd.Series], max_assets: int) -> pd.DataFrame:
    """Align series on the shared index and cap asset count."""
    series = series[:max_assets]
    common_idx = series[0].index
    for s in series[1:]:
        common_idx = common_idx.intersection(s.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping timestamps across series.")
    aligned = [s.loc[common_idx] for s in series]
    names: List[str] = []
    for i, s in enumerate(series):
        name = str(s.name) if s.name not in {None, ""} else f"asset_{i}"
        names.append(name)
    df = pd.concat(aligned, axis=1)
    df.columns = _dedupe_names(names)
    return df


def _filter_assets(df: pd.DataFrame, exclude_assets: List[str] | None) -> pd.DataFrame:
    if not exclude_assets:
        return df
    exclude = [str(e).strip() for e in exclude_assets if str(e).strip()]
    if not exclude:
        return df
    exclude_set = set(exclude)
    exclude_upper = {e.upper() for e in exclude}
    keep_cols = [c for c in df.columns if str(c) not in exclude_set and str(c).upper() not in exclude_upper]
    if not keep_cols:
        raise ValueError("All assets were excluded; nothing left to plot.")
    dropped = [c for c in df.columns if c not in keep_cols]
    if dropped:
        print(f"[INFO] Dropped assets: {', '.join(map(str, dropped))}")
    return df[keep_cols]


def load_panel(data_dir: str, split: str, max_assets: int, exclude_assets: List[str] | None) -> pd.DataFrame:
    """Load and align a panel from a GluonTS dataset directory."""
    split_path_gz = os.path.join(data_dir, split, "data.json.gz")
    split_path_json = os.path.join(data_dir, split, "data.json")
    if os.path.exists(split_path_gz):
        path = split_path_gz
    elif os.path.exists(split_path_json):
        path = split_path_json
    else:
        raise FileNotFoundError(
            f"Could not find data.json(.gz) under {os.path.join(data_dir, split)}"
        )
    series = _load_series(path)
    df = _align_panel(series, max_assets)
    return _filter_assets(df, exclude_assets)


def load_panel_all(data_dir: str, max_assets: int, exclude_assets: List[str] | None) -> pd.DataFrame:
    train_df = load_panel(data_dir, "train", max_assets, exclude_assets)
    try:
        test_df = load_panel(data_dir, "test", max_assets, exclude_assets)
    except FileNotFoundError:
        return train_df
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if not common_cols:
        raise ValueError("No shared assets across train/test splits.")
    full_df = pd.concat([train_df[common_cols], test_df[common_cols]])
    return full_df[~full_df.index.duplicated(keep="last")]

def _try_load_levels_panel(data_dir: str, max_assets: int, exclude_assets: List[str] | None) -> pd.DataFrame | None:
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None
    notes = meta.get("notes", {}) if isinstance(meta, dict) else {}
    if not isinstance(notes, dict):
        return None
    levels_csv = notes.get("levels_csv")
    if not levels_csv:
        return None
    levels_path = levels_csv
    if not os.path.isabs(levels_path):
        levels_path = os.path.join(data_dir, levels_csv)
    if not os.path.exists(levels_path):
        return None
    df = pd.read_csv(levels_path)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    cols = [c for c in df.columns if c != "date"]
    df = df[cols[:max_assets]]
    if df.empty:
        return None
    return _filter_assets(df, exclude_assets)


def _infer_returns_mode(data_dir: str) -> str | None:
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    notes = meta.get("notes", {})
    if not isinstance(notes, dict):
        notes = {}
    returns = notes.get("returns")
    if isinstance(returns, str):
        returns = returns.lower()
        if returns in {"log", "simple", "none"}:
            return returns
    log_returns = notes.get("log_returns")
    if isinstance(log_returns, bool):
        return "log" if log_returns else "simple"
    target_transform = notes.get("target_transform")
    if isinstance(target_transform, str) and "log" in target_transform.lower():
        return "log"
    return None


def load_full_panel(
    data_dir: str,
    max_assets: int,
    exclude_assets: List[str] | None,
) -> Tuple[pd.DataFrame, pd.Timestamp | None, bool]:
    """Load a full panel with a split marker; uses raw levels if available."""
    split_time = None
    try:
        test_df = load_panel(data_dir, "test", max_assets, exclude_assets)
        split_time = test_df.index.min() if len(test_df.index) else None
    except FileNotFoundError:
        test_df = None

    levels_df = _try_load_levels_panel(data_dir, max_assets, exclude_assets)
    if levels_df is not None:
        return levels_df, split_time, True

    train_df = load_panel(data_dir, "train", max_assets, exclude_assets)
    if test_df is None:
        return train_df, split_time, False

    common_cols = [c for c in train_df.columns if c in test_df.columns]
    if not common_cols:
        raise ValueError("No shared assets across train/test splits.")
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]
    full_df = pd.concat([train_df, test_df])
    full_df = full_df[~full_df.index.duplicated(keep="last")]
    return full_df, split_time, False


def load_full_panel_for_split(
    data_dir: str,
    split: str,
    max_assets: int,
    exclude_assets: List[str] | None,
    split_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Timestamp | None, bool]:
    if split == "all":
        return load_full_panel(data_dir, max_assets, exclude_assets)
    levels_df = _try_load_levels_panel(data_dir, max_assets, exclude_assets)
    if levels_df is None:
        return split_df, None, False
    if split_df.empty:
        return split_df, None, False
    start = split_df.index.min()
    end = split_df.index.max()
    common_cols = [c for c in split_df.columns if c in levels_df.columns]
    if not common_cols:
        return split_df, None, False
    sliced = levels_df.loc[start:end, common_cols]
    if sliced.empty:
        return split_df, None, False
    return sliced, None, True


def _compute_cumulative_returns(
    df: pd.DataFrame,
    returns_mode: str | None,
    is_levels: bool,
) -> pd.DataFrame:
    if is_levels or returns_mode in {None, "none"}:
        base = df.iloc[0].replace(0, np.nan)
        return df.div(base) - 1.0
    if returns_mode == "log":
        return np.expm1(df.cumsum())
    if returns_mode == "simple":
        return (1.0 + df).cumprod() - 1.0
    return (1.0 + df).cumprod() - 1.0


def compute_acf_mean(arr: np.ndarray, nlags: int) -> np.ndarray:
    """Mean ACF across assets; arr shape [A, T]."""
    acfs = []
    for row in arr:
        row = np.asarray(row, dtype=float)
        row = row[np.isfinite(row)]
        if row.size < nlags + 2:
            continue
        if np.nanstd(row) < 1e-12:
            continue
        acfs.append(acf(row, nlags=nlags, fft=True))
    if not acfs:
        return np.zeros(nlags + 1, dtype=float)
    return np.nanmean(np.stack(acfs, axis=0), axis=0)


def _plot_full_series_with_split(
    ax: plt.Axes,
    full_df: pd.DataFrame,
    split_time: pd.Timestamp | None,
    title: str,
    ylabel: str,
    show_legend: bool,
    legend_cols: int,
    legend_fontsize: int,
) -> None:
    for col in full_df.columns:
        ax.plot(full_df.index, full_df[col].values, lw=0.6, alpha=0.6, label=str(col))
    if split_time is not None:
        ax.axvline(split_time, color="k", linestyle="--", lw=1.0)
        ax.text(
            split_time,
            0.98,
            "train/test split",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_xlabel("Time", labelpad=1)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30, pad=1)
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            ncol=max(1, legend_cols),
            fontsize=legend_fontsize,
            frameon=False,
        )


def _plot_stylized_facts_axes(
    axes: np.ndarray,
    df: pd.DataFrame,
    dataset_name: str,
    acf_lags: int,
    rolling_window: int,
    full_df: pd.DataFrame | None = None,
    split_time: pd.Timestamp | None = None,
    full_df_is_levels: bool = False,
    full_series_mode: str = "levels",
    full_series_returns_mode: str | None = None,
    full_series_labels: str = "none",
    legend_cols: int = 3,
    legend_fontsize: int = 7,
    panel_labels: List[str] | None = None,
) -> None:
    """Render the 4 stylized-facts panels onto provided axes (2x2)."""
    returns = df.values.T.astype(float)  # [A, T]

    # Drop near-constant assets (e.g. USD==1.0 after per-USD conversion).
    stds = np.nanstd(returns, axis=1)
    keep = np.isfinite(stds) & (stds > 1e-12)
    if not np.any(keep):
        raise ValueError("All assets appear constant/degenerate; cannot compute stylized facts.")
    if np.sum(~keep) > 0:
        dropped = int(np.sum(~keep))
        print(f"[INFO] Dropped {dropped} degenerate asset(s) (std≈0) from plots: {dataset_name}")
    returns = returns[keep]
    flattened = returns.flatten()
    flattened = flattened[np.isfinite(flattened)]
    mu, sigma = np.mean(flattened), np.std(flattened)
    k = kurtosis(flattened, fisher=False, bias=False)

    axes = np.asarray(axes)
    if axes.shape != (2, 2):
        raise ValueError(f"Expected axes shape (2, 2), got {axes.shape}")

    # 1) Heavy tails
    ax = axes[0, 0]
    ax.hist(flattened, bins=60, density=True, alpha=0.6, color="#1f77b4", label="Returns")
    xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(
        xs,
        norm.pdf(xs, mu, sigma),
        color="#2ca02c",
        linestyle="--",
        label=f"Normal ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})",
    )
    ax.set_title(f"Heavy tails (kurtosis={k:.2f})", fontsize=11, pad=2)
    ax.set_xlabel("Log Return", labelpad=1)
    ax.set_ylabel("Density")
    ax.legend()

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
        title = "All series with train/test split"
        ylabel = "Value"
        if full_series_mode == "cum_returns":
            plot_df = _compute_cumulative_returns(
                full_df,
                returns_mode=full_series_returns_mode,
                is_levels=full_df_is_levels,
            )
            title = "Cumulative returns with train/test split"
            ylabel = "Cumulative return"
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

    if panel_labels and len(panel_labels) == 4:
        label_axes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (r, c), label in zip(label_axes, panel_labels):
            axes[r, c].text(
                0.02,
                0.98,
                label,
                transform=axes[r, c].transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            )

def plot_stylized_facts(
    df: pd.DataFrame,
    dataset_name: str,
    out_path: str,
    acf_lags: int,
    rolling_window: int,
    full_df: pd.DataFrame | None = None,
    split_time: pd.Timestamp | None = None,
    full_df_is_levels: bool = False,
    full_series_mode: str = "levels",
    full_series_returns_mode: str | None = None,
    full_series_labels: str = "none",
    legend_cols: int = 3,
    legend_fontsize: int = 7,
):
    """Create and save the 4-panel stylized facts figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Stylized Facts: {dataset_name}", fontsize=14)
    _plot_stylized_facts_axes(
        axes,
        df,
        dataset_name=dataset_name,
        acf_lags=acf_lags,
        rolling_window=rolling_window,
        full_df=full_df,
        split_time=split_time,
        full_df_is_levels=full_df_is_levels,
        full_series_mode=full_series_mode,
        full_series_returns_mode=full_series_returns_mode,
        full_series_labels=full_series_labels,
        legend_cols=legend_cols,
        legend_fontsize=legend_fontsize,
        panel_labels=None,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_stylized_facts_combined(
    entries: List[dict],
    out_path: str,
    acf_lags: int,
    rolling_window: int,
    full_series_mode: str,
    full_series_labels: str,
    legend_cols: int,
    legend_fontsize: int,
    combined_cols: int = 2,
) -> None:
    """Create and save a combined multi-dataset stylized facts figure."""
    if not entries:
        return
    cols = max(1, min(combined_cols, len(entries)))
    rows = int(math.ceil(len(entries) / cols))
    fig = plt.figure(figsize=(10.4 * cols, 7.2 * rows))
    outer = fig.add_gridspec(rows, cols, wspace=0.16, hspace=0.24)
    title_row_ratio = 0.05
    label_idx = 0
    letters = "abcdefghijklmnopqrstuvwxyz"
    for idx, entry in enumerate(entries):
        r, c = divmod(idx, cols)
        inner = outer[r, c].subgridspec(
            3,
            2,
            height_ratios=[title_row_ratio, 1.0, 1.0],
            hspace=0.45,
            wspace=0.18,
        )
        ax_title = fig.add_subplot(inner[0, :])
        ax_title.axis("off")
        ax_title.text(
            0.5,
            0.5,
            f"Stylized Facts: {entry['dataset_name']}",
            ha="center",
            va="center",
            fontsize=12,
        )
        panel_labels = []
        for _ in range(4):
            label = f"({letters[label_idx]})" if label_idx < len(letters) else f"({label_idx + 1})"
            panel_labels.append(label)
            label_idx += 1
        axes = np.array(
            [
                [fig.add_subplot(inner[1, 0]), fig.add_subplot(inner[1, 1])],
                [fig.add_subplot(inner[2, 0]), fig.add_subplot(inner[2, 1])],
            ]
        )
        _plot_stylized_facts_axes(
            axes,
            entry["df"],
            dataset_name=entry["dataset_name"],
            acf_lags=acf_lags,
            rolling_window=rolling_window,
            full_df=entry["full_df"],
            split_time=entry["split_time"],
            full_df_is_levels=entry["full_df_is_levels"],
            full_series_mode=full_series_mode,
            full_series_returns_mode=entry["returns_mode"],
            full_series_labels=full_series_labels,
            legend_cols=legend_cols,
            legend_fontsize=legend_fontsize,
            panel_labels=panel_labels,
        )
    for idx in range(len(entries), rows * cols):
        r, c = divmod(idx, cols)
        ax = fig.add_subplot(outer[r, c])
        ax.axis("off")
    fig.subplots_adjust(top=0.98, bottom=0.06, left=0.05, right=0.97)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined: {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot stylized facts for GluonTS datasets.")
    p.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="List of dataset directories (each must contain train/ or test/ with data.json(.gz)).",
    )
    p.add_argument("--split", default="train", choices=["train", "test", "all"], help="Which split to load.")
    p.add_argument("--max-assets", type=int, default=8, help="Max number of assets to use.")
    p.add_argument("--acf-lags", type=int, default=50, help="Number of lags for ACF.")
    p.add_argument("--rolling-window", type=int, default=60, help="Rolling window (timesteps) for correlation dynamics.")
    p.add_argument(
        "--full-series-mode",
        choices=["levels", "cum_returns"],
        default="levels",
        help="Right-bottom panel: raw levels (default) or cumulative returns.",
    )
    p.add_argument(
        "--exclude-assets",
        nargs="*",
        default=None,
        help="Asset names to exclude (case-insensitive).",
    )
    p.add_argument(
        "--full-series-labels",
        choices=["none", "legend"],
        default="none",
        help="Label mode for the right-bottom panel.",
    )
    p.add_argument("--full-series-legend-cols", type=int, default=3, help="Legend columns for full series plot.")
    p.add_argument("--full-series-legend-fontsize", type=int, default=7, help="Legend font size.")
    p.add_argument("--output-dir", default="assets", help="Directory to save figures.")
    p.add_argument(
        "--combined-output",
        default=None,
        help="Optional path for a combined multi-dataset figure (default: <output-dir>/stylized_facts_<split>_combined.<format>).",
    )
    p.add_argument(
        "--combined-cols",
        type=int,
        default=2,
        help="Number of dataset columns in the combined figure (default: 2).",
    )
    p.add_argument(
        "--output-format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format (default: png).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    fmt = args.output_format.lstrip(".").lower()
    entries = []
    for data_dir in args.data_dirs:
        name = os.path.basename(os.path.abspath(data_dir))
        if args.split == "all":
            df = load_panel_all(data_dir, args.max_assets, args.exclude_assets)
        else:
            df = load_panel(data_dir, args.split, args.max_assets, args.exclude_assets)
        try:
            full_df, split_time, full_df_is_levels = load_full_panel_for_split(
                data_dir, args.split, args.max_assets, args.exclude_assets, df
            )
        except FileNotFoundError:
            full_df, split_time, full_df_is_levels = df, None, False
        returns_mode = _infer_returns_mode(data_dir)
        out_path = os.path.join(args.output_dir, f"stylized_facts_{name}_{args.split}.{fmt}")
        plot_stylized_facts(
            df,
            dataset_name=name,
            out_path=out_path,
            acf_lags=args.acf_lags,
            rolling_window=args.rolling_window,
            full_df=full_df,
            split_time=split_time,
            full_df_is_levels=full_df_is_levels,
            full_series_mode=args.full_series_mode,
            full_series_returns_mode=returns_mode,
            full_series_labels=args.full_series_labels,
            legend_cols=args.full_series_legend_cols,
            legend_fontsize=args.full_series_legend_fontsize,
        )
        entries.append(
            {
                "df": df,
                "dataset_name": name,
                "full_df": full_df,
                "split_time": split_time,
                "full_df_is_levels": full_df_is_levels,
                "returns_mode": returns_mode,
            }
        )

    if args.combined_output or len(entries) > 1:
        if args.combined_output:
            combined_path = args.combined_output
        else:
            combined_path = os.path.join(
                args.output_dir, f"stylized_facts_{args.split}_combined.{fmt}"
            )
        plot_stylized_facts_combined(
            entries,
            out_path=combined_path,
            acf_lags=args.acf_lags,
            rolling_window=args.rolling_window,
            full_series_mode=args.full_series_mode,
            full_series_labels=args.full_series_labels,
            legend_cols=args.full_series_legend_cols,
            legend_fontsize=args.full_series_legend_fontsize,
            combined_cols=args.combined_cols,
        )


if __name__ == "__main__":
    main()
