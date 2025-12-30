#!/usr/bin/env python3
"""
Visualize bull/bear regimes on a GluonTS panel (train/test/all) by using an
equal-weighted portfolio and a rolling log-return filter.

Steps:
- Load panel from data.json(.gz), align assets, cap to --max-assets.
- Build equal-weight portfolio returns.
- Compute cumulative equity curve (log1p compounding).
- Compute rolling log-return over --regime-window, annualize it, and
  classify bull (rolling > threshold) vs bear (<= threshold).
- Plot equity curve with bull/bear shading, and the rolling annualized
  return with the same shading.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_series(path: str) -> List[pd.Series]:
    # Be defensive: some datasets may ship a `.gz` filename that is not actually
    # gzip-compressed.
    def _open_text(p: str):
        with open(p, "rb") as fb:
            magic = fb.read(2)
        if magic == b"\x1f\x8b":
            return gzip.open(p, "rt", encoding="utf-8")
        return open(p, "rt", encoding="utf-8")

    series: List[pd.Series] = []
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


def _align_panel(series: List[pd.Series], max_assets: int) -> pd.DataFrame:
    series = series[:max_assets]
    common_idx = series[0].index
    for s in series[1:]:
        common_idx = common_idx.intersection(s.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping timestamps across series.")
    aligned = [s.loc[common_idx] for s in series]
    df = pd.concat(aligned, axis=1)
    df.columns = [f"asset_{i}" for i in range(df.shape[1])]
    return df


def load_panel(data_dir: str, split: str, max_assets: int) -> pd.DataFrame:
    split_path_gz = os.path.join(data_dir, split, "data.json.gz")
    split_path_json = os.path.join(data_dir, split, "data.json")
    if os.path.exists(split_path_gz):
        path = split_path_gz
    elif os.path.exists(split_path_json):
        path = split_path_json
    else:
        raise FileNotFoundError(f"Could not find data.json(.gz) under {os.path.join(data_dir, split)}")
    series = _load_series(path)
    return _align_panel(series, max_assets)


def compute_regimes(
    eq_ret: pd.Series,
    window: int,
    threshold: float,
    annualize: int,
    warm_start_eq_ret: pd.Series | None = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return cumulative equity curve, rolling annualised log-return, and regime labels (+1 bull, -1 bear)."""
    log_ret = np.log1p(eq_ret)
    cum_equity = np.exp(log_ret.cumsum()) - 1.0

    if warm_start_eq_ret is not None and not warm_start_eq_ret.empty:
        warm = np.log1p(warm_start_eq_ret)
        combined = pd.concat([warm, log_ret])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        min_periods = window if len(warm) >= window - 1 else max(1, window // 2)
        roll = combined.rolling(window, min_periods=min_periods).sum()
        roll = roll.reindex(log_ret.index)
    else:
        roll = log_ret.rolling(window, min_periods=max(1, window // 2)).sum()

    roll_ann = roll * (annualize / float(window))
    regime = pd.Series(index=eq_ret.index, dtype=float)
    regime[roll > threshold] = 1
    regime[roll <= threshold] = -1
    regime = regime.ffill().bfill().astype(int)
    return cum_equity, roll_ann, regime


def _segments(regime: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    segs: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    current = int(regime.iloc[0])
    start = regime.index[0]
    for idx, val in regime.iloc[1:].items():
        if int(val) != current:
            segs.append((start, idx, current))
            start = idx
            current = int(val)
    segs.append((start, regime.index[-1], current))
    return segs


def plot_regimes(
    df: pd.DataFrame,
    dataset: str,
    split: str,
    out_path: str,
    window: int,
    threshold: float,
    annualize: int,
    warm_start_df: pd.DataFrame | None = None,
    split_boundary: pd.Timestamp | None = None,
    boundary_label: str | None = None,
):
    eq_ret = df.mean(axis=1)
    warm_start_eq = None
    if warm_start_df is not None and not warm_start_df.empty:
        split_start = df.index[0]
        warm_eq = warm_start_df.mean(axis=1)
        warm_eq = warm_eq.loc[warm_eq.index < split_start]
        warm_start_eq = warm_eq.tail(max(0, window - 1))

    cum_equity, roll_ann, regime = compute_regimes(
        eq_ret,
        window=window,
        threshold=threshold,
        annualize=annualize,
        warm_start_eq_ret=warm_start_eq,
    )
    segs = _segments(regime)
    bull_color = "#b5e3b5"
    bear_color = "#f5b7ae"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Bull/Bear Regimes: {dataset} ({split})", fontsize=14)

    # Equity curve with shading
    ax = axes[0]
    ax.plot(cum_equity.index, cum_equity.values, color="tab:blue", lw=1.4, label="Equal-weight equity")
    for start, end, label in segs:
        color = bull_color if label == 1 else bear_color
        ax.axvspan(start, end, color=color, alpha=0.6, linewidth=0)
    if split_boundary is not None:
        ax.axvline(split_boundary, color="black", lw=1.0, ls="--", alpha=0.7)
        if boundary_label:
            y_top = ax.get_ylim()[1]
            ax.text(
                split_boundary,
                y_top,
                boundary_label,
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.0),
            )
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Cumulative return")
    ax.legend(loc="upper left")

    # Rolling annualized log-return
    ax = axes[1]
    ax.plot(roll_ann.index, roll_ann.values, color="tab:orange", lw=1.2, label=f"Rolling log return (ann., window={window})")
    ax.axhline(threshold * (annualize / float(window)), color="gray", lw=0.8, ls="--", label="Threshold (annualized)")
    ax.axhline(0.0, color="black", lw=0.8, ls=":")
    for start, end, label in segs:
        color = bull_color if label == 1 else bear_color
        ax.axvspan(start, end, color=color, alpha=0.45, linewidth=0)
    if split_boundary is not None:
        ax.axvline(split_boundary, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_ylabel("Rolling ann. log return")
    ax.set_xlabel("Time")
    bull_patch = mpatches.Patch(color=bull_color, label="Bull")
    bear_patch = mpatches.Patch(color=bear_color, label="Bear")
    ax.legend(handles=[bull_patch, bear_patch], loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    bull_days = int((regime == 1).sum())
    bear_days = int((regime == -1).sum())
    print(f"Saved: {out_path} | bull days={bull_days}, bear days={bear_days}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot bull/bear regimes for GluonTS panels.")
    p.add_argument("--data-dirs", nargs="+", required=True, help="Dataset directories containing train/test splits.")
    p.add_argument("--split", default="train", choices=["train", "test", "all"], help="Split to load.")
    p.add_argument("--max-assets", type=int, default=14, help="Max assets to include.")
    p.add_argument(
        "--warm-start-from-train",
        action="store_true",
        help="When plotting --split test, prepend the last (regime_window-1) points from train split so rolling regimes are available from the first test date.",
    )
    p.add_argument("--regime-window", type=int, default=60, help="Rolling window size (timesteps) for regime detection.")
    p.add_argument(
        "--regime-threshold",
        type=float,
        default=0.0,
        help="Threshold on rolling log-return (non-annualized) to classify bull vs bear.",
    )
    p.add_argument("--annualization", type=int, default=252, help="Timesteps per year for annualizing rolling return.")
    p.add_argument(
        "--show-split-boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --split all, draw a vertical line at the train/test boundary.",
    )
    p.add_argument(
        "--split-boundary-label",
        type=str,
        default=None,
        help="Optional label text to annotate the train/test boundary (only with --split all).",
    )
    p.add_argument("--output-dir", default="assets", help="Directory to save figures.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for data_dir in args.data_dirs:
        name = os.path.basename(os.path.abspath(data_dir))
        split_boundary = None
        if args.split == "all":
            train_df = load_panel(data_dir, "train", args.max_assets)
            test_df = load_panel(data_dir, "test", args.max_assets)
            common_cols = train_df.columns.intersection(test_df.columns)
            if len(common_cols) == 0:
                raise SystemExit("No overlapping assets between train and test.")
            train_df = train_df[common_cols]
            test_df = test_df[common_cols]
            split_boundary = test_df.index[0] if len(test_df.index) else None
            df = pd.concat([train_df, test_df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = load_panel(data_dir, args.split, args.max_assets)
        warm_df = None
        if args.warm_start_from_train and args.split == "test":
            try:
                warm_df = load_panel(data_dir, "train", args.max_assets)
            except Exception:
                warm_df = None
        out_path = os.path.join(args.output_dir, f"bull_bear_{name}_{args.split}.png")
        plot_regimes(
            df,
            dataset=name,
            split=args.split,
            out_path=out_path,
            window=args.regime_window,
            threshold=args.regime_threshold,
            annualize=args.annualization,
            warm_start_df=warm_df,
            split_boundary=split_boundary if args.split == "all" and args.show_split_boundary else None,
            boundary_label=args.split_boundary_label,
        )


if __name__ == "__main__":
    main()
