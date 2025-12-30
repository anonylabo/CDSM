#!/usr/bin/env python3
"""
Label sliding windows (train/val/test) as bull / bear aligned with the same
windowing used by the datamodules (val_ratio<0 / >0, train_val_gap, val_test_gap,
align_tail_windows). Two methods:
- threshold (default): rolling log-return over regime_window; >threshold -> bull.
- hamilton: 2-state Markov switching model on equal-weight returns (Hamilton 1989),
  high-mean state = bull, low-mean state = bear.

Example (match eval params; val_ratio<0 pulls val from train, test from test):
python scripts/label_window_regimes.py \
  --data-dir ~/.gluonts/datasets/ishares14_clean_2 \
  --split test \
  --max-assets 14 \
  --context-len 40 --pred-len 20 --stride 1 --val-ratio -0.1 \
  --train-val-gap -1 --val-test-gap -1 --align-tail-windows \
  --regime-window 60 --regime-threshold 0 \
  --out assets/window_regimes_ishares14_clean_2_test.json \
  --stats-out assets/window_regimes_ishares14_clean_2_test_stats.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from cfdiff.utils.windowing import compute_window_positions


def _load_series(path: str) -> List[pd.Series]:
    # Some datasets ship both `data.json` and `data.json.gz`. We prefer `.gz` when
    # present, but be defensive: a file may be named `.gz` without being actually
    # gzip-compressed (e.g. created by a custom script).
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


def compute_regime_series(eq_ret: pd.Series, window: int, threshold: float) -> pd.Series:
    log_ret = np.log1p(eq_ret)
    roll = log_ret.rolling(window, min_periods=max(1, window // 2)).sum()
    regime = pd.Series(index=eq_ret.index, dtype="object")
    regime[roll > threshold] = "bull"
    regime[roll <= threshold] = "bear"
    return regime.ffill().bfill()


def compute_regime_series_warm_start(
    eq_ret: pd.Series,
    warm_start_eq_ret: pd.Series,
    window: int,
    threshold: float,
) -> pd.Series:
    """
    Warm-start the rolling regime signal by prepending (past) returns.

    This is useful when labelling test split windows: we often want the rolling
    window to be "full" from the first test date, using the tail of train split
    as history.
    """

    if warm_start_eq_ret is None or warm_start_eq_ret.empty:
        return compute_regime_series(eq_ret, window=window, threshold=threshold)

    log_ret = np.log1p(eq_ret)
    warm = np.log1p(warm_start_eq_ret)

    combined = pd.concat([warm, log_ret])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    min_periods = window if len(warm) >= window - 1 else max(1, window // 2)
    roll = combined.rolling(window, min_periods=min_periods).sum()
    roll = roll.reindex(log_ret.index)

    regime = pd.Series(index=eq_ret.index, dtype="object")
    regime[roll > threshold] = "bull"
    regime[roll <= threshold] = "bear"
    return regime.ffill().bfill()


def compute_regime_hamilton(eq_ret: pd.Series) -> pd.Series:
    """
    2-state Markov switching on returns (Hamilton, 1989).
    High-mean state -> bull, low-mean state -> bear.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except Exception as exc:  # pragma: no cover
        raise SystemExit("statsmodels is required for --regime-method hamilton. pip install statsmodels") from exc

    mod = MarkovRegression(eq_ret.values, k_regimes=2, trend="c", switching_variance=True)
    res = mod.fit(disp=False)
    probs = res.smoothed_marginal_probabilities
    probs_arr = probs.values if hasattr(probs, "values") else np.asarray(probs)
    if probs_arr.shape[0] == len(eq_ret) and probs_arr.ndim == 2:
        # shape: time x states
        states = np.argmax(probs_arr, axis=1)
    elif probs_arr.shape[1] == len(eq_ret):
        # shape: states x time
        states = np.argmax(probs_arr, axis=0)
    else:  # fallback
        states = np.argmax(probs_arr, axis=-1)
    state_mean = {}
    for s in [0, 1]:
        mask = np.asarray(states) == s
        state_mean[s] = float(np.mean(eq_ret.values[mask])) if mask.any() else -np.inf
    bull_state = max(state_mean, key=state_mean.get)
    regime_labels = np.where(np.asarray(states) == bull_state, "bull", "bear")
    return pd.Series(regime_labels, index=eq_ret.index)


def iter_windows(df: pd.DataFrame, context_len: int, pred_len: int, stride: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Index]]:
    """Return (start_ts, end_ts, index_slice) for each full window (C+P)."""
    n = len(df)
    L = context_len + pred_len
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Index]] = []
    for start in range(0, n - L + 1, stride):
        end = start + L
        idx = df.index[start:end]
        windows.append((idx[0], idx[-1], idx))
    return windows


def label_windows(df: pd.DataFrame, regime: pd.Series, context_len: int, pred_len: int, stride: int) -> List[dict]:
    windows = iter_windows(df, context_len, pred_len, stride)
    labels: List[dict] = []
    for widx, (start_ts, end_ts, idx) in enumerate(windows):
        # use regime at window end（可改为其他规则）
        reg = regime.loc[end_ts]
        labels.append(
            {
                "window_idx": widx,
                "start": str(start_ts.date()),
                "end": str(end_ts.date()),
                "regime": reg,
            }
        )
    return labels


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Label sliding windows as bull/bear for later evaluation grouping.")
    ap.add_argument("--data-dir", required=True, help="GluonTS dataset root.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to label.")
    ap.add_argument("--max-assets", type=int, default=14, help="Max assets to include.")
    ap.add_argument("--context-len", type=int, required=True, help="Context length C.")
    ap.add_argument("--pred-len", type=int, required=True, help="Prediction length P.")
    ap.add_argument("--stride", type=int, default=1, help="Window stride.")
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Same semantics as datamodule: <0 pulls val from train windows; >=0 pulls val from target split.",
    )
    ap.add_argument(
        "--train-val-gap",
        type=int,
        default=-1,
        help="Gap between train and val when val_ratio<0 (same rule as datamodule; <0 -> auto ceil((L+stride-1)/stride)-1).",
    )
    ap.add_argument(
        "--val-test-gap",
        type=int,
        default=-1,
        help="Gap between val and test when val_ratio>=0 (same rule as datamodule; <0 -> auto ceil((L+stride-1)/stride)-1).",
    )
    ap.add_argument(
        "--align-tail-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match datamodule.align_tail_windows (append final window to touch end if stride misaligns).",
    )
    ap.add_argument(
        "--regime-method",
        choices=["threshold", "hamilton"],
        default="threshold",
        help="Method for regime detection: 'threshold' uses rolling log-return; 'hamilton' uses 2-state Markov switching (Hamilton 1989).",
    )
    ap.add_argument(
        "--warm-start-from-train",
        action="store_true",
        help="When labelling --split test with --regime-method threshold, prepend the last (regime_window-1) points from train split so the rolling regime signal is available from the first test date.",
    )
    ap.add_argument("--regime-window", type=int, default=60, help="Rolling window for threshold method (timesteps).")
    ap.add_argument("--regime-threshold", type=float, default=0.0, help="Threshold on rolling log-return (> -> bull) for threshold method.")
    ap.add_argument("--out", required=True, help="Output JSON manifest path.")
    ap.add_argument("--stats-out", help="Optional JSON path to write summary stats (counts per regime, total).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    window_total = args.context_len + args.pred_len
    stride = int(args.stride)
    align_tail = bool(args.align_tail_windows)
    val_ratio = float(args.val_ratio)

    def _positions(total_len: int) -> List[Tuple[int, int]]:
        return compute_window_positions(total_len, window_total, stride, align_end=align_tail)

    def _regime_for(df: pd.DataFrame, warm_start: pd.DataFrame | None = None) -> pd.Series:
        eq_ret = df.mean(axis=1)
        if args.regime_method == "threshold":
            warm = None
            if warm_start is not None:
                split_start = df.index[0]
                eq_train = warm_start.mean(axis=1)
                eq_train = eq_train.loc[eq_train.index < split_start]
                warm = eq_train.tail(max(0, int(args.regime_window) - 1))
            if warm is not None and not warm.empty:
                return compute_regime_series_warm_start(eq_ret, warm, window=args.regime_window, threshold=args.regime_threshold)
            return compute_regime_series(eq_ret, window=args.regime_window, threshold=args.regime_threshold)
        return compute_regime_hamilton(eq_ret)

    def _label_positions(df: pd.DataFrame, positions: List[Tuple[int, int]], regime: pd.Series, offset: int = 0) -> List[dict]:
        labels: List[dict] = []
        for local_idx, (start, end) in enumerate(positions):
            idx_slice = df.index[start:end]
            if len(idx_slice) == 0:
                continue
            reg = regime.loc[idx_slice[-1]]
            labels.append(
                {
                    "window_idx": offset + local_idx,
                    "start": str(idx_slice[0].date()),
                    "end": str(idx_slice[-1].date()),
                    "regime": reg,
                }
            )
        return labels

    train_df = None
    test_df = None
    if args.split in {"train", "val"} or val_ratio < 0 or args.warm_start_from_train:
        train_df = load_panel(args.data_dir, "train", args.max_assets)
    if args.split in {"test", "val"} or val_ratio >= 0:
        test_df = load_panel(args.data_dir, "test", args.max_assets)

    regime_train = _regime_for(train_df) if train_df is not None else None
    regime_test = None
    if test_df is not None:
        warm_df = train_df if (args.warm_start_from_train and args.regime_method == "threshold") else None
        regime_test = _regime_for(test_df, warm_start=warm_df)

    if train_df is None and args.split in {"train", "val"}:
        raise SystemExit("Train split required but not available.")
    if test_df is None and args.split in {"test", "val"}:
        raise SystemExit("Test split required but not available.")

    labels: List[dict] = []
    stats_total = {}
    val_count = 0

    if val_ratio < 0:
        # val from train split; test from test split.
        train_positions = _positions(len(train_df))
        total_train = len(train_positions)
        if total_train == 0:
            raise SystemExit("No training windows generated; check context_len/pred_len/stride.")
        val_count = int(round(total_train * abs(val_ratio)))
        min_val = 2 if total_train >= 2 else 1
        val_count = max(min_val, val_count)
        if val_count >= total_train:
            raise SystemExit(
                f"val_ratio={val_ratio} requests {val_count} val windows but only {total_train} train windows exist."
            )
        if args.train_val_gap < 0:
            gap = max(0, (window_total + stride - 1) // stride - 1)
        else:
            gap = max(0, int(args.train_val_gap))
        train_end = total_train - val_count - gap
        if train_end <= 0:
            raise SystemExit(
                f"No training windows remain after applying train_val_gap (val_count={val_count}, gap={gap}, total_train={total_train})."
            )
        train_positions_final = train_positions[:train_end]
        val_start = total_train - val_count
        val_positions = train_positions[val_start:]
        test_positions = _positions(len(test_df)) if test_df is not None else []
    else:
        # val/test both from test split.
        train_positions_final = _positions(len(train_df)) if train_df is not None else []
        eval_positions = _positions(len(test_df)) if test_df is not None else []
        if len(eval_positions) == 0:
            raise SystemExit("No evaluation windows generated; check context_len/pred_len/stride.")
        val_count = int(round(len(eval_positions) * val_ratio)) if val_ratio > 0 else 1
        min_val = 2 if len(eval_positions) >= 2 else 1
        val_count = max(min_val, val_count)
        val_count = min(val_count, len(eval_positions))
        val_positions = eval_positions[:val_count]
        if args.val_test_gap < 0:
            gap = max(0, (window_total + stride - 1) // stride - 1)
        else:
            gap = max(0, int(args.val_test_gap))
        test_start = min(len(eval_positions), val_count + gap)
        if test_start >= len(eval_positions):
            raise SystemExit(
                f"No test windows remain after applying val_test_gap (val_count={val_count}, gap={gap}, total_eval={len(eval_positions)})."
            )
        test_positions = eval_positions[test_start:]

    if args.split == "train":
        labels = _label_positions(train_df, train_positions_final, regime_train, offset=0)
        stats_total = {"train_windows": len(train_positions_final)}
    elif args.split == "val":
        if val_ratio < 0:
            labels = _label_positions(train_df, val_positions, regime_train, offset=0)
        else:
            labels = _label_positions(test_df, val_positions, regime_test, offset=0)
        stats_total = {"val_windows": len(val_positions)}
    else:  # test
        labels = _label_positions(test_df, test_positions, regime_test, offset=val_count)
        stats_total = {"test_windows": len(test_positions), "val_offset": val_count}

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Saved regime manifest with {len(labels)} windows to {out_path}")
    counts = {}
    for x in labels:
        counts[x["regime"]] = counts.get(x["regime"], 0) + 1
    bulls = counts.get("bull", 0)
    bears = counts.get("bear", 0)
    print(f"  bull windows: {bulls} | bear windows: {bears}")
    if args.stats_out:
        stats_path = Path(args.stats_out).expanduser()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "split": args.split,
            "total_windows": len(labels),
            "bull": bulls,
            "bear": bears,
            "bull_ratio": bulls / len(labels) if labels else 0.0,
            "bear_ratio": bears / len(labels) if labels else 0.0,
            **stats_total,
        }
        stats_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  stats written to {stats_path}")


if __name__ == "__main__":
    main()
