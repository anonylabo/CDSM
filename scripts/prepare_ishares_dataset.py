#!/usr/bin/env python3
"""
Prepare local iShares price panels as a GluonTS-style dataset.

Example:
python scripts/prepare_ishares_dataset.py \
  --src-file external/conditional_fourier_diffusion/ishares/df_ishares6x4618.csv \
  --dst-dir ~/.gluonts/datasets/ishares6_clean \
  --train-ratio 0.8 \
  --trim-start 2018-01-01 \
  --trim-end 2023-12-31 \
  --returns log \
  --overwrite
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def _compute_returns(series: List[float], mode: str) -> List[float]:
    """Compute simple or log returns from a price series."""
    if len(series) < 2:
        raise ValueError("Series too short to compute returns.")
    returns: List[float] = []
    prev = float(series[0])
    for value_raw in series[1:]:
        value = float(value_raw)
        if mode == "simple":
            if prev == 0.0:
                raise ValueError("Encountered zero price while computing simple returns.")
            ret = (value - prev) / prev
        else:  # log returns
            if prev <= 0.0 or value <= 0.0:
                raise ValueError("Log returns require strictly positive prices.")
            ret = math.log(value) - math.log(prev)
        returns.append(ret)
        prev = value
    return returns


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_jsonl_gz(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as gh:
        for rec in records:
            gh.write(json.dumps(rec) + "\n")


def _normalize_nan_policy(nan_policy: str) -> str:
    policy = str(nan_policy or "none").strip().lower()
    aliases = {
        "keep": "none",
        "error": "raise",
        "forward_fill": "ffill",
        "fill_zero": "zero",
    }
    return aliases.get(policy, policy)


def _apply_nan_policy(df: pd.DataFrame, nan_policy: str) -> pd.DataFrame:
    policy = _normalize_nan_policy(nan_policy)
    if policy == "none":
        return df

    df = df.replace([np.inf, -np.inf], np.nan)
    if policy == "raise":
        arr = df.to_numpy()
        finite = np.isfinite(arr)
        if not finite.all():
            bad = int(arr.size - finite.sum())
            raise ValueError(f"Found {bad} non-finite values with nan_policy='raise'.")
        return df
    if policy == "drop":
        return df.dropna(how="any")
    if policy == "ffill":
        return df.ffill()
    if policy == "zero":
        return df.fillna(0.0)
    raise ValueError(f"Unsupported nan_policy='{nan_policy}'")


def _build_records(
    df: pd.DataFrame,
    freq: str,
    returns: str,
    item_ids: List[str],
) -> List[dict]:
    """Convert a price panel into GluonTS json records."""
    offset = pd.tseries.frequencies.to_offset(freq)
    records: List[dict] = []
    for idx, col in enumerate(item_ids):
        series = df[col].astype(float).tolist()
        if returns == "none":
            target = series
            start_ts = df.index[0]
        else:
            target = _compute_returns(series, returns)
            start_ts = df.index[0] + offset
        records.append(
            {
                "start": str(start_ts),
                "target": target,
                "item_id": col,
                "feat_static_cat": [idx],
            }
        )
    return records


def _write_metadata(
    dst_dir: Path,
    freq: str,
    prediction_length: int,
    num_series: int,
    notes: dict | None = None,
) -> None:
    meta = {
        "freq": freq,
        "target": None,
        "feat_static_cat": [
            {
                "name": "feat_static_cat_0",
                "cardinality": str(num_series),
            }
        ],
        "feat_static_real": [],
        "feat_dynamic_real": [],
        "feat_dynamic_cat": [],
        "prediction_length": prediction_length,
    }
    if notes:
        meta["notes"] = notes
    meta_path = dst_dir / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


def prepare_dataset(
    src_file: Path,
    dst_dir: Path,
    train_ratio: float,
    trim_start: str | None,
    trim_end: str | None,
    returns: str,
    prediction_length: int,
    freq: str,
    overwrite: bool,
    nan_policy: str,
) -> None:
    df = pd.read_csv(src_file)
    if "Date" not in df.columns:
        raise SystemExit("Expected a 'Date' column in the source CSV.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    asset_cols = [c for c in df.columns if c != "Date"]
    if not asset_cols:
        raise SystemExit("No asset columns found in the source CSV.")

    # Trim range
    if trim_start:
        df = df[df.index >= pd.to_datetime(trim_start)]
    if trim_end:
        df = df[df.index <= pd.to_datetime(trim_end)]

    # Handle NaN/Inf after trimming (default: none).
    df = _apply_nan_policy(df, nan_policy)
    if df.empty:
        raise SystemExit("No data left after trimming/cleaning.")

    # Optional safety for log returns
    if returns == "log" and (df <= 0).any().any():
        raise SystemExit("Log returns require strictly positive prices; found non-positive values.")

    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise SystemExit(f"Invalid train_ratio={train_ratio}; results in empty train or test split.")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_records = _build_records(train_df, freq=freq, returns=returns, item_ids=asset_cols)
    test_records = _build_records(test_df, freq=freq, returns=returns, item_ids=asset_cols)

    train_path = dst_dir / "train" / "train.json"
    test_path = dst_dir / "test" / "test.json"
    if not overwrite and (train_path.exists() or test_path.exists()):
        raise FileExistsError(f"{train_path} or {test_path} already exists; use --overwrite to replace.")

    _write_jsonl(train_path, train_records)
    _write_jsonl_gz(train_path.with_name("data.json.gz"), train_records)
    _write_jsonl(test_path, test_records)
    _write_jsonl_gz(test_path.with_name("data.json.gz"), test_records)
    notes = {
        "prepared_by": Path(__file__).name,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "src_file": str(src_file.expanduser().resolve()),
        "trim_start": trim_start,
        "trim_end": trim_end,
        "train_ratio": float(train_ratio),
        "returns": returns,
        "freq": freq,
        "prediction_length": int(prediction_length),
        "num_assets": int(len(asset_cols)),
        "num_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "nan_policy": _normalize_nan_policy(nan_policy),
    }
    _write_metadata(dst_dir, freq=freq, prediction_length=prediction_length, num_series=len(asset_cols), notes=notes)

    print(f"[DONE] Wrote GluonTS dataset to {dst_dir}")
    print(f"  train records: {len(train_records)}  test records: {len(test_records)}")
    print(f"  assets: {len(asset_cols)}  returns mode: {returns}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert local iShares price CSVs to GluonTS json format.")
    ap.add_argument("--src-file", type=Path, required=True, help="Path to iShares CSV (must contain 'Date' column).")
    ap.add_argument("--dst-dir", type=Path, required=True, help="Destination root (will create train/ and test/).")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of rows to place in the train split.")
    ap.add_argument("--trim-start", type=str, default=None, help="Optional inclusive start date (YYYY-MM-DD).")
    ap.add_argument("--trim-end", type=str, default=None, help="Optional inclusive end date (YYYY-MM-DD).")
    ap.add_argument(
        "--returns",
        choices=["none", "simple", "log"],
        default="none",
        help="Convert price levels to returns; 'simple' uses (p_t - p_{t-1})/p_{t-1}, "
        "'log' uses log(p_t) - log(p_{t-1}).",
    )
    ap.add_argument(
        "--nan-policy",
        type=str,
        default="none",
        help="How to handle NaN/Inf after trimming: none, raise, drop, ffill, zero. "
        "ffill leaves leading NaNs intact.",
    )
    ap.add_argument("--prediction-length", type=int, default=30, help="Prediction length to record in metadata.json.")
    ap.add_argument("--freq", type=str, default="1B", help="Pandas/GluonTS frequency string (default business day).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset files.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        src_file=args.src_file,
        dst_dir=args.dst_dir.expanduser(),
        train_ratio=args.train_ratio,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        returns=args.returns,
        prediction_length=args.prediction_length,
        freq=args.freq,
        overwrite=args.overwrite,
        nan_policy=args.nan_policy,
    )


if __name__ == "__main__":
    main()
