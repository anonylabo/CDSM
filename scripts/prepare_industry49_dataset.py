#!/usr/bin/env python3
"""
Prepare the Ken French 49 Industry portfolios (daily) as a GluonTS-style dataset.

Default behaviour:
- Download the zipped CSV from Ken French's site.
- Parse dates, drop empty rows, convert percent returns to decimal.
- Optionally trim by date range.
- Align to a business-day calendar (NYSE-style) by reindexing; handle NaNs per --nan-policy (default: none).
- Split into train/test by train_ratio and write jsonl files under dst_dir/train and dst_dir/test.

Example:
python scripts/prepare_industry49_dataset.py \
  --dst-dir ~/.gluonts/datasets/industry49_clean \
  --train-ratio 0.8 \
  --trim-start 1995-01-01 \
  --trim-end 2000-01-01 \
  --nan-policy none \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests


DEFAULT_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_Daily_CSV.zip"


def _download_csv(url: str) -> pd.DataFrame:
    """
    Ken French CSVs have a preamble; find the header line starting with 'Date' and parse from there.
    """
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("No CSV found inside zip.")
        with zf.open(names[0]) as fh:
            raw_lines = fh.read().decode("latin1").splitlines()
    header_idx = None
    for i, line in enumerate(raw_lines):
        if line.startswith(","):  # header line starts with comma, missing Date label
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not locate header line.")
    header_fields = ["Date"] + [f.strip() for f in raw_lines[header_idx][1:].split(",")]
    data_lines = raw_lines[header_idx + 1 :]
    # Stop at Equal-Weighted section if present
    stop_idx = None
    for i, line in enumerate(data_lines):
        if line.strip().startswith("Average Equal Weighted Returns"):
            stop_idx = i
            break
    if stop_idx is not None:
        data_lines = data_lines[:stop_idx]
    csv_content = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_content), names=header_fields, dtype=str, low_memory=False)
    return df


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


def _clean_dataframe(
    df: pd.DataFrame,
    trim_start: Optional[str],
    trim_end: Optional[str],
    nan_policy: str,
) -> pd.DataFrame:
    df = df[df["Date"].notna() & (df["Date"] != "")]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index()
    # Trim date range if provided
    if trim_start:
        df = df[df.index >= pd.to_datetime(trim_start)]
    if trim_end:
        df = df[df.index <= pd.to_datetime(trim_end)]
    # convert percent strings to numeric; division handled later
    df = df.apply(pd.to_numeric, errors="coerce")
    # align to business-day calendar
    full_idx = pd.bdate_range(start=df.index.min(), end=df.index.max(), freq="B")
    df = df.reindex(full_idx)
    df = _apply_nan_policy(df, nan_policy)
    return df


def _write_jsonl(df: pd.DataFrame, out_path: Path) -> None:
    records = []
    start_date = df.index[0].strftime("%Y-%m-%d")
    for col in df.columns:
        target = df[col].astype(float).tolist()
        records.append({"start": start_date, "target": target, "item_id": str(col)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_metadata(dst_dir: Path, freq: str, prediction_length: int, num_series: int, notes: dict | None = None) -> None:
    meta = {
        "freq": freq,
        "target": None,
        "feat_static_cat": [{"name": "feat_static_cat_0", "cardinality": str(num_series)}],
        "feat_static_real": [],
        "feat_dynamic_real": [],
        "feat_dynamic_cat": [],
        "prediction_length": prediction_length,
    }
    if notes:
        meta["notes"] = notes
    meta_path = dst_dir.expanduser() / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


def prepare_dataset(
    dst_dir: Path,
    url: str,
    train_ratio: float,
    trim_start: Optional[str],
    trim_end: Optional[str],
    overwrite: bool,
    log_returns: bool,
    prediction_length: int,
    nan_policy: str,
) -> None:
    df_raw = _download_csv(url)
    df = _clean_dataframe(df_raw, trim_start, trim_end, nan_policy)
    if len(df) == 0:
        raise RuntimeError("No data after trimming/cleaning.")
    # convert percent to decimal, optionally to log-returns
    df = df / 100.0
    if log_returns:
        if (df <= -1).any().any():
            raise ValueError("Found returns <= -100% which are invalid for log1p.")
        df = df.apply(lambda s: pd.Series(np.log1p(s), index=s.index))
    split_idx = int(len(df) * train_ratio)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError(f"Invalid train_ratio={train_ratio}; results in empty train or test split.")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_path = dst_dir.expanduser() / "train" / "data.json"
    test_path = dst_dir.expanduser() / "test" / "data.json"
    if not overwrite and (train_path.exists() or test_path.exists()):
        raise FileExistsError(f"{train_path} or {test_path} already exists; use --overwrite to replace.")

    _write_jsonl(train_df, train_path)
    _write_jsonl(test_df, test_path)
    notes = {
        "prepared_by": Path(__file__).name,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "url": url,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "train_ratio": float(train_ratio),
        "log_returns": bool(log_returns),
        "freq": "1B",
        "prediction_length": int(prediction_length),
        "num_assets": int(len(df.columns)),
        "num_rows": int(len(df)),
        "actual_start": df.index[0].strftime("%Y-%m-%d"),
        "actual_end": df.index[-1].strftime("%Y-%m-%d"),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "nan_policy": _normalize_nan_policy(nan_policy),
    }
    _write_metadata(dst_dir, freq="1B", prediction_length=prediction_length, num_series=len(df.columns), notes=notes)
    print(f"Wrote train to {train_path} (rows={len(train_df)}), test to {test_path} (rows={len(test_df)})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare Ken French 49 Industry daily returns as a GluonTS dataset.")
    ap.add_argument("--url", type=str, default=DEFAULT_URL, help="Source zip URL.")
    ap.add_argument("--dst-dir", type=Path, required=True, help="Destination directory (root containing train/ and test/).")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of rows to use for train split.")
    ap.add_argument("--trim-start", type=str, default=None, help="Optional start date (YYYY-MM-DD) to keep.")
    ap.add_argument("--trim-end", type=str, default=None, help="Optional end date (YYYY-MM-DD) to keep.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing train/test files if present.")
    ap.add_argument("--log-returns", action="store_true", help="Convert decimal returns to log1p returns (requires all r>-1).")
    ap.add_argument(
        "--nan-policy",
        type=str,
        default="none",
        help="How to handle NaN/Inf after alignment: none, raise, drop, ffill, zero. "
        "ffill leaves leading NaNs intact.",
    )
    ap.add_argument("--prediction-length", type=int, default=30, help="prediction_length to store in metadata.json.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        dst_dir=args.dst_dir,
        url=args.url,
        train_ratio=args.train_ratio,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        overwrite=args.overwrite,
        log_returns=args.log_returns,
        prediction_length=args.prediction_length,
        nan_policy=args.nan_policy,
    )


if __name__ == "__main__":
    main()
