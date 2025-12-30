#!/usr/bin/env python3
"""
Prepare a GluonTS-style FX-30 dataset (per-USD, daily) from the ECB EXR SDMX API.

Usage (similar界面):
python scripts/prepare_fx30_ecb_dataset.py \
  --dst-dir ~/.gluonts/datasets/fx30_ecb \
  --trim-start 2015-01-01 \
  --trim-end 2024-12-31 \
  --train-ratio 0.8 \
  --returns log \
  --prediction-length 30 \
  --exclude-usd \
  --nan-policy none \
  --overwrite
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import requests

ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data/EXR"

# FX-30 currency list (against USD after conversion)
FX30 = [
    "USD",
    "JPY",
    "BGN",
    "CZK",
    "DKK",
    "GBP",
    "HUF",
    "PLN",
    "RON",
    "SEK",
    "CHF",
    "ISK",
    "NOK",
    "TRY",
    "AUD",
    "BRL",
    "CAD",
    "CNY",
    "HKD",
    "IDR",
    "ILS",
    "INR",
    "KRW",
    "MXN",
    "MYR",
    "NZD",
    "PHP",
    "SGD",
    "THB",
    "ZAR",
]


@dataclass
class Outputs:
    csv_path: Path
    train_path: Path
    test_path: Path


def _http_get_csv(url: str, params: dict) -> pd.DataFrame:
    headers = {"Accept": "text/csv"}
    r = requests.get(url, params=params, headers=headers, timeout=60)
    r.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(r.text))
    if df.empty:
        raise RuntimeError("ECB API returned an empty dataset. Check date range and connectivity.")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.upper(): c for c in df.columns}

    def pick(*candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    col_date = pick("TIME_PERIOD", "TIME", "DATE")
    col_val = pick("OBS_VALUE", "OBS", "VALUE")
    col_cur = pick("CURRENCY", "CURRENCY_CODE", "CURRENCY_ID")
    if not col_date or not col_val:
        raise RuntimeError(f"Unexpected ECB CSV schema. Columns={list(df.columns)}")
    if not col_cur:
        df["_CURRENCY_INFERRED_"] = np.nan
        col_cur = "_CURRENCY_INFERRED_"

    out = df[[col_date, col_val, col_cur]].copy()
    out.columns = ["date", "value", "currency"]
    return out


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


def fetch_ecb_fx_vs_eur(currencies: List[str], start: str, end: str) -> pd.DataFrame:
    cur_key = "+".join(currencies)
    series_key = f"D.{cur_key}.EUR.SP00.A"  # daily, spot, average, vs EUR
    url = f"{ECB_API_BASE}/{series_key}"
    params = {"startPeriod": start, "endPeriod": end}
    raw = _http_get_csv(url, params=params)
    tidy = _normalize_columns(raw)

    if tidy["currency"].isna().all():
        raise RuntimeError("ECB CSV missing currency identifier; expected multi-currency response.")

    tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
    tidy = tidy.dropna(subset=["date", "value", "currency"])
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])

    wide = tidy.pivot_table(index="date", columns="currency", values="value", aggfunc="last").sort_index()
    wide = wide.reindex(columns=currencies)
    return wide


def convert_vs_usd(wide_vs_eur: pd.DataFrame) -> pd.DataFrame:
    if "USD" not in wide_vs_eur.columns:
        raise RuntimeError("USD series missing; required for conversion to USD base.")
    usd_per_eur = wide_vs_eur["USD"].copy()
    out = pd.DataFrame(index=wide_vs_eur.index)
    for c in wide_vs_eur.columns:
        if c == "USD":
            out[c] = 1.0
        else:
            out[c] = wide_vs_eur[c] / usd_per_eur
    return out


def _compute_returns(series: List[float], mode: str) -> List[float]:
    if len(series) < 2:
        raise ValueError("Series too short to compute returns.")
    out: List[float] = []
    prev = float(series[0])
    for val_raw in series[1:]:
        val = float(val_raw)
        if mode == "simple":
            if prev == 0.0:
                raise ValueError("Encountered zero price in simple returns.")
            ret = (val - prev) / prev
        else:  # log
            if prev <= 0.0 or val <= 0.0:
                raise ValueError("Log returns require positive values.")
            ret = float(np.log(val) - np.log(prev))
        out.append(ret)
        prev = val
    return out


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_metadata(
    dst_dir: Path,
    freq: str,
    prediction_length: int,
    num_assets: int,
    notes: dict | None = None,
) -> None:
    meta = {
        "freq": freq,
        "target": None,
        "feat_static_cat": [{"name": "feat_static_cat_0", "cardinality": str(num_assets)}],
        "feat_static_real": [],
        "feat_dynamic_real": [],
        "feat_dynamic_cat": [],
        "prediction_length": prediction_length,
    }
    if notes:
        meta["notes"] = notes
    (dst_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def prepare_dataset(
    dst_dir: Path,
    start: str,
    end: str,
    train_ratio: float,
    returns: str,
    prediction_length: int,
    exclude_usd: bool,
    overwrite: bool,
    nan_policy: str,
) -> Outputs:
    dst_dir = dst_dir.expanduser()
    if dst_dir.exists():
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(f"{dst_dir} already exists; use --overwrite to replace.")
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Fetching ECB EXR daily rates vs EUR ({start} to {end})...")
    wide_vs_eur = fetch_ecb_fx_vs_eur(FX30, start=start, end=end)
    print(f"[2/4] Converting to currency-per-USD...")
    df_usd = convert_vs_usd(wide_vs_eur)

    # Trim to requested window (after conversion/fill)
    df_usd = df_usd.loc[pd.to_datetime(start) : pd.to_datetime(end)]
    if df_usd.empty:
        raise SystemExit("No data in requested date range after trimming.")

    out_currencies = [c for c in FX30 if c != "USD"] if exclude_usd else list(FX30)
    missing_cols = [c for c in out_currencies if c not in df_usd.columns]
    if missing_cols:
        raise SystemExit(f"Missing expected currencies in ECB response: {missing_cols}")
    df_usd = df_usd.reindex(columns=out_currencies)
    df_usd = _apply_nan_policy(df_usd, nan_policy)

    # Split along time axis
    n = len(df_usd)
    split_idx = int(n * train_ratio)
    if split_idx <= 0 or split_idx >= n:
        raise SystemExit(f"Invalid train_ratio={train_ratio}; results in empty train or test.")

    # Optionally convert to returns
    data = df_usd.copy()
    if returns != "none":
        ret_df = pd.DataFrame(index=data.index[1:], columns=data.columns, dtype=float)
        for col in data.columns:
            ret_df[col] = _compute_returns(data[col].tolist(), mode=returns)
        data = ret_df

    # Write wide CSVs for reference:
    # - levels: currency-per-USD levels after conversion/fill (before returns)
    # - processed: matches the GluonTS split values (returns if requested, else levels)
    levels_csv = dst_dir / f"fx{len(out_currencies)}_usd_levels_{start}_{end}.csv"
    df_usd.to_csv(levels_csv, index_label="date")
    processed_csv = dst_dir / f"fx{len(out_currencies)}_usd_{returns}_{start}_{end}.csv"
    data.to_csv(processed_csv, index_label="date")

    # Ensure split after returns trimming
    n_effective = len(data)
    split_idx = int(n_effective * train_ratio)
    if split_idx <= 0 or split_idx >= n_effective:
        raise SystemExit(f"Train/test split empty after returns conversion (train_ratio={train_ratio}).")

    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    # Build per-currency records (univariate per line, like exchange_rate dataset)
    train_records: List[dict] = []
    test_records: List[dict] = []
    for idx, cur in enumerate(out_currencies):
        train_records.append(
            {
                "start": train_df.index[0].strftime("%Y-%m-%d"),
                "target": train_df[cur].astype(float).tolist(),
                "item_id": cur,
                "feat_static_cat": [idx],
            }
        )
        test_records.append(
            {
                "start": test_df.index[0].strftime("%Y-%m-%d"),
                "target": test_df[cur].astype(float).tolist(),
                "item_id": cur,
                "feat_static_cat": [idx],
            }
        )

    print(f"[3/4] Writing GluonTS json...")
    train_path = dst_dir / "train" / "train.json"
    test_path = dst_dir / "test" / "test.json"
    _write_jsonl(train_path, train_records)
    _write_jsonl(train_path.with_name("data.json"), train_records)
    _write_jsonl(train_path.with_name("data.json.gz"), train_records)
    _write_jsonl(test_path, test_records)
    _write_jsonl(test_path.with_name("data.json"), test_records)
    _write_jsonl(test_path.with_name("data.json.gz"), test_records)
    notes = {
        "prepared_by": Path(__file__).name,
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "trim_start": start,
        "trim_end": end,
        "train_ratio": float(train_ratio),
        "returns": returns,
        "exclude_usd": bool(exclude_usd),
        "prediction_length": int(prediction_length),
        "num_assets": int(len(out_currencies)),
        "api_base": ECB_API_BASE,
        "levels_csv": str(levels_csv.name),
        "processed_csv": str(processed_csv.name),
        "nan_policy": _normalize_nan_policy(nan_policy),
    }
    _write_metadata(dst_dir, freq="1B", prediction_length=prediction_length, num_assets=len(out_currencies), notes=notes)

    print(f"[4/4] Writing wide CSV for reference...")
    csv_path = processed_csv

    # README
    (dst_dir / "README.txt").write_text(
        "FX daily dataset built from ECB EXR (spot, average) converted to currency-per-USD.\n"
        "Sources: series D.<CUR>.EUR.SP00.A via https://data-api.ecb.europa.eu/service/data/EXR.\n"
        "Conversion: CUR_per_USD = CUR_per_EUR / USD_per_EUR.\n"
        "Note: the USD column would be constant (=1) after conversion, so it may be excluded.\n"
        "Train/test split is chronological with the given train_ratio.\n",
        encoding="utf-8",
    )

    return Outputs(csv_path=csv_path, train_path=train_path, test_path=test_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare FX-30 dataset from ECB EXR (per USD).")
    ap.add_argument("--dst-dir", type=Path, required=True, help="Destination GluonTS dataset root.")
    ap.add_argument("--trim-start", type=str, required=True, help="Start date (YYYY-MM-DD).")
    ap.add_argument("--trim-end", type=str, required=True, help="End date (YYYY-MM-DD).")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of timesteps for train split.")
    ap.add_argument(
        "--returns",
        choices=["none", "simple", "log"],
        default="log",
        help="Convert levels to returns along time axis.",
    )
    ap.add_argument("--prediction-length", type=int, default=30, help="prediction_length to store in metadata.json.")
    ap.add_argument(
        "--exclude-usd",
        action="store_true",
        help="Drop USD from the output series (USD would be constant (=1) after per-USD conversion).",
    )
    ap.add_argument(
        "--nan-policy",
        type=str,
        default="none",
        help="How to handle NaN/Inf after conversion: none, raise, drop, ffill, zero. "
        "ffill leaves leading NaNs intact.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files under dst-dir.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        dst_dir=args.dst_dir,
        start=args.trim_start,
        end=args.trim_end,
        train_ratio=args.train_ratio,
        returns=args.returns,
        prediction_length=args.prediction_length,
        exclude_usd=bool(args.exclude_usd),
        overwrite=args.overwrite,
        nan_policy=args.nan_policy,
    )


if __name__ == "__main__":
    main()
