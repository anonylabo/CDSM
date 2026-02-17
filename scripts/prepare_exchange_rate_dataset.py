#!/usr/bin/env python3
"""
prepare_exchange_rate_dataset.py

One-shot helper to prepare the cleaned GluonTS dataset "exchange_rate_clean"
and resplit it into new train/test sets with no temporal overlap.

Pipeline:
1) Ensure the original GluonTS "exchange_rate" dataset is downloaded
2) Run make_exchange_rate_clean.py to deduplicate test and normalize layout
3) Run resplit_exchange_rate_clean.py to split from the cleaned train split

Example
  python scripts/prepare_exchange_rate_dataset.py \
    --dst-dir ~/.gluonts/datasets/exchange_rate_clean \
    --train-ratio 0.8 \
    --trim-start 2009-01-27 \
    --trim-end 2013-11-04 \
    --nan-policy none
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _default_dst_dir() -> Path:
    return Path.home() / ".gluonts" / "datasets" / "exchange_rate_clean"


def _default_src_dir() -> Path:
    # GluonTS default download location
    return Path.home() / ".mxnet" / "gluon-ts" / "datasets" / "exchange_rate"


def ensure_source_dataset(src_dir: Optional[Path]) -> None:
    """Download the original exchange_rate dataset if missing."""
    try:
        from gluonts.dataset.repository.datasets import get_dataset  # type: ignore

        # get_dataset will ensure files exist under the default location
        get_dataset("exchange_rate")
    except Exception as exc:  # pragma: no cover - best-effort helper
        print("[WARN] Could not auto-download exchange_rate via GluonTS:", exc)
        if src_dir is None:
            src_dir = _default_src_dir()
        # Fall back to existence check only
        if not src_dir.exists():
            raise SystemExit(
                f"Source dataset not found at {src_dir}. Install gluonts and re-run, "
                f"or supply --src-dir to an existing exchange_rate folder."
            )


def _run_script(script_path: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script_path)] + args
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=script_path.parent.parent)


def _count_series(dst_dir: Path) -> tuple[int, int]:
    train_json = dst_dir / "train" / "train.json"
    test_json = dst_dir / "test" / "test.json"
    def _count(p: Path) -> int:
        if not p.exists():
            return 0
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    return _count(train_json), _count(test_json)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _write_jsonl_gz(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as g:
        for rec in records:
            g.write(json.dumps(rec) + "\n")


def _update_metadata_notes(dst_dir: Path, notes_update: dict) -> None:
    meta_path = dst_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "freq": "1B",
            "prediction_length": 30,
            "feat_static_cat": [],
            "feat_static_real": [],
            "feat_dynamic_real": [],
            "feat_dynamic_cat": [],
        }

    notes = meta.get("notes")
    if notes is None:
        notes = {}
    elif not isinstance(notes, dict):
        notes = {"_previous_notes": notes}

    notes.update(notes_update)
    meta["notes"] = notes
    meta_path.write_text(json.dumps(meta, indent=2))


def _normalize_nan_policy(nan_policy: str) -> str:
    policy = str(nan_policy or "none").strip().lower()
    aliases = {
        "keep": "none",
        "error": "raise",
        "forward_fill": "ffill",
        "fill_zero": "zero",
    }
    return aliases.get(policy, policy)


def _read_freq(dst_dir: Path) -> str:
    meta_path = dst_dir / "metadata.json"
    if not meta_path.exists():
        return "1B"
    meta = json.loads(meta_path.read_text())
    return str(meta.get("freq", "1B"))


def _apply_nan_policy_records(records: list[dict], nan_policy: str, freq: str) -> list[dict]:
    policy = _normalize_nan_policy(nan_policy)
    if policy == "none":
        return records

    targets = []
    for rec in records:
        arr = np.asarray(rec.get("target", []), dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        targets.append(arr)

    if policy == "raise":
        for rec, arr in zip(records, targets):
            if not np.isfinite(arr).all():
                bad = int((~np.isfinite(arr)).sum())
                raise ValueError(
                    f"Series {rec.get('item_id', '<unknown>')} contains {bad} non-finite values "
                    "with nan_policy='raise'."
                )
        return records

    if policy == "ffill":
        filled = []
        for rec, arr in zip(records, targets):
            series = pd.Series(arr).ffill().to_numpy()
            new_rec = dict(rec)
            new_rec["target"] = series.tolist()
            filled.append(new_rec)
        return filled

    if policy == "zero":
        filled = []
        for rec, arr in zip(records, targets):
            series = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            new_rec = dict(rec)
            new_rec["target"] = series.tolist()
            filled.append(new_rec)
        return filled

    if policy == "drop":
        starts = {str(rec.get("start", "")) for rec in records}
        lengths = {len(arr) for arr in targets}
        if len(starts) != 1 or len(lengths) != 1:
            raise ValueError("nan_policy='drop' requires aligned start dates and equal lengths across series.")
        data = np.stack(targets, axis=1)
        mask = np.isfinite(data).all(axis=1)
        if not mask.any():
            raise ValueError("All rows contain non-finite values under nan_policy='drop'.")
        first_idx = int(np.argmax(mask))
        data = data[mask]
        offset = pd.tseries.frequencies.to_offset(freq)
        new_start = pd.Timestamp(list(starts)[0]) + first_idx * offset
        dropped = []
        for i, rec in enumerate(records):
            new_rec = dict(rec)
            new_rec["start"] = str(new_start)
            new_rec["target"] = data[:, i].tolist()
            dropped.append(new_rec)
        return dropped

    raise ValueError(f"Unsupported nan_policy='{nan_policy}'")


def _apply_nan_policy_to_split(dst_dir: Path, split: str, nan_policy: str, freq: str) -> None:
    json_name = "train.json" if split == "train" else "test.json"
    json_path = dst_dir / split / json_name
    if not json_path.exists():
        return
    records = _read_jsonl(json_path)
    if not records:
        return
    updated = _apply_nan_policy_records(records, nan_policy, freq)
    _write_jsonl(json_path, updated)
    _write_jsonl_gz(json_path.with_name("data.json.gz"), updated)


def apply_nan_policy(dst_dir: Path, nan_policy: str) -> None:
    policy = _normalize_nan_policy(nan_policy)
    if policy == "none":
        return
    freq = _read_freq(dst_dir)
    for split in ("train", "test"):
        _apply_nan_policy_to_split(dst_dir, split, policy, freq)


def _compute_returns(series: list[float], mode: str) -> list[float]:
    if len(series) < 2:
        raise ValueError("Series too short to compute returns.")
    returns: list[float] = []
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


def convert_to_returns(dst_dir: Path, mode: str) -> None:
    meta_path = dst_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"metadata.json not found under {dst_dir}")
    meta = json.loads(meta_path.read_text())
    freq = meta.get("freq", "1B")
    offset = pd.tseries.frequencies.to_offset(freq)

    for split in ("train", "test"):
        json_name = "train.json" if split == "train" else "test.json"
        json_path = dst_dir / split / json_name
        if not json_path.exists():
            continue
        records = _read_jsonl(json_path)
        transformed: list[dict] = []
        for rec in records:
            target = rec.get("target", [])
            if len(target) < 2:
                continue
            returns = _compute_returns(target, mode)
            start = pd.Timestamp(rec["start"])
            new_rec = dict(rec)
            new_rec["target"] = returns
            new_rec["start"] = str(start + offset)
            transformed.append(new_rec)
        if not transformed:
            raise SystemExit(f"No records remained after return conversion for split '{split}'.")
        _write_jsonl(json_path, transformed)
        gz_path = json_path.with_name("data.json.gz")
        _write_jsonl_gz(gz_path, transformed)
        print(f"[INFO] Converted {json_path} to {mode} returns (records: {len(transformed)})")


def apply_log_transform(dst_dir: Path) -> None:
    for split in ("train", "test"):
        json_name = "train.json" if split == "train" else "test.json"
        json_path = dst_dir / split / json_name
        if not json_path.exists():
            continue
        records = _read_jsonl(json_path)
        transformed: list[dict] = []
        for rec in records:
            target = rec.get("target", [])
            if not target:
                continue
            if any(float(val) <= 0 for val in target):
                raise SystemExit(
                    "Encountered non-positive price while applying log transform. "
                    "Ensure the source data are strictly positive or disable --log-prices."
                )
            new_rec = dict(rec)
            new_rec["target"] = [math.log(float(val)) for val in target]
            transformed.append(new_rec)
        if not transformed:
            continue
        _write_jsonl(json_path, transformed)
        gz_path = json_path.with_name("data.json.gz")
        _write_jsonl_gz(gz_path, transformed)
        print(f"[INFO] Applied log transform to {json_path} (records: {len(transformed)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dst-dir", type=Path, default=_default_dst_dir(), help="Destination cleaned dataset root")
    ap.add_argument("--src-dir", type=Path, default=None, help="Optional source exchange_rate directory (defaults to GluonTS cache)")
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of each series kept in new train")
    ap.add_argument("--trim-start", type=str, default="2009-01-27", help="Optional ISO start bound (inclusive)")
    ap.add_argument("--trim-end", type=str, default="2013-11-04", help="Optional ISO end bound (inclusive)")
    ap.add_argument("--source-split", choices=["train", "test"], default="train", help="Split from cleaned train or test")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination when preparing cleaned dataset")
    ap.add_argument("--no-download", action="store_true", help="Skip attempting to download the source dataset")
    ap.add_argument(
        "--returns",
        choices=["none", "simple", "log"],
        default="none",
        help="Optional conversion from price levels to returns. "
        "'simple' uses (p_t - p_{t-1}) / p_{t-1}; 'log' uses log(p_t) - log(p_{t-1}).",
    )
    ap.add_argument(
        "--log-prices",
        action="store_true",
        help="After resplitting, replace each price observation with its natural logarithm "
        "(requires strictly positive prices). Applied before --returns conversion."
    )
    ap.add_argument(
        "--nan-policy",
        type=str,
        default="none",
        help="How to handle NaN/Inf in the final train/test splits: none, raise, drop, ffill, zero. "
        "ffill leaves leading NaNs intact.",
    )
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    make_script = scripts_dir / "make_exchange_rate_clean.py"
    resplit_script = scripts_dir / "resplit_exchange_rate_clean.py"
    if not make_script.exists() or not resplit_script.exists():
        raise SystemExit("Required scripts not found next to this helper.")

    # 1) Ensure source exists
    if not args.no_download:
        ensure_source_dataset(args.src_dir)

    # 2) Build cleaned dataset
    make_args: list[str] = ["--dst-dir", str(args.dst_dir)]
    if args.src_dir is not None:
        make_args += ["--src-dir", str(args.src_dir)]
    if args.overwrite:
        make_args.append("--overwrite")
    _run_script(make_script, make_args)

    # 3) Resplit from cleaned train/test into new train/test
    resplit_args = [
        "--dataset", str(args.dst_dir),
        "--source-split", str(args.source_split),
        "--train-ratio", str(args.train_ratio),
    ]
    if args.trim_start:
        resplit_args += ["--trim-start", str(args.trim_start)]
    if args.trim_end:
        resplit_args += ["--trim-end", str(args.trim_end)]
    _run_script(resplit_script, resplit_args)

    if args.log_prices:
        apply_log_transform(args.dst_dir)

    if args.returns != "none":
        convert_to_returns(args.dst_dir, args.returns)

    apply_nan_policy(args.dst_dir, args.nan_policy)

    _update_metadata_notes(
        args.dst_dir,
        {
            "prepared_by": Path(__file__).name,
            "prepared_at": datetime.now().isoformat(timespec="seconds"),
            "src_dir": str(args.src_dir.expanduser().resolve()) if args.src_dir is not None else None,
            "source_split": args.source_split,
            "train_ratio": float(args.train_ratio),
            "trim_start": args.trim_start,
            "trim_end": args.trim_end,
            "log_prices": bool(args.log_prices),
            "returns": args.returns,
            "no_download": bool(args.no_download),
            "nan_policy": _normalize_nan_policy(args.nan_policy),
        },
    )

    # 4) Summary
    tr, te = _count_series(args.dst_dir)
    print("[DONE] Prepared dataset at:", args.dst_dir)
    print(f"  train series: {tr}  test series: {te}")
    print("  You can set: export CFDIFF_DATA_DIR=\"%s\"" % args.dst_dir)


if __name__ == "__main__":
    main()
