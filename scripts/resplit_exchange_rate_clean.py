#!/usr/bin/env python3
"""
Re-split a GluonTS-style dataset (exchange_rate_clean) into new train/test
spans with no time overlap. You can choose whether to split from the cleaned
`train` or `test` split as the source. This keeps the format compatible with
downstream datamodules and baselines.

Typical usage when you want to derive both new train and test from the cleaned
train split only:

  python scripts/resplit_exchange_rate_clean.py \
    --dataset ~/.gluonts/datasets/exchange_rate_clean \
    --source-split train \
    --train-ratio 0.8 \
    --trim-start 2009-01-27 \
    --trim-end 2013-11-04
"""

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd


def resplit(
    base_path: Path,
    train_ratio: float,
    trim_start: str | None,
    trim_end: str | None,
    source_split: str,
) -> None:
    """Read the dataset, split each series, and rewrite GluonTS-style files.

    Parameters
    - base_path: dataset root (contains metadata/train/test)
    - train_ratio: fraction of samples to keep in new train per series
    - trim_start/trim_end: optional ISO date bounds applied before splitting
    - source_split: which split to read from ('train' or 'test')
    """
    meta_path = base_path / "metadata.json"
    train_dir = base_path / "train"
    test_dir = base_path / "test"
    train_json_path = train_dir / "train.json"
    train_gz_path = train_dir / "data.json.gz"
    test_json_path = test_dir / "test.json"
    test_gz_path = test_dir / "data.json.gz"

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    freq = meta["freq"]
    pred_len = int(meta["prediction_length"])
    offset = pd.tseries.frequencies.to_offset(freq)

    # Select source file (train.json or test.json) to resplit from
    if source_split.lower() == "train":
        source_path = train_json_path
    elif source_split.lower() == "test":
        source_path = test_json_path
    else:
        raise ValueError("source_split must be 'train' or 'test'")

    with source_path.open("r", encoding="utf-8") as f:
        full_records = [json.loads(line) for line in f]

    new_train = []
    new_test = []
    trim_start_ts = pd.Timestamp(trim_start) if trim_start else None
    trim_end_ts = pd.Timestamp(trim_end) if trim_end else None

    for rec in full_records:
        target = rec["target"]
        start_ts = pd.Timestamp(rec["start"])
        dates = pd.date_range(start_ts, periods=len(target), freq=offset)

        if trim_start_ts or trim_end_ts:
            mask = [True] * len(target)
            if trim_start_ts:
                mask = [flag and (dt >= trim_start_ts) for flag, dt in zip(mask, dates)]
            if trim_end_ts:
                mask = [flag and (dt <= trim_end_ts) for flag, dt in zip(mask, dates)]
            trimmed_target = [val for val, keep in zip(target, mask) if keep]
            trimmed_dates = [dt for dt, keep in zip(dates, mask) if keep]
            if not trimmed_target:
                raise ValueError(
                    f"After trimming, series {rec.get('item_id')} has no samples left."
                )
            target = trimmed_target
            start_ts = trimmed_dates[0]
            dates = pd.date_range(start_ts, periods=len(target), freq=offset)
        else:
            start_ts = pd.Timestamp(rec["start"])

        total_len = len(target)
        max_train = total_len - pred_len
        if max_train <= pred_len:
            raise ValueError(f"Series {rec.get('item_id')} too short for split")

        split_idx = int(total_len * train_ratio)
        split_idx = min(max(split_idx, pred_len), max_train)

        train_rec = dict(rec)
        train_rec["start"] = str(start_ts)
        train_rec["target"] = target[:split_idx]

        test_rec = dict(rec)
        test_rec["start"] = str(pd.Timestamp(start_ts) + split_idx * offset)
        test_rec["target"] = target[split_idx:]

        new_train.append(train_rec)
        new_test.append(test_rec)

    with train_json_path.open("w", encoding="utf-8") as f:
        for rec in new_train:
            f.write(json.dumps(rec) + "\n")
    with gzip.open(train_gz_path, "wt", encoding="utf-8") as g:
        for rec in new_train:
            g.write(json.dumps(rec) + "\n")

    with test_json_path.open("w", encoding="utf-8") as f:
        for rec in new_test:
            f.write(json.dumps(rec) + "\n")
    with gzip.open(test_gz_path, "wt", encoding="utf-8") as g:
        for rec in new_test:
            g.write(json.dumps(rec) + "\n")

    print(
        f"Done. Source={source_split} • Wrote {len(new_train)} series with train_ratio={train_ratio}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / ".gluonts" / "datasets" / "exchange_rate_clean",
        help="Path to the dataset root (contains metadata/train/test)",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        choices=["train", "test"],
        default="train",
        help=(
            "Which split to use as the source for resplitting. "
            "Default: train (derive both new train and test from the cleaned train)."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of each series to keep in train before test",
    )
    parser.add_argument(
        "--trim-start",
        type=str,
        default=None,
        help="Optional ISO date (e.g., 2009-01-27) to drop all earlier samples.",
    )
    parser.add_argument(
        "--trim-end",
        type=str,
        default=None,
        help="Optional ISO date to drop samples after this timestamp.",
    )
    args = parser.parse_args()
    resplit(args.dataset, args.train_ratio, args.trim_start, args.trim_end, args.source_split)


if __name__ == "__main__":
    main()
