#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_exchange_rate_clean.py

Prepare a cleaned GluonTS-style dataset from the official `exchange_rate` repo
dataset by de-duplicating the test split (official test contains 5 rolling
copies per item). The result is written under ~/.gluonts/datasets by default
and is directly consumable by the project datamodules.

Default I/O
-----------
- Source (read-only):   ~/.mxnet/gluon-ts/datasets/exchange_rate
- Destination (write):  ~/.gluonts/datasets/exchange_rate_clean

Outputs
-------
- metadata.json
- train/train.json  (+ train/data.json.gz copy)
- test/test.json    (+ test/data.json.gz re-written from deduplicated records)

Usage
-----
python make_exchange_rate_clean.py \
  --src-dir ~/.mxnet/gluon-ts/datasets/exchange_rate \
  --dst-dir ~/.gluonts/datasets/exchange_rate_clean \
  [--overwrite]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    # Prefer GluonTS to resolve the default source path
    from gluonts.dataset.repository.datasets import get_download_path as _gt_get_download_path  # type: ignore
except Exception:  # pragma: no cover - optional import
    _gt_get_download_path = None  # type: ignore


def _default_src_dir() -> Path:
    if _gt_get_download_path is not None:
        return Path(_gt_get_download_path()) / "datasets" / "exchange_rate"
    # Fallback to standard MXNet path used by GluonTS
    return Path.home() / ".mxnet" / "gluon-ts" / "datasets" / "exchange_rate"


def _default_dst_dir() -> Path:
    return Path.home() / ".gluonts" / "datasets" / "exchange_rate_clean"


def _iter_jsonl_gz(path: Path) -> Iterable[Dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_jsonl_gz(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as g:
        for r in records:
            g.write(json.dumps(r) + "\n")


def _dedupe_test(records: Iterable[Dict]) -> List[Dict]:
    by_id: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_id[str(r.get("item_id"))].append(r)
    kept: List[Dict] = []
    for iid, recs in sorted(by_id.items(), key=lambda kv: kv[0]):
        # Keep the longest target; tie-breaker: latest start string
        recs.sort(key=lambda o: (len(o.get("target", []) or []), str(o.get("start", ""))), reverse=True)
        kept.append(recs[0])
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare cleaned exchange_rate dataset (deduplicated test split)")
    ap.add_argument("--src-dir", type=str, default=str(_default_src_dir()), help="Source exchange_rate dataset directory")
    ap.add_argument("--dst-dir", type=str, default=str(_default_dst_dir()), help="Destination dataset directory")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists")
    args = ap.parse_args()

    src = Path(args.src_dir).expanduser().resolve()
    dst = Path(args.dst_dir).expanduser().resolve()

    train_gz = src / "train" / "data.json.gz"
    test_gz = src / "test" / "data.json.gz"
    meta_src = src / "metadata.json"
    if not train_gz.exists() or not test_gz.exists():
        raise SystemExit(f"Missing source files under {src}: expected train/test data.json.gz")

    if dst.exists() and not args.overwrite:
        print(f"[INFO] Destination exists: {dst}. Use --overwrite to replace.")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "train").mkdir(parents=True, exist_ok=True)
    (dst / "test").mkdir(parents=True, exist_ok=True)

    # metadata.json: copy if present; otherwise synthesize minimal
    if meta_src.exists():
        meta = json.load(meta_src.open("r", encoding="utf-8"))
    else:
        meta = {
            "freq": "1B",
            "prediction_length": 30,
            "feat_static_cat": [{"name": "feat_static_cat_0", "cardinality": "8"}],
            "feat_static_real": [],
            "feat_dynamic_real": [],
            "feat_dynamic_cat": [],
        }
    json.dump(meta, (dst / "metadata.json").open("w", encoding="utf-8"), indent=2)

    # Train: copy gz + write jsonl for convenience
    train_records = list(_iter_jsonl_gz(train_gz))
    _write_jsonl(dst / "train" / "train.json", train_records)
    shutil.copy2(train_gz, dst / "train" / "data.json.gz")

    # Test: dedupe and write both jsonl and gz
    test_records_src = list(_iter_jsonl_gz(test_gz))
    test_records = _dedupe_test(test_records_src)
    _write_jsonl(dst / "test" / "test.json", test_records)
    _write_jsonl_gz(dst / "test" / "data.json.gz", test_records)

    print("[DONE] exchange_rate_clean prepared at:", dst)
    print("  train series:", len(train_records), " test series:", len(test_records))


if __name__ == "__main__":
    main()

