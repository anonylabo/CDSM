#!/usr/bin/env python3
"""
Summarise dataset/window configurations into Markdown and LaTeX tables.

Each dataset is provided via --entry using pipe separators:
LABEL|DATASET|[SPLIT]|CONTEXT|PRED|VAL_RATIO|[DESC]|[PERIOD]|[ASSETS]|[NOTES]
- DATASET can be a name (resolved under ~/.gluonts/datasets) or a path.
- SPLIT defaults to "train" when omitted.
- ASSETS is optional; if absent, the script counts the series in the requested split.

Example mirroring the eigen-window plots:
python scripts/make_dataset_summary.py \
  --entry "FX|~/.gluonts/datasets/exchange_rate_clean|train|30|15|0.3|GluonTS Exchange Rate|1995-1999|8" \
  --entry "ind49 (30,15)|~/.gluonts/datasets/industry49_clean|train|30|15|0.3|Ken French Industry 49|2018-2023|12" \
  --entry "ind49 (60,30)|~/.gluonts/datasets/industry49_clean|train|60|30|0.3|Ken French Industry 49|2018-2023|14" \
  --entry "stock14|~/.gluonts/datasets/ishares14_clean|train|30|15|0.3|iShares ETF panel|recent" \
  --entry "stock6 (val0.3)|~/.gluonts/datasets/ishares6_clean_train0.75|train|100|20|0.3|iShares ETF small|recent|6|val split=0.3" \
  --entry "stock6 (val-0.1)|~/.gluonts/datasets/ishares6_clean_train0.8|train|100|20|-0.1|iShares ETF small|recent|6|val split from tail" \
  --output-markdown assets/dataset_summary.md \
  --output-latex assets/dataset_summary.tex
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class DatasetSpec:
    label: str
    data_dir: Path
    split: str
    context_len: int
    pred_len: int
    val_ratio: float
    description: str
    period: str
    test_period: str
    assets: int
    notes: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Pipe-separated dataset spec: LABEL|DATASET|[SPLIT]|CONTEXT|PRED|VAL_RATIO|[DESC]|[PERIOD]|[ASSETS]|[NOTES]",
    )
    ap.add_argument("--output-markdown", type=Path, default=Path("assets/dataset_summary.md"), help="Markdown table output.")
    ap.add_argument("--output-latex", type=Path, default=Path("assets/dataset_summary.tex"), help="LaTeX table output.")
    ap.add_argument(
        "--infer-period",
        action="store_true",
        help="Best-effort start/end inference when PERIOD is omitted (uses dataset split contents and metadata freq).",
    )
    ap.add_argument(
        "--infer-period-override",
        action="store_true",
        help="Force period to be inferred from the dataset even if a PERIOD string is provided in --entry.",
    )
    ap.add_argument(
        "--period-placeholder",
        default="n/a",
        help="Placeholder text when PERIOD is missing and cannot be inferred.",
    )
    ap.add_argument(
        "--omit-total",
        dest="include_total",
        action="store_false",
        help="Skip the Total (L=C+P) column to save space.",
    )
    ap.add_argument(
        "--latex-tight",
        action="store_true",
        help="Use @{\\,} column padding in LaTeX for a more compact table.",
    )
    ap.set_defaults(include_total=True)
    return ap.parse_args()


def _resolve_dataset_root(dataset: str) -> Path:
    cand = Path(dataset).expanduser()
    if cand.exists():
        return cand
    fallback = Path.home() / ".gluonts" / "datasets" / dataset
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset path does not exist: {dataset}")


def _resolve_split_file(root: Path, split: str) -> Path:
    candidates = [
        root / f"{split}.json",
        root / split / f"{split}.json",
        root / split / "data.json",
        root / split / "data.json.gz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not locate split '{split}' under {root}")


def _iter_json_lines(path: Path):
    if path.suffix == ".gz":
        # Be defensive: some datasets may ship a `.gz` filename that is not
        # actually gzip-compressed.
        with path.open("rb") as fb:
            magic = fb.read(2)
        opener = gzip.open if magic == b"\x1f\x8b" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    else:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def _count_assets(data_dir: Path, split: str) -> int:
    split_file = _resolve_split_file(data_dir, split)
    count = sum(1 for _ in _iter_json_lines(split_file))
    if count == 0:
        raise ValueError(f"No series found in {split_file}")
    return count


def _metadata_freq(data_dir: Path) -> Optional[str]:
    meta = data_dir / "metadata.json"
    if not meta.exists():
        return None
    try:
        with meta.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return str(obj.get("freq")) if obj.get("freq") else None
    except Exception:
        return None


def _freq_to_timedelta(freq: str) -> Optional[timedelta]:
    freq = freq.lower()
    if freq.endswith("b") or freq.endswith("d"):
        return timedelta(days=1)
    if freq.endswith("w"):
        return timedelta(days=7)
    if freq.endswith("h"):
        return timedelta(hours=1)
    if freq.endswith("t") or "min" in freq:
        return timedelta(minutes=1)
    return None


def _infer_period(data_dir: Path, placeholder: str) -> str:
    def _min_max_for_split(split: str) -> Optional[Tuple[datetime, datetime]]:
        try:
            split_file = _resolve_split_file(data_dir, split)
        except FileNotFoundError:
            return None
        freq_hint = _metadata_freq(data_dir)
        offset = None
        if freq_hint:
            try:
                offset = pd.tseries.frequencies.to_offset(freq_hint)
            except Exception:
                offset = _freq_to_timedelta(freq_hint)
        min_start: Optional[datetime] = None
        max_end: Optional[datetime] = None
        for line in _iter_json_lines(split_file):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            start_raw = obj.get("start")
            target = obj.get("target")
            if not start_raw or not target:
                continue
            try:
                start_dt = datetime.fromisoformat(str(start_raw))
            except Exception:
                continue
            if offset is not None:
                try:
                    end_dt = pd.Timestamp(start_dt) + offset * (max(0, len(target) - 1))
                    end_dt = end_dt.to_pydatetime()
                except Exception:
                    end_dt = start_dt
            else:
                end_dt = start_dt
            min_start = start_dt if min_start is None or start_dt < min_start else min_start
            max_end = end_dt if max_end is None or end_dt > max_end else max_end
        if min_start is None or max_end is None:
            return None
        return min_start, max_end

    agg: List[Tuple[datetime, datetime]] = []
    for split in ["train", "test", "val"]:
        res = _min_max_for_split(split)
        if res:
            agg.append(res)
    if not agg:
        return placeholder
    min_start = min(pair[0] for pair in agg)
    max_end = max(pair[1] for pair in agg)
    return f"{min_start.date()} to {max_end.date()}"


def _infer_period_for_split(data_dir: Path, split: str, placeholder: str) -> str:
    try:
        split_file = _resolve_split_file(data_dir, split)
    except FileNotFoundError:
        return placeholder
    freq_hint = _metadata_freq(data_dir)
    offset = None
    if freq_hint:
        try:
            offset = pd.tseries.frequencies.to_offset(freq_hint)
        except Exception:
            offset = _freq_to_timedelta(freq_hint)
    min_start: Optional[datetime] = None
    max_end: Optional[datetime] = None
    for line in _iter_json_lines(split_file):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        start_raw = obj.get("start")
        target = obj.get("target")
        if not start_raw or not target:
            continue
        try:
            start_dt = datetime.fromisoformat(str(start_raw))
        except Exception:
            continue
        if offset is not None:
            try:
                end_dt = pd.Timestamp(start_dt) + offset * (max(0, len(target) - 1))
                end_dt = end_dt.to_pydatetime()
            except Exception:
                end_dt = start_dt
        else:
            end_dt = start_dt
        min_start = start_dt if min_start is None or start_dt < min_start else min_start
        max_end = end_dt if max_end is None or end_dt > max_end else max_end
    if min_start is None or max_end is None:
        return placeholder
    return f"{min_start.date()} to {max_end.date()}"


def _sanitize(cell: str) -> str:
    return cell.replace("|", "/").strip()


def _latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def _parse_entry(entry: str, infer_period: bool, placeholder: str, override_period: bool) -> DatasetSpec:
    parts = [p.strip() for p in entry.split("|")]
    if len(parts) < 5:
        raise SystemExit(f"--entry needs at least LABEL|DATASET|[SPLIT]|CONTEXT|PRED|VAL_RATIO; got: {entry}")

    label = parts[0] or "dataset"
    dataset_field = parts[1]
    split_candidate = parts[2] if len(parts) > 2 else ""
    base_idx = 2
    split: str = "train"
    try:
        # If the 3rd token is numeric, it is the context length; otherwise it is the split name.
        int(split_candidate)
    except ValueError:
        if split_candidate:
            split = split_candidate
        base_idx = 3
    except TypeError:
        base_idx = 3

    try:
        context_len = int(parts[base_idx])
        pred_len = int(parts[base_idx + 1])
        val_ratio = float(parts[base_idx + 2])
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"Failed to parse CONTEXT/PRED/VAL_RATIO in entry: {entry}") from exc

    desc = parts[base_idx + 3] if len(parts) > base_idx + 3 else ""
    period = parts[base_idx + 4] if len(parts) > base_idx + 4 else ""
    assets_str = parts[base_idx + 5] if len(parts) > base_idx + 5 else ""
    notes = parts[base_idx + 6] if len(parts) > base_idx + 6 else ""

    data_dir = _resolve_dataset_root(dataset_field)
    assets = int(assets_str) if assets_str else _count_assets(data_dir, split)
    if override_period or not period:
        # Infer period for the requested split only (e.g., train), so we don't mix train/test ranges.
        period = _infer_period_for_split(data_dir, split, placeholder) if infer_period or override_period else placeholder
    test_period = _infer_period_for_split(data_dir, "test", placeholder)

    return DatasetSpec(
        label=label,
        data_dir=data_dir,
        split=split,
        context_len=context_len,
        pred_len=pred_len,
        val_ratio=val_ratio,
        description=desc,
        period=period,
        test_period=test_period,
        assets=assets,
        notes=notes,
    )


def _write_markdown(specs: List[DatasetSpec], path: Path, include_total: bool) -> None:
    headers = ["Dataset", "Description", "Period (train)", "Period (test)", "Assets (N)", "Window (C, P)"]
    if include_total:
        headers.append("Total (L)")
    headers.append("Split/val")
    headers.append("Notes")

    def row_cells(spec: DatasetSpec) -> List[str]:
        cells: List[str] = [
            spec.label,
            spec.description or "",
            spec.period or "",
            spec.test_period or "",
            str(spec.assets),
            f"({spec.context_len}, {spec.pred_len})",
        ]
        if include_total:
            cells.append(str(spec.context_len + spec.pred_len))
        cells.append(f"{spec.split} (val={spec.val_ratio:g})")
        cells.append(spec.notes or "")
        return cells

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for spec in specs:
        cells = [_sanitize(c) for c in row_cells(spec)]
        lines.append("| " + " | ".join(cells) + " |")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_latex(specs: List[DatasetSpec], path: Path, include_total: bool, tight: bool) -> None:
    cols = "l l l l c c"
    if include_total:
        cols += " c"
    cols += " l l"
    if tight:
        cols = "@{}" + cols.replace(" ", "") + "@{}"
    headers = ["Dataset", "Description", "Period (train)", "Period (test)", "Assets (N)", "Window (C, P)"]
    if include_total:
        headers.append("Total (L)")
    headers.append("Split/val")
    headers.append("Notes")

    def row(spec: DatasetSpec) -> List[str]:
        cells: List[str] = [
            spec.label,
            spec.description or "",
            spec.period or "",
            spec.test_period or "",
            str(spec.assets),
            f"({spec.context_len}, {spec.pred_len})",
        ]
        if include_total:
            cells.append(str(spec.context_len + spec.pred_len))
        cells.append(f"{spec.split} (val={spec.val_ratio:g})")
        cells.append(spec.notes or "")
        return cells

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write(" \\toprule\n")
        f.write(" \\ & ".join(_latex_escape(h) for h in headers) + " \\\\ \n")
        f.write(" \\midrule\n")
        for spec in specs:
            cells = [_latex_escape(c) for c in row(spec)]
            f.write(" \\ & ".join(cells) + " \\\\ \n")
        f.write(" \\bottomrule\n\\end{tabular}\n")


def main() -> None:
    args = parse_args()
    specs = [
        _parse_entry(
            e,
            infer_period=args.infer_period,
            placeholder=args.period_placeholder,
            override_period=args.infer_period_override,
        )
        for e in args.entry
    ]
    _write_markdown(specs, args.output_markdown, include_total=args.include_total)
    _write_latex(specs, args.output_latex, include_total=args.include_total, tight=args.latex_tight)
    print(f"[DONE] Wrote {args.output_markdown} and {args.output_latex}")


if __name__ == "__main__":
    main()
