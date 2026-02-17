#!/usr/bin/env python3
"""
Compute non-overlapping train/val/test window counts for GluonTS-style datasets.

Example (same settings applied to multiple datasets):
python scripts/make_window_count_table.py \
  --dataset "FX:~/.gluonts/datasets/exchange_rate_clean" \
  --dataset "ind49:~/.gluonts/datasets/industry49_clean" \
  --dataset "stock14:~/.gluonts/datasets/ishares14_clean" \
  --dataset "stock6a:~/.gluonts/datasets/ishares6_clean_train0.75" \
  --dataset "stock6b:~/.gluonts/datasets/ishares6_clean_train0.8" \
  --context-len 30 --pred-len 15 --stride 1 --train-ratio 0.8 --val-ratio 0.3 \
  --output-markdown assets/window_counts.md \
  --output-latex assets/window_counts.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cfdiff.dataloaders.conditional_gluonts import (
    _build_windows,
    _load_gluonts_like,
    _resolve_split_file,
)


@dataclass
class WindowCounts:
    label: str
    assets: int
    train_windows: int
    val_windows: int
    test_windows: int


@dataclass
class DatasetConfig:
    context_len: int
    pred_len: int
    stride: int
    val_ratio: float
    train_ratio: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help='Dataset spec as LABEL:PATH[:C][:P][:VAL][:TRAIN][:STRIDE]. '
        "If C/P/VAL/TRAIN/STRIDE are omitted, fall back to global flags.",
    )
    ap.add_argument("--context-len", type=int, help="Global context length C (can be overridden per dataset).")
    ap.add_argument("--pred-len", type=int, help="Global prediction length P (can be overridden per dataset).")
    ap.add_argument("--stride", type=int, default=1, help="Global sliding stride (default: 1).")
    ap.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="Global fraction of training windows to keep (front portion) when --apply-train-ratio is set.",
    )
    ap.add_argument(
        "--train-ratio-from-metadata",
        action="store_true",
        help="If metadata.json has notes.train_ratio, use it to populate train_ratio for display (and trimming).",
    )
    ap.add_argument(
        "--apply-train-ratio",
        dest="apply_train_ratio",
        action="store_true",
        default=True,
        help="Apply train_ratio to trim training windows (default: true).",
    )
    ap.add_argument(
        "--no-apply-train-ratio",
        dest="apply_train_ratio",
        action="store_false",
        help="Do not trim training windows; keep full train split (train_ratio used for display only).",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.3,
        help="Global validation ratio. If >=0, val/test are drawn from test split with non-overlap. "
        "If <0, val is taken from the tail of the train windows with an auto gap.",
    )
    ap.add_argument(
        "--val-test-gap",
        type=int,
        default=-1,
        help="Gap (in windows) between val and test when val_ratio>=0. Default -1 uses ceil((C+P)/stride)-1.",
    )
    ap.add_argument(
        "--train-val-gap",
        type=int,
        default=-1,
        help="Gap (in windows) between train and val when val_ratio<0. Default -1 uses ceil((C+P)/stride)-1.",
    )
    ap.add_argument(
        "--align-tail",
        dest="align_tail",
        action="store_true",
        default=True,
        help="Align windows to the tail (matches align_tail_windows=True in datamodule). Enabled by default.",
    )
    ap.add_argument(
        "--no-align-tail",
        dest="align_tail",
        action="store_false",
        help="Disable tail alignment.",
    )
    ap.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("assets/window_counts.md"),
        help="Markdown output path.",
    )
    ap.add_argument(
        "--output-latex",
        type=Path,
        default=Path("assets/window_counts.tex"),
        help="LaTeX output path.",
    )
    ap.add_argument(
        "--per-entry-markdown-dir",
        type=Path,
        default=None,
        help="If set, also write one Markdown table per dataset into this directory.",
    )
    ap.add_argument(
        "--per-entry-latex-dir",
        type=Path,
        default=None,
        help="If set, also write one LaTeX table per dataset into this directory.",
    )
    ap.add_argument(
        "--omit-stride",
        dest="include_stride",
        action="store_false",
        help="Omit stride column for a tighter table.",
    )
    ap.add_argument(
        "--omit-ratios",
        dest="include_ratios",
        action="store_false",
        help="Omit val/train ratio columns for a tighter table.",
    )
    ap.add_argument(
        "--latex-tight",
        action="store_true",
        help="Use @{\\,} column padding in LaTeX to reduce horizontal whitespace.",
    )
    ap.add_argument(
        "--short-headers",
        action="store_true",
        help="Use abbreviated column headers (Assets -> N, val_ratio -> vr, train_ratio -> tr, Stride -> S).",
    )
    ap.set_defaults(include_stride=True, include_ratios=True)
    return ap.parse_args()


def _resolve_dataset_root(ds: str) -> Path:
    candidate = Path(ds).expanduser()
    if candidate.exists():
        return candidate
    fallback = Path.home() / ".gluonts" / "datasets" / candidate
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Dataset path not found: {candidate}")


def _load_metadata_train_ratio(data_dir: Path) -> Optional[float]:
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    notes = meta.get("notes", {}) if isinstance(meta, dict) else {}
    ratio = notes.get("train_ratio")
    if ratio is None:
        return None
    try:
        return float(ratio)
    except (TypeError, ValueError):
        return None


def _parse_dataset_spec(
    spec: str,
    default_context: Optional[int],
    default_pred: Optional[int],
    default_val_ratio: float,
    default_train_ratio: float,
    default_stride: int,
) -> Tuple[str, Path, DatasetConfig]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise ValueError(f"--dataset must be LABEL:PATH[:C][:P][:VAL][:TRAIN][:STRIDE]; got {spec}")
    label = parts[0].strip() or Path(parts[1]).expanduser().name
    path = _resolve_dataset_root(parts[1])

    def _parse_opt(idx: int, fn, fallback):
        if idx < len(parts) and parts[idx] != "":
            return fn(parts[idx])
        return fallback

    ctx = _parse_opt(2, int, default_context)
    pred = _parse_opt(3, int, default_pred)
    val_ratio = _parse_opt(4, float, default_val_ratio)
    train_ratio = _parse_opt(5, float, default_train_ratio)
    stride = _parse_opt(6, int, default_stride)

    if ctx is None or pred is None:
        raise ValueError(f"Context/pred length missing for dataset {label}; specify globally or in the spec.")
    if stride <= 0:
        raise ValueError("Stride must be positive.")
    return label, path, DatasetConfig(ctx, pred, stride, val_ratio, train_ratio)


def _build_non_overlap_counts(
    data_dir: Path,
    cfg: DatasetConfig,
    align_tail: bool,
    val_test_gap: int,
    train_val_gap: int,
    apply_train_ratio: bool,
) -> Tuple[int, int, int, int]:
    window_total = cfg.context_len + cfg.pred_len
    Xtr = _load_gluonts_like(_resolve_split_file(data_dir, "train"))
    Xt = _load_gluonts_like(_resolve_split_file(data_dir, "test"))
    assets = int(Xtr.shape[1])

    train_windows = _build_windows(Xtr, window=window_total, stride=cfg.stride, align_end=align_tail)
    if train_windows.shape[0] == 0:
        raise ValueError("No train windows generated.")
    if apply_train_ratio:
        if cfg.train_ratio <= 0:
            raise ValueError("train_ratio must be positive.")
        keep = int(round(train_windows.shape[0] * cfg.train_ratio))
        keep = max(1, min(train_windows.shape[0], keep))
        train_windows = train_windows[:keep]

    if cfg.val_ratio < 0:
        total_train = int(train_windows.shape[0])
        val_count = int(round(total_train * abs(cfg.val_ratio)))
        min_val = 2 if total_train >= 2 else 1
        val_count = max(min_val, val_count)
        if val_count >= total_train:
            raise ValueError("val_ratio yields no train windows after split.")
        if train_val_gap < 0:
            gap = max(0, (window_total + cfg.stride - 1) // cfg.stride - 1)
        else:
            gap = max(0, int(train_val_gap))
        train_end = total_train - val_count - gap
        if train_end <= 0:
            raise ValueError("Gap plus val windows exhaust train windows.")
        train_count = train_end
        val_windows = val_count
        eval_windows = _build_windows(Xt, window=window_total, stride=cfg.stride, align_end=align_tail)
        if eval_windows.shape[0] == 0:
            raise ValueError("No test windows generated.")
        test_windows = int(eval_windows.shape[0])
        return assets, train_count, val_windows, test_windows

    # val/test from test split (no overlap)
    eval_windows = _build_windows(Xt, window=window_total, stride=cfg.stride, align_end=align_tail)
    if eval_windows.shape[0] == 0:
        raise ValueError("No evaluation windows generated from test split.")
    total_eval = int(eval_windows.shape[0])

    if cfg.val_ratio <= 0:
        val_count = 1
    else:
        val_count = int(round(total_eval * cfg.val_ratio))
    min_val = 2 if total_eval >= 2 else 1
    val_count = max(min_val, val_count)
    val_count = min(val_count, total_eval)

    if val_test_gap < 0:
        gap = max(0, (window_total + cfg.stride - 1) // cfg.stride - 1)
    else:
        gap = max(0, int(val_test_gap))
    test_start = min(total_eval, val_count + gap)
    if test_start >= total_eval:
        raise ValueError("Gap plus val windows exhaust evaluation windows.")
    test_count = total_eval - test_start
    return assets, int(train_windows.shape[0]), val_count, test_count


def _write_markdown(
    path: Path,
    rows: List[Tuple[str, WindowCounts, DatasetConfig]],
    include_stride: bool,
    include_ratios: bool,
    short_headers: bool,
) -> None:
    headers = ["Dataset", "N" if short_headers else "Assets (N)", "Train", "Val", "Test", "C", "P"]
    if include_stride:
        headers.append("S" if short_headers else "Stride")
    if include_ratios:
        headers.extend(["vr" if short_headers else "val_ratio", "tr" if short_headers else "train_ratio"])
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for label, wc, cfg in rows:
        cells = [
            label,
            str(wc.assets),
            str(wc.train_windows),
            str(wc.val_windows),
            str(wc.test_windows),
            str(cfg.context_len),
            str(cfg.pred_len),
        ]
        if include_stride:
            cells.append(str(cfg.stride))
        if include_ratios:
            cells.extend([f"{cfg.val_ratio:g}", f"{cfg.train_ratio:g}"])
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(
    path: Path,
    rows: List[Tuple[str, WindowCounts, DatasetConfig]],
    include_stride: bool,
    include_ratios: bool,
    tight: bool,
    short_headers: bool,
) -> None:
    headers = ["Dataset", "N" if short_headers else "Assets (N)", "Train", "Val", "Test", "C", "P"]
    if include_stride:
        headers.append("S" if short_headers else "Stride")
    if include_ratios:
        headers.extend(["vr" if short_headers else "val\\_ratio", "tr" if short_headers else "train\\_ratio"])
    base_cols = ["l", "c", "c", "c", "c", "c", "c"]
    if include_stride:
        base_cols.append("c")
    if include_ratios:
        base_cols.extend(["c", "c"])
    if tight:
        colspec = "@{}" + "".join(base_cols) + "@{}"
    else:
        colspec = " ".join(base_cols)

    def esc(s: str) -> str:
        return s.replace("_", "\\_")

    lines = []
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append(" \\toprule")
    lines.append(" \\ & ".join(headers) + " \\\\ ")
    lines.append(" \\midrule")
    for label, wc, cfg in rows:
        cells = [
            esc(label),
            str(wc.assets),
            str(wc.train_windows),
            str(wc.val_windows),
            str(wc.test_windows),
            str(cfg.context_len),
            str(cfg.pred_len),
        ]
        if include_stride:
            cells.append(str(cfg.stride))
        if include_ratios:
            cells.extend([f"{cfg.val_ratio:g}", f"{cfg.train_ratio:g}"])
        lines.append(" \\ & ".join(cells) + " \\\\ ")
    lines.append(" \\bottomrule")
    lines.append("\\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows: List[Tuple[str, WindowCounts, DatasetConfig]] = []

    for ds_spec in args.dataset:
        label, path, cfg = _parse_dataset_spec(
            ds_spec,
            default_context=args.context_len,
            default_pred=args.pred_len,
            default_val_ratio=args.val_ratio,
            default_train_ratio=args.train_ratio,
            default_stride=args.stride,
        )
        if args.train_ratio_from_metadata:
            meta_ratio = _load_metadata_train_ratio(path)
            if meta_ratio is not None:
                cfg.train_ratio = meta_ratio
        assets, train_c, val_c, test_c = _build_non_overlap_counts(
            data_dir=path,
            cfg=cfg,
            align_tail=args.align_tail,
            val_test_gap=args.val_test_gap,
            train_val_gap=args.train_val_gap,
            apply_train_ratio=args.apply_train_ratio,
        )
        rows.append(
            (
                label,
                WindowCounts(
                    label=label,
                    assets=assets,
                    train_windows=train_c,
                    val_windows=val_c,
                    test_windows=test_c,
                ),
                cfg,
            )
        )

    _write_markdown(
        args.output_markdown,
        rows,
        include_stride=args.include_stride,
        include_ratios=args.include_ratios,
        short_headers=args.short_headers,
    )
    _write_latex(
        args.output_latex,
        rows,
        include_stride=args.include_stride,
        include_ratios=args.include_ratios,
        tight=args.latex_tight,
        short_headers=args.short_headers,
    )
    if args.per_entry_markdown_dir is not None:
        md_dir = args.per_entry_markdown_dir.expanduser().resolve()
        md_dir.mkdir(parents=True, exist_ok=True)
        for label, wc, cfg in rows:
            fname = f"window_counts_{label.replace('/', '_')}_C{cfg.context_len}_P{cfg.pred_len}_S{cfg.stride}.md"
            _write_markdown(
                md_dir / fname,
                [(label, wc, cfg)],
                include_stride=args.include_stride,
                include_ratios=args.include_ratios,
                short_headers=args.short_headers,
            )
    if args.per_entry_latex_dir is not None:
        tex_dir = args.per_entry_latex_dir.expanduser().resolve()
        tex_dir.mkdir(parents=True, exist_ok=True)
        for label, wc, cfg in rows:
            fname = f"window_counts_{label.replace('/', '_')}_C{cfg.context_len}_P{cfg.pred_len}_S{cfg.stride}.tex"
            _write_latex(
                tex_dir / fname,
                [(label, wc, cfg)],
                include_stride=args.include_stride,
                include_ratios=args.include_ratios,
                tight=args.latex_tight,
                short_headers=args.short_headers,
            )
    print(f"[DONE] Wrote {args.output_markdown} and {args.output_latex}")


if __name__ == "__main__":
    main()
