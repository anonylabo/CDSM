#!/usr/bin/env python3
"""
Tabular summary for mean correlation/covariance metrics.

This mirrors `plot_correlation_mean.py` (same run/baseline selection and aggregation)
but renders the per-run metrics into a compact table. Useful when you want the
numbers underneath the heatmaps.
"""

from __future__ import annotations

import argparse
import math
import re
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from plot_correlation_mean import (
    MODEL_ORDER,
    _compact_metric_label,
    _resolve_matrix_kind,
    _stage_indices,
    aggregate_baseline,
    aggregate_model,
    load_samples,
    resolve_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument(
        "--blocks-config",
        nargs="+",
        help="List of YAML files describing blocks to concatenate vertically (each with runs, sample-tags, baselines, assets, etc.). If set, overrides CLI runs/sample-tags/baselines and combines all blocks.",
    )
    parser.add_argument(
        "--block",
        action="append",
        help="Inline YAML/JSON string describing one block (label/runs/sample-tags/baseline/assets...). Can be repeated; overrides CLI runs/sample-tags/baselines when provided.",
    )
    parser.add_argument(
        "--per-block-outputs",
        action="store_true",
        default=False,
        help="If set, also write per-block MD/TeX/PNG. Default: only write combined outputs.",
    )
    parser.add_argument("--runs", nargs=4, help="Optional explicit run dirs for time_cond, time_uncond, fourier_cond, fourier_uncond.")
    parser.add_argument("--sample-tags", nargs=4, help="Optional sample tags (use '-' to select latest).")
    parser.add_argument("--batch-aggregate", action="store_true", help="Treat sample-tag/baseline pointing to batch directories and report mean±std across sub-runs.")
    parser.add_argument("--asset-offset", type=int, default=0)
    parser.add_argument("--assets", type=int, default=10)
    parser.add_argument(
        "--metric-name",
        action="append",
        choices=[
            "matrix_corr_fro",
            "matrix_cov_fro",
            "matrix_corr_fro_rel",
            "matrix_cov_fro_rel",
            "matrix_cov_mse",
            "matrix_cov_mae",
            "matrix_cov_diag_mape",
            "matrix_corr_cross_mse",
            "matrix_corr_offdiag_pearson",
            "matrix_corr_offdiag_spearman",
            "matrix_corr_sign_rate",
            "corr_wasserstein",
            "eigen_wasserstein",
        ],
        default=None,
        help="Metric name(s). Can be repeated to include multiple metrics horizontally.",
    )
    parser.add_argument(
        "--matrix-kind",
        action="append",
        choices=["auto", "corr", "cov"],
        default=None,
        help="Matrix kind per metric (auto/corr/cov). Repeat to match metric-name; extra values are ignored, missing values default to 'auto'.",
    )
    parser.add_argument("--stage", choices=["all", "val", "test"], default="test")
    parser.add_argument(
        "--baseline",
        action="append",
        help="Format: RUN|PREFIX|LABEL[|KIND], where KIND optional is 'cov' or 'corr'. When KIND is set, the baseline is used only for matching matrix-kind metrics.",
    )
    parser.add_argument("--baseline-run", type=Path, help="Legacy: baseline run path.")
    parser.add_argument("--baseline-prefix", type=str, help="Required if baseline-run set.")
    parser.add_argument("--baseline-label", type=str, help="Optional label with baseline-run.")
    parser.add_argument("--use-correlation", action="store_true", help="Force correlation matrices.")
    parser.add_argument(
        "--use-augmented-cov",
        action="store_true",
        help="If set, compute predicted covariance on [context; pred] instead of pred-only.",
    )
    parser.add_argument(
        "--augmented-run-substr",
        action="append",
        help="Apply augmented cov only if model run path contains any of these substrings (overrides global off).",
    )
    parser.add_argument(
        "--include-augmented-variants",
        action="store_true",
        help="If set, duplicate conditional runs as +CB variants (pred-only + context-blend rows).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--table-fontsize", type=int, default=9)
    parser.add_argument("--out-csv", type=Path, help="Save the table as CSV.")
    parser.add_argument("--out-md", type=Path, help="Save the table as Markdown.")
    parser.add_argument("--out-png", type=Path, help="Render the table as a PNG (matplotlib table).")
    parser.add_argument("--out-pdf", type=Path, help="Render the table as a PDF (matplotlib table).")
    parser.add_argument("--out-tex", type=Path, help="Save the table as LaTeX.")
    parser.add_argument(
        "--latex-style",
        choices=["simple", "grouped"],
        default="simple",
        help="LaTeX output style: simple (single header row) or grouped (metric header + regime subcolumns).",
    )
    parser.add_argument(
        "--latex-table-env",
        choices=["table", "table*"],
        default="table",
        help="LaTeX table environment to use when emitting grouped tables.",
    )
    parser.add_argument(
        "--png-style",
        choices=["simple", "grouped", "match"],
        default="match",
        help="PNG output style: simple, grouped (metric header + regime subcolumns), or match --latex-style.",
    )
    parser.add_argument("--latex-caption", type=str, help="Optional LaTeX caption for grouped tables.")
    parser.add_argument("--latex-label", type=str, help="Optional LaTeX label for grouped tables (e.g., tab:...).")
    parser.add_argument("--latex-position", type=str, default="ht", help="LaTeX table float position (grouped).")
    parser.add_argument(
        "--latex-size",
        choices=["none", "small", "footnotesize", "scriptsize", "tiny", "normalsize"],
        default="none",
        help="Optional LaTeX font size for grouped tables (e.g., small).",
    )
    parser.add_argument(
        "--latex-tabcolsep",
        type=float,
        default=None,
        help="Optional LaTeX \\tabcolsep (in pt) for grouped tables.",
    )
    parser.add_argument(
        "--latex-arraystretch",
        type=float,
        default=None,
        help="Optional LaTeX \\arraystretch multiplier for grouped tables.",
    )
    parser.add_argument(
        "--latex-resize",
        type=str,
        default=None,
        help="Optional LaTeX width for \\resizebox{<width>}{!}{...}, e.g. \\\\textwidth.",
    )
    parser.add_argument(
        "--pivot-regime",
        action="store_true",
        help="Pivot regime blocks into sub-columns under each metric (e.g., all/bull/bear under each metric).",
    )
    parser.add_argument(
        "--regime-manifest",
        type=Path,
        help="Optional JSON manifest from label_window_regimes.py to split metrics by regime (e.g., bull/bear/all).",
    )
    parser.add_argument(
        "--only-pred-truth",
        action="store_true",
        help="If set, drop the Pred – Context columns and report only Pred – Truth metrics.",
    )
    parser.add_argument(
        "--sig-digits",
        type=int,
        default=None,
        help="Format mean±std values with this many significant digits (overrides fixed .4f formatting).",
    )
    return parser.parse_args()


def _format_sig(value: float, sig: int) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "nan"
    if value == 0:
        return "0"
    v = abs(float(value))
    exp = math.floor(math.log10(v))
    decimals = sig - 1 - exp
    if decimals >= 0:
        return f"{value:.{decimals}f}"
    q = Decimal("1e{}".format(-decimals))
    d = Decimal(str(value)).quantize(q, rounding=ROUND_HALF_UP)
    return format(d, "f")


def _format_sci(value: float, sig: int) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "nan"
    if value == 0:
        return "0"
    # sig significant digits in scientific notation
    text = f"{value:.{sig - 1}e}"
    # normalize exponent like e-01 -> e-1
    text = text.replace("e-0", "e-").replace("e+0", "e+")
    return text


def _markdown_from_df(df: pd.DataFrame, float_fmt: str = ".4f", caption: str | None = None, sig_digits: int | None = None) -> str:
    def _fmt(value) -> str:
        if isinstance(value, str) and "±" in value:
            return _format_pm_cell(value, float_fmt, sig_digits)
        return _format_cell_text(value, float_fmt, sig_digits)

    headers = list(df.columns)
    lines = []
    if caption:
        lines.append(caption)
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


_LATEX_GROUP_RE = re.compile(r"^(.*) \[([^\]]+)\]$")


def _latex_escape(text: str) -> str:
    return text if "$" in text else text.replace("_", r"\_")


def _latex_format_cell(value, bold: bool, float_fmt: str) -> str:
    if isinstance(value, str) and "±" in value:
        left, right = value.split("±", 1)
        if bold:
            return f"$\\mathbf{{{left}}}\\pm\\mathbf{{{right}}}$"
        return f"${left}\\pm{right}$"
    if isinstance(value, float):
        text = format(value, float_fmt)
        return r"\textbf{" + text + "}" if bold else text
    if isinstance(value, str):
        text = _latex_escape(value)
        return r"\textbf{" + text + "}" if bold else text
    return r"\textbf{" + str(value) + "}" if bold else str(value)


def _format_cell_text(value, float_fmt: str, sig_digits: int | None) -> str:
    if isinstance(value, float):
        if sig_digits is not None:
            return _format_sig(value, sig_digits)
        return format(value, float_fmt)
    return str(value)


def _format_pm_cell(value: str, float_fmt: str, sig_digits: int | None) -> str:
    if sig_digits is None or "±" not in value:
        return value
    left, right = value.split("±", 1)
    try:
        left_f = float(left)
        right_f = float(right)
    except ValueError:
        return value
    return f"{_format_sig(left_f, sig_digits)}±{_format_sci(right_f, sig_digits)}"


def _latex_blocks(
    blocks: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    float_fmt: str = ".4f",
    sig_digits: int | None = None,
    caption: str | None = None,
) -> str:

    if not blocks:
        return ""
    headers = list(blocks[0][1].columns)
    col_spec = "l" + "c" * (len(headers) - 1)
    lines: List[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    if caption:
        lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{" + _latex_escape(caption) + r"} \\")
    lines.append(r"\toprule")
    lines.append(" & ".join(_latex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")

    for blk_idx, (label, df, mask) in enumerate(blocks):
        lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{\textbf{" + _latex_escape(label) + r"}} \\")
        lines.append(r"\midrule")
        for r in range(df.shape[0]):
            cells = []
            for c, col in enumerate(headers):
                val = df.iloc[r, c]
                bold = False
                try:
                    mv = mask.iloc[r, c]
                    bold = bool(mv)
                except Exception:
                    bold = False
                text = _format_pm_cell(val, float_fmt, sig_digits) if isinstance(val, str) else _format_cell_text(val, float_fmt, sig_digits)
                cells.append(_latex_format_cell(text, bold, float_fmt))
            lines.append(" & ".join(cells) + r" \\")
        if blk_idx != len(blocks) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _parse_grouped_headers(headers: List[str]) -> list[tuple[str, list[str]]] | None:
    if len(headers) <= 1:
        return None
    groups: list[tuple[str, list[str]]] = []
    seen = set()
    current_label = None
    for col in headers[1:]:
        match = _LATEX_GROUP_RE.match(col)
        if not match:
            return None
        label = match.group(1).strip()
        regime = match.group(2).strip()
        if current_label is None or label != current_label:
            if label in seen:
                return None
            groups.append((label, [regime]))
            seen.add(label)
            current_label = label
        else:
            groups[-1][1].append(regime)
    return groups


def _latex_grouped_blocks(
    blocks: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    float_fmt: str = ".4f",
    caption: str | None = None,
    label: str | None = None,
    position: str = "ht",
    table_env: str = "table",
    size: str | None = None,
    tabcolsep: float | None = None,
    arraystretch: float | None = None,
    resize_to: str | None = None,
    sig_digits: int | None = None,
) -> str:
    if not blocks:
        return ""
    headers = list(blocks[0][1].columns)
    groups = _parse_grouped_headers(headers)
    if groups is None:
        return _latex_blocks(blocks, float_fmt=float_fmt, caption=caption, sig_digits=sig_digits)

    col_spec = "l" + "c" * (len(headers) - 1)
    lines: List[str] = []
    env = table_env or "table"
    lines.append(r"\begin{" + env + r"}[" + position + r"]")
    lines.append(r"\centering")
    if caption:
        lines.append(r"\caption{" + _latex_escape(caption) + r"}")
    if label:
        lines.append(r"\label{" + _latex_escape(label) + r"}")
    if size and size != "none":
        lines.append("\\" + size)
    if tabcolsep is not None:
        lines.append(r"\setlength{\tabcolsep}{" + format(tabcolsep, ".3f").rstrip("0").rstrip(".") + r"pt}")
    if arraystretch is not None:
        lines.append(r"\renewcommand{\arraystretch}{" + format(arraystretch, ".3f").rstrip("0").rstrip(".") + r"}")
    if resize_to:
        lines.append(r"\resizebox{" + resize_to + r"}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    header_cells = [_latex_escape(headers[0])]
    for metric_label, regimes in groups:
        header_cells.append(
            r"\multicolumn{" + str(len(regimes)) + r"}{c}{" + _latex_escape(metric_label) + r"}"
        )
    lines.append(" & ".join(header_cells) + r" \\")

    cmidrules = []
    col_idx = 2
    for _metric_label, regimes in groups:
        end = col_idx + len(regimes) - 1
        cmidrules.append(r"\cmidrule(lr){" + str(col_idx) + "-" + str(end) + r"}")
        col_idx = end + 1
    lines.append(" ".join(cmidrules))

    subheader_cells = [""]
    for _metric_label, regimes in groups:
        for regime in regimes:
            subheader_cells.append(_latex_escape(f"[{regime}]"))
    lines.append(" & ".join(subheader_cells) + r" \\")
    lines.append(r"\midrule")

    for blk_idx, (label_text, df, mask) in enumerate(blocks):
        lines.append(
            r"\multicolumn{" + str(len(headers)) + r"}{l}{\textbf{" + _latex_escape(label_text) + r"}} \\"
        )
        lines.append(r"\midrule")
        for r in range(df.shape[0]):
            cells = []
            for c, col in enumerate(headers):
                val = df.iloc[r, c]
                bold = False
                try:
                    mv = mask.iloc[r, c]
                    bold = bool(mv)
                except Exception:
                    bold = False
                text = _format_pm_cell(val, float_fmt, sig_digits) if isinstance(val, str) else _format_cell_text(val, float_fmt, sig_digits)
                cells.append(_latex_format_cell(text, bold, float_fmt))
            lines.append(" & ".join(cells) + r" \\")
        if blk_idx != len(blocks) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize_to:
        lines.append(r"}")
    lines.append(r"\end{" + env + r"}")
    return "\n".join(lines)


def _higher_is_better(metric_name: str) -> bool:
    return metric_name in {"matrix_corr_offdiag_pearson", "matrix_corr_offdiag_spearman", "matrix_corr_sign_rate"}


def _process_block(
    outputs_root: Path,
    args: argparse.Namespace,
    block: dict,
    metric_name: str,
    matrix_kind: str,
    use_correlation: bool,
    drop_constant_cols: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs_input = block.get("runs", args.runs)
    sample_tags_input = block.get("sample_tags", args.sample_tags)
    baseline_input = block.get("baseline", args.baseline)
    stage = block.get("stage", args.stage)
    asset_offset = int(block.get("asset_offset", args.asset_offset))
    assets = int(block.get("assets", args.assets))
    batch_aggregate = bool(block.get("batch_aggregate", args.batch_aggregate))
    regime_manifest = block.get("regime_manifest", args.regime_manifest)
    if regime_manifest:
        regime_manifest = Path(regime_manifest).expanduser()

    runs = resolve_runs(outputs_root, runs_input)
    run_dirs: Dict[Tuple, Path] = {k: v for k, v in runs.items() if v is not None}
    sample_tags = sample_tags_input or [None] * len(MODEL_ORDER)
    datasets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for idx, key in enumerate(MODEL_ORDER):
        run_dir = runs.get(key)
        if run_dir is None:
            continue
        tag = sample_tags[idx]
        tag_clean = None if tag in {None, "-", ""} else tag
        if batch_aggregate and tag_clean and "batch-" in tag_clean:
            batch_dir = run_dir / "samples_history" / tag_clean
            if not batch_dir.is_dir():
                raise FileNotFoundError(f"Batch directory not found: {batch_dir}")
            payloads: List[Dict[str, object]] = []
            subdirs = sorted([p for p in batch_dir.iterdir() if p.is_dir()])
            if not subdirs:
                raise FileNotFoundError(f"No sub-runs under batch directory {batch_dir}")
            for sub in subdirs:
                payloads.append(load_samples(sub, sample_tag=None))
            datasets[key] = payloads
        else:
            datasets[key] = [load_samples(run_dir, sample_tag=tag_clean)]

    # Optionally duplicate conditional runs as +CB variants.
    if args.include_augmented_variants:
        for key in list(datasets.keys()):
            if not (isinstance(key, tuple) and len(key) == 2 and key[1] == "conditional"):
                continue
            run_dir = run_dirs.get(key)
            if args.augmented_run_substr:
                if run_dir is None or not any(sub in str(run_dir) for sub in args.augmented_run_substr):
                    continue
            cb_key = (key[0], key[1], "cb")
            datasets[cb_key] = datasets[key]
            if run_dir is not None:
                run_dirs[cb_key] = run_dir

    if not datasets:
        raise SystemExit("No model datasets loaded.")

    # Reference contexts/truths (first available model)
    ref_key = None
    ref_payload = None
    for k, payload_list in datasets.items():
        for payload in payload_list:
            if payload.get("context") is not None:
                ref_key = k
                ref_payload = payload
                break
        if ref_payload is not None:
            break
    if ref_key is None or ref_payload is None:
        raise SystemExit("No model with context available to serve as reference.")
    ref_counts = ref_payload.get("stage_counts", {})
    ref_indices = _stage_indices(ref_counts, stage) if ref_counts else list(range(ref_payload["pred"].shape[0]))
    asset_indices = list(range(asset_offset, asset_offset + assets))
    ref_contexts: List = []
    ref_truths: List = []
    for i in range(len(ref_indices)):
        idx = ref_indices[i]
        ref_contexts.append(ref_payload["context"][idx])
        ref_truths.append(ref_payload["truth"][idx])
    ref_indices_set = set(ref_indices)
    ref_contexts_map = {idx: ref_contexts[i] for i, idx in enumerate(ref_indices)}
    ref_truths_map = {idx: ref_truths[i] for i, idx in enumerate(ref_indices)}

    metrics_stats: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    lengths: Dict[Tuple[str, str], Tuple[float, float]] = {}
    row_keys: List[Tuple[str, str]] = []

    # Regime handling
    regime_map = None
    regime_labels = ["all"]
    if regime_manifest:
        import json

        with regime_manifest.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        regime_map = {int(rec["window_idx"]): rec.get("regime", "unknown") for rec in manifest}
        unique_regs = sorted({v for v in regime_map.values()})
        regime_labels = ["all"] + unique_regs
    else:
        regime_labels = ["all"]

    # Helper to process a set of indices (all / per-regime)
    def process_models_for_indices(indices: List[int], regime_label: str) -> None:
        if not indices:
            return
        for key, payload_list in datasets.items():
            vals_pred_ctx: List[float] = []
            vals_pred_truth: List[float] = []
            lens: List[int] = []
            for payload in payload_list:
                base_key = key[:2] if isinstance(key, tuple) and len(key) >= 2 else key
                run_dir = run_dirs.get(key) or run_dirs.get(base_key)
                if args.include_augmented_variants:
                    use_aug = isinstance(key, tuple) and len(key) == 3 and key[2] == "cb"
                else:
                    use_aug = args.use_augmented_cov or (
                        args.augmented_run_substr
                        and run_dir is not None
                        and any(sub in str(run_dir) for sub in args.augmented_run_substr)
                    )
                length, _mats, mets = aggregate_model(
                    key,
                    payload,
                    indices,
                    asset_indices,
                    metric_name,
                    stage,
                    use_correlation,
                    ref_contexts=ref_contexts_map,
                    use_augmented_cov=use_aug,
                )
                vals_pred_ctx.append(mets["pred_minus_context"])
                vals_pred_truth.append(mets["pred_minus_truth"])
                lens.append(length)
            mean_len = float(np.mean(lens))
            std_len = float(np.std(lens)) if len(lens) > 1 else 0.0
            lengths[(regime_label, key)] = (mean_len, std_len)
            metrics_stats[(regime_label, key)] = {
                "pred_minus_context": (
                    float(np.mean(vals_pred_ctx)),
                    float(np.std(vals_pred_ctx) if len(vals_pred_ctx) > 1 else 0.0),
                ),
                "pred_minus_truth": (
                    float(np.mean(vals_pred_truth)),
                    float(np.std(vals_pred_truth) if len(vals_pred_truth) > 1 else 0.0),
                ),
            }
            row_keys.append((regime_label, key))

    # Baselines
    baseline_specs: List[Tuple[str, str, str | None, str | None]] = []
    if baseline_input:
        for spec in baseline_input:
            spec_clean = spec.strip()
            parts = spec_clean.split("|")
            if len(parts) < 2:
                raise ValueError(f"--baseline entries must be 'RUN|PREFIX|LABEL[|KIND]'; got: {spec}")
            run = parts[0].strip()
            prefix = parts[1]
            label = parts[2] if len(parts) > 2 else None
            kind = parts[3].lower() if len(parts) > 3 else None
            if kind not in {None, "cov", "corr"}:
                raise ValueError(f"Baseline KIND must be cov/corr or omitted; got: {kind}")
            baseline_specs.append((run, prefix, label, kind))
    if args.baseline_run:
        if not args.baseline_prefix:
            raise ValueError("--baseline-prefix must be provided when --baseline-run is used.")
        baseline_specs.append((str(args.baseline_run), args.baseline_prefix, args.baseline_label, None))

    # Standardise legacy baseline names to paper terminology (and keep stable ordering).
    label_map = {
        "Train Uncond": "Sample Covariance",
        "Train Unconditional": "RW Predictor",
        "Sample Covariance": "RW Predictor",
        "Window Uncond": "RW Predictor",
        "Window_uncond": "RW Predictor",
        "Local History": "RW Predictor",
        "Local-History": "RW Predictor",
        "Window Local": "RW Predictor",
        "Window_local": "RW Predictor",
        "Global Static Empirical Covariance": "Static Predictor",
        "Global Static": "Static Predictor",
        "Low-rank Factor": "Rolling Factor Model",
        "Window Factor": "Rolling Factor Model",
        "Window_factor": "Rolling Factor Model",
    }

    def _display_label(prefix: str, label: str | None) -> str:
        display = label or prefix.replace("_", " ").title()
        return label_map.get(display, display)

    # Preserve baseline order based on the CLI input order (first occurrence wins).
    baseline_order: Dict[str, int] = {}
    for _run, _prefix, _label, _kind in baseline_specs:
        disp = _display_label(_prefix, _label)
        if disp not in baseline_order:
            baseline_order[disp] = len(baseline_order)

    def process_baselines_for_indices(indices: List[int], regime_label: str) -> None:
        if not indices:
            return
        for run, prefix, label, kind in baseline_specs:
            if kind and kind != matrix_kind:
                continue  # skip baselines not matching this metric kind
            baseline_run = Path(run).resolve()
            runs_to_use: List[Path] = []
            if args.batch_aggregate:
                subdirs = sorted([p for p in baseline_run.iterdir() if p.is_dir()])
                if subdirs:
                    runs_to_use = subdirs
                else:
                    runs_to_use = [baseline_run]
            else:
                runs_to_use = [baseline_run]

            vals_pred_ctx: List[float] = []
            vals_pred_truth: List[float] = []
            lens: List[int] = []
            for run_dir in runs_to_use:
                # Compatibility: when baselines are run with return_corr_only, the runner writes
                # files under a suffixed prefix (e.g., dcc_garch_corr / dcc_garch_cov).
                eff_prefix = prefix
                if kind and not prefix.endswith(f"_{kind}"):
                    candidate = f"{prefix}_{kind}"
                    if (run_dir / f"{candidate}_summary.json").exists():
                        eff_prefix = candidate

                baseline_kind = kind or ("corr" if eff_prefix.endswith("_corr") else "cov")
                use_baseline_cb = False
                if label and "+CB" in label:
                    use_baseline_cb = True
                if eff_prefix.endswith("_cb") or prefix.endswith("_cb"):
                    use_baseline_cb = True

                # Align baseline window indices with the reference stage indices.
                # Some baselines (e.g., window_baselines) may be run with include_val=false,
                # which makes their test windows start at 0 while the reference model test
                # windows start at ref_val_count.
                try:
                    import json as _json

                    with (run_dir / "stage_counts.json").open("r", encoding="utf-8") as f:
                        base_counts = _json.load(f)
                except Exception:
                    base_counts = {}
                base_val_count = int(base_counts.get("val", 0) or 0)
                ref_val_count = int(ref_counts.get("val", 0) or 0)

                if stage == "test":
                    idx_pairs = [(idx_ref, idx_ref - ref_val_count + base_val_count) for idx_ref in indices]
                else:
                    idx_pairs = [(idx_ref, idx_ref) for idx_ref in indices]

                local_contexts = {idx_base: ref_contexts_map[idx_ref] for idx_ref, idx_base in idx_pairs if idx_ref in ref_contexts_map}
                local_truths = {idx_base: ref_truths_map[idx_ref] for idx_ref, idx_base in idx_pairs if idx_ref in ref_truths_map}
                base_indices = [idx_base for idx_ref, idx_base in idx_pairs if idx_ref in ref_contexts_map and idx_ref in ref_truths_map]
                if not base_indices:
                    continue

                length, _mats, mets = aggregate_baseline(
                    run_dir,
                    eff_prefix,
                    local_contexts,
                    local_truths,
                    asset_indices,
                    metric_name,
                    stage,
                    use_correlation,
                    ref_indices=base_indices,
                    use_augmented_cov=use_baseline_cb,
                    baseline_kind=baseline_kind,
                )
                vals_pred_ctx.append(mets["pred_minus_context"])
                vals_pred_truth.append(mets["pred_minus_truth"])
                lens.append(length)

            display_label = _display_label(prefix, label)
            key = ("baseline", display_label)
            lengths[(regime_label, key)] = (float(np.mean(lens)), float(np.std(lens) if len(lens) > 1 else 0.0))
            metrics_stats[(regime_label, key)] = {
                "pred_minus_context": (
                    float(np.mean(vals_pred_ctx)),
                    float(np.std(vals_pred_ctx) if len(vals_pred_ctx) > 1 else 0.0),
                ),
                "pred_minus_truth": (
                    float(np.mean(vals_pred_truth)),
                    float(np.std(vals_pred_truth) if len(vals_pred_truth) > 1 else 0.0),
                ),
            }
            row_keys.append((regime_label, key))

    # Populate rows for all regimes (or single 'all' block)
    regime_is_global = False
    if regime_map is not None:
        # If the manifest already covers the same indices as the selected stage windows
        # (e.g., new manifests with global window_idx), do NOT apply any offset.
        manifest_idx = set(regime_map.keys())
        regime_is_global = all(idx in manifest_idx for idx in ref_indices)

    for reg in regime_labels:
        if reg == "all" or regime_map is None:
            idx_subset = ref_indices
        else:
            if regime_is_global:
                idx_subset = [idx for idx in ref_indices if regime_map.get(idx) == reg]
            else:
                # Backward-compat: treat manifest indices as local-to-stage ordering.
                allowed_local = {idx for idx, r in regime_map.items() if r == reg}
                idx_subset = [ref_indices[i] for i in sorted(allowed_local) if 0 <= i < len(ref_indices)]
        process_models_for_indices(idx_subset, reg)
        process_baselines_for_indices(idx_subset, reg)

    metric_label = _compact_metric_label(metric_name)
    col_ctx = f"Pred \u2013 Context ({metric_label})"
    col_truth = f"Pred \u2013 Truth ({metric_label})"
    title_lookup = {
        ("time", "conditional"): "CDSM-T",
        ("time", "conditional", "cb"): "CDSM-T+CB",
        ("time", "unconditional"): "CDSM-T WOC",
        ("fourier", "conditional"): "CDSM-S",
        ("fourier", "conditional", "cb"): "CDSM-S+CB",
        ("fourier", "unconditional"): "CDSM-S WOC",
    }
    assets_repr = f"{asset_indices[0]}-{asset_indices[-1]}"

    def _human_label(key_tuple):
        # key_tuple: (regime_label, model_key or ("baseline", label))
        regime_label, model_key = key_tuple
        label = title_lookup.get(model_key, model_key[1] if isinstance(model_key, tuple) else str(model_key))
        block_label = regime_label if regime_map is not None else None
        return block_label, label

    regime_order = {name: i for i, name in enumerate(regime_labels)}
    model_order = {k: i for i, k in enumerate(MODEL_ORDER)}
    desired_row_order = [
        "DeepVAR",
        "DeepVAR+CB",
        "CAB",
        "DCC",
        "DCC-GARCH",
        "RW Predictor",
        "Static Predictor",
        "CDSM-S WOC",
        "CDSM-T WOC",
        "CDSM-S",
        "CDSM-T",
        "CDSM-S+CB",
        "CDSM-T+CB",
    ]
    desired_order_map = {label: i for i, label in enumerate(desired_row_order)}

    def _row_label(model_key):
        if model_key in title_lookup:
            return title_lookup[model_key]
        if isinstance(model_key, tuple) and len(model_key) == 2 and model_key[0] == "baseline":
            return model_key[1]
        if isinstance(model_key, tuple):
            return model_key[1]
        return str(model_key)

    def _row_order(key_tuple) -> int:
        regime_label, model_key = key_tuple
        base = regime_order.get(regime_label, 0) * 1000
        label = _row_label(model_key)
        if label in desired_order_map:
            return base + desired_order_map[label]
        if model_key in model_order:
            return base + len(desired_order_map) + model_order[model_key]
        if isinstance(model_key, tuple) and len(model_key) == 2 and model_key[0] == "baseline":
            return base + len(desired_order_map) + len(MODEL_ORDER) + baseline_order.get(model_key[1], 999)
        return base + 999

    rows = []
    rows_mean_for_best = []
    for key in row_keys:
        block_label, label = _human_label(key)
        k = key
        row = {"Model/Baseline": label}
        row_mean = {"Model/Baseline": label}
        row["__order"] = _row_order(k)
        row_mean["__order"] = _row_order(k)
        if not args.only_pred_truth:
            if args.sig_digits is not None:
                row[col_ctx] = (
                    f"{_format_sig(metrics_stats[k]['pred_minus_context'][0], args.sig_digits)}"
                    f"±{_format_sig(metrics_stats[k]['pred_minus_context'][1], args.sig_digits)}"
                )
            else:
                row[col_ctx] = f"{metrics_stats[k]['pred_minus_context'][0]:.4f}±{metrics_stats[k]['pred_minus_context'][1]:.4f}"
            row_mean[col_ctx] = metrics_stats[k]["pred_minus_context"][0]
        if args.sig_digits is not None:
            row[col_truth] = (
                f"{_format_sig(metrics_stats[k]['pred_minus_truth'][0], args.sig_digits)}"
                f"±{_format_sig(metrics_stats[k]['pred_minus_truth'][1], args.sig_digits)}"
            )
        else:
            row[col_truth] = f"{metrics_stats[k]['pred_minus_truth'][0]:.4f}±{metrics_stats[k]['pred_minus_truth'][1]:.4f}"
        row_mean[col_truth] = metrics_stats[k]["pred_minus_truth"][0]
        if block_label is not None:
            row["Block"] = block_label
            row_mean["Block"] = block_label
        rows.append(row)
        rows_mean_for_best.append(row_mean)

    df = pd.DataFrame(rows)
    df_mean = pd.DataFrame(rows_mean_for_best)
    # Align/rename columns for display
    df = df.rename(columns={"Model/Baseline": "Model / Baseline"})
    df_mean = df_mean.rename(columns={"Model/Baseline": "Model / Baseline"})
    wanted_cols = ["Block", "__order", "Model / Baseline"]
    if not args.only_pred_truth:
        wanted_cols.append(col_ctx)
    wanted_cols.append(col_truth)
    df = df[[c for c in wanted_cols if c in df.columns]].loc[:, lambda x: ~x.columns.duplicated()]
    df_mean = df_mean[[c for c in df_mean.columns if c in df.columns]].loc[:, lambda x: ~x.columns.duplicated()]
    # Drop Block if only one regime
    if "Block" in df.columns and df["Block"].nunique() <= 1:
        df = df.drop(columns=["Block"])
        if "Block" in df_mean.columns:
            df_mean = df_mean.drop(columns=["Block"])
    # Ensure best_mask exists and align columns
    if "best_mask" not in locals():
        best_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for c in df.columns:
        if c not in best_mask.columns:
            best_mask[c] = False
    best_mask = best_mask[df.columns].reset_index(drop=True)
    return df, df_mean


def _pivot_regime_wide(df_in: pd.DataFrame, fill_missing) -> pd.DataFrame:
    if "Block" not in df_in.columns:
        return df_in
    metric_cols_orig = [c for c in df_in.columns if "(" in c and ")" in c]
    blocks = list(df_in["Block"].unique())
    seen = set()
    models: list[str] = []
    order_lookup: dict[str, float] = {}
    for _, row in df_in.iterrows():
        name = row["Model / Baseline"]
        if name not in seen:
            seen.add(name)
            models.append(name)
        if "__order" in df_in.columns and name not in order_lookup:
            order_lookup[name] = row["__order"]
    rows: list[dict] = []
    for name in models:
        row: dict = {"Model / Baseline": name}
        if "__order" in df_in.columns:
            row["__order"] = order_lookup.get(name, 0)
        for col in metric_cols_orig:
            for blk in blocks:
                col_name = f"{col} [{blk}]"
                mask = (df_in["Model / Baseline"] == name) & (df_in["Block"] == blk)
                series = df_in.loc[mask, col]
                val = series.iloc[0] if not series.empty else fill_missing
                if pd.isna(val):
                    val = fill_missing
                row[col_name] = val
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    # Normalize metric/matrix_kind lists
    metric_names = args.metric_name or ["matrix_corr_fro"]
    matrix_kinds_raw = args.matrix_kind or []
    matrix_kinds: list[str] = []
    for i in range(len(metric_names)):
        if i < len(matrix_kinds_raw) and matrix_kinds_raw[i]:
            matrix_kinds.append(matrix_kinds_raw[i])
        elif matrix_kinds_raw:
            matrix_kinds.append(matrix_kinds_raw[-1] or "auto")
        else:
            matrix_kinds.append("auto")

    blocks: list[dict] = []
    if args.blocks_config:
        for cfg_path in args.blocks_config:
            with open(cfg_path, "r") as f:
                blocks.append(yaml.safe_load(f))
    if args.block:
        for inline in args.block:
            blocks.append(yaml.safe_load(inline))

    def _run_for_metric(metric_name: str, matrix_kind: str):
        use_corr = _resolve_matrix_kind(metric_name, matrix_kind, args.use_correlation) == "corr"
        if blocks:
            dfs = []
            df_means = []
            labels = []
            per_block_tables_local: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
            for block in blocks:
                label = block.get("label", "block")
                df_block, df_mean_block = _process_block(
                    args.outputs_root, args, block, metric_name, matrix_kind, use_corr, drop_constant_cols=False
                )
                if args.pivot_regime and "Block" in df_block.columns:
                    df_block = _pivot_regime_wide(df_block, fill_missing="-")
                    df_mean_block = _pivot_regime_wide(df_mean_block, fill_missing=np.nan)
                elif "Block" in df_block.columns:
                    df_block = df_block.rename(columns={"Block": "Regime"})
                    df_mean_block = df_mean_block.rename(columns={"Block": "Regime"})
                df_block.insert(0, "Block", label)
                df_mean_block.insert(0, "Block", label)
                dfs.append(df_block)
                df_means.append(df_mean_block)
                labels.append(label)
                per_block_tables_local.append((label, df_block.copy(), df_mean_block.copy()))
            df_local = pd.concat(dfs, ignore_index=True)
            df_mean_local = pd.concat(df_means, ignore_index=True)
            caption_local = "; ".join(labels)
        else:
            df_local, df_mean_local = _process_block(
                args.outputs_root, args, {}, metric_name, matrix_kind, use_corr, drop_constant_cols=True
            )
            caption_local = None
            per_block_tables_local = []
        return df_local, df_mean_local, caption_local, per_block_tables_local

    # First metric
    df, df_mean, caption_text, per_block_tables = _run_for_metric(metric_names[0], matrix_kinds[0])
    # Merge additional metrics horizontally
    for metric_name, matrix_kind in zip(metric_names[1:], matrix_kinds[1:]):
        df_extra, df_mean_extra, _cap, _per = _run_for_metric(metric_name, matrix_kind)
        on_cols = []
        if "Block" in df.columns and "Block" in df_extra.columns:
            on_cols.append("Block")
        if "Regime" in df.columns and "Regime" in df_extra.columns:
            on_cols.append("Regime")
        on_cols.append("Model / Baseline")
        if "__order" in df.columns and "__order" in df_extra.columns:
            on_cols.append("__order")
        df = df.merge(df_extra, on=on_cols, how="outer", sort=False)
        df_mean = df_mean.merge(df_mean_extra, on=on_cols, how="outer", sort=False)

    # Stable ordering: respect input order via __order (and Block order if present)
    if "Block" in df.columns:
        block_order = list(df["Block"].unique())
        df["Block"] = pd.Categorical(df["Block"], categories=block_order, ordered=True)
        df_mean["Block"] = pd.Categorical(df_mean["Block"], categories=block_order, ordered=True)
        sort_cols = ["Block", "__order"]
    else:
        sort_cols = ["__order"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df_mean = df_mean.sort_values(sort_cols).reset_index(drop=True)

    # Optional: pivot regimes into sub-columns per metric for cleaner display
    if args.pivot_regime and "Block" in df.columns and not blocks:
        blocks = list(df["Block"].unique())
        metric_cols_orig = [c for c in df.columns if "(" in c and ")" in c]

        def _pivot_table(df_in: pd.DataFrame, fill_missing="-") -> pd.DataFrame:
            seen = set()
            models: list[str] = []
            for _, r in df_in.iterrows():
                name = r["Model / Baseline"]
                if name not in seen:
                    seen.add(name)
                    models.append(name)
            rows: list[dict] = []
            for name in models:
                row: dict = {"Model / Baseline": name}
                for col in metric_cols_orig:
                    for blk in blocks:
                        col_name = f"{col} [{blk}]"
                        mask = (df_in["Model / Baseline"] == name) & (df_in["Block"] == blk)
                        series = df_in.loc[mask, col]
                        val = series.iloc[0] if not series.empty else fill_missing
                        if pd.isna(val):
                            val = fill_missing
                        row[col_name] = val
                rows.append(row)
            return pd.DataFrame(rows)

        df = _pivot_table(df, fill_missing="-")
        df_mean = _pivot_table(df_mean, fill_missing=np.nan)

    # Hide internal ordering column from outputs
    if "__order" in df.columns:
        df = df.drop(columns=["__order"])
    if "__order" in df_mean.columns:
        df_mean = df_mean.drop(columns=["__order"])

    # Render missing cells (e.g., baselines only applicable to cov/corr metrics)
    metric_cols = [c for c in df.columns if "(" in c and ")" in c]
    if metric_cols:
        df[metric_cols] = df[metric_cols].fillna("-")

    # Compute best-mask before display tweaks
    better_is_higher = _higher_is_better(metric_names[0] if isinstance(metric_names, list) else metric_names)
    best_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for block_label in df["Block"].unique() if "Block" in df.columns else [None]:
        block_idx = df["Block"] == block_label if "Block" in df.columns else slice(None)
        for col in metric_cols:
            series = pd.to_numeric(df_mean.loc[block_idx, col], errors="coerce")
            series_nonan = series.dropna()
            if series_nonan.empty:
                continue
            if better_is_higher:
                best_val = float(series_nonan.max())
                best_mask.loc[block_idx, col] = series == best_val
            else:
                best_val = float(series_nonan.min())
                best_mask.loc[block_idx, col] = series == best_val

    # Split into blocks for output formatting
    block_tables: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    if "Block" in df.columns:
        display_cols = [c for c in df.columns if c != "Block"]
        for blk in df["Block"].unique():
            sub = df[df["Block"] == blk][display_cols].reset_index(drop=True)
            sub_mask = best_mask[df["Block"] == blk][display_cols].reset_index(drop=True)
            block_tables.append((blk, sub, sub_mask))
    else:
        label = caption_text or "dataset"
        block_tables.append((label, df.copy(), best_mask.copy()))

    # Markdown / stdout
    def _markdown_blocks(blocks_md: list[tuple[str, pd.DataFrame]], caption: str | None = None) -> str:
        headers = list(blocks_md[0][1].columns)
        lines: list[str] = []
        if caption:
            lines.append(caption)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for label, subdf in blocks_md:
            lines.append("| **" + label + "** | " + " | ".join([""] * (len(headers) - 1)) + " |")
            for _, row in subdf.iterrows():
                row_cells = []
                for v in row.tolist():
                    if isinstance(v, str) and "±" in v:
                        row_cells.append(_format_pm_cell(v, ".4f", args.sig_digits))
                    else:
                        row_cells.append(_format_cell_text(v, ".4f", args.sig_digits))
                lines.append("| " + " | ".join(row_cells) + " |")
        return "\n".join(lines)

    table_text = _markdown_blocks([(lbl, dfblk) for lbl, dfblk, _ in block_tables], caption=caption_text)
    print(table_text)

    # CSV output keeps an explicit Dataset column
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_tables = []
        for label, subdf, _mask in block_tables:
            tmp = subdf.copy()
            tmp.insert(0, "Dataset", label)
            csv_tables.append(tmp)
        pd.concat(csv_tables, ignore_index=True).to_csv(args.out_csv, index=False)
        print(f"Saved CSV to {args.out_csv}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(table_text)
        print(f"Saved Markdown table to {args.out_md}")
        if blocks and args.per_block_outputs:
            base = args.out_md
            stem, suffix = base.stem, base.suffix or ".md"
            for label, subdf, _mask in block_tables:
                blk_path = base.with_name(f"{stem}_{label}{suffix}")
                blk_path.write_text(_markdown_blocks([(label, subdf)], caption=label))
                print(f"Saved per-block Markdown to {blk_path}")
    if args.out_tex:
        latex_caption = args.latex_caption if args.latex_caption is not None else caption_text
        if args.latex_style == "grouped":
            latex_text = _latex_grouped_blocks(
                block_tables,
                float_fmt=".4f",
                caption=latex_caption,
                label=args.latex_label,
                position=args.latex_position,
                table_env=args.latex_table_env,
                size=args.latex_size,
                tabcolsep=args.latex_tabcolsep,
                arraystretch=args.latex_arraystretch,
                resize_to=args.latex_resize,
                sig_digits=args.sig_digits,
            )
        else:
            latex_text = _latex_blocks(block_tables, float_fmt=".4f", caption=caption_text, sig_digits=args.sig_digits)
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_tex.write_text(latex_text)
        print(f"Saved LaTeX table to {args.out_tex}")
        if blocks and args.per_block_outputs:
            base = args.out_tex
            stem, suffix = base.stem, base.suffix or ".tex"
            for label, subdf, submask in block_tables:
                blk_path = base.with_name(f"{stem}_{label}{suffix}")
                if args.latex_style == "grouped":
                    blk_path.write_text(
                        _latex_grouped_blocks(
                            [(label, subdf, submask)],
                            float_fmt=".4f",
                            caption=label,
                            position=args.latex_position,
                            table_env=args.latex_table_env,
                            size=args.latex_size,
                            tabcolsep=args.latex_tabcolsep,
                            arraystretch=args.latex_arraystretch,
                            resize_to=args.latex_resize,
                            sig_digits=args.sig_digits,
                        )
                    )
                else:
                    blk_path.write_text(_latex_blocks([(label, subdf, submask)], float_fmt=".4f", caption=label, sig_digits=args.sig_digits))
                print(f"Saved per-block LaTeX to {blk_path}")
    def _render_table_figure(out_path: Path, default_suffix: str) -> None:
        fig_style = args.png_style
        if fig_style == "match":
            fig_style = args.latex_style
        use_grouped_fig = fig_style == "grouped"

        def _render_table(df_in: pd.DataFrame, mask_in: pd.DataFrame, caption: str | None, path: Path) -> None:
            def _format_col_label(label: str) -> str:
                # Wrap long metric headers so they don't overlap in the PNG table.
                # Examples:
                #   "Pred – Truth (F(cov Δ)) [bear]" -> "Pred – Truth\n(F(cov Δ))\n[bear]"
                #   "Pred – Truth (F(cov Δ))" -> "Pred – Truth\n(F(cov Δ))"
                if label in {"Block", "Model / Baseline"}:
                    return label
                base = label
                suffix = None
                if label.endswith("]") and " [" in label:
                    base, suffix = label.rsplit(" [", 1)
                    suffix = "[" + suffix  # re-add
                import re

                match = re.match(r"^(.*?)\s*\((.*)\)$", base)
                if match:
                    parts = [match.group(1).strip(), f"({match.group(2).strip()})"]
                else:
                    parts = [base.strip()]
                if suffix:
                    parts.append(suffix.strip())
                return "\n".join([p for p in parts if p])

            col_names = df_in.columns.tolist()
            left_cols = [c for c in col_names if c in {"Block", "Model / Baseline"}]
            left_count = len(left_cols)
            grouped_headers = None
            if use_grouped_fig:
                metric_headers = col_names[left_count:]
                grouped_headers = _parse_grouped_headers(["_"] + metric_headers)
                if grouped_headers is not None:
                    total = sum(len(regs) for _label, regs in grouped_headers)
                    if total != len(metric_headers):
                        grouped_headers = None
            if grouped_headers:
                col_labels = left_cols[:]
                for _metric_label, regimes in grouped_headers:
                    col_labels.extend([f"[{reg}]" for reg in regimes])
            else:
                col_labels = [_format_col_label(c) for c in col_names]

            max_lines = max((lbl.count("\n") + 1 for lbl in col_labels), default=1)
            extra_header = 1 if grouped_headers else 0
            fig_w = max(8.0, 1.55 * len(df_in.columns))
            fig_h = max(2.5, 0.55 * len(df_in) + 1.2 + 0.35 * (max_lines - 1) + 0.45 * extra_header)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
            ax.axis("off")
            cell_vals = [[df_in.iloc[i, j] for j in range(df_in.shape[1])] for i in range(df_in.shape[0])]
            table = ax.table(cellText=cell_vals, colLabels=col_labels, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(args.table_fontsize)
            # Give extra vertical space when headers wrap onto multiple lines.
            table.scale(1.0, 1.25 + 0.25 * (max_lines - 1))
            if caption:
                ax.set_title(caption, fontsize=10, pad=6)
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.get_text().set_fontweight("bold")
                    cell.get_text().set_ha("center")
                    cell.get_text().set_va("center")
                    # Make the header row taller to fit wrapped labels.
                    cell.set_height(cell.get_height() * (1.0 + 0.35 * (max_lines - 1)))
                    continue
                row_idx = r - 1
                if row_idx >= len(df_in):
                    continue
                col_name = df_in.columns[c] if c < len(df_in.columns) else None
                if col_name is not None and bool(mask_in.iloc[row_idx, c]):
                    cell.get_text().set_fontweight("bold")
            if grouped_headers:
                fig.canvas.draw()
                start_col = left_count
                for metric_label, regimes in grouped_headers:
                    if not regimes:
                        continue
                    c0 = start_col
                    c1 = start_col + len(regimes) - 1
                    cell0 = table.get_celld().get((0, c0))
                    cell1 = table.get_celld().get((0, c1))
                    if cell0 is None or cell1 is None:
                        start_col = c1 + 1
                        continue
                    x0 = cell0.get_x()
                    x1 = cell1.get_x() + cell1.get_width()
                    y = cell0.get_y() + cell0.get_height() * 1.05
                    ax.text(
                        (x0 + x1) / 2.0,
                        y,
                        metric_label,
                        ha="center",
                        va="bottom",
                        fontsize=args.table_fontsize,
                        fontweight="bold",
                        transform=ax.transAxes,
                    )
                    start_col = c1 + 1
            fig.tight_layout()
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved table figure to {path}")

        base = out_path
        if base.suffix == "":
            base = base.with_suffix(default_suffix)
        stem = base.stem
        suffix = base.suffix

        # Combined figure (all blocks stacked with a Block column)
        if len(block_tables) > 1:
            combined_tables = []
            combined_masks = []
            for label, subdf, submask in block_tables:
                df_with_block = subdf.copy()
                df_with_block.insert(0, "Block", label)
                mask_with_block = submask.copy()
                mask_with_block.insert(0, "Block", False)
                combined_tables.append(df_with_block)
                combined_masks.append(mask_with_block)
            df_combined = pd.concat(combined_tables, ignore_index=True)
            mask_combined = pd.concat(combined_masks, ignore_index=True)
            _render_table(df_combined, mask_combined, caption_text, base)

            if args.per_block_outputs:
                for label, subdf, submask in block_tables:
                    blk_path = base.with_name(f"{stem}_{label}{suffix}")
                    _render_table(subdf, submask, label, blk_path)
        else:
            _render_table(block_tables[0][1], block_tables[0][2], caption_text, base)

    if args.out_png:
        _render_table_figure(args.out_png, ".png")
    if args.out_pdf:
        _render_table_figure(args.out_pdf, ".pdf")


if __name__ == "__main__":
    main()
