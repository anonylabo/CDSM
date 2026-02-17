#!/usr/bin/env python3
"""
Line plot of Pred-Truth / Pred-Context gaps versus asset count.

Design goals:
- Reuse the same loaders/aggregation as plot_correlation_mean/plot_correlation_table.
- Accept the same block YAML (inline via --block or files via --blocks-config).
- Focus on the four diffusion variants plus baselines; highlight where baselines
  overtake diffusion as assets grow.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from plot_correlation_mean import (
    MODEL_ORDER,
    _resolve_matrix_kind,
    _stage_indices,
    aggregate_baseline,
    aggregate_model,
    load_samples,
    resolve_runs,
)


FOCUS_LABELS = {
    ("time", "conditional"): "Time (Conditional)",
    ("time", "unconditional"): "Time (Unconditional)",
    ("fourier", "conditional"): "Fourier (Conditional)",
    ("fourier", "unconditional"): "Fourier (Unconditional)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument(
        "--blocks-config",
        nargs="+",
        help="YAML files describing blocks (label/runs/sample-tags/baseline/assets...).",
    )
    parser.add_argument(
        "--block",
        action="append",
        help="Inline YAML/JSON block (label/runs/sample-tags/baseline/assets...). Can repeat.",
    )
    parser.add_argument("--runs", nargs=4, help="Optional explicit run dirs for time_cond, time_uncond, fourier_cond, fourier_uncond.")
    parser.add_argument("--sample-tags", nargs=4, help="Optional sample tags (use '-' to select latest).")
    parser.add_argument("--batch-aggregate", action="store_true", help="Treat sample-tag/baseline paths as batch dirs and average sub-runs.")
    parser.add_argument("--asset-offset", type=int, default=0)
    parser.add_argument("--assets", type=int, default=10)
    parser.add_argument(
        "--metric-name",
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
        ],
        default="matrix_corr_fro",
    )
    parser.add_argument("--metric-field", choices=["pred_minus_truth", "pred_minus_context"], default="pred_minus_truth", help="Which gap to plot on y-axis.")
    parser.add_argument("--matrix-kind", choices=["auto", "corr", "cov"], default="auto")
    parser.add_argument("--stage", choices=["all", "val", "test"], default="test")
    parser.add_argument(
        "--baseline",
        action="append",
        help="Format: RUN|PREFIX|LABEL (label optional); falls back to --baseline-run if set.",
    )
    parser.add_argument("--baseline-run", type=Path, help="Legacy single baseline run path.")
    parser.add_argument("--baseline-prefix", type=str, help="Required if baseline-run set.")
    parser.add_argument("--baseline-label", type=str, help="Optional label with baseline-run.")
    parser.add_argument("--use-correlation", action="store_true", help="Force correlation matrices.")
    parser.add_argument(
        "--use-augmented-cov",
        action="store_true",
        help="If set, compute predicted covariance on [context; pred] instead of pred-only for matched runs.",
    )
    parser.add_argument(
        "--augmented-run-substr",
        action="append",
        help="Apply augmented cov only if model run path contains any of these substrings (overrides global off).",
    )
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--out-png", type=Path, required=True, help="Output PNG path for the trend plot.")
    parser.add_argument("--out-csv", type=Path, help="Optional CSV with per-block metrics.")
    parser.add_argument("--title", type=str, help="Optional plot title.")
    parser.add_argument("--per-baseline", action="store_true", help="Plot all baselines instead of only the best per block.")
    parser.add_argument("--per-block-outputs", action="store_true", help="Also write one PNG (and CSV row slice) per block using the block label as suffix.")
    parser.add_argument("--combined-png", type=Path, help="Optional: render one figure with subplots for all blocks (one block per subplot).")
    parser.add_argument(
        "--baseline-annotate",
        choices=["none", "last"],
        default="none",
        help="Annotate best-baseline points: 'none' (default) or only the last point ('last').",
    )
    return parser.parse_args()


def _higher_is_better(metric_name: str) -> bool:
    return metric_name in {"matrix_corr_offdiag_pearson", "matrix_corr_offdiag_spearman", "matrix_corr_sign_rate"}


def _load_block_payloads(
    outputs_root: Path,
    args: argparse.Namespace,
    block: dict,
) -> Tuple[Dict[Tuple[str, str], List[Dict[str, object]]], List[np.ndarray], List[np.ndarray], List[int], int, int, Dict[Tuple[str, str], bool]]:
    metric_name = block.get("metric_name", args.metric_name)
    matrix_kind = _resolve_matrix_kind(metric_name, block.get("matrix_kind", args.matrix_kind), args.use_correlation)
    use_correlation = matrix_kind == "corr"
    runs_input = block.get("runs", args.runs)
    sample_tags_input = block.get("sample_tags", args.sample_tags)
    stage = block.get("stage", args.stage)
    asset_offset = int(block.get("asset_offset", args.asset_offset))
    batch_aggregate = bool(block.get("batch_aggregate", args.batch_aggregate))

    runs = resolve_runs(outputs_root, runs_input)
    use_aug_flags: Dict[Tuple[str, str], bool] = {}
    sample_tags = sample_tags_input or [None] * len(MODEL_ORDER)
    datasets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for idx, key in enumerate(MODEL_ORDER):
        run_dir = runs.get(key)
        if run_dir is None:
            continue
        use_aug = args.use_augmented_cov or (
            args.augmented_run_substr
            and any(sub in str(run_dir) for sub in args.augmented_run_substr)
        )
        use_aug_flags[key] = use_aug
        tag = sample_tags[idx]
        tag_clean = None if tag in {None, "-", ""} else tag
        if batch_aggregate and tag_clean and "batch-" in tag_clean:
            batch_dir = run_dir / "samples_history" / tag_clean
            subdirs = sorted([p for p in batch_dir.iterdir() if p.is_dir()])
            if not subdirs:
                raise FileNotFoundError(f"No sub-runs under batch directory {batch_dir}")
            payloads = [load_samples(sub, sample_tag=None) for sub in subdirs]
            datasets[key] = payloads
        else:
            datasets[key] = [load_samples(run_dir, sample_tag=tag_clean)]

    if not datasets:
        raise SystemExit("No model datasets loaded.")

    # Reference contexts/truths (first available model)
    ref_payload = None
    for payload_list in datasets.values():
        for payload in payload_list:
            if payload.get("context") is not None:
                ref_payload = payload
                break
        if ref_payload is not None:
            break
    if ref_payload is None:
        raise SystemExit("No model with context available to serve as reference.")

    ref_counts = ref_payload.get("stage_counts", {})
    ref_indices = _stage_indices(ref_counts, stage) if ref_counts else list(range(ref_payload["pred"].shape[0]))
    ref_contexts: List[np.ndarray] = []
    ref_truths: List[np.ndarray] = []
    for i in range(len(ref_indices)):
        idx = ref_indices[i]
        ref_contexts.append(ref_payload["context"][idx])
        ref_truths.append(ref_payload["truth"][idx])

    total_assets = ref_payload["truth"].shape[-1]
    return datasets, ref_contexts, ref_truths, ref_indices, total_assets, asset_offset, use_aug_flags


def _collect_metrics(outputs_root: Path, args: argparse.Namespace, block: dict) -> List[Tuple[int, Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]]]:
    metric_name = block.get("metric_name", args.metric_name)
    stage = block.get("stage", args.stage)
    batch_aggregate = bool(block.get("batch_aggregate", args.batch_aggregate))
    use_correlation = _resolve_matrix_kind(metric_name, block.get("matrix_kind", args.matrix_kind), args.use_correlation) == "corr"

    (
        datasets,
        ref_contexts,
        ref_truths,
        ref_indices,
        total_assets,
        asset_offset,
        use_aug_flags,
    ) = _load_block_payloads(outputs_root, args, block)

    # Determine asset counts to sweep
    asset_counts_raw = block.get("asset_counts")
    assets_field = block.get("assets", args.assets)
    counts: List[int] = []

    def _expand_count(entry) -> List[int]:
        if isinstance(entry, str):
            text = entry.strip()
            if text.lower() in {"all", "max"}:
                return [total_assets - asset_offset]
            for sep in ("..", "-", ":"):
                if sep in text:
                    parts = text.split(sep)
                    if len(parts) == 2 and parts[0] and parts[1]:
                        start, end = int(parts[0]), int(parts[1])
                        step = 1 if end >= start else -1
                        return list(range(start, end + step, step))
            return [int(text)]
        return [int(entry)]

    if asset_counts_raw:
        raw_list = asset_counts_raw if isinstance(asset_counts_raw, (list, tuple)) else [asset_counts_raw]
        for c in raw_list:
            counts.extend(_expand_count(c))
    else:
        counts.extend(_expand_count(assets_field))

    counts = sorted(set(counts))

    results: List[Tuple[int, Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]]] = []
    sample_tags_input = block.get("sample_tags", args.sample_tags) or [None] * len(MODEL_ORDER)
    baseline_input = block.get("baseline", args.baseline)
    baseline_specs: List[Tuple[str, str, str | None]] = []
    if baseline_input:
        for spec in baseline_input:
            parts = spec.strip().split("|")
            if len(parts) < 2:
                raise ValueError(f"--baseline entries must be 'RUN|PREFIX|LABEL'; got: {spec}")
            run = parts[0].strip()
            prefix = parts[1]
            label = parts[2] if len(parts) > 2 else None
            baseline_specs.append((run, prefix, label))
    if args.baseline_run:
        if not args.baseline_prefix:
            raise ValueError("--baseline-prefix must be provided when --baseline-run is used.")
        baseline_specs.append((str(args.baseline_run), args.baseline_prefix, args.baseline_label))

    for assets_count in counts:
        if assets_count + asset_offset > total_assets:
            raise ValueError(f"Requested assets={assets_count} with offset {asset_offset} exceeds available {total_assets}")
        asset_indices = list(range(asset_offset, asset_offset + assets_count))
        metrics_stats: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}

        # Models
        for idx, key in enumerate(MODEL_ORDER):
            payload_list = datasets.get(key)
            if payload_list is None:
                continue
            use_aug = use_aug_flags.get(key, False)
            vals_ctx: List[float] = []
            vals_truth: List[float] = []
            for payload in payload_list:
                _length, _mats, mets = aggregate_model(
                    key,
                    payload,
                    ref_indices,
                    asset_indices,
                    metric_name,
                    stage,
                    use_correlation,
                    ref_contexts=ref_contexts,
                    use_augmented_cov=use_aug,
                )
                vals_ctx.append(mets["pred_minus_context"])
                vals_truth.append(mets["pred_minus_truth"])
            metrics_stats[key] = {
                "pred_minus_context": (float(np.mean(vals_ctx)), float(np.std(vals_ctx) if len(vals_ctx) > 1 else 0.0)),
                "pred_minus_truth": (float(np.mean(vals_truth)), float(np.std(vals_truth) if len(vals_truth) > 1 else 0.0)),
            }

        # Baselines
        for run, prefix, label in baseline_specs:
            run_path = Path(run).resolve()
            runs_to_use: List[Path] = []
            if batch_aggregate:
                subdirs = sorted([p for p in run_path.iterdir() if p.is_dir()])
                runs_to_use = subdirs if subdirs else [run_path]
            else:
                runs_to_use = [run_path]

            vals_ctx: List[float] = []
            vals_truth: List[float] = []
            for run_dir in runs_to_use:
                _length, _mats, mets = aggregate_baseline(
                    run_dir,
                    prefix,
                    ref_contexts,
                    ref_truths,
                    asset_indices,
                    metric_name,
                    stage,
                    use_correlation,
                )
                vals_ctx.append(mets["pred_minus_context"])
                vals_truth.append(mets["pred_minus_truth"])

            display_label = label or prefix.replace("_", " ").title()
            key = ("baseline", display_label)
            metrics_stats[key] = {
                "pred_minus_context": (float(np.mean(vals_ctx)), float(np.std(vals_ctx) if len(vals_ctx) > 1 else 0.0)),
                "pred_minus_truth": (float(np.mean(vals_truth)), float(np.std(vals_truth) if len(vals_truth) > 1 else 0.0)),
            }

        results.append((assets_count, metrics_stats))

    return results


def _select_best_baseline(metrics_stats: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]], metric_field: str, higher_is_better: bool) -> Tuple[str, float] | None:
    baseline_items = [(k, v[metric_field][0]) for k, v in metrics_stats.items() if isinstance(k, tuple) and k[0] == "baseline"]
    if not baseline_items:
        return None
    if higher_is_better:
        best_key, best_val = max(baseline_items, key=lambda kv: kv[1])
    else:
        best_key, best_val = min(baseline_items, key=lambda kv: kv[1])
    return best_key[1], best_val


def _render_plot(
    plot_data: Dict[str, List[Tuple[int, float]]],
    best_baseline_series: List[Tuple[int, float, str]],
    metric_field: str,
    title: str | None,
    out_png: Path,
    dpi: int = 140,
    annotate_mode: str = "none",
    highlight_points: List[Tuple[int, float, str]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    markers = ["o", "s", "D", "^", "v", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]
    for idx, (disp_label, points) in enumerate(plot_data.items()):
        if not points:
            continue
        pts_sorted = sorted(points, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)], label=disp_label)

    if best_baseline_series:
        pts_sorted = sorted(best_baseline_series, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker="x", linestyle="--", color="black", label="Best Baseline (per block)")
        if annotate_mode == "last" and pts_sorted:
            x, y, name = pts_sorted[-1]
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    if highlight_points:
        for x, y, text in highlight_points:
            ax.scatter([x], [y], color="red", marker="*", s=60, zorder=5)
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8, color="red")

    y_label = "Pred vs. Truth (lower better)" if metric_field == "pred_minus_truth" else "Pred vs. Context (lower = closer to context)"
    ax.set_xlabel("Assets in block")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trend plot to {out_png}")


def _render_subplots(
    blocks: List[Tuple[str, Dict[str, List[Tuple[int, float]]], List[Tuple[int, float, str]], List[Tuple[int, float, str]]]],
    metric_field: str,
    annotate_mode: str,
    out_path: Path,
    dpi: int = 140,
    title: str | None = None,
) -> None:
    n = len(blocks)
    cols = min(3, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), dpi=dpi, squeeze=False)
    axes_flat = axes.flat
    markers = ["o", "s", "D", "^", "v", "<", ">"]
    linestyles = ["-", "--", "-.", ":"]
    y_label = "Pred vs. Truth (lower better)" if metric_field == "pred_minus_truth" else "Pred vs. Context (lower = closer to context)"
    for ax_idx, (label, plot_data, best_series, highlights) in enumerate(blocks):
        ax = axes_flat[ax_idx]
        # Plot diffusion variants
        for idx, (disp_label, points) in enumerate(plot_data.items()):
            if not points:
                continue
            pts_sorted = sorted(points, key=lambda x: x[0])
            xs = [p[0] for p in pts_sorted]
            ys = [p[1] for p in pts_sorted]
            ax.plot(xs, ys, marker=markers[idx % len(markers)], linestyle=linestyles[idx % len(linestyles)], label=disp_label)
        # Best baseline
        if best_series:
            pts_sorted = sorted(best_series, key=lambda x: x[0])
            xs = [p[0] for p in pts_sorted]
            ys = [p[1] for p in pts_sorted]
            ax.plot(xs, ys, marker="x", linestyle="--", color="black", label="Best Baseline (per block)")
            if annotate_mode == "last" and pts_sorted:
                x, y, name = pts_sorted[-1]
                ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
        # Highlights
        if highlights:
            for x, y, text in highlights:
                ax.scatter([x], [y], color="red", marker="*", s=60, zorder=5)
                ax.annotate(text, (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8, color="red")

        ax.set_xlabel("Assets in block")
        ax.set_ylabel(y_label)
        ax.set_title(label)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
        ax.legend(fontsize=8)

    # Hide unused axes when grid is larger than number of blocks
    for ax in list(axes_flat)[len(blocks) :]:
        ax.axis("off")

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined subplot PNG to {out_path}")


def main() -> None:
    args = parse_args()
    blocks: List[dict] = []
    if args.blocks_config:
        for cfg_path in args.blocks_config:
            with open(cfg_path, "r") as f:
                blocks.append(yaml.safe_load(f))
    if args.block:
        for inline in args.block:
            blocks.append(yaml.safe_load(inline))
    if not blocks:
        blocks.append({})

    higher_is_better = _higher_is_better(args.metric_name)
    metric_field = args.metric_field

    plot_data: Dict[str, List[Tuple[int, float]]] = {label: [] for label in FOCUS_LABELS.values()}
    best_baseline_series: List[Tuple[int, float, str]] = []
    highlight_points: List[Tuple[int, float, str]] = []
    csv_rows: List[Dict[str, object]] = []
    per_block_payloads: List[Tuple[str, Dict[str, List[Tuple[int, float]]], List[Tuple[int, float, str]], List[Tuple[int, float, str]]]] = []

    for block in blocks:
        label = block.get("label", "block")
        metrics_list = _collect_metrics(args.outputs_root, args, block)

        block_plot_data: Dict[str, List[Tuple[int, float]]] = {lbl: [] for lbl in FOCUS_LABELS.values()}
        block_best_series: List[Tuple[int, float, str]] = []
        block_highlights: List[Tuple[int, float, str]] = []

        for assets, metrics_stats in metrics_list:
            best_baseline = _select_best_baseline(metrics_stats, metric_field, higher_is_better)

            # Diffusion variants
            for key, disp_label in FOCUS_LABELS.items():
                if key not in metrics_stats:
                    continue
                mean_val = metrics_stats[key][metric_field][0]
                plot_data[disp_label].append((assets, mean_val))
                block_plot_data[disp_label].append((assets, mean_val))
                csv_rows.append({"label": label, "assets": assets, "model": disp_label, metric_field: mean_val})
                if best_baseline:
                    csv_rows.append({"label": label, "assets": assets, "model": f"Best baseline ({best_baseline[0]})", metric_field: best_baseline[1]})

            # Baselines
            if best_baseline:
                best_baseline_series.append((assets, best_baseline[1], best_baseline[0]))
                block_best_series.append((assets, best_baseline[1], best_baseline[0]))
                # Highlight where best baseline beats all four diffusion models
                model_vals = [metrics_stats[k][metric_field][0] for k in FOCUS_LABELS if k in metrics_stats]
                if model_vals and best_baseline[1] < min(model_vals):
                    txt = f"{best_baseline[0]}<{min(model_vals):.2f}"
                    highlight_points.append((assets, best_baseline[1], txt))
                    block_highlights.append((assets, best_baseline[1], txt))

            if args.per_baseline:
                for key, stats in metrics_stats.items():
                    if isinstance(key, tuple) and key[0] == "baseline":
                        mean_val = stats[metric_field][0]
                        lbl = f"Baseline: {key[1]}"
                        plot_data.setdefault(lbl, []).append((assets, mean_val))
                        block_plot_data.setdefault(lbl, []).append((assets, mean_val))
                        csv_rows.append({"label": label, "assets": assets, "model": lbl, metric_field: mean_val})

        per_block_payloads.append((label, block_plot_data, block_best_series, block_highlights))

    # Combined plot
    _render_plot(plot_data, best_baseline_series, metric_field, args.title, args.out_png, dpi=args.dpi, annotate_mode=args.baseline_annotate, highlight_points=highlight_points)

    # Per-block plots (optional)
    if args.per_block_outputs and len(per_block_payloads) > 1:
        base = args.out_png
        stem = base.stem
        suffix = base.suffix or ".png"
        for label, block_plot_data, block_best_series, block_highlights in per_block_payloads:
            blk_path = base.with_name(f"{stem}_{label}{suffix}")
            _render_plot(block_plot_data, block_best_series, metric_field, label if not args.title else f"{args.title} - {label}", blk_path, dpi=args.dpi, annotate_mode=args.baseline_annotate, highlight_points=block_highlights)

    # Combined subplots
    if args.combined_png and per_block_payloads:
        _render_subplots(per_block_payloads, metric_field, args.baseline_annotate, args.combined_png, dpi=args.dpi, title=args.title)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(args.out_csv, index=False)
        print(f"Saved CSV to {args.out_csv}")
        if args.per_block_outputs and len(per_block_payloads) > 1:
            base = args.out_csv
            stem = base.stem
            suffix = base.suffix or ".csv"
            df_all = pd.DataFrame(csv_rows)
            for label in df_all["label"].unique():
                blk = df_all[df_all["label"] == label]
                blk_path = base.with_name(f"{stem}_{label}{suffix}")
                blk.to_csv(blk_path, index=False)
                print(f"Saved per-block CSV to {blk_path}")


if __name__ == "__main__":
    main()
