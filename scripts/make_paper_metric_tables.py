#!/usr/bin/env python3
"""
Summarise best series / structural metrics for the four diffusion configurations
into a compact table for the paper, optionally including a DeepVAR baseline.

Typical usage (from repo root):

  cd unconditional-time-series-diffusion/external/conditional_fourier_diffusion
  python scripts/make_paper_metric_tables.py \
    --metrics series_crps series_nd series_rmse \
    --table-tex assets/table_series.tex

By default the script locates the latest run under:
  outputs/time/conditional
  outputs/time/unconditional
  outputs/fourier/conditional
  outputs/fourier/unconditional
You can override any of these with explicit paths.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml
import torch


DIFFUSION_RUNS: Dict[str, Tuple[str, str]] = {
    "Time (Conditional)": ("time", "conditional"),
    "Time (Unconditional)": ("time", "unconditional"),
    "Fourier (Conditional)": ("fourier", "conditional"),
    "Fourier (Unconditional)": ("fourier", "unconditional"),
}


def latest_run_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(candidates, key=lambda p: p.name)


def read_best_metrics(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "best_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing best_metrics.json in {run_dir}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    metrics: Dict[str, float] = {}
    for entry in data.get("best_checkpoints", []):
        name = str(entry.get("metric"))
        score = entry.get("best_score")
        if name is None or score is None:
            continue
        metrics[name] = float(score)
    return metrics


def metric_label(name: str) -> str:
    mapping = {
        "series_crps": "CRPS",
        "series_nd": "ND",
        "series_rmse": "RMSE",
    }
    return mapping.get(name, name)


def lookup_metric(metrics_map: Dict[str, float], name: str) -> float:
    """
    Fetch a metric, falling back to MC variants when available.
    Currently supports mapping series_crps <- series_crps_mc if the former
    is absent in metrics_extra.json.
    """
    if name in metrics_map:
        return metrics_map[name]
    if name == "series_crps" and "series_crps_mc" in metrics_map:
        return metrics_map["series_crps_mc"]
    return float("nan")


def lookup_metric_stat(metrics_map: Dict[str, Tuple[float, float]] | Dict[str, float], name: str) -> Tuple[float, float]:
    """
    Return (mean, std) for the requested metric, handling CRPS fallback.
    Accepts maps of floats or (mean, std) tuples.
    """
    def _coerce(val) -> Tuple[float, float]:
        if isinstance(val, tuple) or isinstance(val, list):
            if len(val) >= 2:
                return float(val[0]), float(val[1])
            if len(val) == 1:
                return float(val[0]), 0.0
        return float(val), 0.0

    if name in metrics_map:
        return _coerce(metrics_map[name])
    if name == "series_crps" and "series_crps_mc" in metrics_map:
        return _coerce(metrics_map["series_crps_mc"])
    return float("nan"), float("nan")


def combine_metric_maps(metric_maps: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate a list of metric maps into mean/std tuples.
    """
    agg: Dict[str, List[float]] = {}
    for m in metric_maps:
        for k, v in m.items():
            agg.setdefault(k, []).append(float(v))
    return {k: (float(np.mean(vals)), float(np.std(vals) if len(vals) > 1 else 0.0)) for k, vals in agg.items()}


def read_latest_samples_metrics(run_dir: Path, stage: str | None = None) -> Dict[str, float]:
    """
    Load series/matrix metrics from the latest samples_history/<timestamp>/metrics_extra*.json.
    These come from the sampling/evaluation phase (val+test combined by default).
    """
    hist_root = run_dir / "samples_history"
    if not hist_root.exists():
        raise FileNotFoundError(f"No samples_history found under {run_dir}")
    candidates = [p for p in hist_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No samples_history entries found under {hist_root}")
    latest = max(candidates, key=lambda p: p.name)
    metrics_path = latest / ("metrics_extra.json" if stage is None else f"metrics_extra_{stage}.json")
    fallback_path = latest / "metrics_extra.json"
    if stage is not None and not metrics_path.exists() and fallback_path.exists():
        metrics_path = fallback_path
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path.name} not found in {latest}")
    with metrics_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, float] = {}
    for k, v in data.items():
        if not k.startswith("series_"):
            continue
        out[k] = float(v)
    return out


def read_samples_metrics_at(path: Path, stage: str | None = None) -> Dict[str, float]:
    """
    Load series_* metrics from a specific samples_history/<stamp> directory.
    If stage is provided, looks for metrics_extra_{stage}.json with fallback to metrics_extra.json.
    """
    metrics_path = path / ("metrics_extra.json" if stage is None else f"metrics_extra_{stage}.json")
    fallback_path = path / "metrics_extra.json"
    if stage is not None and not metrics_path.exists() and fallback_path.exists():
        metrics_path = fallback_path
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path.name} not found in {path}")
    with metrics_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, float] = {}
    for k, v in data.items():
        if not k.startswith("series_"):
            continue
        out[k] = float(v)
    return out


def _series_from_samples(path: Path, asset_offset: int, assets: int | None) -> Dict[str, float]:
    """
    Recompute series_* metrics (rmse, nd) from samples.pt/truth with optional asset slicing.
    CRPS is left as NaN since ensemble samples are unavailable here.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    preds = data["samples"]  # (N, P, A)
    truth = data["truth"]
    if assets is not None:
        preds = preds[..., asset_offset : asset_offset + assets]
        truth = truth[..., asset_offset : asset_offset + assets]
    diff = preds - truth
    rmse = torch.sqrt(torch.mean(diff ** 2)).item()
    nd = (torch.sum(diff.abs()) / (torch.sum(truth.abs()) + 1e-8)).item()
    return {
        "series_rmse": float(rmse),
        "series_nd": float(nd),
        "series_crps": float("nan"),
    }


def read_deepvar_series_metrics(path_or_root: Path, stage: str | None = None) -> Dict[str, float]:
    """
    Locate DeepVAR aggregate metrics. Accepts either:
      - a direct deepvar_aggregate_metrics*.json file,
      - a run/batch directory containing such files,
      - or a root directory; falls back to searching under baselines/deepvar/**.
    """
    candidates = []
    if path_or_root.is_file():
        candidates = [path_or_root]
    elif path_or_root.is_dir():
        # First, search under the provided directory
        candidates = sorted(path_or_root.rglob("deepvar_aggregate_metrics*.json"))
        # If nothing found, try the standard baselines/deepvar layout
        if not candidates:
            base = path_or_root / "baselines" / "deepvar"
            if base.exists():
                candidates = sorted(base.rglob("deepvar_aggregate_metrics*.json"))
    if not candidates:
        raise FileNotFoundError(f"No deepvar_aggregate_metrics.json found under {path_or_root}.")
    if stage:
        stage_cands = [p for p in candidates if p.name.endswith(f"metrics_{stage}.json")]
        latest = stage_cands[-1] if stage_cands else candidates[-1]
    else:
        latest = candidates[-1]
    with latest.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    series = data.get("series", {})
    out: Dict[str, float] = {}
    rmse = series.get("rmse")
    if rmse is not None:
        out["series_rmse"] = float(rmse)
    nd = series.get("nd")
    if nd is not None:
        out["series_nd"] = float(nd)
    crps = series.get("crps")
    if crps is not None:
        out["series_crps"] = float(crps)
    return out


def read_mgtsd_series_metrics(root: Path) -> Dict[str, float]:
    """
    Locate an MG-TSD baseline metrics_extra.json and return series_* entries.
    - If root is a file, read it directly.
    - Else if root/metrics_extra.json exists, use that.
    - Else search recursively for metrics_extra.json and pick the latest by mtime.
    """
    path = root
    if path.is_file():
        metrics_path = path
    else:
        candidate = path / "metrics_extra.json"
        if candidate.exists():
            metrics_path = candidate
        else:
            candidates = sorted(path.rglob("metrics_extra.json"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"No metrics_extra.json found under {root}")
            metrics_path = candidates[-1]

    with metrics_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, float] = {}
    for k, v in data.items():
        if k.startswith("series_"):
            out[k] = float(v)
    return out


def _markdown_from_df(df: pd.DataFrame, caption: str | None = None) -> str:
    lines: List[str] = []
    if caption:
        lines.append(caption)
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _latex_blocks(blocks: List[tuple[str, pd.DataFrame, pd.DataFrame]], float_fmt: str = ".4f", caption: str | None = None) -> str:
    def _escape(text: str) -> str:
        return text if "$" in text else text.replace("_", r"\_")

    def _fmt(value, bold: bool) -> str:
        if isinstance(value, str) and "±" in value:
            left, right = value.split("±", 1)
            if bold:
                return f"$\\mathbf{{{left}}}\\pm\\mathbf{{{right}}}$"
            return f"${left}\\pm{right}$"
        if isinstance(value, float):
            text = format(value, float_fmt)
            return r"\textbf{" + text + "}" if bold else text
        if isinstance(value, str):
            text = _escape(value)
            return r"\textbf{" + text + "}" if bold else text
        return r"\textbf{" + str(value) + "}" if bold else str(value)

    if not blocks:
        return ""
    headers = list(blocks[0][1].columns)
    col_spec = "l" + "c" * (len(headers) - 1)
    lines: List[str] = []
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    if caption:
        lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{" + _escape(caption) + r"} \\")
    lines.append(r"\toprule")
    lines.append(" & ".join(_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for blk_idx, (label, df, mask) in enumerate(blocks):
        lines.append(r"\multicolumn{" + str(len(headers)) + r"}{l}{\textbf{" + _escape(label) + r"}} \\")
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
                cells.append(_fmt(val, bold))
            lines.append(" & ".join(cells) + r" \\")
        if blk_idx != len(blocks) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def _markdown_blocks(blocks: List[tuple[str, pd.DataFrame]], caption: str | None = None) -> str:
    if not blocks:
        return ""
    headers = list(blocks[0][1].columns)
    lines: List[str] = []
    if caption:
        lines.append(caption)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for label, df in blocks:
        lines.append("| **" + label + "** | " + " | ".join([""] * (len(headers) - 1)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarise best_metrics.json into a paper-ready LaTeX table.")
    p.add_argument("--blocks-config", nargs="+", help="YAML files describing multiple blocks to stack vertically.")
    p.add_argument("--block", action="append", help="Inline YAML/JSON for one block (label, overrides, samples_history, metrics, etc.).")
    p.add_argument("--per-block-outputs", action="store_true", help="If set, also emit per-block LaTeX/Markdown next to the combined output.")
    p.add_argument("--outputs-root", type=Path, default=Path("outputs"), help="Root outputs directory.")
    p.add_argument("--time-conditional", type=Path, help="Override run dir for time/conditional.")
    p.add_argument("--time-unconditional", type=Path, help="Override run dir for time/unconditional.")
    p.add_argument("--fourier-conditional", type=Path, help="Override run dir for fourier/conditional.")
    p.add_argument("--fourier-unconditional", type=Path, help="Override run dir for fourier/unconditional.")
    p.add_argument("--mgtsd", type=Path, help="Optional MG-TSD baseline run directory (will read metrics_extra.json).")
    p.add_argument("--samples-history", nargs="*", help="Optional mapping of label=path_to_samples_history_entry (e.g., \"Time (Conditional)=outputs/.../samples_history/20251116-193157-best-matrix_cov_fro\").")
    p.add_argument("--batch-aggregate", action="store_true", help="If set, treat samples-history paths or latest samples_history/batch-* as batches and report mean±std across sub-runs.")
    p.add_argument(
        "--metrics",
        nargs="+",
        default=["series_crps", "series_nd", "series_rmse"],
        help="Metric names to extract from best_metrics.json (default: series_crps series_nd series_rmse).",
    )
    p.add_argument(
        "--stage-splits",
        action="store_true",
        help="If set, report metrics separately for val/test/all when available (metrics_extra_val/test).",
    )
    p.add_argument(
        "--table-tex",
        type=Path,
        default=Path("assets/table_series.tex"),
        help="Destination for LaTeX tabular snippet.",
    )
    p.add_argument("--out-md", type=Path, help="Optional Markdown table output.")
    p.add_argument(
        "--use-samples-history",
        action="store_true",
        help="If set, read series_* metrics from the latest samples_history/metrics_extra.json instead of best_metrics.json.",
    )
    p.add_argument(
        "--include-deepvar",
        action="store_true",
        help="If set, append a DeepVAR baseline row using deepvar_aggregate_metrics.json under outputs/baselines/deepvar.",
    )
    p.add_argument(
        "--deepvar-path",
        type=Path,
        help="Optional path to a DeepVAR run/batch or aggregate metrics file. "
        "If omitted, defaults to searching under outputs/baselines/deepvar.",
    )
    p.add_argument("--asset-offset", type=int, default=0, help="Optional asset index offset when recomputing series_* from samples.pt.")
    p.add_argument("--assets", type=int, default=None, help="Optional number of assets to include when recomputing series_* from samples.pt.")
    return p.parse_args()


def _gather_block_rows(cfg: dict, base_args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Build a DataFrame of formatted values and a best-mask for one block.
    """
    outputs_root = cfg.get("outputs_root", base_args.outputs_root)
    metrics = cfg.get("metrics", base_args.metrics)
    stage_splits = bool(cfg.get("stage_splits", base_args.stage_splits))
    use_samples_history = bool(cfg.get("use_samples_history", base_args.use_samples_history))
    batch_aggregate = bool(cfg.get("batch_aggregate", base_args.batch_aggregate))
    include_deepvar = bool(cfg.get("include_deepvar", base_args.include_deepvar))
    asset_offset = int(cfg.get("asset_offset", base_args.asset_offset))
    assets = cfg.get("assets", base_args.assets)
    assets = int(assets) if assets is not None else None

    override_map: Dict[Tuple[str, str], Path] = {}
    for key in ["time_conditional", "time_unconditional", "fourier_conditional", "fourier_unconditional"]:
        if cfg.get(key) is not None:
            override_map[(key.split("_")[0], key.split("_")[1])] = Path(cfg[key]).expanduser().resolve()
        elif getattr(base_args, key):
            override_map[(key.split("_")[0], key.split("_")[1])] = Path(getattr(base_args, key)).expanduser().resolve()

    samples_override: Dict[str, Path] = {}
    samples_items = cfg.get("samples_history", base_args.samples_history)
    if samples_items:
        for item in samples_items:
            if "=" not in item:
                continue
            label, path_str = item.split("=", 1)
            samples_override[label.strip()] = Path(path_str.strip()).expanduser().resolve()

    stage_list = ["val", "test", "all"] if stage_splits else [None]
    header_metrics: List[str] = []
    for m in metrics:
        if stage_splits:
            for st in stage_list:
                suffix = {"val": " (val)", "test": " (test)", "all": " (all)", None: ""}.get(st, f" ({st})")
                header_metrics.append(f"{metric_label(m)}{suffix}")
        else:
            header_metrics.append(metric_label(m))

    rows: List[Tuple[str, List[Tuple[float, float]]]] = []
    for label, (domain, mode) in DIFFUSION_RUNS.items():
        base = override_map.get((domain, mode))
        run_dir = Path(base).expanduser().resolve() if base is not None else latest_run_dir(Path(outputs_root) / domain / mode)
        values: List[Tuple[float, float]] = []
        for name in metrics:
            for stage in stage_list:
                stage_key = None if stage is None or stage == "all" else stage
                if use_samples_history:
                    metric_maps: List[Dict[str, float]] = []
                    if label in samples_override:
                        try:
                            p = samples_override[label]
                            if batch_aggregate and p.name.startswith("batch-"):
                                subdirs = sorted([d for d in p.iterdir() if d.is_dir()])
                                if not subdirs:
                                    raise FileNotFoundError(f"No sub-runs under batch directory {p}")
                                for sub in subdirs:
                                    metric_maps.append(read_samples_metrics_at(sub, stage=stage_key))
                                    # Recompute slice metrics if requested
                                    samples_path = sub / "samples.pt"
                                    if samples_path.exists() and assets is not None:
                                        metric_maps[-1].update(_series_from_samples(samples_path, asset_offset, assets))
                            else:
                                metric_maps.append(read_samples_metrics_at(p, stage=stage_key))
                                samples_path = p / "samples.pt"
                                if samples_path.exists() and assets is not None:
                                    metric_maps[-1].update(_series_from_samples(samples_path, asset_offset, assets))
                        except FileNotFoundError as exc:
                            print(f"[WARN] {label} ({stage or 'all'}): {exc}; falling back to best_metrics.json")
                            metrics_map_best = read_best_metrics(run_dir)
                            metric_maps.append(metrics_map_best)
                    else:
                        try:
                            hist_root = run_dir / "samples_history"
                            if batch_aggregate and hist_root.exists():
                                batch_dirs = sorted([d for d in hist_root.iterdir() if d.is_dir() and d.name.startswith("batch-")])
                                if batch_dirs:
                                    latest_batch = batch_dirs[-1]
                                    subdirs = sorted([d for d in latest_batch.iterdir() if d.is_dir()])
                                    if not subdirs:
                                        raise FileNotFoundError(f"No sub-runs under {latest_batch}")
                                    for sub in subdirs:
                                        metric_maps.append(read_samples_metrics_at(sub, stage=stage_key))
                                        samples_path = sub / "samples.pt"
                                        if samples_path.exists() and assets is not None:
                                            metric_maps[-1].update(_series_from_samples(samples_path, asset_offset, assets))
                                else:
                                    metric_maps.append(read_latest_samples_metrics(run_dir, stage=stage_key))
                            else:
                                metric_maps.append(read_latest_samples_metrics(run_dir, stage=stage_key))
                            # Try to attach sliced metrics from latest samples.pt when available
                            hist_dirs = []
                            hist_root = run_dir / "samples_history"
                            if hist_root.exists():
                                hist_dirs = [d for d in hist_root.iterdir() if d.is_dir()]
                            if hist_dirs:
                                latest_hist = max(hist_dirs, key=lambda p: p.name)
                                samples_path = latest_hist / "samples.pt"
                                if samples_path.exists() and assets is not None and metric_maps:
                                    metric_maps[-1].update(_series_from_samples(samples_path, asset_offset, assets))
                        except FileNotFoundError as exc:
                            print(f"[WARN] {label} ({stage or 'all'}): {exc}; falling back to best_metrics.json")
                            metric_maps.append(read_best_metrics(run_dir))
                    combined = combine_metric_maps(metric_maps)
                else:
                    best_map = read_best_metrics(run_dir)
                    combined = {k: (float(v), 0.0) for k, v in best_map.items()}
                values.append(lookup_metric_stat(combined, name))
        rows.append((label, values))

    if cfg.get("mgtsd", base_args.mgtsd):
        try:
            mgtsd_metrics = read_mgtsd_series_metrics(cfg.get("mgtsd", base_args.mgtsd))
        except FileNotFoundError as exc:
            print(f"[WARN] Skipping MG-TSD baseline: {exc}")
        else:
            values: List[Tuple[float, float]] = []
            if stage_splits:
                for _ in stage_list:
                    for name in metrics:
                        values.append((lookup_metric(mgtsd_metrics, name), 0.0))
            else:
                for name in metrics:
                    values.append((lookup_metric(mgtsd_metrics, name), 0.0))
            rows.append(("MG-TSD (Multi)", values))

    if include_deepvar:
        dv_path = cfg.get("deepvar_path", base_args.deepvar_path)
        dv_root = Path(dv_path).expanduser().resolve() if dv_path else outputs_root
        deepvar_map = {}
        dv_error = None
        try:
            deepvar_map = read_deepvar_series_metrics(dv_root)
        except FileNotFoundError as exc:
            dv_error = exc
            print(f"[WARN] DeepVAR metrics not found: {exc}; filling NaN.")
        values: List[Tuple[float, float]] = []
        if stage_splits:
            for name in metrics:
                for stage in stage_list:
                    dv_map = {}
                    try:
                        dv_map = read_deepvar_series_metrics(
                            dv_root, stage=None if stage is None or stage == "all" else stage
                        )
                    except FileNotFoundError:
                        pass
                    values.append((lookup_metric(dv_map or deepvar_map, name), 0.0))
        else:
            for name in metrics:
                values.append((lookup_metric(deepvar_map, name), 0.0))
        rows.append(("DeepVAR", values))

    # Build DataFrame and best-mask
    cols = ["Model"] + header_metrics
    df_rows = []
    mean_matrix = []
    for label, values in rows:
        row = {"Model": label}
        means_only: List[float] = []
        for col_name, (mean_v, std_v) in zip(header_metrics, values):
            row[col_name] = f"{mean_v:.4f}±{std_v:.4f}"
            means_only.append(mean_v)
        df_rows.append(row)
        mean_matrix.append(means_only)
    df = pd.DataFrame(df_rows, columns=cols)
    mean_arr = np.array(mean_matrix)
    best_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if mean_arr.size:
        for j, col in enumerate(header_metrics):
            col_vals = mean_arr[:, j]
            if np.isnan(col_vals).all():
                continue
            best_val = np.nanmin(col_vals)
            best_mask.loc[col_vals == best_val, col] = True
    return df, best_mask, header_metrics


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

    # Default single-block path
    if not blocks:
        blocks = [{}]

    # Ensure metric layout consistent across blocks
    block_tables: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []
    labels: List[str] = []
    per_blocks: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []
    header_metrics_ref: List[str] | None = None
    for blk in blocks:
        df_blk, mask_blk, header_metrics = _gather_block_rows(blk, args)
        if header_metrics_ref is None:
            header_metrics_ref = header_metrics
        elif header_metrics_ref != header_metrics:
            raise ValueError("All blocks must use the same metrics/stage-splits for vertical concatenation.")
        label = blk.get("label", "block")
        labels.append(label)
        block_tables.append((label, df_blk.copy(), mask_blk.copy()))
        per_blocks.append((label, df_blk.copy(), mask_blk.copy()))

    caption = "; ".join(labels) if len(labels) > 1 else labels[0]

    latex_text = _latex_blocks(block_tables, caption=caption)
    args.table_tex.parent.mkdir(parents=True, exist_ok=True)
    args.table_tex.write_text(latex_text)
    print(f"Wrote LaTeX table to {args.table_tex}")
    if args.per_block_outputs and len(per_blocks) > 1:
        base = args.table_tex
        stem, suffix = base.stem, base.suffix or ".tex"
        for label, df_blk, mask_blk in per_blocks:
            blk_path = base.with_name(f"{stem}_{label}{suffix}")
            blk_path.write_text(_latex_blocks([(label, df_blk, mask_blk)], caption=label))
            print(f"Wrote per-block LaTeX to {blk_path}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_markdown_blocks([(lbl, df) for lbl, df, _m in block_tables], caption=caption))
        print(f"Wrote Markdown table to {args.out_md}")
        if args.per_block_outputs and len(per_blocks) > 1:
            base = args.out_md
            stem, suffix = base.stem, base.suffix or ".md"
            for label, df_blk, _mask_blk in per_blocks:
                blk_path = base.with_name(f"{stem}_{label}{suffix}")
                blk_path.write_text(_markdown_blocks([(label, df_blk)], caption=label))
                print(f"Wrote per-block Markdown to {blk_path}")

    # stdout summary
    print(_markdown_blocks([(lbl, df) for lbl, df, _m in block_tables], caption=caption))


if __name__ == "__main__":
    main()
