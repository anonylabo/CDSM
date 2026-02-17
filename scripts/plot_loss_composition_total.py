#!/usr/bin/env python3
"""Plot total loss composition (score + weighted auxiliary penalties).

This script reads training logs under one or more base directories and
produces a stacked-bar plot of total loss composition by dataset.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

DEFAULT_BASES = [
    Path(__file__).resolve().parents[1] / "outputs/time/conditional",
    Path(__file__).resolve().parents[1] / "outputs/fourier/conditional",
]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "assets/loss_composition_by_dataset.pdf"

METRIC_COLS = {
    "loss": "train/loss_epoch",
    "mean": "train/mean_penalty_epoch",
    "cov": "train/cov_penalty_epoch",
    "corr": "train/corr_penalty_epoch",
    "spec": "train/spectral_penalty_epoch",
    "slide": "train/sliding_cov_penalty_epoch",
}

LAMBDA_KEYS = {
    "mean": "lambda_mean",
    "cov": "lambda_cov",
    "corr": "lambda_corr",
    "spec": "lambda_spectral",
    "slide": "lambda_sliding_cov",
}

LABEL_MAP = {
    "score": r"$\mathcal{L}_{\mathrm{score}}$",
    "mean": r"$\lambda_{\mathrm{mean}}\,\mathcal{L}_{\mathrm{mean}}$",
    "cov": r"$\lambda_{\mathrm{cov}}\,\mathcal{L}_{\mathrm{cov}}$",
    "corr": r"$\lambda_{\mathrm{corr}}\,\mathcal{L}_{\mathrm{corr}}$",
    "spec": r"$\lambda_{\mathrm{spec}}\,\mathcal{L}_{\mathrm{spec}}$",
    "slide": r"$\lambda_{\mathrm{slide}}\,\mathcal{L}_{\mathrm{slide}}$",
}

ORDER = ["score", "mean", "cov", "corr", "spec", "slide"]
DATASET_ORDER = {
    "FX8": 0,
    "FX30": 1,
    "ind49": 2,
    "ishares14": 3,
}


def _format_sci(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    text = f"{value:.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e+").replace("e+", "e")


def _format_lambda_tuple(mean: float, cov: float, corr: float, spec: float, slide: float) -> str:
    vals = [mean, cov, corr, spec]
    if max(vals) - min(vals) < 1e-12 and abs(slide) < 1e-12:
        return rf"$\lambda={_format_sci(mean)}$"
    parts = [
        rf"$\lambda_{{\mathrm{{mean}}}}={_format_sci(mean)}$",
        rf"$\lambda_{{\mathrm{{cov}}}}={_format_sci(cov)}$",
        rf"$\lambda_{{\mathrm{{corr}}}}={_format_sci(corr)}$",
        rf"$\lambda_{{\mathrm{{spec}}}}={_format_sci(spec)}$",
    ]
    if abs(slide) >= 1e-12:
        parts.append(rf"$\lambda_{{\mathrm{{slide}}}}={_format_sci(slide)}$")
    return ", ".join(parts)


def _lambda_annotation(df: pd.DataFrame) -> str | None:
    cols = ["lambda_mean", "lambda_cov", "lambda_corr", "lambda_spec", "lambda_slide"]
    if not all(c in df.columns for c in cols):
        return None
    unique = df[cols].drop_duplicates()
    labels = [
        _format_lambda_tuple(*row)
        for row in unique.itertuples(index=False, name=None)
    ]
    if not labels:
        return None
    if len(labels) == 1:
        return labels[0]
    if len(labels) <= 3:
        return " / ".join(labels)
    return f"{len(labels)} lambda settings"


def _format_panel_label(base_dir: Path) -> str:
    parts = [p.lower() for p in base_dir.parts]
    if "time" in parts:
        return "CSMD (Temporal)"
    if "fourier" in parts:
        return "CSMD (Spectral)"
    return base_dir.name


def _parse_run_descriptor(run_name: str) -> tuple[str, int, int] | None:
    parts = run_name.split("_")
    if len(parts) < 2:
        return None
    tokens = parts[1:]
    for i, token in enumerate(tokens):
        if token.isdigit():
            if i + 1 >= len(tokens):
                return None
            match = re.match(r"(\d+)", tokens[i + 1])
            if not match:
                return None
            dataset = "_".join(tokens[:i])
            if not dataset:
                return None
            c_len = int(token)
            p_len = int(match.group(1))
            return dataset, c_len, p_len
    return None


def read_lambdas(run_dir: Path) -> dict[str, float]:
    cfg = run_dir / "train_config.yaml"
    lambdas = {k: 0.0 for k in LAMBDA_KEYS}
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text())
        model_cfg = data.get("model", {}) if isinstance(data, dict) else {}
        for key, cfg_key in LAMBDA_KEYS.items():
            try:
                lambdas[key] = float(model_cfg.get(cfg_key, 0.0))
            except Exception:
                lambdas[key] = 0.0
    return lambdas


def collect_runs(base_dir: Path, exclude: set[str]) -> pd.DataFrame:
    records = []
    for run_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        metrics_path = run_dir / "lightning_logs/version_0/metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        if METRIC_COLS["loss"] not in df.columns:
            continue
        available = [c for c in METRIC_COLS.values() if c in df.columns]
        sub = df[["epoch"] + available].dropna()
        if sub.empty:
            continue
        best_idx = sub[METRIC_COLS["loss"]].idxmin()
        best_row = sub.loc[best_idx]
        total_loss = float(best_row[METRIC_COLS["loss"]])

        penalties = {}
        for key, col in METRIC_COLS.items():
            if key == "loss":
                continue
            penalties[key] = float(best_row[col]) if col in best_row else 0.0

        lambdas = read_lambdas(run_dir)
        weighted = {k: lambdas.get(k, 0.0) * penalties.get(k, 0.0) for k in lambdas}
        score = total_loss - sum(weighted.values())
        if score < 0:
            score = 0.0

        descriptor = _parse_run_descriptor(run_dir.name)
        if descriptor is None:
            continue
        dataset_raw, c_len, p_len = descriptor
        dataset_base = dataset_raw.replace("_ecb", "")
        if dataset_base in exclude:
            continue
        dataset_label = f"{dataset_base} ({c_len},{p_len})"
        dataset_order = DATASET_ORDER.get(dataset_base, 99)

        records.append({
            "run": run_dir.name,
            "dataset": dataset_label,
            "score": score,
            "mean": weighted.get("mean", 0.0),
            "cov": weighted.get("cov", 0.0),
            "corr": weighted.get("corr", 0.0),
            "spec": weighted.get("spec", 0.0),
            "slide": weighted.get("slide", 0.0),
            "lambda_mean": lambdas.get("mean", 0.0),
            "lambda_cov": lambdas.get("cov", 0.0),
            "lambda_corr": lambdas.get("corr", 0.0),
            "lambda_spec": lambdas.get("spec", 0.0),
            "lambda_slide": lambdas.get("slide", 0.0),
            "dataset_order": dataset_order,
            "context_len": c_len,
            "pred_len": p_len,
        })
    return pd.DataFrame(records)


def plot_panels(panels: dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.rcParams["pdf.fonttype"] = 42

    max_bars = max((len(df["dataset"].unique()) for df in panels.values()), default=6)
    fig_width = max(6.5, 1.1 * max_bars) * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 4.5), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None

    for ax, (label, df) in zip(axes, panels.items()):
        grouped = df.groupby("dataset")[ORDER].mean()
        if "dataset_order" in df.columns:
            order_info = (
                df.groupby("dataset")[["dataset_order", "context_len", "pred_len"]]
                .first()
                .sort_values(["dataset_order", "context_len", "pred_len"])
            )
            grouped = grouped.loc[order_info.index]
        nonzero_cols = [c for c in ORDER if grouped[c].abs().sum() > 0]
        grouped = grouped[nonzero_cols].rename(columns=LABEL_MAP)

        frac = grouped.clip(lower=0)
        frac = frac.div(frac.sum(axis=1).replace(0, 1), axis=0)

        bars = frac.plot(kind="bar", stacked=True, colormap="tab20c", ax=ax, legend=False)
        annotation = _lambda_annotation(df)
        title = label if not annotation else f"{label}\n{annotation}"
        bars.set_title(title)
        bars.set_ylabel("Loss fraction")
        bars.set_ylim(0, 1)
        bars.set_xticklabels(grouped.index, rotation=15, ha="right")

        for container in bars.containers:
            labels = [f"{100 * val:.0f}%" if val > 0.08 else "" for val in container.datavalues]
            bars.bar_label(container, labels=labels, label_type="center", fontsize=8, color="white")

        if legend_handles is None:
            legend_handles, legend_labels = bars.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
        )

    fig.suptitle("Total loss composition by dataset (mean over runs)")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot total loss composition by dataset.")
    parser.add_argument(
        "base_dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_BASES],
        help="Root directories containing run subfolders.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output PDF path (default: assets/loss_composition_by_dataset.pdf).",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated dataset names to exclude (default: none).",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_value",
        type=float,
        default=None,
        help="Filter runs where lambda_mean/cov/corr/spec equal this value and lambda_slide=0.",
    )
    args = parser.parse_args()

    exclude = {d.strip() for d in args.exclude.split(",") if d.strip()}

    panels = {}
    for base in args.base_dirs:
        base_dir = Path(base).resolve()
        df = collect_runs(base_dir, exclude)
        if args.lambda_value is not None and not df.empty:
            target = float(args.lambda_value)
            tol = 1e-12
            df = df[
                ((df["lambda_mean"] - target).abs() <= tol)
                & ((df["lambda_cov"] - target).abs() <= tol)
                & ((df["lambda_corr"] - target).abs() <= tol)
                & ((df["lambda_spec"] - target).abs() <= tol)
                & (df["lambda_slide"].abs() <= tol)
            ]
        if df.empty:
            continue
        label = _format_panel_label(base_dir)
        panels[label] = df

    if not panels:
        raise SystemExit("No runs found.")

    plot_panels(panels, Path(args.out).resolve())


if __name__ == "__main__":
    main()
