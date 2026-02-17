#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ORDER = (
    ("time", "conditional"),
    ("time", "unconditional"),
    ("fourier", "conditional"),
    ("fourier", "unconditional"),
)


def latest_run(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def resolve_runs(outputs_root: Path, overrides: Dict[str, Optional[Path]]) -> Dict[Tuple[str, str], Path]:
    resolved: Dict[Tuple[str, str], Path] = {}
    for domain, variant in ORDER:
        key = f"{domain}/{variant}"
        if overrides.get(key):
            run_dir = Path(overrides[key]).expanduser().resolve()
        else:
            run_dir = latest_run(outputs_root / domain / variant)
        if not run_dir or not (run_dir / "best_metrics.json").exists():
            raise FileNotFoundError(f"Missing best_metrics.json under {outputs_root / domain / variant}")
        resolved[(domain, variant)] = run_dir
    return resolved


def read_best_metrics(run_dir: Path) -> Dict[str, Tuple[float, int]]:
    path = run_dir / "best_metrics.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Tuple[float, int]] = {}
    for entry in data.get("best_checkpoints", []):
        metric = str(entry.get("metric"))
        score = float(entry.get("best_score"))
        epoch = int(entry.get("best_epoch", 0))
        out[metric] = (score, epoch)
    return out


def default_metric_list() -> List[str]:
    return [
        "matrix_cov_fro",
        "matrix_cov_fro_rel",
        "matrix_cov_mse",
        "matrix_cov_mae",
        "matrix_cov_diag_mape",
        "matrix_corr_fro",
        "matrix_corr_fro_rel",
        "matrix_corr_cross_mse",
        "matrix_corr_offdiag_pearson",
        "matrix_corr_offdiag_spearman",
        "matrix_corr_sign_rate",
        "series_mae",
        "series_mse",
        "series_rmse",
    ]


def metric_direction(metric: str) -> str:
    maximize = {
        "matrix_corr_offdiag_pearson",
        "matrix_corr_offdiag_spearman",
        "matrix_corr_sign_rate",
    }
    return "max" if metric in maximize else "min"


def format_cell(value: Optional[float], epoch: Optional[int], bold: bool, decimals: int) -> str:
    if value is None:
        return ""
    fmt = f"{value:.{decimals}f}"
    if epoch is not None:
        fmt = f"{fmt} ({epoch})"
    if bold:
        return f"**{fmt}**"
    return fmt


def latex_escape(s: str) -> str:
    return s.replace("_", "\\_")


def write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def write_latex(path: Path, header: List[str], rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = "l" + "c" * (len(header) - 1)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{%s}\n" % cols)
        f.write(" \\toprule\n")
        f.write(" \\ & ".join(latex_escape(h) for h in header) + " \\\\ \n")
        f.write(" \\midrule\n")
        for r in rows:
            cells = []
            for c in r:
                def convert_middle_dot(s: str) -> str:
                    # Replace UTF-8 middle dot with LaTeX-friendly center dot
                    return s.replace("·", " $\\cdot$ ")

                if c.startswith("**") and c.endswith("**"):
                    core = convert_middle_dot(c[2:-2])
                    cells.append("\\textbf{" + latex_escape(core) + "}")
                else:
                    cells.append(latex_escape(convert_middle_dot(c)))
            f.write(" \\ & ".join(cells) + " \\\\ \n")
        f.write(" \\bottomrule\n\\end{tabular}\n")


def short_metric_name(name: str) -> str:
    mapping = {
        "matrix_cov_fro": "cov_Fro",
        "matrix_cov_fro_rel": "cov_Fro_rel",
        "matrix_cov_mae": "cov_MAE",
        "matrix_cov_mse": "cov_MSE",
        "matrix_cov_diag_mape": "cov_dMAPE",
        "matrix_corr_fro": "corr_Fro",
        "matrix_corr_fro_rel": "corr_Fro_rel",
        "matrix_corr_cross_mse": "corr_MSE",
        "matrix_corr_offdiag_pearson": "corr_P",
        "matrix_corr_offdiag_spearman": "corr_S",
        "matrix_corr_sign_rate": "corr_sign",
        "series_mae": "MAE",
        "series_mse": "MSE",
        "series_rmse": "RMSE",
        "series_nd": "ND",
        "series_nrmse": "NRMSE",
        "series_nd_sum": "ND_sum",
        "series_nrmse_sum": "NRMSE_sum",
        "series_crps": "CRPS",
        "series_crps_sum": "CRPS_sum",
    }
    return mapping.get(name, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise best_metrics.json across runs into CSV/LaTeX.")
    ap.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    ap.add_argument("--time-conditional", type=Path)
    ap.add_argument("--time-unconditional", type=Path)
    ap.add_argument("--fourier-conditional", type=Path)
    ap.add_argument("--fourier-unconditional", type=Path)
    ap.add_argument("--metrics", nargs="+", default=default_metric_list())
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--include-epoch", action="store_true")
    ap.add_argument("--short-labels", action="store_true", help="Use compact metric names in headers")
    ap.add_argument("--prune-near-constant", action="store_true", help="Drop columns whose values are near-zero or near-constant across runs")
    ap.add_argument("--eps", type=float, default=1e-3, help="Epsilon used for pruning (default: 1e-3)")
    ap.add_argument("--short-models", action="store_true", help="Use abbreviated model names (e.g., T·Cond., F·Uncond.)")
    ap.add_argument(
        "--extra-runs",
        nargs="*",
        help="Optional extra rows of the form LABEL|RUN_DIR; used to append baselines such as DeepVAR.",
    )
    ap.add_argument("--table-csv", type=Path, default=Path("assets/best_metrics_summary.csv"))
    ap.add_argument("--table-tex", type=Path, default=Path("assets/best_metrics_summary.tex"))
    args = ap.parse_args()

    overrides = {
        "time/conditional": getattr(args, "time_conditional", None),
        "time/unconditional": getattr(args, "time_unconditional", None),
        "fourier/conditional": getattr(args, "fourier_conditional", None),
        "fourier/unconditional": getattr(args, "fourier_unconditional", None),
    }
    runs = resolve_runs(args.outputs_root, overrides)

    per_run: Dict[Tuple[str, str], Dict[str, Tuple[float, int]]] = {}
    for key, run_dir in runs.items():
        per_run[key] = read_best_metrics(run_dir)

    # Optional extra baselines
    extra_labels: Dict[Tuple[str, str], str] = {}
    if args.extra_runs:
        for entry in args.extra_runs:
            parts = entry.split("|")
            if len(parts) != 2:
                raise ValueError(f"--extra-runs entries must be LABEL|RUN_DIR; got: {entry}")
            label, run_dir = parts[0], Path(parts[1]).expanduser().resolve()
            if not (run_dir / "best_metrics.json").exists():
                raise FileNotFoundError(f"Missing best_metrics.json under {run_dir}")
            key = ("extra", label)
            runs[key] = run_dir
            per_run[key] = read_best_metrics(run_dir)
            extra_labels[key] = label

    # Optional pruning of near-constant metrics
    metrics_list = list(args.metrics)
    if args.prune_near_constant:
        kept: List[str] = []
        for metric in metrics_list:
            values = []
            for key in ORDER:
                se = per_run.get(key, {}).get(metric)
                if se is not None:
                    values.append(abs(se[0]))
            if len(values) < 2:
                continue  # insufficient data to judge; drop silently
            vmin, vmax = min(values), max(values)
            if vmax < args.eps:
                continue  # all effectively zero
            if (vmax - vmin) < args.eps:
                continue  # near-constant across runs
            kept.append(metric)
        metrics_list = kept if kept else metrics_list  # fallback to original if all dropped

    header_metrics = [short_metric_name(m) if args.short_labels else m for m in metrics_list]
    header = ["Model"] + header_metrics
    if args.short_models:
        label_map = {
            ("time", "conditional"): "T · Cond.",
            ("time", "unconditional"): "T · Uncond.",
            ("fourier", "conditional"): "F · Cond.",
            ("fourier", "unconditional"): "F · Uncond.",
        }
    else:
        label_map = {
            ("time", "conditional"): "Time · Conditional",
            ("time", "unconditional"): "Time · Unconditional",
            ("fourier", "conditional"): "Fourier · Conditional",
            ("fourier", "unconditional"): "Fourier · Unconditional",
        }
    # append labels for extra baselines
    for k, lab in extra_labels.items():
        label_map[k] = lab

    winners: Dict[str, Tuple[float, str]] = {}
    for metric in metrics_list:
        direction = metric_direction(metric)
        best_val: Optional[float] = None
        best_key: Optional[str] = None
        for key in list(ORDER) + list(extra_labels.keys()):
            se = per_run.get(key, {}).get(metric)
            if se is None:
                continue
            score = se[0]
            if best_val is None:
                best_val, best_key = score, f"{key[0]}/{key[1]}"
            else:
                if (direction == "min" and score < best_val) or (direction == "max" and score > best_val):
                    best_val, best_key = score, f"{key[0]}/{key[1]}"
        if best_val is not None and best_key is not None:
            winners[metric] = (best_val, best_key)

    rows: List[List[str]] = []
    row_keys = list(ORDER) + list(extra_labels.keys())
    for key in row_keys:
        row = [label_map[key]]
        metrics_map = per_run.get(key, {})
        for metric in metrics_list:
            val_epoch = metrics_map.get(metric)
            if val_epoch is None:
                row.append("")
                continue
            val, ep = val_epoch
            is_best = winners.get(metric, (None, None))[1] == f"{key[0]}/{key[1]}"
            row.append(format_cell(val, ep if args.include_epoch else None, is_best, args.decimals))
        rows.append(row)

    write_csv(args.table_csv, header, rows)
    write_latex(args.table_tex, header, rows)
    print(f"Wrote {args.table_csv} and {args.table_tex}")


if __name__ == "__main__":
    main()
